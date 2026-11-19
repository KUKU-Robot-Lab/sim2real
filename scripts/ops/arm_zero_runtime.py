"""Hardware/ROS adapters are imported only after offline preflight succeeds."""
from __future__ import annotations

import json
import time
import uuid

import numpy as np
import yaml

from arm_zero_calibrate import (Abort, Coordinates, ORDER, Runner, Sample, Settings,
                                candidate_delta, digest, reference, vector)

TOPIC = '/arm_zero_calibration/events'


def validate_artifacts(candidate, history, side, session, model_hash, joint=None):
    if any(candidate.get(k) != v for k, v in dict(version=1, side=side, session=session,
                                                  model_hash=model_hash).items()):
        raise Abort('candidate version/side/session/model mismatch')
    if candidate.get('reviewed_no_contact') is not True:
        raise Abort('candidate requires operator review: exclude body/cable/external force contact')
    if history.get('candidate_id') != candidate.get('candidate_id'):
        raise Abort('history does not belong to candidate')
    if history.get('unknown_joints'):
        raise Abort('unknown zero-write outcome; inspect fresh coordinates, never retry automatically')
    saved = history.get('saved_joints', [])
    if len(set(saved)) != len(saved) or any(type(j) is not int or j not in range(1, 8) for j in saved):
        raise Abort('invalid saved joint history')
    if joint in saved:
        raise Abort('joint already saved; candidate cannot be reused')
    coords = Coordinates(candidate['delta'])
    for j in range(1, 8):
        d = candidate_delta(candidate['hits'][str(j)], reference(side, j))
        if abs(d - coords.delta[j - 1]) > 1e-10:
            raise Abort('candidate delta does not match bag stop readings')
    for j in saved:
        coords.saved(j)
    return coords


def reduce_events(events):
    """Reduce chronological bag events; missing verified event leaves UNKNOWN."""
    candidates = [e for e in events if e['kind'] == 'candidate']
    if len(candidates) != 1:
        raise Abort('supply bags containing exactly one completed candidate')
    candidate = dict(candidates[0])
    candidate['reviewed_no_contact'] = False
    saved, unknown = set(), set()
    for e in events:
        if e['kind'] not in ('zero_attempt', 'zero_verified'):
            continue
        if e.get('candidate_id') != candidate['candidate_id']:
            raise Abort('mixed candidate epochs in supplied bags')
        j = e['joint']
        if e['kind'] == 'zero_attempt':
            if j in saved or j in unknown:
                raise Abort('duplicate zero attempt; history needs manual investigation')
            unknown.add(j)
        elif j not in unknown:
            raise Abort('zero_verified without preceding attempt')
        else:
            unknown.remove(j)
            saved.add(j)
    return candidate, dict(candidate_id=candidate['candidate_id'],
                           saved_joints=sorted(saved), unknown_joints=sorted(unknown))


def validate_sources(candidate, history):
    from pathlib import Path
    from arm_zero_bag import read_bags
    sources = history.get('source_bags', [])
    if not sources or not candidate.get('source_bags'):
        raise Abort('candidate/history must be derived from existing bags')
    for source in sources:
        stat = Path(source['path']).stat()
        if stat.st_size != source['size'] or stat.st_mtime_ns != source['mtime_ns']:
            raise Abort('source bag changed; stop recording and regenerate analysis')
    events, _ = read_bags([Path(source['path']) for source in sources])
    original, actual_history = reduce_events(events)
    for key in ('candidate_id', 'delta', 'hits', 'verify_route', 'side', 'session', 'model_hash'):
        if candidate.get(key) != original.get(key):
            raise Abort(f'candidate differs from bag: {key}')
    for key, value in actual_history.items():
        if history.get(key) != value:
            raise Abort(f'partial-save history differs from bag: {key}')


def validate_attached_baseline(events, candidate_id, route):
    """Require a complete detached run in the same zero epoch before enabling."""
    finished = {e.get('run_id') for e in events
                if e['kind'] == 'finished' and e.get('command') == 'verify'}
    runs = {}
    last_index = {}
    for index, e in enumerate(events):
        if (e['kind'] == 'verification' and e.get('hand') == 'detached'
                and e.get('candidate_id') == candidate_id and e.get('run_id') in finished):
            runs.setdefault(e['run_id'], {})[e['pose_index']] = e['model_target']
            last_index[e['run_id']] = index
    complete = [run for run, poses in runs.items()
                if set(poses) == set(range(len(route)))
                and all(np.allclose(poses[i], pose, atol=1e-10, rtol=0)
                        for i, pose in enumerate(route))]
    if not complete:
        raise Abort('attached verify requires completed detached verification at the same targets')
    last = max(last_index[run] for run in complete)
    if any(e['kind'] in ('zero_attempt', 'zero_verified') for e in events[last + 1:]):
        raise Abort('zero changed after detached baseline; repeat detached verify first')


class Recorder:
    """One reliable event stream, captured by existing rh56f1_record.sh PD bag."""
    def __init__(self, metadata):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        self.rclpy, self.String = rclpy, String
        self.metadata = metadata
        rclpy.init()
        self.node = rclpy.create_node('arm_zero_calibration')
        self.pub = self.node.create_publisher(String, TOPIC, QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=.05)
            if self.recording():
                return
        self.close()
        raise Abort('existing rosbag recorder is not subscribed to calibration topic')

    def recording(self):
        return any('recorder' in ep.node_name for ep in self.node.get_subscriptions_info_by_topic(TOPIC))

    def __call__(self, kind, **data):
        if not self.recording():
            raise Abort('bag recorder disappeared')
        message = self.String()
        message.data = json.dumps(dict(self.metadata, **data, kind=kind, wall_ns=time.time_ns()), allow_nan=False)
        self.pub.publish(message)
        self.rclpy.spin_once(self.node, timeout_sec=0)
        if kind in ('zero_attempt', 'zero_verified'):
            from rclpy.duration import Duration
            if not self.pub.wait_for_all_acked(Duration(nanoseconds=5_000_000)):
                raise Abort(f'bag transport acknowledgement timeout: {kind}')

    def close(self):
        # Called after the bus cleanup: never pause an active holding loop to
        # await final ROS/DDS delivery. Five milliseconds was too short for this.
        from rclpy.duration import Duration
        try:
            if not self.pub.wait_for_all_acked(Duration(seconds=1)):
                if not self.metadata.get('read_only'):
                    raise Abort('bag transport acknowledgement timeout during close')
                import sys
                print('읽기 전용 기록은 종료된 bag의 완료 이벤트로 검증합니다.', file=sys.stderr)
        finally:
            self.node.destroy_node()
            self.rclpy.shutdown()


class CanBus:
    """SDK packet encoder/decoder, SocketCAN kernel timestamps; no cached getters.

    Only IDs 1..7 on one bus exist here. No gripper, no set_zero_all API.
    Every read requests new states; queued pre-request frames cannot count.
    """
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    def __init__(self, interface, deadline):
        import fcntl
        import os
        from pathlib import Path
        import socket
        import struct
        import importlib
        oa = importlib.import_module('openarm_can')
        self.oa, self.struct, self.socket = oa, struct, socket
        self.deadline = deadline
        self.disabled = set(range(1, 8))
        self.touched = False
        self.seq = np.zeros(7, dtype=int)
        self.lock = open('/tmp/arm_zero_calibration.lock', 'a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # CAN receiver registrations expose conflicting SocketCAN users, including
        # ROS drivers; never stop somebody else's process to acquire the bus.
        paths = list(Path('/proc/net/can').glob('rcvlist_*'))
        if not paths:
            raise Abort('cannot inspect CAN ownership registrations')
        for path in paths:
            for line in path.read_text().splitlines():
                fields = line.split()
                if fields and fields[0] in ('can0', 'can1', 'any'):
                    raise Abort(f'CAN receiver already registered: {path.name}: {line.strip()}')
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, 35, 1)  # Linux SO_TIMESTAMPNS
        self.sock.setblocking(False)
        self.sock.bind((interface,))
        types = ['DM8009', 'DM8009', 'DM4340', 'DM4340', 'DM4310', 'DM4310', 'DM4310']
        self.motors = [oa.Motor(getattr(oa.MotorType, t), i + 1, i + 0x11) for i, t in enumerate(types)]
        self.pid = os.getpid()

    def send(self, packet):
        data = bytes(packet.data)
        frame = self.struct.pack('=IBBBB64s', packet.send_can_id, len(data), 0, 0, 0, data)
        if self.sock.send(frame) != len(frame):
            raise Abort('short CAN write')

    def read(self):
        import select
        start_wall, start = time.time(), self.now()
        for motor in self.motors:
            self.send(self.oa.CanPacketEncoder.create_refresh_command(motor))
        states = {}
        frames = 0
        while len(states) < 7 and self.now() - start < self.deadline:
            if not select.select([self.sock], [], [], max(0, self.deadline - (self.now() - start)))[0]:
                break
            raw, ancillary, flags, _ = self.sock.recvmsg(72, 128)
            frames += 1
            if flags or frames > 256:
                raise Abort('truncated/flooded CAN reception')
            if len(raw) not in (16, 72):
                continue
            can_id, length = self.struct.unpack('=IB', raw[:5])
            if can_id not in range(0x11, 0x18) or length != 8:
                continue
            stamps = [self.struct.unpack('@ll', data[:16]) for level, kind, data in ancillary
                      if level == self.socket.SOL_SOCKET and kind == 35]
            if len(stamps) != 1:
                raise Abort('CAN kernel receive timestamp missing')
            stamp_wall = stamps[0][0] + stamps[0][1] / 1e9
            if stamp_wall < start_wall:
                continue
            age = time.time() - stamp_wall
            if age < 0 or age > self.deadline:
                raise Abort('stale CAN kernel timestamp/clock discontinuity')
            idx, data = can_id - 0x11, raw[8:16]
            if data[0] & 0xF != idx + 1:
                raise Abort('CAN state motor ID mismatch')
            status = data[0] >> 4
            if status not in (0, 1):
                raise Abort(f'motor {idx + 1} fault {status}')
            state = self.oa.CanPacketDecoder.parse_motor_state_data(self.motors[idx], list(data))
            if not state.valid or max(state.t_mos, state.t_rotor) > 65:
                raise Abort('invalid/overtemperature motor state')
            self.seq[idx] += 1
            states[idx] = (state.position, state.velocity, state.torque, self.now() - age, status == 1)
        if len(states) != 7:
            raise Abort('stale feedback: not all seven motors replied')
        a = np.array([states[i] for i in range(7)])
        return Sample(a[:, 0], a[:, 1], a[:, 2], a[:, 3], self.seq.copy(), a[:, 4].astype(bool))

    def enable(self):
        self.touched = True
        # Cleanup disables all motors even if this loop fails halfway through.
        for i, motor in enumerate(self.motors, 1):
            self.send(self.oa.CanPacketEncoder.create_enable_command(motor))
            self.disabled.discard(i)

    def command(self, target, gravity, kp, kd):
        for i, motor in enumerate(self.motors):
            if i + 1 not in self.disabled:
                p = self.oa.MITParam(float(kp[i]), float(kd[i]), float(target[i]), 0., float(gravity[i]))
                self.send(self.oa.CanPacketEncoder.create_mit_control_command(motor, p))

    def disable(self, joint):
        self.send(self.oa.CanPacketEncoder.create_disable_command(self.motors[joint - 1]))
        self.disabled.add(joint)

    def zero(self, joint):
        if joint not in self.disabled:
            raise Abort('zero requires disabled motor')
        self.send(self.oa.CanPacketEncoder.create_set_zero_command(self.motors[joint - 1]))

    def invalidate(self):
        # read() accepts only frames timestamped after its next refresh request.
        pass

    def close(self):
        failed = []
        for j in (range(1, 8) if self.touched else []):
            try:
                self.disable(j)
            except Exception as exc:
                failed.append(f'j{j}: {exc}')
        self.sock.close()
        self.lock.close()
        if failed:
            raise Abort('disable failed; operator emergency support required: ' + '; '.join(failed))


def resolve_inputs(args, cfg, model):
    """Read disabled arms and plan before enabling. All snapshots use the bag."""
    import os
    from arm_zero_auto import resolve
    if os.environ.get('ROS_DOMAIN_ID') != '50':
        raise Abort('ROS_DOMAIN_ID=50 required')
    delta = cfg['operation']['initial_delta']
    if args.candidate is not None:
        candidate = yaml.safe_load(args.candidate.read_text())
        if args.history is None:
            raise Abort('candidate requires --history')
        history = yaml.safe_load(args.history.read_text())
        validate_sources(candidate, history)
        delta = validate_artifacts(candidate, history, args.side,
                                   cfg['operation']['session'], model.source_hash,
                                   args.joint if args.command == 'apply' else None).delta
    recorder = Recorder(dict(version=1, side=args.side, hand=args.hand,
                             session=cfg['operation']['session'], run_id=str(uuid.uuid4()),
                             model_hash=model.source_hash, config_hash=digest(cfg), candidate_id=None, read_only=True))
    samples = {}
    try:
        for side in ('right', 'left'):
            bus = CanBus(cfg['sides'][side]['can'], cfg['search']['freshness'] * .8)
            try:
                def record_state(kind, **data):
                    recorder('startup_state', snapshot_side=side, **data)
                try:
                    positions = observe_existing_zero(bus, record_state,
                        cfg['search']['stop_velocity'], cfg['search']['position_tolerance'])
                except Abort as exc:
                    raise Abort(f'{side}: {exc}') from exc
                samples[side] = positions[-1].copy()
                recorder('startup_snapshot', snapshot_side=side, q=samples[side].tolist())
            finally:
                bus.close()  # no enable/disable command: touched is False
    finally:
        recorder.close()
    other = 'left' if args.side == 'right' else 'right'
    return resolve(cfg, args.side, samples[args.side], samples[other], delta)


def record_preflight_failure(args, cfg, model, report):
    recorder = Recorder(dict(version=1, side=args.side, hand=args.hand,
                             session=cfg['operation']['session'], run_id=str(uuid.uuid4()),
                             model_hash=model.source_hash, config_hash=digest(cfg), candidate_id=None))
    try:
        recorder('aborted', phase='preflight', reason='; '.join(report['blockers']),
                 path_samples_checked=report['path_samples_checked'])
    finally:
        recorder.close()


def observe_existing_zero(bus, event, speed_limit, span_limit):
    """Bounded read-only settling: 30 stable samples, at most 150 observations.

    A single velocity outlier is not a disconnect. Real position drift and the
    median reported speed must both pass; persistent motion never passes.
    """
    from collections import deque
    window = deque(maxlen=30)
    span = np.zeros(7)
    velocity = np.zeros(7)
    for _ in range(150):
        sample = bus.read()
        vector(sample.q, 'inspection q')
        vector(sample.dq, 'inspection dq')
        event('zero_inspection_state', q=sample.q.tolist(), dq=sample.dq.tolist(),
              torque=sample.torque.tolist(), stamp=sample.stamp.tolist(),
              sequence=sample.sequence.tolist(), enabled=sample.enabled.tolist())
        if np.any(sample.enabled):
            ids = (np.flatnonzero(sample.enabled) + 1).tolist()
            raise Abort(f'inspection: CAN connected, motors enabled {ids}; no motor commands sent')
        window.append(sample)
        if len(window) == 30:
            positions = np.array([s.q for s in window])
            span = np.ptp(positions, axis=0)
            velocity = np.median(np.abs(np.array([s.dq for s in window])), axis=0)
            if np.max(span) <= span_limit and np.max(velocity) <= speed_limit:
                return positions
        bus.sleep(.02)
    raise Abort(f'inspection stationary timeout: CAN connected; '
                f'q_span_rad={span.tolist()}, median_abs_dq={velocity.tolist()}')


def inspect_zero(args, cfg):
    """Observe existing encoder zeros. Never enable, command, disable or zero."""
    import hashlib
    from arm_zero_calibrate import ROOT
    source = (ROOT / cfg['model']['urdf']).resolve()
    recorder = Recorder(dict(version=1, session=cfg['operation']['session'], side=args.side,
                             hand=args.hand, model_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
                             run_id=str(uuid.uuid4()), candidate_id=None, read_only=True))
    bus = None
    try:
        bus = CanBus(cfg['sides'][args.side]['can'], cfg['search']['freshness'] * .8)
        samples = observe_existing_zero(bus, recorder, cfg['search']['stop_velocity'],
                                        cfg['search']['position_tolerance'])
        recorder('zero_inspection', mean=np.mean(samples, axis=0).tolist(),
                 std=np.std(samples, axis=0).tolist(), count=len(samples),
                 physical_reference='operator confirmed same physical zero reference',
                 zero_written=False, motor_command_sent=False)
        recorder('finished', command='inspect')
    except BaseException as exc:
        try:
            recorder('aborted', phase='inspect', reason=repr(exc))
        except Exception:
            pass
        raise
    finally:
        try:
            if bus is not None:
                bus.close()  # touched=False: no disable command either
        finally:
            recorder.close()


def execute(args, cfg, model):
    import os
    if os.environ.get('ROS_DOMAIN_ID') != '50':
        raise Abort('real calibration requires explicit ROS_DOMAIN_ID=50')
    op = cfg['operation']
    settings = Settings(**{k: v for k, v in cfg['search'].items() if k != 'torque_review'})
    candidate = yaml.safe_load(args.candidate.read_text()) if args.candidate else None
    history = yaml.safe_load(args.history.read_text()) if args.history else None
    if args.command == 'apply' and (candidate is None or history is None):
        raise Abort('apply requires candidate and up-to-date bag-derived --history')
    if candidate is not None:
        if history is None:
            raise Abort('candidate needs bag-derived history')
        validate_sources(candidate, history)
        coords = validate_artifacts(candidate, history, args.side, op['session'], model.source_hash,
                                    args.joint if args.command == 'apply' else None)
        if np.max(np.abs(coords.delta)) > settings.max_offset:
            raise Abort('candidate outside configured offset bound')
    else:
        coords = Coordinates(op['initial_delta'])
        if args.command == 'verify' and args.hand == 'attached':
            raise Abort('attached verification needs detached candidate/history for identical coordinates')
    if candidate is not None and candidate.get('verify_route') != op['verify_route']:
        raise Abort('verification route differs from detached candidate baseline')
    if args.command == 'verify' and args.hand == 'attached':
        if candidate is None or history is None:
            raise Abort('attached verification needs detached candidate/history')
        from pathlib import Path
        from arm_zero_bag import read_bags
        events, _ = read_bags([Path(source['path']) for source in history['source_bags']])
        validate_attached_baseline(events, candidate['candidate_id'], op['verify_route'])
    # Plan every supplied route before CAN is opened. Runtime guard checks actual states too.
    routes = {str(k): v for k, v in op['measure_routes'].items()} if args.command == 'measure' else {'verify': op['verify_route']}
    if args.command == 'measure' and set(map(int, routes)) != set(ORDER):
        raise Abort('measure requires reviewed preparation routes for every manufacturer stage')
    metadata = dict(version=1, session=op['session'], side=args.side, hand=args.hand,
                    model_hash=model.source_hash, config_hash=digest(cfg), run_id=str(uuid.uuid4()),
                    candidate_id=candidate['candidate_id'] if candidate else None)
    model.load_bounds()  # geometry I/O must finish before opening/enabling CAN
    recorder = Recorder(metadata)
    bus = None
    try:
        bus = CanBus(cfg['sides'][args.side]['can'], settings.freshness * .8)
        # Before correction, an encoder reading at a true stop may differ from
        # the nominal URDF limit. Do not confuse that bounded offset with travel.
        slack = settings.max_offset if args.command in ('measure', 'apply') else 0.
        from arm_zero_auto import AutoRunner
        runner_type = AutoRunner if op.get('auto_profile') else Runner
        runner = runner_type(bus, settings, coords, model.gravity,
                        lambda q: model.guard(q, limit_slack=slack), recorder)
        state = runner.read()
        if np.any(state.enabled):
            raise Abort('motors already enabled; exclusive control not established')
        if np.max(np.abs(state.q - vector(op['start_raw'], 'start_raw'))) > .02:
            raise Abort('startup posture differs from reviewed start_raw')
        recorder('started', command=args.command, initial_delta=coords.delta.tolist(),
                 start_raw=cfg['operation']['start_raw'], other_arm_q=cfg['sides'][args.side]['other_arm_q'],
                 measure_routes=op['measure_routes'], control_profile=op.get('auto_profile'))
        bus.enable()
        runner.target = state.q.copy()
        runner.tick(runner.target)
        if args.command == 'measure':
            hits = {}
            original_event = runner.event
            def collect(kind, **data):
                if kind == 'joint_candidate':
                    hits[str(data['joint'])] = data['hits']
                original_event(kind, **data)
            runner.event = collect
            for j in ORDER:
                if isinstance(runner, AutoRunner):
                    runner.stage_joint = j
                recorder('stage', joint=j)
                for waypoint in routes[str(j)]:
                    runner.move(coords.raw(waypoint))
                coords.delta[j - 1] = runner.measure_joint(args.side, j)
            if isinstance(runner, AutoRunner):
                runner.finish_measurement()
            cid = str(uuid.uuid4())
            recorder('candidate', candidate_id=cid, delta=coords.delta.tolist(), hits=hits,
                     reviewed_no_contact=False, power_cycle_verified=False, verify_route=op['verify_route'])
        elif args.command == 'apply':
            if isinstance(runner, AutoRunner):
                runner.stage_joint = args.joint
            # Bind the candidate to the current physical encoder epoch. An old
            # history file alone must not cause a second application of delta.
            for waypoint in op['measure_routes'].get(args.joint, op['measure_routes'].get(str(args.joint), [])):
                runner.move(coords.raw(waypoint))
            observed = runner.measure_joint(args.side, args.joint)
            if abs(observed - coords.delta[args.joint - 1]) > settings.position_tolerance:
                raise Abort('current stop disagrees with candidate; possible changed zero epoch')
            runner.apply_joint(args.joint)
        else:
            for index, pose in enumerate(op['verify_route']):
                state = runner.move(coords.raw(pose))
                recorder('verification', pose_index=index, model_target=list(pose),
                         raw_target=coords.raw(pose).tolist(), q=state.q.tolist(),
                         model_q=coords.model(state.q).tolist(),
                         flange_fk=model.flange(coords.model(state.q)).tolist(),
                         measured_flange=None, measured_hand_relative_flange=None,
                         power_cycle_verified=False)
        recorder('finished', command=args.command)
    except BaseException as exc:
        try:
            recorder('aborted', reason=repr(exc))
        except Exception:
            pass
        raise
    finally:
        try:
            if bus is not None:
                bus.close()  # supported posture required; never unverified home recovery
        finally:
            recorder.close()
