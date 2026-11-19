#!/usr/bin/env python3
"""Single-arm stop calibration. No hardware import/access without --execute.

Angles are radians internally. A saved joint starts a new coordinate epoch.
Raw experiment data goes only to ROS topics recorded by rh56f1_record.sh.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORDER = (4, 3, 5, 6, 7, 2, 1)
REFERENCES = {'right': (-80, -10, 90, 0, 90, 45, 90),
              'left': (80, 10, 90, 0, 90, 45, 90)}


class Abort(RuntimeError):
    """No automatic retry, zero write, or home move after this exception."""


def reference(side, joint):
    return float(np.deg2rad(REFERENCES[side][joint - 1]))


def vector(values, name, n=7):
    result = np.asarray(values, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise Abort(f'{name}: expected {n} finite numbers')
    return result


def candidate_delta(hits, ref):
    hits = vector(hits, 'three stop readings', 3)
    if not np.isfinite(ref) or np.ptp(hits) > np.deg2rad(.5):
        raise Abort('stop repeatability exceeds 0.5 degrees')
    return float(np.mean(hits) - ref)


class Coordinates:
    def __init__(self, delta):
        self.delta = vector(delta, 'delta').copy()

    def raw(self, model):
        return vector(model, 'model q') + self.delta

    def model(self, raw):
        return vector(raw, 'raw q') - self.delta

    def saved(self, joint):
        self.delta[joint - 1] = 0.


@dataclass
class Sample:
    q: np.ndarray
    dq: np.ndarray
    torque: np.ndarray
    stamp: np.ndarray
    sequence: np.ndarray
    enabled: np.ndarray


@dataclass(frozen=True)
class Settings:
    torque_cap: tuple
    contact_torque: tuple
    speed: float = .05
    dt: float = .01
    freshness: float = .05
    max_travel: float = .35
    timeout: float = 30.
    min_travel: float = .035
    backoff: float = .07
    stop_velocity: float = .0125
    stop_duration: float = .3
    position_tolerance: float = .004
    hold_error: float = .08
    disable_duration: float = .3
    max_offset: float = .175
    total_travel: float = 12.
    total_timeout: float = 900.

    def __post_init__(self):
        for key, value in vars(self).items():
            if key in ('torque_cap', 'contact_torque'):
                if np.any(vector(value, key) <= 0):
                    raise Abort(f'{key} must be positive')
            elif not np.isfinite(value) or value <= 0:
                raise Abort(f'invalid {key}')
        if self.speed > .05 or self.dt >= self.freshness:
            raise Abort('speed > 0.05 rad/s or control period >= feedback deadline')
        if np.any(np.asarray(self.contact_torque) >= self.torque_cap):
            raise Abort('contact torque must be below torque cap')
        if not self.min_travel < self.backoff < self.max_travel:
            raise Abort('require min_travel < backoff < max_travel')


class Runner:
    """Bounded control core; bus, clock, geometry and bag publisher are injected."""
    def __init__(self, bus, settings, coordinates, gravity, guard, event: Callable[..., None]):
        self.bus, self.s, self.coordinates = bus, settings, coordinates
        self.gravity, self.guard, self.event = gravity, guard, event
        self.target = None
        self.last_sequence = None
        self.last_tick = None
        self.unknown_epoch = False
        self.started = self.bus.now()
        self.travelled = 0.
        self.previous_q = None

    def read(self):
        if self.unknown_epoch:
            raise Abort('unknown coordinate epoch')
        now = self.bus.now()
        if now - self.started > self.s.total_timeout:
            raise Abort('session timeout')
        if self.last_tick is not None and now - self.last_tick > self.s.freshness:
            raise Abort('stale control cycle')
        state = self.bus.read()
        now = self.bus.now()
        for key in ('q', 'dq', 'torque', 'stamp', 'sequence'):
            vector(getattr(state, key), key)
        age = now - state.stamp
        if np.any(age < 0) or np.any(age > self.s.freshness):
            raise Abort('stale feedback')
        if self.last_sequence is not None and np.any(state.sequence <= self.last_sequence):
            raise Abort('stale/reused feedback sequence')
        self.last_sequence = state.sequence.copy()
        self.last_tick = now
        if np.any(np.abs(state.dq) > self.s.speed * 2):
            joints = (np.flatnonzero(np.abs(state.dq) > self.s.speed * 2) + 1).tolist()
            self.event('limit_violation', limit='velocity', joints=joints,
                       threshold=self.s.speed * 2, q=state.q.tolist(),
                       dq=state.dq.tolist(), torque=state.torque.tolist(),
                       enabled=state.enabled.tolist())
            raise Abort(f'velocity limit exceeded: joints={joints}, '
                        f'dq={state.dq.tolist()}, limit={self.s.speed * 2} rad/s')
        if np.any(np.abs(state.torque) > self.s.torque_cap):
            raise Abort('torque limit exceeded')
        if self.previous_q is not None:
            self.travelled += float(np.sum(np.abs(state.q - self.previous_q)))
        self.previous_q = state.q.copy()
        if self.travelled > self.s.total_travel:
            raise Abort('session total travel exceeded')
        self.guard(self.coordinates.model(state.q))
        self.event('state', q=state.q.tolist(), dq=state.dq.tolist(),
                   torque=state.torque.tolist(), stamp=state.stamp.tolist(),
                   sequence=state.sequence.tolist(), enabled=state.enabled.tolist())
        return state

    def tick(self, target, search=None, disabled=None):
        target = vector(target, 'target')
        state = self.read()
        holding = [i for i in range(7) if i != disabled]
        if not np.all(state.enabled[holding]):
            raise Abort('motor disabled/fault while holding')
        keep = [i for i in range(7) if i != search and i != disabled]
        if np.any(np.abs(state.q[keep] - target[keep]) > self.s.hold_error):
            raise Abort('non-search joint hold error')
        geometry_target = self.coordinates.model(target)
        if search is not None:
            # A bounded PD target may push past the physical stop. Collision and
            # hard-limit checks apply to actual geometry, not the virtual spring.
            geometry_target[search] = self.coordinates.model(state.q)[search]
        self.guard(geometry_target)
        kp, kd, gravity = self.control_parameters(state, target, search)
        if np.any(np.abs(gravity) >= self.s.torque_cap):
            raise Abort('gravity exceeds torque cap')
        # Firmware PD is also bounded: cap total requested effort, not just feedforward.
        effort = kp * (target - state.q) - kd * state.dq + gravity
        if np.any(np.abs(effort) > self.s.torque_cap):
            raise Abort('command torque limit exceeded')
        self.bus.command(target, gravity, kp, kd)
        self.target = target.copy()
        self.event('command', q=target.tolist(), gravity=gravity.tolist(),
                   kp=kp.tolist(), kd=kd.tolist())
        self.bus.sleep(self.s.dt)
        return state

    def control_parameters(self, state, target, search):
        return (np.array([70, 70, 70, 60, 20, 20, 20.]),
                np.array([2.75, 2.5, 2., 2., .7, .6, .5]),
                vector(self.gravity(self.coordinates.model(state.q)), 'gravity'))

    def locate(self, side, joint, direction):
        return self.approach(joint, direction)

    def position_settled(self, state, goal):
        # 저장용 정밀 도달 판정. 영점 FF 제어는 AutoRunner 가 처짐을 고려해 덮어쓴다.
        return np.max(np.abs(state.q - goal)) <= self.s.position_tolerance

    def move(self, goal):
        goal = vector(goal, 'move goal')
        start = self.read().q.copy()
        distance = float(np.max(np.abs(goal - start)))
        # Quintic smoothstep has peak derivative 1.875. Stretch time so
        # peak reference speed stays within the existing configured cap.
        duration = 1.875 * distance / self.s.speed
        if duration + self.s.stop_duration > self.s.timeout:
            raise Abort('move timeout budget insufficient')
        # Check the entire interpolation before sending the first command.
        steps = max(1, int(np.ceil(duration / self.s.dt)))
        movement_started = self.bus.now()
        for i in range(steps + 1):
            if self.bus.now() - movement_started > self.s.timeout:
                raise Abort('move wall-clock timeout during path check')
            self.guard(self.coordinates.model(start + (goal - start) * i / steps))
            self.tick(start)  # prechecking a long path must not interrupt holding
        for i in range(steps + 1):
            if self.bus.now() - movement_started > self.s.timeout:
                raise Abort('move wall-clock timeout')
            u = i / steps
            blend = u ** 3 * (10 + u * (-15 + 6 * u))
            self.tick(start + (goal - start) * blend)
        t0, settled = self.bus.now(), None
        while (self.bus.now() - t0 < self.s.stop_duration + 3
               and self.bus.now() - movement_started < self.s.timeout):
            state = self.tick(goal)
            ok = (self.position_settled(state, goal)
                  and np.max(np.abs(state.dq)) <= self.s.stop_velocity)
            settled = self.bus.now() if ok and settled is None else settled if ok else None
            if settled is not None and self.bus.now() - settled >= self.s.stop_duration:
                return state
        raise Abort('target not reached/stationary or move timeout')

    def approach(self, joint, direction, locating_reference=None):
        idx = joint - 1
        start = self.read().q.copy()
        target = start.copy()
        t0, stopped = self.bus.now(), None
        travelled = 0.
        previous = start[idx]
        while self.bus.now() - t0 <= self.s.timeout:
            target[idx] += direction * self.s.speed * self.s.dt
            state = self.tick(target, search=idx)
            travelled += abs(state.q[idx] - previous)
            previous = state.q[idx]
            if max(travelled, abs(target[idx] - start[idx])) > self.s.max_travel:
                raise Abort('search travel exceeded without stop')
            g = self.gravity(self.coordinates.model(state.q))[idx]
            contact = (abs(state.dq[idx]) <= self.s.stop_velocity
                       and direction * (state.torque[idx] - g) >= self.s.contact_torque[idx])
            if contact and direction * (state.q[idx] - start[idx]) < self.s.min_travel:
                if (locating_reference is None
                        or abs(state.q[idx] - locating_reference) > self.s.max_offset):
                    raise Abort('early stop: insufficient real travel; inspect cable/body contact')
            if contact:
                # Freeze target during confirmation; never keep winding up at a stop.
                target[idx] -= direction * self.s.speed * self.s.dt
                if stopped is None:
                    stopped = (self.bus.now(), state.q[idx])
                if abs(state.q[idx] - stopped[1]) > self.s.position_tolerance:
                    stopped = (self.bus.now(), state.q[idx])
                if self.bus.now() - stopped[0] >= self.s.stop_duration:
                    self.event('stop', joint=joint, q=float(state.q[idx]), travelled=travelled)
                    return float(state.q[idx])
            else:
                stopped = None
        raise Abort('search timeout without stop')

    def measure_joint(self, side, joint):
        direction = -1 if joint == 4 or (side == 'right' and joint in (1, 2)) else 1
        hits = []
        # A first locating pass is discarded. All three measured approaches start
        # after the same backoff, in the same direction.
        hit = self.locate(side, joint, direction)
        for _ in range(3):
            goal = self.read().q.copy()
            goal[joint - 1] = hit - direction * self.s.backoff
            self.move(goal)
            hit = self.approach(joint, direction)
            hits.append(hit)
        delta = candidate_delta(hits, reference(side, joint))
        if abs(delta) > self.s.max_offset:
            raise Abort('offset outside reviewed range')
        self.event('joint_candidate', joint=joint, hits=hits, delta=delta)
        # Unload stop, without returning to an unverified home.
        goal = self.read().q.copy()
        goal[joint - 1] = hit - direction * self.s.backoff
        self.move(goal)
        return delta

    def apply_joint(self, joint):
        idx = joint - 1
        goal = self.read().q.copy()
        goal[idx] = self.coordinates.delta[idx]  # q_model = 0 -> q_raw = delta
        before = self.move(goal)
        self.event('disable_for_zero', joint=joint, q=float(before.q[idx]))
        self.bus.disable(joint)  # other six motors remain held
        t0 = self.bus.now()
        while self.bus.now() - t0 < self.s.disable_duration:
            state = self.tick(goal, disabled=idx)
            if (state.enabled[idx] or abs(state.q[idx] - before.q[idx]) > self.s.position_tolerance
                    or abs(state.dq[idx]) > self.s.stop_velocity):
                raise Abort('movement/not disabled after disable; zero not written')
        self.event('zero_attempt', joint=joint, delta=float(self.coordinates.delta[idx]))
        # From this point an interruption leaves an UNKNOWN epoch, not a retryable candidate.
        try:
            self.bus.zero(joint)
            self.bus.invalidate()
            self.target = None
            # Do not send any old-coordinate targets after the zero write.
            self.coordinates.saved(joint)
            self.previous_q = None  # zero is a coordinate change, not physical travel
            state = self.read()
            if state.enabled[idx] or abs(state.q[idx]) > self.s.position_tolerance:
                raise Abort('zero write not verified by new response')
            if abs(state.dq[idx]) > self.s.stop_velocity:
                raise Abort('motion after zero write')
        except BaseException:
            self.unknown_epoch = True
            self.target = None
            self.bus.invalidate()
            raise
        self.coordinates.saved(joint)
        self.event('zero_verified', joint=joint, delta=self.coordinates.delta.tolist(),
                   power_cycle_verified=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'measure', 'apply', 'verify', 'inspect'))
    parser.add_argument('--side', choices=('right', 'left'), required=True)
    parser.add_argument('--hand', choices=('detached', 'attached'), default='detached')
    parser.add_argument('--joint', type=int, choices=range(1, 8))
    parser.add_argument('--for', dest='for_command', choices=('measure', 'apply', 'verify'),
                        help='plan 전용: 이 명령이 실제로 가는 경로만 검사')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--history', type=Path, help='bag-derived epoch/partial-save analysis')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/arm_zero_calibration.yaml')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        from arm_zero_model import prepare
        cfg = yaml.safe_load(args.config.read_text())
        if args.command == 'inspect':
            if args.execute:
                from arm_zero_runtime import inspect_zero
                inspect_zero(args, cfg)
            else:
                print('기존 영점 읽기 전용 확인: 모터 지령·영점 저장 없음')
            return 0
        plan_for = args.for_command if args.command == 'plan' else args.command
        model, report = prepare(cfg, args.side, args.hand, command=plan_for, joint=args.joint)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.command == 'plan' or not args.execute:
            return 0 if report['ready'] else 2
        if report.get('requires_live_snapshot') and not report['blockers']:
            from arm_zero_runtime import resolve_inputs
            cfg = resolve_inputs(args, cfg, model)
            model, report = prepare(cfg, args.side, args.hand, command=plan_for, joint=args.joint)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report['ready']:
            if cfg['operation'].get('auto_profile'):
                from arm_zero_runtime import record_preflight_failure
                record_preflight_failure(args, cfg, model, report)
            raise Abort('preflight blocked: ' + '; '.join(report['blockers']))
        if args.command != 'verify' and args.hand != 'detached':
            raise Abort('measure/apply require hand AND adaptor detached')
        if args.command == 'apply' and (args.joint is None or args.candidate is None):
            raise Abort('apply requires one --joint and --candidate')
        from arm_zero_runtime import execute
        execute(args, cfg, model)
        return 0
    except (Abort, ValueError, KeyError, OSError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.modules['arm_zero_calibrate'] = sys.modules[__name__]
    raise SystemExit(main())
