import copy

import numpy as np
import pytest
import yaml

from arm_zero_auto import configure, measurement_routes, resolve
from arm_zero_calibrate import ORDER, ROOT
from arm_zero_model import prepare


@pytest.mark.parametrize('side', ['right', 'left'])
def test_manufacturer_preparations_and_bounded_segments(side):
    start = np.zeros(7)
    start[3] = .6
    routes = measurement_routes(start, side)
    assert list(routes) == list(ORDER)
    assert routes[4][-1] == start.tolist()
    for joint in ORDER:
        # Search starts at this joint's current angle, not .225 rad from a stop.
        assert routes[joint][-1][joint - 1] == pytest.approx(start[joint - 1])
    assert routes[5][-1][2] == 0
    assert routes[5][-1][3] == pytest.approx(np.pi / 2)
    assert routes[6][-1][4] == 0
    assert routes[7][-1][5] == 0
    assert routes[2][-1][6] == 0
    assert routes[1][-1][1] == 0


def test_setup_defers_live_state_without_fabricating_operator_checks():
    cfg = yaml.safe_load((ROOT / 'config/arm_zero_calibration.yaml').read_text())
    original = copy.deepcopy(cfg)
    auto = configure(cfg, 'right', 'detached')
    assert cfg == original
    _, report = prepare(auto, 'right', 'detached')
    assert report['requires_live_snapshot'] and not report['ready']
    assert auto['operation']['supported_on_disable'] is False
    assert not any('all seven' in b or 'start_raw' in b for b in report['blockers'])


def test_live_routes_use_current_coordinate_epoch():
    cfg = configure(yaml.safe_load((ROOT / 'config/arm_zero_calibration.yaml').read_text()), 'right', 'detached')
    target = [.1] * 7
    other = [.2] * 7
    delta = [.03] * 7
    result = resolve(cfg, 'right', target, other, delta)
    assert result['operation']['start_raw'] == target
    assert result['sides']['right']['other_arm_q'] == other
    assert result['operation']['initial_delta'] == delta
    assert cfg['operation']['start_raw'] is None


def test_live_snapshot_reads_both_disabled_arms_without_enable(monkeypatch):
    from argparse import Namespace
    from arm_zero_runtime import resolve_inputs
    import arm_zero_runtime as runtime
    from arm_zero_calibrate import Sample
    events, interfaces = [], []
    class Recording:
        def __init__(self, metadata):
            pass
        def __call__(self, kind, **data):
            events.append((kind, data))
        def close(self):
            pass
    class ReadOnlyBus:
        def __init__(self, interface, deadline):
            interfaces.append(interface)
            self.count = 0
        def sleep(self, duration):
            pass
        def read(self):
            self.count += 1
            return Sample(np.zeros(7), np.full(7, .1 if self.count == 1 else .010989), np.zeros(7), np.zeros(7), np.ones(7), np.zeros(7, dtype=bool))
        def close(self):
            pass
    cfg = configure(yaml.safe_load((ROOT / 'config/arm_zero_calibration.yaml').read_text()), 'right', 'detached')
    monkeypatch.setenv('ROS_DOMAIN_ID', '50')
    monkeypatch.setattr(runtime, 'Recorder', Recording)
    monkeypatch.setattr(runtime, 'CanBus', ReadOnlyBus)
    model = Namespace(source_hash='model')
    args = Namespace(side='right', hand='detached', candidate=None, history=None, command='measure')
    actual = resolve_inputs(args, cfg, model)
    assert interfaces == ['can0', 'can1']
    assert sum(kind == 'startup_snapshot' for kind, _ in events) == 2
    assert sum(kind == 'startup_state' for kind, _ in events) >= 60
    assert actual['operation']['start_raw'] == [0.] * 7
    assert list(actual['operation']['measure_routes']) == list(ORDER)


def test_inspection_only_reads_existing_zero_never_commands_motor(monkeypatch):
    from argparse import Namespace
    import arm_zero_runtime as runtime
    from arm_zero_calibrate import Sample
    events = []
    class Recording:
        def __init__(self, metadata):
            self.metadata = metadata
        def __call__(self, kind, **data):
            events.append(dict(self.metadata, kind=kind, **data))
        def close(self):
            pass
    class ReadOnlyBus:
        def __init__(self, interface, deadline):
            assert interface == 'can0'
        def read(self):
            return Sample(np.full(7, .01), np.zeros(7), np.zeros(7), np.zeros(7), np.ones(7), np.zeros(7, dtype=bool))
        def close(self):
            pass
        def sleep(self, duration):
            pass
        def command(self, *args):
            pytest.fail('inspection must not command')
        def enable(self):
            pytest.fail('inspection must not enable')
        def zero(self, joint):
            pytest.fail('inspection must not save zeros')
    monkeypatch.setattr(runtime, 'Recorder', Recording)
    monkeypatch.setattr(runtime, 'CanBus', ReadOnlyBus)
    monkeypatch.setattr(runtime.time, 'sleep', lambda dt: None)
    cfg = configure(yaml.safe_load((ROOT / 'config/arm_zero_calibration.yaml').read_text()), 'right', 'detached')
    runtime.inspect_zero(Namespace(side='right', hand='detached'), cfg)
    result = next(e for e in events if e['kind'] == 'zero_inspection')
    assert result['count'] == 30
    assert np.allclose(result['mean'], .01)
    assert result['zero_written'] is False
    assert result['motor_command_sent'] is False


@pytest.mark.parametrize('mode', ['settles', 'enabled', 'moving'])
def test_inspection_stability_records_evidence_and_distinguishes_failure(mode):
    from arm_zero_runtime import observe_existing_zero
    from arm_zero_calibrate import Abort, Sample
    events = []
    class Bus:
        def __init__(self):
            self.index = 0
        def read(self):
            self.index += 1
            moving = mode == 'moving' or (mode == 'settles' and self.index <= 8)
            q = np.full(7, self.index * .01 if moving else .1)
            dq = np.full(7, .1 if moving else .010989010989)
            enabled = np.full(7, mode == 'enabled', dtype=bool)
            return Sample(q, dq, np.zeros(7), np.zeros(7), np.full(7, self.index), enabled)
        def sleep(self, duration):
            pass
    def event(kind, **data):
        events.append((kind, data))
    if mode == 'settles':
        samples = observe_existing_zero(Bus(), event, .0125, .004)
        assert samples.shape == (30, 7)
        assert np.allclose(samples, .1)
        assert len(events) > 30
    else:
        with pytest.raises(Abort, match='enabled' if mode == 'enabled' else 'stationary timeout'):
            observe_existing_zero(Bus(), event, .0125, .004)
    assert events[0][0] == 'zero_inspection_state'
    assert 'enabled' in events[0][1]


def test_final_bag_flush_is_deferred_until_recorder_close(monkeypatch):
    from types import SimpleNamespace
    import sys
    import arm_zero_runtime as runtime
    calls = []
    recorder = runtime.Recorder.__new__(runtime.Recorder)
    recorder.metadata = {}
    recorder.String = lambda: SimpleNamespace(data='')
    recorder.recording = lambda: True
    recorder.pub = SimpleNamespace(publish=lambda msg: calls.append('publish'),
                                   wait_for_all_acked=lambda duration: calls.append(('ack', duration.nanoseconds)) or True)
    recorder.node = SimpleNamespace(destroy_node=lambda: calls.append('destroy'))
    recorder.rclpy = SimpleNamespace(spin_once=lambda *a, **k: None, shutdown=lambda: calls.append('shutdown'))
    # Exercise actual publisher lifecycle without DDS or hardware.
    class Duration:
        def __init__(self, seconds=0, nanoseconds=0):
            self.nanoseconds = seconds * 1_000_000_000 + nanoseconds
    monkeypatch.setitem(sys.modules, 'rclpy.duration', SimpleNamespace(Duration=Duration))
    recorder('finished')
    assert calls == ['publish']
    recorder.close()
    assert calls == ['publish', ('ack', 1_000_000_000), 'destroy', 'shutdown']


def _live_cfg(q4=0.):
    cfg = configure(yaml.safe_load((ROOT / 'config/arm_zero_calibration.yaml').read_text()), 'right', 'detached')
    raw = np.zeros(7)
    raw[3] = q4
    return resolve(cfg, 'right', raw, np.zeros(7), np.zeros(7))


def test_pre_correction_offset_is_not_rejected_by_path_precheck():
    # 실기 로그: 차렷에서 j4 가 -0.0109 rad(한계 0 아래)로 읽혔다. 런타임은 max_offset 여유를 주므로 사전 검사도 같아야 한다.
    _, report = prepare(_live_cfg(-0.010872), 'right', 'detached', command='measure')
    assert not [b for b in report['blockers'] if b.startswith('path:')], report['blockers']


def test_verify_checks_only_verify_route_not_every_stop_route():
    full = prepare(_live_cfg(), 'right', 'detached')[1]['path_samples_checked']
    verify = prepare(_live_cfg(), 'right', 'detached', command='verify')[1]['path_samples_checked']
    assert 0 < verify < full / 2


class SagBus:
    """영점 제어(FF 0)에서 정적 처짐 q = target - g/kp 를 흉내 낸다(정지마찰·동특성은 아님)."""
    def __init__(self, m, g):
        self.m, self.g, self.t = m, np.asarray(g, float), 0.
        self.q, self.seq, self.commands = np.zeros(7), 0, []

    def now(self): return self.t
    def sleep(self, dt): self.t += dt

    def read(self):
        self.seq += 1
        return self.m.Sample(self.q.copy(), np.zeros(7), np.zeros(7), np.full(7, self.t), np.full(7, self.seq), np.full(7, True))

    def command(self, target, gravity, kp, kd):
        self.commands.append(target.copy())
        self.q = target - self.g / kp


def _sag_runner(g):
    from arm_zero_auto import AutoRunner
    import arm_zero_calibrate as m
    bus = SagBus(m, g)
    settings = m.Settings(torque_cap=(40.,) * 7, contact_torque=(5.,) * 7)
    runner = AutoRunner(bus, settings, m.Coordinates(np.zeros(7)), lambda q: np.asarray(g, float), lambda q: None, lambda *a, **k: None)
    runner.target = bus.q.copy()
    return m, bus, runner


def test_zero_ff_preparation_move_settles_at_gravity_droop():
    # j4 90°: 중력 3.54 N·m, Kp 52 → 정적 오차 0.068 rad. 이동 종료 판정이 이 처짐을 정상으로 봐야 한다.
    g = [0, 0, 0, 3.5423, 0, 0, 0]
    _, bus, runner = _sag_runner(g)
    goal = np.zeros(7)
    goal[3] = np.pi / 2
    state = runner.move(goal)
    assert state.q[3] == pytest.approx(goal[3] - 3.5423 / 52, abs=1e-6)


def test_zero_ff_move_still_rejects_unexplained_error():
    # 모델이 예측한 처짐보다 훨씬 크게 못 따라가면(막힘) 여전히 중단.
    g = [0, 0, 0, 3.5423, 0, 0, 0]
    m, bus, runner = _sag_runner(g)
    bus.command = lambda target, gravity, kp, kd: setattr(bus, 'q', target - np.asarray(g) / kp - np.eye(7)[3] * .05)
    goal = np.zeros(7)
    goal[3] = np.pi / 2
    with pytest.raises(m.Abort):
        runner.move(goal)
