"""Arm-only manufacturer profile contract, with no physical CAN access."""
import numpy as np
import pytest

from arm_zero_auto import AutoRunner, measurement_routes
from arm_zero_calibrate import Coordinates, Settings, reference
from test_arm_zero_calibrate import Bus, module


@pytest.mark.parametrize('side', ['right', 'left'])
def test_no_invented_preposition_before_stop_search(side):
    routes = measurement_routes(np.zeros(7), side)
    assert routes[4] == [[0.] * 7]
    assert routes[3][-1][2] == 0  # search j3 from current pose, not near +90
    assert routes[3][-1][1] == pytest.approx(np.deg2rad(5 if side == 'right' else -5))
    assert routes[3][-1][3] == 0  # j4 located stop
    assert routes[5][-1][3] == pytest.approx(np.pi / 2)
    assert routes[5][-1][4] == 0
    assert routes[6][-1][4] == 0
    assert routes[7][-1][5] == pytest.approx(0)
    assert routes[2][-1][6] == 0
    assert routes[1][-1][1] == 0


def runner_at_stop(stuck=False):
    bus = Bus(module(), stuck=stuck)
    bus.q[3] = bus.stop
    events = []
    runner = AutoRunner(bus, Settings(torque_cap=(40.,) * 7, contact_torque=(2.,) * 7),
                        Coordinates(np.zeros(7)), lambda q: np.zeros(7), lambda q: None,
                        lambda kind, **data: events.append((kind, data)))
    return bus, runner, events


def test_j4_can_start_at_stop_but_three_repeats_must_move():
    bus, runner, events = runner_at_stop()
    assert runner.measure_joint('right', 4) == pytest.approx(bus.stop)
    assert len([e for e in events if e[0] == 'stop']) == 4
    assert not bus.writes
    assert bus.commands[0][3] < bus.stop  # first command searches negative
    bus, runner, _ = runner_at_stop(stuck=True)
    with pytest.raises(module().Abort):
        runner.measure_joint('right', 4)
    assert not bus.writes


def test_manufacturer_search_and_hold_gains():
    bus, runner, events = runner_at_stop()
    runner.tick(bus.q, search=3)
    command = events[-1][1]
    assert command['kp'] == [300, 300, 150, 45, 40, 40, 30]
    assert command['kd'] == [2.5, 2.5, 2.5, 1.2, .8, .8, .8]
    assert command['gravity'] == [0.] * 7
    runner.tick(bus.q, search=0)
    assert events[-1][1]['kp'][0] == 280
    assert events[-1][1]['kd'][0] == 2.1


def test_locating_contact_far_from_reference_is_rejected():
    bus, runner, _ = runner_at_stop(stuck=True)
    bus.q[3] = .5
    with pytest.raises(module().Abort, match='early'):
        runner.measure_joint('right', 4)


def test_manufacturer_j1_reference_not_overwritten_by_gripper_alias():
    assert reference('right', 1) == pytest.approx(np.deg2rad(-80))
    assert reference('left', 1) == pytest.approx(np.deg2rad(80))


@pytest.mark.parametrize('side', ['right', 'left'])
def test_complete_manufacturer_sequence_uses_only_seven_motors(monkeypatch, side):
    from argparse import Namespace
    import yaml
    import arm_zero_runtime as runtime
    from arm_zero_auto import configure, resolve
    from arm_zero_calibrate import ORDER, ROOT
    events = []
    directions = np.array([-1 if j == 4 or (side == 'right' and j in (1, 2)) else 1 for j in range(1, 8)])
    stops = np.array([reference(side, j) for j in range(1, 8)])
    class SevenMotorBus(Bus):
        def __init__(self, *args):
            super().__init__(module())
            self.q[:] = 0
            self.disabled = True
        def command(self, target, gravity, kp, kd):
            assert len(target) == len(kp) == len(kd) == 7
            self.commands.append(target.copy())
            self.q = np.where(directions > 0, np.minimum(target, stops), np.maximum(target, stops))
            self.torque = np.where(directions * (target - stops) > 0,
                                   directions * np.array([5.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]), 0.)
        def enable(self):
            self.disabled = False
        def close(self):
            self.disabled = True
    bus = SevenMotorBus()
    class Recorder:
        def __init__(self, metadata):
            pass
        def __call__(self, kind, **data):
            events.append((kind, data))
        def close(self):
            pass
    cfg = configure(yaml.safe_load((ROOT / 'config/arm_zero_calibration.yaml').read_text()), side, 'detached')
    cfg = resolve(cfg, side, [0.] * 7, [0.] * 7, [0.] * 7)
    model = Namespace(source_hash='test', load_bounds=lambda: None,
                      gravity=lambda q: np.zeros(7), guard=lambda q, **kwargs: None)
    monkeypatch.setenv('ROS_DOMAIN_ID', '50')
    monkeypatch.setattr(runtime, 'CanBus', lambda *args: bus)
    monkeypatch.setattr(runtime, 'Recorder', Recorder)
    runtime.execute(Namespace(side=side, hand='detached', command='measure',
                              joint=None, candidate=None, history=None), cfg, model)
    assert [data['joint'] for kind, data in events if kind == 'stage'] == list(ORDER)
    candidates = [data for kind, data in events if kind == 'candidate']
    assert len(candidates) == 1
    assert candidates[0]['delta'] == pytest.approx([0.] * 7)
    assert len([1 for kind, _ in events if kind == 'stop']) == 28
    assert not bus.writes
    assert bus.q == pytest.approx([0.] * 7, abs=1e-9)
    assert bus.disabled


def test_attached_verification_requires_completed_detached_baseline():
    from arm_zero_runtime import validate_attached_baseline
    route = [[0.] * 7, [0., 0., 0., 1.57, 0., 0., 0.]]
    events = [dict(kind='verification', hand='detached', candidate_id='c', run_id='r',
                   pose_index=i, model_target=pose) for i, pose in enumerate(route)]
    with pytest.raises(module().Abort, match='completed detached'):
        validate_attached_baseline(events, 'c', route)
    events.append(dict(kind='finished', command='verify', run_id='r'))
    validate_attached_baseline(events, 'c', route)
    events.append(dict(kind='zero_verified', candidate_id='c'))
    with pytest.raises(module().Abort, match='zero changed'):
        validate_attached_baseline(events, 'c', route)
