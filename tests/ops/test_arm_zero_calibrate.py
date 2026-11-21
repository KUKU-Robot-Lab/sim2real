"""No ROS, socket or robot access: exercise calibration against a deterministic bus."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

OPS = Path(__file__).resolve().parents[2] / 'scripts/ops'
sys.path.insert(0, str(OPS))


def module():
    spec = importlib.util.find_spec('arm_zero_calibrate')
    assert spec is not None, 'calibration implementation is missing'
    import arm_zero_calibrate
    return arm_zero_calibrate


def test_references_and_coordinate_transform():
    m = module()
    assert m.reference('right', 1) == pytest.approx(np.deg2rad(-80))
    assert m.reference('left', 2) == pytest.approx(np.deg2rad(10))
    delta = np.deg2rad([1, 2, 3, 4, 5, 6, 7])
    state = m.Coordinates(delta)
    assert state.raw(np.zeros(7)) == pytest.approx(delta)
    state.saved(3)
    delta[2] = 0
    assert state.raw(np.zeros(7)) == pytest.approx(delta)


def test_repeatability_and_nonfinite():
    m = module()
    assert m.candidate_delta([1.60, 1.601, 1.602], np.pi / 2) == pytest.approx(1.601 - np.pi / 2)
    for hits in ([1, 1, 1.02], [1, 1], [1, 1, float('nan')]):
        with pytest.raises(m.Abort):
            m.candidate_delta(hits, 0)


class Bus:
    def __init__(self, m, *, stuck=False, stale=False, drift=False, fail_write=False):
        self.m = m
        self.t = 0.
        self.q = np.zeros(7)
        self.q[3] = .12
        self.stop = 0.02
        self.stuck, self.stale, self.drift, self.fail_write = stuck, stale, drift, fail_write
        self.writes = []
        self.commands = []
        self.disabled = False
        self.seq = 0
        self.disabled_joint = None

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt

    def read(self):
        self.seq += 1
        if self.disabled_joint is not None and self.drift:
            self.q[3] += .01
        enabled = np.full(7, not self.disabled)
        if self.disabled_joint is not None:
            enabled[self.disabled_joint - 1] = False
        return self.m.Sample(self.q.copy(), np.zeros(7), self.torque.copy(),
                             np.full(7, self.t - (1 if self.stale else 0)),
                             np.full(7, self.seq), enabled)

    torque = np.zeros(7)

    def command(self, target, gravity, kp, kd):
        old = self.q.copy()
        self.commands.append(target.copy())
        if not self.stuck:
            self.q = target.copy()
            self.q[3] = max(self.stop, self.q[3])
        self.torque = np.zeros(7)
        if self.stuck or target[3] < self.stop:
            self.torque[3] = -3.
        self.last_move = self.q - old

    def disable(self, joint):
        self.disabled_joint = joint

    def zero(self, joint):
        self.writes.append(joint)
        if self.fail_write:
            raise self.m.Abort('write failure')
        self.q[joint - 1] = 0

    def invalidate(self):
        self.commands.clear()


def engine(**kwargs):
    m = module()
    bus = Bus(m, **kwargs)
    settings = m.Settings(torque_cap=(10.,) * 7, contact_torque=(2.,) * 7,
                          max_travel=0.3, timeout=15, min_travel=.03, backoff=.06)
    events = []
    runner = m.Runner(bus, settings, m.Coordinates(np.zeros(7)),
                      lambda q: np.zeros(7), lambda q: None,
                      lambda kind, **data: events.append((kind, data)))
    runner.target = bus.q.copy()
    return m, bus, runner, events


def test_measure_three_approaches_never_writes_zero():
    _, bus, runner, events = engine()
    delta = runner.measure_joint('right', 4)
    assert delta == pytest.approx(.02)
    assert not bus.writes
    assert len([e for e in events if e[0] == 'stop']) == 4  # locating pass + three measured approaches
    assert all(len(q) == 7 for q in bus.commands)


@pytest.mark.parametrize('kwargs,reason', [({'stuck': True}, 'early'), ({'stale': True}, 'stale')])
def test_rejects_early_stop_and_stale(kwargs, reason):
    m, bus, runner, _ = engine(**kwargs)
    with pytest.raises(m.Abort, match=reason):
        runner.measure_joint('right', 4)
    assert bus.writes == []


def test_stop_not_reached():
    m, bus, runner, _ = engine()
    bus.stop = -10
    with pytest.raises(m.Abort, match='travel|timeout'):
        runner.measure_joint('right', 4)
    assert not bus.writes


def test_apply_only_selected_joint_and_clears_coordinate_cache():
    _, bus, runner, events = engine()
    runner.coordinates.delta[3] = .12
    runner.apply_joint(4)
    assert bus.writes == [4]
    assert runner.coordinates.delta[3] == 0
    assert runner.target is None
    assert events[-1][0] == 'zero_verified'


@pytest.mark.parametrize('kwargs,reason,writes', [
    ({'drift': True}, 'disable', []),
    ({'fail_write': True}, 'write failure', [4]),
])
def test_apply_failure_no_retry_or_home(kwargs, reason, writes):
    m, bus, runner, events = engine(**kwargs)
    runner.coordinates.delta[3] = .12
    with pytest.raises(m.Abort, match=reason):
        runner.apply_joint(4)
    assert bus.writes == writes
    assert not any(e[0] == 'zero_verified' for e in events)


@pytest.mark.parametrize('command', ['plan', 'measure', 'apply', 'verify'])
@pytest.mark.parametrize('execute', [False, True])
def test_default_plan_never_imports_hardware(monkeypatch, capsys, command, execute):
    m = module()
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ('openarm_can', 'rclpy', 'arm_zero_runtime'):
            raise AssertionError('plan touched hardware runtime')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert m.main([command, '--side', 'right', '--hand', 'detached'] + (['--execute'] if execute else [])) == 2
    assert 'ready' in capsys.readouterr().out


def test_detached_model_keeps_l7_removes_hand_both_sides():
    module()
    from arm_zero_model import prepare
    import yaml
    cfg = yaml.safe_load((OPS.parents[1] / 'config/arm_zero_calibration.yaml').read_text())
    model, report = prepare(cfg, 'right', 'detached')
    assert 'r_al_7' in model.links and 'l_al_7' in model.links
    assert 'r_hl_base' not in model.links and 'l_hl_base' not in model.links
    assert not report['ready']
    assert model.gravity(np.zeros(7)).shape == (7,)
    assert np.isfinite(model.flange(np.zeros(7))).all()


def test_wrong_removal_boundary_is_rejected():
    m = module()
    from arm_zero_model import prepare
    import yaml
    cfg = yaml.safe_load((OPS.parents[1] / 'config/arm_zero_calibration.yaml').read_text())
    cfg['model']['remove_at'] = ['r_aj_7', 'l_hj_mount']
    with pytest.raises(m.Abort, match='boundary'):
        prepare(cfg, 'right', 'detached')


def test_history_rejects_interrupted_write_and_repeat_apply():
    m = module()
    from arm_zero_runtime import validate_artifacts
    c = {'version': 1, 'side': 'right', 'session': 's', 'model_hash': 'm',
         'delta': [.01] * 7, 'reviewed_no_contact': True, 'candidate_id': 'c',
         'hits': {str(j): [m.reference('right', j) + .01] * 3 for j in range(1, 8)}}
    h = {'candidate_id': 'c', 'saved_joints': [3], 'unknown_joints': []}
    result = validate_artifacts(c, h, 'right', 's', 'm', 4)
    assert result.delta[2] == 0 and result.delta[3] == .01
    with pytest.raises(m.Abort, match='already'):
        validate_artifacts(c, h, 'right', 's', 'm', 3)
    h['unknown_joints'] = [4]
    with pytest.raises(m.Abort, match='unknown'):
        validate_artifacts(c, h, 'right', 's', 'm', 4)


def test_history_refuses_wrong_side_or_unreviewed_contact():
    m = module()
    from arm_zero_runtime import validate_artifacts
    with pytest.raises(m.Abort):
        validate_artifacts({'version': 1, 'side': 'left'}, {}, 'right', 's', 'm', 1)


def test_bag_reduction_marks_attempt_without_verification_unknown():
    module()
    from arm_zero_runtime import reduce_events
    events = [{'kind': 'candidate', 'candidate_id': 'c', 'side': 'right', 'session': 's',
               'model_hash': 'm', 'delta': [0.] * 7, 'hits': {}, 'version': 1},
              {'kind': 'zero_attempt', 'candidate_id': 'c', 'joint': 3}]
    c, h = reduce_events(events)
    assert h['unknown_joints'] == [3] and h['saved_joints'] == []
    events += [{'kind': 'zero_verified', 'candidate_id': 'c', 'joint': 3}]
    _, h = reduce_events(events)
    assert h['unknown_joints'] == [] and h['saved_joints'] == [3]
    assert not c['reviewed_no_contact']


def test_zero_verification_uses_new_epoch():
    _, bus, runner, _ = engine()
    runner.coordinates.delta[3] = .12
    def nonnegative_elbow(q):
        if q[3] < -.001:
            raise runner_abort('old coordinate epoch')
    runner_abort = module().Abort
    runner.guard = nonnegative_elbow
    runner.apply_joint(4)
    assert bus.writes == [4]


def test_contact_setpoint_does_not_cross_geometry_limit():
    m, bus, runner, _ = engine()
    bus.stop = 0
    def limit(q):
        if q[3] < 0:
            raise m.Abort('geometry limit')
    runner.guard = limit
    assert runner.measure_joint('right', 4) == pytest.approx(0)


def test_path_precheck_keeps_sending_hold():
    m, bus, runner, _ = engine()
    def costly_guard(q):
        bus.t += .001
    runner.guard = costly_guard
    goal = bus.q.copy()
    goal[0] = .05
    runner.move(goal)
    assert len(bus.commands) > 100


def test_move_wall_clock_budget():
    m, bus, runner, _ = engine()
    from dataclasses import replace
    runner.s = replace(runner.s, timeout=1.)
    def expensive_guard(q):
        bus.t += .009
    runner.guard = expensive_guard
    goal = bus.q.copy()
    goal[0] = .03
    with pytest.raises(m.Abort, match='timeout'):
        runner.move(goal)


def test_nonselected_motor_disabled_aborts():
    m, bus, runner, _ = engine()
    bus.disabled = True
    with pytest.raises(m.Abort, match='disabled'):
        runner.tick(bus.q, disabled=3)


def test_arm_setup_does_not_require_head_pose_but_requires_routes():
    module()
    from arm_zero_model import prepare
    import yaml
    cfg = yaml.safe_load((OPS.parents[1] / 'config/arm_zero_calibration.yaml').read_text())
    cfg['operation']['measure_routes'] = {1: [[0] * 7]}
    cfg['operation']['verify_route'] = [[0] * 7]
    _, report = prepare(cfg, 'right', 'detached')
    assert not any('head_j_pan' in b for b in report['blockers'])
    assert report['scope'] == 'arms_and_body'
    assert any('all seven' in b for b in report['blockers'])
    assert any('multiple' in b for b in report['blockers'])


def test_zero_command_accepted_but_coordinate_unchanged_is_unknown():
    m, bus, runner, events = engine()
    runner.coordinates.delta[3] = .12
    bus.zero = lambda joint: bus.writes.append(joint)
    with pytest.raises(m.Abort, match='not verified'):
        runner.apply_joint(4)
    assert bus.writes == [4]
    assert runner.unknown_epoch and runner.target is None
    assert any(kind == 'zero_attempt' for kind, _ in events)
    assert not any(kind == 'zero_verified' for kind, _ in events)
    with pytest.raises(m.Abort, match='unknown'):
        runner.read()


def test_cached_stationary_feedback_is_not_fresh():
    m, bus, runner, _ = engine()
    snapshot = bus.read()
    bus.read = lambda: snapshot
    runner.read()
    with pytest.raises(m.Abort, match='reused'):
        runner.read()


def test_asset_manifest_collision_filters_respect_removed_hand():
    module()
    from arm_zero_model import prepare
    import yaml
    cfg = yaml.safe_load((OPS.parents[1] / 'config/arm_zero_calibration.yaml').read_text())
    detached, report = prepare(cfg, 'right', 'detached')
    assert frozenset(('r_al_5', 'r_al_7')) in detached.asset_exclusions
    assert frozenset(('l_al_5', 'l_al_7')) in detached.asset_exclusions
    assert len(detached.asset_exclusions) == 2
    attached, _ = prepare(cfg, 'right', 'attached')
    assert len(attached.asset_exclusions) > 2
    assert frozenset(('r_hl_index_1', 'r_hl_thumb_sensor')) not in attached.asset_exclusions
    assert report['collision_filter_source'].endswith('_manifest.yaml')



def test_calibration_scope_excludes_head_and_preserves_arms_body():
    module()
    from arm_zero_model import prepare
    import yaml
    cfg = yaml.safe_load((OPS.parents[1] / 'config/arm_zero_calibration.yaml').read_text())
    for hand in ('detached', 'attached'):
        model, report = prepare(cfg, 'right', hand)
        assert report['scope'] == 'arms_and_body'
        assert not any(name.startswith('head_') for name in model.links)
        assert not any(j.attrib['name'].startswith('head_') for j in model.joints)
        assert 'body_link' in model.links
        assert all(f'{side}_al_{i}' in model.links for side in 'rl' for i in range(8))
        assert not hasattr(model, 'head_envelopes')


def test_move_has_smooth_start_and_stop_with_speed_cap():
    _, bus, runner, _ = engine()
    goal = bus.q.copy()
    goal[0] += .05
    runner.move(goal)
    positions = np.array(bus.commands)[:, 0]
    increments = np.diff(positions)
    moving = increments[increments > 1e-12]
    assert moving.size > 0
    assert np.max(moving) <= runner.s.speed * runner.s.dt * 1.001
    assert moving[0] < runner.s.speed * runner.s.dt * .01
    assert moving[-1] < runner.s.speed * runner.s.dt * .01
    assert positions[-1] == pytest.approx(goal[0])
