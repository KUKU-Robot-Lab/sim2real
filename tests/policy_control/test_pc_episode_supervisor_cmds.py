"""sup_v0: ROS 없이 명령 통로와 실행 순서를 검증한다."""
import dataclasses
import json
from pathlib import Path

import pytest
from policy_control import episode_spec as S
from policy_control.episode_failure import F
from policy_control.episode_fake import FakeExecutor
from policy_control.episode_runner import READY, RUNNING, STOPPED, EpisodeManager
from policy_control.episode_runner_node import Approvals

REPO = Path(__file__).resolve().parents[2]


def manager():
    ep = S.load(REPO / 'config/episodes/pick_place_right.yaml')
    ex = FakeExecutor(ep)
    return EpisodeManager(ep, ex, approve=lambda *_: True)


def cmd(m, verb, args=None, cid=None, **kw):
    return m.command({'what': 'cmd:' + verb, 'cmd_id': cid or str(len(m.cmd_ack)),
                      'iface': 'sup_v0', 'args': args or {}}, **kw)


def test_boundary_busy_dedup_and_ack_limit():
    m = manager()
    assert m.view()['boundary']
    assert not cmd(m, 'skip', {'node_id': m.node.id}, cid='busy', busy=True)['ok']
    assert m.cmd_ack[-1]['reason'] == 'busy'
    assert cmd(m, 'stop', cid='busy') is None
    m.status = RUNNING
    assert cmd(m, 'stop')['reason'] == 'busy'
    m.status = READY
    for i in range(25):
        cmd(m, 'unknown', cid=f'x{i}')
    assert len(m.view()['cmd_ack']) == 20


def test_insert_executes_before_current_then_resumes_and_is_visible():
    m = manager()
    original = m.node.id
    assert cmd(m, 'insert', {'after': original, 'spec': {'id': 'scene', 'type': 'snapshot'}})['ok']
    inserted = m.node.id
    assert inserted.startswith('sup_') and inserted != 'scene'
    assert inserted in str(m.view()['flow'])
    m.step()
    assert m.node.id == original and m.runs[inserted]['state'] == 'done'
    m.step()
    assert [c[1] for c in m.executor.calls] == [inserted, original]


@pytest.mark.parametrize('spec', [{'type': 'terminal'}, {'type': 'hand', 'action': 'open', 'sides': ['right']},
                                {'type': 'policy', 'name': 'missing'}, {'type': 'trajectory', 'name': 'missing'}])
def test_insert_rejects_invalid_specs(spec):
    m = manager()
    assert not cmd(m, 'insert', {'spec': spec})['ok']
    assert m.index == 0


def test_skip_checks_current_and_terminal_and_logs():
    m = manager()
    rows = []
    m.log = rows.append
    assert not cmd(m, 'skip', {'node_id': 'wrong'})['ok']
    name = m.node.id
    assert cmd(m, 'skip', {'node_id': name, 'reason': 'unneeded'})['ok']
    assert m.runs[name]['state'] == 'skipped' and not m.executor.calls
    assert any(r['event'] == 'skipped' for r in rows)
    assert any(r['event'] == 'cmd' for r in rows)
    m.index = len(m.ep.nodes) - 1
    assert not cmd(m, 'skip', {'node_id': m.node.id})['ok']


def test_stop_uses_safe_stop():
    m = manager()
    assert cmd(m, 'stop', {'reason': 'supervisor stop'})['ok']
    assert m.status == STOPPED and m.executor.stopped == ['supervisor stop']


def test_retry_failed_trajectory_returns_to_original_position_and_is_capped():
    m = manager()
    failed = m.node.id
    m.executor.inject[failed] = [F.TARGET_NOT_FOUND]
    m.step()
    assert m.last['node'] == failed
    m.index = 1
    m.status = READY
    assert cmd(m, 'retry', {'node_id': failed})['ok']
    m.step()
    assert m.index == 1 and m.node.id == 'scene'
    assert not cmd(m, 'retry', {'node_id': failed})['ok']
    assert not cmd(m, 'retry', {'node_id': 'scene'})['ok']


def test_refresh_updates_world_and_forwards_timeout_and_force():
    m = manager()
    seen = []
    class Ex(FakeExecutor):
        def snapshot(self, node, world, timeout_s=None):
            seen.append((node.objects, timeout_s))
            return super().snapshot(node, world)
        def refresh_holders(self, force=False):
            seen.append(force)
            from policy_control.episode_failure import NodeResult
            return NodeResult('completed', '')
    m.executor = Ex(m.ep)
    assert cmd(m, 'refresh_objects', {'objects': ['CUP'], 'timeout_s': 12.0})['ok']
    assert m.world.objects['CUP']['pose']
    assert seen[0] == (('CUP',), 12.0)
    assert cmd(m, 'refresh_holders', {'force': True})['ok'] and seen[-1] is True
    assert not cmd(m, 'refresh_objects', {'objects': ['missing']})['ok']
    assert not cmd(m, 'refresh_objects', {'timeout_s': -1})['ok']


def test_command_lines_filter_and_approvals_never_consume_commands(tmp_path):
    from policy_control.episode_runner_node import supervisor_commands
    base = {'what': 'cmd:stop', 'cmd_id': 'a', 'iface': 'sup_v0', 't': 11, 'runner': 'r', 'args': {}}
    rows = [base, base, {**base, 'cmd_id': 'old', 't': 9}, {**base, 'cmd_id': 'other', 'runner': 'other'},
            {**base, 'cmd_id': 'bad', 'iface': 'other'}, {'what': 'scene', 't': 12, 'runner': 'r'}]
    lines = ['broken', '[]', '{"t": "oops"}'] + [json.dumps(r) for r in rows]
    assert [r['cmd_id'] for r in supervisor_commands(lines, since=10, runner='r', seen=set())] == ['a']
    assert supervisor_commands(lines, since=10, runner='r', seen={'a'}) == []
    path = tmp_path / 'approvals'
    path.write_text('\n'.join(lines))
    approvals = Approvals(path, 10, 'r')
    assert not approvals.take('cmd:stop')
    assert approvals.take('scene')


def test_replay_spec_accepts_existing_single_arm_npz_and_rejects_missing_or_two_sides(tmp_path):
    import yaml
    raw = yaml.safe_load((REPO / 'config/episodes/pick_place_right.yaml').read_text())
    # parser validates existence; replay tool validates contents before publishing.
    path = tmp_path / 'path.npz'
    path.touch()
    from unittest.mock import patch

    from policy_control import _paths
    raw['trajectories']['teach'] = {'kind': 'replay', 'sides': ['right'], 'npz': 'path.npz'}
    with patch.object(_paths, 'SIM2REAL', tmp_path):
        ep = S.parse(raw)
        assert ep.trajectories['teach'].npz == 'path.npz'
        for change in ({'npz': 'missing.npz'}, {'sides': ['right', 'left']}, {'npz': '../path.npz'}):
            bad = {**raw, 'trajectories': {**raw['trajectories'], 'teach': {**raw['trajectories']['teach'], **change}}}
            with pytest.raises(S.EpisodeSpecError):
                S.parse(bad)


def test_replay_runs_start_check_before_arm_replay_and_reports_hand(tmp_path):
    import threading
    from types import SimpleNamespace

    from policy_control.episode_ros import RosExecutor
    from policy_control.episode_world import World
    calls = []
    ex = SimpleNamespace(_wait_pd_tracking=lambda _: True, cancel=threading.Event(), contract='c.json',
                         _run=lambda argv: (calls.append(argv) or (0, '')))
    traj = S.Trajectory('teach', 'replay', ('right',), npz='logs/place_teach/holder1.npz')
    result = RosExecutor.run_trajectory(ex, S.Node('teach', 'trajectory', 'teach'), traj, World())
    assert result.status == 'completed' and 'hand not replayed' in result.reason
    assert [Path(a[1]).name for a in calls] == ['check_path_start.py', 'replay_to_pd.py']
    assert '--tol' not in calls[0] and '--execute' in calls[1]
    calls.clear()
    ex._run = lambda argv: (calls.append(argv) or (1, 'gap'))
    result = RosExecutor.run_trajectory(ex, S.Node('teach', 'trajectory', 'teach'), traj, World())
    assert result.status == 'refused' and len(calls) == 1


def test_teach_start_check_reads_schema_without_fabricated_contract(tmp_path):
    import importlib.util

    import numpy as np
    path = REPO / 'deploy/policy_control/tools/check_path_start.py'
    spec = importlib.util.spec_from_file_location('supervisor_check_path', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = [f'r_aj_{i}' for i in range(1, 8)]
    data = {'meta_schema': np.array('s2r_replay_v1'), 'joint_names': np.array(names),
            'arm_target': np.ones((2, 7)), 'meta_step_dt': np.array(.02)}
    assert mod.replay_start(data) == (names, [1.] * 7)
    assert mod.verdict(dict.fromkeys(names, 1.01), *mod.replay_start(data), .05) == []
    assert mod.verdict(dict.fromkeys(names, 1.06), *mod.replay_start(data), .05)
    for bad in ({'arm_target': np.ones((2, 6))}, {'arm_target': np.full((2, 7), np.nan)},
                {'joint_names': np.array(['r_aj_1'] * 7)}, {'meta_step_dt': np.array(-.02)}):
        with pytest.raises(ValueError):
            mod.replay_start({**data, **bad})


def test_retry_current_failure_executes_once_and_keeps_approval():
    m = manager()
    failed = m.node.id
    m.executor.inject[failed] = [F.TARGET_NOT_FOUND]
    m.step()
    assert m.status == READY and m.pending is not None
    assert cmd(m, 'retry', {'node_id': failed})['ok']
    m.approve = lambda *_: False
    before = len(m.executor.calls)
    m.step()
    assert len(m.executor.calls) == before and m.node.id == failed
    m.approve = lambda *_: True
    m.step()
    assert m.node.id == 'scene' and len(m.executor.calls) == before + 1
    assert m.pending is None


def test_retry_continued_policy_returns_to_home_and_reuses_definition():
    m = manager()
    m.ep = dataclasses.replace(m.ep, continue_on_fail=('place',), failure_policy={'default': 0})
    m.executor.ep = m.ep
    m.executor.inject['place_cup'] = [F.PLACE_FAILED]
    for _ in range(4):
        m.step()
    assert m.node.id == 'go_home_end' and m.place_failed_continued == ['place_cup']
    node = m.ep.nodes[m.ep.node_index('place_cup')]
    assert cmd(m, 'retry', {'node_id': 'place_cup'})['ok']
    assert m.node is node
    m.step()
    assert m.node.id == 'go_home_end'
    assert m.runs['place_cup']['n'] == 2
    assert not cmd(m, 'retry', {'node_id': 'place_cup'})['ok']


def test_failure_and_stopped_are_not_ready_boundaries():
    from policy_control.episode_runner import FAILURE
    m = manager()
    for status in (FAILURE, STOPPED):
        m.status = status
        assert not m.view()['boundary']
        assert cmd(m, 'retry', {'node_id': m.node.id}, cid=status)['reason'] == 'not ready'


def test_approval_commands_remain_deduplicated_across_manager_resets(tmp_path):
    path = tmp_path / 'approvals'
    row = {'what': 'cmd:stop', 'cmd_id': 'a', 'iface': 'sup_v0', 't': 11, 'runner': 'r'}
    path.write_text(json.dumps(row) + '\n')
    approvals = Approvals(path, 10, 'r')
    assert approvals.commands() == [row]
    assert approvals.commands() == []


def test_insert_after_must_match_and_existing_roles_can_be_inserted():
    m = manager()
    assert not cmd(m, 'insert', {'after': 'wrong', 'spec': {'type': 'snapshot'}})['ok']
    m.step()
    m.step()
    spec = {'type': 'policy', 'name': 'rh_aglt_r', 'target_object': 'CUP'}
    assert cmd(m, 'insert', {'spec': spec})['ok']
    inserted = m.node.id
    m.step()
    assert ('policy', inserted, ('rh_aglt_r',)) in m.executor.calls
    assert m.node.id == 'pick_cup'


def test_insert_updates_executor_episode_for_safe_stop_sides():
    from policy_control.episode_ros import RosExecutor
    m = manager()
    left = S.Binding(role='left', kind='aglt', side='left', policy=None)
    m.ep = dataclasses.replace(m.ep, policies={**m.ep.policies, 'left': left})
    m.step()
    m.step()
    assert cmd(m, 'insert', {'spec': {'type': 'policy', 'name': 'left', 'target_object': 'CUP'}})['ok']
    assert m.executor.ep is m.ep
    assert RosExecutor.sides.fget(m.executor) == ('right', 'left')


def test_executor_exception_is_acknowledged_and_safe_stopped():
    m = manager()
    def broken(**_kw):
        raise RuntimeError('holder read failed')
    m.executor.refresh_holders = broken
    ack = cmd(m, 'refresh_holders')
    assert not ack['ok'] and 'holder read failed' in ack['reason']
    assert m.status == STOPPED and m.executor.stopped


def test_force_reload_reads_yaml_even_when_holders_are_present(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from policy_control import episode_ros as R

    path = tmp_path / 'holders.yaml'
    path.write_text('holders:\n  cup_holder_1:\n    position: [1, 2, 3]\n    yaw_rad: 0.5\n')
    monkeypatch.setattr(R._paths, 'SIM2REAL', tmp_path)
    monkeypatch.setattr(R.time, 'sleep', lambda _: None)
    ex = SimpleNamespace(ep=SimpleNamespace(holder_poses='holders.yaml'), holders={1: 'old'}, publish_holders=list)
    assert R.RosExecutor.refresh_holders(ex).status == 'completed'
    assert ex.holders == {1: 'old'}
    assert R.RosExecutor.refresh_holders(ex, force=True).status == 'completed'
    assert ex.holders == {1: ((1., 2., 3.), .5)}
