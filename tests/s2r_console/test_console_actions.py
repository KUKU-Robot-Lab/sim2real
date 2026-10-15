"""Management actions use fake processes, never a robot domain."""
import time

import pytest

from s2r_console.actions import ActionManager
from s2r_console.console import ConsoleError


def setup_actions(console, tiny_repo, commands):
    import yaml
    raw = {'name': 'actions', 'stages': [{'id': key, 'title': key} for key in commands],
           'run': commands}
    (tiny_repo / 'mission.yaml').write_text(yaml.safe_dump(raw))
    import shutil
    from pathlib import Path
    (tiny_repo / 'config').mkdir(exist_ok=True)
    shutil.copy(Path(__file__).resolve().parents[2] / 'config/objects.yaml', tiny_repo / 'config/objects.yaml')
    console.open('t_fake', operator='test')
    return ActionManager(console, settle_s=0)


def wait(manager):
    end = time.monotonic() + 5
    while manager.view().get('active') and time.monotonic() < end:
        time.sleep(.01)
    return manager.view()


def test_side_bundle_stops_at_failed_check(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {
        'hand_right': [{'argv': ['true']}], 'hand_check_right': [{'argv': ['false']}],
        'pd_load_right': [{'argv': ['true']}], 'pd_arm_right': [{'argv': ['true']}]})
    console.session.supervisor.spawn('drivers#0', stage='drivers', note='fake driver', argv=['sleep', '30'], background=True, manual=False)
    m.execute('connect_right', operator='test')
    result = wait(m)
    assert result['outcome'] == 'FAILED'
    assert result['stage'] == 'hand_check_right'
    assert not any(p['stage'] == 'pd_load_right' for p in console.session.supervisor.table())


def test_shutdown_without_drivers_keeps_session(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'shutdown': [{'argv': ['true']}]})
    m.execute('shutdown', operator='test')
    assert wait(m)['outcome'] == 'DONE'
    assert console.session is not None


def test_unknown_and_arbitrary_commands_rejected(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    with pytest.raises(ConsoleError):
        m.describe('shell', {'argv': ['touch', '/tmp/no']})
    with pytest.raises(ConsoleError):
        m.describe('preflight', {'argv': ['false']})


def test_duplicate_action_rejected(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['sleep', '.2']}]})
    m.execute('preflight', operator='test')
    with pytest.raises(ConsoleError):
        m.execute('preflight', operator='test')
    assert wait(m)['outcome'] == 'DONE'


def test_control_guard_explains_missing_live_pd(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'home_right': [{'argv': ['true']}]})
    item = next(a for a in m.catalog() if a['id'] == 'home_right')
    assert not item['enabled']
    assert 'PD' in ' '.join(item['reasons'])


def test_preemption_stops_before_hold_and_blocks_if_stop_fails(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    s = console.session
    s.supervisor.spawn('policy_test_right#0', stage='policy_test_right', note='policy', argv=['sleep', '30'], background=True, manual=False)
    monkeypatch.setattr(console, '_pd_sides', lambda _: ['right'])
    calls = []
    def trigger(session, service, side, **kw):
        calls.append(service)
        if service == 'episode/stop':
            raise ConsoleError('stop failed')
    monkeypatch.setattr(m, '_trigger', trigger)
    with pytest.raises(ConsoleError, match='stop failed'):
        m._preempt(s)
    assert calls == ['episode/stop']
    assert s.supervisor.is_alive('policy_test_right#0')


def test_preemption_order_and_pd_confirmation(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    s = console.session
    s.supervisor.spawn('policy_test_right#0', stage='policy_test_right', note='policy', argv=['sleep', '30'], background=True, manual=False)
    monkeypatch.setattr(console, '_pd_sides', lambda _: ['right'])
    calls = []
    monkeypatch.setattr(m, '_trigger', lambda _, service, side, **kw: calls.append((service, kw)))
    monkeypatch.setattr(m, '_confirm_hold', lambda *args: None)
    m._preempt(s)
    assert [name for name, _ in calls] == ['episode/stop', 'pd/hold']
    assert calls[-1][1]['expect'] == ('TRACKING',)
    assert not s.supervisor.is_alive('policy_test_right#0')


def test_reset_retains_no_old_progress(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    m.execute('preflight', operator='test')
    wait(m)
    m.reset()
    assert m.view() == {'active': False, 'outcome': None, 'steps': []}


def rg3_workspace(console):
    w = console.session.workspace
    obj = {'id': 'source200_blue', 'model': 'src200', 'policy_object': 'source200_real', 'enabled': False, 'input_source': 'fpp', 'locate_supported': True, 'results': {'fpp': {'valid': False, 'timestamp': None}}}
    w.objects = [obj]
    obj.update(enabled=True, input_source='fpp')
    obj['results']['fpp'].update(valid=True, timestamp=time.time(), live=True)
    w.catalog = [{'id': 'rh56f1/aglt/right_rg3_test', 'contract': '/registered/contract.json', 'status': 'verified', 'issues': []}]
    w.cards = [{'id': 'chosen', 'side': 'right', 'policy_id': w.catalog[0]['id'], 'object_id': obj['id'], 'execution_supported': True}]
    return obj


def test_rg3_exact_registered_policy_and_source_mapping(console, tiny_repo):
    from mission_stages import Command
    m = setup_actions(console, tiny_repo, {'episode_grasp_rg3_right': [{'argv': ['true']}]})
    obj = rg3_workspace(console)
    cmd = Command(('python3', 'rh_aglt_node.py', '-p', 'contract:=default', '-p', 'cup_topic:=/objects/source240b_blue/live_pose'))
    out = m._transform(console.session, 'rg3_chosen', {}, cmd)
    assert 'contract:=/registered/contract.json' in out.argv
    assert f"cup_topic:=/objects/{obj['id']}/live_pose" in out.argv
    runner = m._transform(console.session, 'rg3_chosen', {}, Command(('episode_runner_node.py', '-p', 'episode:=default')))
    assert f"policy_object:={obj['policy_object']}" in runner.argv
    assert f"cup_object:={obj['id']}" in runner.argv
    fake = m._transform(console.session, 'rg3_chosen', {}, Command(('fake_cup_pose_pub.py', '--topic', '/objects/source240b_blue/pose')))
    assert fake.argv[-1] == f"/objects/{obj['id']}/pose"


def test_rg3_locate_only_and_snapshot_inputs_blocked(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'episode_grasp_rg3_right': [{'argv': ['true']}]})
    obj = rg3_workspace(console)
    obj['input_source'] = 'locate'
    assert any('locate' in reason for reason in m.describe('rg3_chosen')['reasons'])
    obj['input_source'] = 'fpp'
    obj['results']['fpp']['live'] = False
    assert any('live_pose' in reason for reason in m.describe('rg3_chosen')['reasons'])
    obj['results']['fpp'].update(live=True, timestamp=time.time() - 10)
    assert any('신선한' in reason for reason in m.describe('rg3_chosen')['reasons'])


def test_perception_preview_includes_exact_side_effects(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    w = console.session.workspace
    obj = rg3_workspace(console)
    obj['enabled'] = True
    w.perception.update(fpp=True, locate=True)
    preview = m.describe('perception_refresh')
    assert len(preview['commands']) == 3
    assert preview['commands'][1]['argv'] == w.fpp_argv([obj['id']])
    assert preview['commands'][2]['argv'] == w.locate_argv(obj['id'])


def lifecycle(console, tiny_repo, monkeypatch):
    from types import SimpleNamespace
    import yaml
    commands = [{'argv': ['true'], 'manual': True},
                {'argv': ['true']},
                {'argv': ['true'], 'manual': True},
                {'argv': ['true'], 'note': 'record stop'}]
    m = setup_actions(console, tiny_repo, {'episode_grasp_rg3_right': commands, 'home_right': [{'argv': ['true']}]})
    rg3_workspace(console)
    (tiny_repo / 'config/episodes').mkdir()
    (tiny_repo / 'config/episodes/grasp_rg3_right.yaml').write_text(yaml.safe_dump({
        'objects': {'CUP': {'topic': '/objects/source240b_blue/pose'}},
        'policies': {'rg3': {'policy': 'default'}},
        'sequence': [{'type': 'policy', 'policy_object': 'source240_real'}]}))
    status = {'episode': 'grasp_rg3_right', 'phase': 'READY'}
    observation = SimpleNamespace(status={'episode_runner': status, 'pd_right': {'execute': True, 'phase': 'TRACKING'}},
                                  age_s={'episode_runner': 0, 'pd_right': 0}, joints_age_s=0, joint_ages={}, rosgraph=None, rosgraph_age_s=None)
    monkeypatch.setattr(console.session.feed, 'observed', lambda: observation)
    calls = []
    monkeypatch.setattr(m, '_preempt', lambda s: calls.append('stop -> hold'))
    monkeypatch.setattr(m, '_run_commands', lambda s, sid, cmds: calls.append((sid, cmds)))
    return m, status, calls


def test_rg3_terminal_holds_and_finishes_recording(console, tiny_repo, monkeypatch):
    m, status, calls = lifecycle(console, tiny_repo, monkeypatch)
    original = m._run_commands
    def run(s, sid, cmds):
        original(s, sid, cmds)
        if sid == 'action_episode_run':
            status['phase'] = 'SUCCESS'
    monkeypatch.setattr(m, '_run_commands', run)
    m.execute('rg3_chosen', operator='test')
    assert wait(m)['outcome'] == 'DONE'
    assert calls[-2] == 'stop -> hold'
    assert calls[-1][0] == 'episode_grasp_rg3_right_cleanup'
    assert calls[-1][1][0].note == 'record stop'
    import yaml
    doc = yaml.safe_load((console.session.run_dir / 'rg3_chosen.yaml').read_text())
    assert doc['objects']['CUP']['topic'] == '/objects/source200_blue/pose'
    assert doc['policies']['rg3']['policy'] == 'rh56f1/aglt/right_rg3_test'
    assert doc['sequence'][0]['policy_object'] == 'source200_real'
    assert not any('attach_fallback' in arg or 'fk' in arg.lower() for row in calls if isinstance(row, tuple) for cmd in row[1] for arg in cmd.argv)


def test_rg3_control_intervention_joins_without_console_lock_deadlock(console, tiny_repo, monkeypatch):
    m, status, calls = lifecycle(console, tiny_repo, monkeypatch)
    original = m._run_commands
    started = __import__('threading').Event()
    def run(s, sid, cmds):
        original(s, sid, cmds)
        if sid == 'action_episode_run':
            status['phase'] = 'RUNNING'
            started.set()
    monkeypatch.setattr(m, '_run_commands', run)
    m.execute('rg3_chosen', operator='test')
    assert started.wait(2)
    with console._lock:
        m.execute('home_right', operator='test')
    assert wait(m)['outcome'] == 'DONE'
    assert sum(isinstance(row, tuple) and row[0] == 'action_episode_run' for row in calls) == 1
    cleanup = next(i for i, row in enumerate(calls) if isinstance(row, tuple) and row[0].endswith('_cleanup'))
    home = next(i for i, row in enumerate(calls) if isinstance(row, tuple) and row[0] == 'home_right')
    assert cleanup < home
    assert 'episode_grasp_rg3_right' not in console.session.state.completed


def test_head_command_waits_for_cancelled_scan_to_settle(console, tiny_repo, monkeypatch):
    import threading
    m = setup_actions(console, tiny_repo, {'head_control': [{'argv': ['true']}]})
    rg3_workspace(console)
    begun = threading.Event()
    settled = threading.Event()
    calls = []
    def scan(s, parameters):
        begun.set()
        assert m._cancel_scan.wait(2)
        time.sleep(.02)  # simulated current move and return-pose settlement
        settled.set()
    original = m._run_commands
    def commands(s, sid, cmds):
        if sid == 'head_control':
            assert settled.is_set()
            calls.append('new head command')
        original(s, sid, cmds)
    monkeypatch.setattr(m, '_refresh', scan)
    monkeypatch.setattr(m, '_run_commands', commands)
    m.execute('perception_refresh', operator='test')
    assert begun.wait(2)
    assert any('스캔 취소' in warning for warning in m.describe('head_control')['warnings'])
    with console._lock:
        m.execute('head_control', {'pan': 5, 'tilt': -25}, operator='test')
    assert wait(m)['outcome'] == 'DONE'
    assert calls == ['new head command']
    assert console.session.workspace.head['pan'] == 5


def test_fpp_refresh_starts_all_enabled_groups_before_subset_rescan(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    obj = rg3_workspace(console)
    import copy
    second = copy.deepcopy(obj)
    second['id'] = 'shaker_c_pink'
    console.session.workspace.objects.append(second)
    rows = m._perception_commands(console.session, {'object_ids': [obj['id']]})
    assert 'perception_ctl.py' in rows[0][1].argv[1]
    assert obj['id'] in rows[0][1].argv and second['id'] in rows[0][1].argv
    assert rows[0][1].argv[-3:] == ('--wait', '150', '--require-all')
    assert rows[1][1].argv == tuple(console.session.workspace.fpp_argv([obj['id']]))


def test_preemption_requires_each_targeted_side_not_any_fresh_pd(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    s = console.session
    s.supervisor.spawn('policy_left#0', stage='policy_left', note='left policy', argv=['sleep', '30'], background=True, manual=False)
    monkeypatch.setattr(console, '_pd_sides', lambda _: ['right'])
    calls = []
    monkeypatch.setattr(m, '_trigger', lambda *args, **kw: calls.append(args))
    monkeypatch.setattr(m, '_confirm_hold', lambda *args: None)
    with pytest.raises(ConsoleError, match='left'):
        m._preempt(s)
    assert s.supervisor.is_alive('policy_left#0')
    assert not calls


def test_cancelled_scan_without_restored_pose_blocks_head_command(console, tiny_repo, monkeypatch):
    import threading
    m = setup_actions(console, tiny_repo, {'head_control': [{'argv': ['true']}]})
    rg3_workspace(console)
    begun = threading.Event()
    def scan(s, parameters):
        m._scan_restore_ok = False
        begun.set()
        m._cancel_scan.wait(2)
    monkeypatch.setattr(m, '_refresh', scan)
    m.execute('perception_refresh', operator='test')
    assert begun.wait(2)
    with pytest.raises(ConsoleError, match='복귀|정착'):
        m.execute('head_control', {'pan': 0, 'tilt': -25}, operator='test')
    assert not any(p['stage'] == 'head_control' for p in console.session.supervisor.table())


@pytest.mark.parametrize('home_ok', [True, False])
def test_scan_result_restoration_flag_survives_nonzero_cancel_exit(console, tiny_repo, monkeypatch, home_ok):
    import threading
    import json
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    rg3_workspace(console)
    console.session.workspace.perception.update(fpp=False, locate=True)
    m._cancel_scan = threading.Event()
    m._scan_restore_ok = True
    monkeypatch.setattr(m, '_run_commands', lambda *args: (_ for _ in ()).throw(ConsoleError('cancelled rc=1')))
    monkeypatch.setattr(console.session.supervisor, 'tail', lambda *args: json.dumps({'valid': False, 'home_ok': home_ok, 'reason': 'cancelled'}))
    with pytest.raises(ConsoleError, match='cancelled'):
        m._refresh(console.session, {})
    assert m._scan_restore_ok is home_ok


def test_head_profile_restarts_same_converter_key_and_invalidates_coordinates(console, tiny_repo):
    script = tiny_repo / 'object_pose_node.py'
    script.write_text('import time; time.sleep(30)\n')
    m = setup_actions(console, tiny_repo, {
        'head_control': [{'argv': ['true']}],
        'cups': [{'argv': ['python3', str(script)], 'background': True}]})
    rg3_workspace(console)
    w = console.session.workspace
    (tiny_repo / 'config/calib_profiles').mkdir()
    (tiny_repo / 'config/calib_profiles/selected.yaml').write_text('profile: selected\n')
    s = console.session
    s.supervisor.spawn('cups#0', stage='cups', note='old converter', argv=['python3', str(script)], background=True, manual=False)
    before = next(p['pid'] for p in s.supervisor.table() if p['key'] == 'cups#0')
    parameters = {'pan': 10, 'tilt': -25, 'calib_profile': 'selected'}
    preview = m.describe('head_control', parameters)
    restart = next(c for c in preview['commands'] if c['stop'] == ['cups#0'])
    assert '--head-joint-topic' in restart['argv']
    assert restart['argv'][-2:] == ['--calib-profile', 'selected']
    m.execute('head_control', parameters, operator='test')
    assert wait(m)['outcome'] == 'DONE'
    proc = next(p for p in s.supervisor.table() if p['key'] == 'cups#0')
    assert proc['alive'] and proc['pid'] != before
    assert proc['argv'] == restart['argv']
    assert not w.objects[0]['results']['fpp']['valid']


def test_fpp_execution_starts_all_groups_then_rescans(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    rg3_workspace(console)
    calls = []
    monkeypatch.setattr(m, '_run_commands', lambda s, sid, cmds: calls.append((sid, cmds[0].argv)))
    m.execute('perception_refresh', operator='test')
    assert wait(m)['outcome'] == 'DONE'
    assert [sid for sid, _ in calls] == ['action_fpp_start', 'action_fpp']
    assert '--require-all' in calls[0][1]
    assert not console.session.workspace.objects[0]['results']['fpp']['valid']
