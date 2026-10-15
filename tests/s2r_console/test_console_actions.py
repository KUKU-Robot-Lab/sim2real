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
    s.supervisor.spawn('policy_test#0', stage='policy_test', note='policy', argv=['sleep', '30'], background=True, manual=False)
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
    assert s.supervisor.is_alive('policy_test#0')


def test_preemption_order_and_pd_confirmation(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'preflight': [{'argv': ['true']}]})
    s = console.session
    s.supervisor.spawn('policy_test#0', stage='policy_test', note='policy', argv=['sleep', '30'], background=True, manual=False)
    monkeypatch.setattr(console, '_pd_sides', lambda _: ['right'])
    calls = []
    monkeypatch.setattr(m, '_trigger', lambda _, service, side, **kw: calls.append((service, kw)))
    monkeypatch.setattr(m, '_confirm_hold', lambda *args: None)
    m._preempt(s)
    assert [name for name, _ in calls] == ['episode/stop', 'pd/hold']
    assert calls[-1][1]['expect'] == ('TRACKING',)
    assert not s.supervisor.is_alive('policy_test#0')


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
    assert len(preview['commands']) == 2
    assert preview['commands'][0]['argv'] == w.fpp_argv([obj['id']])
    assert preview['commands'][1]['argv'] == w.locate_argv(obj['id'])


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
