import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_supervisor.interface import Interface, parse_status, write_approval
from vlm_supervisor.judge import Judge, parse_judgement
from vlm_supervisor.planner import Planner, simulate
from vlm_supervisor.supervisor import Supervisor


def plan():
    return [
        ['go_home', 'both', '-', '-'],
        ['scene_snapshot', '-', '-', '-'],
        ['grasp', 'left', 'SHAKER', '-'],
        ['to_tilt_start', 'left', 'SHAKER', '-'],
        ['grasp', 'right', 'SRC_BLUE', '-'],
        ['to_tilt_start', 'right', 'SRC_BLUE', '-'],
        ['tilt', 'both', 'SRC_BLUE', 'SHAKER'],
        ['place', 'right', 'SRC_BLUE', 'HOLDER_1'],
        ['place', 'left', 'SHAKER', 'HOLDER_0'],
        ['go_home', 'both', '-', '-'],
        ['terminal', '-', '-', 'SUCCESS'],
    ]


def test_simulate_valid_and_no_mutation():
    steps = plan()
    assert simulate(steps)['ok']
    assert steps == plan()


@pytest.mark.parametrize('steps', [
    [['tilt', 'both', 'SRC_BLUE', 'SHAKER']],
    [['grasp', 'left', 'SRC_BLUE', '-']],
    plan()[:3] + [['go_home', 'both', '-', '-']],
    plan()[:7] + [['place', 'right', 'SRC_BLUE', 'HOLDER_2']],
    plan()[:4] + [['grasp', 'left', 'SHAKER', '-']],
    plan()[:-1],
    plan() + [['scene_snapshot', '-', '-', '-']],
])
def test_simulate_rejects_invalid(steps):
    assert not simulate(steps)['ok']


class Backend:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def ask(self, images, text, level='low', **kwargs):
        self.prompts.append(text)
        return self.answer, {'model': 'fake'}


def test_planner_uses_sentence_cards_and_simulates():
    backend = Backend('```json\n' + json.dumps(plan()) + '\n```')
    result = Planner(backend).plan('파란컵만 옮겨줘', Image.new('RGB', (2, 2)))
    assert result['simulation']['ok']
    assert '파란컵만 옮겨줘' in backend.prompts[0]
    assert 'HOLDER_1' in backend.prompts[0]


@pytest.mark.parametrize('raw', ['{}', 'nonsense', '{"success": "false", "choice": "A"}',
                                      '{"success": false, "choice": "A"}',
                                      '{"success": true, "choice": "ABCDE"}', '[]'])
def test_bad_judge_stops(raw):
    assert parse_judgement(raw)['action'] == 'stop'


@pytest.mark.parametrize('choice,action', [('A','next'), ('B','retry'), ('C','rescan'), ('D','home'), ('E','stop')])
def test_judge_actions(choice, action):
    raw = json.dumps({'success': choice == 'A', 'choice': choice, 'reason': '관찰'})
    assert parse_judgement('</think>```json\n' + raw + '\n```')['action'] == action


def test_approval_appends_existing_format(tmp_path):
    path = tmp_path / 'episode_approvals.jsonl'
    write_approval('grasp_blue', 'tester', path, 'runner')
    write_approval('tilt_blue', 'tester', path, 'runner')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['what'] for row in rows] == ['grasp_blue', 'tilt_blue']
    assert set(rows[0]) == {'what', 'at', 't', 'operator', 'runner'}
    assert isinstance(rows[0]['t'], float)


class FakeInterface(Interface):
    def __init__(self, path):
        super().__init__('runner', path)
        self.calls = []
        self.current = status()

    def call(self, service):
        self.calls.append(service)
        return {'ok': True, 'reasons': []}

    def status(self):
        return self.current

    def frame(self, after=0, wait=3):
        return Image.new('RGB', (2, 2))


def status():
    return parse_status(json.dumps({'status': 'READY', 'node': 'place_blue', 'world': {},
        'nodes': [{'id': 'place_blue', 'type': 'policy'}],
        'last': {'node': 'tilt_blue', 'code': 'done'}, 'pending': None, 'history': []}))


def test_supervisor_active_approves_and_deduplicates(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    sup = Supervisor(interface, Judge(Backend('{"success":true,"choice":"A","reason":"ok"}')), mode='active')
    assert sup.step(status())['action'] == 'next'
    assert sup.step(status()) is None
    assert interface.calls == ['next']
    assert json.loads(interface.approvals.read_text())['what'] == 'place_blue'


def test_shadow_writes_no_approval_and_calls_nothing(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    sup = Supervisor(interface, Judge(Backend('{"success":false,"choice":"B","reason":"retry"}')))
    result = sup.step(status())
    assert result['result']['error'] == 'NotSupported'
    assert interface.calls == []
    assert not interface.approvals.exists()


@pytest.mark.parametrize('action', ['retry', 'rescan', 'home', 'insert', 'refresh'])
def test_unsupported_records_and_stops(tmp_path, action):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    result = interface.perform(action, active=True)
    assert result['error'] == 'NotSupported'
    assert result['human_required']
    assert interface.calls == ['stop']
    assert not interface.approvals.exists()


def test_busy_and_terminal_not_approved(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    sup = Supervisor(interface, Judge(Backend('{}')), mode='active')
    for phase in ['RUNNING', 'SUCCESS', 'STOPPED']:
        st = status() | {'status': phase}
        assert sup.step(st) is None
    assert interface.calls == []


def test_judge_prompt_contains_full_status():
    backend = Backend('{"success":true,"choice":"A","reason":"ok"}')
    Judge(backend).judge(status(), Image.new('RGB', (2, 2)))
    assert 'world' in backend.prompts[0]
    assert 'tilt_blue' in backend.prompts[0]


def test_invalid_status():
    with pytest.raises(ValueError):
        parse_status('{"status":"READY"}')


def test_stale_vlm_result_never_approves(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')

    class AdvancingJudge:
        def judge(self, st, image):
            interface.current = st | {'node': 'another_node'}
            return {'success': True, 'choice': 'A', 'action': 'next'}

    result = Supervisor(interface, AdvancingJudge(), mode='active').step(status())
    assert result['event'] == 'stale_decision'
    assert interface.calls == []
    assert not interface.approvals.exists()


def test_missing_camera_stops(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    interface.frame = lambda **kw: None
    sup = Supervisor(interface, Judge(Backend('{}')), mode='active')
    assert sup.step(status())['action'] == 'stop'
    assert interface.calls == ['stop']
    assert sup.halted


def test_service_rejection_stops_and_logs(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')

    def call(service):
        interface.calls.append(service)
        return {'ok': service == 'stop'}

    interface.call = call
    log = tmp_path / 'decisions.jsonl'
    sup = Supervisor(interface, Judge(Backend('{"success":true,"choice":"A","reason":"ok"}')),
                     mode='active', log_path=log)
    sup.step(status())
    assert interface.calls == ['next', 'stop']
    assert sup.halted
    assert [json.loads(line)['event'] for line in log.read_text().splitlines()
            if json.loads(line)['event'] != 'error'] == ['decision', 'service_failure']


def test_active_recovery_is_logged_and_never_approved(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    log = tmp_path / 'decisions.jsonl'
    sup = Supervisor(interface, Judge(Backend('{"success":false,"choice":"C","reason":"lost cup"}')),
                     mode='active', log_path=log)
    sup.step(status())
    assert json.loads(log.read_text().splitlines()[-1])['result']['error'] == 'NotSupported'
    assert interface.calls == ['stop']
    assert not interface.approvals.exists()


def test_new_history_allows_same_node_new_attempt(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    sup = Supervisor(interface, Judge(Backend('{"success":true,"choice":"A","reason":"ok"}')))
    assert sup.step(status())
    interface.current = status() | {'history': [{'node': 'tilt_blue', 'attempt': 2}]}
    assert sup.step(interface.current)


def test_runtime_has_no_deploy_imports_and_ros_only_in_interface():
    import ast

    root = Path(__file__).parents[1]
    for path in [*root.glob('*.py'), *(root / 'shell').glob('*.py')]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            if path.name != 'failure_policy.py':
                assert all('deploy' not in name and not name.startswith('policy_control') for name in names)
            if path.name not in {'interface.py', 'shell_node.py'}:
                assert all(name.split('.')[0] not in {'rclpy', 'std_msgs', 'sensor_msgs', 'std_srvs'} for name in names)


def test_run_status_timeout_requests_stop(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    interface.current = {}
    sup = Supervisor(interface, Judge(Backend('{}')), mode='active')
    assert sup.run(status_wait=0) == 2
    assert interface.calls == ['stop']


@pytest.mark.parametrize('phase,code', [('SUCCESS', 0), ('FAILURE', 1), ('STOPPED', 1)])
def test_run_terminal_exit_code(tmp_path, phase, code):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    interface.current = status() | {'status': phase}
    ok = Backend('{"success":true,"choice":"A","reason":"ok"}')
    assert Supervisor(interface, Judge(ok)).run() == code
    assert not interface.calls


def test_success_end_still_judges_final_action(tmp_path):
    # runner closes terminal right after the last action, so READY is never seen for it
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    interface.current = status() | {'status': 'SUCCESS'}
    bad = Backend('{"success":false,"choice":"E","reason":"cup fell"}')
    assert Supervisor(interface, Judge(bad)).run() == 1
    assert bad.prompts


def test_stale_by_status_outage_is_reconsidered(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    sup = Supervisor(interface, Judge(Backend('{"success":true,"choice":"A","reason":"ok"}')), mode='active')
    interface.current = {}
    assert sup.step(status())['event'] == 'stale_decision'
    interface.current = status()
    assert sup.step(status())['action'] == 'next'


def test_initial_ready_approves_without_judge_call(tmp_path):
    interface = FakeInterface(tmp_path / 'approvals.jsonl')
    interface.current = status() | {'last': {}}
    backend = Backend('{}')
    sup = Supervisor(interface, Judge(backend), mode='active')
    sup.step(interface.current)
    assert interface.calls == ['next']
    assert backend.prompts == []


def test_ros_import_is_optional(monkeypatch):
    import builtins
    import importlib
    import vlm_supervisor.interface as boundary

    original = builtins.__import__

    def no_ros(name, *args, **kwargs):
        if name.split('.')[0] in {'rclpy', 'sensor_msgs', 'std_msgs', 'std_srvs'}:
            raise ImportError('ROS deliberately unavailable')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_ros)
    importlib.reload(boundary)
    with pytest.raises(RuntimeError, match='ROS 2'):
        boundary.RosInterface('runner', '/tmp/unused_approvals.jsonl')
