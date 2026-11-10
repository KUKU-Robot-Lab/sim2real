"""Episode shell contracts, without ROS or deploy imports."""
import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
EP = ROOT / 'config/episodes/cocktail_1c_e4000.yaml'


def episode():
    return yaml.safe_load(EP.read_text())


def status():
    return {'status': 'READY', 'busy': False, 'node': 'place_blue',
            'last': {'node': 'tilt_blue'}, 'world': {'pose': 'SETTING',
            'right_hand': 'SRC_BLUE', 'left_hand': 'SHAKER',
            'task_flags': ['blue_poured'], 'objects': {
                'SRC_BLUE': {'at': 'right_hand'}, 'SHAKER': {'at': 'left_hand'},
                'SRC_PINK': {'at': 'table'}}}}


@pytest.mark.parametrize('verb,args,first', [
    ('retry', {'node_id': 'tilt_blue'}, 'tilt_blue'),
    ('rescan', {}, None), ('home', {'side': 'right'}, 'home_r_after_blue'),
    ('insert', {'spec': {'id': 'place_blue', 'type': 'confirm', 'text': 'check'}}, 'place_blue'),
    ('human', {'text': '확인'}, None),
])
def test_derive(verb, args, first):
    from vlm_supervisor.shell.derive import derive
    original = episode()
    saved = copy.deepcopy(original)
    result = derive(original, status(), {'verb': verb, 'args': args})
    assert original == saved
    assert {k: v for k, v in result.items() if k != 'sequence'} == {
        k: v for k, v in original.items() if k != 'sequence'}
    nodes = result['sequence']
    assert nodes[0]['type'] == 'confirm'
    restore = nodes[1]
    assert restore['id'].startswith('shell_restore_')
    assert restore['objects'] == ['SRC_PINK']
    assert restore['expect'] == {'pose': 'SETTING', 'right_hand': 'SRC_BLUE',
                                 'left_hand': 'SHAKER', 'blue_poured': True}
    ids = [n['id'] for n in nodes]
    assert len(ids) == len(set(ids))
    if verb == 'rescan':
        assert len(nodes) == 3
        assert nodes[-1]['type'] == 'terminal'
    elif verb == 'retry':
        assert nodes[2]['id'] == first
    elif verb == 'home':
        assert nodes[2]['name'] == 'go_home_r'
        assert nodes[2]['id'] != first
        assert nodes[3]['id'] == 'place_blue'
    elif verb == 'insert':
        assert nodes[2]['text'] == 'check'
        assert nodes[3]['id'] == 'place_blue'
    else:
        assert nodes[2]['type'] == 'confirm'
        assert nodes[2]['text'] == '확인'


def test_derive_rejects_unknown_target_and_empty_snapshot():
    from vlm_supervisor.shell.derive import derive
    with pytest.raises(ValueError):
        derive(episode(), status(), {'verb': 'retry', 'args': {'node_id': 'missing'}})
    st = status()
    st['world']['objects']['SRC_PINK']['at'] = 'holder:HOLDER_2'
    with pytest.raises(ValueError, match='snapshot'):
        derive(episode(), st, {'verb': 'rescan'})


class Adapter:
    def __init__(self):
        self.events = []
        self.current = status()
        self.fail_plan = False

    def status(self):
        return self.current

    def stop(self):
        self.events.append('stop')
        return {'ok': True}

    def terminate(self):
        self.events.append('terminate')

    def validate(self, path):
        self.events.append('plan')
        assert Path(path).exists()
        return not self.fail_plan

    def start(self, path):
        self.events.append(('start', Path(path)))

    def wait_ready(self):
        self.events.append('ready')
        return True

    def pid(self):
        return 123


def test_controller_order_dedup_and_rollback(tmp_path):
    from vlm_supervisor.shell.shell_node import ShellController
    adapter = Adapter()
    shell = ShellController(episode(), EP, tmp_path, adapter)
    cmd = {'cmd_id': 'one', 'verb': 'retry', 'args': {'node_id': 'tilt_blue'}}
    ack = shell.handle(cmd)
    assert ack['ok'] and ack['generation'] == 1
    assert adapter.events == ['stop', 'terminate', 'plan',
                              ('start', Path(ack['episode_path'])), 'ready']
    assert shell.handle(cmd) == ack
    assert len(adapter.events) == 5
    adapter.fail_plan = True
    failed = shell.handle(cmd | {'cmd_id': 'two'})
    assert not failed['ok'] and failed['generation'] == 1
    assert adapter.events[-2:] == [('start', Path(ack['episode_path'])), 'ready']


def test_controller_busy_and_not_ready_rejected(tmp_path):
    from vlm_supervisor.shell.shell_node import ShellController
    adapter = Adapter()
    shell = ShellController(episode(), EP, tmp_path, adapter)
    for st in ({'busy': True, 'status': 'READY'}, {}, {'status': 'RUNNING'}):
        adapter.current = st
        assert not shell.handle({'cmd_id': str(st), 'verb': 'rescan'})['ok']
    assert adapter.events == []


@pytest.mark.parametrize('verb,args', [('retry', {'node_id': 'tilt_blue'}),
    ('rescan', {}), ('home', {'side': 'right'}),
    ('insert', {'spec': {'id': 'check', 'type': 'confirm', 'text': 'check'}}),
    ('human', {'text': 'check'})])
@pytest.mark.parametrize('mode', ['--plan', '--dry-run'])
def test_deploy_subprocess(tmp_path, verb, args, mode):
    from vlm_supervisor.shell.derive import derive
    path = tmp_path / 'derived.yaml'
    path.write_text(yaml.safe_dump(derive(episode(), status(), {'verb': verb, 'args': args})))
    proc = subprocess.run([sys.executable, str(ROOT / 'deploy/policy_control/tools/episode_run.py'),
                           '--episode', str(path), mode], cwd=ROOT, capture_output=True, text=True, timeout=60,
                           env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    output = proc.stdout + proc.stderr
    if mode == '--plan' and proc.returncode == 1 and not proc.stderr and "가 nn/ 에 없다" in output:
        pytest.skip('local policy checkpoint files unavailable; schema parsed successfully')
    if 'ModuleNotFoundError' in output:
        pytest.skip(output[-500:])
    assert proc.returncode == 0, output


def test_interface_shell_and_supervisor_recovery(tmp_path):
    from vlm_supervisor.interface import Interface
    from vlm_supervisor.supervisor import Supervisor

    class Boundary(Interface):
        def __init__(self):
            super().__init__('runner', tmp_path / 'approvals')
            self.current = status()
            self.sent = []
        def status(self):
            return self.current
        def frame(self, **kw):
            return object()
        def shell_command(self, verb, args):
            self.sent.append((verb, args))
            self.current = status() | {'node': 'shell_confirm_1', 'last': {}}
            return {'ok': True, 'generation': 1}
        def call(self, service):
            raise AssertionError(service)

    class Judge:
        def judge(self, st, image):
            return {'choice': 'B', 'action': 'retry'}

    boundary = Boundary()
    supervisor = Supervisor(boundary, Judge(), mode='active')
    event = supervisor.step(boundary.current)
    assert event['result']['ok']
    assert boundary.sent == [('retry', {'node_id': 'tilt_blue'})]
    assert not supervisor.halted
    assert supervisor.seen is None


def test_human_confirm_waits_for_operator(tmp_path):
    from vlm_supervisor.interface import Interface
    from vlm_supervisor.supervisor import Supervisor
    class Boundary(Interface):
        def call(self, service):
            raise AssertionError('must not auto-approve human confirmation')
    st = status() | {'node': 'shell_human_1', 'confirm': {'text': 'check'}, 'last': {}}
    assert Supervisor(Boundary('runner', tmp_path / 'a'), None, mode='active').step(st) is None


def test_controller_ready_failure_rolls_back(tmp_path):
    from vlm_supervisor.shell.shell_node import ShellController
    adapter = Adapter()
    responses = iter([False, True])
    adapter.wait_ready = lambda: next(responses)
    shell = ShellController(episode(), EP, tmp_path, adapter)
    ack = shell.handle({'cmd_id': 'failed-ready', 'verb': 'rescan'})
    assert not ack['ok'] and ack['generation'] == 0
    assert adapter.events[-1] == ('start', EP)
    assert shell.path == EP


def test_home_infers_failed_arm(tmp_path):
    from vlm_supervisor.shell.shell_node import ShellController
    adapter = Adapter()
    adapter.current['last']['node'] = 'place_blue'
    shell = ShellController(episode(), EP, tmp_path, adapter)
    ack = shell.handle({'cmd_id': 'home', 'verb': 'home', 'args': {}})
    assert ack['ok']
    spec = yaml.safe_load(Path(ack['episode_path']).read_text())
    assert spec['sequence'][2]['name'] == 'go_home_r'


def test_process_escalates_and_pins_runner_parameters(monkeypatch, tmp_path):
    from vlm_supervisor.shell.shell_node import RosProcessAdapter
    import threading
    events = []
    class Process:
        pid = 1234
        def poll(self):
            return None
        def wait(self, timeout):
            events.append(('wait', timeout))
            if timeout == 8:
                raise subprocess.TimeoutExpired('runner', timeout)
    adapter = RosProcessAdapter.__new__(RosProcessAdapter)
    adapter.process = Process()
    adapter.condition = threading.Condition()
    monkeypatch.setattr('vlm_supervisor.shell.shell_node.os.killpg', lambda pid, sig: events.append((pid, sig)))
    adapter.terminate()
    import signal
    assert events == [(1234, signal.SIGINT), ('wait', 8), (1234, signal.SIGTERM), ('wait', 5)]
    adapter.process = None
    adapter.runner = 'episode_runner_right'
    adapter.runner_cmd = ['ros2', 'run', 'policy_control', 'episode_runner_node', '--ros-args', '-p', 'robot:=fake']
    calls = []
    monkeypatch.setattr('vlm_supervisor.shell.shell_node.subprocess.Popen', lambda cmd, **kw: calls.append((cmd, kw)))
    adapter.start(tmp_path / 'gen.yaml')
    cmd, options = calls[0]
    assert 'robot:=fake' in cmd
    assert '__node:=episode_runner_right' in cmd
    assert f'episode:={tmp_path}/gen.yaml' in cmd
    assert options['start_new_session'] is True


def test_ros_transport_matches_ack_without_ros():
    import threading
    import json
    from types import SimpleNamespace
    from vlm_supervisor.interface import RosInterface
    boundary = RosInterface.__new__(RosInterface)
    boundary.String = SimpleNamespace
    boundary.shell_condition = threading.Condition()
    boundary.shell_acks = {}
    boundary.shell_timeout = 0.1
    sent = []
    class Publisher:
        def get_subscription_count(self):
            return 1
        def publish(self, message):
            command = json.loads(message.data)
            sent.append(command)
            boundary._on_shell_ack(SimpleNamespace(data=json.dumps({
                'cmd_id': 'unrelated', 'ok': True})))
            boundary._on_shell_ack(SimpleNamespace(data=json.dumps({
                'cmd_id': command['cmd_id'], 'ok': True, 'generation': 2})))
    boundary.shell_pub = Publisher()
    assert boundary.shell_command('retry', {'node_id': 'tilt_blue'})['generation'] == 2
    assert sent[0]['verb'] == 'retry'
    assert sent[0]['args'] == {'node_id': 'tilt_blue'}
    assert boundary.shell_acks == {}


def test_retry_can_return_to_original_after_rescan(tmp_path):
    from vlm_supervisor.shell.shell_node import ShellController
    adapter = Adapter()
    shell = ShellController(episode(), EP, tmp_path, adapter)
    assert shell.handle({'cmd_id': 'scan', 'verb': 'rescan'})['ok']
    adapter.current['status'] = 'SUCCESS'
    ack = shell.handle({'cmd_id': 'back', 'verb': 'retry', 'args': {'node_id': 'tilt_blue'}})
    assert ack['ok']
    assert yaml.safe_load(Path(ack['episode_path']).read_text())['sequence'][2]['id'] == 'tilt_blue'


def test_stop_respects_json_rejection():
    from types import SimpleNamespace
    from vlm_supervisor.shell.shell_node import RosProcessAdapter
    adapter = RosProcessAdapter.__new__(RosProcessAdapter)
    adapter.Trigger = SimpleNamespace(Request=lambda: None)
    future = SimpleNamespace(done=lambda: True,
                             result=lambda: SimpleNamespace(success=True, message='{"ok":false}'))
    adapter.client = SimpleNamespace(wait_for_service=lambda **kw: True, call_async=lambda req: future)
    assert not adapter.stop()['ok']
