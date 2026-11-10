import pytest

from vlm_supervisor.supervisor import Supervisor


class FakeInterface:
    def __init__(self, failures=1, result=None, outage=0):
        self.state = self.failure()
        self.failures = failures
        self.result = {'ok': True} if result is None else result
        self.outage = outage
        self.missing = 0
        self.calls = []

    @staticmethod
    def failure():
        return {'status': 'FAILURE', 'node': None,
                'last': {'node': 'grasp_to_tilt_start', 'code': 'lift_guard_failed', 'ok': False}}

    def status(self):
        if self.missing:
            self.missing -= 1
            return {}
        return self.state.copy()

    def frame(self, **kwargs):
        return b'frame'

    def perform(self, action, target=None, *, active=False, args=None):
        self.calls.append((action, args, active))
        if active and self.result.get('ok'):
            if action == 'retry':
                self.state = {'status': 'READY', 'node': 'grasp_to_tilt_start'}
                self.missing = self.outage
            elif action == 'next':
                self.failures -= 1
                self.state = self.failure() if self.failures else {'status': 'SUCCESS'}
        return self.result


class Judge:
    def __init__(self, action='retry'):
        self.action = action
        self.statuses = []

    def judge(self, status, image):
        self.statuses.append(status['status'])
        return {'action': self.action, 'choice': 'B'}


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr('vlm_supervisor.supervisor.time.monotonic', lambda: now[0])
    monkeypatch.setattr('vlm_supervisor.supervisor.time.sleep', lambda dt: now.__setitem__(0, now[0] + dt))


def test_failure_retry_restarts_and_survives_status_outage():
    interface, judge = FakeInterface(outage=20), Judge()
    supervisor = Supervisor(interface, judge, mode='active', settle=0)
    assert supervisor.run(status_wait=0.5) == 0
    assert interface.calls == [('retry', {'node_id': 'grasp_to_tilt_start'}, True),
                               ('next', None, True)]
    assert judge.statuses == ['FAILURE']


@pytest.mark.parametrize('limit', [0, 1, 3])
def test_recovery_cap_escalates_to_human_once(limit):
    interface = FakeInterface(failures=10)
    supervisor = Supervisor(interface, Judge(), mode='active', settle=0, max_recover=limit)
    assert supervisor.run() == 1
    actions = [call[0] for call in interface.calls]
    assert actions == ['retry', 'next'] * limit + ['human']


@pytest.mark.parametrize('mode,action,result', [
    ('shadow', 'retry', {'ok': True}),
    ('active', 'stop', {'ok': True}),
    ('active', 'retry', {'ok': False}),
    ('active', 'retry', {'error': 'NotSupported'}),
])
def test_unrecoverable_failure_ends(mode, action, result, tmp_path):
    interface = FakeInterface(result=result)
    log = tmp_path / 'events.jsonl'
    supervisor = Supervisor(interface, Judge(action), mode=mode, settle=0, log_path=log)
    assert supervisor.run() == 1
    assert '"event": "end", "status": "FAILURE"' in log.read_text()


def test_stopped_is_terminal():
    interface, judge = FakeInterface(), Judge()
    interface.state['status'] = 'STOPPED'
    assert Supervisor(interface, judge, mode='active').run() == 1
    assert interface.calls == []
    assert judge.statuses == []

def test_accepted_recovery_does_not_repeat_unchanged_failure():
    interface, judge = FakeInterface(), Judge()
    original = interface.perform

    def perform(*args, **kwargs):
        result = original(*args, **kwargs)
        interface.state = interface.failure()
        return result

    interface.perform = perform
    supervisor = Supervisor(interface, judge, mode='active', settle=0)
    assert supervisor.run(ready_timeout=0.5) == 1
    assert [call[0] for call in interface.calls] == ['retry', 'stop']
    assert judge.statuses == ['FAILURE']


def test_failure_choice_e_stops_even_with_retry_action():
    class StopJudge(Judge):
        def judge(self, status, image):
            return {'choice': 'E', 'action': 'retry'}

    interface = FakeInterface()
    assert Supervisor(interface, StopJudge(), mode='active', settle=0).run() == 1
    assert [call[0] for call in interface.calls] == ['stop']
