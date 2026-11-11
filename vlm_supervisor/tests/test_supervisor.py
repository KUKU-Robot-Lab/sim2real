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
                'last': {'node': 'grasp_to_tilt_start', 'code': 'pour_failed', 'ok': False}}

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


class NormalRetryInterface(FakeInterface):
    """Shell recovery remains available for a normal completed node only."""
    @staticmethod
    def failure():
        return {'status': 'READY', 'node': 'grasp_to_tilt_start',
                'last': {'node': 'grasp_to_tilt_start', 'code': 'none', 'ok': True}}


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


def test_normal_retry_restarts_and_survives_status_outage():
    interface, judge = NormalRetryInterface(outage=20), Judge()
    supervisor = Supervisor(interface, judge, mode='active', settle=0)
    assert supervisor.run(status_wait=0.5) == 0
    assert interface.calls == [('retry', {'node_id': 'grasp_to_tilt_start'}, True),
                               ('next', None, True)]
    assert judge.statuses == ['READY']


@pytest.mark.parametrize('limit', [0, 1, 3])
def test_recovery_cap_escalates_to_human_once(limit):
    interface = NormalRetryInterface(failures=10)
    supervisor = Supervisor(interface, Judge(), mode='active', settle=0, max_recover=limit)
    assert supervisor.run() == 1
    actions = [call[0] for call in interface.calls]
    assert actions == ['retry', 'next'] * limit + ['stop']


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
    interface, judge = NormalRetryInterface(), Judge()
    original = interface.perform

    def perform(*args, **kwargs):
        result = original(*args, **kwargs)
        interface.state = interface.failure()
        return result

    interface.perform = perform
    supervisor = Supervisor(interface, judge, mode='active', settle=0)
    assert supervisor.run(ready_timeout=0.5) == 1
    assert [call[0] for call in interface.calls] == ['retry', 'stop']
    assert judge.statuses == ['READY']


def test_failure_choice_e_stops_even_with_retry_action():
    class StopJudge(Judge):
        def judge(self, status, image):
            return {'choice': 'E', 'action': 'retry'}

    interface = FakeInterface()
    assert Supervisor(interface, StopJudge(), mode='active', settle=0).run() == 1
    assert [call[0] for call in interface.calls] == ['stop']


@pytest.mark.parametrize('code,action', [('lift_guard_failed', 'retry'), ('controller_error', 'retry'),
                                         ('pour_failed', 'next')])
def test_safety_code_or_vlm_success_on_failure_goes_to_human(code, action):
    interface = FakeInterface()
    interface.state['last']['code'] = code
    assert Supervisor(interface, Judge(action), mode='active', settle=0).run() == 1
    assert [call[0] for call in interface.calls] == ['stop']


class ApproachInterface:
    """READY before grasp_blue; next moves to the following node."""

    def __init__(self, node='grasp_blue'):
        self.state = {'status': 'READY', 'node': node, 'next_action': node, 'episode_id': 'ep1',
                      'last': {}, 'attempts': {}, 'world': {'objects': {}}}
        self.calls = []

    def status(self):
        return dict(self.state)

    def frame(self, **kwargs):
        return b'frame'

    def perform(self, action, target=None, *, active=False, args=None):
        self.calls.append((action, target, args, active))
        if active and action == 'next':
            self.state = {'status': 'SUCCESS'}
        return {'ok': True}


class Approach:
    def __init__(self, result):
        self.result, self.steps, self.images, self.worlds = result, [], [], []

    def check(self, image, skill_step, world=None):
        self.steps.append(skill_step)
        self.images.append(image)
        self.worlds.append(world)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


CLEAR = {'valid': True, 'verdict': 'clear', 'action': 'proceed', 'disagree': False, 'step_mismatch': False}
BLOCKED = {'valid': True, 'verdict': 'blocked', 'obstacles': ['SRC_PINK'], 'action': 'human',
           'disagree': False, 'step_mismatch': False}


def test_approach_clear_proceeds_with_card_hand_and_object():
    interface, approach = ApproachInterface(), Approach(CLEAR)
    assert Supervisor(interface, Judge(), mode='active', settle=0, approach_judge=approach).run() == 0
    assert approach.steps == [['grasp', 'right', 'SRC_BLUE', '-']]
    assert [c[0] for c in interface.calls] == ['next']


@pytest.mark.parametrize('result', [
    BLOCKED, BLOCKED | {'action': 'reorder', 'first': 'SRC_PINK'}, BLOCKED | {'action': 'other_hand'},
    CLEAR | {'disagree': True}, CLEAR | {'step_mismatch': True}, {'valid': False, 'verdict': 'unknown', 'action': 'human', 'error': 'invalid JSON'},
    RuntimeError('backend down')])
def test_approach_not_clear_sends_human_and_stops(result, tmp_path):
    interface, log = ApproachInterface(), tmp_path / 'events.jsonl'
    sup = Supervisor(interface, Judge(), mode='active', settle=0, log_path=log, approach_judge=Approach(result))
    assert sup.run() == 1
    assert interface.calls == [('stop', 'grasp_blue', None, True)]
    text = log.read_text()
    assert '"outcome": "human"' in text and '"geom": "none"' in text and '"approach": ' in text


def test_approach_shadow_only_logs(tmp_path):
    interface, log = ApproachInterface(), tmp_path / 'events.jsonl'
    sup = Supervisor(interface, Judge(), mode='shadow', settle=0, log_path=log, approach_judge=Approach(BLOCKED))
    assert sup.step(interface.status())['action'] == 'next'
    assert interface.calls == [('next', 'grasp_blue', None, False)]
    assert '"outcome": "human"' in log.read_text()


def test_approach_checked_once_per_node_attempt():
    interface, approach = ApproachInterface(), Approach(BLOCKED)
    sup = Supervisor(interface, Judge(), mode='shadow', settle=0, approach_judge=approach)
    sup.step(interface.status())
    interface.state['history'] = ['changed']  # new status key, same node and attempts
    sup.step(interface.status())
    assert approach.steps == [['grasp', 'right', 'SRC_BLUE', '-']]
    interface.state['attempts'] = {'grasp_blue': 1}
    sup.step(interface.status())
    assert len(approach.steps) == 2


@pytest.mark.parametrize('missing', ['episode_id', 'attempts'])
def test_approach_without_episode_or_attempts_is_not_deduplicated(missing):
    interface, approach = ApproachInterface(), Approach(BLOCKED)
    del interface.state[missing]
    sup = Supervisor(interface, Judge(), mode='shadow', settle=0, approach_judge=approach)
    sup.step(interface.status())
    interface.state['history'] = ['changed']
    sup.step(interface.status())
    assert len(approach.steps) == 2


def test_approach_key_consumed_after_clear_next():
    interface, approach = ApproachInterface(), Approach(CLEAR)
    sup = Supervisor(interface, Judge(), mode='active', settle=0, approach_judge=approach)
    sup.step(interface.status())
    assert sup.approach_seen is None
    interface.state = ApproachInterface().state  # 재기동: 같은 에피소드·노드·시도
    sup.step(interface.status() | {'history': ['restart']})
    assert len(approach.steps) == 2


def test_approach_uses_target_not_bridge_next_action():
    """sim 브리지는 next_action 에 서비스 이름('next')을 싣는다(test_sim_interface.Bridge)."""
    interface, approach = ApproachInterface(), Approach(CLEAR)
    interface.state |= {'next_action': 'next', 'nodes': [{'id': 'grasp_blue'}]}
    Supervisor(interface, Judge(), mode='shadow', settle=0, approach_judge=approach).step(interface.status())
    assert approach.steps == [['grasp', 'right', 'SRC_BLUE', '-']]
    assert approach.worlds == [{'objects': {}}]  # world 를 판정기로 넘긴다


def test_unknown_grasp_code_stops_before_approach():
    interface, approach = ApproachInterface(), Approach(BLOCKED)
    interface.state = {'status': 'FAILURE', 'node': None, 'episode_id': 'ep1', 'attempts': {'grasp_blue': 1},
                       'last': {'node': 'grasp_blue', 'code': 'grasp_failed', 'ok': False}, 'world': {}}
    sup = Supervisor(interface, Judge('retry'), mode='active', settle=0, approach_judge=approach)
    sup.step(interface.status())
    assert approach.steps == []
    assert interface.calls == [('stop', None, None, True)] and sup.halted


def test_approach_frame_none_goes_to_human():
    from vlm_supervisor.approach import ApproachJudge

    class Backend:
        def ask(self, *a, **k):
            raise AssertionError('backend must not be called without a frame')

    interface = ApproachInterface()
    interface.frame = lambda **k: None
    sup = Supervisor(interface, Judge(), mode='active', settle=0, approach_judge=ApproachJudge(Backend()))
    sup.step(interface.status())
    assert [c[0] for c in interface.calls] == ['stop'] and sup.halted


def test_approach_skips_non_grasp_lanes_and_recover_nodes():
    approach = Approach(BLOCKED)
    for node in ('tilt_blue', 'grasp_to_tilt_start', 'recover:grasp_blue:rollback_retry'):
        interface = ApproachInterface(node)
        Supervisor(interface, Judge(), mode='shadow', settle=0, approach_judge=approach).step(interface.status())
    assert approach.steps == []


class PendingRecoverInterface(FakeInterface):
    """deploy 가 복구를 걸어 둔 READY + pending: 승인은 recover, 매번 다시 실패한다."""
    @staticmethod
    def failure():
        return {'status': 'READY', 'node': 'grasp_to_tilt_start', 'next_action': 'recover:grasp_to_tilt_start:retry',
                'pending': {'node': 'grasp_to_tilt_start', 'action': 'retry', 'reason': 'slip'},
                'last': {'node': 'grasp_to_tilt_start', 'code': 'pour_failed', 'ok': False}}

    def perform(self, action, target=None, *, active=False, args=None):
        self.calls.append((action, target, active))
        if active and action == 'next':  # 복구 승인 = next(next_action) — 다시 실패해 새 시도로 걸린다
            self.state = {**self.failure(), 'attempts': {'grasp_to_tilt_start': len(self.calls)}}
        return {'ok': True}


def test_pending_recover_counts_toward_max_recover():
    interface = PendingRecoverInterface()
    supervisor = Supervisor(interface, Judge('next'), mode='active', settle=0, max_recover=2)
    assert supervisor.run() == 1
    assert [c[0] for c in interface.calls][:2] == ['next', 'next'] and interface.calls[2][0] in {'human', 'stop'}
