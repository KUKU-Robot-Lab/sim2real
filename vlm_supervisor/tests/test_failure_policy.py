import builtins
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vlm_supervisor.supervisor import Supervisor
from vlm_supervisor.judge import Judge
from vlm_supervisor.tests.test_supervisor import FakeInterface, Judge as FixedJudge
from vlm_supervisor.tests.test_package import Backend


@pytest.mark.parametrize('code', ['place_failed', 'object_dropped_left', 'object_dropped_right',
                                  'pour_partial', 'shake_aborted', 'brand_new_code', None])
@pytest.mark.parametrize('action', ['retry', 'rescan', 'home', 'next'])
def test_no_retry_codes_never_execute_recovery(code, action):
    interface = FakeInterface()
    interface.state['last']['code'] = code
    event = Supervisor(interface, FixedJudge(action), mode='active', settle=0).step(interface.status())
    assert event['action'] == 'human'
    assert event['runner_detail']['recovery'] == 'safe_stop'


def test_rule_termination_details_and_none_are_preserved():
    interface = FakeInterface()
    interface.state['last'] = {'node': 'place', 'code': 'place_failed', 'reason': 'opened before latch',
                              'results': [None, {'reason': None, 'signals': None},
                                          {'signals': {'outcome': 'opened_before_latch',
                                                       'cup_in_holder': False, 'hand_empty': None}}]}
    event = Supervisor(interface, FixedJudge(), settle=0).step(interface.status())
    detail = event['runner_detail']
    assert detail['termination_kind'] == 'rule'
    assert 'place_hand_unknown' in detail['rules']
    assert detail['reason'] == 'opened before latch'
    assert detail['results'][-1]['signals']['hand_empty'] is None


@pytest.mark.parametrize('signal,rule', [({'ready_carry': {'still': 'track', 'start_ok': False}}, 'track_abort'),
                                       ({'ready_carry': {'start_ok': False}}, 'start_ok_rejected')])
def test_carry_guard_classified_from_signals(signal, rule):
    interface = FakeInterface()
    interface.state['last'] |= {'code': 'unknown', 'results': [{'signals': signal}]}
    event = Supervisor(interface, FixedJudge(), settle=0).step(interface.status())
    assert rule in event['runner_detail']['rules']


@pytest.mark.parametrize('reason,rule', [('jump_rejected', 'holder_refresh_rejected'),
                                        ('마지막 홀더 재기 failed', 'holder_refresh_rejected'),
                                        ('right rehome: 놓기 실패 컵 자리 모름', 'rehome_refused')])
def test_rule_reason_in_prompt_and_no_retry_offered(reason, rule):
    backend = Backend('{"success":false,"choice":"B","reason":"retry"}')
    decision = Judge(backend).judge({'status': 'FAILURE', 'last': {'code': 'unknown', 'reason': reason}}, b'image')
    prompt = backend.prompts[0]
    assert rule in prompt and reason in prompt
    assert 'B=failure/retry' not in prompt
    assert decision['action'] == 'stop'


def test_pending_rollback_uses_runner_recover_not_shell_retry():
    interface = FakeInterface()
    interface.state |= {'status': 'READY', 'pending': {'action': 'rollback_retry', 'checkpoint': 'cp'},
                        'next_action': 'recover:grasp_to_tilt_start:rollback_retry'}
    interface.state['last']['code'] = 'grasp_failed_right'
    event = Supervisor(interface, FixedJudge(), settle=0).step(interface.status())
    assert event['action'] == 'recover'


def test_unknown_code_logged(caplog):
    interface = FakeInterface()
    interface.state['last']['code'] = 'new_guard'
    event = Supervisor(interface, FixedJudge(), settle=0).step(interface.status())
    assert event['runner_detail']['unknown_code'] is True
    assert 'new_guard' in caplog.text


def test_fallback_warns_when_deploy_is_unavailable(monkeypatch, caplog):
    from vlm_supervisor.failure_policy import load_deploy_policy
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith('policy_control'):
            raise ImportError('deliberately unavailable')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', unavailable)
    codes, table, source = load_deploy_policy()
    assert source == 'fallback' and 'place_failed' in codes
    assert table['grasp_failed_right'][0] == 'rollback_retry'
    assert 'hard-coded fallback' in caplog.text


def test_real_deploy_table_import_without_ros():
    root = Path(__file__).resolve().parents[2]
    script = """
from vlm_supervisor import failure_policy as p
from policy_control.episode_failure import FailureCode, _TABLE, SAFETY
assert p.SOURCE == 'deploy'
assert p.KNOWN_CODES == {c.value for c in FailureCode}
for code in FailureCode:
    detail = p.runner_detail({'status': 'FAILURE', 'last': {'code': code.value}})
    expected = _TABLE.get(code, ('safe_stop', ()))[0]
    if code in SAFETY or code == FailureCode.POLICY_TIMEOUT:
        expected = 'safe_stop'
    assert detail['recovery'] == expected, code
"""
    proc = subprocess.run([sys.executable, '-c', script], cwd=root, capture_output=True, text=True,
                          env={**os.environ, 'PYTHONPATH': str(root / 'deploy/policy_control'),
                               'PYTHONDONTWRITEBYTECODE': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize('code', ['target_not_found', 'holder_not_found', 'lid_not_found',
                                  'grasp_failed_left', 'grasp_failed_right', 'pose_mismatch'])
def test_preparation_and_rollback_never_bypassed_by_shell(code):
    interface = FakeInterface()
    interface.state['last']['code'] = code
    sup = Supervisor(interface, FixedJudge(), mode='active', settle=0)
    assert sup.step(interface.status())['action'] == 'human'
    assert interface.calls[0][0] == 'stop'


def test_unknown_code_even_in_ready_is_closed():
    interface = FakeInterface()
    interface.state |= {'status': 'READY'}
    interface.state['last']['code'] = 'future_guard'
    assert Supervisor(interface, FixedJudge('next'), settle=0).step(interface.status())['action'] == 'human'


@pytest.mark.parametrize('reason', ['pour_failed — 재시도 1/1 다 씀',
                                   'pour_failed — 줄 노드는 복구 없이 정지(다른 줄도 멈춤)',
                                   '복구 준비 refresh_objects 실패: missing'])
def test_runner_no_retry_reason_cannot_be_bypassed(reason):
    interface = FakeInterface()
    interface.state['last']['reason'] = reason
    event = Supervisor(interface, FixedJudge(), settle=0).step(interface.status())
    assert event['action'] == 'human'


def test_pending_recovery_approves_runner_next_service(tmp_path):
    import json
    from vlm_supervisor.tests.test_package import FakeInterface as RealBoundary

    interface = RealBoundary(tmp_path / 'approvals.jsonl')
    interface.current |= {'pending': {'action': 'rollback_retry'},
                          'next_action': 'recover:grasp_blue:rollback_retry',
                          'last': {'node': 'grasp_blue', 'code': 'grasp_failed_right'}}
    event = Supervisor(interface, FixedJudge(), mode='active', settle=0).step(interface.status())
    assert event['action'] == 'recover' and event['result']['ok']
    assert interface.calls == ['next']
    assert json.loads(interface.approvals.read_text())['what'] == 'recover:grasp_blue:rollback_retry'


@pytest.mark.parametrize('code', ['pour_failed', 'lock_failed', 'target_not_found'])
def test_retry_without_pending_is_never_shell_retry(code):
    from vlm_supervisor.failure_policy import allowed_actions, runner_detail
    status = {'status': 'FAILURE', 'last': {'code': code}}
    assert allowed_actions(status, runner_detail(status)) == {'stop', 'human'}


@pytest.mark.parametrize('code,outcome,expected', [
    ('place_failed_continued', None, {'next', 'stop', 'human'}),
    ('place_failed_continued', 'opened_before_latch', {'next', 'stop', 'human'}),
    ('none', 'opened_before_latch', {'stop', 'human'}),
    ('none', 'place_hand_unknown', {'stop', 'human'}),
])
def test_ready_rule_actions_are_restricted(code, outcome, expected):
    from vlm_supervisor.failure_policy import allowed_actions, runner_detail
    status = {'status': 'READY', 'last': {'code': code, 'outcome': outcome}}
    assert allowed_actions(status, runner_detail(status)) == expected


@pytest.mark.parametrize('pending', [{'action': 'retry'}, {'action': 'rollback_retry'}])
def test_initial_ready_with_unmatched_pending_stops(pending):
    from vlm_supervisor.failure_policy import allowed_actions, runner_detail
    interface = FakeInterface()
    interface.state = {'status': 'READY', 'last': None, 'pending': pending,
                       'next_action': 'recover:missing:retry'}
    assert allowed_actions(interface.state, runner_detail(interface.state)) == {'stop', 'human'}
    event = Supervisor(interface, FixedJudge('next'), mode='active', settle=0).step(interface.status())
    assert event['action'] == 'human'
    assert interface.calls == [('stop', None, True)]


def test_next_conversion_cannot_approve_recover_without_pending():
    interface = FakeInterface()
    interface.state = {'status': 'READY', 'last': None, 'node': 'recover:missing:retry'}
    event = Supervisor(interface, FixedJudge(), mode='active', settle=0).step(interface.status())
    assert event['action'] == 'human'
    assert interface.calls == [('stop', None, True)]


@pytest.mark.parametrize('code', ['pour_failed', 'lock_failed'])
def test_retry_code_uses_only_matching_runner_recovery(code):
    interface = FakeInterface()
    interface.state |= {'status': 'READY', 'pending': {'action': 'retry'},
                        'next_action': 'recover:grasp_to_tilt_start:retry'}
    interface.state['last']['code'] = code
    event = Supervisor(interface, FixedJudge(), mode='active', settle=0).step(interface.status())
    assert event['action'] == 'recover'
    assert interface.calls == [('next', None, True)]


def test_deploy_runtime_errors_are_not_hidden_by_fallback(monkeypatch):
    from vlm_supervisor.failure_policy import load_deploy_policy
    original = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith('policy_control'):
            raise RuntimeError('broken deploy initialization')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', broken)
    with pytest.raises(RuntimeError, match='broken deploy initialization'):
        load_deploy_policy()


def test_active_fallback_logs_error_event(monkeypatch, tmp_path, caplog):
    import json
    monkeypatch.setattr('vlm_supervisor.supervisor.SOURCE', 'fallback')
    log = tmp_path / 'events.jsonl'
    Supervisor(FakeInterface(), FixedJudge(), mode='active', log_path=log)
    event = json.loads(log.read_text())
    assert event['event'] == 'error' and event['policy_source'] == 'fallback'
    assert 'active supervisor uses fallback failure policy' in caplog.text


def test_initial_ready_with_empty_last_frame_allows_next():
    # sim 브리지는 첫 상태에서 last 를 node·code None 인 틀로 보낸다.
    from vlm_supervisor.failure_policy import allowed_actions, runner_detail
    status = {'status': 'READY', 'pending': None, 'next_action': 'go_home_start',
              'last': {'node': None, 'code': None, 'reason': None}}
    detail = runner_detail(status)
    assert not detail['unknown_code']
    assert 'next' in allowed_actions(status, detail)


@pytest.mark.parametrize('status', [
    {'status': 'FAILURE', 'pending': None, 'last': {'node': None, 'code': None, 'reason': None}},
    {'status': 'READY', 'pending': None, 'last': {'node': None, 'code': None, 'reason': 'rehome refused'}},
])
def test_empty_node_code_with_failure_or_reason_does_not_allow_next(status):
    from vlm_supervisor.failure_policy import allowed_actions, runner_detail
    assert 'next' not in allowed_actions(status, runner_detail(status))
