"""deploy 복구 표와 runner 상태의 보수적 해석(ROS 의존 없음)."""
from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)
# deploy 미설치 환경의 기존 코드 목록. 알려지지 않은 코드는 항상 safe_stop.
_FALLBACK_CODES = {
    'none', 'policy_timeout', 'target_not_found', 'grasp_failed_left', 'grasp_failed_right',
    'object_dropped_left', 'object_dropped_right', 'pour_failed', 'pour_partial', 'place_failed',
    'holder_not_found', 'lock_failed', 'lid_not_found', 'shake_aborted', 'pose_mismatch',
    'controller_error', 'observation_error', 'operator_declined', 'lift_guard_failed', 'unknown',
}
_FALLBACK_TABLE = {
    'target_not_found': ('retry', ('refresh_objects',)),
    'holder_not_found': ('retry', ('refresh_holders',)),
    'lid_not_found': ('retry', ('refresh_objects',)),
    'grasp_failed_left': ('rollback_retry', ('open_hand',)),
    'grasp_failed_right': ('rollback_retry', ('open_hand',)),
    'pose_mismatch': ('rollback_retry', ()),
    'pour_failed': ('retry', ()), 'lock_failed': ('retry', ()),
}


def load_deploy_policy():
    try:
        from policy_control.episode_failure import FailureCode, RecoveryManager, SAFETY, _TABLE  # pyright: ignore[reportMissingImports]
        # _TABLE is the module-level table consumed by RecoveryManager.decide.
        if not callable(RecoveryManager.decide):
            raise TypeError('RecoveryManager.decide unavailable')
        codes = {code.value for code in FailureCode}
        safety = {code.value for code in SAFETY} | {FailureCode.POLICY_TIMEOUT.value}
        table = {code.value: (action, tuple(pre)) for code, (action, pre) in _TABLE.items()
                 if code.value not in safety}
        return codes, table, 'deploy'
    except ImportError as exc:
        LOG.warning('deploy failure policy import failed; using hard-coded fallback: %s', exc)
        return _FALLBACK_CODES.copy(), _FALLBACK_TABLE.copy(), 'fallback'


KNOWN_CODES, TABLE, SOURCE = load_deploy_policy()
SAFE_STOP_CODES = KNOWN_CODES - TABLE.keys() - {'none'}
# Runner metadata codes, not FailureCode members. 'done' is used by the proto bridge.
NORMAL_CODES = {'none', 'done', 'confirmed', 'waiting_confirm'}
CONTINUED_CODE = 'place_failed_continued'


def _no_completed_node(status):
    """첫 READY: last 가 없거나, sim 브리지처럼 node·code 가 모두 None 인 빈 틀."""
    last = status.get('last')
    return not isinstance(last, dict) or (last.get('node') is None and last.get('code') is None)


def runner_detail(status):
    last = status.get('last')
    last = last if isinstance(last, dict) else {}
    code = last.get('code')
    if _no_completed_node(status):
        code = 'none'
    known = isinstance(code, str) and code in KNOWN_CODES | NORMAL_CODES | {CONTINUED_CODE}
    action, pre = TABLE.get(code, ('safe_stop', ())) if isinstance(code, str) else ('safe_stop', ())
    results = last.get('results')
    results = results if isinstance(results, list) else []
    rows: list[dict] = [r for r in results if isinstance(r, dict)]
    signals: list[dict] = []
    for row in rows:
        signal = row.get('signals')
        if isinstance(signal, dict):
            signals.append(signal)
    if isinstance(last.get('signals'), dict):
        signals.append(last['signals'])
    outcomes = [s.get('outcome') for s in signals if s.get('outcome') is not None]
    if last.get('outcome') is not None:
        outcomes.append(last['outcome'])
    reason = last.get('reason')
    text = ' '.join(str(r or '') for r in [code, reason, *outcomes, *(r.get('reason') for r in rows)]).lower()
    rules = []
    if 'opened_before_latch' in outcomes or 'opened before latch' in text:
        rules.append('opened_before_latch')
    if 'place_hand_unknown' in text or any(s.get('outcome') == 'opened_before_latch'
                                          and s.get('cup_in_holder') is False for s in signals):
        rules.append('place_hand_unknown')
    for s in signals:
        carry = s.get('ready_carry')
        if isinstance(carry, dict):
            if carry.get('still') == 'track':
                rules.append('track_abort')
            if carry.get('start_ok') is False:
                rules.append('start_ok_rejected')
    if 'track_abort' in text or "'still': 'track'" in text:
        rules.append('track_abort')
    if '준비 자세 이송 끝 문턱 미달' in text or 'start_ok rejection' in text:
        rules.append('start_ok_rejected')
    if any(t in text for t in ('jump_rejected', '홀더 재기', 'holder refresh', 'holder refresh rejected')):
        rules.append('holder_refresh_rejected')
    if 'rehome' in text and any(t in text for t in ('refused', '거부', '자리 모름', 'hand holds', 'no safe path')):
        rules.append('rehome_refused')
    failure = status.get('status') == 'FAILURE' or code not in NORMAL_CODES
    return {'code': code, 'reason': reason, 'outcome': last.get('outcome'), 'outcomes': outcomes,
            'results': results, 'unknown_code': not known,
            'termination_kind': 'rule' if rules else ('physical_or_unclassified' if failure else 'normal'),
            'rules': sorted(set(rules)), 'recovery': action, 'pre': list(pre), 'policy_source': SOURCE}


def allowed_actions(status, detail):
    """runner가 제시한 pending 복구만 승인. shell로 rollback/준비 단계를 우회하지 않는다."""
    pending = status.get('pending')
    if pending:
        last = status.get('last')
        last = last if isinstance(last, dict) else {}
        expected = f"recover:{last.get('node')}:{detail['recovery']}"
        if (isinstance(pending, dict) and not detail['unknown_code']
                and last.get('node') and detail['code'] != CONTINUED_CODE
                and status.get('status') == 'READY' and detail['recovery'] != 'safe_stop'
                and pending.get('action') == detail['recovery']
                and status.get('next_action') == expected):
            return {'recover', 'stop', 'human'}
        return {'stop', 'human'}
    if _no_completed_node(status) and status.get('status') == 'READY':
        return {'next', 'wait', 'stop', 'human'}
    if detail['unknown_code']:
        return {'stop', 'human'}
    if detail['code'] == CONTINUED_CODE and status.get('status') != 'FAILURE':
        return {'next', 'stop', 'human'}
    if detail['rules']:
        return {'stop', 'human'}
    if detail['code'] in NORMAL_CODES and status.get('status') != 'FAILURE':
        return {'next', 'wait', 'retry', 'rescan', 'home', 'insert', 'stop', 'human'}
    return {'stop', 'human'}
