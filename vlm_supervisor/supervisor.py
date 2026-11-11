"""ROS 없는 의사결정 코어. 실행기 계획을 수정하거나 업로드하지 않는다."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time

from .interface import node_id
from .planner import load_cards
from .failure_policy import SOURCE, allowed_actions, runner_detail

END = {'SUCCESS', 'STOPPED'}
RECOVERY = {'retry', 'rescan', 'home', 'insert', 'human'}


class Supervisor:
    def __init__(self, interface, judge, mode='shadow', log_path=None, settle=0.5, max_recover=3,
                 approach_judge=None):
        if mode not in {'shadow', 'active'}:
            raise ValueError('mode must be shadow or active')
        self.interface, self.judge, self.mode = interface, judge, mode
        self.log_path = Path(log_path) if log_path else None
        self.settle = settle
        self.max_recover = max_recover
        self.recoveries = {}
        self.seen = None
        self.halted = False
        self.approach_judge = approach_judge
        self.approach_seen = None
        if mode == 'active' and SOURCE == 'fallback':
            logging.getLogger(__name__).error('active supervisor uses fallback failure policy')
            self.log({'event': 'error', 'mode': mode, 'policy_source': SOURCE,
                      'reason': 'active supervisor uses fallback failure policy'})
        # grasp 카드의 노드 예시: id → (손, 물체). 손은 정책 이름 끝 _r/_l.
        self.grasp_nodes = {} if approach_judge is None else {
            n['id']: ({'_r': 'right', '_l': 'left'}.get(str(n.get('name', ''))[-2:]), n.get('target_object'))
            for card in load_cards()['스킬'] if card['id'] == 'grasp'
            for n in card['실행_방법']['노드_예시']}

    def log(self, event):
        row = {'t': time.time(), **event}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row

    @staticmethod
    def key(status):
        return json.dumps({name: status.get(name) for name in
                           ('episode_id', 'status', 'node', 'last', 'pending', 'history', 'attempts', 'next_action')}, sort_keys=True)

    def step(self, status):
        if self.halted or status.get('status') not in {'READY', 'FAILURE'} or status.get('busy'):
            return None
        if status.get('confirm'):
            return None  # Human confirmations require an external operator approval.
        key = self.key(status)
        if key == self.seen:
            return None
        self.seen = key
        last = status.get('last') or {}
        last = last if isinstance(last, dict) else {}
        detail = runner_detail(status)
        allowed = allowed_actions(status, detail)
        if detail['unknown_code'] and (last or status.get('status') == 'FAILURE'):
            logging.getLogger(__name__).warning('unknown runner failure code: %r', detail['code'])
        if last:
            try:
                image = self.interface.frame(after=time.monotonic() + self.settle,
                                             wait=self.settle + 3)
                decision = self.judge.judge(status, image)
            except Exception as exc:
                decision = {'choice': 'E', 'success': False, 'action': 'stop',
                            'reason': f'VLM/frame failure: {exc}'}
        else:
            decision = {'choice': 'A', 'action': 'next', 'reason': 'initial READY; no completed node'}
        target = node_id(status.get('node'))
        action = 'stop' if decision.get('choice') == 'E' else decision['action']
        failed_node = node_id(last.get('node')) or target
        capped = action in RECOVERY and self.recoveries.get(failed_node, 0) >= self.max_recover
        if 'recover' in allowed and action in {'next', 'retry', 'rescan', 'home'}:
            action = 'recover'  # deploy owns pre-actions, rollback checkpoint and retry budget
        if action not in allowed:
            capped = True
        if capped:
            action = 'human'
        if action == 'next':
            if status.get('pending') or (target and target.startswith('recover:')):
                action = 'recover'
            elif target is None:
                action = 'wait'
        if action not in allowed:
            action = 'human' if 'human' in allowed else 'stop'
        # 접근 판정 대상: next 로 갈 노드, 또는 retry 로 다시 돌 노드(재실행도 접근이다).
        approach_node = None
        if action == 'next' and status.get('status') == 'READY':
            nxt = node_id(status.get('next_action'))
            approach_node = target or (nxt if nxt in {node_id(n) for n in status.get('nodes') or []} else None)
        elif action == 'retry':
            approach_node = failed_node
        approach_reason = self.check_approach(status, approach_node) if approach_node else None
        approach_stop = approach_reason is not None and self.mode == 'active'  # shadow: only logs
        if approach_stop:
            action, failed_node = 'human', approach_node
        # A slow VLM must not approve a different node after the human has advanced.
        latest = self.interface.status()
        if self.key(latest) != key:
            if not latest:
                self.seen = None  # status outage, not a real change: reconsider this state when it returns
            self.approach_seen = None  # the approach verdict was not acted on: judge again
            return self.log({'event': 'stale_decision', 'decision': decision})
        try:
            recovery_args = {'node_id': failed_node} if action in RECOVERY else {}
            if action == 'human':
                result = self.interface.perform('stop', target, active=self.mode == 'active')
            elif action == 'recover':
                result = self.interface.perform('next', status['next_action'], active=self.mode == 'active')
            elif action in RECOVERY:
                result = self.interface.perform(action, target, active=self.mode == 'active', args=recovery_args)
            else:
                result = self.interface.perform(action, target, active=self.mode == 'active')
        except Exception as exc:
            result = {'ok': False, 'reasons': [str(exc)]}
        event = self.log({'event': 'decision', 'mode': self.mode, 'node': last.get('node'),
                          'target': target, 'action': action, 'decision': decision, 'result': result, 'runner_detail': detail,
                          **({'approach': approach_reason} if approach_stop else {})})
        if approach_node and not approach_stop and self.mode == 'active' and result.get('ok'):
            self.approach_seen = None  # 판정을 쓴 시도는 소비: 재시도·재기동 때 다시 판정한다
        if self.mode == 'active' and (action in {'stop', 'human'} or approach_stop or result.get('error') == 'NotSupported'):
            self.halted = True
        elif self.mode == 'active' and result.get('ok') is False:
            self.log({'event': 'service_failure', 'result': self.interface.perform('stop', active=True)})
            self.halted = True
        if self.mode == 'active' and action in RECOVERY and result.get('ok'):
            self.recoveries[failed_node] = self.recoveries.get(failed_node, 0) + 1
            if capped:
                self.halted = True
            elif status.get('status') == 'READY':
                self.seen = None
        return event

    def check_approach(self, status, node):
        """grasp 카드 노드 직전 경로 판정. 사람에게 넘길 사유를 돌려준다. 통과·대상 아님·이미 본 시도면 None.

        lanes 노드(grasp_to_tilt_start 등)는 손·물체를 몰라 판정하지 않는다."""
        if self.approach_judge is None or node not in self.grasp_nodes:
            return None
        hand, obj = self.grasp_nodes[node]
        # 에피소드·시도 횟수를 모르면 같은 시도인지 알 수 없어 중복 제거를 하지 않는다.
        if status.get('episode_id') is not None and status.get('attempts') is not None:
            seen = (status.get('episode_id'), node, json.dumps(status['attempts'].get(node)
                                                               if isinstance(status['attempts'], dict) else None))
            if seen == self.approach_seen:
                return None
            self.approach_seen = seen
        try:
            image = self.interface.frame(after=time.monotonic() + self.settle, wait=self.settle + 3)
            result = self.approach_judge.check(image, ['grasp', hand, obj, '-'], status.get('world'))
        except Exception as exc:  # noqa: BLE001 - judge failure must not let the approach run
            result = {'valid': False, 'verdict': 'unknown', 'action': 'human', 'error': f'{type(exc).__name__}: {exc}'}
        clear = (result.get('valid') and result.get('verdict') == 'clear' and result.get('action') == 'proceed'
                 and not result.get('disagree') and not result.get('step_mismatch'))
        if clear:
            reason = None
        elif not result.get('valid'):
            reason = 'invalid/error: ' + str(result.get('error'))
        elif result.get('disagree') or result.get('step_mismatch'):
            reason = 'VLM and geometry/step disagree'
        elif result.get('action') == 'reorder':
            reason = 'reorder NotSupported: runner cannot change node order'
        elif result.get('action') == 'other_hand':
            reason = 'other_hand NotSupported: no execution path'
        else:
            reason = 'blocked: human'
        geom = result.get('geom')
        # geom none = 좌표가 없어 VLM 단독 판정.
        self.log({'event': 'approach', 'mode': self.mode, 'node': node, 'step': ['grasp', hand, obj, '-'],
                  'outcome': 'proceed' if clear else 'human', 'reason': reason,
                  'geom': geom.get('verdict') if isinstance(geom, dict) else 'none',
                  'result': {k: v for k, v in result.items() if k != 'meta'}})
        return reason

    def final_judged(self, status):
        """Runner closes terminal right after the last action, so READY is never seen for it: judge that action here."""
        last = status.get('last') or {}
        if not last.get('node') or self.key(status) == self.seen:
            return True
        self.seen = self.key(status)
        try:
            image = self.interface.frame(after=time.monotonic() + self.settle, wait=self.settle + 3)
            decision = self.judge.judge(status, image)
        except Exception as exc:
            decision = {'choice': 'E', 'success': False, 'action': 'stop', 'reason': f'VLM/frame failure: {exc}'}
        self.log({'event': 'final_judge', 'node': last.get('node'), 'decision': decision,
                  'runner_detail': runner_detail(status)})
        return (decision.get('action') == 'next'
                and 'next' in allowed_actions(status, runner_detail(status)))

    def run(self, status_wait=10.0, ready_timeout=60.0):
        recovery_key = None
        deadline = time.monotonic() + status_wait
        while True:
            status = self.interface.status()
            if not status:
                if time.monotonic() >= deadline:
                    self.log({'event': 'status_timeout', 'result': self.interface.perform(
                        'stop', active=self.mode == 'active')})
                    return 2
            else:
                if recovery_key is not None:
                    if self.key(status) == recovery_key:
                        if time.monotonic() >= deadline:
                            self.log({'event': 'status_timeout', 'result': self.interface.perform(
                                'stop', active=self.mode == 'active')})
                            self.log({'event': 'end', 'status': 'FAILURE'})
                            return 1
                        time.sleep(0.1)
                        continue
                    recovery_key = None
                    self.seen = None
                deadline = time.monotonic() + status_wait
                if status.get('status') in END:
                    if status['status'] == 'SUCCESS' and not self.final_judged(status):
                        return 1
                    self.log({'event': 'end', 'status': status['status']})
                    return 0 if status['status'] == 'SUCCESS' else 1
                event = self.step(status)
                recovered = (event is not None and event['event'] == 'decision'
                             and event['action'] in RECOVERY and event['result'].get('ok')
                             and self.mode == 'active' and not self.halted)
                if recovered:
                    recovery_key = self.key(status)
                    deadline = time.monotonic() + max(status_wait, ready_timeout)
                elif status.get('status') == 'FAILURE' and not status.get('busy'):
                    if event is not None and event['event'] == 'stale_decision':
                        continue
                    self.log({'event': 'end', 'status': 'FAILURE'})
                    return 1
                if self.halted:
                    return 1
            time.sleep(0.1)
