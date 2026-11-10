"""ROS 없는 의사결정 코어. 실행기 계획을 수정하거나 업로드하지 않는다."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .interface import node_id

END = {'SUCCESS', 'FAILURE', 'STOPPED'}


class Supervisor:
    def __init__(self, interface, judge, mode='shadow', log_path=None, settle=0.5):
        if mode not in {'shadow', 'active'}:
            raise ValueError('mode must be shadow or active')
        self.interface, self.judge, self.mode = interface, judge, mode
        self.log_path = Path(log_path) if log_path else None
        self.settle = settle
        self.seen = None
        self.halted = False

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
                           ('episode_id', 'status', 'node', 'last', 'pending', 'history', 'attempts')}, sort_keys=True)

    def step(self, status):
        if self.halted or status.get('status') != 'READY' or status.get('busy'):
            return None
        if status.get('confirm'):
            return None  # Human confirmations require an external operator approval.
        key = self.key(status)
        if key == self.seen:
            return None
        self.seen = key
        last = status.get('last') or {}
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
        action = decision['action']
        if action == 'next':
            if status.get('pending') or (target and target.startswith('recover:')):
                action = 'recover'
            elif target is None:
                action = 'wait'
        # A slow VLM must not approve a different node after the human has advanced.
        latest = self.interface.status()
        if self.key(latest) != key:
            if not latest:
                self.seen = None  # status outage, not a real change: reconsider this state when it returns
            return self.log({'event': 'stale_decision', 'decision': decision})
        try:
            recovery_args = {'node_id': last.get('node') or target} if action == 'retry' else {}
            if action in {'retry', 'rescan', 'home', 'insert', 'human'}:
                result = self.interface.perform(action, target, active=self.mode == 'active', args=recovery_args)
            else:
                result = self.interface.perform(action, target, active=self.mode == 'active')
        except Exception as exc:
            result = {'ok': False, 'reasons': [str(exc)]}
        event = self.log({'event': 'decision', 'mode': self.mode, 'node': last.get('node'),
                          'target': target, 'action': action, 'decision': decision, 'result': result})
        if self.mode == 'active' and (action == 'stop' or result.get('error') == 'NotSupported'):
            self.halted = True
        elif self.mode == 'active' and result.get('ok') is False:
            self.log({'event': 'service_failure', 'result': self.interface.perform('stop', active=True)})
            self.halted = True
        if action in {'retry', 'rescan', 'home', 'insert', 'human'} and result.get('ok'):
            self.seen = None
        return event

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
        self.log({'event': 'final_judge', 'node': last.get('node'), 'decision': decision})
        return decision.get('action') == 'next'

    def run(self, status_wait=10.0):
        deadline = time.monotonic() + status_wait
        while True:
            status = self.interface.status()
            if not status:
                if time.monotonic() >= deadline:
                    self.log({'event': 'status_timeout', 'result': self.interface.perform(
                        'stop', active=self.mode == 'active')})
                    return 2
            else:
                deadline = time.monotonic() + status_wait
                if status.get('status') in END:
                    if status['status'] == 'SUCCESS' and not self.final_judged(status):
                        return 1
                    self.log({'event': 'end', 'status': status['status']})
                    return 0 if status['status'] == 'SUCCESS' else 1
                self.step(status)
                if self.halted:
                    return 1
            time.sleep(0.1)
