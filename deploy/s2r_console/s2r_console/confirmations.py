"""Displayed command approvals are bound to a run, lease and configuration."""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import asdict, replace
from pathlib import Path

class Confirmations:
    TTL_S = 120.0

    def __init__(self, console):
        self.console = console
        self.pending = {}

    def _error(self, text, code=409):
        from .console import ConsoleError
        return ConsoleError(text, code=code)

    def describe(self, action, parameters):
        from .console import EPISODE_NODE, STALE_S, MC
        from mission_stages import commands_for
        c, s = self.console, self.console._need()
        if action == 'stage':
            if set(parameters) - {'stage', 'restart'} or not isinstance(parameters.get('stage'), str):
                raise self._error('단계 설정이 올바르지 않다', 400)
            if 'restart' in parameters and not isinstance(parameters['restart'], bool):
                raise self._error('restart 는 true/false 여야 한다', 400)
            stage = c._current_stage(s, parameters['stage'])
            evidence = c._evidence(s)
            evidence = replace(evidence, approvals=evidence.approvals | {stage.id})
            gate = MC.gate(s.mission, stage.id, s.state, evidence)
            if not gate.ok:
                raise self._error('지금 실행할 수 없다: ' + '; '.join(gate.reasons))
            commands = commands_for(s.runbook, s.mission, stage.id, repo=c.repo, execute=True)
            targets = [p['key'] for p in s.supervisor.table() if p['alive'] and p['stage'] == stage.id] if parameters.get('restart') else []
            warnings = ['실기 로봇을 움직이는 명령이다'] if stage.touches_real else []
            if targets:
                warnings.append('재시작 전에 종료: ' + ', '.join(targets))
            return {'title': stage.title, 'commands': [asdict(cmd) for cmd in commands],
                    'warnings': warnings, 'stop_targets': targets}
        if action == 'episode':
            if set(parameters) - {'episode_action', 'want'}:
                raise self._error('에피소드 설정이 올바르지 않다', 400)
            requested = parameters.get('episode_action')
            if requested not in ('next', 'run', 'reset'):
                raise self._error('지원하지 않는 에피소드 동작', 400)
            with s.feed_lock:
                obs = s.feed.observed()
            state = obs.status.get(EPISODE_NODE)
            if state is None or obs.age_s.get(EPISODE_NODE, float('inf')) > STALE_S:
                raise self._error('에피소드 실행기 상태가 안 온다')
            want = state.get('next_action') if requested == 'next' else f"episode:{state.get('episode')}" if requested == 'run' else ''
            if requested != 'reset' and (state.get('busy') or not want):
                raise self._error('실행기가 실행 중이거나 다음 동작이 없다')
            return {'title': f'에피소드 {requested} {want}', 'commands': [{'episode_action': requested, 'approve': want}],
                    'warnings': [], 'want': want}
        description = c.actions.describe(action, parameters)
        if not description['enabled']:
            raise self._error('지금 실행할 수 없다: ' + '; '.join(description['reasons']))
        return description

    def basis(self, action, parameters, description):
        c, s = self.console, self.console._need()
        files = {str(s.profile.mission), *map(str, (c.repo / rel for rel in s.mission.artifacts.values()))}
        files.update(str(c.repo / ck['path']) for ck in s.mission.checkpoints.values())
        for path in (c.repo / 'config').glob('calib_profiles/*.yaml'):
            files.add(str(path))
        # Policy cards can refer to deployments other than the mission defaults.
        workspace = s.workspace.snapshot()
        for item in workspace.get('catalog', []):
            if any(card.get('policy_id') == item.get('id') for card in workspace.get('cards', [])):
                for key in ('contract_path', 'checkpoint_path', 'policy_file'):
                    if item.get(key):
                        files.add(str(c.repo / item[key]))
        digests = {path: s.cache.digest(Path(path), 'sha256') or 'missing' for path in sorted(files)}
        payload = {'run_id': s.run_id, 'action': action, 'parameters': parameters,
                   'commands': description.get('commands'), 'want': description.get('want'),
                   'stop_targets': description.get('stop_targets'), 'warnings': description.get('warnings'),
                   'files': digests, 'workspace': s.workspace.fingerprint()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def prepare(self, data, *, token):
        with self.console._lock:
            self.console.lease.check(token)
            if not isinstance(data.get('action'), str) or 'argv' in json.dumps(data):
                raise self._error('동작 이름과 등록된 설정만 받는다', 400)
            action = data['action']
            parameters = {k: v for k, v in data.items() if k != 'action'}
            description = self.describe(action, parameters)
            now = time.monotonic()
            self.pending = {key: value for key, value in self.pending.items() if value['expires'] > now}
            key = secrets.token_urlsafe(24)
            basis = self.basis(action, parameters, description)
            self.pending[key] = {'token': token, 'action': action, 'parameters': parameters,
                                 'basis': basis, 'expires': now + self.TTL_S}
            return {'ok': True, **description, 'confirmation_id': key, 'expires_in_s': self.TTL_S}

    def execute(self, key, *, token, operator):
        with self.console._lock:
            c, s = self.console, self.console._need()
            c.lease.check(token)
            record = self.pending.pop(key, None)
            if record is None or record['expires'] <= time.monotonic() or record['token'] != token:
                raise self._error('승인이 없거나 만료되었다 — 실행 내용을 다시 확인할 것')
            action, params = record['action'], record['parameters']
            description = self.describe(action, params)
            if record['basis'] != self.basis(action, params, description):
                raise self._error('계약 또는 설정이 바뀌었다 — 다시 승인할 것')
            if action == 'stage':
                stage = c._current_stage(s, params['stage'])
                if stage.touches_real:
                    c.approve(stage.id, operator=operator, typed=stage.id, note='일회용 명령 승인')
                try:
                    c.run_stage(stage.id, operator=operator, restart=params.get('restart', False))
                finally:
                    if stage.touches_real:
                        c.revoke(stage.id, operator=operator, note='일회용 승인 사용 완료')
            elif action == 'episode':
                c.episode(params['episode_action'], operator=operator, typed=description.get('want', ''))
            else:
                c.actions.execute(action, params, operator=operator)
            c._intent(s, 'actions/execute', {'action_id': action, 'parameters': params, 'operator': operator,
                                           'basis': record['basis']})
