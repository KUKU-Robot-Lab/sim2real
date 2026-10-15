"""Approved management actions built from the robot's registered runbook.

HTTP callers supply identifiers and bounded settings, never command lines.  The
legacy runner retains process keys, launch checks, and stop-on-first-failure.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import json
import math
import os
import signal
import threading
import time
import yaml

from .console import ConsoleError
from .console_state import STALE_S
from .runner import StageRunner, DONE
from . import pd_names as PD
from mission_stages import Command, commands_for
import mission_core as MC


class ActionManager:
    def __init__(self, console, *, settle_s=2.0):
        self.console = console
        self.settle_s = settle_s
        self._status = {'active': False, 'outcome': None, 'steps': []}
        self._lock = threading.RLock()

    def view(self):
        with self._lock:
            result = copy.deepcopy(self._status)
            runner = getattr(self, '_runner', None)
            if result['active'] and runner is not None:
                result['steps'] = runner.view()['steps']
            return result

    def abort(self):
        self._cancel_policy.set() if hasattr(self, '_cancel_policy') else None
        if hasattr(self, '_cancel_scan'):
            self._cancel_scan.set()
        scanning = self._status.get('id') == 'perception_refresh' and self._status.get('active')
        if scanning:
            for proc in self.console._need().supervisor.alive():
                if proc.stage.startswith('action_locate_'):
                    try:
                        os.killpg(proc.popen.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        runner = getattr(self, '_runner', None)
        if runner is not None and runner.active and not self._status.get('id', '').startswith('rg3_') and not scanning:
            runner.abort()
        thread = getattr(self, '_thread', None)
        if thread is not None:
            thread.join(timeout=110 if scanning else 35)
            if thread.is_alive():
                raise ConsoleError('관리 작업이 아직 정지되지 않았다')

    def reset(self):
        self.abort()
        with self._lock:
            self._status = {'active': False, 'outcome': None, 'steps': []}
            self._runner = None

    def _specs(self, session):
        known = {stage.id: stage for stage in session.mission.stages}
        specs = {}
        for name in ('preflight', 'drivers', 'head_home', 'cups', 'shutdown'):
            if name in known:
                specs[name] = (known[name].title, 'setting', (name,))
        for side in ('right', 'left'):
            bundle = tuple(f'{part}_{side}' for part in ('hand', 'hand_check', 'pd_load', 'pd_arm'))
            if all(name in known for name in bundle):
                specs[f'connect_{side}'] = (f'{side} 팔·손 연결', 'setting', bundle)
            for part in ('home', 'rehome', 'return'):
                name = f'{part}_{side}'
                if name in known:
                    specs[name] = (known[name].title, 'control', (name,))
            if f'home_{side}' in known:
                specs[f'engage_{side}'] = (f'{side} PD engage', 'control', (f'home_{side}',))
        if 'head_control' in known:
            specs['head_control'] = (known['head_control'].title, 'control', ('head_control',))
        if getattr(session, 'workspace', None) is not None:
            specs['perception_refresh'] = ('선택 물체 인지 재추출', 'control', ())
            specs['policy_stop'] = ('정책 정지 · 기록 마무리', 'control', ())
            for card in session.workspace.cards:
                sid = f"episode_grasp_rg3_{card['side']}"
                if card.get('execution_supported') and sid in known:
                    specs['rg3_' + card['id']] = (f"rg3 {card['side']} {card['object_id']}", 'policy', (sid,))
        return specs

    @staticmethod
    def _card(s, action_id):
        return next(c for c in s.workspace.cards if 'rg3_' + c['id'] == action_id)

    def catalog(self):
        s = self.console._need()
        return [self.describe(name) for name in self._specs(s)]

    def describe(self, action_id, parameters=None):
        parameters = dict(parameters or {})
        if set(parameters) - {'object_ids', 'pan', 'tilt', 'calib_profile'}:
            raise ConsoleError('지원하지 않는 작업 설정', code=400)
        if 'object_ids' in parameters and (not isinstance(parameters['object_ids'], list) or
                                          any(not isinstance(value, str) for value in parameters['object_ids'])):
            raise ConsoleError('물체 ID 목록이 필요하다', code=400)
        s = self.console._need()
        spec = self._specs(s).get(action_id)
        if spec is None:
            raise ConsoleError(f'모르는 관리 작업: {action_id}', code=404)
        title, group, stages = spec
        reasons = []
        interruptible = (self._status.get('id', '').startswith('rg3_') and group == 'control') or (
            self._status.get('id') == 'perception_refresh' and action_id == 'head_control')
        if self.view()['active'] and not interruptible:
            reasons.append('다른 관리 작업이 실행 중이다')
        active = [r.stage_id for r in s.runners.values() if r.active]
        nonpolicy = [sid for sid in active if not sid.startswith(('policy_', 'episode_'))]
        if nonpolicy:
            reasons.append(f"단계 {', '.join(nonpolicy)} 가 실행 중이다")
        if (group == 'control' and action_id not in ('perception_refresh', 'head_control', 'policy_stop')) or group == 'policy':
            side = self._card(s, action_id)['side'] if group == 'policy' else action_id.rsplit('_', 1)[-1]
            with s.feed_lock:
                obs = s.feed.observed()
            st = obs.status.get(PD.name(side))
            if not st or obs.age_s.get(PD.name(side), 1e9) > STALE_S:
                reasons.append(f'{side} PD 실시간 상태가 없다 — 발행 모드 연결과 상태 수신 필요')
            elif not st.get('execute', False):
                reasons.append(f'{side} PD 가 무발행 모드다')
            elif st.get('phase') in ('HOLD', 'UNKNOWN', 'RELEASING', 'RAMPING'):
                reasons.append(f"{side} PD 상태 {st.get('phase')} — 안전 복구/정착 필요")
            if obs.joints_age_s is None or obs.joints_age_s > STALE_S:
                reasons.append('관절 실측이 없거나 오래됐다')
            joint_ages = getattr(obs, 'joint_ages', {})
            if joint_ages:
                expected = [n for values in s.contract_sides.values() if values.get('side') == side
                            for kind in ('arm_joints', 'hand_joints') for n in values.get(kind, [])]
                ages = dict(joint_ages)
                for source, canonical in s.joint_alias.items():
                    if source in ages:
                        ages[canonical] = ages[source]
                if any(ages.get(n, 1e9) > STALE_S for n in expected):
                    reasons.append(f'{side} 팔 또는 손 관절 실측이 없거나 오래됐다')
        if action_id.startswith('connect_'):
            if not any(p['alive'] and p['stage'] == 'drivers' for p in s.supervisor.table()):
                reasons.append('양팔 드라이버가 연결되지 않았다')
        if action_id == 'head_control':
            head = {**s.workspace.head, **parameters}
            for axis, low, high in (('pan', -90, 90), ('tilt', -60, 30)):
                value = head.get(axis)
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                    reasons.append(f'{axis} 설정이 운용 한계를 벗어났다')
        if group == 'policy':
            card = self._card(s, action_id)
            obj = next((o for o in s.workspace.objects if o['id'] == card['object_id']), None)
            if obj is None:
                return {'id': action_id, 'title': title, 'group': group, 'enabled': False,
                        'reasons': ['정책 물체를 선택할 것'], 'commands': [], 'warnings': [], 'stages': list(stages), 'parameters': parameters}
            result = obj['results']['fpp']
            policy = next(p for p in s.workspace.catalog if p['id'] == card['policy_id'])
            if not obj['enabled']:
                reasons.append('정책 물체를 활성화할 것')
            if not s.workspace.perception['fpp'] or obj['input_source'] != 'fpp':
                reasons.append('현재 rg3는 FPP 실시간 추적을 요구한다 — locate 입력만으로 실행할 수 없다')
            if not result['valid'] or not result.get('timestamp') or time.time() - result['timestamp'] > 1:
                reasons.append('선택 물체의 신선한 FPP 추적 좌표가 없다')
            if not result.get('live'):
                reasons.append('rg3는 FPP live_pose 추적을 요구한다 — 저장한 pose 좌표만으로 실행할 수 없다')
            if policy['status'] not in ('verified', 'deployed') or policy.get('issues'):
                reasons.append('선택 정책이 배포 검사를 통과하지 않았다')
        if action_id == 'perception_refresh':
            ids = parameters.get('object_ids')
            selected = [o for o in s.workspace.objects if o['enabled'] and (ids is None or o['id'] in ids)]
            if not selected:
                reasons.append('인지를 실행할 물체를 선택할 것')
            if ids is not None and (not isinstance(ids, list) or any(i not in {o['id'] for o in selected} for i in ids)):
                reasons.append('활성화한 등록 물체만 인지할 수 있다')
            if not any(s.workspace.perception.values()):
                reasons.append('FPP 또는 locate 인지 노드를 선택할 것')
        commands, warnings = [], []
        if self.view()['active'] and self._status.get('id') == 'perception_refresh' and action_id == 'head_control':
            warnings.append('목 스캔 취소 → 현재 이동과 운용 자세 복귀 정착 확인 → 새 목 자세 실행')
        for stage_index, sid in enumerate(stages):
            stage = MC.stage_by_id(s.mission, sid)
            independent = replace(stage, needs=() if group != 'setting' else stage.needs, touches_real=False)
            mission = replace(s.mission, stages=tuple(independent if x.id == sid else x for x in s.mission.stages))
            planned = replace(s.state, completed=tuple(dict.fromkeys((*s.state.completed, *stages[:stage_index]))))
            reasons.extend(MC.gate(mission, sid, planned, self.console._evidence(s)).reasons)
            for index, cmd in enumerate(commands_for(s.runbook, s.mission, sid, repo=self.console.repo, execute=True)):
                if cmd.manual:
                    warnings.append(cmd.note)
                else:
                    effective = self._transform(s, action_id, parameters, cmd)
                    commands.append({'stage': sid, 'index': index, 'note': effective.note,
                                     'argv': list(effective.argv), 'stop': list(effective.stop), 'background': effective.background})
        targets = self._policy_targets(s)
        if group == 'control' and targets:
            warnings.append('관련 정책과 실행기를 정지 → PD 현재 자세 유지 확인 → 요청 실행. 정책은 자동 재개하지 않는다.')
        if action_id == 'shutdown':
            warnings.append('팔을 받침에 놓으십시오. PD 해제와 팔 드라이버 종료로 모터 토크가 해제됩니다.')
            with s.feed_lock:
                observed = s.feed.observed()
            procs = {p['key']: p for p in s.supervisor.table()}
            for side in ('right', 'left'):
                if self.console._pd_phase(s, observed, procs, side=side) == 'UNKNOWN':
                    reasons.append(f'{side} PD 상태를 모른다 — 토크 해제 확인 전 종료할 수 없다')
        if action_id == 'perception_refresh':
            for source, cmd in self._perception_commands(s, parameters):
                commands.append({'stage': source, 'index': 0, 'note': cmd.note,
                                 'argv': list(cmd.argv), 'stop': [], 'background': False})
        if action_id.startswith('engage_'):
            commands = [c for c in commands if 'pd_engage' in c['argv']]
            if not commands:
                reasons.append('이 로봇에 독립 engage 명령이 등록되지 않았다')
        return {'id': action_id, 'title': title, 'group': group, 'enabled': not reasons,
                'card_id': self._card(s, action_id)['id'] if group == 'policy' else None,
                'reasons': reasons, 'commands': commands, 'warnings': warnings,
                'stages': list(stages), 'parameters': parameters, 'stop_targets': targets if group == 'control' else []}

    @staticmethod
    def _policy_targets(s):
        return [p['key'] for p in s.supervisor.table() if p['alive'] and
                p['stage'].startswith(('policy_', 'episode_'))]

    def execute(self, action_id, parameters=None, *, operator):
        if self.view()['active'] and self._status.get('id') == 'perception_refresh' and action_id == 'head_control':
            preview = self.describe(action_id, parameters)
            if not preview['enabled']:
                raise ConsoleError('목 제어 설정을 실행할 수 없다', reasons=tuple(preview['reasons']))
            self._cancel_scan.set()
            for proc in self.console._need().supervisor.alive():
                if proc.stage.startswith('action_locate_'):
                    try:
                        os.killpg(proc.popen.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            self._thread.join(timeout=100)
            if self._thread.is_alive():
                raise ConsoleError('목 스캔 취소 후 정착을 확인하지 못했다')
        # A control intervention cancels the monitored episode and waits for its
        # stop/hold/record cleanup before the new command can begin.
        if self.view()['active'] and self._status.get('id', '').startswith('rg3_'):
            preview = self.describe(action_id, parameters)
            if preview['group'] == 'control' and preview['enabled']:
                self._cancel_policy.set()
                self._thread.join(timeout=30)
                if self._thread.is_alive() or self._status.get('outcome') == 'FAILED':
                    raise ConsoleError('정책 정지와 PD 자세 유지 확인이 실패했다')
        with self._lock:
            description = self.describe(action_id, parameters)
            if not description['enabled']:
                raise ConsoleError('작업을 지금 실행할 수 없다', reasons=tuple(description['reasons']))
            s = self.console._need()
            self._status = {'active': True, 'id': action_id, 'operator': operator,
                            'outcome': None, 'stage': None, 'note': '', 'steps': []}
            self._cancel_policy = threading.Event()
            self._cancel_scan = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(s, description), daemon=True)
            self._thread.start()

    def _commands(self, s, sid, description):
        original = commands_for(s.runbook, s.mission, sid, repo=self.console.repo, execute=True)
        # Preserve indexes: manual checks were acknowledged in this action's preview.
        commands = [replace(cmd, manual=False, argv=('true',)) if cmd.manual else cmd for cmd in original]
        if description['id'] == 'shutdown' and not {'head_home', 'head_control'} & set(s.state.completed):
            commands = [Command(argv=('true',)) if '--release' in cmd.argv and
                        any('head_home.py' in a for a in cmd.argv) else cmd for cmd in commands]
        if description['id'].startswith('engage_'):
            commands = [cmd if 'pd_engage' in cmd.argv else Command(argv=('true',)) for cmd in commands]
        return [self._transform(s, description['id'], description['parameters'], c) for c in commands]

    def _transform(self, s, action_id, parameters, cmd):
        if action_id == 'head_control' and any('head_home.py' in a for a in cmd.argv):
            head = {**s.workspace.head, **parameters}
            cmd = replace(cmd, argv=(*cmd.argv, '--pan', str(head['pan']), '--tilt', str(head['tilt'])))
        if action_id.startswith('rg3_'):
            card = self._card(s, action_id)
            obj = next(o for o in s.workspace.objects if o['id'] == card['object_id'])
            policy = next(p for p in s.workspace.catalog if p['id'] == card['policy_id'])
            argv = tuple(f"contract:={policy['contract']}" if a.startswith('contract:=') else
                         f"cup_topic:=/objects/{obj['id']}/live_pose" if a.startswith('cup_topic:=') else a for a in cmd.argv)
            episode_path = str(s.run_dir / f"rg3_{card['id']}.yaml")
            argv = tuple(f'episode:={episode_path}' if a.startswith('episode:=') else a for a in argv)
            if '--episode' in argv:
                values = list(argv)
                values[values.index('--episode') + 1] = episode_path
                argv = tuple(values)
            if any('episode_runner_node.py' in a for a in argv):
                argv += ('-p', f"policy_object:={obj['policy_object']}", '-p', f"cup_object:={obj['id']}")
            cmd = replace(cmd, argv=argv)
            cmd = replace(cmd, argv=tuple(a.replace('/objects/source240b_blue/', f"/objects/{obj['id']}/") for a in cmd.argv))
        return cmd

    def _run_commands(self, s, sid, commands):
        with self._lock:
            self._status['stage'] = sid
            self._runner = StageRunner(sid, commands, s.supervisor, on_done=lambda *args: None,
                                       settle_s=self.settle_s)
            runner = self._runner
        runner.start()
        runner.join()
        with self._lock:
            self._status['steps'] = runner.view()['steps']
        if runner.outcome != DONE:
            raise ConsoleError(f'{sid}: {runner.note}')

    def _trigger(self, s, service, side, *, expect=()):
        from .console import _TRIGGER
        flag = '--episode-ns' if service.startswith('episode/') else '--side'
        argv = ['python3', str(_TRIGGER), service, flag, side, '--execute', '--service-timeout', '5']
        if expect:
            argv += ['--expect-pd', *expect, '--phase-timeout', '10']
        self._run_commands(s, f'action_{service.replace("/", "_")}_{side}', [Command(tuple(argv), note=service)])

    def _preempt(self, s):
        targets = self._policy_targets(s)
        if not targets:
            return
        # Stop the sequencer first; otherwise it could advance while PD is held.
        from .console import _EPISODE_CMD
        with s.feed_lock:
            sequencer = s.feed.observed().status.get('episode_runner')
        if sequencer:
            self._run_commands(s, 'action_episode_stop', [Command(
                ('python3', str(_EPISODE_CMD), 'stop', '--execute', '--operator', self._status['operator']),
                note='에피소드 실행기 정지')])
        sides = self.console._pd_sides(s)
        if not sides:
            raise ConsoleError('정책 정지 후 PD 상태를 확인할 수 없다')
        for side in sides:
            self._trigger(s, 'episode/stop', side)
            began_ns = time.time_ns()
            self._trigger(s, 'pd/hold', side, expect=('TRACKING',))
            self._confirm_hold(s, side, began_ns)
        for runner in s.runners.values():
            if runner.active and runner.stage_id.startswith(('policy_', 'episode_')):
                runner.abort()
                runner.join(timeout=8)
        s.supervisor.stop(targets)
        if any(s.supervisor.is_alive(key) for key in targets):
            raise ConsoleError('정책 프로세스 정지 실패 — 요청을 실행하지 않는다')

    @staticmethod
    def _confirm_hold(s, side, began_ns):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with s.feed_lock:
                obs = s.feed.observed()
            status = obs.status.get(PD.name(side), {})
            if (obs.age_s.get(PD.name(side), 1e9) <= STALE_S and status.get('t_pub_ns', 0) > began_ns
                    and status.get('phase') == 'TRACKING' and status.get('target') == 'internal'):
                return
            time.sleep(.1)
        raise ConsoleError(f'{side} PD 현재 자세 유지 확인 실패 — 후속 명령을 실행하지 않는다')

    def _run(self, s, description):
        outcome, note = DONE, ''
        try:
            action = description['id']
            if description['group'] == 'control' or action == 'shutdown':
                self._preempt(s)
            if action == 'shutdown':
                for side in self.console._pd_sides(s):
                    self._trigger(s, 'pd/release', side, expect=('IDLE',))
            if action == 'perception_refresh':
                self._refresh(s, description['parameters'])
            for sid in description['stages']:
                if description['group'] == 'policy':
                    self._run_policy(s, sid, description)
                else:
                    self._run_commands(s, sid, self._commands(s, sid, description))
                if not (description['group'] == 'policy' and self._cancel_policy.is_set()):
                    with self.console._lock:
                        s.state = replace(s.state, completed=MC.settle_completed(s.mission, s.state.completed, sid))
            if action == 'head_control' and description['parameters']:
                s.workspace.update({'head': description['parameters']})
            if action == 'shutdown':
                # Also cover managed children added after the generated stop list.
                for prefix in ('episode', 'policy', 'cups', 'head', 'pd', 'hand', 'drivers'):
                    keys = [p['key'] for p in s.supervisor.table() if p['alive'] and p['stage'].startswith(prefix)]
                    if keys:
                        s.supervisor.stop(keys)
        except Exception as exc:  # retain failure and exact bundle position in the UI
            outcome, note = 'FAILED', str(exc)
            s.event('action', f"✗ {description['id']}: {note}")
            if description['id'] != 'shutdown' and description['group'] != 'policy':
                failure_stage, failure_runner = self._status.get('stage'), getattr(self, '_runner', None)
                with s.feed_lock:
                    observed = s.feed.observed()
                side = description['id'].rsplit('_', 1)[-1]
                for candidate in self.console._pd_sides(s):
                    if side in ('left', 'right') and side != candidate:
                        continue
                    if observed.status.get(PD.name(candidate), {}).get('phase') == 'TRACKING':
                        try:
                            self._trigger(s, 'pd/hold', candidate, expect=('TRACKING',))
                        except Exception as hold_error:
                            note += f' · 자세 유지 요청 실패: {hold_error}'
                self._status['stage'], self._runner = failure_stage, failure_runner
        finally:
            with self._lock:
                self._status.update(active=False, outcome=outcome, note=note)
            s.event('action', f"{description['id']} {outcome}")

    def _run_policy(self, s, sid, description):
        from .console import _EPISODE_CMD
        original = commands_for(s.runbook, s.mission, sid, repo=self.console.repo, execute=True)
        commands = self._commands(s, sid, description)
        card = self._card(s, description['id'])
        doc = yaml.safe_load((self.console.repo / f"config/episodes/grasp_rg3_{card['side']}.yaml").read_text())
        for policy in doc['policies'].values():
            policy['policy'] = card['policy_id']
        obj = next(o for o in s.workspace.objects if o['id'] == card['object_id'])
        doc['objects']['CUP']['topic'] = f"/objects/{obj['id']}/pose"
        for node in doc['sequence']:
            if node.get('type') == 'policy':
                node['policy_object'] = obj['policy_object']
        (s.run_dir / f"rg3_{card['id']}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        manuals = [i for i, c in enumerate(original) if c.manual]
        if len(manuals) < 2:
            raise ConsoleError('rg3 에피소드 완료/정리 경계가 등록되지 않았다')
        boundary = manuals[-1]
        launched = False
        try:
            launched = True
            self._run_commands(s, sid, commands[:boundary])
            deadline = time.monotonic() + 15
            while True:
                with s.feed_lock:
                    obs = s.feed.observed()
                status = obs.status.get('episode_runner')
                if status and obs.age_s.get('episode_runner', 1e9) <= STALE_S:
                    break
                if time.monotonic() >= deadline:
                    raise ConsoleError('rg3 실행기 상태 수신 시간 초과')
                time.sleep(.1)
            self._run_commands(s, 'action_episode_run', [Command(
                ('python3', str(_EPISODE_CMD), 'run', '--execute', '--operator', self._status['operator'],
                 '--approve', f"episode:{status['episode']}"), note='승인한 rg3 에피소드 실행')])
            deadline = time.monotonic() + 300
            while True:
                if self._cancel_policy.is_set():
                    self._preempt(s)
                    break
                with s.feed_lock:
                    obs = s.feed.observed()
                status = obs.status.get('episode_runner', {})
                if obs.age_s.get('episode_runner', 1e9) > STALE_S:
                    raise ConsoleError('rg3 실행기 상태가 오래됐다')
                if status.get('phase') in ('SUCCESS', 'FAILURE', 'STOPPED', 'ABORTED'):
                    if status['phase'] != 'SUCCESS':
                        raise ConsoleError(f"rg3 종료: {status['phase']}")
                    break
                if time.monotonic() > deadline:
                    raise ConsoleError('rg3 실행 시간 초과')
                time.sleep(.1)
        finally:
            if launched:
                # A service failure blocks cleanup rather than dropping a live
                # external target and letting the watchdog release the posture.
                self._preempt(s)
                self._run_commands(s, f'{sid}_cleanup', commands[boundary + 1:])

    def _refresh(self, s, parameters):
        workspace = s.workspace
        ids = parameters.get('object_ids')
        selected = [o for o in workspace.objects if o['enabled'] and (ids is None or o['id'] in ids)]
        for source in ('fpp', 'locate'):
            if self._cancel_scan.is_set():
                raise ConsoleError('목 스캔 취소 후 운용 자세 정착 완료')
            if not workspace.perception[source]:
                continue
            for obj in selected:
                workspace.set_result(obj['id'], source, {'valid': False, 'reason': 'acquisition in progress'})
            try:
                if source == 'fpp':
                    supported = [o['id'] for o in selected if o['model'] != 'holder']
                    if supported:
                        self._run_commands(s, 'action_fpp', [Command(tuple(workspace.fpp_argv(supported)), note='FPP 재추출')])
                else:
                    for obj in selected:
                        if self._cancel_scan.is_set():
                            raise ConsoleError('목 스캔 취소 후 운용 자세 정착 완료')
                        if not obj['locate_supported']:
                            workspace.set_result(obj['id'], source, {'valid': False, 'reason': obj['locate_reason']})
                            continue
                        key = f"action_locate_{obj['id']}"
                        self._run_commands(s, key, [Command(tuple(workspace.locate_argv(obj['id'])), note='locate 재추출')])
                        tail = s.supervisor.tail(f'{key}#0', 32768)
                        result = next((json.loads(line) for line in reversed(tail.splitlines()) if line.startswith('{')), None)
                        if not result:
                            raise ConsoleError('locate 결과가 없다')
                        workspace.set_result(obj['id'], source, result)
            except Exception:
                for obj in selected:
                    workspace.set_result(obj['id'], source, {'valid': False, 'reason': 'acquisition failed'})
                raise

    @staticmethod
    def _perception_commands(s, parameters):
        workspace = s.workspace
        ids = parameters.get('object_ids')
        selected = [o for o in workspace.objects if o['enabled'] and (ids is None or o['id'] in ids)]
        commands = []
        if workspace.perception['fpp']:
            supported = [o['id'] for o in selected if o['model'] != 'holder']
            if supported:
                commands.append(('action_fpp', Command(tuple(workspace.fpp_argv(supported)), note='FPP 재추출')))
        if workspace.perception['locate']:
            for obj in selected:
                if obj['locate_supported']:
                    commands.append((f"action_locate_{obj['id']}", Command(tuple(workspace.locate_argv(obj['id'])), note='locate 재추출')))
        return commands
