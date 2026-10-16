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
import re
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


ARMS = ('right', 'left')
ALL_RESOURCES = frozenset({'right', 'left', 'head', 'perception', 'drivers'})


class _Job:
    """One management action.  Jobs on disjoint resources run side by side."""

    def __init__(self, action_id, group, operator, resources):
        self.id, self.group, self.resources = action_id, group, frozenset(resources)
        self.status = {'active': True, 'id': action_id, 'group': group, 'operator': operator,
                       'outcome': None, 'stage': None, 'note': '', 'steps': [],
                       'resources': sorted(self.resources), 'started': time.time()}
        self.cancel_policy = threading.Event()
        self.cancel_scan = threading.Event()
        self.runner = None
        self.thread = None

    @property
    def active(self):
        return self.status['active']

    def view(self):
        result = copy.deepcopy(self.status)
        if result['active'] and self.runner is not None:
            result['steps'] = self.runner.view()['steps']
        return result


class ActionManager:
    def __init__(self, console, *, settle_s=2.0):
        self.console = console
        self.settle_s = settle_s
        self._jobs = {}           # action id -> latest job for that action
        self._order = []          # action ids, most recently started last
        self._lock = threading.RLock()
        self._scan_restore_ok = True
        self._tls = threading.local()
        self._profile_cache = {}  # calib profile path -> (mtime, summary)
        self._detached = _Job('', '', 'direct', ())   # direct helper calls outside a job thread
        self._detached.status = {'active': False, 'id': '', 'operator': 'direct', 'outcome': None, 'steps': []}

    # The helpers below run inside a job thread and address "their" job.
    def _job(self):
        return getattr(self._tls, 'job', None) or self._detached

    @property
    def _status(self):
        return self._job().status

    @property
    def _cancel_scan(self):
        return self._job().cancel_scan

    @_cancel_scan.setter
    def _cancel_scan(self, event):
        self._job().cancel_scan = event

    @property
    def _cancel_policy(self):
        return self._job().cancel_policy

    @_cancel_policy.setter
    def _cancel_policy(self, event):
        self._job().cancel_policy = event

    @property
    def _runner(self):
        return self._job().runner

    @_runner.setter
    def _runner(self, runner):
        self._job().runner = runner

    def active_jobs(self):
        with self._lock:
            return [job for job in self._jobs.values() if job.active]

    def view(self):
        """Every job, plus the most recently started one at top level (older callers read that)."""
        with self._lock:
            if not self._jobs:
                return {'active': False, 'outcome': None, 'steps': []}
            jobs = [self._jobs[key].view() for key in self._order if key in self._jobs]
            result = dict(jobs[-1])
            result['active'] = any(job['active'] for job in jobs)
            result['jobs'] = jobs
            return result

    def workspace_busy(self, patch):
        """A workspace edit must not change the inputs of a running perception/head/policy job."""
        for job in self.active_jobs():
            if job.id.startswith('rg3_') and set(patch) <= {'head'}:
                continue
            if job.resources & {'head', 'perception'} or job.id.startswith('rg3_'):
                return job.id
        return None

    def abort(self):
        jobs = self.active_jobs()
        for job in jobs:
            job.cancel_policy.set()
            job.cancel_scan.set()
            if job.id == 'perception_refresh':
                self._kill_scan()
            elif job.runner is not None and job.runner.active and not job.id.startswith('rg3_'):
                job.runner.abort()
        for job in jobs:
            if job.thread is not None:
                job.thread.join(timeout=110 if job.id == 'perception_refresh' else 35)
                if job.thread.is_alive():
                    raise ConsoleError(f'관리 작업 {job.id} 이(가) 아직 정지되지 않았다')

    def _kill_scan(self):
        for proc in self.console._need().supervisor.alive():
            if proc.stage.startswith('action_locate_'):
                try:
                    os.killpg(proc.popen.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def reset(self):
        self.abort()
        with self._lock:
            self._jobs, self._order = {}, []
            self._scan_restore_ok = True

    def _resources(self, s, action_id, group, parameters):
        if action_id == 'shutdown':
            return ALL_RESOURCES
        if action_id == 'drivers':
            return frozenset({'drivers', *ARMS})
        if action_id == 'preflight':                 # regenerates the contract both arms load
            return frozenset(ARMS)
        if action_id in ('head_home', 'head_control'):
            return frozenset({'head'})
        if action_id == 'cups':
            return frozenset({'head', 'perception'})
        if action_id == 'perception_refresh':      # locate moves the neck; FPP needs it still
            return frozenset({'perception', 'head'})
        if action_id == 'policy_stop':
            return frozenset(ARMS)
        if group == 'policy':                         # live FPP input + the arm
            return frozenset({self._card(s, action_id)['side'], 'head', 'perception'})
        side = action_id.rsplit('_', 1)[-1]
        return frozenset({side}) if side in ARMS else frozenset({action_id})

    @staticmethod
    def _interrupts(job, action_id, group):
        """A control command cancels a policy episode; a head command cancels a scan."""
        return (job.id.startswith('rg3_') and group == 'control') or (
            job.id == 'perception_refresh' and action_id == 'head_control')

    def _conflicts(self, s, action_id, group, parameters):
        resources = self._resources(s, action_id, group, parameters)
        return [job for job in self.active_jobs()
                if job.id == action_id or (job.resources & resources and not self._interrupts(job, action_id, group))]

    @staticmethod
    def _action_sides(resources):
        return {side for side in ARMS if side in resources}

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
        resources = self._resources(s, action_id, group, parameters)
        for job in self._conflicts(s, action_id, group, parameters):
            if job.id == action_id:
                reasons.append(f'{action_id} 이(가) 이미 실행 중이다')
            else:
                shared = ', '.join(sorted(job.resources & resources))
                reasons.append(f'{job.id} 실행 중 — 같은 대상({shared})을 쓴다')
        scanning = any(job.id == 'perception_refresh' for job in self.active_jobs())
        if action_id in ('head_control', 'perception_refresh') and not scanning and not self._scan_restore_ok:
            reasons.append('이전 목 스캔의 운용 자세 복귀 정착을 확인하지 못했다 — head_home 복구 필요')
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
            for axis, low, high in (('pan', -90, 90), ('tilt', -60, -10)):
                value = head.get(axis)
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                    reasons.append(f'{axis} 설정이 운용 한계를 벗어났다')
            profile = head.get('calib_profile', '')
            if profile and (not isinstance(profile, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', profile) or
                            not (s.repo / 'config/calib_profiles' / f'{profile}.yaml').is_file()):
                reasons.append('선택한 카메라 보정 프로필이 등록되지 않았다')
            known = self.head_profiles(s).get(profile) if profile else None
            if known is not None and known.get('error'):
                reasons.append(f"보정 프로필을 읽지 못했다: {known['error']}")
            elif known is not None and known['strict'] and all(isinstance(head.get(k), (int, float)) for k in ('pan', 'tilt')):
                tol = known['tolerance']
                if not any(abs(a['pan'] - head['pan']) <= tol and abs(a['tilt'] - head['tilt']) <= tol for a in known['angles']):
                    reasons.append(f"pan {head['pan']} · tilt {head['tilt']} 는 {profile} 가 보정한 각도(±{tol}°)가 아니다 "
                                   "— 이 자세에서는 locate · FP++ 좌표를 쓸 수 없다")
            drivers = [unit for unit in s.units.values() if any('head_joint_publisher.py' in a for a in unit.argv)]
            if drivers and not any(s.supervisor.is_alive(unit.key) for unit in drivers):
                reasons.append('목 드라이버가 연결되지 않았다 — setting head_home 필요')
        if group == 'policy':
            card = self._card(s, action_id)
            obj = next((o for o in s.workspace.objects if o['id'] == card['object_id']), None)
            if obj is None:
                return {'id': action_id, 'title': title, 'group': group, 'enabled': False,
                        'reasons': ['정책 물체를 선택할 것'], 'commands': [], 'warnings': [], 'stages': list(stages), 'parameters': parameters}
            result = obj['results']['fpp']
            policy = next((p for p in s.workspace.catalog if p['id'] == card['policy_id']), None)
            if policy is None:
                return {'id': action_id, 'title': title, 'group': group, 'enabled': False,
                        'reasons': ['정책 파일이 없거나 등록되지 않았다'], 'commands': [], 'warnings': [],
                        'stages': list(stages), 'parameters': parameters, 'card_id': card['id']}
            if not obj['enabled']:
                reasons.append('정책 물체를 활성화할 것')
            if not s.workspace.perception['fpp'] or obj['input_source'] != 'fpp':
                reasons.append('현재 rg3는 FPP 실시간 추적을 요구한다 — locate 입력만으로 실행할 수 없다')
            if not result['valid'] or not result.get('timestamp') or time.time() - result['timestamp'] > 1:
                reasons.append('선택 물체의 신선한 FPP 추적 좌표가 없다')
            if not result.get('live'):
                reasons.append('rg3는 FPP live_pose 추적을 요구한다 — 저장한 pose 좌표만으로 실행할 수 없다')
            if (policy['status'] not in ('candidate', 'verified', 'deployed') or
                    not policy.get('contract') or policy.get('issues')):
                reasons.append('선택 정책을 실행할 수 없다 — 계약·등록부 검사와 hold 상태를 확인할 것')
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
        if group == 'policy' and policy['status'] == 'candidate':
            warnings.append('candidate 정책 실기 시험 — 실기 검증 전 정책입니다. 계약·등록부 검사 후 운영자 승인으로 실행합니다.')
        if scanning and action_id == 'head_control':
            warnings.append('목 스캔 취소 → 현재 이동과 운용 자세 복귀 정착 확인 → 새 목 자세 실행')
        for job in self.active_jobs():
            if job.id.startswith('rg3_') and group == 'control' and job.resources & resources:
                warnings.append(f'{job.id} 에피소드를 멈추고 정리한 뒤 실행한다')
        for stage_index, sid in enumerate(stages):
            stage = MC.stage_by_id(s.mission, sid)
            independent = replace(stage, needs=() if group != 'setting' else stage.needs, touches_real=False)
            mission = replace(s.mission, stages=tuple(independent if x.id == sid else x for x in s.mission.stages))
            planned = replace(s.state, completed=tuple(dict.fromkeys((*s.state.completed, *stages[:stage_index]))))
            reasons.extend(MC.gate(mission, sid, planned, self.console._evidence(s)).reasons)
            try:
                stage_commands = commands_for(s.runbook, s.mission, sid, repo=self.console.repo, execute=True)
            except (KeyError, ValueError, OSError) as exc:
                reasons.append(f'{sid}: 명령 구성 불가: {exc}')
                continue
            for index, cmd in enumerate(stage_commands):
                if cmd.manual:
                    warnings.append(cmd.note)
                else:
                    effective = self._transform(s, action_id, parameters, cmd)
                    commands.append({'stage': sid, 'index': index, 'note': effective.note,
                                     'argv': list(effective.argv), 'stop': list(effective.stop), 'background': effective.background})
        targets = self._policy_targets(s, self._preempt_sides(s, action_id, group, parameters)) if group == 'control' else []
        if group == 'control' and targets:
            try:
                needed = self._policy_sides(s, targets)
                missing = needed - set(self.console._pd_sides(s))
                if missing:
                    reasons.append(f"정책 대상 {', '.join(sorted(missing))} PD 상태가 없거나 오래됐다")
            except ConsoleError as exc:
                reasons.append(str(exc))
            warnings.append('정지 대상: ' + ', '.join(targets) + ' → PD 현재 자세 유지 확인 → 요청 실행. 정책은 자동 재개하지 않는다.')
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
        if action_id in ('head_control', 'cups', 'perception_refresh'):
            for key, command in self._converter_updates(s, parameters):
                commands.append({'stage': 'cups', 'index': int(key.split('#')[1]), 'note': '카메라 좌표 변환 보정 변경',
                                 'argv': list(command.argv), 'stop': [key], 'background': True})
        if action_id.startswith('engage_'):
            commands = [c for c in commands if 'pd_engage' in c['argv']]
            if not commands:
                reasons.append('이 로봇에 독립 engage 명령이 등록되지 않았다')
        return {'id': action_id, 'title': title, 'group': group, 'enabled': not reasons,
                'job': self._job_summary(action_id), 'live': self._live(s, action_id, stages),
                'card_id': self._card(s, action_id)['id'] if group == 'policy' else None,
                'reasons': reasons, 'commands': commands, 'warnings': warnings,
                'stages': list(stages), 'parameters': parameters, 'stop_targets': targets if group == 'control' else [],
                **({'calib_profiles': self.head_profiles(s)} if action_id == 'head_control' else {})}

    def head_profiles(self, s):
        """config/calib_profiles 의 보정 각도표 — 목 제어는 camera 세션이 보정한 각도로만 간다(10.15 사용자: 그래야 locate 사용 가능)."""
        import calib_profile as CP
        out = {}
        for path in sorted((s.repo / 'config/calib_profiles').glob('*.yaml')):
            try:
                mtime = path.stat().st_mtime
                cached = self._profile_cache.get(path)
                if cached is None or cached[0] != mtime:
                    prof = CP.load_profile(path)
                    summary = {'strict': bool(prof.strict), 'tolerance': float(prof.angle_tolerance_deg),
                               'angles': [{'pan': round(e.pan_deg, 3), 'tilt': round(e.tilt_deg, 3), 'n_bolts': int(e.n_bolts),
                                           'resid_mm': None if not math.isfinite(e.bolt_resid_mm) else round(e.bolt_resid_mm, 2)}
                                          for e in prof.entries]}
                    self._profile_cache[path] = cached = (mtime, summary)
                out[path.stem] = cached[1]
            except Exception as exc:      # 한 파일이 깨져도 나머지 프로필은 보인다
                out[path.stem] = {'error': str(exc), 'strict': True, 'tolerance': 0.0, 'angles': []}
        return out

    def _job_summary(self, action_id):
        job = self._jobs.get(action_id)
        if job is None:
            return None
        return {k: job.status.get(k) for k in ('active', 'outcome', 'note', 'stage', 'started', 'finished')}

    def _live(self, s, action_id, stages):
        """What this action left running: its background processes, and for an arm its PD phase."""
        procs = [p for p in s.supervisor.table() if p['stage'] in stages]
        alive = [p.get('note') or p['key'] for p in procs if p['alive']]
        dead = [f"{p.get('note') or p['key']} rc={p.get('rc')}" for p in procs
                if not p['alive'] and not p.get('stopped') and p.get('rc') not in (0, None) and p.get('background')]
        result = {'alive': alive, 'dead': dead}
        side = action_id.rsplit('_', 1)[-1]
        if action_id.startswith(('connect_', 'home_', 'rehome_', 'return_', 'engage_')) and side in ARMS:
            with s.feed_lock:
                obs = s.feed.observed()
            status = obs.status.get(PD.name(side))
            fresh = status and obs.age_s.get(PD.name(side), 1e9) <= STALE_S
            result['pd'] = (status.get('phase') if fresh else '상태 없음') if status else '상태 없음'
            hold = (status or {}).get('hold') or {}
            if fresh and hold.get('err') is not None:
                result['pd_err'] = round(float(hold['err']), 3)
                result['pd_settled'] = bool(hold.get('settled'))
        return result

    def _policy_targets(self, s, sides=None):
        targets = [p['key'] for p in s.supervisor.table() if p['alive'] and
                   p['stage'].startswith(('policy_', 'episode_'))]
        if sides is None:
            return targets
        kept = []
        for key in targets:
            try:
                if self._policy_sides(s, [key]) & set(sides):
                    kept.append(key)
            except ConsoleError:
                kept.append(key)       # side unknown: treat as touching this arm
        return kept

    def _preempt_sides(self, s, action_id, group, parameters):
        """Arm commands stop only that arm's policies; head/perception/stop-all commands stop every policy."""
        if group == 'control':
            sides = self._action_sides(self._resources(s, action_id, group, parameters))
            if len(sides) == 1 and action_id != 'policy_stop':
                return sides
        return None

    def execute(self, action_id, parameters=None, *, operator):
        s = self.console._need()
        preview = self.describe(action_id, parameters)
        group = preview['group']
        resources = self._resources(s, action_id, group, preview['parameters'])
        # A head command cancels a running scan and waits for the return pose to settle; a control
        # command on the same arm cancels the monitored episode and waits for stop/hold/record cleanup.
        for job in self.active_jobs():
            if job.id == action_id or not (job.resources & resources) or not self._interrupts(job, action_id, group):
                continue
            if not preview['enabled']:
                raise ConsoleError('작업을 지금 실행할 수 없다', reasons=tuple(preview['reasons']))
            if job.id == 'perception_refresh':
                job.cancel_scan.set()
                self._kill_scan()
                job.thread.join(timeout=100)
                if job.thread.is_alive() or not self._scan_restore_ok:
                    raise ConsoleError('목 스캔 취소 후 정착을 확인하지 못했다')
            else:
                job.cancel_policy.set()
                job.thread.join(timeout=30)
                if job.thread.is_alive() or job.status.get('outcome') == 'FAILED':
                    raise ConsoleError('정책 정지와 PD 자세 유지 확인이 실패했다')
        with self._lock:
            description = self.describe(action_id, parameters)
            if not description['enabled']:
                raise ConsoleError('작업을 지금 실행할 수 없다', reasons=tuple(description['reasons']))
            job = _Job(action_id, description['group'], operator,
                       self._resources(s, action_id, description['group'], description['parameters']))
            self._jobs[action_id] = job
            self._order = [key for key in self._order if key != action_id] + [action_id]
            job.thread = threading.Thread(target=self._run_job, args=(job, s, description), daemon=True,
                                          name=f'action-{action_id}')
            job.thread.start()

    def _run_job(self, job, s, description):
        self._tls.job = job
        try:
            self._run(s, description)
        finally:
            self._tls.job = None

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
        if action_id in ('head_control', 'cups') and any('head_move.py' in a for a in cmd.argv):
            head = {**s.workspace.head, **parameters}
            cmd = replace(cmd, argv=(*cmd.argv, '--pan', str(head['pan']), '--tilt', str(head['tilt'])))
        if any('object_pose_node.py' in a for a in cmd.argv):
            argv = list(cmd.argv)
            if '--head-joint-topic' not in argv:
                argv += ['--head-joint-topic', '/head/joint_states']
            if '--calib-profile' in argv:
                index = argv.index('--calib-profile')
                del argv[index:index + 2]
            profile = parameters.get('calib_profile', s.workspace.head['calib_profile'])
            if profile:
                argv += ['--calib-profile', profile]
            cmd = replace(cmd, argv=tuple(argv))
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

    def _run_commands(self, s, sid, commands, *, process_keys=None):
        with self._lock:
            self._status['stage'] = sid
            self._runner = StageRunner(sid, commands, s.supervisor, on_done=lambda *args: None,
                                       settle_s=self.settle_s)
            runner = self._runner
            if process_keys is not None:
                for step, key in zip(runner.steps, process_keys):
                    step.key = key
        runner.start()
        runner.join()
        with self._lock:
            self._status['steps'] = runner.view()['steps']
        if runner.outcome != DONE:
            raise ConsoleError(f'{sid}: {runner.note}')

    def _converter_updates(self, s, parameters):
        if not any(st.id == 'cups' for st in s.mission.stages):
            return []
        commands = commands_for(s.runbook, s.mission, 'cups', repo=self.console.repo, execute=True)
        procs = {p['key']: p for p in s.supervisor.table()}
        updates = []
        for index, command in enumerate(commands):
            if not command.background or not any('object_pose_node.py' in a for a in command.argv):
                continue
            key = f'cups#{index}'
            existing = procs.get(key)
            if not existing or not existing['alive']:
                continue
            desired = self._transform(s, 'head_control', parameters, command)
            if tuple(existing['argv']) == desired.argv:
                continue
            updates.append((key, desired))
        return updates

    def _sync_converter(self, s, parameters):
        """Keep the existing mission process key when changing camera calibration."""
        for key, desired in self._converter_updates(s, parameters):
            s.workspace.invalidate_results('camera conversion calibration changed')
            s.supervisor.stop([key])
            if s.supervisor.is_alive(key):
                raise ConsoleError('카메라 좌표 변환 노드 정지 실패')
            self._run_commands(s, 'cups', [desired], process_keys=[key])

    def _trigger(self, s, service, side, *, expect=()):
        from .console import _TRIGGER
        flag = '--episode-ns' if service.startswith('episode/') else '--side'
        argv = ['python3', str(_TRIGGER), service, flag, side, '--execute', '--service-timeout', '5']
        if expect:
            argv += ['--expect-pd', *expect, '--phase-timeout', '10']
        self._run_commands(s, f'action_{service.replace("/", "_")}_{side}', [Command(tuple(argv), note=service)])

    def _preempt(self, s, sides=None):
        targets = self._policy_targets(s, sides)
        if not targets:
            return
        needed = self._policy_sides(s, targets)
        fresh = set(self.console._pd_sides(s))
        missing = needed - fresh
        if missing:
            raise ConsoleError(f"정책 대상 {', '.join(sorted(missing))} PD 상태가 없거나 오래됐다 — 정지/자세 유지 확인 필요")
        # Stop the sequencer first; otherwise it could advance while PD is held.
        from .console import _EPISODE_CMD
        with s.feed_lock:
            sequencer = s.feed.observed().status.get('episode_runner')
        if sequencer:
            self._run_commands(s, 'action_episode_stop', [Command(
                ('python3', str(_EPISODE_CMD), 'stop', '--execute', '--operator', self._status['operator']),
                note='에피소드 실행기 정지')])
        for side in sorted(needed):
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
    def _policy_sides(s, targets):
        needed = set()
        unknown = []
        for proc in s.supervisor.table():
            if proc['key'] not in targets:
                continue
            sides = set(re.findall(r'(?:^|_)(right|left)(?:_|$)', proc['stage']))
            stage = next((st for st in s.mission.stages if st.id == proc['stage']), None)
            if stage:
                lane = next((la for la in s.mission.lanes if la.id == stage.lane), None)
                if lane and lane.side in ('right', 'left'):
                    sides.add(lane.side)
                elif lane and lane.id == 'both':
                    sides.update(('right', 'left'))
            for arg in proc.get('argv', []):
                if arg.startswith(('ns:=', 'side:=')) and arg.split(':=', 1)[1] in ('right', 'left'):
                    sides.add(arg.split(':=', 1)[1])
                if arg.startswith('contract:='):
                    path = s.repo / arg.split(':=', 1)[1]
                    try:
                        contract = json.loads(path.read_text())
                    except (OSError, ValueError):
                        continue
                    for name, spec in contract.get('sides', {}).items():
                        side = spec.get('side', name)
                        if side in ('right', 'left'):
                            sides.add(side)
            if not sides:
                unknown.append(proc['key'])
            needed.update(sides)
        if unknown:
            # Ancillary recorders share their policy's stage, so a stage whose
            # side cannot be established must not be silently treated as free.
            raise ConsoleError('정책 대상 팔을 확인할 수 없다: ' + ', '.join(unknown))
        return needed

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
                sides = self._preempt_sides(s, action, description['group'], description['parameters'])
                self._preempt(s) if sides is None else self._preempt(s, sides)
            if action == 'shutdown':
                for side in self.console._pd_sides(s):
                    self._trigger(s, 'pd/release', side, expect=('IDLE',))
            if action in ('head_home', 'head_control'):
                s.workspace.invalidate_results('head motion or calibration changed')
            if action == 'perception_refresh':
                self._sync_converter(s, description['parameters'])
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
            if action == 'head_home':
                self._scan_restore_ok = True
            if action in ('head_control', 'cups'):
                self._sync_converter(s, description['parameters'])
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
                own = self._action_sides(self._job().resources)
                for candidate in self.console._pd_sides(s):
                    if candidate not in own:       # never disturb an arm another job may be driving
                        continue
                    if observed.status.get(PD.name(candidate), {}).get('phase') == 'TRACKING':
                        try:
                            self._trigger(s, 'pd/hold', candidate, expect=('TRACKING',))
                        except Exception as hold_error:
                            note += f' · 자세 유지 요청 실패: {hold_error}'
                self._status['stage'], self._runner = failure_stage, failure_runner
        finally:
            with self._lock:
                job = self._job()
                if job.runner is not None:
                    job.status['steps'] = job.runner.view()['steps']
                job.status.update(active=False, outcome=outcome, note=note, finished=time.time())
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
                    self._preempt(s, {card['side']})
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
                self._preempt(s, {card['side']})
                self._run_commands(s, f'{sid}_cleanup', commands[boundary + 1:])

    def _refresh(self, s, parameters):
        workspace = s.workspace
        ids = parameters.get('object_ids')
        selected = [o for o in workspace.objects if o['enabled'] and (ids is None or o['id'] in ids)]
        failures = []
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
                        workspace.invalidate_results('FPP acquisition restarted', sources=('fpp',))
                        start = next(c for key, c in self._perception_commands(s, parameters) if key == 'action_fpp_start')
                        self._run_commands(s, 'action_fpp_start', [start])
                        self._run_commands(s, 'action_fpp', [Command(tuple(workspace.fpp_argv(supported)), note='FPP 재추출')])
                else:
                    for obj in selected:
                        if self._cancel_scan.is_set():
                            raise ConsoleError('목 스캔 취소 후 운용 자세 정착 완료')
                        if not obj['locate_supported']:
                            workspace.set_result(obj['id'], source, {'valid': False, 'reason': obj['locate_reason']})
                            continue
                        key = f"action_locate_{obj['id']}"
                        self._scan_restore_ok = False
                        try:
                            self._run_commands(s, key, [Command(tuple(workspace.locate_argv(obj['id'])), note='locate 재추출')])
                        finally:
                            tail = s.supervisor.tail(f'{key}#0', 32768)
                            result = next((json.loads(line) for line in reversed(tail.splitlines()) if line.startswith('{')), None)
                            self._scan_restore_ok = bool(result and result.get('home_ok') is True)
                        if not result:
                            raise ConsoleError('locate 결과가 없다')
                        workspace.set_result(obj['id'], source, result)
            except Exception as exc:
                for obj in selected:
                    workspace.set_result(obj['id'], source, {'valid': False, 'reason': f'acquisition failed: {exc}'})
                if source != 'fpp' or self._cancel_scan.is_set():
                    raise
                failures.append(f'FPP: {exc}')
        if failures:
            raise ConsoleError(' · '.join(failures))

    @staticmethod
    def _perception_commands(s, parameters):
        workspace = s.workspace
        ids = parameters.get('object_ids')
        selected = [o for o in workspace.objects if o['enabled'] and (ids is None or o['id'] in ids)]
        commands = []
        if workspace.perception['fpp']:
            supported = [o['id'] for o in selected if o['model'] != 'holder']
            if supported:
                all_enabled = [o['id'] for o in workspace.objects if o['enabled'] and o['model'] != 'holder']
                commands.append(('action_fpp_start', Command(('python3', str(s.repo / 'scripts/ops/perception_ctl.py'),
                                                              'start', *all_enabled, '--viewer', '--wait', '150', '--require-all'),
                                                             note='선택한 모든 FPP 묶음 연결 확인')))
                commands.append(('action_fpp', Command(tuple(workspace.fpp_argv(supported)), note='FPP 재추출')))
        if workspace.perception['locate']:
            for obj in selected:
                if obj['locate_supported']:
                    commands.append((f"action_locate_{obj['id']}", Command(tuple(workspace.locate_argv(obj['id'])), note='locate 재추출')))
        return commands
