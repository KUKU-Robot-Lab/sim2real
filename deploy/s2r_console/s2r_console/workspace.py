"""Run-scoped perception and policy workspace; selecting cards never runs a robot."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from . import _paths  # installs shared registry imports
from object_registry import load_registry
from policy_control.policy_registry import scan

TASKS = ('approaching', 'grasping', 'placing', 'locking', 'pouring')


def task_of(policy_id: str) -> str:
    """정책 id → 작업(카드 종류). <로봇>/<계열>/<이름>: 이름에 approach → approaching(경로 C 접근) · 계열 pour* → pouring ·
    place* → placing · lock* → locking · 그 밖(aglt · cup_grasp · cup_pick · grasp_fj …) → grasping. 순수."""
    parts = policy_id.split('/')
    family, name = (parts[1] if len(parts) > 2 else ''), parts[-1]
    if 'approach' in name:
        return 'approaching'
    for prefix, task in (('pour', 'pouring'), ('place', 'placing'), ('lock', 'locking')):
        if family.startswith(prefix):
            return task
    return 'grasping'
MODELS = {'source200': ('src200', 'source200_real'), 'source240': ('src240', 'source240_real'),
          'source240b': ('src240', 'source240_real'), 'shaker_c': ('shaker', 'shaker_c_real'),
          # ★10.28 칵테일 분홍 — 분홍만 든 묶음(파랑 source240b 와 같은 방식, 정책 물체는 같은 source240)
          'source240p': ('src240', 'source240_real')}


def _empty(source, reason='not acquired') -> dict[str, Any]:
    return dict(position=None, frame='base', timestamp=None, source=source, valid=False, reason=reason, live=False)



def _arm_start(entry) -> str:
    """★10.20 계약의 학습 시작 자세(home | ready) — 없거나 못 읽으면 home(옛 계약)."""
    if not getattr(entry, 'contract', ''):
        return 'home'
    try:
        return str(json.loads((entry.path / entry.contract).read_text()).get('arm_start') or 'home')
    except (OSError, ValueError):
        return 'home'

class Workspace:
    def __init__(self, repo: Path, robot_id: str, slots: dict | None = None):
        self.repo = Path(repo).resolve()
        self.robot_id = str(robot_id)
        self.revision = 0
        self.perception = {'fpp': True, 'locate': False}
        self.head = {'pan': 0.0, 'tilt': -20.0, 'calib_profile': ''}
        self.objects = []
        self.cards = []
        self._fpp_after = {}
        registry_repo = self.repo if (self.repo / 'config/objects.yaml').exists() else _paths.SIM2REAL
        self.registry = load_registry(registry_repo / 'config/objects.yaml')
        for name, spec in self.registry.objects.items():
            group = spec.fpp.get('group')
            if group not in MODELS:
                continue
            # source240b is the measured sky-blue variant of source240.
            color = name.rsplit('_', 1)[-1]
            if color not in ('blue', 'orange', 'pink'):
                continue
            model, policy_object = MODELS[group]
            self.objects.append(dict(id=name, model=model, color='skyblue' if color == 'blue' else color,
                                     policy_object=policy_object, enabled=False, input_source='fpp',
                                     measured='hue_measured' in spec.fpp, hue_range=copy.deepcopy(spec.fpp['color']),
                                     locate_supported=True, results={s: _empty(s) for s in ('fpp', 'locate')}))
        holders = yaml.safe_load((registry_repo / 'config/cup_holders.yaml').read_text()) if (registry_repo / 'config/cup_holders.yaml').exists() else {}
        for holder in holders.get('holders', []):
            self.objects.append(dict(id=holder['name'], model='holder', color=None, marker_id=holder['marker_id'],
                                     policy_object=None, enabled=False, input_source='fpp', measured=True,
                                     locate_supported=False, locate_reason='marker holder: color locate unsupported',
                                     results={s: _empty(s) for s in ('fpp', 'locate')}))
        self.catalog = self._catalog(slots or {})
        self._catalog_at = time.monotonic()
        key = hashlib.sha256(self.robot_id.encode()).hexdigest()[:20]
        self.layout_path = self.repo / 'runs/s2r_console/workspaces' / (key + '.json')
        if self.layout_path.is_file():
            try:
                doc = json.loads(self.layout_path.read_text())
                self.update({'cards': doc.get('cards', [])}, persist=False)
            except (OSError, ValueError, TypeError):
                self.cards = []

    def _catalog(self, slots):
        out = []
        for entry in scan(self.repo / 'deploy/policies', deep=False):
            if entry.id.split('/')[0] not in self.robot_id and ('rh56f1' in self.robot_id or 'dg5f' in self.robot_id):
                continue
            task = task_of(entry.id)
            # ★10.20 사용자 "경로 a는 폐지, -> 경로d가 대체함" · "1-c : ap05+grasp": 잡기 카드 = 경로 D(ap05 접근 → 사전 파지 자리에서
            #   이 정책이 넘겨받는다) — 사전 파지 자리에서 시작하는 정책(계약 arm_start ready)만. 홈에서 잡는 옛 rg3 흐름(경로 A)은 없앴다.
            #   접근 카드 = 경로 C(이 접근 정책 → 사전 파지 자리 정지 → 스크립트 닫기 · 들기).
            rg3 = False
            path_d = task == 'grasping' and _arm_start(entry) == 'ready'
            path_c = task == 'approaching'
            deploy = entry.card.get('deploy') if isinstance(entry.card.get('deploy'), dict) else {}
            blocked = deploy.get('execution') is False
            if blocked:
                rg3 = path_c = path_d = False
            reason = str(deploy.get('execution_reason') or '실행 흐름 준비 전') if blocked else '배치만 지원'
            out.append(dict(id=entry.id, policy_id=entry.id, file=str(entry.path / 'policy.yaml'),
                            checkpoint=str(entry.path / 'nn' / entry.checkpoint) if entry.checkpoint else '',
                            contract=str(entry.path / entry.contract) if entry.contract else '',
                            policy_file=str((entry.path / 'policy.yaml').relative_to(self.repo)),
                            checkpoint_path=str((entry.path / 'nn' / entry.checkpoint).relative_to(self.repo)) if entry.checkpoint else '',
                            contract_path=str((entry.path / entry.contract).relative_to(self.repo)) if entry.contract else '',
                            side=entry.card.get('side'), task=task, status=entry.status, issues=list(entry.issues),
                            supported_objects=['src200', 'src240', 'shaker'] if rg3 or path_c or path_d else [],
                            default_rg3=False, execution_supported=rg3 or path_c or path_d,
                            execution_reason='' if rg3 or path_c or path_d else reason))
        return out

    def snapshot(self) -> dict[str, Any]:
        if time.monotonic() - self._catalog_at >= 5:
            self.catalog = self._catalog({})
            self._catalog_at = time.monotonic()
        return copy.deepcopy(dict(revision=self.revision, perception=self.perception, head=self.head,
                                  objects=self.objects, cards=self.cards, catalog=self.catalog))

    def fingerprint(self):
        # Measurements update continuously; approvals bind only to selected settings.
        config = dict(perception=self.perception, head=self.head, cards=self.cards,
                      objects=[{k: o[k] for k in ('id', 'enabled', 'input_source')} for o in self.objects])
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def update(self, patch: dict, *, persist=True):
        if not isinstance(patch, dict) or set(patch) - {'perception', 'head', 'objects', 'cards', 'reset_layout'}:
            raise ValueError('unknown workspace settings')
        state = self.snapshot()
        for key in ('perception', 'head'):
            if key in patch:
                if not isinstance(patch[key], dict) or set(patch[key]) - set(state[key]):
                    raise ValueError(f'invalid {key}')
                state[key].update(patch[key])
        if any(type(v) is not bool for v in state['perception'].values()):
            raise ValueError('perception switches must be boolean')
        head = state['head']
        for key, limits in (('pan', (-90, 90)), ('tilt', (-60, -10))):
            try:
                value = float(head[key])
            except (TypeError, ValueError):
                raise ValueError(f'invalid head {key}') from None
            if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
                raise ValueError(f'head {key} outside limits')
            head[key] = value
        if not isinstance(head['calib_profile'], str) or (head['calib_profile'] and not re.fullmatch(r'[A-Za-z0-9_-]+', head['calib_profile'])):
            raise ValueError('invalid calibration profile name')
        objects = {o['id']: o for o in state['objects']}
        changes = patch.get('objects', [])
        if not isinstance(changes, list):
            raise ValueError('objects must be a list')
        for change in changes:
            if not isinstance(change, dict) or change.get('id') not in objects or set(change) - {'id', 'enabled', 'input_source'}:
                raise ValueError('unknown object or settings')
            obj = objects[change['id']]
            obj.update(change)
            if type(obj['enabled']) is not bool or obj['input_source'] not in ('fpp', 'locate'):
                raise ValueError('invalid object selection')
            if obj['input_source'] == 'locate' and not obj['locate_supported']:
                raise ValueError(obj['locate_reason'])
        if 'cards' in patch:
            if not isinstance(patch['cards'], list):
                raise ValueError('cards must be a list')
            state['cards'] = []
            seen = set()
            catalog = {p['id']: p for p in self.catalog}
            for item in patch['cards']:
                if not isinstance(item, dict) or set(item) - {'id', 'task', 'policy_id', 'side', 'object_id', 'x', 'y', 'execution_supported', 'execution_reason'}:
                    raise ValueError('invalid policy card')
                card = copy.deepcopy(item)
                card.setdefault('object_id', '')
                policy = catalog.get(card.get('policy_id'))
                if not policy or card.get('task') not in TASKS or (card.get('object_id') not in (None, '') and card.get('object_id') not in objects):
                    raise ValueError('unknown policy, task or object')
                cid = card.get('id')
                if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', cid) or cid in seen:
                    raise ValueError('invalid or duplicate card id')
                seen.add(cid)
                card.setdefault('side', policy['side'])
                if card['side'] != policy['side']:
                    raise ValueError('policy arm mismatch')
                for axis in ('x', 'y'):
                    try:
                        card[axis] = float(card.get(axis, 0))
                    except (ValueError, TypeError):
                        raise ValueError('invalid card position') from None
                    if not math.isfinite(card[axis]) or abs(card[axis]) > 100000:
                        raise ValueError('invalid card position')
                card['execution_supported'] = policy['execution_supported'] and card['task'] in ('grasping', 'approaching')
                card['execution_reason'] = '' if card['execution_supported'] else (policy['execution_reason'] or '배치만 지원')   # ★10.20 정책이 막은 이유를 카드에도
                state['cards'].append(card)
        if patch.get('reset_layout'):
            for i, card in enumerate(state['cards']):
                card.update(x=i * 260, y=0)
        self.perception, self.head, self.objects, self.cards = (state[k] for k in ('perception', 'head', 'objects', 'cards'))
        self.revision += 1
        if persist:
            self.layout_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.layout_path.with_suffix('.tmp')
            temp.write_text(json.dumps({'cards': self.cards}, allow_nan=False))
            temp.replace(self.layout_path)
        return self.snapshot()

    def set_result(self, object_id, source, result):
        obj = next((o for o in self.objects if o['id'] == object_id), None)
        if obj is None or source not in ('fpp', 'locate'):
            raise ValueError('unknown object or input source')
        row = _empty(source, str(result.get('reason', 'not acquired')))
        if result.get('valid'):
            pos = result.get('position', result.get('pos'))
            if not isinstance(pos, (list, tuple)) or len(pos) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos):
                raise ValueError('invalid position')
            row.update(position=list(pos), frame=str(result.get('frame', 'base')),
                       timestamp=float(result.get('timestamp', time.time())), valid=True, reason='',
                       live=source == 'fpp' and result.get('live') is True)
        obj['results'][source] = row

    def object_topics(self):
        return [f"/objects/{o['id']}/{kind}" for o in self.objects for kind in ('pose', 'live_pose')]

    def invalidate_results(self, reason, sources=('fpp', 'locate')):
        cutoff = time.time()
        for obj in self.objects:
            for source in sources:
                self.set_result(obj['id'], source, {'valid': False, 'reason': reason})
                if source == 'fpp':
                    self._fpp_after[obj['id']] = cutoff

    def ingest_fpp(self, observation):
        if isinstance(observation, dict):
            data = observation.get('objects', observation)
            report_age = observation.get('objects_age_s', 0.0)
        else:
            data = observation.objects
            report_age = observation.objects_age_s
        for obj in self.objects:
            if not self.perception['fpp']:
                self.set_result(obj['id'], 'fpp', {'valid': False, 'reason': 'FPP not selected'})
                continue
            topic = f"/objects/{obj['id']}/pose"
            live_topic = f"/objects/{obj['id']}/live_pose"
            live = data.get(live_topic) is not None
            summary = data.get(live_topic) if live else data.get(topic, data.get(obj['id']))
            if summary is None or report_age is None:
                self.set_result(obj['id'], 'fpp', {'valid': False, 'reason': 'FPP result unavailable'})
                continue
            age = summary.get('age_s', float('inf')) + report_age
            stamp = summary.get('stamp', time.time() - age)
            fresh = math.isfinite(age) and age <= 1 and stamp > self._fpp_after.get(obj['id'], 0)
            result = dict(position=summary.get('pos'), valid=fresh,
                          timestamp=stamp, frame=summary.get('frame', 'base_link'), reason='stale FPP result', live=live)
            self.set_result(obj['id'], 'fpp', result)

    def locate_argv(self, object_id):
        obj = next((o for o in self.objects if o['id'] == object_id), None)
        if not obj or not obj['locate_supported']:
            raise ValueError('color locate unsupported for this object')
        argv = [sys.executable, str(self.repo / 'scripts/ops/registered_locate.py'), object_id,
                '--repo', str(self.repo), '--current-pose', '--execute']
        if self.head['calib_profile']:
            argv += ['--calib-profile', self.head['calib_profile']]
        return argv

    def fpp_argv(self, object_ids=None):
        ids = object_ids if object_ids is not None else [o['id'] for o in self.objects if o['enabled'] and o['model'] != 'holder']
        known = {o['id'] for o in self.objects if o['model'] != 'holder'}
        if not ids or any(i not in known for i in ids):
            raise ValueError('select registered FPP objects')
        return [sys.executable, str(self.repo / 'scripts/ops/fpp_rescan.py'), *ids]
