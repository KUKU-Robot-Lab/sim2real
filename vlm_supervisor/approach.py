"""접근(ap05) 전 사진으로 경로 방해물을 판정한다. ap05 는 회피를 못 하므로 막혔으면 다른 조치를 권한다."""
from __future__ import annotations

import json
import math

from .planner import load_cards, parse_json

VERDICTS = {'clear', 'blocked'}
ACTIONS = {'clear': {'proceed'}, 'blocked': {'reorder', 'other_hand', 'human'}}
# 경로 직선과 방해물 중심의 XY 거리가 이보다 작으면 막힘. 컵 반지름 약 0.04 m + 손바닥 반폭 약 0.045 m
# + 여유 0.015 m. shortcut: 실측 손 폭이 아닌 어림값, ap05 충돌 사례가 쌓이면 다시 잰다.
CLEAR_M = 0.10
# 선분 끝(목표 쪽)을 이만큼 짧게 자른다. 컵 반지름 약 0.04 m: 목표 바로 옆 물체는 손이 목표를 잡으러 가는
# 마지막 구간에서만 가까우므로 막힘으로 세지 않는다. shortcut: 컵 한 종류 기준 어림값, 다른 크기 물체면 물체별로.
GOAL_R_M = 0.04


def _name(value):
    return value.strip().upper().replace(' ', '_') if isinstance(value, str) else value


def parse_approach(raw: str, names: set[str], step_hand=None, step_obj=None) -> dict:
    """VLM 응답을 검사한다. 깨졌거나 이름이 목록 밖이면 unknown/human 으로 떨어뜨린다."""
    fallback = {'hand': step_hand, 'object': step_obj, 'verdict': 'unknown', 'obstacles': [],
                'action': 'human', 'first': None, 'reason': '', 'valid': False}
    try:
        v = parse_json(raw, dict)
    except (ValueError, TypeError, AttributeError):
        return {**fallback, 'error': 'invalid JSON'}
    obs_raw = v.get('obstacles')
    if not (all(isinstance(v.get(k), str) for k in ('hand', 'object', 'verdict', 'action'))
            and isinstance(v.get('first'), (str, type(None))) and (obs_raw is None or (
                isinstance(obs_raw, list) and all(isinstance(o, str) for o in obs_raw)))):
        return {**fallback, 'error': 'field type'}
    hand, verdict, action = (v[k].strip().lower() for k in ('hand', 'verdict', 'action'))
    obj, first, obstacles = _name(v['object']), _name(v.get('first')), [_name(o) for o in obs_raw or []]
    errors = []
    if hand not in ('left', 'right'):
        errors.append(f'hand {hand!r}')
    if obj not in names:
        errors.append(f'object {obj!r}')
    if any(o not in names for o in obstacles):
        errors.append(f'obstacles {v.get("obstacles")!r}')
    if verdict not in VERDICTS or action not in ACTIONS[verdict]:
        errors.append(f'verdict/action {verdict!r}/{action!r}')
    if verdict == 'blocked' and not obstacles:
        errors.append('blocked without obstacles')
    if verdict == 'clear' and obstacles:
        errors.append('clear with obstacles')
    if action == 'reorder' and first not in names:
        errors.append(f'first {first!r}')
    if errors:
        return {**fallback, 'error': '; '.join(errors)}
    return {'hand': hand, 'object': obj, 'verdict': verdict, 'obstacles': obstacles, 'action': action,
            'first': first if action == 'reorder' else None, 'reason': str(v.get('reason', ''))[:200],
            'valid': True, 'step_mismatch': (step_hand, step_obj) != (None, None)
            and (hand, obj) != (step_hand, step_obj)}


def _seg_dist_xy(p, a, b) -> float:
    # hdgp rh_place_ap_r/ap_logic.py seg_dist(bf9c8d21) 와 같은 식의 스칼라판(torch 의존 없이).
    ax, ay, bx, by, px, py = a[0], a[1], b[0], b[1], p[0], p[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _xy_ok(pos) -> bool:
    return (isinstance(pos, (list, tuple)) and len(pos) >= 2
            and all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in pos[:2]))


def geom_check(world: dict, hand: str, obj: str, names: set[str]) -> dict | None:
    """손 시작점–목표 직선과 다른 물체의 XY 거리로 판정. 손·목표 좌표가 없으면 None(VLM 단독 판정).
    좌표가 숫자·유한값이 아니면 verdict unknown(판정 불가, 호출자는 불일치로 보고 사람에게 넘긴다).

    world 는 월드 프레임 m 좌표. 두 형태를 받는다: sim 브리지 /status 형
    {"objects": {name: {"pos": [x, y, z], "in_hand": ...}}} 와 평평한 {name: {"pos": [...]}}.
    손 위치 키는 right_hand/left_hand(최상위 또는 objects 안). 목표와 판정하는 손이 쥔(in_hand == hand) 물체는
    장애물에서 뺀다. 반대 손이 든 물체는 그 손과 함께 길에 있을 수 있어 남긴다.
    """
    objs = world.get('objects')
    items = {**{k: w for k, w in world.items() if k != 'objects'}, **(objs if isinstance(objs, dict) else {})}
    items = {k: w for k, w in items.items() if isinstance(w, dict) and w.get('pos')}
    start, goal = items.get(f'{hand}_hand'), items.get(obj)
    if not start or not goal:
        return None
    others = {n: w for n, w in items.items() if n in names and n != obj and w.get('in_hand') != hand}
    bad = sorted(n for n, w in [(f'{hand}_hand', start), (obj, goal), *others.items()] if not _xy_ok(w['pos']))
    if bad:
        return {'verdict': 'unknown', 'obstacles': [], 'error': f'bad pos {bad}', 'clear_m': CLEAR_M}
    a, b = start['pos'], goal['pos']
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    if d > 0:  # 목표 쪽 끝을 GOAL_R_M 만큼 자른다(손이 목표보다 가까우면 시작점으로 줄어듦)
        k = max(0.0, d - GOAL_R_M) / d
        b = [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]
    # 손은 위에서 내려오지만 높이를 무시한 XY 투영이 더 보수적이다.
    dists = {n: round(_seg_dist_xy(w['pos'], a, b), 3) for n, w in others.items()}
    obstacles = sorted((n for n, d in dists.items() if d < CLEAR_M), key=lambda n: dists[n])
    return {'verdict': 'blocked' if obstacles else 'clear', 'obstacles': obstacles,
            'dist_m': dists, 'clear_m': CLEAR_M}


class ApproachJudge:
    def __init__(self, backend, max_new_tokens: int = 400):
        self.backend, self.max_new_tokens = backend, max_new_tokens
        cards = load_cards()
        self.objects = {o['시스템_이름']: {'color': o['색'], 'hand': o['손'], 'holder': o['홀더']}
                        for o in cards['객체_사전']}
        self.names = set(self.objects) | {k for k in cards['홀더_사전'] if k.startswith('HOLDER_')}

    def prompt(self, hand, obj) -> str:
        return ('Before the robot hand approaches a cup, check the CURRENT head-camera photograph. '
                'The approach policy moves the hand in a straight line from its current position to the target '
                'and cannot avoid obstacles. Identify any object lying between the hand and the target, '
                'or so close to the target that the fingers would hit it. '
                'Use ONLY these names: ' + json.dumps(self.objects, ensure_ascii=False)
                + f' and holders {sorted(n for n in self.names if n.startswith("HOLDER_"))}. '
                f'Planned step: hand={hand}, object={obj}. '
                'If blocked, choose action reorder (move "first" object away first), other_hand, or human. '
                'Return only JSON {"hand":"left"|"right","object":NAME,"verdict":"clear"|"blocked",'
                '"obstacles":[NAME...],"action":"proceed"|"reorder"|"other_hand"|"human",'
                '"first":NAME|null,"reason":"one line"}. action is proceed exactly when verdict is clear.')

    def check(self, image, skill_step, world=None) -> dict:
        """skill_step = [skill, hand, object, target] (planner 4원소). world = geom_check 의 world 형태."""
        _, hand, obj, _ = skill_step
        if image is None:
            out = parse_approach('', self.names, hand, obj) | {'error': 'no camera frame', 'raw': None, 'meta': None}
        else:
            try:
                raw, meta = self.backend.ask([image], self.prompt(hand, obj), max_new_tokens=self.max_new_tokens)
            except Exception as e:  # noqa: BLE001 — 백엔드가 죽어도 접근을 그냥 진행시키지 않는다
                raw, meta = '', {'error': f'{type(e).__name__}: {str(e)[:200]}'}
            try:
                out = parse_approach(raw, self.names, hand, obj)
            except Exception as e:  # noqa: BLE001 — 예상 못 한 응답 모양도 fallback
                out = parse_approach('', self.names, hand, obj) | {'error': f'parse {type(e).__name__}'}
            out |= {'raw': raw, 'meta': meta}
        geom = geom_check(world, hand, obj, self.names) if world else None
        return out | {'geom': geom, 'disagree': geom is not None and geom['verdict'] != out['verdict']}
