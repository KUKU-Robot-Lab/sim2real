"""놓기 직전 후보 경로 고르기(PIVOT 류 시각 프롬프팅). VLM 은 좌표를 말하지 않고 번호만 고른다.

좌표는 base 프레임 m. 경로는 손 안 컵 중심의 웨이포인트(cup)와, 같은 경로를 손 위치로 옮긴 것(hand = cup - T_in).
"""
from __future__ import annotations

import json
import math

from PIL import ImageDraw

from .approach import _seg_dist_xy, _xy_ok
from .planner import parse_json

# shortcut: 아래 높이·옆 거리는 어림값(컵 높이 약 0.1 m 기준), 실기·sim 경로 사례가 쌓이면 다시 잡는다.
ABOVE_M = 0.08   # 홀더 위 이 높이까지 와서 수직으로 내린다
LIFT_M = 0.10    # "높이 들고" 후보가 더 올리는 높이
SIDE_M = 0.12    # 좌·우 우회 후보가 중간점에서 옆으로 비키는 거리
BLOCK_M = 0.05   # 경로(컵·손 선분)에서 다른 물체 중심까지 이보다 가까우면 막힘
NAMES = ('direct', 'lift_high', 'detour_left', 'detour_right')
TEXT = {'direct': 'lower straight onto the holder', 'lift_high': 'lift higher first, then move over and lower',
        'detour_left': 'swing out to the left of the straight line (facing the holder), then lower',
        'detour_right': 'swing out to the right of the straight line (facing the holder), then lower'}
COLORS = {'direct': (255, 40, 40), 'lift_high': (40, 200, 40), 'detour_left': (40, 120, 255),
          'detour_right': (255, 200, 0)}
COLOR_NAMES = {'direct': 'red', 'lift_high': 'green', 'detour_left': 'blue', 'detour_right': 'yellow'}


def _items(world):
    objs = world.get('objects')
    items = {**{k: w for k, w in world.items() if k != 'objects'}, **(objs if isinstance(objs, dict) else {})}
    return {k: w for k, w in items.items() if isinstance(w, dict) and w.get('pos')}


def _xyz_ok(pos):
    return _xy_ok(pos) and len(pos) >= 3 and _xy_ok(pos[1:3])


def held_cup(world, hand):
    """판정하는 손이 쥔 물체 이름(in_hand == hand). 없으면 None."""
    return next((n for n, w in _items(world).items() if w.get('in_hand') == hand), None)


def candidates(world: dict, hand: str, holder_pos, opts: dict | None = None) -> list[dict]:
    """후보 경로 4개. 손·홀더 좌표가 없거나 숫자가 아니면 빈 목록.

    opts: t_in(컵 중심 - 손 위치, base 프레임 3벡터; 없으면 world 의 쥔 컵 좌표로 계산, 그것도 없으면 0),
    above_m·lift_m·side_m(위 상수 덮어쓰기)."""
    opts = opts or {}
    items = _items(world)
    h = items.get(f'{hand}_hand')
    if not h or not _xyz_ok(h['pos']) or not _xyz_ok(holder_pos):
        return []
    hp = [float(c) for c in h['pos'][:3]]
    t_in = opts.get('t_in')
    if t_in is None:
        cup = items.get(held_cup(world, hand) or '')
        t_in = [c - p for c, p in zip(cup['pos'][:3], hp)] if cup and _xyz_ok(cup['pos']) else [0.0, 0.0, 0.0]
    if not _xyz_ok(t_in):
        return []
    c0 = [p + t for p, t in zip(hp, t_in)]
    hx, hy, hz = (float(c) for c in holder_pos[:3])
    above, lift, side = (opts.get(k, d) for k, d in (('above_m', ABOVE_M), ('lift_m', LIFT_M), ('side_m', SIDE_M)))
    z0 = max(c0[2], hz + above)
    end = [hx, hy, hz]
    dx, dy = hx - c0[0], hy - c0[1]
    d = math.hypot(dx, dy)
    left = (-dy / d, dx / d) if d > 1e-6 else (0.0, 1.0)  # 진행 방향 왼쪽(z 위 기준), 제자리면 +y
    mid = ((c0[0] + hx) / 2, (c0[1] + hy) / 2)
    paths = {
        'direct': [c0, [hx, hy, z0], end],
        'lift_high': [c0, [c0[0], c0[1], z0 + lift], [hx, hy, z0 + lift], end],
        'detour_left': [c0, [mid[0] + left[0] * side, mid[1] + left[1] * side, z0], [hx, hy, z0], end],
        'detour_right': [c0, [mid[0] - left[0] * side, mid[1] - left[1] * side, z0], [hx, hy, z0], end],
    }
    return [{'id': i + 1, 'name': n, 'cup': paths[n], 'hand': [[p - t for p, t in zip(w, t_in)] for w in paths[n]]}
            for i, n in enumerate(NAMES)]


def _seg_dist(p, a, b):
    d = [y - x for x, y in zip(a, b)]
    L2 = sum(c * c for c in d)
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, sum((q - x) * c for q, x, c in zip(p, a, d)) / L2))
    return math.dist(p, [x + t * c for x, c in zip(a, d)])


def blocked(cands: list[dict], world: dict, hand: str, ignore=()) -> dict:
    """{후보 번호: [가까운 물체...]}. 판정하는 손·쥔 물체·ignore(목표 홀더 이름)는 뺀다. 좌표가 깨진 물체는 무시하지 않고 막힘으로 센다."""
    others = {n: w for n, w in _items(world).items()
              if n != f'{hand}_hand' and w.get('in_hand') != hand and n not in ignore}
    out = {}
    for c in cands:
        segs = [(p[i], p[i + 1]) for p in (c['cup'], c['hand']) for i in range(len(p) - 1)]
        near = sorted(n for n, w in others.items()
                      if not _xyz_ok(w['pos']) or min(_seg_dist(w['pos'][:3], a, b) for a, b in segs) < BLOCK_M)
        if near:
            out[c['id']] = near
    return out


def project(p, K, T_cam_base):
    """base 점 → 픽셀 (u, v). T_cam_base 는 base 좌표를 카메라 좌표로 바꾸는 4x4(p_cam = T @ p_base),
    카메라 축은 OpenCV(x 오른쪽, y 아래, z 앞). 카메라 뒤면 None."""
    x, y, z = (sum(T_cam_base[r][k] * p[k] for k in range(3)) + T_cam_base[r][3] for r in range(3))
    if z <= 1e-6:
        return None
    return K[0][0] * x / z + K[0][1] * y / z + K[0][2], K[1][1] * y / z + K[1][2]


def draw(image, cands: list[dict], K, T_cam_base):
    """사진 사본에 후보별 색 화살표와 번호. K·T 가 없으면 None(호출자는 텍스트 설명만 쓴다)."""
    if image is None or K is None or T_cam_base is None:
        return None
    img = image.convert('RGB')  # 새 사본
    g = ImageDraw.Draw(img)
    for c in cands:
        pts = [q for q in (project(w, K, T_cam_base) for w in c['cup']) if q is not None]
        if len(pts) < 2:
            continue
        color = COLORS[c['name']]
        g.line(pts, fill=color, width=3)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        a = math.atan2(y1 - y0, x1 - x0)
        g.polygon([(x1, y1), *[(x1 - 12 * math.cos(a + s), y1 - 12 * math.sin(a + s)) for s in (0.45, -0.45)]],
                  fill=color)
        lx, ly = _along(pts, 0.4)
        g.rectangle([lx - 8, ly - 9, lx + 8, ly + 9], fill=(0, 0, 0))
        g.text((lx - 4, ly - 6), str(c['id']), fill=color)
    return img


def _along(pts, frac):
    # 번호 자리: 픽셀 경로 길이의 frac 지점(모든 후보가 같은 시작점·홀더 위를 지나 끝점에 두면 겹친다)
    lens = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    left = frac * sum(lens)
    for (a, b), L in zip(zip(pts, pts[1:]), lens):
        if L >= left and L > 0:
            t = left / L
            return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        left -= L
    return pts[-1]


def parse_choice(raw: str, n: int, bad: dict) -> dict:
    """{"choice": int, "reason": str} 검사. 범위 밖·막힌 후보면 valid False."""
    try:
        v = parse_json(raw, dict)
    except (ValueError, TypeError, AttributeError):
        return {'choice': None, 'reason': '', 'valid': False, 'error': 'invalid JSON'}
    choice, reason = v.get('choice'), v.get('reason', '')
    if isinstance(choice, str) and choice.strip().isdigit():
        choice = int(choice.strip())
    if not isinstance(choice, int) or isinstance(choice, bool) or not isinstance(reason, str):
        return {'choice': None, 'reason': '', 'valid': False, 'error': 'field type'}
    if not 1 <= choice <= n:
        return {'choice': choice, 'reason': reason, 'valid': False, 'error': f'choice {choice} out of 1..{n}'}
    if choice in bad:
        return {'choice': choice, 'reason': reason, 'valid': False, 'error': f'choice {choice} is BLOCKED by {bad[choice]}'}
    return {'choice': choice, 'reason': reason, 'valid': True}


class PlacePathJudge:
    def __init__(self, backend, K=None, T_cam_base=None, opts=None, max_new_tokens: int = 300):
        self.backend, self.K, self.T, self.opts, self.max_new_tokens = backend, K, T_cam_base, opts, max_new_tokens

    @staticmethod
    def prompt(cands, bad, drawn: bool, hand, holder) -> str:
        lines = []
        for c in cands:
            pts = ', '.join('(' + ', '.join(f'{v:.2f}' for v in w) + ')' for w in c['cup'])
            mark = f'  BLOCKED (passes within {BLOCK_M * 1000:.0f} mm of {", ".join(bad[c["id"]])}) - do not choose' \
                if c['id'] in bad else ''
            lines.append(f'{c["id"]}. {COLOR_NAMES[c["name"]]}: {TEXT[c["name"]]}. cup waypoints [m]: {pts}{mark}')
        seen = ('The numbered colored arrows drawn on the photograph show the path of the held cup for each option. '
                if drawn else 'No arrows are drawn; use the waypoints below (robot base frame: x forward, y left, z up). ')
        return ('The robot is about to place the cup held in its ' + str(hand) + ' hand onto ' + str(holder) + '. '
                + seen + 'Pick the ONE option whose path avoids every other object, the other arm and the table, '
                'preferring the simplest safe path. Options:\n' + '\n'.join(lines)
                + '\nReturn only JSON {"choice": <option number>, "reason": "<short reason>"}.')

    def _ask(self, image, text, n, bad):
        try:
            raw, meta = self.backend.ask([image], text, max_new_tokens=self.max_new_tokens)
        except Exception as e:  # noqa: BLE001 - 백엔드 오류는 폴백으로
            raw, meta = '', {'error': f'{type(e).__name__}: {str(e)[:200]}'}
        return parse_choice(raw, n, bad) | {'raw': raw, 'meta': meta}

    def choose(self, image, world: dict, hand: str, holder) -> dict:
        """holder = 홀더 이름(world 에서 좌표를 찾음) 또는 base 좌표. 카메라는 생성자 값, 없으면 world['camera']
        {"K": 3x3, "T_cam_base": 4x4}. 실패하면 choice 1(바로 내리기) + valid False."""
        world = world or {}
        holder_pos = (_items(world).get(holder) or {}).get('pos') if isinstance(holder, str) else holder
        cands = candidates(world, hand, holder_pos, self.opts)
        # 목표 홀더는 경로 끝이라 장애물이 아니다. 좌표로 받았으면 그 자리에 있는 물체를 뺀다.
        ignore = (holder,) if isinstance(holder, str) else tuple(
            n for n, w in _items(world).items() if holder_pos and math.dist(w['pos'][:3], holder_pos[:3]) < BLOCK_M)
        bad = blocked(cands, world, hand, ignore=ignore)
        base = {'choice': 1, 'reason': '', 'valid': False, 'blocked': bad}
        if not cands:
            return base | {'error': 'no geometry (hand/holder pos)'}
        if image is None:
            return base | {'error': 'no camera frame'}
        cam = world.get('camera') if isinstance(world.get('camera'), dict) else {}
        drawn = draw(image, cands, cam.get('K') if self.K is None else self.K,
                     cam.get('T_cam_base') if self.T is None else self.T)
        text = self.prompt(cands, bad, drawn is not None, hand, holder)
        out = self._ask(drawn or image, text, len(cands), bad)
        meta = out.get('meta')
        if not out['valid'] and not (isinstance(meta, dict) and meta.get('error')):
            first = out
            out = self._ask(drawn or image, f'Your previous answer was invalid: {first["error"]}. ' + text, len(cands), bad) \
                | {'retried_from': {k: first.get(k) for k in ('error', 'raw')}}
        if not out['valid']:
            out = out | {'choice': 1, 'rejected': out['choice']}
        return out | {'blocked': bad, 'drawn': drawn is not None, 'prompt': text,
                      'candidates': [{'id': c['id'], 'name': c['name'], 'hand': c['hand']} for c in cands]}
