#!/usr/bin/env python3
"""차렷(팔 관절 0) -> 계약 홈 팔 경로를 오프라인으로 계획·충돌 검증해 replay_to_pd.py 가 바로 재생할 npz 로 쓴다.

ROS·실기·GPU 무관(순수 파이썬, sim2real/.venv: mujoco·trimesh·numpy·yaml). 실기에 아무것도 보내지 않는다.

    .venv/bin/python deploy/policy_control/tools/plan_home_path.py                    # 계획 + npz
    .venv/bin/python deploy/policy_control/tools/plan_home_path.py --check-only       # 직선만 검사
    .venv/bin/python deploy/policy_control/tools/plan_home_path.py --check-only --npz logs/policy_control/home_path_right.npz

세계·충돌 형상은 home_path_world.py 참고. 판정 규칙:
  * 세계(테이블 상자·컵)·몸통·머리·반대팔 쌍: dist >= --margin.
  * 같은 팔 사슬에서 관절 2개 이내로 떨어진 쌍(r_al_0-r_al_2, r_al_5-손 등): 관통만 금지(dist >= 0).
    볼록 껍질이 관절 주변에서 늘 몇 mm 안쪽이라 margin 을 걸면 어떤 자세도 통과하지 못한다.
  * 시작 자세에서 이미 관통한 자기충돌 쌍은 '항상 접촉'으로 보고하고 제외한다.
  * 시작 자세에서 margin 을 못 지키는 세계 쌍(차렷의 손끝이 받침판 속)은 '탈출 구간'으로 다룬다:
    탈출 영역(시작에서 L-inf --escape-radius 안)에서만 그 쌍의 여유(상자와 관통이면 -꼭짓점 관통 깊이)가
    시작값보다 ESCAPE_TOL 이상 나빠지지 않으면 되고, 영역을 나가기 전에 모두 요구 여유를 넘어야 한다.
    한 번 넘은 쌍과 영역 밖은 일반 규칙. 이 구간은 '모델 속 시작 자세가 이미 관통'이라 검증이 약하다(보고서에 표시).
시간 매개화: L-inf 호 길이로 매개하고 구간마다 코사인 가감속(정지-출발)이라 모든 관절이 ramp_speed 이하.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import home_path_world as W  # noqa: E402

PD_CONFIG_DEFAULT = W.SIM2REAL / "deploy/policy_control/config/pd_dg5f_m_short.yaml"
OUT_DEFAULT = W.SIM2REAL / "logs/policy_control/home_path_right.npz"
FRAME_DT = 0.02
CHECK_STEP = 0.005          # [rad] 조밀 검사 간격(L-inf)
NEAR_ADJ_GAP = 2            # 같은 팔 사슬에서 관절 이 개수 이내 = 관통만 검사
ESCAPE_TOL = 0.01           # [m] 탈출 쌍이 시작보다 더 파고들어도 되는 한도(차렷 모델 자세가 이미 관통이라 단조 조건은 풀 수 없다)
RAMP_TIME = 1.0             # [s] 구간 시작·끝 코사인 가감속 시간
ESCAPE_RADIUS = 1.0         # [rad] 시작 자세 탈출 영역(L-inf). 차렷 손끝이 z −0.048 이라 받침판 위 +0.02 까지 올리려면 어깨·팔꿈치가 0.5 rad 안팎 돈다
ESCAPE_EXIT_EXTRA = 0.003   # [m] 탈출 지점은 모든 탈출 쌍이 요구 여유 + 이 값을 넘은 곳(경계에서 다시 스치지 않게)
ALWAYS_SAMPLES = 300        # '항상 접촉' 판정용 무작위 자세 수
ALWAYS_FRAC = 0.9           # 이 비율 이상에서 관통하면 구조적 겹침으로 본다


# ---------------------------------------------------------------- 설정 읽기

def read_ramp_speed(path: Path) -> float:
    import yaml
    with open(path) as f:
        v = yaml.safe_load(f)["ramp_speed"]
    v = float(v)
    if not 0.0 < v <= 0.5:
        raise SystemExit(f"{path}: ramp_speed {v} 가 (0, 0.5] 밖 — 거부")
    return v


def load_contract(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_q(text: str, n: int = 7) -> np.ndarray:
    q = np.array([float(v) for v in text.split(",")])
    if q.shape != (n,) or not np.isfinite(q).all():
        raise SystemExit(f"관절값 {n} 개(CSV)가 필요하다: {text!r}")
    return q


# ---------------------------------------------------------------- 검사기

@dataclass
class Verdict:
    ok: bool
    worst_pair: tuple = ("", "")
    worst_slack: float = np.inf       # dist - 요구 여유 (음수 = 위반)
    worst_dist: float = np.inf
    min_clear: float = np.inf         # margin 규칙 쌍의 최소 dist (탈출 중인 쌍 제외)
    min_clear_pair: tuple = ("", "")
    dists: dict = field(default_factory=dict)
    escaping: set = field(default_factory=set)


class Checker:
    """충돌·여유 판정. 규칙은 모듈 docstring. 상태 기반(check)과 순서 기반(check_path) 두 가지."""

    def __init__(self, world: W.World, scenes: list[dict], margin: float, start: np.ndarray,
                 limits: tuple[np.ndarray, np.ndarray], escape_radius: float = ESCAPE_RADIUS, seed: int = 0):
        self.w, self.scenes, self.margin = world, scenes, margin
        self.start = np.asarray(start, dtype=float).copy()
        self.escape_radius = escape_radius
        self.n_checks = 0
        self.lenient = True          # False 면 탈출 완화 없음(탈출 지점 이후 계획용)
        self.chain = {b for b, g in world.body_group.items() if g > 0} | {f"{world.moving_joints[0][:2]}al_0"}
        d0 = self.raw(self.start)
        pen0 = [k for k, v in d0.items() if v < 0 and not self._touches_world(k)]
        # '항상 접촉' = 시작에서 관통 AND 한계 안 무작위 자세 ALWAYS_FRAC 이상에서 관통(구조적 껍질 겹침)
        rng = np.random.default_rng(seed)
        samples = rng.uniform(limits[0], limits[1], size=(ALWAYS_SAMPLES, 7))
        hits = dict.fromkeys(pen0, 0)
        for q in samples:
            d = self.raw(q)
            for k in pen0:
                if d.get(k, np.inf) < 0:
                    hits[k] += 1
        self.always_freq = {k: hits[k] / ALWAYS_SAMPLES for k in pen0}
        self.always_contact = sorted(k for k in pen0 if self.always_freq[k] >= ALWAYS_FRAC)
        world.excluded_pairs = set(self.always_contact)
        self.raw(self.start)                                   # kinematics 를 시작 자세로
        d0 = {k: v for k, v in d0.items() if k not in world.excluded_pairs}
        self.escape0 = {k: self._escape_measure(k, v) for k, v in d0.items() if v < self.required(k)}

    def _touches_world(self, k) -> bool:
        return self.w.is_world(k[0]) or self.w.is_world(k[1])

    def required(self, k) -> float:
        a, b = k
        if a in self.chain and b in self.chain:
            ga, gb = self.w.body_group.get(a, 0), self.w.body_group.get(b, 0)
            if abs(ga - gb) <= NEAR_ADJ_GAP:
                return 0.0
        return self.margin

    def raw(self, q: np.ndarray) -> dict:
        out: dict = {}
        for sc in self.scenes:
            self.w.set_q(sc)
            for k, v in self.w.pair_distances(q).items():
                if v < out.get(k, np.inf):
                    out[k] = v
        self.n_checks += 1
        return out

    def _escape_measure(self, k, dist: float) -> float:
        """탈출 쌍의 여유: dist >= 0 이면 dist, 상자와 관통이면 -(움직이는 몸 꼭짓점의 상자 관통 깊이)."""
        if dist >= 0 or not self._touches_world(k):
            return dist
        box, body = (k[0], k[1]) if self.w.is_world(k[0]) else (k[1], k[0])
        return -self.box_vertex_depth(body, box)

    def box_vertex_depth(self, body: str, box: str) -> float:
        import mujoco
        m, d = self.w.model, self.w.data
        g_box = int(np.nonzero(m.geom_bodyid == m.body(box).id)[0][0])
        R, c, h = d.geom_xmat[g_box].reshape(3, 3), d.geom_xpos[g_box], m.geom_size[g_box]
        depth = 0.0
        for g in np.nonzero(m.geom_bodyid == m.body(body).id)[0]:
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            inside = np.min(h - np.abs((v - c) @ R), axis=1)     # >0 이면 상자 안, 값 = 가장 가까운 면까지
            depth = max(depth, float(inside.max()))
        return depth

    def in_escape_zone(self, q: np.ndarray) -> bool:
        return bool(self.escape0) and float(np.abs(q - self.start).max()) <= self.escape_radius + 1e-12

    def check(self, q: np.ndarray, cleared: set | None = None) -> Verdict:
        """상태 기반 판정. 탈출 영역(시작에서 L-inf escape_radius 안)에서 아직 풀리지 않은(cleared 밖) 탈출 쌍은
        '시작보다 나빠지지 않음'만 요구한다. 그 밖은 dist >= 요구 여유."""
        dists = self.raw(q)
        v = Verdict(True, dists=dists)
        zone = self.lenient and self.in_escape_zone(q)
        cleared = cleared or set()
        for k, dist in dists.items():
            req = self.required(k)
            if dist < req and zone and k in self.escape0 and k not in cleared:
                meas = self._escape_measure(k, dist)
                slack = meas - (self.escape0[k] - ESCAPE_TOL)
                v.escaping.add(k)
            else:
                slack = dist - req
                if req > 0 and dist < v.min_clear:
                    v.min_clear, v.min_clear_pair = dist, k
            if slack < v.worst_slack:
                v.worst_slack, v.worst_pair, v.worst_dist = slack, k, dist
                if slack < 0:
                    v.ok = False
        return v

    def check_path(self, path: np.ndarray, step: float = CHECK_STEP) -> dict:
        """꺾은선 경로를 step(L-inf) 이하 간격으로 순서대로 검사. 한 번 margin 을 넘은 탈출 쌍은 다시 일반 규칙."""
        cleared: set = set()
        rep = {"ok": True, "min_clear": np.inf, "min_clear_pair": ("", ""), "min_clear_q": None,
               "min_clear_ne": np.inf, "min_clear_ne_pair": ("", ""),
               "fails": [], "samples": 0, "escape_end_s": 0.0}
        s_acc = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            L = float(np.abs(b - a).max())
            n = max(1, int(np.ceil(L / step)))
            for t in np.linspace(0.0, 1.0, n + 1)[(1 if i else 0):]:
                q = a + (b - a) * t
                v = self.check(q, cleared)
                cleared |= {k for k in self.escape0 if v.dists.get(k, np.inf) >= self.required(k) + ESCAPE_EXIT_EXTRA}
                rep["samples"] += 1
                s_here = s_acc + L * t
                if v.escaping:
                    rep["escape_end_s"] = s_here
                if v.min_clear < rep["min_clear"]:
                    rep["min_clear"], rep["min_clear_pair"], rep["min_clear_q"] = v.min_clear, v.min_clear_pair, q.copy()
                for k, dist in v.dists.items():             # 시작부터 margin 을 지킨 쌍만(탈출 쌍 제외)
                    if k not in self.escape0 and self.required(k) > 0 and dist < rep["min_clear_ne"]:
                        rep["min_clear_ne"], rep["min_clear_ne_pair"] = dist, k
                if not v.ok:
                    rep["ok"] = False
                    rep["fails"].append({"s": s_here, "q": q.round(4).tolist(), "pair": v.worst_pair,
                                         "dist": v.worst_dist, "slack": v.worst_slack})
            s_acc += L
        if len(path) == 1:
            v = self.check(path[0])
            rep.update(ok=v.ok, min_clear=v.min_clear, min_clear_pair=v.min_clear_pair, samples=1)
        return rep


# ---------------------------------------------------------------- 계획

def edge_ok(chk: Checker, a, b, step=CHECK_STEP) -> bool:
    """상태 기반 간선 검사(가운데부터 쪼개 일찍 실패)."""
    n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
    for i in _bisect_order(n):
        if not chk.check(a + (b - a) * i / n).ok:
            return False
    return True


def _bisect_order(n: int) -> list[int]:
    out, seen, stack = [], set(), [(1, n)]
    while stack:
        lo, hi = stack.pop(0)
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
        stack += [(lo, mid - 1), (mid + 1, hi)]
    return out


def rrt_connect(chk: Checker, start, goal, lo, hi, rng, max_iter=4000, step_size=0.25, edge_step=0.02, log=print):
    """7D RRT-Connect. 노드 = 관절 벡터, 거리 = L-inf. 경로(꼭짓점 배열) 또는 None."""
    Ta, Tb = [start.copy()], [goal.copy()]
    Pa, Pb = [-1], [-1]

    def nearest(T, q):
        arr = np.asarray(T)
        return int(np.argmin(np.abs(arr - q).max(axis=1)))

    def steer(a, b):
        d = np.abs(b - a).max()
        return b.copy() if d <= step_size else a + (b - a) * (step_size / d)

    def extend(T, P, q):
        i = nearest(T, q)
        qn = steer(T[i], q)
        if edge_ok(chk, T[i], qn, edge_step):
            T.append(qn)
            P.append(i)
            return ("reached" if np.allclose(qn, q) else "advanced"), len(T) - 1
        return "trapped", -1

    def connect(T, P, q):
        while True:
            st, idx = extend(T, P, q)
            if st != "advanced":
                return st, idx

    swapped = False
    for it in range(max_iter):
        if it and it % 500 == 0:
            log(f"[rrt] 반복 {it} · 트리 {len(Ta)}+{len(Tb)} · 검사 {chk.n_checks}")
        qr = goal.copy() if rng.random() < 0.05 else rng.uniform(lo, hi)
        st, ia = extend(Ta, Pa, qr)
        if st != "trapped":
            st2, ib = connect(Tb, Pb, Ta[ia])
            if st2 == "reached":
                pa, pb = _trace(Ta, Pa, ia), _trace(Tb, Pb, ib)
                path = pa[::-1] + pb[1:] if not swapped else pb[::-1] + pa[1:]
                path = np.array(path)
                if swapped:
                    path = path[::-1]
                if not np.allclose(path[0], start):
                    path = path[::-1]
                log(f"[rrt] 연결 — 반복 {it + 1}, 트리 {len(Ta)}+{len(Tb)}, 꼭짓점 {len(path)}")
                return path
        Ta, Tb, Pa, Pb = Tb, Ta, Pb, Pa
        swapped = not swapped
    log(f"[rrt] 실패 — {max_iter} 반복")
    return None


def _trace(T, P, i):
    out = []
    while i >= 0:
        out.append(T[i])
        i = P[i]
    return out


def shortcut(chk: Checker, path: np.ndarray, rng, iters=200, edge_step=0.02) -> np.ndarray:
    path = [p.copy() for p in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if edge_ok(chk, path[i], path[j], edge_step):
            path = path[: i + 1] + path[j:]
    return np.array(path)


def escape_prefix(chk: Checker, path: np.ndarray, step: float = CHECK_STEP) -> list | None:
    """경로를 순서대로 따라가 탈출 쌍이 모두 풀리는 첫 자세까지의 꼭짓점 목록(끝 = 탈출 지점).
    그 전에 위반이 있거나 끝까지 안 풀리면 None."""
    cleared: set = set()
    pending = set(chk.escape0)
    out = [path[0].copy()]
    if not pending:
        return out
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            q = a + (b - a) * t
            v = chk.check(q, cleared)
            if not v.ok:
                return None
            cleared |= {k for k in pending if v.dists.get(k, np.inf) >= chk.required(k) + ESCAPE_EXIT_EXTRA}
            if cleared >= pending:
                out.append(q)
                return out
        out.append(b.copy())
    return None


def plan_rrt_two_phase(chk: Checker, start, goal, lo, hi, rng, args):
    """1단계: 탈출 완화를 켠 RRT 로 경로를 찾아 탈출이 끝나는 지점까지만 쓴다.
    2단계: 그 지점에서 완화 없이 RRT·지름길. 합친 경로를 조밀·순서 검사."""
    chk.lenient = True
    core = rrt_connect(chk, start, goal, lo, hi, rng, args.rrt_iters, edge_step=args.rrt_edge_step)
    if core is None:
        return None
    prefix = escape_prefix(chk, core, args.step)
    if prefix is None:
        print("[rrt] 1단계 경로가 탈출 전에 위반 — 버림")
        return None
    q_e = prefix[-1]
    chk.lenient = False
    try:
        if not chk.check(q_e).ok:
            return None
        rest = rrt_connect(chk, q_e, goal, lo, hi, rng, args.rrt_iters, edge_step=args.rrt_edge_step)
        if rest is None:
            return None
        rest = shortcut(chk, rest, rng, edge_step=args.rrt_edge_step)
    finally:
        chk.lenient = True
    path = np.array(prefix[:-1] + list(rest))
    rep = chk.check_path(path, args.step)
    if not rep["ok"]:
        f0 = min(rep["fails"], key=lambda f: f["slack"])
        print(f"[rrt] 조밀·순서 검사 실패 {fmt_pair(f0['pair'])} dist {f0['dist']:+.4f} s {f0['s']:.3f}")
        return None
    return path


# ---------------------------------------------------------------- 시간 매개화

def segment_profile(L: float, vmax: float, ramp: float, dt: float) -> np.ndarray:
    """길이 L(L-inf) 구간의 s(t) 샘플(0..L, dt 간격, 끝 포함). 코사인 가감속 사다리꼴, 정지-출발."""
    if L <= 0:
        return np.array([0.0])
    v = vmax
    if L < v * ramp:                    # 짧은 구간: 최고 속도를 낮춰 가감속만으로 L
        v = L / ramp
    T = L / v + ramp
    n = int(np.ceil(T / dt))
    t = np.minimum(np.arange(n + 1) * dt, T)

    def s_of(tt):
        if tt < ramp:                   # ∫ v(1-cos(πτ/ramp))/2
            return v / 2 * (tt - ramp / np.pi * np.sin(np.pi * tt / ramp))
        if tt <= T - ramp:
            return v * ramp / 2 + v * (tt - ramp)
        r = T - tt
        return L - v / 2 * (r - ramp / np.pi * np.sin(np.pi * r / ramp))

    s = np.array([s_of(x) for x in t])
    s[-1] = L
    return np.clip(s, 0.0, L)


def time_parametrize(path: np.ndarray, vmax: float, dt: float = FRAME_DT, ramp: float = RAMP_TIME) -> np.ndarray:
    """꼭짓점 경로 -> (N,7) 프레임. 각 구간 s 는 L-inf 거리라 |dq_j| <= ds <= vmax·dt."""
    frames = [path[0][None, :]]
    for a, b in zip(path[:-1], path[1:]):
        L = float(np.abs(b - a).max())
        if L <= 0:
            continue
        s = segment_profile(L, vmax, ramp, dt)[1:]
        frames.append(a + np.outer(s / L, b - a))
    return np.vstack(frames)


# ---------------------------------------------------------------- 엄밀 거리(검증용)

def hull_distance_lower_bound(A: np.ndarray, B: np.ndarray, iters: int = 20000, gap: float = 1e-5) -> tuple[float, float]:
    """conv(A)·conv(B) 거리의 (하한, 상한). Minkowski 차에 Frank-Wolfe, 하한은 분리 평면 지지값."""
    z = A.mean(0) - B.mean(0)
    lb = 0.0
    for _ in range(iters):
        nz = np.linalg.norm(z)
        if nz < 1e-9:
            return 0.0, 0.0
        s = A[np.argmin(A @ z)] - B[np.argmax(B @ z)]
        lb = max(lb, float(s @ z) / nz)          # 방향 z 로 본 차집합의 지지값 = 거리 하한
        if nz - lb < gap:
            break
        d = s - z
        dd = float(d @ d)
        if dd < 1e-18:
            break
        gam = np.clip(-float(z @ d) / dd, 0.0, 1.0)
        if gam <= 0:
            break
        z = z + gam * d
    return max(lb, 0.0), float(np.linalg.norm(z))


def body_vertices(world: W.World, body: str) -> list[np.ndarray]:
    import mujoco
    m, d = world.model, world.data
    out = []
    for g in np.nonzero(m.geom_bodyid == m.body(body).id)[0]:
        R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            out.append(m.mesh_vert[a:a + n] @ R.T + c)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            h = m.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
            out.append(corners @ R.T + c)
    return out


def certify_pairs(chk: Checker, frames: np.ndarray, top: int = 12) -> list[dict]:
    """경로 위 가장 가까운 margin 규칙 쌍들(탈출 중인 쌍 제외)을 볼록 껍질 엄밀 거리(하한)로 다시 잰다."""
    best: dict = {}
    cleared: set = set()
    escaping_at: dict = {}
    for i, q in enumerate(frames):
        v = chk.check(q, cleared)
        cleared |= {k for k in chk.escape0 if v.dists.get(k, np.inf) >= chk.required(k) + ESCAPE_EXIT_EXTRA}
        for k in v.escaping:
            escaping_at.setdefault(k, set()).add(i)
        for k, dist in v.dists.items():
            if chk.required(k) <= 0 or k in v.escaping:
                continue
            if dist < best.get(k, (np.inf, 0))[0]:
                best[k] = (dist, i)
    rows = []
    for k, (dist, i) in sorted(best.items(), key=lambda kv: kv[1][0])[:top]:
        lbs = []
        for j in range(max(0, i - 3), min(len(frames), i + 4)):
            if j in escaping_at.get(k, ()):
                continue
            lb_scene = []
            for sc in chk.scenes:
                chk.w.set_q(sc)
                chk.w.pair_distances(frames[j])          # kinematics 갱신
                va, vb = body_vertices(chk.w, k[0]), body_vertices(chk.w, k[1])
                lb_scene.append(min(hull_distance_lower_bound(a, b)[0] for a in va for b in vb))
            lbs.append(min(lb_scene))
        rows.append({"pair": k, "mujoco_dist": dist, "frame": i, "exact_lb": min(lbs)})
    return rows


# ---------------------------------------------------------------- 메인

def parse_hand_q(text: str | None) -> dict[str, float]:
    """`r_hj_index_2=1.62,r_hj_thumb_3=1.16,...` → dict. 실측 손 자세(09.22 실기 차렷에서 손은 주먹을 쥐고 있었다)."""
    out: dict[str, float] = {}
    for item in (text or "").split(","):
        if item.strip():
            k, _, v = item.partition("=")
            out[k.strip()] = float(v)
    return out


def build_scenes(contract: dict, side: str, other_mode: str, hand_mode: str, hand_q: dict | None = None) -> list[dict]:
    """고정 관절 조합 목록. 경로는 모든 조합에서 통과해야 한다(반대팔 home/zero × 손 contract/zeros [+ 실측 손])."""
    other = "left" if side == "right" else "right"
    sd, od = contract["sides"][side], contract["sides"][other]
    hands = []
    if hand_mode in ("contract", "both"):
        hands.append(dict(sd["home_hand"]))
    if hand_mode in ("zeros", "both"):
        hands.append({k: 0.0 for k in sd["home_hand"]})
    if hand_q:
        hands.append({**{k: 0.0 for k in sd["home_hand"]}, **{k: v for k, v in hand_q.items() if k in sd["home_hand"]}})
    arms = []
    if other_mode in ("home", "both"):
        arms.append(dict(zip(od["arm_joints"], od["home_arm"])))
    if other_mode in ("zero", "both"):
        arms.append({j: 0.0 for j in od["arm_joints"]})
    base = {**od["home_hand"], "head_j_pan": 0.0, "head_j_tilt": 0.0}
    return [{**base, **h, **a} for h in hands for a in arms]


def resolve_goal(text: str, contract: dict, env: dict, side: str) -> tuple[np.ndarray, str]:
    if text == "contract":
        return np.array(contract["sides"][side]["home_arm"], dtype=float), "contract sides.%s.home_arm" % side
    if text == "env_reset":
        if side != "right":
            raise SystemExit("env_reset 는 right_aglt env.yaml(우팔) 전용")
        return np.array(env["arm_reset_joint_pos_override"], dtype=float), "env.yaml arm_reset_joint_pos_override"
    return parse_q(text), "cli"


def abduction_box(lo: np.ndarray, hi: np.ndarray, side: str, cap: float, start, goal) -> tuple[np.ndarray, np.ndarray]:
    """RRT 샘플 범위를 좁힌다 — j2(옆 벌림) ≤ cap, j3 ∈ [-0.3, 0.6] (좌팔은 부호 반대). 시작·목표는 늘 범위 안에 둔다."""
    lo, hi = lo.copy(), hi.copy()
    sgn = 1.0 if side == "right" else -1.0
    box = {1: (-np.inf, cap), 2: (-0.3, 0.6)}
    for j, (a, b) in box.items():
        a, b = (a, b) if sgn > 0 else (-b, -a)
        a = min(a, start[j], goal[j]) if np.isfinite(a) else lo[j]
        b = max(b, start[j], goal[j]) if np.isfinite(b) else hi[j]
        lo[j], hi[j] = max(lo[j], a), min(hi[j], b)
    return lo, hi


J1J4_GRID_J1 = np.round(np.arange(-1.4, 1.41, 0.2), 2)
J1J4_GRID_J4 = np.round(np.arange(0.2, 1.41, 0.2), 2)


def plan_j1j4(chk, start, goal, lo, hi, step=CHECK_STEP, vmax=0.1, dt=FRAME_DT, ramp=RAMP_TIME):
    """start → A(j1·j4 만) → B(A 의 j1·j4 + 나머지는 목표) → goal(j1·j4 만). 격자에서 통과하는 A 중 여유가 최대인 것들의
    **가운데**를 고른다(경계에 붙은 자세는 모델 오차에 약하다). 없으면 None.

    꼭짓점 사이 검사를 통과해도 **시간을 붙인 프레임**으로 다시 검사해 통과한 것만 쓴다 — 09.22 여유가 정확히 2 cm 에 걸린
    후보가 꼭짓점 검사는 통과하고 프레임 검사(1e-5 m 부족)에서 떨어졌다."""
    passing = []
    for j1 in J1J4_GRID_J1:
        for j4 in J1J4_GRID_J4:
            if not (lo[0] <= j1 <= hi[0] and lo[3] <= j4 <= hi[3]):
                continue
            a = start.copy()
            a[0], a[3] = j1, j4
            b = goal.copy()
            b[0], b[3] = j1, j4
            path = np.stack([start, a, b, goal])
            rep = chk.check_path(path, step)
            if rep["ok"] and chk.check_path(time_parametrize(path, vmax, dt, ramp), step)["ok"]:
                passing.append((round(float(rep["min_clear_ne"]), 3), float(j1), float(j4), path))
    if not passing:
        return None
    best = max(p[0] for p in passing)
    top = sorted((p for p in passing if p[0] >= best - 1e-9), key=lambda p: (p[1], p[2]))
    pick = top[len(top) // 2]
    print(f"[j1j4] 통과 {len(passing)} 개 · 최대 여유 {best:.3f} m 인 것 {len(top)} 개 중 가운데 — 빼기 자세 j1 {pick[1]:+.2f} · j4 {pick[2]:.2f}")
    return pick[3]


def fmt_pair(k) -> str:
    return f"{k[0]}<->{k[1]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), default="right")
    ap.add_argument("--start", default="zeros", help="'zeros' 또는 7개 CSV [rad]")
    ap.add_argument("--goal", default="contract", help="'contract'(계약 home_arm) | 'env_reset'(env.yaml 리셋 자세) | CSV")
    ap.add_argument("--other-arm", choices=("home", "zero", "both"), default="both",
                    help="반대팔 고정 자세 — both 면 두 자세 모두에서 통과해야 한다")
    ap.add_argument("--hand-start", choices=("contract", "zeros", "both", "measured"), default="contract",
                    help="이동 중 손 자세 — contract = 계약 home_hand(엄지_2 만 ±1.57), zeros = 전부 0, both = 둘 다 통과")
    ap.add_argument("--with-cup", action="store_true", help="스폰 중심에 cup_big_s100 상자를 둔다")
    ap.add_argument("--hand-q", default=None,
                    help="실측 손 자세 'r_hj_index_2=1.62,...' — --hand-start 자세들에 **더해** 이 손으로도 통과해야 한다")
    ap.add_argument("--margin", type=float, default=0.02, help="세계·몸통·반대팔 최소 여유 [m]")
    ap.add_argument("--inset", type=float, default=0.05, help="RRT 샘플 관절한계 안쪽 여유 [rad]")
    ap.add_argument("--step", type=float, default=CHECK_STEP, help="조밀 검사 간격 [rad]")
    ap.add_argument("--ramp-time", type=float, default=RAMP_TIME)
    ap.add_argument("--dt", type=float, default=FRAME_DT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rrt-iters", type=int, default=4000)
    ap.add_argument("--rrt-restarts", type=int, default=5)
    ap.add_argument("--rrt-edge-step", type=float, default=0.02,
                    help="RRT·지름길 간선 검사 간격 [rad] (탐색용 — 최종 경로는 --step 으로 다시 조밀 검사)")
    ap.add_argument("--escape-radius", type=float, default=ESCAPE_RADIUS,
                    help="시작 자세 탈출 영역 L-inf 반경 [rad] — 시작에서 이미 margin 미달인 쌍만 이 안에서 완화")
    ap.add_argument("--force-rrt", action="store_true", help="직선이 통과해도 RRT 로 계획(시험용)")
    ap.add_argument("--via", choices=("rrt", "j1j4"), default="rrt",
                    help="j1j4 = j1·j4 로 테이블에서 빼기 → 공중에서 나머지 관절 → j1·j4 로 들어가기(09.22 사용자 방식). "
                         "격자에서 통과하는 빼기 자세 중 여유가 가장 큰 것들의 가운데를 고른다")
    ap.add_argument("--max-abduction", type=float, default=None,
                    help="어깨 옆 벌림(j2) 상한 [rad] — 손이 옆으로 크게 나가지 않게(09.22 사용자: 옆이 아니라 j1·j4 로). "
                         "j3(상완 회전)도 [-0.3, 0.6] 으로 묶는다. 좌팔은 부호를 뒤집는다. 없으면 관절한계 전부")
    ap.add_argument("--check-only", action="store_true", help="계획하지 않고 직선(또는 --npz)만 검사")
    ap.add_argument("--npz", type=Path, default=None, help="--check-only 대상 npz(arm_target)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--contract", type=Path, default=W.CONTRACT_DEFAULT)
    ap.add_argument("--env-yaml", type=Path, default=W.ENV_YAML_DEFAULT)
    ap.add_argument("--urdf", type=Path, default=W.URDF_DEFAULT)
    ap.add_argument("--profile", type=Path, default=W.PROFILE_DEFAULT)
    ap.add_argument("--pd-config", type=Path, default=PD_CONFIG_DEFAULT)
    args = ap.parse_args(argv)
    if args.step > CHECK_STEP + 1e-12:
        raise SystemExit(f"--step 은 {CHECK_STEP} rad 이하")

    t0 = time.time()
    contract = load_contract(args.contract)
    env = W.load_env_yaml(args.env_yaml)
    vmax = read_ramp_speed(args.pd_config)
    world = W.build_world(W.WorldSpec(urdf=args.urdf, env_yaml=args.env_yaml, side=args.side, with_cup=args.with_cup,
                                      detect_margin=max(0.08, args.margin * 3)))
    limits = W.load_profile_limits(args.profile)
    lo = np.array([limits[j][0] for j in world.moving_joints])
    hi = np.array([limits[j][1] for j in world.moving_joints])
    start = np.zeros(7) if args.start == "zeros" else parse_q(args.start)
    goal, goal_src = resolve_goal(args.goal, contract, env, args.side)
    if args.hand_start == "measured" and not args.hand_q:
        raise SystemExit("--hand-start measured 에는 --hand-q(실측 손 자세)가 필요하다")
    scenes = build_scenes(contract, args.side, args.other_arm, args.hand_start, parse_hand_q(args.hand_q))
    for name, q in (("start", start), ("goal", goal)):
        bad = [(j, round(float(v), 4), limits[j]) for j, v in zip(world.moving_joints, q)
               if not limits[j][0] - 1e-9 <= v <= limits[j][1] + 1e-9]
        if bad:
            raise SystemExit(f"{name} 가 관절한계 밖: {bad}")

    if args.max_abduction is not None:
        lo, hi = abduction_box(lo, hi, args.side, args.max_abduction, start, goal)
    chk = Checker(world, scenes, args.margin, start, (lo, hi), args.escape_radius, args.seed)
    print(f"[plan] 세계: 로봇 {args.urdf.name} (메쉬 볼록 껍질, 몸통 {world.mesh_notes.get('body_link:body_link0_symp_cut_nohousing_top730.stl')}) "
          f"· 테이블 상자 {len(world.table_boxes)} · 컵 {'있음' if world.cup_box else '없음'} · 반대팔 {args.other_arm} "
          f"· 손 {args.hand_start} · margin {args.margin} m · ramp_speed {vmax} rad/s")
    for b in world.table_boxes + ([world.cup_box] if world.cup_box else []):
        print(f"    {b['name']:<8} lo {np.round(b['lo'], 4).tolist()} hi {np.round(b['hi'], 4).tolist()}")
    print("[plan] 시작 자세 관통 자기충돌 쌍(무작위 자세 관통 비율): "
          + (", ".join(f"{fmt_pair(k)} {f:.2f}" for k, f in chk.always_freq.items()) or "없음"))
    print(f"[plan] 항상 접촉으로 제외(비율 >= {ALWAYS_FRAC}): {[fmt_pair(k) for k in chk.always_contact] or '없음'}")
    if chk.escape0:
        print("[plan] 시작 자세 margin 미달 세계 쌍(탈출 구간): "
              + ", ".join(f"{fmt_pair(k)} {v:+.4f}" for k, v in sorted(chk.escape0.items(), key=lambda kv: kv[1])))
    v_goal = chk.check(goal)
    if not v_goal.ok:
        raise SystemExit(f"목표 자세가 충돌/여유 미달: {fmt_pair(v_goal.worst_pair)} dist {v_goal.worst_dist:+.4f}")
    print(f"[plan] 목표 {goal_src} {goal.tolist()} — 최소 여유 {v_goal.min_clear:.4f} m ({fmt_pair(v_goal.min_clear_pair)})")

    if args.check_only:
        if args.npz:
            d = np.load(args.npz)
            path = np.asarray(d["arm_target"], dtype=float)
            label = str(args.npz)
        else:
            path, label = np.stack([start, goal]), "직선 start->goal"
        rep = chk.check_path(path, args.step)
        _print_check(label, rep)
        return 0 if rep["ok"] else 1

    straight = chk.check_path(np.stack([start, goal]), args.step)
    _print_check("직선 start->goal", straight)
    rng = np.random.default_rng(args.seed)
    if straight["ok"] and not args.force_rrt:
        method, path = "straight", np.stack([start, goal])
    elif args.via == "j1j4":
        method, path = "j1j4", plan_j1j4(chk, start, goal, lo, hi, args.step, vmax, args.dt, args.ramp_time)
        if path is None:
            raise SystemExit("[plan] j1·j4 구조 경로를 찾지 못했다(격자 전부 실패) — --via rrt 를 쓰거나 세계를 다시 볼 것")
    else:
        method = "rrt"
        path = None
        for attempt in range(args.rrt_restarts):
            path = plan_rrt_two_phase(chk, start, goal, lo + args.inset, hi - args.inset, rng, args)
            if path is not None:
                break
            print(f"[rrt] 시도 {attempt + 1} 실패 — 다시")
        if path is None:
            raise SystemExit("[plan] RRT-Connect 실패")
    final = chk.check_path(path, args.step)
    _print_check(f"최종 경로({method}, 꼭짓점 {len(path)})", final)
    if not final["ok"]:
        raise SystemExit("[plan] 최종 경로 재검사 실패 — 쓰지 않는다")

    frames = time_parametrize(path, vmax, args.dt, args.ramp_time)
    vel = np.abs(np.diff(frames, axis=0)).max() / args.dt
    in_lim = bool(((frames >= lo - 1e-9) & (frames <= hi + 1e-9)).all())
    frames_rep = chk.check_path(frames, args.step)
    if not frames_rep["ok"]:
        _print_check("시간 매개화 프레임", frames_rep)
    if not (frames_rep["ok"] and in_lim and vel <= vmax + 1e-9):
        raise SystemExit(f"[plan] 프레임 검사 실패 ok={frames_rep['ok']} 한계={in_lim} 최고속도={vel:.4f}")
    cert = certify_pairs(chk, frames)
    worst = min(cert, key=lambda r: r["exact_lb"]) if cert else None
    out = args.out or (OUT_DEFAULT if args.goal == "contract" else
                       OUT_DEFAULT.with_name(f"home_path_{args.side}_{args.goal if args.goal == 'env_reset' else 'custom'}.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, arm_target=frames.astype(np.float64), meta_step_dt=np.float64(args.dt),
             meta_min_clearance=np.float64(frames_rep["min_clear"]),
             meta_worst_pair=np.array(fmt_pair(frames_rep["min_clear_pair"])),
             meta_start=start, meta_goal=goal, meta_method=np.array(method),
             meta_contract_sha1=np.array(W.sha1_of(args.contract)), meta_urdf_sha1=np.array(W.sha1_of(args.urdf)),
             meta_joints=np.array(world.moving_joints), meta_waypoints=path,
             meta_margin=np.float64(args.margin), meta_max_joint_speed=np.float64(vel),
             meta_escape_end_s=np.float64(frames_rep["escape_end_s"]),
             meta_min_clearance_non_escape=np.float64(frames_rep["min_clear_ne"]),
             meta_worst_pair_non_escape=np.array(fmt_pair(frames_rep["min_clear_ne_pair"])),
             meta_escape_pairs=np.array([fmt_pair(k) for k in chk.escape0]),
             meta_always_contact=np.array([fmt_pair(k) for k in chk.always_contact]),
             meta_other_arm=np.array(args.other_arm), meta_hand_start=np.array(args.hand_start),
             meta_hand_q=np.array(args.hand_q or ""),
             meta_with_cup=np.bool_(args.with_cup), meta_goal_source=np.array(goal_src),
             meta_exact_min_lb=np.float64(worst["exact_lb"] if worst else np.nan))
    print(f"[plan] 방법 {method} · 프레임 {len(frames)} × dt {args.dt} = {(len(frames) - 1) * args.dt:.1f} s "
          f"· 최고 관절속도 {vel:.4f} rad/s (한계 {vmax}) · 관절한계 {'안' if in_lim else '밖'}")
    print(f"[plan] 최소 여유(margin 규칙 쌍, 탈출 후) {frames_rep['min_clear']:.4f} m — {fmt_pair(frames_rep['min_clear_pair'])}"
          f" · 탈출 쌍 제외 {frames_rep['min_clear_ne']:.4f} m — {fmt_pair(frames_rep['min_clear_ne_pair'])}"
          f" · 탈출 구간 s <= {frames_rep['escape_end_s']:.3f} rad(L-inf 호 길이)")
    print("[plan] 엄밀 검증(볼록 껍질 Frank-Wolfe 하한) 가까운 쌍:")
    for r in cert:
        print(f"    {fmt_pair(r['pair']):<36} mujoco {r['mujoco_dist']:.4f}  엄밀하한 {r['exact_lb']:.4f}  frame {r['frame']}")
    print(f"[plan] 저장 {out}  ({time.time() - t0:.1f} s, 검사 {chk.n_checks} 회)")
    print(f"[plan] 재생: python3 deploy/policy_control/tools/replay_to_pd.py --npz {out} "
          f"--joints {','.join(world.moving_joints)} --rate-scale 1.0   (무발행 확인; --execute 는 사용자 승인 후)")
    return 0


def _print_check(label: str, rep: dict) -> None:
    head = "통과" if rep["ok"] else "실패"
    print(f"[check] {label}: {head} · 샘플 {rep['samples']} · 최소 여유(margin 규칙) {rep['min_clear']:.4f} m "
          f"({fmt_pair(rep['min_clear_pair'])}) · 탈출 쌍 제외 {rep['min_clear_ne']:.4f} m "
          f"({fmt_pair(rep['min_clear_ne_pair'])}) · 탈출 구간 끝 s={rep['escape_end_s']:.3f} rad")
    groups: dict = {}
    for f in rep["fails"]:
        g = groups.setdefault(f["pair"], {"first_s": f["s"], "worst": f})
        g["last_s"] = f["s"]
        if f["slack"] < g["worst"]["slack"]:
            g["worst"] = f
    for k, g in sorted(groups.items(), key=lambda kv: kv[1]["worst"]["slack"]):
        w = g["worst"]
        print(f"    위반 {fmt_pair(k):<36} s {g['first_s']:.3f}~{g['last_s']:.3f} rad · 최악 dist {w['dist']:+.4f} "
              f"(slack {w['slack']:+.4f}) at q={w['q']}")


if __name__ == "__main__":
    raise SystemExit(main())
