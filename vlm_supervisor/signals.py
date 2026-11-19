"""센서 판정 — 손(RH56F1 각도 · 명령 · 관절 힘 · 손끝 촉각) · 물체(FP++ live) 기록으로 단계 끝에 성공 · 실패 · 애매를 낸다.

순수 논리(ROS 없음). 실기 노드는 메시지가 올 때마다 feed_*() 를 부르고, 노드가 끝나면 judge(kind, side, t0, t1) 를 부른다.
같은 함수를 bench/replay_bags.py 가 bag 으로 재생한다. 임계값은 config/signals.yaml(근거 · 건수는 그 파일 주석).

손 배열은 벤더 슬롯 순(새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전). 각도는 0.1°(1740 근처 = 편 손),
힘(force_actual)은 g. angle_set 의 -1 은 "그 축 유지" 라 차이 계산에서 뺀다.

판정 원칙(11.17 실기 bag 32개에서):
- 쥠은 힘이 크면 확실하다(네 손가락 합 · 엄지 굽힘 힘). 힘이 작아도 물체에 막혀 손가락이 명령까지 못 간 경우가 있어
  "명령까지 다 닫혔고(gap 작음) 힘도 작다" 일 때만 빈손으로 본다.
- 쥔 뒤 물체를 놓치면 손가락이 다시 닫히지 않아 gap 이 남는다(벤더 힘 제한). 그래서 이어 쥐는 단계(이송 · 기울이기)는
  단계 안 힘 급락(drop)과 물체 자리로 본다. 힘만 작으면 실패가 아니라 애매다.
- 물체 자리(FP++ live)가 있으면 손바닥 거리 · 높이로 보탠다. 놓은 자리(홀더 안)는 센서로 확정하지 않고 VLM 컵별 질문에 넘긴다.
"""
from __future__ import annotations

import bisect
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

OK, FAIL, UNSURE = "ok", "fail", "unsure"
OPEN_RAW = 1740.0        # 편 손 각도(0.1°) — 실기 bag 시작 값 1733~1742
CLOSE_SPAN = 840.0       # 닫힘 비율 = (OPEN_RAW - 각도) / CLOSE_SPAN. 비율일 뿐 판정에는 gap · 힘을 쓴다
FINGERS = slice(0, 4)    # 새끼 · 약지 · 중지 · 검지
TH_BEND, TH_ROT = 4, 5

KINDS = ("grasp", "hold", "release")
_CFG = Path(__file__).resolve().parent / "config" / "signals.yaml"


def load_cfg(path: Path | str | None = None) -> dict:
    with open(path or _CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================ 버퍼
class _Series:
    """(t, 값 튜플) 시계열. 오래된 것은 maxlen 으로 버린다. t 는 초(단조 증가로 넣는다).
    ROS 콜백 스레드가 넣고 러너 스레드가 읽으므로 넣기 · 읽기를 한 락으로 묶는다(t · v 짝이 어긋나지 않게)."""

    def __init__(self, maxlen: int) -> None:
        self.t: deque = deque(maxlen=maxlen)
        self.v: deque = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.dropped = 0                             # 유한하지 않아 버린 표본 수(NaN · inf 하나가 누적 합을 망친다)

    def add(self, t: float, vals: Sequence[float]) -> None:
        vals = tuple(float(x) for x in vals)
        if not (math.isfinite(t) and all(map(math.isfinite, vals))):
            self.dropped += 1
            return
        self._put(t, vals)

    def _put(self, t: float, vals: tuple) -> None:  # 유한 검사 없이 — target 의 nan 은 "아직 모름" 표시
        with self.lock:
            if self.t and t < self.t[-1]:
                return                               # 늦게 온 메시지는 버린다(창 계산이 정렬을 전제)
            self.t.append(float(t))
            self.v.append(vals)

    def window(self, t0: float, t1: float) -> list[tuple]:
        with self.lock:
            ts, vs = list(self.t), list(self.v)
        a, b = bisect.bisect_left(ts, t0), bisect.bisect_right(ts, t1)
        return list(zip(ts[a:b], vs[a:b]))

    def last_before(self, t: float, max_age: float) -> tuple | None:
        with self.lock:
            i = bisect.bisect_right(self.t, t) - 1
            if i < 0 or t - self.t[i] > max_age:
                return None
            return self.v[i]


def _mean(rows: Sequence[tuple]) -> list[float] | None:
    if not rows:
        return None
    n = len(rows[0][1])
    return [sum(r[1][k] for r in rows) / len(rows) for k in range(n)]


@dataclass
class HandBuffer:
    """한 손의 최근 기록. 60 Hz 손 토픽 기준 maxlen 6000 ≈ 100 s."""
    maxlen: int = 6000
    angle: _Series = field(init=False)
    target: _Series = field(init=False)
    force: _Series = field(init=False)
    tip: _Series = field(init=False)

    def __post_init__(self) -> None:
        self.angle, self.target = _Series(self.maxlen), _Series(self.maxlen)
        self.force, self.tip = _Series(self.maxlen), _Series(self.maxlen)

    def feed_angle(self, t: float, vals: Sequence[float]) -> None:      # /hand_<s>/angle_actual joint_values
        self.angle.add(t, vals)

    def feed_target(self, t: float, vals: Sequence[float]) -> None:     # /hand_<s>/angle_set joint_values (-1 = 유지)
        vals = [float(x) for x in vals]
        if not (math.isfinite(t) and all(map(math.isfinite, vals))):
            self.target.dropped += 1
            return
        prev = self.target.v[-1] if self.target.v else (math.nan,) * len(vals)   # 넣는 스레드는 하나뿐
        self.target._put(t, tuple(p if v < 0 else v for v, p in zip(vals, prev)))

    def feed_force(self, t: float, vals: Sequence[float]) -> None:      # /hand_<s>/force_actual joint_values (g)
        self.force.add(t, vals)

    def feed_tip(self, t: float, vals: Sequence[float]) -> None:        # /hand_<s>/tip_forces_zeroed data (N)
        self.tip.add(t, vals)


@dataclass(frozen=True)
class HandSnap:
    """창 평균 한 장. 없는 값은 None."""
    clos: float              # 네 손가락 닫힘 비율 평균
    gap: float | None        # 네 손가락 (실제 − 명령) 평균, 0.1°. + = 명령보다 덜 닫힘(막힘)
    f4: float                # 네 손가락 힘 합(g)
    ft: float                # 엄지 굽힘 힘(g)
    tip: float | None        # 손끝 촉각 합(N, 영점 뺀 값)
    n: int                   # 창 안 각도 표본 수

    def as_dict(self) -> dict:
        return {k: (None if v is None else round(v, 3)) for k, v in self.__dict__.items()}


def snap(h: HandBuffer, t: float, win: float = 0.5, force_stale_s: float = 0.2) -> HandSnap | None:
    """[t - win, t] 평균. 각도 · 힘이 없거나 마지막 힘 표본이 force_stale_s 넘게 오래됐으면(토픽 끊김) None(판정 보류)."""
    a, f = h.angle.window(t - win, t), h.force.window(t - win, t)
    am = _mean(a) or (lambda v: list(v) if v else None)(h.angle.last_before(t, 1.0))
    fl = h.force.last_before(t, force_stale_s)      # grip_drop 의 stale 보호와 같은 한도
    if am is None or fl is None:
        return None
    fm = _mean(f) or list(fl)
    s = h.target.last_before(t, 1e9)
    gap = None
    if s is not None:
        d = [am[k] - s[k] for k in range(4) if not math.isnan(s[k])]
        gap = sum(d) / len(d) if d else None
    tp = _mean(h.tip.window(t - win, t))
    return HandSnap(clos=sum(OPEN_RAW - x for x in am[FINGERS]) / 4 / CLOSE_SPAN, gap=gap,
                    f4=sum(fm[FINGERS]), ft=fm[TH_BEND], tip=None if tp is None else sum(tp), n=len(a))


def grip_drop(h: HandBuffer, t0: float, t1: float, smooth_s: float = 0.2, within_s: float = 0.5,
              stale_s: float = 0.2) -> dict | None:
    """[t0, t1] 안 쥠 힘(네 손가락 + 엄지 굽힘) — 최대값 · 끝값 · within_s 안에 떨어진 최대 낙폭(이동 평균 smooth_s).
    마지막 힘 표본이 t1 보다 stale_s 넘게 오래됐으면(토픽 끊김) None. stale_s < 0.3 이라 끝값 창도 비지 않는다."""
    rows = h.force.window(t0, t1)
    if len(rows) < 5 or t1 - rows[-1][0] > stale_s:
        return None
    ts = [r[0] for r in rows]
    g = [sum(r[1][:5]) for r in rows]
    gs, acc, j = [], 0.0, 0
    for i, x in enumerate(g):                        # 시간 창 이동 평균
        acc += x
        while ts[i] - ts[j] > smooth_s:
            acc -= g[j]
            j += 1
        gs.append(acc / (i - j + 1))
    best, best_t, j = 0.0, None, 0
    win: deque = deque()                             # within_s 창 최대값(단조 큐)
    for i, x in enumerate(gs):
        while win and gs[win[-1]] <= x:
            win.pop()
        win.append(i)
        while ts[i] - ts[win[0]] > within_s:
            win.popleft()
        d = gs[win[0]] - x
        if d > best:
            best, best_t = d, ts[i] - t0
    tail = [x for t, x in zip(ts, gs) if t >= t1 - 0.3]
    return {"gmax": max(gs), "gend": sum(tail) / len(tail), "drop": best, "drop_t": best_t}


class ObjectBuffer:
    """물체 이름 → (t, x, y, z) base 좌표. FP++ live 를 넣는다."""

    def __init__(self, maxlen: int = 3000) -> None:
        self.maxlen, self.s = maxlen, {}

    def feed(self, name: str, t: float, xyz: Sequence[float]) -> None:
        self.s.setdefault(name, _Series(self.maxlen)).add(t, xyz[:3])

    def at(self, name: str, t: float, win: float = 0.6) -> tuple | None:
        """[t - win, t + 0.05] 중앙값."""
        s = self.s.get(name)
        rows = s.window(t - win, t + 0.05) if s else []
        if not rows:
            return None
        return tuple(sorted(r[1][k] for r in rows)[len(rows) // 2] for k in range(3))

    def names(self) -> list[str]:
        return list(self.s)


# ============================================================================ 판정
@dataclass(frozen=True)
class Verdict:
    label: str                    # ok | fail | unsure
    reasons: tuple[str, ...]
    feats: Mapping = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"label": self.label, "reasons": list(self.reasons), "feats": dict(self.feats)}


def hold_state(s: HandSnap | None, c: Mapping) -> tuple[str, str]:
    """쥔 손 한 장 → (held | empty | unsure, 이유)."""
    if s is None:
        return UNSURE, "손 기록 없음"
    if s.f4 >= c["held_f4_min"] or (s.ft >= c["held_ft_min"] and (s.tip or 0.0) >= c["held_tip_min"]):
        return "held", f"쥠 힘 F4 {s.f4:.0f} g · 엄지 {s.ft:.0f} g · 촉각 {s.tip or 0:.1f} N"
    gap_ok = s.gap is not None and s.gap <= c["empty_gap_max"]
    if gap_ok and s.f4 < c["empty_f4_max"] and s.ft < c["empty_ft_max"] and (s.tip or 0.0) < c["empty_tip_max"]:
        return "empty", f"명령까지 닫힘(gap {s.gap:.0f}) · 힘 작음 F4 {s.f4:.0f} g · 엄지 {s.ft:.0f} g"
    return UNSURE, f"힘 애매 F4 {s.f4:.0f} g · 엄지 {s.ft:.0f} g · gap {s.gap if s.gap is None else round(s.gap)}"


def _obj_vote(objs: ObjectBuffer | None, palm: tuple | None, t0: float, t1: float, kind: str,
              c: Mapping, name: str | None) -> tuple[str | None, str, dict]:
    """물체 자리 → (held | lost | None, 이유, 특징). palm 이 없으면 손 거리 없이 높이만 본다."""
    if objs is None:
        return None, "", {}
    if not name and palm is None:
        return None, "", {}                          # 고를 근거가 없다 — 아무 물체나 보지 않는다(물체 추적 없음과 같게)
    cands = [name] if name else objs.names()
    best = None
    for nm in cands:
        p1 = objs.at(nm, t1)
        if p1 is None:
            continue
        d = math.dist(p1, palm) if palm is not None else None
        key = d if d is not None else 0.0
        if best is None or key < best[0]:
            best = (key, nm, p1, d)
    if best is None:
        return None, "물체 기록 없음", {}
    _, nm, p1, d = best
    p0 = objs.at(nm, t0 + 0.5, 1.0)
    rise = None if p0 is None else p1[2] - p0[2]
    f = {"obj": nm, "obj_z1": round(p1[2], 4), "obj_rise": None if rise is None else round(rise, 4),
         "palm_d": None if d is None else round(d, 4)}
    near = d is not None and d <= c["obj_palm_near_m"]
    far = d is not None and d >= c["obj_palm_far_m"]
    if kind == "grasp":
        if rise is not None and rise >= c["obj_rise_min_m"] and (d is None or near):
            return "held", f"{nm} {rise * 100:.1f} cm 들림", f
        if rise is not None and abs(rise) < c["obj_still_m"] and (d is None or far):
            return "lost", f"{nm} 제자리({rise * 100:+.1f} cm)", f
    elif kind == "hold":
        if near:
            return "held", f"{nm} 손바닥 {d * 100:.1f} cm", f
        if far:
            return "lost", f"{nm} 손에서 {d * 100:.1f} cm", f
    return None, f"{nm} 애매(거리 {f['palm_d']} · 상승 {f['obj_rise']})", f


def judge(kind: str, hand: HandBuffer, t0: float, t1: float, cfg: Mapping, *,
          objs: ObjectBuffer | None = None, palm: tuple | None = None, obj_name: str | None = None) -> Verdict:
    """단계 [t0, t1] 판정.

    kind: grasp(잡고 들기 — 끝에 쥐고 있어야), hold(이미 쥔 채 이송 · 기울이기 — 끝까지 놓치면 안 됨),
          release(놓기 — 끝에 손이 비고 열려 있어야. 놓은 자리는 VLM).
    palm: 단계 끝 손바닥 위치(base, m). 있으면 물체 거리를 쓴다.
    """
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} — {KINDS} 중 하나")
    c = cfg[kind]
    s1 = snap(hand, t1, cfg["window_s"], cfg["force_stale_s"])
    feats: dict = {"end": None if s1 is None else s1.as_dict()}
    if kind == "release":
        if s1 is None:
            return Verdict(UNSURE, ("손 기록 없음",), feats)
        opened = s1.clos <= c["open_clos_max"]
        light = s1.f4 + s1.ft <= c["empty_force_max"]
        if opened and light:
            return Verdict(OK, (f"손 열림(닫힘 {s1.clos:.2f}) · 힘 {s1.f4 + s1.ft:.0f} g — 놓은 자리는 VLM",), feats)
        if s1.f4 >= cfg["grasp"]["held_f4_min"]:
            return Verdict(FAIL, (f"놓기 끝에도 쥠 힘 F4 {s1.f4:.0f} g",), feats)
        return Verdict(UNSURE, (f"손 열림 애매(닫힘 {s1.clos:.2f} · 힘 {s1.f4 + s1.ft:.0f} g)",), feats)

    st, why = hold_state(s1, c)
    dr = grip_drop(hand, t0, t1, cfg["smooth_s"], cfg["drop_within_s"], cfg["force_stale_s"])
    feats["grip"] = None if dr is None else {k: (None if v is None else round(v, 2)) for k, v in dr.items()}
    reasons = [why]
    lost_grip = (dr is not None and dr["gmax"] >= c["drop_gmax_min"]
                 and dr["gend"] <= c["drop_end_ratio"] * dr["gmax"] and dr["gend"] < c["drop_end_max"])
    if lost_grip:
        reasons.append(f"쥠 상실 — 힘 {dr['gmax']:.0f} → {dr['gend']:.0f} g(낙폭 {dr['drop']:.0f} g @ {dr['drop_t']:.1f} s)")
    ov, owhy, of = _obj_vote(objs, palm, t0, t1, kind, c, obj_name)
    feats.update(of)
    if owhy:
        reasons.append(owhy)

    if ov == "lost" or (lost_grip and st != "held"):
        return Verdict(FAIL, tuple(reasons), feats)
    if st == "empty":
        return Verdict(FAIL if kind == "grasp" else UNSURE, tuple(reasons), feats)
    if st == "held" and ov != "lost":
        return Verdict(OK, tuple(reasons), feats)
    if ov == "held" and not lost_grip:
        return Verdict(OK, tuple(reasons), feats)
    return Verdict(UNSURE, tuple(reasons), feats)


# ============================================================================ 노드 → 단계 종류
def stage_kind(node_id: str, name: str = "", cfg: Mapping | None = None) -> str | None:
    """노드 id · 이름 → grasp | hold | release | None(판정 안 함). 규칙은 signals.yaml kinds(앞에서부터 첫 일치)."""
    rules = (cfg or load_cfg())["kinds"]
    key = f"{node_id} {name}".lower()
    for r in rules:
        if any(p in key for p in r["match"]):
            return r["kind"]
    return None


def combine(signal: Mapping, verdicts: Iterable[Verdict]) -> dict:
    """러너 신호 + 센서 판정 → 새 신호(원본은 그대로). 센서 실패가 하나라도 있으면 ok=False 로 내린다.
    센서 성공은 러너 실패를 올리지 않는다(러너 · 사진 판정은 그대로 간다). 애매는 신호를 바꾸지 않는다."""
    vs = list(verdicts)
    out = dict(signal)
    out["sensor"] = [v.as_dict() for v in vs]
    fails = [v for v in vs if v.label == FAIL]
    if fails and out.get("ok"):
        out["ok"] = False
        out["reason"] = ("센서: " + " / ".join(r for v in fails for r in v.reasons[:2])
                         + (f" | 러너: {signal.get('reason')}" if signal.get("reason") else ""))
    return out


# ============================================================================ 노드 → 검사 · 묶음
@dataclass(frozen=True)
class Check:
    side: str
    kind: str                     # grasp | hold | release
    obj: str | None               # FP++ 이름(/objects/<obj>/live_pose). None = 손만 본다
    node: str


def _fpp_name(ep, obj: str | None) -> str | None:
    """에피소드 물체 키(SRC_BLUE) → FP++ 이름(source240_blue). 토픽이 /objects/<이름>/pose 꼴이 아니면 None."""
    if not obj:
        return None
    topic = str((ep.objects.get(obj) or {}).get("topic", ""))
    parts = topic.strip("/").split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "objects" else None


def _node_checks(ep, n, held: Mapping[str, str | None]) -> list[Check]:
    if n.type in ("policy", "parallel_policy"):
        out = []
        for j in n.jobs:
            b = ep.policies.get(j.role)
            if b is None:
                continue
            if j.target_holder or b.kind == "place":
                out.append(Check(b.side, "release", _fpp_name(ep, j.source_object), n.id))
            elif j.target_object:
                out.append(Check(b.side, "grasp", _fpp_name(ep, j.target_object), n.id))
        return out
    if n.type == "trajectory":
        tr = ep.trajectories.get(n.name)
        sides = tr.sides if tr is not None else ()
        return [Check(s, "hold", held.get(s), n.id) for s in sides if held.get(s)]
    if n.type == "hand":
        kind = "grasp" if n.action == "grasp" else "release" if n.action == "open" else None
        return [Check(s, kind, None, n.id) for s in n.sides] if kind else []
    return []


def plan_checks(ep, node, held: Mapping[str, str | None]) -> list[Check]:
    """경계 노드 → 검사 목록. lanes 는 팔마다 줄 전체를 한 구간으로 본다: 줄에 잡기가 있으면 grasp(끝에 쥐고 있어야),
    아니면 그 줄 마지막 항목의 검사. held = 팔 → 지금 쥔 FP++ 물체(SensorHub 가 판정으로 갱신)."""
    if node.type != "lanes":
        return _node_checks(ep, node, held)
    out = []
    members = [m for m in ep.nodes if m.group == node.id]
    for side in dict.fromkeys(m.lane for m in members):
        cs = [c for m in members if m.lane == side for c in _node_checks(ep, m, held)]
        g = [c for c in cs if c.kind == "grasp"]
        if g:
            out.append(Check(side, "grasp", g[-1].obj, node.id))
        elif cs:
            out.append(Check(side, cs[-1].kind, cs[-1].obj, node.id))
    return out


class ClockSkewError(RuntimeError):
    """버퍼에 표본은 있는데 판정 시각 근처에 하나도 없다 — 러너 시계와 메시지 시각이 다른 시계다."""


_CFG_KEYS = ("window_s", "smooth_s", "drop_within_s", "force_stale_s", "clock_skew_max_s",
             *(f"{k}.{x}" for k in ("grasp", "hold") for x in (
                 "held_f4_min", "held_ft_min", "held_tip_min", "empty_gap_max", "empty_f4_max", "empty_ft_max",
                 "empty_tip_max", "drop_gmax_min", "drop_end_ratio", "drop_end_max", "obj_rise_min_m", "obj_still_m",
                 "obj_palm_near_m", "obj_palm_far_m")),
             "release.open_clos_max", "release.empty_force_max")


def _missing(cfg: Mapping) -> list[str]:
    out = []
    for key in _CFG_KEYS:
        c = cfg
        for part in key.split("."):
            c = c.get(part) if isinstance(c, Mapping) else None
        if c is None:
            out.append(key)
    return out


class SensorHub:
    """실기 러너가 쓰는 묶음 — 메시지 버퍼 + 노드 경계 판정. BoundaryRunner(sensors=hub) 로 넘긴다.

    ROS 쪽은 콜백에서 hub.hand[side].feed_*(t, values) · hub.objs.feed(name, t, xyz) 를 부른다(t = 같은 시계, 초).
    palm(side, t) 는 그 시각 손바닥 위치(base, m) 를 주는 함수(손바닥 FK). None 이면 물체 손 거리는 안 본다.
    clock 은 BoundaryRunner 와 같은 시계여야 한다(t0 · t1 이 그 시계로 온다).
    """

    def __init__(self, ep, cfg: Mapping | None = None, palm=None, maxlen: int = 6000) -> None:
        self.ep, self.cfg, self.palm = ep, cfg or load_cfg(), palm
        if miss := _missing(self.cfg):
            raise ValueError(f"signals 설정에 키가 없다: {', '.join(miss)}")
        self.hand = {"right": HandBuffer(maxlen), "left": HandBuffer(maxlen)}
        self.objs = ObjectBuffer(maxlen)
        self.held: dict[str, str | None] = {"right": None, "left": None}

    def __call__(self, node, t0: float, t1: float) -> list[Verdict]:
        out, checks = [], plan_checks(self.ep, node, self.held)
        if checks:
            self._check_clock(t1)
        for c in checks:
            p = self.palm(c.side, t1) if self.palm is not None else None
            v = judge(c.kind, self.hand[c.side], t0, t1, self.cfg, objs=self.objs, palm=p, obj_name=c.obj)
            v = Verdict(v.label, v.reasons, {**v.feats, "side": c.side, "kind": c.kind, "node": c.node})
            out.append(v)
            if c.kind == "release" and v.label == OK:
                self.held[c.side] = None
            elif c.kind == "grasp":
                self.held[c.side] = None if v.label == FAIL else (c.obj or "?")
            elif c.kind == "hold" and v.label == FAIL:
                self.held[c.side] = None
        return out

    def _check_clock(self, t1: float) -> None:
        """표본이 있는데 t1 ± clock_skew_max_s 안에 하나도 없으면 ClockSkewError(모두 unsure 로 조용히 지나가지 않게).
        실기에선 미래 표본이 없어 "가장 새 표본이 t1 과 어긋남" 과 같다. bag 재생(앞으로 다 넣어 둠)도 오경보 없다."""
        sk = self.cfg["clock_skew_max_s"]
        ss = [s for h in self.hand.values() for s in (h.angle, h.target, h.force, h.tip)] + list(self.objs.s.values())
        ss = [s for s in ss if s.t]
        if ss and not any(s.window(t1 - sk, t1 + sk) for s in ss):
            newest = max(s.t[-1] for s in ss)
            raise ClockSkewError(f"판정 시각 {t1:.3f} 근처 ±{sk} s 에 표본 없음(가장 새 표본 {newest:.3f}) — 시계가 다르다")
