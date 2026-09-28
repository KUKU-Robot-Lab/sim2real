"""실기 joint 정책 기록(npz)을 읽어 "제어기가 정책을 따라갔는가" 를 잰다. 순수(numpy 만).

09.28 사용자: "실기에서는 원하는대로 어느정도 된것 같은데 제어기 등이 잘 따라갔는지를 토대로 재학습할수도 있으니까".
기록기(tools/joint_recorder.py)가 남긴 스트림을 시각으로 맞춰
  · 정책 주기 — 목표(joint_target) 간격 · 끊김 · 한 스텝 계산 시간(proc_ms)
  · 추종 — 목표 q* 와 실측 q 의 차이(지연 0 과 가장 잘 맞는 지연에서), 관절별 RMS · 최대
  · pd 가 목표를 바꿨는가 — pd applied 와 목표의 차이(속도 · 한계 클립)
  · 행동 포화 — |a| ≥ 1 비율(팔 증분 · 손 절대)
  · 물체 입력 — 출처(live · held · attached · missing) 횟수
를 낸다. 판정(좋다/나쁘다)은 하지 않는다 — 숫자와 기준(학습 trace)을 나란히 보는 것은 사람이 한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

#: 추종 지연을 찾아볼 범위 [s] — pd 램프 · CAN 왕복을 넉넉히 덮는다
LAGS_S = np.arange(0.0, 0.301, 0.01)


@dataclass(frozen=True)
class Track:
    names: tuple
    rms0: np.ndarray        # 지연 0 에서 관절별 RMS [rad]
    max0: np.ndarray        # 지연 0 에서 관절별 |err| 최대
    lag_s: float            # 전체 RMS 를 가장 작게 하는 지연
    rms_lag: np.ndarray     # 그 지연에서 관절별 RMS
    n: int


def statuses(t: np.ndarray, texts) -> list[tuple[float, dict]]:
    out = []
    for ti, s in zip(t, texts):
        try:
            out.append((float(ti), json.loads(str(s))))
        except ValueError:
            continue
    return out


def running_window(jn: list[tuple[float, dict]], t_tgt: np.ndarray) -> tuple[float, float]:
    """정책이 돈 구간 = joint_node phase 'running' 의 처음 ~ 끝. status 가 없으면 목표가 온 구간."""
    run = [t for t, s in jn if s.get("phase") == "running"]
    if run:
        return min(run), max(run)
    if t_tgt.size:
        return float(t_tgt.min()), float(t_tgt.max())
    return 0.0, 0.0


def _interp(t_q: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, t_q, q[:, j]) for j in range(q.shape[1])], axis=1)


def tracking(names, t_tgt, q_tgt, t_meas, q_meas, lags=LAGS_S) -> Track | None:
    """목표 시각 t 의 q*(t) 와 실측 q(t + lag). 실측 구간 밖의 목표는 버린다."""
    if len(t_tgt) < 2 or len(t_meas) < 2:
        return None
    best = (np.inf, 0.0, None)
    rms0 = max0 = None
    n = 0
    for lag in lags:
        tt = t_tgt + lag
        keep = (tt >= t_meas[0]) & (tt <= t_meas[-1])
        if keep.sum() < 2:
            continue
        err = _interp(t_meas, q_meas, tt[keep]) - q_tgt[keep]
        rms = np.sqrt(np.mean(err ** 2, axis=0))
        if lag == lags[0]:
            rms0, max0, n = rms, np.abs(err).max(axis=0), int(keep.sum())
        tot = float(np.sqrt(np.mean(rms ** 2)))
        if tot < best[0]:
            best = (tot, float(lag), rms)
    if rms0 is None or max0 is None or best[2] is None:
        return None
    return Track(tuple(names), rms0, max0, best[1], best[2], n)


def pick(names_all, want) -> list[int]:
    idx = {n: i for i, n in enumerate(names_all)}
    return [idx[w] for w in want if w in idx]


def summarize(d) -> dict:
    """npz(dict 처럼 읽히는 것) → 요약 dict. 스트림이 비면 그 항목은 None."""
    hz = float(d["meta_policy_hz"])
    jn = statuses(d["jn_t"], d["jn_json"])
    t_tgt = np.asarray(d["tgt_t"], float)
    t0, t1 = running_window(jn, t_tgt)
    win = (t_tgt >= t0) & (t_tgt <= t1)
    out: dict = {"window_s": t1 - t0, "policy_hz": hz, "steps": int(win.sum())}
    gaps = np.diff(t_tgt[win]) if win.sum() > 1 else np.zeros(0)
    out["rate_hz"] = float(len(gaps) / gaps.sum()) if gaps.size and gaps.sum() > 0 else None
    out["gap_max_ms"] = float(gaps.max() * 1e3) if gaps.size else None
    out["gaps_over_2dt"] = int((gaps > 2.0 / hz).sum()) if gaps.size else 0
    proc = np.array([s["proc_ms"] for t, s in jn if t0 <= t <= t1 and "proc_ms" in s], float)
    out["proc_ms"] = ({"p50": float(np.percentile(proc, 50)), "p95": float(np.percentile(proc, 95)),
                       "max": float(proc.max())} if proc.size else None)
    src: dict = {}
    for t, s in jn:
        if t0 <= t <= t1 and s.get("obj_source"):
            src[s["obj_source"]] = src.get(s["obj_source"], 0) + 1
    out["obj_source"] = src
    tnames = [str(n) for n in d["tgt_names"]]
    arm = [str(n) for n in d["arm_names"]]
    hand = [str(n) for n in d["hand_names"]]
    q_tgt = np.asarray(d["tgt_q"], float)
    out["arm"] = out["hand"] = None
    if win.sum() > 1:
        ia = pick(tnames, arm)
        if len(ia) == len(arm) and len(d["arm_t"]):
            out["arm"] = tracking(arm, t_tgt[win], q_tgt[win][:, ia], np.asarray(d["arm_t"], float),
                                  np.asarray(d["arm_q"], float))
        hn = [h for h in hand if h in tnames]
        if hn and len(d["hand_t"]):
            ih_t, ih_m = pick(tnames, hn), pick(hand, hn)
            out["hand"] = tracking(hn, t_tgt[win], q_tgt[win][:, ih_t], np.asarray(d["hand_t"], float),
                                   np.asarray(d["hand_q"], float)[:, ih_m])
    out["applied_vs_target"] = None
    anames = [str(n) for n in d["app_names"]]
    if win.sum() > 1 and len(d["app_t"]):
        common = [n for n in arm if n in anames and n in tnames]
        if common:
            ta = np.asarray(d["app_t"], float)
            keep = (ta >= t0) & (ta <= t1)
            if keep.sum() > 1:
                qa = np.asarray(d["app_q"], float)[keep][:, pick(anames, common)]
                qt = _interp(t_tgt[win], q_tgt[win][:, pick(tnames, common)], ta[keep])
                diff = qa - qt
                out["applied_vs_target"] = {"names": common, "rms": np.sqrt(np.mean(diff ** 2, axis=0)),
                                            "max": np.abs(diff).max(axis=0)}
    out["tips"] = None
    if "tip_t" in d and len(d["tip_t"]):
        tt, ti, tw = np.asarray(d["tip_t"], float), np.asarray(d["tip_idx"], int), np.asarray(d["tip_wrench"], float)
        keep = (tt >= t0) & (tt <= t1)
        rows = {}
        for i in range(1, 6):
            k = keep & (ti == i)
            if k.any():
                fn = np.linalg.norm(tw[k, :3], axis=1)
                rows[i] = {"n": int(k.sum()), "f_max": float(fn.max()), "f_p50": float(np.percentile(fn, 50)),
                           "t_max": float(tt[k][int(np.argmax(fn))] - t0)}
        out["tips"] = rows
    out["tactile"] = None
    if "tac_t" in d and len(d["tac_t"]):
        ct, ci, cv = np.asarray(d["tac_t"], float), np.asarray(d["tac_idx"], int), np.asarray(d["tac"], float)
        rows = {}
        for i in range(1, 6):
            k = ci == i
            if not k.any():
                continue
            pre = k & (ct < t0) & (ct >= t0 - 1.0)          # 영점 = 시작 전 1 s 평균(팔 정책 시작 자세 · 무접촉)
            base = np.nanmean(cv[pre], axis=0) if pre.any() else np.nanmean(cv[k][:30], axis=0)
            w = k & (ct >= t0) & (ct <= t1)
            if not w.any():
                continue
            delta = cv[w] - base
            tot = np.nansum(delta, axis=1)
            rows[i] = {"n": int(w.sum()), "base_sum": float(np.nansum(base)), "sum_max": float(tot.max()),
                       "cell_max": float(np.nanmax(delta)), "t_max": float(ct[w][int(np.argmax(tot))] - t0)}
        out["tactile"] = rows
    act = np.asarray(d["act"], float)
    ta = np.asarray(d["act_t"], float)
    keep = (ta >= t0) & (ta <= t1) if ta.size else np.zeros(0, bool)
    if keep.sum():
        a = act[keep]
        na = len(arm)
        out["action_sat"] = {"arm": float((np.abs(a[:, :na]) >= 1.0).mean()),
                             "hand": float((np.abs(a[:, na:]) >= 1.0).mean()) if a.shape[1] > na else None,
                             "per_dim": (np.abs(a) >= 1.0).mean(axis=0)}
    else:
        out["action_sat"] = None
    return out


def _row(names, *cols, fmt="{:+.4f}") -> list[str]:
    return [f"  {n:16s} " + "  ".join(fmt.format(float(c[i])) for c in cols) for i, n in enumerate(names)]


def render(s: dict) -> str:
    L = [f"정책 구간 {s['window_s']:.1f} s · 목표 {s['steps']} 개 · 계약 {s['policy_hz']:.0f} Hz"]
    if s["rate_hz"] is not None:
        L.append(f"실측 주기 {s['rate_hz']:.1f} Hz · 최대 간격 {s['gap_max_ms']:.0f} ms · 2 dt 넘은 간격 {s['gaps_over_2dt']}")
    if s["proc_ms"]:
        p = s["proc_ms"]
        L.append(f"한 스텝 계산 p50 {p['p50']:.1f} · p95 {p['p95']:.1f} · 최대 {p['max']:.1f} ms")
    L.append(f"물체 입력 출처 {s['obj_source'] or '없음'}")
    for key, title in (("arm", "팔"), ("hand", "손")):
        tr = s[key]
        if tr is None:
            L.append(f"{title} 추종: 기록 없음")
            continue
        L.append(f"{title} 추종 (목표 q* − 실측 q, {tr.n} 표본) — 가장 잘 맞는 지연 {tr.lag_s * 1e3:.0f} ms")
        L.append(f"  {'관절':16s} {'RMS@0':>8s}  {'최대@0':>8s}  {'RMS@지연':>8s}")
        L += _row(tr.names, tr.rms0, tr.max0, tr.rms_lag, fmt="{:8.4f}")
    av = s["applied_vs_target"]
    if av:
        L.append("pd 가 보낸 값 − 정책 목표 (0 이 아니면 pd 가 속도 · 한계로 목표를 깎았다)")
        L += _row(av["names"], av["rms"], av["max"], fmt="{:8.4f}")
    tips = s.get("tips")
    if tips is None:
        L.append("손끝 F/T: 기록 없음(F/T 센서 손이 아니거나 드라이버 ft_broadcaster 가 꺼져 있었다)")
    else:
        names = {1: "엄지", 2: "검지", 3: "중지", 4: "약지", 5: "새끼"}
        L.append("손끝 힘 |F| [N] — 손끝별 중앙값 · 최대 (최대 시각)")
        L += [f"  {names[i]}  {r['f_p50']:.2f} · {r['f_max']:.2f} ({r['t_max']:.1f} s, {r['n']} 표본)" for i, r in sorted(tips.items())]
    tac = s.get("tactile")
    if tac:
        names = {1: "엄지", 2: "검지", 3: "중지", 4: "약지", 5: "새끼"}
        L.append("손끝 촉각(원시 단위, 시작 전 1 s 평균을 0 으로) — 손끝별 칸 합 최대 · 한 칸 최대 (최대 시각) · 영점 합")
        L += [f"  {names[i]}  {r['sum_max']:.0f} · {r['cell_max']:.0f} ({r['t_max']:.1f} s) · 영점 {r['base_sum']:.0f}"
              for i, r in sorted(tac.items())]
    sat = s["action_sat"]
    if sat:
        hand = "—" if sat["hand"] is None else f"{sat['hand'] * 100:.1f} %"
        L.append(f"행동 포화 |a| ≥ 1 — 팔 {sat['arm'] * 100:.1f} % · 손 {hand}")
    return "\n".join(L)
