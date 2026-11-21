#!/usr/bin/env python3
"""pd 보상 조건 비교 분석 — tilt_comp_ab.py 출력 폴더 하나 → 조건별 열 표(report.md · report.json). 파일만 읽는다.

★판정 = 손끝 오차(관절 오차는 참고). 구간(단계)마다 · 팔마다 · 정지/이동(지령 applied 가 0.1 s 동안 MOVE_RAD_S 넘게 움직이면 이동)마다:
  · 손끝 |Δ| [mm] = |FK(실측) − FK(applied ZOH)|, palm sensor(policy_control.tin_live.palm_fk = fk_numpy.UrdfChainFK) — 평균 · p95 · 최대
  · 관절 j1..j7 [°] = 실측 − applied — 평균(부호) · |오차| p95 · |오차| 최대
  조건별로 사이클 평균 [최소~최대]. 첫 조건 대비 차이가 두 조건 중 큰 사이클 간 범위(최대 − 최소) 안이면 "구분 안 됨",
  어느 조건이든 사이클 1 개면 "사이클 부족". 재생기 끝 줄(발행 · 늦음 · 최대 지연 · 간격 p95)은 회차 메타에서 표로(늦음 > 0 이면 ★).
아래는 tilt 전용 지표(예전 A/B 표 그대로):

오차 = 실측 /joint_states − pd 가 실제로 낸 목표(/policy_control/pd_<팔>/applied, ZOH). 처짐을 지우지 않는다
(arm_latency_report 는 2 s 조각 평균을 빼서 처짐이 사라진다 — 읽기 함수 read_bag · zoh 만 쓴다).
  · 정지 = tilt 직전 정지 구간 끝 2 s
  · 최대 기울기 = tilt 재생 중 applied 오른팔 FK × T_in_r(경로 check.json)의 병 축 기울기가 최대 − 3° 안인 구간(A · B 같은 명령)
  · 왼손 끝 = l_hl_palm_sensor 위치, FK(실측) − FK(applied) [mm] — z 와 3D 크기
  · tilt_comp_tau = /policy_control/pd_<팔>/diagnostics(tilt 재생 동안) 관절별 |τ| 최대,
    한계 = 런 폴더 pd yaml 사본(pd_config_left.yaml)의 tilt_comp.<팔>.limit(블록 없으면 None)
  · 모델 부족 토크 = 구간 평균(실측 /joint_states effort − applied effort(pd 가 보낸 τ_ff)) [N·m]
목표: 양팔 j1..j7 |평균 오차| ≤ 0.3°(정지 · 최대 기울기 각각) · 손 끝 |오차| ≤ 3 mm.

    python3 deploy/policy_control/tools/tilt_comp_ab_report.py logs/runs/<YYMMDD>/tilt_comp_ab/<HHMMSS>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bag as RB  # noqa: E402  (deploy/policy_control 를 sys.path 에 올린다)
from arm_latency_report import read_bag, zoh  # noqa: E402
from tilt_spout_report import MAX_BAND_DEG  # noqa: E402

PRE_S = 2.0
JOINT_GOAL_DEG, TIP_GOAL_MM = 0.3, 3.0
FK_DT_S = 1 / 75           # 손끝 FK 는 약 13 ms 마다(시각 기준 — 표본 수 기준이면 /joint_states 빈도에 따라 바뀐다). 짧은 봉우리는 놓칠 수 있다
MIN_TIP_N = 3              # 손끝 표본이 이보다 적은 칸은 '—'
MIN_PAIRS = 3              # 짝(같은 회차의 조건 쌍)이 이보다 적으면 '사이클 부족'
MOVE_RAD_S, MOVE_WIN_S = 0.01, 0.1   # 지령이 0.1 s 동안 이 속도 넘게 움직이면 이동 구간
MIN_STEP_S = 0.3           # 이보다 짧은 단계(검사 등)는 구간 표에서 뺀다
#: 참고(하중 · 지표가 다르다): 150115 n07b 실기, 왼손에 셰이커 117 g · 지표는 셰이커 입구(손 끝 아님)
REF_150115 = {"j4_pre_deg": -0.87, "j4_max_deg": -1.15, "inlet_sag_mm": "8~14"}


def window(t: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    return (t >= w[0]) & (t <= w[1])


def mean_gap(tm, xm, ta, xa, w) -> np.ndarray | None:
    """구간 평균(실측 − applied ZOH), 관절별 — 단위는 입력 그대로."""
    sel = window(tm, w)
    if sel.sum() < 2 or len(ta) == 0:
        return None
    return (xm[sel] - zoh(ta, xa, tm[sel])).mean(axis=0)


def joint_err_deg(tm, qm, ta, qa, w) -> np.ndarray | None:
    """구간 평균(실측 − applied) [°], 관절별."""
    g = mean_gap(tm, qm, ta, qa, w)
    return None if g is None else np.degrees(g)


def effort_series(db: Path, side: str, w) -> tuple:
    """구간의 실측 /joint_states · pd applied effort — (tm, em, ta, ea), 팔 관절 7개 [N·m]."""
    names = {"/joint_states": [f"openarm_{side}_joint{i}" for i in range(1, 8)],
             f"/policy_control/pd_{side}/applied": [f"{side[0]}_aj_{i}" for i in range(1, 8)]}
    got = RB.read(db, list(names), w[0] - 0.1, w[1])   # applied ZOH 가 구간 첫 표본 앞 값을 쓰게 0.1 s 앞부터
    out: list = []
    for topic, ns in names.items():
        t, v = [], []
        for ts, m in got[topic]:
            idx = {n: i for i, n in enumerate(m.name)}
            if len(m.effort) == len(m.name) and all(n in idx for n in ns):
                t.append(ts)
                v.append([m.effort[idx[n]] for n in ns])
        out += [np.asarray(t), np.asarray(v, float).reshape(-1, 7)]
    return tuple(out)


def goal_ok(left, right, tip_mm) -> bool:
    """양팔 j1..j7 |오차| ≤ JOINT_GOAL_DEG 이고 손 끝 ≤ TIP_GOAL_MM — 하나라도 없으면 실패."""
    if left is None or right is None or tip_mm is None:
        return False
    return float(np.abs(np.r_[left, right]).max()) <= JOINT_GOAL_DEG and tip_mm <= TIP_GOAL_MM


def fk_pick(t: np.ndarray) -> np.ndarray:
    """t(오름차순) 에서 FK_DT_S 간격으로 고른 색인."""
    if len(t) == 0:
        return np.zeros(0, int)
    return np.unique(np.searchsorted(t, np.arange(t[0], t[-1] + 1e-9, FK_DT_S)))


def tip_err_mm(tm, qm, ta, qa, w, fk) -> tuple[float, float] | None:
    """손 끝 위치 오차 평균 — (z, 3D 크기) [mm]."""
    sel = np.flatnonzero(window(tm, w))
    sel = sel[fk_pick(tm[sel])]
    if len(sel) < MIN_TIP_N or len(ta) == 0:
        return None
    qa_z = zoh(ta, qa, tm[sel])
    d = np.array([fk(m)[:3, 3] - fk(c)[:3, 3] for m, c in zip(qm[sel], qa_z)]) * 1e3
    return float(d[:, 2].mean()), float(np.linalg.norm(d, axis=1).mean())


def tilt_window(ta_r, qa_r, R_in_r, w_tilt, fk_r) -> tuple[tuple[float, float], float] | None:
    """applied 오른팔 → 병 축 기울기 → 최대 − MAX_BAND_DEG 안 구간(첫 · 끝 시각)과 최대 기울기[°]."""
    sel = np.flatnonzero(window(ta_r, w_tilt))
    if len(sel) < 2:
        return None
    tilt = np.array([np.degrees(np.arccos(np.clip((fk_r(q)[:3, :3] @ R_in_r)[2, 2], -1, 1))) for q in qa_r[sel]])
    hit = sel[tilt >= tilt.max() - MAX_BAND_DEG]
    return (float(ta_r[hit[0]]), float(ta_r[hit[-1]])), float(tilt.max())


def tau_stats(rows: list[tuple[float, dict]], w, limit) -> dict:
    """diagnostics [(시각, dict)] → tilt_comp 활성 표본 수 · 관절별 |τ| 최대 · 관절별 한계 도달(limit None 이면 None)."""
    tau = np.array([d["tilt_comp_tau"] for t, d in rows if w[0] <= t <= w[1] and "tilt_comp_tau" in d], float).reshape(-1, 7)
    if not len(tau):
        return {"n": 0, "max_nm": None, "limit_hit": None}
    amax = np.abs(tau).max(axis=0)
    hit = None
    if limit is not None:
        lim = np.asarray(limit, float)
        hit = ((lim > 0) & (amax >= lim - 1e-6)).tolist()
    return {"n": int(len(tau)), "max_nm": amax.round(4).tolist(), "limit_hit": hit}


def payload_check(row: dict, meta: dict, diag: list[tuple[float, dict]]) -> float | None:
    """★11.17 회차의 실제 pd payload kg — diagnostics(tilt 재생 동안) 첫 값, 없으면 회차 시작 때 status 값. 의도(meta payload_kg)와 다르면 메모."""
    got = next((d["tilt_comp_payload"][0] for _, d in diag if d.get("tilt_comp_payload")), None)
    if got is None and meta.get("pd_payload"):
        got = meta["pd_payload"][0]
    want = meta.get("payload_kg")
    if got is None:
        row["notes"].append("pd payload 실제값 기록 없음")
    elif want is not None and abs(got - want) > 1e-6:
        row["notes"].append(f"pd payload {got:g} kg ≠ 의도 {want:g} kg")
    return got


def step_t(meta: dict, sid: str) -> tuple[float, float] | None:
    s = next((s for s in meta["steps"] if s["id"] == sid and s["rc"] == 0), None)
    return None if s is None else (float(s["t0"]), float(s["t1"]))


def moving(ta, qa, t) -> np.ndarray:
    """t 시각마다 지령(applied)이 움직이는 중인가 — 지난 MOVE_WIN_S 동안 관절 변위 최대 / 창 > MOVE_RAD_S."""
    if len(ta) < 2:
        return np.zeros(len(t), bool)
    j = np.searchsorted(ta, ta - MOVE_WIN_S)
    return zoh(ta, np.abs(qa - qa[j]).max(axis=1) / MOVE_WIN_S > MOVE_RAD_S, t)


def err_stats(meas, applied, w, fk) -> dict | None:
    """구간 오차 — {"move" | "still": {n, mean_deg, p95_deg, max_deg (j1..j7), tip_mean_mm, tip_p95_mm, tip_max_mm} | None}."""
    (tm, qm), (ta, qa) = meas, applied
    sel = window(tm, w)
    if sel.sum() < 2 or len(ta) == 0:
        return None
    t, q = tm[sel], qm[sel]
    c = zoh(ta, qa, t)
    e = np.degrees(q - c)
    mv = moving(ta, qa, t)
    k = fk_pick(t)
    tip = np.array([np.linalg.norm(fk(q[i])[:3, 3] - fk(c[i])[:3, 3]) for i in k]) * 1e3
    out = {}
    for ph, m in (("move", mv), ("still", ~mv)):
        if m.sum() < 2:
            out[ph] = None
            continue
        ae, tp = np.abs(e[m]), tip[m[k]]
        ok = len(tp) >= MIN_TIP_N
        out[ph] = {"n": int(m.sum()), "mean_deg": e[m].mean(axis=0).round(4).tolist(),
                   "p95_deg": np.percentile(ae, 95, axis=0).round(4).tolist(), "max_deg": ae.max(axis=0).round(4).tolist(),
                   "tip_n": int(len(tp)),
                   "tip_mean_mm": float(tp.mean()) if ok else None,
                   "tip_p95_mm": float(np.percentile(tp, 95)) if ok else None,
                   "tip_max_mm": float(tp.max()) if ok else None}
    return out


def step_metrics(meta: dict, arms: dict, fk: dict) -> dict:
    """단계마다 · 팔마다 err_stats. MIN_STEP_S 보다 짧은 단계는 뺀다."""
    out = {}
    for st in meta["steps"]:
        w = (float(st["t0"]), float(st["t1"]))
        if w[1] - w[0] >= MIN_STEP_S:
            out[st["id"]] = {s: err_stats(a["meas"], a["applied"], w, fk[s]) for s, a in arms.items()}
    return out


def cycle_limits(meta: dict, fallback: Path) -> dict:
    """회차의 팔별 tilt_comp limit — 회차 메타 pd_config(조건 yaml)에서, 없으면 런 폴더 pd_config_left.yaml."""
    out = {}
    for s in ("left", "right"):
        p = Path((meta.get("pd_config") or {}).get(s) or fallback)
        cfg = yaml.safe_load((p if p.is_file() else fallback).read_text(encoding="utf-8")) or {}
        out[s] = ((cfg.get("tilt_comp") or {}).get(s) or {}).get("limit")
    return out


POSE_TRIM_S = 0.3     # 창 단계 끝 = 재생 프로세스 끝(마지막 프레임 뒤 노드 정리) — 그만큼 뺀다
POSE_SETTLE_S = 0.5   # 유지 첫 구간(접근 끝 과도)도 뺀다 — 유지 3 s 면 2.5 s 평균. 끝난 뒤도 pd 가 마지막 목표를 붙드니 정리가 길어도 같은 자리다


def pose_rows(meta: dict, arms: dict) -> list[dict]:
    """--pose 창마다 유지 구간 평균(실측 − applied) [°], 팔 · 관절별. 창 정보는 준비가 쓴 windows.json."""
    p = Path(meta.get("pose") or "")
    if not p.is_file():
        return []
    out = []
    for wn in json.loads(p.read_text(encoding="utf-8"))["windows"]:
        t = step_t(meta, wn["id"])
        if t is None:
            continue
        w = (t[1] - POSE_TRIM_S - wn["hold_s"] + POSE_SETTLE_S, t[1] - POSE_TRIM_S)
        e = {s: joint_err_deg(*a["meas"], *a["applied"], w) for s, a in arms.items()}
        out.append({**{k: wn[k] for k in ("id", "pose", "d", "rep", "grip")}, "t": list(w),
                    **{s: None if v is None else np.round(v, 4).tolist() for s, v in e.items()}})
    return out


def render_pose(rows: list[dict]) -> list[str]:
    """창별 표 + (조건, 자세, d) 평균 표. d = +1 위(목표 + offset)에서 · −1 아래에서 접근."""
    P = [{**r, "cond": row["condition"], "cycle": row["cycle"]} for row in rows for r in row.get("pose") or []]
    if not P:
        return []
    J = " | ".join(f"j{j}" for j in range(1, 8))
    sep = "|" + "---|" * 7
    out = ["", "## 자세 유지 정지 오차 [°] (실측 − applied, 창 끝 유지 구간 평균)", "", "### 창별", "",
           f"| 사이클 | 조건 | 자세 | d | 회 | 손 | 팔 | {J} |", sep + "---|" * 7]
    for r in P:
        for s in ("left", "right"):
            out.append(f"| {r['cycle']} | {r['cond']} | {r['pose']} | {r['d']:+d} | {r['rep']} | {r['grip']} | {s} | "
                       + " | ".join(_fmt(x) for x in (r[s] or [None] * 7)) + " |")
    out += ["", "### d 별 평균", "", f"| 조건 | 자세 | d | 창 | 팔 | {J} |", sep + "---|" * 5]
    for key in dict.fromkeys((r["cond"], r["pose"], r["d"]) for r in P):
        g = [r for r in P if (r["cond"], r["pose"], r["d"]) == key]
        for s in ("left", "right"):
            v = [r[s] for r in g if r[s] is not None]
            m = np.mean(v, axis=0) if v else [None] * 7
            out.append(f"| {key[0]} | {key[1]} | {key[2]:+d} | {len(v)} | {s} | " + " | ".join(_fmt(x) for x in m) + " |")
    return out


def cycle_report(meta: dict, limits: dict, R_in_r, fk: dict) -> dict:
    row = {"cycle": meta["cycle"], "condition": meta["condition"], "notes": [],
           "replay": [{"step": st["id"], **x} for st in meta["steps"] for x in (st.get("extra") or {}).get("replay", [])]}
    hold, tilt = step_t(meta, "pre_tilt_hold"), step_t(meta, "tilt")
    bag = Path(meta.get("bag") or "")
    if not (bag / "arm").is_dir():
        row["notes"].append("bag 없음 — 표에서 뺌")
        return row
    L, R = read_bag(bag / "arm", "left"), read_bag(bag / "arm", "right")
    row["steps"] = step_metrics(meta, {"left": L, "right": R}, fk)
    row["pose"] = pose_rows(meta, {"left": L, "right": R})
    if hold is None or tilt is None:
        row["notes"].append("tilt 단계 없음 — tilt 지표에서 뺌")
        return row
    wins = {"pre": (hold[1] - PRE_S, hold[1])}
    tw = tilt_window(*R["applied"], R_in_r, tilt, fk["right"])
    if tw is None:
        row["notes"].append("tilt 중 오른팔 applied 없음")
    else:
        wins["max"], row["max_tilt_deg"] = tw
        row["max_window_s"] = tw[0][1] - tw[0][0]
    arm_db = RB.db_of(bag, "arm")
    for k, w in wins.items():
        e = joint_err_deg(*L["meas"], *L["applied"], w)
        er = joint_err_deg(*R["meas"], *R["applied"], w)
        tip = tip_err_mm(*L["meas"], *L["applied"], w, fk["left"])
        row[k] = {"left_deg": None if e is None else e.round(4).tolist(),
                  "right_deg": None if er is None else er.round(4).tolist(),
                  "tip_z_mm": None if tip is None else tip[0], "tip_mm": None if tip is None else tip[1],
                  "pass": goal_ok(e, er, None if tip is None else tip[1])}
        for s in ("left", "right"):
            g = mean_gap(*effort_series(arm_db, s, w), w)
            row[k][f"{s}_effort_gap_nm"] = None if g is None else g.round(3).tolist()
    topics = {s: f"/policy_control/pd_{s}/diagnostics" for s in ("left", "right")}
    got = RB.read(RB.db_of(bag, "pd"), list(topics.values()), tilt[0], tilt[1])
    diag = {s: [(t, json.loads(m.data)) for t, m in got[tp]] for s, tp in topics.items()}
    row["tau"] = {s: tau_stats(diag[s], tilt, limits[s]) for s in topics}
    row["payload_kg"] = payload_check(row, meta, diag["left"])
    n_on = sum(v["n"] for v in row["tau"].values())
    want_on = meta["tilt_comp"] if "tilt_comp" in meta else meta["condition"] == "B"   # 옛 런(A/B 고정)은 B = 켬
    if want_on != (n_on > 0):
        row["notes"].append(f"조건 {meta['condition']} 인데 tilt_comp 활성 표본 {n_on}")
    return row


METRICS = [  # (표 이름, 꺼내는 함수)
    ("pd payload [kg]", lambda r: r.get("payload_kg")),
    ("정지 왼 j1..j7 [°]", lambda r: r.get("pre", {}).get("left_deg")),
    ("정지 오른 j1..j7 [°]", lambda r: r.get("pre", {}).get("right_deg")),
    ("최대 기울기 왼 j1..j7 [°]", lambda r: r.get("max", {}).get("left_deg")),
    ("최대 기울기 오른 j1..j7 [°]", lambda r: r.get("max", {}).get("right_deg")),
    ("정지 왼 |오차| 최대 [°]", lambda r: _amax(r, "pre", "left_deg")),
    ("정지 오른 |오차| 최대 [°]", lambda r: _amax(r, "pre", "right_deg")),
    ("최대 기울기 왼 |오차| 최대 [°]", lambda r: _amax(r, "max", "left_deg")),
    ("최대 기울기 오른 |오차| 최대 [°]", lambda r: _amax(r, "max", "right_deg")),
    ("정지 손 끝 z [mm]", lambda r: r.get("pre", {}).get("tip_z_mm")),
    ("정지 손 끝 |Δ| [mm]", lambda r: r.get("pre", {}).get("tip_mm")),
    ("최대 기울기 손 끝 z [mm]", lambda r: r.get("max", {}).get("tip_z_mm")),
    ("최대 기울기 손 끝 |Δ| [mm]", lambda r: r.get("max", {}).get("tip_mm")),
    ("정지 모델 부족 토크 왼 j1..j7 [N·m]", lambda r: r.get("pre", {}).get("left_effort_gap_nm")),
    ("정지 모델 부족 토크 오른 j1..j7 [N·m]", lambda r: r.get("pre", {}).get("right_effort_gap_nm")),
    ("tilt_comp_tau 왼 |τ| 최대 j1..j7 [N·m]", lambda r: r.get("tau", {}).get("left", {}).get("max_nm")),
    ("tilt_comp_tau 오른 |τ| 최대 j1..j7 [N·m]", lambda r: r.get("tau", {}).get("right", {}).get("max_nm")),
    ("tilt_comp 한계 도달 왼 · 오른", lambda r: None if "tau" not in r else f"{_hits(r['tau']['left'])} · {_hits(r['tau']['right'])}"),
    ("최대 기울기 [°] · 구간 [s]", lambda r: None if "max_tilt_deg" not in r else f"{r['max_tilt_deg']:.1f} · {r['max_window_s']:.1f}"),
    ("통과(정지 · 최대)", lambda r: None if "pre" not in r else f"{_ok(r.get('pre'))} · {_ok(r.get('max'))}"),
]
SCALARS = {"정지 왼 |오차| 최대 [°]", "정지 오른 |오차| 최대 [°]", "최대 기울기 왼 |오차| 최대 [°]", "최대 기울기 오른 |오차| 최대 [°]",
           "정지 손 끝 z [mm]", "정지 손 끝 |Δ| [mm]", "최대 기울기 손 끝 z [mm]", "최대 기울기 손 끝 |Δ| [mm]"}


def _amax(r, k, key):
    v = r.get(k, {}).get(key)
    return None if v is None else float(np.abs(v).max())


def _hits(s):
    """관절별 한계 도달 → 'j4 j5' · 없으면 '없음' · 한계 블록 없거나 활성 표본 0 이면 '-'."""
    h = s.get("limit_hit")
    return "-" if h is None else (" ".join(f"j{i + 1}" for i, x in enumerate(h) if x) or "없음")


def _ok(d):
    return "-" if d is None else ("✓" if d.get("pass") else "✗")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(f"{x:+.2f}" for x in v)
    return f"{v:+.3f}" if isinstance(v, float) else str(v)


def stat(vals) -> dict | None:
    """회차 순 값들 → 평균 · 최소 · 최대 · 개수(None 은 뺀다) + "v" = 회차 순 값 그대로(None 포함, 짝 비교용)."""
    raw = [None if x is None else float(x) for x in vals]
    v = np.array([x for x in raw if x is not None], float)
    return None if not len(v) else {"mean": float(v.mean()), "lo": float(v.min()), "hi": float(v.max()), "n": int(len(v)), "v": raw}


def verdict(base: dict | None, other: dict | None) -> str:
    """짝지은 차이 — 조건마다 k 번째 회차끼리(ABAB 교차 배치의 같은 회) other − base 의 평균 [최소~최대].
    짝이 MIN_PAIRS 보다 적으면 '사이클 부족', 범위가 0 을 넘나들면 '구분 안 됨'. 다중 비교 보정 없음(칸별 참고용)."""
    if base is None or other is None:
        return "-"
    d = np.array([o - b for b, o in zip(base["v"], other["v"]) if b is not None and o is not None])
    if len(d) < MIN_PAIRS:
        return "사이클 부족"
    txt = f"{d.mean():+.2f} [{d.min():+.2f}~{d.max():+.2f}]"
    return f"구분 안 됨 {txt}" if d.min() <= 0 <= d.max() else txt


def conds_of(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["condition"] for r in rows))


def summarize(rows: list[dict]) -> dict:
    """조건별 평균 · 사이클 간 범위 — 스칼라 지표만."""
    out = {}
    for cond in conds_of(rows):
        rs = [r for r in rows if r["condition"] == cond]
        out[cond] = {name: stat(get(r) for r in rs) for name, get in METRICS if name in SCALARS}
        for w, key in (("정지", "goal_pass_pre"), ("최대 기울기", "goal_pass_max")):
            v = [out[cond].get(n) for n in (f"{w} 왼 |오차| 최대 [°]", f"{w} 오른 |오차| 최대 [°]", f"{w} 손 끝 |Δ| [mm]")]
            out[cond][key] = goal_ok(*[None if x is None else x["mean"] for x in v])
    return out


TIP_KEYS = (("tip_mean_mm", "평균"), ("tip_p95_mm", "p95"), ("tip_max_mm", "최대"))
JOINT_KEYS = (("mean_deg", "평균"), ("p95_deg", "p95"), ("max_deg", "최대"))


def summarize_steps(rows: list[dict]) -> dict:
    """{단계: {팔: {정지/이동: {조건: {손끝 지표: stat, 관절 지표: 관절별 사이클 평균, "joint_max_worst": stat}}}}}."""
    out: dict = {}
    order = list(dict.fromkeys(k for r in rows for k in r.get("steps", {})))
    for sid in order:
        for side in ("left", "right"):
            for ph in ("move", "still"):
                cell = {}
                for cond in conds_of(rows):
                    ea = [((r.get("steps", {}).get(sid) or {}).get(side) or {}).get(ph) for r in rows if r["condition"] == cond]
                    es = [e for e in ea if e]          # ea 는 회차 순(None 포함) — 짝 비교가 회차를 맞춘다
                    if not es:
                        continue
                    c = {k: stat((e or {}).get(k) for e in ea) for k, _ in TIP_KEYS}
                    c.update({k: np.mean([e[k] for e in es], axis=0).round(3).tolist() for k, _ in JOINT_KEYS})
                    c["joint_max_worst"] = stat(max(e["max_deg"]) for e in es)
                    cell[cond] = c
                if cell:
                    out.setdefault(sid, {}).setdefault(side, {})[ph] = cell
    return out


def _ms(x: dict | None) -> str:
    return "—" if x is None else f"{x['mean']:.2f} [{x['lo']:.2f}~{x['hi']:.2f}]"


def tip_overall(rows: list[dict]) -> dict:
    """{팔: {정지/이동: {조건: stat}}} — 회차마다 구간 손끝 평균을 다시 평균(구간 가중 없음), 그 회차 값들의 stat."""
    out: dict = {}
    for side in ("left", "right"):
        for ph in ("still", "move"):
            for cond in conds_of(rows):
                per = []
                for r in rows:
                    if r["condition"] != cond:
                        continue
                    v = [x for st in (r.get("steps") or {}).values()
                         for x in [((st or {}).get(side) or {}).get(ph)] if x and x.get("tip_mean_mm") is not None]
                    per.append(float(np.mean([x["tip_mean_mm"] for x in v])) if v else None)
                if (st := stat(per)) is not None:
                    out.setdefault(side, {}).setdefault(ph, {})[cond] = st
    return out


def joint_overall(rows: list[dict], ph: str = "still") -> dict:
    """{팔: {관절 0..6: {조건: {"mean": stat, "p95": stat}}}} — 회차마다 구간(정지) 관절 오차를 구간 평균(가중 없음)한 값의 stat.
    mean = 부호 있는 평균 [°], p95 = |오차| p95 [°]. 양팔 j1..j7 14 행 판정용."""
    out: dict = {}
    for side in ("left", "right"):
        for cond in conds_of(rows):
            per = {k: [] for k in ("mean_deg", "p95_deg")}
            for r in rows:
                if r["condition"] != cond:
                    continue
                v = [x for st in (r.get("steps") or {}).values() for x in [((st or {}).get(side) or {}).get(ph)] if x]
                for k in per:
                    per[k].append(np.mean([x[k] for x in v], axis=0) if v else None)
            for j in range(7):
                cell = {name: stat(None if a is None else a[j] for a in per[k]) for k, name in (("mean_deg", "mean"), ("p95_deg", "p95"))}
                if cell["mean"] is not None:
                    out.setdefault(side, {}).setdefault(j, {})[cond] = cell
    return out


def render_joints(rows: list[dict]) -> list[str]:
    """양팔 j1..j7 정지 오차 표(14 행) — 조건별 평균 · p95 [최소~최대], 첫 조건 대비 짝 차이와 관절별 판정."""
    jo, conds = joint_overall(rows), conds_of(rows)
    if not jo:
        return []
    base = conds[0]
    head = [f"{c} 평균" for c in conds] + [f"{c} p95" for c in conds]
    head += [f"{c}−{base} 평균 차" for c in conds[1:]] + [f"{c}−{base} p95 차" for c in conds[1:]] + ["판정(p95 차)"]
    lines = ["", "## 관절별 정지 오차 [°] — 양팔 j1..j7, 실측 − applied · 회차마다 정지 구간 평균 → 조건별 평균 [최소~최대]",
             f"평균은 부호 있음, p95 는 |오차|. 차이 = 같은 회차 쌍의 차 평균 [최소~최대]. 목표 |오차| ≤ {JOINT_GOAL_DEG}°. "
             "관절별 판정은 14 개를 따로 본 것(다중 비교 보정 없음).", "",
             "| 팔 | 관절 | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    for side in ("left", "right"):
        for j in range(7):
            cell = jo.get(side, {}).get(j, {})
            if not cell:
                continue
            m = lambda c, k: "—" if not (x := (cell.get(c) or {}).get(k)) else f"{x['mean']:+.3f} [{x['lo']:+.3f}~{x['hi']:+.3f}]"  # noqa: E731
            vals = [m(c, "mean") for c in conds] + [m(c, "p95") for c in conds]
            vals += [verdict((cell.get(base) or {}).get("mean"), (cell.get(c) or {}).get("mean")) for c in conds[1:]]
            d95 = [verdict((cell.get(base) or {}).get("p95"), (cell.get(c) or {}).get("p95")) for c in conds[1:]]
            vals += d95 + [" · ".join(f"{c} {_joint_call(v)}" for c, v in zip(conds[1:], d95)) or "-"]
            lines.append(f"| {side} | j{j + 1} | " + " | ".join(vals) + " |")
    return lines


def _joint_call(v: str) -> str:
    """p95 짝 차이 문구 → 판정 한 단어(음수 = 오차가 줄었다)."""
    if v in ("-", "사이클 부족") or v.startswith("구분 안 됨"):
        return v if v != "-" else "—"
    return "줄어듦" if v.startswith("-") else "늘어남"


def render_summary(rows: list[dict]) -> list[str]:
    """맨 위 결론 줄 — 팔별 손끝 정지 · 이동 평균과 첫 조건 대비 짝 차이."""
    tov, conds = tip_overall(rows), conds_of(rows)
    if not tov:
        return []
    base = conds[0]
    lines = ["## 요약 — 손끝 오차 [mm], 회차마다 구간 평균 → 조건별 평균 [최소~최대] · 첫 조건 대비 짝 차이", ""]
    ph_name = {"move": "이동", "still": "정지"}
    for side, phs in tov.items():
        for ph, cell in phs.items():
            parts = [f"{c} {_ms(cell.get(c))}" for c in conds]
            parts += [f"{c}−{base} {verdict(cell.get(base), cell.get(c))}" for c in conds[1:]]
            lines.append(f"- {side} {ph_name[ph]}: " + " · ".join(parts))
    return lines


def render_steps(rows: list[dict], steps: dict) -> list[str]:
    conds = conds_of(rows)
    base, others = conds[0], conds[1:]
    ph_name = {"move": "이동", "still": "정지"}
    head = [f"{c} {n}" for c in conds for _, n in TIP_KEYS] + [f"{c}−{base} {n}" for c in others for _, n in TIP_KEYS[::2]]
    lines = ["", "## 구간별 손끝 오차 [mm] — |FK(실측) − FK(지령)|, palm sensor · 사이클 평균 [최소~최대]",
             "칸별 판정은 참고용(다중 비교 보정 없음), 결론은 맨 위 손끝 정지 · 이동 요약 줄로.",
             f"차이 칸: 첫 조건({base}) 대비 같은 회차 쌍의 차 평균 [최소~최대], 범위가 0 을 넘나들면 '구분 안 됨', "
             f"짝 {MIN_PAIRS} 미만이면 '사이클 부족'. 손끝 표본 {MIN_TIP_N} 개 미만 칸은 '—'.", "",
             "| 구간 | 팔 | 상태 | " + " | ".join(head) + " |", "|---|---|---|" + "---|" * len(head)]
    for sid, sides in steps.items():
        for side, phs in sides.items():
            for ph, cell in phs.items():
                vals = [_ms((cell.get(c) or {}).get(k)) for c in conds for k, _ in TIP_KEYS]
                vals += [verdict((cell.get(base) or {}).get(k), (cell.get(c) or {}).get(k)) for c in others for k, _ in TIP_KEYS[::2]]
                lines.append(f"| {sid} | {side} | {ph_name[ph]} | " + " | ".join(vals) + " |")
    lines += ["", "## 구간별 관절 오차 [°] — 실측 − 지령 · 사이클 평균(평균은 부호, p95 · 최대는 |오차|)", "",
              "| 구간 | 팔 | 상태 | 조건 | 평균 j1..j7 | p95 j1..j7 | 최대 j1..j7 | 최대 최악 관절 [범위] |", "|---|---|---|---|---|---|---|---|"]
    for sid, sides in steps.items():
        for side, phs in sides.items():
            for ph, cell in phs.items():
                for c, v in cell.items():
                    lines.append(f"| {sid} | {side} | {ph_name[ph]} | {c} | " + " | ".join(_fmt(v[k]) for k, _ in JOINT_KEYS)
                                 + f" | {_ms(v['joint_max_worst'])} |")
    rp = [(r, x) for r in rows for x in r.get("replay", [])]
    if rp:
        late = [f"c{r['cycle']:02d} {x['step']} {x['late']}" for r, x in rp if x["late"] > 0]
        lines += ["", "## 재생 발행(replay_to_pd 끝 줄)"] + ([f"★늦은 프레임 있음: {' · '.join(late)}"] if late else []) + [
            "", "| 사이클 | 조건 | 구간 | 발행 | 늦음 | 최대 지연 [ms] | 간격 p95 [ms] |", "|---|---|---|---|---|---|---|"]
        lines += [f"| c{r['cycle']:02d} | {r['condition']} | {x['step']} | {x['n']} | {x['late']}{' ★' if x['late'] else ''} | "
                  f"{x['max_lag_ms']:.1f} | {x['p95_ms']:.1f} |" for r, x in rp]
    return lines


def render(rows: list[dict], summ: dict, run: Path, steps: dict | None = None) -> str:
    conds = conds_of(rows)
    cols = [f"c{r['cycle']:02d} {r['condition']}" for r in rows] + [f"{c} 평균 [범위]" for c in conds] + ["참고 150115"]
    ref = {"정지 왼 j1..j7 [°]": f"j4 {REF_150115['j4_pre_deg']:+.2f}", "최대 기울기 왼 j1..j7 [°]": f"j4 {REF_150115['j4_max_deg']:+.2f}",
           "최대 기울기 손 끝 |Δ| [mm]": f"(셰이커 입구 {REF_150115['inlet_sag_mm']})"}
    ms = lambda c, n: "" if not summ[c].get(n) else f"{summ[c][n]['mean']:+.3f} [{summ[c][n]['lo']:+.3f}~{summ[c][n]['hi']:+.3f}]"  # noqa: E731
    lines = [f"# pd 보상 조건 비교 — {run}", ""]
    lines += [f"- {c}: {_cond_text(next(r for r in rows if r['condition'] == c))}" for c in conds]
    lines += [""] + render_summary(rows) + render_joints(rows)
    lines += ["", f"오차 = 실측 − pd applied(처짐 포함). 목표 양팔 j1..j7 |오차| ≤ {JOINT_GOAL_DEG}° "
              f"(정지 · 최대 기울기 각각) · 손 끝 ≤ {TIP_GOAL_MM:g} mm. 손은 빈손. 조건 칸은 회차별 값의 평균 [최소~최대].", "",
              "## tilt 지표", "", "| 지표 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, get in METRICS:
        vals = [_fmt(get(r)) for r in rows] + [ms(c, name) for c in conds] + [_fmt(ref.get(name))]
        cell = name.replace("|", "\\|")   # 이름 속 |오차| · |Δ| 가 표 칸을 나누지 않게
        lines.append(f"| {cell} | " + " | ".join(vals) + " |")
    ok = lambda c, k: "✓" if summ[c][k] else "✗"  # noqa: E731
    lines += ["", "목표 통과(평균): 정지 " + " · ".join(f"{c} {ok(c, 'goal_pass_pre')}" for c in conds) + " / 최대 기울기 "
              + " · ".join(f"{c} {ok(c, 'goal_pass_max')}" for c in conds),
              "", "참고 150115 n07b 는 왼손에 셰이커 117 g 를 쥔 회차이고 처짐은 셰이커 입구 기준이다 — 이 실험(빈손 · 손 끝)과 하중 · "
                  "지표가 달라 직접 비교하지 않는다."]
    if steps:
        lines += render_steps(rows, steps)
    notes = [f"c{r['cycle']:02d}: {n}" for r in rows for n in r["notes"]]
    if notes:
        lines += ["", "메모: " + " · ".join(notes)]
    return "\n".join(lines) + "\n"


def _cond_text(row: dict) -> str:
    c = row.get("cond")
    if not c:
        return "tilt_comp 켬(tilt 재생 --tilt-comp)" if row["condition"] == "B" else "tilt_comp 끔"
    args = " ".join(c.get("replay_args", []))
    return " · ".join(x for x in (c.get("note"), f"pd {json.dumps(c['pd'], ensure_ascii=False)}" if c.get("pd") else "",
                                  f"재생 {args} @ {','.join(c.get('replay_on', []))}" if args else "") if x) or "바꾼 것 없음"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("run", type=Path)
    a = ap.parse_args(argv)
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(a.run.glob("cycle_*.json"))]
    if not metas:
        raise SystemExit(f"✗ {a.run}: cycle_*.json 없음")
    chk = json.loads((Path(metas[0]["traj"]).parent / "check.json").read_text(encoding="utf-8"))
    R_in_r = np.asarray(chk["T_in_r"], float)[:3, :3]
    from policy_control.tin_live import palm_fk
    fk = {s: palm_fk(s) for s in ("left", "right")}
    rows = [{**cycle_report(m, cycle_limits(m, a.run / "pd_config_left.yaml"), R_in_r, fk), "cond": m.get("cond")} for m in metas]
    summ, steps = summarize(rows), summarize_steps(rows)
    md = render(rows, summ, a.run, steps) + "\n".join(render_pose(rows))
    (a.run / "report.md").write_text(md, encoding="utf-8")
    (a.run / "report.json").write_text(json.dumps({"cycles": rows, "summary": summ, "steps": steps, "tip_summary": tip_overall(rows), "joint_still": joint_overall(rows), "ref_150115": REF_150115,
                                                   "goal": {"joint_deg": JOINT_GOAL_DEG, "tip_mm": TIP_GOAL_MM}},
                                                  ensure_ascii=False, indent=1,
                                                  default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                        encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
