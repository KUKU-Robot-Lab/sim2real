#!/usr/bin/env python3
"""실기 팔 bag → 명령 → 움직임 지연 표. 파일만 읽는다.

10.07 Grasping 요청(hdgp rh_aglt 팔 지연 DR 근거)으로 10.03 home_return bag 에서 잰 방법을 도구로 옮겼다(10.08, 실기 전 세팅).
    python3 tools/arm_latency_report.py logs/bags/<런>/arm [--side right] [--json out.json]

필요한 토픽(rh56f1_record.sh EXTRA 가 싣는다): /joint_states · /policy_control/pd_<팔>/applied · /policy_control/joint_target ·
/policy_control/status/pd_<팔>. 정의:
  · 집어감   = joint_target 수신 → pd applied 가 그 값을 낸 시각(pd 틱)
  · 로봇 응답 = applied(ZOH) → 실측 q, 2 s 조각마다 RMS 최소 시프트(조각 평균 오차 = 처짐 제거) — 관절별 p10/50/90
  · 정책 틱 = joint_target 간격
움직임 0.03 rad 미만 조각 · 개선 30 % 미만 · 상한(400 ms)에 걸린 조각은 뺀다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LAGS = np.arange(0.0, 0.4001, 0.004)
SEG_S, STEP_S, GRID_S = 2.0, 1.0, 0.004


def read_bag(path: Path, side: str) -> dict:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    meas_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    cmd_names = [f"{side[0]}_aj_{i}" for i in range(1, 8)]
    want = {"/joint_states": ("meas", meas_names), f"/policy_control/pd_{side}/applied": ("applied", cmd_names),
            "/policy_control/joint_target": ("target", cmd_names)}
    out: dict = {k: ([], []) for k in ("meas", "applied", "target")}
    status: list = []
    while r.has_next():
        topic, data, t = r.read_next()
        if topic in want:
            key, names = want[topic]
            m = deserialize_message(data, get_message(types[topic]))
            idx = {n: i for i, n in enumerate(m.name)}
            if not all(n in idx for n in names):
                continue
            out[key][0].append(t * 1e-9)
            out[key][1].append([m.position[idx[n]] for n in names])
        elif topic == f"/policy_control/status/pd_{side}":
            m = deserialize_message(data, get_message(types[topic]))
            try:
                d = json.loads(m.data)
            except ValueError:
                continue
            a = d.get("arms", {}).get(side, {})
            status.append((t * 1e-9, a.get("phase"), a.get("target"), d.get("stage_cfg")))
    res: dict = {k: (np.asarray(v[0]), np.asarray(v[1], float).reshape(-1, 7)) for k, v in out.items()}
    res["status"] = status
    return res


def moving_windows(status: list, external_only: bool) -> list[tuple[float, float]]:
    ok = [(t, (ph == "TRACKING" and tg == "external") if external_only else ph in ("RAMPING", "TRACKING"))
          for t, ph, tg, _ in status]
    wins, start = [], None
    for t, o in ok:
        if o and start is None:
            start = t
        elif not o and start is not None:
            wins.append((start, t))
            start = None
    if start is not None and ok:
        wins.append((start, ok[-1][0]))
    return [(a, b) for a, b in wins if b - a > SEG_S + 0.5]


def zoh(t_src: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return q[np.clip(np.searchsorted(t_src, t, side="right") - 1, 0, len(t_src) - 1)]


def segment_lags(applied, meas, wins) -> dict[int, list[float]]:
    (ta, qa), (tm, qm) = applied, meas
    out: dict[int, list[float]] = {j: [] for j in range(7)}
    for a, b in wins:
        for s in np.arange(a + 0.5, b - SEG_S, STEP_S):
            t = np.arange(s, s + SEG_S, GRID_S)
            for j in range(7):
                u = zoh(ta, qa[:, j], t)
                if np.ptp(u) < 0.03:
                    continue
                rms = []
                for lag in LAGS:
                    e = np.interp(t + lag, tm, qm[:, j]) - u
                    rms.append(float(np.sqrt(np.mean((e - e.mean()) ** 2))))
                i = int(np.argmin(rms))
                if i < len(LAGS) - 1 and rms[i] < 0.7 * rms[0]:
                    out[j].append(float(LAGS[i]))
    return out


def pickup_ms(target, applied) -> np.ndarray:
    (tt, qt), (ta, qa) = target, applied
    d = []
    for i in range(1, len(tt)):
        if np.max(np.abs(qt[i] - qt[i - 1])) < 1e-4:
            continue
        k = np.where((ta >= tt[i]) & (ta < tt[i] + 0.1))[0]
        hit = [kk for kk in k if np.max(np.abs(qa[kk] - qt[i])) < 1e-6]
        if hit:
            d.append((ta[hit[0]] - tt[i]) * 1e3)
    return np.asarray(d)


def report(bag: Path, side: str) -> dict:
    d = read_bag(bag, side)
    if len(d["applied"][0]) < 10 or len(d["meas"][0]) < 10:
        raise SystemExit(f"{bag}: {side} applied · /joint_states 가 없다(rh56f1_record.sh EXTRA 에 pd applied 를 넣었는가)")
    stages = sorted({s for *_, s in d["status"] if s})
    rep: dict = {"bag": str(bag), "side": side, "stage_cfg": stages}
    for tag, ext in (("external", True), ("all_moving", False)):
        lags = segment_lags(d["applied"], d["meas"], moving_windows(d["status"], ext))
        rep[tag] = {f"j{j + 1}": ({"n": len(v), "p10": float(np.percentile(v, 10)) * 1e3, "p50": float(np.percentile(v, 50)) * 1e3,
                                   "p90": float(np.percentile(v, 90)) * 1e3} if len(v) >= 3 else {"n": len(v)})
                    for j, v in lags.items()}
    if len(d["target"][0]) > 2:
        pk = pickup_ms(d["target"], d["applied"])
        tick = np.diff(d["target"][0]) * 1e3
        rep["pickup_ms"] = {k: float(np.percentile(pk, q)) for k, q in (("p10", 10), ("p50", 50), ("p90", 90))} if pk.size else {}
        rep["target_period_ms"] = {"p50": float(np.median(tick)), "p99": float(np.percentile(tick, 99))}
    return rep


def render(rep: dict) -> str:
    L = [f"[latency] {rep['bag']} · {rep['side']} · pd stage {rep['stage_cfg']}"]
    if "pickup_ms" in rep:
        p, q = rep["pickup_ms"], rep["target_period_ms"]
        L.append(f"  목표 → applied(pd 틱) p10/50/90 = {p.get('p10', float('nan')):.1f}/{p.get('p50', float('nan')):.1f}/"
                 f"{p.get('p90', float('nan')):.1f} ms · 목표 간격 중앙 {q['p50']:.1f} ms (p99 {q['p99']:.1f})")
    for tag in ("external", "all_moving"):
        L.append(f"  applied → 실측 [{tag}] 관절별 p10/p50/p90 ms (조각 수)")
        for j, v in rep[tag].items():
            L.append(f"    {j}  " + (f"{v['p10']:5.0f} {v['p50']:5.0f} {v['p90']:5.0f}  ({v['n']})" if "p50" in v else f"조각 {v['n']} — 부족"))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("bag", type=Path, help="팔 bag 폴더(…/arm)")
    ap.add_argument("--side", choices=("right", "left"), default="right")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rep = report(args.bag, args.side)
    print(render(rep))
    if args.json:
        args.json.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
