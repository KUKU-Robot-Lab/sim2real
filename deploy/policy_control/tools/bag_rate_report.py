#!/usr/bin/env python3
"""bag(sqlite3) 토픽별 주기 · 간격 · 끊김 보고 — ROS 없이 db3 를 직접 읽는다. 10.03 사용자: 기록 · 병목 확인.

    python3 deploy/policy_control/tools/bag_rate_report.py logs/bags/20261003_150351_home_return_left/arm
    python3 deploy/policy_control/tools/bag_rate_report.py <bag 폴더> --gap-x 3 --active-gap 0.5

★기록 중인 bag 에는 쓰지 않는다 — sqlite 잠금이 기록기를 죽인다(10.03 'database is locked' 로 arm 기록기가 끝났다).
  열기 전에 metadata.yaml(기록기가 정상 종료할 때 쓴다)이 있는지 본다. 없으면 `ros2 bag reindex -s sqlite3 <폴더>` 뒤에.

기록 시각(bag 수신 시각)으로 계산한다. 'active' 열은 토픽이 끊김 없이 나온 구간만의 주기(명령 토픽처럼 동작 중에만 나오는 것용).
"""
from __future__ import annotations

import argparse
import glob
import sqlite3
import sys
from pathlib import Path

import numpy as np


def topic_stats(ts_ns: np.ndarray, gap_x: float, active_gap_s: float) -> dict:
    """수신 시각(ns, 정렬) → 통계. 순수."""
    n = len(ts_ns)
    if n < 2:
        return {"n": n}
    dt = np.diff(ts_ns) / 1e9
    dur = (ts_ns[-1] - ts_ns[0]) / 1e9
    med = float(np.median(dt))
    active = dt[dt < active_gap_s]
    return {"n": n, "dur_s": dur, "hz": (n - 1) / dur if dur > 0 else float("nan"),
            "hz_active": 1.0 / float(np.mean(active)) if len(active) else float("nan"),
            "med_ms": med * 1e3, "p99_ms": float(np.percentile(dt, 99)) * 1e3, "max_ms": float(dt.max()) * 1e3,
            "gaps": int(np.sum(dt > gap_x * med)) if med > 0 else 0,
            "pauses": int(np.sum(dt >= active_gap_s))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag", type=Path)
    ap.add_argument("--gap-x", type=float, default=3.0, help="중앙 간격의 몇 배를 끊김으로 셀지")
    ap.add_argument("--active-gap", type=float, default=0.5, help="이 간격[s] 이상이면 '쉼'(동작 중에만 나오는 토픽)")
    args = ap.parse_args(argv)
    if not (args.bag / "metadata.yaml").is_file():
        print(f"✗ {args.bag}/metadata.yaml 없음 — 기록 중이거나 비정상 종료. 기록이 끝났으면 ros2 bag reindex -s sqlite3 {args.bag}")
        return 1
    dbs = sorted(glob.glob(str(args.bag / "*.db3")))
    rows: dict[str, list[int]] = {}
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        names = dict(con.execute("select id, name from topics"))
        for tid, ts in con.execute("select topic_id, timestamp from messages"):
            rows.setdefault(names[tid], []).append(ts)
        con.close()
    print(f"{args.bag}  ({len(dbs)} db3)")
    print(f"{'topic':44s} {'n':>8s} {'Hz':>8s} {'active':>8s} {'med ms':>7s} {'p99 ms':>7s} {'max ms':>8s} {'끊김':>5s} {'쉼':>4s}")
    for name in sorted(rows):
        s = topic_stats(np.sort(np.asarray(rows[name], dtype=np.int64)), args.gap_x, args.active_gap)
        if s["n"] < 2:
            print(f"{name:44s} {s['n']:8d}")
            continue
        print(f"{name:44s} {s['n']:8d} {s['hz']:8.1f} {s['hz_active']:8.1f} {s['med_ms']:7.2f} {s['p99_ms']:7.2f} "
              f"{s['max_ms']:8.1f} {s['gaps']:5d} {s['pauses']:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
