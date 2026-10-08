#!/usr/bin/env python3
"""실기 런 동안 프로세스별 CPU(코어 수)를 1 s 마다 CSV 로 남긴다 — 읽기만 한다(프로세스를 건드리지 않는다).

10.08 사용자: 실기 테스트 전에 미리 세팅. 10.03 밤 측정은 정책 노드가 15 s 만에 멈춰 추론 부하를 못 쟀다 —
미션 aglt · 에피소드 단계가 bag 과 같이 이걸 띄워 정책이 실제로 도는 동안의 부하를 남긴다.

    python3 tools/proc_cpu_record.py --out logs/cpu/<이름>.csv            # SIGINT · SIGTERM 에 요약을 찍고 끝
    python3 tools/proc_cpu_record.py --summary logs/cpu/<이름>.csv        # 기록 요약만(평균 · p95 코어)

대상 = 명령줄에 KEYS 중 하나가 든 프로세스(정책 · pd · 손 · 팔 브링업 · FP++ · 카메라 · 기록기). 이름은 짧은 꼬리표로 묶는다.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

#: (꼬리표, 실행 파일 · 스크립트 이름에 들어 있는 문자열) — 위에서부터 처음 맞는 것. ★10.08 리뷰: 명령줄 전체를 보면 팔 bag 기록기
#: (EXTRA 토픽에 status/rh_aglt_node_right 가 든다)가 정책으로 잡힌다 — 기록기를 먼저 가르고, 나머지는 앞 세 토큰의 이름만 본다.
RECORDER = ("bag", "bag record")
KEYS = (("rh_aglt", "rh_aglt_node"), ("rh_place", "rh_place_node"), ("pour_fj", "pour_fj_node"), ("pd", "pd_node"),
        ("episode_runner", "episode_runner_node"), ("hand_ecat_master", "rh56f1_ecat_master"), ("hand_ecat", "rh56f1_ecat_node"),
        ("hand_state", "rh56f1_state_node"), ("controller_manager", "ros2_control_node"), ("fpp", "foundationpose"),
        ("fpp", "object_pose_node"), ("camera", "realsense"), ("console", "s2r_console"), ("isaac_train", "train.py"))
FIELDS: list[str] = ["t", "label", "pid", "cores", "rss_mb"]


def label_of(cmdline: str) -> str | None:
    tokens = cmdline.split()
    if tokens and Path(tokens[0]).name in ("bash", "sh"):
        return None                                  # 래퍼 셸(EXTRA=… 를 든 것)은 세지 않는다 — 그 자식이 따로 잡힌다
    if RECORDER[1] in cmdline:
        return RECORDER[0]
    head = " ".join(Path(t).name for t in tokens[:3])
    for label, key in KEYS:
        if key in head:
            return label
    return None


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    """꼬리표별 (시각마다 합친 코어의) 평균 · p95 · 최대, 최대 RSS."""
    per_t: dict[str, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    rss: dict[str, float] = defaultdict(float)
    for r in rows:
        per_t[r["label"]][float(r["t"])] += float(r["cores"])
        rss[r["label"]] = max(rss[r["label"]], float(r["rss_mb"]))
    out = {}
    for label, series in per_t.items():
        v = sorted(series.values())
        out[label] = {"mean": sum(v) / len(v), "p95": v[min(len(v) - 1, int(0.95 * len(v)))], "max": v[-1],
                      "n": float(len(v)), "rss_mb": rss[label]}
    return out


def render(summary: dict[str, dict[str, float]]) -> str:
    lines = ["꼬리표               평균   p95   최대 (코어)  RSS MB   표본"]
    for label, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        lines.append(f"{label:18s} {s['mean']:6.2f} {s['p95']:5.2f} {s['max']:5.2f}      {s['rss_mb']:6.0f}  {int(s['n'])}")
    lines.append(f"합계 평균 {sum(s['mean'] for s in summary.values()):.2f} 코어")
    return "\n".join(lines)


def record(out: Path, period: float) -> int:
    import psutil
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = {"now": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(now=True))
    procs: dict[int, tuple[psutil.Process, str]] = {}
    rows: list[dict] = []
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        t0 = time.time()
        while not stop["now"]:
            for p in psutil.process_iter(["pid", "cmdline"]):
                if p.pid in procs:
                    continue
                label = label_of(" ".join(p.info.get("cmdline") or []))
                if label:
                    try:
                        p.cpu_percent(None)             # 첫 호출은 기준점
                        procs[p.pid] = (p, label)
                    except psutil.Error:
                        pass
            time.sleep(period)
            t = round(time.time() - t0, 2)
            for pid, (p, label) in list(procs.items()):
                try:
                    row = {"t": t, "label": label, "pid": pid, "cores": round(p.cpu_percent(None) / 100.0, 3),
                           "rss_mb": round(p.memory_info().rss / 2**20, 1)}
                except psutil.Error:
                    procs.pop(pid)
                    continue
                w.writerow(row)
                rows.append(row)
            f.flush()
    print(f"[cpu] {out} · {len(rows)} 행")
    print(render(summarize(rows)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path, help="기록할 CSV")
    g.add_argument("--summary", type=Path, help="기록된 CSV 요약만")
    ap.add_argument("--period", type=float, default=1.0)
    args = ap.parse_args(argv)
    if args.summary:
        with args.summary.open() as f:
            print(render(summarize(list(csv.DictReader(f)))))
        return 0
    return record(args.out, args.period)


if __name__ == "__main__":
    sys.exit(main())
