#!/usr/bin/env python3
"""정책 실기 기록 — `ros2 bag record` 두 묶음(policy · sensors)을 띄우고, 정지 신호에 둘을 정상 종료한다. 구독만 한다.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/policy_bag.py \
        --contract deploy/policies/dg5f_m/cup_grasp/left_i01/joint_contract.json \
        --robot deploy/policy_control/config/robots/dg5f_m_left_real.yaml
    → logs/policy_control/real_runs/<시각>__<task>__<side>/{policy/, sensors/, meta.json}

09.28 사용자: "기록 파일들은 모두 bag 파일 아니야? … 따로 따로 관리하면 될 것 같은데?" — Python 기록기 하나가 모든 토픽을
받다 촉각을 버렸다(손가락당 9 Hz). 기록은 rosbag2(C++)가 하고, 정책 · 제어와 손끝 센서를 다른 bag 으로 나눈다.
분석은 tools/bag_to_trace.py(→ trace.npz) · tools/joint_trace_report.py --latest.
콘솔은 SIGTERM 을 보내고 5 s 뒤 SIGKILL 한다 — 받자마자 자식에게 SIGINT 를 넘겨 sqlite3 bag 을 닫게 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402,F401
from policy_control.joint_contract import load_contract  # noqa: E402
from policy_control.sources import load_robot_cfg, select_side  # noqa: E402
from policy_control.trace_acc import BAG_GROUPS, topics  # noqa: E402

SIM2REAL = Path(__file__).resolve().parents[3]
OUT_DIR = SIM2REAL / "logs" / "policy_control" / "real_runs"
STOP_WAIT_S = 4.0            # 콘솔 KILL_GRACE_S(5) 안에 끝낸다
CACHE_BYTES = 1 << 20        # 작게 — 강제 종료돼도 잃는 양이 적게


def run_dir(out_dir: Path, task: str, side: str, now: float | None = None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
    return out_dir / f"{stamp}__{task.replace('/', '_')}__{side}"


def record_argv(out: Path, group: str, tops: list[str]) -> list[str]:
    return ["ros2", "bag", "record", "-o", str(out / group), "--max-cache-size", str(CACHE_BYTES),
            "--include-hidden-topics", *tops]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--robot", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seconds", type=float, default=0.0, help=">0 이면 그 시간 뒤 스스로 끝낸다(시험용)")
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")
    c = load_contract(args.contract)
    cfg = select_side(load_robot_cfg(args.robot), c.side)
    tmap = topics(c, cfg)
    out = run_dir(args.out_dir, c.task, c.side)
    out.mkdir(parents=True, exist_ok=False)
    meta = {"task": c.task, "side": c.side, "contract": str(args.contract.resolve()), "robot": str(args.robot.resolve()),
            "t_start": time.time(), "topics": {t: list(v) for t, v in tmap.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    procs = []
    for g in BAG_GROUPS:
        tops = [t for t, (_, _, grp) in tmap.items() if grp == g]
        procs.append(subprocess.Popen(record_argv(out, g, tops), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                      text=True))
    print(f"[bag] {c.task} · {c.side} · 기록 시작 → {out} ({', '.join(BAG_GROUPS)})", flush=True)

    stop = {"now": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(now=True))
    t0 = time.time()
    while not stop["now"] and (args.seconds <= 0 or time.time() - t0 < args.seconds):
        if any(p.poll() is not None for p in procs):
            dead = [(g, p.returncode, (p.stderr.read() or "")[-300:]) for g, p in zip(BAG_GROUPS, procs) if p.poll() is not None]
            print(f"✗ [bag] 녹화가 먼저 끝났다: {dead}", flush=True)
            break
        time.sleep(0.2)
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    deadline = time.time() + STOP_WAIT_S
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
    meta["t_end"] = time.time()
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    sizes = {g: sum(f.stat().st_size for f in (out / g).glob("*")) if (out / g).is_dir() else 0 for g in BAG_GROUPS}
    print(f"[bag] 저장 {out} · {meta['t_end'] - meta['t_start']:.1f} s · " +
          " · ".join(f"{g} {s / 1e6:.1f} MB" for g, s in sizes.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
