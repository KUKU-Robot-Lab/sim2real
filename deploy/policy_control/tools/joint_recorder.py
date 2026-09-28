#!/usr/bin/env python3
"""joint 정책 실기 기록기 — 정책이 도는 동안 입력 · 출력 · 제어기 응답을 npz 로 남긴다. 구독만 한다(발행 없음).

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/joint_recorder.py \
        --contract deploy/policies/left_cp_e4280/joint_contract.json \
        --robot deploy/policy_control/config/robots/dg5f_m_left_real.yaml

09.28 사용자: "이번에 정책 실행시켰을때 로그가 같은게 있나? … 제어기 등이 잘 따라갔는지를 토대로 재학습할수도 있으니까".
그날 policy_left 는 실기에서 돌았지만 obs · 행동 · 목표 · 실측이 어디에도 남지 않았다(콘솔은 status 만 본다).

남기는 것(전부 받은 시각 t 와 함께):
  obs · action          /policy_control/{obs,action}             (seq)
  목표 q* · q̇*          /policy_control/joint_target             (seq = frame_id '<episode>:<seq>')
  pd 가 실제 보낸 값     /policy_control/pd_<side>/applied         (q · q̇ · τ)
  팔 · 손 실측           robot yaml 의 arm · ee 소스 → canonical  (정책 노드와 같은 SourceSet 변환)
  손 전류               /dg5f_<side>/joint_states effort          (드라이버 원래 이름 · mA)
  물체 자세             robot yaml 의 object 소스
  status               joint_node · pd_<side> JSON
  손끝 F/T              /dg5f_<side>/fingertip_1~5_broadcaster/wrench (F/T 센서 손에서, 드라이버 fingertip_sensor · ft_broadcaster)
  손끝 촉각             /dg5f_<side>/tactile/finger_1~5 (촉각 센서 손, 09.28 왼손 = TACTILE_M 3×5 mono8 · TACTILE_S 3×6 mono16)
                        칸 값 그대로(원시 단위, 18 칸으로 채움 — 모자라면 NaN). 영점은 분석이 시작 전 평균을 뺀다
                        tip 순서 = 벤더 finger 1~5(엄지 · 검지 · 중지 · 약지 · 새끼), 팁 로컬 프레임 [N · N·m]

SIGTERM(콘솔의 정지) · SIGINT 을 받으면 그때까지를 저장하고 끝난다. --seconds 를 주면 그 시간 뒤에도 끝난다.
분석: tools/joint_trace_report.py --latest --side <side>
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402,F401
from policy_control import codec  # noqa: E402
from policy_control.joint_contract import load_contract  # noqa: E402
from policy_control.sources import SourceSet, load_robot_cfg, select_side  # noqa: E402

SIM2REAL = Path(__file__).resolve().parents[3]
OUT_DIR = SIM2REAL / "logs" / "policy_control" / "real_runs"
NS = "/policy_control"


def out_path(out_dir: Path, task: str, side: str, now: float | None = None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
    return out_dir / f"{stamp}__{task}__{side}.npz"


def newest(out_dir: Path, side: str) -> Path | None:
    files = sorted(out_dir.glob(f"*__{side}.npz"))
    return files[-1] if files else None


def _stack(rows, width: int) -> np.ndarray:
    return np.stack(rows) if rows else np.zeros((0, width))


def main() -> int:
    """실시간 구독 기록(시험 · 짧은 확인용). 정책 단계는 policy_bag.py(rosbag2)를 쓴다 — 이 노드는 촉각을 버렸다(09.28)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", type=Path, required=True, help="joint_contract.json")
    ap.add_argument("--robot", type=Path, required=True, help="robot yaml (정책 노드와 같은 파일)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seconds", type=float, default=0.0, help=">0 이면 그 시간 뒤 저장하고 끝난다")
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")
    from policy_control.trace_acc import TraceAccumulator, topics
    from rosidl_runtime_py.utilities import get_message

    c = load_contract(args.contract)
    cfg = select_side(load_robot_cfg(args.robot), c.side)
    acc = TraceAccumulator(c, cfg)

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions

    qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    stop = {"now": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(now=True))
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(f"joint_recorder_{c.side}")
    for topic, (kind, typ, _grp) in topics(c, cfg).items():
        node.create_subscription(get_message(typ), topic, (lambda k: lambda m: acc.add(k, time.time(), m))(kind), qos)
    out = out_path(args.out_dir, c.task.replace("/", "_"), c.side)
    print(f"[recorder] {c.task} · {c.side} · 기록 시작 → {out.name} (정지 신호에 저장)", flush=True)
    t_start = time.time()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    while not stop["now"] and (args.seconds <= 0 or time.time() - t_start < args.seconds):
        ex.spin_once(timeout_sec=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **acc.to_arrays({"contract": str(args.contract), "t_start": t_start}))  # 압축 안 함 — 콘솔 KILL 5 s
    node.destroy_node()
    rclpy.shutdown()
    print(f"[recorder] 저장 {out} · {time.time() - t_start:.1f} s · {acc.counts()}"
          + (f" · 디코드 실패 {acc.errors}" if acc.errors else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
