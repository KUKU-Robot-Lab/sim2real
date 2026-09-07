#!/usr/bin/env python3
"""기록 궤적을 pd 노드의 입력(/policy_control/joint_target, JointState)으로 재생한다.

정책 없이 pd 경로만 시험하는 shadow 단계. 속도는 **유한차분**으로 실어(q̇*) 실기에서
velocity 인터페이스를 처음 쓰는 지점이다. 시작은 실측에서 첫 프레임까지 0.1 rad/s 램프
(shadow_replay_core.approach_ramp — robotctl 과 같은 값·이유), rate_scale ≤ 1.

    python3 policy_control/tools/replay_to_pd.py --npz tests/fixtures/policy_control/left_v2B25_end.npz --env 0 \
        --joints l_aj_1,...,l_aj_7 --rate-scale 0.25 --execute
    python3 policy_control/tools/replay_to_pd.py --hdf5 tests/fixtures/policy_control/g1_y00.hdf5 --key arm_q_cmd ...
★실기 동작이다 — 사용자 승인 후에만 --execute.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402,F401

from shadow_replay_core import PARK_SPEED_RAD_PER_SEC, approach_ramp  # noqa: E402

TOPIC = "/policy_control/joint_target"
# npz 안 관절목표 열 후보 — 앞에서부터 있는 것을 쓴다
NPZ_JOINT_KEYS = ("fabric_q", "arm_target", "arm_q_cmd")
STATE_TOPIC = "/joint_states"


def load_frames(args) -> tuple[np.ndarray, float]:
    """(N, J) 관절 목표와 기록 주기 dt."""
    if args.npz:
        d = np.load(args.npz)
        # 기록마다 관절목표 열 이름이 다르다: 정책 shadow 는 fabric_q, 자세 이동 궤적
        # (reset_*.npz)은 arm_target. 하나를 박아두면 다른 쪽을 못 읽는다(2026-09-07).
        wanted = getattr(args, "npz_key", None)
        keys = [wanted] if wanted else [k for k in NPZ_JOINT_KEYS if k in d]
        if not keys or keys[0] not in d:
            raise SystemExit(
                f"{args.npz.name}: 관절목표 열을 못 찾았다 (찾은 이름 {wanted or NPZ_JOINT_KEYS}). "
                f"파일에 있는 키: {sorted(d.keys())}")
        q = np.asarray(d[keys[0]])
        if q.ndim == 3:
            q = q[:, args.env, :]
        # 프레임 하나 = 정책 스텝 하나(벽시계 step_dt). fabric 내부 서브스텝 수는 관계없다.
        dt = float(d["meta_step_dt"].item()) if "meta_step_dt" in d else (
            float(d["meta_fabric_dt"].item()) if "meta_fabric_dt" in d else args.dt)
        return q[:, : len(args.joints)], dt
    import h5py

    with h5py.File(args.hdf5, "r") as f:
        q = np.asarray(f[args.key])
        dt = float(f.attrs.get("step_dt", args.dt))
    return q[:, : len(args.joints)], dt


# 되짚기 진입 램프가 메워도 되는 최대 간극 [rad]. 이 램프만이 관절공간 직선이고,
# 되짚기는 "왔던 자리에 서 있다"가 전제라 간극이 크면 전제가 깨진 것이다(2026-09-07 실기).
REVERSE_MAX_RAMP_RAD = 0.35


def build_plan(frames: np.ndarray, start: np.ndarray, pub_dt: float, reverse: bool,
               max_ramp_rad: float = REVERSE_MAX_RAMP_RAD) -> tuple[np.ndarray, np.ndarray, int]:
    """(발행할 관절목표, 속도 전향, 램프 프레임 수).

    ★preset 에서 내려올 때는 goto_home 이 아니라 이 함수의 reverse 를 쓴다. goto_home 은
    계약 홈을 향한 관절공간 직선이라 손이 지나는 곳을 계산하지 않는다 — 2026-09-07 에
    그걸로 테이블을 향해 내려가 사용자가 모터를 강제 종료했다. 올라온 길을 거꾸로 가면
    지나온 공간만 지난다.
    """
    frames = frames[::-1] if reverse else frames
    gap = float(np.abs(np.asarray(start) - frames[0]).max())
    if reverse and gap > max_ramp_rad:
        raise SystemExit(
            f"[replay] --reverse 거부: 실측이 기록 끝에서 {gap:.3f} rad 떨어져 있다 "
            f"(한계 {max_ramp_rad}). 되짚기는 왔던 자리에서만 안전하다 — 진입 램프는 "
            f"관절공간 직선이라 그 간극을 테이블을 모른 채 지난다.")
    ramp = approach_ramp(np.asarray(start, dtype=float), frames[0],
                         speed=PARK_SPEED_RAD_PER_SEC, dt=pub_dt)
    plan = np.vstack([ramp, frames])
    vel = np.vstack([np.zeros((1, plan.shape[1])), np.diff(plan, axis=0) / pub_dt])
    vel[: len(ramp)] = 0.0                                   # 램프 구간은 속도 전향 없음
    return plan, vel, len(ramp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=Path)
    src.add_argument("--hdf5", type=Path)
    ap.add_argument("--key", default="arm_q_cmd", help="hdf5 dataset (기본 arm_q_cmd)")
    ap.add_argument("--npz-key", default=None,
                    help=f"npz 관절목표 열 이름 (기본: {NPZ_JOINT_KEYS} 중 있는 것)")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--joints", required=True, help="canonical 관절 이름 CSV (기록 열 순서)")
    ap.add_argument("--dt", type=float, default=0.02, help="기록 주기 fallback (s)")
    ap.add_argument("--rate-scale", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=0, help="0 = 전부")
    ap.add_argument("--reverse", action="store_true",
                    help="기록을 거꾸로 재생한다 — preset 에서 차렷으로 **내려올 때**. "
                         "goto_home 은 관절공간 직선이라 테이블을 모른다(2026-09-07 실기)")
    ap.add_argument("--max-ramp-rad", type=float, default=REVERSE_MAX_RAMP_RAD,
                    help="되짚기 진입 램프가 메워도 되는 최대 간극 [rad]")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    args.joints = [j.strip() for j in args.joints.split(",")]
    if not 0.0 < args.rate_scale <= 1.0:
        raise SystemExit("--rate-scale 는 (0, 1]")

    frames, dt = load_frames(args)
    if args.frames:
        frames = frames[: args.frames]
    if not np.isfinite(frames).all():
        raise SystemExit("[replay] 기록에 비유한 값 — 보간하지 않고 거부한다")
    pub_dt = dt / args.rate_scale
    print(f"[replay] {frames.shape[0]} 프레임 × {len(args.joints)} 관절 · 기록 dt {dt:.4f} s → "
          f"발행 {pub_dt * 1000:.1f} ms ({args.rate_scale:g}×) · "
          f"{'되짚기(역재생)' if args.reverse else '정방향'} · {'★발행' if args.execute else '무발행'}")
    if not args.execute:
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("replay_to_pd")
    meas: dict = {}
    node.create_subscription(JointState, STATE_TOPIC,
                             lambda m: meas.update(zip(m.name, m.position)), qos_profile_sensor_data)
    pub = node.create_publisher(JointState, TOPIC, 10)
    t0 = time.time()
    while time.time() - t0 < 5.0 and not meas:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not meas:
        raise SystemExit(f"[replay] {STATE_TOPIC} 수신 없음")
    # canonical → source 이름은 pd 가 처리한다; 여기서는 실측을 이름으로 찾는다(l_aj_i ↔ openarm_left_jointi)
    side = "left" if args.joints[0].startswith("l_") else "right"
    start = np.array([meas[f"openarm_{side}_joint{j.split('_')[-1]}"] for j in args.joints])
    plan, vel, n_ramp = build_plan(frames, start, pub_dt, args.reverse, args.max_ramp_rad)
    for k, (q, qd) in enumerate(zip(plan, vel)):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = f"replay:{k}"
        msg.name, msg.position, msg.velocity, msg.effort = list(args.joints), q.tolist(), qd.tolist(), [0.0] * len(q)
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(pub_dt)
    print(f"[replay] 완료 · 램프 {n_ramp} + 기록 {len(frames)} 프레임"
          f"{' (되짚기)' if args.reverse else ''}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
