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
  손끝 F/T              /dg5f_<side>/fingertip_1~5_broadcaster/wrench (드라이버 fingertip_sensor · ft_broadcaster 켤 때)
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", type=Path, required=True, help="joint_contract.json")
    ap.add_argument("--robot", type=Path, required=True, help="robot yaml (정책 노드와 같은 파일)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seconds", type=float, default=0.0, help=">0 이면 그 시간 뒤 저장하고 끝난다")
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")
    c = load_contract(args.contract)
    cfg = select_side(load_robot_cfg(args.robot), c.side)
    src = SourceSet(cfg)
    arm_s, ee_s, obj_s = cfg.sources["arm"], cfg.sources["ee"], cfg.sources.get("object")

    import rclpy
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String

    stop = {"now": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(now=True))
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(f"joint_recorder_{c.side}")
    R: dict[str, list] = {k: [] for k in ("obs", "act", "tgt", "app", "arm", "hand", "cur", "obj", "jn", "pd", "tip")}
    names: dict[str, tuple | None] = {"tgt": None, "app": None, "cur": None}
    errors: dict[str, int] = {}

    def guard(key, fn):
        def cb(m):
            try:
                fn(time.time(), m)
            except (codec.CodecError, ValueError, KeyError) as exc:
                errors[f"{key}: {type(exc).__name__}"] = errors.get(f"{key}: {type(exc).__name__}", 0) + 1
        return cb

    def on_float(key):
        def fn(t, m):
            s = codec.decode_float_array(m)
            R[key].append((t, s.seq, s.data.copy()))
        return fn

    def on_names(key):
        def fn(t, m):
            n = tuple(m.name)
            if names[key] is None:
                names[key] = n
            if n != names[key]:
                return
            q = np.asarray(m.position, float)
            qd = np.asarray(m.velocity, float) if len(m.velocity) == len(n) else np.full(len(n), np.nan)
            tau = np.asarray(m.effort, float) if len(m.effort) == len(n) else np.full(len(n), np.nan)
            seq = -1
            if key == "tgt":
                _, _, s = str(m.header.frame_id).rpartition(":")
                seq = int(s) if s.isdigit() else -1
            R[key].append((t, seq, q, qd, tau))
        return fn

    def on_joint(key, source):
        def fn(t, m):
            src.update_from_joint_state(source.name, codec.decode_joint_state(m), time.monotonic())
            st = src.snapshot(time.monotonic())
            if key == "arm" and st.arm_q is not None:
                R["arm"].append((t, np.asarray(st.arm_q, float), np.asarray(st.arm_qd, float)))
            if key == "hand" and st.ee_q is not None:
                R["hand"].append((t, np.asarray(st.ee_q, float)))
                on_names("cur")(t, m)                            # 드라이버 원래 이름으로 전류(effort)
        return fn

    def on_obj(t, m):
        p = codec.decode_pose(m)
        R["obj"].append((t, np.asarray(p.pos, float), np.asarray(p.quat, float)))

    def on_tip(i):
        def fn(t, m):
            f, w = m.wrench.force, m.wrench.torque
            R["tip"].append((t, i, np.array([f.x, f.y, f.z, w.x, w.y, w.z], float)))
        return fn

    def on_status(key):
        return lambda t, m: R[key].append((t, str(m.data)))

    node.create_subscription(Float64MultiArray, f"{NS}/obs", guard("obs", on_float("obs")), 100)
    node.create_subscription(Float64MultiArray, f"{NS}/action", guard("act", on_float("act")), 100)
    node.create_subscription(JointState, f"{NS}/joint_target", guard("tgt", on_names("tgt")), 100)
    node.create_subscription(JointState, f"{NS}/pd_{c.side}/applied", guard("app", on_names("app")), 100)
    node.create_subscription(JointState, arm_s.topic, guard("arm", on_joint("arm", arm_s)), qos_profile_sensor_data)
    node.create_subscription(JointState, ee_s.topic, guard("hand", on_joint("hand", ee_s)), qos_profile_sensor_data)
    if obj_s is not None:
        node.create_subscription(PoseStamped, obj_s.topic, guard("obj", on_obj), qos_profile_sensor_data)
    for i in range(1, 6):
        node.create_subscription(WrenchStamped, f"/dg5f_{c.side}/fingertip_{i}_broadcaster/wrench",
                                 guard("tip", on_tip(i)), qos_profile_sensor_data)
    node.create_subscription(String, f"{NS}/status/joint_node", guard("jn", on_status("jn")), 50)
    node.create_subscription(String, f"{NS}/status/pd_{c.side}", guard("pd", on_status("pd")), 50)
    out = out_path(args.out_dir, c.task.replace("/", "_"), c.side)
    print(f"[recorder] {c.task} · {c.side} · 기록 시작 → {out.name} (정지 신호에 저장)", flush=True)
    t_start = time.time()
    while not stop["now"] and (args.seconds <= 0 or time.time() - t_start < args.seconds):
        rclpy.spin_once(node, timeout_sec=0.05)

    ee_names = list(ee_s.joints) + list(ee_s.mirror)
    tn, an, cn = names["tgt"] or (), names["app"] or (), names["cur"] or ()
    data = {
        "meta_task": np.array(c.task), "meta_side": np.array(c.side), "meta_contract": np.array(str(args.contract)),
        "meta_policy_hz": np.float64(c.policy_hz), "meta_t_start": np.float64(t_start),
        "meta_errors": np.array([f"{k} ×{v}" for k, v in errors.items()]),
        "obs_t": np.array([r[0] for r in R["obs"]]), "obs_seq": np.array([r[1] for r in R["obs"]], int),
        "obs": _stack([r[2] for r in R["obs"]], c.obs_dim),
        "act_t": np.array([r[0] for r in R["act"]]), "act_seq": np.array([r[1] for r in R["act"]], int),
        "act": _stack([r[2] for r in R["act"]], c.action_dim),
        "tgt_t": np.array([r[0] for r in R["tgt"]]), "tgt_seq": np.array([r[1] for r in R["tgt"]], int),
        "tgt_names": np.array(tn), "tgt_q": _stack([r[2] for r in R["tgt"]], len(tn)),
        "tgt_qd": _stack([r[3] for r in R["tgt"]], len(tn)),
        "app_t": np.array([r[0] for r in R["app"]]), "app_names": np.array(an),
        "app_q": _stack([r[2] for r in R["app"]], len(an)), "app_qd": _stack([r[3] for r in R["app"]], len(an)),
        "app_tau": _stack([r[4] for r in R["app"]], len(an)),
        "arm_t": np.array([r[0] for r in R["arm"]]), "arm_names": np.array(c.arm_joints),
        "arm_q": _stack([r[1] for r in R["arm"]], 7), "arm_qd": _stack([r[2] for r in R["arm"]], 7),
        "hand_t": np.array([r[0] for r in R["hand"]]), "hand_names": np.array(ee_names),
        "hand_q": _stack([r[1] for r in R["hand"]], len(ee_names)),
        "cur_t": np.array([r[0] for r in R["cur"]]), "cur_names": np.array(cn),
        "cur_mA": _stack([r[4] for r in R["cur"]], len(cn)),
        "obj_t": np.array([r[0] for r in R["obj"]]), "obj_pos": _stack([r[1] for r in R["obj"]], 3),
        "obj_quat": _stack([r[2] for r in R["obj"]], 4),
        "jn_t": np.array([r[0] for r in R["jn"]]), "jn_json": np.array([r[1] for r in R["jn"]]),
        "pd_t": np.array([r[0] for r in R["pd"]]), "pd_json": np.array([r[1] for r in R["pd"]]),
        "tip_t": np.array([r[0] for r in R["tip"]]), "tip_idx": np.array([r[1] for r in R["tip"]], int),
        "tip_wrench": _stack([r[2] for r in R["tip"]], 6),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)                      # 압축하지 않는다 — 콘솔은 SIGTERM 뒤 5 s 에 SIGKILL 한다
    node.destroy_node()
    rclpy.shutdown()
    print(f"[recorder] 저장 {out} · {time.time() - t_start:.1f} s · 목표 {len(R['tgt'])} · obs {len(R['obs'])} · "
          f"팔 {len(R['arm'])} · 손 {len(R['hand'])} · 물체 {len(R['obj'])} · 손끝 {len(R['tip'])}"
          + (f" · 디코드 실패 {errors}" if errors else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
