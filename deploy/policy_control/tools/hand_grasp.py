#!/usr/bin/env python3
"""경로 B 손 쥐기 — 팔은 지금 자리에 둔 채 한 손을 닿을 때까지 닫고 막힘 래치로 쥔 판정, 3 s 유지 뒤 pd 가 붙든다.

10.17 사용자(docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md 경로 B): 사람이 컵을 손에 대면 닫는다. 규칙은 경로 C 와 같다
(policy_control/hand_grasp.py). 팔 · 손 목표를 /policy_control/joint_target 으로 policy_hz 에 보낸다 — 팔은 pd 의 지금
세트포인트(status arms.<팔>.q_set) 그대로(실측으로 다시 잡으면 처짐만큼 팔이 움직인다), 손은 닫는 목표. 끝나면 그 팔 episode stop
→ pd 가 마지막 팔 · 손 목표를 내부 목표로 붙든다(성공이든 실패든 손을 펴지 않는다). SIGTERM(실행기 정지)도 stop 을 내고 끝난다.

종료 코드: 0 = 쥠 + 유지 · 2 = 못 쥠(not grasped — 들지 않았다, 손은 그 자리) · 3 = pd 이상(HOLD · estop · TRACKING 아님 ·
          세트포인트 없음) · 4 = 측정 없음 · 5 = 정지 신호(SIGTERM) · 1 = 그 밖. ★실기 동작이다 — 사용자 승인 후에만 --execute(에피소드 실행기는 confirm 노드 뒤에 부른다).

    python3 deploy/policy_control/tools/hand_grasp.py --side right \\
        --contract config/hand/rh56f1_right_hand.json --out logs/x --execute
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import rh_aglt as A  # noqa: E402
from policy_control.hand_grasp import GraspCfg, HandGrasp, HandGraspError  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_to_pd import EPISODE_TOPIC_SIDE, PD_STATUS, STATE_TOPIC, STOP_WAIT_S, TOPIC, replay_episode_id  # noqa: E402

HAND_STATE = "/hand_{side}/joint_states"     # rh56f1_state_node — canonical r_hj_* · l_hj_*(robot yaml ee)
MEAS_WAIT_S = 5.0
STALE_S = 0.5
SET_GAP_RAD = 0.10          # pd 세트포인트 ↔ 실측 — 이보다 멀면 무언가 이상하다(처짐은 수 mrad ~ 수십 mrad)


def arm_hold_target(pd: dict, side: str, q_meas: np.ndarray) -> tuple[np.ndarray | None, str]:
    """팔 목표 = pd 지금 세트포인트. 없거나 실측과 SET_GAP_RAD 넘게 다르면 (None, 이유) — 아무것도 보내지 않는다. 순수."""
    q_set = ((pd.get("arms") or {}).get(side) or {}).get("q_set")
    if q_set is None or len(q_set) != len(q_meas) or not np.all(np.isfinite(np.asarray(q_set, float))):
        return None, f"pd_{side} status 에 세트포인트(q_set)가 없다 — pd 를 이 코드로 다시 띄울 것"
    q_set = np.asarray(q_set, float)
    gap = float(np.abs(q_set - q_meas).max())
    if gap > SET_GAP_RAD:
        return None, f"pd 세트포인트 ↔ 실측 {gap:.3f} rad > {SET_GAP_RAD} — 팔이 세트포인트를 못 따르고 있다"
    return q_set, ""


def build(contract: str, cfg_json: str) -> tuple[HandGrasp, A.RaContract, GraspCfg]:
    c = A.load_contract(contract)
    cfg = GraspCfg.from_mapping(json.loads(cfg_json) if cfg_json else {})
    return HandGrasp(c, cfg, c.policy_hz), c, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--contract", required=True, help="손 열림 · 쥠 자세와 막힘 명목값을 줄 rh_aglt 계약(경로 C 접근 정책 계약)")
    ap.add_argument("--cfg", default="", help='닫기 · 판정 값 JSON — {"close_rad_s": 0.6, ...}(policy_control.hand_grasp.GraspCfg)')
    ap.add_argument("--out", type=Path, default=None, help="결과 hand_grasp_<side>.json 을 쓸 폴더")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    try:
        hg, c, cfg = build(a.contract, a.cfg)
    except (HandGraspError, A.RhAgltError, OSError, ValueError) as exc:
        print(f"[hand_grasp] ✗ {exc}")
        return 1
    side = c.side()
    if not side.hand_joints[0].startswith(a.side[0] + "_"):
        print(f"[hand_grasp] ✗ 계약 손 {side.hand_joints[0]} 은 {a.side} 손이 아니다")
        return 1
    print(f"[hand_grasp] {a.side} · {c.policy_hz:g} Hz · {cfg} · 최악 {cfg.worst_s:.1f} s · {'★발행' if a.execute else '무발행'}")
    if not a.execute:
        return 0
    rc, res = run(a.side, hg, c)
    res = {**res, "side": a.side, "rc": rc, "cfg": cfg.__dict__, "contract": a.contract, "at": time.strftime("%F %T")}
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"hand_grasp_{a.side}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[hand_grasp] {'✓' if rc == 0 else '✗'} {res.get('phase')} — {res.get('why')}")
    return rc


def run(side: str, hg: HandGrasp, c: A.RaContract) -> tuple[int, dict]:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    s = c.side()
    arm_names, hand_names = list(s.arm_joints), list(s.hand_joints)
    arm_src = {j: f"openarm_{side}_joint{j.split('_')[-1]}" for j in arm_names}
    rclpy.init()
    node = rclpy.create_node(f"hand_grasp_{side}")
    arm, hand, pd = {}, {}, {}

    def on_arm(m) -> None:
        arm.update({n: (p, time.monotonic()) for n, p in zip(m.name, m.position)})

    def on_hand(m) -> None:
        hand.update({n: (p, time.monotonic()) for n, p in zip(m.name, m.position)})

    def on_pd(m) -> None:
        try:
            pd.update(json.loads(m.data), _t=time.monotonic())
        except ValueError:
            pass
    node.create_subscription(JointState, STATE_TOPIC, on_arm, qos_profile_sensor_data)
    node.create_subscription(JointState, HAND_STATE.format(side=side), on_hand, qos_profile_sensor_data)
    node.create_subscription(String, PD_STATUS.format(side=side), on_pd, 10)
    pub = node.create_publisher(JointState, TOPIC, 10)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ep_pub = node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=side), latched)

    def fresh(store: dict, names: list) -> np.ndarray | None:
        now = time.monotonic()
        if not all(n in store and now - store[n][1] < STALE_S for n in names):
            return None
        return np.array([store[n][0] for n in names], float)

    def pd_fault() -> str | None:
        if not pd or time.monotonic() - float(pd.get("_t", 0)) > 1.0:
            return "pd status 없음"
        if pd.get("estop") or pd.get("phase") != "TRACKING":
            return f"pd {pd.get('phase')} estop={pd.get('estop')} {pd.get('reasons', '')}"
        return None

    stop_sig = {"on": False}

    def on_term(signum, _frame) -> None:          # 실행기 정지(safe_stop SIGTERM) — 루프를 빠져나가 stop 을 내고 끝낸다
        stop_sig["on"] = True
    prev_term = signal.signal(signal.SIGTERM, on_term)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < MEAS_WAIT_S:
            rclpy.spin_once(node, timeout_sec=0.05)
            if fresh(arm, list(arm_src.values())) is not None and fresh(hand, hand_names) is not None and pd:
                break
        q_arm, q_hand = fresh(arm, list(arm_src.values())), fresh(hand, hand_names)
        if q_arm is None or q_hand is None:
            return 4, {"phase": "failed", "why": f"측정 없음 — {STATE_TOPIC} 또는 {HAND_STATE.format(side=side)} 가 안 들어온다"}
        why = pd_fault()
        if why:
            return 3, {"phase": "failed", "why": f"시작 전 {why} — 아무것도 보내지 않았다"}
        q_arm, why = arm_hold_target(pd, side, q_arm)
        if q_arm is None:
            return 3, {"phase": "failed", "why": f"시작 전 {why} — 아무것도 보내지 않았다"}
        hg.start(q_hand)
        ep_id, dt = replay_episode_id(), 1.0 / float(c.policy_hz)

        def send(k: int, qh: np.ndarray) -> None:
            m = JointState()
            m.header.stamp = node.get_clock().now().to_msg()
            m.header.frame_id = f"{ep_id}:{k}"
            m.name = arm_names + hand_names
            m.position = [float(v) for v in np.concatenate([q_arm, qh])]
            m.velocity = [0.0] * len(m.name)
            m.effort = [0.0] * len(m.name)
            pub.publish(m)

        k, rc, trace = 0, 0, []
        t_next = time.monotonic()
        while not hg.finished:
            rclpy.spin_once(node, timeout_sec=0.0)
            if stop_sig["on"]:
                rc, hg.phase, hg.why = 5, "failed", "정지 신호(SIGTERM) — 손은 그 자리"
                break
            meas = fresh(hand, hand_names)
            why = pd_fault()
            if meas is None or why:
                rc = 3 if why else 4
                hg.phase, hg.why = "failed", why or "손 측정이 끊겼다"
                break
            q_cmd = hg.step(meas)
            send(k, q_cmd)
            trace.append([round(time.monotonic() - t0, 3), *np.round(meas, 4).tolist(), *np.round(q_cmd, 4).tolist(),
                          hg.latched_fingers()])
            k += 1
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))
        if rc == 0 and hg.phase == "failed":
            rc = 2
        stop = json.dumps({"episode": ep_id, "event": "stop", "object_anchor": None, "home_q": {},
                           "reasons": [f"hand_grasp {hg.phase}: {hg.why}"], "t_ns": time.time_ns()})
        ep_pub.publish(String(data=stop))
        t1 = time.monotonic()
        while time.monotonic() - t1 < STOP_WAIT_S and pd.get("target") != "internal":
            send(k, hg.q_cmd if hg.q_cmd is not None else q_hand)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)
        return rc, {**hg.as_dict(), "ticks": k, "pd_target": pd.get("target"), "trace_cols": ["t", "meas*6", "cmd*6", "latched"],
                    "trace": trace[-240:]}
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
