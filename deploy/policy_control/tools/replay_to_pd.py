#!/usr/bin/env python3
"""기록 궤적을 pd 노드의 입력(/policy_control/joint_target, JointState)으로 재생한다.

정책 없이 pd 경로만 시험하는 shadow 단계. 속도는 **유한차분**으로 실어(q̇*) 실기에서
velocity 인터페이스를 처음 쓰는 지점이다. 시작은 실측에서 첫 프레임까지 0.1 rad/s 램프
(shadow_replay_core.approach_ramp — robotctl 과 같은 값·이유), rate_scale ≤ 1.

    python3 deploy/policy_control/tools/replay_to_pd.py --npz tests/fixtures/policy_control/left_v2B25_end.npz --env 0 \
        --joints l_aj_1,...,l_aj_7 --rate-scale 0.25 --execute
    python3 deploy/policy_control/tools/replay_to_pd.py --hdf5 tests/fixtures/policy_control/g1_y00.hdf5 --key arm_q_cmd ...
★실기 동작이다 — 사용자 승인 후에만 --execute.

10.17 두 팔 동시 재생(tilt · 양팔 이송): --joints 에 r_aj_* 와 l_aj_* 를 같이 주면 한 JointState 에 14 관절을 실어
한 시각표로 보낸다(pd 는 이름으로 팔별 분배). pd 상태 구독 · 끝의 episode stop 은 팔마다 그 팔 토픽으로 낸다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402,F401

from shadow_replay_core import PARK_SPEED_RAD_PER_SEC, approach_ramp  # noqa: E402
from policy_control.rh_replay import resample  # noqa: E402

TOPIC = "/policy_control/joint_target"
EPISODE_TOPIC = "/policy_control/episode"            # 공용(양팔) — 재생은 쓰지 않는다
EPISODE_TOPIC_SIDE = "/policy_control/{side}/episode"   # 팔별 — pd_node 가 그 팔에만 적용
#: pd status 는 팔마다 따로다(09.23) — `--joints` 접두어가 쪽을 알려 준다.
PD_STATUS = "/policy_control/status/pd_{side}"
PD_NODE = "pd_node"        # pd_node.node_name: 쪽 하나면 pd_node_<side>, 여러 쪽이면 pd_node
SUB_STABLE_N, SUB_STABLE_DT = 5, 0.05   # 구독 조건이 연속 5회 × 0.05 s 유지돼야 첫 프레임을 낸다
MEAS_FRESH_S = 0.5         # 대기 통과 뒤 실측을 새로 받을 시간
STOP_WAIT_S = 1.0          # episode stop 이 pd 에 닿을 때까지(pd_selftest 와 같은 값)
# npz 안 관절목표 열 후보 — 앞에서부터 있는 것을 쓴다
NPZ_JOINT_KEYS = ("fabric_q", "arm_target", "arm_q_cmd")
STATE_TOPIC = "/joint_states"
#: 손 실측은 /joint_states 에 없다(11.18 bag 004902: 16 이름 = 팔 14 + *_finger_joint1 2). rh56f1_state_node 가
#  팔과 따로 canonical 이름(r_hj_thumb_1 …)으로 낸다 — 손 열을 재생할 때만 여기를 같이 구독한다.
HAND_STATE_TOPIC = "/hand_{side}/joint_states"
HAND_TAG = "_hj_"          # canonical 손 관절 표시(팔은 _aj_)


def arm_sides(joints: list[str]) -> list[str]:
    """관절 이름 접두어(r_ · l_) → 팔 목록(처음 나온 순서). 다른 접두어는 거부한다."""
    sides: list[str] = []
    for j in joints:
        side = {"r_": "right", "l_": "left"}.get(j[:2])
        if side is None:
            raise SystemExit(f"[replay] 관절 이름 {j!r} — r_ · l_ 접두어만(canonical)")
        if side not in sides:
            sides.append(side)
    return sides


def measured_name(joint: str) -> str:
    """canonical → 실측 이름. 팔(l_aj_i · r_aj_i) → /joint_states 의 openarm_<side>_joint<i>.

    ★11.18: 손(*_hj_*)은 마지막 토큰이 관절 번호가 아니라 이름이라(thumb_1 · index_1 …) 옛 식에 넣으면
    r_hj_index_1 → openarm_right_joint1 처럼 **팔 관절**이 됐다. 그 값이 진입 램프의 출발점이 되어
    손 목표가 팔 각도(−1.21 rad)에서 출발했다(네 손가락은 하한 0 에 잘려 편 채, 엄지만 28 s 회전).
    손 상태 토픽은 canonical 이름을 그대로 쓴다 — 바꿀 것이 없다.
    """
    if HAND_TAG in joint:
        return joint
    return f"openarm_{arm_sides([joint])[0]}_joint{joint.split('_')[-1]}"


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


def check_start_gap(start: np.ndarray, first: np.ndarray, tol: float | None) -> float:
    """실측 ↔ 첫 프레임 최대 간극 [rad]. tol 을 주면 넘을 때 거부 — 실측에서 계획한 궤적(tilt · 이송)은 계획 뒤
    팔이 움직였으면 진입 램프(관절공간 직선)가 검사하지 않은 길로 간다."""
    gap = float(np.abs(np.asarray(start, dtype=float) - np.asarray(first, dtype=float)).max())
    if tol is not None and gap > tol:
        raise SystemExit(f"[replay] 거부: 실측이 첫 프레임에서 {gap:.3f} rad 떨어져 있다(한계 {tol}) — 궤적을 다시 계획할 것")
    return gap


def stretch(frames: np.ndarray, dt: float, rate_scale: float) -> np.ndarray:
    """경로는 그대로, 시간축만 1/rate_scale 배 — 발행은 기록 dt 격자 그대로(11.17: dt/rate_scale 로 늦추면 30 Hz 계단).
    1.0 은 원본 그대로(보간 부동소수 오차도 넣지 않는다)."""
    return frames if rate_scale == 1.0 else resample(frames, dt, rate_scale, dt)[0]


def schedule_wait(t_start: float, k: int, pub_dt: float, now: float) -> float:
    """k 번째 프레임 발행 시각(t_start + k·pub_dt)까지 남은 잠 [s] — 절대 시각이라 지연이 쌓이지 않는다."""
    return max(0.0, t_start + k * pub_dt - now)


def pd_sides_seen(names: set[str], sides) -> dict:
    """구독자 이름 중 이 재생의 팔을 맡은 pd 노드 이름 {이름: 맡은 팔 집합}.
    pd_node 는 joint_target 을 폴링 노드 `pd_node_poll_<팔>[_<팔>]`(pd_node.py:240)로 구독한다.
    옛 이름 `pd_node_<side>`·양팔 `pd_node` 도 받는다."""
    out = {}
    for n in (x.lstrip("/") for x in names):
        if n == PD_NODE:
            out[n] = set(sides)
        elif n.startswith(f"{PD_NODE}_poll_"):
            out[n] = set(n[len(f"{PD_NODE}_poll_"):].split("_")) & set(sides)
        elif n.startswith(f"{PD_NODE}_") and n[len(PD_NODE) + 1:] in sides:
            out[n] = {n[len(PD_NODE) + 1:]}
    return {n: c for n, c in out.items() if c}


def missing_pd(names: set[str], sides) -> list[str]:
    """재생하는 팔 중 구독자 이름에 그 팔의 pd 가 안 보이는 팔."""
    seen = set().union(*pd_sides_seen(names, sides).values()) if names else set()
    return [s for s in sides if s not in seen]


def need_subs(names: set[str], sides, min_subs: int) -> int:
    """필요한 구독자 수 = max(min_subs, 보이는 이 재생의 pd 노드 수) — 이름만 보이고 아직 매칭 안 된 pd 를 걸러낸다."""
    return max(min_subs, len(pd_sides_seen(names, sides)))


def wait_stable(ok, spin, n: int, timeout: float, clock=time.monotonic) -> bool:
    """spin() 사이로 ok() 가 연속 n 번 참이면 True, timeout 안에 못 채우면 False."""
    t0, streak = clock(), 0
    while streak < n:
        if clock() - t0 >= timeout:
            return False
        spin()
        streak = streak + 1 if ok() else 0
    return True


def late_state(lag: float, pub_dt: float, max_late: float) -> str:
    """예정 시각 대비 지연 판정: 'resync'(max_late 넘음 → 일정 재기준) · 'late'(2·pub_dt 넘음, 센다) · 'ok'."""
    if lag > max_late:
        return "resync"
    return "late" if lag > 2 * pub_dt else "ok"


def run_schedule(n: int, pub_dt: float, max_late: float, abort_late: float, max_resync: int, send,
                 clock=time.monotonic, sleep=time.sleep) -> dict:
    """프레임 0..n-1 을 절대 시각표로 send(k). max_late 를 넘게 늦으면 그 프레임부터 일정을 지금으로 민다(재기준) —
    밀린 프레임을 몰아 보내지 않고 같은 목표에서 잠깐 멈췄다 이어 가는 꼴. 한 번 늦음 > abort_late 이거나
    재기준이 max_resync 를 넘으면 aborted 에 사유를 적고 멈춘다(11.18 tilt_comp_ab 020453: 프레임 19 한 번 314 ms 로 abort)."""
    t_start = clock()
    st = {"t_pub": [], "n_late": 0, "worst": 0.0, "n_resync": 0, "shift": 0.0, "aborted": None}
    for k in range(n):
        now = clock()
        lag = now - (t_start + k * pub_dt)
        st["worst"] = max(st["worst"], lag)
        state = late_state(lag, pub_dt, max_late)
        if state == "resync":
            why = (f"--abort-late {abort_late * 1000:.0f} ms" if lag > abort_late else
                   f"재기준 {max_resync} 회 넘음(--max-resync)" if st["n_resync"] >= max_resync else None)
            if why:
                st["aborted"] = f"프레임 {k}/{n} 이 {lag * 1000:.0f} ms 늦음({why})"
                break
            t_start += lag
            st["n_resync"] += 1
            st["shift"] += lag
        st["n_late"] += state == "late"
        send(k)
        st["t_pub"].append(now)
        sleep(schedule_wait(t_start, k + 1, pub_dt, clock()))
    return st


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
    ap.add_argument("--max-start-gap", type=float, default=None,
                    help="정방향도 실측 ↔ 첫 프레임 간극이 이보다 크면 거부 [rad](실측에서 계획한 궤적용, 기본 검사 안 함)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--min-subs", type=int, default=1,
                    help=f"첫 프레임 전 {TOPIC} 발행자에 맞물려야 할 구독자 수(pd · bag · 콘솔 모두 센다)")
    ap.add_argument("--sub-timeout", type=float, default=3.0,
                    help="구독자 수 · 팔마다 pd 노드(pd_node_<side> 또는 pd_node)를 기다리는 시간 [s] — 넘으면 아무것도 내지 않고 거부")
    ap.add_argument("--max-late", type=float, default=0.1,
                    help="프레임이 예정 시각보다 이만큼 [s] 넘게 늦으면 그 프레임부터 일정을 지금으로 민다(재기준, 몰아 보내지 않음)")
    ap.add_argument("--abort-late", type=float, default=1.0,
                    help="한 번 늦음이 이만큼 [s] 넘으면 발행을 멈추고 episode abort · 종료 코드 2")
    ap.add_argument("--max-resync", type=int, default=3,
                    help="재기준이 이 횟수를 넘으면 발행을 멈추고 episode abort · 종료 코드 2")
    ap.add_argument("--tilt-comp", action="store_true",
                    help="★11.16 재생 동안 pd tilt_comp(payload ff + 적분)를 켠다 — pd yaml 에 tilt_comp 블록이 있는 팔만, stop 사건에서 꺼진다")
    ap.add_argument("--no-stop-event", action="store_true", default=False,
                    help="끝에 episode stop 을 내지 않는다(그러면 스트림 두절로 pd 가 watchdog HOLD 로 간다)")
    args = ap.parse_args()
    args.joints = [j.strip() for j in args.joints.split(",")]
    if not 0.0 < args.rate_scale <= 1.0:
        raise SystemExit("--rate-scale 는 (0, 1]")
    if args.min_subs < 1:
        raise SystemExit("--min-subs 는 1 이상")

    frames, dt = load_frames(args)
    if args.frames:
        frames = frames[: args.frames]
    if not np.isfinite(frames).all():
        raise SystemExit("[replay] 기록에 비유한 값 — 보간하지 않고 거부한다")
    n_rec = len(frames)
    frames = stretch(frames, dt, args.rate_scale)
    pub_dt = dt
    print(f"[replay] {n_rec} 프레임 × {len(args.joints)} 관절 · 기록 dt {dt:.4f} s → "
          f"발행 {pub_dt * 1000:.1f} ms 고정 · 시간 {1 / args.rate_scale:g}배({len(frames)} 프레임) · "
          f"{'되짚기(역재생)' if args.reverse else '정방향'} · {'★발행' if args.execute else '무발행'}")
    if not args.execute:
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    sides = arm_sides(args.joints)
    rclpy.init()
    node = Node("replay_to_pd")
    meas: dict = {}
    pd = {side: None for side in sides}                      # 팔마다 pd 목표 출처
    node.create_subscription(JointState, STATE_TOPIC,
                             lambda m: meas.update(zip(m.name, m.position)), qos_profile_sensor_data)
    # 손 열을 재생할 때만(팔만 쓰는 재생은 집합이 비어 구독이 늘지 않는다) — 손 실측은 팔 토픽에 없다
    for side in {arm_sides([j])[0] for j in args.joints if HAND_TAG in j}:
        node.create_subscription(JointState, HAND_STATE_TOPIC.format(side=side),
                                 lambda m: meas.update(zip(m.name, m.position)), qos_profile_sensor_data)
    for side in sides:
        node.create_subscription(String, PD_STATUS.format(side=side),
                                 lambda m, side=side: pd.update({side: _target_of(m.data)}), 10)
    pub = node.create_publisher(JointState, TOPIC, 10)
    # episode stop 발행자는 **지금** 만든다 — 끝에 만들면 discovery 전에 끝나 latched 메시지가 사라진다(pd_selftest 와 같은 이유)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    # ★10.01 실기: 공용 /policy_control/episode 로 stop 을 내면 **다른 팔의 pd** 도 받아 홈에서 정착한 hold 를 새 hold(settle False)로
    #   덮었다(좌 재생 뒤 우 hold err 0.037 굳음). pd 는 팔마다 /policy_control/<side>/episode 를 그 팔에만 적용한다 — 이 팔 토픽으로 낸다.
    ep_pubs = [node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=side), latched) for side in sides]
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0 and not meas:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not meas:
        raise SystemExit(f"[replay] {STATE_TOPIC} 수신 없음")
    # ★11.17 실측: 새 발행자는 DDS 매칭 전에 낸 프레임을 잃는다(앞 50~450 ms) → pd 가 0.2 rad 계단을 max_vel 로 메워 팔이 튄다.
    #   맞물릴 때까지 아무것도 내지 않고, 맞물린 뒤 시작 자세를 새로 읽어 검사 · 계획한다.
    #   bag 기록기 · 콘솔도 이 토픽을 구독하니 수만 세면 pd 보다 먼저 통과할 수 있다 → 재생하는 팔의 pd 노드 이름도 본다.
    def _names() -> set:
        return {i.node_name for i in node.get_subscriptions_info_by_topic(TOPIC)}

    def _ready() -> bool:
        names = _names()
        return not missing_pd(names, sides) and pub.get_subscription_count() >= need_subs(names, sides, args.min_subs)

    def _spin_for(sec: float) -> None:          # spin_once 는 콜백 하나에 돌아오니(250 Hz 실측) 시각으로 채운다
        t_end = time.monotonic() + sec
        while (left := t_end - time.monotonic()) > 0:
            rclpy.spin_once(node, timeout_sec=left)

    if not wait_stable(_ready, lambda: _spin_for(SUB_STABLE_DT), SUB_STABLE_N, args.sub_timeout):
        names, n_subs = _names(), pub.get_subscription_count()
        raise SystemExit(f"[replay] 거부: {TOPIC} 구독자 {n_subs}(필요 {need_subs(names, sides, args.min_subs)}, "
                         f"--min-subs {args.min_subs}), 안 보이는 pd {missing_pd(names, sides)} "
                         f"({args.sub_timeout:g} s) — 아무것도 내지 않았다")
    n_subs = pub.get_subscription_count()
    # 대기 동안의 실측은 낡았을 수 있다 — 비우고 재생하는 팔의 실측이 새로 들어오는지 본다
    want_meas = [measured_name(j) for j in args.joints]
    meas.clear()
    t0 = time.monotonic()
    while time.monotonic() - t0 < MEAS_FRESH_S and not all(n in meas for n in want_meas):
        rclpy.spin_once(node, timeout_sec=0.02)
    stale = [n for n in want_meas if n not in meas]
    if stale:
        raise SystemExit(f"[replay] 거부: 구독 대기 뒤 {MEAS_FRESH_S:g} s 동안 {STATE_TOPIC} 실측 갱신 없음 {stale} "
                         f"— 아무것도 내지 않았다")
    print(f"[replay] {TOPIC} 구독자 {n_subs}: "
          f"{sorted(i.node_name for i in node.get_subscriptions_info_by_topic(TOPIC))}")
    # canonical → source 이름은 pd 가 처리한다; 여기서는 실측을 이름으로 찾는다(l_aj_i ↔ openarm_left_jointi)
    start = np.array([meas[measured_name(j)] for j in args.joints])
    check_start_gap(start, frames[::-1][0] if args.reverse else frames[0], args.max_start_gap)
    plan, vel, n_ramp = build_plan(frames, start, pub_dt, args.reverse, args.max_ramp_rad)
    # ★10.04 실기: 목표 프레임 'replay:k' 와 stop 사건(episode 0)의 번호가 달라, 끝에서 pd 가 stop 을 받아 붙든 뒤에도
    #   stop 확인용으로 보내던 마지막 프레임이 붙들기를 지웠다(pd stopped_episode 는 같은 번호만 거른다) → 1 s 뒤 스트림이
    #   끊기자 워치독 HOLD ↔ 해제가 2 s 동안 번갈았다(home · return 끝). 이 재생만의 번호로 프레임과 stop 을 맞춘다.
    ep_id = replay_episode_id()
    def send(k, q, qd):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = f"{ep_id}:{k}"
        msg.name, msg.position, msg.velocity, msg.effort = list(args.joints), q.tolist(), qd.tolist(), [0.0] * len(q)
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)

    if args.tilt_comp:
        on = json.dumps({"episode": ep_id, "event": "tilt_comp_on", "t_ns": time.time_ns()})
        for ep_pub in ep_pubs:
            ep_pub.publish(String(data=on))
    st = run_schedule(len(plan), pub_dt, args.max_late, args.abort_late, args.max_resync,
                      lambda k: send(k, plan[k], vel[k]))
    t_pub, aborted = st["t_pub"], st["aborted"]
    p95 = float(np.percentile(np.diff(t_pub), 95)) * 1000 if len(t_pub) > 1 else float("nan")
    # tilt_comp_ab.REPLAY_LINE 이 앞부분을 읽는다 — 앞 순서를 바꾸지 말고 뒤에만 붙인다
    print(f"[replay] 발행 {len(t_pub)} 프레임 · 늦음(> {2 * pub_dt * 1000:.0f} ms) {st['n_late']} · "
          f"최대 지연 {st['worst'] * 1000:.1f} ms · 간격 p95 {p95:.1f} ms · "
          f"재기준 {st['n_resync']} · 누적 밀림 {st['shift'] * 1000:.0f} ms")
    if aborted:
        # 목표는 더 내지 않고 abort 사건만 낸다 — pd 는 지금 세트포인트를 붙들고(pd_arm.on_episode) tilt_comp 도 끈다.
        # 사건이 안 닿아도 끊긴 스트림은 pd watchdog 이 HOLD 로 받는다.
        ab = json.dumps({"episode": ep_id, "event": "abort", "reasons": [f"replay_to_pd late: {aborted}"],
                         "t_ns": time.time_ns()})
        for ep_pub in ep_pubs:
            ep_pub.publish(String(data=ab))
        rclpy.spin_once(node, timeout_sec=0.1)
        print(f"[replay] 중단: {aborted} — 발행을 멈추고 episode abort(현 세트포인트 유지 · tilt_comp 끔)")
        node.destroy_node()
        rclpy.shutdown()
        return 2
    if not args.no_stop_event:
        # 재생이 끝났다는 뜻으로 episode stop — pd 가 마지막 세트포인트를 내부 목표로 붙든다(스트림 두절 HOLD 가 아니라).
        # 그래야 뒤의 goto_home 정착 · pd/hand_home 이 받아들여진다(09.22 홈 경로 재생).
        stop = json.dumps({"episode": ep_id, "event": "stop", "object_anchor": None, "home_q": {},
                           "reasons": ["replay_to_pd done"], "t_ns": time.time_ns()})
        for ep_pub in ep_pubs:
            ep_pub.publish(String(data=stop))
        t1 = time.monotonic()
        while time.monotonic() - t1 < STOP_WAIT_S and any(v != "internal" for v in pd.values()):
            send(len(plan), plan[-1], np.zeros_like(plan[-1]))
            time.sleep(pub_dt)
        print(f"[replay] episode stop → pd target {pd!r}")
    print(f"[replay] 완료 · 램프 {n_ramp} + 기록 {len(frames)} 프레임(원본 {n_rec})"
          f"{' (되짚기)' if args.reverse else ''}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


def replay_episode_id() -> int:
    """재생마다 다른 에피소드 번호(목표 프레임 '<번호>:<seq>' 와 stop 사건이 같은 번호) — 정책 노드 번호(작은 정수)와 겹치지 않게
    큰 수로. pd 는 stop 받은 번호의 늦은 프레임만 거르고, 다음 재생은 새 번호라 받는다."""
    return 900_000_000 + (time.time_ns() // 1_000_000) % 99_000_000


def _target_of(text: str):
    try:
        return json.loads(text).get("target")
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
