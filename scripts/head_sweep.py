#!/usr/bin/env python3
"""목 스윕 기록 — 팔 · 손은 그대로 두고 목만 보정 각도표를 돌며 bag 에 남긴다(11.10 사용자: C 조사 자동으로, 한 번만).

C = 카메라(외부 파라미터 · 머리 기구학) 오차와 팔 FK(관절 영점 · 손 장착) 오차 분리. 손이 가만히 있는 동안 목만 바꿔 보면
카메라 쪽 오차는 목 자세에 따라 달라지고 FK 쪽 오차는 그대로다(logs/runs/261109/analysis_hand_offset 의 E6 · Kj+KM 모델).

    python head_sweep.py --center 19.56,67.47              # 계획만 출력(목 · 기록 둘 다 안 건드린다)
    python head_sweep.py --execute                         # 지금 목 자세(/head/joint_states)를 중심으로 격자 → 끝나면 그 자세로

★11.10 수정: 예전엔 OpenArm 보정 프로필 각도(tilt −20 ~ −30)를 그대로 보내 RH56F1 머리가 천장을 봤다. RH56F1 드라이버 각도는
camset 이 pan +19.56 · tilt +67.47 이다(11.10 bag 020124_head_sweep 의 /head/joint_states). 이제 시작 순간 실측 자세를 중심으로
작은 격자만 돈다(손이 계속 화면 안에 있게). tilt 상한 82 — 드라이버도 --max-tilt 82 로 떠 있어야 한다(미션 head_sweep 단계가 띄운다).

★`--execute` 는 목을 움직인다. 팔은 명령하지 않는다 — 시작 전에 손을 카메라 앞 자세로 두고 pd 가 붙들고 있어야 한다.
기록은 rosbag 만(CLAUDE.md 11.10 규칙): rh56f1_record.sh 의 arm/hand/pd + camera(색 · 정렬 깊이 · /head/joint_states).
각도마다 목 상태(/head/move_status)도 arm bag 에 들어가 정착 구간을 bag 만으로 찾는다. 끝나면(실패 · 중단 포함) 처음 자세로
되돌리고 기록을 끈다. 종료 코드: 0 전부 정착, 1 일부 각도 실패, 2 시작 못 함.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

MAX_TILT = 82.0   # = make_rh56f1_missions.PLACE_MAX_TILT(놓기 학생 머리와 같은 상한)
RECORD_SH = REPO / "deploy/policy_control/tools/rh56f1_record.sh"
CAMERA_TOPICS = ("/camera/camera/color/image_raw/compressed",
                 "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth",
                 "/camera/camera/color/camera_info", "/camera/camera/aligned_depth_to_color/camera_info",
                 "/head/joint_states")
EXTRA_TOPICS = ("/head/move_status", "/head/joint_states")
BAG_NAME = "head_sweep"
RECORD_WARMUP_S = 3.0   # bag 기록기가 토픽을 찾는 시간(bc07954: DDS 발견 지연)


def record(action: str) -> int:
    """rh56f1_record.sh start|stop — arm/hand/pd bag + camera bag(색 · 깊이 · /head/joint_states)."""
    env = {**os.environ, "CAMERA": " ".join(CAMERA_TOPICS), "CAMERA_HZ": "10",
           "EXTRA": " ".join(EXTRA_TOPICS), "HAND_SIDES": "right"}
    argv = ["bash", str(RECORD_SH), action] + ([BAG_NAME] if action == "start" else [])
    return subprocess.run(argv, env=env).returncode


def current_pose(timeout: float = 5.0) -> tuple[float, float]:
    """지금 목 자세 — /head/joint_states 한 장(deg). 드라이버가 없으면 RuntimeError."""
    import math
    import rclpy
    from sensor_msgs.msg import JointState
    own = not rclpy.ok()
    if own:
        rclpy.init()
    node = rclpy.create_node("head_sweep_pose")
    got: list[JointState] = []
    node.create_subscription(JointState, "/head/joint_states", got.append, 10)
    try:
        deadline = time.monotonic() + timeout
        while not got and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if own:
            rclpy.shutdown()
    if not got:
        raise RuntimeError(f"/head/joint_states 가 {timeout:g} s 동안 없다 — 목 드라이버가 떠 있나")
    pos = dict(zip(got[-1].name, got[-1].position))
    return round(math.degrees(pos["head_j_pan"]), 2), round(math.degrees(pos["head_j_tilt"]), 2)


def plan_angles(center: tuple[float, float], pan_offsets: list[float], tilt_offsets: list[float],
                max_tilt: float) -> list[tuple[float, float]]:
    """기준 자세 둘레 격자 — tilt 줄마다 pan 을 번갈아(뱀 모양) 돈다. 기준 자세 자체를 맨 앞에 둔다(같은 손을 기준에서 한 번 더)."""
    pan0, tilt0 = center
    out = [(pan0, tilt0)]
    for row, dt in enumerate(sorted(tilt_offsets)):
        tilt = tilt0 + dt
        if tilt > max_tilt:
            raise ValueError(f"tilt {tilt:+.1f}° 가 드라이버 상한 {max_tilt:g}° 를 넘는다 — --tilt-offsets 를 줄일 것")
        pans = sorted(pan0 + dp for dp in pan_offsets)
        for pan in (pans if row % 2 == 0 else pans[::-1]):
            if (pan, tilt) != out[0]:
                out.append((pan, tilt))
    return out


def _floats(text: str) -> list[float]:
    return [float(t) for t in text.split(",") if t.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center", default="", help="pan,tilt(deg). 비우면 지금 /head/joint_states")
    ap.add_argument("--pan-offsets", default="-10,0,10", help="중심 대비 pan(deg)")
    ap.add_argument("--tilt-offsets", default="-8,0,8", help="중심 대비 tilt(deg) — + 가 아래(테이블) 쪽")
    ap.add_argument("--max-tilt", type=float, default=MAX_TILT, help="tilt 상한 — 드라이버 --max-tilt 와 같게")
    ap.add_argument("--hold", type=float, default=3.0, help="정착 뒤 머무는 시간 s(10 Hz 카메라 약 30장)")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    try:
        center = tuple(_floats(args.center)) if args.center else current_pose()
        if len(center) != 2:
            raise ValueError("--center 는 pan,tilt 두 값")
        angles = plan_angles(center, _floats(args.pan_offsets), _floats(args.tilt_offsets), args.max_tilt)
    except Exception as exc:  # noqa: BLE001 — 자세를 못 읽음 · 범위 밖
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    back = f"처음 자세 pan {center[0]:+.2f} / tilt {center[1]:+.2f}"
    est = len(angles) * (args.hold + 4.0)
    print(f"목 스윕 {len(angles)}개 각도(중심 pan {center[0]:+.2f} tilt {center[1]:+.2f}) · 각 {args.hold:.1f} s · 약 {est / 60:.1f} 분 · 끝나면 {back}")
    for pan, tilt in angles:
        print(f"  pan {pan:+6.1f}  tilt {tilt:+6.1f}")
    if not args.execute:
        print("--execute 없이 실행함 — 목 · 기록 둘 다 건드리지 않았다")
        return 0

    import head_move as HM
    import head_motion as hm
    if record("start") != 0:
        print("❌ 기록 시작 실패 — 목을 움직이지 않는다", file=sys.stderr)
        return 2
    failed: list[str] = []
    try:
        time.sleep(RECORD_WARMUP_S)
        for i, (pan, tilt) in enumerate(angles, 1):
            code, status = HM.request_move(pan, tilt, max_tilt=args.max_tilt)
            if code == 2:
                raise RuntimeError(f"목 요청 거절 · 드라이버 없음(pan {pan:+.1f} tilt {tilt:+.1f}) — 스윕 중단")
            lines = hm.format_result_lines(status) if status else []
            mark = "✅" if code == 0 else "⚠️"
            print(f"{mark} [{i}/{len(angles)}] pan {pan:+.1f} tilt {tilt:+.1f}  {' · '.join(lines)}", flush=True)
            if code != 0:
                failed.append(f"{pan:+.1f}/{tilt:+.1f}")
            time.sleep(args.hold)
    except KeyboardInterrupt:
        print("중단 — 처음 자세로 되돌리고 기록을 끈다", file=sys.stderr)
        failed.append("interrupted")
    except Exception as exc:  # noqa: BLE001 — 어떤 실패든 목 복귀 · 기록 정지는 한다
        print(f"❌ {exc}", file=sys.stderr)
        failed.append("error")
    finally:
        try:
            code, _ = HM.request_move(*center, max_tilt=args.max_tilt)
            print(("✅" if code == 0 else "❌") + f" 복귀 {back}", flush=True)
            if code != 0:
                failed.append("return")
        finally:
            time.sleep(1.0)
            record("stop")
    if failed:
        print(f"⚠️ 실패 각도: {', '.join(failed)}")
        return 1
    print("✅ 스윕 끝 — bag: logs/runs/<YYMMDD>/bags/<시각>_head_sweep/{arm,hand,pd,camera}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
