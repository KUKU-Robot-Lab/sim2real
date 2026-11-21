#!/usr/bin/env python3
"""목 이동 요청 클라이언트 — 시리얼 포트를 열지 않고 `head_joint_publisher`(단독 소유자)에게 이동을 요청한다.

    python head_move.py                          # 계획만 출력(요청하지 않는다)
    python head_move.py --execute                # 기준 자세로(목이 움직인다)
    python head_move.py --execute --pan 20 --tilt -30
    python head_move.py --execute --extrinsics config/global_camera_extrinsics_arm4090_live.yaml   # 외부 파라미터를 잰 명령 각으로(없으면 기준 자세)

★`--execute` 는 목을 움직인다 — 사용자 허락 뒤에만(허락은 이 명령 한 번에만). 출력은 `head_home.py` 검증 줄과 같은 형식이라
`survey.parse_head_home` 이 그대로 읽는다. 종료 코드: 0 정착·목표 도달, 1 이동은 했지만 실패, 2 요청 거절·드라이버 없음.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import head_motion as hm  # noqa: E402

ACCEPT_TIMEOUT_S = 5.0
DONE_TIMEOUT_S = 40.0
DRIVER_WAIT_S = 5.0


def request_move(pan: float | None, tilt: float | None, *, max_tilt: float = hm.MAX_TILT_DEG) -> tuple[int, dict | None]:
    """요청을 보내고 결과를 기다린다. (종료 코드, 마지막 상태)."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    hm.parse_request(hm.encode_request(hm.MoveRequest("check", pan, tilt)), max_tilt)  # 범위 검사(드라이버도 다시 한다)
    own_init = not rclpy.ok()
    if own_init:
        rclpy.init()
    node = Node(f"head_move_{uuid.uuid4().hex[:6]}")
    seen: list[dict] = []
    request_id = uuid.uuid4().hex[:12]
    node.create_subscription(String, hm.STATUS_TOPIC, lambda m: seen.append(hm.parse_status(m.data)), 10)
    pub = node.create_publisher(String, hm.REQUEST_TOPIC, 10)
    try:
        deadline = time.monotonic() + DRIVER_WAIT_S
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            print("❌ 목 드라이버(head_joint_publisher)가 요청을 듣고 있지 않다 — 먼저 띄울 것", file=sys.stderr)
            return 2, None
        pub.publish(String(data=hm.encode_request(hm.MoveRequest(request_id, pan, tilt))))
        return _await(node, seen, request_id, rclpy)
    finally:
        node.destroy_node()
        if own_init:
            rclpy.shutdown()


def _await(node, seen: list[dict], request_id: str, rclpy) -> tuple[int, dict | None]:
    accepted = False
    t0 = time.monotonic()
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        while seen:
            status = seen.pop(0)
            if status["id"] != request_id:
                continue
            if status["state"] == hm.STATE_REJECTED:
                print(f"❌ 요청 거절: {status.get('reason', '')}", file=sys.stderr)
                return 2, status
            if status["state"] == hm.STATE_ACCEPTED:
                accepted = True
            if status["state"] in (hm.STATE_DONE, hm.STATE_FAILED):
                return (0 if status.get("ok") else 1), status
        waited = time.monotonic() - t0
        if not accepted and waited > ACCEPT_TIMEOUT_S:
            print("❌ 드라이버가 요청을 받았다고 답하지 않았다", file=sys.stderr)
            return 2, None
        if waited > DONE_TIMEOUT_S:
            print("❌ 이동 완료 응답이 오지 않았다", file=sys.stderr)
            return 1, None


def extrinsics_cmd_deg(path: Path) -> tuple[float, float] | None:
    """외부 파라미터 파일이 잰 자세의 명령 각(인코더 deg, pan · tilt). 파일이 없으면 None. 순수.

    ★11.18 head_view_check 가 마커를 화면 안으로 들이려고 목을 camset 에서 조금 돌려 잴 수 있다(cd5aaf1) — 그 뒤 기준
    자세(head_home.yaml)로 가면 object_pose_node 가 head_pose ±tol_tick 밖이라 발행을 막는다(11.18 FP++ 재추출 rc=1).
    그래서 잰 때와 같은 명령(provenance.cmd_deg, 없으면 head_pose 틱)으로 간다."""
    import yaml
    if not path.is_file():
        return None
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cmd = (d.get("provenance") or {}).get("cmd_deg")
    if cmd and len(cmd) == 2:
        return float(cmd[0]), float(cmd[1])
    hp = d.get("head_pose") or {}
    if "pan_tick" in hp and "tilt_tick" in hp:
        return tuple((int(hp[k]) - 2048) * 360 / 4096 for k in ("pan_tick", "tilt_tick"))
    raise ValueError(f"{path} 에 provenance.cmd_deg 도 head_pose 틱도 없다")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pan", type=float, default=None)
    parser.add_argument("--tilt", type=float, default=None)
    parser.add_argument("--max-tilt", type=float, default=None,
                        help="tilt 상한. 없으면 --config 의 max_tilt_deg, 그것도 없으면 head_motion.MAX_TILT_DEG")
    parser.add_argument("--config", type=Path, default=None,
                        help="드라이버와 같은 머리 설정(★11.18 드라이버는 설정 max_tilt_deg 82 를 쓰는데 여기서 -10 으로 미리 거절했다)")
    parser.add_argument("--extrinsics", type=Path, default=None,
                        help="--pan/--tilt 가 없을 때 이 외부 파라미터 파일을 잰 명령 각으로 간다(파일이 없으면 기준 자세)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (args.pan is None) != (args.tilt is None):
        print("❌ --pan 과 --tilt 는 같이 준다(둘 다 없으면 기준 자세)", file=sys.stderr)
        return 2
    if args.max_tilt is None:
        args.max_tilt = hm.MAX_TILT_DEG
        if args.config is not None:
            import head_home
            try:
                args.max_tilt = head_home.max_tilt_for(head_home.load_head_home(args.config), None, hm.MAX_TILT_DEG)
            except (ValueError, OSError) as exc:
                print(f"❌ {args.config}: {exc}", file=sys.stderr)
                return 2
    if args.pan is None and args.extrinsics is not None:
        try:
            cmd = extrinsics_cmd_deg(args.extrinsics)
        except (ValueError, OSError) as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2
        if cmd is None:
            print(f"  {args.extrinsics} 없음 — 기준 자세(head_home.yaml)로")
        else:
            args.pan, args.tilt = cmd
            print(f"  외부 파라미터를 잰 자세로: {args.extrinsics.name}")
    target = "기준 자세(head_home.yaml)" if args.pan is None else f"pan {args.pan:+.2f} / tilt {args.tilt:+.2f}"
    print(f"요청: {target} → {hm.REQUEST_TOPIC}")
    if not args.execute:
        print("--execute 없이 실행함 — 요청하지 않았다")
        return 0
    try:
        code, status = request_move(args.pan, args.tilt, max_tilt=args.max_tilt)
    except hm.BadRequest as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    if status is not None and status.get("state") in (hm.STATE_DONE, hm.STATE_FAILED):
        if lines := hm.format_result_lines(status):
            print("\n".join(lines))
        if status.get("reason"):
            print(f"  사유: {status['reason']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
