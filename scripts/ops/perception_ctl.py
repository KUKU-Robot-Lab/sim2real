#!/usr/bin/env python3
"""노드 2 의 사용자 면 — 물체를 골라 인지 체인을 켜고 끈다.

  perception_ctl.py start shaker_closed cup_big_s100 [--viewer]
  perception_ctl.py stop [--camera]
  perception_ctl.py viewer on|off
  perception_ctl.py status
  perception_ctl.py list

이름은 레지스트리(config/objects.yaml)로 검증·alias 해석 후 /perception/cmd 에 발행한다.
perception_launcher_node.py 가 떠 있어야 한다. 단 stop 은 런처가 없으면 ssh 로 저 PC 의 내리는 스크립트를 직접 부른다
(09.28 실기: shutdown 이 런처를 먼저 내려 sensors_off 가 "구독자 0" 으로 실패했고 FP++ · 카메라가 켜진 채 남았다).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))


from object_registry import load_registry  # noqa: E402
from perception_launcher_core import shell_argv  # noqa: E402


def build_payload(args, registry) -> dict | None:
    if args.op == "start":
        payload = {"op": "start", "objects": [registry.resolve(n) for n in args.objects]}
        if args.viewer:
            payload["viewer"] = True      # 키를 빼면 런처는 뷰어를 건드리지 않는다(False 는 '내려라')
        return payload
    if args.op == "stop":
        return {"op": "stop", "camera": bool(args.camera)}
    if args.op == "viewer":
        return {"op": "viewer", "on": args.on == "on"}
    return None


REMOTE_SIM2REAL = "rl_ws/sim2real"          # 저 PC 홈 기준(ssh 는 홈에서 시작, PC 마다 사용자 이름이 다르다)


def direct_stop_scripts(camera: bool) -> list[tuple[str, ...]]:
    """런처 없이 내릴 때 저 PC 에서 부를 스크립트(런처의 stop 과 같은 것) — 자세 송신 → FP++ 전부 → (카메라)."""
    out = [("pose_tx_down.sh",), ("fpp_down.sh", "all")]
    if camera:
        out.append(("camera_down.sh",))
    return out


def direct_stop(host: str, camera: bool) -> int:
    import subprocess

    rc = 0
    for script, *argv in direct_stop_scripts(camera):
        cmd = f"bash {REMOTE_SIM2REAL}/scripts/vision/{script} " + " ".join(f"'{a}'" for a in argv)
        proc = subprocess.run(shell_argv(host, cmd), capture_output=True, text=True, timeout=60, cwd=str(Path.home()))
        print(f"[direct] {host}: {script} {' '.join(argv)} → rc={proc.returncode} {(proc.stdout or proc.stderr).strip()[-120:]}")
        rc = rc or proc.returncode
    return rc


def print_status(payload: dict) -> None:
    print(f"camera: {'up' if payload['camera_up'] else 'down'} ({payload['camera_hz']} Hz)"
          f" · viewer: {payload['viewer']} · busy: {payload['busy']}")
    for name, info in payload["objects"].items():
        age = info["pose_age_s"]
        print(f"  {name:16s} container={info['container'] or '-':22s} "
              f"pose={'-' if age is None else f'{age:.2f}s ago'}")
    if payload.get("error"):
        print(f"  ERROR: {payload['error']}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="op", required=True)
    s = sub.add_parser("start")
    s.add_argument("objects", nargs="+")
    s.add_argument("--viewer", action="store_true")
    s.add_argument("--wait", type=float, default=0.0,
                   help="런처가 일을 끝낼 때까지(busy=False) 최대 이 초만큼 기다리고, 오류면 1 로 끝난다(0 = 보내기만)")
    st = sub.add_parser("stop")
    st.add_argument("--camera", action="store_true", help="카메라까지 내린다")
    st.add_argument("--host", default="vision-3090", help="런처가 없을 때 직접 내릴 곳(ssh) · local = 이 PC")
    st.add_argument("--wait", type=float, default=0.0,
                    help="런처가 다 내릴 때까지 최대 이 초만큼 기다린다(FP++ 재등록: stop 이 끝난 뒤 start)")
    v = sub.add_parser("viewer")
    v.add_argument("on", choices=("on", "off"))
    sub.add_parser("status")
    sub.add_parser("list")
    return ap


def main() -> int:
    args = parser().parse_args()
    registry = load_registry()
    if args.op == "list":
        for name in registry.names():
            print(f"{name:16s} {registry.get(name).real}")
        for alias, target in registry.aliases.items():
            print(f"{alias:16s} → {target}")
        return 0
    payload = build_payload(args, registry)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("perception_ctl")
    if payload is not None:
        pub = node.create_publisher(String, "/perception/cmd", 10)
        deadline = time.monotonic() + 3.0
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            if args.op == "stop":
                print(f"perception_launcher_node 가 안 떠 있다 — ssh 로 {args.host} 에서 직접 내린다", file=sys.stderr)
                node.destroy_node()
                rclpy.shutdown()
                return direct_stop(args.host, bool(args.camera))
            print("perception_launcher_node 가 안 떠 있다 (/perception/cmd 구독자 0)", file=sys.stderr)
            return 1
        pub.publish(String(data=json.dumps(payload)))
        print(f"sent {payload}")
    box: list[str] = []
    node.create_subscription(String, "/perception/status", lambda m: box.append(m.data), 10)
    deadline = time.monotonic() + 3.0
    while not box and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    rc = 0
    wait = float(getattr(args, "wait", 0.0) or 0.0)
    if wait > 0 and payload is not None:
        # 09.22 실기: start 는 보내기만 하고 0 으로 끝나 카메라 기동 실패(camera_up.sh rc=1)를 콘솔이 몰랐다.
        rc = _wait_done(node, box, wait)
    elif box:
        print_status(json.loads(box[-1]))
    else:
        print("/perception/status 가 안 온다", file=sys.stderr)
    node.destroy_node()
    rclpy.shutdown()
    return rc


IDLE_AFTER_S = 3.0     # 보낸 뒤 이만큼 지났는데 busy 가 아니면 끝난 것(이미 켜져 있어 "변경 없음" 이면 busy 가 켜지지 않는다)


def wait_verdict(seen_busy: bool, busy: bool, elapsed: float, error: str | None) -> str | None:
    """'ok' · 'fail' · None(계속 기다린다). 09.22 fake: 런처가 '변경 없음' 으로 즉시 끝나 busy 를 한 번도 안 켰는데
    busy→끝 전환만 기다리다 150 s 를 넘겼다."""
    if busy:
        return None
    if seen_busy or elapsed >= IDLE_AFTER_S:
        return "fail" if error else "ok"
    return None


def _wait_done(node, box: list, wait: float) -> int:
    """런처가 일을 끝낼 때까지(또는 wait 초). 끝난 상태에 error 가 있으면 1."""
    import rclpy

    t0, seen_busy = time.monotonic(), False
    while time.monotonic() - t0 < wait:
        rclpy.spin_once(node, timeout_sec=0.2)
        if not box:
            continue
        st = json.loads(box[-1])
        seen_busy |= bool(st.get("busy"))
        v = wait_verdict(seen_busy, bool(st.get("busy")), time.monotonic() - t0, st.get("error"))
        if v is None:
            continue
        print_status(st)
        if v == "fail":
            print(f"✗ 인지 기동 실패: {st['error']}", file=sys.stderr)
            return 1
        return 0
    print(f"✗ {wait:.0f} s 안에 런처가 끝나지 않았다", file=sys.stderr)
    if box:
        print_status(json.loads(box[-1]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
