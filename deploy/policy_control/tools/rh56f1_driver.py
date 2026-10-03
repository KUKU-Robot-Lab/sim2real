#!/usr/bin/env python3
"""RH56F1 손 드라이버(robot_control rh56f1_driver)를 config/rh56f1_ports.yaml 의 그 손 값으로 띄운다.

09.29 사용자: 손마다 개별 포트, USB RS485 / CANFD 를 상황에 따라 바꿔 쓴다 — 미션 명령은 그대로 두고 이 파일만 바꾼다.
10.02 사용자: RS485 를 더 쓰지 않고 EtherCAT 으로 — transport: ethercat 이면 policy_control/rh56f1_ecat_node.py 를 띄운다
(같은 토픽 · 메시지라 pd · 정책 · 점검 도구는 그대로). rs485 · canfd 는 벤더 드라이버로 남겨 둔다.

    python3 tools/rh56f1_driver.py --side right            # exec ros2 launch rh56f1_driver rh56f1_right_driver.launch.py …
    python3 tools/rh56f1_driver.py --side right --print    # 명령만 보인다
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[1] / "config" / "rh56f1_ports.yaml"
VENDOR_TRANSPORTS = ("rs485", "canfd")
TRANSPORTS = VENDOR_TRANSPORTS + ("ethercat",)
ECAT_NODE = Path(__file__).resolve().parents[1] / "policy_control" / "rh56f1_ecat_node.py"


def argv_for(cfg: dict, side: str, ports_path: str | Path = DEFAULT, no_op: bool = False,
             transport: str | None = None, extra: list[str] | None = None) -> list[str]:
    """그 손의 실행 명령. 순수 — 값이 이상하거나 좌우 포트(NIC)가 같으면 ValueError.
    transport='rs485' 면 cfg['rs485'] 블록(비상용 옛 배선)을 쓴다. extra 는 EtherCAT 노드에 그대로 붙는다."""
    if transport == "rs485":
        cfg = dict(cfg.get("rs485") or {})
        if not cfg:
            raise ValueError("rh56f1_ports.yaml 에 rs485 블록이 없다")
    elif transport not in (None, "ethercat"):
        raise ValueError(f"--transport 는 ethercat · rs485: {transport!r}")
    hand = cfg.get(side) or {}
    other = cfg.get("left" if side == "right" else "right") or {}
    if hand.get("transport") not in TRANSPORTS:
        raise ValueError(f"{side}: transport 는 {TRANSPORTS} 중 하나: {hand.get('transport')!r}")
    if hand["transport"] == "ethercat":
        if not hand.get("ifname"):
            raise ValueError(f"{side}: 'ifname' 이 없다")
        if other.get("transport") == "ethercat" and other.get("ifname") == hand["ifname"]:
            raise ValueError(f"좌우 손이 같은 NIC {hand['ifname']} — 손 하나에 NIC 하나(일반 스위치는 폭주한다)")
        return ([sys.executable, str(ECAT_NODE), "--side", side, "--ports", str(ports_path)] + (["--no-op"] if no_op else [])
                + list(extra or []))
    for key in ("port", "baud", "hand_id"):
        if key not in hand:
            raise ValueError(f"{side}: '{key}' 가 없다")
    if other.get("port") == hand["port"]:
        raise ValueError(f"좌우 손이 같은 포트 {hand['port']} — 벤더 설정은 포트로 장치를 묶어 한 손이 덮인다")
    if not 1 <= int(hand["hand_id"]) <= 254:
        raise ValueError(f"{side}: hand_id 1..254")
    return ["ros2", "launch", "rh56f1_driver", f"rh56f1_{side}_driver.launch.py",
            f"transport:={hand['transport']}", f"port:={hand['port']}", f"baud:={int(hand['baud'])}",
            f"hand_id:={int(hand['hand_id'])}"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--ports", default=str(DEFAULT))
    ap.add_argument("--print", action="store_true", dest="dry", help="명령만 보인다")
    ap.add_argument("--no-op", action="store_true", help="ethercat: SAFE_OP 에 머문다(상태만 · 손은 명령을 쓰지 않는다)")
    ap.add_argument("--transport", choices=("ethercat", "rs485"), default=None,
                    help="설정 파일 기본(ethercat)을 덮는다 — rs485 는 비상용 옛 배선(USB-RS485 를 다시 꽂아야 한다)")
    ap.add_argument("--op-enable", action="store_true", help="ethercat OP 실험: 명령 전에도 ENABLE_SET 을 켠다")
    ap.add_argument("--sync-type", type=int, default=None, help="ethercat OP 실험: 0x1C32/33:01 값")
    args = ap.parse_args(argv)
    extra = (["--op-enable"] if args.op_enable else []) + (["--sync-type", str(args.sync_type)] if args.sync_type is not None else [])
    if extra and args.transport == "rs485":
        raise SystemExit("--op-enable · --sync-type 는 ethercat 전용")
    cmd = argv_for(yaml.safe_load(Path(args.ports).read_text()) or {}, args.side, args.ports, args.no_op,
                   transport=args.transport, extra=extra)
    print(" ".join(cmd), flush=True)
    if args.dry:
        return 0
    os.execvp(cmd[0], cmd)
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
