#!/usr/bin/env python3
"""RH56F1 손 드라이버(robot_control rh56f1_driver)를 config/rh56f1_ports.yaml 의 그 손 값으로 띄운다.

09.29 사용자: 손마다 개별 포트, USB RS485 / CANFD 를 상황에 따라 바꿔 쓴다 — 미션 명령은 그대로 두고 이 파일만 바꾼다.

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
TRANSPORTS = ("rs485", "canfd")


def argv_for(cfg: dict, side: str) -> list[str]:
    """그 손의 launch 명령. 순수 — 값이 이상하거나 좌우 포트가 같으면 ValueError."""
    hand = cfg.get(side) or {}
    other = cfg.get("left" if side == "right" else "right") or {}
    for key in ("transport", "port", "baud", "hand_id"):
        if key not in hand:
            raise ValueError(f"{side}: '{key}' 가 없다")
    if hand["transport"] not in TRANSPORTS:
        raise ValueError(f"{side}: transport 는 {TRANSPORTS} 중 하나: {hand['transport']!r}")
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
    args = ap.parse_args(argv)
    cmd = argv_for(yaml.safe_load(Path(args.ports).read_text()) or {}, args.side)
    print(" ".join(cmd), flush=True)
    if args.dry:
        return 0
    os.execvp(cmd[0], cmd)
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
