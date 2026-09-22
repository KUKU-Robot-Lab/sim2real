#!/usr/bin/env python3
"""CAN 에서 모터 응답이 실제로 들어오는지 — 수신 패킷 카운터가 늘어나는가. 읽기만 한다(sysfs).

    python3 scripts/setup/check_can_rx.py can0 can1 [--seconds 1.0]

09.22 실기: 모터 전원을 켰다고 여겼는데 can0/can1 모두 RX 0 · ERROR-PASSIVE 였다. 팔 브링업은 "activated" 를 찍고
관절 상태를 전부 0.0 으로 냈고, pd engage · 경로 재생까지 통과한 뒤에야 팔이 안 움직인 것을 알았다.
브링업 직후 이 검사가 실패하면 drivers 단계에서 멈춘다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def rx_packets(iface: str) -> int:
    return int(Path(f"/sys/class/net/{iface}/statistics/rx_packets").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ifaces", nargs="+")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--min-rate", type=float, default=50.0, help="초당 최소 수신 패킷(모터가 응답하면 수백~수천)")
    args = ap.parse_args()
    try:
        before = {i: rx_packets(i) for i in args.ifaces}
    except OSError as exc:
        print(f"✗ CAN 인터페이스를 읽을 수 없다: {exc}", file=sys.stderr)
        return 2
    time.sleep(args.seconds)
    bad = []
    for i in args.ifaces:
        rate = (rx_packets(i) - before[i]) / args.seconds
        ok = rate >= args.min_rate
        print(f"  {'✓' if ok else '✗'} {i}: 수신 {rate:.0f} 패킷/s")
        if not ok:
            bad.append(i)
    if bad:
        print(f"✗ {', '.join(bad)} 에서 모터 응답이 없다 — 모터 전원(비상정지 · 전원 스위치) · CAN 케이블을 확인하고 "
              "CAN 을 down → 설정 → up 으로 다시 올릴 것", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
