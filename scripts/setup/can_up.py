#!/usr/bin/env python3
"""팔 CAN(can0 · can1)을 CAN FD 1 M / 5 M 로 올린다 — 이미 그렇게 올라와 있으면 건드리지 않는다.

10.15 사용자: "can ip up 같은 걸 sudo 로 맨날 내가 했는데 자동으로 하게 해야 할 것 같아". 이날 home_right 가
움직이지 않은 run 은 이 수동 단계를 건너뛴 뒤였다.

- 판단은 `ip -details -json link show <if>` 만 읽는다. 바꿀 필요가 없으면 아무 명령도 내지 않는다
  (드라이버가 버스를 쓰는 중에 링크를 내리면 팔이 끊긴다).
- 바꿀 때는 `sudo --non-interactive /usr/sbin/ip link set …` 세 줄(down · type · up)만 쓴다. 이 세 줄은
  scripts/setup/sudoers-s2r-can 이 인자 글자까지 정확히 같을 때만 비밀번호 없이 허용한다. 설치가 안 됐으면
  실패하고 운영자 명령을 알려 준다.
- BUS-OFF 는 restart-ms 0 이라 스스로 돌아오지 않는다 — down · up 으로 다시 올린다.

  python3 scripts/setup/can_up.py can0 can1            # 계획만
  python3 scripts/setup/can_up.py can0 can1 --execute  # 필요하면 올린다
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

IP = "/usr/sbin/ip"
BITRATE = 1_000_000
DBITRATE = 5_000_000
ALLOWED_IFS = ("can0", "can1")      # sudoers 가 허용하는 이름과 같아야 한다


def link_commands(name: str) -> list[list[str]]:
    """sudoers-s2r-can 의 허용 줄과 글자 하나까지 같은 명령."""
    return [[IP, "link", "set", name, "down"],
            [IP, "link", "set", name, "type", "can", "bitrate", str(BITRATE), "dbitrate", str(DBITRATE), "fd", "on"],
            [IP, "link", "set", name, "up"]]


def why_reconfigure(link: dict | None) -> str | None:
    """None = 지금 그대로 쓴다. 문자열 = 다시 올려야 하는 이유. 순수."""
    if link is None:
        return "인터페이스가 없다"
    data = (link.get("linkinfo") or {}).get("info_data") or {}
    if "UP" not in link.get("flags", []):
        return "꺼져 있다(DOWN)"
    if (data.get("bittiming") or {}).get("bitrate") != BITRATE:
        return f"bitrate {(data.get('bittiming') or {}).get('bitrate')} ≠ {BITRATE}"
    if (data.get("data_bittiming") or {}).get("bitrate") != DBITRATE:
        return f"dbitrate {(data.get('data_bittiming') or {}).get('bitrate')} ≠ {DBITRATE}"
    if "FD" not in data.get("ctrlmode", []):
        return "FD 모드가 아니다"
    if data.get("state") == "BUS-OFF":
        return "BUS-OFF — 스스로 돌아오지 않는다(restart-ms 0)"
    return None


def read_link(name: str) -> dict | None:
    out = subprocess.run([IP, "-details", "-json", "link", "show", name], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    rows = json.loads(out.stdout or "[]")
    return rows[0] if rows else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ifaces", nargs="+", choices=ALLOWED_IFS)
    ap.add_argument("--execute", action="store_true", help="필요하면 sudo(비밀번호 없이 허용된 세 줄)로 올린다")
    args = ap.parse_args()
    bad = 0
    for name in args.ifaces:
        link = read_link(name)
        if link is None:
            print(f"✗ {name}: 인터페이스가 없다 — USB-CAN(PCAN-USB Pro FD) 연결 확인")
            bad += 1
            continue
        why = why_reconfigure(link)
        if why is None:
            print(f"✓ {name}: UP · FD 1M/5M — 그대로 둔다")
            continue
        print(f"· {name}: {why} → 다시 올린다")
        if not args.execute:
            for cmd in link_commands(name):
                print("    sudo " + " ".join(cmd))
            continue
        for cmd in link_commands(name):
            r = subprocess.run(["sudo", "--non-interactive", *cmd], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"✗ {name}: {' '.join(cmd)} 실패 — {(r.stderr or r.stdout).strip()}")
                if "password" in (r.stderr or "").lower() or "sudo" in (r.stderr or "").lower():
                    print("  비밀번호 없는 허용이 설치되지 않았다. 운영자 한 번:\n"
                          "    sudo install -m 0440 -o root -g root scripts/setup/sudoers-s2r-can /etc/sudoers.d/s2r-can"
                          " && sudo visudo -c")
                bad += 1
                break
        else:
            after = why_reconfigure(read_link(name))
            if after is None:
                print(f"✓ {name}: 다시 올림 · UP · FD 1M/5M")
            else:
                print(f"✗ {name}: 다시 올렸지만 아직 {after}")
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
