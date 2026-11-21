"""사이트(설치 장소)별 포트 설정 — config/sites/<이름>.yaml 을 rh56f1_ports.yaml 위에 얹는다.

10.26 사용자: 환경(설치 장소)에 따라 포트가 달라지니 포트를 사이트 이름으로 따로 저장한다.
기존 배선은 kitech, 10.26 새 배선은 arm. 손 프로토콜 값(ethercat 블록 · 보호 · 어드민턴스)은 rh56f1_ports.yaml 그대로 두고,
사이트 파일은 그 장소에서 달라지는 값(손 NIC · CAN 이름 · 목 시리얼)만 갖는다.

사이트 고르기(앞이 이긴다): --site 인자 > 환경변수 S2R_SITE > 이 PC 의 ~/.config/s2r/site(한 줄) > 없음.
없으면 rh56f1_ports.yaml 의 right · left 값을 그대로 쓴다(= kitech 와 같다, tests 가 확인).
사이트 파일에서 손 ifname 이 null 이면 그 손은 아직 연결 전이다 — 그 손을 띄우면 이유와 함께 거부한다.

ROS 없음 · 순수 함수(입력 dict 를 고치지 않는다).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Mapping

import yaml

SITES_DIR = Path(__file__).resolve().parents[1] / "config" / "sites"
ENV_KEY = "S2R_SITE"
HOST_FILE = Path.home() / ".config" / "s2r" / "site"
SIDES = ("right", "left")


class SiteError(ValueError):
    pass


def available() -> list[str]:
    return sorted(p.stem for p in SITES_DIR.glob("*.yaml"))


def resolve_name(cli: str | None = None, env: Mapping[str, str] | None = None,
                 host_file: Path | None = None) -> str | None:
    """고른 사이트 이름(소문자) 또는 None."""
    if cli:
        return cli.strip().lower()
    env = os.environ if env is None else env
    if env.get(ENV_KEY, "").strip():
        return env[ENV_KEY].strip().lower()
    hf = HOST_FILE if host_file is None else host_file
    if hf.exists():
        name = hf.read_text().strip().lower()
        return name or None
    return None


def load_site(name: str) -> dict:
    path = SITES_DIR / f"{name}.yaml"
    if not path.exists():
        raise SiteError(f"사이트 '{name}' 설정이 없다({path}) — 있는 것: {', '.join(available()) or '없음'}")
    site = yaml.safe_load(path.read_text()) or {}
    if site.get("site") != name:
        raise SiteError(f"{path}: site 키({site.get('site')!r})가 파일 이름 '{name}' 과 다르다")
    return site


def apply_site(ports: dict, site: dict | None) -> dict:
    """rh56f1_ports dict 에 사이트의 손 NIC 를 얹은 새 dict. site 가 None 이면 복사본만."""
    out = copy.deepcopy(ports)
    if not site:
        return out
    hands = (site.get("rh56f1") or {})
    for side in SIDES:
        if side not in hands:
            continue
        ifname = (hands[side] or {}).get("ifname")
        hand = dict(out.get(side) or {})
        hand["transport"] = "ethercat"
        hand["ifname"] = ifname or ""
        if not ifname:
            hand["unconnected"] = f"사이트 {site['site']}: {side} 손 NIC 미정(config/sites/{site['site']}.yaml)"
        out[side] = hand
    out["site"] = site["site"]
    return out


def load_ports(ports_path: str | Path, site_name: str | None = None) -> dict:
    """rh56f1_ports.yaml + (고른) 사이트. site_name 은 resolve_name() 결과."""
    ports = yaml.safe_load(Path(ports_path).read_text()) or {}
    return apply_site(ports, load_site(site_name) if site_name else None)


def check_side(ports: dict, side: str) -> None:
    """그 손이 사이트에서 아직 연결 전이면 SiteError."""
    msg = (ports.get(side) or {}).get("unconnected")
    if msg:
        raise SiteError(msg)


#: ★11.18 RealSense 시리얼(rs_cameras)을 읽는 사이트. 손 사이트(resolve_name)를 쓰지 않는다 — 카메라가 붙은 arm4090 의
#   ~/.config/s2r/site 는 motion_acq 손 마스터용 arm4090_macq(git 밖)라 rs_cameras 가 없다. 다른 설치 장소면 S2R_CAMERA_SITE.
CAMERA_SITE_ENV, CAMERA_SITE_DEFAULT = "S2R_CAMERA_SITE", "arm_v2"


def camera_serial(role: str, env: Mapping[str, str] | None = None) -> str:
    """rs_cameras.<role>.serial(rs_launch serial_no 값, 밑줄 없이). 없으면 SiteError — 시리얼 없이 띄우면 두 대 중 아무거나 잡는다."""
    env = os.environ if env is None else env
    name = env.get(CAMERA_SITE_ENV, "").strip().lower() or CAMERA_SITE_DEFAULT
    serial = str(((load_site(name).get("rs_cameras") or {}).get(role) or {}).get("serial") or "").strip()
    if not serial.isdigit():
        raise SiteError(f"사이트 {name}: rs_cameras.{role}.serial 이 없다(config/sites/{name}.yaml)")
    return serial


if __name__ == "__main__":   # scripts/vision/camera_up.sh: python3 site_ports.py camera <head|global>
    import sys
    if sys.argv[1:2] != ["camera"] or len(sys.argv) != 3:
        sys.exit("usage: site_ports.py camera <head|global>")
    try:
        print(camera_serial(sys.argv[2]))
    except SiteError as err:
        sys.exit(f"site_ports: {err}")
