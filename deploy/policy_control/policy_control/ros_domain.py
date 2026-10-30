"""실기 배포 ROS 2 도메인 — config/ros_domain.env 한 곳에서 읽는다(bash 는 같은 파일을 source 한다).

10.30 사용자: 배포 도메인은 motion_acq(nova2 글러브 126 · macq 콘솔 7)와 분리. 숫자는 그 파일에만 있다.
"""
from __future__ import annotations

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[3] / "config" / "ros_domain.env"
KEY = "S2R_REAL_DOMAIN"
#: ROS 2 도메인 범위. 0 은 실기 기본값이라 launch 들이 거부하므로 1 부터.
DOMAIN_MIN, DOMAIN_MAX = 1, 232


def read_real_domain(path: Path = ENV_FILE) -> int:
    """`KEY=VALUE` 줄만 읽는다(주석 · 빈 줄 무시). 줄이 없거나 1..232 밖이면 ValueError."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or key.strip() != KEY:
            continue
        domain = int(value.strip().strip('"').strip("'"))
        if not DOMAIN_MIN <= domain <= DOMAIN_MAX:
            raise ValueError(f"{path}: {KEY}={domain} 은 {DOMAIN_MIN}..{DOMAIN_MAX} 밖이다")
        return domain
    raise ValueError(f"{path}: {KEY}= 줄이 없다")


REAL_DOMAIN: int = read_real_domain()
REAL_DOMAIN_STR: str = str(REAL_DOMAIN)
