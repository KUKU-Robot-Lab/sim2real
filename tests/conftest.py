"""테스트가 `scripts/` 트리의 모듈을 이름으로 임포트할 수 있게 한다.

★모듈은 역할별 하위 디렉토리(`nodes/`·`ops/`·`calib/`·…)로 나뉘어 있고 라이브러리만
  `scripts/` 최상위에 있다. 예전에는 전부 한 디렉토리에 평평하게 있어 pytest 가
  알아서 경로를 잡아줬는데, 나눈 뒤로는 **이 파일 하나가** 그 역할을 대신한다.
  테스트마다 sys.path 를 손대지 말 것 — 여기 한 곳만 고치면 된다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache"}
for _d in [SCRIPTS, *(d for d in sorted(SCRIPTS.iterdir())
                      if d.is_dir() and d.name not in _SKIP)]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# policy_control 패키지(ament_python)도 이름으로 임포트할 수 있게 한다 — tests/policy_control/ 이 쓴다.
_PKG = Path(__file__).resolve().parents[1] / "deploy" / "policy_control"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


@pytest.fixture(scope="session", autouse=True)
def _dds_stays_on_this_host():
    """시험의 DDS 는 이 PC 안에만 둔다. 가짜 미션 · 시험 도메인(96/97/99)은 다른 호스트의 시험과 같은 번호이고, 10.30 부터 실기 셸은
    ROS_LOCALHOST_ONLY=0 이라 그 값이 남은 채 시험을 돌리면 시험 노드가 모든 NIC 에 광고된다(10.30 5090 실측: 0 인 노드는 랜 · Wi-Fi ·
    tailscale · docker 브리지 전부에 locator 를 낸다). rclpy.init 과 자식 프로세스(ros2 CLI · 가짜 노드)가 os.environ 을 읽으므로
    세션 동안 1 을 박고, 랜 프로파일(config/dds/lan_profile.sh)은 1 과 같이 두면 격리가 깨지니 지운다. env 를 스스로 만들어 넘기는
    시험(test_dds_lan_profile._source)은 영향이 없다."""
    keys = ("ROS_LOCALHOST_ONLY", "FASTRTPS_DEFAULT_PROFILES_FILE")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
