"""벤더 손 통신 계층 스냅샷이 상위의 수정을 담고 있어야 한다 (ROS 없음).

2026-09-07 공식 레포 대조에서 나온 것:

★`delto_tcp_comm` 스냅샷이 2026-03-26 상태라 `8c85809 (2026-04-01) Correct velocity scale
  from 0.1deg/s to rpm and remove sign inversion` 이전이다. 그래서 손 관절 **속도가 60배 작고
  부호가 반대**였다(raw 는 rpm 인데 0.1 deg/s 로 해석). 이 값은 우 g1 정책 관측 155차원 중
  `hand_qd` 20칸으로 들어가므로 sim↔실기 정합이 그만큼 깨진다.

같이 들어오는 것: POSIX 소켓 전환(boost 의존 제거, 공개 API 동일), 소켓 강화(poll 500 ms 수신
타임아웃·TCP keepalive), 펌웨어 3.0+ 오류코드 보고. 오늘 손 접속이 **오류 없이 무한 대기**한
증상이 그 강화가 겨냥한 것이다.

⚠**이 테스트는 "벤더와 일치"를 잠글 뿐 "물리적으로 옳다"를 잠그지 않는다.** 교체 후 실측
(∫v dt vs Δ위치, index/middle/ring · 양방향): 보고 속도가 실제 관절 속도보다 **4.4~4.9배 크고**
그 비율이 지령 속도에 따라 변한다(1 s 4.86 · 3 s 4.70 · 6 s 4.42). 손가락 결합은 아니다
(`rj_dg_2_2` 만 지령했을 때 같은 손가락의 나머지 3관절은 안 움직였다). 빠른 지령에서
|v|max 가 정확히 2π rad/s(=60 rpm)로 포화한다. ⇒ 드라이버의 velocity 필드는 **관절 속도가
아니다**. 정책 관측의 hand_qd 로 그대로 쓰면 안 된다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TCP = Path("/home/user/rl_ws/robot_control/ros_ws/src/delto_m_ros2/delto_tcp_comm")
HDR = TCP / "include/delto_tcp_comm/delto_developer_TCP.hpp"
SRC = TCP / "src/delto_developer_TCP.cpp"
needs = pytest.mark.skipif(not HDR.exists(), reason="벤더 손 드라이버 없음")


@needs
def test_velocity_scale_is_rpm_not_tenth_degree():
    """raw 는 rpm 이다 — π/30. π/1800 은 0.1 deg/s 해석으로 **60배 작다**."""
    m = re.search(r"VELOCITY_SCALE\s*=\s*\(?\s*M_PI\s*/\s*([0-9.]+)", HDR.read_text(encoding="utf-8"))
    assert m, "VELOCITY_SCALE 을 못 찾았다"
    assert float(m.group(1)) == pytest.approx(30.0), (
        f"M_PI/{m.group(1)} — rpm 이면 M_PI/30 이어야 한다. M_PI/1800 이면 속도가 60배 작다")


@needs
def test_velocity_has_no_sign_inversion():
    """상위가 부호 반전을 제거했다(8c85809)."""
    src = SRC.read_text(encoding="utf-8")
    bad = [l.strip() for l in src.splitlines() if "VELOCITY_SCALE" in l and "-1" in l]
    assert not bad, f"속도에 부호 반전이 남아 있다: {bad}"


@needs
def test_socket_layer_has_a_receive_timeout():
    """오늘 손 접속이 오류 없이 무한 대기했다 — poll 기반 수신 타임아웃이 있어야 한다."""
    src = SRC.read_text(encoding="utf-8")
    assert "poll(" in src, "poll() 기반 타임아웃이 없다"
    assert "timeout_ms" in HDR.read_text(encoding="utf-8"), "RecvAll 에 타임아웃 인자가 없다"


@needs
def test_no_boost_dependency():
    """상위는 boost 를 걷어냈다(368117d) — 공개 API 는 그대로다."""
    assert "boost" not in HDR.read_text(encoding="utf-8").lower(), "헤더에 boost 가 남아 있다"
    assert "boost" not in (TCP / "package.xml").read_text(encoding="utf-8").lower()


@needs
def test_public_api_delto_hardware_uses_is_intact():
    """`delto_hardware` 가 부르는 8개 메서드는 교체 후에도 그대로여야 한다."""
    hdr = HDR.read_text(encoding="utf-8")
    for m in ("Connect", "Disconnect", "GetData", "GetFirmwareVersion",
              "SendDuty", "SetFTSensorOffset", "SetGPIO", "IsConnected"):
        assert re.search(rf"\b{m}\s*\(", hdr), f"{m} 이 헤더에서 사라졌다"
