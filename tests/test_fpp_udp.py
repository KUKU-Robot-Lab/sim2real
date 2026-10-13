"""FP++ 자세 UDP 형식 — vision-3090 → 로봇 PC. ROS·네트워크 불필요.

받는 쪽이 잘못된 패킷을 **발행하지 않고 버리는지**가 핵심이다: 이 자세는 정책 입력으로 들어간다.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fpp_udp as U  # noqa: E402

KNOWN = ("cup_big_s100", "shaker_closed")
POSE = U.PosePacket(seq=7, name="cup_big_s100", frame="camera_color_optical_frame", stamp=(1790399673, 485416039),
                    p=(0.012, -0.034, 0.512), q=(0.0, 0.0, math.sin(0.3), math.cos(0.3)))


def test_a_pose_survives_the_round_trip_exactly():
    got = U.decode(U.encode(POSE), KNOWN)
    assert got == POSE
    assert len(U.encode(POSE)) < U.MAX_BYTES // 4                  # 30 Hz 로 보내도 가볍다


def test_a_heartbeat_carries_the_camera_rate():
    assert U.decode(U.encode(U.Heartbeat(seq=3, camera_hz=29.8)), KNOWN) == U.Heartbeat(seq=3, camera_hz=29.8)


def _raw(**over):
    body = json.loads(U.encode(POSE))
    body.update(over)
    return json.dumps(body).encode()


@pytest.mark.parametrize("data, why", [
    (b"\xff\xfe", "not JSON"),
    (b"[1,2]", "version"),
    (_raw(v=2), "version"),
    (_raw(kind="odd"), "unknown kind"),
    (_raw(name="knife"), "unknown object"),
    (_raw(frame=""), "frame"),
    (_raw(stamp=[1]), "stamp"),
    (_raw(stamp=[1, 2_000_000_000]), "nanosec"),
    (_raw(p=[0, 0]), "p must"),
    (_raw(p=[0, "x", 0]), "p\\[1\\]"),
    (_raw(q=[0, 0, 0, 2]), "unit quaternion"),
    (_raw(seq=True), "seq"),
    (b'{"v":1,"kind":"hb","seq":1,"camera_hz":-1}', "camera_hz"),
    (b'{"v":1,"kind":"pose","seq":1,"name":"cup_big_s100","frame":"f","stamp":[0,0],"p":[0,0,NaN],"q":[0,0,0,1]}',
     "finite"),
    (b"x" * (U.MAX_BYTES + 1), "too large"),
])
def test_anything_malformed_is_refused_not_published(data, why):
    with pytest.raises(ValueError, match=why):
        U.decode(data, KNOWN)


def test_the_rate_meter_counts_only_the_recent_window():
    m = U.RateMeter(window_s=2.0)
    for i in range(60):                                                # 30 Hz 로 2 초
        m.tick(i / 30.0)
    assert m.hz(59 / 30.0) == pytest.approx(30.0)
    assert m.hz(10.0) == 0.0                                           # 끊기면 0


def test_the_camera_rate_drops_to_zero_when_heartbeats_stop():
    assert U.camera_hz_at(None, 5.0) == 0.0
    assert U.camera_hz_at((4.0, 29.8), 5.0) == 29.8
    assert U.camera_hz_at((4.0, 29.8), 4.0 + U.STALE_S + 0.1) == 0.0


def test_only_newer_packets_pass_per_object_and_a_restarted_sender_is_accepted():
    g = U.SeqGate()
    assert g.accept("cup", 5) and g.accept("shaker", 1)
    assert not g.accept("cup", 5) and not g.accept("cup", 4)          # 겹침 · 역순은 버린다
    assert g.accept("cup", 6)
    assert g.accept("cup", 6 + 5000) and g.accept("cup", 0)           # 크게 뒤로 = 송신기 재기동


def test_the_sender_sends_each_frame_once_per_object():
    """같은 stamp 의 자세가 다시 들어오면(되돌아온 사본) 보내지 않는다 — 순환을 송신기에서도 끊는다."""
    gate = U.StampGate()
    assert gate.accept("cyl60", (10, 5))
    assert not gate.accept("cyl60", (10, 5))
    assert not gate.accept("cyl60", (10, 4))
    assert gate.accept("cyl60_blue", (10, 5))
    assert gate.accept("cyl60", (11, 0))


# ---- 10.13 live 추적도 UDP 로(arm4090 실기: 컨테이너 localhost DDS → 정책 노드가 live_pose 를 못 받았다)
def test_a_live_pose_round_trips_and_is_marked():
    live = U.PosePacket(**{**POSE.__dict__, "live": True})
    got = U.decode(U.encode(live), KNOWN)
    assert got == live and got.live and got.key == "cup_big_s100#live"
    assert b'"live":true' in U.encode(live) and b"live" not in U.encode(POSE)   # 찍기 패킷은 예전 형식 그대로
    assert U.decode(U.encode(POSE), KNOWN).key == "cup_big_s100"


def test_a_non_bool_live_flag_is_refused():
    raw = json.loads(U.encode(POSE)); raw["live"] = 1
    with pytest.raises(ValueError):
        U.decode(json.dumps(raw).encode(), KNOWN)


def test_live_and_snapshot_seq_are_gated_separately():
    g = U.SeqGate()
    assert g.accept("cup_big_s100#live", 10)
    assert g.accept("cup_big_s100", 5)                 # live 가 앞서 있어도 찍기는 막히지 않는다


def test_the_relay_nodes_carry_the_live_topic():
    src = Path(__file__).resolve().parents[1] / "scripts" / "nodes"
    tx, rx = (src / "fpp_pose_tx.py").read_text(), (src / "fpp_pose_rx.py").read_text()
    assert "live_input_topic(n)" in tx and "live=live" in tx
    assert "live_input_topic(n)" in rx and "packet.live" in rx
