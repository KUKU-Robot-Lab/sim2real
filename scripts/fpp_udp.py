#!/usr/bin/env python3
"""FP++ 자세를 vision-3090 → 로봇 PC 로 넘기는 UDP 패킷 형식. ROS 없이 import 된다(test_fpp_udp.py 대상).

왜 DDS 가 아니라 UDP 인가(09.26 실측):
  · 이 wifi AP 는 멀티캐스트를 막아 두 PC 의 DDS 가 서로를 찾지 못한다.
  · vision-3090 의 wifi 상향이 원격 데스크톱(RustDesk, 약 9 MB/s)으로 차 있으면 DDS 쓰기가 송신 버퍼에서 막혀
    카메라 노드가 기동 중에 멈췄다(깊이 → 컬러 센서 49 s, "Node Is Up" 까지 4 분, 그 뒤에도 무발행).
  · 카메라를 localhost 전용으로 띄우면 0.15 s 에 뜨고 29.6 Hz 로 낸다.
그래서 영상 · FP++ 는 vision-3090 안에서만 돌고(ROS_LOCALHOST_ONLY=1), 정책 입력인 **물체 자세만** 이 형식으로
넘긴다. UDP 는 보내는 쪽을 막지 않고, 30 Hz 로 새 값이 오므로 한두 개 잃어도 다음 것이 대신한다.

한 패킷 = JSON 한 개(utf-8). 종류 둘:
  pose       {"v":1,"kind":"pose","seq":n,"name":"cup_big_s100","frame":"camera_color_optical_frame",
              "stamp":[sec,nanosec],"p":[x,y,z],"q":[x,y,z,w]}          — FP++ 의 카메라 프레임 자세 그대로
  heartbeat  {"v":1,"kind":"hb","seq":n,"camera_hz":29.8}              — 물체가 안 잡혀도 카메라가 사는지
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Collection

VERSION = 1
PORT = 51126
MAX_BYTES = 1024
HEARTBEAT_S = 0.5
#: 이 시간 동안 heartbeat 가 없으면 카메라 hz 를 0 으로 본다(송신기 · 링크가 죽었다)
STALE_S = 2.0
#: 받는 쪽이 카메라 hz 를 내는 토픽(std_msgs/Float32) — 인지 런처가 읽는다
CAMERA_HZ_TOPIC = "/perception/camera_hz"
#: seq 가 이만큼 뒤로 가면 송신기가 다시 떴다고 본다(0 부터 새로 센다)
SEQ_RESTART_GAP = 1000
_QUAT_NORM_TOL = 0.05


@dataclass(frozen=True)
class PosePacket:
    seq: int
    name: str
    frame: str
    stamp: tuple[int, int]
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]


@dataclass(frozen=True)
class Heartbeat:
    seq: int
    camera_hz: float


def encode(packet: PosePacket | Heartbeat) -> bytes:
    if isinstance(packet, PosePacket):
        body = {"v": VERSION, "kind": "pose", "seq": packet.seq, "name": packet.name, "frame": packet.frame,
                "stamp": list(packet.stamp), "p": list(packet.p), "q": list(packet.q)}
    else:
        body = {"v": VERSION, "kind": "hb", "seq": packet.seq, "camera_hz": round(float(packet.camera_hz), 2)}
    data = json.dumps(body, separators=(",", ":")).encode()
    if len(data) > MAX_BYTES:
        raise ValueError(f"packet {len(data)} B > {MAX_BYTES} B")
    return data


def decode(data: bytes, known: Collection[str]) -> PosePacket | Heartbeat:
    """패킷 → 값. 형식이 틀리거나 모르는 물체면 ValueError — 받는 쪽은 버리고 센다(발행하지 않는다)."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"packet too large: {len(data)} B")
    try:
        raw = json.loads(data.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("v") != VERSION:
        raise ValueError(f"version must be {VERSION}: {raw!r}"[:200])
    seq = _int(raw.get("seq"), "seq")
    kind = raw.get("kind")
    if kind == "hb":
        hz = _finite(raw.get("camera_hz"), "camera_hz")
        if hz < 0:
            raise ValueError(f"camera_hz must be >= 0, got {hz}")
        return Heartbeat(seq=seq, camera_hz=hz)
    if kind != "pose":
        raise ValueError(f"unknown kind: {kind!r}")
    name = raw.get("name")
    if name not in known:
        raise ValueError(f"unknown object: {name!r}")
    frame = raw.get("frame")
    if not isinstance(frame, str) or not frame:
        raise ValueError("frame must be a non-empty string")
    stamp = raw.get("stamp")
    if not isinstance(stamp, list) or len(stamp) != 2:
        raise ValueError("stamp must be [sec, nanosec]")
    sec, nsec = _int(stamp[0], "stamp.sec"), _int(stamp[1], "stamp.nanosec")
    if not 0 <= nsec < 1_000_000_000:
        raise ValueError(f"stamp.nanosec out of range: {nsec}")
    p = _vector(raw.get("p"), 3, "p")
    q = _vector(raw.get("q"), 4, "q")
    norm = math.sqrt(sum(v * v for v in q))
    if abs(norm - 1.0) > _QUAT_NORM_TOL:
        raise ValueError(f"q is not a unit quaternion (|q| = {norm:.3f})")
    return PosePacket(seq=seq, name=str(name), frame=frame, stamp=(sec, nsec), p=p, q=q)


def _int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _vector(value, n: int, name: str) -> tuple:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{name} must be a list of {n} numbers")
    return tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(value))


class RateMeter:
    """최근 `window_s` 동안 받은 횟수로 hz 를 잰다(카메라 fps · 수신 자세 fps)."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._stamps: list[float] = []

    def tick(self, now: float) -> None:
        self._stamps = [t for t in self._stamps if now - t < self.window_s] + [now]

    def hz(self, now: float) -> float:
        return len([t for t in self._stamps if now - t < self.window_s]) / self.window_s


def camera_hz_at(last: tuple[float, float] | None, now: float) -> float:
    """마지막 heartbeat `(받은 시각, hz)` → 지금 믿을 카메라 hz. heartbeat 가 끊기면 0."""
    if last is None or now - last[0] > STALE_S:
        return 0.0
    return last[1]


class SeqGate:
    """물체마다 더 새 seq 만 통과시킨다 — UDP 는 순서가 뒤바뀌거나 겹쳐 올 수 있다. 옛 자세를 다시 내지 않는다.
    seq 가 크게 뒤로 가면(`SEQ_RESTART_GAP`) 송신기가 다시 뜬 것으로 보고 받아들인다."""

    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def accept(self, key: str, seq: int) -> bool:
        last = self._last.get(key)
        if last is not None and last - SEQ_RESTART_GAP < seq <= last:
            return False
        self._last[key] = seq
        return True
