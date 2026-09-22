"""relay -> viewer UDP 데이터그램 형식. 순수 파이썬(ROS·Isaac 무관, py3.10/3.11 공용).

형식: UTF-8 JSON 한 덩어리 `{"t": <float epoch s>, "names": [str...], "positions": [float...]}`.
names 는 sim 자산 canonical 이름(r_aj_1, r_hj_index_2, ...), positions 는 rad.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47811
#: UDP 페이로드 상한(IPv4 65507)보다 여유 있게. 관절 ~60 개면 3 KB 안팎이다.
MAX_DATAGRAM_BYTES = 60000
MAX_JOINTS = 256


@dataclass(frozen=True)
class JointPacket:
    t: float
    names: tuple[str, ...]
    positions: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.positions))


def _validate(t: float, names: Sequence[str], positions: Sequence[float]) -> JointPacket:
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(float(t)):
        raise ValueError(f"t 가 유한한 수가 아니다: {t!r}")
    if len(names) != len(positions):
        raise ValueError(f"names {len(names)} / positions {len(positions)} 길이 불일치")
    if len(names) > MAX_JOINTS:
        raise ValueError(f"관절 수 {len(names)} > {MAX_JOINTS}")
    if len(set(names)) != len(names):
        raise ValueError("names 중복")
    out_pos = []
    for n, p in zip(names, positions):
        if not isinstance(n, str) or not n:
            raise ValueError(f"관절 이름이 빈 문자열이거나 str 이 아니다: {n!r}")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(float(p)):
            raise ValueError(f"{n} 위치가 유한한 수가 아니다: {p!r}")
        out_pos.append(float(p))
    return JointPacket(t=float(t), names=tuple(names), positions=tuple(out_pos))


def encode(t: float, joints: Mapping[str, float]) -> bytes:
    """canonical -> rad 사전을 데이터그램으로. 이름순 정렬(결정론)."""
    names = sorted(joints)
    pkt = _validate(t, names, [joints[n] for n in names])
    data = json.dumps({"t": pkt.t, "names": list(pkt.names), "positions": list(pkt.positions)},
                      separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_DATAGRAM_BYTES:
        raise ValueError(f"데이터그램 {len(data)} B > {MAX_DATAGRAM_BYTES} B")
    return data


def decode(data: bytes) -> JointPacket:
    """데이터그램 -> JointPacket. 형식이 틀리면 ValueError(뷰어는 그 패킷만 버린다)."""
    if len(data) > MAX_DATAGRAM_BYTES:
        raise ValueError(f"데이터그램 {len(data)} B > {MAX_DATAGRAM_BYTES} B")
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 해석 실패: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("최상위가 객체가 아니다")
    missing = {"t", "names", "positions"} - set(obj)
    if missing:
        raise ValueError(f"필드 누락: {sorted(missing)}")
    names, positions = obj["names"], obj["positions"]
    if not isinstance(names, list) or not isinstance(positions, list):
        raise ValueError("names/positions 는 배열이어야 한다")
    return _validate(obj["t"], names, positions)
