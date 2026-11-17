"""rosbag2 sqlite(.db3) 읽기 — rclpy 없이 CDR 을 직접 푼다(판정에 쓰는 타입만). 읽기 전용 연결.

ROS 가 없는 분석 PC(server)에서 실기 bag 을 재생하려고 둔다. 타입이 늘면 DEC 에 더한다.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Iterator

import numpy as np


def _str(b: bytes, o: int) -> tuple[str, int]:
    n = struct.unpack_from("<I", b, o)[0]
    o += 4
    return (b[o:o + n - 1].decode("utf-8", "replace") if n else ""), o + n


def _al(o: int, a: int) -> int:        # CDR 정렬은 캡슐 헤더(4 바이트) 뒤 기준
    r = (o - 4) % a
    return o + (a - r if r else 0)


def _header(b: bytes) -> int:
    _, o = _str(b, 12)
    return o


def _multi(b: bytes, dt: str, w: int) -> np.ndarray:
    o = 4
    nd = struct.unpack_from("<I", b, o)[0]
    o += 4
    for _ in range(nd):
        o = _al(o, 4)
        _, o = _str(b, o)
        o = _al(o, 4) + 8
    o = _al(o, 4) + 4
    o = _al(o, 4)
    n = struct.unpack_from("<I", b, o)[0]
    o = _al(o + 4, w)
    return np.frombuffer(b, dt, n, o).astype(np.float64)


def rh_act(b: bytes) -> np.ndarray:     # GetAngleAct1 · GetForceAct1 · GetCurrentAct1 · SetAngle1: header, int32 hand_id, int32[6]
    o = _al(_header(b), 4)
    return np.frombuffer(b, "<i4", 6, o + 4).astype(np.float64)


def jstate(b: bytes) -> tuple[list[str], np.ndarray]:
    o = _al(_header(b), 4)
    n = struct.unpack_from("<I", b, o)[0]
    o += 4
    names = []
    for _ in range(n):
        o = _al(o, 4)
        s, o = _str(b, o)
        names.append(s)
    o = _al(o, 4)
    m = struct.unpack_from("<I", b, o)[0]
    o = _al(o + 4, 8)
    return names, np.frombuffer(b, "<f8", m, o)


def pose(b: bytes) -> np.ndarray:       # PoseStamped → x y z qx qy qz qw
    o = _al(_header(b), 8)
    return np.frombuffer(b, "<f8", 7, o)


def string(b: bytes) -> str:
    return _str(b, 4)[0]


DEC = {"rh56f1_interfaces/msg/GetAngleAct1": rh_act, "rh56f1_interfaces/msg/GetForceAct1": rh_act,
       "rh56f1_interfaces/msg/GetCurrentAct1": rh_act, "rh56f1_interfaces/msg/SetAngle1": rh_act,
       "std_msgs/msg/Float64MultiArray": lambda b: _multi(b, "<f8", 8), "geometry_msgs/msg/PoseStamped": pose,
       "std_msgs/msg/String": string, "sensor_msgs/msg/JointState": jstate}


def topics(bag_dir: Path) -> dict[str, str]:
    out = {}
    for db in sorted(Path(bag_dir).glob("*.db3")):
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        out.update({n: t for n, t in c.execute("select name, type from topics")})
        c.close()
    return out


def read(bag_dir: Path, names) -> Iterator[tuple[str, float, object]]:
    """bag 폴더(여러 .db3) → (토픽, 기록 시각 s, 값) 시각 순. names = 읽을 토픽 이름 집합(또는 판별 함수)."""
    want = names if callable(names) else (lambda n, s=set(names): n in s)
    rows = []
    for db in sorted(Path(bag_dir).glob("*.db3")):
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for tid, name, typ in c.execute("select id, name, type from topics").fetchall():
            if not want(name) or typ not in DEC:
                continue
            f = DEC[typ]
            for ts, data in c.execute("select timestamp, data from messages where topic_id = ? order by timestamp", (tid,)):
                try:
                    rows.append((ts * 1e-9, name, f(data)))
                except (struct.error, ValueError):
                    continue
        c.close()
    rows.sort(key=lambda r: r[0])
    for t, n, v in rows:
        yield n, t, v
