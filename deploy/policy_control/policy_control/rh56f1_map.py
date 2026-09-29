"""Inspire RH56F1 — sim 관절(rad) ↔ 벤더 각도 레지스터(0.1°) 변환. 순수(ROS 없음).

09.29 사용자: "sim2real 과 robot_control 쪽에서 rh56f1 제어 part 연결". 드라이버(robot_control inspire_rh56f1)는
벤더 단위만 말한다 — 슬롯 순서가 sim 과 반대이고(새끼부터), 네 손가락은 레지스터가 **클수록 펴진다**.
변환표는 config/rh56f1_hand_map.yaml 하나에만 있다. pd 백엔드 · 상태 노드 · fake 손 · 축 확인 도구가 같이 읽는다.

  HandMap.to_register(q)    손 6관절(joint_order, rad) → 슬롯 순 int[6]. 확인 안 된 축은 -1(드라이버가 그 축을 두고 간다)
  HandMap.to_rad(reg)       슬롯 순 레지스터 → joint_order rad (확인 안 된 축도 값은 낸다 — 읽기는 막지 않는다)
  HandMap.with_mimic(q)     구동 6 → 자산의 종속 6 까지 {이름: rad}
  HandMap.touch_sim_order   벤더 손가락 힘(새끼부터) → sim 순(엄지부터) N
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

SCHEMA = "policy_control/rh56f1_hand_map/v1"
N = 6
#: 드라이버가 SetAngle1 에서 "이 축은 움직이지 않는다" 로 읽는 값
LEAVE = -1
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "rh56f1_hand_map.yaml"


class HandMapError(ValueError):
    pass


@dataclass(frozen=True)
class Axis:
    name: str               # 접두어 없는 관절 이름(thumb_1 …)
    slot: int               # 드라이버 슬롯 0..5
    rad: tuple[float, float]
    reg: tuple[int, int]    # rad[0] · rad[1] 에서의 레지스터
    verified: bool

    def to_reg(self, q: float) -> int:
        lo, hi = self.rad
        t = (float(np.clip(q, min(lo, hi), max(lo, hi))) - lo) / (hi - lo)
        return int(round(self.reg[0] + t * (self.reg[1] - self.reg[0])))

    def to_rad(self, r: float) -> float:
        a, b = self.reg
        t = (float(np.clip(r, min(a, b), max(a, b))) - a) / (b - a)
        return self.rad[0] + t * (self.rad[1] - self.rad[0])


@dataclass(frozen=True)
class HandMap:
    axes: tuple[Axis, ...]                          # joint_order 순
    mimic: tuple[tuple[str, str, float], ...]       # (종속, 원본, 배율) — 원본이 먼저 계산돼 있게 정렬됨
    touch_perm: tuple[int, ...]                     # sim 순 i ← 벤더 칸 touch_perm[i]
    touch_unit_n: float
    command_max_hz: float

    @property
    def joint_order(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.axes)

    def names(self, side: str) -> list[str]:
        return [f"{side[0]}_hj_{a.name}" for a in self.axes]

    def unverified(self) -> list[str]:
        return [a.name for a in self.axes if not a.verified]

    def to_register(self, q: Sequence[float], *, allow_unverified: bool = False) -> list[int]:
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape[0] != N or not np.all(np.isfinite(arr)):
            raise HandMapError(f"손 목표는 유한한 {N} 개여야 한다: {arr}")
        out = [LEAVE] * N
        for a, v in zip(self.axes, arr):
            out[a.slot] = a.to_reg(v) if (a.verified or allow_unverified) else LEAVE
        return out

    def to_rad(self, reg: Sequence[float]) -> np.ndarray:
        reg = list(reg)
        if len(reg) != N:
            raise HandMapError(f"레지스터는 {N} 개여야 한다: {reg}")
        return np.array([a.to_rad(reg[a.slot]) for a in self.axes])

    def with_mimic(self, q: Sequence[float]) -> dict[str, float]:
        out = {a.name: float(v) for a, v in zip(self.axes, q)}
        for name, of, k in self.mimic:
            out[name] = k * out[of]
        return out

    def touch_sim_order(self, vendor: Sequence[float]) -> np.ndarray:
        v = np.asarray(vendor, dtype=float).reshape(-1)
        if v.shape[0] != len(self.touch_perm):
            raise HandMapError(f"손가락 힘은 {len(self.touch_perm)} 개여야 한다: {v.shape[0]}")
        return v[list(self.touch_perm)] * self.touch_unit_n


def parse(raw: Mapping) -> HandMap:
    if raw.get("schema") != SCHEMA:
        raise HandMapError(f"schema 는 {SCHEMA} 여야 한다")
    slots = list(raw["slot_order"])
    order = list(raw["joint_order"])
    if sorted(slots) != sorted(order) or len(order) != N:
        raise HandMapError(f"slot_order {slots} 와 joint_order {order} 는 같은 {N} 관절이어야 한다")
    axes = []
    for name in order:
        j = raw["joints"][name]
        rad, reg = tuple(float(v) for v in j["rad"]), tuple(int(v) for v in j["reg"])
        if len(rad) != 2 or len(reg) != 2 or rad[0] == rad[1] or reg[0] == reg[1]:
            raise HandMapError(f"{name}: rad · reg 는 서로 다른 두 끝점이어야 한다")
        if min(reg) < 0:
            raise HandMapError(f"{name}: 레지스터는 0 이상(-1 은 '움직이지 않음' 예약값)")
        axes.append(Axis(name, slots.index(name), rad, reg, bool(j.get("verified", False))))
    mimic, known = [], set(order)
    pending = dict(raw.get("mimic") or {})
    while pending:
        ready = [k for k, v in pending.items() if v["of"] in known]
        if not ready:
            raise HandMapError(f"mimic 원본을 풀 수 없다: {sorted(pending)}")
        for k in ready:
            mimic.append((k, pending[k]["of"], float(pending[k]["k"])))
            known.add(k)
            del pending[k]
    t = raw["touch"]
    vendor, sim = list(t["vendor_order"]), list(t["sim_order"])
    if sorted(vendor) != sorted(sim):
        raise HandMapError("touch vendor_order · sim_order 는 같은 손가락이어야 한다")
    return HandMap(axes=tuple(axes), mimic=tuple(mimic), touch_perm=tuple(vendor.index(f) for f in sim),
                   touch_unit_n=float(t["unit_n"]), command_max_hz=float(raw.get("command_max_hz", 30.0)))


def load(path: str | Path = DEFAULT_PATH) -> HandMap:
    return parse(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
