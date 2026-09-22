#!/usr/bin/env python3
"""RH56F1 손: sim 라디안 ↔ 드라이버 레지스터 정수 순수 변환.

이 모듈은 ROS에 의존하지 않는다. Isaac Sim 정책이 내보내는 canonical drive 관절
각도(라디안)와 `inspire_control_ros2` 드라이버가 받는 `SetAngle1`/`GetAngleAct1`의
정수 레지스터 값 사이를 선형으로 변환한다.

한 손은 6개의 drive 관절을 가진다 (control_joint_order 손 부분):
    thumb_1, thumb_2, index_1, middle_1, ring_1, pinky_1
드라이버 `SetAngle1.joint_values`는 int32[6]이고, 각 슬롯은 액추에이터 ID(1~6)에
대응한다. sim 관절 순서와 드라이버 슬롯 순서가 다르므로 `driver_index`로 순열을 준다.

선형 맵:  reg = reg_lo + (rad - rad_lo)/(rad_hi - rad_lo) * (reg_hi - reg_lo)
역변환:   rad = rad_lo + (reg - reg_lo)/(reg_hi - reg_lo) * (rad_hi - rad_lo)

기본 캘리브레이션(아래 default_hand_calibration)은 URDF 관절 범위와 RH56F1 매뉴얼의
레지스터 범위로 채워져 있다. 단, **손가락↔액추에이터 ID 순서**와 **레지스터 증가 방향
(굽힘 부호)**은 하드웨어 배선/펌웨어에 따라 달라질 수 있으므로 실물에서 확인해야 한다.
그 값들은 코드가 아니라 config yaml로 노출되어 있으니 yaml만 고치면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

# 한 손의 drive 관절 개수 (RH56F1은 손가락당 1 drive DOF + 엄지 2 DOF).
HAND_DRIVE_DOF = 6

# sim canonical drive 관절 이름 (control_joint_order의 손 부분, 접두사 제외).
SIM_DRIVE_SUFFIXES = (
    "thumb_1",
    "thumb_2",
    "index_1",
    "middle_1",
    "ring_1",
    "pinky_1",
)


@dataclass(frozen=True)
class DriveCalibration:
    """sim drive 관절 하나 ↔ RH56F1 레지스터 슬롯 하나의 선형 캘리브레이션.

    Attributes:
        sim_joint: canonical 관절 이름 (예: "r_hj_index_1").
        driver_index: `SetAngle1.joint_values` 안에서의 0-based 슬롯 위치 (0~5).
        rad_lo, rad_hi: sim 관절 각도의 하한/상한 (라디안, URDF limit).
        reg_lo, reg_hi: 위 rad_lo/rad_hi에 대응하는 드라이버 레지스터 정수값.
            reg_lo > reg_hi 이면 레지스터가 각도 증가에 대해 감소하는 배선을 뜻한다.
    """

    sim_joint: str
    driver_index: int
    rad_lo: float
    rad_hi: float
    reg_lo: int
    reg_hi: int

    def __post_init__(self) -> None:
        if not 0 <= self.driver_index < HAND_DRIVE_DOF:
            raise ValueError(
                f"driver_index must be 0..{HAND_DRIVE_DOF - 1}, got {self.driver_index}"
            )
        if self.rad_hi == self.rad_lo:
            raise ValueError(f"{self.sim_joint}: rad_lo and rad_hi must differ")
        if self.reg_hi == self.reg_lo:
            raise ValueError(f"{self.sim_joint}: reg_lo and reg_hi must differ")

    @property
    def _reg_bounds(self) -> tuple[int, int]:
        return (min(self.reg_lo, self.reg_hi), max(self.reg_lo, self.reg_hi))

    def rad_to_reg(self, rad: float) -> int:
        clamped = min(max(rad, min(self.rad_lo, self.rad_hi)), max(self.rad_lo, self.rad_hi))
        ratio = (clamped - self.rad_lo) / (self.rad_hi - self.rad_lo)
        reg = self.reg_lo + ratio * (self.reg_hi - self.reg_lo)
        low, high = self._reg_bounds
        return int(min(max(round(reg), low), high))

    def reg_to_rad(self, reg: int) -> float:
        low, high = self._reg_bounds
        clamped = min(max(reg, low), high)
        ratio = (clamped - self.reg_lo) / (self.reg_hi - self.reg_lo)
        rad = self.rad_lo + ratio * (self.rad_hi - self.rad_lo)
        return min(max(rad, min(self.rad_lo, self.rad_hi)), max(self.rad_lo, self.rad_hi))


@dataclass(frozen=True)
class HandCalibration:
    """한 손 전체(6 drive 관절)의 캘리브레이션."""

    joints: tuple[DriveCalibration, ...]

    def __post_init__(self) -> None:
        if len(self.joints) != HAND_DRIVE_DOF:
            raise ValueError(
                f"hand needs {HAND_DRIVE_DOF} drive joints, got {len(self.joints)}"
            )
        slots = sorted(j.driver_index for j in self.joints)
        if slots != list(range(HAND_DRIVE_DOF)):
            raise ValueError(f"driver_index must be a permutation of 0..5, got {slots}")

    @property
    def sim_joint_order(self) -> tuple[str, ...]:
        """명령 배열이 따르는 sim 관절 순서 (입력 순서 그대로)."""
        return tuple(j.sim_joint for j in self.joints)

    def sim_to_registers(self, rads: list[float]) -> list[int]:
        """sim 관절 각도(입력 순서) → 드라이버 슬롯 순서 레지스터 int[6]."""
        if len(rads) != HAND_DRIVE_DOF:
            raise ValueError(f"expected {HAND_DRIVE_DOF} angles, got {len(rads)}")
        regs = [0] * HAND_DRIVE_DOF
        for cal, rad in zip(self.joints, rads):
            regs[cal.driver_index] = cal.rad_to_reg(rad)
        return regs

    def registers_to_sim(self, regs: list[int]) -> dict[str, float]:
        """드라이버 슬롯 순서 레지스터 int[6] → {sim 관절: 라디안}."""
        if len(regs) != HAND_DRIVE_DOF:
            raise ValueError(f"expected {HAND_DRIVE_DOF} registers, got {len(regs)}")
        return {cal.sim_joint: cal.reg_to_rad(regs[cal.driver_index]) for cal in self.joints}


# 손가락↔액추에이터 ID 기본 순서 (Inspire 관례: ID1~4 = pinky/ring/middle/index,
# ID5 = 엄지 굽힘, ID6 = 엄지 회전). driver_index = ID - 1.
# TODO(hardware): 실물에서 각 손가락을 하나씩 굽혀 어느 슬롯이 반응하는지 확인하고,
#   레지스터 증가 방향(reg_lo/reg_hi 스왑 여부)도 검증할 것.
_DEFAULT_SLOTS = {
    "pinky_1": 0,
    "ring_1": 1,
    "middle_1": 2,
    "index_1": 3,
    "thumb_2": 4,  # 엄지 굽힘 (좁은 범위)
    "thumb_1": 5,  # 엄지 회전/벌림 (넓은 범위)
}

# 관절별 (rad_lo, rad_hi, reg_lo, reg_hi). rad은 URDF limit, reg은 RH56F1 매뉴얼.
_DEFAULT_RANGES = {
    "index_1": (0.0, 1.5285593588966337, 900, 1740),
    "middle_1": (0.0, 1.5285593588966337, 900, 1740),
    "ring_1": (0.0, 1.5285593588966337, 900, 1740),
    "pinky_1": (0.0, 1.5285593588966337, 900, 1740),
    "thumb_2": (0.0, 0.4745550236172582, 1100, 1350),
    "thumb_1": (0.0, 2.0943951023931953, 600, 1800),
}


def default_hand_calibration(prefix: str) -> HandCalibration:
    """기본 캘리브레이션 생성. prefix는 "r_hj_" 또는 "l_hj_".

    관절 순서는 SIM_DRIVE_SUFFIXES(=명령 배열 순서)를 따른다.
    """
    joints = []
    for suffix in SIM_DRIVE_SUFFIXES:
        rad_lo, rad_hi, reg_lo, reg_hi = _DEFAULT_RANGES[suffix]
        joints.append(
            DriveCalibration(
                sim_joint=f"{prefix}{suffix}",
                driver_index=_DEFAULT_SLOTS[suffix],
                rad_lo=rad_lo,
                rad_hi=rad_hi,
                reg_lo=reg_lo,
                reg_hi=reg_hi,
            )
        )
    return HandCalibration(joints=tuple(joints))


def hand_calibration_from_config(prefix: str, entries: list[dict]) -> HandCalibration:
    """yaml에서 읽은 항목 리스트로 캘리브레이션 생성.

    각 항목: {suffix, driver_index, rad_lo, rad_hi, reg_lo, reg_hi}.
    관절 순서는 entries 순서를 따른다 (= 명령 배열 순서).
    """
    joints = []
    for e in entries:
        joints.append(
            DriveCalibration(
                sim_joint=f"{prefix}{e['suffix']}",
                driver_index=int(e["driver_index"]),
                rad_lo=float(e["rad_lo"]),
                rad_hi=float(e["rad_hi"]),
                reg_lo=int(e["reg_lo"]),
                reg_hi=int(e["reg_hi"]),
            )
        )
    return HandCalibration(joints=tuple(joints))
