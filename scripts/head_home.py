#!/usr/bin/env python3
"""head 를 기준 자세로 보내고 게인까지 **올바른 순서로** 적용한다. local5090 기본값.

값은 `config/head_home.yaml` 하나에 모여 있다 — 자세·baud·게인의 단일 진실원천이다.

★★**순서가 핵심이다.** DYNAMIXEL 은 Operating Mode 를 쓰면 제어 게인을 그 모드의
기본값으로 **리셋한다.** 게인을 모드보다 먼저 넣으면 조용히 지워지고 처짐만 남는다
(2026-09-01 에 겪었다 — `apply_head_gains.py` 로 I=400 을 넣은 뒤 `head_goto_hold.py`
를 돌렸더니 I 가 0 으로 돌아가 tilt 가 +1.56° 처졌다).

    토크 off → Operating Mode → **게인** → 목표 위치 → 토크 on

Position I Gain 은 RAM 이라 **전원을 끄면 사라진다.** bringup 마다 이 스크립트를 돌린다.

    python head_home.py                 # 계획만 출력
    python head_home.py --execute       # 적용 + 검증
    python head_home.py --execute --pan 5 --tilt -25   # 일회성 다른 자세
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from head_compliant_hold import (
    ADDR_GOAL_POSITION,
    ADDR_OPERATING_MODE,
    ADDR_POSITION_I_GAIN,
    ADDR_TORQUE_ENABLE,
    TORQUE_OFF,
    TORQUE_ON,
    CompliantController,
    tick_to_deg,
)
from head_position_hold_node import (
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    OP_MODE_EXTENDED_POSITION,
    deg_to_tick,
)
from head_turns import (
    ADDR_HOMING_OFFSET,
    ADDR_PRESENT_POSITION,
    OP_MODE_POSITION,
    TICKS_PER_TURN,
    HeadReachError,
    droop_tick,
    physical_tick,
    torque_on_starts,
    torque_on_tick,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "head_home.yaml"
GAIN_MAX = 16383
SETTLE_SECONDS = 4.0
VERIFY_SAMPLES = 30
VERIFY_TOLERANCE_DEG = 0.3
#: goal_bias_tick 상한(5.3°) — head_pose_check.MAX_GOAL_OFFSET 과 같다. 이보다 크면 보정이 아니라 다른 문제다
GOAL_BIAS_MAX = 60
#: 처짐 복구(lift_droop): 확장 위치(4)로 올라가 이 틱 안에 닿기를 기다린다. 42° 위로 · 프로파일 속도 50(약 69°/s)이면 1 s 안이다
LIFT_TOL_TICK = 12
LIFT_TIMEOUT_S = 5.0
#: 위치 모드(3)로 돌아가며 토크를 잠깐 끌 때 고개와 경계 사이에 남아야 하는 여유
LIFT_EDGE_MARGIN_DEG = 10.0
#: 설정 droop_limit_tick 이 경계(offset+4096) 너머 이 범위 안이어야 한다(실측 3298 = 경계 + 22.9°)
DROOP_LIMIT_SPAN_TICK = round(30.0 * TICKS_PER_TURN / 360.0)

_REQUIRED = ("port", "baud", "motors", "position_i_gain", "operating_mode",
             "profile_acceleration", "profile_velocity")


@dataclass(frozen=True)
class HeadHome:
    port: str
    baud: int
    targets_deg: dict[int, float]
    names: dict[int, str]
    position_i_gain: int
    operating_mode: int
    profile_acceleration: int
    profile_velocity: int
    #: 목표 레지스터에 더해 쓰는 틱(모터별, 없으면 0). ★2026-10-28 RH56F1 머리는 목표보다 일정 틱 큰 자리에 선다 —
    #  펌웨어 궤적 끝(Position Trajectory)부터 목표와 다르다(pan +19 · tilt +3, 1805→1824 · 1815→1834 · 2807→2810).
    #  pan 1.67° 는 드라이버 보정 상한(1°)을 넘어 트림으로 못 메운다. 판정(verify · 드라이버)은 원래 목표 기준이다.
    goal_bias_tick: dict[int, int] = field(default_factory=dict)
    #: 토크를 풀면 고개가 중력으로 내려가 구조물에 막히는 자리(모터별 틱, offset 적용 후 프레임). 있는 모터만 처짐 복구를 한다.
    droop_limit_tick: dict[int, int] = field(default_factory=dict)


def goal_tick(config: HeadHome, dxl_id: int, deg: float | None = None) -> int:
    """모터에 쓸 목표 틱 = 목표 각(없으면 설정 목표)의 틱 + 그 모터의 goal_bias_tick."""
    target = config.targets_deg[dxl_id] if deg is None else deg
    return deg_to_tick(target) + int(config.goal_bias_tick.get(dxl_id, 0))


def load_head_home(path: Path) -> HeadHome:
    """설정을 읽고 **경계에서 검증한다** — 잘못된 값이 모터까지 가면 안 된다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"설정에 빠진 항목: {', '.join(missing)}")

    gain = int(raw["position_i_gain"])
    if not 0 <= gain <= GAIN_MAX:
        raise ValueError(f"position_i_gain 게인은 0~{GAIN_MAX}: {gain}")

    targets: dict[int, float] = {}
    names: dict[int, str] = {}
    biases: dict[int, int] = {}
    droops: dict[int, int] = {}
    for name, motor in dict(raw["motors"]).items():
        dxl_id = int(motor["id"])
        deg = float(motor["deg"])
        if dxl_id in targets:
            raise ValueError(f"id 가 중복이다: {dxl_id}")
        if not -180.0 <= deg <= 180.0:
            raise ValueError(f"{name}: 각도는 -180~180: {deg}")
        bias = int(motor.get("goal_bias_tick", 0))
        if abs(bias) > GOAL_BIAS_MAX:
            raise ValueError(f"{name}: goal_bias_tick 은 ±{GOAL_BIAS_MAX} 틱 안: {bias}")
        targets[dxl_id] = deg
        names[dxl_id] = str(name)
        if bias:
            biases[dxl_id] = bias
        if motor.get("droop_limit_tick") is not None:
            droops[dxl_id] = int(motor["droop_limit_tick"])

    return HeadHome(
        port=str(raw["port"]), baud=int(raw["baud"]), targets_deg=targets, names=names,
        position_i_gain=gain, operating_mode=int(raw["operating_mode"]),
        profile_acceleration=int(raw["profile_acceleration"]),
        profile_velocity=int(raw["profile_velocity"]),
        goal_bias_tick=biases,
        droop_limit_tick=droops,
    )


def _read4(controller: CompliantController):
    """제어표 4바이트 칸을 부호 있는 값으로 읽는 함수 — head_turns 는 이것만 쓴다(읽기 전용)."""
    return lambda dxl_id, address: controller.read4_signed(dxl_id, address, f"addr {address}")


def check_reach(controller: CompliantController, config: HeadHome, dxl_ids) -> dict[int, int]:
    """토크를 켜기 전에 — 켜는 순간 목이 반대쪽으로 거의 한 바퀴 돌 자리면 HeadReachError(아무것도 쓰지 않는다).

    2026-10-28 RH 머리 tilt: 위치 모드는 토크를 켤 때 '한 바퀴 안 값 + Homing Offset' 으로 다시 맞춘다. 고개가 경계
    (홈보다 18° 더 숙인 곳) 너머에 있으면 반대쪽 끝으로 잡혀 홈까지 더 숙이는 쪽으로 ~340° 를 돈다(head_turns).
    """
    goals = {dxl_id: goal_tick(config, dxl_id) for dxl_id in dxl_ids}   # 실제로 쓸 목표(goal_bias_tick 포함)
    return torque_on_starts(_read4(controller), goals, operating_mode=config.operating_mode, names=config.names)


def here_tick(controller: CompliantController, config: HeadHome, dxl_id: int) -> int:
    """토크를 켠 뒤 모터가 읽을 지금 자리 — '지금 자리를 목표로' 쓸 때는 토크 꺼짐 값 대신 이것을 쓴다."""
    read4 = _read4(controller)
    return torque_on_tick(read4(dxl_id, ADDR_PRESENT_POSITION), read4(dxl_id, ADDR_HOMING_OFFSET),
                          config.operating_mode)


def apply_one(controller: CompliantController, config: HeadHome, dxl_id: int) -> None:
    """★순서가 전부다. 두 가지 펌웨어 동작에 각각 물린 적이 있다(2026-09-01).

    0. **토크를 켜기 전에 역회전 검사**(check_reach, 2026-10-28) — 걸리면 아무것도 쓰지 않는다.
    1. **Operating Mode 를 쓰면 제어 게인이 그 모드 기본값으로 리셋된다**
       → 게인은 모드 **뒤에**.
    2. **Torque Enable 이 0→1 이 되면 Goal Position 이 Present Position 으로 덮어써진다**
       (급격한 점프 방지) → 목표는 토크 **뒤에**. 앞에 쓰면 토크가 꺼진 동안 중력에
       떨어진 자리가 목표가 되어 조용히 어긋난다.

        검사 → 토크off → 모드 → 게인 → 프로파일 → 토크on → 목표
    """
    check_reach(controller, config, (dxl_id,))
    controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
    controller.write1(dxl_id, ADDR_OPERATING_MODE, config.operating_mode, "operating mode")
    controller.write2(dxl_id, ADDR_POSITION_I_GAIN, config.position_i_gain, "position i gain")
    controller.write4(dxl_id, ADDR_PROFILE_ACCELERATION, config.profile_acceleration, "accel")
    controller.write4(dxl_id, ADDR_PROFILE_VELOCITY, config.profile_velocity, "vel")
    controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON, "torque on")
    controller.write4(dxl_id, ADDR_GOAL_POSITION, goal_tick(config, dxl_id), "goal")


def droop_ticks(controller: CompliantController, config: HeadHome, dxl_ids) -> dict[int, int]:
    """경계 너머로 처져 짧은 길로 올라올 수 있는 모터 {id: 펼친 실제 틱}(head_turns.droop_tick) — 읽기만 한다."""
    if int(config.operating_mode) != OP_MODE_POSITION:
        return {}
    read4 = _read4(controller)
    found: dict[int, int] = {}
    edge_ticks = round(LIFT_EDGE_MARGIN_DEG * TICKS_PER_TURN / 360.0)
    for dxl_id in dxl_ids:
        limit = config.droop_limit_tick.get(dxl_id)
        if limit is None:
            continue
        name = config.names.get(dxl_id, f"id {dxl_id}")
        offset, goal = read4(dxl_id, ADDR_HOMING_OFFSET), goal_tick(config, dxl_id)
        edge = offset + TICKS_PER_TURN  # 위치 모드(3) 프레임 offset ~ offset+4095 바로 너머
        if not edge <= limit <= edge + DROOP_LIMIT_SPAN_TICK:
            raise HeadReachError(f"{name}: 설정 오류 — droop_limit_tick {limit} 은 경계 {edge} ~ {edge + DROOP_LIMIT_SPAN_TICK}"
                                 f" 안이어야 한다(offset {offset})")
        actual = droop_tick(read4(dxl_id, ADDR_PRESENT_POSITION), offset, goal, limit)
        if actual is None:
            continue
        if not (goal < actual and offset <= goal <= edge - 1 - edge_ticks):
            raise HeadReachError(f"{name}: 처짐 복구 거부 — 목표 {goal} 이 위치 모드 경계 안쪽 {LIFT_EDGE_MARGIN_DEG:g}° "
                                 f"({offset} ~ {edge - 1 - edge_ticks})에 없거나 처진 자리({actual})보다 위가 아니다")
        found[dxl_id] = actual
    return found


def lift_droop(controller: CompliantController, config: HeadHome, dxl_id: int) -> None:
    """경계 너머로 처진 고개를 확장 위치(4)로 짧은 길(위로)로 목표까지 올린다 — 위치 모드(3) 복귀는 apply_one 이 한다.

    2026-11-03: 토크를 풀면 tilt 가 기계 하한(+109.85° ≈ 3298)까지 내려가 경계(3071) 너머에 선다. 위치 모드로 켜면 −798 로
    잡혀 ~317° 를 반대로 돈다. 확장 위치는 토크를 켜도 바퀴 수를 다시 맞추지 않고 위치 한계도 안 쓴다. 목표 = 토크를 켠 뒤
    읽은 자리 − 펼친 거리라서 Present 가 어느 바퀴로 세든(3298 이든 −798 이든) 같은 물리 자리다.
    순서는 apply_one 과 같다: 토크off → 모드(게인 리셋) → 게인 → 프로파일 → 토크on(Goal=Present, 제자리) → 목표.
    실패하면 토크는 켜진 채 모드 4 로 지금 자리를 잡는다. 다음 이동 요청의 bring_to 가 다시 검사해 모드 3 으로 돌린다.
    """
    name = config.names.get(dxl_id, f"id {dxl_id}")
    goal, limit = goal_tick(config, dxl_id), config.droop_limit_tick[dxl_id]
    controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
    controller.write1(dxl_id, ADDR_OPERATING_MODE, OP_MODE_EXTENDED_POSITION, "operating mode 4")
    controller.write2(dxl_id, ADDR_POSITION_I_GAIN, config.position_i_gain, "position i gain")
    controller.write4(dxl_id, ADDR_PROFILE_ACCELERATION, config.profile_acceleration, "accel")
    controller.write4(dxl_id, ADDR_PROFILE_VELOCITY, config.profile_velocity, "vel")
    controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON, "torque on")
    read4 = _read4(controller)
    present = read4(dxl_id, ADDR_PRESENT_POSITION)
    lift = physical_tick(present, limit) - goal  # 토크를 켠 뒤 자리로 다시 잰다(검사 뒤 고개가 움직였어도)
    if not 0 < lift < TICKS_PER_TURN // 2:
        controller.write4(dxl_id, ADDR_GOAL_POSITION, present, "lift stop")
        raise HeadReachError(f"{name}: 처짐 복구 — 토크를 켠 뒤 자리({present})에서 올릴 거리 {lift} 틱이 반 바퀴 안의 양수가"
                             f" 아니다. 토크는 켜진 채(모드 4) 지금 자리를 잡고 있다")
    goal4 = present - lift
    controller.write4(dxl_id, ADDR_GOAL_POSITION, goal4, "lift goal")
    deadline = time.monotonic() + LIFT_TIMEOUT_S
    try:
        while abs((present := read4(dxl_id, ADDR_PRESENT_POSITION)) - goal4) > LIFT_TOL_TICK:
            if time.monotonic() > deadline:
                raise HeadReachError(f"{name}: 처짐 복구 — 확장 위치로 {LIFT_TIMEOUT_S:g}s 안에 목표 {goal4} 에 닿지 못했다"
                                     f"(지금 {present}). 토크는 켜진 채(모드 4) 지금 자리를 잡고 있다")
            time.sleep(0.05)
    except Exception:
        try:
            controller.write4(dxl_id, ADDR_GOAL_POSITION, present, "lift stop")  # 막힌 채 계속 밀지 않게(마지막으로 읽은 자리)
        except Exception:  # noqa: BLE001 - 원래 오류를 올린다
            pass
        raise
    offset = read4(dxl_id, ADDR_HOMING_OFFSET)
    margin = (offset + TICKS_PER_TURN - 1 - torque_on_tick(present, offset)) * 360.0 / TICKS_PER_TURN
    if margin < LIFT_EDGE_MARGIN_DEG:  # 위치 모드로 돌아가며 토크를 끄는 순간 처져 경계를 넘으면 안 된다
        raise HeadReachError(f"{name}: 처짐 복구 — 올라온 자리가 경계까지 {margin:.1f}° 뿐이다(필요 "
                             f"{LIFT_EDGE_MARGIN_DEG:g}°). 토크는 켜진 채(모드 4) 둔다")


def bring_to(controller: CompliantController, config: HeadHome, dxl_ids, apply=apply_one) -> None:
    """모든 모터를 쓰기 전에 확인하고(처진 모터는 droop_ticks, 나머지는 check_reach), 처진 모터를 먼저 올린 뒤 모두 apply.

    head_home.py --execute 와 목 드라이버(head_driver_core) 가 같이 쓴다. 걸리면 아무것도 쓰지 않는다 — 한쪽만 움직이지 않게.
    """
    drooped = droop_ticks(controller, config, dxl_ids)
    check_reach(controller, config, tuple(i for i in dxl_ids if i not in drooped))
    for dxl_id, actual in drooped.items():
        print(f"  {config.names.get(dxl_id, dxl_id)}: 경계 너머로 처져 있다(실제 {actual} 틱) — 확장 위치로 짧은 길로 올린다")
        lift_droop(controller, config, dxl_id)
    for dxl_id in dxl_ids:
        apply(controller, config, dxl_id)


def verify(controller: CompliantController, config: HeadHome) -> bool:
    time.sleep(SETTLE_SECONDS)
    ok = True
    for dxl_id, target in config.targets_deg.items():
        samples = [tick_to_deg(controller.read_present_tick(dxl_id))
                   for _ in range(VERIFY_SAMPLES)]
        mean = statistics.fmean(samples)
        error = mean - target
        gain = controller.read2_signed(dxl_id, ADDR_POSITION_I_GAIN, "i")
        good = abs(error) <= VERIFY_TOLERANCE_DEG and gain == config.position_i_gain
        ok &= good
        print(f"  {config.names[dxl_id]}: 목표 {target:+.2f}° · 실제 {mean:+.3f}° · "
              f"오차 {error:+.3f}° · σ {statistics.pstdev(samples):.3f}° · "
              f"I={gain} · {'✓' if good else '❌'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pan", type=float, default=None, help="일회성 덮어쓰기")
    parser.add_argument("--tilt", type=float, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release", action="store_true", help="목 토크 해제만 수행(받침 확인 후 --execute)")
    args = parser.parse_args(argv)

    try:
        config = load_head_home(Path(args.config))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"❌ 설정을 읽을 수 없다: {exc}", file=sys.stderr)
        return 2

    if args.release:
        print(f"목 토크 해제 — {config.port} · 모터 {list(config.targets_deg)} (중력으로 처질 수 있다)")
        if not args.execute:
            print("--execute 없이 실행함 — 아무것도 쓰지 않았다")
            return 0
        controller = CompliantController(config.port, config.baud)
        try:
            for dxl_id in config.targets_deg:
                controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
            return 0
        finally:
            controller.close()

    overrides = {"pan": args.pan, "tilt": args.tilt}
    targets = dict(config.targets_deg)
    for dxl_id, name in config.names.items():
        if overrides.get(name) is not None:
            targets[dxl_id] = float(overrides[name])
    config = HeadHome(**{**config.__dict__, "targets_deg": targets})

    print(f"설정 {args.config}")
    print(f"포트 {config.port} @ {config.baud} · 모드 {config.operating_mode} · "
          f"Position I Gain {config.position_i_gain}")
    for dxl_id, target in config.targets_deg.items():
        bias = config.goal_bias_tick.get(dxl_id, 0)
        print(f"  {config.names[dxl_id]}(id={dxl_id}) → {target:+.2f}° "
              f"(tick {deg_to_tick(target)}" + (f" · 쓰는 목표 {goal_tick(config, dxl_id)} = 보정 {bias:+d}" if bias else "") + ")")
    print("순서: 토크off → 모드 → 게인 → 목표 → 토크on  "
          "(모드가 게인을 리셋하므로 뒤바꾸면 안 된다)")

    if not args.execute:
        print("\n--execute 없이 실행함 — 아무것도 쓰지 않았다")
        return 0

    controller = CompliantController(config.port, config.baud)
    try:
        for dxl_id in config.targets_deg:
            controller.ping(dxl_id)
        bring_to(controller, config, tuple(config.targets_deg))
        print(f"\n{SETTLE_SECONDS:g}s 정착 후 검증")
        ok = verify(controller, config)
        print("\n★Position I Gain 은 RAM 이다 — 전원을 끄면 사라진다. bringup 마다 실행할 것")
        return 0 if ok else 1
    except HeadReachError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        return 1
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
