#!/usr/bin/env python3
"""FP++ 전에 머리 자세를 카메라 캘리브 자세와 맞춘다 — 확인(기본, 읽기만) · 맞춤(--execute, 머리가 조금 움직인다).

카메라 외부 파라미터(config/global_camera_extrinsics_arm4090.yaml)는 머리 자세 하나에서만 맞는 정적 스냅샷이다. 그 자세(틱)를
같은 파일의 head_pose 에 적어 두고(scripts/calib/table_cad_extrinsics.py --write --head-config), 컵 · 홀더 자세를 재기 전에
지금 자세와 비교한다. 10.04 사용자: "fpp 진행 전에 자동으로 각도 확인하고 세팅을 제대로 맞춘 다음에 진행".

    python3 scripts/head_pose_check.py --config config/head_home_rh56f1.yaml \\
        --extrinsics config/global_camera_extrinsics_arm4090.yaml          # 읽기만 — 어긋나면 rc 1
    python3 scripts/head_pose_check.py ... --execute                       # 어긋나면 캘리브 자세로 맞춘 뒤 다시 확인
    python3 scripts/head_pose_check.py ... --execute --home                # 미션 head_home: 멀리 있어도 그 자세로(게인 · 토크도)

맞추는 법: 이 머리 pan 은 목표보다 18 틱(1.6°) 앞에서 멈춘다(10.04 실측: 목표 2015 → 2033, 그 전 1997 → 2015 — 낮은 유지력 ·
마찰). 목표를 그대로 보내면 캘리브 자세에 닿지 않으니 '목표 += 캘리브 자세 − 멈춘 자리'를 몇 번 되풀이한다. 목표는 캘리브
자세 ± MAX_GOAL_OFFSET 안에서만 옮긴다. I 게인은 RAM 이라 전원을 껐다 켜면 0 — 토크 · 모드 · I 게인이 설정과 다르면
head_home 순서(토크 off → 모드 → 게인 → 프로파일 → 토크 on → 목표)로 먼저 맞춘다.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOL_TICK = 4              # ±4 틱 = ±0.35° — 테이블(약 0.6 m)에서 약 3.5 mm
MAX_GOAL_OFFSET = 60      # 목표를 캘리브 자세에서 이 틱(5.3°) 넘게 옮기지 않는다
MAX_START_ERR = 120       # 지금 자세가 캘리브 자세에서 이 틱(10.5°) 넘게 벗어나 있으면 맞추지 않는다(head_home 먼저)
SETTLE_S = 1.5
MAX_ITER = 6
DEG_PER_TICK = 360.0 / 4096


@dataclass(frozen=True)
class HeadPose:
    ticks: dict[str, int]     # 모터 이름(pan · tilt) → 캘리브 때 멈춰 있던 틱
    tol_tick: int


def load_head_pose(extrinsics_yaml: Path) -> HeadPose | None:
    """외부 파라미터 파일의 head_pose — 없으면 None(그 캘리브는 머리 자세를 적지 않았다)."""
    raw = yaml.safe_load(Path(extrinsics_yaml).read_text(encoding="utf-8")) or {}
    hp = raw.get("head_pose")
    if not isinstance(hp, dict):
        return None
    ticks = {k[: -len("_tick")]: int(v) for k, v in hp.items() if k.endswith("_tick") and k != "tol_tick"}
    if not ticks:
        return None
    return HeadPose(ticks=ticks, tol_tick=int(hp.get("tol_tick", TOL_TICK)))


def next_goal(goal: int, present: int, target: int, max_offset: int = MAX_GOAL_OFFSET) -> int:
    """멈춘 자리의 오차만큼 목표를 옮긴다 — 캘리브 자세 ± max_offset 안으로. 순수."""
    g = int(goal) + (int(target) - int(present))
    return max(int(target) - max_offset, min(int(target) + max_offset, g))


def align(read_present: Callable[[int], int], read_goal: Callable[[int], int], write_goal: Callable[[int, int], None],
          targets: dict[int, int], tol: int, *, settle_s: float = SETTLE_S, max_iter: int = MAX_ITER,
          sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print) -> bool:
    """모든 모터가 targets(id → 틱) ± tol 안에 멈출 때까지 목표를 고쳐 보낸다. I/O 는 주입 — 시험용."""
    for it in range(max_iter + 1):
        present = {i: read_present(i) for i in targets}
        off = {i: present[i] - t for i, t in targets.items()}
        log(f"  [{it}] " + " · ".join(f"id {i}: 지금 {present[i]} · 차 {off[i]:+d}" for i in targets))
        if all(abs(d) <= tol for d in off.values()):
            return True
        if it == max_iter:
            break
        for i, t in targets.items():
            if abs(off[i]) > tol:
                write_goal(i, next_goal(read_goal(i), present[i], t))
        sleep(settle_s)
    return False


def read_head_ticks(config_path: Path) -> dict[str, int]:
    """머리 모터 현재 틱(이름 → 틱) — 포트를 열어 읽기만 한다."""
    from head_compliant_hold import CompliantController
    from head_home import load_head_home

    cfg = load_head_home(Path(config_path))
    c = CompliantController(cfg.port, cfg.baud)
    try:
        return {cfg.names[i]: c.read_present_tick(i) for i in cfg.targets_deg}
    finally:
        c.close()


def _report(now: dict[str, int], hp: HeadPose) -> bool:
    ok = True
    for name, t in hp.ticks.items():
        d = now[name] - t
        good = abs(d) <= hp.tol_tick
        ok &= good
        print(f"  {name}: 지금 {now[name]} · 캘리브 {t} · 차 {d:+d} 틱({d * DEG_PER_TICK:+.2f}°) {'✓' if good else '✗'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", type=Path, required=True, help="머리 설정(포트 · 게인 · 모터 id) — head_home_rh56f1.yaml")
    ap.add_argument("--extrinsics", type=Path, required=True, help="head_pose 가 든 카메라 외부 파라미터 yaml")
    ap.add_argument("--execute", action="store_true", help="어긋나면 캘리브 자세로 맞춘다(머리가 움직인다)")
    ap.add_argument("--home", action="store_true",
                    help="head_home 대신: 캘리브 자세에서 멀리 있어도 맞춘다(없으면 MAX_START_ERR 넘게 벗어나면 거부)")
    args = ap.parse_args(argv)

    hp = load_head_pose(args.extrinsics)
    if hp is None:
        print(f"✗ {args.extrinsics} 에 head_pose 가 없다 — 이 캘리브를 잰 머리 자세를 모른다. "
              "scripts/calib/table_cad_extrinsics.py --write --head-config <머리 설정> 으로 다시 캘리브한다")
        return 2
    from head_compliant_hold import ADDR_POSITION_I_GAIN, ADDR_TORQUE_ENABLE, CompliantController, tick_to_deg
    from head_home import HeadHome, apply_one, load_head_home
    from head_position_hold_node import ADDR_GOAL_POSITION, ADDR_OPERATING_MODE

    cfg = load_head_home(args.config)
    ids = {name: i for i, name in cfg.names.items()}
    missing = [n for n in hp.ticks if n not in ids]
    if missing:
        print(f"✗ head_pose 의 모터 {missing} 가 머리 설정에 없다")
        return 2
    print(f"[head] 캘리브 자세 = {args.extrinsics.name} head_pose · 허용 ±{hp.tol_tick} 틱(±{hp.tol_tick * DEG_PER_TICK:.2f}°)")
    c = CompliantController(cfg.port, cfg.baud)
    try:
        now = {name: c.read_present_tick(ids[name]) for name in hp.ticks}
        if _report(now, hp):
            print("[head] OK — 카메라 외부 파라미터가 맞는 자세다")
            return 0
        if not args.execute:
            print("[head] ✗ 캘리브 자세와 다르다 — --execute 로 맞추거나(머리가 움직인다), 이 자세에서 다시 캘리브한다")
            return 1
        far = [n for n in hp.ticks if abs(now[n] - hp.ticks[n]) > MAX_START_ERR]
        if far and not args.home:
            print(f"[head] ✗ {far} 가 캘리브 자세에서 {MAX_START_ERR} 틱 넘게 벗어나 있다 — head_home 을 먼저 돌린다")
            return 1
        targets = {ids[n]: t for n, t in hp.ticks.items()}
        for n, i in ids.items():                    # 전원을 껐다 켰으면 I 게인 0 · 토크 off — head_home 순서로 먼저
            if n not in hp.ticks:
                continue
            ready = (c.read1(i, ADDR_TORQUE_ENABLE, "torque") == 1
                     and c.read1(i, ADDR_OPERATING_MODE, "mode") == cfg.operating_mode
                     and c.read2_signed(i, ADDR_POSITION_I_GAIN, "i") == cfg.position_i_gain)
            if not ready:
                print(f"  {n}: 토크 · 모드 · I 게인이 설정과 다르다 — head_home 순서로 적용(목표 = 지금 자리, 옮기기는 다음에)")
                here = c.read_present_tick(i)
                one = HeadHome(**{**cfg.__dict__, "targets_deg": {**cfg.targets_deg, i: tick_to_deg(here)}})
                apply_one(c, one, i)
        time.sleep(SETTLE_S)
        ok = align(lambda i: c.read_present_tick(i), lambda i: c.read4_signed(i, ADDR_GOAL_POSITION, "goal"),
                   lambda i, g: c.write4(i, ADDR_GOAL_POSITION, int(g), "goal"), targets, hp.tol_tick)
        now = {name: c.read_present_tick(ids[name]) for name in hp.ticks}
        print("[head] 맞춘 뒤")
        ok = _report(now, hp) and ok
        print("[head] OK — 캘리브 자세로 맞췄다" if ok else
              "[head] ✗ 맞추지 못했다 — 머리 전원 · 걸림을 보고, 이 자세에서 다시 캘리브한다(table_cad_extrinsics.py)")
        return 0 if ok else 1
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
