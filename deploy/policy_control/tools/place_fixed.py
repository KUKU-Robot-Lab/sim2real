#!/usr/bin/env python3
"""고정 놓기 궤적 — 경로 F 로 잡고 든 뒤 control tilt_now 로 tilt 시작 자세에 온 지금 자세에서 쥔 컵을 홀더에 놓는다.
상황판 control 탭 [place_fixed_<컵>_<끝>_<팔>] 이 부른다(tilt_now 처럼). ★11.18 사용자.
tilt 끝 자세에서만 누른다. 하강까지 · 들기까지 모두 놓는 팔이 명목 시작 자세에서 3° 넘게 벗어나면 계획이 거부한다(확인용).

  1. 지금 두 팔 관절(실측, plan_tilt.current_q) · 그 팔 pd 가 지금 쓰는 손 명령(/policy_control/pd_<팔>/applied 의 손 6 —
     terminal 뒤 pd 가 붙든 정책의 마지막 손 목표, rad) → <out>/q.json · hand.json.
     손 실측(/hand_<팔>/joint_states)은 접촉 때문에 명령보다 덜 닫혀 있어 쓰지 않는다. applied 에 손이 없으면 rc 4(추측하지 않는다).
  2. plan_place_fixed.py --cup <컵> --q-json q.json --hand-grip <손 명령> --out <out>/plan — 걸리면 rc 그대로 · 움직이지 않음.
  3. replay_traj_rh.py --npz <out>/plan/place_cmd.npz [--until descend] --hand-start applied — --execute 없으면 드라이런만.
     rate_scale = 계획이 낸 reduced 단계 허용 최대(내림) · 시작 간격 0.05 rad.
기록: logs/runs/<YYMMDD>/place_fixed/<컵>_<HHMMSS>/(meta.json · q.json · hand.json · plan/).

    python3 deploy/policy_control/tools/place_fixed.py --cup blue --until descend            # 계획 · 검사 · 드라이런
    python3 deploy/policy_control/tools/place_fixed.py --cup blue --until all --execute      # ★실기 — 사용자 승인 뒤
종료 코드: 0 = 끝 · 2 = 검사 실패 · 3 = 핀 · 입력 불일치 · 4 = pd 손 명령 없음 · 그 밖 = 재생 실패.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import yaml

from tilt_pour import TOOLS, run, write_meta, run_paths
from plan_place_fixed import CFG_DIR, CUPS

APPLIED = "/policy_control/pd_{side}/applied"
HAND_WAIT_S = 3.0
START_GAP_RAD = 0.05


def pick_hand(names, positions, want: tuple[str, ...]) -> list[float] | None:
    """applied 메시지(name · position)에서 want 손 관절을 그 순서로. 하나라도 없으면 None."""
    got = dict(zip(names, positions))
    return [float(got[n]) for n in want] if all(n in got for n in want) else None


def pd_hand(side: str, names: tuple[str, ...]) -> list[float] | None:
    """pd 가 지금 손에 쓰는 명령(applied 의 손 이름들). pd 가 RAMPING/TRACKING 이 아니면 applied 에 손이 없다 → None."""
    import rclpy
    from sensor_msgs.msg import JointState

    hand: list = [None]
    rclpy.init()
    try:
        node = rclpy.create_node("place_fixed_hand")
        try:
            node.create_subscription(JointState, APPLIED.format(side=side),
                                     lambda m: hand.__setitem__(0, pick_hand(m.name, m.position, names) or hand[0]), 10)
            t0 = time.time()
            while time.time() - t0 < HAND_WAIT_S and hand[0] is None:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
    return hand[0]


def failed_checks(plan_out: Path) -> str:
    try:
        chk = json.loads((plan_out / "check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "(check.json 없음)"
    return " · ".join(f"{k} {c.get('value')} (한계 {c.get('limit')})" for k, c in chk.items() if not c.get("ok", True)) or "(없음)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cup", choices=tuple(CUPS), required=True)
    ap.add_argument("--until", choices=("descend", "all"), required=True, help="descend = 하강 끝(손 쥔 채)까지 · all = 들기까지")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    from plan_tilt import current_q
    from policy_control import place_teach as PT

    side = yaml.safe_load((CFG_DIR / f"{CUPS[a.cup]}.yaml").read_text(encoding="utf-8"))["side"]
    out = run_paths.unique_dir(run_paths.day_dir("place_fixed") / f"{a.cup}_{time.strftime(run_paths.TIME_FMT)}")
    write_meta(out, tool="place_fixed", cup=a.cup, side=side, until=a.until, execute=a.execute)
    q = current_q("")
    hand = pd_hand(side, PT.hand_names(side))
    if hand is None:
        print(f"[place_fixed] ✗ {APPLIED.format(side=side)} 에 손 명령이 {HAND_WAIT_S:.0f} s 안에 없다 — pd {side} 가 TRACKING 이 아니거나 "
              f"손 그룹이 없다. 손 실측으로 대신하지 않는다(컵을 놓칠 수 있다) · 움직이지 않았다 · {out}", flush=True)
        return 4
    (out / "q.json").write_text(json.dumps(q))
    (out / "hand.json").write_text(json.dumps({"side": side, "names": list(PT.hand_names(side)), "q": hand,
                                               "source": APPLIED.format(side=side)}))
    print(f"[place_fixed] {a.cup} · {side} · 손 명령 {[round(v, 3) for v in hand]} · {out}", flush=True)

    plan_out = out / "plan"
    rc = run([sys.executable, str(TOOLS / "plan_place_fixed.py"), "--cup", a.cup, "--q-json", str(out / "q.json"),
              "--hand-grip", ",".join(repr(v) for v in hand), "--out", str(plan_out)])
    if rc != 0:
        print(f"[place_fixed] 계획 · 검사 rc {rc} — 움직이지 않았다 · 걸린 항목: {failed_checks(plan_out)} · {plan_out}", flush=True)
        return rc
    meta = json.loads((plan_out / "meta.json").read_text(encoding="utf-8"))
    rate = math.floor(float(meta["replay_max_rate_scale_reduced"]) * 100 - 1e-9) / 100   # 부동소수 오차로 한계를 넘지 않게 내림
    if rate <= 0:
        print(f"[place_fixed] ✗ 재생 허용 rate_scale {meta['replay_max_rate_scale_reduced']} — 움직이지 않았다", flush=True)
        return 2
    argv_replay = [sys.executable, str(TOOLS / "replay_traj_rh.py"), "--npz", str(plan_out / "place_cmd.npz"),
                   "--rate-scale", repr(rate), "--max-start-gap", repr(START_GAP_RAD), "--hand-start", "applied"]
    if a.until != "all":
        argv_replay += ["--until", a.until]
    rc = run(argv_replay + (["--execute"] if a.execute else []))
    print(f"[place_fixed] {'재생' if a.execute else '드라이런(--execute 로 재생)'} rc {rc} · {out}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
