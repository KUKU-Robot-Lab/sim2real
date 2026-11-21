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
★11.18 콘솔 고정 놓기 튜닝 탭(s2r_console/place_tune.py) — --until 대신 하나:
  --jog dx,dy,dz,dyaw                  하강 끝에서 손 쥔 채 조금(plan_place_fixed --jog) → place_jog_exec(board_jog precise 방식:
                                       120 Hz 5차 ≤ 0.30 rad/s · 유지 보정 c = pd 세트포인트 − 정지 실측)
  --save-offset                        지금 손 ↔ 하강 끝 계획 차이를 설정 offset 에 더한다(움직이지 않음 · 손 명령 안 읽음)
  --resume                             지금 자세에서 release → lift → hold(plan_place_fixed --resume)
  --out <폴더>                         기록 폴더를 정해 준다(콘솔이 결과 · plan/ 을 읽는다). 이미 있으면 rc 3.
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
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--until", choices=("descend", "all"), help="descend = 하강 끝(손 쥔 채)까지 · all = 들기까지")
    m.add_argument("--jog", default=None, help="dx,dy,dz,dyaw — 홀더 좌표 mm · 컵 축 둘레 deg(손 쥔 채)")
    m.add_argument("--save-offset", action="store_true")
    m.add_argument("--resume", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    mode = a.until or ("jog" if a.jog is not None else "save" if a.save_offset else "resume")
    from plan_tilt import current_q
    from policy_control import place_teach as PT

    side = yaml.safe_load((CFG_DIR / f"{CUPS[a.cup]}.yaml").read_text(encoding="utf-8"))["side"]
    if a.out is not None:
        try:
            a.out.mkdir(parents=True)
        except FileExistsError:
            print(f"[place_fixed] ✗ --out {a.out} 가 이미 있다 — 움직이지 않았다", flush=True)
            return 3
        out = a.out
    else:
        out = run_paths.unique_dir(run_paths.day_dir("place_fixed") / f"{a.cup}_{time.strftime(run_paths.TIME_FMT)}")
    write_meta(out, tool="place_fixed", cup=a.cup, side=side, until=a.until, execute=a.execute, mode=mode)
    q = current_q("")
    if mode == "save":                       # 움직이지 않는다 — 손 명령 · 재생 없이 계산 · 저장만
        (out / "q.json").write_text(json.dumps(q))
        rc = run([sys.executable, str(TOOLS / "plan_place_fixed.py"), "--cup", a.cup, "--q-json", str(out / "q.json"),
                  "--save-offset", "--out", str(out / "plan")])
        print(f"[place_fixed] 오프셋 저장 rc {rc} · {out / 'plan' / 'offset.json'}", flush=True)
        return result(out, rc, mode, a)
    hand = pd_hand(side, PT.hand_names(side))
    if hand is None:
        print(f"[place_fixed] ✗ {APPLIED.format(side=side)} 에 손 명령이 {HAND_WAIT_S:.0f} s 안에 없다 — pd {side} 가 TRACKING 이 아니거나 "
              f"손 그룹이 없다. 손 실측으로 대신하지 않는다(컵을 놓칠 수 있다) · 움직이지 않았다 · {out}", flush=True)
        return result(out, 4, mode, a)
    (out / "q.json").write_text(json.dumps(q))
    (out / "hand.json").write_text(json.dumps({"side": side, "names": list(PT.hand_names(side)), "q": hand,
                                               "source": APPLIED.format(side=side)}))
    print(f"[place_fixed] {a.cup} · {side} · 손 명령 {[round(v, 3) for v in hand]} · {out}", flush=True)

    plan_out = out / "plan"
    extra = (["--resume"] if mode == "resume" else ["--jog", a.jog] if mode == "jog" else [])
    rc = run([sys.executable, str(TOOLS / "plan_place_fixed.py"), "--cup", a.cup, "--q-json", str(out / "q.json"),
              "--hand-grip", ",".join(repr(v) for v in hand), "--out", str(plan_out), *extra])
    if rc != 0:
        print(f"[place_fixed] 계획 · 검사 rc {rc} — 움직이지 않았다 · 걸린 항목: {failed_checks(plan_out)} · {plan_out}", flush=True)
        return result(out, rc, mode, a)
    if mode == "jog":                        # 계획 끝 관절로 precise 실행(120 Hz 5차 · 유지 보정) — replay 는 1 mm 를 pd 오차 안에서 잃었다
        rc = run([sys.executable, str(TOOLS / "place_jog_exec.py"), "--npz", str(plan_out / "place_cmd.npz"),
                  "--max-start-gap", repr(START_GAP_RAD)] + (["--execute"] if a.execute else []))
        print(f"[place_fixed] jog {'실행' if a.execute else '드라이런(--execute 로 실행)'} rc {rc} · {out}", flush=True)
        return result(out, rc, mode, a)
    try:
        meta = json.loads((plan_out / "meta.json").read_text(encoding="utf-8"))
        rate = math.floor(float(meta["replay_max_rate_scale_reduced"]) * 100 - 1e-9) / 100
    except (OSError, ValueError, KeyError, TypeError) as exc:   # 계획이 rc 0 인데 meta 를 못 읽으면 움직이지 않는다
        print(f"[place_fixed] ✗ {plan_out / 'meta.json'} 읽기 실패 {type(exc).__name__}: {exc} — 움직이지 않았다", flush=True)
        return result(out, 3, mode, a)   # 부동소수 오차로 한계를 넘지 않게 내림
    if rate <= 0:
        print(f"[place_fixed] ✗ 재생 허용 rate_scale {meta['replay_max_rate_scale_reduced']} — 움직이지 않았다", flush=True)
        return result(out, 2, mode, a)
    argv_replay = [sys.executable, str(TOOLS / "replay_traj_rh.py"), "--npz", str(plan_out / "place_cmd.npz"),
                   "--rate-scale", repr(rate), "--max-start-gap", repr(START_GAP_RAD), "--hand-start", "applied"]
    if a.until == "descend":
        argv_replay += ["--until", a.until]
    rc = run(argv_replay + (["--execute"] if a.execute else []))
    print(f"[place_fixed] {'재생' if a.execute else '드라이런(--execute 로 재생)'} rc {rc} · {out}", flush=True)
    return result(out, rc, mode, a)


def result(out: Path, rc: int, mode: str, a) -> int:
    """끝난 런의 rc 를 남긴다 — 콘솔 튜닝 탭이 마지막 결과를 여기서 읽는다."""
    (out / "result.json").write_text(json.dumps({"rc": rc, "mode": mode, "cup": a.cup, "execute": a.execute, "jog": a.jog,
                                                 "time": time.time()}), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
