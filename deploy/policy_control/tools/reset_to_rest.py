#!/usr/bin/env python3
"""실기 팔을 **저장 경로를 거꾸로 되짚어** 차렷(경로 시작 자세)으로 되돌린다. 실패한 자리에서 빠져나오는 길.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/reset_to_rest.py --side right            # 검사만
    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/reset_to_rest.py --side right --execute  # 실제 이동

09.23 사용자: "이 상태에서 원래 차렷 자세로 돌아가는 방법이 필요해". 그날 그 복귀를 손으로 밟은 절차 그대로다.

  ① 실측 팔·손을 읽는다
  ② 저장 경로에서 **지금과 가장 가까운 프레임**을 찾는다 — 경로 위 어디에서 멈췄든 거기서 되돌아온다
  ③ 지금 자세 → 그 프레임까지 진입 램프를 저장 경로와 같은 세계에서 충돌 검사한다.
     goto_home 직선으로 내려가지 않는 이유: 그것은 손이 지나는 곳을 계산하지 않는다(09.07 에 테이블로 내려갔다).
  ④ 통과하면 그 프레임까지 잘라낸 npz 를 쓰고 pd engage → `replay_to_pd --reverse` → release 를 부른다.
     진입 간극이 기본 한계(0.35 rad)보다 크면 ③을 통과한 만큼만 `--max-ramp-rad` 를 올려 넘긴다.
  ⑤ pd 가 이미 내려가 forward 컨트롤러만 남았으면 `--restore-controllers` 로 JTC 로 되돌린다(이동 없음).

검사만(기본)은 발행하지 않는다. `--execute` 가 있어야 로봇이 움직인다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM2REAL = HERE.parents[2]
SAMPLE = SIM2REAL / "deploy" / "s2r_console" / "tools" / "sample_joints.py"
PATHS = SIM2REAL / "deploy" / "policy_control" / "paths"
RAMP_HEADROOM = 0.05            # [rad] 검사에 통과한 간극 위로 두는 여유 — 재생 직전 미세 변화를 덮는다


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def nearest_frame(frames: np.ndarray, now: np.ndarray) -> tuple[int, float]:
    """저장 경로에서 지금과 가장 가까운 프레임 (index, L-inf 거리). 순수.

    경로 위 어디서 멈췄든 그 지점부터 되짚으면 지나온 공간만 지난다 — 되돌아가는 길은 그것뿐이다."""
    dists = np.abs(np.asarray(frames, dtype=float) - np.asarray(now, dtype=float)).max(axis=1)
    k = int(np.argmin(dists))
    return k, float(dists[k])


def measure() -> dict[str, float]:
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.6"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"✗ 관절 상태를 못 읽었다: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])["q"]


def run(argv: list[str], what: str) -> None:
    print(f"\n$ {' '.join(argv)}", flush=True)
    rc = subprocess.run(argv, cwd=str(SIM2REAL)).returncode
    if rc != 0:
        raise SystemExit(f"✗ {what} 실패 (rc={rc}) — 멈춘다")


def restore_controllers(side: str, execute: bool) -> int:
    """pd 가 없는데 forward 컨트롤러만 active 로 남은 상태를 JTC 로 되돌린다(자세는 그대로)."""
    fwd = [f"{side}_forward_{k}_controller" for k in ("position", "velocity", "effort")]
    jtc = f"{side}_joint_trajectory_controller"
    print(f"[reset] 제어 되돌리기: {jtc} 활성 · {', '.join(fwd)} 비활성")
    if not execute:
        print("[reset] 검사만 — --execute 를 붙이면 전환한다")
        return 0
    run(["ros2", "control", "switch_controllers", "--activate", jtc, "--deactivate", *fwd], "컨트롤러 전환")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--npz", type=Path, default=None, help="저장 홈 경로(기본 deploy/policy_control/paths/home_<side>.npz)")
    ap.add_argument("--execute", action="store_true", help="실제로 움직인다 — 없으면 검사만")
    ap.add_argument("--margin", type=float, default=0.02, help="진입 램프 최소 여유 [m]")
    ap.add_argument("--max-gap", type=float, default=0.6,
                    help="경로에서 이만큼 넘게 벗어나 있으면 멈춘다 [rad] — 사람이 볼 일이다")
    ap.add_argument("--at-rest-tol", type=float, default=0.05, help="이 안이면 이미 차렷 [rad]")
    ap.add_argument("--hold-s", type=float, default=3.0, help="engage 뒤 제자리 유지 [s]")
    ap.add_argument("--restore-controllers", action="store_true",
                    help="이동 없이 컨트롤러만 JTC 로 되돌린다(pd 가 죽은 뒤 정리)")
    args = ap.parse_args()

    if args.restore_controllers:
        return restore_controllers(args.side, args.execute)

    npz = args.npz or (PATHS / f"home_{args.side}.npz")
    if not npz.is_file():
        raise SystemExit(f"✗ 저장 경로가 없다: {npz}")
    P = _load("plan_home_path")
    W = P.W
    d = np.load(npz)
    joints = [str(j) for j in d["meta_joints"]]
    goal, start = np.asarray(d["meta_goal"], float), np.asarray(d["meta_start"], float)
    q = measure()
    missing = [j for j in joints if j not in q]
    if missing:
        raise SystemExit(f"✗ 관절 상태 없음 {missing} — 드라이버가 떠 있는가")
    now = np.array([q[j] for j in joints])
    frames = np.asarray(d["arm_target"], dtype=float)
    k, gap = nearest_frame(frames, now)
    print(f"[reset] {args.side} 지금 {np.round(now, 4).tolist()}")
    print(f"[reset] 저장 경로에서 가장 가까운 프레임 {k}/{len(frames) - 1} · 거리 {gap:.4f} rad "
          f"(끝까지 {float(np.abs(now - goal).max()):.4f} · 시작까지 {float(np.abs(now - start).max()):.4f})")
    if gap > args.max_gap:
        raise SystemExit(f"✗ 저장 경로에서 {gap:.3f} rad 벗어나 있다(한계 {args.max_gap}) — 되짚기로는 못 돌아온다. "
                         "사람이 자세를 보고 판단할 것")
    if float(np.abs(now - start).max()) <= args.at_rest_tol:
        print(f"[reset] 이미 차렷 안({args.at_rest_tol} rad) — 움직일 것이 없다")
        return 0
    target = frames[k]

    hand_q = {j: float(v) for j, v in q.items() if j.startswith(("r_hj_", "l_hj_"))}
    contract = P.load_contract(W.CONTRACT_DEFAULT)
    world = W.build_world(W.WorldSpec(side=args.side, detect_margin=max(0.08, args.margin * 3)))
    limits = W.load_profile_limits(W.PROFILE_DEFAULT)
    lo = np.array([limits[j][0] for j in world.moving_joints])
    hi = np.array([limits[j][1] for j in world.moving_joints])
    scenes = P.build_scenes(contract, args.side, str(d["meta_other_arm"]), "measured", hand_q)
    chk = P.Checker(world, scenes, args.margin, now, (lo, hi))
    rep = chk.check_path(np.stack([now, target]))
    P._print_check(f"진입 램프(지금 → 프레임 {k})", rep)
    if not rep["ok"]:
        print("✗ 진입 램프가 충돌 · 여유 미달 — 팔을 움직이지 말 것. 사람이 보고 판단할 것", file=sys.stderr)
        return 1
    print(f"[reset] 진입 램프 통과 · 최소 여유 {rep['min_clear']:.4f} m")
    if not args.execute:
        print("[reset] 검사만 — --execute 를 붙이면 engage → 되짚기 → release 를 실행한다")
        return 0

    cut = SIM2REAL / "logs" / "policy_control" / f"reset_{args.side}.npz"      # 그 프레임까지 잘라낸 경로
    cut.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cut, arm_target=frames[:k + 1], meta_step_dt=d["meta_step_dt"], meta_joints=d["meta_joints"],
             meta_start=start, meta_goal=target, meta_method=np.array("reset"),
             meta_contract_sha1=d["meta_contract_sha1"])
    print(f"[reset] 되짚을 구간 {cut} · 프레임 {k + 1} × {float(d['meta_step_dt'])} = "
          f"{k * float(d['meta_step_dt']):.1f} s")
    ctl = str(HERE / "episode_ctl.py")
    run([sys.executable, ctl, "--only", "pd_engage", "--hold-s", str(args.hold_s),
         "--execute", "--approve", "pd_engage"], "pd engage")
    run([sys.executable, str(HERE / "replay_to_pd.py"), "--npz", str(cut), "--joints", ",".join(joints),
         "--rate-scale", "1.0", "--reverse", "--max-ramp-rad", f"{gap + RAMP_HEADROOM:.3f}", "--execute"],
        "되짚기 재생")
    run([sys.executable, ctl, "--only", "pd_release", "--execute", "--approve", "pd_release"], "pd release")

    back = measure()
    err = float(np.abs(np.array([back[j] for j in joints]) - start).max())
    print(f"\n[reset] 완료 · 시작 자세와 최대 오차 {err:.4f} rad")
    return 0 if err < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
