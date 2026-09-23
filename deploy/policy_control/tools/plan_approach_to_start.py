#!/usr/bin/env python3
"""지금 자세 → 저장 홈 경로의 시작점까지 짧은 접근 구간을 만들고 충돌 검사한다. 구독만 한다(발행 없음).

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/plan_approach_to_start.py \
        --npz deploy/policy_control/paths/home_right.npz

09.23 사용자: "실기 세팅이 trajectory 시작과 다르면 먼저 자세를 맞추고 시작하게 만들면 되는 것 아니냐" — 맞다.
저장 경로는 고정해 두고(다시 계획하지 않는다), 시작점까지만 이 접근 구간으로 맞춘다.

  · 실측 팔 · 손을 읽는다(손은 그대로 두고 팔만 움직인다 — pd home_hand: keep)
  · 지금 자세 → 경로 시작점 **관절공간 직선**을 저장 경로와 같은 세계(테이블 · 몸통 · 반대 팔, 여유 2 cm)에서 검사한다
  · 통과하면 0.1 rad/s 로 시간을 붙여 <out>(기본 logs/policy_control/approach_<side>.npz)에 쓴다 — replay_to_pd 가 재생한다
  · 이미 시작점 안(--tol)이어도 그 짧은 정렬을 그대로 쓴다 — 재생 한 번으로 늘 시작점에 선다
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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def measure() -> dict[str, float]:
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.6"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"✗ 관절 상태를 못 읽었다: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])["q"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", type=Path, required=True, help="저장 홈 경로(시작점을 여기서 읽는다)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=0.05, help="이 안이면 접근이 필요 없다 [rad]")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--max-rad", type=float, default=0.6, help="접근 구간 최대 관절 이동 [rad] — 넘으면 사람이 볼 일이다")
    args = ap.parse_args()

    P = _load("plan_home_path")
    W = P.W
    d = np.load(args.npz)
    joints = [str(j) for j in d["meta_joints"]]
    side = "right" if joints[0].startswith("r_") else "left"
    start = np.asarray(d["meta_start"], dtype=float)
    q = measure()
    missing = [j for j in joints if j not in q]
    if missing:
        raise SystemExit(f"✗ 관절 상태 없음 {missing} — 드라이버가 떠 있는가")
    now = np.array([q[j] for j in joints])
    err = np.abs(now - start)
    print(f"[approach] 지금 {np.round(now, 4).tolist()}")
    print(f"[approach] 시작점 {np.round(start, 4).tolist()} · 최대 차이 {err.max():.4f} rad ({joints[int(np.argmax(err))]})")
    if err.max() <= args.tol:
        print("[approach] 이미 시작점 안 — 짧은 정렬만 만든다")
    if err.max() > args.max_rad:
        raise SystemExit(f"✗ 시작점까지 {err.max():.3f} rad — {args.max_rad} 를 넘는다. 자세를 눈으로 확인하고 경로를 다시 만들 것")

    hand_q = {k: float(v) for k, v in q.items() if k.startswith(("r_hj_", "l_hj_"))}
    contract = P.load_contract(W.CONTRACT_DEFAULT)
    world = W.build_world(W.WorldSpec(urdf=W.URDF_DEFAULT, env_yaml=W.ENV_YAML_DEFAULT, side=side,
                                      with_cup=False, detect_margin=max(0.08, args.margin * 3)))
    limits = W.load_profile_limits(W.PROFILE_DEFAULT)
    lo = np.array([limits[j][0] for j in world.moving_joints])
    hi = np.array([limits[j][1] for j in world.moving_joints])
    scenes = P.build_scenes(contract, side, "both", "measured", hand_q)
    chk = P.Checker(world, scenes, args.margin, now, (lo, hi), P.ESCAPE_RADIUS, 0)
    path = np.stack([now, start])
    rep = chk.check_path(path)
    P._print_check("접근 구간(지금 → 시작점)", rep)
    if not rep["ok"]:
        print("✗ 접근 구간이 충돌 · 여유 미달 — 팔을 움직이지 말 것", file=sys.stderr)
        return 1
    vmax = P.read_ramp_speed(P.PD_CONFIG_DEFAULT)
    frames = P.time_parametrize(path, vmax, P.FRAME_DT, P.RAMP_TIME)
    frames_rep = chk.check_path(frames)
    if not frames_rep["ok"]:
        print("✗ 시간 매개화 프레임 검사 실패", file=sys.stderr)
        return 1
    out = args.out or (SIM2REAL / "logs" / "policy_control" / f"approach_{side}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, arm_target=frames.astype(np.float64), meta_step_dt=np.float64(P.FRAME_DT),
             meta_joints=np.array(joints), meta_start=now, meta_goal=start, meta_method=np.array("approach"),
             meta_min_clearance=np.float64(frames_rep["min_clear"]),
             meta_min_clearance_non_escape=np.float64(frames_rep["min_clear_ne"]),
             meta_contract_sha1=np.array(W.sha1_of(W.CONTRACT_DEFAULT)))
    print(f"[approach] 저장 {out} · {len(frames)} 프레임 × {P.FRAME_DT} = {(len(frames) - 1) * P.FRAME_DT:.1f} s "
          f"· 최소 여유 {frames_rep['min_clear_ne']:.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
