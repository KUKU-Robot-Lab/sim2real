#!/usr/bin/env python3
"""손을 **저장 홈 경로가 검사한 자세**로 맞춘다 — 검사하고 나서 보낸다. 팔은 움직이지 않는다.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/hand_to_path_pose.py --side right            # 검사만
    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/hand_to_path_pose.py --side right --execute  # 실제 이동

09.23 사용자: "어떤 hand 자세든 trajectory 시작 자세로 세팅을 맞추게 하는 거지?" — 그렇다. 손 전원을 껐다 켜면
손가락이 다른 자세로 자리 잡아 경로 시작점 검사(check_path_start)가 막는다. 저장 경로를 다시 계획하는 대신
**손을 그 모양으로 되돌린다.** 기준 자세는 pd yaml `hand_path_pose` 하나이고 경로 계획도 같은 값을 읽는다
(`plan_home_path --hand-start pd`) — 둘이 어긋날 수 없다.

09.23 사용자(봉투): "큰 구를 달아논 상태에서 팔을 움직이게 하고, 그다음에 손가락을 피면 되잖아?" —
`--sphere-npz <경로>` 를 주면 그 경로를 계획한 **봉투 구** 안에 손이 들어가는지만 본다. 자세를 정확히
맞추지 않아도 되고, 이미 구 안이면 손을 아예 보내지 않는다. 손가락을 펴는 것은 팔이 도착한 뒤 pd/hand_home 이 한다.

  ① pd 설정에서 기준 자세를, 로봇에서 지금 팔·손을 읽는다
  ② **지금 손 → 기준 손** 을 지금 팔 자세에서 충돌 검사한다(손가락이 상판 모서리를 지날 수 있다)
  ③ 통과하면 `episode_ctl --only pd_hand_path` 로 보낸다(pd 가 속도 제한 램프로 간다)
  ④ 도착을 실측으로 확인한다 — 못 닿으면 실패로 돌려준다(막힌 것이다)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SIM2REAL = HERE.parents[2]
SAMPLE = SIM2REAL / "deploy" / "s2r_console" / "tools" / "sample_joints.py"
PD_CONFIG = SIM2REAL / "deploy" / "policy_control" / "config" / "pd_dg5f_m_short.yaml"
STEPS = 12                      # 손 이동 구간을 몇 자세로 나눠 볼 것인가
SETTLE_S = 4.0                  # 지령 뒤 손이 갈 시간 — pd hand max_vel 1.0 rad/s
REACH_TOL = 0.15                # [rad] 도착 판정
#: 끝점이 이미 좁은 쌍은 그 값보다 이만큼까지 더 가까워져도 둔다 [m]. 관통(0)은 어떤 경우에도 넘지 않는다.
#: 09.23 실기: 차렷에서 새끼손가락이 받침판 가장자리에 겹쳐 있어(모델이 보수적) 3 mm 접근에 막혔다.
SWEEP_DIP = 0.005
#: **이미 닿아 있는** 쌍(끝점에서 관통)이 도중에 더 깊어져도 두는 한도 [m].
#: 09.23 실기: 차렷에서 편 손가락이 받침판(z 0) 위에 얹혀 있다 — 주먹으로 말면 멀어지지만 도중에 몇 mm 내려간다.
#: 손 힘은 PID p 1.5 에 lead clamp 0.2 rad 로 묶여 있어 걸리면 지령이 먼저 멈춘다.
TOUCH_SLACK = 0.010


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


def sweep(now: dict[str, float], want: dict[str, float], steps: int) -> list[dict[str, float]]:
    """지금 손 → 기준 손 관절공간 직선을 steps 자세로. 순수."""
    return [{j: now.get(j, want[j]) + (want[j] - now.get(j, want[j])) * t
             for j in want} for t in np.linspace(0.0, 1.0, steps + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--pd-config", type=Path, default=PD_CONFIG)
    ap.add_argument("--execute", action="store_true", help="실제로 손을 보낸다 — 없으면 검사만")
    ap.add_argument("--sphere-npz", type=Path, default=None,
                    help="손 봉투 구로 계획한 경로 npz — 그 반지름(meta_hand_sphere) 안에 들어가면 자세를 정확히 "
                         "맞출 필요가 없다. 이미 구 안이면 손을 보내지 않고, 보낸 뒤에도 구 안이면 통과로 본다")
    ap.add_argument("--margin", type=float, default=0.02, help="검사 최소 여유 [m]")
    ap.add_argument("--tol", type=float, default=REACH_TOL, help="도착 판정 [rad]")
    ap.add_argument("--settle-s", type=float, default=SETTLE_S)
    args = ap.parse_args()

    pose = (yaml.safe_load(args.pd_config.read_text()).get("hand_path_pose") or {}).get(args.side)
    if not pose:
        raise SystemExit(f"✗ {args.pd_config} 에 hand_path_pose.{args.side} 가 없다")
    want = {str(j): float(v) for j, v in pose.items()}
    q = measure()
    missing = [j for j in want if j not in q]
    if missing:
        raise SystemExit(f"✗ 손 관절 상태 없음 {missing[:4]} — 손 드라이버가 떠 있는가")
    radius = None
    if args.sphere_npz is not None:
        d = np.load(args.sphere_npz)
        if "meta_hand_sphere" not in d or float(d["meta_hand_sphere"]) != float(d["meta_hand_sphere"]):
            raise SystemExit(f"✗ {args.sphere_npz.name} 은 손 봉투 구로 계획한 경로가 아니다")
        radius = float(d["meta_hand_sphere"])

    def in_sphere(now: dict[str, float]) -> tuple[bool, str]:
        P0 = _load("home_path_world")
        lk, jt, _ = P0.parse_urdf(P0.URDF_DEFAULT)
        hand = {j: float(v) for j, v in now.items() if f"{args.side[0]}_hj_" in j}
        r, worst = P0.hand_radius(lk, jt, hand, args.side)
        return r <= radius, f"{r * 100:.1f} / {radius * 100:.1f} cm ({worst})"

    if radius is not None:
        ok, how = in_sphere(q)
        if ok:
            print(f"[hand] 손 봉투 {how} — 구 안이다. 손을 보내지 않는다(팔 경로는 이 구로 검사됐다)")
            return 0
        print(f"[hand] 손 봉투 {how} — 구를 넘는다. 기준 자세로 오므린다")

    err = {j: q[j] - want[j] for j in want}
    worst = max(err, key=lambda j: abs(err[j]))
    print(f"[hand] {args.side} 기준 자세와 최대 차이 {abs(err[worst]):.3f} rad ({worst})")

    P = _load("plan_home_path")
    W = P.W
    arm = [f"{args.side[0]}_aj_{i}" for i in range(1, 8)]
    now_arm = np.array([q[j] for j in arm])
    contract = P.load_contract(W.CONTRACT_DEFAULT)
    world = W.build_world(W.WorldSpec(side=args.side, detect_margin=max(0.08, args.margin * 3)))
    limits = W.load_profile_limits(W.PROFILE_DEFAULT)
    lo = np.array([limits[j][0] for j in world.moving_joints])
    hi = np.array([limits[j][1] for j in world.moving_joints])
    poses = sweep({j: q[j] for j in want}, want, STEPS)
    scenes = [P.build_scenes(contract, args.side, "both", "measured", h) for h in poses]
    #: 검사기는 **한 번만** 만든다 — 만들 때마다 구조적 자기충돌 제외(r_al_5<->r_al_7)가 초기화된다(09.23 fake 가 잡았다).
    chk = P.Checker(world, scenes[0], args.margin, now_arm, (lo, hi))

    def dists(sc) -> dict:
        chk.scenes = sc
        return chk.raw(now_arm)

    #: 차렷에서 엄지는 원래 상판에서 2 cm 안쪽이다 — 저장 경로도 그 쌍을 탈출 규칙으로 인정한다.
    #: 그래서 **양 끝보다 가까워지지 않는 것**을 기준으로 삼는다(절대 2 cm 가 아니라).
    d0, dn = dists(scenes[0]), dists(scenes[-1])
    floor = {k: min(args.margin, d0.get(k, np.inf), dn.get(k, np.inf))
             for k in set(d0) | set(dn)}
    tight = sorted((k for k in floor if floor[k] < args.margin - 1e-9), key=lambda k: floor[k])
    if tight:
        print("[hand] 끝점이 이미 좁은 쌍(그 값을 하한으로 쓴다): "
              + ", ".join(f"{P.fmt_pair(k)} {floor[k]:.4f}" for k in tight[:4]))
    worst_clear, worst_pair = np.inf, ("", "")
    for i, sc in enumerate(scenes):
        for k, d in dists(sc).items():
            need = min(chk.required(k), floor.get(k, args.margin))
            if 0.0 < need < args.margin:              # 끝점이 이미 좁다 — 조금 더 가까워지는 것은 두되 관통은 막는다
                need = max(0.0, need - SWEEP_DIP)
            elif need <= 0.0:                         # 이미 닿아 있다 — 더 깊어지는 것을 한도 안에서만 둔다
                need -= TOUCH_SLACK
            if need > 0 and d < worst_clear:
                worst_clear, worst_pair = d, k
            if d < need - 1e-6:
                print(f"✗ 손 이동 {i}/{STEPS} 자세가 충돌 · 여유 미달 — {P.fmt_pair(k)} {d:.4f} m "
                      f"(하한 {need:.4f}). 손을 보내지 않는다", file=sys.stderr)
                return 1
    print(f"[hand] 손 이동 구간 {STEPS + 1} 자세 통과 · 최소 여유 {worst_clear:.4f} m ({P.fmt_pair(worst_pair)})")
    if not args.execute:
        print("[hand] 검사만 — --execute 를 붙이면 pd/hand_path 로 보낸다")
        return 0

    ctl = [sys.executable, str(HERE / "episode_ctl.py"), "--only", "pd_hand_path", "--side", args.side,
           "--execute", "--approve", "pd_hand_path"]
    print(f"\n$ {' '.join(ctl)}", flush=True)
    if subprocess.run(ctl, cwd=str(SIM2REAL)).returncode != 0:
        print("✗ pd/hand_path 거부 — pd 가 engage 돼 있는가", file=sys.stderr)
        return 1
    time.sleep(args.settle_s)
    back = measure()
    if radius is not None:
        ok, how = in_sphere(back)
        if not ok:
            print(f"✗ 손이 봉투 구를 넘는다 — {how}. 손가락이 걸렸는지 눈으로 볼 것", file=sys.stderr)
            return 1
        print(f"[hand] 손 봉투 {how} — 구 안. 자세가 기준과 달라도 저장 경로는 유효하다")
        return 0
    got = {j: back.get(j, np.nan) - want[j] for j in want}
    bad = max(got, key=lambda j: abs(got[j]))
    if abs(got[bad]) > args.tol:
        print(f"✗ 손이 기준 자세에 닿지 못했다 — {bad} 오차 {got[bad]:+.3f} rad (허용 {args.tol}). "
              "손가락이 걸렸는지 눈으로 볼 것", file=sys.stderr)
        return 1
    print(f"[hand] 도착 · 최대 오차 {abs(got[bad]):.3f} rad ({bad})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
