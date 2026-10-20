#!/usr/bin/env python3
"""1-B 들기 — 쥔 뒤 지금 자세에서 손바닥 방향을 그대로 둔 채 수직으로 lift_m 올리고 hold_s 유지하는 팔 궤적을 계획 · 검사한다.

10.19 사용자 "1-B: READY CARRY 로 컵으로 접근 + 스크립트 파지/리프트" · 결정 1(10.17) "들어 올린 뒤 3 s 유지 = 1 회 성공".
경로 C 의 들기(approach_grasp lift — 손바닥 방향 고정 수직 10 cm)를 정책 노드 없이 계획 궤적으로 한다. 움직이지 않는다
(관절 실측 구독만). 재생은 에피소드 실행기가 replay_to_pd 로(그 팔 7 관절 — 손 목표는 pd 가 쥔 채 붙든다).

  손바닥 점 = 지금 실측 FK(ready_carry 사슬 · 1판 URDF). 목표 = 같은 방향 · 같은 xy · z + lift_m, 최소 저크 time_s → hold_s 정지.
  IK = ready_carry posture.solve(pose 모드), 앞 칸에서 이어 푼다(같은 IK 가지).
  검사(하나라도 걸리면 종료 코드 2 = 재생 금지): 위치 2 mm · 방향 1° · 관절 여유 ≥ 5° · 직선 3 mm · 60 Hz 관절 속도 ≤ 2 rad/s.

    python3 deploy/policy_control/tools/plan_lift.py --side right --out logs/x/lift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_reach import (ANG_TOL_DEG, LINE_DEV_M, LINE_ITERS, MARGIN_MIN_DEG, POS_TOL_M, RATE_HZ, VMAX_RAD_S,  # noqa: E402
                        ik_pose, line_deviation, margin_deg, minjerk)
from plan_tilt import ARMS, ASSET, H, current_q  # noqa: E402

LIFT_M = 0.10                   # 결정 1 · 경로 C 와 같은 10 cm
TIME_S = 3.0                    # 평균 3.3 cm/s(최소 저크 최대 6.25 cm/s) — 경로 C lift_speed 0.05 m/s 와 비슷
HOLD_S = 3.0                    # 결정 1: 3 s 유지


def plan(side: str, out: Path, q: dict, hdgp: Path, *, asset: Path = ASSET, lift_m: float = LIFT_M, time_s: float = TIME_S,
         hold_s: float = HOLD_S) -> tuple[bool, dict]:
    cfg = H.load_policy(asset)
    H.verify(hdgp, cfg["hdgp"]["md5"])
    RC = H.load_ready_carry(hdgp)
    ch = RC.chain.Chain.from_urdf(RC.chain.default_urdf(), f"{side[0]}_hl_palm_sensor", ARMS[side])
    q0 = torch.tensor([q[side]], dtype=torch.float32)
    T0 = ch.fk(q0)[0].double().numpy()
    R, p0 = T0[:3, :3], T0[:3, 3]
    n = max(2, int(round(time_s * RATE_HZ)) + 1)
    s = minjerk(n)
    rows, qk, e_p, e_a = [q0[0].clone()], q0.clone(), 0.0, 0.0
    for k in range(1, n):
        qk, ep, ea = ik_pose(RC.posture, ch, qk, p0 + np.array([0.0, 0.0, lift_m * s[k]]), R, LINE_ITERS)
        e_p, e_a = max(e_p, float(ep[0])), max(e_a, float(ea[0]))
        rows.append(qk[0].clone())
    path = torch.stack(rows)
    palm = ch.fk(path)[:, :3, 3].double().numpy()
    full = np.vstack([path.double().numpy(), np.repeat(path[-1:].double().numpy(), int(round(hold_s * RATE_HZ)), 0)])
    meta = {"side": side, "lift_m": lift_m, "time_s": time_s, "hold_s": hold_s, "palm_start": [round(float(v), 4) for v in p0],
            "palm_end": [round(float(v), 4) for v in palm[-1]], "rise_m": round(float(palm[-1, 2] - p0[2]), 4),
            "max_pos_err_mm": round(e_p * 1000, 2), "max_ang_err_deg": round(e_a, 2),
            "line_dev_mm": round(line_deviation(palm, palm[0], palm[-1]) * 1000, 2),
            "margin_min_deg": round(float(margin_deg(ch, path).min()), 2),
            "vmax_rad_s": round(float(np.abs(np.diff(full, axis=0)).max()) * RATE_HZ, 3),
            "q_start": [round(float(v), 4) for v in q[side]], "n_cmd": int(len(full)), "T_end_s": round((len(full) - 1) / RATE_HZ, 2),
            "hdgp": str(hdgp), "hdgp_commit_pin": cfg["hdgp"].get("commit"), "problems": []}
    if e_p > POS_TOL_M or e_a > ANG_TOL_DEG:
        meta["problems"].append(f"IK 오차 {meta['max_pos_err_mm']} mm · {meta['max_ang_err_deg']}°")
    if meta["line_dev_mm"] > LINE_DEV_M * 1000:
        meta["problems"].append(f"손바닥이 직선에서 {meta['line_dev_mm']} mm")
    if meta["margin_min_deg"] < MARGIN_MIN_DEG:
        meta["problems"].append(f"관절 여유 {meta['margin_min_deg']}° < {MARGIN_MIN_DEG}°")
    if meta["vmax_rad_s"] > VMAX_RAD_S:
        meta["problems"].append(f"관절 속도 {meta['vmax_rad_s']} rad/s > {VMAX_RAD_S}")
    # ★10.20 사용자 "테이블, 자기 충돌, 관절 제약 등을 고려해야함": reach 와 같은 세계(reach_world — 테이블 · 벽 · 몸통 · 머리 ·
    #   반대팔 실측 2 cm, 같은 팔 이웃 관절 관통 금지) · 프로필 관절 한계 여유. 컵은 손 안이라 세계에 두지 않는다.
    from reach_world import ReachWorld
    other = "left" if side == "right" else "right"
    rw = ReachWorld(side, hdgp, q[side], q[other], None)
    pa = path.double().numpy()
    col = rw.check(pa, 0.0)
    meta["collision"] = col
    meta["profile_margin_min_deg"] = round(float(rw.profile_margin_deg(pa).min()), 2)
    if not col["ok"]:
        meta["problems"].append("충돌 " + json.dumps(col.get("worst"), ensure_ascii=False))
    if meta["profile_margin_min_deg"] < MARGIN_MIN_DEG:
        meta["problems"].append(f"프로필 관절 여유 {meta['profile_margin_min_deg']}° < {MARGIN_MIN_DEG}°")
    meta["ok"] = not meta["problems"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "lift_cmd.npz").unlink(missing_ok=True)
    if meta["ok"]:
        np.savez_compressed(out / "lift_cmd.npz", arm_target=full, joint_names=np.array(ARMS[side]),
                            meta_step_dt=np.array(1.0 / RATE_HZ))
    (out / "check.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta["ok"], meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=tuple(ARMS), required=True)
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} — 없으면 실측')
    ap.add_argument("--lift", type=float, default=LIFT_M)
    ap.add_argument("--time", type=float, default=TIME_S)
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not (0.02 <= a.lift <= 0.15 and 1.5 <= a.time <= 10.0 and 0.0 <= a.hold <= 10.0):
        print(f"✗ --lift {a.lift}(0.02~0.15) · --time {a.time}(1.5~10) · --hold {a.hold}(0~10)")
        return 1
    try:
        ok, meta = plan(a.side, a.out, current_q(a.q_json), H.hdgp_dir(), asset=a.asset, lift_m=a.lift, time_s=a.time,
                        hold_s=a.hold)
    except H.PinError as exc:
        print(f"✗ {exc}")
        return 3
    print(json.dumps({k: meta[k] for k in ("ok", "rise_m", "T_end_s", "margin_min_deg", "vmax_rad_s", "line_dev_mm", "problems")},
                     ensure_ascii=False))
    if not ok:
        print(f"✗ 계획 거부 — {'; '.join(meta['problems'])}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
