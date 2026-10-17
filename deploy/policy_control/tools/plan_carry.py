#!/usr/bin/env python3
"""잡기 뒤 두 팔을 붓기 시작 자세로 옮길 궤적을 계획 · 검사한다 — 움직이지 않는다(관절 상태 구독만).

10.17 사용자 결정 12: 팔 이송은 ready_carry(hdgp Mover, md5 고정) + 끝 자세는 고정 이정표(rh56f1_pour_start.yaml)
→ Mover.start(q_goal=이정표). 컵 축을 먼저 세우고(앞 35 %) 세운 채 컵 점은 최소 저크로 곧게, 남는 자유는 관절 기준.
컵 = 손 안 컵 자리(tilt 와 같은 기준 출발 single734_t120 의 자리, 실측 아님).
열린 고리로 굴린다(q ← q + dq, 완벽 추종 가정) → 60 Hz 관절 목표. 실기에서는 pd 가 그 목표를 따라간다.
검사: 계획 ok(이정표의 컵 점 · 축이 앵커와 3 mm · 3°, 관절 여유 ≥ 1°) · 도착(컵 점 1 cm · 축 2° · 관절 0.05 rad · 정지) ·
탁자 여유(컵 최저점 − 탁자 ≥ 1 cm) · 60 Hz 관절 속도 · ★두 팔 사이(리뷰: 한 팔씩이면 다른 팔은 실측에 서 있다 —
팔 관절 4~7 원점 · 손바닥 점끼리 ≥ ARM_POINT_GAP_M, 두 손 안 컵 겉면끼리 ≥ CUP_GAP_M). 하나라도 걸리면 종료 코드 2(재생 금지).
재생(실기, 사용자 승인 뒤): replay_to_pd.py --npz <out>/carry_cmd.npz --joints <그 팔 관절> --rate-scale 1.0 --max-start-gap 0.05 --execute
★10.17 계획 경로 A "ready_carry(한 팔씩)": --sides right 면 오른팔만 계획한다(npz 열 = 그 팔 7 관절, 다른 팔은 pd 가 붙든 채).
빈손 이송(경로 B)도 같은 경로다 — 컵이 없어도 손 안 컵 자리를 가상 컵으로 두고 검사한다(탁자 여유는 그만큼 보수적).

    python3 deploy/policy_control/tools/plan_carry.py --out logs/tilt/<run>/carry [--sides right]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_tilt import ARMS, ASSET, SIM2REAL, H, current_q, table_z  # noqa: E402

RATE_HZ = 60.0
CUP_CENTER_M = 0.085                               # 컵 원점 = 바닥 위 85 mm(D4 README, 두 컵 같음)
CUP_TOP_M = {"right": 0.155, "left": 0.045}        # 원점 → 입구(병 src240 · shaker)
CUP_R_M = 0.03
TABLE_MARGIN_M = 0.01
VMAX_RAD_S = 2.0
ARM_POINT_GAP_M = 0.10        # 두 팔 관절 4~7 원점 · 손바닥 점 사이 최소 거리(링크 반지름 약 4~5 cm 두 개)
CUP_GAP_M = 0.03              # 두 손 안 컵(반지름 CUP_R_M 원기둥) 겉면 사이 최소
ROOT_KEY = {"right": "src_root", "left": "rcv_root"}


def quat_up(q: np.ndarray) -> np.ndarray:
    """wxyz → 물체 +z 축(세계)."""
    w, x, y, z = (float(v) for v in q / np.linalg.norm(q))
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def palm_frame(al: np.ndarray, pl: np.ndarray) -> torch.Tensor:
    """손바닥 → 컵 (1,4,4): 원점 pl, +z = al(축 대칭 컵이라 나머지 축은 아무 직교축)."""
    z = al / np.linalg.norm(al)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = pl
    return torch.tensor(T, dtype=torch.float32)[None]


def cup_in_palm(RC, ch, ref: dict, side: str) -> torch.Tensor:
    """기준 출발(single734 행 0)의 손 안 컵 자리 — tilt_real_start(DR.start_state)와 같은 정의."""
    names = [str(n) for n in ref["joint_names"]]
    q = torch.tensor([[float(ref["joint_pos"][0, names.index(j)]) for j in ARMS[side]]])
    T0 = ch.fk(q)[0].double().numpy()
    root = ref[ROOT_KEY[side]][0].astype(float)
    R0, p0 = T0[:3, :3], T0[:3, 3]
    return palm_frame(R0.T @ quat_up(root[3:7]), R0.T @ (root[:3] - p0))


def lowest_z(P, ch, path: torch.Tensor, T_to: torch.Tensor, side: str) -> float:
    """경로 전체에서 컵(원기둥 바닥 · 입구 원)의 가장 낮은 점 z."""
    _, p, z, _, _ = P.object_state(ch, path, T_to.expand(len(path), 4, 4), None)
    rim = CUP_R_M * torch.sqrt((1.0 - z[:, 2] ** 2).clamp(min=0.0))
    lo = torch.minimum((p - CUP_CENTER_M * z)[:, 2], (p + CUP_TOP_M[side] * z)[:, 2]) - rim
    return float(lo.min())


def arm_points(ch, path: torch.Tensor) -> torch.Tensor:
    """(N,7) → (N,5,3) 관절 4~7 원점 + 손바닥 점(로봇 루트)."""
    T, _ax, org = ch.fk(path, frames=True)
    return torch.cat([org[:, 3:7], T[:, None, :3, 3]], dim=1)


def cup_points(P, ch, path: torch.Tensor, T_to: torch.Tensor, side: str, k: int = 6) -> torch.Tensor:
    """(N,7) → (N,k,3) 손 안 컵 축 위 점(바닥 → 입구)."""
    _, p, z, _, _ = P.object_state(ch, path, T_to.expand(len(path), 4, 4), None)
    s = torch.linspace(-CUP_CENTER_M, CUP_TOP_M[side], k, dtype=p.dtype)
    return p[:, None, :] + s[None, :, None] * z[:, None, :]


def arms_clear(P, chains: dict, paths: dict, T_tos: dict) -> dict:
    """두 팔 경로(같은 길이) 사이 최소 거리 — 팔 점끼리 · 컵 겉면끼리."""
    a = {s: arm_points(chains[s], paths[s]) for s in ARMS}
    c = {s: cup_points(P, chains[s], paths[s], T_tos[s], s) for s in ARMS}
    d_arm = torch.cdist(a["right"], a["left"]).amin(dim=(1, 2))
    d_cup = torch.cdist(c["right"], c["left"]).amin(dim=(1, 2)) - 2 * CUP_R_M
    out = {"min_arm_point_m": round(float(d_arm.min()), 4), "min_cup_gap_m": round(float(d_cup.min()), 4)}
    out["ok"] = bool(out["min_arm_point_m"] >= ARM_POINT_GAP_M and out["min_cup_gap_m"] >= CUP_GAP_M)
    return out


def roll(RC, ch, q0: torch.Tensor, qg: torch.Tensor, T_to: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Mover 를 열린 고리로 굴린다 → (경로 (N,7) 첫 행 = q0, 요약)."""
    P, cfg = RC.posture, RC.mover.MoverCfg()
    _, pa, za, _, _ = P.object_state(ch, qg, T_to, None)
    reg = P.line_region(pa, za, 0.0, 1, 0.0, 0.0)
    mv = RC.mover.Mover(ch, cfg, 1, "cpu")
    plan = mv.start(torch.arange(1), q0, T_to, None, pa, za, reg, q_goal=qg)
    q, rows, arrived = q0.clone(), [q0[0].clone()], False
    for _ in range(cfg.max_steps + cfg.settle_steps + 5):
        dq, on, done, arr = mv.step(q)
        q = q + dq
        rows.append(q[0].clone())
        arrived = arrived or bool(arr[0])
        if not mv.on.any():
            break
    info = {"plan_ok": bool(plan.ok[0]), "anchor_err_mm": round(float(plan.err_pos[0]) * 1000, 2),
            "anchor_err_deg": round(float(plan.err_axis[0]), 2), "goal_margin_deg": round(float(np.degrees(float(plan.margin[0]))), 1),
            "arrived": arrived, "steps": len(rows) - 1, "end_q_err_rad": round(float((q - qg).abs().max()), 4),
            "end_cup_err_mm": round(float(mv.last_err[0]) * 1000, 2), "end_axis_err_deg": round(float(mv.last_axis_deg[0]), 2)}
    return torch.stack(rows), info


def plan(asset: Path, out: Path, q: dict, hdgp: Path, sides: tuple = tuple(ARMS)) -> tuple[bool, dict]:
    cfg = H.load_policy(asset)
    H.verify(hdgp, cfg["hdgp"]["md5"])
    H.verify_refs(asset, cfg.get("ref") or {})
    RC = H.load_ready_carry(hdgp)
    pose = yaml.safe_load((SIM2REAL / cfg["start_pose"]).read_text(encoding="utf-8"))
    ref = dict(np.load(asset / "ref" / "single734_t120.npz", allow_pickle=False))
    tz = table_z(cfg)
    paths, checks, chains, T_tos = {}, {}, {}, {}
    for side in ARMS:
        chains[side] = RC.chain.Chain.from_urdf(RC.chain.default_urdf(), f"{side[0]}_hl_palm_sensor", ARMS[side])
        T_tos[side] = cup_in_palm(RC, chains[side], ref, side)
    for side in sides:
        ch, T_to = chains[side], T_tos[side]
        q0 = torch.tensor([q[side]], dtype=torch.float32)
        qg = torch.tensor([pose[side]], dtype=torch.float32)
        path, info = roll(RC, ch, q0, qg, T_to)
        info["table_clear_m"] = round(lowest_z(RC.posture, ch, path, T_to, side) - tz, 4)
        info["vmax_rad_s"] = round(float((path[1:] - path[:-1]).abs().max()) * RATE_HZ, 3)
        info["ok"] = bool(info["plan_ok"] and info["arrived"] and info["table_clear_m"] >= TABLE_MARGIN_M
                          and info["vmax_rad_s"] <= VMAX_RAD_S)
        paths[side], checks[side] = path.double().numpy(), info
    n = max(len(p) for p in paths.values())
    full = {s: np.vstack([paths[s], np.repeat(paths[s][-1:], n - len(paths[s]), 0)]) if s in paths
            else np.repeat(np.asarray([q[s]], float), n, 0) for s in ARMS}      # 움직이지 않는 팔 = 실측 그대로
    arms = arms_clear(RC.posture, chains, {s: torch.tensor(full[s], dtype=torch.float32) for s in ARMS}, T_tos)
    arm = np.concatenate([full[s] for s in sides], 1)
    out.mkdir(parents=True, exist_ok=True)
    names = [j for s in sides for j in ARMS[s]]
    np.savez_compressed(out / "carry_cmd.npz", arm_target=arm, joint_names=np.array(names), meta_step_dt=np.array(1.0 / RATE_HZ))
    ok = all(c["ok"] for c in checks.values()) and arms["ok"]
    meta = {"ok": ok, "sides": list(sides), "checks": checks, "arms_clear": arms, "n_cmd": int(n), "T_end_s": round((n - 1) / RATE_HZ, 3),
            "table_z": tz, "q_start": {s: q[s] for s in sides}, "q_goal": {s: pose[s] for s in sides}, "start_pose": cfg["start_pose"],
            "hdgp": str(hdgp), "hdgp_commit_pin": cfg["hdgp"].get("commit")}
    (out / "check.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return ok, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} — 없으면 실측')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sides", default="right,left", help="right · left · right,left(한 팔씩이면 하나)")
    a = ap.parse_args(argv)
    sides = tuple(x.strip() for x in a.sides.split(",") if x.strip())
    if not sides or any(x not in ARMS for x in sides) or len(set(sides)) != len(sides):
        print(f"✗ --sides {a.sides!r} — right · left 중")
        return 1
    sides = tuple(x for x in ARMS if x in sides)                 # 열 순서는 언제나 우 → 좌
    try:
        ok, meta = plan(a.asset, a.out, current_q(a.q_json), H.hdgp_dir(), sides)
    except H.PinError as exc:
        print(f"✗ {exc}")
        return 3
    print(json.dumps({"ok": ok, "T_end_s": meta["T_end_s"], "arms_clear": meta["arms_clear"], "checks": meta["checks"],
                      "npz": str(a.out / "carry_cmd.npz")},
                     ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
