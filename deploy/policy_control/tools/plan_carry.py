#!/usr/bin/env python3
"""잡기 뒤 두 팔을 붓기 시작 자세로 옮길 궤적을 계획 · 검사한다 — 움직이지 않는다(관절 상태 구독만).

10.17 사용자 결정 12: 팔 이송은 ready_carry(hdgp Mover, md5 고정) + 끝 자세는 고정 이정표(rh56f1_pour_start.yaml)
→ Mover.start(q_goal=이정표). 컵 축을 먼저 세우고(앞 35 %) 세운 채 컵 점은 최소 저크로 곧게, 남는 자유는 관절 기준.
컵 = 손 안 컵 자리(tilt 와 같은 기준 출발 single734_t120 의 자리, 실측 아님).
열린 고리로 굴린다(q ← q + dq, 완벽 추종 가정) → 60 Hz 관절 목표. 실기에서는 pd 가 그 목표를 따라간다.
검사: 계획 ok(이정표의 컵 점 · 축이 앵커와 3 mm · 3°, 관절 여유 ≥ 1°) · 도착(컵 점 1 cm · 축 2° · 관절 0.05 rad · 정지) ·
탁자 여유(컵 최저점 − 탁자 ≥ 1 cm) · 60 Hz 관절 속도. 하나라도 걸리면 종료 코드 2(재생 금지).
재생(실기, 사용자 승인 뒤): replay_to_pd.py --npz <out>/carry_cmd.npz --joints <14 관절> --rate-scale 1.0 --max-start-gap 0.05 --execute

    python3 deploy/policy_control/tools/plan_carry.py --out logs/tilt/<run>/carry
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
from plan_tilt import ARMS, ASSET, JOINTS, SIM2REAL, H, current_q, table_z  # noqa: E402

RATE_HZ = 60.0
CUP_CENTER_M = 0.085                               # 컵 원점 = 바닥 위 85 mm(D4 README, 두 컵 같음)
CUP_TOP_M = {"right": 0.155, "left": 0.045}        # 원점 → 입구(병 src240 · shaker)
CUP_R_M = 0.03
TABLE_MARGIN_M = 0.01
VMAX_RAD_S = 2.0
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


def plan(asset: Path, out: Path, q: dict, hdgp: Path) -> tuple[bool, dict]:
    cfg = H.load_policy(asset)
    H.verify(hdgp, cfg["hdgp"]["md5"])
    H.verify_refs(asset, cfg.get("ref") or {})
    RC = H.load_ready_carry(hdgp)
    pose = yaml.safe_load((SIM2REAL / cfg["start_pose"]).read_text(encoding="utf-8"))
    ref = dict(np.load(asset / "ref" / "single734_t120.npz", allow_pickle=False))
    tz = table_z(cfg)
    paths, checks = {}, {}
    for side in ARMS:
        ch = RC.chain.Chain.from_urdf(RC.chain.default_urdf(), f"{side[0]}_hl_palm_sensor", ARMS[side])
        T_to = cup_in_palm(RC, ch, ref, side)
        q0 = torch.tensor([q[side]], dtype=torch.float32)
        qg = torch.tensor([pose[side]], dtype=torch.float32)
        path, info = roll(RC, ch, q0, qg, T_to)
        info["table_clear_m"] = round(lowest_z(RC.posture, ch, path, T_to, side) - tz, 4)
        info["vmax_rad_s"] = round(float((path[1:] - path[:-1]).abs().max()) * RATE_HZ, 3)
        info["ok"] = bool(info["plan_ok"] and info["arrived"] and info["table_clear_m"] >= TABLE_MARGIN_M
                          and info["vmax_rad_s"] <= VMAX_RAD_S)
        paths[side], checks[side] = path.double().numpy(), info
    n = max(len(p) for p in paths.values())
    arm = np.concatenate([np.vstack([p, np.repeat(p[-1:], n - len(p), 0)]) for p in (paths["right"], paths["left"])], 1)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "carry_cmd.npz", arm_target=arm, joint_names=np.array(JOINTS), meta_step_dt=np.array(1.0 / RATE_HZ))
    ok = all(c["ok"] for c in checks.values())
    meta = {"ok": ok, "checks": checks, "n_cmd": int(n), "T_end_s": round((n - 1) / RATE_HZ, 3), "table_z": tz,
            "q_start": q, "q_goal": {s: pose[s] for s in ARMS}, "start_pose": cfg["start_pose"],
            "hdgp": str(hdgp), "hdgp_commit_pin": cfg["hdgp"].get("commit")}
    (out / "check.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return ok, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} — 없으면 실측')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        ok, meta = plan(a.asset, a.out, current_q(a.q_json), H.hdgp_dir())
    except H.PinError as exc:
        print(f"✗ {exc}")
        return 3
    print(json.dumps({"ok": ok, "T_end_s": meta["T_end_s"], "checks": meta["checks"], "npz": str(a.out / "carry_cmd.npz")},
                     ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
