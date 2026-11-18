#!/usr/bin/env python3
"""place_shaker 직전 왼팔 shaker 섞기(잔 돌리기) 궤적을 지금 자세에서 검사한다 — 움직이지 않는다(관절 상태 구독만).

11.18 사용자: 동작은 기울여 돌리기(swirl), 실기 시연용이라 비드가 쏟아지지 않게 순하게.
궤적은 계획하지 않고 자산의 고정 npz(hdgp scripts/tools/shaker_mix/motion.py 가 만든 왼팔 60 Hz arm_target)를 그대로 쓴다.
시작 · 끝 = rh56f1_pour_start.yaml left(tilt 궤적 처음 · 끝), 손 안 shaker = tilt traj_v12_240 tin_fixed_1107.json T_in_l.
검사: ref npz md5(자산 policy.yaml) · 실측 q · npz 유한 · 왼팔 실측 ↔ npz 첫 프레임 ≤ start_gap_rad · npz 끝 ↔ start_pose left ≤ start_gap_rad ·
60 Hz 관절 속도 ≤ VMAX_RAD_S · 배속 r 재생 가속도 ≤ AMAX_RAD_S2 · 관절 한계 여유 ≥ MARGIN_MIN_RAD · shaker 최저점 ≥ 탁자 + TABLE_MARGIN_M ·
두 팔 사이(오른팔은 실측에 서 있다 — plan_carry.arms_clear 그대로). 통과하면 <out>/mix_cmd.npz(ref 복사). 시작할 때 옛 mix_cmd.npz · check.json 을 지운다.
오른손은 항상 빈손이라고 본다(--empty 없음) — 컵 겉면 검사는 shaker ↔ 오른팔 점만 한다.
재생(실기, 사용자 승인 뒤): replay_to_pd.py --npz <out>/mix_cmd.npz --joints l_aj_1,..,l_aj_7 --rate-scale <policy.yaml> --max-start-gap <start_gap_rad> --execute

    python3 deploy/policy_control/tools/plan_mix.py --out logs/runs/<YYMMDD>/<run>/mix_<id>              # 실측 관절
    python3 deploy/policy_control/tools/plan_mix.py --q-json q.json --out <폴더>                          # 로봇 없이
종료 코드: 0 = 재생해도 됨 · 2 = 검사 실패(재생 금지) · 3 = 핀 · 자산 불일치 · 1 = 인자 · 관절 입력(--q-json 없음 · 키 · 길이 · 유한) 오류.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_carry import (ARMS, ASSET as TABLE_ASSET, RATE_HZ, SIM2REAL, TABLE_MARGIN_M, VMAX_RAD_S, H,  # noqa: E402
                        arms_clear, current_q, lowest_z, table_z)

ASSET = SIM2REAL / "deploy" / "policy_control" / "config" / "mix" / "shaker_swirl"
SIDE = "left"
MARGIN_MIN_RAD = 0.05        # hdgp shaker_mix/motion.py MARGIN_MIN_RAD 와 같음(실기 추종 오차 · 외란 몫)
AMAX_RAD_S2 = 8.0            # hdgp shaker_mix/motion.py AMAX_RAD_S2 와 같음(순한 동작 실측 4.6~5.3 위 여유 — 비드가 쏟아지지 않게)


def load_asset(asset: Path, hdgp) -> tuple[dict, dict, torch.Tensor, np.ndarray]:
    """(policy.yaml, ref npz, T_to (1,4,4) 손바닥 → shaker 원점, start_pose left (7,)). 핀 · 자산이 맞지 않으면 PinError."""
    cfg = H.load_policy(asset)
    for k, hi in (("rate_scale", 1.0), ("start_gap_rad", 0.1)):
        v = cfg.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < v <= hi:   # NaN · 없음 · 문자열도 여기서 걸린다
            raise H.PinError(f"{asset}/policy.yaml: {k} {v!r} 가 (0, {hi}] 안의 숫자가 아니다")
    pose_path = SIM2REAL / str(cfg.get("start_pose"))
    if not pose_path.is_file():
        raise H.PinError(f"{asset}/policy.yaml: start_pose {pose_path} 없음")
    q_end = np.asarray(yaml.safe_load(pose_path.read_text(encoding="utf-8"))[SIDE], float)
    H.verify_refs(asset, cfg.get("ref") or {})
    name = cfg.get("ref_npz")
    if name not in (cfg.get("ref") or {}):
        raise H.PinError(f"{asset}/policy.yaml: ref_npz {name!r} 가 ref(md5 핀)에 없다")
    tin = cfg["tin"]
    tin_path = SIM2REAL / tin["from"]
    if not tin_path.is_file() or H.md5(tin_path) != str(tin["md5"]):
        raise H.PinError(f"{tin_path}: md5 {H.md5(tin_path) if tin_path.is_file() else '없음'} ≠ 고정 {tin['md5']}")
    H.verify(hdgp, cfg["hdgp"]["md5"])
    with np.load(asset / "ref" / name, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    if ([str(j) for j in ref.get("joint_names", [])] != ARMS[SIDE] or ref["arm_target"].ndim != 2
            or ref["arm_target"].shape[1] != 7 or len(ref["arm_target"]) < 2
            or abs(float(ref.get("meta_step_dt", 0.0)) - 1.0 / RATE_HZ) > 1e-9):
        raise H.PinError(f"{name}: 왼팔 7 관절 · 60 Hz arm_target 이 아니다")
    if not np.isfinite(ref["arm_target"]).all() or q_end.shape != (7,) or not np.isfinite(q_end).all():
        raise H.PinError(f"{name} 또는 {pose_path}: 유한한 왼팔 7 관절이 아니다")
    T = np.asarray(json.loads(tin_path.read_text(encoding="utf-8"))["T_in_l"]["T"], float)
    Tz = np.eye(4)
    Tz[2, 3] = -float(tin["z_off_m"])                  # T_in_l 은 shaker 원점 + z_off·z 를 가리킨다(tin_live.Z_OFF)
    return cfg, ref, torch.tensor(T @ Tz, dtype=torch.float32)[None], q_end


def plan(asset: Path, out: Path, q: dict, hdgp) -> tuple[bool, dict]:
    try:
        cfg, ref, T_to_l, q_end = load_asset(asset, hdgp)
    except (TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:   # 깨진 npz · yaml · json — 자산 문제는 트레이스백 대신 rc 3
        raise H.PinError(f"{asset}: 자산 읽기 실패 {exc!r}") from exc
    r = float(cfg["rate_scale"])
    RC = H.load_ready_carry(hdgp)
    chains = {s: RC.chain.Chain.from_urdf(RC.chain.default_urdf(), f"{s[0]}_hl_palm_sensor", ARMS[s]) for s in ARMS}
    path = np.asarray(ref["arm_target"], float)
    n = len(path)
    paths = {SIDE: path, "right": np.repeat(np.asarray([q["right"]], float), n, 0)}      # 오른팔 = 실측 그대로
    gap = float(np.abs(np.asarray(q[SIDE], float) - path[0]).max())
    end_gap = float(np.abs(path[-1] - q_end).max())
    vmax = float(np.abs(np.diff(path, axis=0)).max() * RATE_HZ)
    amax = float(np.abs(np.diff(path, n=2, axis=0)).max() * (RATE_HZ * r) ** 2)       # 배속 r 재생 = 시간 1/r 배 → 가속도 r² 배
    path_t = torch.tensor(path, dtype=torch.float32)
    marg = float(chains[SIDE].margin(path_t).min())
    # plan_carry 컵 기하(원점 바닥 위 85 mm · 반지름 30 mm)를 그대로 쓴다 — shaker(65 mm · 35.4 mm)보다 바닥을 약 17 mm 낮게 잰다(안전 쪽 오차)
    table = lowest_z(RC.posture, chains[SIDE], path_t, T_to_l, SIDE) - table_z(H.load_policy(TABLE_ASSET))   # 탁자 = plan_carry 와 같은 출처
    arms = arms_clear(RC.posture, chains, {s: torch.tensor(paths[s], dtype=torch.float32) for s in ARMS}, {SIDE: T_to_l}, ("right",))
    lim = float(cfg["start_gap_rad"])
    checks = {"start_gap_rad": round(gap, 4), "start_gap_max_rad": lim, "end_gap_rad": round(end_gap, 4), "vmax_rad_s": round(vmax, 3),
              "amax_rad_s2": round(amax, 2), "limit_margin_rad": round(marg, 4), "table_clear_m": round(table, 4)}
    # not (x <= 한도) 꼴 — NaN 이 통과하지 않게
    why = [k for k, bad in (("start_gap", not gap <= lim), ("end_pose", not end_gap <= lim), ("vmax", not vmax <= VMAX_RAD_S),
                            ("amax", not amax <= AMAX_RAD_S2), ("limit_margin", not marg >= MARGIN_MIN_RAD),
                            ("table", not table >= TABLE_MARGIN_M), ("arms_clear", not arms["ok"])) if bad]
    ok = not why
    out.mkdir(parents=True, exist_ok=True)
    if ok:
        shutil.copyfile(asset / "ref" / cfg["ref_npz"], out / "mix_cmd.npz")
    meta = {"ok": ok, "fail": why, "side": SIDE, "checks": checks, "arms_clear": arms, "n_cmd": n, "T_end_s": round((n - 1) / RATE_HZ, 3),
            "rate_scale": r, "T_end_real_s": round((n - 1) / RATE_HZ / r, 3),
            "q_start": q, "ref_npz": cfg["ref_npz"], "asset": str(asset), "hdgp": str(hdgp), "hdgp_commit_pin": cfg["hdgp"].get("commit")}
    (out / "check.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return ok, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} — 없으면 실측')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    for f in ("mix_cmd.npz", "check.json"):                        # 실패 · 거부로 끝나도 옛 통과본 · 옛 판정이 남지 않게
        (a.out / f).unlink(missing_ok=True)
    try:
        q = current_q(a.q_json)
        bad = [s for s in ARMS if np.asarray(q[s], float).shape != (7,) or not np.isfinite(np.asarray(q[s], float)).all()]
    except (OSError, ValueError, KeyError, TypeError) as exc:          # 파일 없음 · JSON 깨짐 · 키 누락 · 숫자 아님
        print(f"✗ 관절 입력: {exc!r}")
        return 1
    if bad:
        print(f"✗ 관절 입력: {bad} 가 유한한 7 관절이 아니다 — {q}")
        return 1
    try:
        ok, meta = plan(a.asset, a.out, q, H.hdgp_dir())
    except (H.PinError, KeyError) as exc:
        print(f"✗ {exc!r}" if isinstance(exc, KeyError) else f"✗ {exc}")
        return 3
    print(json.dumps({"ok": ok, "fail": meta["fail"], "T_end_s": meta["T_end_s"], "checks": meta["checks"], "arms_clear": meta["arms_clear"],
                      "npz": str(a.out / "mix_cmd.npz") if ok else None}, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
