#!/usr/bin/env python3
"""고정 place 궤적 — 붓기 시작 자세(tilt_now)에서 쥔 컵을 홀더에 놓는 팔 + 손 궤적을 계획 · 검사한다. 움직이지 않는다.

11.17 사용자: RL place 정책 대신 홀더가 고정이므로 계산한 궤적으로 놓는다. 실기에서 jog/손 교시로 놓기 지점만 맞춰
그 차이를 컵별 설정(config/place_fixed/<cup>.yaml) offset 에 적고 다시 계산한다(궤적 점을 손으로 고치지 않는다).

  컵 놓기 지점 = place_carry.carry_target(홀더, 물체 원점 높이, place_dz) + offset(x, y, z).
  손바닥 방향 = 시작 자세 그대로(컵 똑바로), 컵은 손바닥 기준 상대 위치 유지 → 손바닥 목표 = 시작 손바닥 + (컵 목표 − 컵 시작).
  offset.yaw_deg = 컵 축(z) 둘레로 손바닥을 돌린다(컵 자리는 그대로).
  구간: approach(시작 → 놓기 + approach_dz, 직선 최소 저크) → descend(수직) → release(손 펴기 + 정지)
        → lift(손 편 채 +lift_m) → hold.
  IK · 보간 · 검사 기준은 plan_lift 와 같다(위치 2 mm · 방향 1° · 관절 여유 ≥ 5° · 직선 3 mm · 60 Hz 관절 속도 ≤ 2 rad/s)
  + 충돌(reach_world: 탁자 · 몸 · 반대팔) + 홀더 여유(hdgp holder_clear) + 컵 ↔ 탁자(plan_carry TABLE_MARGIN_M).
  놓는 순간 검사: 컵 바닥 − 링 윗면(drop_m, yaml release_drop_m) · 컵 xy ↔ 홀더 xy ≤ 구멍 반지름 − 컵 반지름 ·
  손이 다 편 뒤 lift 전 정지 ≥ 0.3 s · 두 팔 실측 시작 자세 ↔ 명목 ≤ 3°.
종료 코드: 0 = 재생해도 됨 · 2 = 검사 실패(place_cmd.npz 안 씀) · 3 = 핀 · 입력 불일치. 시작할 때 --out 의 옛 산출물을 지운다.
재생(실기, 사용자 승인 뒤): replay_traj_rh.py --npz <out>/place_cmd.npz ...

    python3 deploy/policy_control/tools/plan_place_fixed.py --cup blue --out logs/runs/<YYMMDD>/place_fixed/blue
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))
from policy_control.place_carry import carry_target  # noqa: E402

SIM2REAL = TOOLS.parents[2]
CFG_DIR = TOOLS.parent / "config" / "place_fixed"
CUPS = {"blue": "blue_h1", "pink": "pink_h2", "shaker": "shaker_h0"}
START_YAML = TOOLS.parent / "config" / "homes" / "rh56f1_pour_start.yaml"
HOLDERS_YAML = SIM2REAL / "config" / "cup_holder_poses_5090_v1.yaml"
HOLDER_CLEAR_REL = "scripts/tools/tilt_newton/holder_clear.py"
HOLDER_RIM_DZ = 0.030          # assets/meshes/cup_holder.stl z 최대 30 mm(원점 = 받침 윗면) — 빈 내 홀더 링 윗면
#                                (= hdgp tasks/rh_place_r/place_geom.py HOLDER_TOP_Z · 상판 = 원점 − place_carry.HOLDER_BASE_DZ)
HOLDER_BORE_R = 0.035          # hdgp tasks/rh_place_r/place_geom.py HOLDER_BORE_R — 컵이 들어가는 구멍 반지름
HAND_MAP_YAML = TOOLS.parent / "config" / "rh56f1_hand_map.yaml"   # 손 관절 rad 범위(0 = 편 손 ~ 굽힘 한계)
OPEN_SETTLE_S = 0.3            # 손이 다 편 뒤 lift 시작까지 최소 정지
RELEASE_MIN_S = 1.5
Q_START_TOL_DEG = 3.0          # --q-json 실측 ↔ 명목 시작 자세(cup_start 가 정의된 자세), 놓는 팔만
TIME_KEYS = ("approach", "descend", "release", "lift", "hold")
HOLDER_CLEAR_M = 0.02          # 다른 홀더(+ 꽂힌 컵) 여유 — plan_tilt_traj LIMITS clr_min_m
OWN_RIM_CLEAR_M = 0.01         # 접근 중 컵 ↔ 내 홀더 링
STAGE_REDUCED_ARM, HAND_VMAX, SPEED_MARGIN = 0.25, 2.1, 0.8   # replay_traj_rh.py 와 같은 값(재생 rate_scale 안내용)


def rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cup_place(holder_pos, origin_z: float, place_dz: float, offset: dict) -> np.ndarray:
    """컵 원점 놓기 지점(base) = carry_target + offset xyz."""
    return carry_target(holder_pos, origin_z, place_dz) + np.array([offset["x"], offset["y"], offset["z"]], float)


def palm_pose(p0: np.ndarray, R0: np.ndarray, cup0: np.ndarray, cup: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """컵이 cup 에 있고 컵 축 둘레로 yaw_deg 돌았을 때 손바닥 (위치, 방향) — 시작 손바닥 ↔ 컵 상대 자세 유지."""
    Rz = rz(yaw_deg)
    return np.asarray(cup, float) + Rz @ (np.asarray(p0, float) - np.asarray(cup0, float)), Rz @ R0


def load_cfg(path: Path) -> dict:
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(c, dict):
        raise ValueError(f"{path}: yaml 최상위가 dict 아님")
    need = ("cup", "object", "side", "holder", "geometry", "cup_start", "place_dz", "approach_dz", "lift_m", "time_s", "offset")
    miss = [k for k in need if k not in c]
    if miss:
        raise ValueError(f"{path}: 키 없음 {miss}")
    if not isinstance(c["geometry"], dict) or any(k not in c["geometry"] for k in ("contract", "object")):
        raise ValueError(f"{path}: geometry 에 contract · object 필요")
    if isinstance(c["holder"], bool) or not isinstance(c["holder"], int) or c["holder"] < 0:
        raise ValueError(f"{path}: holder {c['holder']!r} — 0 이상 정수")
    cs = c["cup_start"]
    if not isinstance(cs, (list, tuple)) or len(cs) != 3 or not np.all(np.isfinite(np.asarray(cs, float))):
        raise ValueError(f"{path}: cup_start {cs!r} — 유한값 3 개")
    float(c["place_dz"])
    miss = [k for k in ("x", "y", "z", "yaw_deg") if k not in c["offset"]] + [k for k in TIME_KEYS if k not in c["time_s"]]
    if miss:
        raise ValueError(f"{path}: offset · time_s 키 없음 {miss}")
    off = {k: float(c["offset"][k]) for k in ("x", "y", "z", "yaw_deg")}
    # 상한 = 11.17 손 교시(logs/place_teach/*_c2) 와 0 오프셋 계산의 차이(z 최대 +7.9 cm · yaw 최대 −46°)를 담는 폭
    if max(abs(off[k]) for k in "xyz") > 0.10 or abs(off["yaw_deg"]) > 60.0:
        raise ValueError(f"{path}: offset {off} — xyz 10 cm · yaw 60° 넘음(교시 값 확인)")
    if c["side"] not in ("right", "left") or not 0.02 <= float(c["approach_dz"]) <= 0.15 or not 0.02 <= float(c["lift_m"]) <= 0.15:
        raise ValueError(f"{path}: side · approach_dz(0.02~0.15) · lift_m(0.02~0.15)")
    if any(not 0.5 <= float(c["time_s"][k]) <= 10.0 for k in TIME_KEYS) or float(c["time_s"]["release"]) < RELEASE_MIN_S:
        raise ValueError(f"{path}: time_s 0.5~10 s · release ≥ {RELEASE_MIN_S} s")
    lo, hi = (float(v) for v in c.get("release_drop_m", (-0.005, 0.03)))
    if not -0.03 <= lo < hi <= 0.10:
        raise ValueError(f"{path}: release_drop_m [{lo}, {hi}] — min < max, −0.03~0.10 m")
    return {**c, "offset": off, "release_drop_m": [lo, hi]}


def check_grip(grip) -> list[float]:
    """손 명령 6 값이 각 관절 0 ~ 굽힘 한계(rh56f1_hand_map.yaml rad) 안인지."""
    m = yaml.safe_load(HAND_MAP_YAML.read_text(encoding="utf-8"))
    hi = [float(m["joints"][k]["rad"][1]) for k in m["joint_order"]]
    g = [float(v) for v in grip]
    if len(g) != len(hi) or any(not 0.0 <= v <= h for v, h in zip(g, hi)):
        raise ValueError(f"hand_grip {g} — 6 값, 각 0 ~ {hi}")
    return g


def hand_schedule(grip, open_, n: int, r0: int, step: float) -> tuple[np.ndarray, int]:
    """쥔 채 → r0 부터 open_ 로 틱당 step 제한. (손 명령 [n, 6], 다 편 첫 칸 — 못 펴면 n)."""
    from policy_control import place_teach as PT
    hand = np.repeat(np.asarray(grip, float)[None], n, 0)
    hand[r0:] = open_
    hand = PT._rate_limited(hand, step)
    hit = np.all(np.abs(hand[r0:] - np.asarray(open_, float)) < 1e-9, axis=1)
    return hand, (r0 + int(np.argmax(hit)) if hit.any() else n)


def q_start_dev_deg(q: dict, nominal: dict, side: str) -> float:
    """한 팔 실측 ↔ 명목 시작 자세(cup_start 가 정의된 자세) 관절 최대 편차 [deg].
    3° 검사는 놓는 팔만 한다. 반대 팔은 앞 놓기 뒤 자세(예 lift 끝)일 수 있고, 실측 q 가 충돌 세계(ReachWorld)에 그대로 들어간다."""
    return float(np.degrees(np.abs(np.asarray(q[side], float) - np.asarray(nominal[side], float))).max())


def release_checks(drop_m: float, drop_lim, xy_err: float, xy_tol: float, settle_s: float, open_after_s: float,
                   q_dev: float) -> dict:
    """놓는 순간 검사 4 개: 바닥 − 링 윗면 ∈ [lo, hi] · xy ≤ 구멍 여유 · 다 편 뒤 정지 ≥ OPEN_SETTLE_S · 시작 편차 ≤ 3°."""
    lo, hi = drop_lim
    settle = round(settle_s, 3)
    return {
        "hand_open_settle_s": {"value": settle, "limit": OPEN_SETTLE_S, "ok": bool(settle >= OPEN_SETTLE_S),
                               "open_after_s": round(open_after_s, 3)},
        "release_drop_m": {"value": round(drop_m, 4), "limit": [lo, hi], "ok": bool(lo <= drop_m <= hi)},
        "release_xy_m": {"value": round(xy_err, 4), "limit": round(xy_tol, 4), "ok": bool(xy_err <= xy_tol)},
        "q_start_dev_deg": {"value": round(q_dev, 2), "limit": Q_START_TOL_DEG, "ok": bool(q_dev <= Q_START_TOL_DEG)},
    }


def all_ok(checks: dict) -> bool:
    """limit 만 있는 검사는 value ≥ limit 로 "ok" 를 채우고(checks 를 제자리에서 고친다 — meta 에 그대로 쓰려고),
    하나라도 실패면 False(npz 안 씀)."""
    for c in checks.values():
        if "ok" not in c:
            c["ok"] = bool(c["value"] >= c["limit"])
    return all(c["ok"] for c in checks.values())


def geometry(cfg: dict) -> dict:
    """origin_z · 반지름 · 바닥/입구 — place 정책 계약 object_geometry(rh_place.origin_z_of 와 같은 값)."""
    p = SIM2REAL / cfg["geometry"]["contract"]
    g = json.loads(p.read_text(encoding="utf-8"))["object_geometry"][cfg["geometry"]["object"]]
    return {"contract": str(p), "contract_md5": None, "object": cfg["geometry"]["object"], "origin_z": float(g["origin_offset_z"]),
            "radius": float(g["radius"]), "bottom_z": float(g["bottom_z"]), "mouth_z": float(g["mouth_z"]), **fit_radius(cfg, g)}


def fit_radius(cfg: dict, g: dict) -> dict:
    """구멍에 들어가는 아랫부분 반지름 — 설정 fit_radius_m(위가 넓은 컵) 있으면 그 값, 없으면 계약 반지름(가장 넓은 곳)."""
    if "fit_radius_m" not in cfg:
        return {"fit_radius": float(g["radius"]), "fit_radius_source": "contract.radius"}
    r = float(cfg["fit_radius_m"])
    if not 0.0 < r <= float(g["radius"]):
        raise ValueError(f"fit_radius_m {r} — 0 초과 · 계약 반지름 {g['radius']} 이하")
    return {"fit_radius": r, "fit_radius_source": "cfg.fit_radius_m"}


def _holder_clear(hdgp):
    path = Path(hdgp / HOLDER_CLEAR_REL)
    spec = importlib.util.spec_from_file_location("hdgp_holder_clear", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def plan(cfg: dict, q: dict, holders_path: Path, hand_grip, out: Path, hdgp) -> tuple[bool, dict]:
    import torch
    from plan_carry import TABLE_MARGIN_M
    from plan_reach import (ANG_TOL_DEG, LINE_DEV_M, LINE_ITERS, MARGIN_MIN_DEG, POS_TOL_M, RATE_HZ, VMAX_RAD_S,
                            ik_pose, line_deviation, margin_deg, minjerk)
    from plan_tilt import ARMS, ASSET, H, table_z
    from reach_world import ReachWorld
    from policy_control import place_teach as PT, rh_replay as RR

    pol = H.load_policy(ASSET)
    H.verify(hdgp, pol["hdgp"]["md5"])
    RC = H.load_ready_carry(hdgp)
    HC, hc_path = _holder_clear(hdgp)
    side, off, T = cfg["side"], cfg["offset"], cfg["time_s"]
    geo = geometry(cfg)
    geo["contract_md5"] = H.md5(geo["contract"])
    holders = HC.load_holders(str(holders_path))
    k_h = int(cfg["holder"])
    hpos = holders[k_h]

    ch = RC.chain.Chain.from_urdf(RC.chain.default_urdf(), f"{side[0]}_hl_palm_sensor", ARMS[side])
    q0 = torch.tensor([q[side]], dtype=torch.float32)
    T0 = ch.fk(q0)[0].double().numpy()
    R0, p0 = T0[:3, :3], T0[:3, 3]
    # 컵 ↔ 손바닥 상대 위치는 명목 시작 자세(cup_start 가 정의된 자세)에서 정하고, 실측 q 면 손바닥 프레임으로 옮긴다
    nominal = yaml.safe_load(START_YAML.read_text(encoding="utf-8"))
    qn = nominal[side]
    Tn = ch.fk(torch.tensor([qn], dtype=torch.float32))[0].double().numpy()
    off_local = Tn[:3, :3].T @ (np.asarray(cfg["cup_start"], float) - Tn[:3, 3])
    cup0 = p0 + R0 @ off_local
    cup_pl = cup_place(hpos, geo["origin_z"], float(cfg["place_dz"]), off)
    cup_ap = cup_pl + np.array([0.0, 0.0, float(cfg["approach_dz"])])
    yaw = off["yaw_deg"]

    def frames(seconds: float) -> int:
        return max(2, int(round(seconds * RATE_HZ)) + 1)

    # (위치, 방향) 목표 열 — approach 는 컵 직선 + yaw 를 같은 s 로, descend 는 수직, lift 는 손바닥 수직
    s_a, s_b, s_d = minjerk(frames(T["approach"])), minjerk(frames(T["descend"])), minjerk(frames(T["lift"]))
    goals = {"approach": [palm_pose(p0, R0, cup0, cup0 + s * (cup_ap - cup0), yaw * s) for s in s_a[1:]],
             "descend": [palm_pose(p0, R0, cup0, cup_ap + s * (cup_pl - cup_ap), yaw) for s in s_b[1:]]}
    p_pl, R_pl = palm_pose(p0, R0, cup0, cup_pl, yaw)
    goals["lift"] = [(p_pl + np.array([0.0, 0.0, float(cfg["lift_m"]) * s]), R_pl) for s in s_d[1:]]

    rows, qk, e_p, e_a = [q0[0].clone()], q0.clone(), 0.0, 0.0
    bounds, seg_err = {}, {}

    def solve(label: str) -> None:
        nonlocal qk, e_p, e_a
        a, sp, sa = len(rows), 0.0, 0.0
        for p, R in goals[label]:
            qk, ep, ea = ik_pose(RC.posture, ch, qk, p, R, LINE_ITERS)
            sp, sa = max(sp, float(ep[0])), max(sa, float(ea[0]))
            rows.append(qk[0].clone())
        e_p, e_a = max(e_p, sp), max(e_a, sa)
        seg_err[label] = {"pos_mm": round(sp * 1000, 2), "ang_deg": round(sa, 3)}
        bounds[label] = (0 if label == "approach" else a, len(rows) - 1)

    def hold(label: str, seconds: float) -> None:
        a = len(rows)
        rows.extend([rows[-1].clone()] * int(round(seconds * RATE_HZ)))
        bounds[label] = (a, len(rows) - 1)

    solve("approach")
    solve("descend")
    hold("release", float(T["release"]))
    solve("lift")
    hold("hold", float(T["hold"]))
    path = torch.stack(rows)
    arm = path.double().numpy()
    n = len(arm)

    # 손: 쥔 채(approach · descend) → release 첫 칸부터 OPEN_RAD 로 place_teach 와 같은 속도 제한 → 편 채
    grip, open_ = np.asarray(hand_grip, float), np.asarray(PT.OPEN_RAD, float)
    r0 = bounds["release"][0]
    hand, hand_open_at = hand_schedule(grip, open_, n, r0, PT.HAND_VEL_RAD_S / RATE_HZ)

    Tfk = ch.fk(path).double().numpy()
    palm = Tfk[:, :3, 3]
    cup = palm + np.einsum("nij,j->ni", Tfk[:, :3, :3], off_local)
    seg = {k: slice(a, b + 1) for k, (a, b) in bounds.items()}
    line = {"approach": line_deviation(cup[seg["approach"]], cup0, cup_ap),
            "descend": line_deviation(cup[seg["descend"]], cup[bounds["descend"][0] - 1], cup_pl),
            "lift": line_deviation(palm[seg["lift"]], palm[bounds["lift"][0] - 1], p_pl + [0, 0, float(cfg["lift_m"])])}

    # 탁자 · 몸 · 반대팔(plan_lift 와 같은 세계) + 빈 내 홀더 링(원기둥 r HOLDER_R, 탁자 ~ 링 윗면)을 reach_world 의 '컵' 자리에 넣어
    #   손가락 · 팔 충돌 형상으로 관통만 금지한다(쥔 컵은 손 안이라 넣지 않는다)
    other = "left" if side == "right" else "right"
    own_top = float(hpos[2]) + HOLDER_RIM_DZ
    ring = {"x": float(hpos[0]), "y": float(hpos[1]), "z_lo": float(hpos[2]) - 0.030, "z_hi": own_top, "radius": HC.HOLDER_R}
    rw = ReachWorld(side, hdgp, q[side], q[other], ring)
    col = rw.check(arm, 0.0)
    # 홀더 여유: 다른 홀더(+ 꽂힌 컵, 탁자 ~ 0.45 m 원기둥) ↔ 컵 · 팔 · 손 / 내 홀더 링 ↔ 컵(접근 중) · 팔 · 손(전 구간)
    others = np.delete(holders, k_h, 0)
    segs = [(geo["bottom_z"], geo["mouth_z"], geo["radius"])]
    eye = np.repeat(np.eye(3)[None], n, 0)
    cup_pts = HC.cup_axis_points(cup, eye, segs)
    cup_other = np.min([HC.column_dist(p, others).min(-1) - r for p, r in cup_pts], 0)
    ap = seg["approach"]
    # 접근 중 컵(원기둥) ↔ 내 홀더 링: 축 사이 거리가 반지름 합보다 작으면 컵 바닥 − 링 윗면, 아니면 옆 거리
    radial = np.hypot(*(cup[ap, :2] - hpos[:2]).T) - (HC.HOLDER_R + geo["radius"])
    vert = cup[ap, 2] + geo["bottom_z"] - own_top
    cup_own_ap = float(np.min(np.where((radial > 0) & (vert > 0), np.hypot(radial, vert), np.maximum(radial, vert))))
    ap_pts = HC.arm_points(ch, arm)
    arm_segs = [HC._seg_points(ap_pts[:, i], ap_pts[:, i + 1]) for i in range(ap_pts.shape[1] - 1)]
    arm_other = float(np.min([HC.column_dist(sg, others).min() for sg in arm_segs]) - HC.ARM_R)
    hand_other = float(HC.column_dist(ap_pts[:, -1], others).min() - HC.HAND_R)
    tz = table_z(pol)
    cup_table_ap = float((cup[ap, 2] + geo["bottom_z"]).min() - tz)

    # 놓는 순간(descend 끝 = 손 펴기 시작) 컵: 바닥 − 링 윗면, 축 ↔ 홀더 축 xy 거리
    cup_rel = cup[bounds["descend"][1]]
    drop_m = float(cup_rel[2] + geo["bottom_z"] - own_top)
    xy_err = float(np.hypot(*(cup_rel[:2] - hpos[:2])))
    xy_tol = HOLDER_BORE_R - geo["fit_radius"]
    q_dev = q_start_dev_deg(q, nominal, side)
    vmax = float(np.abs(np.diff(arm, axis=0)).max()) * RATE_HZ
    checks = {
        "ik_pos_mm": {"value": round(e_p * 1000, 2), "limit": POS_TOL_M * 1000, "ok": e_p <= POS_TOL_M},
        "ik_ang_deg": {"value": round(e_a, 3), "limit": ANG_TOL_DEG, "ok": e_a <= ANG_TOL_DEG},
        "joint_margin_deg": {"value": round(float(margin_deg(ch, path).min()), 2), "limit": MARGIN_MIN_DEG},
        "profile_margin_deg": {"value": round(float(rw.profile_margin_deg(arm).min()), 2), "limit": MARGIN_MIN_DEG},
        "line_dev_mm": {"value": {k: round(v * 1000, 2) for k, v in line.items()}, "limit": LINE_DEV_M * 1000,
                        "ok": max(line.values()) <= LINE_DEV_M},
        "vmax_rad_s": {"value": round(vmax, 3), "limit": VMAX_RAD_S, "ok": vmax <= VMAX_RAD_S},
        "collision": {"value": col, "ok": bool(col["ok"])},
        "cup_other_holders_m": {"value": round(float(cup_other.min()), 4), "limit": HOLDER_CLEAR_M},
        "arm_other_holders_m": {"value": round(arm_other, 4), "limit": HOLDER_CLEAR_M},
        "hand_other_holders_m": {"value": round(hand_other, 4), "limit": HOLDER_CLEAR_M},
        "cup_own_rim_approach_m": {"value": round(float(cup_own_ap), 4), "limit": OWN_RIM_CLEAR_M},
        "cup_table_approach_m": {"value": round(cup_table_ap, 4), "limit": TABLE_MARGIN_M},
        **release_checks(drop_m, cfg["release_drop_m"], xy_err, xy_tol, (bounds["lift"][0] - hand_open_at) / RATE_HZ,
                         (hand_open_at - r0) / RATE_HZ, q_dev),
        # 정보만(판정 안 함): 반대 팔 편차. 충돌은 위 collision 이 반대 팔 실측 q 로 본다
        "q_other_dev_deg": {"value": round(q_start_dev_deg(q, nominal, other), 2), "ok": True, "info": True},
    }
    cup_err = float(np.linalg.norm(cup_rel - cup_pl))
    ok = all_ok(checks)

    segments = [{"label": k, "start": a, "end": b, "hand_command": "position"} for k, (a, b) in bounds.items()]
    out.mkdir(parents=True, exist_ok=True)
    npz, tmp = out / "place_cmd.npz", out / "place_cmd.tmp.npz"
    meta = {"ok": ok, "cup": cfg["cup"], "object": cfg["object"], "side": side, "holder": k_h,
            "place_xyz": [round(float(v), 4) for v in cup_pl], "approach_xyz": [round(float(v), 4) for v in cup_ap],
            "cup_start_xyz": [round(float(v), 4) for v in cup0], "cup_place_err_mm": round(cup_err * 1000, 2),
            "palm_start": [round(float(v), 4) for v in p0], "palm_place": [round(float(v), 4) for v in p_pl],
            "offset": off, "place_dz": float(cfg["place_dz"]), "approach_dz": float(cfg["approach_dz"]), "lift_m": float(cfg["lift_m"]),
            "time_s": T, "geometry": geo, "table_z": tz, "holder_pos": [float(v) for v in hpos], "own_rim_z": own_top,
            "hand_grip": [float(v) for v in grip], "hand_grip_source": cfg["_grip_src"], "hand_open": [float(v) for v in open_],
            "release_drop_m": round(drop_m, 4), "release_xy_m": round(xy_err, 4), "release_xy_tol_m": round(xy_tol, 4),
            "segments": segments, "n_cmd": n, "T_end_s": round((n - 1) / RATE_HZ, 2), "q_start": [float(v) for v in q[side]],
            "inputs": {"cfg": cfg["_path"], "cfg_md5": H.md5(cfg["_path"]), "holders": str(holders_path),
                       "holders_md5": H.md5(holders_path), "start_yaml": str(START_YAML), "start_yaml_md5": H.md5(START_YAML),
                       "q_json": cfg.get("_q_json"), "holder_clear": str(hc_path), "holder_clear_md5": H.md5(hc_path),
                       "hdgp": str(hdgp), "hdgp_commit_pin": pol["hdgp"].get("commit")},
            "ik_err_by_segment": seg_err, "vmax_by_segment": {
                k: round(float(np.abs(np.diff(arm[max(a - 1, 0):b + 1], axis=0)).max()) * RATE_HZ, 3) for k, (a, b) in bounds.items()},
            "checks": checks}
    if ok:
        np.savez(tmp, meta_schema=np.array(PT.SCHEMA), meta_side=np.array(side), meta_arm_names=np.array(PT.arm_names(side)),
                 meta_hand_names=np.array(PT.hand_names(side)), meta_step_dt=np.array(1.0 / RATE_HZ),
                 meta_segments=np.array(json.dumps(segments)), meta_source=np.array("plan_place_fixed"),
                 meta_retreat=np.array("none"), arm_q=arm, hand_q_cmd=hand, arm_target=arm,
                 joint_names=np.array(PT.arm_names(side)))
        traj = RR.load_replay(tmp)                       # replay_traj_rh 와 같은 읽기로 형식 확인
        rs = RR.max_rate_scale(traj, arm_vmax=STAGE_REDUCED_ARM * SPEED_MARGIN, hand_vmax=HAND_VMAX * SPEED_MARGIN)
        meta["replay_max_rate_scale_reduced"] = float(np.floor(rs * 1000) / 1000)   # 내림: 반올림이 허용 최대를 넘지 않게
        meta["wall_s"] = round(meta["T_end_s"] / rs, 2)
        meta["npz_md5"] = H.md5(tmp)
    (out / "check.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    if ok:
        os.replace(tmp, npz)                             # 검증 · meta 가 끝난 뒤에만 재생 이름으로
    return ok, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cup", choices=tuple(CUPS), required=True)
    ap.add_argument("--cfg", type=Path, default=None, help="컵별 설정(기본 config/place_fixed/<cup>_h<k>.yaml)")
    ap.add_argument("--holders", type=Path, default=HOLDERS_YAML)
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} 실측 — 없으면 rh56f1_pour_start.yaml')
    ap.add_argument("--hand-grip", default="", help="쥔 구간 손 명령 6 값(rad, 쉼표) — 없으면 place_teach.GRIP_RAD")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from plan_tilt import H
    from policy_control import place_teach as PT

    for f in ("place_cmd.npz", "place_cmd.tmp.npz", "meta.json", "check.json"):   # 실패해도 옛 산출물이 남지 않게
        (a.out / f).unlink(missing_ok=True)
    cfg_path = a.cfg or CFG_DIR / f"{CUPS[a.cup]}.yaml"
    try:
        cfg = load_cfg(cfg_path)
        if cfg["cup"] != a.cup:
            raise ValueError(f"{cfg_path}: cup {cfg['cup']} ≠ --cup {a.cup}")
        q = (json.loads(Path(a.q_json).read_text(encoding="utf-8")) if a.q_json
             else {k: v for k, v in yaml.safe_load(START_YAML.read_text(encoding="utf-8")).items() if k in ("right", "left")})
        if any(len(q[s]) != 7 for s in ("right", "left")):
            raise ValueError("q 팔마다 7 값")
        src = "--hand-grip" if a.hand_grip else "cfg.hand_grip" if "hand_grip" in cfg else "place_teach.GRIP_RAD"
        grip = check_grip(a.hand_grip.split(",") if a.hand_grip else cfg.get("hand_grip", PT.GRIP_RAD))
        cfg = {**cfg, "_path": str(cfg_path), "_q_json": a.q_json or None, "_grip_src": src}
        hdgp = H.hdgp_dir()
    except (H.PinError, ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        print(f"✗ {type(exc).__name__}: {exc}")
        return 3
    try:
        ok, meta = plan(cfg, q, a.holders, grip, a.out, hdgp)
    except (H.PinError, ValueError, KeyError, IndexError) as exc:   # 계약 · 홀더 파일의 키 · 번호 불일치
        print(f"✗ {type(exc).__name__}: {exc}")
        return 3
    bad = [k for k, c in meta["checks"].items() if not c["ok"]]
    print(json.dumps({"ok": ok, "place_xyz": meta["place_xyz"], "T_end_s": meta["T_end_s"], "failed": bad,
                      "rate_scale_reduced": meta.get("replay_max_rate_scale_reduced"), "wall_s": meta.get("wall_s"),
                      "release_drop_m": meta["release_drop_m"], "release_xy_m": meta["release_xy_m"],
                      "release_xy_tol_m": meta["release_xy_tol_m"],
                      "hand_open_settle_s": meta["checks"]["hand_open_settle_s"]["value"]}, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
