#!/usr/bin/env python3
"""tilt 1차(tilt_traj 궤적 재생, 잔차 0)를 지금 두 팔 자세에서 계획 · 검사한다 — 움직이지 않는다(관절 상태 구독만).

10.26 learn:e2e-aglt 요청(sim grasp2tilt, hdgp 24bfd4dc): 잡기 → ready_carry 이송(plan_carry, 끝 = rh56f1_pour_start) →
컵별 최종 CMA 파라미터 궤적. sim 과 같은 계산을 쓴다:
  · 궤적 = hdgp agnostic/modules/tilt_traj.build_batch(작업 트리 + md5 핀, config/tilt/traj_v10/policy.yaml)
  · 고정 입력(q0 · mouth_ref · D4 표 · Geo · 여유 설정) = traj_v10/setup_<컵>.npz(build_tilt_traj_asset.py 가 sim Context 에서 꺼냄)
  · 시작 = capture: 실측 관절에서 궤적 첫 knot 까지 pre 0.5 s 관절 min-jerk(rh_tilt env._teach_at 과 같은 식)
  · knot 1/30 s 를 60 Hz 로 선형 보간(env 와 같음). 복귀(t_ret 4 s, q0 로 min-jerk)까지 재생한다 — sim 판정은 정착(settle)까지.
다른 점(남은 차이): 홀더는 --holders(실측 yaml), 탁자 높이는 실기 head_extrinsics table_top_z, 손 안 컵 자리(T_in)는
기본값(e2e 짝 중앙값)이거나 --tin-json(실측) — 실기는 쥔 자리를 모른다.

  python3 deploy/policy_control/tools/plan_tilt_traj.py --cup 240 --out logs/tilt/<run>          # 실측 관절
  python3 deploy/policy_control/tools/plan_tilt_traj.py --cup 240 --q-json q.json --out /tmp/t   # 로봇 없이
재생(실기, 사용자 승인 뒤):
  replay_to_pd.py --npz <out>/tilt_cmd.npz --joints r_aj_1,..,r_aj_7,l_aj_1,..,l_aj_7 --rate-scale 1.0 --max-start-gap 0.05 --execute
종료 코드: 0 = 재생해도 됨 · 2 = 검사 실패(재생 금지) · 3 = 핀 · 자산 불일치 · 그 밖 = 계획 못 함.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

SIM2REAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM2REAL / "deploy" / "policy_control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_control import hdgp_pin as H  # noqa: E402

ASSET = SIM2REAL / "deploy" / "policy_control" / "config" / "tilt" / "traj_v10"
MODULES = "source/openarm/openarm/agnostic/modules"
URDF = "assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
TIP = {"r": "r_hl_palm_sensor", "l": "l_hl_palm_sensor"}
ARMS = {"r": [f"r_aj_{i}" for i in range(1, 8)], "l": [f"l_aj_{i}" for i in range(1, 8)]}
JOINTS = ARMS["r"] + ARMS["l"]
PKG = "s2r_hdgp_modules"          # openarm 패키지 초기화(학습 과제 import)를 거치지 않게 modules 폴더만 따로 이름 붙인다
RATE_HZ = 60.0
# 검사 한계 — sim 과 같은 값(tilt_newton_batch.pre_fail: IK 20 mm · 홀더 여유 20 mm · 탁자 10 mm) + 실기 쪽
LIMITS = {"ik_pos_mm": 20.0, "clr_min_m": 0.02, "table_clear_m": 0.01, "cmd_vel_rad_s": 2.0, "joint_margin_rad": 0.0,
          "start_gap_rad": 0.35}


def load_modules(root: Path):
    """hdgp modules 폴더를 PKG 이름으로 올리고 tilt_traj · ready_carry 를 돌려준다(상대 import ..ready_carry 가 그대로 돈다)."""
    if PKG not in sys.modules:
        d = Path(root) / MODULES
        spec = importlib.util.spec_from_file_location(PKG, d / "__init__.py", submodule_search_locations=[str(d)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{PKG}.tilt_traj"), importlib.import_module(f"{PKG}.ready_carry.chain")


def load_holders(path: Path) -> np.ndarray:
    """홀더 yaml(cup_holder_pose.py 형식) → (H,3) 축 x, y · 받침 윗면 z(hdgp holder_clear.load_holders 와 같은 순서)."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    hs = d.get("holders")
    if not isinstance(hs, dict) or not hs:
        raise H.PinError(f"{path}: holders 가 없다")
    return np.array([h["position"] for _, h in sorted(hs.items())], dtype=np.float64)


def table_z(cfg: dict) -> float:
    src = SIM2REAL / str(cfg["table"]["from"])
    z = (yaml.safe_load(src.read_text(encoding="utf-8")) or {}).get("table_top_z")
    if not isinstance(z, (int, float)):
        raise H.PinError(f"{src}: table_top_z 가 없다")
    return float(z)


def current_q(q_json: str) -> dict[str, list[float]]:
    """{"right": [7], "left": [7]} — 파일이 있으면 그것, 없으면 /joint_states 실측."""
    if q_json:
        return json.loads(Path(q_json).read_text(encoding="utf-8"))
    from check_path_start import measure
    q = measure(JOINTS)
    return {"right": [float(q[j]) for j in ARMS["r"]], "left": [float(q[j]) for j in ARMS["l"]]}


def minjerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def resample(q_start: np.ndarray, knots: np.ndarray, dt_knot: float, pre_s: float, rate: float = RATE_HZ) -> np.ndarray:
    """rh_tilt env._teach_at 과 같은 식: s < pre → 실측 → knot 0 관절 min-jerk, 그 뒤 knot 선형 보간. (N,J)."""
    K = knots.shape[0]
    t_end = pre_s + (K - 1) * dt_knot
    s = np.arange(int(math.floor(t_end * rate + 1e-9)) + 1) / rate
    tau = np.clip(s - pre_s, 0.0, None) / dt_knot
    i0 = np.clip(np.floor(tau).astype(int), 0, K - 1)
    i1 = np.minimum(i0 + 1, K - 1)
    f = np.clip(tau - i0, 0.0, 1.0)[:, None]
    q = knots[i0] * (1 - f) + knots[i1] * f
    if pre_s > 0:
        m = minjerk(s / pre_s)[:, None]
        q = np.where((s < pre_s)[:, None], q_start + (knots[0] - q_start) * m, q)
    return q


def plan(cfg: dict, asset: Path, cup: str, q: dict, holders: Path, tin_json: str, root: Path) -> dict:
    import torch
    TT, CH = load_modules(root)
    from dataclasses import fields
    S = np.load(asset / "ref" / f"setup_{cup}.npz", allow_pickle=False)
    params = json.loads((asset / "ref" / f"params_{cup}.json").read_text(encoding="utf-8"))["params"]
    geo_d, clr_d = json.loads(str(S["geo_json"])), json.loads(str(S["clear_json"]))
    clr_d = {k: (tuple(tuple(x) for x in v) if k.endswith("_segs") else v) for k, v in clr_d.items()}
    clr_d["table_z"] = table_z(cfg)                                    # sim 0.205 → 실기 탁자
    t64 = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))  # noqa: E731
    ch = {s: CH.Chain.from_urdf(str(root / URDF), TIP[s], ARMS[s], device="cpu", dtype=torch.float32) for s in "rl"}
    geo = TT.TiltGeo(**{f.name: geo_d[f.name] for f in fields(TT.TiltGeo)})
    CL = importlib.import_module(f"{PKG}.tilt_traj.clearance")
    setup = TT.TiltSetup(chain_r=ch["r"], chain_l=ch["l"], q0_r=t64(S["q0_r"]), q0_l=t64(S["q0_l"]),
                         mouth_ref=t64(S["mouth_ref"]), th_tab=t64(S["th_tab"]), q_tab=t64(S["q_tab"]),
                         holders=t64(load_holders(holders)), geo=geo,
                         clear=CL.ClearCfg(**{k: v for k, v in clr_d.items() if k in {f.name for f in fields(CL.ClearCfg)}}),
                         fast=False, cuda_graph=False)
    if tin_json:
        tj = json.loads(Path(tin_json).read_text(encoding="utf-8"))
        T_r, T_l, tin_src = np.asarray(tj["T_in_r"]), np.asarray(tj["T_in_l"]), tin_json
    else:
        T_r, T_l, tin_src = S["T_in_r"], S["T_in_l"], "asset 기본값(e2e 짝 중앙값)"
    tb = TT.build_batch(setup, TT.params_tensor([params]), t64(T_r)[None], t64(T_l)[None])
    L = int(tb.length[0])
    knots = np.concatenate([tb.q_r[0, :L].numpy(), tb.q_l[0, :L].numpy()], 1)       # (L,14)
    q_meas = np.array(list(q["right"]) + list(q["left"]), dtype=np.float64)
    cmd = resample(q_meas, knots, geo.dt, float(cfg["pre_s"]))
    info = {k: float(v[0]) for k, v in tb.info.items()}
    lo = np.concatenate([ch["r"].lo.numpy(), ch["l"].lo.numpy()])
    hi = np.concatenate([ch["r"].hi.numpy(), ch["l"].hi.numpy()])
    vel = float(np.abs(np.diff(cmd, axis=0)).max() * RATE_HZ)
    margin = float(np.minimum(cmd - lo, hi - cmd).min())
    gap = float(np.abs(q_meas - knots[0]).max())
    e = tb.knots.edges[0].numpy()
    checks = {
        "ik_pos_mm": {"value": round(info["ik_pos_mm"], 2), "ok": info["ik_pos_mm"] <= LIMITS["ik_pos_mm"]},
        "clr_min_m": {"value": round(info["clr_min"], 4), "ok": info["clr_min"] >= LIMITS["clr_min_m"]},
        "table_clear_m": {"value": round(info["table_clear"], 4), "ok": info["table_clear"] >= LIMITS["table_clear_m"]},
        "cmd_vel_rad_s": {"value": round(vel, 3), "ok": vel <= LIMITS["cmd_vel_rad_s"]},
        "joint_margin_rad": {"value": round(margin, 4), "ok": margin >= LIMITS["joint_margin_rad"]},
        "start_gap_rad": {"value": round(gap, 4), "ok": gap <= LIMITS["start_gap_rad"]},
    }
    return {"cmd": cmd, "knots": knots, "info": info, "checks": checks, "ok": all(c["ok"] for c in checks.values()),
            "params": params, "tin": tin_src, "T_in_r": np.asarray(T_r).tolist(), "T_in_l": np.asarray(T_l).tolist(),
            "secs": {"pre": float(cfg["pre_s"]), "app_end": round(float(e[2]), 3), "tilt_end": round(float(e[3]), 3),
                     "settle_end": round(float(e[6]), 3), "T_end_s": round(len(cmd) / RATE_HZ, 3)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cup", required=True, choices=("240", "200"), help="붓는 컵(오른손)")
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--holders", type=Path, default=None, help="홀더 yaml — 없으면 policy.yaml holders.default")
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} — 없으면 실측')
    ap.add_argument("--tin-json", default="", help='{"T_in_r": 4x4, "T_in_l": 4x4} 손끝 → 컵(왼손은 rcv_z_off 포함) — 없으면 기본값')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        cfg = H.load_policy(a.asset)
        root = H.hdgp_tree()
        H.verify(root, cfg["hdgp"]["md5"])
        H.verify_refs(a.asset, cfg.get("ref") or {})
        holders = a.holders or (SIM2REAL / cfg["holders"]["default"])
        r = plan(cfg, a.asset, a.cup, current_q(a.q_json), holders, a.tin_json, root)
    except H.PinError as exc:
        print(f"✗ {exc}")
        return 3
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "tilt_cmd.npz", arm_target=r["cmd"], knots=r["knots"], joint_names=np.array(JOINTS),
                        meta_step_dt=np.array(1.0 / RATE_HZ))
    chk = {k: v for k, v in r.items() if k not in ("cmd", "knots")}
    chk.update({"cup": a.cup, "holders": str(holders), "holders_md5": H.md5(holders), "asset": str(a.asset)})
    (a.out / "check.json").write_text(json.dumps(chk, indent=1, ensure_ascii=False), encoding="utf-8")
    bad = {k: v["value"] for k, v in r["checks"].items() if not v["ok"]}
    print(json.dumps({"ok": r["ok"], "fail": bad, "secs": r["secs"], "tin": r["tin"],
                      "npz": str(a.out / "tilt_cmd.npz")}, ensure_ascii=False))
    return 0 if r["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
