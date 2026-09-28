#!/usr/bin/env python3
"""Build `deploy_contract.json` — from a training run dump, or control-only from a robot asset.

    # policy contract from a run (numbers come from params/env.yaml, agent.yaml, nn/*.pth)
    python3 deploy/policy_control/tools/build_deploy_contract.py --run logs/policy/left_v2B25 --grasp-band v1
    python3 deploy/policy_control/tools/build_deploy_contract.py --run logs/policy/right_g1 --gains <control_gains.yaml>
    # the same run re-based onto the 09.05 bimanual DG-5F-M asset (fabric URDF/params of that asset)
    python3 deploy/policy_control/tools/build_deploy_contract.py --run logs/policy/right_g1 --asset openarm_dg5f-m_bi_rl
    # bimanual pour (own schema): the sim meta is REQUIRED — it is the <trace_out>_meta.json of hdgp play.py
    #   (play.py --task open-short_b_pour_fab --checkpoint <pth> --trace_steps N --trace_out <path>)
    python3 deploy/policy_control/tools/build_deploy_contract.py --run logs/policy/pour_i11 --sim-meta <..._meta.json>
    #   -> logs/policy/pour_i11/pour_contract.json
    # control-only contract (no policy) for pd/fabric tests, one arm at a time
    python3 deploy/policy_control/tools/build_deploy_contract.py --asset openarm_dg5f-m_bi_rl --home zero \
        --out logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_control import contract as C  # noqa: E402
from policy_control.contract_assets import ASSETS, DEFAULT_ASSET, build_asset_contract  # noqa: E402
from policy_control import pour_contract as PC  # noqa: E402
from policy_control.contract_build import SIM_META_HOWTO, build_contract, build_pour, detect_family  # noqa: E402

POUR_OUT_NAME = "pour_contract.json"  # own schema: never the single-arm deploy_contract.json name
JOINT_OUT_NAME = "joint_contract.json"  # joint_direct family (arm joint increments + hand absolute, no fabric)
JOINT_DEPLOY_ASSET = "openarm_dg5f-m-short_bi_rl"   # 실기 손 = 20 관절 short; 학습 자산의 용접 관절은 계약이 붙잡는다


def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=None, help="run dir holding params/ and nn/ (omit for asset-only)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="explicit .pth (default: the only one in nn/)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <run>/deploy_contract.json (pour_bimanual: <run>/pour_contract.json)")
    ap.add_argument("--gains", type=Path, default=None, help="control_gains.yaml to compare trained kp/kd with")
    ap.add_argument("--grasp-band", default=None,
                    help="gripper_left only: v1 | v2 | lo,hi (table-height m). v2B25 was trained with v1")
    ap.add_argument("--asset", default=None, choices=sorted(ASSETS),
                    help=f"bind to an hdgp/assets/robot asset; without --run: control-only contract (default {DEFAULT_ASSET})")
    ap.add_argument("--sim-meta", type=Path, default=None,
                    help="pour_bimanual only (required): <trace_out>_meta.json from hdgp play.py --trace_out")
    ap.add_argument("--fill-default", type=float, default=None,
                    help="pour_bimanual only: source-cup fill level 0..1 used when no estimator publishes one")
    ap.add_argument("--sides", default="right,left", help="asset-only: sides to include (comma list)")
    ap.add_argument("--primary", default="right", help="asset-only: side mirrored into the legacy top-level sections")
    ap.add_argument("--home", default="zero", help="asset-only: zero | run:<run dir> (init_state, mirrored) | "
                         "pour:<pour_contract.json> (bimanual pour reset pose, arms + hands)")
    ap.add_argument("--hand-obs-order", default=None,
                    help="joint_direct only (required): <json list> of the 19/20 hand joint names in simulator order "
                         "(from a play trace meta 'joint_names'), or 'assumed' (20-joint order minus welded joints — "
                         "marks the contract so it cannot be verified)")
    ap.add_argument("--mirror-other-arm", action="store_true",
                    help="asset-only, run: 홈 전용 — 반대 팔 홈을 init_state 대신 부호 미러로(좌우 대칭)")
    args = ap.parse_args(argv)
    if args.run is None and args.out is None:
        ap.error("--out is required without --run")
    return args


def _main_pour(args) -> int:
    if args.sim_meta is None:
        raise SystemExit(f"[contract] pour_bimanual needs --sim-meta: {SIM_META_HOWTO}")
    c = build_pour(args.run, args.sim_meta, checkpoint=args.checkpoint, asset=args.asset,
                   fill_default=args.fill_default)
    out = args.out or (args.run / POUR_OUT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    PC.save_contract(c, out)
    print(f"[contract] {c.task} · pour_bimanual · obs {c.obs_dim} / act {c.action_dim} · "
          f"{c.policy_hz:.0f} Hz · asset {c.asset} → {out}")
    return 0


def _main_joint(args) -> int:
    import json

    from policy_control.joint_build import build_joint_contract
    from policy_control.joint_contract import save_contract

    spec = args.hand_obs_order
    if spec is None:
        raise SystemExit("[contract] joint_direct needs --hand-obs-order <json> | assumed (the simulator hand joint "
                         "order is not in the URDF — dump it with a play trace meta)")
    if spec == "assumed":
        order, source = None, "assumed: 20-joint PhysX order (pour i24 trace) minus welded joints"
    else:
        raw = json.loads(Path(spec).read_text())
        names = raw.get("joint_names", raw) if isinstance(raw, dict) else raw
        order, source = [n for n in names if "_hj_" in n], f"measured: {spec}"
    hdgp = Path(__file__).resolve().parents[4] / "hdgp"
    c = build_joint_contract(args.run, hdgp, args.asset or JOINT_DEPLOY_ASSET, hand_obs_order=order,
                             order_source=source, checkpoint=args.checkpoint)
    out = args.out or (args.run / JOINT_OUT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_contract(c, out)
    print(f"[contract] {c.task} · joint_direct · obs {c.obs_dim} / act {c.action_dim} · {c.policy_hz:.0f} Hz · "
          f"{c.side} · train {c.train_asset} → deploy {c.asset} · hand obs order {source.split(':')[0]} → {out}")
    return 0


def main(argv=None) -> int:
    args = _parse(argv)
    family = detect_family(args.run / "params/env.yaml") if args.run is not None else None
    if family == "pour_bimanual":
        return _main_pour(args)
    if family == "joint_direct":
        return _main_joint(args)
    if args.run is not None:
        c = build_contract(args.run, checkpoint=args.checkpoint, grasp_band=args.grasp_band, asset=args.asset)
        out = args.out or (args.run / "deploy_contract.json")
    else:
        c = build_asset_contract(args.asset or DEFAULT_ASSET, sides=tuple(args.sides.split(",")),
                                 primary=args.primary, home=args.home,
                                 mirror_other=args.mirror_other_arm)
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    C.save_contract(c, out)
    asset = c.asset.name if c.asset else "run asset (training-time fabric URDF)"
    print(f"[contract] {c.run.task} · obs {c.policy.obs_dim} / act {c.policy.action_dim} · "
          f"{c.rate.policy_hz:.0f} Hz · gravity {c.pd.gravity.mode} · sides {c.side_names} (primary {c.primary_side}) · "
          f"asset {asset} → {out}")
    if args.gains:
        return _report_gains(c, args.gains)
    return 0


def _report_gains(c: C.DeployContract, gains: Path) -> int:
    rc = 0
    for side in c.side_names:
        rep = C.compare_gains(c, gains, side=side)
        print(f"[contract] {side} gains {'OK' if rep.ok else 'MISMATCH'}"
              + ("" if rep.ok else ": " + "; ".join(rep.reasons))
              + (f"\n  kd note: {rep.kd_note}" if rep.kd_note else ""))
        rc = rc or (0 if rep.ok else 3)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
