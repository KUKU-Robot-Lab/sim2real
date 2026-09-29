#!/usr/bin/env python3
"""RH56F1 양팔 붓기(hdgp open-rh_b_pour_fj) 런 → pour_fj 계약(policy_control/pour_fj.py). 09.29 사용자: 곧 나올 정책 미리 준비.

    python3 tools/build_pour_fj_contract.py --run deploy/policies/both_rh_pour_fjXX [--checkpoint nn/<pth>] \\
        [--hand-obs-order <trace_meta.json>] [--out <run>/pour_fj_contract.json]

--hand-obs-order 가 없으면 손 관측 순서는 **추정**(URDF 너비 우선 · 알파벳)으로 적고 계약에 assumed 라고 남긴다 —
배포 등록부가 verified 로 올리지 못한다. trace_meta.json 은 hdgp play --trace_out 이 남긴 것으로,
`hand_joint_names: {src: [...], rcv: [...]}`(robot.joint_names[hand_ids]) 를 담아야 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402
from policy_control import pour_fj as F  # noqa: E402
from policy_control.pour_profiles import load_mimic_pair  # noqa: E402

ASSET = "openarm_rh56f1_bi_rl"
URDF = _paths.RL_WS / "hdgp" / "assets" / "robot" / ASSET / f"{ASSET}.urdf"


def _checkpoint(run: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else run / p
    found = sorted((run / "nn").glob("*.pth"))
    if len(found) != 1:
        raise SystemExit(f"{run}/nn 에 .pth 가 {len(found)} 개 — --checkpoint 로 고를 것")
    return found[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="params/{env,agent}.yaml 이 있는 런 폴더")
    ap.add_argument("--checkpoint", help="nn/ 에 하나가 아니면 명시")
    ap.add_argument("--hand-obs-order", help="trace_meta.json — 손 관측(PhysX) 순서 실측")
    ap.add_argument("--hdgp", default=str(_paths.RL_WS / "hdgp"))
    ap.add_argument("--out", help="기본 <run>/pour_fj_contract.json")
    args = ap.parse_args(argv)
    run = Path(args.run).resolve()
    order, source = None, ""
    if args.hand_obs_order:
        meta = json.loads(Path(args.hand_obs_order).read_text())
        order = meta.get("hand_joint_names")
        if not order or set(order) != set(F.ROLES):
            raise SystemExit(f"{args.hand_obs_order}: hand_joint_names {{src: [...], rcv: [...]}} 가 없다")
        source = f"measured:{Path(args.hand_obs_order).name}"
    c = F.build(run, _checkpoint(run, args.checkpoint), load_mimic_pair(Path(args.hdgp)), URDF, asset=ASSET,
                hand_obs_order=order, obs_order_source=source)
    out = Path(args.out) if args.out else run / "pour_fj_contract.json"
    out.write_text(c.to_json())
    print(f"[pour_fj] {c.task} · obs {c.obs_dim} / act {c.action_dim} · {c.policy_hz:.0f} Hz · 팔 {c.arm_mode} · "
          f"손 {c.hand_range}{' · 동결' if c.hand_freeze else ''} · hold {c.hold_steps} · 손 관측 순서 {c.hand_obs_order_source} → {out}")
    for n in c.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
