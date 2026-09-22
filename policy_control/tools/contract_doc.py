#!/usr/bin/env python3
"""deploy_contract.json → 사람이 읽는 계약 문서(markdown). 문서는 생성물이지 원본이 아니다.

    python3 policy_control/tools/contract_doc.py logs/policy/left_v2B25/deploy_contract.json \
        logs/policy/right_g1/deploy_contract.json --out docs/CONTRACT_policy_control.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_control import contract as C  # noqa: E402
from policy_control import pour_contract as PC  # noqa: E402
from contract_doc_pour import RUNBOOK, render_pour  # noqa: E402

HAND_WRITTEN = ["## hand-written companions (not generated, never overwritten by this tool)", "",
                f"- [{RUNBOOK}]({RUNBOOK}): bimanual pour (`pour_bimanual`), files to drop in, commands, "
                "fill_level meaning, remaining hardware calibrations", ""]


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render(c: C.DeployContract, src: str = "") -> str:
    # 제목은 계약 **파일** 로 구분한다. run.task/run.dir 만 쓰면 같은 런의 변종 계약
    # (right_g1 의 deploy_contract.json vs .dg5f-m.json)과 asset 계약들이 같은 제목으로 겹친다.
    out = [f"## {c.run.task} — `{src or c.run.dir}`", "",
           f"- checkpoint `{c.run.checkpoint}` md5 `{c.run.checkpoint_md5}` · experiment `{c.run.experiment}`",
           f"- rate: policy {c.rate.policy_hz:.0f} Hz (step_dt {c.rate.step_dt:.5f} s) · episode {c.rate.episode_steps} steps",
           f"- policy: obs {c.policy.obs_dim} / action {c.policy.action_dim} · rnn {c.policy.rnn or 'none'} · "
           f"mlp {c.policy.mlp_units} · action_clip {c.policy.action_clip} · obs_clip {c.policy.obs_clip}", "",
           "### obs segments", _row(("#", "name", "dim [offset]", "builder", "params")), _row(("---",) * 5)]
    off = 0
    for i, s in enumerate(c.obs.segments):
        params = ", ".join(f"{k}={v}" for k, v in s.params.items() if k != "default")
        out.append(_row((i, f"`{s.name}`", f"{s.dim} [{off}:{off + s.dim}]", s.builder, params[:120])))
        off += s.dim
    out += ["", "joint orders: " + ", ".join(f"{k}={len(v)}" for k, v in c.obs.joint_orders.items()),
            f"fk: {c.obs.fk}", "", "### action", _row(("group", "slice")), _row(("---", "---"))]
    out += [_row((g.name, g.slice)) for g in c.action.groups]
    out += _action_lines(c) + ["", "### fabric",
            f"- {c.fabric.class_name} · {c.fabric.robot_dir} · {c.fabric.params} · world {c.fabric.world}",
            f"- dt {c.fabric.dt:.5f} × decimation {c.fabric.decimation} · damping {c.fabric.damping} · "
            f"vel_ff {c.fabric.vel_ff_scale} · hand_sync {c.fabric.hand_sync} · table_z {c.fabric.table_z} · "
            f"body_repulsion_pairs {c.fabric.use_body_repulsion_pairs}",
            "", "### pd",
            f"- groups {c.pd.groups} · gravity `{c.pd.gravity.mode}`"
            + (f" gain {c.pd.gravity.gain} limit {c.pd.gravity.limit}" if c.pd.gravity.gain else "")
            + f" · sim gravity disabled {c.pd.gravity.sim_gravity_disabled}",
            f"- trained gains kp {c.pd.sim_gains.kp} / kd {c.pd.sim_gains.kd}",
            f"- home arm {[round(v, 4) for v in c.pd.home_arm]}", ""]
    out += _sides_lines(c)
    return "\n".join(out)


def _action_lines(c: C.DeployContract) -> list:
    p = c.action.palm
    if p is None:
        return ["", "- control-only contract: no policy, no action decoders (fabric takes palm_cmd / hand_cmd)"]
    palm = f"- palm `{p.convention}` box {p.box_lo}–{p.box_hi} · pos rate {p.pos_rate_limit}"
    if p.euler_center:
        palm += f" · euler_center {p.euler_center} · max_pose_angle {p.max_pose_angle}"
    if p.delta_xyz:
        palm += f" · delta {p.delta_xyz}/{p.delta_rot_deg}° · anchor {p.anchor}"
    hand = ", ".join(f"{k}={v}" for k, v in c.action.hand.params.items()
                     if k not in ("open_pose", "grip_pose", "soft_limits"))
    return ["", palm, f"- hand `{c.action.hand.decoder}` joints {len(c.action.hand.joints)} · {hand[:300]}"]


def _sides_lines(c: C.DeployContract) -> list:
    asset = f"`{c.asset.name}` ({c.asset.ee_kind}) urdf `{c.asset.urdf}`" if c.asset else "run asset (training-time fabric URDF)"
    out = ["### sides (v2)", f"- asset {asset} · primary `{c.primary_side}` · control_only {c.control_only}", "",
           _row(("side", "ee", "hand joints", "palm body", "tips", "pd groups", "fabric", "action groups", "gravity")),
           _row(("---",) * 9)]
    for name in c.side_names:
        s = c.side(name)
        fab = f"{s.fabric.class_name} / {s.fabric.robot_dir}" if s.fabric else "none"
        out.append(_row((name, s.ee_kind, len(s.hand_joints), f"`{s.palm_body}`", len(s.tip_bodies), s.pd_groups,
                         fab, s.action_groups, s.gravity.mode)))
    return out + [""]


def _rel(path: Path) -> str:
    """저장소 기준 상대 경로 — 제목의 유일 키이자 재생성 명령에 그대로 쓸 수 있는 인자."""
    repo = Path(__file__).resolve().parents[2]
    try:
        return str(Path(path).resolve().relative_to(repo))
    except ValueError:
        return str(path)


def render_any(path: Path) -> str:
    """Dispatch on the JSON `schema`: pour_contract/v1 -> pour section, anything else -> single-arm contract."""
    path = Path(path)
    try:
        schema = json.loads(path.read_text()).get("schema")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise SystemExit(f"[contract_doc] cannot read {path}: {exc}") from exc
    src = _rel(path)
    if schema == PC.SCHEMA:
        return render_pour(PC.load_contract(path), src)
    return render(C.load_contract(path), src)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("contracts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cmd = "python3 policy_control/tools/contract_doc.py \\\n    " + " \\\n    ".join(
        _rel(p) for p in args.contracts) + f" \\\n    --out {_rel(args.out)}"
    parts = ["# policy_control 배포 계약(생성물 — 원본은 deploy_contract.json)", "",
             "obs → policy → fabric → pd 4노드가 읽는 계약을 사람이 읽을 수 있게 펼친 것. "
             "수정은 `tools/build_deploy_contract.py` 로 계약을 다시 만들고 이 문서를 재생성한다.", "",
             "이 문서를 만든 명령(그대로 다시 치면 재생성된다):", "", "```bash", cmd, "```", ""]
    parts += [render_any(p) for p in args.contracts] + HAND_WRITTEN
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"[contract_doc] {len(args.contracts)} contracts → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
