"""pour_contract.json -> markdown section (generated; the source of truth is the contract JSON).

Used by tools/contract_doc.py, which dispatches on the JSON `schema` field.
"""
from __future__ import annotations

from policy_control import pour_contract as PC
from policy_control.pour_obs import segment_slices

RUNBOOK = "RUNBOOK_pour_bimanual.md"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _fmt(values, nd: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{nd}f}" for v in values) + "]"


def _layout_lines(c: PC.PourContract) -> list:
    out = [f"### obs {c.obs_dim} (generated from `pour_obs.segment_slices`)",
           _row(("segment", "slice", "width")), _row(("---",) * 3)]
    for name, sl in segment_slices(c).items():
        out.append(_row((f"`{name}`", f"[{sl.start}:{sl.stop}]", sl.stop - sl.start)))
    return out + [""]


def _side_lines(c: PC.PourContract) -> list:
    out = ["### sides", _row(("role", "side", "arm", "hand", "palm body", "tips", "fabric class", "robot_dir", "params")),
           _row(("---",) * 9)]
    for s in c.sides:
        out.append(_row((s.role, s.side, len(s.arm_joints), len(s.hand_joints), f"`{s.palm_body}`",
                         len(s.tip_bodies), s.fabric_class, s.fabric_robot_dir, s.fabric_params)))
    out.append("")
    for s in c.sides:
        out += [f"- `{s.role}` anchor (fabric frame, xyz+euler) {_fmt(s.anchor)} · fab_to_env {_fmt(s.fab_to_env)}",
                f"  - delta lo {_fmt(s.delta_lo)} / hi {_fmt(s.delta_hi)} · box {_fmt(s.box_lo)} .. {_fmt(s.box_hi)}",
                f"  - arm reset {_fmt(s.arm_reset)}"]
    return out + [""]


def render_pour(c: PC.PourContract, src: str = "") -> str:
    f, fl = c.fabric, c.fill_level
    out = [f"## family `{c.family}` - {c.task} - `{src or c.run_dir}`", "",
           f"- schema `{c.schema}` · asset `{c.asset}` · runbook [{RUNBOOK}]({RUNBOOK})",
           f"- checkpoint `{c.checkpoint}` md5 `{c.checkpoint_md5}`",
           f"- env.yaml sha1 `{c.env_yaml_sha1}` · agent.yaml sha1 `{c.agent_yaml_sha1}`",
           f"- rate: policy {c.policy_hz:.0f} Hz · fabric dt {c.fabric_dt:.5f} x decimation {c.fabric_decimation}",
           f"- policy: obs {c.obs_dim} / action {c.action_dim} · mlp {list(c.mlp_units)} · "
           f"normalize_input {c.normalize_input} · obs_clip {c.obs_clip} · action_clip {c.action_clip}", ""]
    out += _layout_lines(c)
    out += [f"### action {c.action_dim} = per role [palm 6, hand 3]",
            f"- hold_steps {c.hold_steps} · palm_ema_alpha {c.palm_ema_alpha} · hand_action_mode `{c.hand_action_mode}`",
            f"- synergy_close_speed {c.synergy_close_speed} · contact_freeze {c.synergy_contact_freeze} "
            f"(threshold {c.contact_force_threshold} N)",
            f"- close gate enabled {c.close_gate_enabled} · radius {c.close_gate_radius} · ramp {c.close_gate_ramp}",
            f"- cup_mouth_z {c.cup_mouth_z} · joint_pos_err_max {c.joint_pos_err_max}", "",
            "### fill_level", f"- source `{fl.source}` · range [{fl.lo}, {fl.hi}] · default {fl.default}",
            f"- {fl.note}", "",
            "### fabric (one shared world)",
            f"- damping {f.damping} · vel_ff {f.vel_ff_scale} · max_objects {f.max_objects} · "
            f"hand_repulsion {f.use_hand_repulsion} · body_repulsion_pairs {f.use_body_repulsion_pairs}",
            f"- table obstacle {f.table_obstacle} · margin_xy {f.table_margin_xy} · thickness {f.table_thickness} · "
            f"table_z {f.table_z}", ""]
    return "\n".join(out + _side_lines(c))
