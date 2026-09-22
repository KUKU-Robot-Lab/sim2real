"""Deploy contract for the bimanual pour family (`pour_bimanual`, hdgp task open-short_b_pour_fab).

Separate from `DeployContract` on purpose: that schema carries ONE obs/action/fabric section and
its nodes resolve one side. The pour policy owns both arms in a single 223/18 vector, so it gets
its own frozen contract; `contract_build.detect_family` recognises the run and points here.

Sources of every number:
  env.yaml / agent.yaml of the run  -> rates, clips, EMA alpha, delta boxes, gates, noise, network
  hdgp robot profile (pair "short") -> joint names, synergy poses/channels, palm box, bodies
  asset URDF                        -> hand joint limits (sim soft limits, factor from env.yaml)
  sim meta (trace_meta.json)        -> PhysX DOF order, fabric-FK anchors, fab_to_env offsets.
     These three are computed inside Isaac at env init and are NOT in the run dump.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "policy_control/pour_contract/v1"
FAMILY = "pour_bimanual"
ROLES = ("src", "rcv")


class PourContractError(ValueError):
    """The run dump, profile or sim meta is inconsistent with the pour family."""


@dataclass(frozen=True)
class PourSideCfg:
    role: str                 # src | rcv
    side: str                 # right | left
    arm_joints: tuple         # canonical arm order (== sim arm_ids order, checked at build)
    hand_obs_joints: tuple    # hand joints in PhysX articulation DOF order (obs hand_q order)
    hand_joints: tuple        # finger-major order (synergy targets, fabric hand order)
    palm_body: str
    tip_bodies: tuple
    fingers: tuple
    thumb: str
    grp_a: tuple                 # finger indices of the opposing contact group (thumb side)
    grp_b: tuple                 # finger indices of the other contact group
    joint_finger: tuple       # per hand_joints entry: index into fingers
    joint_channel: tuple      # per hand_joints entry: 0/1/2
    joint_is_mid: tuple
    joint_is_dist: tuple
    arm_reset: tuple          # per arm_joints entry: episode reset pose (profile.arm_reset_joint_pos)
    hand_open: tuple
    hand_grip: tuple          # oppose_grip_delta already applied
    hand_lo: tuple
    hand_hi: tuple
    anchor: tuple             # (6,) fabric-frame palm pose at the reset posture
    fab_to_env: tuple         # (3,) env palm pos - fabric palm pos at reset
    delta_lo: tuple           # (6,) m / rad
    delta_hi: tuple
    box_lo: tuple             # (3,) env frame
    box_hi: tuple
    fabric_joint_order: tuple
    fabric_class: str
    fabric_robot_dir: str
    fabric_params: str


@dataclass(frozen=True)
class FillLevelCfg:
    source: str               # always "external": operator / estimator input, no sensor exists
    lo: float
    hi: float
    default: float | None     # None = refuse to run without a value (never a silent 0)
    note: str


@dataclass(frozen=True)
class PourFabricCfg:
    """Env-level fabric settings shared by both sides (one world: the table box)."""

    damping: float
    max_objects: int
    vel_ff_scale: float
    use_hand_repulsion: bool
    use_body_repulsion_pairs: bool
    table_obstacle: bool
    table_margin_xy: float
    table_thickness: float
    table_z: float


@dataclass(frozen=True)
class PourContract:
    schema: str
    family: str
    task: str
    run_dir: str
    checkpoint: str
    checkpoint_md5: str
    env_yaml_sha1: str
    agent_yaml_sha1: str
    asset: str
    policy_hz: float
    fabric_dt: float
    fabric_decimation: int
    obs_dim: int
    action_dim: int
    mlp_units: tuple
    normalize_input: bool
    obs_clip: float | None    # rl_games wrapper clip_observations; applied before the network
    action_clip: float
    hold_steps: int
    palm_ema_alpha: float
    cup_mouth_z: float
    joint_pos_err_max: float
    hand_action_mode: str
    synergy_close_speed: float
    synergy_contact_freeze: bool
    contact_force_threshold: float
    close_gate_enabled: bool
    close_gate_radius: float
    close_gate_ramp: float
    fill_level: FillLevelCfg
    fabric: PourFabricCfg
    sides: tuple              # (PourSideCfg src, PourSideCfg rcv)

    @property
    def roles(self) -> tuple:
        return tuple(s.role for s in self.sides)

    def side(self, role: str) -> PourSideCfg:
        for s in self.sides:
            if s.role == role:
                return s
        raise PourContractError(f"no role {role!r}; have {self.roles}")



def side_block_dim(s: PourSideCfg) -> int:
    """arm_q, arm_qd, hand_q, palm pos 3 + rot6, tips-palm, cup-palm, tips-cup, joint_err, cup_up."""
    na, nh, nt = len(s.arm_joints), len(s.hand_obs_joints), len(s.tip_bodies)
    return 2 * na + nh + 9 + 3 * nt + 3 + 3 * nt + nh + 3


def validate(c: PourContract) -> PourContract:
    if c.schema != SCHEMA or c.family != FAMILY:
        raise PourContractError(f"schema/family {c.schema!r}/{c.family!r} is not {SCHEMA}/{FAMILY}")
    if c.roles != ROLES:
        raise PourContractError(f"sides must be {ROLES}, got {c.roles}")
    if c.hand_action_mode != "grip3":
        raise PourContractError(f"only hand_action_mode grip3 is supported, run has {c.hand_action_mode!r}")
    want_act = len(c.sides) * 9
    common = 3 + 3 + 1 + want_act
    want_obs = sum(side_block_dim(s) for s in c.sides) + common
    if c.action_dim != want_act or c.obs_dim != want_obs:
        raise PourContractError(f"dims {c.obs_dim}/{c.action_dim} != layout {want_obs}/{want_act}")
    if not 0.0 < c.palm_ema_alpha <= 1.0:
        raise PourContractError(f"palm_ema_alpha {c.palm_ema_alpha} outside (0, 1]")
    if c.fabric_dt <= 0.0 or c.fabric_decimation < 1:
        raise PourContractError(f"fabric_dt/decimation {c.fabric_dt}/{c.fabric_decimation} invalid")
    if c.fabric.table_obstacle and (c.fabric.table_thickness <= 0.0 or c.fabric.table_margin_xy < 0.0):
        raise PourContractError("fabric table box needs thickness > 0 and margin_xy >= 0")
    if c.fabric.max_objects < 1 or c.fabric.damping < 0.0:
        raise PourContractError("fabric max_objects must be >= 1 and damping >= 0")
    if c.fill_level.source != "external":
        raise PourContractError("fill_level.source must be 'external'")
    for s in c.sides:
        n = len(s.hand_joints)
        if sorted(s.hand_obs_joints) != sorted(s.hand_joints):
            raise PourContractError(f"{s.role}: obs and synergy hand joint sets differ")
        for f in ("joint_finger", "joint_channel", "joint_is_mid", "joint_is_dist",
                  "hand_open", "hand_grip", "hand_lo", "hand_hi"):
            if len(getattr(s, f)) != n:
                raise PourContractError(f"{s.role}.{f}: {len(getattr(s, f))} values for {n} hand joints")
        if len(s.arm_reset) != len(s.arm_joints):
            raise PourContractError(f"{s.role}.arm_reset: {len(s.arm_reset)} values for {len(s.arm_joints)} arm joints")
        for f, k in (("anchor", 6), ("delta_lo", 6), ("delta_hi", 6), ("fab_to_env", 3), ("box_lo", 3), ("box_hi", 3)):
            if len(getattr(s, f)) != k:
                raise PourContractError(f"{s.role}.{f} must have {k} values")
        if not s.grp_a or not s.grp_b or max(tuple(s.grp_a) + tuple(s.grp_b)) >= len(s.fingers):
            raise PourContractError(f"{s.role}: contact groups {s.grp_a}/{s.grp_b} do not index fingers")
        if s.thumb not in s.fingers:
            raise PourContractError(f"{s.role}: thumb {s.thumb!r} not in fingers")
    return c


def _tuplify(v):
    if isinstance(v, list):
        return tuple(_tuplify(x) for x in v)
    return v


def to_dict(c: PourContract) -> dict:
    return dataclasses.asdict(c)


def from_dict(raw: dict) -> PourContract:
    try:
        sides = tuple(PourSideCfg(**{k: _tuplify(v) for k, v in s.items()}) for s in raw["sides"])
        body = {k: _tuplify(v) for k, v in raw.items() if k not in ("sides", "fill_level", "fabric")}
        return validate(PourContract(**body, fill_level=FillLevelCfg(**raw["fill_level"]),
                                     fabric=PourFabricCfg(**raw["fabric"]), sides=sides))
    except (KeyError, TypeError) as exc:
        raise PourContractError(f"malformed pour contract: {exc}") from exc


def save_contract(c: PourContract, path: Path) -> None:
    Path(path).write_text(json.dumps(to_dict(validate(c)), indent=1, ensure_ascii=False) + "\n")


def load_contract(path: Path) -> PourContract:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PourContractError(f"cannot read {path}: {exc}") from exc
    return from_dict(raw)


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
