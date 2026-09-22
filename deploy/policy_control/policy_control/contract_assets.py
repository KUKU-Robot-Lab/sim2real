"""Robot assets a contract can be bound to — ``hdgp/assets/robot/<name>`` (09.05 line-up).

Two entry points:

* ``build_asset_contract`` — a **control-only** contract (no policy) straight from
  the asset manifest: both arms as ``sides``, fabric + pd per side, homes and
  gains from data files. This is what the one-arm-at-a-time pd/fabric tests run
  before any policy exists for the asset.
* ``bind_asset`` — re-base a run-built contract onto an asset: fabric URDF/params
  of that asset, soft limits from its URDF, per-side bodies from its manifest.

The default asset is the bimanual DG-5F-M (``DEFAULT_ASSET``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from . import _paths
from .contract import (SCHEMA, ActionCfg, AssetInfo, ContractError, DeployContract, FabricCfg, Gains,
                       GravityCfg, ObsCfg, PdCfg, PolicyCfg, RateCfg, RunInfo, SideCfg, file_sha1,
                       from_dict, legacy_sides, load_driver_gains, to_dict, validate)

RL_WS = _paths.RL_WS
ASSET_ROOT = RL_WS / "hdgp/assets/robot"
DEFAULT_ASSET = "openarm_dg5f-m_bi_rl"
GAINS_YAML = RL_WS / "urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
#: control-only cadence: the grasp_s2r fabric loop (60 Hz, 2 sub-steps, damping 10) — the same numbers
#: every DG-5F run dump carried; a policy contract overrides them from its own dump.
CONTROL_FABRIC_DT = 1.0 / 60.0
CONTROL_FABRIC_DECIMATION = 2
CONTROL_FABRIC_DAMPING = 10.0
CONTROL_MAX_OBJECTS = 8


@dataclass(frozen=True)
class FabricSpec:
    class_name: str
    robot_dir: str
    params: str
    world: dict


@dataclass(frozen=True)
class AssetSpec:
    name: str
    ee_kind: str                          # dg5f | gripper | rh56f1 | none
    fabric: dict                          # side → FabricSpec | None

    @property
    def dir(self) -> Path:
        return ASSET_ROOT / self.name

    @property
    def urdf(self) -> Path:
        return self.dir / f"{self.name}.urdf"

    @property
    def manifest(self) -> Path:
        return self.dir / f"{self.name}_manifest.yaml"


def _tesollo(side: str, family: str) -> FabricSpec:
    cls = "OpenArmTeoslloPoseFabric" if side == "right" else "OpenArmTeoslloLeftPoseFabric"
    world = "open_tesollo_boxes_no_table" if side == "right" else "open_tesollo_left_boxes_no_table"
    return FabricSpec(cls, f"openarm_{family}_bi_{side}", f"openarm_{family}_{side}_pose_params.yaml",
                      {"filename": world})


ASSETS = {
    "openarm_dg5f-m_bi_rl": AssetSpec("openarm_dg5f-m_bi_rl", "dg5f",
                                      {"right": _tesollo("right", "dg5f-m"), "left": _tesollo("left", "dg5f-m")}),
    # DG-5F short base: same hand, mount/base 47.8mm shorter. Identical joint names and
    # control_joint_order, so ee_kind "dg5f" and every name-derived field carry over.
    "openarm_dg5f-m-short_bi_rl": AssetSpec(
        "openarm_dg5f-m-short_bi_rl", "dg5f",
        {"right": _tesollo("right", "dg5f-m-short"), "left": _tesollo("left", "dg5f-m-short")}),
    "openarm_dg5f-s_bi_rl": AssetSpec("openarm_dg5f-s_bi_rl", "dg5f",
                                      {"right": _tesollo("right", "dg5f-s"), "left": _tesollo("left", "dg5f-s")}),
    "openarm_gripper_bi_rl": AssetSpec("openarm_gripper_bi_rl", "gripper", {
        "right": None,                    # no right-gripper fabric class exists (left-only track)
        "left": FabricSpec("OpenArmGripperLeftPoseFabric", "openarm_gripper_bi_left",
                           "openarm_gripper_left_pose_params.yaml", {"filename": "open_gripper_left_boxes_no_table"})}),
    "openarm_rh56f1_bi_rl": AssetSpec("openarm_rh56f1_bi_rl", "rh56f1", {"right": None, "left": None}),
}


def asset_spec(name: str) -> AssetSpec:
    try:
        spec = ASSETS[name]
    except KeyError:
        raise ContractError(f"unknown asset {name!r}; known: {sorted(ASSETS)}") from None
    for p in (spec.urdf, spec.manifest):
        if not p.exists():
            raise ContractError(f"asset {name}: missing {p}")
    return spec


def load_manifest(spec: AssetSpec) -> dict:
    data = yaml.safe_load(spec.manifest.read_text())
    for key in ("control_joint_order", "link_order"):
        if key not in data:
            raise ContractError(f"{spec.manifest}: no {key}")
    return data


# ------------------------------------------------------------------ manifest → per-side names
def side_joints(manifest: dict, side: str) -> tuple[list, list]:
    """(arm joints, hand joints) of one side in the asset's control order."""
    p = side[0]
    order = list(manifest["control_joint_order"])
    arm = [j for j in order if j.startswith(f"{p}_aj_")]
    hand = [j for j in order if j.startswith(f"{p}_hj_")]
    if len(arm) != 7:
        raise ContractError(f"manifest: side {side} has {len(arm)} arm joints, expected 7")
    return arm, hand


def side_bodies(manifest: dict, side: str, ee_kind: str) -> tuple[str, list]:
    """(palm link, fingertip links) of one side, checked against the manifest link_order."""
    p = side[0]
    links = set(manifest["link_order"])
    palm = f"{p}_hl_palm" if ee_kind in ("dg5f", "rh56f1") else f"{p}_hl_gripper_base"
    if ee_kind == "none":
        palm = f"{p}_al_7"
    tips = [f"{p}_hl_{f}_tip" for f in FINGERS] if ee_kind in ("dg5f", "rh56f1") else []
    missing = [b for b in [palm, *tips] if b not in links]
    if missing:
        raise ContractError(f"manifest link_order lacks {missing} for side {side} ({ee_kind})")
    return palm, tips


def pd_groups_for(side: str, ee_kind: str) -> list:
    tail = {"dg5f": "hand", "rh56f1": "hand", "gripper": "gripper"}.get(ee_kind)
    return [f"{side}_arm"] + ([f"{side}_{tail}"] if tail else [])


# ------------------------------------------------------------------ homes (data sources only)
def _mirror_signs() -> tuple[list, list]:
    """(arm, hand) mirror signs right→left, from the tesollo left preset (FK-verified 07-28)."""
    from openarm.tesollo.left.grasp_v1 import grasp_left_preset as P
    return [float(s) for s in P._ARM_SIGN], [float(s) for s in P._HAND_SIGN]


POUR_HOME = "pour:"


def _load_pour(mode: str):
    """The pour contract behind a ``pour:<pour_contract.json>`` home (path relative to sim2real)."""
    from . import pour_contract as PC
    path = Path(mode[len(POUR_HOME):])
    if not path.is_absolute():
        path = _paths.SIM2REAL / path
    if not path.is_file():
        raise ContractError(f"pour home: pour contract not found ({path})")
    try:
        return PC.load_contract(path), path
    except PC.PourContractError as exc:
        raise ContractError(f"pour home: {path} is not a pour contract: {exc}") from exc


def pour_homes(pour, sides: tuple) -> tuple[dict, dict]:
    """(arm homes, hand homes) per side = the pour contract's reset pose (``arm_reset`` / ``hand_open``)."""
    by_side = {s.side: s for s in pour.sides}
    missing = [s for s in sides if s not in by_side]
    if missing:
        raise ContractError(f"pour contract has no side {missing}; has {sorted(by_side)}")
    arm = {s: [float(v) for v in by_side[s].arm_reset] for s in sides}
    hand = {s: {j: float(v) for j, v in zip(by_side[s].hand_joints, by_side[s].hand_open)} for s in sides}
    return arm, hand


def arm_homes(mode: str, sides: tuple) -> tuple[dict, str]:
    """``zero`` (차렷, all joints 0), ``run:<dir>`` (that run's init_state, mirrored to the other arm) or
    ``pour:<pour_contract.json>`` (the bimanual pour reset pose, both arms explicit)."""
    if mode == "zero":
        return {s: [0.0] * 7 for s in sides}, "zero (차렷)"
    if mode.startswith(POUR_HOME):
        pour, path = _load_pour(mode)
        return pour_homes(pour, sides)[0], f"pour:{path.name} arm_reset ({pour.run_dir})"
    if not mode.startswith("run:"):
        raise ContractError(f"home must be 'zero', 'run:<run dir>' or 'pour:<pour_contract.json>', got {mode!r}")
    from .contract_build import _home_values, _text, detect_family
    run = Path(mode[4:])
    if not run.is_absolute():
        run = _paths.SIM2REAL / run
    env_text = _text(run / "params/env.yaml")
    src_side = "left" if detect_family(run / "params/env.yaml") == "gripper_left" else "right"
    home = _reset_override(env_text) or _home_values(env_text, [f"{src_side[0]}_aj_{i}" for i in range(1, 8)])
    src_how = "arm_reset_joint_pos_override" if _reset_override(env_text) else "init_state"
    sign, _ = _mirror_signs()
    out, how = {}, []
    for s in sides:
        if s == src_side:
            out[s] = home
            continue
        try:                                        # 반대 팔도 init_state 에 있으면 그 값 — 정책 환경과 같아야 한다(09.22)
            out[s] = _home_values(env_text, [f"{s[0]}_aj_{i}" for i in range(1, 8)])
            how.append(f"{s} = init_state")
        except SystemExit:
            out[s] = [g * v for g, v in zip(sign, home)]
            how.append(f"{s} = _ARM_SIGN mirror")
    return out, f"run:{run.name} ({src_side} = {src_how}; {', '.join(how) or 'one arm'})"


def _reset_override(env_text: str) -> list[float] | None:
    """정책 팔의 **에피소드 시작 자세** — env 가 리셋마다 쓰는 `arm_reset_joint_pos_override`(7개). 없거나 비면 None.

    09.22: 계약 홈을 init_state 로 잡았더니 정책이 실제로 본 시작 자세(grasp_fj_env.py:115)와 달랐다. 사용자 결정 —
    팔이 먼저 갈 곳은 에피소드 리셋 자세다. 반대 팔은 리셋에서 건드리지 않으므로 init_state 그대로다.
    """
    m = re.search(r"^arm_reset_joint_pos_override:[^\n]*\n((?:- [^\n]+\n)+)", env_text, re.M)
    if m is None:
        return None
    vals = [float(v) for v in re.findall(r"^- (-?[0-9.eE+]+)\s*$", m.group(1), re.M)]
    if not vals:
        return None
    if len(vals) != 7:
        raise ContractError(f"arm_reset_joint_pos_override 가 7개가 아니다: {vals}")
    return vals


def hand_home(ee_kind: str, side: str, hand_joints: list) -> dict:
    if ee_kind == "dg5f":
        from grasp_s2r_synergy import HAND_JOINT_NAMES, HAND_OPEN_POSE
        right = dict(zip(HAND_JOINT_NAMES, HAND_OPEN_POSE))
        if side == "right":
            vals = right
        else:
            _, sign = _mirror_signs()
            vals = {f"l{n[1:]}": s * v for (n, v), s in zip(right.items(), sign)}
        missing = [j for j in hand_joints if j not in vals]
        if missing:
            raise ContractError(f"hand open pose lacks {missing}")
        return {j: float(vals[j]) for j in hand_joints}
    if ee_kind == "gripper":
        from openarm.gripper.left.grasp_sensor import grasp_left_preset as P
        return {j: float(P.GRIPPER_OPEN_POS) for j in hand_joints}
    return {j: 0.0 for j in hand_joints}


def driver_gains(gains_yaml: Path, arm_joints: list) -> Gains:
    real = load_driver_gains(gains_yaml)
    return Gains(joints=list(arm_joints), kp=[real[i][0] for i in range(1, 8)], kd=[real[i][1] for i in range(1, 8)])


# ------------------------------------------------------------------ control-only contract
def _asset_info(spec: AssetSpec) -> AssetInfo:
    return AssetInfo(name=spec.name, urdf=str(spec.urdf.relative_to(RL_WS)),
                     manifest=str(spec.manifest.relative_to(RL_WS)), manifest_sha1=file_sha1(spec.manifest),
                     ee_kind=spec.ee_kind)


def _side_fabric(spec: AssetSpec, side: str, arm: list, hand: list, home_arm: list, home_hand: dict,
                 home_source: str) -> FabricCfg | None:
    fs = spec.fabric.get(side)
    if fs is None:
        return None
    fabric_hand = hand if spec.ee_kind == "dg5f" else []     # the gripper is not a fabric joint
    return FabricCfg(class_name=fs.class_name, robot_dir=fs.robot_dir, params=fs.params, world=dict(fs.world),
                     dt=CONTROL_FABRIC_DT, decimation=CONTROL_FABRIC_DECIMATION, damping=CONTROL_FABRIC_DAMPING,
                     max_objects=CONTROL_MAX_OBJECTS, joint_order=arm + fabric_hand,
                     home_q=home_arm + [home_hand[j] for j in fabric_hand], vel_ff_scale=1.0,
                     hand_vel_ff_scale=1.0 if fabric_hand else None, hand_sync="syn_target" if fabric_hand else None,
                     use_body_repulsion_pairs=bool(fabric_hand), home_source=home_source)


def _hand_override(home_hand: dict, hand_joints: list) -> dict:
    missing = [j for j in hand_joints if j not in home_hand]
    if missing:
        raise ContractError(f"pour hand_open lacks {missing}")
    return {j: float(home_hand[j]) for j in hand_joints}


def _pour_hand_homes(home: str, asset: str, sides: tuple) -> dict:
    """Hand homes of a ``pour:`` home (same asset only); {} for every other home mode."""
    if not home.startswith(POUR_HOME):
        return {}
    pour, path = _load_pour(home)
    if pour.asset != asset:
        raise ContractError(f"pour contract {path.name} is for asset {pour.asset!r}, not {asset!r}")
    return pour_homes(pour, sides)[1]


def _run_hand_homes(home: str, spec: AssetSpec, manifest: dict, sides: tuple) -> dict:
    """``run:`` 홈의 손 = 그 런의 init_state 손 값(정책의 첫 상태와 같아야 한다, 09.22 사용자).

    init_state 에 없는 손 관절(잠근 관절 등)은 기본 open pose 값을 쓴다. 반대 팔은 손 부호로 미러한다.
    다른 홈 모드면 {} — 호출자가 기본 open pose 를 쓴다.
    """
    if not home.startswith("run:"):
        return {}
    from .contract_build import _block, _robot_block, _text, detect_family
    run = Path(home[4:])
    if not run.is_absolute():
        run = _paths.SIM2REAL / run
    env_path = run / "params/env.yaml"
    src = "left" if detect_family(env_path) == "gripper_left" else "right"
    joint_pos = _block(_robot_block(_text(env_path)), r"^\s*joint_pos:\s*$")
    out = {}
    for side in sides:
        hand = side_joints(manifest, side)[1]
        base = hand_home(spec.ee_kind, side, hand)                      # 없는 관절의 자리
        vals = {}
        for j in hand:
            own = re.search(rf"^\s*{re.escape(j)}:\s*(-?[0-9.eE+]+)\s*$", joint_pos, re.M)
            if own is not None:                                         # 그 팔의 값이 있으면 그대로
                vals[j] = float(own.group(1))
                continue
            name = f"{src[0]}{j[1:]}"                                   # 없으면 원본 팔 값을 미러
            m = re.search(rf"^\s*{re.escape(name)}:\s*(-?[0-9.eE+]+)\s*$", joint_pos, re.M)
            if m is None:
                vals[j] = base[j]
                continue
            v = float(m.group(1))
            if side != src:                                             # 부호표는 오른손 관절 순서(HAND_JOINT_NAMES)다
                from grasp_s2r_synergy import HAND_JOINT_NAMES
                _, sign = _mirror_signs()
                v *= sign[list(HAND_JOINT_NAMES).index(f"r{name[1:]}")]
            vals[j] = v
        out[side] = vals
    return out


def _control_side(spec: AssetSpec, manifest: dict, side: str, home_arm: list, home_source: str,
                  gains_yaml: Path, home_hand: dict | None = None) -> SideCfg:
    arm, hand = side_joints(manifest, side)
    palm, tips = side_bodies(manifest, side, spec.ee_kind)
    home_h = hand_home(spec.ee_kind, side, hand) if home_hand is None else _hand_override(home_hand, hand)
    return SideCfg(side=side, arm_joints=arm, hand_joints=hand, ee_kind=spec.ee_kind, palm_body=palm,
                   tip_bodies=tips, home_arm=list(home_arm), home_hand=home_h,
                   pd_groups=pd_groups_for(side, spec.ee_kind),
                   # ★`contract_build._gravity_mode` 규칙("sim 중력 ON → 실기 off")의 **예외**다.
                   #   제어 전용 계약에는 정책이 없다 — 재현해야 할 학습 조건 자체가 없으므로
                   #   `sim_gravity_disabled` 는 자리표시자이고, 여기서 τ_ff 를 쓰는 것은
                   #   순수한 운용 선택이다(직접 제어·preset 이동에서 추종오차를 줄인다).
                   #   `pd_dg5f_m.yaml` 의 모델은 새 자산 기준이다: 배율 전부 1.0, payload 는
                   #   자산 URDF 손가락 질량 1.763 kg 실계산(구 우팔의 1.1·0.835 hack 아님).
                   gravity=GravityCfg(mode="model_tau_ff", sim_gravity_disabled=False),
                   sim_gains=driver_gains(gains_yaml, arm),
                   fabric=_side_fabric(spec, side, arm, hand, home_arm, home_h, home_source),
                   palm=None, hand=None, action_groups=[])


def build_asset_contract(asset: str = DEFAULT_ASSET, sides: tuple = ("right", "left"), primary: str = "right",
                         home: str = "zero", gains_yaml: Path = GAINS_YAML) -> DeployContract:
    """Control-only contract for ``asset``: fabric + pd per side, no policy (obs/action empty)."""
    spec = asset_spec(asset)
    manifest = load_manifest(spec)
    if primary not in sides:
        raise ContractError(f"primary {primary!r} not in sides {sides}")
    homes, home_source = arm_homes(home, tuple(sides))
    hands = _pour_hand_homes(home, asset, tuple(sides)) or _run_hand_homes(home, spec, manifest, tuple(sides))
    side_cfgs = {s: _control_side(spec, manifest, s, homes[s], home_source, gains_yaml, hands.get(s))
                 for s in sides}
    main = side_cfgs[primary]
    if main.fabric is None:
        raise ContractError(f"asset {asset}: primary side {primary} has no fabric — pick the other side")
    c = DeployContract(
        schema=SCHEMA,
        run=RunInfo(dir="", task=f"asset:{asset}", experiment="control_only", checkpoint="", checkpoint_md5="",
                    env_yaml_sha1="", agent_yaml_sha1=""),
        rate=RateCfg(policy_hz=1.0 / CONTROL_FABRIC_DT, step_dt=CONTROL_FABRIC_DT, episode_steps=0),
        policy=PolicyCfg(obs_dim=0, action_dim=0, rnn=None, mlp_units=[], normalize_input=False,
                         action_clip=None, obs_clip=None),
        obs=ObsCfg(joint_orders={"arm": main.arm_joints, "hand_profile": main.hand_joints, "tips": main.tip_bodies},
                   fk={"kind": "urdf_chain", "urdf": str(spec.urdf.relative_to(RL_WS))}, segments=()),
        action=ActionCfg(groups=(), palm=None, hand=None),
        fabric=main.fabric,
        pd=PdCfg(groups=main.pd_groups, home_arm=main.home_arm, home_hand=main.home_hand, gravity=main.gravity,
                 sim_gains=main.sim_gains),
        sides=side_cfgs, primary_side=primary, asset=_asset_info(spec), control_only=True)
    return validate(from_dict(to_dict(c)))


# ------------------------------------------------------------------ bind a run contract to an asset
def bind_asset(c: DeployContract, asset: str) -> DeployContract:
    """Re-base a run-built (single-arm) contract onto ``asset``: that asset's fabric URDF/params, soft
    limits from its URDF and per-side bodies from its manifest. Run semantics (obs, action, gains,
    rates, world, homes) are untouched — the policy is still the run's; only the robot model moves."""
    from .contract_build import _urdf_joint_limits

    spec = asset_spec(asset)
    manifest = load_manifest(spec)
    sides, primary = legacy_sides(c.obs, c.action, c.fabric, c.pd)
    side = c.primary_side or primary
    fs = spec.fabric.get(side)
    if fs is None:
        raise ContractError(f"asset {asset} has no fabric for side {side}")
    fabric = replace(c.fabric, class_name=fs.class_name, robot_dir=fs.robot_dir, params=fs.params)
    hand = c.action.hand
    if hand is not None and "soft_limits" in hand.params:
        params = {**hand.params, "soft_limits": _urdf_joint_limits(spec.urdf, list(hand.joints))}
        hand = replace(hand, params=params)
    action = replace(c.action, hand=hand)
    palm, tips = side_bodies(manifest, side, spec.ee_kind)
    arm, hand_joints = side_joints(manifest, side)
    old = sides[side]
    if list(old.arm_joints) != arm:
        raise ContractError(f"run arm joints {old.arm_joints} != asset {arm}")
    new_side = replace(old, palm_body=palm, tip_bodies=tips, ee_kind=spec.ee_kind, fabric=fabric, hand=hand,
                       hand_joints=hand_joints if spec.ee_kind == "dg5f" else old.hand_joints)
    fk = {**c.obs.fk, "urdf": str(spec.urdf.relative_to(RL_WS))}
    out = replace(c, schema=SCHEMA, fabric=fabric, action=action, obs=replace(c.obs, fk=fk),
                  sides={side: new_side}, primary_side=side, asset=_asset_info(spec))
    return validate(from_dict(to_dict(out)))
