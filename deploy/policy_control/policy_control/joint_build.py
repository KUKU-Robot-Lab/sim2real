"""run dump(params/env.yaml · agent.yaml · nn/) → JointContract. 빌드 때만 hdgp 를 읽는다(프로필 · 자산 URDF).

값의 출처(전부 이 런이 학습한 것):
  팔 한계         학습 자산 URDF (soft_joint_pos_limit_factor 1.0 을 확인한다)
  손 한계         학습 자산 URDF ∩ 프로필 `hand_action_limit_override`(re.fullmatch — IsaacLab 과 같다)
  손 시작 자세    env.yaml robot.init_state.joint_pos 를 손 한계로 clip (hdgp `_hand_reset_q`)
  팔 시작 자세    env.yaml `arm_reset_joint_pos_override`
  k_arm · ema     env.yaml
  목표            goal_first_z_range 가운데(가로 0) · goal_box_min/max
  관측 손 순서    인자로 받는다 — 학습 자산의 시뮬레이터 관절 순서는 URDF 로 알 수 없다
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from .joint_contract import FAMILY, SCHEMA, JointContract, JointContractError, file_md5, file_sha1, validate
from .pour_build import _yaml

#: 20-관절 DG-5F 의 시뮬레이터 관절 순서(깊이 우선: 모든 _1, 그다음 _2 … · 손가락 index, middle, pinky, ring, thumb).
#: pour i24 trace_meta 의 joint_names 와 scripts/grasp_s2r_obs_builder.py 가 같은 순서다.
_PHYSX_FINGERS = ("index", "middle", "pinky", "ring", "thumb")


def assumed_obs_order(hand_joints: Sequence[str], side_prefix: str) -> tuple:
    """용접 관절을 뺀 20-관절 순서 — **가정**이다. 용접 링크가 합쳐지면 달라질 수 있어 실측으로 바꿔야 한다."""
    order = [f"{side_prefix}_hj_{f}_{d}" for d in (1, 2, 3, 4) for f in _PHYSX_FINGERS]
    kept = [n for n in order if n in set(hand_joints)]
    if sorted(kept) != sorted(hand_joints):
        raise JointContractError(f"assumed order does not cover the hand joints {sorted(set(hand_joints) - set(kept))}")
    return tuple(kept)


def urdf_joints(urdf: Path) -> dict:
    """이름 → (type, lower, upper, origin xyz, origin rpy)."""
    out = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        lim, org = j.find("limit"), j.find("origin")
        lo = hi = None
        if lim is not None and lim.get("lower") is not None:
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        xyz = tuple(float(v) for v in (org.get("xyz", "0 0 0") if org is not None else "0 0 0").split())
        rpy = tuple(float(v) for v in (org.get("rpy", "0 0 0") if org is not None else "0 0 0").split())
        out[j.get("name")] = (j.get("type"), lo, hi, xyz, rpy)
    return out


def _need(env: dict, key: str):
    if key not in env or env[key] is None:
        raise JointContractError(f"env.yaml has no '{key}'")
    return env[key]


def _robot_cfg(env: dict) -> dict:
    scene = env.get("scene")
    for cand in ((scene or {}).get("robot") if isinstance(scene, dict) else None, env.get("robot_cfg"), env.get("robot")):
        if isinstance(cand, dict) and "init_state" in cand:
            return cand
    raise JointContractError("env.yaml has no robot articulation cfg with init_state")


def _pick_checkpoint(run: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return Path(checkpoint)
    pths = sorted((run / "nn").glob("*.pth"))
    if len(pths) != 1:
        raise JointContractError(f"{run}/nn has {len(pths)} .pth — pass --checkpoint")
    return pths[0]


def _hand_limits(profile, urdf: dict, names: Sequence[str]) -> tuple[list, list]:
    lo, hi = [], []
    for n in names:
        t, a, b, _, _ = urdf.get(n, (None, None, None, None, None))
        if t not in ("revolute", "prismatic") or a is None:
            raise JointContractError(f"hand joint {n} is not a limited movable joint in the training URDF")
        lo.append(a)
        hi.append(b)
    for regex, (olo, ohi) in dict(profile.hand_action_limit_override).items():
        hits = [i for i, n in enumerate(names) if re.fullmatch(regex, n)]
        if not hits:
            raise JointContractError(f"override {regex!r} matches no hand joint (hdgp would refuse to boot)")
        for i in hits:
            lo[i] = max(lo[i], float(olo)) if olo is not None else lo[i]
            hi[i] = min(hi[i], float(ohi)) if ohi is not None else hi[i]
    return lo, hi


def build_joint_contract(run: Path, hdgp_root: Path, deploy_asset: str, *, hand_obs_order: Sequence[str] | None,
                         order_source: str, checkpoint: Path | None = None) -> JointContract:
    from .pour_profiles import load_profile

    run = Path(run)
    env_yaml, agent_yaml = run / "params" / "env.yaml", run / "params" / "agent.yaml"
    env, agent = _yaml(env_yaml), _yaml(agent_yaml)
    if not env.get("hand_direct"):
        raise JointContractError("env.yaml hand_direct is not true — not a joint_direct run")
    profile = load_profile(Path(hdgp_root), str(_need(env, "profile_name")))
    side = "right" if str(env.get("hand_side", "r")) == "r" else "left"
    prefix = side[0]
    robot = _robot_cfg(env)
    if float(robot.get("soft_joint_pos_limit_factor", 1.0)) != 1.0:
        raise JointContractError("soft_joint_pos_limit_factor != 1.0 — limits would differ from the URDF")
    usd = Path(str(robot["spawn"]["usd_path"]))
    train_asset = usd.parent.name
    train_urdf = Path(hdgp_root) / "assets" / "robot" / train_asset / f"{train_asset}.urdf"
    deploy_urdf = Path(hdgp_root) / "assets" / "robot" / deploy_asset / f"{deploy_asset}.urdf"
    for p in (train_urdf, deploy_urdf):
        if not p.is_file():
            raise JointContractError(f"asset URDF missing: {p}")
    tj, dj = urdf_joints(train_urdf), urdf_joints(deploy_urdf)

    arm = tuple(f"{prefix}_aj_{i}" for i in range(1, int(profile.num_arm_joints) + 1))
    if not all(re.fullmatch(profile.arm_joint_regex, n) for n in arm):
        raise JointContractError(f"arm joints {arm} do not match profile regex {profile.arm_joint_regex!r}")
    arm_lo, arm_hi = [tj[n][1] for n in arm], [tj[n][2] for n in arm]
    hand = tuple(profile.hand_joint_names)
    hand_lo, hand_hi = _hand_limits(profile, tj, hand)
    init = dict(robot["init_state"]["joint_pos"])
    hand_reset = [min(max(float(init.get(n, 0.0)), a), b) for n, a, b in zip(hand, hand_lo, hand_hi)]
    arm_reset = [float(v) for v in _need(env, "arm_reset_joint_pos_override")]

    # 학습 자산에서 용접됐지만 배포 자산(실기 손)에서는 움직이는 관절 — 용접 각도로 붙잡는다
    welded = {}
    for n, (t, *_rest) in tj.items():
        if n.startswith(f"{prefix}_hj_") and t == "fixed" and dj.get(n, ("fixed",))[0] in ("revolute", "prismatic"):
            if tj[n][3:] != dj[n][3:]:
                raise JointContractError(f"{n}: welded origin differs from the deploy URDF — weld angle is not 0")
            welded[n] = 0.0

    order = tuple(hand_obs_order) if hand_obs_order is not None else assumed_obs_order(hand, prefix)
    ckpt = _pick_checkpoint(run, checkpoint)
    net = agent["params"]["network"]
    cfg = agent["params"]["config"]
    agent_env = agent["params"].get("env") or {}
    z_lo, z_hi = (float(v) for v in _need(env, "goal_first_z_range"))
    notes = (
        f"train_urdf {train_urdf.name} · profile {profile.name}",
        "관측 노이즈 · 지연은 학습 전용이라 넣지 않는다",
        "목표는 한 점(학습은 가로 ±goal_first_xy_range · 높이 goal_first_z_range 에서 뽑는다)",
    )
    return validate(JointContract(
        schema=SCHEMA, family=FAMILY, task=str(cfg["name"]), run_dir=str(run),
        checkpoint=str(ckpt), checkpoint_md5=file_md5(ckpt),
        env_yaml_sha1=file_sha1(env_yaml), agent_yaml_sha1=file_sha1(agent_yaml),
        asset=deploy_asset, train_asset=train_asset, side=side,
        policy_hz=1.0 / (float(env["sim"]["dt"]) * int(_need(env, "decimation"))),
        obs_dim=int(_need(env, "observation_space")), action_dim=int(_need(env, "action_space")),
        # rl_games 래퍼(IsaacLab)의 clip 은 agent.yaml 최상위 params.env 에 있다(pour_build 와 같은 자리)
        obs_clip=None if agent_env.get("clip_observations") is None else float(agent_env["clip_observations"]),
        action_clip=float(agent_env.get("clip_actions", 1.0)), recurrent="rnn" in net,
        arm_joints=arm, arm_lo=tuple(arm_lo), arm_hi=tuple(arm_hi), arm_reset=tuple(arm_reset),
        k_arm=float(_need(env, "k_arm")), arm_ema=float(_need(env, "arm_ema")),
        hand_joints=hand, hand_lo=tuple(hand_lo), hand_hi=tuple(hand_hi), hand_reset=tuple(hand_reset),
        hand_ema=float(_need(env, "hand_ema")),
        hand_obs_order=order, hand_obs_order_source=str(order_source), welded=welded,
        palm_body=str(profile.palm_body), tip_bodies=tuple(profile.fingertip_bodies),
        keypoint_half_height=0.5 * float(_need(env, "keypoint_scale")) * float(_need(env, "keypoint_fixed_height")),
        keypoint_axial_unit=((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0 / 3.0), (0.0, 0.0, -1.0 / 3.0)),
        goal_offset=(0.0, 0.0, 0.5 * (z_lo + z_hi)),
        goal_box_lo=tuple(float(v) for v in _need(env, "goal_box_min")),
        goal_box_hi=tuple(float(v) for v in _need(env, "goal_box_max")),
        notes=notes,
    ))
