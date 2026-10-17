import torch
import math


def _per_env(value, ref: torch.Tensor) -> torch.Tensor:
    """float, 0-d, (1,) or (N,) -> (N,) tensor on ref's dtype/device."""
    t = torch.as_tensor(value, dtype=ref.dtype, device=ref.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(ref.shape[0])
    return t


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _ramp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 for x <= lo, 1 for x >= hi, linear in between."""
    return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)


def _radial(p: torch.Tensor, origin: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Part of (p - origin) perpendicular to the object axis (horizontal for an upright object)."""
    v = p - origin
    return v - (v * up).sum(dim=-1, keepdim=True) * up


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    palm = ctx.palm_pos

    # =====================================================================================
    # APPROACH (rh_aglt_ap iter_02) - constants, terms and weights unchanged
    # =====================================================================================
    PREGRASP_RADII = 2.5    # palm 2.5 object radii from the object axis
    PALM_RISE = 0.03        # palm 0.03 m ABOVE the object origin along the axis
    DIR_N = 0.53            # palm->axis direction = 0.53 * palm normal ...
    DIR_F = 0.83            # ... + 0.83 * finger direction
    JOINT_OPEN_MAX = 0.55   # every joint target up to 0.55 is still "open / pre-shaped"

    W_TRAVEL = 0.5
    W_OPEN_TRAVEL = 0.15
    W_ALIGN = 0.4
    W_POSE_POS = 0.8
    W_POSE_SHAPE = 0.8
    W_POSE_OPEN = 0.5
    W_SETTLE = 0.15
    W_CONTACT = 0.2
    W_TABLE = 0.2
    W_TIP = 0.5

    S_LEVEL = 0.6
    S_FACE = 0.25
    S_THUMB = 0.15
    ORIENT_FLOOR = 0.4
    LEVEL_LO = 0.15
    LEVEL_HI = 0.45
    FACE_LO = 0.5
    FACE_HI = 0.9
    THUMB_TOL = 0.01
    OPPOSE_FLOOR = 0.4
    OPPOSE_FULL = 0.9

    TRAVEL_SCALE = 0.30
    POSE_SCALE = 0.05
    THUMB_SCALE = 0.02
    OPEN_SCALE = 0.3
    SETTLE_SCALE = 0.05
    TILT_DEADBAND = 0.02
    TILT_SCALE = 0.05
    SPIN_DEADBAND = 0.12
    SPIN_SCALE = 0.15
    TIP_SCALE = 0.30
    CONTACT_SCALE = 1.0
    FOREIGN_SCALE = 2.0
    ROOM_MARGIN = 0.05
    ROOM_FLOOR = 0.85

    # =====================================================================================
    # GRASP + LIFT + HOLD (new) - ladder: deep 1.5 < fingers 9 < lift 10 < goal hold 3 + 8
    # =====================================================================================
    DEEP_RADII = 1.5        # user 10.14: palm about 1.5 radii from the axis before closing ...
    DEEP_H = -0.01          # ... and a little below the object origin
    W_DEEP = 1.5            # open hand pre-grasp -> deep (ladder rung 1)
    W_FINGERS = 9.0         # distinct-finger contact income (thumb + 2 others = 0.7 of it)
    W_LIFT = 10.0           # object height towards the goal height x floored grasp quality
    W_GOAL = 3.0            # object at the goal x floored grasp quality
    W_STILL = 8.0           # object at the goal AND arm/palm still (velocities, works before the hold)
    SQUEEZE_SHARE = 0.15    # over-grip costs up to 15 % of the grasp incomes (only far above contact-level force)
    W_DRAG = 2.0            # object pushed/pulled across the table while the hand touches it
    # distinct-finger ladder (shares of W_FINGERS), thumb required: 2 -> 0.10, 3 -> 0.70, 4 -> 0.85, 5 -> 1.0
    Q_PAIR = 0.10
    Q_TRI = 0.60
    Q_MORE = 0.30
    F_ON = 0.5              # N: per-finger contact saturates at ~1-2 N (tanh(2) = 0.96 at 1 N)
    FING_FLOOR = 0.10       # floored finger factor on lift / goal: 2 fingers keep 0.19, 3 keep 0.73
    CMD_FLOOR = 0.30        # floored close-until-contact factor (four fingers): commands at the contact angle keep ~0.56
    THUMB_CMD_FLOOR = 0.50  # floored thumb-flexion command factor: thumb flexion commanded open keeps half
    DEPTH_FLOOR = 0.40      # floored depth factor (palm depth + proximal-link wrap)
    PALM_SHARE = 0.5        # depth evidence = 0.5 palm depth + 0.5 proximal-link wrap of the touching fingers
    LIFT_ON = 0.005         # m above spawn: the object is still on the table below ~1 cm (G6)
    LIFT_OFF = 0.015
    MIN_GOAL_RISE = 0.05    # m: guard for the lift-progress denominator
    KP_SCALE = 0.04         # m: goal closeness exp(-kp / 4 cm)
    NEAR_SHARE = 0.2        # part of the goal closeness paid by the env's near_goal flag
    QD_SCALE = 0.5          # rad/s (norm over the 7 arm joints)
    V_SCALE = 0.05          # m/s palm speed
    F_SAFE = 5.0            # N per finger: no squeeze cost below
    F_SQ_SCALE = 5.0        # N
    DRAG_DEAD = 0.005       # m horizontal displacement from the spawn point
    DRAG_SCALE = 0.02       # m
    DRAG_V_DEAD = 0.02      # m/s horizontal object speed
    DRAG_V_SCALE = 0.05     # m/s
    SUCCESS_MULT = 10.0     # success-step lump sum = 10 x that step's own income (>= gamma^240 / (1 - gamma) = 9)
    # iter_01 (rg3we ep210: hand never closed, closure 0.075 flat, n_fingers 0.01): closing before the first
    # contact paid nothing up to 0.55 per joint and lost the open-hand incomes beyond it, so the close command
    # had no gradient from the approach checkpoint's fully open hand. Pay the close COMMAND (shape-independent,
    # linear from fully open) in proportion to the depth reached, and release the open-hand requirement there.
    W_CLOSE = 0.8           # close-command income at the deep ring (fully closed command, no contact needed)
    CLOSE_THUMB_SHARE = 0.25  # part of the close command taken from the thumb flexion (rest: four fingers)

    # ---- palm relative to the object axis (cylinder coordinates) ----
    radius = _per_env(ctx.cup_radius, palm)
    up = _unit(ctx.cup_up)
    h = ((palm - ctx.cup_pos) * up).sum(dim=-1)
    radial = _radial(palm, ctx.cup_pos, up)
    r = radial.norm(dim=-1)
    # distance to the pre-grasp ring, extended INWARD along the measured pre-grasp -> deep corridor
    # (2.5 R, +0.03) -> (1.5 R, -0.01): t = 0 for every point on or outside the pre-grasp pose (identical to the
    # approach reward there), so going deeper never loses approach income.
    seg_r = (DEEP_RADII - PREGRASP_RADII) * radius
    seg_h = DEEP_H - PALM_RISE
    seg_len2 = seg_r * seg_r + seg_h * seg_h
    t_cor = (((r - PREGRASP_RADII * radius) * seg_r + (h - PALM_RISE) * seg_h) / seg_len2).clamp(0.0, 1.0)
    e_r = r - (PREGRASP_RADII * radius + t_cor * seg_r)
    e_h = h - (PALM_RISE + t_cor * seg_h)
    dist = torch.sqrt(e_r * e_r + e_h * e_h + 1e-8)
    # deep progress: linear cone around the deep ring (1.5 R, -0.01), 0 at the pre-grasp pose and everywhere
    # outside it, no extra pay for r < 1.5 R (palm at the wall). Used by deep, the open-hand release and close.
    seg_len = torch.sqrt(seg_len2)
    d_deep = torch.sqrt(torch.relu(r - DEEP_RADII * radius) ** 2 + (h - DEEP_H) ** 2 + 1e-8)
    deep_q = (1.0 - d_deep / seg_len).clamp(0.0, 1.0)

    # ---- orientation (unchanged) ----
    n = _unit(ctx.palm_axes[:, 0:3])
    f = _unit(ctx.palm_axes[:, 3:6])
    to_axis = -radial / r.unsqueeze(-1).clamp_min(1e-4)
    tilt_sin = torch.sqrt(n[:, 2] ** 2 + f[:, 2] ** 2 + 1e-8).clamp(max=1.0)
    want = _unit(DIR_N * n + DIR_F * f)
    facing = (to_axis * want).sum(dim=-1)
    tips = ctx.tips_pos
    thumb_dz = tips[:, 0, 2] - tips[:, -1, 2]
    thumb_up = 0.5 * (1.0 + torch.tanh(thumb_dz / THUMB_SCALE))
    orient_lin = S_LEVEL * (1.0 - tilt_sin) + S_FACE * 0.5 * (1.0 + facing) + S_THUMB * thumb_up
    orient_q = ((1.0 - _ramp(tilt_sin, LEVEL_LO, LEVEL_HI))
                * _ramp(facing, FACE_LO, FACE_HI)
                * _ramp(thumb_dz, -THUMB_TOL, THUMB_TOL))
    gate_orient = ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * orient_q

    # ---- thumb / finger opposition (unchanged) ----
    u_thumb = _unit(_radial(tips[:, 0], ctx.cup_pos, up))
    u_fingers = _unit(_radial(tips[:, 1:].mean(dim=1), ctx.cup_pos, up))
    oppose = 0.5 * (1.0 - (u_thumb * u_fingers).sum(dim=-1))
    oppose_q = _ramp(oppose, 0.0, OPPOSE_FULL)
    gate_oppose = OPPOSE_FLOOR + (1.0 - OPPOSE_FLOOR) * oppose_q
    gate = gate_orient * gate_oppose

    # ---- object on the table? (height above its spawn, G6: not tilt, not the grasp flag) ----
    rise = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    on_table = 1.0 - _ramp(rise, LIFT_ON, LIFT_OFF)

    # ---- object undisturbed: judged only while it stands on the table (in-hand tilt during a lift is normal) ----
    tilt_axis = torch.acos(up[:, 2].clamp(-1.0, 1.0))
    tilt = torch.maximum(_per_env(ctx.cup_tilt, palm), tilt_axis)
    w = ctx.cup_ang_vel
    w_tip = (w - (w * up).sum(dim=-1, keepdim=True) * up).norm(dim=-1)
    tilt_ex = torch.relu(tilt - TILT_DEADBAND)
    spin_ex = torch.relu(w_tip - SPIN_DEADBAND)
    undisturbed_tbl = torch.exp(-(tilt_ex / TILT_SCALE) ** 2) * torch.exp(-(spin_ex / SPIN_SCALE) ** 2)
    undisturbed = 1.0 - on_table * (1.0 - undisturbed_tbl)

    # ---- per-finger contact, distinct fingers (any link counts, both links of one finger count once) ----
    ff = ctx.finger_force.clamp_min(0.0)
    c = torch.tanh(ff / F_ON)                                  # (N,5) saturating per finger
    touch = 1.0 - torch.prod(1.0 - c, dim=-1)                  # any finger on the object
    k_other = c[:, 1:].sum(dim=-1)                             # 0..4 other fingers, treated alike
    finger_q = c[:, 0] * (Q_PAIR * _ramp(k_other, 0.0, 1.0)
                          + Q_TRI * _ramp(k_other, 1.0, 2.0)
                          + Q_MORE * _ramp(k_other, 2.0, 4.0))

    # ---- hand openness PER JOINT (unchanged), released once a finger touches the object ----
    fold = (torch.relu(ctx.hand_q_target_closure - JOINT_OPEN_MAX).sum(dim=-1)
            + torch.relu(ctx.hand_closure - JOINT_OPEN_MAX))
    # iter_01: also released with depth - open at the pre-grasp pose (deep_q = 0), free to close at the deep ring
    open_q = 1.0 - (1.0 - torch.exp(-(fold / OPEN_SCALE) ** 2)) * (1.0 - touch) * (1.0 - deep_q)

    still = torch.exp(-ctx.palm_lin_vel.norm(dim=-1) / SETTLE_SCALE)
    room = (ctx.arm_limit_margin.min(dim=-1).values / ROOM_MARGIN).clamp(0.0, 1.0)
    room_factor = ROOM_FLOOR + (1.0 - ROOM_FLOOR) * room

    # ---- approach incomes (unchanged) ----
    far = torch.exp(-dist / TRAVEL_SCALE)
    travel = W_TRAVEL * far * room_factor
    open_hand = W_OPEN_TRAVEL * far * open_q * room_factor
    align = W_ALIGN * orient_lin * undisturbed * room_factor
    near = torch.exp(-dist / POSE_SCALE) * undisturbed * room_factor
    pose_pos = W_POSE_POS * near * gate
    pose_shape = W_POSE_SHAPE * near * orient_q * oppose_q
    pose_open = W_POSE_OPEN * near * open_q * gate
    settle = W_SETTLE * near * still * gate

    obj_force = ff.sum(dim=-1) + ctx.palm_force
    contact = -W_CONTACT * torch.tanh(obj_force / CONTACT_SCALE)
    table_contact = -W_TABLE * torch.tanh(ctx.hand_foreign_force / FOREIGN_SCALE)
    tipping = -W_TIP * torch.tanh(spin_ex / TIP_SCALE) * on_table

    # =====================================================================================
    # grasp stage
    # =====================================================================================
    # deep: linear cone around the deep ring (1.5 R, -0.01), 0 at the pre-grasp pose and everywhere outside it,
    # no extra pay for r < 1.5 R (palm at the wall)
    # gated by orientation only: the open-hand opposition measure is ill-defined once the tips pass around the object
    deep = W_DEEP * deep_q * undisturbed * gate_orient

    # depth evidence for the grasp: palm depth + proximal links of the touching fingers (wrap, not finger tips only)
    wrap = (torch.tanh(ctx.finger_prox_force[:, 1:].clamp_min(0.0) / F_ON).sum(dim=-1)
            / k_other.clamp_min(1.0)).clamp(0.0, 1.0)
    depth_f = DEPTH_FLOOR + (1.0 - DEPTH_FLOOR) * (PALM_SHARE * deep_q + (1.0 - PALM_SHARE) * wrap)

    # close-until-contact (shape-independent): the four finger COMMANDS near +1 (squared: stopping at the contact
    # angle is clearly worse) AND the thumb flexion command closing (real robot: thumb_2 left open, the thumb only
    # pushed back by the index). Commands, not hand_q_target_closure (frozen at contact).
    a_fing = ctx.actions[:, 9:13].clamp(0.0, 1.0)
    cmd_q = (a_fing * a_fing).mean(dim=-1)
    thumb_cmd_q = ctx.actions[:, 8].clamp(0.0, 1.0)
    cmd_f = ((CMD_FLOOR + (1.0 - CMD_FLOOR) * cmd_q)
             * (THUMB_CMD_FLOOR + (1.0 - THUMB_CMD_FLOOR) * thumb_cmd_q))

    # iter_01 close: the close COMMAND (linear from fully open -1 to fully closed +1, contact not needed) x depth.
    # Fingers stop at their own contact (sim contact freeze, real lead cap), so this is shape-independent.
    fing_lin = (0.5 * (ctx.actions[:, 9:13].clamp(-1.0, 1.0) + 1.0)).mean(dim=-1)
    thumb_lin = 0.5 * (ctx.actions[:, 8].clamp(-1.0, 1.0) + 1.0)
    close_q = (1.0 - CLOSE_THUMB_SHARE) * fing_lin + CLOSE_THUMB_SHARE * thumb_lin
    close = W_CLOSE * deep_q * close_q * undisturbed * gate_orient

    finger_f = FING_FLOOR + (1.0 - FING_FLOOR) * finger_q
    grip = touch * finger_f * cmd_f * depth_f                  # floored grasp quality on lift / goal (hand on the object)

    fingers = W_FINGERS * finger_q * cmd_f * depth_f

    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp_min(MIN_GOAL_RISE)
    lift_q = (rise / goal_rise).clamp(0.0, 1.0)
    lift = W_LIFT * lift_q * grip

    kp = ctx.goal_kp_dist
    goal_q = ((1.0 - NEAR_SHARE) * torch.exp(-kp / KP_SCALE) + NEAR_SHARE * ctx.near_goal.to(palm.dtype)) * (1.0 - on_table)
    goal = W_GOAL * goal_q * grip
    still_q = (torch.exp(-ctx.arm_qd.norm(dim=-1) / QD_SCALE)
               * torch.exp(-ctx.palm_lin_vel.norm(dim=-1) / V_SCALE))
    hold_still = W_STILL * goal_q * still_q * grip

    squeeze_q = torch.tanh(torch.relu(ff - F_SAFE) / F_SQ_SCALE).mean(dim=-1)
    squeeze = -SQUEEZE_SHARE * squeeze_q * (fingers + lift + goal + hold_still)

    disp = (ctx.cup_pos[:, 0:2] - ctx.cup_spawn_pos[:, 0:2]).norm(dim=-1)
    v_xy = ctx.cup_lin_vel[:, 0:2].norm(dim=-1)
    drag_q = 0.5 * (torch.tanh(torch.relu(disp - DRAG_DEAD) / DRAG_SCALE)
                    + torch.tanh(torch.relu(v_xy - DRAG_V_DEAD) / DRAG_V_SCALE))
    drag = -W_DRAG * drag_q * on_table * torch.tanh(obj_force / F_ON)

    components = {
        "travel": travel,
        "open_hand": open_hand,
        "align": align,
        "pose_pos": pose_pos,
        "pose_shape": pose_shape,
        "pose_open": pose_open,
        "settle": settle,
        "contact": contact,
        "table_contact": table_contact,
        "tipping": tipping,
        "deep": deep,
        "close": close,
        "fingers": fingers,
        "lift": lift,
        "goal": goal,
        "still": hold_still,
        "squeeze": squeeze,
        "drag": drag,
    }
    step = (travel + open_hand + align + pose_pos + pose_shape + pose_open + settle
            + contact + table_contact + tipping
            + deep + close + fingers + lift + goal + hold_still + squeeze + drag)

    # success lump sum: success truncates the episode 240 hold steps later, so hovering at the goal without
    # completing would out-earn success. Pay, once, at the success step a bounded multiple of the income being
    # earned there (held at the goal), which covers the discounted income lost after the hold.
    success = SUCCESS_MULT * step.clamp_min(0.0) * ctx.goal_success.to(palm.dtype)
    reward = step + success
    return reward, {**components, "success": success}
