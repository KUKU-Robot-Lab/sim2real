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
    # GRASP + LIFT + HOLD - ladder (iter_06): deep 1.5 < held 4 + qualities 6 < lift 10 < goal 3 + still 8
    # =====================================================================================
    DEEP_RADII = 1.5        # user 10.14: palm about 1.5 radii from the axis before closing ...
    DEEP_H = -0.01          # ... and a little below the object origin
    W_DEEP = 1.5            # open hand pre-grasp -> deep (ladder rung 1)
    W_LIFT = 14.0           # object height towards the goal height x held grasp quality (iter_06 grip)
    W_GOAL = 3.0            # object at the goal x floored grasp quality
    W_STILL = 8.5           # object at the goal AND arm/palm still (iter_07 8.0 -> 8.5: the keeping income dilutes M6)
    SQUEEZE_SHARE = 0.15    # over-grip costs up to 15 % of the grasp incomes (only far above contact-level force)
    W_DRAG = 2.0            # object pushed/pulled across the table while the hand touches it
    F_ON = 0.5              # N: per-finger contact saturates at ~1-2 N (tanh(2) = 0.96 at 1 N)
    CMD_FLOOR = 0.30        # close-command quality factor (four fingers): commands at the contact angle keep ~0.56
    THUMB_CMD_FLOOR = 0.50  # thumb-flexion command quality factor: thumb flexion commanded open keeps half
    DEPTH_FLOOR = 0.40      # depth quality factor (palm radially in + proximal-link wrap)
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
    W_CLOSE = 2.5           # close-command income at the deep ring (iter_05 3.0 -> iter_06 2.5: no-contact plateau)
    # iter_05 (rg3wl ep1400, 10.15 counterfactual replay, closing forced from step 300): the grasp paid +3.4..+5.0 per
    # step once held, but the first 5-9 closing steps netted 0.00 (right: +0.11 income, -0.11 loss) - the approach
    # measures kept taxing the closing hand before any grasp: the open-hand opposition (oppose_q 0.71 -> 0.61 as the
    # fingers wrap, cutting pose_shape / pose_pos), the object-contact cost on finger force (approach iter_00: 'the
    # open hand parks without touching', never released for the grasp) and the 1 deg undisturbed factor on the first
    # touches (released only once held). One sigma (0.4) of one finger closing paid +0.024/step (0.4 %); rg3wl started
    # with the finger mu shifted +0.6 and drifted back open (middle -0.84, thumb flexion -0.90). Release all three with
    # depth (as open_q, iter_01) and pay the close command x4, only while the tips are at the surface (a fist in the
    # air beside the object earns nothing). Tipping (spin) and drag costs stay.
    CLOSE_THUMB_SHARE = 0.25  # part of the close command taken from the thumb flexion (rest: four fingers)
    # iter_02 (rg3wh ep3100, 10.15 counterfactual: closing all fingers at the parked pose -> 4 fingers on, grasped
    # 0.88, but the step total fell 3.88 -> 3.39): the first contact tilts / nudges the object (tilt 4 deg, 1-3 cm),
    # the undisturbed factor (1 deg deadband) and the distance-based pose incomes then drop the approach + deep
    # incomes by ~3.2, more than the finger income adds. Both hands parked open (right) or closed short of contact
    # (left). Once fingers hold the object the approach is done: its incomes are floored at their full value and the
    # disturbance factor no longer taxes deep / close. Fingers counted by kind (any link), the thumb not required.
    HELD_LO = 1.0             # distinct fingers touching: no hold below 1 ...
    HELD_HI = 3.0             # ... full hold from 3 (user 10.14: at least 3 kinds of links)
    APPROACH_HELD = W_TRAVEL + W_OPEN_TRAVEL + W_ALIGN + W_POSE_POS + W_POSE_SHAPE + W_POSE_OPEN + W_SETTLE
    # iter_02: goal_tol 0.10 = goal offset 0.10 -> the env flags success with the object still on the table
    # (10.15 probe: 15 of 16 envs). The success lump sum needs the object lifted at least half way to the goal.
    SUCC_LIFT_LO = 0.5
    SUCC_LIFT_HI = 0.8
    # iter_03 (rg3wi ep1100, 10.15): lift income saturated at the goal height and nothing paid less above it, so the
    # lifted cup kept rising (lifted envs: mean rise 0.24 m at ep400 -> 0.35 m at ep1100, goal +0.10 m; kp_dist
    # 0.135 -> 0.237, goal_success 0). User 10.15: lift and stop at the goal height, the arm must not keep moving
    # after the lift. Deterministic replay (8 env, 2700 steps): every lifted cup kept rising to 0.38-0.55 m above
    # spawn until the episode ended and was carried 0.10-0.55 m sideways, arm 0.3-0.6 rad/s. Lift progress now peaks
    # at the goal and decays with the excess: rise past the goal height + horizontal distance from the goal.
    OVER_SCALE = 0.08       # m: lift income x exp(-excess / 8 cm) (0.53 at 5 cm, 0.29 at 10 cm, 0.04 at 25 cm past the goal)
    XY_DEAD = 0.01          # m: horizontal distance from the goal that costs nothing
    # same replay: the held cup tilted, median 46 deg (right) / 27 deg (left) while held. The goal pose is upright.
    TILT_DEAD = 0.087       # rad (5 deg): held-cup tilt that costs nothing
    TILT_HOLD_SCALE = 0.26  # rad (15 deg): lift income x exp(-(tilt - 5 deg) / 15 deg) (0.51 at 15 deg, 0.14 at 35 deg)
    # user video review (10.15, rg3wi ep1100): (3) the cup is held high, at the bottle neck - the palm should sit below the
    # origin (replay: palm +2.1 cm above the origin while held, index tip +4.6 cm). Earlier trial-and-error fixed this at
    # origin -20 mm (rh_aglt_r iter_21~25, d6 iter_06 GRASP_H) with this height factor on every post-grasp income.
    GRASP_H = -0.020        # m: palm target height along the axis (origin -20 mm)
    # (2) fingers inside the cup (replay: tip jumps 10-24 mm through the 5 mm hollow SDF wall in one step and stays) is a
    # physics fault, not a reward problem: the grasp env uses solid colliders (*_rl_solid.usd) and terminates on it.
    # iter_04 (rg3wk ep1100, 10.15): palm parked at the deep ring (xy 48-52 mm, origin -5..-8 mm), fingers open, no finger
    # ever touched in 1100 + 600 (rg3wj) epochs; finger action mu -1.07..-1.15 (open, beyond the clip). rg3wi (same start)
    # found the grasp by chance at ep380 (right, mu -1.19 -> +0.03 in one jump) and ep800 (left). Between the open hand and
    # the first contact only the close COMMAND paid (0.05 measured). Pay each fingertip for closing in on the object
    # surface at the deep ring - a dense path from open to touching (user 10.15: "the pose is very good, the fingers just
    # do not touch").
    W_REACH = 1.0           # all five tips on the surface at the deep ring (iter_05 2.0 -> iter_06 1.0)
    REACH_SCALE = 0.015     # m: exp(-gap / 1.5 cm) (parked open tips 20-45 mm off the surface -> 0.05-0.26)
    REACH_THUMB_SHARE = 0.3  # thumb part, rest averaged over the four fingers
    REACH_TOP = 0.8         # x mouth height: tips above it (over the rim / at the neck) count as away
    REACH_PAD = 0.008       # m: the tip point sits off the pad (rg3wi touching tips: r - R median -1..+5 mm, thumb +9..+18 mm)
    # iter_06 (rg3wm ep600-1400, 10.15 deterministic replays 64 env x 900, ~/rl_ws/videos/grasp_hw_1015_wm_reset): the grasp was
    # found (both hands by ep500-700) and then dropped by the left hand (ep1025 on: contact 0.8 -> 0.04, 300 epochs).
    # (1) the held grasp paid little: fingers = 9 x finger_q x cmd_f x depth_f x height_f, five floored factors. Right hand,
    #     all five fingers on the object: thumb flexion command -0.83 (cmd_f ~0.2) and palm +3 mm above the origin
    #     (height_f 0.5) left 0.71 of 9. Held on the table 7.2-7.9 / step vs the open hand at the deep ring 4.8-5.4.
    # (2) the closing shaping became a no-contact plateau: left ep1400 parks the half-closed hand (finger actions ~0) with
    #     the tips 5-15 mm off the surface and earns close 0.91 + reach 1.40 -> 6.59 / step, risk-free.
    # (3) the env ended 0.2-0.3 % of held steps per step by physics faults (tip penetration 25 of 38 ends, phys_cup, mimic),
    #     each a terminal loss of V ~ 785 (gamma 0.99): ~2.5 / step, the whole held-over-hover margin (fixed in the env:
    #     phys_fault_truncate, bootstrapped like a time-out).
    # The grasp incomes are now ADDITIVE on the held measure (a missing quality costs its own share, not half of everything),
    # the held grasp clearly above any no-contact pose (ledger: G3 >= 2 x G1), close / reach halved, lift concave so the
    # first centimetres pay.
    W_GRASP = 14.0          # held = 3 kinds of fingers (user 10.14: at least 3 kinds of links; 2 -> 0) x grasp quality
    GRASP_FLOOR = 0.3       # ONE floor on the whole quality product (iter_05 floored each of five factors, then multiplied)
    THUMB_FLOOR = 0.5       # quality factor without the thumb on the object (thumb tip not required, user 10.14)
    MORE_3 = 0.6            # quality factor at 3 kinds of fingers, 1.0 at 5 (iter_07: 0.7 -> 0.6, user 10.16 "all five")
    LIFT_DEAD = 0.005       # m: rise below this is pushing / tilting on the table, not a lift
    # iter_07 (rg3wn ep4900, 10.16 replays 64 env x 900 with substep 4, ~/rl_ws/videos/grasp_hw_1016_wn_pen/_raw/phys): with
    # physics faults ending the episode again (terminated, as in the earlier runs), the right hand's "touching, not yet
    # held" state is worth less than the closed hand hovering without contact (two fingers on V 532-561 vs 569): touching
    # ends 0.35-0.43 % of steps, hovering 0 %. Contact itself must clearly pay. User 10.16: (1) a closing guide that ends at
    # the contact - no extra income for closing further, a finger touching with both links is done; (2) a contact-keeping
    # income - no digging in, a proper force; a gradient up to the lift. User 10.16: touch with as many of the five fingers
    # as possible (tips need not touch), the thumb is a support (its first link), not a tip. Replay: the four fingers touch
    # with the tip only (first link ~0 %), the thumb with its first link (47 %); touching tips sit at wall - 2.4 mm (p50),
    # +1.2 mm (p99) inside the wall; the env ends at 10 mm.
    W_KEEP = 3.5            # contact keeping, all five fingers (each 1/5): any link 0.6, both links 1.0 (thumb: first link 1.0)
    KEEP_ANY = 0.6          # share paid for one link of a finger (four fingers); both links of the finger pay the rest
    F_KEEP_HI = 8.0         # N per finger: keeping income x exp(-(f - 8 N) / 10 N) above (thumb support p50 18 N in the replay)
    F_KEEP_SCALE = 10.0
    DIG_DEAD = 0.001        # m: tip point inside the outer wall that costs nothing (touching tips p99 +1.2 mm with substep 4)
    NEAR_MIX = 0.9          # iter_08 open-finger nearness: hand-wide (reach_q) share, rest the finger's own tip
    GUIDE_ANY = 0.8         # iter_08 closing guide share reached at the first contact of a finger (both links: 1)
    DIG_SCALE = 0.003       # m: keep x exp(-((pen - 1 mm) / 3 mm)^2) (0.37 at 4 mm, 0.0 at 10 mm = env termination)

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
    # iter_05: the open-hand opposition is a pre-grasp measure (ill-defined once the tips wrap), released with depth
    oppose_q = 1.0 - (1.0 - _ramp(oppose, 0.0, OPPOSE_FULL)) * (1.0 - deep_q)
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
    held = _ramp(c.sum(dim=-1), HELD_LO, HELD_HI)               # iter_02: fingers holding the object (kinds, any link)
    held3 = _ramp(c.sum(dim=-1), HELD_HI - 1.0, HELD_HI)        # 2 kinds -> 0, 3 kinds -> 1
    # palm depth judged radially (the grasp height is height_f's): 1 at the deep ring radius, 0 at the pre-grasp radius
    palm_in = (1.0 - torch.relu(r - DEEP_RADII * radius) / ((PREGRASP_RADII - DEEP_RADII) * radius)).clamp(0.0, 1.0)
    # iter_02: disturbance no longer taxes a held object; iter_05: nor the first touches of the closing hand at the deep ring
    released = torch.maximum(held, touch * deep_q)
    undisturbed_g = 1.0 - (1.0 - undisturbed) * (1.0 - released)

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
    align = W_ALIGN * orient_lin * undisturbed_g * room_factor
    near = torch.exp(-dist / POSE_SCALE) * undisturbed_g * room_factor
    pose_pos = W_POSE_POS * near * gate
    pose_shape = W_POSE_SHAPE * near * orient_q * oppose_q
    pose_open = W_POSE_OPEN * near * open_q * gate
    settle = W_SETTLE * near * still * gate

    obj_force = ff.sum(dim=-1) + ctx.palm_force
    # iter_05: finger force on the object is the grasp at the deep ring - only the palm keeps the approach charge there
    contact_force = ff.sum(dim=-1) * (1.0 - torch.maximum(deep_q, held)) + ctx.palm_force
    contact = -W_CONTACT * torch.tanh(contact_force / CONTACT_SCALE)
    table_contact = -W_TABLE * torch.tanh(ctx.hand_foreign_force / FOREIGN_SCALE)
    tipping = -W_TIP * torch.tanh(spin_ex / TIP_SCALE) * on_table

    # =====================================================================================
    # grasp stage
    # =====================================================================================
    # deep: linear cone around the deep ring (1.5 R, -0.01), 0 at the pre-grasp pose and everywhere outside it,
    # no extra pay for r < 1.5 R (palm at the wall)
    # gated by orientation only: the open-hand opposition measure is ill-defined once the tips pass around the object
    # iter_06: once 3 kinds of fingers hold the object the ring's height no longer matters (height_f judges the grasp height)
    deep = W_DEEP * torch.maximum(deep_q, held3 * palm_in) * undisturbed_g * gate_orient

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
    thumb_lin = 0.5 * (ctx.actions[:, 8].clamp(-1.0, 1.0) + 1.0)

    # iter_04 reach: per-tip gap to the object surface (radial gap outside the wall + height above REACH_TOP x mouth)
    tip_rel = tips - ctx.cup_pos.unsqueeze(1)
    tip_h = (tip_rel * up.unsqueeze(1)).sum(dim=-1)
    tip_r = (tip_rel - tip_h.unsqueeze(-1) * up.unsqueeze(1)).norm(dim=-1)
    mouth_h = _per_env(ctx.cup_mouth_z, palm).unsqueeze(1)
    tip_gap = torch.sqrt(torch.relu(tip_r - radius.unsqueeze(1) - REACH_PAD) ** 2
                         + torch.relu(tip_h - REACH_TOP * mouth_h) ** 2 + 1e-8)
    # iter_07: a finger touching the object with any link has arrived (the thumb supports with its first link, the tip
    # may stay off the surface)
    tip_near = torch.maximum(torch.exp(-tip_gap / REACH_SCALE), c)
    reach_q = REACH_THUMB_SHARE * tip_near[:, 0] + (1.0 - REACH_THUMB_SHARE) * tip_near[:, 1:].mean(dim=-1)

    # iter_07 digging: tip point inside the outer wall at its height (the env's own penetration measure; the wall radius
    # comes from the object mesh - the narrow bottom of a bottle is not the cylinder radius, so without it nothing is judged)
    wall_r = ctx.tip_wall_r if ctx.tip_wall_r is not None else torch.zeros_like(tip_r)
    pen = torch.where(wall_r > 0.0, wall_r - tip_r, torch.full_like(tip_r, -1.0))
    dig = torch.exp(-(torch.relu(pen - DIG_DEAD) / DIG_SCALE) ** 2)              # (N,5) 1 = not digging in

    # iter_07 contact keeping, per finger: touching (any link) x links (four fingers: one link KEEP_ANY, both 1; thumb: its
    # first link alone is the full support) x proper force x not digging in. Paid for every finger, so each added finger
    # pays (one finger already beats the closed hand hovering at the surface).
    cp = torch.tanh(ctx.finger_prox_force.clamp_min(0.0) / F_ON)
    ct = torch.tanh(ctx.finger_tip_force.clamp_min(0.0) / F_ON)
    both = cp * ct
    link_four = KEEP_ANY * c[:, 1:] + (1.0 - KEEP_ANY) * both[:, 1:]
    link_thumb = KEEP_ANY * c[:, 0] + (1.0 - KEEP_ANY) * torch.maximum(cp[:, 0], both[:, 0])
    links = torch.cat([link_thumb.unsqueeze(1), link_four], dim=1)
    f_band = torch.exp(-torch.relu(ff - F_KEEP_HI) / F_KEEP_SCALE)
    keep_q = (links * f_band * dig).mean(dim=-1)
    # touching fingers that are not digging in (share), carried into the grasp quality
    dig_q = (c * dig).sum(dim=-1) / c.sum(dim=-1).clamp_min(1e-3)
    # depth gate until the fingers hold the object; once held the grasp height is judged by height_f, not by the ring
    reach_depth = deep_q + (1.0 - deep_q) * held

    # iter_03 grasp height (previous verified form, rh_aglt_r iter_18/21): 1 within +-0.1 x mouth of the target, then
    # drops to 0.3 above it (towards the neck) and to 0.85 below it
    mouth = _per_env(ctx.cup_mouth_z, palm)
    dzh = (h - GRASP_H) / mouth.clamp_min(1e-3)
    up_ex = (dzh - 0.1).clamp(min=0.0)
    lo_ex = (-dzh - 0.1).clamp(min=0.0)
    height_f = (1.0 - 0.7 * torch.tanh((up_ex / 0.2) ** 2)) * (1.0 - 0.15 * torch.tanh((lo_ex / 0.3) ** 2))

    disp = (ctx.cup_pos[:, 0:2] - ctx.cup_spawn_pos[:, 0:2]).norm(dim=-1)
    v_xy = ctx.cup_lin_vel[:, 0:2].norm(dim=-1)
    drag_q = 0.5 * (torch.tanh(torch.relu(disp - DRAG_DEAD) / DRAG_SCALE)
                    + torch.tanh(torch.relu(v_xy - DRAG_V_DEAD) / DRAG_V_SCALE))
    # iter_06: grasp income = held3 x W_GRASP x (floor + (1 - floor) x quality product), ONE floor. The iter_05 product of five
    # separately floored factors kept 0.71 of 9 with all five fingers on (right ep600: thumb flexion open, palm +3 mm).
    # Each defect still scales the part above the floor (ledger ratio rules), the held grasp never falls to the hover level.
    thumb_f = THUMB_FLOOR + (1.0 - THUMB_FLOOR) * c[:, 0]
    more_f = MORE_3 + (1.0 - MORE_3) * _ramp(c.sum(dim=-1), HELD_HI, 5.0)
    depth_q = PALM_SHARE * palm_in + (1.0 - PALM_SHARE) * wrap
    depth_f2 = DEPTH_FLOOR + (1.0 - DEPTH_FLOOR) * depth_q
    quality = thumb_f * more_f * cmd_f * depth_f2 * height_f * dig_q    # iter_07: x touching fingers not digging in
    grip = held3 * (GRASP_FLOOR + (1.0 - GRASP_FLOOR) * quality)      # also carried by lift / goal / still
    fingers = W_GRASP * grip                                          # iter_07: touch_inc -> keep (per finger)

    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp_min(MIN_GOAL_RISE)
    over = (torch.relu(rise - goal_rise)
            + torch.relu((ctx.cup_pos[:, 0:2] - ctx.goal_pos[:, 0:2]).norm(dim=-1) - XY_DEAD))
    upright_q = torch.exp(-torch.relu(tilt_axis - TILT_DEAD) / TILT_HOLD_SCALE)
    # iter_06: concave in the first centimetres (sqrt), dead zone 5 mm (pushing / tilting on the table)
    lift_frac = (torch.relu(rise - LIFT_DEAD) / (goal_rise - LIFT_DEAD)).clamp(0.0, 1.0)
    lift_q = (torch.sqrt(lift_frac + 1e-12) * torch.exp(-over / OVER_SCALE)
              * upright_q)                                   # iter_03: peaks at the goal point, held upright
    lift = W_LIFT * lift_q * grip

    kp = ctx.goal_kp_dist
    goal_q = ((1.0 - NEAR_SHARE) * torch.exp(-kp / KP_SCALE) + NEAR_SHARE * ctx.near_goal.to(palm.dtype)) * (1.0 - on_table)
    goal = W_GOAL * goal_q * grip
    still_q = (torch.exp(-ctx.arm_qd.norm(dim=-1) / QD_SCALE)
               * torch.exp(-ctx.palm_lin_vel.norm(dim=-1) / V_SCALE))
    hold_still = W_STILL * goal_q * still_q * grip

    squeeze_q = torch.tanh(torch.relu(ff - F_SAFE) / F_SQ_SCALE).mean(dim=-1)
    squeeze = -SQUEEZE_SHARE * squeeze_q * (fingers + lift + goal + hold_still)

    drag = -W_DRAG * drag_q * on_table * torch.tanh(obj_force / F_ON)
    # iter_04: no reach income while the object is being dragged across the table (ledger D2g)
    reach = W_REACH * reach_depth * reach_q * undisturbed_g * gate_orient * (1.0 - drag_q * on_table)
    # iter_05: close command x tips at the surface (no income for a fist closed in the air) x drag gate as reach
    # (ledger D2g). Depth: the deep ring, or once held the grasp depth evidence (palm depth + proximal wrap, as the
    # finger income) - a low grasp at the target height keeps it (H3l), a shallow tips-only grasp does not (S3).
    close_depth = deep_q + (1.0 - deep_q) * held * depth_f
    # iter_08 closing guide per finger, in three steps that only go up (user 10.16: close where the object is, one link
    # touching is good, both links better, no income for closing further once done):
    #   open      : GUIDE_ANY x the finger's close command x its OWN tip nearness (iter_07: hand-wide nearness - the right
    #               hand held with ring/pinky commanded open, -0.95/-1.0, 56-59 % of held steps, and a closing command was
    #               paid the same whether or not that finger could reach the object)
    #   one link  : GUIDE_ANY + rest x the missing tip's nearness (first link only -> close the tip onto the surface);
    #               tip only -> the first link needs a deeper hand, paid by the grasp depth (depth_f), so it stays here
    #   both links: 1 (thumb: its first link alone is the full support)
    lin5 = torch.cat([thumb_lin.unsqueeze(1), 0.5 * (ctx.actions[:, 9:13].clamp(-1.0, 1.0) + 1.0)], dim=1)
    own_near = torch.exp(-tip_gap / REACH_SCALE)
    near5 = torch.cat([torch.maximum(own_near[:, 0], own_near[:, 1:].mean(dim=-1)).unsqueeze(1), own_near[:, 1:]], dim=1)
    pre = lin5 * (NEAR_MIX * reach_q.unsqueeze(1) + (1.0 - NEAR_MIX) * near5)
    # four fingers after contact: both links 1, first link only -> its tip nearing the surface (<= cp, continuous into both)
    post_four = GUIDE_ANY + (1.0 - GUIDE_ANY) * (both + cp * (1.0 - ct) * own_near)[:, 1:]
    post = torch.cat([torch.ones_like(c[:, :1]), post_four], dim=1)
    guide = c * torch.maximum(post, pre) + (1.0 - c) * pre                  # a contact never lowers the guide
    guide_q = CLOSE_THUMB_SHARE * guide[:, 0] + (1.0 - CLOSE_THUMB_SHARE) * guide[:, 1:].mean(dim=-1)
    close = W_CLOSE * close_depth * guide_q * undisturbed_g * gate_orient * (1.0 - drag_q * on_table)
    keep = W_KEEP * reach_depth * keep_q * (1.0 - drag_q * on_table)

    # iter_02: the approach is done once the fingers hold the object - its incomes never fall below their full value
    approach_sum = travel + open_hand + align + pose_pos + pose_shape + pose_open + settle
    approach_hold = torch.relu(held * APPROACH_HELD - approach_sum)

    components = {
        "travel": travel,
        "open_hand": open_hand,
        "align": align,
        "pose_pos": pose_pos,
        "pose_shape": pose_shape,
        "pose_open": pose_open,
        "settle": settle,
        "approach_hold": approach_hold,
        "contact": contact,
        "table_contact": table_contact,
        "tipping": tipping,
        "deep": deep,
        "close": close,
        "reach": reach,
        "keep": keep,
        "fingers": fingers,
        "lift": lift,
        "goal": goal,
        "still": hold_still,
        "squeeze": squeeze,
        "height_factor": height_f.detach(),
        "drag": drag,
    }
    step = (approach_sum + approach_hold
            + contact + table_contact + tipping
            + deep + close + reach + keep + fingers + lift + goal + hold_still + squeeze + drag)

    # success lump sum: success truncates the episode 240 hold steps later, so hovering at the goal without
    # completing would out-earn success. Pay, once, at the success step a bounded multiple of the income being
    # earned there (held at the goal), which covers the discounted income lost after the hold.
    success = (SUCCESS_MULT * step.clamp_min(0.0) * ctx.goal_success.to(palm.dtype)
               * _ramp(lift_q, SUCC_LIFT_LO, SUCC_LIFT_HI))      # iter_02: lifted, not standing at the goal tolerance
    reward = step + success
    return reward, {**components, "success": success}
