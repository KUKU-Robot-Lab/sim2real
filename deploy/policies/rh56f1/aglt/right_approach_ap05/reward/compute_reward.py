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

    # ---- measured OPEN-hand pre-grasp pose (20 steps before closing, F14b; not object dimensions) ----
    PREGRASP_RADII = 2.5    # palm 2.5 object radii from the object axis
    # iter_03 (user 10.19 "the hand centre must be at about the bottle origin height - now 2-3 cm high"): +0.03 -> 0.
    #   Same pose as rh_aglt env.ready_pre_h_m (0, 0) so the approach warmstart and the ready_carry start agree.
    PALM_RISE = 0.0         # palm at the object origin height along the axis
    DIR_N = 0.53            # palm->axis direction = 0.53 * palm normal ...
    DIR_F = 0.83            # ... + 0.83 * finger direction
    JOINT_OPEN_MAX = 0.55   # every joint target up to 0.55 is still "open / pre-shaped" (F16)

    # ---- incomes (per-step maxima; right pose total 3.30, pose income 0.8+0.8+0.5+0.15 = 2.25) ----
    W_TRAVEL = 0.5          # plain distance income, never gated by orientation / opposition / hand (F3)
    W_OPEN_TRAVEL = 0.15    # open-hand bonus on the travel profile: fist < 0.9x open at every distance
    W_ALIGN = 0.4           # linear orientation score at EVERY distance (was 0.3): 30 deg up costs 0.12/step
    W_POSE_POS = 0.8        # steep final-10-cm position income  x orientation gate x opposition gate
    W_POSE_SHAPE = 0.8      # orientation quality x opposition quality at the pose (no floor)
    W_POSE_OPEN = 0.5       # per-joint open / pre-shaped hand at the pose x both gates (was 0.4):
                            # two folded thumb joints lose 0.65 of 3.30 -> 0.80x
    W_SETTLE = 0.15         # stay still at the pose x both gates
    # ---- mild costs (unchanged) ----
    W_CONTACT = 0.2         # any finger/palm force on the object
    W_TABLE = 0.2           # hand on table / body
    W_TIP = 0.5             # bounded, only while the object is rotating

    # ---- linear orientation score (F11a): shares sum to 1, gradient at every tilt ----
    S_LEVEL = 0.6           # 1 - sin(palm-plane tilt)
    S_FACE = 0.25           # (1 + cos(palm->axis vs measured direction)) / 2
    S_THUMB = 0.15          # thumb tip above the little-finger tip
    # ---- floored orientation gate on every final-10-cm income (F10, F11b) ----
    ORIENT_FLOOR = 0.4      # wrong orientation keeps 40 % (was 0.5): 25 deg tilt -> 0.51x of the pose
    LEVEL_LO = 0.15         # sin tilt: full gate up to ~9 deg ...
    LEVEL_HI = 0.45         # ... linear down to the floor at ~27 deg
    FACE_LO = 0.5           # facing cos: floor beyond 60 deg off the measured direction ...
    FACE_HI = 0.9           # ... full gate within ~26 deg (the F13 direction 0.93n+0.37f has 0.81 -> 0.78)
    THUMB_TOL = 0.01        # m: full with the thumb tip >= 1 cm above the little-finger tip, 0 at 1 cm below
    # ---- floored thumb/finger opposition gate on every final-10-cm income (F15) ----
    OPPOSE_FLOOR = 0.4      # thumb and fingers on the same side (F13, oppose 0.01) keep 40 %
    OPPOSE_FULL = 0.9       # oppose >= 0.9 (measured 0.95 at the pose, 0.94 at 2 cm) = full gate

    # ---- shaping scales ----
    TRAVEL_SCALE = 0.30     # m: smooth from the 0.46 m start all the way in
    POSE_SCALE = 0.05       # m: pose income 0.135 at 10 cm, 0.45 at 4 cm, 0.67 at 2 cm, 1.0 at 0
    THUMB_SCALE = 0.02      # m: thumb-above-little-finger softness in the linear score
    OPEN_SCALE = 0.3        # summed per-joint excess: 0.1 -> 0.89, one thumb joint folded -> 0.1, two -> 0
    SETTLE_SCALE = 0.05     # m/s
    TILT_DEADBAND = 0.02    # rad: resting tilt is free
    TILT_SCALE = 0.05       # rad
    SPIN_DEADBAND = 0.12    # rad/s: above the 0.094 rad/s solver jitter (F8)
    SPIN_SCALE = 0.15       # rad/s
    TIP_SCALE = 0.30        # rad/s
    CONTACT_SCALE = 1.0     # N: 0.35 N touch -> -0.07
    FOREIGN_SCALE = 2.0     # N
    ROOM_MARGIN = 0.05      # full joint room above 5 % margin (F7)
    ROOM_FLOOR = 0.85       # mild joint-room multiplier on positive income

    # ---- palm relative to the object axis (cylinder coordinates) ----
    radius = _per_env(ctx.cup_radius, palm)
    up = _unit(ctx.cup_up)
    h = ((palm - ctx.cup_pos) * up).sum(dim=-1)
    radial = _radial(palm, ctx.cup_pos, up)
    r = radial.norm(dim=-1)
    e_r = r - PREGRASP_RADII * radius
    e_h = h - PALM_RISE
    dist = torch.sqrt(e_r * e_r + e_h * e_h + 1e-8)  # distance to the open-hand pre-grasp ring

    # ---- orientation (linear / cosine-like errors, no narrow peak) ----
    n = _unit(ctx.palm_axes[:, 0:3])
    f = _unit(ctx.palm_axes[:, 3:6])
    to_axis = -radial / r.unsqueeze(-1).clamp_min(1e-4)
    # sin of the palm-plane tilt: 0 when normal and fingers are both horizontal; equals sin(a) for the
    # right pose rotated by angle a about any horizontal axis
    tilt_sin = torch.sqrt(n[:, 2] ** 2 + f[:, 2] ** 2 + 1e-8).clamp(max=1.0)
    want = _unit(DIR_N * n + DIR_F * f)
    facing = (to_axis * want).sum(dim=-1)  # cos: 1 = object where the measured open pose has it
    tips = ctx.tips_pos
    thumb_dz = tips[:, 0, 2] - tips[:, -1, 2]  # thumb tip minus little-finger tip
    thumb_up = 0.5 * (1.0 + torch.tanh(thumb_dz / THUMB_SCALE))

    # every-distance shaping score in [0, 1], linear in sin(tilt) and in the facing cosine
    orient_lin = S_LEVEL * (1.0 - tilt_sin) + S_FACE * 0.5 * (1.0 + facing) + S_THUMB * thumb_up
    # pose-quality in [0, 1]: 1 only for level + facing + thumb-up
    orient_q = ((1.0 - _ramp(tilt_sin, LEVEL_LO, LEVEL_HI))
                * _ramp(facing, FACE_LO, FACE_HI)
                * _ramp(thumb_dz, -THUMB_TOL, THUMB_TOL))
    gate_orient = ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * orient_q

    # ---- object between thumb and fingers (F15a): opposite sides of the axis -> 1, same side -> 0 ----
    u_thumb = _unit(_radial(tips[:, 0], ctx.cup_pos, up))
    u_fingers = _unit(_radial(tips[:, 1:].mean(dim=1), ctx.cup_pos, up))  # four fingers alike
    oppose = 0.5 * (1.0 - (u_thumb * u_fingers).sum(dim=-1))
    oppose_q = _ramp(oppose, 0.0, OPPOSE_FULL)  # 0.19 at 10 cm, 0.71 at 4 cm, 1 at 2 cm and at the pose
    gate_oppose = OPPOSE_FLOOR + (1.0 - OPPOSE_FLOOR) * oppose_q
    gate = gate_orient * gate_oppose  # both floored; only the final-10-cm incomes carry it

    # ---- object undisturbed (upright and not rotating); jitter is inside the deadbands ----
    tilt_axis = torch.acos(up[:, 2].clamp(-1.0, 1.0))
    tilt = torch.maximum(_per_env(ctx.cup_tilt, palm), tilt_axis)
    w = ctx.cup_ang_vel
    w_tip = (w - (w * up).sum(dim=-1, keepdim=True) * up).norm(dim=-1)  # ignore spin about own axis
    tilt_ex = torch.relu(tilt - TILT_DEADBAND)
    spin_ex = torch.relu(w_tip - SPIN_DEADBAND)
    undisturbed = torch.exp(-(tilt_ex / TILT_SCALE) ** 2) * torch.exp(-(spin_ex / SPIN_SCALE) ** 2)

    # ---- hand openness PER JOINT (F16): a folded joint cannot hide behind open ones ----
    fold = (torch.relu(ctx.hand_q_target_closure - JOINT_OPEN_MAX).sum(dim=-1)
            + torch.relu(ctx.hand_closure - JOINT_OPEN_MAX))
    open_q = torch.exp(-(fold / OPEN_SCALE) ** 2)

    # ---- steadiness at the pose ----
    still = torch.exp(-ctx.palm_lin_vel.norm(dim=-1) / SETTLE_SCALE)

    # ---- mild joint-room factor (same for every joint) ----
    room = (ctx.arm_limit_margin.min(dim=-1).values / ROOM_MARGIN).clamp(0.0, 1.0)
    room_factor = ROOM_FLOOR + (1.0 - ROOM_FLOOR) * room

    # ---- incomes ----
    far = torch.exp(-dist / TRAVEL_SCALE)
    travel = W_TRAVEL * far * room_factor  # ungated (F3)
    # keeps the hand open at every distance; peaks at the (open) pose -> no parking spot
    open_hand = W_OPEN_TRAVEL * far * open_q * room_factor
    # additive orientation income at every distance: gradient all the way, travel stays plain
    align = W_ALIGN * orient_lin * undisturbed * room_factor
    near = torch.exp(-dist / POSE_SCALE) * undisturbed * room_factor
    # every final-10-cm income carries BOTH floored gates (orientation F10, opposition F15c)
    pose_pos = W_POSE_POS * near * gate
    pose_shape = W_POSE_SHAPE * near * orient_q * oppose_q
    pose_open = W_POSE_OPEN * near * open_q * gate
    settle = W_SETTLE * near * still * gate

    # ---- mild, bounded costs ----
    obj_force = ctx.finger_force.sum(dim=-1) + ctx.palm_force
    contact = -W_CONTACT * torch.tanh(obj_force / CONTACT_SCALE)
    table_contact = -W_TABLE * torch.tanh(ctx.hand_foreign_force / FOREIGN_SCALE)
    tipping = -W_TIP * torch.tanh(spin_ex / TIP_SCALE)

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
    }
    reward = (travel + open_hand + align + pose_pos + pose_shape + pose_open + settle
              + contact + table_contact + tipping)
    return reward, components
