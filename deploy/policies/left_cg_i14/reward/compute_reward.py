import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """0 off the object, ~0.6 at 0.1 N, ->1 above a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate and extra force buys nothing."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe normalisation: a zero vector stays zero (its dot products are then 0)."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _sidewall(normal_z: torch.Tensor) -> torch.Tensor:
    """How HORIZONTAL a surface normal is: 1 on a vertical side wall, 0 on a rim face, a lid or the
    table, ~0.3 on a 45 deg rim edge. Used only from a few centimetres in, always with a floor;
    NEVER on the long-range approach (lesson 2)."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def _smoothstep_near(dist: torch.Tensor, full: float, zero: float) -> torch.Tensor:
    """1 at dist <= full, 0 at dist >= zero, C1-smooth (cubic smoothstep) in between."""
    t = ((zero - dist) / (zero - full)).clamp(min=0.0, max=1.0)
    return t * t * (3.0 - 2.0 * t)


def _gentle_open(dev: torch.Tensor, half: float) -> torch.Tensor:
    """Hypothesis A: gentle open-pose credit, 1 / (1 + (dev/half)^4). Zero slope at dev = 0, so once
    the hand is near the open pose there is no pull towards (or beyond) the open-side action limit;
    it falls to 0.5 at dev = half and towards 0 only for a clearly hooked or fisted hand."""
    x = (dev.clamp(min=0.0) / half)
    return 1.0 / (1.0 + x.pow(4))


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Per-step ceilings (iter_13 code, unchanged except the Hypothesis A changes):
    #   reach            ~0.60  pre-contact only, x (1 - held)
    #                           = 0.55 origin-relative long-range reach (x0.45 with an empty hand)
    #                           + 0.24 origin-relative palm seating (x0.35 engage floor, empty hand)
    #                           + 0.14 closing (segment nearness x max(palm seat, palm band))
    #   open hand         0.22  Hypothesis A: modest (was 0.30) and GENTLE for small deviations; the
    #                           open-POSE requirement is released by a first touch OR the palm band
    #                           (release band of iter_13 unchanged)
    #   finger targets    ---   Hypothesis A: removed (it acted on the saturating commanded targets)
    #   keep the grasp   1.20   links-per-digit + count ladder + palm contact + opposition
    #                    +0.20  pre-hold palm-seat bonus
    #                    +0.15  segment pull (proximal + middle link towards the surface)
    #   lift             2.50   x uprightness x real hold (`held`); NOT x env_mult
    #   transport        2.50   x off_table x env_mult
    #   hold at rest    10.00   real hold, upright, at the goal, still, x env_mult
    #   first success   60.00   one-off; later successes 2.0, x env_mult
    #
    # Hypothesis A (this round): weaken the pressure that pushes the finger commands beyond the
    # open-side action limit.
    #   - open_pose = gentle(mean deviation, half 0.35) x gentle(worst joint, half 0.75), each
    #     1/(1+(d/half)^4): slope 0 at the open pose. Estimates (not measured):
    #       slightly curled (mean 0.1, worst 0.15): 0.99 -> loses ~1 % of what the hook loses
    #       curled 0.2 (worst 0.3):                 0.88 -> loses ~12 % of what the hook loses
    #       measured hook (mean ~0.61, worst ~0.98): ~0.025
    #   - no contact, palm facing a side wall, finger links 2 cm beyond the palm (estimate):
    #       10 cm: release 0     -> open beats hook by ~0.22 x 0.975           = ~0.21 (target >= 0.08)
    #        7 cm: release ~0.5  -> ~0.22 x 0.975 x 0.5 - closing ~0.008       = ~0.10 (target >= 0.08)
    #     <= 5 cm: release ~1    -> curling costs no open income and no target cost; only the
    #              closing credit moves (non-decreasing, rising at 3-4 cm), as in iter_13.
    # =============================================================================================
    W_REACH = 0.55        # origin-relative reach; x0.45 empty hand, fades once held
    W_OPEN = 0.22         # Hypothesis A: modest open-hand income (was 0.30), gentle pose measure
    W_PALM_IN = 0.24      # palm seating beside the origin, level with it, facing the axis
    W_CLOSE = 0.14        # measured flexion x SEGMENT nearness to a wall x max(palm seat, palm band)
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm
    W_DIGIT = 0.30        # per-digit SEGMENT COUNT, every digit weighted alike
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap
    W_PALM_SEAT = 0.20    # palm contact while >=1 digit touches, before a held lift
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)
    W_SEG_PULL = 0.15     # pull of proximal/middle links towards the wall; far below lift and hold
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty)
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    SHOVE_RELIEF = 0.80   # share of the shove cost waived while a digit touches AND the palm closes in
    P_SQUEEZE = 0.50      # force beyond what a hold needs
    P_TABLE = 0.60        # palm or finger links driven onto the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: small, so lifting and carrying beat standing still
    P_CUP_SPIN = 0.06
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    RELEASE_SCALE = 0.50  # N: soft contact scale of the first-touch release (unchanged)

    # palm band of the release and of the closing gate (palm surface distance, iter_13 unchanged)
    RELEASE_FULL = 0.05   # m: fully released at or inside this palm-to-surface distance
    RELEASE_ZERO = 0.09   # m: not released at or beyond this (0.50 at 7 cm, 0.16 at 8 cm)
    # pre-contact closing nearness of the proximal and middle links (iter_13 unchanged)
    CLOSE_SEG_SCALE = 0.03   # m: beyond LINK_DEAD; 4 cm -> 0.61, 5 cm -> 0.43, 7 cm -> 0.22, 10 cm -> 0.08
    CLOSE_SIDE_FLOOR = 0.10  # a rim-facing segment keeps only 10 % of the side-wall nearness

    PAD_SHARE = 0.25      # a digit touching ONLY with its fingertip pad counts this much
    PULL_DEAD = 0.012     # m: a touching link point reads 1-3 cm (link points sit inside the links)
    PULL_SCALE = 0.02     # m: 1.8 cm -> 0.74, 3.3 cm -> 0.35, 5 cm -> 0.15

    # origin-relative approach (shaping scales only, no object dimension)
    LEVEL_K = 2.0         # axial offset from the origin counts twice the radial offset
    REACH_FAR = 0.35      # m: far scale; weighted start distance ~0.35-0.6 m keeps a slope there
    REACH_NEAR = 0.15     # m: near scale
    SEAT_SCALE = 0.08     # m: palm-seating nearness to the origin (compact: stays local to the object)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # scale of the surface-based `seat` measure (grasp ladder)
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link
    # Hypothesis A: gentle open-pose measure (normalised joint deviation from the open default pose)
    OPEN_MEAN_HALF = 0.35    # mean deviation giving half credit (0.1 -> 0.99, 0.2 -> 0.90, 0.6 -> 0.10)
    OPEN_MAX_HALF = 0.75     # worst movable joint giving half credit (lenient: 0.3 -> 0.97, 1.0 -> 0.24)
    OPEN_NEAR = 0.08      # shove relief only: palm "closing in" gate zero at 8 cm or beyond
    OPEN_CONTACT = 0.02   # and full by 2 cm (not used by the open-hand income)
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale
    TILT_DEAD = math.radians(5.0)
    TILT_SCALE = math.radians(10.0)
    SQ_TOTAL = 40.0
    SQ_LINK = 15.0
    FALL_DEAD = 0.03
    FALL_SCALE = 0.20
    SET_DEAD = 0.02
    SET_SCALE = 0.05
    SET_LIFTOFF = 0.005
    SET_BAND = 0.04
    GOAL_TOL_MULT = 3.0
    GOAL_TOL_MIN = 0.03
    GOAL_FINE = 0.04
    SETTLE_FLOOR = 0.40
    DWELL_BASE = 0.70
    DWELL_SCALE = 1.5
    ENV_FLOOR = 0.10
    ENV_W_COUNT = 0.30
    ENV_W_DEPTH = 0.35
    ENV_W_PALM = 0.25
    ENV_W_OPP = 0.10
    ENV_DEPTH_LINKS = 2.0
    ENV_NEAR_SHARE = 0.25
    ENV_NEAR_SCALE = 0.01
    FOLLOW_DEAD = 0.010      # m: resolution of the surface distance, NOT a pose target
    FOLLOW_SCALE = 0.006     # m: 1.1 cm -> 0.85, 1.9 cm -> 0.22, 2.0-2.5 cm -> 0.19-0.08
    WRAP_FOLLOW_FLOOR = 0.30  # a digit on the object by its segments but not following the wall keeps 0.3
    WRAP_FLOOR = 0.30         # wrap factor of env_q with no digit wrapped
    FLOOR_BAND_TOP = 0.020
    FLOOR_BAND_FULL = 0.010

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0
    digit_joint[1, 4:7] = 1.0
    digit_joint[2, 8:11] = 1.0
    digit_joint[3, 12:15] = 1.0
    digit_joint[4, 17:19] = 1.0
    n_digit_joint = digit_joint.sum(-1)

    # ---- height and vertical velocity ------------------------------------------------------------
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    off_table = torch.tanh(height / TABLE_RISE)
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)

    # ---- contact and grasp quality ---------------------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3) middle, distal, pad
    prox_c = _soft_contact(ctx.prox_cup_force)                     # (N,5)   proximal link
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)

    # the four segments of each digit, from the palm outwards: [proximal, middle, distal, pad].
    # Segments count fully, the pad alone only PAD_SHARE.
    seg_c = torch.cat([prox_c.unsqueeze(-1), link_c], dim=-1)      # (N,5,4)
    seg_w = torch.tensor([1.0, 1.0, 1.0, PAD_SHARE], device=device, dtype=dtype)
    seg_cw = seg_c * seg_w                                         # (N,5,4)
    digit_touch = seg_cw.amax(dim=-1)                              # (N,5) held by finger segments
    n_touch = digit_touch.sum(dim=-1)
    finger_any = digit_touch[:, 1:].amax(dim=-1)

    # plain "touching at all" (any segment, the pad included at full weight)
    touch_any = seg_c.amax(dim=-1)                                 # (N,5)
    n_any = touch_any.sum(dim=-1)
    any_digit = touch_any.amax(dim=-1)                             # (N,)
    grip = (0.8 * n_any / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    link_side = _sidewall(ctx.link_surface_normal[..., 2])         # (N,5,3)
    prox_side = _sidewall(ctx.prox_surface_normal[..., 2])         # (N,5)
    seg_side = torch.cat([prox_side.unsqueeze(-1), link_side], dim=-1)
    seg_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * seg_side
    digit_links = ((seg_cw * seg_side_w).sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)  # (N,5)

    # ---- open-pose measure (mean AND worst movable joint) ----------------------------------------
    # Hypothesis A: GENTLE for small deviations (zero slope at the open pose, so nothing pushes the
    # finger commands beyond the open-side action limit); only a clearly hooked or fisted hand loses
    # the credit. The worst joint is a lenient multiplier that only catches one strongly curled joint.
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = _gentle_open(dev_mean, OPEN_MEAN_HALF) * _gentle_open(dev_max, OPEN_MAX_HALF)

    # ---- measured per-digit flexion --------------------------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint
    flex = (closure / 0.5).clamp(max=1.0)

    # ---- palm against the object's real surface (finger terms, open release, shove relief) -------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    # geometric palm arrival (the non-contact part of `seat`, used by the grasp ladder)
    palm_arrive = palm_near * (0.3 + 0.7 * face) * palm_side_w
    seat = torch.maximum(palm_arrive, palm_c)
    cq = 0.35 + 0.65 * seat

    # palm band (iter_13 unchanged): smoothstep in the palm distance (1 at <= 5 cm, 0.50 at 7 cm,
    # 0 at >= 9 cm) x facing x side-wall weight (not faded with distance)
    band_side_w = SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side
    palm_band = (_smoothstep_near(palm_d, RELEASE_FULL, RELEASE_ZERO)
                 * (0.3 + 0.7 * face) * band_side_w)

    # palm "closing in" gate: full by 2 cm, zero at 8 cm or beyond (shove relief only)
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)

    # ---- palm relative to the object ORIGIN (iter_10, unchanged) ---------------------------------
    rel = ctx.palm_pos - ctx.cup_pos                               # (N,3) origin -> palm
    h_ax = (rel * ctx.cup_axis).sum(-1)                            # (N,) axial offset (+ above)
    radial = rel - h_ax.unsqueeze(-1) * ctx.cup_axis               # (N,3) axis -> palm, perpendicular
    r_rad = radial.norm(dim=-1)                                    # (N,) distance from the axis
    d_org = torch.sqrt(r_rad.pow(2) + (LEVEL_K * h_ax).pow(2) + 1e-12)
    face_org = (-(ctx.palm_normal * _unit(radial)).sum(-1)).clamp(min=0.0, max=1.0)

    # ---- opposition from the surface normals at the touching segments ----------------------------
    seg_normals = torch.cat([ctx.prox_surface_normal.unsqueeze(2), ctx.link_surface_normal], dim=2)
    thumb_n = _unit((seg_cw[:, 0, :].unsqueeze(-1) * seg_normals[:, 0]).sum(dim=1))
    finger_n = _unit((seg_cw[:, 1:, :].unsqueeze(-1) * seg_normals[:, 1:]).sum(dim=(1, 2)))
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)

    # hold gate: counts digits held by their segments (a pad alone is PAD_SHARE of a digit)
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # ---- envelope quality (multiplier on transport, hold and success only) ------------------------
    core_c = torch.maximum(prox_c, link_c[:, :, 0])                                   # (N,5)
    tip_c = torch.maximum(link_c[:, :, 1], link_c[:, :, 2])                           # (N,5)
    core_dist = torch.minimum(ctx.prox_surface_dist, ctx.link_surface_dist[:, :, 0])  # (N,5)
    core_gap = (core_dist - LINK_DEAD).clamp(min=0.0)
    core_eng = torch.maximum(torch.maximum(core_c, PAD_SHARE * tip_c),
                             ENV_NEAR_SHARE * torch.exp(-core_gap / ENV_NEAR_SCALE))  # (N,5)
    link_dist = ctx.link_surface_dist
    lift_off = link_dist.amax(dim=-1) - link_dist.amin(dim=-1)
    follow = torch.exp(-(lift_off - FOLLOW_DEAD).clamp(min=0.0) / FOLLOW_SCALE)       # (N,5)
    digit_wrap = core_eng * (WRAP_FOLLOW_FLOOR + (1.0 - WRAP_FOLLOW_FLOOR) * follow)  # (N,5)
    hand_wrap = digit_wrap.mean(dim=-1)                                               # (N,)
    env_count = hand_wrap
    core_depth = prox_c + link_c[:, :, 0] + PAD_SHARE * (link_c[:, :, 1] + link_c[:, :, 2])
    count_depth = (core_depth / ENV_DEPTH_LINKS).clamp(max=1.0)                       # (N,5)
    env_depth = count_depth.mean(dim=-1)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)
    wrap_factor = WRAP_FLOOR + (1.0 - WRAP_FLOOR) * hand_wrap
    env_q = env_base * wrap_factor
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- fading out once the object is really held
    # =============================================================================================
    fade = loose
    reach_far = 1.0 - torch.tanh(d_org / REACH_FAR)
    reach_near = 1.0 - torch.tanh(d_org / REACH_NEAR)
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    # =============================================================================================
    # OPEN HAND -- always on during the approach; the pose requirement is released smoothly by a
    # first touch OR by the palm band (iter_13 release unchanged). Hypothesis A: modest weight and
    # the gentle open_pose above; the finger-target cost is removed.
    # =============================================================================================
    digit_force = torch.maximum(ctx.prox_cup_force.clamp(min=0.0),
                                ctx.link_cup_force.clamp(min=0.0).amax(dim=-1))       # (N,5)
    touch_force = torch.maximum(digit_force.amax(dim=-1), ctx.palm_cup_force.clamp(min=0.0))
    first_touch = _soft_contact(touch_force, RELEASE_SCALE)                           # (N,)
    # release = first touch OR palm band. Fully released by 5 cm, half at 7 cm, none from 9 cm on.
    release = 1.0 - (1.0 - first_touch) * (1.0 - palm_band)
    open_keep = open_pose + release * (1.0 - open_pose)
    # before release the level falls with grip; once released it no longer falls with contact
    # while the object is on the table (a first touch cannot cost open income); in the air 1 - grip
    open_level = (1.0 - grip * (1.0 - release * on_table)).clamp(min=0.0, max=1.0)
    r_open = W_OPEN * open_level * open_keep

    # flexion counts in the engage factor only for digits that touch the object (iter_11, unchanged)
    touch_flex = (flex * touch_any).mean(dim=-1)                   # (N,)
    engage = (0.5 * touch_flex + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    seat_org = torch.exp(-d_org / SEAT_SCALE)
    r_palm_in = W_PALM_IN * fade * seat_org * (0.3 + 0.7 * face_org) * seat_engage

    # pre-contact closing credit from the FINGER SEGMENTS (iter_13 unchanged): per digit, the
    # side-wall-weighted nearness of its proximal and middle links x its measured flexion, gated by
    # max(palm seat, palm band); far from the object flexion pays nothing.
    prox_close = (torch.exp(-(ctx.prox_surface_dist - LINK_DEAD).clamp(min=0.0) / CLOSE_SEG_SCALE)
                  * (CLOSE_SIDE_FLOOR + (1.0 - CLOSE_SIDE_FLOOR) * prox_side))        # (N,5)
    mid_close = (torch.exp(-(ctx.link_surface_dist[:, :, 0] - LINK_DEAD).clamp(min=0.0)
                           / CLOSE_SEG_SCALE)
                 * (CLOSE_SIDE_FLOOR + (1.0 - CLOSE_SIDE_FLOOR) * link_side[:, :, 0]))  # (N,5)
    near_digit = 0.5 * (prox_close + mid_close)                                       # (N,5)
    close_gate = torch.maximum(seat, palm_band)
    r_close = W_CLOSE * fade * close_gate * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total (+ pre-hold palm-seat bonus + segment pull)
    # =============================================================================================
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_any / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    pre_hold = (1.0 - held * (0.5 + 0.5 * off_table)).clamp(min=0.0, max=1.0)
    r_palm_seat = W_PALM_SEAT * palm_c * any_digit * pre_hold

    prox_near = torch.exp(-(ctx.prox_surface_dist - PULL_DEAD).clamp(min=0.0) / PULL_SCALE)       # (N,5)
    mid_near = torch.exp(-(ctx.link_surface_dist[:, :, 0] - PULL_DEAD).clamp(min=0.0) / PULL_SCALE)
    seg_pull = touch_any * 0.5 * (prox_near + mid_near)                                           # (N,5)
    r_seg_pull = W_SEG_PULL * seg_pull.mean(dim=-1)

    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_palm_seat = r_palm_seat * table_decay
    r_oppose = r_oppose * table_decay
    r_seg_pull = r_seg_pull * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total, gated by `held` and uprightness only
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    r_lift = W_LIFT * held * upright * lift_shape
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, x off_table, keeps env_mult
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, x off_table, keeps env_mult
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded, no latch
    # =============================================================================================
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    shove_raw = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                            + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    palm_closing = torch.maximum((1.0 - open_gate) * (0.5 + 0.5 * face), palm_c)
    shove_relief = SHOVE_RELIEF * any_digit * palm_closing
    p_shove = shove_raw * (1.0 - shove_relief)
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    hand_rel = ctx.hand_z_min - ctx.table_z
    floor_t = ((FLOOR_BAND_TOP - hand_rel) / (FLOOR_BAND_TOP - FLOOR_BAND_FULL)).clamp(min=0.0, max=1.0)
    floor_pen = floor_t * floor_t * (3.0 - 2.0 * floor_t)
    p_table = P_TABLE * 0.5 * (palm_pen + floor_pen)
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = held * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + P_CUP_SPIN * torch.tanh(spin / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_reach + r_open + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_palm_seat + r_oppose + r_seg_pull
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero

    components = {
        "reach": r_reach,
        "open_hand": r_open,
        "palm_in": r_palm_in,
        "close": r_close,
        "grasp_count": r_count,
        "grasp_digit": r_digit,
        "grasp_palm": r_palm,
        "grasp_palm_seat": r_palm_seat,
        "grasp_oppose": r_oppose,
        "grasp_seg_pull": r_seg_pull,
        "lift_height": r_lift,
        "lift_rise": r_rise,
        "goal_reach": r_goal,
        "goal_progress": r_progress,
        "goal_hold": r_hold,
        "success_bonus": r_success,
        "pen_drop": -p_drop,
        "pen_setdown": -p_setdown,
        "pen_stall": -p_stall,
        "pen_drift": -p_drift,
        "pen_descend": -p_descend,
        "pen_tilt": -p_tilt,
        "pen_shove": -p_shove,
        "pen_squeeze": -p_squeeze,
        "pen_table": -p_table,
        "pen_arm_vel": -p_arm_vel,
        "pen_arm_fast": -p_arm_fast,
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components
