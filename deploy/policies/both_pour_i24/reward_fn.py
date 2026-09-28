import torch
import math


# ---------------------------------------------------------------------------------------------
# geometry constant: the pouring lip sits this far from the rim centre (ctx doc, r = 4.1 cm)
_LIP_R = 0.041


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def _step(x: torch.Tensor) -> torch.Tensor:
    # smoothstep on an already normalised argument
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _carry_phase(grasped_f: torch.Tensor, h: torch.Tensor,
                 grasp_lo: float, grasp_hi: float, free_lo: float, free_hi: float) -> torch.Tensor:
    # (N,) in [0,1]: 0 on the table, 1 once the cup is grasped AND off the table.
    # high above the table the (flickering) grasp flag is no longer needed.
    held_ramp = grasped_f * _step((h - grasp_lo) / (grasp_hi - grasp_lo))
    high_ramp = _step((h - free_lo) / (free_hi - free_lo))
    return torch.maximum(held_ramp, high_ramp)


def _grasp_posture(finger_force: torch.Tensor, palm_force: torch.Tensor,
                   wrap_count: torch.Tensor, dt: torch.dtype):
    """Posture quality of ONE hand on its own cup, in [0,1], plus its two defect measures.

    This is the heart of round 21. It is built only from quantities that describe the shape of
    the contact, so it stays meaningful while the cup rotates:
      * wrap   - fingers whose middle/distal link touches the cup (an envelope needs >= 3)
      * palm   - palm contact force (the traces show it collapsing 5.5 N -> 1.25 N during the pour)
      * thumb  - thumb contact force (the thumb is the finger that gets pushed open when the cup
                 slides out of the palm toward the fingertips)
      * tip_f  - fingers touching with the fingertip ONLY = (contacting fingers) - (wrapped fingers);
                 a pinch multiplies the whole quality by as little as 0.4
    """
    wrap_n = wrap_count.to(dt)
    wrap_q = torch.clamp(wrap_n / 3.0, 0.0, 1.0)                       # 1.0 at >= 3 wrapped fingers
    n_contact = torch.sum((finger_force > 0.3).to(dt), dim=-1)          # 0.3 N = contact threshold
    tip_only = torch.clamp(n_contact - wrap_n, min=0.0)
    tip_f = torch.clamp(tip_only / 2.0, 0.0, 1.0)                       # 2 tip-only fingers = full defect
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)         # 0.76 at the required 3 N
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    q = (0.45 * wrap_q + 0.30 * palm_q + 0.25 * thumb_q) * (1.0 - 0.6 * tip_f)
    return q, tip_f, wrap_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """How closely the hand still holds the cup the way it did at the moment it lifted it, in [0,1].

    All three parts are RELATIVE to that moment (operator 09.22: no absolute force target):
      * pose  - palm + fingertips in cup coordinates; a hand that keeps its grip has them fixed no matter
                how the cup is rotated (i18/i23 traces: they drift 2.4-4.2 cm while tilting)
      * share - palm/finger force DISTRIBUTION (i18/i23: about half of it moves while tilting)
      * mag   - total grip force relative to the lift moment (i18/i23: 3-5x harder while tilting)
    Before the reference exists (cup not lifted yet) the factor is 1 - nothing to keep yet.
    """
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = 0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim
    return torch.where(valid, keep, torch.ones_like(keep))


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    N = ctx.src_palm_pos.shape[0]
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ======================================================================== weights / constants
    # --- stage 1-3 (reach, grasp, lift): deliberately at the iter_18 level. iter_17 cut this
    #     income and the source cup was then never lifted at all, so nothing here gets weaker.
    APPROACH_W = 0.6
    APPROACH_K = 4.0
    REACH_W = 0.3
    REACH_FAR, REACH_NEAR = 0.26, 0.14
    GRASP_W = 0.8
    CLOSE_W = 0.8
    CLOSE_NORM = 0.35
    BOTH_W = 0.8
    LIFT_SRC_W = 3.0          # iter_18 value that lifted reliably
    LIFT_RCV_W = 1.5
    LIFT_TARGET_SRC = 0.25
    LIFT_TARGET_RCV = 0.10
    LIFTOFF_H = 0.03          # [m] e-fold of the lift-off part: the first cm pays the most
    RCV_LIFT_FREE, RCV_LIFT_SIGMA = 0.15, 0.08

    # --- goal 1: grasp posture that survives rotation. Paid EVERY held step, phase independent,
    #     and amplified (not replaced) during the pour, so losing a wrapped finger at step 600
    #     costs more than at step 200 instead of less.
    POSTURE_SRC_W = 3.0       # comparable with lift_src 3.0: the posture is a first-class income
    POSTURE_RCV_W = 1.2
    POSTURE_KEEP_W = 1.5      # after the lift: "still the grasp you lifted with" (half the pre-lift weight)
    POSTURE_KEEP_RCV_W = 0.8
    SIDE_W = 1.5              # pour side: palm away from the receiver while carrying toward it
    GEO_PALM_FROM = 0.06      # [m] palm this far UNDER the tilted cup = geometry 0 (i23 sits at +6 cm)
    GEO_TIP_FROM = 0.04       # [m] fingertip this far ON TOP of the tilted cup = 0 (i23 sits at -4 cm)
    POSTURE_POUR_BONUS = 0.6  # x1.6 while pouring / after the pour
    TIP_W = 1.2               # explicit charge for fingertip-only holding (logged separately)
    TIP_POUR_BONUS = 1.0      # x2.0 during the pour
    PALM_SLIP_W = 1.5         # charge for the palm force collapsing below PALM_MIN while held
    PALM_MIN = 3.0            # [N] acceptance target for the pour/late phases
    GRIP_FLOOR = 0.15         # rotating with a ruined grasp keeps only 15 % of the tilt income
    HOLD_FLOOR = 0.20         # iter_25: rotation income kept when the hand geometry is wrong
    GEO_MARGIN = 0.03         # [m] palm this far ABOVE / fingertips this far BELOW the cup axis = full credit
    GEO_TILT_LO = math.radians(15.0)   # geometry requirement starts here ...
    GEO_TILT_HI = math.radians(45.0)   # ... and is fully required from here on
    KEEP_POS = 0.02           # [m] hand-in-cup drift from the lift moment at which pose similarity is 0.37
    KEEP_MAG = 0.7            # |log(force / force at lift)| at which force similarity is 0.37 (2x -> 0.37)

    # --- transport / aiming
    CONV_W = 2.0
    CONV_FAR = 0.40
    BRING_W = 1.5
    BT_K = 10.0
    AIM_W = 2.5
    AIM_K = 15.0
    LIP_DEADBAND = 0.02
    APPR_FULL, APPR_ZERO = 0.08, 0.24       # lip xy window in which tilting may start to pay
    SAFE_FULL, SAFE_ZERO = 0.050, 0.075     # release window, inside the env latch radius 8.1 cm
    HOVER_LO, HOVER_HI = 0.06, 0.11
    HOVER_FREE0, HOVER_SIGMA = 0.10, 0.08
    LIP_Z_LO, LIP_Z_HI = 0.0, 0.02
    LIP_Z_FREE0, LIP_Z_SIGMA = 0.07, 0.10
    LIP_ABOVE_FLOOR = 0.35
    DIR_FULL, DIR_ZERO = 0.0, 0.28          # pour_dir_xy[:,0] > 0.3 voids success
    DIR_W, DIR_PEN_W = 1.5, 1.0
    SIDE_VOID, SIDE_OK = 0.03, 0.10
    SIDE_PEN_W = 2.0
    HOME_K, HOME_W = 10.0, 0.8
    HOME_SRC_FLOOR = 0.2

    # --- goal 3: fill dependent pour angle
    THETA_REL_BASE = 1.25     # [rad] first bead leaves a FULL cup at ~72 deg
    THETA_REL_SPAN = 0.60     # ... and a nearly empty one at ~106 deg
    TILT_PEAK_ABOVE = 0.10    # income peaks at theta_rel + 0.10 rad
    TILT_ZERO_ABOVE = 0.60    # ... and is zero at theta_rel + 0.60 rad
    OVERTILT_ABOVE = 0.25     # linear, unsaturated penalty starts here
    OVERTILT_W, OVERTILT_MAX = 6.0, 6.0
    TILT_PRE_W = 4.0          # below the premature limit, inside the approach zone
    TILT_MAIN_W = 20.0        # the real pour income: only with the lip over the receiver mouth
    POSE_W = 6.0              # holding the release pose
    PRE_MARGIN = 0.17         # [rad] theta_pre = premature limit - 10 deg
    SRC_TILT_FREE = 0.20      # [rad] tilt that is free far from the receiver
    SRC_TILT_SCALE = 0.20
    SRC_TILT_SIGMA = 0.25
    SRC_UP_W_TABLE, SRC_UP_W_CARRY = 1.0, 1.5
    LATCH_W = 1.0
    BAND_BELOW, BAND_ABOVE = 0.10, 0.40

    # --- beads
    POUR_DELTA_W = 120.0
    FLOW_CAP = 0.02           # per-step cap on the increment that is paid
    FLOW_W = 20.0
    TRANSFER_W = 5.0
    SPILL_W = 50.0
    SETTLE_FRAC = 0.05
    POST_POUR_FRAC = 0.5      # the task succeeds at half the beads
    HOLD_SRC_W, HOLD_RCV_W = 1.5, 0.5

    # --- receiver constraints
    RCV_TILT_FREE, RCV_TILT_SIGMA = 0.12, 0.20
    RCV_PEN_SCALE = 0.25
    RCV_GATE_FLOOR = 0.3
    RCV_UP_W, RCV_UP_POUR_W = 1.0, 1.0
    RCV_STILL_FREE, RCV_STILL_SCALE = 0.05, 0.15
    RCV_STILL_W, RCV_STILL_NEAR_W, RCV_STILL_POUR_W = 0.25, 0.25, 0.50
    RCV_TOPPLED = 1.2
    DROP_DEPTH = 0.03

    # --- safety / collisions (real robot)
    FORCE_SCALE = 5.0
    CONTACT_W, CONTACT_SCALE = 2.0, 2.0
    HIT_FORCE, HIT_EVENT_W, HIT_LATCH_W = 0.5, 5.0, 0.3
    FOREIGN_DEADBAND, CLEAN_DEADBAND = 1.0, 0.5
    FOREIGN_W_FREE, FOREIGN_W_HELD = 0.3, 1.5
    CLEAN_FLOOR = 0.5
    PRELIFT_SAFE, PRELIFT_ZERO, PRELIFT_W = 0.14, 0.10, 2.0
    NEAR_DIST, CLOSE_V_FREE, CLOSE_V_SCALE = 0.15, 0.15, 0.20

    # --- goal 2: stable control. Base rates are the iter_18 values (iter_19 multiplied them by 10
    #     in every phase and made both the grasp and the success rate worse). Everything stronger
    #     is gated on the carry phase or on the hand channels only.
    PALM_RATE_W, HAND_RATE_W = 0.01, 0.005
    PALM_RATE_CARRY_W, HAND_RATE_CARRY_W = 0.05, 0.02
    SAT_LO, SAT_W = 0.7, 0.15
    ADR_RAMP_FULL = 0.2       # control charges reach full weight at ADR progress 0.2 (6/30)
    FLIP_CARRY_W = 0.6        # sign flip of consecutive raw palm+hand commands, carry gate only
    HAND_FLIP_W = 0.4         # flip of the 6 hand channels while a cup is held: it directly
                              # opens/closes the fingers and is what ruins the posture
    SRC_V_FREE, SRC_V_SCALE = 0.15, 0.15
    SRC_W_FREE, SRC_W_SCALE = 1.5, 1.5
    RCV_V_FREE, RCV_V_SCALE = 0.05, 0.10
    RCV_W_FREE, RCV_W_SCALE = 0.5, 1.0
    ROT_SPEED_FREE = 1.0

    SUCCESS_W = 20.0

    # ======================================================================== basic state
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    src_carry = _carry_phase(g_src, h_src, 0.015, 0.05, 0.06, 0.10)
    rcv_carry = _carry_phase(g_rcv, h_rcv, 0.015, 0.05, 0.06, 0.10)
    held_src = torch.maximum(g_src, src_carry)     # a thumb contact that flickers is not fatal
    held_rcv = torch.maximum(g_rcv, rcv_carry)
    carry_raw = src_carry * rcv_carry               # geometric fact: both cups are off the table
    not_nested_f = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress / SETTLE_FRAC)

    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    clean_soft = CLEAN_FLOOR + (1.0 - CLEAN_FLOOR) * clean

    # ======================================================================== posture quality
    q_src, tip_src, wrapq_src = _grasp_posture(ctx.src_finger_force, ctx.src_palm_force,
                                               ctx.src_wrap_count, dt)
    q_rcv, tip_rcv, wrapq_rcv = _grasp_posture(ctx.rcv_finger_force, ctx.rcv_palm_force,
                                               ctx.rcv_wrap_count, dt)
    # envelope factor per hand: 0 at <= 1 wrapped finger, 1 at >= 3
    env_src = _step((ctx.src_wrap_count.to(dt) - 1.0) / 2.0)
    env_rcv = _step((ctx.rcv_wrap_count.to(dt) - 1.0) / 2.0)
    # the gate every rotation / pour income is multiplied by: a degrading grasp earns much less
    grip_src = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * (0.5 * q_src + 0.5 * env_src)
    grip_rcv = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * (0.5 * q_rcv + 0.5 * env_rcv)

    # ======================================================================== receiver upright gate
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    rcv_home = torch.exp(-HOME_K * d_home)
    dir_x = ctx.pour_dir_xy[:, 0].to(dt)
    dir_ok = 1.0 - _step((dir_x - DIR_FULL) / (DIR_ZERO - DIR_FULL))

    # carry gate of every transport-stage income: both cups lifted AND both ENVELOPED AND the
    # receiver on its own side. The envelope factors are what keep the posture alive after pickup.
    carry_both = carry_raw * (0.25 + 0.75 * env_src) * (0.25 + 0.75 * env_rcv) * (0.2 + 0.8 * side_ok)

    # ======================================================================== pouring geometry
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    lip_off = torch.clamp(lip_xy - LIP_DEADBAND, min=0.0)
    approach_gate = 1.0 - _step((lip_xy - APPR_FULL) / (APPR_ZERO - APPR_FULL))
    safe = 1.0 - _step((lip_xy - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))

    tilt_mag = ctx.src_cup_tilt
    limit = ctx.premature_tilt_limit.to(dt)
    release_tilt = limit + math.radians(20.0)
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)
    # signed tilt toward the receiver, never larger than the total tilt
    theta_eff = torch.minimum(torch.clamp(ctx.src_tilt_toward_rcv.to(dt), min=0.0), tilt_mag)
    # ---- iter_25 ---- hand geometry while the cup is tilted (operator 09.22): the palm must be on the
    # UPPER side of the tilted cup and the other fingers UNDER it, carrying the cup. i23 did the
    # opposite (palm 6 cm on the gravity side, the 4 fingers 4 cm on top, no finger underneath).
    # Grasp maintenance is what ENABLES the rotation income: wrong geometry keeps only 20 %.
    u_src = ctx.src_cup_up.to(dt)
    down = torch.zeros_like(u_src)
    down[:, 2] = -1.0
    d_perp = down - torch.sum(down * u_src, dim=-1, keepdim=True) * u_src          # gravity across the axis
    d_low = d_perp / torch.clamp(torch.norm(d_perp, dim=-1, keepdim=True), min=1e-6)
    palm_low = torch.sum((ctx.src_palm_pos - ctx.src_cup_pos) * d_low, dim=-1)       # + = under the cup
    tips_low = torch.sum((ctx.src_tips_pos - ctx.src_cup_pos.unsqueeze(1)) * d_low.unsqueeze(1), dim=-1)
    # continuous over the whole range the hand can be in, so a policy that starts with the palm UNDER
    # the cup (i23: +6 cm) still sees which way is better: 0 at palm 6 cm under -> 1 at 3 cm above,
    # and per finger 0 at 4 cm on top -> 1 at 3 cm under (touching)
    palm_top = _step((GEO_PALM_FROM - palm_low) / (GEO_PALM_FROM + GEO_MARGIN))
    under = _step((tips_low[:, 1:] + GEO_TIP_FROM) / (GEO_TIP_FROM + GEO_MARGIN)) \
        * (ctx.src_finger_force[:, 1:] > 0.3).to(dt)
    fingers_under = torch.clamp(torch.sum(under, dim=-1) / 2.0, 0.0, 1.0)             # >= 2 fingers carry it
    geo_src = palm_top * fingers_under
    geo_on = _step((tilt_mag - GEO_TILT_LO) / (GEO_TILT_HI - GEO_TILT_LO))
    # pour side, defined while the cup is still upright: the palm must sit on the side of the source cup
    # AWAY from the receiver so that tipping toward the receiver turns the palm on top. i23 carried the
    # source cup past the receiver and tipped toward the palm (cos +0.77). Linear, so there is a
    # gradient from any arrangement: cos +1 -> 0, cos -0.6 or less -> 1.
    palm_side = (ctx.src_palm_pos - ctx.src_cup_pos)[:, :2]
    palm_side = palm_side / torch.clamp(torch.norm(palm_side, dim=-1, keepdim=True), min=1e-6)
    side_cos = torch.sum(palm_side * ctx.pour_dir_xy.to(dt), dim=-1)
    side_lin = torch.clamp((1.0 - side_cos) / 1.6, 0.0, 1.0)
    # operator 09.22: no absolute force target - the grasp must stay as it was when the cup was first
    # lifted (hand pose in cup coordinates, force distribution, force level), and keeping it is what
    # unlocks the rotation income together with the palm-on-top / fingers-under geometry
    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    rot_hold_src = HOLD_FLOOR + (1.0 - HOLD_FLOOR) * env_src \
        * ((1.0 - geo_on) + geo_on * geo_src * keep_src)

    # height budget grows with the tilt: the lip drops r*sin(theta) below the rim centre, so a cup
    # that must turn further has to ride higher. Rotating and raising together must not be taxed.
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    hover_free = HOVER_FREE0 + _LIP_R * sin_t
    lip_free = LIP_Z_FREE0 + _LIP_R * sin_t
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_eq - hover_free, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _step((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _step((dz_eq + 0.04) / (HOVER_HI + 0.04)) * hover_decay
    lip_above_raw = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(lip_dz - lip_free, min=0.0) / LIP_Z_SIGMA) ** 2)
    lip_above = LIP_ABOVE_FLOOR + (1.0 - LIP_ABOVE_FLOOR) * lip_above_raw
    z_ok = torch.maximum(hover, safe * lip_above_raw * lip_high)

    # fill dependent release angle and the phase switch used by the posture income
    f_rem = torch.clamp(ctx.bead_fill_level.to(dt) * ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    theta_rel = THETA_REL_BASE + (1.0 - f_rem) * THETA_REL_SPAN
    theta_peak = theta_rel + TILT_PEAK_ABOVE
    post_pour = _step(ctx.bead_in_target_frac / POST_POUR_FRAC)
    turning = _step((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.1))
    pour_phase = torch.maximum(carry_raw * safe * turning, post_pour)

    # ======================================================================== stage 1: approach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-APPROACH_K * d_src)
    approach_rcv = APPROACH_W * torch.exp(-APPROACH_K * d_rcv)
    reach_src = REACH_W * _step((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _step((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ======================================================================== stage 2: grasp
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = GRASP_W * held_src + CLOSE_W * torch.exp(-6.0 * d_src) * clos_src
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * torch.exp(-6.0 * d_rcv) * clos_rcv
    both_grasped = BOTH_W * held_src * held_rcv

    # ---- GOAL 1 ---- posture income: unconditional on phase, amplified while pouring ------------
    # before the lift the envelope quality q builds the grasp; from the lift on the posture income is
    # "the grasp is still the one you lifted with", and once tilted it also needs the pour geometry
    v_src = ctx.src_grasp_ref_valid.to(dt)
    v_rcv = ctx.rcv_grasp_ref_valid.to(dt)
    q_hold_src = (1.0 - v_src) * q_src + v_src * keep_src
    q_hold_rcv = (1.0 - v_rcv) * q_rcv + v_rcv * keep_rcv
    # after the lift the keep income is smaller than the pre-lift envelope income and is NOT multiplied
    # by the pour geometry: holding a lifted cup still must never out-earn turning it (local minimum),
    # the geometry is enforced through the rotation income only
    posture_src = held_src * ((1.0 - v_src) * POSTURE_SRC_W * q_src + v_src * POSTURE_KEEP_W * keep_src) \
        * (1.0 + POSTURE_POUR_BONUS * pour_phase)
    posture_rcv = held_rcv * ((1.0 - v_rcv) * POSTURE_RCV_W * q_rcv + v_rcv * POSTURE_KEEP_RCV_W * keep_rcv)
    # iter_24: no pour-phase amplification - tilting further must never raise a grasp penalty by itself
    tip_only_src = -TIP_W * held_src * tip_src
    tip_only_rcv = -TIP_W * held_rcv * tip_rcv

    # ======================================================================== stage 3: lift
    h_src_pos = torch.clamp(h_src, min=0.0)
    lift_shape_src = 0.5 * (1.0 - torch.exp(-h_src_pos / LIFTOFF_H)) \
        + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    tilt_over = torch.clamp(tilt_mag - (SRC_TILT_FREE + approach_gate
                                        * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)), min=0.0) * (1.0 - safe)
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)
    # the lift pays at the iter_18 strength; the tilt corridor may halve it but never zero it,
    # and the envelope adds up to +25 % instead of gating the lift away
    lift_src = LIFT_SRC_W * held_src * lift_shape_src * (0.5 + 0.5 * src_up) * (1.0 + 0.25 * env_src)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = LIFT_RCV_W * held_rcv * torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0) \
        * rcv_band * (0.5 + 0.5 * rcv_up) * (0.25 + 0.75 * env_rcv)
    both_lifted = BOTH_W * carry_both

    # ======================================================================== stage 4: transport
    conv_xy = torch.clamp(1.0 - lip_off / CONV_FAR, 0.0, 1.0)
    converge = CONV_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy
    bring_together = BRING_W * carry_both * (0.3 + 0.7 * dir_ok) * not_nested_f * clean_soft \
        * src_up * torch.exp(-BT_K * lip_off) * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean_soft
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim = AIM_W * stack_gate * grip_src * aim_xy * z_ok * rcv_up_soft * (0.3 + 0.7 * dir_ok)
    pour_dir = DIR_W * carry_both * not_nested_f * conv_xy * dir_ok
    # bring the source cup in from the side that will put the palm on top (paid while carrying toward
    # the receiver, before and during the tilt)
    pour_side = SIDE_W * carry_both * not_nested_f * conv_xy * side_lin
    pour_dir_pen = -DIR_PEN_W * carry_raw * approach_gate * _step((dir_x - 0.10) / 0.25)
    home_src_f = HOME_SRC_FLOOR + (1.0 - HOME_SRC_FLOOR) * src_carry
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_f = 1.0 - torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)
    home_rcv = HOME_W * home_src_f * rcv_carry * grip_rcv * rcv_up * rcv_band * side_ok * rcv_home * still_f
    side_margin = -SIDE_PEN_W * held_rcv * (1.0 - side_ok)

    # ======================================================================== stage 5: pour angle
    # ---- GOAL 3 ---- one bell per episode: rises to theta_rel + 0.10 rad, zero at + 0.60 rad.
    # There is NO flat income above the peak, which is what pinned every earlier run at 150 deg.
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    rise = torch.clamp(theta_eff / torch.clamp(theta_peak, min=0.3), 0.0, 1.0)
    decline = torch.clamp((theta_rel + TILT_ZERO_ABOVE - tilt_mag) / (TILT_ZERO_ABOVE - TILT_PEAK_ABOVE),
                          0.0, 1.0)
    bell = rise * decline
    # every rotation income is multiplied by the CURRENT posture, so turning the cup with a
    # degrading grasp is worth a fraction of turning it with an envelope
    tilt_gate = carry_both * not_nested_f * clean_soft * clear * beads_left * rcv_up_soft \
        * dir_ok * rot_hold_src
    # (a) below the premature limit and inside the approach zone: safe, cannot latch, gives the
    #     gradient that starts the rotation once the cup is high and on its way
    tilt_pre = TILT_PRE_W * tilt_gate * (torch.minimum(theta_eff, theta_pre)
                                         / torch.clamp(theta_peak, min=0.3)) * approach_gate * hover_wide
    # (b) the real pour: only with the lip inside the release radius and above the receiver rim
    # ---- iter_23 FIX ---- the subtraction exists only so the rotation below theta_pre is not paid
    # twice (tilt_pre already pays it at TILT_PRE_W). Subtracting the FULL fraction cancelled the
    # whole bell for every theta <= theta_pre (decline is still 1 there), so tilt_main was
    # identically 0 below 50 deg and iter_22 stalled at 22 deg for 1180 epochs. Remove only what
    # tilt_pre actually paid, scaled into tilt_main units.
    tilt_main = TILT_MAIN_W * tilt_gate * torch.clamp(
        bell - (TILT_PRE_W / TILT_MAIN_W) * torch.minimum(theta_eff, theta_pre)
        / torch.clamp(theta_peak, min=0.3), min=0.0) \
        * safe * lip_above
    # holding the release pose (lip over the mouth, angle in the release band, receiver upright)
    pour_band = _step((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = POSE_W * stack_gate * rot_hold_src * beads_left * safe * lip_above * aim_xy \
        * pour_band * decline * rcv_up * dir_ok
    # linear, unsaturated charge past theta_rel + 0.25 rad, only while the cup is carried
    overtilt = -torch.clamp(OVERTILT_W * torch.clamp(tilt_mag - (theta_rel + OVERTILT_ABOVE), min=0.0),
                            max=OVERTILT_MAX) * src_carry
    # keep the cup near upright until it is lifted AND near the receiver
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src \
        * torch.tanh(tilt_over / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ======================================================================== stage 6: beads
    d_in = ctx.d_in_target.to(dt)
    pour_delta = POUR_DELTA_W * torch.clamp(d_in, -1.0, FLOW_CAP)   # increments, per-step cap
    flow_excess = -FLOW_W * torch.relu(d_in - FLOW_CAP)             # dumping everything at once
    transferred = TRANSFER_W * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0) * not_nested_f * rcv_up
    spill_delta = -SPILL_W * settled * torch.clamp(ctx.d_spill, min=0.0)
    hold_src = HOLD_SRC_W * post_pour * held_src * q_src
    hold_rcv = HOLD_RCV_W * post_pour * held_rcv * q_rcv * rcv_up

    # ======================================================================== receiver constraints
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry \
        * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)
    still_w = RCV_STILL_W + RCV_STILL_NEAR_W * approach_gate * src_carry + RCV_STILL_POUR_W * pour_phase
    rcv_still = -still_w * rcv_carry * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ======================================================================== safety / collisions
    cup_contact = -CONTACT_W * torch.tanh(ctx.cup_cup_force / CONTACT_SCALE)
    hit_now = (ctx.cup_cup_force > HIT_FORCE).to(dt)
    cup_hit = -HIT_EVENT_W * settled * hit_now - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) \
        * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) \
        * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos
    cup_dist = torch.norm(rel, dim=-1)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_close = torch.sum((ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel) * u, dim=-1)
    closing_speed = -1.0 * torch.maximum(src_carry, rcv_carry) * (cup_dist < NEAR_DIST).to(dt) \
        * torch.tanh(torch.clamp(v_close - CLOSE_V_FREE, min=0.0) / CLOSE_V_SCALE)
    nested = -0.5 * ctx.cups_nested.to(dt)
    prelift_near = -PRELIFT_W * (1.0 - carry_raw) * _step((PRELIFT_SAFE - cup_dist) / (PRELIFT_SAFE - PRELIFT_ZERO))

    # ======================================================================== goal 2: stable control
    da = ctx.actions - ctx.prev_actions
    # exploration-killing charges are off for a policy that has not learned the task yet
    # (adr_progress stays 0 until the running success rate first passes 0.3) and are fully on
    # from ADR 6/30. The iter_18 base rates stay on always.
    ramp = _step(torch.full_like(carry_raw, float(ctx.adr_progress) / ADR_RAMP_FULL))
    palm_w = PALM_RATE_W + (PALM_RATE_CARRY_W - PALM_RATE_W) * carry_raw * ramp
    hand_w = HAND_RATE_W + (HAND_RATE_CARRY_W - HAND_RATE_W) * carry_raw * ramp
    palm_rate_src = -palm_w * torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_rate_rcv = -palm_w * torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_rate = -hand_w * (torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1))
    flips = torch.relu(-(ctx.actions * ctx.prev_actions))
    act_flip = -FLIP_CARRY_W * ramp * carry_raw * torch.sum(flips, dim=-1)
    # the 6 finger channels are charged for flipping whenever a cup is held: an open/close
    # oscillation is exactly what pushes the thumb out and lets the cup creep to the fingertips
    hand_flip = -HAND_FLIP_W * ramp * torch.maximum(held_src, held_rcv) \
        * (torch.sum(flips[:, 6:9], dim=-1) + torch.sum(flips[:, 15:18], dim=-1))
    a_abs = torch.abs(ctx.actions)
    palm_sat_src = -SAT_W * ramp * torch.mean(_step((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat_rcv = -SAT_W * ramp * torch.mean(_step((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    v_src = torch.norm(ctx.src_cup_lin_vel, dim=-1)
    src_speed = -carry_raw * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = -carry_raw * torch.tanh(torch.clamp(src_ang_speed - SRC_W_FREE, min=0.0) / SRC_W_SCALE)
    w_rcv = torch.norm(ctx.rcv_cup_ang_vel, dim=-1)
    rcv_motion = -0.5 * carry_raw * pour_phase * (
        torch.tanh(torch.clamp(v_rcv - RCV_V_FREE, min=0.0) / RCV_V_SCALE)
        + torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE))

    # ======================================================================== success
    # the terminal payoff itself is scaled by the posture: a successful pour held with a pinch
    # is worth half of one held with an envelope
    success = SUCCESS_W * ctx.success.to(dt) * clear * clean \
        * (0.5 + 0.5 * held_src * q_src) * (0.6 + 0.4 * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "both_grasped": both_grasped,
        "posture_src": posture_src,
        "posture_rcv": posture_rcv,
        "tip_only_src": tip_only_src,
        "tip_only_rcv": tip_only_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "converge": converge,
        "bring_together": bring_together,
        "aim": aim,
        "pour_dir": pour_dir,
        "pour_side": pour_side,
        "pour_dir_pen": pour_dir_pen,
        "home_rcv": home_rcv,
        "side_margin": side_margin,
        "tilt_pre": tilt_pre,
        "tilt_main": tilt_main,
        "pour_pose": pour_pose,
        "overtilt": overtilt,
        "upright_src": upright_src,
        "premature_latch": premature_latch,
        "rot_speed": rot_speed,
        "pour_delta": pour_delta,
        "flow_excess": flow_excess,
        "transferred": transferred,
        "spill_delta": spill_delta,
        "hold_src": hold_src,
        "hold_rcv": hold_rcv,
        "upright_rcv": upright_rcv,
        "rcv_still": rcv_still,
        "drop": drop,
        "cup_contact": cup_contact,
        "cup_hit": cup_hit,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "prelift_near": prelift_near,
        "palm_rate_src": palm_rate_src,
        "palm_rate_rcv": palm_rate_rcv,
        "hand_rate": hand_rate,
        "act_flip": act_flip,
        "hand_flip": hand_flip,
        "palm_sat_src": palm_sat_src,
        "palm_sat_rcv": palm_sat_rcv,
        "src_speed": src_speed,
        "src_spin": src_spin,
        "rcv_motion": rcv_motion,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
