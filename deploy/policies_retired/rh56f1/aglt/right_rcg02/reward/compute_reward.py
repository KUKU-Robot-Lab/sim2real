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
    # neo iter_03 (10.19, 2판 new_02_x T2R path check): the first touch of a closing hand rocks the object - successful
    # 1판 grasps (neo03m ep2200, 14/16 lifted) tilt p90 1.6 deg / spin p90 0.36 rad/s at one finger, 4 deg / 0.9 rad/s at
    # two, and the 90-step closing replay from the approach pose up to 0.6-0.9 rad/s. The 1.1 deg / 0.12 rad/s dead bands
    # cut the approach incomes (undisturbed factor) and charged the tipping cost on that wobble: closing -> first touch
    # -1.69 (-57 %, 0 % of envs). Dead bands above the acquisition wobble; real tipping (ledger T1: 20 deg, 1.5 rad/s;
    # T2 85 deg) is still charged in full.
    TILT_DEADBAND = 0.05    # rad (2.9 deg, iter_02 0.02)
    TILT_SCALE = 0.08       # iter_02 0.05
    SPIN_DEADBAND = 0.80    # rad/s (iter_02 0.12)
    SPIN_SCALE = 0.40       # iter_02 0.15
    TIP_SCALE = 0.30
    CONTACT_SCALE = 1.0
    FOREIGN_SCALE = 2.0
    ROOM_MARGIN = 0.05
    ROOM_FLOOR = 0.85

    # =====================================================================================
    # GRASP + LIFT + HOLD - ladder (iter_06): deep 1.5 < held 4 + qualities 6 < lift 10 < goal 3 + still 8
    # =====================================================================================
    DEEP_RADII = 1.5        # user 10.14: palm about 1.5 radii from the axis before closing ...
    # iter_12 (user 10.18 "origin -20 mm 에서 최대한 잡게"): the deep ring at the grasp height GRASP_H (was -0.01).
    DEEP_H = -0.02          # ... and below the object origin, at the grasp height
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
    HOLD_SCALE = 0.12       # iter_09 m: hold-point closeness exp(-|cup - goal| / 12 cm), position only (was exp(-kp / 4 cm) + near flag)
    QD_SCALE = 0.5          # rad/s (norm over the 7 arm joints)
    V_SCALE = 0.05          # m/s palm speed
    F_SAFE = 5.0            # N per finger: no squeeze cost below
    F_SQ_SCALE = 5.0        # N
    DRAG_DEAD = 0.005       # m horizontal displacement from the spawn point
    DRAG_SCALE = 0.02       # m
    DRAG_V_DEAD = 0.05      # m/s horizontal object speed (neo iter_03: 0.02 - the first touch nudges the object at 2-5 cm/s)
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
    XY_DEAD = 0.03          # m: horizontal distance from the goal that costs nothing (iter_09 1 -> 3 cm)
    # same replay: the held cup tilted, median 46 deg (right) / 27 deg (left) while held. The goal pose is upright.
    # neo iter_02 (user 10.19 "컵 hand tcp 기준에서 컵이 기울어지는건 arm을 조절해서, cup 자체를 똑바로 세울 수 있다고 봄",
    # "fpp로 cup들의 z는 추출할수 있음"; the actor already sees the FP++ cup axis, cup_up): the in-hand tilt is the grasp's,
    # the WORLD tilt is the arm's to fix. neo03 ep5500 grid (10 seeds x 15 cells, held steps p50): world tilt src240 22 deg,
    # shaker 28 deg, while the real grasp photos (10.19) hold the cup 1-5 deg off the finger row. Upright is now a real
    # income (was 0.35 of 46.8 at the goal, tilted 30 deg kept 98 %). Dead band 5 deg > FP++ tilt bias (0-3 deg).
    # The arm latch (env arm_hold_rise_m 0.07, k_arm 0.05 rad) freezes the arm pose 7 cm up, so the income starts with the
    # lift (lift_root), where the arm is still incremental.
    TILT_DEAD = 0.087       # rad (5 deg): held-cup world tilt that costs nothing (iter_01 20 deg)
    TILT_HOLD_SCALE = 0.26  # rad (15 deg): upright_q = exp(-(tilt - 5 deg) / 15 deg) (0.72 at 10, 0.37 at 20, 0.19 at 30 deg)
    # iter_10 (10.18 lift5 pair replay, rg3wo ep10000, nf0 + substep 4): closing the free fingers on a lifted cup pushed it
    # 2.6-5.8 cm up in the hand and tilted it 4 -> 27-37 deg with the palm in place; the lift income fell 1.2-2.2 (tilt x
    # over-height), more than the fingers income gained. User 10.17: "5점 이상파지 > 리프트 > 안정성 >>> 컵 똑바로 들기",
    # a five-point grasp has to tilt / lift / shift the cup a little. Upright is no longer a factor on the lift income but
    # an income of its own (W_UPRIGHT / W_LIFT = 3 %; 1.0 broke the ledger M6 / M4 still ratios from either side).
    # neo iter_02: upright back on the lift group as a factor (lift, goal, still), floored. As an own income (W 8) it added
    # the same constant to every lifted state and diluted the still / overshoot / carried contrasts (ledger M6 0.87,
    # G7 0.86, G7z 0.95). With five_q on the same group, closing more fingers still multiplies the whole group.
    UP_FLOOR = 0.6          # lift group x (0.6 + 0.4 upright_q): 30 deg tilted keeps 0.68 of it (ledger grasp5d: < 85 % of upright)
    # user video review (10.15, rg3wi ep1100): (3) the cup is held high, at the bottle neck - the palm should sit below the
    # origin (replay: palm +2.1 cm above the origin while held, index tip +4.6 cm). Earlier trial-and-error fixed this at
    # origin -20 mm (rh_aglt_r iter_21~25, d6 iter_06 GRASP_H) with this height factor on every post-grasp income.
    GRASP_H = -0.020        # m: palm target height along the axis (origin -20 mm)
    # iter_12: rg3wo grasped high (s32 replay, palm while held: src240 right +39 mm, first contact +17..+52 mm at r 2.5 R).
    # The approach start rg3wc ep800 parks at the pre-grasp ring (palm +29..+30 mm, 4-finger tips +28..+32 mm), fingers
    # reached the cup from there, and (1) height_f was flat (0.30-0.33) above origin +12 mm - no gradient down, (2) reach /
    # keep / close / deep paid the held grasp at any height. Above the band height_f now decays exponentially (no flat top)
    # and it scales reach, keep, close and the held part of deep as well.
    HF_FLOOR = 0.3          # height factor far above the target (as before)
    HF_UP_SCALE = 0.25      # x mouth height: decay above the +-0.1 x mouth band (src 21 mm, shaker 16 mm)
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
    MORE_3 = 0.4            # quality factor at 3 kinds of fingers, 1.0 at 5 (iter_07: 0.7 -> 0.6, iter_09: 0.4, user 10.17 "5점 파지")
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
    KEEP_ANY = 0.6          # thumb: share paid for any link, its first link pays the rest (four fingers: neo iter_01 LINK_*)
    F_KEEP_HI = 8.0         # N per finger: keeping income x exp(-(f - 8 N) / 10 N) above (thumb support p50 18 N in the replay)
    F_KEEP_SCALE = 10.0
    DIG_DEAD = 0.001        # m: tip point inside the outer wall that costs nothing (touching tips p99 +1.2 mm with substep 4)
    NEAR_MIX = 0.9          # iter_08 open-finger nearness: hand-wide (reach_q) share, rest the finger's own tip
    GUIDE_ANY = 0.8         # iter_08 closing guide share reached at the first contact of a finger (both links: 1)
    DIG_SCALE = 0.003       # m: keep x exp(-((pen - 1 mm) / 3 mm)^2) (0.37 at 4 mm, 0.0 at 10 mm = env termination)
    # iter_11 (user 10.18 "5점 이상파지 > 리프트 > 안정성", success = 7 cm lift with all five fingers on for 60 steps): the
    # closing incomes average the fingers (reach_q: thumb 0.3 + mean of four; guide_q likewise), so one finger left open
    # 2 cm off the cup costs only ~1/4 of a small share, and rg3wo held 3-4 finger grasps with ring / pinky commanded
    # open. Isaac Lab dexsuite and DextrAH-G pay the FARTHEST finger instead (1 - tanh(d_max / 0.4), exp(-10 d_max)):
    # the whole gradient goes to the lagging finger. As an income it adds a constant to every five-finger state and
    # dilutes the held-state contrasts (ledger grasp5 M6 moving / still 0.849 -> 0.86, G7 overshoot 0.85 with +2), and
    # gated by the deep ring it would tax the arrival of the open hand (the iter_05 offset). So it is the shortfall of
    # the farthest finger, only once three fingers hold the object (held3): 0 with all five on, the 3 -> 4 -> 5 steps pay.
    W_LAG = 2.0             # held: - (1 - min over fingers of the tip nearness), contact = nearness 1
    # neo iter_01 (aglt_{r,l}_neo ep1700 = neo iter_00, 10.18, user chose "3지 바닥값을 없애고 5지 접촉과 파지 높이에 비례"): the lift was
    # learned (lifted 0.96) while the five-finger grasp collapsed exactly when the lift / goal / still incomes rose
    # (ep1000-1400): held touch5 0.31 / 0.56 -> 0.025 / 0.053 of held steps, three-finger tip grasps 70 / 57 %, right palm
    # at the mouth (h / mouth -0.29 -> +1.04, height_f 0.3). Those incomes carried grip = held3 x (0.3 + 0.7 x quality):
    # the 0.3 floor paid any three-finger grasp at any height, and five fingers / the grasp height only scaled the 0.7
    # part through a product of six factors. The lift group now pays in proportion to the fingers on the object
    # (3 -> 1/3, 4 -> 2/3, 5 -> 1) and to the grasp height (height_f), without that floor; the rest of the grasp quality
    # (thumb, close command, depth, digging) keeps its floor inside. The held income (fingers) is unchanged.
    # height_f keeps its own HF_FLOOR (0.3 above the target band). Replay of ep1700 (s42, lifted >= 5 cm steps): quality
    # was 0.02-0.06 for 3, 4 AND 5 fingers (more x cmd 0.2-0.4 x depth 0.4-0.5 x height 0.3-0.7), so the floor carried
    # the whole income and the lifted step paid 14.4 / 14.6 / 14.3 (right, high grasp) for 3 / 4 / 5 fingers - no
    # gradient towards five. iter_01 on the same steps: 9.1 / 10.1 / 11.0 (left low: 3f 9.9, 4f 11.6, 5f 14.5); lifted
    # still clearly above grasped-on-the-table (6.5-7.6, unchanged).
    FIVE_LO = 2.0           # fingers on the object where the lift group starts (held3 is 0 there as well) ...
    FIVE_HI = 5.0           # ... and where it is full
    # neo iter_01 (user 10.18 "손바닥쪽에 깊숙히 들어가서 하는 파지. 팁도 닿게 권장하는거지 팁만 닿으라고 한게 아님"):
    # ep1700 replays (neo_lift5 s42, 32 env x 900 per hand) held the cup with the tips only - palm point 41 / 55 mm off
    # the wall (right / left p50; the deep ring 1.5 R is 15-18 mm), the first links of the four fingers 0.00 touching,
    # 2.5 tips; grid 75 env: first-link share 3-6 %. The depth sat inside two floors (depth_f2 0.4 inside GRASP_FLOOR
    # 0.3), so a tip grasp at arm's length kept nearly the whole held and lift income. The grasp depth is now its own
    # factor outside those floors on the held AND the lift incomes: palm radially in (palm_in, full at the deep ring)
    # and the links of the touching four fingers (first link 0.75, both links 1, tip only 0.25). The keeping income
    # pays the links the same way (was: any one link 0.6). The palm no longer pays the approach contact charge once held.
    DEPTH_PALM = 0.5        # grasp depth = 0.5 palm_in + 0.5 link quality of the touching four fingers
    # neo iter_02 (10.19): the real RH56F1 holds with the first links - s2r_1st grip measurement (both hands, src240 /
    # shaker) tip tactile ~0, the user's photos: tips curl past the cup without touching; user "구조적 핸드가 잡는게 쉽지
    # 않음 팁 5점 파지는", "tip을 닿게 하려면 컵자체가 핸드입장에선 기울여져서 잡게 될수 밖에 없음". neo03 ep5500 held the
    # cup with 3.6 tip-only fingers. The tip is now neutral: the first link alone is the full link quality, a tip adds
    # nothing to it and a tip-only finger has none (it still counts as a finger on the object).
    LINK_PROX = 1.0         # first (proximal) link alone (iter_01 0.75; with the tip as well 1.0, unchanged)
    LINK_TIP = 0.0          # tip alone (iter_01 0.25)
    DEEP_GRIP_FLOOR = 0.2   # held / lift incomes x (0.2 + 0.8 x depth)
    # neo iter_02 (10.19 user "링크1 접촉과 누르는힘 수정 필요 - 결국 컵을 누르긴 해야함. 그렇지 않으면 미끄러지게 됨"):
    # the first link has to PRESS the cup, a touch is not a grasp. With F_ON (0.5 N) the first-link quality was full at
    # 1 N. The real hand held with 3-8 N per first link (s2r_1st grip measurement), sim neo03 ep5500 with the tips
    # 2-5 N and the first links ~0. First-link quality (link quality, grasp depth, closing guide after contact) =
    # tanh(F / 1.5 N): 0.5 N 0.32, 1 N 0.58, 2 N 0.87, 3 N 0.96. The squeeze cost above 5 N per finger is unchanged.
    # Touching (fingers on the object, held, success) still counts any contact (F_ON).
    F_PRESS = 1.5           # N
    # neo iter_02 (10.19 user "중요한건 컵을 잡고 들어올려서 안정적으로 잡을 때의 이 파지 자세를 정책이 아는게 제일
    # 중요" - the later ready-carry moves the object from this grasp pose): the lift group (lift, goal, still) pays the
    # in-hand pose kept since lift-off. ctx.cup_hand_drift = how far the cup origin / mouth centre moved in the palm
    # frame since the cup was first grasped AND lifted (env HandDrift: jitter cancels, the cup turning in the hand moves
    # the mouth point; 0 before lift-off). neo03 ep5500 grid: in-hand slip median 8 mm (rg3wo 3-4 mm).
    FIX_DEAD = 0.003        # m: drift that costs nothing (jitter)
    FIX_SCALE = 0.010       # m: fix_q = exp(-(drift - 3 mm) / 10 mm) (0.61 at 8 mm, 0.18 at 20 mm)
    FIX_FLOOR = 0.3         # lift group x (0.3 + 0.7 fix_q): the main condition of the lift incomes

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
    # neo iter_03: released by any touch (was touch x deep_q). The approach pose of the RH56F1 hand has the tips 0.4-2.4 cm
    # off the wall at 2.5 R (approach_r parked), so the first touch comes long before the deep ring (deep_q 0.1-0.2):
    # the approach measure taxed the first step of the grasp (T2R iter_04 lesson).
    released = torch.maximum(held, touch)
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
    # neo iter_01: the palm keeps it only until the fingers hold the object (a palm-side grasp is the goal)
    # neo iter_03: the finger charge is released by the touch itself (the grasp has begun), the palm keeps it until held
    contact_force = ff.sum(dim=-1) * (1.0 - torch.maximum(torch.maximum(deep_q, held), touch)) + ctx.palm_force * (1.0 - held)
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
    # iter_12: deep is computed after height_f (the held part is judged at the grasp height), see below keep

    # depth evidence for the grasp: palm depth + proximal links of the touching fingers (wrap, not finger tips only)
    wrap = (torch.tanh(ctx.finger_prox_force[:, 1:].clamp_min(0.0) / F_PRESS).sum(dim=-1)
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
    cp = torch.tanh(ctx.finger_prox_force.clamp_min(0.0) / F_PRESS)     # neo iter_02: pressing, not touching
    ct = torch.tanh(ctx.finger_tip_force.clamp_min(0.0) / F_ON)
    both = cp * ct
    # neo iter_01: links per finger - first link 0.75, both 1, tip only 0.25 (keeping income and grasp depth)
    lnk = cp * (LINK_PROX + (1.0 - LINK_PROX) * ct) + (1.0 - cp) * ct * LINK_TIP
    link_four = lnk[:, 1:]
    link_thumb = KEEP_ANY * c[:, 0] + (1.0 - KEEP_ANY) * torch.maximum(cp[:, 0], both[:, 0])
    links = torch.cat([link_thumb.unsqueeze(1), link_four], dim=1)
    f_band = torch.exp(-torch.relu(ff - F_KEEP_HI) / F_KEEP_SCALE)
    keep_q = (links * f_band * dig).mean(dim=-1)
    # touching fingers that are not digging in (share), carried into the grasp quality
    dig_q = (c * dig).sum(dim=-1) / c.sum(dim=-1).clamp_min(1e-3)
    # depth gate until the fingers hold the object; once held the grasp height is judged by height_f, not by the ring
    # neo iter_03: depth gate of the closing incomes (reach, close, keep) from the pre-grasp pose on the corridor (0.4) up to the
    # deep ring (1). With deep_q alone, closing at the approach pose paid nothing (+0.0 %, 25 % of envs) - and that is where
    # the tips already are; neo03m touches at 2.4 R and holds at 1.8 R. Off the corridor the floor fades (exp(-dist / 5 cm)).
    DEPTH_GATE_FLOOR = 0.4
    depth_g = torch.maximum(deep_q, DEPTH_GATE_FLOOR * torch.exp(-dist / POSE_SCALE))
    reach_depth = depth_g + (1.0 - depth_g) * held

    # iter_03 grasp height (previous verified form, rh_aglt_r iter_18/21): 1 within +-0.1 x mouth of the target, then
    # drops to 0.3 above it (towards the neck) and to 0.85 below it
    mouth = _per_env(ctx.cup_mouth_z, palm)
    # iter_10: h runs along the object axis, so tilting the object in the hand moved h by ~ r sin(tilt) and rewarded the
    # tilt itself (ledger: 30 deg tilt raised height_f 0.87 -> 1.0 on every post-grasp income; rg3wo held cups tilted
    # median 46 / 27 deg). Take the lower of the axis height and the world-vertical height (equal while upright).
    def _height_f(hh: torch.Tensor) -> torch.Tensor:
        dzh = (hh - GRASP_H) / mouth.clamp_min(1e-3)
        up_ex = (dzh - 0.1).clamp(min=0.0)
        lo_ex = (-dzh - 0.1).clamp(min=0.0)
        up_f = HF_FLOOR + (1.0 - HF_FLOOR) * torch.exp(-up_ex / HF_UP_SCALE)    # iter_12: was 1 - 0.7 tanh((x/0.2)^2)
        return up_f * (1.0 - 0.15 * torch.tanh((lo_ex / 0.3) ** 2))

    height_f = torch.minimum(_height_f(h), _height_f(palm[:, 2] - ctx.cup_pos[:, 2]))

    disp = (ctx.cup_pos[:, 0:2] - ctx.cup_spawn_pos[:, 0:2]).norm(dim=-1)
    v_xy = ctx.cup_lin_vel[:, 0:2].norm(dim=-1)
    drag_q = 0.5 * (torch.tanh(torch.relu(disp - DRAG_DEAD) / DRAG_SCALE)
                    + torch.tanh(torch.relu(v_xy - DRAG_V_DEAD) / DRAG_V_SCALE))
    # iter_06: grasp income = held3 x W_GRASP x (floor + (1 - floor) x quality product), ONE floor. The iter_05 product of five
    # separately floored factors kept 0.71 of 9 with all five fingers on (right ep600: thumb flexion open, palm +3 mm).
    # Each defect still scales the part above the floor (ledger ratio rules), the held grasp never falls to the hover level.
    thumb_f = THUMB_FLOOR + (1.0 - THUMB_FLOOR) * c[:, 0]
    more_f = MORE_3 + (1.0 - MORE_3) * _ramp(c.sum(dim=-1), HELD_HI, 5.0)
    # neo iter_01: grasp depth outside the quality floor (palm in + links of the touching four fingers)
    link_q = (lnk[:, 1:].sum(dim=-1) / k_other.clamp_min(1.0)).clamp(0.0, 1.0)
    grasp_depth = DEPTH_PALM * palm_in + (1.0 - DEPTH_PALM) * link_q
    deep_gf = DEEP_GRIP_FLOOR + (1.0 - DEEP_GRIP_FLOOR) * grasp_depth
    quality = thumb_f * more_f * cmd_f * height_f * dig_q    # iter_07: x touching fingers not digging in
    grip = held3 * deep_gf * (GRASP_FLOOR + (1.0 - GRASP_FLOOR) * quality)   # the held income (neo iter_01: not lift / goal / still)
    fingers = W_GRASP * grip                                          # iter_07: touch_inc -> keep (per finger)
    # neo iter_01: lift-group grip = fingers on the object (3 -> 1/3, 5 -> 1, no floor) x grasp height (height_f, own floor)
    # x grasp depth
    five_q = _ramp(c.sum(dim=-1), FIVE_LO, FIVE_HI)
    q_rest = thumb_f * cmd_f * dig_q
    lift_grip = held3 * five_q * height_f * deep_gf * (GRASP_FLOOR + (1.0 - GRASP_FLOOR) * q_rest)

    goal_rise = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp_min(MIN_GOAL_RISE)
    # iter_10: height off the hold point counts only where the cup AND the palm (target: cup origin + GRASP_H) are off on the
    # same side - the arm drifting up / down moves both, the cup sliding in a closing hand moves only the cup.
    dz_cup = rise - goal_rise
    dz_palm = palm[:, 2] - (ctx.cup_spawn_pos[:, 2] + goal_rise + GRASP_H)
    dz_hold = torch.where(dz_cup * dz_palm > 0.0,
                          torch.sign(dz_cup) * torch.minimum(dz_cup.abs(), dz_palm.abs()),
                          torch.zeros_like(dz_cup))
    over = (torch.relu(dz_hold)
            + torch.relu((ctx.cup_pos[:, 0:2] - ctx.goal_pos[:, 0:2]).norm(dim=-1) - XY_DEAD))
    upright_q = torch.exp(-torch.relu(tilt_axis - TILT_DEAD) / TILT_HOLD_SCALE)
    upright_f = UP_FLOOR + (1.0 - UP_FLOOR) * upright_q          # neo iter_02: world tilt, the arm's to fix
    # neo iter_02: in-hand pose kept since lift-off (no slip, no turning in the hand) - the main condition of the lift
    # group, next to the world tilt (the arm uprights the cup without changing the grasp: drift stays 0)
    drift = ctx.cup_hand_drift if ctx.cup_hand_drift is not None else torch.zeros_like(rise)
    fix_q = torch.exp(-torch.relu(drift - FIX_DEAD) / FIX_SCALE)
    fix_f = FIX_FLOOR + (1.0 - FIX_FLOOR) * fix_q
    lift_f = upright_f * fix_f
    # iter_06: concave in the first centimetres (sqrt), dead zone 5 mm (pushing / tilting on the table)
    lift_frac = (torch.relu(rise - LIFT_DEAD) / (goal_rise - LIFT_DEAD)).clamp(0.0, 1.0)
    lift_root = torch.sqrt(lift_frac + 1e-12)
    lift_q = lift_root * torch.exp(-over / OVER_SCALE)            # iter_10: upright left the lift income
    lift = W_LIFT * lift_q * lift_grip * lift_f

    # iter_09 (user 10.17): goal transport is not the task - the goal is only the hold point that keeps the incremental
    # arm from drifting. kp included the cup attitude (upright, the last priority) and exp(-kp / 4 cm) went flat ~15 cm
    # away, where a jolted arm then drifted further (no gradient back). Position only, wide basin.
    hold_d = torch.sqrt(((ctx.cup_pos[:, 0:2] - ctx.goal_pos[:, 0:2]) ** 2).sum(dim=-1) + dz_hold ** 2 + 1e-12)
    goal_q = torch.exp(-hold_d / HOLD_SCALE) * (1.0 - on_table)
    goal = W_GOAL * goal_q * lift_grip * lift_f
    still_q = (torch.exp(-ctx.arm_qd.norm(dim=-1) / QD_SCALE)
               * torch.exp(-ctx.palm_lin_vel.norm(dim=-1) / V_SCALE))
    hold_still = W_STILL * goal_q * still_q * lift_grip * lift_f

    squeeze_q = torch.tanh(torch.relu(ff - F_SAFE) / F_SQ_SCALE).mean(dim=-1)
    squeeze = -SQUEEZE_SHARE * squeeze_q * (fingers + lift + goal + hold_still)

    drag = -W_DRAG * drag_q * on_table * torch.tanh(obj_force / F_ON)
    # iter_04: no reach income while the object is being dragged across the table (ledger D2g)
    reach = W_REACH * reach_depth * reach_q * undisturbed_g * gate_orient * (1.0 - drag_q * on_table) * height_f
    # iter_05: close command x tips at the surface (no income for a fist closed in the air) x drag gate as reach
    # (ledger D2g). Depth: the deep ring, or once held the grasp depth evidence (palm depth + proximal wrap, as the
    # finger income) - a low grasp at the target height keeps it (H3l), a shallow tips-only grasp does not (S3).
    close_depth = depth_g + (1.0 - depth_g) * held * depth_f      # neo iter_03: depth_g (was deep_q)
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
    # four fingers after contact: neo iter_02 the first link is done (1); tip only stays at GUIDE_ANY (the first link needs
    # a deeper hand, paid by the grasp depth). iter_01 paid the first-link-only finger for pulling its tip onto the
    # surface - the tip contact that tilts the cup in the hand.
    post_four = GUIDE_ANY + (1.0 - GUIDE_ANY) * cp[:, 1:]
    post = torch.cat([torch.ones_like(c[:, :1]), post_four], dim=1)
    guide = c * torch.maximum(post, pre) + (1.0 - c) * pre                  # a contact never lowers the guide
    guide_q = CLOSE_THUMB_SHARE * guide[:, 0] + (1.0 - CLOSE_THUMB_SHARE) * guide[:, 1:].mean(dim=-1)
    close = W_CLOSE * close_depth * guide_q * undisturbed_g * gate_orient * (1.0 - drag_q * on_table) * height_f
    keep = W_KEEP * reach_depth * keep_q * (1.0 - drag_q * on_table) * height_f
    # iter_12: the deep ring itself is at GRASP_H (height_f 1 there); the held part only at the grasp height
    deep = W_DEEP * torch.maximum(deep_q, held3 * palm_in * height_f) * undisturbed_g * gate_orient
    laggard = -W_LAG * held3 * (1.0 - tip_near.min(dim=-1).values)

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
        "laggard": laggard,
        "fingers": fingers,
        "lift": lift,
        "upright_factor": upright_f.detach(),
        "fix_factor": fix_f.detach(),
        "goal": goal,
        "still": hold_still,
        "squeeze": squeeze,
        "height_factor": height_f.detach(),
        "drag": drag,
    }
    step = (approach_sum + approach_hold
            + contact + table_contact + tipping
            + deep + close + reach + keep + laggard + fingers + lift + goal + hold_still + squeeze + drag)

    # success lump sum: success truncates the episode 240 hold steps later, so hovering at the goal without
    # completing would out-earn success. Pay, once, at the success step a bounded multiple of the income being
    # earned there (held at the goal), which covers the discounted income lost after the hold.
    success = (SUCCESS_MULT * step.clamp_min(0.0) * ctx.goal_success.to(palm.dtype)
               * _ramp(lift_q, SUCC_LIFT_LO, SUCC_LIFT_HI))      # iter_02: lifted, not standing at the goal tolerance
    reward = step + success
    return reward, {**components, "success": success}
