# rh_aglt_d6 iter_03 (10.12) — iter_02 + 들린 뒤 안정화 · 쥐기 전 밀림 2항(나머지 그대로).
# 사용자(10.12): "cup 을 위로 들고 나서 모든 관절을 최대한 고정 — 떨림·진동·팅김 방지".
#   stabilize = −held·off · [min(0.05·Σ arm_qd², 2) + 0.5·mean(Δa²) + 0.2·mean(a_arm²)]
#     (팔 1 rad/s −0.05 · 튐 3 rad/s −0.45). iter_02 motion_penalty 의 tanh(|qd|/1) 는 1 rad/s 에서 포화돼 튐을 못 막았다.
# 실기 rg3(10.12 14:50): 접근 중 손이 컵을 쳐서 쓰러뜨림 — sim 성공률에 가려짐(insight pre-grasp-push-tilt).
#   pre_push = −free·(1−off) · [0.5·tanh(|v_cup| / 0.1) + 0.5·tanh((기울기 − 0.1) / 0.1)₊]
# 진단(합산 안 함): flick_rate(들린 뒤 max|arm_qd| > 1.5 rad/s, flick_detect 와 같은 문턱) · pre_tilt.
# rh_aglt_d6 iter_02 (10.12) — iter_01 + 목표 끌림 · 똑바로 유지 2항. 나머지 그대로.
# 진단(aglt_r_rg2, rigid_hold 성공 · 보상 iter_01): 성공이 "쥐고 들고 2 s 손 안 이동 ≤5 mm"라 위치를 안 봤고, 성공해도 에피소드가
#   안 끝나 2 s 마다 success_bonus(40) 를 다시 받았다 → 컵을 시작점에서 옆 28 cm · 바닥 위 51 cm 까지 휘둘렀다(trace env0, ep1300).
#   목표 10 cm 위인데 들림 최대높이 중앙 38–50 cm, D6 위치 성공 0.00 (ep200·300·1300 실기 지연 1.0).
#   lift 는 (height / goal_h).clamp(max=1) 이라 목표보다 높아도 손해가 없고, precision(exp(−err/1 cm))·dwell 은 멀면 0 이라 끌어오지 못한다.
# 사용자(10.12): "목표 10 cm 고정, 똑바로 들게(기울이기 없게 최대한)", 허용 오차는 넉넉히, 이송 무시 경향 고치기.
# env 쪽(rg3): rigid_hold 끔 → 키포인트 성공(기울기 포함) · tol 3 cm 고정 · goal_max 1(첫 성공 = 에피소드 끝) ·
#   목표 = 정착 컵 바로 위 10 cm(goal_first_xy_range 0, z 0.10).
# 추가:
#   goal_pull   = −1.0 · held · off · tanh(|컵 원점 − 목표| / 0.10)   (2 cm −0.20 · 10 cm −0.76 · 30 cm −0.995) — 들린 동안 멀수록 손해
#   upright_hold= −0.5 · held · off · tanh(기울기 / 10°)              (5° −0.23 · 10° −0.38 · 20° −0.48)
# rh_aglt_d6 iter_01 — iter_00 과 수식 같음, 컵 형상 값만 env 별 텐서로(10.11 다물체 A: per_env_geometry).
#   iter_00 은 rad·mouth 를 float() 로 받아 (N,) 텐서면 예외. 여기서는 (N,) 로 받고 (N,F) 식에는 열 모양(rad_c·mouth_c)을 쓴다.
#   스칼라 ctx 를 주면 iter_00 과 값이 같다(test: reward_gen/rh_aglt_d6/iter_01/check_equal.py).
# rh_aglt_d6 iter_00 — 양손 공통 D6 보상 (10.11, 손으로 씀) · 새 계열(사용자 10.11 "학습 다시 시작할 때 ITER 리셋")
# 출처: rh_aglt_r iter_27(f28a 학습 보상) 그대로 + 목표 근처 정밀 유지 항 1개. 왼손도 이 파일을 쓴다(사용자 "양손 같은 세팅").
#   iter_27 은 ctx.finger_*_pad 가 없으면 옛 접촉력으로 떨어지는 분기를 이미 가진다 — rh_aglt_l RewardContext 에도 같은 필드가 있다.
# 사용자 D6(10.11): 실기 성공 = 쥐고 들어 정해진 목표에서 2 s 연속 유지 · 허용 1~2 cm.
# 진단(10.11 D6 평가, f28a ep2200 · 256 env · 실기 지연 S2R-DELAY-1011 + FP++ attach, source_cup/d6_eval.sh):
#   2 cm 안 2 s 유지 0.41~0.67(결정론), 1 cm 0.04~0.09 · 지연 없어도 1 cm 0.10~0.18 · 유지 중 위치 오차 중앙 12~13 mm.
#   원인: 학습 성공 규약이 "kp ≤ tol 을 10 스텝 누적 → 다음 목표"(f28a tol 2.3 cm) — 1 cm 정밀도도 2 s 유지도 요구하지 않았고,
#   dwell 은 tol 안/밖 0·1 이라 tol 안에서 더 가까이 갈 기울기가 없었다.
# 환경 쪽(d6.sh): goal_max 1 · goal_success_steps 120(60 Hz × 2 s) · goal_force_consecutive True(벗어나면 누적 0) ·
#   tol 2 cm → 1 cm 커리큘럼. 성공 = 연속 유지 완료 → finish_bonus(남은 스텝 × (lift + grasp_hold)) 가 그대로 큰 보상.
# 바꾼 것(나머지 iter_27 그대로):
#   precision = 2.0 · room_post · carry_gate · post_f · still · exp(−|컵 원점 − 목표| / 1 cm)
#     (0 mm 1.0 · 10 mm 0.37 · 20 mm 0.14) — tol 안에서도 더 가까이 · 가만히 있을수록 이득. 위치 오차는 D6 판정과 같은 컵 원점 기준.
#   dwell(tol 안 3.0)·success_bonus(첫 성공 60)·finish_bonus 는 값 그대로 — 성공 정의가 env 에서 바뀐다.

import torch
import math


def _per_env(v, ref):
    """스칼라 또는 (N,) → (N,) 텐서(ref 의 dtype·device)."""
    t = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
    return t.expand(ref.shape[0]) if t.ndim == 0 else t.reshape(-1)

GRASP_H = -0.020        # 손바닥 목표 높이 [m] (원점 기준, 축 방향) — iter_18 은 +0.15 × mouth(+12.8 mm)
TOP_RAMP = 0.015        # 윗면 아래 이 거리부터 "윗면 위" 판정이 0 → 1 [m]


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _axial_radial(p: torch.Tensor, cup_pos: torch.Tensor, up: torch.Tensor):
    # cylinder coordinates of point(s) p around the cup axis (broadcasts over a finger dim)
    v = p - cup_pos
    h = (v * up).sum(-1)
    rv = v - h.unsqueeze(-1) * up
    return h, rv


def _horizontal(v: torch.Tensor) -> torch.Tensor:
    # world-horizontal part of a (N,3) vector (z component removed)
    return torch.stack([v[:, 0], v[:, 1], torch.zeros_like(v[:, 0])], dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    up = _unit(ctx.cup_up)
    # iter_25: palmar-side (pad) contact only — outer/side contacts of a link do not count
    if ctx.finger_prox_pad is not None:
        f_prox, f_tip = ctx.finger_prox_pad, ctx.finger_tip_pad
    else:
        f_prox, f_tip = ctx.finger_prox_force, ctx.finger_tip_force
    f_all = f_prox + f_tip
    held = ctx.grasped.float()
    free = 1.0 - held                                   # approach income fades once held
    rad = _per_env(ctx.cup_radius, ctx.cup_pos).clamp(min=1e-3)        # iter_01: (N,) — 물체마다 다름(per_env_geometry)
    mouth = _per_env(ctx.cup_mouth_z, ctx.cup_pos).clamp(min=1e-3)
    rad_c, mouth_c = rad.unsqueeze(1), mouth.unsqueeze(1)          # iter_01: (N,F) 식에 쓰는 열 모양

    # ---------------- joint room (unchanged) ----------------
    m = ctx.arm_limit_margin.clamp(0.0, 1.0)
    room_j = (m / 0.05).clamp(0.0, 1.0)                 # (N,7)
    room = 0.5 * room_j.min(-1).values + 0.5 * room_j.mean(-1)
    room_pre = 0.5 + 0.5 * room
    # iter_18g: held income uses its own, wider ramp (last 10 % of range) and drops clearly at a limit (was 0.8 + 0.2·room)
    room_j10 = (m / 0.10).clamp(0.0, 1.0)
    room10 = 0.5 * room_j10.min(-1).values + 0.5 * room_j10.mean(-1)
    room_post = 0.5 + 0.5 * room10

    # ---------------- palm vs. the measured grasp ring (unchanged) ----------------
    h_p, rv_p = _axial_radial(ctx.palm_pos, ctx.cup_pos, up)
    r_p = rv_p.norm(dim=-1)
    h_target = GRASP_H                                  # iter_21: grasp height origin −20 mm (was +0.15 x mouth)
    d_ring = torch.sqrt((r_p - 1.5 * rad) ** 2 + (h_p - h_target) ** 2 + 1e-12)

    far_k = 1.0 - torch.tanh(d_ring / 0.25)             # long-range pull from the far start
    near_k = 1.0 - torch.tanh(d_ring / 0.05)            # last few cm
    # iter_18 (from iter_13): grasp-height factor — 1 within +-0.1 x mouth of the target, then drops to 0.3
    dzh = (h_p - h_target) / mouth
    up_ex = (dzh - 0.1).clamp(min=0.0)
    lo_ex = (-dzh - 0.1).clamp(min=0.0)
    height_f = (1.0 - 0.7 * torch.tanh((up_ex / 0.2) ** 2)) * (1.0 - 0.15 * torch.tanh((lo_ex / 0.3) ** 2))
    at_pose = torch.exp(-d_ring / 0.015)                # essentially at the grasp pose

    # ---------------- hand orientation quality (unchanged) ----------------
    n = _unit(ctx.palm_axes[:, 0:3])
    f = _unit(ctx.palm_axes[:, 3:6])
    n_ax = (n * up).sum(-1)
    f_ax = (f * up).sum(-1)
    horiz = ((1.0 - n_ax ** 2) * (1.0 - f_ax ** 2)).clamp(0.0, 1.0)
    w = 0.67 * n + 0.73 * f                              # measured palm->axis direction
    w_h = _unit(w - (w * up).sum(-1, keepdim=True) * up)
    to_axis = _unit(-rv_p)
    align = 0.5 * (1.0 + (to_axis * w_h).sum(-1))
    pos_q = torch.exp(-d_ring / 0.02)
    pose_q = (horiz + align + pos_q) / 3.0

    s_tilt = torch.sqrt(n_ax ** 2 + f_ax ** 2 + 1e-12)
    plane_q = (1.0 - (s_tilt - 0.2) / 0.5).clamp(0.0, 1.0)
    plane_level = (1.0 - (s_tilt - 0.2) / 0.3).clamp(0.0, 1.0)
    level_pre = 0.5 + 0.5 * plane_level

    thumb_dz = ctx.tips_pos[:, 0, 2] - ctx.tips_pos[:, 4, 2]
    thumb_up = ((thumb_dz + 0.01) / 0.03).clamp(0.0, 1.0)

    # ---------------- fingertips in cup cylinder coordinates ----------------
    h_t, rv_t = _axial_radial(ctx.tips_pos, ctx.cup_pos.unsqueeze(1), up.unsqueeze(1))
    r_t = rv_t.norm(dim=-1)                              # (N,5)
    r_thumb = r_t[:, 0]
    h_thumb = h_t[:, 0]
    # iter_21: object wall radius at each fingertip height (bottle shoulder narrows) and the object's top
    if ctx.tip_wall_r is not None:
        wall_t = ctx.tip_wall_r
    else:
        wall_t = torch.zeros_like(r_t) + rad_c
    top = ctx.cup_top_z.unsqueeze(1) if ctx.cup_top_z is not None else torch.zeros_like(h_t[:, :1]) + mouth_c
    # "over the top" replaces the open-cup "inside the mouth" rule: 0 below top − 15 mm, 1 at/above the top
    over = ((h_t - (top - TOP_RAMP)) / TOP_RAMP).clamp(0.0, 1.0)      # (N,5)
    wall_thumb = wall_t[:, 0]

    thumb_inside = over[:, 0]
    thumb_out = 1.0 - thumb_inside

    # ---------------- NEW (iter_10): four fingers onto the OUTER wall ----------------
    # same "inside the mouth" rule as the thumb: radially inside the wall AND above 0.4 x mouth -> counts as far
    f_r = r_t[:, 1:]                                     # (N,4) index, middle, ring, little alike
    f_h = h_t[:, 1:]
    f_outside = 1.0 - over[:, 1:]                        # iter_21: was "inside the mouth"
    # radial gap to the outer wall at that height (iter_21: profile, was fixed rad) + distance outside [-0.7, 0.8] x mouth
    gap_r = (f_r - wall_t[:, 1:]).clamp(min=0.0)
    gap_h = (-0.7 * mouth_c - f_h).clamp(min=0.0) + (f_h - 0.8 * mouth_c).clamp(min=0.0)
    d_wall = torch.sqrt(gap_r ** 2 + gap_h ** 2 + 1e-12)
    wall_close = torch.exp(-d_wall / rad_c) * f_outside   # 1 = tip on the outer wall inside the window
    wall = (wall_close ** 2).mean(-1)                   # convex (squared) mean: each finger has its own gradient
    # iter_18c: geometric four-finger grasp quality (force-independent — under the admittance hand the contact force
    # is small, so a force score cannot tell a 2-finger pinch from a wrap). Per finger: tip proximity to the outer wall
    # inside the height window on a 0.25 r scale; soft-min over the four fingers (tau 0.1) -> every finger counts.
    near_wall = torch.exp(-d_wall / (0.25 * rad_c)) * f_outside
    q_geo = -0.1 * torch.log(torch.exp(-near_wall / 0.1).mean(-1))
    # g(q) as in grasp_fj (09.13 i2, 5-finger wrap): floor 0.5, full at q >= 0.25. measured on traces (held+lifted):
    #   env17 / admittance env17 fine-tunes q 0.20-0.35, aglt_r_i18 thumb+ring pinch 0.03
    g_geo = 0.5 + 0.5 * ((q_geo - 0.05) / 0.20).clamp(0.0, 1.0)

    # post-grasp finger factor: cube of the mean four-finger outer-wall contact score
    f_score = (torch.tanh(f_all[:, 1:].clamp(min=0.0) / 0.7) * f_outside).mean(-1)
    # floor 0.8: a firm 3-finger grasp keeps ~0.82 (still far above letting go); the 4th finger adds the most
    # iter_18b: linear in the number of fingers on the wall — each of the four adds 0.10 (floor 0.6).
    # iter_18 (0.8 + 0.2 f^3) paid a thumb+1-finger pinch 0.80 of a full grasp; aglt_r_i18 learned exactly that.
    finger_f = 0.6 + 0.4 * f_score

    # ---------------- contacts (unchanged) ----------------
    c = (f_all / 1.0).clamp(0.0, 1.0)
    thumb_c = c[:, 0] * thumb_out
    fing_c = c[:, 1:].mean(-1)
    wrap = thumb_c * fing_c

    pc = (f_prox / 0.5).clamp(0.0, 1.0)
    envelope = pc[:, 1:].mean(-1)
    # iter_21: index & middle curled onto the wall AT THEIR OWN HEIGHT — fingertip contact x radial closeness to the
    #   profile wall there (narrower on the bottle shoulder -> must curl further). Proximal contact was ~0 on every
    #   g5362b trace (fingertip grasp), so a proximal-contact score would give no signal; tip contact: index 0.23 on
    #   the 200 bottle (vs 0.99 cyl60), middle 0.98.
    near_im = torch.exp(-gap_r[:, :2] / (0.25 * rad_c)) * f_outside[:, :2]    # (index, middle)
    im_wrap_q = (pc[:, 1:3] * near_im).mean(-1)                          # iter_22: proximal (was tip) contact
    env_f = 0.7 + 0.3 * envelope                                          # iter_22: fingertip-only grasp keeps 70 %

    u = _unit(rv_t)
    u_thumb = u[:, 0]
    u_fing = _unit(u[:, 1:].mean(1))
    oppose = 0.5 * (1.0 - (u_thumb * u_fing).sum(-1))

    dr_box = (r_thumb - (wall_thumb + 0.45 * rad)).clamp(min=0.0) + ((wall_thumb + 0.15 * rad) - r_thumb).clamp(min=0.0)
    # iter_18: thumb box height measured from the palm grasp height (object centre), not from the mouth
    thumb_lo = h_target + 0.10 * mouth
    thumb_hi = h_target + 0.50 * mouth
    dh_box = (h_thumb - thumb_hi).clamp(min=0.0) + (thumb_lo - h_thumb).clamp(min=0.0)
    d_box = torch.sqrt(dr_box ** 2 + dh_box ** 2 + 1e-12)
    box_attract = torch.exp(-d_box / rad)

    outer_wall = ((r_thumb - wall_thumb) / (0.1 * rad)).clamp(0.0, 1.0)
    not_above = ((top[:, 0] - h_thumb) / (0.08 * mouth)).clamp(0.0, 1.0)
    not_far = (((wall_thumb + 0.8 * rad) - r_thumb) / (0.35 * rad)).clamp(0.0, 1.0)
    thumb_place = outer_wall * not_above * not_far
    thumb_f = 0.25 + 0.75 * thumb_place

    deep = ((2.3 * rad - r_p) / (0.8 * rad)).clamp(0.0, 1.0)     # iter_22: full at 1.5 r (was 1.8 r)
    deep = deep * ((r_p - 0.9 * rad) / (0.3 * rad)).clamp(0.0, 1.0)   # iter_25: overshoot past 1.2 r -> 0 at 0.9 r
    proj = (rv_t[:, 1:] * to_axis.unsqueeze(1)).sum(-1)
    past = (0.5 + proj / (0.5 * rad_c)).clamp(0.0, 1.0).mean(-1)
    fc = c[:, 1:]
    fingers_all = 0.5 * fc.mean(-1) + 0.5 * fc.min(-1).values
    parts = torch.stack([fingers_all, plane_level, past], dim=-1)
    blend = 0.5 * parts.mean(-1) + 0.5 * parts.min(-1).values
    shape_f = (0.3 + 0.7 * deep) * (0.6 + 0.4 * blend)

    inside_f = 1.0 - 0.5 * thumb_inside
    # scales grasp_hold, lift, transport, goal_hold; NEW: x finger_f
    post_f = shape_f * inside_f * thumb_f * finger_f * height_f * g_geo * env_f

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

    # ---------------- travel (unchanged) ----------------
    approach = room_pre * 0.35 * (0.6 * far_k + 0.4 * level_pre * near_k)
    pre_pose = room_pre * level_pre * 0.07 * near_k * pose_q
    # iter_26: all five proximal pads (thumb weighted most) face the cup axis near the cup, before contact
    if ctx.finger_pad_facing is not None:
        q_face = ctx.finger_pad_facing.clamp(0.0, 1.0)
        face_q = 0.4 * q_face[:, 0] + 0.3 * q_face[:, 1:].mean(-1) + 0.3 * q_face.min(-1).values
    else:
        face_q = torch.zeros_like(near_k)
    pad_face = room_pre * level_pre * 0.07 * near_k * face_q
    # iter_23: target hand shape by distance to the grasp ring — open far, close (to 0.6) inside ~12 mm
    x_c = ((0.045 - d_ring) / 0.025).clamp(0.0, 1.0)            # iter_24: 45 mm -> 20 mm (was 30 -> 12)
    g_near = x_c * x_c * (3.0 - 2.0 * x_c)
    # iter_24: open hand never loses near the cup (no reason to pull away), closing near adds up to +0.10
    hand_shape = room_pre * level_pre * 0.10 * (
        (1.0 - g_near) * open_ + g_near + g_near * (closure / 0.6).clamp(max=1.0) * (0.3 + 0.7 * envelope))

    level = 1.0 - n[:, 2].abs()
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))
    orient_fade = 1.0 - 0.5 * near_k
    approach_orient = room_pre * level_pre * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    thumb_steer = room_pre * level_pre * 0.04 * near_k * box_attract * (0.4 + 0.6 * oppose)

    hand_up = 0.25 + 0.75 * thumb_up

    # NEW: fingertips onto the outer wall, before AND while grasping (not gated by free, so the
    # ring/little fingers keep a pull toward the wall while the cup is held); small (0.10) and
    # gated by the palm being at the grasp ring so it cannot create a parking spot away from the cup
    finger_wall = 0.10 * room_pre * level_pre * near_k * pose_q * hand_up * upright * wall

    # ---------------- grasp (unchanged form; post_f now includes finger_f) ----------------
    grasp_contact = 0.10 * upright * hand_up * wrap * thumb_f
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * deep * past + 0.20 * envelope
    grasp_hold = room_post * held * upright * post_f * (1.05 + hand_up * plane_q * grasp_quality)
    # iter_21: index·middle wrap while held (user "curl index/middle more around the narrowing bottle")
    im_wrap = 0.15 * room_post * held * upright * height_f * im_wrap_q

    # ---------------- lift / transport / hold (unchanged form) ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    lift = room_post * 2.5 * held * upright * post_f * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright
    kp = ctx.goal_kp_dist

    tol = max(float(ctx.goal_tol), 1e-3)
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    # (from iter_17): progress (potential form) instead of distance-state income
    to_goal = ctx.goal_pos - ctx.cup_pos
    toward = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v_toward = (ctx.cup_lin_vel * toward).sum(-1).clamp(-0.3, 0.3)
    progress = room_post * carry_gate * post_f * (250.0 / 60.0) * v_toward
    dwell = room_post * 3.0 * carry_gate * post_f * (kp < tol).float() * still
    pos_err = (ctx.cup_pos - ctx.goal_pos).norm(dim=-1)
    precision = room_post * 2.0 * carry_gate * post_f * still * torch.exp(-pos_err / 0.01)   # d6 iter_00

    first = (ctx.goal_success & (ctx.goal_successes <= 1)).float()
    later = (ctx.goal_success & (ctx.goal_successes > 1)).float()
    success_bonus = 60.0 * first + 40.0 * later
    # iter_18f: finishing the episode's last goal truncates the episode, so the per-step standing income
    # (lift + grasp_hold) of the remaining time is lost. Pay that remaining income once at completion so that
    # finishing early is never worse than holding the cup lifted until timeout (10.07 aglt_l_g52a: lift 0.99 -> 1.41,
    # grasp_hold 0.68 -> 0.86 while successes 4.40 -> 0.33 under command delay).
    # 900 = policy steps per episode (episode_length_s 15 x 60 Hz, rh_aglt_env_cfg).
    done_all = (ctx.goal_success & (ctx.goal_successes >= float(ctx.goal_max))).float()
    remaining_steps = (1.0 - ctx.episode_progress).clamp(0.0, 1.0) * 900.0

    finish_bonus = done_all * remaining_steps * (lift + grasp_hold)

    # iter_02 (10.12): 들린 동안 목표에서 멀수록 · 기울수록 손해(위 머리말)
    goal_pull = -1.0 * held * off * torch.tanh(pos_err / 0.10)
    upright_hold = -0.5 * held * off * torch.tanh(ctx.cup_tilt / math.radians(10.0))

    # ---------------- short-lived, bounded penalties (unchanged) ----------------
    cup_w = ctx.cup_ang_vel.norm(dim=-1)
    tilt = ctx.cup_tilt
    tilt_gate = torch.tanh((tilt - 0.05).clamp(min=0.0) / 0.1)
    axis_dot = torch.cross(ctx.cup_ang_vel, up, dim=-1)
    axis_rate = axis_dot.norm(dim=-1)
    axis_move = torch.tanh((axis_rate - 0.2).clamp(min=0.0) / 0.3)
    tilt_rate = -axis_dot[:, 2] / torch.sin(tilt).clamp(min=1e-3)
    rising = torch.tanh(tilt_rate.clamp(min=0.0) / 0.3)
    tip_penalty = -0.6 * free * (1.0 - off) * tilt_gate * axis_move * (0.5 + 0.5 * rising)
    fall_speed = (-ctx.cup_lin_vel[:, 2]).clamp(min=0.0)
    drop_penalty = -1.0 * ctx.lifted.float() * free * torch.tanh(fall_speed / 0.5)
    motion_penalty = -held * (0.10 * torch.tanh(cup_v / 0.5)
                              + 0.05 * torch.tanh(cup_w / 2.0)
                              + 0.10 * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0))
    action_rate = -0.01 * ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)

    # iter_03 (10.12 사용자): 컵을 든 뒤 팔·손 안정화(떨림·진동·팅김 방지) — 들린 뒤엔 관절을 최대한 고정.
    #   motion_penalty 의 팔 속도 항은 1 rad/s 에서 포화돼 튐(> 1.5 rad/s)을 못 막는다 → 제곱(상한 2)으로.
    #   팔 액션 = 관절 증분이라 0 이 곧 고정 — 크기도 벌한다. 액션 변화(팔·손)는 iter_02 의 50 배.
    post = held * off
    arm_qd2 = (ctx.arm_qd ** 2).sum(-1)
    da = (ctx.actions - ctx.prev_actions) ** 2
    stabilize = -post * ((0.05 * arm_qd2).clamp(max=2.0)
                         + 0.5 * da.mean(-1)
                         + 0.2 * (ctx.actions[:, :7] ** 2).mean(-1))
    # iter_03 (10.12 실기 rg3: 접근 중 손이 컵을 쳐서 넘어뜨림 · insight pre-grasp-push): 쥐기 전(안 든 채) 컵이
    #   움직이거나 기울면 벌한다 — 속도(밀기 순간)와 0.1 rad 넘는 기울기(받아 쥐기 전략).
    pre = free * (1.0 - off)
    pre_push = -pre * (0.5 * torch.tanh(cup_v / 0.10)
                       + 0.5 * torch.tanh((ctx.cup_tilt - 0.10).clamp(min=0.0) / 0.10))
    flick = (ctx.lifted.float() * (ctx.arm_qd.abs().max(dim=-1).values > 1.5).float())   # 진단 — flick_detect 와 같은 1.5 rad/s
    foreign_contact = -0.10 * torch.tanh(ctx.hand_foreign_force / 5.0)

    comps = {
        "approach": approach,
        "hand_shape": hand_shape,
        "pre_pose": pre_pose,
        "pad_face": pad_face,
        "approach_orient": approach_orient,
        "thumb_steer": thumb_steer,
        "finger_wall": finger_wall,
        "grasp_contact": grasp_contact,
        "grasp_hold": grasp_hold,
        "im_wrap": im_wrap,
        "lift": lift,
        "progress": progress,
        "dwell": dwell,
        "precision": precision,
        "success_bonus": success_bonus,
        "finish_bonus": finish_bonus,
        "goal_pull": goal_pull,
        "upright_hold": upright_hold,
        "tip_penalty": tip_penalty,
        "drop_penalty": drop_penalty,
        "motion_penalty": motion_penalty,
        "action_rate": action_rate,
        "foreign_contact": foreign_contact,
        "stabilize": stabilize,
        "pre_push": pre_push,
        "flick_rate": flick,         # diagnostic only (not summed)
        "pre_tilt": pre * ctx.cup_tilt,   # diagnostic only (not summed) — 쥐기 전 기울기
        "joint_room": room,          # diagnostic only (not summed)
        "finger_factor": finger_f,   # diagnostic only (not summed)
        "height_factor": height_f,   # diagnostic only (not summed)
        "grasp_q_geo": q_geo,        # diagnostic only (not summed)
        "envelope_prox": envelope,   # diagnostic only (not summed) — 네 손가락 첫마디 접촉 평균
        "palm_deep": deep,           # diagnostic only (not summed)
    }
    diag = ("flick_rate", "pre_tilt", "joint_room", "finger_factor", "height_factor", "grasp_q_geo", "envelope_prox", "palm_deep")
    reward = torch.zeros_like(approach)
    for k, v in comps.items():
        if k in diag:
            continue
        reward = reward + v
    return reward, comps
