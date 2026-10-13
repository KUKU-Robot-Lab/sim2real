"""cup_attach — 파지 전 FP++, 쥔 뒤 손바닥 FK. 10.04 사용자 원칙 · sim(rh_aglt hdgp 2721a946) 규칙:
FP++ 프레임이 찍힌 시각의 손바닥 FK 로 상대 자세를 만들고, 파지 시작 뒤 100 ms 넘어 찍힌 프레임부터 붙인다."""
from __future__ import annotations

import numpy as np
import pytest

from policy_control.cup_attach import AttachCfg, CupAttach, grasp_flag

I3 = np.eye(3)
DT = 1 / 60
LAT = 0.29                      # FP++ 지연(실측 중앙 288 ms)
Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def _rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def test_grasp_needs_the_thumb_and_one_other_finger_over_one_newton():
    assert grasp_flag([1.2, 0.0, 1.1, 0.0, 0.0], 1.0)
    assert not grasp_flag([1.2, 0.0, 0.0, 0.0, 0.0], 1.0)           # 엄지만
    assert not grasp_flag([0.5, 3.0, 3.0, 3.0, 3.0], 1.0)           # 엄지 없이
    assert not grasp_flag([1.0, 1.0, 0, 0, 0], 1.0)                 # 문턱은 '넘어야'
    assert not grasp_flag([], 1.0)


class _World:
    """컵을 쥐자마자 0.7 m/s 로 들어 올리는 손 + 0.29 s 늦게 오는 FP++(10.5 Hz)."""

    def __init__(self, grasp_t=1.0, lift=0.7):
        self.grasp_t, self.lift = grasp_t, lift
        self.cup_rel = np.array([0.0, 0.0, -0.04])               # 손바닥 아래 4 cm 에 컵

    def palm(self, t):
        z = 0.30 + max(0.0, t - self.grasp_t) * self.lift
        return np.array([0.30, -0.10, z]), _rz(30.0 * max(0.0, t - self.grasp_t))

    def cup_true(self, t):
        if t < self.grasp_t:
            return np.array([0.30, -0.10, 0.26]), Q0
        p, R = self.palm(t)
        return p + R @ self.cup_rel, None

    def fpp(self, t):
        """시각 t 에 도착한 마지막 프레임(찍힌 시각 = 도착 − LAT, 95 ms 마다)."""
        shot = np.floor((t - LAT) / 0.095) * 0.095
        p, _ = self.cup_true(shot)
        return p, Q0, shot


def test_attaches_with_the_palm_at_the_frame_time_not_the_current_palm():
    w, c = _World(), CupAttach(AttachCfg())
    t, attached_at, out = 0.0, None, None
    while t < 2.0:
        p, R = w.palm(t)
        grasped = t >= w.grasp_t
        out = c.step(grasped, t, p, R, w.fpp(t))
        if c.source == "attached" and attached_at is None:
            attached_at = t
        t += DT
    assert attached_at is not None
    assert w.grasp_t + LAT + 0.1 - 0.02 <= attached_at <= w.grasp_t + LAT + 0.1 + 0.12   # 파지 뒤 약 0.4 s
    p, R = w.palm(t - DT)
    truth = p + R @ w.cup_rel
    assert np.linalg.norm(out[0] - truth) < 0.002                     # 손을 따라간다(시각을 맞춰 붙였다)


def test_the_naive_rule_would_be_off_by_centimetres():
    """비교: 같은 프레임을 '지금' 손바닥에 붙이면 쥐자마자 들어 올린 만큼 틀린다(sim 77 mm 버그)."""
    w = _World()
    t = w.grasp_t + LAT + 0.12
    fp, _, _ = w.fpp(t)
    p_now, R_now = w.palm(t)
    rel_naive = R_now.T @ (fp - p_now)
    assert np.linalg.norm(rel_naive - w.cup_rel) > 0.05


def test_frames_shot_before_the_grasp_plus_100ms_do_not_attach():
    c = CupAttach(AttachCfg())
    pp, cup = np.array([0.3, -0.1, 0.3]), np.array([0.3, -0.1, 0.26])   # 1.0 s 에 쥐기 시작
    for k in range(12):                                                    # 1.05 s 에 찍힌 프레임은 이르다
        c.step(True, 1.0 + k * DT, pp, I3, (cup, Q0, 1.05))
    assert c.source == "live"
    c.step(True, 1.0 + 12 * DT, pp, I3, (cup, Q0, 1.10))                  # 1.10 s 에 찍힌 프레임 — 붙는다
    assert c.source == "attached"


def test_a_broken_grasp_restarts_the_clock():
    c = CupAttach(AttachCfg())
    pp = np.array([0.3, -0.1, 0.3])
    c.step(True, 1.00, pp, I3, None)
    c.step(False, 1.05, pp, I3, None)                               # 끊김 — 시작 시각을 잊는다
    c.step(True, 1.10, pp, I3, None)
    c.step(True, 1.50, pp, I3, (np.zeros(3), Q0, 1.15))            # 1.10 + 0.1 이전 프레임
    assert c.source == "live" and c.grasp_since == pytest.approx(1.10)
    c.step(True, 1.52, pp, I3, (np.zeros(3), Q0, 1.21))
    assert c.source == "attached"


def test_releases_after_fifteen_lost_steps_and_keeps_the_history():
    c = CupAttach(AttachCfg())
    pp = np.array([0.3, -0.1, 0.3])
    t = 0.0
    for _ in range(40):
        c.step(True, t, pp, I3, (np.array([0.3, -0.1, 0.26]), Q0, t - 0.3 if t > 0.4 else None))
        t += DT
    assert c.source == "attached"
    for k in range(14):
        c.step(False, t, pp, I3, None)
        t += DT
        assert c.source == "attached"
    out = c.step(False, t, pp, I3, (np.ones(3), Q0, t - 0.3))
    assert c.source == "live" and np.allclose(out[0], 1.0)
    assert c.palm_at(t - 0.2) is not None                           # 기록은 이어 간다


def test_no_attach_when_the_frame_is_older_than_the_palm_history():
    c = CupAttach(AttachCfg(history_s=0.2))
    pp = np.array([0.3, -0.1, 0.3])
    for k in range(31):                                                    # 1.0 → 1.5 s 쥔 채, 프레임 없음
        c.step(True, 1.0 + k * DT, pp, I3, None)
    c.step(True, 1.52, pp, I3, (np.zeros(3), Q0, 1.15))                   # 1.15 s 는 기록(최근 0.2 s) 밖
    assert c.source == "live"


def test_palm_interpolation_is_linear_in_position_and_slerp_in_rotation():
    c = CupAttach(AttachCfg())
    c.record(0.0, [0.0, 0.0, 0.0], _rz(0))
    c.record(0.1, [0.0, 0.0, 0.1], _rz(90))
    p, R = c.palm_at(0.05)
    assert p == pytest.approx([0.0, 0.0, 0.05]) and R == pytest.approx(_rz(45), abs=1e-9)
    assert c.palm_at(-0.01) is None and c.palm_at(0.15) is None    # 기록 밖 · 20 ms 넘는 외삽


def test_reset_forgets_everything():
    c = CupAttach(AttachCfg(attach_after_s=0.0))
    c.step(True, 0.0, np.zeros(3), I3, None)
    c.step(True, 0.1, np.zeros(3), I3, (np.ones(3), Q0, 0.05))
    assert c.source == "attached"
    c.reset()
    assert c.source == "live" and c.as_dict()["rel_pos"] is None and c.palm_at(0.05) is None


def test_joint_force_grasp_needs_a_thumb_axis_and_another_finger():
    from policy_control.cup_attach import grasp_flag_joint
    assert grasp_flag_joint([400, 0, 350, 0, 0, 0], 300)              # 엄지 굽힘 + 검지
    assert grasp_flag_joint([0, 400, 0, 0, 0, 310], 300)              # 엄지 회전 + 새끼
    assert not grasp_flag_joint([400, 400, 0, 0, 0, 0], 300)          # 엄지만
    assert not grasp_flag_joint([0, 0, 900, 900, 900, 900], 300)      # 엄지 없이
    assert not grasp_flag_joint([400, 0, 350, 0, 0], 300)             # 칸 수가 모자라면 아니다


def test_either_signal_attaches_on_a_thumb_phalanx_grasp_that_the_tips_miss():
    """10.04 Grasping: cyl60 정책은 엄지를 첫마디로 감싸 손끝 촉각엔 엄지가 잘 안 잡힌다 — 관절 힘으로는 잡힌다."""
    from policy_control.cup_attach import grasp_signal
    tips = [0.2, 1.5, 1.4, 0.0, 0.0]                                  # 엄지 손끝 0.2 N
    joints = [450, 120, 420, 380, 0, 0]                               # 엄지 굽힘 450 g
    assert not grasp_signal(tips, joints, AttachCfg(signal="tip"))
    assert grasp_signal(tips, joints, AttachCfg(signal="joint"))
    assert grasp_signal(tips, joints, AttachCfg(signal="either"))
    assert not grasp_signal(tips, None, AttachCfg(signal="either"))  # 관절 힘 소스가 없으면 손끝만
    assert grasp_signal([1.5, 1.2, 0, 0, 0], None, AttachCfg(signal="either"))
    with pytest.raises(ValueError):
        grasp_signal(tips, joints, AttachCfg(signal="bogus"))


def test_attach_records_how_many_steps_after_the_grasp_and_checks_fpp_against_fk():
    """Grasping 요청: 쥔 뒤 몇 스텝 만에 붙었는지, 붙은 뒤 FP++ 프레임 vs 같은 시각 FK 추정 차이(손 안 밀림 · 부착 오차)."""
    w, c = _World(), CupAttach(AttachCfg())
    t = 0.0
    while t < 2.0:
        p, R = w.palm(t)
        c.step(t >= w.grasp_t, t, p, R, w.fpp(t))
        t += DT
    d = c.as_dict()
    assert 20 <= d["attach_steps"] <= 32                              # 지연 0.29 s + 100 ms ≈ 24 스텝(프레임 간격만큼 더)
    assert d["check_n"] >= 5 and d["check_max_m"] < 0.002              # 맞게 붙었으면 뒤 프레임들도 FK 와 2 mm 안


def test_a_cup_slipping_in_the_hand_shows_up_in_the_check():
    w, c = _World(), CupAttach(AttachCfg())
    t = 0.0
    while t < 2.0:
        p, R = w.palm(t)
        if t > 1.6:
            w.cup_rel = np.array([0.0, 0.0, -0.06])                  # 컵이 손 안에서 2 cm 밀렸다
        c.step(t >= w.grasp_t, t, p, R, w.fpp(t))
        t += DT
    assert c.as_dict()["check_max_m"] > 0.015                         # 관측은 FK 그대로지만 기록은 밀림을 보인다


def test_optional_palm_distance_gate_blocks_finger_on_finger_false_grasps():
    c = CupAttach(AttachCfg(attach_after_s=0.0, max_palm_dist_m=0.12))
    far = (np.array([0.30, 0.20, 0.26]), Q0, 0.0)                     # 컵은 30 cm 떨어져 있는데 손가락끼리 닿아 힘이 실린다
    for k in range(20):
        c.step(True, k * DT, np.array([0.30, -0.10, 0.30]), I3, far)
    assert c.source == "live" and c.grasp_steps == 0
    near = (np.array([0.30, -0.10, 0.26]), Q0, 0.36)               # 파지가 인정된 21 스텝(0.35 s) 뒤에 찍힌 프레임
    c.step(True, 21 * DT, np.array([0.30, -0.10, 0.30]), I3, near)
    c.step(True, 22 * DT, np.array([0.30, -0.10, 0.30]), I3, near)
    assert c.source == "attached"


def test_a_release_after_attaching_is_counted_for_the_episode_runner():
    """10.04 에피소드 실행기(Step 2): 붙었다가 떨어진 횟수 = 컵 놓침 신호(OBJECT_DROPPED). 새 에피소드(reset)면 0."""
    c = CupAttach(AttachCfg(attach_after_s=0.0))
    pp = np.array([0.3, -0.1, 0.3])
    c.step(True, 0.0, pp, I3, None)
    c.step(True, 0.02, pp, I3, (np.array([0.3, -0.1, 0.26]), Q0, 0.01))
    assert c.source == "attached" and c.as_dict()["releases"] == 0
    for k in range(15):
        c.step(False, 0.04 + k * DT, pp, I3, None)
    assert c.source == "live" and c.as_dict()["releases"] == 1
    c.reset()
    assert c.as_dict()["releases"] == 0


def _run_static(cfg: AttachCfg, w: _World, until: float = 2.0):
    """에피소드 snapshot 재발행: 값은 닿기 전 컵 자리 그대로, stamp 는 '지금'(30 Hz 재발행)."""
    snap, _ = w.cup_true(0.0)
    c, t, attached_at, out = CupAttach(cfg), 0.0, None, None
    while t < until:
        p, R = w.palm(t)
        out = c.step(t >= w.grasp_t, t, p, R, (snap, Q0, t))
        if c.source == "attached" and attached_at is None:
            attached_at = t
        t += DT
    p, R = w.palm(t - DT)
    return c, attached_at, out, p + R @ w.cup_rel


def test_static_snapshot_attaches_with_the_palm_at_the_grasp_start():
    """10.04 사용자 "잡을 때 위치를 기준으로 핸드와 함께 FK": 정지 기록은 파지 시작 시각의 손바닥과 짝짓는다."""
    w = _World()
    c, attached_at, out, truth = _run_static(AttachCfg(static_source=True), w)
    assert attached_at is not None and w.grasp_t + 0.1 - 1e-9 <= attached_at <= w.grasp_t + 0.1 + 2 * DT
    assert np.linalg.norm(out[0] - truth) < 0.002                     # 들어 올린 손을 따라간다
    assert c.check_n == 0 and c.as_dict()["static"] is True          # 정지 기록과는 비교하지 않는다


def test_the_frame_time_rule_on_a_static_snapshot_would_be_off_by_the_lift():
    """비교: 정지 기록을 프레임 시각(= 지금) 규칙으로 붙이면 파지 뒤 100 ms 동안 든 만큼(0.7 m/s → 7 cm) 틀린다."""
    w = _World()
    _, _, out, truth = _run_static(AttachCfg(), w)
    assert np.linalg.norm(out[0] - truth) > 0.05


# ---------------------------------------------------------------- 10.12 A안: live FP++ 를 쥘 때까지 받는다
def _run_occluded(cfg: AttachCfg, w: _World, hidden_from: float, until: float = 2.0):
    """손이 컵을 가려 hidden_from 뒤로는 새 프레임이 없다 — 정책 노드는 마지막 프레임을 그대로 준다(hold-last)."""
    c, t, attached_at, out = CupAttach(cfg), 0.0, None, None
    last = None
    while t < until:
        p, R = w.palm(t)
        if t < hidden_from:
            last = w.fpp(t)
        out = c.step(t >= w.grasp_t, t, p, R, last)
        if c.source == "attached" and attached_at is None:
            attached_at = t
        t += DT
    p, R = w.palm(t - DT)
    return c, attached_at, out, p + R @ w.cup_rel


def test_an_occluded_grasp_falls_back_to_the_grasp_start_palm():
    """10.09 실기 live: 손이 다가가면 0.3~2.3 s 끊긴다. 파지 뒤 프레임이 안 오면 fallback_after_s 에 정지 컵 규칙(파지 시작 FK)."""
    w = _World()
    c, attached_at, out, truth = _run_occluded(AttachCfg(fallback_after_s=0.7), w, hidden_from=w.grasp_t - 0.2)
    assert attached_at is not None and w.grasp_t + 0.7 - 1e-9 <= attached_at <= w.grasp_t + 0.7 + 2 * DT
    assert np.linalg.norm(out[0] - truth) < 0.002                     # 컵은 닿기 전 자리 그대로였다 → 손을 따라간다
    assert c.as_dict()["fallback"] is True


def test_a_frame_that_arrives_before_the_fallback_wins():
    """프레임이 오면 sim PalmAttach 규칙(파지 + 100 ms 뒤 찍힌 프레임, 그 시각 손바닥) — 대개 파지 뒤 약 0.4 s."""
    w, c = _World(), CupAttach(AttachCfg(fallback_after_s=0.7))
    t, attached_at = 0.0, None
    while t < 2.0:
        p, R = w.palm(t)
        c.step(t >= w.grasp_t, t, p, R, w.fpp(t))
        if c.source == "attached" and attached_at is None:
            attached_at = t
        t += DT
    assert attached_at is not None and attached_at < w.grasp_t + 0.7
    assert c.as_dict()["fallback"] is False


def test_without_the_fallback_an_occluded_grasp_never_attaches():
    w = _World()
    c, attached_at, out, truth = _run_occluded(AttachCfg(), w, hidden_from=w.grasp_t - 0.2)
    assert attached_at is None and c.source == "live"
    assert np.linalg.norm(out[0] - truth) > 0.3                      # 정책은 탁자 위 컵을 계속 본다


@pytest.mark.parametrize("bad", [-0.1, 0.05, 0.1, 0.9])
def test_the_fallback_must_come_after_the_attach_delay_and_inside_the_palm_history(bad):
    with pytest.raises(ValueError, match="fallback_after_s"):
        AttachCfg(fallback_after_s=bad)                               # 0 = 끔, 아니면 attach_after_s < x ≤ history_s


# ---- 10.13 부착 감시(guard) — 실기 20261013_112100: 스친 오판으로 붙은 뒤 FP++ 컵과 FK 추정이 9 cm 까지 벌어졌다
def _false_attach(cfg, seconds=3.0, palm_v=0.05, regrasp=True):
    """컵(FP++, 지연 0)은 테이블에 그대로, 손바닥은 붙은 뒤 palm_v m/s 로 멀어진다 — 쥔 판정은 계속 참(오판)."""
    c = CupAttach(cfg)
    cup = (np.array([0.3, -0.1, 0.29]), Q0)
    t, seen = 0.0, []
    while t < seconds:
        p = np.array([0.25, -0.1, 0.30]) + (np.array([0.0, palm_v, 0.0]) * max(0.0, t - 0.5))
        out = c.step(True, t, p, I3, (cup[0], cup[1], t))
        seen.append((t, c.source, np.linalg.norm(out[0] - cup[0])))
        t += DT
    return c, seen


def test_without_guard_a_false_attachment_follows_the_hand_forever():
    c, seen = _false_attach(AttachCfg())
    assert seen[-1][1] == "attached" and seen[-1][2] > 0.1          # 10 cm 넘게 틀린 컵을 계속 준다


def test_guard_drops_a_false_attachment_and_returns_the_live_cup():
    c, seen = _false_attach(AttachCfg(guard_m=0.04, guard_frames=3))
    drop_t = next(t for t, s, _ in seen if s == "live" and t > 0.5)
    assert 0.5 + 0.04 / 0.05 <= drop_t <= 0.5 + 0.04 / 0.05 + 4 * DT   # 4 cm 넘은 뒤 3 프레임
    assert seen[-1][1] == "live" and seen[-1][2] < 1e-9               # 그 뒤로는 FP++ 그대로
    assert c.guard_drops == 1 and c.releases == 0 and c.guard_blocked  # 떨굼으로 세지 않는다
    assert c.as_dict()["guard_drops"] == 1


def test_guard_waits_for_the_grasp_to_break_before_attaching_again():
    c, _ = _false_attach(AttachCfg(guard_m=0.04, guard_frames=3))
    cup = (np.array([0.3, -0.1, 0.29]), Q0)
    t = 3.0
    for _ in range(30):                                                  # 쥠이 이어지는 동안: 다시 안 붙는다
        c.step(True, t, cup[0] + [0, 0, 0.01], I3, (cup[0], cup[1], t)); t += DT
    assert c.source == "live"
    c.step(False, t, cup[0], I3, (cup[0], cup[1], t)); t += DT           # 한 번 끊기면
    for _ in range(30):
        c.step(True, t, cup[0] + [0, 0, 0.01], I3, (cup[0], cup[1], t)); t += DT
    assert c.source == "attached"


def test_guard_keeps_a_real_grasp_with_small_slip():
    c, seen = _false_attach(AttachCfg(guard_m=0.04, guard_frames=3), palm_v=0.0)
    assert seen[-1][1] == "attached" and c.guard_drops == 0


def test_new_episode_clears_the_guard_count():
    c, _ = _false_attach(AttachCfg(guard_m=0.04, guard_frames=3))
    c.reset()
    assert c.guard_drops == 1                                            # 떼기 경로의 reset 은 누계를 지우지 않는다
    c.new_episode()
    assert c.guard_drops == 0 and not c.guard_blocked


@pytest.mark.parametrize("bad", [dict(guard_m=-0.01), dict(guard_m=float("nan")), dict(guard_frames=0)])
def test_guard_refuses_bad_values(bad):
    with pytest.raises(ValueError):
        AttachCfg(**bad)
