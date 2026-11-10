"""★10.25 준비 자세 이송(place_ready_carry) — newton_07_r · 06_l 계약 · PlaceChain 이송 → 정책 순서 · n05 는 그대로."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_control import place_ready_carry as RCY
from policy_control import rh_aglt as A
from policy_control import rh_place as P

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy/policies/rh56f1/place"
HDGP = REPO.parent / "hdgp"


@pytest.mark.parametrize("run", ["right_n07", "left_n06"])
def test_n07_n06_contracts_carry_to_the_trained_anchor(run):
    c = P.load_contract(POL / run / "rh_place_contract.json")
    cfg = RCY.ReadyCarryCfg.from_dict(c.ready_carry)
    assert (cfg.dz, cfg.v_max, cfg.qd_max, cfg.plan_down, cfg.max_steps) == (0.085, 0.0015, 0.012, 0.04, 600)
    assert c.latch_s2r and not c.carry and c.attach_source == "bundle"


@pytest.mark.parametrize("run", ["right_n05", "left_n05"])
def test_n05_has_no_ready_carry(run):
    assert P.load_contract(POL / run / "rh_place_contract.json").ready_carry == {}


def test_carry_end_runs_refuse_to_build_without_ready_carry(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "deploy/policy_control/tools"))
    import build_rh_aglt_contract as B
    assert "carry_end" in A.read_env(POL / "right_n07/params/env.yaml")["start_bank_path"]
    with pytest.raises(P.RhPlaceError, match="--ready-carry"):
        B.main(["--run", str(POL / "right_n07"), "--attach", "bundle", "--out", str(tmp_path / "c.json")])


class _Pol:
    def __init__(self):
        self.n, self.resets = 0, 0

    def reset(self):
        self.resets += 1

    def forward(self, obs):
        self.n += 1
        return np.full(13, 0.3)


class _Est:
    def __init__(self):
        self.n, self.resets = 0, 0

    def reset(self):
        self.resets += 1

    def step(self, obs):
        self.n += 1
        return np.zeros(6)

    def rel_pose(self, p, q):
        from policy_control import place_latch_est as LE
        return LE.rel_pose(p, q, np.zeros(6))

    def as_dict(self):
        return {}


NO_HDGP = not (HDGP / "source/openarm/openarm/agnostic/modules/ready_carry").is_dir()


def _setup(policy="right_n07"):
    c = P.load_contract(POL / f"{policy}/rh_place_contract.json")
    s = c.side()
    RC, mch = RCY.load_chain(s.palm_body, list(s.arm_joints))
    pol, est = _Pol(), _Est()
    ch = P.PlaceChain(c, pol)
    ch.object_name, ch.est = "source240_real", est
    hand = np.asarray(s.hand_grip, float) * 0.5
    q = np.array([-0.0276, 0.9205, -0.6622, 0.7405, -0.1584, 0.7484, 1.5155])   # sim 잡기 뱅크 1023 r 8 행(들기 끝 실측)
    att = (np.array([0.0167, 0.0376, 0.0313]), np.array([0.4015, 0.5551, 0.4426, 0.5786]))   # 같은 행 컵 ← 손바닥
    holder = np.array([0.38, -0.20, 0.205 + P.PC.HOLDER_BASE_DZ])

    def meas(qa):
        import torch
        T = mch.fk(torch.tensor([list(qa)], dtype=torch.float32))[0].double().numpy()
        palm = T[:3, 3]
        return {"arm": A.RaMeas(arm_q=np.asarray(qa), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, hand)),
                                palm_pos=palm, palm_R=T[:3, :3], tips=np.tile(palm, (5, 1)), cup_pos=np.zeros(3),
                                cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5), joint_force=None)}

    return c, RC, mch, pol, est, ch, hand, q, att, holder, meas


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_chain_carries_first_then_starts_the_policy_from_zero():
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup()
    with pytest.raises(P.RhPlaceError, match="Mover 사슬"):
        ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    assert ch.carrying and ch.rcarry.plan["T_steps"] > 0
    r0 = pol.resets
    for _ in range(c.ready_carry["max_steps"] + 200):
        _, a, _ = ch.step(meas(q))
        if not ch.carrying:
            break
        assert pol.n == 0 and est.n == 0 and not ch.latch                 # 이송 중 정책 · 추정기 안 돈다
        q = ch.dec.state["arm"].arm_target.copy()                  # 디코더 팔 목표(EMA · ±1 행동) 완벽 추종
    d = ch.rcarry.as_dict()
    assert not ch.carrying and d["carry_ok"] and d["arrived"] and d["err_mm"] < 10.0
    assert pol.resets == r0 + 1 and est.resets >= 1 and ch.step_i == 0 and np.all(ch.prev == 0)
    ch.step(meas(q))
    assert pol.n == 1 and est.n == 1                                   # 다음 틱부터 정책
    assert ch.as_dict()["phase"] == "place" and ch.as_dict()["ready_carry"]["arrived"]


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
@pytest.mark.parametrize("policy", ["right_n07", "right_n10_e4000"])
def test_carry_dq_never_exceeds_the_action_limit(policy):
    """★11.10 Mover 관절 상한 = k_arm·arm_ema — 행동이 ±1 에 잘리지 않아 계획대로 도착한다(실기 161342: 잘린 틱 95 %, 끝 err 48 mm)."""
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup(policy)
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    rc = ch.rcarry
    assert rc.qd_max == pytest.approx(c.k_arm * c.arm_ema) and rc.qd_max < c.ready_carry["qd_max"]
    raw, orig = [], rc.mv.step

    def step(qq, _o=orig):
        out = _o(qq)
        raw.append(float(out[0].abs().max()) / rc.scale)
        return out
    rc.mv.step = step
    while ch.carrying:
        ch.step(meas(q))
        q = ch.dec.state["arm"].arm_target.copy()
    assert max(raw) <= 1.0 + 1e-5
    assert rc.start_ok and rc.as_dict()["err_mm"] < 10.0


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_carry_that_does_not_reach_the_start_does_not_start_the_policy():
    """★11.10 팔이 따라가지 못해 컵이 sim 시작 분포 밖에서 이송이 끝나면 정책 상태를 열지 않고 다음 틱을 거부한다(노드는 abort)."""
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup()
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    r0 = pol.resets
    for _ in range(c.ready_carry["max_steps"] + 200):              # 실측 q 고정 = 팔이 안 움직임 → 계획 시간 + settle 뒤 끝
        ch.step(meas(q))
        if not ch.carrying:
            break
    d = ch.rcarry.as_dict()
    assert not ch.carrying and not d["start_ok"] and d["err_mm"] > RCY.START_ERR_MAX_M * 1000
    assert pol.resets == r0 and pol.n == 0
    with pytest.raises(P.RhPlaceError, match="시작 분포 밖"):
        ch.step(meas(q))
    assert pol.n == 0
    assert d["still"] == "track"                                     # ★11.11 팔이 명령을 못 따라간다 → 이탈 가드가 먼저 끊는다


def _run_carry(policy, wobble):
    """★11.11 이송 끝까지 굴린다. 실측 q = 명령 + wobble(틱) → (팔 행동 열, 끝 as_dict).
    Mover 가 끝난 뒤 2 틱 동안 실측 q 가 실제로 흔들리다 멈춘다(arm_qd 는 늘 0.2 — 판정에 안 쓴다)."""
    from dataclasses import replace
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup(policy)
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    acts, k = [], 0
    while ch.carrying:
        jig = 0.01 * (k % 2) if not ch.rcarry.moving and k <= ch.rcarry.t + 2 else 0.0   # Mover 끝 뒤 2 틱 흔들림
        mm = meas(q + wobble(k) + jig)
        mm["arm"] = replace(mm["arm"], arm_qd=np.full(7, 0.2))
        _, a, _ = ch.step(mm)
        acts.append(a[:7].copy())
        q = ch.dec.state["arm"].arm_target.copy()
        k += 1
    return np.asarray(acts), ch.rcarry.as_dict()


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_carry_follows_the_command_not_the_measured_arm_then_waits_until_still():
    """★11.11 사용자 "mover 말고 ready-approach 처럼 명령 목표로": 실측 q 가 떨려도 이송 행동은 같다(개루프).
    Mover 가 끝나면 팔 행동 0 으로 실측이 STILL_TICKS 연속 멈출 때까지 기다린 뒤 정책으로 넘긴다."""
    clean, d0 = _run_carry("right_n10_e4000", lambda k: np.zeros(7))
    shaky, d1 = _run_carry("right_n10_e4000", lambda k: 0.014 * np.sin(2 * np.pi * 1.75 * k / 60.0) * np.ones(7))
    t = d0["t"]                                                      # 이동 구간은 개루프 — 같다. 멈춤 확인 구간은 처짐 보정이 실측을 본다
    assert d1["t"] == t and np.allclose(clean[:t], shaky[:t], atol=1e-6)
    assert d0["arrived"] and d0["start_ok"] and d0["still"] == "ok"
    assert np.all(clean[t:] == 0.0) and d0["still_ticks"] == len(clean) - t
    assert d0["still_ticks"] >= RCY.STILL_TICKS + 2                  # 흔들리는 2 틱 동안은 세지 않는다


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
@pytest.mark.parametrize("off,fails", [(0.03, False), (0.15, True)])
def test_carry_aborts_when_the_arm_stays_off_the_command(off, fails):
    """★11.11 이탈 가드: held(디코더 시작 명령)와 실측이 off 만큼 어긋난 채 이송 — TRACK_ABORT_RAD 넘으면
    TRACK_ABORT_TICKS 틱에 이송을 끝내고 start_ok False("track"). 184024 시작 차이 0.026 rad 은 걸리지 않는다."""
    from dataclasses import replace
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup("right_n10_e4000")
    ch.rc_chain = (RC, mch)
    dq = np.zeros(7)
    dq[0] = off
    ch.reset(meas(q + dq), holder=holder, held=(q, hand), attach=att)
    n = 0
    while ch.carrying and n < 2000:
        mm = meas(ch.dec.state["arm"].arm_target + dq)
        mm["arm"] = replace(mm["arm"], arm_qd=np.zeros(7))
        cmd_before = ch.dec.state["arm"].arm_target.copy()
        _, _, tg = ch.step(mm)
        n += 1
    d = ch.rcarry.as_dict()
    assert not d["on"]
    assert d["track_dev_max"] == pytest.approx(off, abs=1e-6)          # L1 — 이송 중 이탈 최대가 로그에 남는다
    if fails:
        assert d["still"] == "track" and not d["start_ok"] and not d["carry_ok"] and d["t"] == RCY.TRACK_ABORT_TICKS - 1   # 발동 틱은 Mover 를 굴리지 않는다
        meas_q = np.asarray(mm["arm"].arm_q, float)                     # H1 — 발동 틱 목표 = 실측(현 위치, pd 튐 0)
        assert np.allclose(tg["arm"][0], meas_q) and np.allclose(ch.dec.state["arm"].arm_target, meas_q)
        assert np.max(np.abs(cmd_before - meas_q)) == pytest.approx(off)   # 덮기 전 명령은 실측보다 off 앞서 있었다
    else:
        assert d["still"] != "track" and d["t"] > RCY.TRACK_ABORT_TICKS and d["carry_ok"]


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_track_guard_also_runs_while_waiting_for_still():
    """★11.11 M2 — Mover 가 끝난 뒤 멈춤 확인 구간에서 팔이 명령에서 벗어나도 TRACK_ABORT_TICKS 에 "track"."""
    from dataclasses import replace
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup("right_n10_e4000")
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    off = np.zeros(7)
    k = 0
    while ch.carrying and k < 2000:
        if not ch.rcarry.moving:
            off[3] = 0.2                                                # 이송 끝 뒤 팔이 밀려 났다
            k += 1
        mm = meas(ch.dec.state["arm"].arm_target + off)
        mm["arm"] = replace(mm["arm"], arm_qd=np.zeros(7))
        ch.step(mm)
    d = ch.rcarry.as_dict()
    assert d["still"] == "track" and not d["start_ok"] and not d["carry_ok"]
    assert k == RCY.TRACK_ABORT_TICKS and d["still_ticks"] == RCY.TRACK_ABORT_TICKS - 1


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
@pytest.mark.parametrize("field", ["arm_q", "arm_qd"])
def test_nan_measurement_is_not_silent(field):
    """★11.11 M1 — 실측 q 가 NaN 이면 이탈로 세어 "track", 팔 목표는 유한(NaN 관절은 직전 명령).
    arm_qd 는 멈춤 판정에 안 쓴다 — NaN 이어도 실측 q 가 멈췄으면 "ok"(q 차분 NaN 은 아래 test_q_nan_while_waiting_is_not_still)."""
    from dataclasses import replace
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup("right_n10_e4000")
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    while ch.carrying:
        cmd = ch.dec.state["arm"].arm_target.copy()
        mm = meas(cmd)
        bad = cmd.copy() if field == "arm_q" else np.zeros(7)
        bad[2] = np.nan
        mm["arm"] = replace(mm["arm"], **{field: bad}) if field == "arm_qd" else replace(mm["arm"], arm_q=bad, arm_qd=np.zeros(7))
        _, _, tg = ch.step(mm)
    d = ch.rcarry.as_dict()
    if field == "arm_q":
        assert d["still"] == "track" and d["t"] == RCY.TRACK_ABORT_TICKS - 1 and d["track_dev_max"] is None
        assert np.all(np.isfinite(tg["arm"][0])) and np.all(np.isfinite(ch.dec.state["arm"].arm_target))
    else:
        assert d["still"] == "ok" and d["start_ok"]


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_q_nan_while_waiting_is_not_still():
    """★11.11 Mover 끝 뒤 실측 q 한 관절이 NaN 이면 멈춤으로 세지 않는다(arm_qd 0 이어도) — 이탈 가드가 "track" 으로 끊는다."""
    from dataclasses import replace
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup("right_n10_e4000")
    ch.rc_chain = (RC, mch)
    ch.reset(meas(q), holder=holder, held=(q, hand), attach=att)
    while ch.carrying:
        cmd = ch.dec.state["arm"].arm_target.copy()
        if not ch.rcarry.moving:
            cmd[2] = np.nan
        mm = meas(cmd)
        mm["arm"] = replace(mm["arm"], arm_q=cmd, arm_qd=np.zeros(7))
        ch.step(mm)
    d = ch.rcarry.as_dict()
    assert d["still"] == "track" and d["sag_ticks"] == 0


def _run_sag_plant(sag_j7, hold_qd_ticks=0, sag_max_rad=None, out=None, alpha=(1 / 60.0) / 0.14):
    """★11.11 처짐 플랜트: 실측 q 가 (명령 − 처짐)을 1차 지연(τ 0.14 s, 60 Hz)으로 따라간다. q̇ = 실측 차분.
    멈춤 확인 첫 hold_qd_ticks 틱은 q̇ 를 0.2 rad/s 로 거짓 보고(판정은 q 차분이라 무시돼야 한다) → (멈춤 확인 구간 팔 행동, 보고 q̇, 끝 as_dict, 끝 q_ref − q)."""
    from dataclasses import replace
    c, RC, mch, pol, est, ch, hand, q, att, holder, meas = _setup("right_n10_e4000")
    ch.rc_chain = (RC, mch)
    sag = np.zeros(7)
    sag[6] = sag_j7
    ch.reset(meas(q - sag), holder=holder, held=(q, hand), attach=att)
    ch.rcarry.track_abort_rad = 0.2                    # sim 런타임 값 — 지연 + 처짐을 이동 중 이탈로 끊지 않는다
    if sag_max_rad is not None:
        ch.rcarry.sag_max_rad = sag_max_rad
    qm, qd, acts, qds, n = q - sag, np.zeros(7), [], [], 0
    while ch.carrying and n < 3000:
        moving = ch.rcarry.moving
        if not moving and hold_qd_ticks > 0:
            qd_rep, hold_qd_ticks = np.full(7, 0.2), hold_qd_ticks - 1
        else:
            qd_rep = qd
        mm = meas(qm)
        mm["arm"] = replace(mm["arm"], arm_qd=qd_rep)
        _, a, tg = ch.step(mm)
        q_seen = qm
        if not moving:
            acts.append(a[:7].copy())
            qds.append(ch.rcarry.qd_est)                     # 판정에 쓴 q 차분 속도(보고 q̇ 는 안 쓴다)
        q_new = qm + alpha * (ch.dec.state["arm"].arm_target - sag - qm)
        qd, qm, n = (q_new - qm) * 60.0, q_new, n + 1
    d = ch.rcarry.as_dict()
    if out is not None:
        out.update(ch=ch, tg=tg)
    return np.asarray(acts), np.asarray(qds), d, ch.rcarry.q_ref - q_seen     # 끝 틱에 본 실측 기준


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_sag_correction_closes_steady_droop_without_integrating_while_moving():
    """★11.11 안 A — 처짐 j7 0.06 rad: 보정 뒤 max|q_ref − q| < SAG_DONE_RAD 로 "ok", 누적은 SAG_MAX_RAD 안,
    실측이 움직이는(q̇ ≥ STILL_QD_RAD_S) 틱에는 팔 행동 0(적분 없음)."""
    acts, qds, d, err = _run_sag_plant(0.06, hold_qd_ticks=10)
    assert d["still"] == "ok" and d["start_ok"] and d["carry_ok"]
    assert np.max(np.abs(err)) < RCY.SAG_DONE_RAD
    assert max(d["sag_max"]) <= RCY.SAG_MAX_RAD + 1e-9 and d["sag_max"][6] > 0.05
    moving = qds >= RCY.STILL_QD_RAD_S
    assert moving.any() and np.all(acts[moving] == 0.0)            # 보정이 만든 실측 움직임 틱은 적분하지 않는다
    assert d["sag_ticks"] == int((~moving).sum())
    assert d["still_ticks"] <= RCY.STILL_MAX_TICKS


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_sag_correction_ignores_nonzero_reported_qd():
    """★11.11 sim Newton joint_vel 처럼 정지 중에도 q̇ 0.2 rad/s 를 계속 보고해도 실측 q 가 멈추면 보정이 돌고 "ok"."""
    acts, qds, d, err = _run_sag_plant(0.06, hold_qd_ticks=10 ** 6)
    assert d["still"] == "ok" and d["sag_ticks"] > 0 and np.max(np.abs(err)) < RCY.SAG_DONE_RAD


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_sag_correction_never_exceeds_the_cap():
    """★11.11 처짐 0.12 rad(상한 0.08 넘음) — 누적이 SAG_MAX_RAD 에 멈추고 "timeout" 으로 끝난다."""
    _, _, d, err = _run_sag_plant(0.12)
    assert d["still"] == "timeout"
    assert max(d["sag_max"]) == pytest.approx(RCY.SAG_MAX_RAD, abs=1e-9)
    assert np.max(np.abs(err)) == pytest.approx(0.12 - RCY.SAG_MAX_RAD, abs=2e-3)


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_sag_timeout_returns_command_to_q_ref():
    """★11.11 timeout 이면 보정 누적을 정책에 넘기지 않는다 — 디코더 목표 · 내보낸 팔 목표 = q_ref. sag_max_rad 0.04(실기 첫 시험값)도 생성 뒤 바꿀 수 있다."""
    out = {}
    _, _, d, _ = _run_sag_plant(0.06, sag_max_rad=0.04, out=out)
    ch, tg = out["ch"], out["tg"]
    assert d["still"] == "timeout"
    assert max(d["sag_max"]) == pytest.approx(0.04, abs=1e-9)
    np.testing.assert_allclose(ch.dec.state["arm"].arm_target, ch.rcarry.q_ref, atol=1e-12)
    np.testing.assert_allclose(tg["arm"][0], ch.rcarry.q_ref, atol=1e-12)


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_policy_start_prev_action_is_zero():
    """★11.11 이송 끝 틱 마지막 보정 행동이 정책 첫 관측의 직전 행동으로 새지 않는다 — _place_begin 이 prev 0(학습 시작과 같다)."""
    out = {}
    acts, _, d, _ = _run_sag_plant(0.06, out=out)
    assert d["still"] == "ok" and d["start_ok"] and np.any(acts != 0.0)
    assert np.all(out["ch"].prev == 0.0)


@pytest.mark.skipif(NO_HDGP, reason="hdgp 없음")
def test_sag_not_interrupted_when_arm_follows_correction_steps(monkeypatch):
    """★11.11 sim e2e: 팔이 보정 명령을 바로 따라가 0.005 rad/틱(0.3 rad/s)으로 움직여도 추종 잔차는 그대로라 보정이 끊기지 않고 "ok".
    (실측 q 차분만 보던 판정은 이 움직임을 '안 멈춤'으로 봐 보정을 쉬었다 — sag_ticks 0~17.)"""
    monkeypatch.setattr(RCY, "K_SAG", 0.1)                          # 처짐 0.06 에서 스텝이 상한 SAG_STEP_MAX 에 닿게
    out = {}
    acts, _, d, err = _run_sag_plant(0.06, alpha=1.0, out=out)
    assert d["still"] == "ok" and d["start_ok"] and np.max(np.abs(err)) < RCY.SAG_DONE_RAD
    steps = np.abs(acts[:, 6]) * out["ch"].rcarry.scale
    assert np.max(steps) == pytest.approx(RCY.SAG_STEP_MAX, abs=1e-9)
    run = np.flatnonzero(steps > 0.0)
    assert len(run) == d["sag_ticks"] >= 5 and np.all(np.diff(run) == 1)   # 보정 틱이 끊기지 않고 이어진다
