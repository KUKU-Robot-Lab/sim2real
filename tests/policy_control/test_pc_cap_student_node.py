"""11.18 rh_cap 깊이 학생 노드(rh_cap_student_node)의 ROS 없는 절반 — 구간 기계 · 구간별 실행 행동 · 학생 예외 뒤 상태 ·
LSTM 리셋 시점 · 머리 명령 본문 · 계약 대조 · stage/next · 틱의 머리 발행 조건. rclpy.init · 노드 기동 없음(가짜 정책).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from policy_control import cap_student as CS
from policy_control import place_student as PS
from policy_control import rh_aglt as A
from policy_control import rh_cap as C
from policy_control import rh_cap_student_node as N
from policy_control.pour_fj_node import FAMILIES, PourFjNode

REPO = Path(__file__).resolve().parents[2]
CARD = yaml.safe_load((REPO / "deploy/policies/rh56f1/cap/right_student_h01/student.yaml").read_text(encoding="utf-8"))
TICK_ERRORS = (*FAMILIES["rh_cap"].errors, *N.STUDENT_ERRORS)          # 노드가 틱 오류(→ gap → abort)로 받는 예외
HEAD = np.array([0.0, 40.0])


@pytest.fixture(scope="module")
def c() -> C.CapContract:
    return C.load_contract(Path(__file__).resolve().parent / "fixtures" / "rh_cap_seat_contract.json")


class FakePolicy:
    """CapStudent 자리 — 인자 순서 · 모양을 검사하고 정해 둔 행동(또는 예외)을 낸다."""

    def __init__(self) -> None:
        self.resets, self.obs, self.a, self.exc = 0, [], np.full(13, 0.5), None

    def reset(self) -> None:
        self.resets += 1

    def step(self, depth, obs, age):
        assert depth.shape == CS.IMG_HW and obs.shape == (CS.N_OBS,) and isinstance(age, float)
        if self.exc is not None:
            raise self.exc
        self.obs.append(obs.copy())
        return self.a.copy(), CS.head_deg([0.2, -0.4])


def _meas(c, arm_q=N.LIVE_ARM_START_Q):
    s = c.side()
    m = A.RaMeas(arm_q=np.asarray(arm_q, float), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, s.hand_open)),
                 palm_pos=np.array([0.3, -0.2, 0.4]), palm_R=np.eye(3), tips=np.zeros((5, 3)), cup_pos=np.zeros(3),
                 cup_quat=np.array([1.0, 0.0, 0.0, 0.0]), tactile_n=np.zeros(5))
    return {"arm": m, "depth": (np.full(CS.IMG_HW, 0.8, np.float32), 0.01), "head": HEAD}


class FakeScripts:
    """cap_scripts.CapScripts 자리 — 하강 · 이송 팔 행동은 A_SCRIPT, 끝 신호는 done 으로 정한다."""
    A_SCRIPT = np.full(7, 0.2)

    def __init__(self) -> None:
        self.done, self.calls, self.resets, self.carry_missed = False, [], 0, False

    def reset(self) -> None:
        self.resets += 1

    def descend_step(self, m, hand_q, arm_target, cap):
        self.calls.append(("descend", cap))
        return self.A_SCRIPT.copy(), self.done

    def carry_step(self, m, arm_target, rcv):
        self.calls.append(("carry", rcv))
        return self.A_SCRIPT.copy(), self.done

    def as_dict(self) -> dict:
        return {}


POSES = {"cap": (np.array([0.215, -0.185, 0.219]), np.array([1.0, 0, 0, 0])),
         "rcv": (np.array([0.25, 0.12, 0.41]), np.array([1.0, 0, 0, 0]))}


def _chain(c, to: str = "search"):
    ch = N.CapStudentChain(c, FakePolicy())
    ch.scripts = FakeScripts()
    ch.reset(_meas(c))
    ch.poses = POSES
    while ch.stages.phase != to:
        ch.step(_meas(c))
        ch.next()
    return ch


def _targets(ch):
    st = ch.dec.state["arm"]
    return st.arm_target.copy(), st.hand_target.copy()


# ---------------------------------------------------------------- 구간 기계
def test_next_walks_all_phases_then_done():
    st = N.Stages()
    assert [st.phase] + [st.next() for _ in N.PHASES] == [*N.PHASES, N.DONE]
    assert set(N.STAGE_OF) == set(N.PHASES) and set(N.STAGE_OF.values()) == set(CS.STAGES)


def test_auto_only_moves_scripted_phases():
    st = N.Stages(carry_wait=3)
    for ph in ("search", "approach"):                                  # 학생 구간은 끝 신호가 와도 스스로 안 넘어간다
        assert not st.auto(True) and st.phase == ph
        st.next()
    assert st.phase == "descend" and not st.auto(False)
    assert st.auto(True) and st.phase == "grasp"                       # 하강 끝 → 잡기
    assert not st.auto(True) and st.next() == "carry"
    assert [st.auto(True), st.auto(False), st.auto(True), st.auto(True)] == [False] * 4   # 끝 신호가 끊기면 다시 센다
    assert st.auto(True) and st.phase == "seat"                        # 이송 끝 뒤 3 틱째
    assert not st.auto(True) and st.phase == "seat"


def test_default_carry_wait_is_training_value():
    st = N.Stages()
    st.phase = "carry"
    assert [st.auto(True) for _ in range(60)] == [False] * 59 + [True]


# ---------------------------------------------------------------- 체인
# 묶음 obs.stages [grasp, carry, seat, approach, search] 순 one-hot — 하강은 잡기 칸(리터럴로 고정, N.STAGE_OF 를 다시 쓰지 않는다)
ONEHOT = {"search": [0, 0, 0, 0, 1], "approach": [0, 0, 0, 1, 0], "descend": [1, 0, 0, 0, 0], "grasp": [1, 0, 0, 0, 0],
          "carry": [0, 1, 0, 0, 0], "seat": [0, 0, 1, 0, 0]}


def test_stage_column_follows_phase_and_prev_is_executed_action(c):
    ch = _chain(c)
    for ph in N.PHASES:
        assert ch.stages.phase == ph
        obs, a_out, _ = ch.step(_meas(c))
        want = ONEHOT[ph]
        np.testing.assert_allclose(obs[55:60], want)
        np.testing.assert_allclose(obs[60:62], CS.head_obs(HEAD))      # 실측 인코더 각 그대로
        obs2, _, _ = ch.step(_meas(c))
        np.testing.assert_allclose(obs2[42:55], a_out, atol=1e-6)      # 직전 행동 = 실행한 행동
        ch.next()
    assert ch.stages.phase == N.DONE
    np.testing.assert_allclose(ch.policy.obs[4][55:60], [1, 0, 0, 0, 0])   # 하강 구간의 단계 칸 = grasp


def test_executed_action_per_phase(c):
    s, k = c.side(), slice(7, 13)
    ch = _chain(c)
    for _ in ("search", "approach"):                                   # 팔 = 학생 그대로 · 손 = 편 손
        arm0, _ = _targets(ch)
        _, a, _ = ch.step(_meas(c))
        np.testing.assert_allclose(a[:7], 0.5)
        np.testing.assert_allclose(a[k], ch.a_open)
        arm1, hand1 = _targets(ch)
        assert np.all(arm1 > arm0)
        np.testing.assert_allclose(hand1, s.hand_open)
        ch.next()
    _, a, _ = ch.step(_meas(c))                                        # descend: 팔 = 하강 스크립트 · 손 폄
    np.testing.assert_allclose(a, np.concatenate([FakeScripts.A_SCRIPT, ch.a_open]))
    assert ch.scripts.calls[-1] == ("descend", POSES["cap"])
    ch.scripts.done = True
    ch.step(_meas(c))                                                  # 하강 끝 → 잡기(자동)
    ch.scripts.done = False
    arm0, _ = _targets(ch)
    assert ch.stages.phase == "grasp"                                  # grasp: 손 절대 · 팔 상자(중심 = 넘긴 순간 팔 목표)
    np.testing.assert_allclose(ch.center, arm0)
    ch.policy.a = np.ones(13)
    for _ in range(400):
        _, a, _ = ch.step(_meas(c))
    np.testing.assert_allclose(a[k], 1.0)
    assert np.all(_targets(ch)[0] <= np.minimum(ch.center + c.phase_arm_box_rad, s.arm_hi) + 1e-9)
    assert np.any(_targets(ch)[0] > ch.center + 0.25)
    held = _targets(ch)
    ch.next()                                                          # carry: 팔 = 이송 스크립트 · 손 = 잡기 끝 손 목표
    for _ in range(5):
        _, a, _ = ch.step(_meas(c))
    np.testing.assert_allclose(a[:7], FakeScripts.A_SCRIPT)
    assert ch.scripts.calls[-1] == ("carry", POSES["rcv"])
    np.testing.assert_allclose(_targets(ch)[1], held[1], atol=1e-9)
    ch.scripts.done = True
    for _ in range(N.CARRY_WAIT_TICKS):                                # 이송 끝 뒤 60 틱 → 끼우기(자동)
        assert ch.stages.phase == "carry"
        ch.step(_meas(c))
    held = _targets(ch)
    assert ch.stages.phase == "seat"                                   # seat: 손 = 넘긴 목표 + 잔차 · 팔 상자 중심 = 넘긴 팔 목표
    np.testing.assert_allclose(ch.center, held[0])
    ch.policy.a = np.concatenate([np.zeros(7), np.full(6, -0.1)])
    _, a, _ = ch.step(_meas(c))
    np.testing.assert_allclose(a[k], CS.seat_hand_action(np.full(6, -0.1), held[1], *ch.dec.law.bounds()))
    assert np.all(_targets(ch)[1] <= held[1] + 1e-9) and np.any(_targets(ch)[1] < held[1])


@pytest.mark.parametrize("exc", [CS.StaleDepth("낡음"), ValueError("모양"), FloatingPointError("NaN")])
@pytest.mark.parametrize("phase", ["approach", "carry", "seat"])
def test_student_error_is_a_tick_error_and_leaves_state(c, exc, phase):
    ch = _chain(c, phase)
    ch.step(_meas(c))
    before = (*_targets(ch), ch.prev.copy(), ch.step_i, ch.stages.phase, ch.head_cmd.copy())
    ch.policy.exc = exc
    with pytest.raises(TICK_ERRORS):
        ch.step(_meas(c))
    after = (*_targets(ch), ch.prev, ch.step_i, ch.stages.phase, ch.head_cmd)
    for b, a in zip(before, after):
        np.testing.assert_array_equal(b, a)
    ch.policy.exc = None
    ch.step(_meas(c))                                                  # 다음 정상 틱은 그대로 잇는다
    assert ch.step_i == before[3] + 1


def test_stale_depth_from_real_core_reaches_chain_unchanged(c):
    """실제 CapStudent.step 의 인자 순서(깊이, 관측, 나이)로 불리는지 — 낡은 나이는 StaleDepth, 모델은 안 돈다."""
    pol = CS.CapStudent.__new__(CS.CapStudent)
    pol.max_depth_age_s, pol.reset = 0.2, lambda: None
    ch = N.CapStudentChain(c, pol)
    ch.reset(_meas(c))
    with pytest.raises(CS.StaleDepth):
        ch.step({**_meas(c), "depth": (np.zeros(CS.IMG_HW, np.float32), 0.5)})
    assert ch.step_i == 0


def test_lstm_reset_only_at_episode_reset(c):
    ch = _chain(c, "seat")                                             # 모든 구간 전환을 지났다
    ch.step(_meas(c))
    assert ch.policy.resets == 1
    ch.reset(_meas(c))
    assert ch.policy.resets == 2 and ch.stages.phase == "search" and not ch.prev.any() and ch.step_i == 0


# ---------------------------------------------------------------- 순수 함수
def test_head_command_is_the_student_output_untransformed():
    deg = CS.head_deg([0.0, 0.0])
    assert json.loads(N.head_stream_body(7, deg)) == {"seq": 7, "pan_deg": -7.5, "tilt_deg": 40.0}
    assert json.loads(N.head_stream_body(8, CS.head_deg([5.0, -5.0]))) == {"seq": 8, "pan_deg": 30.0, "tilt_deg": -2.0}
    look = PS.HeadLook(None)                                           # 놓기 학생이 내는 본문과 같은 형식
    look.cmd = deg
    assert N.head_stream_body(1, deg) == look.command(None, None, None, hold=True)


def test_start_refusal_outside_training_start_pose():
    assert N.start_refusals(N.LIVE_ARM_START_Q, 0.15) == []
    far = np.array(N.LIVE_ARM_START_Q) + [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]
    assert "0.200 rad" in N.start_refusals(far, 0.15)[0]


def test_contract_must_match_student_card(c):
    bad = N.contract_mismatch(c, CARD)                                 # 고정본 = 1판 자산 · 어드민턴스 · hand_block 없음
    assert sorted(b.split(":")[0] for b in bad) == ["cap_phase", "checkpoint_md5", "hand_block", "hand_command", "profile"]
    ok = replace(c, profile=CARD["train"]["profile_name"], hand_command="position", hand_block={"g_rad": 0.02},
                 cap_phase=C.CHAIN, checkpoint_md5=CARD["ckpt"]["md5"])
    assert N.contract_mismatch(ok, CARD) == []
    assert [b.split(":")[0] for b in N.contract_mismatch(replace(ok, k_arm=0.04), CARD)] == ["k_arm"]
    assert len(N.contract_mismatch(ok, {"side": "right"})) >= 8        # 묶음에 값이 없으면 통과가 아니다


# ---------------------------------------------------------------- 노드 메서드(ROS 초기화 없이 빈 객체에)
def _node(c, phase="running", to="search"):
    n = N.CapStudentNode.__new__(N.CapStudentNode)
    n.ended, n.sent, n._publish, n._gap, n._lg_stop = [], [], True, 0, None
    n.book, n.chain = SimpleNamespace(phase=phase), _chain(c, to)
    n._end = lambda kind, why: (n.ended.append((kind, why)), setattr(n.book, "phase", "stopped"))
    n._drain = lambda: None
    n._head, n._head_on, n._head_seq = SimpleNamespace(latest=lambda now: HEAD), True, 0
    n._String = lambda data: data
    n._pub_head = SimpleNamespace(publish=n.sent.append)
    n._depth_cfgs = {json.dumps(FREE): None}                           # 받은 깊이 노드 config = free-head 하나
    return n


def _call_next(n):
    res = N.CapStudentNode._srv_next(n, None, SimpleNamespace())
    return res.success, json.loads(res.message)["reasons"]


def test_stage_next_service(c):
    n = _node(c, phase="armed")
    assert _call_next(n)[0] is False and n.chain.stages.phase == "search"      # 실행 중이 아니면 거부
    n = _node(c)
    ok, why = _call_next(n)
    assert not ok and "성공한 틱" in why[0] and n.chain.stages.phase == "search"   # 이 구간에서 아직 한 틱도 안 돌았다
    n.chain.step(_meas(c))
    n._gap = 2
    ok, why = _call_next(n)
    assert not ok and "오류" in why[0]                                          # 직전 틱이 오류
    n._gap = 0
    assert _call_next(n) == (True, ["search → approach"]) and not n.ended
    assert _call_next(n)[0] is False                                           # 넘긴 구간에서 다시 세기 시작
    n = _node(c, to="seat")
    n.chain.step(_meas(c))
    assert _call_next(n) == (True, ["seat → done"])
    assert n.ended[0][0] == "stop"                                             # 끼우기 뒤 = 에피소드 stop(pd 가 붙든다)


def test_tick_publishes_head_only_after_a_good_student_step(c, monkeypatch):
    n = _node(c)
    good = lambda self: self.chain.step(_meas(c))  # noqa: E731
    monkeypatch.setattr(PourFjNode, "_on_tick", good)
    n._on_tick()
    assert [json.loads(s) for s in n.sent] == [{"seq": 1, "pan_deg": 0.0, "tilt_deg": 23.2}]   # head_deg([0.2, −0.4])
    monkeypatch.setattr(PourFjNode, "_on_tick", lambda self: None)             # 틱 오류(스텝이 안 나아감) — 머리 명령도 없다
    n._on_tick()
    assert len(n.sent) == 1

    def stale(now):
        raise PS.PlaceInputStale("목 끊김")
    n._head.latest = stale
    monkeypatch.setattr(PourFjNode, "_on_tick", good)
    n._on_tick()
    assert n.ended[0][0] == "abort" and len(n.sent) == 1 and n.chain.step_i == 1   # 정책 틱 전에 멈춘다


def test_no_head_stream_when_targets_are_not_published(c, monkeypatch):
    """무발행 점검(publish_target false)에서는 목도 움직이지 않는다."""
    n = _node(c)
    n._publish = False
    monkeypatch.setattr(PourFjNode, "_on_tick", lambda self: self.chain.step(_meas(c)))
    n._on_tick()
    assert n.chain.step_i == 1 and n.sent == []


def test_start_refused_when_hand_is_not_open(c):
    s = c.side()
    assert N.hand_open_refusals(s.hand_open, s.hand_open) == []
    near = np.asarray(s.hand_open, float) + [0.0, 0.0, 0.14, 0.0, 0.0, 0.0]
    assert N.hand_open_refusals(near, s.hand_open) == []
    closed = np.asarray(s.hand_open, float) + [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]                   # 쥔 채 reset → start
    assert "1.000 rad" in N.hand_open_refusals(closed, s.hand_open)[0]


def test_next_refused_in_scripted_phases():
    for ph in N.SCRIPTED:
        assert "스크립트" in N.next_refusal(ph, 10, 0)[0]
    assert N.next_refusal("grasp", 1, 0) == []
    assert N.next_refusal("carry", 10, 0, carry_missed=True) == []                          # 이송 미도달 — 사람이 넘긴다
    assert N.next_refusal("descend", 10, 0, carry_missed=True)


def test_carry_missed_holds_without_auto_seat(c):
    ch = _chain(c, "carry")
    ch.scripts.carry_missed = True                                                          # Mover 가 도착 없이 끝남 → 끝 신호 False
    for _ in range(N.CARRY_WAIT_TICKS + 5):
        ch.step(_meas(c))
    assert ch.stages.phase == "carry" and ch.as_dict()["note"]


def test_scripted_phase_without_poses_is_a_tick_error(c):
    ch = _chain(c, "descend")
    ch.poses = None
    with pytest.raises(TICK_ERRORS):
        ch.step(_meas(c))


def test_stale_receiver_needs_rcv_allow_fixed(c):
    from policy_control import cap_scripts as CSC
    stale = CSC.pick_pose((0.0, [0.3, -0.1, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0), (CSC.RCV_FIXED_POS, CSC.RCV_FIXED_QUAT), 9.0, 0.5)
    assert "rcv_allow_fixed" in CSC.rcv_fixed_refusals(stale, False, 0.5)[0]
    assert CSC.rcv_fixed_refusals(stale, True, 0.5) == []
    n = _node(c)                                                       # 노드 경로: FP++ 수신 없음 → 고정 자리
    n.contract, n.cups = c, {}
    n._fixed = {"cap": (CSC.CAP_FIXED_POS, CSC.CAP_FIXED_QUAT), "rcv": (CSC.RCV_FIXED_POS, CSC.RCV_FIXED_QUAT)}
    n._rcv_allow_fixed = False
    assert any("rcv_allow_fixed" in b for b in N.CapStudentNode._start_refusals(n, _meas(c), 0.15))
    n._rcv_allow_fixed = True
    assert N.CapStudentNode._start_refusals(n, _meas(c), 0.15) == []


def test_lg_guard_stop_silences_head_stream(c, monkeypatch):
    n = _node(c)
    n._lg_stop = "guard"
    monkeypatch.setattr(PourFjNode, "_on_tick", lambda self: self.chain.step(_meas(c)))
    n._on_tick()
    assert n.sent == []


def test_stage_next_passes_missed_carry_to_seat(c):
    n = _node(c, to="carry")
    n.chain.step(_meas(c))
    assert _call_next(n)[0] is False                                   # 미도달이 아니면 스크립트 구간 — 자동만
    n.chain.scripts.carry_missed = True
    assert _call_next(n) == (True, ["carry → seat"])


# ---------------------------------------------------------------- grasp_near_m 거리(11.18 fake 97: 첫 search 스텝 RhAgltError)
H01 = REPO / "deploy/policies/rh56f1/cap/right_student_h01/rh_cap_contract.json"


def _walk(ch, c, meas=None):
    """search → seat 를 다 돈다(하강 끝 신호는 FakeScripts.done). 구간마다 한 틱 이상."""
    m = meas or _meas(c)
    for ph in N.PHASES:
        assert ch.stages.phase == ph
        ch.scripts.done = ph == "descend"
        ch.step(m)
        if ch.stages.phase == ph:
            ch.next()


def test_decoder_gets_cap_palm_distance(c):
    """dec.step 에 obj_palm_m = |캡 원점 − 손바닥| — 하강 끝 전 = start 공급 자세, 뒤 = 그 순간 손바닥에 붙인 캡."""
    ch = _chain(c)
    got, step = [], ch.dec.step
    ch.dec.step = lambda a, **kw: (got.append((ch.stages.phase, kw)), step(a, **kw))[1]
    m0 = _meas(c)
    d0 = float(np.linalg.norm(POSES["cap"][0] - m0["arm"].palm_pos))
    _walk(ch, c)
    assert [g[0] for g in got] == list(N.PHASES) and all("joint_force" in kw for _, kw in got)
    np.testing.assert_allclose([kw["obj_palm_m"] for _, kw in got], d0)   # 손바닥이 그대로면 붙인 뒤도 같은 거리
    m1 = _meas(c)                                                       # 잡은 뒤 손바닥이 움직이고 돌면 캡이 따라간다
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    m1["arm"] = replace(m1["arm"], palm_pos=m1["arm"].palm_pos + [0.1, 0.0, 0.2], palm_R=R)
    np.testing.assert_allclose(ch.cap_pos(m1["arm"]), m1["arm"].palm_pos + R @ (POSES["cap"][0] - m0["arm"].palm_pos))


def test_real_h01_decoder_runs_every_phase():
    """회귀: 실제 RaDecoder + h01 계약(grasp_near_m 0.08 · hand_block) 으로 모든 구간 스텝이 RhAgltError 없이 돈다."""
    c1 = C.load_contract(H01)
    assert c1.grasp_near_m > 0.0 and c1.hand_block is not None
    ch = N.CapStudentChain(c1, FakePolicy())
    ch.scripts = FakeScripts()
    ch.reset(_meas(c1))
    ch.poses = POSES
    _walk(ch, c1)
    assert ch.stages.phase == N.DONE and ch.step_i == len(N.PHASES)


@pytest.mark.parametrize("phase", ["search", "grasp"])
def test_nan_cap_or_palm_is_a_tick_error(phase):
    """캡 자세가 없거나 NaN 이면 큰 값 · 0 으로 바꾸지 않고 틱 오류 — 디코더 목표 · 구간 그대로."""
    c1 = C.load_contract(H01)
    ch = N.CapStudentChain(c1, FakePolicy())
    ch.scripts = FakeScripts()
    ch.reset(_meas(c1))
    ch.poses = POSES
    while ch.stages.phase != phase:
        ch.scripts.done = ch.stages.phase == "descend"
        ch.step(_meas(c1))
        if ch.stages.phase not in (phase, "grasp"):
            ch.next()
    before = (*_targets(ch), ch.step_i)
    bad = _meas(c1)
    if phase == "search":                                              # 캡 NaN = 이 노드의 검사
        ch.poses = {**POSES, "cap": (np.array([np.nan, 0.0, 0.2]), POSES["cap"][1])}
    else:                                                               # 잡은 뒤 손바닥 NaN = 관측 검사가 먼저 막는다
        bad["arm"] = replace(bad["arm"], palm_pos=np.array([np.nan, 0.0, 0.4]))
    with pytest.raises(TICK_ERRORS, match="NaN" if phase == "search" else "palm_pos"):
        ch.step(bad)
    for b, a in zip(before, (*_targets(ch), ch.step_i)):
        np.testing.assert_array_equal(b, a)
    ch.poses = None
    with pytest.raises(TICK_ERRORS, match="캡 자세가 없다"):
        ch.step(_meas(c1))


# ---------------------------------------------------------------- 깊이 노드 config 대조(11.18)
FREE = {"block": 8, "out_wh": [160, 90], "floor_cut": False, "head_gate": False, "clip_m": [0.01, 10.0], "extrinsics_md5": None}
FIXED = {"block": 8, "out_wh": [160, 90], "floor_cut_z_m": 0.155, "extrinsics_md5": "691ade15493f240e6c7b98c4393446a9"}


def test_depth_config_refusals_free_head_passes_others_refused():
    assert N.depth_config_refusals(FREE) == []
    assert "config 없음" in N.depth_config_refusals(None)[0]
    bad = N.depth_config_refusals(FIXED)                                 # 머리 고정 모드: 새 키 셋이 없다
    assert len(bad) == 3 and all("(키 없음)" in b for b in bad), bad
    assert [b.split(" = ")[0].split()[-1] for b in N.depth_config_refusals({**FREE, "clip_m": [0.2, 1.5]})] == ["clip_m"]
    assert N.depth_config_refusals({**FREE, "floor_cut": 0}) != []       # 0 == False 로 통과하지 않는다
    assert N.depth_config_refusals({**FREE, "out_wh": [320, 180]}) != []


def test_depth_configs_refusals_any_mismatch_refuses():
    """latched 발행자마다 한 통 — 하나라도 어긋나면(도착 순서와 무관하게) 거부."""
    ok, fixed = json.dumps(FREE), json.dumps(FIXED)
    assert N.depth_configs_refusals([ok]) == []
    assert "config 없음" in N.depth_configs_refusals([])[0]
    assert N.depth_configs_refusals([ok, fixed]) and N.depth_configs_refusals([fixed, ok])
    assert "JSON 아님" in N.depth_configs_refusals(["{"])[0]
    assert N.depth_configs_refusals(["[1]"]) != []


class _PastDepthCheck(Exception):
    """대조를 통과해 다음 단계(_wait_depth)까지 왔다는 표시."""


def _depth_node(c, cfgs):
    from policy_control import cap_scripts as CSC
    n = _node(c)
    n._depth_cfgs = dict.fromkeys(cfgs)
    n.contract, n.cups, n.reset_tol, n._rcv_allow_fixed = c, {}, 0.15, True
    n._fixed = {"cap": (CSC.CAP_FIXED_POS, CSC.CAP_FIXED_QUAT), "rcv": (CSC.RCV_FIXED_POS, CSC.RCV_FIXED_QUAT)}
    n._held = lambda meas: "held"

    def past():
        raise _PastDepthCheck
    n._wait_depth = past
    n._measure = lambda advance=False: _meas(c)
    n.fam = SimpleNamespace(refusals=lambda _c, meas, tol: N.CapStudentNode._start_refusals(n, meas, tol), errors=TICK_ERRORS)
    n.book = SimpleNamespace(phase="armed", start=lambda: (None, ["book.start"]))   # 대조를 통과하면 여기까지 온다
    return n


@pytest.mark.parametrize("cfgs, ok", [([], False), ([json.dumps(FREE), json.dumps(FIXED)], False), ([json.dumps(FREE)], True)],
                         ids=["none", "fixed_mixed", "free_only"])
def test_reset_and_start_check_depth_configs(c, cfgs, ok):
    """reset 과 start 둘 다 받은 깊이 config 를 대조한다 — reset 뒤 머리 고정 노드가 올라와도 start 가 거부한다."""
    n = _depth_node(c, cfgs)
    with pytest.raises(_PastDepthCheck if ok else N.PourFjNodeError):
        N.CapStudentNode._reset_chain(n, _meas(c))
    res = N.CapStudentNode._srv_start(n, None, SimpleNamespace())
    why = json.loads(res.message)["reasons"]
    assert not res.success and (why == ["book.start"] if ok else any(N.CONFIG_TOPIC in b for b in why)), why


def test_real_free_head_config_and_fake_publisher_pass():
    """head_depth_node.free_head_config 실제 dict 가 통과하고 fake 발행기 값이 그것과 같다(키 이름 어긋남을 잡는다)."""
    from policy_control import hdgp_pin
    if not (hdgp_pin.hdgp_tree() / ".git").exists():
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    hdn = pytest.importorskip("head_depth_node")
    import fake_head_depth_pub as F
    real = hdn.free_head_config(hdn.real_depth.BLOCK, hdn.real_depth.MIN_VALID)
    assert N.depth_config_refusals(real) == []
    assert {k: real[k] for k in F.CONFIG} == F.CONFIG
    assert N.depth_configs_refusals([json.dumps({**F.CONFIG, "fake": True})]) == []


@pytest.mark.parametrize("script", ["fake_head_depth_pub", "fake_head_stream_driver", "cap_student_fake_drive"])
@pytest.mark.parametrize("dom", ["", "0", "50", "050", "abc", " "])
def test_cap_fakes_refuse_real_or_bad_domain(script, dom, monkeypatch):
    """fake 는 ROS 를 붙이기 전에 실기 · 기본 · 잘못된 도메인을 거부한다("050" = 실기 50)."""
    import importlib
    import sys
    from policy_control.ros_domain import REAL_DOMAIN
    assert REAL_DOMAIN == 50
    mod = importlib.import_module(script)
    monkeypatch.setenv("ROS_DOMAIN_ID", dom)
    monkeypatch.setattr(sys, "argv", [script, *(["--out", "unused"] if script == "cap_student_fake_drive" else [])])
    monkeypatch.setitem(sys.modules, "rclpy", None)                 # 검사를 지나 ROS 를 붙이려 하면 ImportError
    with pytest.raises(SystemExit, match="fake 도메인에서만"):
        mod.main()
