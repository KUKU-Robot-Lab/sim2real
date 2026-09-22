"""pour_node_core: ROS-free inbox (messages -> PourMeasure) and outbox (PourStep -> joint_target)."""
import dataclasses
import numpy as np
import pytest

from policy_control.fabric_core import JointTarget
from policy_control.pour_chain import PourChain, PourStep, RecordedPolicy
from policy_control.pour_node_core import PourInbox, PourNodeError, fabric_home, joint_target_arrays

from pour_trace_util import contract, make_fk, trace


def _fill(inbox, c, z, meta, t, e, now, skip=()):
    names = meta["joint_names"]
    if "joints" not in skip:
        inbox.put_joints(names, z["joint_pos"][t, e], z["joint_vel"][t, e], now)
    for r in c.roles:
        if r not in skip:
            inbox.put_cup(r, z[f"{r}_cup_pos"][t, e], z[f"{r}_cup_quat"][t, e], now)


def test_measure_lists_everything_missing():
    c = contract()
    inbox = PourInbox(c, stale_sec=0.5)
    with pytest.raises(PourNodeError) as exc:
        inbox.measure(now=0.0)
    text = str(exc.value)
    assert "cup:src" in text and "cup:rcv" in text and c.side("rcv").arm_joints[0] in text


def test_measure_requires_the_rcv_side():
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0, skip=("rcv",))
    with pytest.raises(PourNodeError, match="cup:rcv"):
        inbox.measure(now=1.0)


def test_measure_flags_stale_inputs():
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0)
    inbox.measure(now=1.4)
    with pytest.raises(PourNodeError, match="stale"):
        inbox.measure(now=1.6)


def test_partial_joint_messages_merge_by_name():
    c = contract()
    z, meta = trace()
    names = list(meta["joint_names"])
    inbox = PourInbox(c, stale_sec=0.5)
    half = len(names) // 2
    for sl in (slice(0, half), slice(half, None)):
        inbox.put_joints(names[sl], z["joint_pos"][0, 0][sl], z["joint_vel"][0, 0][sl], 1.0)
    for r in c.roles:
        inbox.put_cup(r, z[f"{r}_cup_pos"][0, 0], z[f"{r}_cup_quat"][0, 0], 1.0)
    m = inbox.measure(now=1.0)
    for s in c.sides:
        for n in s.arm_joints + s.hand_joints:
            assert m.joint_pos[n] == pytest.approx(z["joint_pos"][0, 0][names.index(n)])
    assert m.fill_level is None and m.forces is None


def test_rejects_non_finite_and_bad_shapes():
    c = contract()
    inbox = PourInbox(c, stale_sec=0.5)
    with pytest.raises(PourNodeError):
        inbox.put_joints(["a"], [np.nan], [0.0], 0.0)
    with pytest.raises(PourNodeError):
        inbox.put_joints(["a", "b"], [0.0], [0.0], 0.0)
    with pytest.raises(PourNodeError):
        inbox.put_joints(["a"], [0.0], None, 0.0)
    with pytest.raises(PourNodeError):
        inbox.put_cup("src", [0, 0], [1, 0, 0, 0], 0.0)
    with pytest.raises(PourNodeError):
        inbox.put_cup("nope", [0, 0, 0], [1, 0, 0, 0], 0.0)
    with pytest.raises(PourNodeError):
        inbox.put_fill(1.5, 0.0)


def test_fill_level_is_latched_for_the_episode():
    """hdgp measures fill_level once at hold end and keeps it for the episode (pour_fabric_env.py:520-524),
    so an operator value published once must not go stale."""
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    inbox.put_fill(0.4, 1.0)
    _fill(inbox, c, z, meta, 0, 0, now=60.0)
    assert inbox.measure(now=60.0).fill_level == pytest.approx(0.4)


def test_fill_and_forces_pass_through():
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0)
    inbox.put_fill(0.4, 1.0)
    n = len(c.side("src").fingers)
    for r in c.roles:
        inbox.put_forces(r, np.ones(n), 2 * np.ones(n), 1.0)
    m = inbox.measure(now=1.0)
    assert m.fill_level == pytest.approx(0.4)
    assert np.allclose(m.forces["rcv"][1], 2.0)


def test_fabric_home_is_in_fabric_joint_order():
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0)
    home = fabric_home(c, inbox.measure(1.0))
    names = list(meta["joint_names"])
    for s in c.sides:
        want = [z["joint_pos"][0, 0][names.index(n)] for n in s.fabric_joint_order]
        assert np.allclose(home[s.role], want)


class _Fabric:
    def reset(self, home):
        self.home = home

    def step(self, palm, hand, hold=False):
        return {r: JointTarget(q_arm=np.full(7, 0.1 if r == "src" else 0.2), qd_arm=np.full(7, 0.01),
                               q_full=np.zeros(27), substeps=np.zeros((1, 27))) for r in palm}


def test_joint_target_has_both_sides_arm_from_fabric_hand_from_decoder():
    c = contract()
    z, meta = trace()
    fks = {s.role: make_fk(s) for s in c.sides}
    chain = PourChain(c, RecordedPolicy(z["actions"][:, 0]), fks, fabric=_Fabric())
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0)
    inbox.put_fill(1.0, 1.0)
    m = inbox.measure(1.0)
    chain.reset(home=fabric_home(c, m))
    step = chain.step(m)
    names, q, qd = joint_target_arrays(c, step)
    assert len(names) == len(set(names)) == 2 * (7 + len(c.side("src").hand_joints))
    for s in c.sides:
        ia = [names.index(n) for n in s.arm_joints]
        ih = [names.index(n) for n in s.hand_joints]
        assert np.allclose(q[ia], 0.1 if s.role == "src" else 0.2) and np.allclose(qd[ia], 0.01)
        assert np.allclose(q[ih], step.targets[s.role].hand_target) and np.allclose(qd[ih], 0.0)


def test_joint_target_needs_a_fabric():
    c = contract()
    step = PourStep(np.zeros(1), np.zeros(1), True, {}, {}, {}, None)
    with pytest.raises(PourNodeError, match="fabric"):
        joint_target_arrays(c, step)


def test_contract_arm_reset_is_the_trace_reset_pose():
    """hdgp side_rig.py:73-75 resets the arm to profile.arm_reset_joint_pos; trace row 0 is that pose."""
    c = contract()
    z, meta = trace()
    names = list(meta["joint_names"])
    for s in c.sides:
        assert len(s.arm_reset) == len(s.arm_joints)
        row0 = [float(z["joint_pos"][0, 0][names.index(j)]) for j in s.arm_joints]
        assert np.allclose(s.arm_reset, row0, atol=2e-3)


def test_tip_forces_map_to_contract_finger_order_with_zero_mid():
    from policy_control.pour_node_core import tip_forces_to_inputs
    c = contract()
    s = c.side("src")
    tips = [f"r_hl_{f}_tip" for f in ("pinky", "thumb", "index", "middle", "ring")]
    xyz = np.array([[0, 0, 5.0], [3.0, 4.0, 0], [0, 0, 0], [0, 1.0, 0], [2.0, 0, 0]])
    f_mid, f_dist = tip_forces_to_inputs(s, tips, xyz)
    want = {"pinky": 5.0, "thumb": 5.0, "index": 0.0, "middle": 1.0, "ring": 2.0}
    assert np.allclose(f_dist, [want[f] for f in s.fingers])
    assert np.allclose(f_mid, 0.0) and f_mid.shape == f_dist.shape
    with pytest.raises(PourNodeError):
        tip_forces_to_inputs(s, tips[:4], xyz[:4])


def test_reset_pose_error_is_max_arm_deviation():
    from policy_control.pour_node_core import reset_pose_error
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0)
    m = inbox.measure(now=1.0)
    assert reset_pose_error(c, m) < 2e-3
    s = c.side("rcv")
    bad = dataclasses.replace(m, joint_pos={**m.joint_pos, s.arm_joints[3]: m.joint_pos[s.arm_joints[3]] + 0.3})
    assert reset_pose_error(c, bad) == pytest.approx(0.3, abs=2e-3)


def test_start_refusals_cover_pose_and_missing_fill():
    from policy_control.pour_node_core import start_refusals
    c = contract()
    assert c.fill_level.default is None          # the iter_11 contract has no documented default
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0)
    no_fill = inbox.measure(now=1.0)
    reasons = start_refusals(c, no_fill, tol=0.15)
    assert len(reasons) == 1 and "fill_level" in reasons[0]
    inbox.put_fill(0.6, now=1.0)
    ok = inbox.measure(now=1.0)
    assert start_refusals(c, ok, tol=0.15) == []
    s = c.side("src")
    far = dataclasses.replace(ok, joint_pos={**ok.joint_pos, s.arm_joints[1]: ok.joint_pos[s.arm_joints[1]] + 0.4})
    reasons = start_refusals(c, far, tol=0.15)
    assert len(reasons) == 1 and "reset pose" in reasons[0]


def test_fill_level_cannot_change_while_the_episode_holds_it():
    """Training keeps fill_level constant after the hold; a mid-episode republish must not step the obs."""
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    inbox.put_fill(0.4, 1.0)
    inbox.hold_fill(True)
    with pytest.raises(PourNodeError, match="held"):
        inbox.put_fill(0.9, 2.0)
    _fill(inbox, c, z, meta, 0, 0, now=2.0)
    assert inbox.measure(now=2.0).fill_level == pytest.approx(0.4)
    inbox.hold_fill(False)
    inbox.put_fill(0.9, 3.0)
    assert inbox.measure(now=2.0).fill_level == pytest.approx(0.9)


# --------------------------------------------------------------- inputs (status 의 입력 생사 목록)
def _by_name(rows):
    return {r["name"]: r for r in rows}


def test_inputs_lists_every_input_even_before_anything_arrives():
    c = contract()
    rows = _by_name(PourInbox(c, stale_sec=0.5).inputs(now=0.0))
    want = {f"{r}:{k}" for r in c.roles for k in ("arm", "hand", "cup", "force")} | {"fill"}
    assert set(rows) == want
    for r in c.roles:
        assert rows[f"{r}:arm"]["state"] == rows[f"{r}:hand"]["state"] == rows[f"{r}:cup"]["state"] == "missing"
        assert rows[f"{r}:force"]["state"] == "off"            # 접촉력은 선택 입력이다
    assert rows["fill"]["state"] == "off" and rows["fill"]["age_ms"] is None


def test_inputs_age_is_the_oldest_joint_of_the_group_and_turns_stale():
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0)
    arm0 = c.side("src").arm_joints[0]
    inbox.put_joints([arm0], [0.0], [0.0], 1.3)               # 한 관절만 새로 와도 그룹 나이는 가장 늙은 것
    rows = _by_name(inbox.inputs(now=1.4))
    assert rows["src:arm"] == {"name": "src:arm", "state": "live", "age_ms": pytest.approx(400.0)}
    assert rows["rcv:cup"]["state"] == "live"
    assert _by_name(inbox.inputs(now=1.6))["src:arm"]["state"] == "stale"


def test_inputs_agrees_with_measure_about_what_blocks_the_tick():
    c = contract()
    z, meta = trace()
    inbox = PourInbox(c, stale_sec=0.5)
    _fill(inbox, c, z, meta, 0, 0, now=1.0, skip=("rcv",))
    bad = {r["name"] for r in inbox.inputs(now=1.0) if r["state"] in ("missing", "stale")}
    assert bad == {"rcv:cup"}
    with pytest.raises(PourNodeError, match="cup:rcv"):
        inbox.measure(now=1.0)


def test_inputs_force_for_one_role_only_marks_the_other_missing_and_fill_is_held():
    c = contract()
    inbox = PourInbox(c, stale_sec=0.5)
    n = len(c.side("src").fingers)
    inbox.put_forces("src", np.ones(n), np.ones(n), 1.0)
    inbox.put_fill(0.4, 1.0)
    rows = _by_name(inbox.inputs(now=9.0))
    assert rows["src:force"]["state"] == "stale" and rows["rcv:force"]["state"] == "missing"
    assert rows["fill"]["state"] == "held"                     # 래치 값 — 오래돼도 stale 이 아니다
