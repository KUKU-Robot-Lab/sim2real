"""★10.21 손 안 컵 추정기(inhand_est) — rcg_04 정책 폴더에 두면 rh_aglt 노드가 기본으로 켠다.

사용자: "추정기들 붙여야 s2r 될것 같은 디폴트 세팅으로 해둬야함". learn:grasp/lift(hdgp b23642a4 · 16dd2aa09):
GRU(과제 관측판) = 끊긴 뒤 정책이 보는 컵(손바닥 FK ⊗ 추정) · grasp_state = 들기 성공 끝 넘김 묶음.
기준 구현 = server our_source/rh_aglt_runs/source_cup/gru_loop_hook.py perceive() · grasp_handover_hook.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from policy_control import approach_grasp as AG
from policy_control import inhand_est as IE
from policy_control import rh_aglt as A
from policy_control.cup_dropout import CupDropout, DropoutCfg, DropoutStop
from policy_control.pour_fj_node import AgltChain, PourFjNode

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies" / "rh56f1" / "aglt"
CON = {k: POL / k / "rh_aglt_contract.json" for k in ("right_approach_ap05", "right_rcg04", "left_rcg04", "right_rcg03")}
URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
Q_MID = np.array([-0.6, 0.2026, 0.6538, 1.7608, 0.3791, 0.2, 0.6646])   # test_pc_approach_handoff 와 같은 자세
OBJ = "source240_real"


def _spec_or_skip(side: str) -> IE.InhandSpec:
    spec = IE.find_spec(CON[f"{side}_rcg04"])
    assert spec is not None
    for f in (spec.gru, spec.grasp_state):
        if not (spec.root / f).is_file():
            pytest.skip(f"추정기 ckpt {spec.root / f} 없음(git 밖 — inhand.yaml source 에서 받는다)")
    return spec


@pytest.fixture(scope="module")
def rcg04() -> A.RaContract:
    return A.load_contract(CON["right_rcg04"])


@pytest.fixture(scope="module")
def ap05() -> A.RaContract:
    return A.load_contract(CON["right_approach_ap05"])


@pytest.fixture(scope="module")
def run(rcg04) -> IE.InhandRun:
    return IE.InhandRun(_spec_or_skip("right"), rcg04)


@pytest.fixture(scope="module")
def fk(ap05):
    if not URDF.is_file():
        pytest.skip("RH56F1 URDF 없음")
    from policy_control.fk_numpy import UrdfChainFK
    s = ap05.side()
    return UrdfChainFK(URDF, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)


# ---------------------------------------------------------------- 설정 · 파일
def test_only_the_rcg04_folders_carry_an_estimator():
    for side, t in (("right", "r"), ("left", "l")):
        spec = IE.find_spec(CON[f"{side}_rcg04"])
        assert spec is not None and spec.side == t and spec.drop_m == pytest.approx(0.10)
        assert 0.065 <= spec.drop_m <= 0.16                      # sim 학습 끊김 거리 U(0.065, 0.16) 안
        assert set(spec.module_md5) >= {"geometry.py", "latch.py", "runtime.py", "grasp_state.py", "model.py"}
    assert IE.find_spec(CON["right_approach_ap05"]) is None     # 접근 정책 폴더에는 없다 — 경로 D 는 넘겨받는 rcg 폴더에서 찾는다
    assert IE.find_spec(CON["right_rcg03"]) is None             # 경로 E(rcg03)는 추정기가 학습되지 않았다 — 꺼진 채
    assert IE.find_spec("", CON["right_rcg04"]) is not None      # 빈 경로는 건너뛴다


def test_the_estimator_files_are_ignored_by_git():
    import subprocess
    p = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(POL / "right_rcg04" / "inhand" / "gru_rcg04_obs_r.pt")])
    assert p.returncode == 0
    p = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(POL / "right_rcg04" / "inhand" / "inhand.yaml")])
    assert p.returncode == 1                                    # 설정(md5 핀)은 추적한다


def test_a_changed_ckpt_is_refused(tmp_path, rcg04):
    spec = _spec_or_skip("right")
    d = tmp_path / "inhand"
    shutil.copytree(spec.root, d)
    raw = yaml.safe_load((d / IE.INHAND_YAML).read_text(encoding="utf-8"))
    raw["gru"]["md5"] = "0" * 32
    (d / IE.INHAND_YAML).write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(IE.InhandError, match="md5"):
        IE.InhandRun(IE.load_spec(d / IE.INHAND_YAML), rcg04)


def test_a_wrong_side_is_refused(rcg04):
    with pytest.raises(IE.InhandError, match="side"):
        IE.InhandRun(_spec_or_skip("left"), rcg04)


def test_a_bad_spec_is_refused(tmp_path):
    p = tmp_path / "inhand.yaml"
    p.write_text("side: r\ndrop_m: 0.5\n", encoding="utf-8")
    with pytest.raises(IE.InhandError):
        IE.load_spec(p)


def test_the_pinned_code_loads_and_the_gru_layout_is_the_deploy_actor_layout(run, rcg04):
    assert run.width == rcg04.obs_dim == 94
    assert run.gru.prior_input and not run.gru.proprio
    assert [tuple(x) for x in run.gru.layout] == [tuple(x) for x in A.actor_layout(rcg04)]
    assert not any(f in run.gru.ck["fields"] for f in ("cup_rel_palm", "cup_up", "tips_rel_cup", "kp_rel_goal"))   # 컵 칸은 뺀다
    assert run.names == {"source240_real": "source240_cvx", "source200_real": "source200_cvx", "shaker_c_real": "shaker_c_cvx"}
    assert IE.PKG in run.pkg.__name__ and ".cache/s2r_hdgp" in run.pkg.__file__   # 공유 작업 트리가 아니라 고정 커밋 사본


# ---------------------------------------------------------------- 래치 · 관측
def _meas(c, palm_pos, cup_pos, hand=None):
    s = c.side()
    return A.RaMeas(arm_q=Q_MID.copy(), arm_qd=np.zeros(7), hand_q=dict(zip(s.hand_joints, hand if hand is not None else s.hand_open)),
                    palm_pos=np.asarray(palm_pos, float), palm_R=np.eye(3), tips=np.zeros((5, 3)),
                    cup_pos=np.asarray(cup_pos, float), cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5))


def test_far_cups_pass_through_and_the_first_near_tick_latches(run, rcg04):
    run.reset(OBJ)
    q = np.asarray(rcg04.side().hand_open, float)
    m = _meas(rcg04, [0.3, 0.0, 0.2], [0.3, -0.2, 0.2])
    out = run.perceive(m, q, q)
    assert out is m and not run.dropped and run.k == 1           # 끊기기 전 = 지각 그대로
    run.remember(np.zeros(94))
    m = _meas(rcg04, [0.3, 0.0, 0.2], [0.3, -0.09, 0.2])         # 9 cm < 0.10 → 이 틱에 끊는다
    out = run.perceive(m, q, q)
    assert run.dropped and run.drop_tick == 1
    assert np.allclose(run.latch.prior_p[0].numpy(), [0.0, -0.09, 0.0], atol=1e-6)   # 그 틱의 손바닥 기준 지각 자세
    assert np.allclose(out.cup_pos, m.palm_pos + run.est_p, atol=1e-6)               # 컵 = 손바닥 FK ⊗ GRU(손바닥 R = I)
    m2 = _meas(rcg04, [0.3, 0.0, 0.25], [0.3, -0.5, 0.2])       # 끊긴 뒤 지각이 튀어도 정책은 보지 않는다
    out2 = run.perceive(m2, q, q)
    assert run.dropped and np.allclose(out2.cup_pos, m2.palm_pos + run.est_p, atol=1e-6)
    assert np.allclose(run.latch.prior_p[0].numpy(), [0.0, -0.09, 0.0], atol=1e-6)   # 사전값은 끊긴 순간 그대로


def test_reset_clears_the_latch_and_the_first_tick_is_not_live(run, rcg04):
    q = np.asarray(rcg04.side().hand_open, float)
    run.reset(OBJ)
    assert not run.dropped and run.k == 0 and run.prev_obs is None and run.bundle() is None
    run.perceive(_meas(rcg04, [0.3, 0.0, 0.2], [0.3, -0.05, 0.2]), q, q)
    assert not run.dropped                                       # 리셋 뒤 첫 틱(live False)은 끊지 않는다(훅 episode_length_buf > 0)
    run.perceive(_meas(rcg04, [0.3, 0.0, 0.2], [0.3, -0.05, 0.2]), q, q)
    assert run.dropped


def test_unknown_objects_are_refused(run):
    with pytest.raises(IE.InhandError):
        run.reset("cyl60")


def test_the_bundle_is_written_after_the_drop_and_reads_back(run, rcg04, tmp_path):
    q = np.asarray(rcg04.side().hand_open, float)
    run.reset(OBJ)
    assert run.save_bundle(None, {"node": "n"}, tmp_path) is None   # 아무것도 못 봤으면 묶음 없음
    for k in range(40):
        run.perceive(_meas(rcg04, [0.3, 0.0, 0.2 + 0.001 * k], [0.3, -0.06, 0.2]), q, q + 0.01 * k)
        run.remember(np.zeros(94))
    path = run.save_bundle(None, {"node": "rh_aglt_node_right", "episode": 3}, tmp_path)
    assert path is not None and path.endswith("_rh_aglt_node_right_3.npz")
    b = run.pkg.grasp_state.load_bundle(path)
    assert b["format"] == "grasp_handover/1" and b["side"] == "r" and b["objects"] == ["source240_cvx"]
    assert bool(b["valid"][0]) and int(b["n_steps"][0]) == run.gs.window      # 끊긴 뒤 39 틱 ≥ 끝 창 30
    assert np.allclose(b["hand_q"][0].numpy(), q + 0.39, atol=1e-6)            # 마지막 perceive 의 손 관절
    assert b["meta"]["gru_md5"] == run.spec.gru_md5 and b["meta"]["drop_tick"] == 1
    pw, _ = run.pkg.grasp_state.object_world(run.torch.zeros(1, 3), run.torch.tensor([[1.0, 0, 0, 0]]), b)
    assert np.allclose(pw[0].numpy(), b["rel_p"][0].numpy())                   # 받는 쪽: 손바닥 FK ⊗ 묶음


# ---------------------------------------------------------------- 체인(경로 F: ap05 → rcg04)
class Toward:
    def __init__(self):
        self.ch = None

    def forward(self, obs):
        c, st = self.ch.c, self.ch.dec.state["arm"]
        a_arm = np.clip((Q_MID - st.arm_target) / np.asarray(c.k_arm, float), -1.0, 1.0)
        return np.concatenate([a_arm, AG.hand_action_for(self.ch.dec.law, np.asarray(c.side().hand_open, float))])

    def reset(self):
        pass


class Grasp:
    def __init__(self):
        self.obs = []

    def forward(self, obs):
        self.obs.append(np.asarray(obs, float).copy())
        return np.concatenate([np.zeros(7), np.ones(6) * 0.5])

    def reset(self):
        pass


class Plant:
    def __init__(self, c, fk):
        s = c.side()
        self.fk, self.names = fk, list(s.hand_joints)
        self.hand = np.asarray(s.hand_open, float).copy()
        self.arm = Q_MID.copy()
        r = float(c.object_geometry[OBJ]["radius"]) * float(c.ready_r_mult[0])
        self.cup = self.fk.palm_pose(self.arm, self.hand).palm_pos - np.array([r, 0.0, 0.0])
        self.arm = np.asarray(s.arm_home, float).copy()
        self.arm_prev = self.arm.copy()

    def meas(self):
        p = self.fk.palm_pose(self.arm, self.hand)
        m = A.RaMeas(arm_q=self.arm.copy(), arm_qd=(self.arm - self.arm_prev) * 60.0, hand_q=dict(zip(self.names, self.hand)),
                     palm_pos=p.palm_pos, palm_R=p.extra["palm_rot"], tips=p.tips, cup_pos=self.cup.copy(),
                     cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5))
        return {"arm": m}

    def apply(self, targets):
        arm, hand = targets["arm"]
        self.arm_prev, self.arm = self.arm.copy(), np.asarray(arm, float).copy()
        self.hand = np.asarray(hand, float).copy()


def _cup_obs(c, obs):
    i = 0
    for name, w in A.actor_layout(c):
        if name == "cup_rel_palm":
            return np.asarray(obs[i:i + w])
        i += w
    raise AssertionError


def test_route_f_runs_the_estimator_through_the_handover(ap05, rcg04, run, fk):
    nxt = Grasp()
    pol = Toward()
    ch = AgltChain(ap05, pol)
    pol.ch = ch
    ch.object_name = OBJ
    ch.ag = AG.ApproachGrasp(AG.ApproachGraspCfg(handoff=1), ap05.policy_hz)
    ch.ag.next = (rcg04, nxt)
    ch.inhand = run
    plant = Plant(A.select_object(ap05, OBJ), fk)
    ch.reset(plant.meas())
    assert run.k == 0 and run.object == OBJ
    seen = []
    for _ in range(60 * 10):
        m = plant.meas()["arm"]
        obs, _, tg = ch.step({"arm": m})
        seen.append((run.dropped, np.linalg.norm(m.cup_pos - m.palm_pos), _cup_obs(ap05, obs), m))
        plant.apply(tg)
        if ch.ag.phase == "policy":
            break
    assert ch.ag.phase == "policy" and run.dropped               # 사전 파지 고리(7.5 cm) < 끊김 0.10 m — 넘길 때는 늘 끊긴 상태
    assert np.allclose(ch.goals.cup_reset, plant.cup)            # 넘김 판정 · 목표는 측정 컵(sim 참값 컵) — 추정 컵이 아니다
    first = next(i for i, s in enumerate(seen) if s[0])
    assert seen[first - 1][1] >= run.drop_m > seen[first][1]     # 처음 0.10 m 안으로 든 틱
    for dropped, _, cup_obs, m in seen[:first]:                  # 그 전 = FP++ 그대로
        assert np.allclose(cup_obs, np.clip(m.cup_pos - m.palm_pos, -5, 5), atol=1e-9)
    k0 = run.k
    for _ in range(20):                                          # 넘겨받은 rcg 도 같은 GRU 상태로 잇는다(리셋 없음)
        obs, _, tg = ch.step(plant.meas())
        plant.apply(tg)
    assert run.k == k0 + 20 and len(nxt.obs) == 20
    assert np.allclose(run.prev_obs, nxt.obs[-1])                # 다음 틱 GRU 입력 = 이번 행동을 낸 관측
    m = plant.meas()["arm"]
    assert not np.allclose(_cup_obs(rcg04, nxt.obs[-1]), m.cup_pos - m.palm_pos, atol=1e-6)   # 정책은 추정 컵을 본다


# ---------------------------------------------------------------- 끊김 규칙과 같이
def _dropout():
    return CupDropout(DropoutCfg(stop_s=1.0, near_m=0.2, near_max_s=2.5, follow_s=0.5))


def test_the_near_hand_dropout_stop_waits_for_the_estimator():
    cup = (np.array([0.3, -0.08, 0.2]), np.array([1.0, 0, 0, 0]), 10.0)
    got = (0.0, cup[0], cup[1], 10.0)
    d = _dropout()
    d.step(0.0, got, False, 10.0, np.array([0.3, 0.0, 0.2]), np.eye(3), cup)
    with pytest.raises(DropoutStop):                              # 추정기 없이: 손 근처 2.5 s 넘게 끊기면 멈춤(10.19)
        d.step(3.0, got, False, 13.0, np.array([0.3, 0.0, 0.2]), np.eye(3), cup)
    d = _dropout()
    d.step(0.0, got, False, 10.0, np.array([0.3, 0.0, 0.2]), np.eye(3), cup)
    out = d.step(3.0, got, False, 13.0, np.array([0.3, 0.0, 0.2]), np.eye(3), cup, covered=True)
    assert out is cup and d.mode == "est" and d.stop_reason is None


def test_a_released_cup_still_stops_with_the_estimator():
    cup = (np.array([0.3, -0.04, 0.2]), np.array([1.0, 0, 0, 0]), 10.0)
    got = (0.0, cup[0], cup[1], 10.0)
    d = _dropout()
    t = 0.0
    for i in range(35):                                           # 쥔 상태(15 틱) → 풀림(15 틱) — 그동안 FP++ 는 새 프레임
        got = (t, cup[0], cup[1], 10.0 + t)
        d.step(t, got, i < 20, 10.0 + t, np.array([0.3, 0.0, 0.2]), np.eye(3), cup, covered=True)
        t += 1 / 60
    assert d.released and d.stop_reason is None
    with pytest.raises(DropoutStop, match="쥠이 풀린 뒤"):
        d.step(t + 1.0, got, False, 11.0 + t, np.array([0.3, 0.0, 0.2]), np.eye(3), cup, covered=True)


# ---------------------------------------------------------------- 노드 기본값(auto)
class _Node:
    """PourFjNode._inhand_run 만 부르는 대역 — 파라미터 · 로그."""

    def __init__(self, contract_path, contract, inhand_est="auto", handoff_path="", drop=0.0):
        self.contract = contract
        self.params = {"inhand_est": inhand_est, "contract": str(contract_path), "handoff_contract": str(handoff_path),
                       "inhand_drop_m": drop}
        self.logs = []

    def get_parameter(self, n):
        return SimpleNamespace(value=self.params[n])

    def get_logger(self):
        return SimpleNamespace(info=self.logs.append, warning=self.logs.append)


def test_route_f_turns_the_estimator_on_by_default(ap05, rcg04):
    _spec_or_skip("right")
    node = _Node(CON["right_approach_ap05"], ap05, handoff_path=CON["right_rcg04"])
    run = PourFjNode._inhand_run(node, rcg04)
    assert run is not None and run.spec.root.parent == CON["right_rcg04"].parent.resolve() and run.drop_m == pytest.approx(0.10)
    assert PourFjNode._inhand_run(node, rcg04) is run            # 한 번만 읽는다
    assert any("손 안 컵 추정기 켬" in s for s in node.logs)


def test_route_e_and_off_leave_it_off(ap05, rcg04):
    rcg03 = A.load_contract(CON["right_rcg03"])
    assert PourFjNode._inhand_run(_Node(CON["right_approach_ap05"], ap05, handoff_path=CON["right_rcg03"]), rcg03) is None
    assert PourFjNode._inhand_run(_Node(CON["right_approach_ap05"], ap05), None) is None   # 경로 C · 접근 단독
    off = _Node(CON["right_approach_ap05"], ap05, inhand_est="off", handoff_path=CON["right_rcg04"])
    assert PourFjNode._inhand_run(off, rcg04) is None


def test_the_drop_distance_can_be_overridden_by_the_node(ap05, rcg04):
    _spec_or_skip("right")
    node = _Node(CON["right_approach_ap05"], ap05, handoff_path=CON["right_rcg04"], drop=0.08)
    assert PourFjNode._inhand_run(node, rcg04).drop_m == pytest.approx(0.08)
