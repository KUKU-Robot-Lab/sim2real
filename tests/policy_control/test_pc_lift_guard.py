"""★11.08 들기 가드 G40H8(cocktail_1c_e4000 grasp_blue) — job 파라미터 범위 · 노드 멈춤 조건 · 가드 정지 뒤 들기 판정."""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from policy_control import episode_ros as R
from policy_control import episode_spec as E
from policy_control import lift_guard as LG
from policy_control import pour_fj_node as N
from policy_control.contract_assets import ASSETS
from policy_control.tick_budget import TickBudget
from policy_control import episode_failure as EF
from policy_control.episode_runner import EpisodeManager, PolicyPlan, _lift_guard
from policy_control.fk_numpy import UrdfChainFK

REPO = Path(__file__).resolve().parents[2]
EPS = REPO / "config" / "episodes"
RCG04 = REPO / "deploy/policies/rh56f1/aglt/right_rcg04/rh_aglt_contract.json"
PINK = [0.3048, -0.3394, 0.2899]


def _aglt_params(ep_file: str) -> dict:
    ep = E.load(EPS / ep_file)
    world = NS(objects={"SRC_PINK": {"pose": [PINK, [1, 0, 0, 0]], "at": "table"}})
    out = {}
    for n in ep.nodes:
        for j in n.jobs:
            b = ep.policies[j.role]
            if b.kind == "aglt":
                p = PolicyPlan(role=j.role, kind="aglt", side=b.side, policy=b.policy, target_object=j.target_object,
                               source_object=None, target_holder=None, holder_id=None, setting=None, object_pose=None,
                               policy_object=j.policy_object, lift_guard=_lift_guard(j, world))
                out[n.id] = {k: v for k, v in R.job_params(ep, p).items() if k.startswith("lift_")}
    return out


def test_every_right_grasp_job_sends_the_guard_and_the_off_jobs_send_zero_and_nan():
    """★11.08 리뷰 1(a) — 오른손 잡기 job 은 에피소드와 상관없이 늘 4개를 보낸다(끄는 job = 0 · nan). 왼손 · 놓기는 안 보낸다."""
    p = _aglt_params("cocktail_1c_e4000.yaml")
    assert p["grasp_blue"] == {"lift_guard_mm": 40.0, "lift_hstop_m": 0.08, "lift_guard_x": PINK[0], "lift_guard_y": PINK[1]}
    assert p["grasp_shaker"] == {}                                # 왼손
    for ep_file in ("cocktail_1c_e4000.yaml", "cocktail_1c_e1000.yaml"):
        for nid, v in _aglt_params(ep_file).items():
            if nid == "grasp_blue" and ep_file == "cocktail_1c_e4000.yaml" or not v:
                continue
            assert v["lift_guard_mm"] == 0.0 and v["lift_hstop_m"] == 0.0, (ep_file, nid)
            assert np.isnan(v["lift_guard_x"]) and np.isnan(v["lift_guard_y"]), (ep_file, nid)
    assert "grasp_pink" in _aglt_params("cocktail_1c_e1000.yaml") and _aglt_params("cocktail_1c_e1000.yaml")["grasp_pink"]


class _Params:
    """노드 파라미터 저장소(get_parameter · set_parameters) — rclpy 없이 _take_lift_guard 를 돌린다."""

    def __init__(self, **kw):
        self.v = {n: d for n, d in N.LG_PARAMS_OFF}
        self.v.update(kw)

    def get_parameter(self, n):
        return NS(value=self.v[n])

    def set_parameters(self, ps):
        for x in ps:
            self.v[x.name] = x.value
        return [NS(successful=True, reason="")] * len(ps)


def test_the_node_uses_the_guard_values_once_and_a_manual_reset_gets_the_guard_off():
    """★11.08 리뷰 1(b) — reset 이 읽자마자 끔으로 되돌린다. 다음(콘솔 수동) reset 은 끈 값을 읽는다."""
    store = _Params(lift_guard_mm=40.0, lift_hstop_m=0.08, lift_guard_x=PINK[0], lift_guard_y=PINK[1])
    me = NS(get_parameter=store.get_parameter, set_parameters=store.set_parameters)
    assert N.PourFjNode._take_lift_guard(me) == (40.0, 0.08, PINK[0], PINK[1])
    assert store.v["lift_guard_mm"] == 0.0 and np.isnan(store.v["lift_guard_x"])
    g, h, x, y = N.PourFjNode._take_lift_guard(me)
    assert g == 0.0 and h == 0.0 and np.isnan(x) and np.isnan(y) and not me._lg.on
    bad = NS(get_parameter=store.get_parameter, set_parameters=lambda ps: [NS(successful=False, reason="no")] * len(ps))
    with pytest.raises(N.PourFjNodeError):
        N.PourFjNode._take_lift_guard(bad)


def test_a_chain_without_the_guard_hooks_or_a_latched_cup_with_hstop_refuses_the_reset():
    """★11.08 리뷰 2 · 10 — 가드를 요청했는데 체인이 못 받거나(lift_grasp_palm_m 없음) cup_latch + 높이 정지면 거부."""
    on = (0.0, 0.08, PINK[0], PINK[1])
    me = NS(chain=NS(), _lg_clear=None, get_logger=lambda: NS(info=lambda *_: None), cup_latch=False)
    with pytest.raises(N.PourFjNodeError, match="lift_grasp_palm_m"):
        N.PourFjNode._reset_lift_guard(me, on)
    me.chain, me.cup_latch = NS(lift_grasp_palm_m=0.08), True
    with pytest.raises(N.PourFjNodeError, match="cup_latch"):
        N.PourFjNode._reset_lift_guard(me, on)
    me.cup_latch = False
    N.PourFjNode._reset_lift_guard(me, on)
    assert me._lg.on and me._lg.hstop_m == 0.08
    N.PourFjNode._reset_lift_guard(NS(chain=NS(), _lg_clear=None), (0.0, 0.0, float("nan"), float("nan")))   # 끔은 어느 체인이든


def test_the_student_chain_takes_the_guard():
    """★11.08 리뷰가 엇갈린 것 — 학생 체인(StudentChain, 학생 노드 체인)은 AgltChain 을 이어받아 lift_grasp_palm_m · goals 를
    가진다 → _reset_lift_guard 가 받아들이고 거부하지 않는다."""
    from policy_control import rh_aglt as A
    from policy_control.student_chain import StudentChain
    d = REPO / "deploy/policies/rh56f1/aglt/right_student1104e"
    c = A.load_contract(REPO / "deploy/policies/rh56f1/aglt/right_approach_ap05/rh_aglt_contract.json")
    cfg = {**yaml.safe_load((d / "student.yaml").read_text()), "dir": d}
    ch = StudentChain(c, NS(policy_cfg=cfg, handover=dict(tmin_s=0.5, still_s=0.3, still_mps=0.01)))
    me = NS(chain=ch, _lg_clear=None, get_logger=lambda: NS(info=lambda *_: None), cup_latch=False)
    N.PourFjNode._reset_lift_guard(me, (0.0, 0.08, PINK[0], PINK[1]))
    assert me._lg.on and hasattr(ch, "goals")


def _runner_status(status: dict):
    me = NS(_lock=threading.Lock(), cancel=threading.Event(), _status=status)
    return me, []


def test_the_runner_refuses_before_start_unless_the_node_status_shows_the_guard_on(monkeypatch):
    """★11.08 리뷰 2 — reset 뒤 status lift_guard 가 on · 자리 1 mm 안이어야 start. 아니면 refused(start 를 부르지 않는다)."""
    monkeypatch.setattr(R, "LG_CHECK_WAIT_S", 0.2)
    want = (40.0, 0.08, PINK[0], PINK[1])
    key = "rh_aglt_node_right"
    t = time.monotonic()
    good = {"on": True, "guard_mm": 40.0, "hstop_m": 0.08, "x": PINK[0], "y": PINK[1]}
    for st, ok in (({"phase": "armed", "lift_guard": good}, True),
                   ({"phase": "armed", "lift_guard": {**good, "on": False}}, False),
                   ({"phase": "armed", "lift_guard": {**good, "x": PINK[0] + 0.002}}, False),
                   ({"phase": "armed", "lift_guard": {**good, "x": None}}, False),
                   ({"phase": "idle", "lift_guard": good}, False),        # reset 전 status
                   ({"phase": "armed"}, False)):                          # 가드를 모르는 노드
        me, _ = _runner_status({key: {**st, "_t": t + 1.0}})
        assert (R.RosExecutor._lift_guard_on(me, key, want, t) == "") == ok, st
    ep = E.load(EPS / "cocktail_1c_e4000.yaml")
    node = next(n for n in ep.nodes if n.id == "grasp_blue")
    world = NS(objects={"SRC_PINK": {"pose": [PINK, [1, 0, 0, 0]], "at": "table"}})
    (plan,) = EpisodeManager._plans(NS(ep=ep), node, world)
    assert plan.lift_guard == want
    calls = []
    me = NS(ep=ep, _wait_node_ready=lambda key: None, _lock=threading.Lock(), _events={}, cancel=threading.Event(),
            _tin_json={}, _tin_why={}, _status={key: {"phase": "armed", "lift_guard": {**good, "on": False}, "_t": 0.0}},
            _set_params=lambda node, params: calls.append(("set", node)) or (True, []),
            _call=lambda path: calls.append(("call", path)) or (True, []))
    me._lift_guard_on = lambda *a: R.RosExecutor._lift_guard_on(me, *a)
    r = R.RosExecutor._policy(me, plan)
    assert r.status == "refused" and r.reason.startswith("lift_guard:")
    assert [c for c in calls if c[0] == "call"] == [("call", "/policy_control/right/episode/reset")]   # start 는 부르지 않는다


Q0 = np.array([-0.5814, 0.2908, -0.0098, 1.6084, 0.1172, 0.1432, 0.6816])


def _node(cfg: LG.GuardCfg, lift_dz: float, held: bool):
    c = json.loads(RCG04.read_text())["sides"]["arm"]
    urdf = ASSETS["openarm_rh56f1_bi_rl"].urdf
    fk = UrdfChainFK(urdf, c["arm_joints"], c["hand_joints"], c["palm_body"], c["tip_bodies"])
    n = NS(_lg=cfg, _lg_clear=LG.Clearance(fk, urdf), _lg_min=None, _lg_stop=None, _lg_hcount=0, fk={"arm": fk},
           chain=NS(goals=NS(lift_dz=lift_dz, held=held)), cup_fresh={"arm": True})
    n._lg_palm = lambda meas: N.PourFjNode._lg_palm(n, meas)
    m = NS(arm_q=Q0.copy(), hand_q={j: 0.0 for j in c["hand_joints"]}, palm_pos=np.array([0.3, -0.1, 0.4]))
    return n, {"arm": m}, (list(c["arm_joints"]), Q0.copy())


def _tick(n, meas, cmd, q=None):
    return N.PourFjNode._lift_guard_tick(n, meas, cmd[0], cmd[1] if q is None else q)["lift_guard"]


def test_the_node_stops_on_the_smaller_of_measured_and_commanded_clearance():
    n, meas, cmd = _node(LG.GuardCfg(0.04, 0.0, *PINK[:2]), 0.02, True)
    g = _tick(n, meas, cmd)
    assert n._lg_stop is None and g["clear_mm"] > 40 and g["clear_mm"] == g["clear_meas_mm"] == g["clear_cmd_mm"]
    assert g["palm"] == [0.3, -0.1, 0.4] and g["lg_ms"] > 0
    n._lg = LG.GuardCfg(g["clear_mm"] / 1e3 + 0.001, 0.0, *PINK[:2])     # 지금 틈보다 1 mm 큰 가드 → 이 틱에 멈춤
    _tick(n, meas, cmd)
    assert n._lg_stop.startswith("guard: clear")
    # 명령 관절만 분홍 쪽으로(어깨 yaw 를 돌림) — 실측은 멀어도 명령 쪽 틈으로 멈춘다(bag: 멈춘 뒤 33 mm 까지 더 감)
    n._lg = LG.GuardCfg(0.04, 0.0, *PINK[:2])
    far = g["clear_mm"]
    for dq in np.linspace(0.0, 1.5, 31):
        q = Q0.copy(); q[0] += dq
        g2 = _tick(n, meas, cmd, q)
        if g2["clear_cmd_mm"] < g2["clear_meas_mm"] - 5:
            break
    assert g2["clear_mm"] == g2["clear_cmd_mm"] < g2["clear_meas_mm"] == far


def test_a_nonfinite_clearance_stops_at_once_and_a_nonfinite_rise_is_only_shown():
    """★11.08 리뷰 3 — 틈이 비유한(관절 nan)이면 바로 멈춘다. 높이 정지의 dz 가 비유한이면 status 표시만."""
    n, meas, cmd = _node(LG.GuardCfg(0.04, 0.08, *PINK[:2]), float("nan"), True)
    meas["arm"].arm_q[3] = float("nan")
    g = _tick(n, meas, cmd)
    assert n._lg_stop == "guard: nonfinite clearance" and g["rise_nonfinite"] and g["rise_m"] is None
    n, meas, cmd = _node(LG.GuardCfg(0.04, 0.08, *PINK[:2]), float("nan"), True)
    g = _tick(n, meas, cmd)
    assert n._lg_stop is None and g["rise_nonfinite"] and g["hstop_ticks"] == 0
    with pytest.raises(LG.LiftGuardError):                          # 명령에 팔 관절이 없으면 예외 → _on_tick 이 abort
        N.PourFjNode._lift_guard_tick(n, meas, ["x"], [0.0])


def test_the_height_stop_needs_five_ticks_in_a_row_with_grasp():
    """★11.08 리뷰 7 — 실기 FP++ 잡음 한두 프레임으로 멈추지 않는다."""
    n, meas, cmd = _node(LG.GuardCfg(0.0, 0.08, *PINK[:2]), 0.09, True)
    for k in range(1, N.LG_HSTOP_TICKS):
        _tick(n, meas, cmd)
        assert n._lg_stop is None and n._lg_hcount == k
    n.chain.goals.lift_dz = 0.07                                    # 한 틱 끊김 → 처음부터
    _tick(n, meas, cmd)
    n.chain.goals.lift_dz = 0.09
    for _ in range(N.LG_HSTOP_TICKS - 1):
        _tick(n, meas, cmd)
    assert n._lg_stop is None
    _tick(n, meas, cmd)
    assert n._lg_stop.startswith("hstop:")
    n, meas, cmd = _node(LG.GuardCfg(0.0, 0.08, *PINK[:2]), 0.09, False)   # 쥐지 않았으면 몇 틱이든 안 멈춘다
    for _ in range(10):
        _tick(n, meas, cmd)
    assert n._lg_stop is None


def _on_tick(lg_tick):
    """_on_tick 의 가드 부분만 — 측정 · 체인 · 목표는 가짜, 발행 · 끝은 기록."""
    out = {"target": 0, "end": [], "status": []}
    me = NS(_drain=lambda: None, book=NS(phase="running", episode=1), chain=NS(step=lambda meas: (np.zeros(2), np.zeros(1), {}),
            step_i=1), _measure=lambda advance=False: {"arm": None}, cup_fresh={}, contract=NS(hold_steps=0),
            fam=NS(targets=lambda c, t: (["a"], np.zeros(1), np.zeros(1)), errors=(ValueError,)), _gap=0, _seq=0, max_gap=3, _budget=TickBudget(16.7),
            _lg_stop=None, _publish=True, _pub_obs=NS(publish=lambda m: None), _pub_action=NS(publish=lambda m: None),
            _pub_target=NS(publish=lambda m: out.__setitem__("target", out["target"] + 1)),
            _status=lambda ok, reasons, extra=None: out["status"].append((ok, reasons)),
            _end=lambda *a: out["end"].append(a), _after_step=lambda: True, max_s=0, _t_start=0.0)
    me._lift_guard_tick = lambda meas, names, q: lg_tick(me)
    N.PourFjNode._on_tick(me)
    return out


def test_a_guard_tick_error_aborts_and_a_stop_tick_sends_no_new_target():
    """★11.08 리뷰 4 · 5 — 가드 계산 예외 = abort('lift_guard tick: …'). 가드가 멈춘 틱은 joint_target 을 내지 않는다."""
    def boom(me):
        raise RuntimeError("fk 실패")
    out = _on_tick(boom)
    assert out["end"] == [("abort", "lift_guard tick: RuntimeError: fk 실패")] and out["target"] == 0

    def stop(me):
        me._lg_stop = "guard: clear 39.0 mm < 40 mm at r_al_7"
        return {"lift_guard": {}}
    assert _on_tick(stop)["target"] == 0
    assert _on_tick(lambda me: {})["target"] == 1


PALM_NEAR = np.array([0.3, -0.1 + 0.07, 0.29 + 0.06])         # 컵(들린 뒤)에서 7 cm — bag 손 안 실측 0.064~0.084 m


def _hold(rises, dt=1 / 12, palm=PALM_NEAR, repeat=1):
    """FP++ live 프레임을 시간 순으로 흘려 넣고(같은 값 repeat 번) 노드 status 손바닥을 같이 갱신해 _guard_hold 판정을 받는다."""
    live, key = {"SRC_BLUE": []}, "rh_aglt_node_right"
    s = NS(_live=live, _lock=threading.Lock(), cancel=threading.Event(), _status={})
    s._guard_palm = lambda k: R.RosExecutor._guard_palm(s, k)
    p = NS(target_object="SRC_BLUE", object_pose=([0.3, -0.1, 0.29], [1, 0, 0, 0]))

    def feed():
        for i, r in enumerate(rises):
            for _ in range(repeat):
                time.sleep(dt / repeat)
                with s._lock:
                    live["SRC_BLUE"].append(((0.3, -0.1, 0.29 + r + 1e-5 * i), (1, 0, 0, 0), time.monotonic()))
                    if palm is not None:
                        s._status[key] = {"lift_guard": {"palm": list(palm)}, "_t": time.monotonic()}
    threading.Thread(target=feed, daemon=True).start()
    return R.RosExecutor._guard_hold(s, p, key)


def test_the_guard_stop_counts_as_lifted_only_after_a_second_in_hand_above_5_cm(monkeypatch):
    """★11.08 리뷰 8 — 손 안(손바닥 FK 12 cm 안) · 같은 값 반복은 세지 않음 · 0.20 m 넘는 튐은 버림 · 끊김 0.5 s."""
    monkeypatch.setattr(R, "GUARD_HOLD_WAIT_S", 2.5)
    ok, _, sig = _hold([0.06] * 16, repeat=2)                       # 실기처럼 약 12 Hz 수신 · 6 Hz 갱신
    assert ok and sig["guard_hold_s"] >= 1.0
    ok, why, _ = _hold([0.06] * 8 + [0.04] + [0.06] * 8)            # 중간 한 프레임이 5 cm 아래 → 둘 다 1 s 못 됨
    assert not ok and "최장" in why
    ok, _, _ = _hold([0.06] * 8 + [0.5] + [0.06] * 8)               # 튐 한 프레임은 버린다 — 이어진다
    assert ok
    ok, why, _ = _hold([0.06] * 16, palm=PALM_NEAR + [0.0, 0.08, 0.0])   # 손바닥에서 15 cm — 손 밖
    assert not ok and "손바닥" in why
    ok, why, _ = _hold([0.06] * 16, palm=None)                      # 손바닥 모름 = 세지 않는다
    assert not ok
    ok, why, _ = _hold_same([0.06] * 16)                           # 같은 값만 계속(FP++ 가 붙든 값) — 새 프레임 1개
    assert not ok and "반복 15" in why


def _hold_same(rises, dt=1 / 12):
    """값이 한 번도 바뀌지 않는 live(같은 값 반복만)."""
    live, key = {"SRC_BLUE": []}, "rh_aglt_node_right"
    s = NS(_live=live, _lock=threading.Lock(), cancel=threading.Event(), _status={})
    s._guard_palm = lambda k: R.RosExecutor._guard_palm(s, k)
    p = NS(target_object="SRC_BLUE", object_pose=([0.3, -0.1, 0.29], [1, 0, 0, 0]))

    def feed():
        for r in rises:
            time.sleep(dt)
            with s._lock:
                live["SRC_BLUE"].append(((0.3, -0.1, 0.29 + r), (1, 0, 0, 0), time.monotonic()))
                s._status[key] = {"lift_guard": {"palm": list(PALM_NEAR)}, "_t": time.monotonic()}
    threading.Thread(target=feed, daemon=True).start()
    return R.RosExecutor._guard_hold(s, p, key)


def test_a_guard_failure_is_its_own_code_and_always_a_safe_stop():
    """★11.08 리뷰 9 — 가드 판정 실패 · 거부 · 틱 오류는 GRASP_FAILED(되돌아가 손 펴기)가 아니라 lift_guard_failed → safe stop.
    가드 단계는 재시도 0 이어야 한다(spec validate)."""
    ep = E.load(EPS / "cocktail_1c_e4000.yaml")
    node = next(n for n in ep.nodes if n.id == "grasp_blue")
    for reason in (f"{R.GUARD_FAIL_TEXT}: FP++ 새 프레임 3개 (guard: clear 39.0 mm < 40 mm at r_al_7)",
                   "reset: lift_guard: 분홍 컵 자리 x,y 가 없다", "lift_guard tick: RuntimeError: x",
                   "lift_guard: reset 뒤 1 s 안에 노드 status 에서 요청한 가드가 켜진 것을 못 봤다"):
        res = EF.NodeResult("aborted", reason, 1.0, "rh_aglt_r", {"attached": False})
        code = EF.FailureDetector()._one(ep, node, res)
        assert code is EF.F.LIFT_GUARD_FAILED, reason
        retry3 = dataclasses.replace(ep, failure_policy={"default": 3})     # 재시도가 남아도
        assert EF.RecoveryManager().decide(retry3, node, code, 0, None, {}).action == EF.SAFE_STOP
    raw = yaml.safe_load((EPS / "cocktail_1c_e4000.yaml").read_text())
    with pytest.raises(E.EpisodeSpecError, match="재시도 상한 0"):
        E.parse({**raw, "failure_policy": {"default": 1}})
