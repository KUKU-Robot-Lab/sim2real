"""★10.28 칵테일(SEP cocktail_1 · B cocktail_1b) — 같은 팔이 두 물체를 잡고 두 홀더에 놓는다.

정책 노드는 시작 때 첫 job 의 컵 · 홀더로 뜬다. 실행기가 reset 전에 job 마다 object · cup_topic(aglt) · holder(place)를
넣고, 노드는 reset 마다 다시 읽는다. 이전에는 노드가 cup_topic · holder 를 시작 때 한 번만 읽어서(파라미터 서비스는 받기만
하고 아무것도 바꾸지 않았다) 분홍 잡기가 파랑을 쫓고, 분홍 놓기가 파랑이 든 홀더 1 을 노렸다(10.28 탐색).
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control import episode_spec as S
from policy_control import rh_place_node as RPN
from policy_control.episode_ros import RESET_CUP_STALE_TEXT, RosExecutor, job_params
from policy_control.episode_runner import EpisodeManager
from policy_control.pour_fj_node import PourFjNode, PourFjNodeError

EPISODES = Path(__file__).resolve().parents[2] / "config" / "episodes"
BLUE_LIVE = "/objects/source240b_blue/live_pose"
PINK_LIVE = "/objects/source240p_pink/live_pose"


def _plan(ep, node_id: str):
    node = next(n for n in ep.nodes if n.id == node_id)
    (plan,) = EpisodeManager._plans(SimpleNamespace(ep=ep), node, SimpleNamespace(objects={}))
    return plan


# ---------------------------------------------------------------- 실행기
@pytest.mark.parametrize("name", ["cocktail_1", "cocktail_1b"])
def test_each_job_gets_its_own_cup_and_holder(name):
    ep = S.load(EPISODES / f"{name}.yaml")
    got = {nid: job_params(ep, _plan(ep, nid))
           for nid in ("grasp_blue", "grasp_shaker", "place_blue", "grasp_pink", "place_pink", "place_shaker")}
    assert got == {"grasp_blue": {"object": "source240_real", "cup_topic": BLUE_LIVE},
                   "grasp_shaker": {"object": "shaker_c_real", "cup_topic": "/objects/shaker_c_orange/live_pose"},
                   "place_blue": {"object": "source240_real", "holder": 1, "holder_tol_m": 0.02},
                   "grasp_pink": {"object": "source240_real", "cup_topic": PINK_LIVE},
                   "place_pink": {"object": "source240_real", "holder": 2, "holder_tol_m": 0.07},   # ★10.29 카드 허용
                   "place_shaker": {"object": "shaker_c_real", "holder": 0, "holder_tol_m": 0.02}}


def _runner(ep, set_reply=(True, []), reset_reply=(False, ["reset: 홀더 2: y 가 학습 범위 밖"])):
    calls = []
    me = SimpleNamespace(ep=ep, _wait_node_ready=lambda key: None, _lock=threading.Lock(), _events={},
                         cancel=threading.Event(),
                         _set_params=lambda node, params: calls.append(("set", node, params)) or set_reply,
                         _call=lambda path: calls.append(("call", path)) or reset_reply)
    return me, calls


def test_the_runner_sets_the_job_params_before_reset():
    ep = S.load(EPISODES / "cocktail_1b.yaml")
    me, calls = _runner(ep)
    r = RosExecutor._policy(me, _plan(ep, "place_pink"))
    assert r.status == "refused" and "학습 범위 밖" in r.reason
    assert calls == [("set", "/rh_place_node_right", {"object": "source240_real", "holder": 2, "holder_tol_m": 0.07}),
                     ("call", "/policy_control/right_place/episode/reset")]


def test_a_refused_param_set_stops_before_reset():
    ep = S.load(EPISODES / "cocktail_1b.yaml")
    me, calls = _runner(ep, set_reply=(False, ["/rh_aglt_node_right/set_parameters: 서비스가 없다"]))
    r = RosExecutor._policy(me, _plan(ep, "grasp_pink"))
    assert r.status == "refused" and "서비스가 없다" in r.reason
    assert calls == [("set", "/rh_aglt_node_right", {"object": "source240_real", "cup_topic": PINK_LIVE})]


def test_the_param_set_sends_the_holder_tolerance_as_a_double():
    """★10.29 노드의 holder_tol_m 은 double 로 선언돼 있다. 문자열로 보내면 SetParameters 가 타입을 거부해 놓기 job 이 모두
    reset 전에 막힌다."""
    from rcl_interfaces.msg import ParameterType
    sent = []

    class _Cli:
        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, req):
            sent.append(req)
            done = SimpleNamespace(results=[SimpleNamespace(successful=True)] * len(req.parameters))
            return SimpleNamespace(done=lambda: True, result=lambda: done)

    me = SimpleNamespace(node=SimpleNamespace(create_client=lambda *a, **k: _Cli(), destroy_client=lambda cli: None), _cbg=None)
    ok, why = RosExecutor._set_params(me, "/rh_place_node_right", {"object": "source240_real", "holder": 2, "holder_tol_m": 0.07})
    assert ok and why == []
    got = {p.name: p.value for p in sent[0].parameters}
    assert got["object"].type == ParameterType.PARAMETER_STRING and got["object"].string_value == "source240_real"
    assert got["holder"].type == ParameterType.PARAMETER_INTEGER and got["holder"].integer_value == 2
    assert got["holder_tol_m"].type == ParameterType.PARAMETER_DOUBLE and got["holder_tol_m"].double_value == 0.07


# ---------------------------------------------------------------- 잡기 노드 — 컵 토픽
def _aglt(active: str, wanted: str):
    log = []
    me = SimpleNamespace(fam=SimpleNamespace(cups=(("arm", "cup_topic", "/objects/aglt_cup_s065/pose"),), roles=("arm",)),
                         get_parameter=lambda n: SimpleNamespace(value=wanted),
                         _cup_topics={"arm": active}, _poll_cups={"arm": f"sub:{active}"},
                         cups={"arm": (1.0, np.zeros(3), np.array([1.0, 0, 0, 0]), 1.0)},
                         _poll=SimpleNamespace(destroy_subscription=lambda s: log.append(("destroy", s))),
                         _make_cup_poll=lambda t: log.append(("make", t)) or f"sub:{t}",
                         get_logger=lambda: SimpleNamespace(info=lambda m: None))
    return me, log


def test_a_new_cup_topic_swaps_the_subscription_and_forgets_the_old_cup():
    me, log = _aglt(BLUE_LIVE, PINK_LIVE)
    assert PourFjNode._switch_cups(me) == [f"arm → {PINK_LIVE}"]
    assert log == [("destroy", f"sub:{BLUE_LIVE}"), ("make", PINK_LIVE)]
    assert me._poll_cups == {"arm": f"sub:{PINK_LIVE}"} and me._cup_topics == {"arm": PINK_LIVE}
    assert "arm" not in me.cups                                  # 파랑 마지막 값으로 분홍을 쫓지 않는다


def test_the_same_cup_topic_keeps_the_subscription_and_the_cup():
    me, log = _aglt(BLUE_LIVE, BLUE_LIVE)
    assert PourFjNode._switch_cups(me) == [] and log == [] and "arm" in me.cups


def test_after_a_switch_the_live_cup_reset_is_refused_as_stale_so_the_runner_retries():
    me, _ = _aglt(BLUE_LIVE, PINK_LIVE)
    me.__dict__.update(_drain=lambda: None, attach={}, gates={}, dropouts={}, cup_fresh=set(), cup_latch=False,
                       cup_hold_last=True, book=SimpleNamespace(phase="armed"), _fresh_cup=lambda role: None,
                       _reply=lambda res, ok, why: (ok, why))
    me._switch_cups = lambda: PourFjNode._switch_cups(me)
    ok, why = PourFjNode._srv_reset(me, None, None)
    assert not ok and RESET_CUP_STALE_TEXT in why[0] and me.book.phase == "idle"
    assert me._cup_topics == {"arm": PINK_LIVE}


# ---------------------------------------------------------------- 놓기 노드 — 홀더
@pytest.mark.parametrize("text, first, want", [("", 1, [1]), ("1,2", 1, [1, 2]), ("2, 1", 1, [1, 2]), ("2", 1, [1, 2])])
def test_holder_candidates_always_include_the_launch_holder(text, first, want):
    assert RPN.holder_candidates(text, first) == want


def test_holder_candidates_reject_garbage():
    with pytest.raises(PourFjNodeError):
        RPN.holder_candidates("1,b", 1)


def _place(*, active=1, wanted=2, poses=None, holder_range=True, tol=0.02):
    contract = SimpleNamespace(holder_range={"x": [0.375, 0.38], "y": [-0.223, -0.037]} if holder_range else None,
                               target_holders=[1])
    params = {"holder": wanted, "holder_tol_m": tol}
    me = SimpleNamespace(contract=contract, holder_id=active, holder_tol=0.02, get_parameter=lambda n: SimpleNamespace(value=params[n]),
                         _holder_topics={1: "/objects/cup_holder_1/pose", 2: "/objects/cup_holder_2/pose"},
                         _holder_poses=dict(poses or {}), _holder=None, _holder_excess={"y": 0.06},
                         get_logger=lambda: SimpleNamespace(info=lambda m: None))
    me._check_holder_id = lambda k: RPN.PlaceNode._check_holder_id(me, k)
    return me


H1, H2 = np.array([0.3783, -0.0586, -0.0035]), np.array([0.3806, -0.2827, -0.0035])


def test_reset_moves_the_place_to_the_requested_holder():
    me = _place(poses={1: H1, 2: H2})
    RPN.PlaceNode._select_holder(me)
    assert me.holder_id == 2 and np.allclose(me._holder, H2) and me._holder_excess == {}


def test_reset_takes_the_jobs_holder_tolerance():
    """★10.29 실행기가 job 마다 넣은 holder_tol_m 을 reset 이 다시 읽는다. 분홍 → 홀더 2 만 넓히고, 다음 파랑 → 홀더 1 은
    0.02 로 돌아온다."""
    me = _place(poses={1: H1, 2: H2}, tol=0.07)
    RPN.PlaceNode._select_holder(me)
    assert me.holder_id == 2 and me.holder_tol == 0.07
    back = _place(active=2, wanted=1, poses={1: H1, 2: H2}, tol=0.02)
    back.holder_tol = 0.07
    RPN.PlaceNode._select_holder(back)
    assert back.holder_id == 1 and back.holder_tol == 0.02


@pytest.mark.parametrize("tol", [0.0, 0.5, float("nan")])
def test_a_bad_holder_tolerance_refuses_the_reset_before_the_switch(tol):
    me = _place(poses={1: H1, 2: H2}, tol=tol)
    with pytest.raises(PourFjNodeError, match="holder_tol_m"):
        RPN.PlaceNode._select_holder(me)
    assert me.holder_id == 1 and me.holder_tol == 0.02


def test_the_same_holder_keeps_its_latest_pose():
    me = _place(active=1, wanted=1, poses={1: H1})
    RPN.PlaceNode._select_holder(me)
    assert me.holder_id == 1 and np.allclose(me._holder, H1)


def test_a_holder_the_node_does_not_listen_to_is_refused():
    me = _place(wanted=0, poses={1: H1, 2: H2})
    with pytest.raises(PourFjNodeError, match="holders"):
        RPN.PlaceNode._select_holder(me)
    assert me.holder_id == 1


def test_a_holder_without_a_pose_yet_is_refused():
    me = _place(poses={1: H1})
    with pytest.raises(PourFjNodeError, match="cup_holder_2"):
        RPN.PlaceNode._select_holder(me)
    assert me.holder_id == 1


def test_an_old_contract_still_only_takes_its_trained_holders():
    me = _place(poses={1: H1, 2: H2}, holder_range=False)
    with pytest.raises(PourFjNodeError, match="학습 목표 홀더"):
        RPN.PlaceNode._select_holder(me)


def test_holder_poses_arrive_per_holder_and_only_the_active_one_drives_the_place(monkeypatch):
    me = _place(active=1, poses={})
    me._errors = {}
    monkeypatch.setattr(RPN.codec, "decode_pose", lambda msg: SimpleNamespace(pos=msg))
    RPN.PlaceNode._on_holder(me, H2, 2)
    assert np.allclose(me._holder_poses[2], H2) and me._holder is None
    RPN.PlaceNode._on_holder(me, H1, 1)
    assert np.allclose(me._holder, H1)


def test_a_refused_holder_switch_refuses_the_reset_before_anything_else():
    me = _place(wanted=0, poses={1: H1, 2: H2})
    me.book = SimpleNamespace(phase="armed")
    me._reply = lambda res, ok, why: (ok, why)
    me._select_holder = lambda: RPN.PlaceNode._select_holder(me)
    ok, why = RPN.PlaceNode._srv_reset(me, None, None)
    assert not ok and "holders" in why[0] and me.book.phase == "idle"


# ---------------------------------------------------------------- --plan — 놓기 홀더가 계약 학습 범위 안인가
REPO = Path(__file__).resolve().parents[2]
HOLDERS_5090 = REPO / "config" / "cup_holder_poses_5090_v1.yaml"


def _place_contracts(ep):
    from policy_control import rh_place as P
    return {r: P.load_contract(REPO / "deploy" / "policies" / b.policy / "rh_place_contract.json")
            for r, b in ep.policies.items() if b.kind == "place"}


@pytest.mark.parametrize("name", ["cocktail_1", "cocktail_1b"])
def test_plan_takes_pink_into_holder_2_with_the_cards_tolerance(name):
    """★10.29 사용자(sim_e2e 경유) 2번: 홀더 2(y −0.2827)는 n07 학습 범위에서 6.0 cm 밖이지만 그 자리에 두고 받는다(sim B
    평가가 이 자리에서 잰 값). 카드 place_pink 의 holder_tol_m 0.07 안이라 막지 않고 알림만 남긴다."""
    from policy_control.episode_ros import load_holder_poses
    from policy_control.rh_place import HOLDER_TOL_M
    ep = S.load(EPISODES / f"{name}.yaml")
    bad, notes = S.place_holder_checks(ep, load_holder_poses(HOLDERS_5090), _place_contracts(ep), HOLDER_TOL_M)
    assert bad == []
    assert len(notes) == 1, notes                              # 파랑 → 1 · shaker → 0 은 범위 안
    assert notes[0].startswith("place_pink: rh_place_r → HOLDER_2(2)") and "y 6.0 cm" in notes[0] and "허용 7.0 cm" in notes[0]


@pytest.mark.parametrize("name", ["cocktail_1", "cocktail_1b"])
def test_without_the_cards_tolerance_holder_2_is_still_refused(name):
    """넓힌 허용은 카드에 적은 그 job 에만 있다. 빼면 기본 2 cm 로 다시 막는다(다른 job · 다른 에피소드의 검사는 그대로)."""
    from dataclasses import replace
    from policy_control.episode_ros import load_holder_poses
    from policy_control.rh_place import HOLDER_TOL_M
    ep = S.load(EPISODES / f"{name}.yaml")
    bare = replace(ep, nodes=tuple(replace(n, jobs=tuple(replace(j, holder_tol_m=None) for j in n.jobs)) for n in ep.nodes))
    bad, _ = S.place_holder_checks(bare, load_holder_poses(HOLDERS_5090), _place_contracts(bare), HOLDER_TOL_M)
    assert len(bad) == 1, bad
    assert bad[0].startswith("place_pink: rh_place_r → HOLDER_2(2)") and "6.0 cm 밖(허용 2.0 cm)" in bad[0]


def test_plan_passes_holders_inside_the_range():
    ep = S.load(EPISODES / "cocktail_1b.yaml")
    wide = SimpleNamespace(holder_range={"x": [0.37, 0.39], "y": [-0.33, 0.20]})
    poses = {k: ((0.38, y, 0.235), 0.0) for k, y in ((0, 0.16), (1, -0.06), (2, -0.28))}
    assert S.place_holder_checks(ep, poses, {"rh_place_r": wide, "rh_place_l": wide}, 0.02) == ([], [])


def _raw(name: str) -> dict:
    import yaml
    return yaml.safe_load((EPISODES / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("where, value", [
    ("place_pink", 0.5),             # 상한 0.10 m 위(오타로 검사가 꺼지지 않게)
    ("place_pink", 0),
    ("place_pink", "7cm"),
    ("grasp_pink", 0.07),            # 홀더가 없는 잡기 job
])
def test_the_card_tolerance_is_only_a_small_distance_on_a_place_job_with_a_holder(where, value):
    raw = _raw("cocktail_1")
    seq = [dict(n, holder_tol_m=value) if n["id"] == where else n for n in raw["sequence"]]
    with pytest.raises(S.EpisodeSpecError, match="holder_tol_m"):
        S.validate(S.parse({**raw, "sequence": seq}, path="t"))


@pytest.mark.parametrize("name", ["cocktail_1", "cocktail_1b"])
def test_the_plan_tool_passes_with_holder_2_and_shows_the_accepted_tolerance(name, capsys, monkeypatch):
    """시작 검사(--plan --require-holders)가 홀더 2 로 막지 않고, 받은 범위 밖 홀더를 알림 줄(!)로 보인다. 정책 파일(ckpt)이
    없는 PC 에서도 홀더 검사만 보도록 등록부 판정은 통과로 둔다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("episode_run", REPO / "deploy/policy_control/tools/episode_run.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(S, "availability", lambda ep, entries: {j.role: "" for n in ep.nodes for j in n.jobs})
    rc = tool.main(["--episode", str(EPISODES / f"{name}.yaml"), "--plan", "--require-holders"])
    out = capsys.readouterr().out
    assert "✗ 홀더" not in out and "! 홀더: place_pink: rh_place_r → HOLDER_2(2)" in out
    assert rc == 0, out
