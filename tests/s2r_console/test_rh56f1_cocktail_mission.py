"""★10.28 칵테일 SEP(cocktail_1) · B(cocktail_1b) 실기 미션 연결 — 사용자 "SEP, B 과정 모두 세팅 완료 해둬야 함"(sim_e2e 경유).

같은 오른손 노드가 파랑 → 분홍을 잡고 홀더 1 → 2 에 놓는다. 노드는 첫 job 값으로 뜨고(실행기가 job 마다 바꾼다), 놓기 노드는
두 홀더를 다 구독해 둔다. 분홍 FP++ 는 분홍만 든 묶음 source240p 라 cups 단계가 그 컨테이너도 띄운다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "deploy" / "policy_control"))
COCKTAILS = ("cocktail_1", "cocktail_1b")


def _generator():
    spec = importlib.util.spec_from_file_location("make_rh56f1_missions_cocktail", REPO / "scripts/ops/make_rh56f1_missions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _generator()


def _node_argv(cmds, script: str, side: str) -> list:
    tag = f"__node:={script.removesuffix('.py')}_{side}"
    (argv,) = [c["argv"] for c in cmds if tag in c.get("argv", [])]
    return argv


def test_both_cocktail_orders_are_console_episodes():
    assert set(COCKTAILS) <= set(GEN.EPISODES)


def test_the_cups_stage_starts_the_pink_only_group():
    assert "source240p_pink" in GEN._cups() and "source240p" in GEN._groups()
    assert GEN._cups()[:2] == [GEN.REAL_CUPS["right"], GEN.REAL_CUPS["left"]]      # 팔 물체가 먼저(그대로)


@pytest.mark.parametrize("name", COCKTAILS)
def test_the_right_place_node_listens_to_both_of_its_holders(name):
    cmds = GEN._episode_run(name, True)
    right = _node_argv(cmds, "rh_place_node.py", "right")
    assert "holder:=1" in right and "holders:=1,2" in right
    left = _node_argv(cmds, "rh_place_node.py", "left")
    assert "holder:=0" in left and not any(a.startswith("holders:=") for a in left)


@pytest.mark.parametrize("name", COCKTAILS)
def test_the_grasp_node_starts_on_blue_with_the_topic_the_runner_uses(name):
    from policy_control.episode_ros import job_params
    from policy_control.episode_runner import EpisodeManager
    ep = GEN._episode(name)
    argv = _node_argv(GEN._episode_run(name, True), "rh_aglt_node.py", "right")
    node = next(n for n in ep.nodes if n.id == "grasp_blue")
    (plan,) = EpisodeManager._plans(SimpleNamespace(ep=ep), node, SimpleNamespace(objects={}))
    assert f"cup_topic:={job_params(ep, plan)['cup_topic']}" in argv     # 생성기 _live 와 실행기 live_topic 이 같은 규칙


@pytest.mark.parametrize("name", COCKTAILS)
def test_fake_objects_sit_inside_their_regions(name):
    ep = GEN._episode(name)
    fakes = {c["argv"][c["argv"].index("--topic") + 1]: c["argv"] for c in GEN._episode_run(name, False)
             if "fake_cup_pose_pub.py" in " ".join(c.get("argv", []))}
    regions = 0
    for oname, obj in ep.objects.items():
        reg = obj.get("region")
        if not reg:
            continue
        regions += 1
        a = fakes[obj["topic"]]
        x, y = float(a[a.index("--x") + 1]), float(a[a.index("--y") + 1])
        assert reg["x"][0] <= x <= reg["x"][1] and reg["y"][0] <= y <= reg["y"][1], (oname, x, y)
    assert regions == 1


def test_sep_tells_the_operator_to_keep_pink_off_the_table_at_the_start():
    note = next(c["note"] for c in GEN._episode_run("cocktail_1", True) if c.get("note", "").startswith("★[에피소드 cocktail_1]"))
    assert "물체(SRC_BLUE, SHAKER)" in note and "SRC_PINK 는 아직 탁자에 두지 않는다" in note


def _first_jobs(ep) -> tuple[dict, dict]:
    """정책 노드(스크립트_팔) → (첫 job 의 에피소드 노드 id, 그 job_params) · 놓기 노드 → 모든 놓기 job 홀더."""
    from policy_control.episode_ros import job_params
    from policy_control.episode_runner import EpisodeManager
    first, holders = {}, {}
    for n in ep.nodes:
        for plan in EpisodeManager._plans(SimpleNamespace(ep=ep), n, SimpleNamespace(objects={})):
            if plan.kind not in ("aglt", "place"):
                continue
            side = "right" if plan.role.endswith("_r") else "left"
            key = f"{'rh_aglt_node' if plan.kind == 'aglt' else 'rh_place_node'}_{side}"
            jp = job_params(ep, plan)
            first.setdefault(key, (n.id, jp))
            if "holder" in jp:
                holders.setdefault(key, set()).add(jp["holder"])
    return first, holders


#: aglt · place job 이 있는 콘솔 에피소드(grasp_b_* · tilt_base_b 처럼 다른 정책만 쓰는 것은 실행기가 넣는 값이 없다)
POLICY_EPISODES = sorted(n for n in GEN.EPISODES if _first_jobs(GEN._episode(n))[0])


def test_the_first_job_check_covers_both_cocktails_and_the_single_object_episodes():
    assert set(COCKTAILS) <= set(POLICY_EPISODES) and len(POLICY_EPISODES) > len(COCKTAILS)


@pytest.mark.parametrize("real", (True, False))
@pytest.mark.parametrize("name", POLICY_EPISODES)
def test_every_policy_node_starts_on_its_first_job_and_listens_to_all_its_holders(name, real):
    """★10.28 실행기는 job 마다 cup_topic · holder 를 넣고 노드는 reset 에서 다시 읽는다 — 노드가 뜨는 값이 첫 job 값과 같아야
    칵테일이 아닌 에피소드는 아무것도 바뀌지 않는다. 놓기 job 의 홀더는 노드가 다 구독한다(밖이면 그 job 의 reset 이 거부된다)."""
    ep, cmds = GEN._episode(name), GEN._episode_run(name, real)
    argvs = {a[len("__node:="):]: c["argv"] for c in cmds for a in c.get("argv", []) if a.startswith("__node:=")}
    first, holders = _first_jobs(ep)
    for key, (nid, jp) in first.items():
        got = dict(a.split(":=", 1) for a in argvs[key] if ":=" in a and not a.startswith("__"))
        for param in ("cup_topic", "holder"):
            if param in jp:
                assert got.get(param) == str(jp[param]), (key, nid, param)
        if key in holders:
            listen = {int(x) for x in got.get("holders", "").split(",") if x} | {int(got["holder"])}
            assert holders[key] <= listen, (key, sorted(holders[key]), sorted(listen))
