"""pd 이름은 팔마다 갈린다 — 서비스 · status · 노드 (ROS 없음, 순수 부분).

09.23 사용자: "pd 를 구분하는 게 맞을 것 같음. 앞으로도 양팔 또는 한팔 정책들이 많을 거고
개별 제어를 하는 게 맞을 것 같음."

그 전에는 `/policy_control/pd/*` 하나뿐이라 왼팔 pd 를 띄우려면 오른팔 pd 를 내려야 했다
(미션이 실제로 그렇게 적혀 있었다). 여기서 잠그는 것:
  ① 이름 규칙이 한 곳(pd_node)에 있고, 부르는 쪽(episode_ctl · trigger)이 같은 규칙을 쓴다
  ② 노드 이름은 쪽 하나면 갈리고, 양쪽을 맡으면 하나다
  ③ `--side` 없이 pd 를 부를 수 없다(기본값이 있더라도 경로에 쪽이 들어간다)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
TOOLS = SIM2REAL / "deploy/policy_control/tools"
LAUNCH = SIM2REAL / "deploy/policy_control/launch"

pytestmark = pytest.mark.unit


def _load(name: str, where: Path = TOOLS):
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("episode_ctl")
T = _load("trigger")


def _pd_node_names():
    from policy_control import pd_node

    return pd_node


@pytest.mark.parametrize("side", ("right", "left"))
def test_the_name_rule_lives_in_one_place_and_every_caller_uses_it(side):
    pd_node = _pd_node_names()
    assert pd_node.pd_ns(side) == f"/policy_control/pd_{side}"
    assert pd_node.status_node(side) == f"pd_{side}"
    # 부르는 쪽이 같은 경로를 만든다 — 어긋나면 서비스가 없다고만 나오고 이유를 모른다
    assert E.pd_service(side, "engage") == f"{pd_node.pd_ns(side)}/engage"
    assert E.pd_status(side) == f"/policy_control/status/{pd_node.status_node(side)}"
    assert T.resolve("pd/engage", side) == f"{pd_node.pd_ns(side)}/engage"


def test_episode_services_keep_no_side():
    """에피소드는 체인 하나에 하나다 — 쪽을 붙이면 없는 서비스를 부른다."""
    assert T.resolve("episode/start", "right") == "/policy_control/episode/start"
    assert T.resolve("episode/abort", "left") == "/policy_control/episode/abort"


def test_the_node_name_splits_only_when_it_owns_one_arm():
    pd_node = _pd_node_names()
    assert pd_node.node_name(["right"]) == "pd_node_right"
    assert pd_node.node_name(["left"]) == "pd_node_left"
    assert pd_node.node_name(["right", "left"]) == "pd_node"      # 한 노드가 양팔 — 이름은 하나
    assert pd_node.node_name([]) == "pd_node"


@pytest.mark.parametrize("stage_id", ("pd_engage", "pd_goto_home", "pd_hand_home", "pd_hand_rest",
                                      "pd_hand_path", "pd_release"))
def test_every_pd_stage_resolves_to_a_sided_service(stage_id):
    stage = E.stage_by_id(stage_id)
    for side in ("right", "left"):
        path = E.service_of(stage, side)
        assert path.startswith(f"/policy_control/pd_{side}/"), (stage_id, path)
    #: 표에는 쪽이 없다 — 쪽은 부를 때 붙는다(두 곳에 적히면 어긋난다)
    assert "/" not in str(stage.service) or not str(stage.service).startswith("/policy_control")


def test_episode_stages_are_untouched_by_the_side():
    for sid in ("ep_reset", "ep_start", "ep_stop"):
        stage = E.stage_by_id(sid)
        assert E.service_of(stage, "right") == E.service_of(stage, "left")


def test_an_unknown_side_is_refused_by_the_cli():
    with pytest.raises(SystemExit):
        E.main(["--side", "middle", "--only", "pd_engage"])


def test_the_launch_names_the_node_after_its_arm(monkeypatch):
    launch = _load("pd_controller.launch", LAUNCH) if (LAUNCH / "pd_controller.launch.py").exists() else None
    assert launch is not None
    made = {}

    def fake_make_node(name, params, *, use_source, node_name=None, **kw):
        made["name"], made["node_name"], made["params"] = name, node_name, params
        return object()

    monkeypatch.setattr(launch, "make_node", fake_make_node)
    monkeypatch.setattr(launch, "check_domain", lambda fake: None)
    monkeypatch.setattr(launch, "require_file", lambda p, what: Path(p))
    monkeypatch.setattr(launch, "resolve_path", lambda p: Path(str(p)))
    monkeypatch.setattr(launch, "resolve_robot", lambda p: Path(str(p)))
    monkeypatch.setattr(launch, "resolve_pd_config", lambda p: Path(str(p)))
    cfg = {"contract": "c.json", "robot": "r.yaml", "pd_config": "pd.yaml", "fake": "true", "execute": "true"}
    launch.pd_nodes({**cfg, "sides": "right"})
    assert made["node_name"] == "pd_node_right"
    launch.pd_nodes({**cfg, "sides": "left"})
    assert made["node_name"] == "pd_node_left"
    launch.pd_nodes({**cfg, "sides": "both"})
    assert made["node_name"] == "pd_node"                          # 양팔 한 노드 — 이름은 하나
