"""손 봉투 구 — 손가락 자세를 맞추는 대신 구에 담고 팔을 옮긴다 (ROS 없음, 순수 부분).

09.23 사용자: "핸드를 최대한 안 걸리게 할 거니까 그냥 큰 구를 달아논 상태에서 팔을 움직이게 하고,
그다음에 손가락을 피면 되잖아?"

여기서 잠그는 것:
  ① 구 반지름 0.11 m 가 **주먹은 담고 편 손은 못 담는다** — 그 경계가 운용 규칙이다
  ② 구 세계는 손바닥 **아래**(손가락)만 지운다. 손바닥 위 어댑터는 실물 그대로 남는다
     (처음 구현은 `_hl_` 이름만 보고 어댑터까지 지워 손이 통째로 사라진 세계에서 "통과"가 나왔다)
  ③ 구는 **세계·몸통·반대팔**에 대한 봉투다 — 같은 팔 링크와는 검사하지 않는다
  ④ 저장 경로가 그 반지름을 메타로 들고 다니고, 실기 검사가 그것을 읽는다
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SIM2REAL = Path(__file__).resolve().parents[2]
TOOLS = SIM2REAL / "deploy/policy_control/tools"
PATHS = SIM2REAL / "deploy/policy_control/paths"

pytestmark = pytest.mark.unit


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


W = _load("home_path_world")
C = _load("check_path_start")


@pytest.fixture(scope="module")
def urdf():
    return W.parse_urdf(W.URDF_DEFAULT)


@pytest.fixture(scope="module")
def fist():
    pose = yaml.safe_load((SIM2REAL / "deploy/policy_control/config/pd_dg5f_m_short.yaml").read_text())
    return {side: {k: float(v) for k, v in pose["hand_path_pose"][side].items()} for side in ("right", "left")}


@pytest.mark.parametrize("side", ("right", "left"))
def test_the_default_sphere_holds_a_fist_but_not_an_open_hand(urdf, fist, side):
    links, joints, _ = urdf
    closed, _ = W.hand_radius(links, joints, fist[side], side)
    opened, worst = W.hand_radius(links, joints, {}, side)
    assert closed < W.HAND_SPHERE_DEFAULT, "주먹은 기본 구 안에 들어가야 한다"
    assert opened > W.HAND_SPHERE_DEFAULT, "편 손까지 담는 구였다면 팔 경로가 안 나온다"
    assert "middle" in worst or "index" in worst           # 편 손에서 제일 먼 것은 가운뎃손가락 끝
    assert closed == pytest.approx(0.0932, abs=5e-4)       # 09.23 실측 — 바뀌면 운용 규칙이 바뀐 것이다


def test_a_half_closed_hand_is_still_inside_but_a_quarter_closed_is_not(urdf, fist):
    links, joints, _ = urdf
    r80 = W.hand_radius(links, joints, {k: v * 0.8 for k, v in fist["right"].items()}, "right")[0]
    r40 = W.hand_radius(links, joints, {k: v * 0.4 for k, v in fist["right"].items()}, "right")[0]
    assert r80 < W.HAND_SPHERE_DEFAULT < r40


def test_the_sphere_world_drops_only_the_fingers():
    world = W.build_world(W.WorldSpec(side="right", hand_sphere=W.HAND_SPHERE_DEFAULT))
    bodies = set(world.body_group)
    assert "r_hl_palm" in bodies
    for keep in ("r_hl_base", "r_hl_adapter", "r_hl_flange_adapter"):
        assert keep in bodies, f"{keep} 는 손가락이 아니다 — 자세와 무관한 실물이라 남는다"
    assert not [b for b in bodies if "_hl_" in b and b.split("_hl_")[1].startswith(("thumb", "index", "middle", "ring", "pinky"))]


def test_without_the_sphere_the_fingers_are_still_there():
    world = W.build_world(W.WorldSpec(side="right"))
    assert [b for b in world.body_group if "r_hl_thumb" in b]


def test_the_sphere_is_not_checked_against_its_own_arm():
    """구는 손목 뒤로 튀어나온다 — 그 헛걸림(r_al_5<->r_hl_palm)으로 목표 자세가 막혔었다."""
    world = W.build_world(W.WorldSpec(side="right", hand_sphere=W.HAND_SPHERE_DEFAULT))
    world.set_q({f"r_aj_{i}": 0.0 for i in range(1, 8)})
    palm = [(a, b) for a, b in world.pair_distances(np.zeros(7)) if "r_hl_palm" in (a, b)]
    assert not [p for p in palm if any(x.startswith("r_al_") for x in p)], "같은 팔은 제외한다"
    others = {x for p in palm for x in p} - {"r_hl_palm"}
    assert any(x.startswith(("table", "wall", "back_box", "body", "l_")) for x in others), \
        f"세계·몸통·반대팔과는 여전히 검사해야 한다: {sorted(others)}"


@pytest.mark.parametrize("side", ("right", "left"))
def test_the_saved_home_paths_carry_the_sphere_they_were_planned_with(side):
    d = np.load(PATHS / f"home_{side}.npz")
    assert "meta_hand_sphere" in d, "저장 경로가 봉투 반지름을 들고 다녀야 실기 검사가 그것을 읽는다"
    assert float(d["meta_hand_sphere"]) == pytest.approx(W.HAND_SPHERE_DEFAULT)
    assert str(d["meta_method"]) == "rrt"           # j1j4 는 구로는 격자 전부 실패한다(09.23)
    assert float(d["meta_min_clearance_non_escape"]) >= 0.02


@pytest.mark.parametrize("side", ("right", "left"))
def test_the_real_check_passes_a_fist_and_names_what_pokes_out(fist, side):
    assert C.sphere_verdict(fist[side], side, W.HAND_SPHERE_DEFAULT) == []
    half = {k: v * 0.5 for k, v in fist[side].items()}
    why = C.sphere_verdict(half, side, W.HAND_SPHERE_DEFAULT)
    assert len(why) == 1 and "더 오므릴 것" in why[0]
    assert "cm" in why[0]                            # 얼마나 넘는지 숫자로 말한다


def test_the_check_says_so_when_the_hand_state_is_missing():
    assert "손 관절 상태가 없다" in C.sphere_verdict({"r_aj_1": 0.0}, "right", W.HAND_SPHERE_DEFAULT)[0]
