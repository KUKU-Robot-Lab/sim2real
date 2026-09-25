"""실측 손 자세에서 접기 — 손가락끼리 안 닿는 만큼만 (ROS 없음, 순수 부분).

09.23 사용자: "손가락들이 서로 충돌이 일어나지 않게 모을 순 없는건가? 현재 JOINT STATE 기반해서."

여기서 잠그는 것:
  ① **볼록 껍질로는 손을 판정할 수 없다** — 편 손에서도 이웃 손가락 껍질이 겹친다고 나온다.
     원본 메쉬로는 안 겹친다. 이 차이가 이 도구가 껍질 대신 메쉬를 쓰는 이유 전부다.
  ② 접기는 실측에서 출발한다 — 손가락마다 갈 수 있는 만큼만 가고, 못 가면 거기서 멈춘다
  ③ 굳은 손가락은 실측값에 묶이고, 그 때문에 봉투 구를 넘으면 **넘는다고 말한다**(조용히 통과시키지 않는다)
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

pytestmark = pytest.mark.unit


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


F = _load("plan_hand_fold")
W = F.W


@pytest.fixture(scope="module")
def urdf():
    return W.parse_urdf(W.URDF_DEFAULT)


@pytest.fixture(scope="module")
def fist():
    pose = yaml.safe_load((SIM2REAL / "deploy/policy_control/config/pd_dg5f_m_short.yaml").read_text())
    return {s: {k: float(v) for k, v in pose["hand_path_pose"][s].items()} for s in ("right", "left")}


@pytest.fixture(scope="module")
def geo(urdf):
    links, joints, _ = urdf
    return F.HandGeometry(links, joints, "right")


def test_only_fingers_count_as_fingers():
    assert F.finger_of("r_hl_index_2") == "index" and F.finger_of("r_hl_thumb_tip") == "thumb"
    #: 손바닥·어댑터를 손가락으로 세면 그들끼리 늘 붙어 있어 접기가 0 % 에서 멈춘다(09.23 에 그렇게 났다)
    for other in ("r_hl_palm", "r_hl_base", "r_hl_adapter", "r_hl_flange_adapter", "r_al_7"):
        assert F.finger_of(other) == ""


def test_the_convex_hull_model_cannot_judge_the_hand(fist):
    """껍질 세계는 **편 손에서도** 손가락끼리 겹쳤다고 한다 — 그래서 게이트로 쓰지 않는다."""
    pytest.importorskip("mujoco")
    world = W.build_world(W.WorldSpec(side="right", hand_self=True, detect_margin=0.03))
    for label, q in (("편 손", {k: 0.0 for k in fist["right"]}), ("주먹", fist["right"])):
        world.set_q(q)
        pairs = {k: v for k, v in world.pair_distances(np.zeros(7)).items()
                 if F.finger_of(k[0]) and F.finger_of(k[1]) and F.finger_of(k[0]) != F.finger_of(k[1])}
        assert pairs, label
        assert min(pairs.values()) < 0, f"{label}: 껍질이 겹치지 않으면 이 테스트의 전제가 바뀐 것이다"


def test_the_raw_mesh_says_the_fist_is_clear(geo, fist):
    """같은 자세를 원본 메쉬로 보면 떨어져 있다 — 0.95 mm 는 나란히 도는 약지↔새끼다."""
    d, a, b = geo.worst_pair(fist["right"])
    assert d > 0, (d, a, b)
    assert d == pytest.approx(0.00095, abs=2e-4), (d, a, b)      # 09.23 실측 — 바뀌면 손 자산이 바뀐 것이다
    assert {F.finger_of(a), F.finger_of(b)} == {"ring", "pinky"}


def test_folding_an_open_hand_reaches_the_fist(geo, urdf, fist):
    links, joints, _ = urdf
    now = {k: 0.0 for k in fist["right"]}
    q = F.fold(geo, now, fist["right"], F.MARGIN_M)
    for k, v in fist["right"].items():
        assert q[k] == pytest.approx(v), k                       # 걸리는 것이 없으면 끝까지 간다
    assert W.hand_radius(links, joints, q, "right")[0] < W.HAND_SPHERE_DEFAULT


def test_a_finger_that_cannot_move_stays_where_it_is(geo, urdf, fist):
    """새끼가 굳으면(오른손 error 409) 나머지를 다 접어도 봉투를 못 맞춘다 — 그 사실을 숫자로 낸다."""
    links, joints, _ = urdf
    now = {k: 0.0 for k in fist["right"]}
    frozen = {k: 0.0 for k in fist["right"] if "pinky" in k}
    target = {**fist["right"], **frozen}
    q = F.fold(geo, now, target, F.MARGIN_M)
    assert all(q[k] == 0.0 for k in frozen)                      # 굳은 관절은 그대로
    r, worst = W.hand_radius(links, joints, q, "right")
    assert r > W.HAND_SPHERE_DEFAULT and "pinky" in worst        # 멀리 남는 것은 그 새끼손가락이다
    assert r == pytest.approx(0.148, abs=2e-3)


def test_folding_stops_before_the_fingers_touch(geo, fist):
    """여유를 크게 요구하면 덜 접힌다 — 접기가 실제로 충돌을 보고 멈추는지."""
    now = {k: 0.0 for k in fist["right"]}
    #: 1.5 mm 는 편 손(2.1 mm)은 만족하고 주먹(1.0 mm)은 못 만족한다 — 그 사이에서 멈춰야 한다.
    #: (4 mm 처럼 **출발 자세부터** 못 만족하는 값을 주면 접기는 0 % 에서 선다 — 그건 손 구조의 한계다.)
    loose = F.fold(geo, now, fist["right"], 0.0015)
    assert geo.worst_pair(loose)[0] >= 0.0015 - 1e-9
    assert any(loose[k] != pytest.approx(fist["right"][k]) for k in fist["right"])


def test_the_pruning_never_hides_the_closest_pair(geo, fist):
    """경계구 가지치기가 최소 거리를 바꾸면 안 된다 — 가지치기 없이 센 값과 같아야 한다."""
    q = fist["right"]
    pts = geo.placed(q)
    from scipy.spatial import cKDTree

    brute = min(float(cKDTree(pts[b]).query(pts[a])[0].min()) for a, b in geo.pairs)
    assert geo.worst_pair(q)[0] == pytest.approx(brute, abs=1e-9)
