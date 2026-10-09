"""CAD 하나로 FP++ 물체 등록 · 색 맞춤 · 활성 묶음 — 10.09 사용자: "cad 만 올리면 자동으로 추출하게, 매번 오래 걸리면 안됨".

손으로 하던 것: 메쉬 단위 · 원점 맞추기, objects.yaml 항목 쓰기, 색상(hue) 범위 손 튜닝, 미션 · 테스트의 물체 이름 고치기.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "ops"))

import fpp_color_pick as C  # noqa: E402
import fpp_object as F  # noqa: E402
from object_registry import load_registry  # noqa: E402


def _bottle_stl(path: Path, units_mm: bool = True, z0: float = -95.0, h: float = 240.0) -> Path:
    m = trimesh.creation.cylinder(radius=30.0, height=h)
    m.apply_translation([0, 0, z0 + h / 2])               # CAD 원점 = 바닥 위 95 mm
    if not units_mm:
        m.apply_scale(0.001)
    m.export(path)
    return path


@pytest.fixture
def repo(tmp_path):
    """임시 저장소 — objects.yaml(카메라만) · objects.d · assets/meshes."""
    (tmp_path / "config" / "objects.d").mkdir(parents=True)
    (tmp_path / "assets" / "meshes").mkdir(parents=True)
    (tmp_path / "config" / "objects.yaml").write_text(yaml.safe_dump(
        {"camera_extrinsics": str(ROOT / "config" / "global_camera_extrinsics_arm4090.yaml"), "objects": {}}))
    return tmp_path


def test_a_mm_cad_becomes_a_metre_mesh_with_the_policy_origin(repo, tmp_path):
    stl = _bottle_stl(tmp_path / "bottle.stl")
    out = F.convert_mesh(stl, repo / "assets" / "meshes" / "bottle.obj", origin_above_bottom=0.085)
    m = trimesh.load(out)
    assert m.bounds[0][2] == pytest.approx(-0.085, abs=1e-6) and m.bounds[1][2] == pytest.approx(0.155, abs=1e-6)
    assert m.bounds[1][0] == pytest.approx(0.030, abs=1e-4)
    assert out.read_text().splitlines()[0].startswith("#")      # 출처 · 원점이 첫 줄에


def test_a_metre_cad_is_kept_and_the_default_origin_is_mid_height(repo, tmp_path):
    stl = _bottle_stl(tmp_path / "b_m.stl", units_mm=False)
    out = F.convert_mesh(stl, repo / "assets" / "meshes" / "b_m.obj", origin_above_bottom=None)
    m = trimesh.load(out)
    assert m.bounds[0][2] == pytest.approx(-0.120, abs=1e-6) and m.bounds[1][2] == pytest.approx(0.120, abs=1e-6)


def test_add_writes_one_object_per_colour_in_its_own_group_and_the_registry_loads_it(repo, tmp_path):
    stl = _bottle_stl(tmp_path / "bottle.stl")
    names = F.add_object(repo, stl, name="bottle", colors=["orange", "pink"], origin_above_bottom=0.085)
    assert names == ["bottle_orange", "bottle_pink"]
    reg = load_registry(repo / "config" / "objects.yaml")
    for n, c in zip(names, ("orange", "pink")):
        spec = reg.get(n)
        assert spec.fpp["group"] == "bottle" and spec.fpp["color"] == c
        assert spec.fpp["mesh_path"] == "assets/s2r_meshes/bottle.obj"
        assert spec.origin_above_bottom_m == pytest.approx(0.085)
        assert spec.aabb[0][2] == pytest.approx(-0.085, abs=1e-4) and spec.aabb[1][2] == pytest.approx(0.155, abs=1e-4)
        assert spec.symmetry_axis == (0.0, 0.0, 1.0) and not spec.symmetry_flip
    again = F.add_object(repo, stl, name="bottle", colors=["blue"], origin_above_bottom=0.085)   # 다시 하면 갈아 끼운다
    assert again == ["bottle_blue"] and "bottle_orange" not in load_registry(repo / "config" / "objects.yaml").names()


def test_a_name_already_in_objects_yaml_is_refused(repo, tmp_path):
    with pytest.raises(ValueError, match="cyl60"):
        F.add_object(ROOT, _bottle_stl(tmp_path / "x.stl"), name="cyl60", colors=["yellow"], origin_above_bottom=0.085,
                     dry_run=True)


def test_an_unknown_colour_name_is_refused(repo, tmp_path):
    with pytest.raises(ValueError, match="purple"):
        F.add_object(repo, _bottle_stl(tmp_path / "b.stl"), name="b", colors=["purple"], origin_above_bottom=0.085)


def test_a_measured_hue_range_replaces_the_colour_name(repo, tmp_path):
    """색 맞춤(calib)이 잰 hue 를 [lo, hi] 로 적는다 — 이름표의 고정 범위보다 우선."""
    F.add_object(repo, _bottle_stl(tmp_path / "b.stl"), name="b", colors=["orange", "pink"], origin_above_bottom=0.085)
    F.set_hues(repo, "b", {"b_orange": (14.0, 1.5), "b_pink": (157.0, 2.0)})
    reg = load_registry(repo / "config" / "objects.yaml")
    assert reg.get("b_orange").fpp["color"] == [7.0, 21.0]     # 중앙 ± max(7, 3σ)
    assert reg.get("b_pink").fpp["color"] == [150.0, 164.0]


def test_explicit_hue_ranges_work_like_names_and_wrap_around_red():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (183, 111, 24)                                      # 주황 hue 16
    m = np.ones((10, 10), bool)
    assert C.color_fraction(img, m, [10.0, 20.0]) == 1.0
    assert C.color_fraction(img, m, [20.0, 30.0]) == 0.0
    img[:] = (200, 20, 30)                                       # 빨강 hue ~178
    assert C.color_fraction(img, m, [172.0, 4.0]) == 1.0        # lo > hi = 0/180 을 넘는 구간
    with pytest.raises(ValueError):
        C.color_fraction(img, m, [10.0])


def test_calibration_orders_blobs_left_to_right_and_measures_each_hue():
    """테이블 위 색 덩어리를 base y 큰 쪽(왼쪽)부터 — 사용자가 왼쪽부터 놓은 순서와 짝짓는다."""
    hues = [np.full(500, 157.0), np.full(500, 14.0)]
    ys = [-0.10, 0.07]                                           # 첫 덩어리가 오른쪽
    got = F.pair_left_to_right(ys, hues, ["bottle_orange", "bottle_pink"])
    assert {k: round(v[0]) for k, v in got.items()} == {"bottle_orange": 14, "bottle_pink": 157}
    with pytest.raises(ValueError, match="2"):
        F.pair_left_to_right(ys[:1], hues[:1], ["bottle_orange", "bottle_pink"])


def test_hue_statistics_handle_the_red_wraparound():
    med, sd = F.hue_stats(np.array([178.0, 179.0, 1.0, 2.0]))
    assert med == pytest.approx(0.0, abs=1.6) or med == pytest.approx(180.0, abs=1.6)
    assert sd < 3.0


def test_activate_writes_the_side_mapping_the_mission_reads(repo):
    (repo / "config" / "objects.d" / "x.yaml").write_text(yaml.safe_dump({"objects": {}}))
    F.write_active(repo, group="bottle", right="bottle_pink", left="bottle_orange")
    doc = yaml.safe_load((repo / "config" / "fpp_active.yaml").read_text())
    assert doc == {"group": "bottle", "sides": {"right": "bottle_pink", "left": "bottle_orange"}}
