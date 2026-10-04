"""object_registry 순수 로직 검증. numpy+yaml 만, ROS 불필요."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from object_registry import (  # noqa: E402
    DEFAULT_REGISTRY, container_name, extrinsics_for, input_topic, load_registry,
    output_topic, render_fpp_yaml, status_topic,
)


def test_default_registry_loads_real_objects():
    reg = load_registry(DEFAULT_REGISTRY)
    assert set(reg.names()) == {"shaker_closed", "cup_big_s100", "aglt_cup_s065", "cyl60", "cup_holder"}
    assert reg.get("shaker_closed").origin_above_bottom_m == pytest.approx(0.0921)
    assert reg.get("shaker_closed").symmetry_axis == (0.0, 0.0, 1.0)
    assert reg.get("cup_big_s100").symmetry_axis == (0.0, 1.0, 0.0)


def test_aglt_cup_is_shaker_scaled_065():
    """10.01 arm4090: rh_aglt 학습 컵(shaker_closed_thick × 0.65)의 실물 — 메시 · 원점 · cad_to_body 가 같은 배율."""
    reg = load_registry(DEFAULT_REGISTRY)
    sh, ag = reg.get("shaker_closed"), reg.get("aglt_cup_s065")
    assert ag.fpp["mesh_path"] == sh.fpp["mesh_path"] and ag.fpp["mesh_scale_to_meters"] == pytest.approx(0.65)
    assert ag.origin_above_bottom_m == pytest.approx(0.65 * sh.origin_above_bottom_m, abs=1e-5)
    assert ag.cad_to_body_pos == pytest.approx(0.65 * sh.cad_to_body_pos, abs=1e-5)
    assert ag.fpp["detection_pick"] == "bright"


def test_alias_resolves_to_canonical_and_unknown_raises():
    reg = load_registry(DEFAULT_REGISTRY)
    assert reg.resolve("cup_big_s080") == "cup_big_s100"
    assert reg.get("cup_big_s120").name == "cup_big_s100"
    with pytest.raises(ValueError, match="unknown object"):
        reg.resolve("teapot")


def test_topic_and_container_names_follow_convention():
    assert input_topic("shaker_closed") == "/perception_plus_plus/shaker_closed/pose"
    assert status_topic("shaker_closed") == "/perception_plus_plus/shaker_closed/tracking_status"
    assert output_topic("shaker_closed") == "/objects/shaker_closed/pose"
    assert container_name("shaker_closed") == "fpp_shaker_closed"


def test_render_fpp_yaml_matches_node_parameter_schema():
    reg = load_registry(DEFAULT_REGISTRY)
    doc = yaml.safe_load(render_fpp_yaml(reg.get("shaker_closed")))
    params = doc["cup_tracking"]["ros__parameters"]
    assert params["mesh_path"] == "assets/meshes/shaker_cad.obj"          # 09.07 사용자 지정 CAD
    assert params["mesh_scale_to_meters"] == 1.0
    assert params["detection_pick"] == "blue"
    assert params["pose_topic"] == "/perception_plus_plus/shaker_closed/pose"
    assert params["status_topic"] == "/perception_plus_plus/shaker_closed/tracking_status"
    assert params["child_frame_id"] == "shaker_closed"
    assert params["rgb_topic"] == "/camera/camera/color/image_raw"


def test_extrinsics_for_uses_shared_camera_but_object_cad_to_body():
    reg = load_registry(DEFAULT_REGISTRY)
    cam_yaml = DEFAULT_REGISTRY.parent / "global_camera_extrinsics.yaml"
    ext_s = extrinsics_for(reg.get("shaker_closed"), cam_yaml)
    ext_c = extrinsics_for(reg.get("cup_big_s100"), cam_yaml)
    assert np.allclose(ext_s.cam_pos, ext_c.cam_pos)
    assert np.allclose(ext_s.cad_to_body_quat, [1, 0, 0, 0])
    assert np.allclose(ext_c.cad_to_body_quat, [0.707107, -0.707107, 0, 0], atol=1e-5)
    # rot_x(−90°): body z 가 CAD +y(cup.obj 의 위)로 가야 한다
    from pose_symmetry import quat_axis_direction
    assert np.allclose(quat_axis_direction(ext_c.cad_to_body_quat, [0, 0, 1]), [0, 1, 0], atol=1e-5)
    assert ext_s.base_frame == "base_link"


def test_invalid_registry_fails_fast(tmp_path):
    bad = tmp_path / "objects.yaml"
    bad.write_text(
        "camera_extrinsics: config/global_camera_extrinsics.yaml\n"
        "objects:\n  x:\n    real: r\n    fpp: {mesh_path: a, mesh_scale_to_meters: 1.0,"
        " cup_class_id: 41, detection_pick: red, yolo_confidence: 0.3}\n"
        "    cad_to_body: {position: [0,0,0], orientation_wxyz: [2,0,0,0]}\n"
        "    sim: {usd: u, origin_above_bottom_m: 0.1}\n"
        "    aabb: [[0,0,0],[1,1,1]]\n")
    with pytest.raises(ValueError, match="not normalized"):
        load_registry(bad)
    cyc = tmp_path / "cyc.yaml"
    cyc.write_text("camera_extrinsics: c\nobjects: {}\naliases: {a: b, b: a}\n")
    with pytest.raises(ValueError, match="alias"):
        load_registry(cyc)


def test_cup_holder_fpp_mesh_is_the_marker_stl_in_meters():
    """10.03 FP++ 재확인용 — FP++ 메쉬(m)와 마커 추정 STL(mm)이 같은 형상 · 같은 원점, 컨테이너가 보는 자리에 있다."""
    import trimesh
    h = load_registry(DEFAULT_REGISTRY).get("cup_holder")
    assert h.symmetry_axis is None and h.origin_above_bottom_m == pytest.approx(0.030)
    rel = h.fpp["mesh_path"]
    assert rel.startswith("assets/s2r_meshes/")                       # fpp_up.sh 가 sim2real assets/meshes 를 여기에 붙인다
    repo = DEFAULT_REGISTRY.parents[1]
    obj = trimesh.load(repo / "assets/meshes" / Path(rel).name, force="mesh")
    stl = trimesh.load(repo / "assets/meshes/cup_holder.stl", force="mesh")
    assert np.allclose(obj.bounds, stl.bounds / 1000.0, atol=1e-6)
    assert np.allclose(obj.bounds, np.array(h.aabb), atol=1e-4)
    assert "s2r_meshes" in (repo / "scripts/vision/fpp_up.sh").read_text()


def test_cyl60_fpp_mesh_matches_the_sim_cylinder():
    """10.04 rh_aglt cyl60 정책용 — ⌀60 × 170 mm, 원점 = 중심(바닥 +0.085), 대칭축 z."""
    import trimesh
    c = load_registry(DEFAULT_REGISTRY).get("cyl60")
    assert c.origin_above_bottom_m == pytest.approx(0.085) and c.symmetry_axis == (0.0, 0.0, 1.0)
    repo = DEFAULT_REGISTRY.parents[1]
    m = trimesh.load(repo / "assets/meshes" / Path(c.fpp["mesh_path"]).name, force="mesh")
    assert np.allclose(m.bounds, [[-0.03, -0.03, -0.085], [0.03, 0.03, 0.085]], atol=1e-4)
    assert np.allclose(m.bounds, np.array(c.aabb), atol=1e-4)
