import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cad_body_alignment import mesh_aabb, cad_to_body_yup_to_zup  # noqa: E402
from pour_obs_geometry import quat_apply  # noqa: E402

CUP_OBJ = str(Path(__file__).resolve().parents[2] / "perception/assets/Cup/cup.obj")

def test_mesh_aabb_matches_measured_cup():
    mn, mx = mesh_aabb(CUP_OBJ)
    assert np.allclose(mn, [-0.0463, -0.0773, -0.0440], atol=1e-3)
    assert np.allclose(mx, [ 0.0437,  0.1003,  0.0460], atol=1e-3)

def test_yup_to_zup_rotation_maps_mesh_Y_to_body_Z():
    _, quat = cad_to_body_yup_to_zup(
        np.array([-0.0463, -0.0773, -0.0440]),
        np.array([ 0.0437,  0.1003,  0.0460]))
    # mesh +Y(높이축)가 body +Z 로 가야 한다
    assert np.allclose(quat_apply(quat, [0, 1, 0]), [0, 0, 1], atol=1e-9)

def test_yup_to_zup_translation_puts_bottom_center_at_origin():
    pos, quat = cad_to_body_yup_to_zup(
        np.array([-0.0463, -0.0773, -0.0440]),
        np.array([ 0.0437,  0.1003,  0.0460]))
    # mesh 바닥면(y=min)의 x/z 중심점이 변환 후 body 원점(≈0,0,0) 근처
    bottom_center_mesh = np.array([(-0.0463 + 0.0437) / 2, -0.0773,
                                   (-0.0440 + 0.0460) / 2])
    mapped = np.array(quat_apply(quat, bottom_center_mesh)) + pos
    assert np.allclose(mapped, [0.0, 0.0, 0.0], atol=1e-6)
