"""★11.19 커피박스 — plan_reach.face_azimuth: 박스 면 법선(±축) 중 회귀 방위에 가까운 쪽과 돌릴 각도(부호 = 시계)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
from scipy.spatial.transform import Rotation

TOOLS = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"


def _plan_reach():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("plan_reach", TOOLS / "plan_reach.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plan_reach"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("yaw, want", [(-100, 0.0), (-80, 20.0), (80, 0.0), (-145, -45.0), (170, 90.0)])
def test_face_azimuth_picks_the_nearer_normal(yaw, want):
    f = _plan_reach().face_azimuth
    x, y, z, w = Rotation.from_euler("z", yaw, degrees=True).as_quat()
    phi0 = math.radians(-100.0)
    phi, dd = f([w, x, y, z], [1.0, 0.0, 0.0], phi0)
    assert abs(abs(dd) - abs(want)) < 1e-6 and (want == 0 or abs(want) == 90 or math.copysign(1, dd) == math.copysign(1, want))
    assert math.isclose(math.remainder(phi - phi0, 2 * math.pi), math.radians(dd), abs_tol=1e-9)


def test_box_surface_dist_is_a_box_not_a_cylinder():
    """★11.20 박스 손 점 검사 — 면 앞 · 모서리 밖 · 옆면 곁(원기둥이면 안으로 잡히는 점)이 박스 기준으로 맞는지."""
    d = _plan_reach().cup_surface_dist
    box = (math.radians(90.0), 0.055, 0.040)          # 박스 x 축이 세계 +y — 반폭 x 55 · y 40 mm
    pts = [[0.0, -0.065, 0.05],                        # 80.5 면(세계 -y) 앞 10 mm
           [0.050, 0.0, 0.05],                         # 옆면(세계 +x) 밖 10 mm — 원기둥 r 55 면 안(-5 mm)
           [0.0, 0.0, 0.05],                           # 한가운데 — 가까운 옆면까지 40 mm 안
           [0.050, 0.065, 0.05]]                       # 모서리 밖 (10, 10) mm
    got = d(pts, [0.0, 0.0], 0.0, 0.126, 0.055, box) * 1000
    assert [round(float(v), 1) for v in got] == [10.0, 10.0, -40.0, 14.1]
    assert d(pts[1:2], [0.0, 0.0], 0.0, 0.126, 0.055)[0] < 0   # 원기둥(box=None)은 그대로
