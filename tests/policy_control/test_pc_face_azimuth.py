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
