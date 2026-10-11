"""fpp_scene — FP++ 표본 → 방안 2 장면 JSON(로봇 base = env-local xy 그대로)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fpp_scene as F  # noqa: E402

UP = (0.0, 0.0, 0.0, 1.0)


def _tilted(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def test_tilt_deg():
    assert F.tilt_deg(UP) == pytest.approx(0.0, abs=1e-6)
    assert F.tilt_deg(_tilted(30)) == pytest.approx(30.0, abs=1e-4)


def test_summarize_takes_median_and_spread():
    s = [(0.25, -0.20, 0.27, *UP), (0.252, -0.20, 0.27, *UP), (0.40, -0.20, 0.27, *UP)]
    out = F.summarize(s)
    assert out["xy"] == [0.252, -0.2] and out["n"] == 3
    assert out["spread_m"] == pytest.approx(0.148, abs=1e-3)


def test_check_rejects_tipped_and_jittery():
    ok = F.summarize([(0.25, -0.2, 0.27, *UP)] * 5)
    F.check("c", ok)
    with pytest.raises(SystemExit, match="기울기"):
        F.check("c", F.summarize([(0.25, -0.2, 0.27, *_tilted(30))] * 5))
    with pytest.raises(SystemExit, match="흩어짐"):
        F.check("c", F.summarize([(0.25, -0.2, 0.27, *UP), (0.26, -0.2, 0.27, *UP), (0.25, -0.2, 0.27, *UP)]))


def test_build_scene_matches_search_scene_format():
    cup = F.summarize([(0.25, -0.2, 0.27, *UP)] * 3)
    obs = F.summarize([(0.30, -0.05, 0.27, *UP)] * 3)
    scene = F.build_scene(cup, 2, [("source240_real", obs)])
    assert scene["cup"] == [0.25, -0.2] and scene["holder"] == 2
    assert scene["obstacles"] == [{"object": "source240_real", "pos": [0.3, -0.05]}]


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        F.summarize([])
