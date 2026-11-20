"""11.20 reach_world 홀더 상자 거리: 꼭짓점이 모두 밖이어도 면·모서리가 뚫으면 충돌로 본다(codex 지적)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"
sys.path.insert(0, str(TOOLS))
import reach_world as R  # noqa: E402


def _box(c, h):
    c, h = np.asarray(c, float), np.asarray(h, float)
    return np.array([c + h * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)])


def test_big_piece_enclosing_small_box_is_a_collision():
    v = _box([0, 0, 0], [0.10, 0.04, 0.04])          # 꼭짓점 8개 모두 홀더 밖
    assert R._mesh_box_gap(v, np.array([0.02, 0.03, 0.03])) < 0


def test_gap_never_exceeds_true_distance():
    rng = np.random.default_rng(0)
    h = np.array([0.02, 0.03, 0.03])
    for _ in range(500):
        v = rng.normal(size=(12, 3)) * 0.02 + rng.normal(size=3) * 0.08
        # 볼록 조각 위 표본 점의 상자 거리 최소 = 실제 거리의 위쪽 근사
        wts = rng.dirichlet(np.ones(len(v)), size=4000)
        p = wts @ v
        true_ub = float(np.linalg.norm(p - np.clip(p, -h, h), axis=1).min())
        assert R._mesh_box_gap(v, h) <= true_ub + 1e-9


def test_separated_face_on_gap_is_exact():
    v = _box([0.10, 0, 0], [0.01, 0.01, 0.01])        # 상자 +x 면에서 7 cm 떨어짐
    assert abs(R._mesh_box_gap(v, np.array([0.02, 0.03, 0.03])) - 0.07) < 1e-9
