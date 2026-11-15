"""fit_board: 알려진 관절 영점 오차로 만든 접촉 점에서 그 관절·크기를 되찾는다(로봇 없음)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy/policy_control/tools'))
import board_jog as B  # noqa: E402
import fit_board as F  # noqa: E402

pytestmark = pytest.mark.skipif(not B.URDF_DEFAULT.exists(), reason='hdgp URDF 없음')


def test_recovers_single_joint_offset_from_varied_poses():
    kin = B.Kin(B.URDF_DEFAULT, 'left', np.asarray(B.POINT))
    true = np.zeros(7)
    true[5] = np.radians(-6.0)
    rng = np.random.default_rng(0)
    base = np.radians([25, -10, 15, 80, 75, -5, -5])
    pts = []
    for k in range(8):
        q = base + np.radians(rng.uniform(-20, 20, 7))
        pts.append((f'p{k}', q, np.asarray(B.POINT), kin.tip(q + true)[:3, 3]))   # 실제 = FK(읽기 + 오차)
    rows = F.report(F.Problem(kin, pts, None), max_joints=1, top=2)
    assert rows[0]['joints'] == [6]
    assert rows[0]['deg'][0] == pytest.approx(-6.0, abs=0.05)
    assert rows[0]['jog_rms_mm'] < 0.1
