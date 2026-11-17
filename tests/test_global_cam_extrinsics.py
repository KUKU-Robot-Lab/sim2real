"""★11.18 글로벌 D435i 외부 파라미터 풀이(scripts/calib/global_cam_extrinsics.py) — 합성 코너로 자세 복원 · 마커 하나면 거부."""
import pytest
from scipy.spatial.transform import Rotation

import global_cam_extrinsics as gce
import head_view_feedback as hv


def test_selftest():
    gce.selftest()


def test_global_only_markers_join_solve():
    """머리에 70 · 71 만, 글로벌에 70 · 71 · 75 · 76 → 75 · 76 이 초기 자세 부호로 풀이에 들어간다.
    글로벌에서 71 이 가려져 머리 부호 마커가 70 하나면 --init 없이 거부, 틀어진 --init(2 cm · 2°)로는 같은 자세로 돌아온다."""
    cfg, K, dist, R_h, t_h, R_g, eye, hm, gm, _ = gce._synth()
    pick = lambda m, ids: {i: m[i] for i in ids}  # noqa: E731
    res = gce.solve_global(pick(hm, (70, 71)), K, dist, pick(gm, (70, 71, 75, 76)), K, dist, cfg, R_h, t_h)
    g = res["global"]
    assert g["ids"] == [70, 71, 75, 76] and res["init"] == "head"
    assert g["sign_src"] == {70: "head", 71: "head", 75: "global-init", 76: "global-init"}
    dmm, ddeg = hv.extrinsics_diff(g["R"], g["t"], R_g, eye)
    assert dmm < 1.0 and ddeg < 0.05 and g["rms_px"] < 0.01, (dmm, ddeg, g["rms_px"])

    with pytest.raises(ValueError):
        gce.solve_global(pick(hm, (70, 71)), K, dist, pick(gm, (70, 75, 76)), K, dist, cfg, R_h, t_h)
    init = (Rotation.from_euler("xyz", [2, -1, 1], degrees=True).as_matrix() @ R_g, eye + [0.02, -0.01, 0.01])
    res = gce.solve_global(pick(hm, (70, 71)), K, dist, pick(gm, (70, 75, 76)), K, dist, cfg, R_h, t_h, init)
    g = res["global"]
    assert res["init"] == "file" and g["ids"] == [70, 75, 76] and set(g["sign_src"].values()) == {"global-init"}
    dmm, ddeg = hv.extrinsics_diff(g["R"], g["t"], R_g, eye)
    assert dmm < 1.0 and ddeg < 0.05, (dmm, ddeg)
