"""한 컨테이너에서 같은 모양 · 다른 색 컵 여럿 — 색 판정 · 검출 배정 · 자세 대표값(순수, GPU 없음).

10.08 사용자: "컨테이너 하나에, 색깔이 다른 동일 모델을 추출" · 파랑 · 핑크 · 노랑 3 개. 컵은 가만히 있으니 한 번 찍는다.
색 값은 10.08 arm4090 머리 카메라 한 장에서 잰 것(컵 몸통 · 안쪽 · 테이블).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fpp_color_pick as C  # noqa: E402

BLUE, BLUE_IN = (1, 135, 190), (0, 82, 165)
YELLOW, YELLOW_RIM = (249, 195, 20), (253, 238, 0)
PINK = (193, 96, 166)
TABLE = (103, 112, 112)


def _scene(*patches):
    """검은 테이블 위 색 칸들 — (rgb, (y0, y1, x0, x1)). 칸마다 마스크를 돌려준다."""
    img = np.empty((120, 200, 3), np.uint8)
    img[:] = TABLE
    masks = []
    for rgb, (y0, y1, x0, x1) in patches:
        img[y0:y1, x0:x1] = rgb
        m = np.zeros(img.shape[:2], bool)
        m[y0:y1, x0:x1] = True
        masks.append(m)
    return img, masks


@pytest.mark.parametrize("rgb, color", [(BLUE, "blue"), (BLUE_IN, "blue"), (YELLOW, "yellow"), (YELLOW_RIM, "yellow"),
                                        (PINK, "pink")])
def test_each_measured_cup_colour_scores_high_only_for_its_own_name(rgb, color):
    img, (m,) = _scene((rgb, (10, 60, 10, 60)))
    scores = {c: C.color_fraction(img, m, c) for c in C.COLORS}
    assert scores[color] > 0.9
    assert all(v < 0.05 for c, v in scores.items() if c != color), scores


def test_the_black_table_and_an_empty_mask_score_zero():
    img, (m,) = _scene((TABLE, (0, 50, 0, 50)))
    assert all(C.color_fraction(img, m, c) == 0.0 for c in C.COLORS)
    assert C.color_fraction(img, np.zeros(img.shape[:2], bool), "blue") == 0.0


def test_a_cup_with_its_dark_inside_still_counts_by_the_coloured_share():
    """마스크에 그림자 · 테이블이 섞여도 색 비율로 고른다 — 절반이 컵 색이면 0.5."""
    img, (m,) = _scene((TABLE, (10, 60, 10, 60)))
    img[10:35, 10:60] = YELLOW
    assert C.color_fraction(img, m, "yellow") == pytest.approx(0.5)


def test_three_cups_each_go_to_their_own_colour_whatever_the_detection_order():
    img, masks = _scene((PINK, (10, 50, 10, 40)), (YELLOW, (10, 50, 60, 90)), (BLUE, (10, 50, 110, 140)))
    want = {"cyl60": "yellow", "cyl60_blue": "blue", "cyl60_pink": "pink"}
    got = C.assign(img, masks, want)
    assert got == {"cyl60_pink": 0, "cyl60": 1, "cyl60_blue": 2}
    got_rev = C.assign(img, masks[::-1], want)
    assert got_rev == {"cyl60_pink": 2, "cyl60": 1, "cyl60_blue": 0}


def test_a_missing_colour_is_left_out_not_given_another_cup():
    """핑크 컵이 없으면 핑크는 비운다 — 남은 노랑 · 파랑 컵을 대신 주지 않는다(정책 입력이다)."""
    img, masks = _scene((YELLOW, (10, 50, 60, 90)), (BLUE, (10, 50, 110, 140)))
    got = C.assign(img, masks, {"cyl60": "yellow", "cyl60_blue": "blue", "cyl60_pink": "pink"})
    assert got == {"cyl60": 0, "cyl60_blue": 1}


def test_two_detections_of_the_same_colour_go_to_the_stronger_one_once():
    """같은 색 후보가 둘이면(뚜껑 · 누운 원통) 색 비율이 높은 하나만 — 한 후보를 두 물체에 주지 않는다."""
    img, masks = _scene((BLUE, (10, 50, 10, 40)), (TABLE, (10, 50, 60, 90)))
    img[10:25, 60:90] = BLUE                     # 둘째 후보는 일부만 파랑
    got = C.assign(img, masks, {"cyl60_blue": "blue", "cyl60": "yellow"})
    assert got == {"cyl60_blue": 0}


def test_unknown_colour_name_is_refused():
    img, (m,) = _scene((BLUE, (0, 10, 0, 10)))
    with pytest.raises(ValueError):
        C.color_fraction(img, m, "purple")


def _pose(x, y, z, yaw=0.0):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def test_the_snapshot_pose_is_the_frame_nearest_the_median_and_reports_the_spread():
    poses = [_pose(0.100, 0.0, 0.5), _pose(0.101, 0.0, 0.5), _pose(0.102, 0.0, 0.5, 0.1), _pose(0.103, 0.0, 0.5),
             _pose(0.30, 0.0, 0.5)]
    T, spread_mm = C.representative(poses)
    np.testing.assert_allclose(T, poses[2])                 # 위치 중앙값(0.102)인 장 — 회전도 그 장 것
    assert spread_mm == pytest.approx(198.0, abs=0.1)      # 튄 한 장이 흔들림으로 드러난다
    with pytest.raises(ValueError):
        C.representative([])


def test_snapshot_node_quaternion_matches_scipy_for_any_rotation():
    """대표 자세를 PoseStamped 로 낼 때의 행렬 → 사원수(x, y, z, w) — 180° 가까운 회전도."""
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "nodes"))
    import fpp_snapshot_node as N
    rng = np.random.default_rng(0)
    rots = list(Rotation.random(200, random_state=1)) + [Rotation.from_rotvec([np.pi, 0, 0]),
                                                        Rotation.from_rotvec([0, np.pi * 0.999, 0])]
    for r in rots:
        q, want = N._quat_xyzw(r.as_matrix()), r.as_quat()
        assert min(np.abs(q - want).max(), np.abs(q + want).max()) < 1e-9
    del rng


def test_snapshot_config_needs_every_key_and_a_known_colour(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "nodes"))
    import fpp_snapshot_node as N
    good = {"name": "cyl60", "color": "yellow", "mesh_path": "m.obj", "mesh_scale_to_meters": 1.0,
            "pose_topic": "/perception_plus_plus/cyl60/pose"}
    f = tmp_path / "g.yaml"
    f.write_text(json.dumps({"objects": [good]}))
    assert N.load_config(f)["objects"][0]["name"] == "cyl60"
    for bad in ({**good, "color": "purple"}, {k: v for k, v in good.items() if k != "pose_topic"}):
        f.write_text(json.dumps({"objects": [bad]}))
        with pytest.raises(ValueError):
            N.load_config(f)
    f.write_text(json.dumps({"objects": []}))
    with pytest.raises(ValueError):
        N.load_config(f)
