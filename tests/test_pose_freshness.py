"""FP++ 가 실패 뒤 이전 pose 를 새 시각으로 다시 내는 것을 거르는 규칙 — 순수부 테스트."""
from pose_freshness import StaleGuard

P1, P2 = (0.1, -0.2, 0.5), (0.11, -0.2, 0.5)


def test_same_value_while_the_head_is_still_is_a_normal_hold():
    g = StaleGuard(min_move_deg=0.5)
    assert not g.is_stale("cup", P1, (10.0, -30.0))
    assert not g.is_stale("cup", P1, (10.05, -30.0))       # 센서 잡음 수준의 흔들림
    assert not g.is_stale("cup", P1, (10.0, -30.0))


def test_same_camera_frame_value_after_the_head_moved_is_stale():
    """고정된 물체는 목이 돌면 카메라 프레임 값이 바뀐다. 똑같다면 그 값은 옛 자세에서 본 것이다."""
    g = StaleGuard(min_move_deg=0.5)
    assert not g.is_stale("cup", P1, (10.0, -30.0))
    assert g.is_stale("cup", P1, (30.0, -30.0))
    assert g.is_stale("cup", P1, (30.0, -30.0))            # 계속 낡은 값이다


def test_a_new_value_resets_the_reference_head_angle():
    g = StaleGuard(min_move_deg=0.5)
    g.is_stale("cup", P1, (10.0, -30.0))
    assert g.is_stale("cup", P1, (30.0, -30.0))
    assert not g.is_stale("cup", P2, (30.0, -30.0))        # 새 검출
    assert not g.is_stale("cup", P2, (30.1, -30.0))        # 새 자세 기준으로 다시 정지 상태
    assert g.is_stale("cup", P2, (50.0, -30.0))


def test_objects_are_tracked_independently():
    g = StaleGuard(min_move_deg=0.5)
    g.is_stale("a", P1, (0.0, -30.0))
    g.is_stale("b", P1, (0.0, -30.0))
    assert g.is_stale("a", P1, (20.0, -30.0))
    assert not g.is_stale("b", P2, (20.0, -30.0))


def test_without_a_head_angle_nothing_is_called_stale():
    g = StaleGuard(min_move_deg=0.5)
    assert not g.is_stale("cup", P1, None)
    assert not g.is_stale("cup", P1, None)


def test_tilt_motion_also_counts():
    g = StaleGuard(min_move_deg=0.5)
    g.is_stale("cup", P1, (0.0, -20.0))
    assert g.is_stale("cup", P1, (0.0, -30.0))
