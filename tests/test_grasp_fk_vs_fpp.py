"""grasp_fk_vs_fpp — 단계 · 끊김 · 놓침 · 기록 시점 · bag 찾기(순수 로직). bag · FK 는 실제 에피소드로 따로 확인했다(11.04 131817)."""
from __future__ import annotations

import numpy as np

import grasp_fk_vs_fpp as G

LINES = [
    {"timestamp": "2026-11-04T13:18:56.000", "event": "enter", "state_id": "grasp_blue"},
    {"timestamp": "2026-11-04T13:19:04.000", "event": "exit", "state_id": "grasp_blue", "result": "COMPLETED"},
    {"timestamp": "2026-11-04T13:20:53.000", "event": "enter", "state_id": "place_blue"},
    {"timestamp": "2026-11-04T13:21:04.000", "event": "failure", "state_id": "place_blue"},
    {"timestamp": "2026-11-04T13:21:04.100", "event": "safe_stop", "state_id": "place_blue", "result": "FAILURE"},
    {"timestamp": "2026-11-04T13:21:43.000", "event": "stop", "state_id": "go_home_start"},     # 운영자 stop 은 단계가 아니다
]


def test_steps_end_at_the_first_terminal_event_and_gaps_are_named_after_the_previous_step():
    steps = G.parse_steps(LINES)
    assert [(s[0], round(s[2] - s[1], 1), s[3]) for s in steps] == [("grasp_blue", 8.0, "COMPLETED"), ("place_blue", 11.0, "failure")]
    t0 = steps[0][1]
    assert G.phase_at(steps, t0 + 1) == "grasp_blue"
    assert G.phase_at(steps, t0 + 20) == "after_grasp_blue"
    assert G.record_times(steps, "grasp_blue", t0 + 999) == {"end": t0 + 8, "end_plus_1s": t0 + 9, "before_next": steps[1][1] - 0.2}
    assert G.record_times(steps, "place_blue", steps[1][2] + 5)["before_next"] == steps[1][2] + 4.8     # 다음 단계 없음 = bag 끝


def test_stale_is_age_over_half_a_second_and_lost_needs_a_sustained_run():
    grid = np.arange(0.0, 3.0, 0.1)
    recv = np.array([0.05, 0.3, 0.6, 2.5])                 # 0.6 → 2.5 사이 1.9 s 비었다
    st = G.stale_mask(grid, recv)
    assert not st[(grid > 0.05) & (grid < 1.05)].any() and st[(grid > 1.15) & (grid < 2.45)].all() and not st[grid >= 2.5].any()
    assert st[0] and grid[0] < recv[0]                       # 받기 전도 끊김
    bad = st.copy()
    bad[5] = True                                            # 0.5 s 의 한 칸 튐은 놓침이 아니다
    assert 1.05 < G.first_sustained(grid[1:], bad[1:], 1.0) < 1.25         # 0.6 s 에 마지막으로 받고 0.5 s 뒤
    assert G.first_sustained(grid, np.zeros_like(st), 1.0) is None


def test_axis_angle_ignores_twist_about_the_cup_axis():
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    R = rx(np.radians(20))
    assert abs(G.axis_deg(R, R @ rz(1.0))) < 1e-6 and abs(G.rot_deg(R, R @ rz(1.0)) - np.degrees(1.0)) < 1e-6
    assert abs(G.axis_deg(np.eye(3), R) - 20.0) < 1e-6


def test_bag_is_the_latest_one_started_before_the_episode(tmp_path, monkeypatch):
    for n in ("20261104_125800_episode_x", "20261104_131813_episode_x", "20261104_131900_episode_x", "20261104_131810_episode_y"):
        (tmp_path / n).mkdir()
    monkeypatch.setattr(G, "BAG_DIR", tmp_path)
    assert G.find_bag(tmp_path / "20261104_131817_x", "x").name == "20261104_131813_episode_x"
