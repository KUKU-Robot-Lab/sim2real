"""★11.06 scene snapshot 은 FP++ live 추적을 먼저 본다 — …/pose 는 다시 찍기 전까지 옛 자리를 계속 낸다."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from policy_control.episode_ros import SNAPSHOT_WINDOW_S, RosExecutor, pick_scene_pose

STALE, LIVE = (0.2907, -0.3213, 0.289), (0.3048, -0.3396, 0.290)   # 11.06 150952 실기 bag 값


def _fr(pos, n: int = 10, step=(0.0, 0.0, 0.0)):
    now = time.monotonic()
    return [(tuple(p + k * s for p, s in zip(pos, step)), (1.0, 0.0, 0.0, 0.0), now - 0.01 * k) for k in range(n)]


def _me(tmp_path, live):
    return SimpleNamespace(fpp_rescan=False, ep=None, _lock=threading.Lock(), cancel=threading.Event(), _relay={},
                           run_dir=tmp_path, _frames={"SRC_PINK": _fr(STALE)}, _live={"SRC_PINK": live})


def test_snapshot_takes_the_live_xy_over_the_stale_static_pose(tmp_path):
    res, got = RosExecutor.snapshot(_me(tmp_path, _fr(LIVE)), SimpleNamespace(objects=["SRC_PINK"]), None)
    assert res.status == "completed" and got["SRC_PINK"][0] == (LIVE[0], LIVE[1], STALE[2])   # z 는 known_z 쪽
    scene = json.loads((tmp_path / "scene.json").read_text())
    assert scene["source"] == {"SRC_PINK": "live"} and abs(scene["live_static_gap_m"]["SRC_PINK"] - 0.0224) < 1e-3


def test_snapshot_falls_back_to_the_static_pose_without_live_frames(tmp_path):
    res, got = RosExecutor.snapshot(_me(tmp_path, []), SimpleNamespace(objects=["SRC_PINK"]), None)
    assert res.status == "completed" and got["SRC_PINK"][0] == STALE


def test_moving_live_never_falls_back_to_the_stale_pose():
    snap, kind, why = pick_scene_pose(_fr(LIVE, step=(0.003, 0, 0)), _fr(STALE), False, 99.0)
    assert snap is None and "live 움직임" in why


def test_short_live_waits_one_window_before_static():
    assert pick_scene_pose(_fr(LIVE, n=2), _fr(STALE), False, 0.1)[0] is None
    assert pick_scene_pose(_fr(LIVE, n=2), _fr(STALE), False, SNAPSHOT_WINDOW_S)[1] == "pose"


def test_rescanned_static_pose_wins_over_live():
    snap, kind, _ = pick_scene_pose(_fr(LIVE), _fr(STALE), True, 0.0)
    assert kind == "pose" and snap[0] == STALE
