"""11.20 tilt_now --goal-poses — act_tilt 카드가 브리지 전에 오른팔만 ACT 시연 시작 자세로(실기 · ROS 없음, run 을 가짜로)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "deploy" / "policy_control" / "tools"
R = [-0.5, 0.2, 0.2, 1.7, 0.0, 0.5, 0.4]


@pytest.fixture()
def tilt_now(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "rclpy", None)           # 패치를 빼먹은 시험이 실 ROS 로 가지 않게
    import tilt_now as T
    monkeypatch.setattr(T, "current_q", lambda: {"right": [0.0] * 7, "left": [0.0] * 7})
    calls, plan_rc = [], {}

    def run(argv):
        calls.append(argv)
        if argv[1].endswith("plan_carry.py"):
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "check.json").write_text(json.dumps({"ok": not plan_rc, "checks": {}, "arms_clear": {"ok": True}}))
            return plan_rc.get("right", 0)
        return 0

    monkeypatch.setattr(T, "run", run)
    poses = tmp_path / "poses.yaml"
    poses.write_text(yaml.safe_dump({"poses": [{"id": 7, "ep": 3, "right": R}, {"id": 9, "right": [0.0] * 7}]}))
    return T, calls, plan_rc, poses, tmp_path / "tilt"


def test_goal_pose_moves_only_the_right_arm_to_the_picked_pose(tilt_now):
    T, calls, _, poses, root = tilt_now
    argv = ["--out-root", str(root), "--sides", "right", "--goal-poses", str(poses), "--pose-id", "7", "--execute"]
    assert T.main(argv) == 0
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py", "replay_to_pd.py"]
    plan, play = calls
    assert plan[plan.index("--sides") + 1] == "right"
    assert json.loads(Path(plan[plan.index("--goal-json") + 1]).read_text()) == {"right": R}
    assert play[play.index("--joints") + 1] == ",".join(T.ARM_JOINTS["right"])   # 오른팔 관절만 · 손 명령 없음


def test_bad_pose_file_or_failed_check_never_replays(tilt_now):
    T, calls, plan_rc, poses, root = tilt_now
    base = ["--out-root", str(root), "--sides", "right", "--goal-poses", str(poses), "--execute"]
    assert T.main(base + ["--pose-id", "nope"]) == 1 and calls == []
    with pytest.raises(ValueError):
        T.pick_goal(poses, ["right", "left"])                  # 왼팔 관절 없는 목록으로 두 팔을 계획하지 않는다
    plan_rc["right"] = 3
    assert T.main(base + ["--seed", "0"]) == 3
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py"]   # 걸리면 이동 없음 → 카드 rc≠0 → runner 가 브리지를 건너뜀


def _poses(path, rows):
    path.write_text(yaml.safe_dump({"poses": rows}))
    return path


def test_head_follows_the_same_pose_after_the_arm_and_its_failure_stops_the_card(tilt_now, monkeypatch):
    T, calls, _, _, root = tilt_now
    poses = _poses(root.parent / "h.yaml", [{"id": "a", "right": R, "head": [-12.5, 61.0]}])
    head = "python3 /r/scripts/head_move.py --config /r/config/h.yaml"
    base = ["--out-root", str(root), "--sides", "right", "--goal-poses", str(poses), "--head-argv", head]
    assert T.main(base) == 0                                    # dry: 계획만, 목 명령 없음
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py"]
    calls.clear()
    assert T.main(base + ["--execute"]) == 0
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py", "replay_to_pd.py", "head_move.py"]
    assert calls[-1] == head.split() + ["--pan", "-12.5", "--tilt", "61.0", "--execute"]   # yaml 인코더 deg 그대로
    meta = json.loads(next(root.rglob("meta.json")).read_text())
    assert meta["head_goal_deg"] == [-12.5, 61.0] and meta["goal_pose"]["id"] == "a"
    calls.clear()
    monkeypatch.setattr(T, "run", lambda argv: calls.append(argv) or (5 if argv[1].endswith("head_move.py") else 0))
    assert T.main(base + ["--execute"]) == 5                     # 목 실패 → rc≠0 → runner 가 브리지를 건너뜀
    no_head = _poses(root.parent / "n.yaml", [{"id": "b", "right": R}])
    calls.clear()
    assert T.main(["--out-root", str(root), "--sides", "right", "--goal-poses", str(no_head), "--head-argv", head, "--execute"]) == 1
    assert calls == []                                           # head 없는 자세면 팔도 움직이지 않는다


def test_palm_tilt_filter_drops_out_of_range_rows_only_from_the_random_pick(tilt_now, tmp_path, capsys):
    T = tilt_now[0]
    rows = [{"id": i, "right": R, "palm_tilt_deg": d} for i, d in enumerate([50.0, 76.7, 95.0])] + [{"id": 9, "right": R}]
    p = _poses(tmp_path / "p.yaml", rows)
    assert {T.pick_goal(p, ["right"], seed=s, palm_range=(61, 92))["id"] for s in range(30)} == {1}
    assert T.pick_goal(p, ["right"], "0", palm_range=(61, 92))["id"] == 0          # id 지정은 필터와 무관
    assert len({T.pick_goal(p, ["right"], seed=s)["id"] for s in range(60)}) == 4   # off
    q = _poses(tmp_path / "q.yaml", [{"id": 1, "right": R}, {"id": 2, "right": R}])
    assert len({T.pick_goal(q, ["right"], seed=s, palm_range=(61, 92))["id"] for s in range(30)}) == 2
    assert "palm_tilt_deg 없음" in capsys.readouterr().out
    with pytest.raises(ValueError):
        T.pick_goal(_poses(tmp_path / "r.yaml", rows[:1]), ["right"], palm_range=(61, 92))
