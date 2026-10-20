"""★10.19 사용자 "좌우 동시 실행" — 팔마다 에피소드 실행기(episode_runner_<팔>) · control 탭 tilt_now · tilt_pour.

실행기 이름이 서비스 · 상태 · 승인 · snapshot 재발행을 가른다. safe_stop 은 그 에피소드의 팔 정책만 멈춘다.
tilt_now = 세팅 자세로(rehome 처럼): 팔마다 plan_carry(검사) → 통과면 그 팔 replay_to_pd — 걸리면 그 팔부터 안 움직인다.
tilt_pour = 지금 자세에서 plan_tilt(검사) → 통과면 replay_to_pd(두 팔) — 검사가 걸리면 재생하지 않는다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "deploy" / "policy_control" / "tools"


def test_approvals_belong_to_their_runner(tmp_path):
    from policy_control.episode_runner_node import NAME, Approvals, write_approval

    path = tmp_path / "approvals.jsonl"
    right, left, both = (Approvals(path, 0.0, r) for r in ("episode_runner_right", "episode_runner_left", NAME))
    write_approval("go_home_start", "op", path, runner="episode_runner_right")
    assert not left.take("go_home_start") and not both.take("go_home_start")   # 같은 노드 이름이어도 다른 실행기 것
    assert right.take("go_home_start") and not right.take("go_home_start")     # 한 줄은 한 번
    path.write_text(path.read_text() + json.dumps({"what": "x", "t": 1.0}) + "\n")   # runner 없는 옛 줄 = 기본 실행기
    assert both.take("x") and not Approvals(path, 0.0, "episode_runner_right").take("x")
    assert json.loads(path.read_text().splitlines()[0])["runner"] == "episode_runner_right"


def test_relay_topic_per_runner():
    from policy_control.episode_ros import OBJECT_RELAY, relay_topic

    assert relay_topic("episode_runner", "CUP") == OBJECT_RELAY.format("CUP") == "/episode/objects/CUP/pose"
    assert relay_topic("episode_runner_right", "CUP") == "/episode/right/objects/CUP/pose"
    assert relay_topic("episode_runner_left", "CUP") == "/episode/left/objects/CUP/pose"


def test_one_arm_episodes_stop_only_their_arm():
    from policy_control import episode_spec as S

    for side in ("right", "left"):
        assert S.episode_sides(S.load(REPO / f"config/episodes/grasp_rg3_{side}.yaml")) == [side]
    assert S.episode_sides(S.load(REPO / "config/episodes/tilt_base_b.yaml")) == ["right", "left"]


def test_episode_cmd_targets_the_named_runner(monkeypatch, capsys):
    sys.path.insert(0, str(TOOLS))
    import episode_cmd as E

    seen = {}
    monkeypatch.setenv("ROS_DOMAIN_ID", "98")
    monkeypatch.setattr(E, "read_status", lambda timeout=3.0, runner=E.NAME: seen.setdefault("status", runner) and
                        {"episode": "grasp_rg3_left", "phase": "READY", "next_action": "go_home_start", "busy": False})
    monkeypatch.setattr(E, "call", lambda action, timeout=5.0, runner=E.NAME: (seen.setdefault("call", runner), (True, []))[1])
    rows = []
    monkeypatch.setattr(E, "write_approval", lambda what, op, path, runner=E.NAME: rows.append((what, runner)))
    assert E.main(["run", "--approve", "episode:grasp_rg3_left", "--execute", "--runner", "episode_runner_left"]) == 0
    assert seen == {"status": "episode_runner_left", "call": "episode_runner_left"}
    assert rows == [("episode:grasp_rg3_left", "episode_runner_left")]
    assert E.main(["stop", "--execute", "--runner", "rm -rf"]) == 1               # 이름 형식 밖은 거부


# ── tilt_pour(지금 자세에서 붓기) ───────────────────────────────────────
@pytest.fixture()
def tilt_pour(tmp_path, monkeypatch):
    sys.path.insert(0, str(TOOLS))
    import tilt_pour as T

    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "policy.yaml").write_text("rate_scale: 1.0\nstart_gap_rad: 0.05\n")
    calls = []

    def fake(plan_rc):
        def run(argv):
            calls.append(argv)
            if argv[1].endswith("plan_tilt.py"):
                out = Path(argv[argv.index("--out") + 1])
                out.mkdir(parents=True, exist_ok=True)
                checks = {"src_start_cup_in_sim_box": {"value": [0.11, -0.27, 0.43], "limit": [[0.18, -0.2, 0.34], [0.32, -0.05, 0.48]],
                                                       "ok": plan_rc == 0}}
                (out / "check.json").write_text(json.dumps({"ok": plan_rc == 0, "T_end_s": 21.4, "checks": checks}))
                return plan_rc
            return 0
        monkeypatch.setattr(T, "run", run)
    return T, asset, tmp_path / "tilt", calls, fake


def test_tilt_pour_refuses_to_move_when_the_plan_check_fails(tilt_pour, capsys):
    T, asset, root, calls, fake = tilt_pour
    fake(2)
    assert T.main(["--asset", str(asset), "--out-root", str(root), "--no-record", "--execute"]) == 2
    assert len(calls) == 1 and calls[0][1].endswith("plan_tilt.py")               # replay 를 부르지 않았다
    assert "src_start_cup_in_sim_box" in capsys.readouterr().out                  # 콘솔 로그에 걸린 항목(rc 만이 아니라)


def test_tilt_pour_replays_both_arms_from_the_current_pose(tilt_pour):
    T, asset, root, calls, fake = tilt_pour
    fake(0)
    assert T.main(["--asset", str(asset), "--out-root", str(root), "--no-record"]) == 0     # 계획만
    assert len(calls) == 1
    assert T.main(["--asset", str(asset), "--out-root", str(root), "--no-record", "--execute"]) == 0
    replay = calls[-1]
    assert replay[1].endswith("replay_to_pd.py") and "--execute" in replay
    assert replay[replay.index("--joints") + 1] == ",".join([*(f"r_aj_{i}" for i in range(1, 8)), *(f"l_aj_{i}" for i in range(1, 8))])
    assert replay[replay.index("--max-start-gap") + 1] == "0.05" and replay[replay.index("--rate-scale") + 1] == "1.0"
    assert replay[replay.index("--npz") + 1].endswith("/tilt_cmd.npz") and "_pour" in replay[replay.index("--npz") + 1]
    assert len({c[c.index("--out") + 1] for c in calls if "--out" in c}) == 2                # 두 번 눌러도 기록 폴더가 따로


# ── tilt_now(세팅 자세로, rehome 처럼) ──────────────────────────────────
GOAL = {"right": [-0.58, 0.29, -0.01, 1.61, 0.12, 0.14, 0.68], "left": [0.59, -0.40, -0.07, 1.70, -0.06, -0.27, -0.55]}
HOME = {"right": [-1.21, 0.2, 0.65, 1.76, 0.37, 0.58, 0.67], "left": [1.21, -0.2, -0.65, 1.76, -0.38, -0.58, -0.66]}


@pytest.fixture()
def tilt_now(tmp_path, monkeypatch):
    sys.path.insert(0, str(TOOLS))
    import tilt_now as T

    asset = tmp_path / "asset"
    asset.mkdir()
    (tmp_path / "pour_start.yaml").write_text(json.dumps(GOAL))
    (asset / "policy.yaml").write_text(f"start_gap_rad: 0.05\nstart_pose: {tmp_path / 'pour_start.yaml'}\n")
    monkeypatch.setattr(T, "current_q", lambda: {s: list(v) for s, v in HOME.items()})
    calls, qs = [], []

    def fake(plan_rc: dict):
        def run(argv):
            calls.append(argv)
            if argv[1].endswith("plan_carry.py"):
                side = argv[argv.index("--sides") + 1]
                if "--q-json" in argv:
                    qs.append((side, json.loads(Path(argv[argv.index("--q-json") + 1]).read_text())))
                out = Path(argv[argv.index("--out") + 1])
                out.mkdir(parents=True, exist_ok=True)
                rc = plan_rc.get(side, 0)
                chk = {side: {"ok": rc == 0, "plan_ok": True, "arrived": True, "table_clear_m": -0.02 if rc else 0.05, "vmax_rad_s": 1.0}}
                (out / "check.json").write_text(json.dumps({"ok": rc == 0, "checks": chk, "arms_clear": {"ok": True}}))
                return rc
            return 0
        monkeypatch.setattr(T, "run", run)
    return T, asset, tmp_path / "tilt", calls, qs, fake


def test_tilt_now_moves_right_then_left_each_after_its_own_check(tilt_now):
    T, asset, root, calls, qs, fake = tilt_now
    fake({})
    assert T.main(["--asset", str(asset), "--out-root", str(root), "--execute"]) == 0
    kinds = [(Path(c[1]).name, c[c.index("--sides") + 1] if "--sides" in c else c[c.index("--joints") + 1]) for c in calls]
    assert kinds == [("plan_carry.py", "right"), ("replay_to_pd.py", ",".join(f"r_aj_{i}" for i in range(1, 8))),
                     ("plan_carry.py", "left"), ("replay_to_pd.py", ",".join(f"l_aj_{i}" for i in range(1, 8)))]
    for c in calls:
        if c[1].endswith("replay_to_pd.py"):
            assert "--execute" in c and c[c.index("--max-start-gap") + 1] == "0.05"
            assert c[c.index("--npz") + 1].endswith("/carry_cmd.npz")
        else:
            assert "--q-json" not in c                                         # 실행이면 팔마다 그때 실측에서 계획


def test_tilt_now_stops_before_the_arm_whose_check_fails(tilt_now, capsys):
    T, asset, root, calls, qs, fake = tilt_now
    fake({"left": 2})
    assert T.main(["--asset", str(asset), "--out-root", str(root), "--execute"]) == 2
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py", "replay_to_pd.py", "plan_carry.py"]   # 왼팔은 재생 안 함
    assert "탁자 여유" in capsys.readouterr().out
    calls.clear()
    fake({"right": 2})
    assert T.main(["--asset", str(asset), "--out-root", str(root), "--execute"]) == 2
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py"]                                       # 오른팔이 걸리면 아무것도


def test_tilt_now_plan_only_plans_the_left_arm_as_if_the_right_had_arrived(tilt_now):
    T, asset, root, calls, qs, fake = tilt_now
    fake({})
    assert T.main(["--asset", str(asset), "--out-root", str(root)]) == 0
    assert [Path(c[1]).name for c in calls] == ["plan_carry.py", "plan_carry.py"]                     # 재생 없음
    assert qs[0] == ("right", HOME)
    assert qs[1] == ("left", {"right": GOAL["right"], "left": HOME["left"]})
    assert T.main(["--asset", str(asset), "--out-root", str(root), "--sides", "both"]) == 1
