"""10.19 sim_traj — sim 궤적 고르기 · 계획 검사 · 실행 거부 · 재생 순서(순수 부분)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"
sys.path.insert(0, str(TOOLS))
import sim_traj as T  # noqa: E402

POL = Path(__file__).resolve().parents[2] / "deploy" / "policies_retired"   # ★10.20 옛 neo03(시험 고정본)


@pytest.fixture(scope="module")
def side():
    from policy_control import rh_aglt as A
    return A.load_contract(POL / "rh56f1/aglt/left_neo03" / "rh_aglt_contract.json").sides["arm"]


def test_deploy_object_names_map_back_to_the_training_colliders():
    names = ("source240_cvx", "source200_cvx", "shaker_c_cvx")
    assert T.sim_cup_name(names, "shaker_c_real") == "shaker_c_cvx"
    assert T.sim_cup_name(names, "source240_real") == "source240_cvx"
    with pytest.raises(T.SimTrajError):
        T.sim_cup_name(names, "cyl60")


def _hand(n=240, grasp=184, start=172):
    h = np.zeros((n, 6))
    h[:, 0] = 1.57
    h[:, 2:] = 0.05 + 0.01 * np.sin(np.arange(n) / 7.0)[:, None]      # 접근 중 손가락 꼼지락(0.02 rad 안)
    h[start:, 2:] += np.linspace(0, 0.6, n - start)[:, None] * 3       # 닫기
    return h, grasp


def test_close_start_is_where_the_fingers_leave_their_approach_values():
    h, g = _hand()
    k = T.close_start(h, g)
    assert 172 <= k <= 178 and k < g


def test_the_medoid_is_the_path_in_the_middle():
    t = np.linspace(0, 1, 50)[:, None]
    paths = [np.hstack([t, t * 0 + dy, t * 0]) for dy in (0.0, 0.01, 0.05)]
    assert T.medoid(paths) == 1


def _trace(n_env=6, n=300, ok=(2, 5), grasp=184, succ=260):
    tr = {"dt": np.float32(1 / 60), "success": np.zeros((n, n_env), bool), "grasped": np.zeros((n, n_env), bool),
          "palm": np.zeros((n, n_env, 3)), "hand_pd": np.zeros((n, n_env, 6)), "arm_q": np.zeros((n, n_env, 7)),
          "cup": np.zeros((n, n_env, 3))}
    h, _ = _hand(n, grasp)
    for e in range(n_env):
        tr["hand_pd"][:, e] = h
        tr["palm"][:, e, 0] = np.linspace(0, 0.2, n) + 0.001 * e
        if e in ok:
            tr["success"][succ:, e] = True
            tr["grasped"][grasp:, e] = True
    return tr


def test_choose_takes_only_successes_of_the_requested_cup():
    spec = {"envs": [{"cup": o} for _ in range(2) for o in ("source240_cvx", "source200_cvx", "shaker_c_cvx")]}
    rows = [{"env": e, "end": "success" if e in (2, 5) else "time_out"} for e in range(6)]
    pick = T.choose(_trace(), rows, spec, "shaker_c_cvx")
    assert pick["env"] in (2, 5) and pick["n_ok"] == 2 and pick["n"] == 2
    assert pick["success"] == 260 and pick["grasp"] == 184 and 172 <= pick["close"] < 184
    rows = [{"env": e, "end": "time_out"} for e in range(6)]
    with pytest.raises(T.SimTrajError, match="성공 0"):
        T.choose(_trace(), rows, spec, "shaker_c_cvx")


def test_respec_keeps_the_object_cycle_of_the_env_index():
    spec = {"objects": ["a", "b", "c"], "xs": [0.1, 0.2], "ys": [0.3], "cells": [[0.1, 0.3], [0.2, 0.3]], "envs": []}
    out = T.respec(spec, 0.266, 0.080, 3)
    assert [e["cup"] for e in out["envs"]] == ["a", "b", "c"] * 3
    assert out["cells"] == [[0.266, 0.080]] and {(e["x"], e["y"]) for e in out["envs"]} == {(0.266, 0.080)}


def test_plan_arrays_and_checks(side):
    tr = _trace()
    for e in range(6):
        tr["arm_q"][:, e] = np.asarray(side.arm_home)
        tr["hand_pd"][:, e] = np.clip(tr["hand_pd"][:, e], side.hand_lim_lo, side.hand_lim_hi)
    pick = {"env": 2, "success": 260, "grasp": 184, "close": 175}
    plan = T.plan_arrays(tr, pick, side)
    assert plan["q"].shape == (261, 13) and list(plan["names"][:7]) == list(side.arm_joints)
    assert T.check_plan(plan, side) == []
    fast = {**plan, "q": plan["q"].copy()}
    fast["q"][100:, 0] += 0.2                            # 한 프레임에 0.2 rad = 12 rad/s × 0.5
    assert any("속도" in b for b in T.check_plan(fast, side))
    far = {**plan, "q": plan["q"].copy()}
    far["q"][:, 3] = np.asarray(side.arm_hi)[3] + 0.1
    assert any("한계" in b for b in T.check_plan(far, side))


def test_run_refuses_unless_everything_lines_up(side):
    q0 = np.concatenate([side.arm_home, side.hand_open])
    plan = {"q": q0[None, :].repeat(3, 0)}
    meta = {"cup_xy": [0.266, 0.080]}
    pd = {"phase": "TRACKING", "hand_command": "position"}
    ok = T.run_refusals(plan, meta, np.asarray(side.arm_home), [0.268, 0.081, 0.27], pd, False)
    assert ok == []
    assert T.run_refusals(plan, meta, np.asarray(side.arm_home), [0.266, 0.095, 0.27], pd, False)[0].startswith("컵이")
    why = T.run_refusals(plan, meta, np.asarray(side.arm_home) + 0.3, None, {"phase": "HOLD", "hand_command": "admittance"},
                         True)
    assert len(why) == 5


def test_the_schedule_waits_once_right_before_the_fingers_close():
    sched = T.schedule({"close": 3, "q": np.zeros((6, 13))})
    assert sched == [("frame", 0), ("frame", 1), ("frame", 2), ("settle", 3), ("frame", 3), ("frame", 4), ("frame", 5)]


class _FakeServer:
    """server 파일 = {경로: md5}. ssh/scp 명령을 흉내 낸다(10.19 잘린 업로드 회귀)."""

    def __init__(self, files: dict[str, str], scp_md5: str):
        self.files, self.scp_md5, self.log = dict(files), scp_md5, []

    def ssh(self, cmd, timeout=120, check=True):
        self.log.append(cmd)
        if cmd.startswith("md5sum "):
            return self.files.get(cmd.split()[1], "") + "\n"
        if cmd.startswith("for r in"):
            name = cmd.split("-name '")[1].split("'")[0]
            return "\n".join(p for p in self.files if p.endswith("/" + name) and "/job/" not in p)
        if cmd.startswith("cp "):
            src, dst = [s.strip("'") for s in cmd.split(" && ")[0].split()[1:3]]
            self.files[dst] = self.files[src]
        if " && mv " in cmd or cmd.startswith("mv "):
            a, b = [s.strip("'") for s in cmd.split("mv ")[1].split()[:2]]
            self.files[b] = self.files.pop(a)
        if cmd.startswith("rm -f "):
            self.files.pop(cmd.split()[2].strip("'"), None)
        return ""

    def sh(self, cmd, timeout, check=True):
        assert cmd[0] == "scp"
        self.log.append("scp")
        self.files[cmd[-1].split(":", 1)[1]] = self.scp_md5
        return ""


def _fake(monkeypatch, files, scp_md5="good"):
    s = _FakeServer(files, scp_md5)
    monkeypatch.setattr(T, "ssh", s.ssh)
    monkeypatch.setattr(T, "_sh", s.sh)
    return s


def test_upload_reuses_the_training_run_on_the_server(monkeypatch):
    s = _fake(monkeypatch, {"/log/run/nn/a.pth": "good"})
    how = T.upload_ckpt(Path("/x/a.pth"), "good", "/job/src/nn")
    assert how.startswith("server 복사") and s.files["/job/src/nn/a.pth"] == "good" and "scp" not in s.log


def test_a_truncated_copy_is_replaced_not_trusted(monkeypatch):
    s = _fake(monkeypatch, {"/job/src/nn/a.pth": "cut", "/log/run/nn/a.pth": "good"})
    T.upload_ckpt(Path("/x/a.pth"), "good", "/job/src/nn")
    assert s.files["/job/src/nn/a.pth"] == "good"


def test_a_bad_upload_leaves_nothing_behind(monkeypatch):
    s = _fake(monkeypatch, {"/log/run/nn/a.pth": "other"}, scp_md5="cut")
    with pytest.raises(T.SimTrajError, match="md5"):
        T.upload_ckpt(Path("/x/a.pth"), "good", "/job/src/nn")
    assert not any(p.startswith("/job/") for p in s.files)


def test_upload_falls_back_to_scp_then_renames(monkeypatch):
    s = _fake(monkeypatch, {})
    assert T.upload_ckpt(Path("/x/a.pth"), "good", "/job/src/nn") == "scp 업로드"
    assert s.files == {"/job/src/nn/a.pth": "good"}


def test_prep_grid_is_divisible_and_respec_collapses_it_to_the_real_cup():
    assert T.PREP_GRID >= 2                       # _grid 는 (nx-1) 로 나눈다
    prep = {"objects": ["a", "b"], "xs": [0.2, 0.3], "ys": [0.0, 0.1], "dropped": [[0.3, 0.1]],
            "cells": [[0.2, 0.0], [0.2, 0.1], [0.3, 0.0]], "envs": [{"cell": 2}] * 12, "trials": 2}
    s = T.respec(prep, 0.266, 0.08, 3)
    assert s["cells"] == [[0.266, 0.08]] and s["dropped"] == [] and len(s["envs"]) == 6
    assert {e["cell"] for e in s["envs"]} == {0} and [e["cup"] for e in s["envs"][:2]] == ["a", "b"]


def test_a_console_stop_signal_still_lets_the_run_send_the_episode_stop():
    """★10.19 콘솔 정지 = SIGTERM. 그대로면 rclpy 가 context 를 닫아 episode stop 이 안 나간다(실측) — Ctrl-C 처럼 바꾼다."""
    import signal
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        T._interrupt(signal.SIGTERM, None)


def test_the_console_plan_writes_under_the_card_folder(monkeypatch):
    got = {}
    monkeypatch.setattr(T, "cmd_plan", lambda a: got.setdefault("a", a) and 0)
    T.main(["plan", "--policy", "p", "--object", "o", "--cup-topic", "/t", "--out-root", "/x/card_1"])
    assert got["a"].out_root == "/x/card_1" and got["a"].out == ""
