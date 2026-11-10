"""★11.07 tilt 직전 T_in 실측(tin_live) — 실기 bag 151437 의 분석 창(logs/analysis/tilt_tin_1107 hold_windows.json)과 1 mm 안,
흔들림 · 프레임 부족이면 fallback, place 노드가 읽는 json 왕복."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from policy_control import tin_live as T

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "policy_control" / "derived" / "tin_live_151437.npz"


def _fk(side: str):
    return T.palm_fk(side)          # 실행기와 같은 FK — 분석 값과 1 mm 안이면 FK 도 같다


@pytest.mark.parametrize("s,side", [("r", "right"), ("l", "left")])
def test_matches_analysis_within_1mm(s, side):
    D = dict(np.load(FIX))
    t0, t1 = D[f"{s}_win"]
    r = T.measure(D[f"{s}_t"], D[f"{s}_hs"], D[f"{s}_X"], D[f"{s}_arm_t"], D[f"{s}_arm_q"], _fk(side), t0, t1,
                  z_off=T.Z_OFF[side])
    assert r["ok"], r
    assert r["n_fresh"] == int(D[f"{s}_n_fresh"])
    got, ref = np.asarray(r["T_in"]), D[f"{s}_T_in"]
    assert np.linalg.norm(got[:3, 3] - ref[:3, 3]) < 1e-3
    assert T.ang_deg(got[:3, :3], ref[:3, :3]) < 0.5


def _synthetic(n=20, rate=6.0):
    """손바닥 = 원점(FK 항등), 컵 약 76 mm(고정 T_in_r 74 mm 근처)에서 표본마다 10 µm 씩 바뀜(갱신), 6 Hz."""
    t = np.arange(n + 1) / rate
    X = np.tile([0.01, 0.03, 0.07, 1.0, 0.0, 0.0, 0.0], (n + 1, 1))
    X[:, 0] += np.arange(n + 1) * 1e-5
    return t, t + 100.0, X, np.array([-1.0, 10.0]), np.zeros((2, 7)), (lambda q: np.eye(4))


def test_rejects_jitter_and_few_frames_then_fallback(tmp_path):
    t, hs, X, at, aq, fk = _synthetic()
    assert T.measure(t, hs, X, at, aq, fk, 0.0, 10.0)["ok"]
    Xj = X.copy()
    Xj[:, 0] += 0.012 * (np.arange(len(X)) % 2)
    bad = T.measure(t, hs, Xj, at, aq, fk, 0.0, 10.0)
    assert not bad["ok"] and "위치 흔들림" in bad["reason"]
    few = T.measure(t, hs, X, at, aq, fk, 0.0, 0.5)
    assert not few["ok"] and "갱신" in few["reason"]
    stale = T.measure(t, np.full_like(hs, 5.0), X, at, aq, fk, 0.0, 10.0)   # stamp 가 안 늘면 갱신 아님
    assert stale["n_fresh"] == 0 and not stale["ok"]

    fixed = tmp_path / "tin_fixed.json"
    M = np.eye(4); M[:3, 3] = [0.01, 0.02, 0.07]
    fixed.write_text(json.dumps({"T_in_r": {"T_meas_fullrot": M.tolist(), "T": np.eye(4).tolist()}}))
    rec = T.record(bad, "right", fixed, node="tilt_blue")
    assert rec["source"] == "fallback" and "위치 흔들림" in rec["fallback_reason"] and rec["pos_m"] == [0.01, 0.02, 0.07]
    p = tmp_path / "tin.json"
    p.write_text(json.dumps(rec))
    rel_p, rel_q, d = T.load_tin(p, "right")
    assert np.allclose(rel_p, [0.01, 0.02, 0.07]) and np.allclose(rel_q, [1, 0, 0, 0]) and d["node"] == "tilt_blue"
    with pytest.raises(T.TinError):
        T.load_tin(p, "left")


def test_late_arm_holds_end_value_ratio(tmp_path):
    """팔 표본이 끝보다 ARM_HOLD_S 안 앞서면 끝값 유지 · 더 늦은 프레임만 버림 · 갱신 비율 0.3."""
    t, hs, X, at, aq, fk = _synthetic()
    end = float(t[-1])
    held = T.measure(t, hs, X, np.array([-1.0, end - 0.15]), aq, fk, 0.0, 10.0)
    assert held["ok"] and "n_late_dropped" not in held
    cut = T.measure(t, hs, X, np.array([-1.0, end - 1.0]), aq, fk, 0.0, 10.0)
    assert cut["ok"] and cut["n_late_dropped"] == 5 and cut["n_fresh"] == held["n_fresh"] - 5
    rep = np.repeat(np.arange(len(t)), 3)              # 같은 프레임 3번 재발행 = 갱신 비율 1/3
    tt = np.arange(len(rep)) / 18.0
    assert T.measure(tt, hs[rep], X[rep], np.array([-1.0, 10.0]), aq, fk, 0.0, 10.0)["ok"]


@pytest.mark.parametrize("slip_m", [0.020, 0.030])
def test_slipped_low_tin_is_used_and_gap_to_fixed_is_recorded(tmp_path, slip_m):
    """★11.10 사용자: 낮게 잡아 컵이 20~30 mm 미끄러진 정상 실측도 그대로 쓴다 — 고정값과 차이는 vs_fixed 에 기록만."""
    t, hs, X, at, aq, fk = _synthetic()
    fixed = tmp_path / "f.json"
    M = np.eye(4); M[:3, 3] = X[0, :3]
    fixed.write_text(json.dumps({"T_in_r": {"T_meas_fullrot": M.tolist()}}))
    Xs = X.copy()
    Xs[:, 2] -= slip_m
    ok = T.measure(t, hs, Xs, at, aq, fk, 0.0, 10.0)
    assert ok["ok"], ok
    r = T.record(ok, "right", fixed)
    assert r["source"] == "measured" and np.allclose(r["T_in"], ok["T_in"], atol=1e-6)
    vs = r["vs_fixed"]
    assert vs["dz_mm"] == pytest.approx(-slip_m * 1000, abs=0.5) and vs["pos_mm"] == pytest.approx(slip_m * 1000, abs=0.5)
    assert abs(vs["dx_mm"]) < 0.5 and abs(vs["dy_mm"]) < 0.5 and vs["ang_deg"] == 0.0
    assert "error" in T.record(ok, "right", tmp_path / "없음.json")["vs_fixed"]   # 고정값을 못 읽어도 실측은 쓴다


def test_object_far_from_palm_is_rejected():
    """손에서 15 cm 떨어진 물체(엉뚱한 물체 추적) — NEAR_M 기하 검사로 거부."""
    t, hs, X, at, aq, fk = _synthetic()
    Xf = X.copy()
    Xf[:, :3] += np.array([0.0, 0.0, 0.15]) - X[0, :3]      # 컵 약 (0, 0, 0.15) m — 갱신(µm 변화)은 그대로
    far = T.measure(t, hs, Xf, at, aq, fk, 0.0, 10.0)
    assert not far["ok"] and "손바닥–컵" in far["reason"]


def test_left_palm_limit_is_tighter():
    """★11.10 왼손 상한 90 mm — 100 mm 물체는 오른손 상한(120)은 통과, 왼손 상한에서는 거부."""
    t, hs, X, at, aq, fk = _synthetic()
    Xf = X.copy()
    Xf[:, :3] += np.array([0.0, 0.0, 0.10]) - X[0, :3]
    assert T.NEAR_M_SIDE == {"right": 0.12, "left": 0.09}
    assert T.measure(t, hs, Xf, at, aq, fk, 0.0, 10.0, near_m=T.NEAR_M_SIDE["right"])["ok"]
    left = T.measure(t, hs, Xf, at, aq, fk, 0.0, 10.0, near_m=T.NEAR_M_SIDE["left"])
    assert not left["ok"] and "> 90" in left["reason"]


# ---------------------------------------------------------------- 실행기 · place 노드 연결
def _logger(lines: list):
    from types import SimpleNamespace
    say = lambda *a, **k: lines.append(a[0])      # noqa: E731
    return SimpleNamespace(get_logger=lambda: SimpleNamespace(info=say, warning=say, error=say))


class _Cancel:
    """정지 Event 대역 — wait 는 기다리지 않고 on_wait(초) 를 부른다(가짜 시계를 움직이는 시험용)."""
    def __init__(self, on_wait=None):
        self.flag, self.on_wait, self.waits = False, on_wait, []

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.on_wait:
            self.on_wait(timeout)
        return self.flag


def _executor(tmp_path, live, arm, fallback="fixed/tin.json", rescan=True, still=None):
    import threading
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    lines: list = []
    ep = SimpleNamespace(tin_fallback=fallback, name="cocktail_1c_e4000", objects={"SRC_BLUE": {"topic": "/objects/cyl60_blue/pose"}},
                         policies={"rh_place_r": SimpleNamespace(attach_source="tin_json", side="right")})
    ex = SimpleNamespace(ep=ep, _lock=threading.Lock(), _tin_rec={"live": live, "arm": arm}, _tin_json={}, _tin_note=[],
                         _tin_why={}, node=_logger(lines), run_dir=tmp_path, fpp_rescan=rescan, cancel=_Cancel(), runs=[])
    ex._run = lambda argv: (ex.runs.append(argv), (0, ""))[1]
    ex._wait_arm_still = still or (lambda sides: {"arm_still": True})
    ex._write_tin = lambda *a: R.RosExecutor._write_tin(ex, *a)
    ex._drop_tin = lambda *a: R.RosExecutor._drop_tin(ex, *a)
    return ex, lines


def test_executor_window_matches_analysis_and_writes_run_json(tmp_path, monkeypatch):
    """bag 151437 오른손 정지 창을 실행기 버퍼 모양(수신 시각 · stamp · x7 / 시각 · {관절: rad})으로 넣으면 분석 T_in 과 1 mm 안."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    D = dict(np.load(FIX))
    t0, t1 = D["r_win"]
    live = {"SRC_BLUE": [(float(a), float(b), tuple(x)) for a, b, x in zip(D["r_t"], D["r_hs"], D["r_X"])]}
    arm = [(float(a), {f"r_aj_{i + 1}": float(q[i]) for i in range(7)}) for a, q in zip(D["r_arm_t"], D["r_arm_q"])]
    monkeypatch.setattr(R.time, "monotonic", lambda: float(t1))
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    ex, lines = _executor(tmp_path, live, arm, fallback="deploy/policy_control/config/tilt/traj_v12_240/ref/tin_fixed_1107.json")   # 고정 평균과 20 mm 안이어야 measured
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(t0), {"arm_still": True})
    assert why == "" and ex._tin_rec is None
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert out["source"] == "measured" and ex._tin_json["right"] == str(tmp_path / "tin_tilt_blue_right.json")
    assert np.linalg.norm(np.asarray(out["pos_m"]) - D["r_T_in"][:3, 3]) < 1e-3
    assert len(ex._tin_note) == 1 and ex._tin_note[0].startswith("T_in right measured · 고정값과 ") and "measured" in lines[0]
    assert out["measured"]["n_fresh"] >= T.MIN_FRESH and out["measured"]["window_s"] >= 2.0   # 창 시작 + FP++ 지연부터


def test_executor_rescans_then_stops_instead_of_fixed_fallback(tmp_path, monkeypatch):
    """★11.10 실기: 실측이 안 되면(팔이 움직임 · 측정 예외) 컵을 FP++ 로 다시 잡고 TIN_RETRY_N 회 다시 잰다 — 그래도 안 되면
    고정값 없이 tilt 를 멈추고 회차마다 어긴 판정을 json 에 남긴다. fake(재잡기 없음)는 옛 fallback."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    D = dict(np.load(FIX))
    live = {"SRC_BLUE": [(float(a), float(b), tuple(x)) for a, b, x in zip(D["r_t"], D["r_hs"], D["r_X"])]}
    arm = [(float(a), {f"r_aj_{i + 1}": float(q[i]) for i in range(7)}) for a, q in zip(D["r_arm_t"], D["r_arm_q"])]
    monkeypatch.setattr(R.time, "monotonic", lambda: float(D["r_win"][1]))
    fb = "deploy/policy_control/config/tilt/traj_v12_240/ref/tin_fixed_1107.json"
    ex, lines = _executor(tmp_path, live, arm, fallback=fb, still=lambda sides: {"arm_still": False, "arm_span_rad": 0.02})
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]),
                                     {"arm_still": False, "arm_span_rad": 0.02})
    assert "고정값으로 진행하지 않는다" in why and "tilt 를 시작하지 않는다" in why and "right" not in ex._tin_json
    assert len(ex.runs) == R.TIN_RETRY_N and ex.runs[0][-4:] == ["cyl60_blue", "--wait", str(R.RESCAN_WAIT_S), "--require-all"]
    assert not (tmp_path / "tin_tilt_blue_right.json").exists()
    bad = json.loads((tmp_path / "tin_tilt_blue_right_failed.json").read_text())
    assert bad["source"] == "failed" and len(bad["attempts"]) == R.TIN_RETRY_N + 1
    assert all("팔이 멈추지 않아" in a["reason"] for a in bad["attempts"]) and ex._tin_why["right"] in why

    def boom(*a, **k):
        raise IndexError("x")
    monkeypatch.setattr(T, "measure", boom)
    ex, lines = _executor(tmp_path, live, arm, fallback=fb, rescan=False)        # fake: 재잡기 · 재시도 없이 옛 fallback
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]), {"arm_still": True})
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert why == "" and ex.runs == [] and out["source"] == "fallback" and out["fallback_reason"] == "측정 실패(IndexError('x'))"
    assert "실측 아님" in ex._tin_note[0] and "실측 아님" in lines[-1]


def test_executor_rescans_and_measures_again_at_the_still_pose(tmp_path, monkeypatch):
    """★11.10 첫 창에 FP++ 새 프레임이 없으면(추적 잃음) 재잡기 뒤 정지 자세에서 다시 연 창으로 measured — 실패 회차는 json 에."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    D = dict(np.load(FIX))
    now = [float(D["r_win"][1])]
    monkeypatch.setattr(R.time, "monotonic", lambda: now[0])
    live = {"SRC_BLUE": [(float(a), float(b), tuple(x)) for a, b, x in zip(D["r_t"], D["r_hs"], D["r_X"])]}
    arm = [(float(a), {f"r_aj_{i + 1}": float(q[i]) for i in range(7)}) for a, q in zip(D["r_arm_t"], D["r_arm_q"])]

    def still(sides):                              # 재잡기 뒤 다시 연 기록에 FP++ 가 다시 잡은 프레임이 들어온다
        ex._tin_rec["live"].update(live)
        ex._tin_rec["arm"].extend(arm)
        now[0] = float(D["r_win"][0])
        return {"arm_still": True}
    ex, lines = _executor(tmp_path, {}, arm, fallback="deploy/policy_control/config/tilt/traj_v12_240/ref/tin_fixed_1107.json", still=still)
    ex.cancel = _Cancel(lambda t: now.__setitem__(0, float(D["r_win"][1])))   # 창 기다림 = 가짜 시계를 창 끝으로
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]), {"arm_still": True})
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert why == "" and len(ex.runs) == 1 and out["source"] == "measured"
    assert len(out["failed_attempts"]) == 1 and len(ex._tin_note) == 1 and ex._tin_note[0].startswith("T_in right measured(2회차) · 고정값과 ")
    assert np.linalg.norm(np.asarray(out["pos_m"]) - D["r_T_in"][:3, 3]) < 1e-3


def test_executor_falls_back_and_refuses_only_without_any_value(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    monkeypatch.setattr(R._paths, "SIM2REAL", tmp_path)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    ex, lines = _executor(tmp_path, {}, [], rescan=False)   # fake 경로(실기는 위 시험)
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, R.time.monotonic(), {"arm_still": True})
    assert "tilt 를 시작하지 않는다" in why and "right" not in ex._tin_json       # 측정도 고정값도 없다 → 멈춤
    assert "fixed/tin.json" in ex._tin_why["right"]                              # place 거부 문구에 쓸 실제 원인

    M = np.eye(4); M[:3, 3] = [0.01, 0.02, 0.07]
    (tmp_path / "fixed").mkdir()
    (tmp_path / "fixed/tin.json").write_text(json.dumps({"T_in_r": {"T_meas_fullrot": M.tolist()}}))
    ex, lines = _executor(tmp_path, {}, [], rescan=False)
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, R.time.monotonic(), {"arm_still": True})
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert why == "" and out["source"] == "fallback" and out["fallback_file"] == "fixed/tin.json"
    assert ex._tin_note[0].startswith("T_in right fallback(실측 아님") and "실측 아님" in lines[-1]


def _right_fix():
    D = dict(np.load(FIX))
    live = {"SRC_BLUE": [(float(a), float(b), tuple(x)) for a, b, x in zip(D["r_t"], D["r_hs"], D["r_X"])]}
    arm = [(float(a), {f"r_aj_{i + 1}": float(q[i]) for i in range(7)}) for a, q in zip(D["r_arm_t"], D["r_arm_q"])]
    return D, live, arm


FB = "deploy/policy_control/config/tilt/traj_v12_240/ref/tin_fixed_1107.json"


def test_executor_cancel_during_rescan_is_a_stop_not_a_failure(tmp_path, monkeypatch):
    """H1: 재잡기 도중 정지 — failed.json · 거부 사유를 쓰지 않고 바로 돌아간다(호출부가 aborted)."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    D, live, arm = _right_fix()
    monkeypatch.setattr(R.time, "monotonic", lambda: float(D["r_win"][1]))
    ex, lines = _executor(tmp_path, live, arm, fallback=FB)
    ex._run = lambda argv: (ex.runs.append(argv), ex.cancel.set(), (0, ""))[2]
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]),
                                     {"arm_still": False})
    assert why == "episode runner stop" and len(ex.runs) == 1
    assert not list(tmp_path.glob("*failed.json")) and "right" not in ex._tin_why


def test_run_planned_returns_aborted_when_stopped_during_tin(tmp_path):
    """H1 호출부: T_in 측정 중 정지 → NodeResult aborted · 'episode runner stop'(같은 함수 다른 정지 경로와 같은 문구)."""
    import threading
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    from policy_control import episode_spec as S
    ep = S.load(Path(__file__).resolve().parents[2] / "config" / "episodes" / "cocktail_1c_e4000.yaml")

    def measure(node, objs, t0, still, required=None):
        me.cancel.set()
        return "episode runner stop"
    me = SimpleNamespace(_tin_rec=None, _tin_json={}, _tin_why={}, ep=ep, _lock=threading.Lock(), _wait_pd_tracking=lambda side: True,
                         _wait_arm_still=lambda sides: {"arm_still": True}, _pd_view=lambda side: {}, run_dir=tmp_path,
                         _run=lambda argv: pytest.fail("정지 뒤 도구를 띄웠다"), cancel=threading.Event(), _measure_tin=measure,
                         node=_logger([]))
    me._drop_tin = lambda *a: R.RosExecutor._drop_tin(me, *a)
    me._tin_json = {"right": {"T_in": "stale"}}
    r = R.RosExecutor._run_planned(me, SimpleNamespace(id="tilt"), ep.trajectories["tilt"],
                                   SimpleNamespace(hand=lambda side: {"right": "SRC_BLUE", "left": "RCV"}[side]))
    assert (r.status, r.reason) == ("aborted", "episode runner stop")
    assert "right" not in me._tin_json  # 정지로 tilt 가 안 돈 팔의 T_in 은 place 에 남기지 않는다


def test_executor_stops_retrying_after_the_time_limit(tmp_path, monkeypatch):
    """시간 상한: 첫 회차 뒤 시계가 TIN_RETRY_S 를 넘으면 재잡기 없이 끝낸다."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    D, live, arm = _right_fix()
    clock = iter([0.0] + [R.TIN_RETRY_S + 1.0] * 50)
    monkeypatch.setattr(R.time, "monotonic", lambda: next(clock))
    ex, lines = _executor(tmp_path, live, arm, fallback=FB)
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, 0.0, {"arm_still": False})
    assert "tilt 를 시작하지 않는다" in why and ex.runs == []
    assert len(json.loads((tmp_path / "tin_tilt_blue_right_failed.json").read_text())["attempts"]) == 1


def test_executor_two_arms_reports_all_failures_and_forgets_the_good_arm(tmp_path, monkeypatch):
    """H2: 두 팔 다 실패면 두 팔 failed.json · 사유를 다 쓰고 한 번에 거부. M3: 한 팔만 실패해도 거부하고, 성공한 팔 T_in 은
    place 가 쓰지 않게 잊는다(json 파일은 남김)."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    D, live, arm = _right_fix()
    monkeypatch.setattr(R.time, "monotonic", lambda: float(D["r_win"][1]))
    objs = {"right": "SRC_BLUE", "left": "RCV"}
    ex, lines = _executor(tmp_path, live, arm, fallback=FB, still=lambda sides: {"arm_still": False})
    ex.ep.objects["RCV"] = {"topic": "/objects/shaker/pose"}
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt"), objs, float(D["r_win"][0]), {"arm_still": False})
    assert "T_in right" in why and "T_in left" in why and why.count("tilt 를 시작하지 않는다") == 1
    assert (tmp_path / "tin_tilt_right_failed.json").exists() and (tmp_path / "tin_tilt_left_failed.json").exists()
    assert set(ex._tin_why) == {"right", "left"} and ex._tin_json == {}

    ex, lines = _executor(tmp_path, live, arm, fallback=FB)          # 오른손은 measured, 왼손은 프레임 없음
    ex.ep.objects["RCV"] = {"topic": "/objects/shaker/pose"}
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt"), objs, float(D["r_win"][0]), {"arm_still": True})
    assert "T_in left" in why and "T_in right" not in why
    assert (tmp_path / "tin_tilt_right.json").exists() and "right" not in ex._tin_json   # 기록은 남고 place 는 못 쓴다
    assert "다른 팔" in ex._tin_why["right"] and not any(n.startswith("T_in right") for n in ex._tin_note)
    assert all(argv[-4] == "shaker" for argv in ex.runs)                             # 재잡기는 실패한 팔 컵만

    ex, lines = _executor(tmp_path, live, arm, fallback=FB)          # 왼손이 계획 · place 어디에도 안 쓰이면 진행
    ex.ep.objects["RCV"] = {"topic": "/objects/shaker/pose"}
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt"), objs, float(D["r_win"][0]), {"arm_still": True},
                                     required={"right"})
    assert why == "" and "right" in ex._tin_json and "T_in left 없음" in ex._tin_note and "left" in ex._tin_why


def test_executor_records_rescan_rc_and_argv_error(tmp_path, monkeypatch):
    """M2: 재잡기 rc · 끝줄 · rescan_argv ValueError 가 회차 기록 · failed.json · 사유에 남는다."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    D, live, arm = _right_fix()
    monkeypatch.setattr(R.time, "monotonic", lambda: float(D["r_win"][1]))
    ex, lines = _executor(tmp_path, live, arm, fallback=FB, still=lambda sides: {"arm_still": False})
    ex._run = lambda argv: (ex.runs.append(argv), (3, "cyl60_blue not found"))[1]
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]), {"arm_still": False})
    bad = json.loads((tmp_path / "tin_tilt_blue_right_failed.json").read_text())
    assert "rescan" not in bad["attempts"][0] and bad["attempts"][1]["rescan"] == "FP++ 재잡기 rc 3: cyl60_blue not found"
    assert "FP++ 재잡기 rc 3" in why

    ex, lines = _executor(tmp_path, live, arm, fallback=FB, still=lambda sides: {"arm_still": False})
    ex.ep.objects["SRC_BLUE"] = {"topic": "/cup"}                     # 실물 이름을 못 읽는 토픽
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]), {"arm_still": False})
    assert ex.runs == [] and "FP++ 재잡기 못 함" in why


def test_executor_uses_measured_tin_far_from_fixed_and_records_gap(tmp_path, monkeypatch):
    """★11.10 고정값과 멀어도 기하 검사를 통과한 실측은 재잡기 없이 쓴다 · 차이는 json(vs_fixed) · 노드 결과 reason 에 남는다.
    고정값 파일을 못 읽어도 실측은 쓴다(차이 자리에 사유)."""
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    D, live, arm = _right_fix()
    monkeypatch.setattr(R.time, "monotonic", lambda: float(D["r_win"][1]))
    monkeypatch.setattr(R._paths, "SIM2REAL", tmp_path)
    M = np.eye(4); M[:3, 3] = [0.3, 0.3, 0.3]                           # 실측과 한참 먼 고정 평균
    (tmp_path / "fixed").mkdir()
    (tmp_path / "fixed/tin.json").write_text(json.dumps({"T_in_r": {"T_meas_fullrot": M.tolist()}}))

    def still(sides):
        ex._tin_rec["live"].update(live)
        ex._tin_rec["arm"].extend(arm)
        return {"arm_still": True}
    ex, lines = _executor(tmp_path, live, arm, still=still)
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]), {"arm_still": True})
    assert why == "" and ex.runs == []
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert out["source"] == "measured" and out["vs_fixed"]["pos_mm"] > 300 and {"dx_mm", "dy_mm", "dz_mm", "ang_deg"} <= set(out["vs_fixed"])
    assert any("고정값과" in n and "dz" in n for n in ex._tin_note) and any("고정값과" in ln for ln in lines)
    assert not (tmp_path / "tin_tilt_blue_right_failed.json").exists()

    ex, lines = _executor(tmp_path, live, arm, fallback="fixed/none.json", still=still)
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]), {"arm_still": True})
    assert why == "" and ex.runs == []
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert out["source"] == "measured" and "못 읽음" in out["vs_fixed"]["error"]

def test_cocktail_place_jobs_get_the_tin_json_of_their_arm():
    from types import SimpleNamespace
    from policy_control import episode_spec as E
    from policy_control.episode_ros import job_params
    from policy_control.episode_runner import EpisodeManager
    ep = E.load(Path(__file__).resolve().parents[2] / "config/episodes/cocktail_1c_e4000.yaml")
    assert ep.tin_fallback == "deploy/policy_control/config/tilt/traj_v12_240/ref/tin_fixed_1107.json"
    plans = [p for n in ep.nodes for p in EpisodeManager._plans(SimpleNamespace(ep=ep), n, SimpleNamespace(objects={}))
             if p.kind == "place"]
    got = {p.role: job_params(ep, p, {"right": "run/tin_tilt_blue_right.json", "left": "run/tin_tilt_pink_left.json"}) for p in plans}
    assert [p.role for p in plans].count("rh_place_r") == 2                       # 파랑 · 분홍 놓기
    assert got["rh_place_r"]["attach_source"] == "tin_json" and got["rh_place_r"]["tin_json"] == "run/tin_tilt_blue_right.json"
    assert got["rh_place_l"]["attach_source"] == "tin_json" and got["rh_place_l"]["tin_json"] == "run/tin_tilt_pink_left.json"   # ★11.08 왼손 shaker 도
    # 앞 에피소드가 tin_json 이어도 바인딩 없는 다음 에피소드의 같은 노드엔 빈 값이 간다(노드에 남은 경로를 덮는다)
    import dataclasses
    plain = dataclasses.replace(ep, policies={**ep.policies, "rh_place_r": dataclasses.replace(ep.policies["rh_place_r"], attach_source=None)})
    p_r = next(p for p in plans if p.role == "rh_place_r")
    assert job_params(plain, p_r, {"right": "run/old.json"})["attach_source"] == "" == job_params(plain, p_r, {"right": "x"})["tin_json"]
    with pytest.raises(E.EpisodeSpecError, match="상대 경로"):
        E.validate(dataclasses.replace(ep, tin_fallback="/tmp/tin.json"))
    with pytest.raises(E.EpisodeSpecError, match="tin_fallback .*없는.json"):            # 시작 전에 파일을 읽어 본다
        E.validate(dataclasses.replace(ep, tin_fallback="logs/없는.json"))
    raw = __import__("yaml").safe_load((Path(__file__).resolve().parents[2] / "config/episodes/cocktail_1c_e4000.yaml").read_text())
    raw["policies"]["rh_place_r"]["attach_sorce"] = "tin_json"
    with pytest.raises(E.EpisodeSpecError, match="모르는 키"):
        E.parse(raw)


def test_place_refusal_names_the_tin_cause_and_new_episode_forgets_tin(tmp_path):
    from types import SimpleNamespace
    from policy_control.episode_ros import RosExecutor
    ex = SimpleNamespace(cancel=__import__("threading").Event(), _tin_json={"right": "old.json"}, _tin_why={"right": "x"})
    RosExecutor.clear_cancel(ex)
    assert ex._tin_json == {} and ex._tin_why == {}

    from policy_control import episode_spec as E
    from policy_control.episode_runner import EpisodeManager
    ep = E.load(Path(__file__).resolve().parents[2] / "config/episodes/cocktail_1c_e4000.yaml")
    p = next(p for n in ep.nodes for p in EpisodeManager._plans(SimpleNamespace(ep=ep), n, SimpleNamespace(objects={}))
             if p.role == "rh_place_r")
    me = SimpleNamespace(ep=ep, _tin_json={}, _tin_why={"right": "[tilt_blue] T_in right 측정 실패(흔들림) · fallback 못 읽음"},
                         _wait_node_ready=lambda k: None, _lock=__import__("threading").Lock(), _events={})
    r = RosExecutor._policy(me, p)
    assert r.status == "refused" and "측정 실패(흔들림)" in r.reason
    me._tin_why = {}
    assert "tilt 가 돌지 않았다" in RosExecutor._policy(me, p).reason


def test_place_node_attaches_tin_and_logs_fpp_only_when_plausible(tmp_path):
    """노드: tin_json 경로 → 붙인 자세(묶음 자리, 오프셋 없음) · 제어 컵 = FK ⊗ T_in · FP++ 는 서 있고 탁자 위일 때만 차이 로그."""
    from types import SimpleNamespace
    from policy_control.rh_place_node import PlaceNode
    rec = {"side": "right", "pos_m": [0.01, 0.02, 0.07], "quat_wxyz": [1, 0, 0, 0], "source": "fallback", "node": "tilt_blue",
           "fallback_reason": "흔들림", "measured": {"n_fresh": 9}}
    (tmp_path / "t.json").write_text(json.dumps(rec))
    par = {"tin_json": str(tmp_path / "t.json"), "attach_source": "tin_json"}
    me = SimpleNamespace(get_parameter=lambda k: SimpleNamespace(value=par[k]),
                         contract=SimpleNamespace(attach_source="bundle", side=lambda: SimpleNamespace(side="right")))
    assert PlaceNode._attach_source(me) == "tin_json"
    b = PlaceNode._tin_bundle(me)
    assert b.side == "r" and np.allclose(b.rel_p, rec["pos_m"]) and b.meta["source"] == "fallback" and b.n_steps == 9
    me._bundle, me._tin = b, True
    info = PlaceNode._attach_info(me)
    assert info["attach_source"] == "tin_json" and info["source"] == "fallback" and info["fallback_reason"] == "흔들림"
    par["tin_json"] = ""
    with pytest.raises(T.TinError):
        PlaceNode._tin_bundle(me)

    lines: list = []
    fk = (np.array([0.4, -0.2, 0.1]), np.array([1.0, 0, 0, 0]))
    fresh = {"cup": None}
    me = SimpleNamespace(chain=SimpleNamespace(live_cup="stale", cup_obs=lambda p, R: fk),
                         _fresh_cup=lambda role: fresh["cup"], **vars(_logger(lines)))
    assert PlaceNode._tin_cup(me, "arm", np.zeros(3), np.eye(3)) is fk and me.chain.live_cup is None and not lines
    fresh["cup"] = (fk[0] + [0.01, 0, 0], [0.9659, 0.2588, 0, 0], 0.0)          # 30° 누움 — 기록 안 함
    PlaceNode._tin_cup(me, "arm", np.zeros(3), np.eye(3))
    assert not lines
    fresh["cup"] = (fk[0] + [0.01, 0, 0], [1.0, 0, 0, 0], 0.0)
    assert PlaceNode._tin_cup(me, "arm", np.zeros(3), np.eye(3)) is fk                 # 제어 값은 그대로 FK
    assert len(lines) == 1 and "[10.0, 0.0, 0.0] mm" in lines[0]


V12 = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "config" / "tilt" / "traj_v12_240"
V10 = V12.parent / "traj_v10"


def test_axis_aligned_rebuilds_the_fixed_planner_tin_and_planner_tin_keeps_v12_for_the_fixed_value(tmp_path):
    """★11.09 문제 1: tin_fixed_1107 의 T(계획기용) = 실측 전체 회전의 컵 축 + traj_v10 기본 방위 — 같은 식이 그대로 다시 만든다.
    고정값을 v12 setup 기준으로 넣으면 v12 계획 T_in 이 그대로 나온다(실측이 고정값과 같으면 계획도 그대로)."""
    d = json.loads((V12 / "ref" / "tin_fixed_1107.json").read_text())
    with np.load(V10 / "ref" / "setup_240.npz") as S10:
        for side, key in T.FALLBACK_KEY.items():
            out, deg = T.axis_aligned(np.asarray(d[key]["T_meas_fullrot"]), S10[key])
            assert np.abs(out - np.asarray(d[key]["T"])).max() < 1e-5 and 15 < deg < 25
    meas = {s: np.asarray(d[k]["T_meas_fullrot"]) for s, k in T.FALLBACK_KEY.items()}
    p = T.planner_tin(meas, V12 / "ref" / "setup_240.npz", node="tilt")
    with np.load(V12 / "ref" / "setup_240.npz") as S12:
        for key in T.FALLBACK_KEY.values():
            assert np.abs(np.asarray(p[key]) - S12[key]).max() < 1e-5
    assert p["sides"]["right"]["source"].startswith("run") and p["sides"]["right"]["pos_vs_default_mm"] < 0.1

    low = meas["right"].copy()
    low[:3, 3] += low[:3, 2] * 0.010                     # 컵 축 방향으로 10 mm(손이 컵을 더 아래에서 쥔 것)
    p = T.planner_tin({"right": low}, V12 / "ref" / "setup_240.npz")
    assert abs(p["sides"]["right"]["pos_vs_default_mm"] - 10.0) < 0.1 and p["sides"]["left"] == {"source": "setup 기본값"}
    with pytest.raises(T.TinError):
        flip = meas["right"].copy(); flip[:3, 1:3] *= -1
        T.planner_tin({"right": flip}, V12 / "ref" / "setup_240.npz")


def test_tilt_traj_measures_before_planning_and_plans_with_the_run_tin(tmp_path, monkeypatch):
    """tilt_traj(v12): 정지 → T_in 측정 → 계획(--tin-json = 이 런 값) → 재생. 측정이 계획보다 먼저."""
    import threading
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    from policy_control import episode_spec as S
    ep = S.load(Path(__file__).resolve().parents[2] / "config" / "episodes" / "cocktail_1c_e4000.yaml")
    order: list = []
    d = json.loads((V12 / "ref" / "tin_fixed_1107.json").read_text())

    def measure(node, objs, t0, still, required=None):
        order.append(("required", tuple(sorted(required))))
        order.append(("measure", tuple(sorted(objs))))
        for s in objs:
            path = tmp_path / f"tin_{node.id}_{s}.json"
            path.write_text(json.dumps({"T_in": d[T.FALLBACK_KEY[s]]["T_meas_fullrot"]}))
            me._tin_json[s] = str(path)
        return ""

    def run(argv):
        order.append(Path(argv[1]).name)
        order.append(list(argv))
        return 0, "tail"
    lines: list = []
    me = SimpleNamespace(_tin_rec=None, _tin_json={}, _tin_why={}, ep=ep, _lock=threading.Lock(), _wait_pd_tracking=lambda side: True,
                         _wait_arm_still=lambda sides: {"arm_still": True}, _pd_view=lambda side: {}, run_dir=tmp_path, _run=run,
                         cancel=threading.Event(), _measure_tin=measure, node=_logger(lines))
    me._planner_tin = lambda *a: R.RosExecutor._planner_tin(me, *a)
    r = R.RosExecutor._run_planned(me, SimpleNamespace(id="tilt"), ep.trajectories["tilt"],
                                   SimpleNamespace(hand=lambda side: {"right": "SRC_BLUE", "left": "RCV"}[side]))
    assert r.status == "completed"
    names = [o for o in order if isinstance(o, (str, tuple))]
    assert names == [("required", ("left", "right")), ("measure", ("left", "right")), "plan_tilt_traj.py", "replay_to_pd.py"]
    plan = order[3]
    tj = json.loads(Path(plan[plan.index("--tin-json") + 1]).read_text())
    assert tj["sides"]["right"]["source"].startswith("run") and tj["sides"]["left"]["source"].startswith("run")
    assert "--tin-json" not in order[5]
