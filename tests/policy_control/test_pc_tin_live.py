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
    """손바닥 = 원점(FK 항등), 컵 0.1 m 안에서 표본마다 10 µm 씩 바뀜(갱신), 6 Hz."""
    t = np.arange(n + 1) / rate
    X = np.tile([0.1, 0.05, 0.05, 1.0, 0.0, 0.0, 0.0], (n + 1, 1))
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


def test_late_arm_holds_end_value_ratio_and_fixed_gap(tmp_path):
    """팔 표본이 끝보다 ARM_HOLD_S 안 앞서면 끝값 유지 · 더 늦은 프레임만 버림 · 갱신 비율 0.3 · 오른손 고정 평균과 20 mm 넘으면 fallback."""
    t, hs, X, at, aq, fk = _synthetic()
    end = float(t[-1])
    held = T.measure(t, hs, X, np.array([-1.0, end - 0.15]), aq, fk, 0.0, 10.0)
    assert held["ok"] and "n_late_dropped" not in held
    cut = T.measure(t, hs, X, np.array([-1.0, end - 1.0]), aq, fk, 0.0, 10.0)
    assert cut["ok"] and cut["n_late_dropped"] == 5 and cut["n_fresh"] == held["n_fresh"] - 5
    rep = np.repeat(np.arange(len(t)), 3)              # 같은 프레임 3번 재발행 = 갱신 비율 1/3
    tt = np.arange(len(rep)) / 18.0
    assert T.measure(tt, hs[rep], X[rep], np.array([-1.0, 10.0]), aq, fk, 0.0, 10.0)["ok"]

    fixed = tmp_path / "f.json"
    M = np.eye(4); M[:3, 3] = [0.1, 0.05, 0.05]
    fixed.write_text(json.dumps({"T_in_r": {"T_meas_fullrot": M.tolist()}, "T_in_l": {"T_meas_fullrot": M.tolist()}}))
    ok = T.measure(t, hs, X, at, aq, fk, 0.0, 10.0)
    assert T.record(ok, "right", fixed)["source"] == "measured"
    far = {**ok, "T_in": (np.asarray(ok["T_in"]) + np.pad(np.array([[0, 0, 0, 0.03]]), ((0, 3), (0, 0)))).tolist()}
    r = T.record(far, "right", fixed)
    assert r["source"] == "fallback" and "고정 평균과 차이" in r["fallback_reason"] and r["measured"]["pos_mm"]
    assert T.record(far, "left", fixed)["source"] == "measured"          # 왼손은 비교 안 함


# ---------------------------------------------------------------- 실행기 · place 노드 연결
def _logger(lines: list):
    from types import SimpleNamespace
    say = lambda *a, **k: lines.append(a[0])      # noqa: E731
    return SimpleNamespace(get_logger=lambda: SimpleNamespace(info=say, warning=say, error=say))


def _executor(tmp_path, live, arm, fallback="fixed/tin.json"):
    import threading
    from types import SimpleNamespace
    lines: list = []
    ep = SimpleNamespace(tin_fallback=fallback, name="cocktail_1c_e4000",
                         policies={"rh_place_r": SimpleNamespace(attach_source="tin_json", side="right")})
    ex = SimpleNamespace(ep=ep, _lock=threading.Lock(), _tin_rec={"live": live, "arm": arm}, _tin_json={}, _tin_note=[],
                         _tin_why={}, node=_logger(lines), run_dir=tmp_path)
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
    assert ex._tin_note == ["T_in right measured"] and "measured" in lines[0]
    assert out["measured"]["n_fresh"] >= T.MIN_FRESH and out["measured"]["window_s"] >= 2.0   # 창 시작 + FP++ 지연부터


def test_executor_falls_back_when_the_arm_moves_or_measure_raises(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    D = dict(np.load(FIX))
    live = {"SRC_BLUE": [(float(a), float(b), tuple(x)) for a, b, x in zip(D["r_t"], D["r_hs"], D["r_X"])]}
    arm = [(float(a), {f"r_aj_{i + 1}": float(q[i]) for i in range(7)}) for a, q in zip(D["r_arm_t"], D["r_arm_q"])]
    monkeypatch.setattr(R.time, "monotonic", lambda: float(D["r_win"][1]))
    ex, lines = _executor(tmp_path, live, arm, fallback="deploy/policy_control/config/tilt/traj_v12_240/ref/tin_fixed_1107.json")
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]),
                                     {"arm_still": False, "arm_span_rad": 0.02})
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert why == "" and out["source"] == "fallback" and "팔이 멈추지 않아" in out["fallback_reason"]
    assert "실측 아님" in ex._tin_note[0] and "실측 아님" in lines[0]

    def boom(*a, **k):
        raise IndexError("x")
    monkeypatch.setattr(T, "measure", boom)
    ex, lines = _executor(tmp_path, live, arm, fallback="deploy/policy_control/config/tilt/traj_v12_240/ref/tin_fixed_1107.json")
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, float(D["r_win"][0]), {"arm_still": True})
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert why == "" and out["source"] == "fallback" and out["fallback_reason"] == "측정 실패(IndexError('x'))"


def test_executor_falls_back_and_refuses_only_without_any_value(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from policy_control import episode_ros as R
    monkeypatch.setattr(R._paths, "SIM2REAL", tmp_path)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    ex, lines = _executor(tmp_path, {}, [])
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, R.time.monotonic(), {"arm_still": True})
    assert "tilt 를 시작하지 않는다" in why and "right" not in ex._tin_json       # 측정도 고정값도 없다 → 멈춤
    assert "fixed/tin.json" in ex._tin_why["right"]                              # place 거부 문구에 쓸 실제 원인

    M = np.eye(4); M[:3, 3] = [0.01, 0.02, 0.07]
    (tmp_path / "fixed").mkdir()
    (tmp_path / "fixed/tin.json").write_text(json.dumps({"T_in_r": {"T_meas_fullrot": M.tolist()}}))
    ex, lines = _executor(tmp_path, {}, [])
    why = R.RosExecutor._measure_tin(ex, SimpleNamespace(id="tilt_blue"), {"right": "SRC_BLUE"}, R.time.monotonic(), {"arm_still": True})
    out = json.loads((tmp_path / "tin_tilt_blue_right.json").read_text())
    assert why == "" and out["source"] == "fallback" and out["fallback_file"] == "fixed/tin.json"
    assert ex._tin_note[0].startswith("T_in right fallback(실측 아님") and "실측 아님" in lines[0]


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
