"""★11.20 왼손 place 이송 끝 FP++ T_in 재측정(tin_refresh_at_carry_end) · 컵−홀더 xy 보정 이송 1회(carry_fpp_xy_tol_m).
품질 판정(갱신 · 흩어짐 2 mm · 기울기 · 옛 값과 차)과 정지 창 → 보정 이송(최대 1회) → 정지 창 → 정책 순서를 ROS 없이 본다."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

N = pytest.importorskip("policy_control.rh_place_node")
TL = N.TL

PALM = np.array([0.37, 0.16, 0.30])          # 손바닥 = 관절 앞 3개(회전 없음) — 시험용 FK
TRUE = np.array([0.0, 0.03, -0.03])          # 참 T_in(손바닥 → shaker 원점)
Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def fk(q):
    M = np.eye(4)
    M[:3, 3] = np.asarray(q, float)[:3]
    return M


def _samples(t0, noise_m=0.0003, tilt_deg=0.0, stale=False, cup_off=(0.0, 0.0, 0.0)):
    """창 시작 t0 기준 1.3 s — 팔 60 Hz · FP++ 10 Hz(stamp 증가, 값은 조금씩 바뀜)."""
    arm = [(t0 + k / 60, np.r_[PALM, 0, 0, 0, 0]) for k in range(int(1.3 * 60))]
    a = np.radians(tilt_deg) / 2
    q = np.array([np.cos(a), np.sin(a), 0.0, 0.0])
    live = []
    for k in range(13):
        sgn = 1 if k % 2 else -1
        pos = PALM + TRUE + np.asarray(cup_off) + np.array([sgn * noise_m, 0.0, 0.0])
        live.append((t0 + 0.05 + k * 0.1, 100.0 if stale else 100.0 + k * 0.1, (*pos, *q)))
    return live, arm


def test_quality_gates():
    t0 = 10.0
    live, arm = _samples(t0)
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE + [0.0, -0.01, 0.0], Q0))
    assert rec["ok"] and rel is not None and rec["n_fresh"] >= N.TIN_REFRESH_MIN_FRESH
    assert np.allclose(rel[0], TRUE, atol=5e-4)              # z_off 를 더해 재고 to_origin 으로 되돌려 원점 틀
    assert rec["d_mm"][1] == pytest.approx(10.0, abs=0.6)
    cases: list[tuple[dict[str, Any], str]] = [({"noise_m": 0.003}, "흩어짐"), ({"tilt_deg": 20.0}, "기울기"), ({"stale": True}, "갱신")]
    for kw, why in cases:
        live, arm = _samples(t0, **kw)
        rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE, Q0))
        assert not rec["ok"] and rel is None and why in rec["reason"], (kw, rec)
    live, arm = _samples(t0)                                  # 옛 T_in 과 30 mm — 다른 물체로 보고 그대로
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE + [0.03, 0.0, 0.0], Q0))
    assert not rec["ok"] and rel is None and "옛 T_in" in rec["reason"]


def test_settle_uses_only_frames_stamped_after_settle():
    """settle 끝(stamp_min) 전에 찍힌 영상은 늦게 받아 수신 창 안에 들어와도 버린다 — 이송 중 잘못 따라간 FP++ 를 섞지 않는다."""
    t0 = 10.0
    live, arm = _samples(t0)
    live = [(t, hs, (*(np.asarray(x[:3]) + [0.008, 0.0, 0.0]), *x[3:])) if hs < 100.85 else (t, hs, x) for t, hs, x in live]
    rec, _ = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE, Q0))
    assert not rec["ok"] and "흩어짐" in rec["reason"]                 # stamp 제한 없으면 settle 전 영상(8 mm 어긋남)이 섞인다
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE, Q0), stamp_min=100.85)
    assert rec["ok"] and rec["n_fresh"] == 4 and np.allclose(rel[0], TRUE, atol=5e-4)


def test_one_tilted_fpp_frame_passes_but_a_tilted_majority_is_refused():
    """★11.20 실기: 서 있는 shaker 를 FP++ 가 127.8° 로 내는 프레임이 섞인다 — 한 프레임은 빼고 재서 통과(n_tilted_dropped),
    창 과반(중앙값)이 TIN_FPP_TILT_DEG 를 넘으면 거부."""
    t0 = 10.0
    a = np.radians(127.8) / 2
    bad_q = (np.cos(a), np.sin(a), 0.0, 0.0)
    live, arm = _samples(t0)
    in_win = [k for k, f in enumerate(live) if f[0] >= t0 + TL.FPP_LATENCY_S]
    one = [(t, hs, (*x[:3], *bad_q)) if k == in_win[2] else (t, hs, x) for k, (t, hs, x) in enumerate(live)]
    rec, rel = N.tin_refresh(one, arm, fk, t0, t0 + 1.3, "left", (TRUE, Q0))
    assert rec["ok"] and rel is not None and rec["n_tilted_dropped"] == 1, rec
    assert np.allclose(rel[0], TRUE, atol=5e-4)
    most = set(in_win[: len(in_win) // 2 + 1])
    many = [(t, hs, (*x[:3], *bad_q)) if k in most else (t, hs, x) for k, (t, hs, x) in enumerate(live)]
    rec, rel = N.tin_refresh(many, arm, fk, t0, t0 + 1.3, "left", (TRUE, Q0))
    assert not rec["ok"] and rel is None and "기울기 중앙값" in rec["reason"], rec


def _rxrz(tilt_deg, spin_deg):
    a, b = np.radians(tilt_deg), np.radians(spin_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    return Rx @ Rz                                            # 컵 축 기울기 a · 축 둘레 b(새 측정 = 단위 회전 기준)


def test_axis_gate_ignores_spin_and_keeps_old_spin():
    """★11.20 실기 회차 성분(축 3.9° · 축 둘레 12.9° · 위치 10.6 mm) — 통과, 채택 T_in = 새 위치 · 새 축 · 옛 축 둘레."""
    t0 = 10.0
    live, arm = _samples(t0)
    Ro = _rxrz(3.9, 12.9)
    old_p = TRUE - np.array([1.6, -7.7, -7.1]) / 1000
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (old_p, TL.R_to_quat(Ro)))
    assert rec["ok"] and rel is not None, rec
    assert rec["d_axis_deg"] == pytest.approx(3.9, abs=0.1) and rec["d_spin_deg"] == pytest.approx(12.9, abs=0.1)
    assert rec["d_rot_deg"] > N.TIN_REFRESH_MAX_DROT_DEG and rec["d_norm_mm"] == pytest.approx(10.6, abs=0.6)
    Ra = TL.quat_to_R(rel[1])
    assert np.allclose(rel[0], TRUE, atol=5e-4) and np.allclose(Ra[:, 2], [0, 0, 1], atol=1e-6)   # 위치 · 축 = 새 값
    assert np.allclose(Ra, _rxrz(0.0, 12.9), atol=1e-6)      # 축 둘레 = 옛 값(옛 회전을 새 축으로 최소 회전)


def test_axis_gate_rejects_tilt_11deg():
    t0 = 10.0
    live, arm = _samples(t0)
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE, TL.R_to_quat(_rxrz(11.0, 0.0))))
    assert not rec["ok"] and rel is None and "컵 축 기울기" in rec["reason"]
    assert rec["d_axis_deg"] == pytest.approx(11.0, abs=0.1) and rec["d_spin_deg"] == pytest.approx(0.0, abs=0.1)


def test_adopted_quat_same_hemisphere_and_opposite_axis_records_none():
    t0 = 10.0
    live, arm = _samples(t0)
    qo = -TL.R_to_quat(_rxrz(3.9, 12.9))                      # 옛 rel_q 를 -q 로(같은 회전, 다른 부호)
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE, qo))
    assert rec["ok"] and float(np.asarray(rel[1]) @ qo) > 0, rec
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE, TL.R_to_quat(_rxrz(180.0, 0.0))))
    assert not rec["ok"] and rel is None and "반대" in rec["reason"]   # TinError 경로
    assert rec["d_axis_deg"] is None and rec["d_spin_deg"] is None


@pytest.mark.parametrize("p, q, why", [([np.nan, 0.0, 0.0], [1.0, 0, 0, 0], "유한"), ([0.0, 0.03, -0.03], [1.2, 0, 0, 0], "노름")])
def test_new_rel_must_be_finite_unit(monkeypatch, p, q, why):
    t0 = 10.0
    live, arm = _samples(t0)
    monkeypatch.setattr(N.TL, "to_origin", lambda *_: (np.array(p), np.array(q)))
    rec, rel = N.tin_refresh(live, arm, fk, t0, t0 + 1.3, "left", (TRUE, Q0))
    assert not rec["ok"] and rel is None and why in rec["reason"]


class _RC:
    def __init__(self, target):
        self.target, self.on, self.holds, self.starts = np.asarray(target, float), False, [], []
        self.still, self.start_err, self.plan_sleep = "ok", (0.005, 1.0), 0.0

    def hold(self, ticks):
        self.holds.append(ticks)
        self.on = True

    def start(self, q, pos, quat, target):
        self.starts.append(np.asarray(pos, float).copy())
        time.sleep(self.plan_sleep)
        self.on = True
        return {"ok": True}


def _node(tol, monkeypatch, want=True, side="left"):
    monkeypatch.setattr(N, "start_err_with", lambda rc, q, p, quat: rc.start_err)   # 실제 식은 test_pc_rh_place_ready_carry
    node = N.PlaceNode.__new__(N.PlaceNode)
    node.get_parameter = lambda n: SimpleNamespace(value={"tin_subtract_offset": False}[n])
    logs = []
    node.get_logger = lambda: SimpleNamespace(info=lambda m, **k: logs.append(("info", m)),
                                              warning=lambda m, **k: logs.append(("warn", m)))
    node.contract = SimpleNamespace(policy_hz=60, side=lambda: SimpleNamespace(role="arm", side=side))
    rc = _RC([*(PALM + TRUE)[:2] + [0.0, -0.005], 0.40])     # 홀더 xy 가 참 컵에서 y −5 mm
    chain = SimpleNamespace(rel=(TRUE + [0.0, -0.01, 0.0], np.array([1.0, 0, 0, 0])), rcarry=rc, attach_info={})
    chain.cup_from_palm = lambda p, R: (np.asarray(p) + np.asarray(R) @ chain.rel[0], chain.rel[1])
    node.chain = chain
    node._rf, node._rf_want, node._rf_recarried, node._rf_tol, node._rf_fk = None, want, False, tol, fk
    node._rf_settle = N.TIN_REFRESH_SETTLE_S
    node._end = lambda kind, why: logs.append(("end", kind, why))
    return node, rc, logs


def _fill(node):
    """열린 정지 창에 1.3 s 표본을 넣는다(창 시작을 과거로 당겨 지금 끝나게)."""
    t0 = time.monotonic() - 1.31
    live, arm = _samples(t0)
    node._rf.update(t0=t0, stamp_min=100.0, live=live, arm=arm, palm=(PALM, np.eye(3), np.r_[PALM, 0, 0, 0, 0]))


@pytest.mark.parametrize("side", ["left", "right"])                # ★11.20 오른손도 켬 — Z_OFF · NEAR_M 이 팔마다 다르다
def test_window_recarry_once_then_policy(monkeypatch, side):
    node, rc, logs = _node(0.003, monkeypatch, side=side)
    t_open, wall_open = time.monotonic(), time.time()
    assert node._carry_end_refresh(rc) == "wait"
    assert rc.holds == [round((N.TIN_REFRESH_SETTLE_S + TL.FPP_LATENCY_S + N.TIN_REFRESH_WINDOW_S) * 60)]
    assert node._rf["t0"] >= t_open + N.TIN_REFRESH_SETTLE_S and node._rf["stamp_min"] >= wall_open + N.TIN_REFRESH_SETTLE_S
    _fill(node)
    assert node._carry_end_refresh(rc) == "wait" and len(rc.starts) == 1     # 새 T_in 컵 xy 5 mm > 3 → 보정 이송
    assert np.allclose(node.chain.rel[0], TRUE, atol=5e-4)
    assert node._carry_end_refresh(rc) == "wait" and len(rc.holds) == 2      # 보정 이송 끝 → 다시 정지 창
    _fill(node)
    assert node._carry_end_refresh(rc) == "" and len(rc.starts) == 1        # 두 번째도 넘지만 보정은 1회 — 경고만
    assert node._carry_end_refresh(rc) == ""                                  # 정책
    recs = node.chain.attach_info["tin_refresh"]
    assert [r["round"] for r in recs] == [1, 2] and all(r["ok"] for r in recs)
    assert recs[0]["xy_err_mm"] == pytest.approx(5.0, abs=0.6) and recs[0]["start_err_mm"] == 5.0
    assert 0.0 <= node.chain.attach_info["tin_refresh_recarry"]["plan_s"] < N.RECARRY_PLAN_WARN_S
    assert any(k == "warn" and "보정 이송 뒤에도" in m for k, m, *_ in logs)


def test_within_tol_and_off_keep_old_flow(monkeypatch):
    node, rc, _ = _node(0.01, monkeypatch)
    node._carry_end_refresh(rc)
    _fill(node)
    assert node._carry_end_refresh(rc) == "" and rc.starts == []             # 5 mm ≤ 10 mm — 보정 없음
    node, rc, _ = _node(0.003, monkeypatch, want=False)                                    # 끔 = 기존 그대로(창 · 이송 없음)
    assert node._carry_end_refresh(rc) == "" and rc.holds == [] and node.chain.attach_info == {}


def _old_kept(node, old, logs, why):
    rec = node.chain.attach_info["tin_refresh"][-1]
    assert node.chain.rel[0] is old[0] and node.chain.rel[1] is old[1]      # 반쯤 바뀐 rel 없음
    assert not rec["ok"] and why in rec["reason"] and any(k == "warn" and why in m for k, m, *_ in logs)


def test_arm_not_still_skips_the_refresh(monkeypatch):
    node, rc, logs = _node(0.003, monkeypatch)
    rc.still, old = "timeout", node.chain.rel
    assert node._carry_end_refresh(rc) == "" and rc.holds == [] and rc.starts == []
    _old_kept(node, old, logs, "멈춤")


def test_new_rel_outside_start_band_keeps_old(monkeypatch):
    node, rc, logs = _node(0.003, monkeypatch)
    rc.start_err, old = (0.040, 1.0), node.chain.rel                          # 40 mm > START_ERR_MAX_M 29 mm
    node._carry_end_refresh(rc)
    _fill(node)
    assert node._carry_end_refresh(rc) == "" and rc.starts == []              # 갱신 버림 → 보정 이송도 없음, abort 아님
    _old_kept(node, old, logs, "시작 문턱")
    assert not any(k == "end" for k, *_ in logs)


def test_refresh_crash_keeps_old_and_records(monkeypatch):
    node, rc, logs = _node(0.003, monkeypatch)
    old = node.chain.rel
    node._carry_end_refresh(rc)
    _fill(node)
    monkeypatch.setattr(N, "start_err_with", lambda *_: 1 / 0)                 # 새 rel 을 다 만든 뒤 던진다
    assert node._carry_end_refresh(rc) == "" and rc.starts == []
    _old_kept(node, old, logs, "ZeroDivisionError")


def test_slow_recarry_plan_warns(monkeypatch):
    node, rc, logs = _node(0.003, monkeypatch)
    rc.plan_sleep = N.RECARRY_PLAN_WARN_S + 0.01
    node._carry_end_refresh(rc)
    _fill(node)
    assert node._carry_end_refresh(rc) == "wait" and node.chain.attach_info["tin_refresh_recarry"]["plan_s"] > N.RECARRY_PLAN_WARN_S
    assert any(k == "warn" and "보정 이송 계획" in m for k, m, *_ in logs)


def test_runner_waits_longer_for_refresh_jobs():
    from policy_control.episode_runner import TIN_REFRESH_EXTRA_S, _node_max_s
    b = SimpleNamespace(kind="place", student=None, approach_grasp=None, tin_refresh_at_carry_end=False)
    assert _node_max_s(b) == 15.0
    assert _node_max_s(SimpleNamespace(**{**vars(b), "tin_refresh_at_carry_end": True})) == pytest.approx(15.0 + 21.4)
    assert TIN_REFRESH_EXTRA_S == pytest.approx(2 * round((N.TIN_REFRESH_SETTLE_MAX_S + TL.FPP_LATENCY_S + N.TIN_REFRESH_WINDOW_S) * 60) / 60
                                                + (600 + 30 + N.RCY_STILL_MAX_TICKS) / 60 + N.READY_CARRY_SLACK_S)


def test_cocktail_e2e_turns_it_on_for_both_places(tmp_path):
    from pathlib import Path
    from policy_control.episode_spec import EpisodeSpecError, load as load_episode
    path = Path(__file__).resolve().parents[2] / "config/episodes/cocktail_e2e.yaml"
    ep = load_episode(path)
    left, right = ep.policies["rh_place_l"], ep.policies["rh_place_r"]
    assert (left.tin_refresh_at_carry_end, left.carry_fpp_xy_tol_m) == (True, 0.003)
    assert (right.tin_refresh_at_carry_end, right.carry_fpp_xy_tol_m) == (True, 0.003)
    bad = tmp_path / "x.yaml"                                  # 문턱만 켜고 재측정은 끈 것은 거부
    bad.write_text(path.read_text().replace("tin_refresh_at_carry_end: true", "tin_refresh_at_carry_end: false"))
    with pytest.raises(EpisodeSpecError, match="carry_fpp_xy_tol_m"):
        load_episode(bad)
