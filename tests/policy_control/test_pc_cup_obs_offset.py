"""★11.04 cup_obs_offset_base_m — 접근 · 잡기 정책이 보는 컵 = 측정 컵 − t(t = 카메라가 본 손 − FK 손). 부호 · 범위 검사.
목표(SETTING)는 base 설정값이라 빼지 않는다. place 로 가는 넘김 묶음은 보정 전(카메라) 틀."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control import approach_grasp as AG
from policy_control import inhand_est as IE
from policy_control import place_handover as PH
from policy_control import pour_fj_node as N
from test_pc_rh_aglt_goals import CUP_R, _meas, _right, _Zero

T = np.array([-0.0116, 0.0338, -0.0143])   # 오른손 정지 자세 Rm 실측(cocktail_1c_e1000, logs/analysis/frame_offset_1104)


def test_offset_range_is_checked_per_axis():
    assert AG.cup_obs_offset(None).tolist() == [0.0, 0.0, 0.0]
    assert AG.cup_obs_offset([0.05, -0.05, 0.0]).tolist() == [0.05, -0.05, 0.0]
    for bad in ([0.051, 0, 0], [0, 0, -0.06], [0.01, 0.01], [0, float("nan"), 0]):
        with pytest.raises(AG.ApproachGraspError):
            AG.cup_obs_offset(bad)


def test_chain_sees_cup_minus_t_and_the_first_goal_follows_the_seen_cup():
    c = _right()
    ch = N.AgltChain(c, _Zero())
    ch.cup_offset = T
    raw = {"arm": _meas(c, CUP_R)}
    ch.reset(raw)
    assert ch.goals.cup_reset == pytest.approx(CUP_R - T)          # 부호: C_obs = C_camera − t
    assert ch.goal.pos == pytest.approx(CUP_R - T + c.goal_offset)
    assert raw["arm"].cup_pos == pytest.approx(CUP_R)                # 노드 측정(로그 · FP++ 원값)은 그대로
    obs, _, _ = ch.step(raw)
    ref = N.AgltChain(c, _Zero())
    ref.reset({"arm": _meas(c, CUP_R - T)})
    assert obs == pytest.approx(ref.step({"arm": _meas(c, CUP_R - T)})[0])   # 정책 관측 = 보정 컵을 측정한 것과 같다


def test_a_requested_goal_is_the_base_setting_and_is_not_shifted_by_t():
    c = _right()
    ch = N.AgltChain(c, _Zero())
    ch.cup_offset = T
    ch.reset({"arm": _meas(c, CUP_R)})
    target = CUP_R - T + np.array([0.04, -0.03, 0.11])          # 첫 목표 영역 안 → 곧바로 바뀐다
    assert ch.request_goal(target) == []
    assert ch.goals.current == pytest.approx(target)               # t 를 빼지 않는다(옛 버그: target − t)


# ---- 넘김 묶음 = 보정 전(카메라) 틀 ---------------------------------------------------------------
def _rot(ax, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)]["xyz".index(ax)]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


@pytest.fixture(scope="module")
def rcg04():
    from test_pc_inhand_est import CON
    from policy_control import rh_aglt as A
    return A.load_contract(CON["right_rcg04"])


@pytest.fixture(scope="module")
def run(rcg04):
    from test_pc_inhand_est import _spec_or_skip
    return IE.InhandRun(_spec_or_skip("right"), rcg04)


def _held(run, c, R, n=40):
    """손바닥이 R 로 돌아간 채 컵(보정 틀)을 쥐고 n 틱 — 끊긴 뒤 grasp_state 가 끝 창을 채운다."""
    from test_pc_inhand_est import OBJ
    from test_pc_inhand_est import _meas as im
    q = np.asarray(c.side().hand_open, float)
    palm = np.array([0.3, -0.1, 0.25])
    run.reset(OBJ)
    for k in range(n):
        m = replace(im(c, palm, palm + R @ np.array([0.0, -0.06, 0.01])), palm_R=R)
        run.perceive(m, q, q)
        run.remember(np.zeros(run.width))
    return palm


def test_the_handover_bundle_is_in_the_uncorrected_camera_frame(run, rcg04):
    R = _rot("z", 90) @ _rot("x", 30)                              # 손바닥이 돌아간 자세(R_palmᵀ·t ≠ t)
    palm = _held(run, rcg04, R)
    corr, raw = run.bundle(None), run.bundle(None, cup_offset=T)
    assert run.bundle(None, cup_offset=np.zeros(3)) is not None
    d = R.T @ T
    assert not np.allclose(d, T, atol=1e-3)
    assert np.allclose((raw["rel_p"] - corr["rel_p"]).numpy()[0], d, atol=1e-6)          # rel_p_raw = rel_p + R_palmᵀ·t
    assert np.allclose(raw["rel_q"].numpy(), corr["rel_q"].numpy())                      # 회전 · 축 그대로
    assert np.allclose(raw["obj_axis_palm"].numpy(), corr["obj_axis_palm"].numpy())
    axis = corr["obj_axis_palm"].numpy()[0]
    assert float((raw["grasp_height"] - corr["grasp_height"])[0]) == pytest.approx(-axis @ d, abs=1e-6)
    torch = run.torch
    pp = torch.as_tensor(palm, dtype=torch.float32)[None]
    pq = run.geo.matrix_to_quat(torch.as_tensor(R, dtype=torch.float32)[None])
    wc, _ = run.pkg.grasp_state.object_world(pp, pq, corr)
    wr, _ = run.pkg.grasp_state.object_world(pp, pq, raw)
    assert np.allclose(wr.numpy()[0], wc.numpy()[0] + T, atol=1e-5)                      # place: FK 손바닥 ⊗ 묶음 = 카메라 컵


def test_the_node_capture_writes_the_raw_bundle_with_t_in_meta(run, rcg04, tmp_path, monkeypatch):
    R = _rot("y", -40) @ _rot("z", 60)
    _held(run, rcg04, R)
    corr = run.bundle(None)
    monkeypatch.setattr(IE, "BUNDLE_DIR", tmp_path)
    logs = []
    node = SimpleNamespace(chain=SimpleNamespace(inhand=run, dec=None, cup_offset=T), node_name="rh_aglt_node_right",
                           book=SimpleNamespace(episode=7), get_parameter=lambda n: SimpleNamespace(value=""),
                           get_logger=lambda: SimpleNamespace(info=logs.append, warning=logs.append))
    N.PourFjNode._inhand_capture(node, "lift_ok")
    b = PH.load(run.bundle_path)                                   # place 쪽 읽기 코드 · 형식 그대로
    assert np.allclose(b.rel_p, corr["rel_p"].numpy()[0] + R.T @ T, atol=1e-6)
    assert b.meta["cup_obs_offset_base_m"] == pytest.approx(T.tolist())
    assert b.meta["cup_obs_offset_palm_m"] == pytest.approx((R.T @ T).tolist(), abs=1e-5)


# ---- 11.04 리뷰: 시작 거부 틀 · palm_R 없음 · 조용한 0 ----------------------------------------------
def test_the_start_check_judges_the_spawn_box_with_the_corrected_cup():
    """원값 컵은 소환 박스 y 밖 4 cm(여유 2 cm 넘음), 보정 컵(측정 − t)은 안 — 정책과 같은 틀로 판정해 시작을 받는다."""
    c = _right()
    t = np.array([0.0, -0.045, 0.0])
    raw = {"arm": _meas(c, np.array([0.25, c.goal_box_min[1] - 0.04, CUP_R[2]]))}
    assert any("spawn box" in r for r in N.aglt_start_refusals(c, raw, 0.15))
    ch = N.AgltChain(c, _Zero())
    ch.cup_offset = t
    ch.reset(raw)
    node = SimpleNamespace(_drain=lambda: None, chain=ch, contract=c, _measure=lambda: raw, reset_tol=0.15,
                           fam=N.FAMILIES["rh_aglt"], book=SimpleNamespace(start=lambda: (None, ["book"])),
                           _reply=lambda _res, ok, reasons: (ok, reasons))
    assert N.PourFjNode._srv_start(node, None, None) == (False, ["book"])   # 거부 검사는 통과 · 장부까지 갔다
    ch.cup_offset = np.zeros(3)
    ok, why = N.PourFjNode._srv_start(node, None, None)
    assert not ok and any("spawn box" in r for r in why)                     # 보정이 없으면 원값으로 거부


def test_a_nonzero_offset_without_a_palm_rotation_is_refused(run):
    run.reset(None)
    assert run.palm_R is None
    assert run._palm_shift(None).tolist() == [0.0, 0.0, 0.0]               # t 없음 · 0 은 회전 없이도 0
    assert run._palm_shift(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(IE.InhandError, match="손바닥 회전"):
        run._palm_shift(T)


def test_the_offset_is_logged_even_when_zero_and_written_with_each_grip(tmp_path, monkeypatch):
    """시작 로그 문구와 grips.jsonl. 상태 토픽 필드(_offset_status)는 노드 __init__ 에서 채워 노드를 띄워야 보인다 — 여기서는 보지 않는다."""
    import json
    assert N.cup_offset_log_line(np.zeros(3)) == "cup_obs_offset_base_m 보정 없음(0,0,0)"
    assert str(np.round(T, 4).tolist()) in N.cup_offset_log_line(T)
    monkeypatch.setattr(N.PourFjNode, "GRIP_LOG", tmp_path / "grips.jsonl")
    ag = SimpleNamespace(grip={"palm_cup_m": 0.05}, end=("stop", "ok"))
    for off in (T, None):
        chain = SimpleNamespace(object_name="x") if off is None else SimpleNamespace(object_name="x", cup_offset=off)
        node = SimpleNamespace(node_name="rh_aglt_node_right", book=SimpleNamespace(episode=1), chain=chain,
                               GRIP_LOG=N.PourFjNode.GRIP_LOG, has_parameter=lambda n: False)
        N.PourFjNode._log_grip(node, ag)
    rows = [json.loads(x) for x in (tmp_path / "grips.jsonl").read_text().splitlines()]
    assert rows[0]["cup_obs_offset_base_m"] == pytest.approx(T.tolist()) and rows[1]["cup_obs_offset_base_m"] == [0.0, 0.0, 0.0]
    assert rows[0]["palm_cup_m"] == 0.05                                    # 기존 필드 그대로
