"""11.08 놓기 학생 실기 경로(place_student · rh_place_student_node) — prop · img2 · 머리 규칙 · ckpt 머리글 검사 · 체인 RNN 규칙."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_control import hdgp_pin
from policy_control import place_cup_gate as CG
from policy_control import place_head_look as HL
from policy_control import place_head_pose as HP
from policy_control import place_holder_proj as HPJ
from policy_control import place_student as PS
from policy_control import rh_aglt as A
from policy_control import rh_place as P
from policy_control import student_mask as SM

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "deploy" / "policies" / "rh56f1" / "place" / "right_n10_e4000" / "rh_place_contract.json"


@pytest.fixture(scope="module")
def c():
    if not CONTRACT.is_file():
        pytest.skip(f"계약 없음 {CONTRACT}")
    return P.load_contract(CONTRACT)


@pytest.fixture(scope="module")
def rd():
    from policy_control.student_policy import REAL_DEPTH, load_pinned
    try:
        return load_pinned(REAL_DEPTH, "hdgp_real_depth")[0]
    except Exception as exc:   # noqa: BLE001 — hdgp 스냅숏이 없는 호스트
        pytest.skip(f"hdgp real_depth 스냅숏 없음: {exc}")


def test_prop_layout_and_goal(c):
    cols = PS.prop_cols(c)
    assert [n for n, _ in cols] == [*PS.PARTS, "goal_pos", "prev_action", "carry_on", "settle_on", "head_pan", "head_tilt"]
    lay = A.actor_layout(c)
    obs = np.arange(sum(w for _, w in lay), dtype=np.float64) * 0.01
    prop = PS.prop_from_obs(c, obs, True, False, [0.5, -0.5])
    assert prop.shape == (sum(w for _, w in cols),)
    s = PS._slices(c)
    i = sum(w for n, w in cols[:7])
    np.testing.assert_allclose(prop[i:i + 3], obs[s["goal_rel_palm"]] + obs[s["palm_pos"]], atol=1e-6)
    np.testing.assert_allclose(prop[-4:], [1.0, 0.0, 0.5, -0.5])


def test_palm_from_obs_rotation(c):
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    obs = np.zeros(sum(w for _, w in A.actor_layout(c)))
    s = PS._slices(c)
    obs[s["palm_rot"]] = np.concatenate([R[:, 0], R[:, 1]])
    obs[s["palm_pos"]] = [0.3, -0.1, 0.4]
    p, Rg = PS.palm_from_obs(c, obs)
    np.testing.assert_allclose(Rg, R)
    np.testing.assert_allclose(p, [0.3, -0.1, 0.4])


def test_to_action_limits():
    np.testing.assert_allclose(PS.to_action([HL.PAN_LIM[0], HL.TILT_LIM[1]]), [-1.0, 1.0])
    np.testing.assert_allclose(PS.to_action([-7.5, 40.0]), [0.0, 0.0])
    np.testing.assert_allclose(PS.to_action([99.0, -99.0]), [1.0, -1.0])


def test_block_k_matches_pixel_centres(rd):
    K = np.asarray(rd.D435I_K_1103, float)
    kb = PS.block_k(K, 8)
    np.testing.assert_allclose(kb, rd.scale_k(K, 8), atol=1e-9)
    x = np.array([0.05, -0.03, 0.7])
    u = K @ x
    ub = kb @ x
    np.testing.assert_allclose(ub[:2] / ub[2], (u[:2] / u[2] - 3.5) / 8.0, atol=1e-9)


def test_head_limits_match_driver():
    sys.path.insert(0, str(REPO / "scripts"))
    import head_motion as hm
    assert (PS.HEAD_VEL_DPS, PS.HEAD_ACC_DPS2) == (hm.STREAM_MAX_VEL_DPS, hm.STREAM_MAX_ACC_DPS2)
    assert (tuple(HL.PAN_LIM), tuple(HL.TILT_LIM)) == (hm.STREAM_PAN_RANGE_DEG, hm.STREAM_TILT_RANGE_DEG)
    assert PS.HEAD_MAX_AGE_S > 1.0 / 20 and PS.HEAD_MAX_AGE_S < hm.STREAM_STALE_S


def test_head_state_interp_and_stale():
    h = PS.HeadState()
    h.on_state(1_000_000_000, PS.HEAD_JOINTS, [0.0, 0.0])
    h.on_state(1_050_000_000, PS.HEAD_JOINTS, [0.1, 0.2])
    q0 = np.array(HP.fk.encoder_from_urdf(0.0, 0.0))
    q1 = np.array(HP.fk.encoder_from_urdf(np.degrees(0.1), np.degrees(0.2)))
    np.testing.assert_allclose(h.at(1_025_000_000), 0.5 * (q0 + q1), atol=1e-9)
    np.testing.assert_allclose(h.latest(1_100_000_000), q1)
    with pytest.raises(PS.PlaceInputStale):
        h.latest(1_050_000_000 + int(0.3e9))
    with pytest.raises(PS.PlaceInputStale):
        h.at(1_050_000_000 + int(0.2e9))
    with pytest.raises(PS.PlaceInputStale):
        PS.HeadState().latest(0)


def test_head_look_hold_after_settle():
    hc = HP.HeadCamPose()
    look = PS.HeadLook(hc)
    q = np.array(hc.q0_enc_deg)
    cup, hold = np.array([0.38, -0.20, 0.33]), np.array([0.38, -0.28, 0.235])
    a = json.loads(look.command(q, cup, hold, hold=False))
    np.testing.assert_allclose([a["pan_deg"], a["tilt_deg"]], HL.look_at(hc, HP.look_target(cup, hold), q), atol=1e-3)
    b = json.loads(look.command(q, cup + 0.1, hold, hold=True))      # 손 뗀 뒤 = 마지막 명령 그대로, seq 만 는다
    assert (b["pan_deg"], b["tilt_deg"]) == (a["pan_deg"], a["tilt_deg"]) and b["seq"] == a["seq"] + 1


def _frame(rd, d_m=0.6):
    hc = HP.HeadCamPose()
    T = hc.cam_pose_from_encoder(*hc.q0_enc_deg)
    pos, yaw = HPJ.load_holder_poses()["cup_holder_1"]
    pre = {"block": 8, "min_valid": 0.5, "floor_cut_z_m": PS.TABLE_Z - rd.FLOOR_MARGIN_M}
    im = PS.PlaceImage(rd, pre, 0.1, 2.0, "blue")
    d = np.full((720, 1280), int(d_m * 1000), np.uint16)
    return im, im.frame(d, 5, T, pos, yaw), T


def test_frame_holder_and_occlusion(rd):
    im, far, T = _frame(rd, 2.0)            # 측정이 홀더보다 뒤 → 홀더 보임
    assert far.depth.shape == (90, 160) and far.holder.shape == (90, 160) and far.holder.sum() > 10
    _, near, _ = _frame(rd, 0.05)           # 측정이 홀더 앞(가림) → 홀더 화소 없음
    assert near.holder.sum() == 0


def test_img2_cup_gate_keep_last_and_first_tick(rd):
    im, fr, T = _frame(rd, 1.0)
    p, R = np.array([0.38, -0.12, 0.25]), np.eye(3)
    e = CG.expected_px(p, R, im.c_cup, T, im.kb)
    bits = np.zeros((90, 160), bool)
    v, u = int(round(e[1])), int(round(e[0]))
    bits[v - 2:v + 3, u - 2:u + 3] = True
    d0, o0 = im.img2(fr, bits, p, R)
    assert not d0.any() and not o0.any() and im.vis == 1.0          # 첫 틱 = 0, 컵 유지는 갱신
    d1, o1 = im.img2(fr, None, p, R)                                 # 결측 = 마지막 유지 + 보임 0
    assert im.vis == 0.0 and im.last.sum() == 25 and not o1[..., 2].any()
    np.testing.assert_allclose(o1[..., 1], np.where((d1 >= 0.1) & (d1 <= 2.0), d1 / 2.0, 0) * im.last, atol=1e-6)
    far = np.zeros((90, 160), bool)
    far[5:10, 5:10] = True                                           # 기대 화소에서 멀다 → 거부(유지)
    _, o2 = im.img2(fr, far, p, R)
    assert im.vis == 0.0 and im.dist_px > CG.GATE_PX and o2[5:10, 5:10, 1].sum() == 0
    dn = np.where((fr.depth >= 0.1) & (fr.depth <= 2.0), fr.depth / 2.0, 0)
    np.testing.assert_allclose(o2[..., 0], dn * fr.holder, atol=1e-6)


def _meta(c, extr_md5, z):
    return {"kind": "depth", "task": "rh_place", "arch": "v3", "delta": False, "n_obs": sum(w for _, w in PS.prop_cols(c)),
            "n_act": c.action_dim, "prop_cols": PS.prop_cols(c),
            "mask": {"mode": "P", "img2": list(PS.P_IMG2), "vis_min_px": SM.VIS_MIN_PX, "holder_occ_m": HPJ.HOLDER_OCC_M,
                     "holder_mesh_md5": hdgp_pin.md5(HPJ.HOLDER_MESH)},
            "head": {"rule": "look_at_mid", "look_dz": 0.0, "holder_top_z": HP.HOLDER_TOP_Z, "pan_lim": list(HL.PAN_LIM),
                     "tilt_lim": list(HL.TILT_LIM), "sag": True, "sag_table_md5": HP.SAG_TABLE_MD5, "extr_md5": extr_md5,
                     "rate_eval_dps": 60.0, "accel_eval_dps2": 300.0, "obs": ["head_pan", "head_tilt"],
                     "cup_c_m": {k: list(v) for k, v in CG.C_DEFAULT.items()}},
            "depth_pre": {"dynamic_pose": "x", "extrinsics_md5": extr_md5, "block": 8, "floor_cut_z_m": z}}


def test_check_meta(c):
    assert c.side().side == "right"
    d = _meta(c, "e" * 32, 0.185)
    PS.check_meta(d, c, "e" * 32, 0.185)
    for path, v in ((("head", "accel_eval_dps2"), 400.0), (("head", "extr_md5"), "x"), (("mask", "mode"), "A"),
                    (("depth_pre", "floor_cut_z_m"), 0.2)):
        bad = json.loads(json.dumps(d))
        bad[path[0]][path[1]] = v
        with pytest.raises(ValueError):
            PS.check_meta(bad, c, "e" * 32, 0.185)
    bad = json.loads(json.dumps(d))
    bad["prop_cols"] = bad["prop_cols"][:-2]
    with pytest.raises(ValueError):
        PS.check_meta(bad, c, "e" * 32, 0.185)


class _Pol:
    def __init__(self):
        self.n_reset, self.carry = 0, []

    def reset(self):
        self.n_reset += 1

    def forward(self, obs, carry_on=False):
        self.carry.append(carry_on)
        raise PS.PlaceInputStale("x")


def test_chain_rnn_reset_only_at_episode_start(monkeypatch):
    ch = PS.StudentPlaceChain.__new__(PS.StudentPlaceChain)
    ch.policy = _Pol()
    monkeypatch.setattr(P.PlaceChain, "reset", lambda self, *a, **k: self._policy_reset())
    monkeypatch.setattr(P.PlaceChain, "_carry_tick", lambda self, meas: (np.zeros(3), np.zeros(2), {}))
    ch.reset({})
    ch._policy_reset()                       # 이송 끝(_place_begin) — sim 은 RNN 을 비우지 않는다
    assert ch.policy.n_reset == 1
    ch._carry_tick({})                       # 이송 틱에도 학생이 돈다(carry_on), 입력 낡음은 이송을 막지 않는다
    assert ch.policy.carry == [True] and "x" in ch.carry_input_err


def test_head_state_thread_safe():
    """H1 — 콜백(on_state)이 쓰는 동안 일꾼(at)이 읽어도 deque 변경 예외가 나지 않는다."""
    import threading
    h, stop = PS.HeadState(keep=64), threading.Event()

    def writer():
        t = 1_000_000_000
        while not stop.is_set():
            t += 1_000_000
            h.on_state(t, PS.HEAD_JOINTS, [0.0, 0.0], now_ns=t)

    sw = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                  # 스레드 전환을 잦게 — lock 없던 판은 여기서 RuntimeError(deque mutated)
    th = threading.Thread(target=writer)
    th.start()
    try:
        for _ in range(3000):
            try:
                h.at(1_000_500_000)              # 버퍼가 지나간 stamp → min 경로까지 돈다
            except PS.PlaceInputStale:
                pass
    finally:
        stop.set()
        th.join()
        sys.setswitchinterval(sw)


def test_head_state_bad_and_future_dropped():
    """H3 · M4 — 비유한 · 미래 stamp 는 예외 없이 불량으로 세고 버린다, 순서 기준을 오염시키지 않는다."""
    h, now = PS.HeadState(), 10_000_000_000
    h.on_state(now - 100_000_000, PS.HEAD_JOINTS, [0.0, 0.0], now_ns=now)
    h.on_state(now - 50_000_000, PS.HEAD_JOINTS, [float("nan"), 0.0], now_ns=now)
    h.on_state(now - 40_000_000, PS.HEAD_JOINTS, [0.0, float("inf")], now_ns=now)
    assert h.bad == 2 and len(h.buf) == 1
    h.on_state(now + int(1e9), PS.HEAD_JOINTS, [0.1, 0.1], now_ns=now)       # 1 s 미래
    assert h.bad == 3 and "미래" in h.bad_why and len(h.buf) == 1
    h.on_state(now - 10_000_000, PS.HEAD_JOINTS, [0.1, 0.2], now_ns=now)     # 정상 다음 상태는 그대로 들어간다
    assert h.bad == 0 and len(h.buf) == 2 and h.buf[-1][0] == now - 10_000_000
    assert PS.age_refusal(-0.1, 0.1, "깊이") and PS.age_refusal(float("nan"), 0.1, "깊이")
    assert PS.age_refusal(-0.04, 0.1, "깊이") is None and PS.age_refusal(0.11, 0.1, "깊이")


def test_head_state_pair_gap_and_dt():
    """M5 — 보간 간격 > HEAD_PAIR_S · 깊이가 마지막 상태보다 HEAD_PAIR_S 넘게 앞섬 = 낡음, 고른 상태와의 시각 차를 남긴다."""
    h = PS.HeadState()
    h.on_state(1_000_000_000, PS.HEAD_JOINTS, [0.0, 0.0], now_ns=2_000_000_000)
    h.on_state(1_050_000_000, PS.HEAD_JOINTS, [0.0, 0.0], now_ns=2_000_000_000)
    h.on_state(1_300_000_000, PS.HEAD_JOINTS, [0.0, 0.0], now_ns=2_000_000_000)
    h.at(1_040_000_000)
    assert h.dt_ms == pytest.approx(40.0)
    with pytest.raises(PS.PlaceInputStale, match="간격"):
        h.at(1_100_000_000)                  # 1.05 ~ 1.30 사이(0.25 s 간격)
    h.at(1_390_000_000)
    assert h.dt_ms == pytest.approx(90.0)
    with pytest.raises(PS.PlaceInputStale):
        h.at(1_300_000_000 + int(PS.HEAD_PAIR_S * 1e9) + 1_000_000)
    g = PS.HeadState()                                    # 20 Hz 에서 상태 하나 빠짐(0.1 s 간격)은 보간한다
    g.on_state(1_000_000_000, PS.HEAD_JOINTS, [0.0, 0.0], now_ns=2_000_000_000)
    g.on_state(1_100_000_000, PS.HEAD_JOINTS, [0.0, 0.0], now_ns=2_000_000_000)
    g.at(1_050_000_000)
    from policy_control.student_policy import MAX_DEPTH_AGE_S
    assert 0.1 < PS.HEAD_PAIR_S < min(PS.HEAD_MAX_AGE_S, MAX_DEPTH_AGE_S)


def test_start_refusal():
    """M6 — reset 때 목 각이 ckpt 시작 자세에서 HEAD_START_TOL_DEG 밖이면 거부."""
    q0 = np.array([10.0, 50.0])
    assert PS.start_refusal(q0 + [1.9, -1.9], q0) is None
    assert "벗어났다" in PS.start_refusal(q0 + [0.0, PS.HEAD_START_TOL_DEG + 0.1], q0)
    assert PS.start_refusal([float("nan"), 50.0], q0)


def test_chain_stale_input_does_not_advance_base(monkeypatch):
    """M3 — 입력이 낡으면 베이스 step(타이머 · est)을 부르기 전에 실패. 이송 중에는 검사하지 않는다."""
    calls = []
    monkeypatch.setattr(P.PlaceChain, "step", lambda self, meas: calls.append(1) or "ok")

    class _Fresh(_Pol):
        def fresh(self):
            raise PS.PlaceInputStale("낡음")

    ch = PS.StudentPlaceChain.__new__(PS.StudentPlaceChain)
    ch.policy, ch.rcarry = _Fresh(), None
    with pytest.raises(PS.PlaceInputStale):
        ch.step({})
    assert calls == []
    ch.rcarry = type("R", (), {"on": True})()
    assert ch.step({}) == "ok" and calls == [1]


def _node():
    N = pytest.importorskip("policy_control.rh_place_student_node")
    hc = HP.HeadCamPose()
    sp = type("SP", (), {})()
    sp.look, sp.now_ns, sp.ok_ns = PS.HeadLook(hc), 5, 5
    sp.last_palm, sp.image = (np.array([0.38, -0.20, 0.30]), np.eye(3)), type("I", (), {"c_cup": np.zeros(3)})()
    sp.head = PS.HeadState()
    node = N.PlaceStudentNode.__new__(N.PlaceStudentNode)
    node._policy, node.chain, node._String = sp, type("C", (), {"settling": False})(), (lambda data: data)
    node._head_fail, node._head_err, node._frame_fail, node._frame_err = 0, "", 0, ""
    node._worker = type("W", (), {"is_alive": lambda self: True})()
    node.pub, node.ends, node.stat = [], [], []
    node._pub_head = type("P", (), {"publish": lambda self, m: node.pub.append(json.loads(m))})()
    node._end = lambda kind, why: node.ends.append((kind, why))
    node._status = lambda ok, reasons, extra=None: node.stat.append((ok, list(reasons)))
    node._holder = np.array([0.38, -0.28, 0.235])
    return N, node, sp, np.array(hc.q0_enc_deg)


def test_node_head_cmd_fail_abort_and_hold():
    """H2 · M2 — 풀이 실패는 연속 HEAD_CMD_FAIL_MAX 틱까지 status 경고, 넘으면 abort. 낡은 틱은 새로 풀지 않고 마지막 명령 유지."""
    N, node, sp, q = _node()
    pub, ends = node.pub, node.ends
    node._head_tick(q)
    assert len(pub) == 1 and node._head_fail == 0
    sp.now_ns = 6                                            # 정책 틱 낡음 → 같은 명령, 손바닥이 움직여도 새로 안 푼다
    sp.last_palm = (np.array([0.10, 0.30, 0.50]), np.eye(3))
    node._head_tick(q)
    assert (pub[1]["pan_deg"], pub[1]["tilt_deg"]) == (pub[0]["pan_deg"], pub[0]["tilt_deg"])
    sp.ok_ns = 6
    node._holder = np.array([np.nan, 0.0, 0.0])               # look_at ValueError
    for i in range(N.HEAD_CMD_FAIL_MAX):
        node._head_tick(q)
    assert not ends and node._head_fail == N.HEAD_CMD_FAIL_MAX and "머리 명령" in node._warnings()[0]
    node._head_tick(q)
    assert ends and ends[0][0] == "abort" and len(pub) == 2


def test_node_head_fail_count_kept_on_hold():
    """재리뷰 3 — 실패 셈은 새로 푼 명령이 성공할 때만 비운다. 낡은 틱의 유지 재전송은 셈을 비우지 않는다."""
    N, node, sp, q = _node()
    node._head_tick(q)                                       # 첫 명령
    holder = node._holder
    node._holder = np.array([np.nan, 0.0, 0.0])
    for _ in range(3):
        node._head_tick(q)
    assert node._head_fail == 3
    sp.now_ns = 6                                            # 낡은 틱 — 유지 재전송은 하지만 셈은 그대로
    node._head_tick(q)
    assert len(node.pub) == 2 and node._head_fail == 3 and node._head_err
    sp.ok_ns, node._holder = 6, holder                       # 신선한 틱에서 풀이 성공 → 비운다
    node._head_tick(q)
    assert len(node.pub) == 3 and node._head_fail == 0 and not node.ends


def test_node_head_unexpected_error_aborts_and_raises():
    """재리뷰 4 — ValueError 가 아닌 예외는 status 에 남기고 abort 한 뒤 다시 던진다."""
    N, node, sp, q = _node()
    sp.last_palm = None                                      # 신선 틱인데 손바닥 없음 = 버그(TypeError)
    with pytest.raises(TypeError):
        node._head_tick(q)
    assert node.stat and node.stat[-1][0] is False and "TypeError" in node.stat[-1][1][0]
    assert node.ends and node.ends[0][0] == "abort" and "TypeError" in node.ends[0][1] and not node.pub
