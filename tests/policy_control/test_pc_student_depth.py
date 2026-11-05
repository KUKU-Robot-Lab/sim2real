"""11.04 FP++ 없는 깊이 학생(rh_aglt 잡기) 배포 모듈 — 적재 · 관측 칸 · 넘김 규칙.

hdgp model.py 는 고정 커밋 사본에서 읽는다(config/hdgp_student_snapshot.yaml, hdgp git = HDGP_DIR 기본 ~/rl_ws/hdgp).
hdgp 저장소 · torchvision 이 없으면 (a) 는 건너뛴다.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_control import hdgp_pin
from policy_control import rh_aglt as A
from policy_control import student_obs as SO
from policy_control.student_handover import StudentHandover
from policy_control.student_policy import STUDENT_CARD

REPO = Path(__file__).resolve().parents[2]
RCG04 = REPO / "deploy" / "policies" / "rh56f1" / "aglt" / "right_rcg04" / "rh_aglt_contract.json"
HO = {"tmin_s": 2.0, "still_s": 0.5, "still_mps": 0.03}
HZ = 60.0
K160 = [[113.54415130615234, 0.0, 79.45823669433594], [0.0, 113.48487854003906, 45.04754638671875], [0.0, 0.0, 1.0]]   # scale_k(D435I_K_1103, 8)
#: hdgp real_pipe.meta() 키 구성(origin/main ed11c791) — 값은 r20u5 · D435I_K_1103 · 블록 8 학습 기본
DEPTH_PRE = {"d_min": 0.2, "d_max": 1.3, "normalize": "d / d_max", "block": 8, "min_valid": 0.5, "rule": "블록 중앙값",
             "render_wh": [1280, 720], "out_wh": [160, 90], "K": K160, "K_convention": "OpenCV", "K_source": "const:D435I_K_1103",
             "extrinsics": "/train/config/global_camera_extrinsics_arm4090_r20u5.yaml",
             "extrinsics_md5": "624677fee7e8ff6502f33a3ed8a96a1f", "intrinsics": "", "intrinsics_md5": "",
             "cam_hz": 30.0, "hold_ticks": 2, "delay_ticks": [0, 2], "floor_cut_z_m": 0.185, "dr": None}
NO_HDGP = not (hdgp_pin.hdgp_tree() / ".git").exists()


# ---------------------------------------------------------------- (a) 적재 · step · reset
def _ckpt(tmp_path, arch="v2", **over):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from policy_control import student_policy as SP

    if NO_HDGP:
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    mod = SP.load_model_module()
    torch.manual_seed(0)
    n_vis = 6 if arch == "v3" else 0
    m = mod.make_depth_student(60, 13, hid=64, n_aux=3, pretrained=False, arch=arch, n_vis=n_vis)
    pre = {**DEPTH_PRE, "rgb": "블록 평균 0~1"} if arch == "v3" else DEPTH_PRE
    d = {"model": m.state_dict(), "n_obs": 60, "n_act": 13, "hid": 64, "img_hw": [90, 160], "it": 1, "d_min": 0.2,
         "d_max": 1.3, "kind": "depth", "arch": arch, "task": "rh_aglt", "stages": ["approach", "grasp"], "n_aux": 3,
         "n_vis": n_vis, "head": False, "delta": False, "delta_cols": [], "depth_pre": pre,
         "prop_cols": [list(x) for x in SO.PROP_COLS], "handover": {"rule": "DeployHandover", **HO}}
    d.update(over)
    p = tmp_path / "student.pt"
    torch.save(d, p)
    return SP, p, hashlib.md5(p.read_bytes()).hexdigest()


def _inputs(k):
    rng = np.random.default_rng(k)
    return rng.uniform(0.0, 1.5, (90, 160)).astype(np.float32), rng.normal(size=60).astype(np.float32)


def test_load_step_shape_and_reset_is_deterministic(tmp_path):
    SP, p, md5 = _ckpt(tmp_path)
    pol = SP.StudentPolicy(p, md5, task="rh_aglt")
    assert pol.handover == HO and pol.img_hw == (90, 160)
    seq = [_inputs(k) for k in range(4)]
    a1 = [pol.step(*x, 0.0) for x in seq]
    assert a1[0].shape == (13,) and np.all(np.abs(a1[0]) <= 1.0)
    assert not np.allclose(a1[1], a1[2]) or not np.allclose(a1[2], a1[3])    # LSTM 상태가 이어진다
    pol.reset()
    a2 = [pol.step(*x, 0.0) for x in seq]
    np.testing.assert_allclose(np.array(a1), np.array(a2), atol=1e-6)


def test_first_tick_after_reset_ignores_depth(tmp_path):
    SP, p, md5 = _ckpt(tmp_path)
    pol = SP.StudentPolicy(p, md5, task="rh_aglt")
    d1, x = _inputs(1)
    a = pol.step(d1, x, 0.0)
    pol.reset()
    b = pol.step(np.zeros_like(d1), x, 0.0)
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("over, match", [({"delta": True}, "형식"), ({"arch": "v1"}, "형식"), ({"kind": "rgb"}, "형식"),
                                         ({"task": "rh_aglt_l"}, "형식"), ({"head": True}, "형식"),
                                         ({"prop_cols": [list(x) for x in SO.PROP_COLS[::-1]]}, "prop_cols"),
                                         ({"handover": {"rule": "sim6"}}, "넘김"),
                                         ({"handover": {"rule": "DeployHandover", **HO, "deploy_uses_rr": True}}, "넘김"), ({"depth_pre": None}, "depth_pre"),
                                         ({"depth_pre": {}}, "depth_pre")])
def test_wrong_ckpt_is_refused(tmp_path, over, match):
    SP, p, md5 = _ckpt(tmp_path, **over)
    with pytest.raises(ValueError, match=match):
        SP.StudentPolicy(p, md5, task="rh_aglt")


def test_wrong_md5_and_missing_weight_are_refused(tmp_path):
    SP, p, md5 = _ckpt(tmp_path)
    with pytest.raises(ValueError, match="md5"):
        SP.StudentPolicy(p, "0" * 32, task="rh_aglt")
    import torch

    d = torch.load(p, weights_only=False)
    d["model"].pop("head.bias")
    torch.save(d, p)
    with pytest.raises(RuntimeError, match="head.bias"):
        SP.StudentPolicy(p, hashlib.md5(p.read_bytes()).hexdigest(), task="rh_aglt")


def test_depth_after_the_first_tick_changes_the_action(tmp_path):
    SP, p, md5 = _ckpt(tmp_path)
    pol = SP.StudentPolicy(p, md5, task="rh_aglt")
    (d1, x), (d2, _), (d3, _) = _inputs(1), _inputs(2), _inputs(3)
    pol.step(d1, x, 0.0)
    a = pol.step(d2, x, 0.0)
    pol.reset()
    pol.step(d1, x, 0.0)
    b = pol.step(d3, x, 0.0)
    assert not np.allclose(a, b, atol=1e-6)


def test_stale_depth_is_raised_and_empty_depth_is_flagged(tmp_path):
    SP, p, md5 = _ckpt(tmp_path)
    pol = SP.StudentPolicy(p, md5, task="rh_aglt", max_depth_age_s=0.2, empty_warn_ticks=3)
    d, x = _inputs(1)
    for age in (0.21, float("nan")):
        with pytest.raises(SP.StaleDepth):
            pol.step(d, x, age)
    flags = [(pol.step(np.zeros_like(d), x, 0.0), pol.depth_empty)[1] for _ in range(3)]
    assert flags == [False, False, True]
    pol.step(d, x, 0.0)
    assert not pol.depth_empty


@pytest.mark.parametrize("key, val", [("block", 4), ("out_wh", [320, 180]), ("render_wh", [640, 360]),
                                      ("extrinsics_md5", "0" * 32), ("min_valid", 0.5 + 1e-5),
                                      ("floor_cut_z_m", 0.186), ("K", np.add(K160, [[0, 0, 2e-3], [0, 0, 0], [0, 0, 0]]).tolist()),
                                      ("K", None)])
def test_check_depth_config_refuses_a_mismatch(key, val):
    from policy_control import student_policy as SP

    node = {k: DEPTH_PRE[k] for k in ("block", "min_valid", "floor_cut_z_m", "out_wh", "render_wh", "K", "extrinsics_md5")}
    SP.check_depth_config(DEPTH_PRE, {**node, "K": np.add(K160, 5e-4).tolist(), "extrinsics": "/deploy/r20u5.yaml"})
    with pytest.raises(A.RhAgltError, match=key):
        SP.check_depth_config(DEPTH_PRE, {**node, key: val})


def test_pinned_files_are_refused_on_md5_mismatch(tmp_path, monkeypatch):
    if NO_HDGP:
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    import yaml

    from policy_control import student_policy as SP

    cfg = yaml.safe_load(SP.SNAPSHOT_CFG.read_text())
    _, src = SP.load_pinned(SP.REAL_DEPTH, "t_rd")
    assert src["commit"] == cfg["commit"] and src["md5"] == cfg["md5"][SP.REAL_DEPTH] and "/s2r_hdgp/" in src["path"]
    bad = tmp_path / "pin.yaml"
    bad.write_text(yaml.safe_dump({**cfg, "md5": {**cfg["md5"], SP.REAL_DEPTH: "0" * 32}}))
    with pytest.raises(hdgp_pin.PinError, match="md5"):
        SP.load_pinned(SP.REAL_DEPTH, "t_rd", bad)
    monkeypatch.setenv("HDGP_STUDENT_SNAPSHOT_COMMIT", "b5e6857a8^")      # real_depth.py 가 다른 옛 커밋 — 핀은 그대로라 거부
    with pytest.raises(hdgp_pin.PinError, match="md5"):
        SP.load_pinned(SP.REAL_DEPTH, "t_rd")


# ---------------------------------------------------------------- (a2) v3 깊이 + RGB · 학생 묶음 폴더
STUDENTS = {"right": ("rh_aglt", REPO / "deploy/policies/rh56f1/aglt/right_student1104e"),
            "left": ("rh_aglt_l", REPO / "deploy/policies/rh56f1/aglt/left_student1104e")}


def _rgb(k):
    return np.random.default_rng(100 + k).uniform(0.0, 1.0, (90, 160, 3)).astype(np.float32)


def test_v3_needs_rgb_and_rgb_changes_the_action(tmp_path):
    SP, p, md5 = _ckpt(tmp_path, arch="v3")
    pol = SP.StudentPolicy(p, md5, task="rh_aglt")
    assert pol.rgb_on and pol.model.n_vis == 6
    d, x = _inputs(1)
    with pytest.raises(ValueError, match="rgb"):
        pol.step(d, x, 0.0)
    for bad in (_rgb(0)[:, :, :2], _rgb(0) * 2.0, np.full((90, 160, 3), np.nan, np.float32)):
        with pytest.raises(ValueError, match="rgb"):
            pol.step(d, x, 0.0, rgb=bad)
    pol.reset()
    first = pol.step(d, x, 0.0, rgb=_rgb(1))
    pol.reset()
    np.testing.assert_allclose(first, pol.step(np.zeros_like(d), x, 0.0, rgb=np.zeros((90, 160, 3), np.float32)), atol=1e-6)
    a = pol.step(d, x, 0.0, rgb=_rgb(1))
    pol.reset()
    pol.step(d, x, 0.0, rgb=_rgb(1))
    b = pol.step(d, x, 0.0, rgb=_rgb(2))
    assert not np.allclose(a, b, atol=1e-6)


def test_v2_refuses_rgb_and_v3_without_depth_pre_rgb_is_refused(tmp_path):
    SP, p, md5 = _ckpt(tmp_path)
    pol = SP.StudentPolicy(p, md5, task="rh_aglt")
    with pytest.raises(ValueError, match="rgb"):
        pol.step(*_inputs(1), 0.0, rgb=_rgb(1))
    SP, p, md5 = _ckpt(tmp_path, arch="v3", depth_pre=DEPTH_PRE)
    with pytest.raises(ValueError, match="rgb"):
        SP.StudentPolicy(p, md5, task="rh_aglt")


def test_check_depth_config_needs_the_node_rgb_flag_for_v3():
    from policy_control import student_policy as SP

    node = {k: DEPTH_PRE[k] for k in ("block", "min_valid", "floor_cut_z_m", "out_wh", "render_wh", "K", "extrinsics_md5")}
    pre = {**DEPTH_PRE, "rgb": "블록 평균 0~1"}
    SP.check_depth_config(pre, {**node, "rgb": True})
    with pytest.raises(A.RhAgltError, match="rgb"):
        SP.check_depth_config(pre, {**node, "rgb": False})
    with pytest.raises(A.RhAgltError, match="rgb"):
        SP.check_depth_config(DEPTH_PRE, {**node, "rgb": True})


def _bundle(tmp_path, **over):
    """오른손 student.yaml 을 tmp 로 옮긴 묶음(ckpt 는 빈 파일) → 폴더."""
    import yaml

    src = STUDENTS["right"][1]
    d = tmp_path / "b"
    (d / "student").mkdir(parents=True)
    (d / "camera").mkdir()
    (d / "student/student_it02600.pt").write_bytes(b"")
    (d / "camera/extrinsics_r20u5_40e6496.yaml").write_bytes((src / "camera/extrinsics_r20u5_40e6496.yaml").read_bytes())
    cfg = yaml.safe_load((src / STUDENT_CARD).read_text())
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    (d / STUDENT_CARD).write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return d


@pytest.mark.skipif(not (REPO / "deploy/policies/rh56f1/aglt/right_student1104e" / STUDENT_CARD).is_file(), reason="학생 묶음 없음")
def test_student_dir_refuses_moved_extrinsics_commit_and_escaping_paths(tmp_path, monkeypatch):
    from policy_control import student_policy as SP

    cfg = SP.load_student_dir(_bundle(tmp_path))
    assert cfg["rgb"] and cfg["task"] == "rh_aglt" and cfg["extrinsics"]["md5"] == "624677fee7e8ff6502f33a3ed8a96a1f"
    d = _bundle(tmp_path / "x")
    (d / "camera/extrinsics_r20u5_40e6496.yaml").write_text("camera: {}\n")
    with pytest.raises(hdgp_pin.PinError, match="외부 파라미터"):
        SP.load_student_dir(d)
    with pytest.raises(hdgp_pin.PinError, match="hdgp.commit"):
        SP.load_student_dir(_bundle(tmp_path / "y", hdgp={"commit": "0" * 40}))
    with pytest.raises(hdgp_pin.PinError, match="폴더 밖"):
        SP.load_student_dir(_bundle(tmp_path / "z", extrinsics={"path": "../../../../etc/hostname"}))
    monkeypatch.setenv("HDGP_STUDENT_SNAPSHOT_COMMIT", "bd06f8c06")
    with pytest.raises(hdgp_pin.PinError, match="hdgp.commit"):
        SP.load_student_dir(_bundle(tmp_path / "w"))


@pytest.mark.parametrize("side", ["right", "left"])
def test_real_1104e_students_load_from_their_bundle_and_step(side):
    pytest.importorskip("torchvision")
    if NO_HDGP:
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    task, d = STUDENTS[side]
    if not (d / STUDENT_CARD).is_file() or not any((d / "student").glob("*.pt")):
        pytest.skip(f"학생 ckpt 없음: {d}")
    from policy_control import student_policy as SP

    pol = SP.StudentPolicy.from_dir(d)
    assert pol.rgb_on and pol.meta["arch"] == "v3" and pol.meta["task"] == task and pol.model.n_vis == 6
    assert pol.handover == HO and pol.depth_pre["floor_cut_z_m"] == pytest.approx(0.185)
    acts = [pol.step(dd, x, 0.03, rgb=_rgb(k)) for k, (dd, x) in enumerate(_inputs(k) for k in range(3))]
    assert all(a.shape == (13,) and np.isfinite(a).all() and np.abs(a).max() <= 1.0 for a in acts)


# ---------------------------------------------------------------- (b) 관측 칸
@pytest.fixture(scope="module")
def c():
    return A.select_object(A.load_contract(RCG04), "source200_real")


def _meas(c, cup=(0.35, -0.16, 0.29)):
    s = c.side()
    return A.RaMeas(arm_q=np.asarray(s.arm_home) + 0.05, arm_qd=np.linspace(-0.3, 0.3, 7),
                    hand_q={j: q + 0.1 for j, q in zip(s.hand_joints, s.hand_open)},
                    palm_pos=np.array([0.2, -0.16, 0.3]), palm_R=np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                    tips=np.tile([0.25, -0.16, 0.28], (5, 1)), cup_pos=np.asarray(cup, float),
                    cup_quat=np.array([1.0, 0, 0, 0]), tactile_n=np.zeros(5))


def test_student_columns_match_prop_cols_and_teacher_values(c):
    assert SO.student_layout(c) == SO.PROP_COLS and sum(w for _, w in SO.PROP_COLS) == 60
    m, dec, prev = _meas(c), A.RaDecoder(c), np.linspace(-1, 1, 13)
    teacher = A.build_obs(c, m, dec, A.first_goal(c, m.cup_pos, m.cup_quat), prev)
    got = SO.build_student_obs(c, m, dec, prev, "approach")
    assert got.shape == (60,)
    i, j = 0, 0
    span = {}
    for name, w in A.actor_layout(c):
        span[name] = teacher[i:i + w]
        i += w
    for name, w in SO.PROP_COLS[:-2]:
        np.testing.assert_allclose(got[j:j + w], span[name], err_msg=name)
        j += w
    assert list(got[-2:]) == [1.0, 0.0] and list(SO.build_student_obs(c, m, dec, prev, "grasp")[-2:]) == [0.0, 1.0]
    np.testing.assert_allclose(got[55:58], np.eye(3)[c.object_names.index("source200_real")])   # object_id


def test_student_obs_ignores_the_cup_and_refuses_a_tactile_contract(c):
    dec, prev = A.RaDecoder(c), np.zeros(13)
    a = SO.build_student_obs(c, _meas(c), dec, prev, "grasp")
    b = SO.build_student_obs(c, _meas(c, cup=(np.nan, np.nan, np.nan)), dec, prev, "grasp")
    np.testing.assert_array_equal(a, b)
    tac = A.with_run(c, actor_tactile_obs=True, obs_dim=c.obs_dim + 5)
    with pytest.raises(A.RhAgltError, match="학생 관측 칸"):
        SO.build_student_obs(tac, _meas(c), A.RaDecoder(tac), prev, "grasp")


def test_student_obs_is_not_clipped_and_refuses_non_finite_inputs(c):
    dec, prev = A.RaDecoder(c), np.zeros(13)
    m = replace(_meas(c), arm_qd=np.full(7, 8.0))
    got = SO.build_student_obs(c, m, dec, prev, "grasp")
    np.testing.assert_allclose(got[7:14], 8.0)                       # arm_qd 칸 — obs_clip(5) 를 거치지 않는다
    for k, v in (("arm_q", np.full(7, np.nan)), ("palm_pos", np.array([0.2, np.inf, 0.3])),
                 ("hand_q", {j: np.nan for j in c.side().hand_joints})):
        with pytest.raises(A.RhAgltError, match=k):
            SO.build_student_obs(c, replace(_meas(c), **{k: v}), dec, prev, "grasp")


# ---------------------------------------------------------------- (c) 넘김
def _ho(**kw):
    return StudentHandover(hz=HZ, max_s=kw.pop("max_s", 15.0), on_max=kw.pop("on_max", "fail"), **{**HO, **kw})


def _run(h, palms):
    return [h.step(p) for p in palms]


def _still(n, at=(0.3, -0.1, 0.2)):
    return [np.asarray(at, float)] * n


def test_no_handover_before_tmin_then_handover_on_the_first_still_window():
    h = _ho()
    out = _run(h, _still(121))              # 호출 0..120: sim t = n − 1, T_min 120 틱 → 호출 121 에서 처음
    assert all(s == ("approach", False, False) for s in out)
    assert h.step(_still(1)[0]) == ("grasp", True, False)
    assert h.n == 122


def test_a_jump_inside_the_window_delays_handover_until_it_leaves_the_window():
    p0 = np.array([0.3, -0.1, 0.2])
    h = _ho()
    _run(h, _still(110, p0))
    out = _run(h, _still(30, p0 + [0.02, 0, 0]))     # 2 cm 이동(창 평균 0.04 m/s) — 그 위치가 창(30 틱)에 남는 동안 못 넘긴다
    assert all(s[0] == "approach" for s in out)
    assert h.step(p0 + [0.02, 0, 0])[0] == "grasp" and h.n == 141      # 멈춤만이면 호출 121 에서 넘긴다
    h2 = _ho()                                        # 창 평균 경계: 창 동안 1.4 cm(0.028 m/s)는 넘긴다
    _run(h2, [p0 + [0.0014 * k / 3, 0, 0] for k in range(121)])
    assert h2.step(p0 + [0.0014 * 121 / 3, 0, 0])[0] == "grasp"


def test_after_handover_stage_never_goes_back_and_stop_lasts_half_a_second():
    h = _ho()
    _run(h, _still(121))
    first = h.step(_still(1)[0])
    rest = _run(h, [np.array([1.0, 1.0, 1.0]) * k for k in range(100)])   # 넘긴 뒤에는 위치를 보지 않는다
    stops = [first[1]] + [s[1] for s in rest]
    assert stops[:30] == [True] * 30 and not any(stops[30:])          # 정지 30 틱(0.5 s) = 넘김 틱 포함
    assert all(s[0] == "grasp" and not s[2] for s in rest)
    hold, stu = np.full(13, -1.0), np.full(13, 0.7)
    assert np.array_equal(h.action(True, stu, hold), hold) and np.array_equal(h.action(False, stu, hold), stu)


@pytest.mark.parametrize("on_max, want", [("fail", ("approach", False, True)), ("handover", ("grasp", True, False))])
def test_approach_limit_fails_or_hands_over(on_max, want):
    h = _ho(max_s=3.0, on_max=on_max)
    moving = [np.array([0.001 * k, 0, 0]) for k in range(180)]          # 0.06 m/s — 끝까지 안 멈춘다
    out = _run(h, moving)
    assert all(s == ("approach", False, False) for s in out)
    assert h.step(np.array([0.18, 0, 0])) == want


def test_tmin_shorter_than_the_window_is_refused():
    with pytest.raises(ValueError, match="tmin_s"):
        StudentHandover(hz=HZ, max_s=6.0, on_max="fail", tmin_s=0.2, still_s=0.5, still_mps=0.03)


def test_handover_args_are_required():
    with pytest.raises(TypeError):
        StudentHandover(hz=HZ, **HO)
    with pytest.raises(ValueError, match="on_max"):
        StudentHandover(hz=HZ, max_s=6.0, on_max="maybe", **HO)
