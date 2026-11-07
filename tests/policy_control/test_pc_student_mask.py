"""11.07 목표 마스크 학생 A 실기 입력(policy_control.student_mask) — hdgp 78e75059 mask_scene.py 와 bit 단위 동등성 · 거부 규칙.

hdgp 원본은 `git -C <hdgp> show 78e75059:<경로>` 로 꺼내 그대로 import 한다(torch · math · os 만 쓴다).
hdgp 저장소 · torchvision · 실제 ckpt 가 없으면 해당 시험은 건너뛴다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from policy_control import hdgp_pin
from policy_control import rh_aglt as A
from policy_control import student_mask as SM
from policy_control import student_obs as SO
from policy_control.student_policy import STUDENT_CARD

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "78e75059b75e3754733e34182fefc5569b7fd0a6"
REF_MASK = "scripts/tools/student_adapters/mask_scene.py"
REF_LEARNER = "source/openarm/openarm/distillation/student/learner.py"
AGLT = REPO / "deploy/policies/rh56f1/aglt"
BUNDLES = {"right": ("rh_aglt", AGLT / "right_student1105mA", 0), "left": ("rh_aglt_l", AGLT / "left_student1105mA", 2)}
NO_HDGP = not (hdgp_pin.hdgp_tree() / ".git").exists()
MASK = {"mode": "A", "img2": list(SM.IMG2), "vis_min_px": 3, "occluded": "마지막 보인 마스크 유지 + 보임 0", "center_frame": ""}
HO = {"tmin_s": 2.0, "still_s": 0.5, "still_mps": 0.03}
K160 = [[113.54415130615234, 0.0, 79.45823669433594], [0.0, 113.48487854003906, 45.04754638671875], [0.0, 0.0, 1.0]]
DEPTH_PRE = {"d_min": 0.2, "d_max": 1.3, "block": 8, "min_valid": 0.5, "render_wh": [1280, 720], "out_wh": [160, 90], "K": K160,
             "extrinsics_md5": "624677fee7e8ff6502f33a3ed8a96a1f", "floor_cut_z_m": 0.185, "rgb": "블록 평균 0~1"}


def _ref_src(rel: str) -> str:
    if NO_HDGP:
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    r = subprocess.run(["git", "-C", str(hdgp_pin.hdgp_tree()), "show", f"{REF_COMMIT}:{rel}"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"hdgp {REF_COMMIT[:8]} {rel} 없음: {r.stderr.strip()}")
    return r.stdout


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """hdgp 78e75059 mask_scene 모듈."""
    p = tmp_path_factory.mktemp("ref") / "mask_scene_ref.py"
    p.write_text(_ref_src(REF_MASK))
    spec = importlib.util.spec_from_file_location("mask_scene_ref", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref_target_mask(ref, n, d_min=0.2, d_max=1.3):
    """TargetMask.__init__ 은 env · 카메라를 요구한다 — post 가 읽는 필드만 aug False(배포) 값으로 채운다."""
    tm = object.__new__(ref.TargetMask)
    tm.mode, tm.dev, tm.n, tm.d_min, tm.d_max = "A", torch.device("cpu"), n, d_min, d_max
    tm.last, tm.aug, tm._aug_on, tm._seed = None, None, False, 0
    tm.vis, tm.center = torch.zeros(n), torch.zeros(n, 3)
    tm.stat = torch.zeros(4, dtype=torch.float64)
    return tm


def _masks(rng, n, h, w, k):
    """(n, k, h, w) bool: 블록 · 경계를 고르게 섞으려고 사각형 몇 개 + 화소 잡음. 일부 칸은 VIS_MIN_PX 근처로 작게."""
    m = rng.random((n, k, h, w)) < 0.02
    for i in range(n):
        for c in range(k):
            for _ in range(rng.integers(0, 3)):
                y, x = rng.integers(0, h), rng.integers(0, w)
                m[i, c, y:y + rng.integers(1, h // 3 + 2), x:x + rng.integers(1, w // 3 + 2)] = True
    return m


# ---------------------------------------------------------------- (a) 동등성
def test_mask_post_is_bit_identical_to_hdgp_post(ref, monkeypatch):
    monkeypatch.delenv("DISTILL_MASK_DUMP", raising=False)
    assert SM.VIS_MIN_PX == ref.VIS_MIN_PX
    rng = np.random.default_rng(0)
    n, h, w = 4, 90, 160
    tm, mp = _ref_target_mask(ref, n), SM.MaskPost(0.2, 1.3)
    for t in range(80):
        m = torch.from_numpy(_masks(rng, n, h, w, 2).astype(np.float32))
        if t % 7 == 3:                                       # 목표가 안 보이는(< 3 px) 틱 — 마지막 마스크 유지
            m[:, 0] = 0.0
            m[0, 0, 5, 5:7] = 1.0
        m = torch.cat([m, torch.zeros(n, 1, h, w)], 1)       # hdgp tm.raw + z(rgb 칸 3 채널)
        d = torch.from_numpy(rng.uniform(0.0, 1.6, (n, h, w)).astype(np.float32))
        d[torch.from_numpy(rng.random((n, h, w)) < 0.1)] = 0.0
        d[0, 0, 0], d[0, 0, 1] = 0.2, 1.3                     # 경계값(포함)
        new_ep = torch.from_numpy(rng.random(n) < 0.1) if t else torch.ones(n, dtype=torch.bool)
        a, b = tm.post(m.clone(), d.clone(), new_ep.clone()), mp.post(m.clone(), d.clone(), new_ep.clone())
        assert a.dtype == b.dtype and torch.equal(a, b), f"tick {t}: 최대 차이 {(a - b).abs().max()}"
        assert torch.equal(tm.last, mp.last) and torch.equal(tm.vis, mp.vis)


def test_encode_decode_matches_hdgp_full_block_rule(ref):
    src = _ref_src(REF_MASK)
    line = "(m.float().reshape(n, H // b, b, W // b, b).mean((2, 4)) >= 0.5).float() for m in (t, o)"
    assert line in src and "o &= ~t" in src, "hdgp full() 블록 규칙이 바뀌었다 — student_mask.encode_bits 를 다시 맞춘다"
    rng = np.random.default_rng(1)
    b, H, W = 8, 720, 1280
    for k in range(3):
        cls = torch.from_numpy(_masks(rng, 1, H, W, 3)[0])
        bits = SM.encode_bits(cls, b).numpy()
        assert bits.shape == (90, 160) and bits.dtype == np.uint8
        for tgt in range(3):
            t = cls[tgt][None]
            o = torch.stack([cls[j] for j in range(3) if j != tgt]).any(0)[None]
            o &= ~t                                                       # hdgp full(): o &= ~t
            n = 1
            want = torch.stack([(m.float().reshape(n, H // b, b, W // b, b).mean((2, 4)) >= 0.5).float() for m in (t, o)], 1)[0]
            assert np.array_equal(SM.decode(bits, tgt), want.numpy()), f"seed {k} target {tgt}"


def test_student_input_matches_hdgp_learner_order(ref, tmp_path, monkeypatch):
    """learner.on_step: img = depth_input(d) (3 채널) ++ pipe.rgb(= post 출력) → 6 채널. 배포 StudentPolicy 가 같은 텐서를 넣는지."""
    monkeypatch.delenv("DISTILL_MASK_DUMP", raising=False)
    src = _ref_src(REF_LEARNER)
    assert "img = depth_input(d, self.d_min, self.d_max)" in src and "img = torch.cat([img, rgb.float()], 1)" in src
    SP, p, md5 = _ckpt(tmp_path)
    pol = SP.StudentPolicy(p, md5, task="rh_aglt")
    pol.set_target("pink")
    seen = []
    fwd = pol.model.forward
    pol.model.forward = lambda img, *a, **kw: (seen.append(img.clone()), fwd(img, *a, **kw))[1]
    tm = _ref_target_mask(ref, 1)
    rng = np.random.default_rng(2)
    for t in range(6):
        bits = SM.encode_bits(torch.from_numpy(_masks(rng, 1, 720, 1280, 3)[0]), 8).numpy()
        if t == 3:
            bits &= ~np.uint8(1 << 1)                                     # 목표(분홍) 사라짐 — 유지
        d = rng.uniform(0.0, 1.6, (90, 160)).astype(np.float32)
        pol.step(d, np.zeros(60, np.float32), 0.0, rgb=bits)
        m = torch.from_numpy(SM.decode(bits, 1))[None]
        want = torch.cat([pol.mod.depth_input(torch.from_numpy(d)[None], 0.2, 1.3),
                          tm.post(m, torch.from_numpy(d)[None], torch.tensor([t == 0])).float()], 1)
        if t == 0:                                                        # learner blank_new = 배포 reset 뒤 첫 틱 0
            want = torch.zeros_like(want)
        assert torch.equal(seen[-1], want), f"tick {t}"


# ---------------------------------------------------------------- 거부 · 짝짓기
def _ckpt(tmp_path, **over):
    pytest.importorskip("torchvision")
    if NO_HDGP:
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    from policy_control import student_policy as SP

    mod = SP.load_model_module()
    torch.manual_seed(0)
    m = mod.make_depth_student(60, 13, hid=64, n_aux=3, pretrained=False, arch="v3", n_vis=6)
    d = {"model": m.state_dict(), "n_obs": 60, "n_act": 13, "hid": 64, "img_hw": [90, 160], "d_min": 0.2, "d_max": 1.3,
         "kind": "depth", "arch": "v3", "task": "rh_aglt", "stages": ["approach", "grasp"], "n_aux": 3, "n_vis": 6,
         "head": False, "delta": False, "depth_pre": DEPTH_PRE, "prop_cols": [list(x) for x in SO.PROP_COLS],
         "handover": {"rule": "DeployHandover", **HO}, "mask": MASK, "one_obj": "source240_cvx"}
    d.update(over)
    p = tmp_path / "student.pt"
    torch.save(d, p)
    return SP, p, hashlib.md5(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("over,match", [({"mask": {**MASK, "mode": "B"}}, "mask"), ({"mask": {**MASK, "vis_min_px": 5}}, "mask"),
                                        ({"mask": {**MASK, "img2": SM.IMG2[::-1]}}, "mask"), ({"arch": "v2", "n_vis": 0}, "v3")])
def test_unported_mask_ckpt_is_refused(tmp_path, over, match):
    SP, p, md5 = _ckpt(tmp_path, **over)
    with pytest.raises(ValueError, match=match):
        SP.StudentPolicy(p, md5, task="rh_aglt")


def test_mask_student_needs_target_and_uint8_bits(tmp_path):
    SP, p, md5 = _ckpt(tmp_path)
    pol = SP.StudentPolicy(p, md5, task="rh_aglt")
    assert pol.mask_on and not pol.rgb_on
    d, x = np.full((90, 160), 0.6, np.float32), np.zeros(60, np.float32)
    with pytest.raises(ValueError, match="목표"):
        pol.step(d, x, 0.0, rgb=np.zeros((90, 160), np.uint8))
    pol.set_target("blue")
    for bad in (None, np.zeros((90, 160, 3), np.float32), np.zeros((90, 160), np.float32), np.zeros((45, 80), np.uint8)):
        with pytest.raises(ValueError, match="마스크"):
            pol.step(d, x, 0.0, rgb=bad)
    with pytest.raises(ValueError, match="target_class"):
        pol.set_target("green")


def _bundle(tmp_path, ckpt: Path, **over):
    """마스크 묶음(오른손 student.yaml)을 tmp 로 옮긴다 — ckpt 는 주어진 파일, YOLO 는 가짜 바이트."""
    import yaml

    d = tmp_path / "b"
    (d / "student").mkdir(parents=True)
    (d / "camera").mkdir()
    (d / "student/student.pt").write_bytes(ckpt.read_bytes())
    (d / "student/yolo.pt").write_bytes(b"yolo")
    src = BUNDLES["right"][1]
    (d / "camera/extrinsics_r20u5_40e6496.yaml").write_bytes((src / "camera/extrinsics_r20u5_40e6496.yaml").read_bytes())
    cfg = yaml.safe_load((src / STUDENT_CARD).read_text())
    cfg["ckpt"] = {**cfg["ckpt"], "path": "student/student.pt", "md5": hashlib.md5(ckpt.read_bytes()).hexdigest()}
    cfg["mask"]["yolo"] = {"path": "student/yolo.pt", "md5": hashlib.md5(b"yolo").hexdigest()}
    for k, v in over.items():
        cfg[k] = v
    (d / STUDENT_CARD).write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return d


@pytest.mark.skipif(not (BUNDLES["right"][1] / STUDENT_CARD).is_file(), reason="마스크 묶음 없음")
def test_bundle_kind_must_match_the_ckpt_kind(tmp_path):
    SP, p, _ = _ckpt(tmp_path)
    pol = SP.StudentPolicy.from_dir(_bundle(tmp_path / "ok", p))
    assert pol.mask_on and SM.CLASSES[pol.target] == "blue"
    assert SM.CLASSES[SP.StudentPolicy.from_dir(_bundle(tmp_path / "pk", p), target_class="pink").target] == "pink"
    with pytest.raises(A.RhAgltError, match="rgb"):                       # 마스크 ckpt → rgb 묶음(1104e 꼴)
        SP.StudentPolicy.from_dir(_bundle(tmp_path / "rgb", p, rgb=True, mask=None))
    with pytest.raises(A.RhAgltError, match="mask"):                      # 마스크 ckpt → 깊이 묶음(rgb · mask 없음)
        SP.StudentPolicy.from_dir(_bundle(tmp_path / "dep", p, mask=None))
    q = tmp_path / "rgbck"
    q.mkdir()
    SP, p2, _ = _ckpt(q, mask=None)
    with pytest.raises(A.RhAgltError, match="rgb"):                       # rgb ckpt → 마스크 묶음
        SP.StudentPolicy.from_dir(_bundle(tmp_path / "rev", p2))
    with pytest.raises(hdgp_pin.PinError, match="YOLO"):
        d = _bundle(tmp_path / "y", p)
        (d / "student/yolo.pt").write_bytes(b"other")
        SP.StudentPolicy.from_dir(d)


@pytest.mark.skipif(not (BUNDLES["right"][1] / STUDENT_CARD).is_file(), reason="마스크 묶음 없음")
def test_target_class_must_fit_the_ckpt_object(tmp_path):
    SP, p, _ = _ckpt(tmp_path)                                            # one_obj source240_cvx(학습 목표 blue)
    assert SP.StudentPolicy.from_dir(_bundle(tmp_path / "b", p)).target_warn is None
    assert "blue" in SP.StudentPolicy.from_dir(_bundle(tmp_path / "p", p), target_class="pink").target_warn
    with pytest.raises(A.RhAgltError, match="orange"):                    # 오른손 source240 학생에 shaker 목표
        SP.StudentPolicy.from_dir(_bundle(tmp_path / "o", p), target_class="orange")
    q = tmp_path / "sh"
    q.mkdir()
    SP, p2, _ = _ckpt(q, one_obj="shaker_c_cvx")
    with pytest.raises(A.RhAgltError, match="blue"):                      # 왼손 shaker 학생에 묶음 기본 blue
        SP.StudentPolicy.from_dir(_bundle(tmp_path / "l", p2))
    assert SM.check_target("shaker_c_cvx", "orange") is None
    with pytest.raises(ValueError, match="대응"):
        SM.check_target("cup_x", "blue")


def test_depth_and_mask_node_configs_are_checked():
    from policy_control import student_policy as SP

    node = {k: DEPTH_PRE[k] for k in ("block", "min_valid", "floor_cut_z_m", "out_wh", "render_wh", "K", "extrinsics_md5")}
    SP.check_depth_config(DEPTH_PRE, {**node, "rgb": False}, mask=True)
    with pytest.raises(A.RhAgltError, match="rgb"):
        SP.check_depth_config(DEPTH_PRE, {**node, "rgb": True}, mask=True)
    SP.check_depth_config(DEPTH_PRE, {**node, "rgb": True})                # 기존 rgb 학생 규칙 그대로
    yolo = {"md5": "abc", "conf": 0.25, "imgsz": 640, "retina_masks": True, "top1_per_class": True, "cross_class_iou": 0.5}
    cfg = {"weights_md5": "abc", "classes": list(SM.CLASSES), "block": 8, "out_wh": [160, 90], "render_wh": [1280, 720],
           "conf": 0.25, "imgsz": 640, "retina_masks": True, "top1_per_class": True, "cross_class_iou": 0.5}
    SM.check_mask_config(cfg, yolo, DEPTH_PRE)
    for k, v in (("weights_md5", "x"), ("classes", ["a", "b", "c"]), ("block", 4), ("out_wh", [320, 180]), ("conf", 0.5),
                 ("imgsz", 1280), ("retina_masks", False), ("top1_per_class", False), ("cross_class_iou", 0.0)):
        with pytest.raises(ValueError, match=k):
            SM.check_mask_config({**cfg, k: v}, yolo, DEPTH_PRE)
    with pytest.raises(ValueError, match="conf"):                         # 묶음에 검출 값이 없으면 대조하지 못한다
        SM.check_mask_config(cfg, {"md5": "abc"}, DEPTH_PRE)


def test_pick_mask_pairs_within_half_a_frame_else_uses_the_newest_with_its_age():
    ms = 1_000_000
    a, b, c = (np.full((2, 2), i, np.uint8) for i in (1, 2, 3))
    buf = [(100 * ms, a), (133 * ms, b), (166 * ms, c)]
    assert SM.pick_mask([], 0) == (None, "none", None)
    m, src, age = SM.pick_mask(buf, 140 * ms)
    assert m is b and src == "paired" and age == pytest.approx(0.007)
    m, src, age = SM.pick_mask(buf, 260 * ms)                             # 마스크가 늦다 — 가장 최근 것, 짝 아님
    assert m is c and src == "latest" and age == pytest.approx(0.094)


def _mmsg(st_ms: int, target_px: int = 10, bad: bool = False):
    bits = np.zeros((90, 160), np.uint8)
    bits.flat[:target_px] = 1                                             # 비트 0 = blue
    sec, ns = divmod(st_ms * 1_000_000, 10**9)
    return type("M", (), {"encoding": "8UC3" if bad else "mono8", "height": 90, "width": 160, "step": 160,
                          "data": bits.tobytes(), "header": type("H", (), {"stamp": type("S", (), {"sec": sec, "nanosec": ns})})})


def test_mask_stream_errors_only_on_stale_or_broken_stream_not_on_empty_detection():
    ms = 1_000_000
    s = SM.MaskStream((90, 160), max_age_s=0.2, bad_max=3, window=4)
    with pytest.raises(SM.MaskStale, match="못 받았다"):                    # 한 번도 없음
        s.bits_for(1000 * ms, 0)
    assert "못 받았다" in s.alive(1000 * ms)
    s.on_msg(_mmsg(1000))
    assert s.alive(1100 * ms) is None and "끊겼나" in s.alive(1300 * ms)    # stamp 기준 생존
    assert s.bits_for(1010 * ms, 0).shape == (90, 160) and s.src == "paired"
    s.bits_for(1150 * ms, 0)                                              # 짝 없음, 나이 150 ms < 상한
    assert s.src == "latest"
    with pytest.raises(SM.MaskStale, match="나이"):                        # 단절 — 나이 상한 넘음
        s.bits_for(1250 * ms, 0)
    for k in range(5):                                                    # 빈 검출(목표 0 px)은 오류가 아니다 — 놓침만 센다
        s.on_msg(_mmsg(1300 + 33 * k, target_px=0))
        s.bits_for((1300 + 33 * k) * ms, 0)
    assert s.status()["target_miss"] == 5 and s.status()["bad_streak"] == 0
    for _ in range(3):                                                    # 불량 연속
        s.on_msg(_mmsg(1500, bad=True))
    assert "불량" in s.alive(1500 * ms)
    with pytest.raises(SM.MaskStale, match="불량"):
        s.bits_for(1500 * ms, 0)
    s.reset()                                                             # 에피소드 경계 — 불량 · 통계만 비운다
    assert s.status() == {"src": "none", "age_ms": None, "unpaired_ratio": None, "bad_streak": 0, "target_miss": 0}
    with pytest.raises(SM.MaskStale, match="나이"):                        # 남은 마지막 정상 마스크(1432)도 나이 상한으로 거른다
        s.bits_for(1700 * ms, 0)


def test_mask_stream_reset_keeps_fresh_mask_so_first_tick_is_not_stale():
    ms = 1_000_000
    s = SM.MaskStream((90, 160), max_age_s=0.2, bad_max=3, window=4)
    s.on_msg(_mmsg(1000))
    s.on_msg(_mmsg(1010, bad=True))
    s.reset()                                                             # 에피소드 시작 직후 YOLO 다음 프레임이 아직 없다
    assert s.status()["bad_streak"] == 0
    assert s.bits_for(1150 * ms, 0).shape == (90, 160) and s.src == "latest"   # 0.2 s 안 마스크 → MaskStale 아님


def test_mask_stream_unpaired_ratio_counts_late_pairs():
    ms = 1_000_000
    s = SM.MaskStream((90, 160), max_age_s=0.5, window=4)
    s.on_msg(_mmsg(1000))
    for k, t in enumerate((1000, 1033, 1066, 1100)):                      # 깊이 4 프레임, 마스크는 첫 것만
        s.bits_for(t * ms, 0)
    assert s.unpaired_ratio() == 0.75
    s.on_msg(_mmsg(1066))                                                 # 늦게 온 짝도 짝으로 센다
    assert s.unpaired_ratio() == 0.5
    with pytest.raises(ValueError):
        SM.MaskStream((90, 160), max_age_s=0.0)


def test_mask_array_refuses_wrong_messages():
    class Msg:
        encoding, height, width, step, data = "mono8", 90, 160, 160, bytes(90 * 160)

    assert SM.mask_array(Msg, (90, 160)).shape == (90, 160)
    for k, v in (("encoding", "8UC3"), ("width", 80), ("step", 161), ("data", bytes([64]) + bytes(90 * 160 - 1))):
        bad = type("Bad", (Msg,), {k: v})
        with pytest.raises(ValueError):
            SM.mask_array(bad, (90, 160))


# ---------------------------------------------------------------- (c) 실제 ckpt
@pytest.mark.parametrize("side", ["right", "left"])
def test_real_mask_students_load_from_their_bundle_and_step(side):
    pytest.importorskip("torchvision")
    if NO_HDGP:
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    task, d, tgt = BUNDLES[side]
    if not (d / STUDENT_CARD).is_file() or not list((d / "student").glob("student_it*.pt")):
        pytest.skip(f"마스크 학생 ckpt 없음: {d}")
    from policy_control import student_policy as SP

    pol = SP.StudentPolicy.from_dir(d)
    assert pol.mask_on and not pol.rgb_on and pol.target == tgt and pol.meta["task"] == task and pol.model.n_vis == 6
    rng = np.random.default_rng(3)
    acts = []
    for k in range(4):
        bits = SM.encode_bits(torch.from_numpy(_masks(rng, 1, 720, 1280, 3)[0]), 8).numpy()
        acts.append(pol.step(rng.uniform(0.0, 1.5, (90, 160)).astype(np.float32), rng.normal(size=60).astype(np.float32), 0.03,
                             rgb=bits))
    assert all(a.shape == (13,) and np.isfinite(a).all() and np.abs(a).max() <= 1.0 for a in acts)
