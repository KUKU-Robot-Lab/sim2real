"""11.08 놓기 학생 실기 입력 기반 모듈(place_head_pose · place_holder_proj · place_cup_gate) — hdgp 학습 정의와 수치 대조.

hdgp 원본은 `git -C <hdgp> show 43074945:<경로>` 로 꺼내 import 한다(head.py · holder_proj.py 는 torch 만 쓴다).
hdgp 저장소 · bag 분석 자료(logs/cup_seg_1107)가 없으면 해당 시험은 건너뛴다.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_control import hdgp_pin
from policy_control import place_cup_gate as G
from policy_control import place_head_pose as P
from policy_control import place_holder_proj as H

REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "43074945a017d2b25a3e23f9311a1009c3950bbc"
PLACE_COMMIT = "2079548988b6277de9e40dc511350a946166abaf"   # place_head.py · real_pipe set_pose(head.py · holder_proj.py 는 43074945 와 같음)
SEG = REPO / "logs" / "cup_seg_1107"
K_FULL = np.array([[908.3532104492188, 0.0, 639.1658935546875], [0.0, 907.8790283203125, 363.88037109375], [0.0, 0.0, 1.0]])
CAMSET_RAD = (0.3413940734670227, 1.1776177590379322)   # 11.06 bag /head/joint_states(pan 1825 · tilt 2815 틱)


def _git_show(rel: str, commit: str = REF_COMMIT) -> str:
    if not (hdgp_pin.hdgp_tree() / ".git").exists():
        pytest.skip(f"hdgp 저장소 없음: {hdgp_pin.hdgp_tree()}")
    r = subprocess.run(["git", "-C", str(hdgp_pin.hdgp_tree()), "show", f"{commit}:{rel}"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"hdgp {commit[:9]} {rel} 없음: {r.stderr.strip()}")
    return r.stdout


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclass 가 모듈을 sys.modules 에서 찾는다
    spec.loader.exec_module(mod)
    return mod


def _ref(tmp_path_factory, rel: str, name: str, commit: str = REF_COMMIT):
    p = tmp_path_factory.mktemp(name) / f"{name}.py"
    p.write_text(_git_show(rel, commit))
    return _load(p, name)


def test_head_pose_start_equals_extrinsics():
    hc = P.HeadCamPose()
    assert np.abs(hc.cam_pose_from_joint_state(*CAMSET_RAD) - hc.T_base_cam0).max() < 1e-6
    with pytest.raises(ValueError):
        hc.cam_pose_from_joint_state(float("nan"), 1.0)


@pytest.mark.parametrize("sag", [False, True])
def test_head_pose_matches_hdgp_head_model(tmp_path_factory, sag):
    """hdgp HeadModel.cam_pose(ideal=True: 처짐 차는 넣고 yaw · 오르내림 무작위는 끔)와 여러 각도에서 같다. 표는 sim2real 사본(md5 같음)."""
    torch = pytest.importorskip("torch")
    head = _ref(tmp_path_factory, "source/openarm/openarm/distillation/student/head.py", "head_ref", PLACE_COMMIT)
    hc = P.HeadCamPose(sag=sag)
    d = __import__("yaml").safe_load(P.EXTRINSICS.read_text())["camera"]
    cfg = head.HeadCfg(cam_pos=tuple(d["position"]), cam_quat_wxyz=tuple(d["orientation_wxyz"]), calib_enc_deg=hc.q0_enc_deg,
                       home_enc_deg=hc.q0_enc_deg, sag=sag, sag_table=str(P.SAG_TABLE), randomize=False, yaw_deg=0.0)
    hm = head.HeadModel(1, cfg=cfg)
    err = 0.0
    for pan, tilt in [(-19.56, 67.47), (0.0, 75.0), (-45.0, 82.0), (30.0, -2.0), (-30.0, 40.0), (25.0, 10.0), (-5.0, 55.0)]:
        t, q = hm.cam_pose(torch.tensor([[pan, tilt]], dtype=torch.float32), ideal=True)
        T = hc.cam_pose_from_joint_state(math.radians(-pan), math.radians(tilt))      # joint_states = URDF, pan 부호 반대
        err = max(err, np.abs(T[:3, 3] - t[0].double().numpy()).max(),
                  np.abs(T[:3, :3] - P.quat_wxyz_to_mat(q[0].double().numpy())).max())
    print(f"sag={sag} 최대 오차 {err:.2e}")
    assert err < 2e-5


def test_sag_table_md5_and_shift():
    hc = P.HeadCamPose()
    assert hc.sag.md5 == P.SAG_TABLE_MD5
    a, b = hc.cam_pose_from_encoder(0.0, 75.0), P.HeadCamPose(sag=False).cam_pose_from_encoder(0.0, 75.0)
    assert np.abs(a - b).max() > 1e-4          # 시작 자세가 아닌 곳에서는 처짐이 실제로 들어간다
    with pytest.raises(ValueError):
        P.SagTable(md5="0" * 32)


def _real_pipe_ref(tmp_path_factory):
    """hdgp real_pipe.py(+ 상대 import 하는 real_depth · head)를 패키지째 꺼낸다."""
    root = tmp_path_factory.mktemp("rp_ref") / "rp_ref"
    (root / "student").mkdir(parents=True)
    src = "source/openarm/openarm/distillation/"
    for rel, dst in [("real_depth.py", "real_depth.py"), ("student/real_pipe.py", "student/real_pipe.py"),
                     ("student/head.py", "student/head.py")]:
        (root / dst).write_text(_git_show(src + rel, PLACE_COMMIT))
    (root / "__init__.py").write_text(""); (root / "student" / "__init__.py").write_text("")
    sys.path.insert(0, str(root.parent))
    try:
        return (importlib.import_module("rp_ref.student.real_pipe"), importlib.import_module("rp_ref.real_depth"),
                importlib.import_module("rp_ref.student.head"))
    finally:
        sys.path.remove(str(root.parent))


def test_floor_cut_matches_hdgp_real_pipe(tmp_path_factory):
    """같은 깊이 · 자세로 hdgp RealDepthPipe.set_pose → _front(블록 1, 증강 끔 = 바닥 컷 + mm 반올림)와 비교 + 1280x720 시간."""
    import time

    torch = pytest.importorskip("torch")
    rp, rd, hm = _real_pipe_ref(tmp_path_factory)
    hc = P.HeadCamPose()
    z_m = 0.0 - rd.FLOOR_MARGIN_M
    rng = np.random.default_rng(0)
    for q in [CAMSET_RAD, (0.0, 1.25), (0.6, 0.9), (-0.5, 0.3)]:
        T = hc.cam_pose_from_joint_state(*q)
        d = rng.uniform(0.2, 2.5, (720, 1280)).astype(np.float32)
        pipe = rp.RealDepthPipe.__new__(rp.RealDepthPipe)
        pipe.dev, pipe.rays, pipe._k_wh = torch.device("cpu"), None, (K_FULL, 1280, 720)
        pipe.aug, pipe.block, pipe.z_min = False, 1, z_m
        pipe.set_pose(T[None, :3, 3], hm.mat_to_quat_wxyz(torch.tensor(T[None, :3, :3])).numpy())
        ref = pipe._front(torch.from_numpy(d)[None])[0].numpy()
        got = rd.mm_quantize(P.floor_cut(d, T, K_FULL, z_m))
        cut = (got == 0).mean()
        assert 0.01 < cut < 0.99, (q, cut)                       # 컷이 실제로 걸리는 자세
        nd = int((np.abs(got - ref) > 1e-6).sum()); print(f"floor_cut q={q} 컷 {cut:.3f} 다른 px {nd}")
        assert nd <= 10, (q, nd)
    d16 = (rng.uniform(0.2, 2.5, (720, 1280)) * 1000).astype(np.uint16)
    P.floor_cut(d16.astype(np.float32) / np.float32(1000.0), T, K_FULL, z_m)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        P.floor_cut(d16.astype(np.float32) / np.float32(1000.0), T, K_FULL, z_m)
        ts.append(time.perf_counter() - t0)
    print(f"floor_cut 1280x720 p50 {np.median(ts) * 1e3:.1f} ms p95 {np.percentile(ts, 95) * 1e3:.1f} ms")


def test_look_target_matches_hdgp_place_head(tmp_path_factory, monkeypatch):
    """hdgp place_head.PlaceHead.target(20795489)을 잡음 0 으로 돌려 look_target 과 비교. 컵 점 c 와 HOLDER_TOP_Z 도 같은지 본다."""
    import types

    torch = pytest.importorskip("torch")
    ph = _ref(tmp_path_factory, "scripts/tools/student_adapters/place_head.py", "place_head_ref", PLACE_COMMIT)
    pg = _ref(tmp_path_factory, "source/openarm/openarm/agnostic/tasks/rh_place_r/place_geom.py", "place_geom_ref", PLACE_COMMIT)
    assert P.HOLDER_TOP_Z == pg.HOLDER_TOP_Z
    assert {k: v[1] for k, v in ph.CUP_C.items()} == G.C_DEFAULT
    pkg = types.ModuleType("openarm.agnostic.tasks.rh_place_r"); pkg.place_geom = pg
    for name in ["openarm", "openarm.agnostic", "openarm.agnostic.tasks"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "openarm.agnostic.tasks.rh_place_r", pkg)
    rng = np.random.default_rng(1)
    n = 4
    palm = torch.tensor(rng.uniform(-0.3, 0.5, (n, 3)), dtype=torch.float32)
    Rp = torch.linalg.qr(torch.tensor(rng.normal(size=(n, 3, 3)), dtype=torch.float32))[0]
    hol = torch.tensor(rng.uniform(-0.3, 0.3, (n, 3, 3)), dtype=torch.float32)
    tgt = torch.tensor([0, 2, 1, 2])
    env = types.SimpleNamespace(num_envs=n, device="cpu", _palm_pos=lambda: palm, _palm_R=lambda: Rp, _holder_pos=hol, _target=tgt)
    me = ph.PlaceHead.__new__(ph.PlaceHead)
    c = np.array(G.C_DEFAULT["source240"])
    me.cup_c, me.cup_off = torch.tensor(c, dtype=torch.float32).expand(n, 3), torch.zeros(n, 3)
    ref = me.target(env).double().numpy()
    for i in range(n):
        cup = palm[i].double().numpy() + Rp[i].double().numpy() @ c
        assert np.abs(P.look_target(cup, hol[i, tgt[i]].double().numpy()) - ref[i]).max() < 1e-6


def test_holder_proj_matches_hdgp(tmp_path_factory):
    torch = pytest.importorskip("torch")
    hp = _ref(tmp_path_factory, "scripts/tools/student_adapters/holder_proj.py", "holder_proj_ref")
    v, f = H.load_obj()
    hc = P.HeadCamPose()
    for q in [CAMSET_RAD, (0.0, 1.25), (0.6, 0.9)]:
        cam = hc.cam_pose_from_joint_state(*q)
        for pos, yaw in H.load_holder_poses().values():
            Th = H.holder_T(pos, yaw)
            small, zb = H.holder_mask(v, f, Th, cam, K_FULL)
            T = np.linalg.inv(cam) @ Th
            R, t = torch.tensor(T[None, :3, :3], dtype=torch.float32), torch.tensor(T[None, :3, 3], dtype=torch.float32)
            zr = hp.raster_depth(hp.to_cam(v.astype(np.float32), R, t), f, K_FULL, 1280, 720)[0].numpy()
            assert ((zb > 0) == np.isfinite(zr)).all()
            fin = np.isfinite(zr)
            assert np.abs(zb[fin] - zr[fin]).max() < 1e-5
            assert (small == (hp.block_reduce(torch.from_numpy(fin)[None], 8)[0].numpy() > 0.5)).all()
            assert small.shape == (90, 160) and small.sum() > 100


def test_holder_occlusion_rule():
    v, f = H.load_obj()
    hc = P.HeadCamPose()
    pos, yaw = H.load_holder_poses()["cup_holder_1"]
    cam = hc.cam_pose_from_joint_state(*CAMSET_RAD)
    full, zb = H.holder_mask(v, f, H.holder_T(pos, yaw), cam, K_FULL)
    same, _ = H.holder_mask(v, f, H.holder_T(pos, yaw), cam, K_FULL, depth_m=np.where(zb > 0, zb, 0.0))
    front, _ = H.holder_mask(v, f, H.holder_T(pos, yaw), cam, K_FULL, depth_m=np.where(zb > 0, zb - 0.05, 0.0))
    assert (same == full).all() and front.sum() == 0
    with pytest.raises(ValueError):
        H.holder_mask(v, f, H.holder_T(pos, yaw), cam, K_FULL, depth_m=np.zeros((90, 160)))


def test_cup_gate_rule():
    k = K_FULL
    p, R, c = np.array([0.0, 0.0, 0.5]), np.eye(3), np.zeros(3)
    e = G.expected_px(p, R, c, np.eye(4), k)
    m_near = np.zeros((720, 1280), bool); m_near[int(e[1]) - 5:int(e[1]) + 5, int(e[0]) + 30:int(e[0]) + 40] = True
    m_far = np.zeros((720, 1280), bool); m_far[:10, :10] = True
    # 클래스 2(orange) 중 conf 최고 하나만 본다: 0.2 는 conf 미달, 0.6 이 0.5 보다 먼저
    got, d, j = G.pick_cup([(0, 0.9, m_far), (2, 0.2, m_near), (2, 0.5, m_far), (2, 0.6, m_near)], "orange", p, R, np.eye(4), k, c=c)
    assert j == 3 and got is m_near and 30 < d < 40
    got, d, j = G.pick_cup([(2, 0.9, m_far), (2, 0.5, m_near)], "orange", p, R, np.eye(4), k, c=c)   # top1 이 멀면 버림(다음 후보로 넘어가지 않음)
    assert got is None and j == -1 and d > G.GATE_PX


@pytest.mark.skipif(not (SEG / "yolo11s_A" / "results" / "frame_cup_new.csv").exists(), reason="logs/cup_seg_1107/yolo11s_A 없음")
def test_cup_gate_reproduces_vision_rule():
    """s2r:vision 11.08 규칙(yolo11s_A analyze_new.py: keep25 클래스 인스턴스)을 같은 dets 로 재현하고, 80 px 문턱이 good 프레임의 맞는 컵을 버리지 않는지 본다."""
    pd = pytest.importorskip("pandas")
    A = SEG / "yolo11s_A"; n = 720 * 1280
    F = pd.read_csv(A / "results" / "frame_cup_new.csv"); F = F[F.cup.isin(["blue", "shaker"])].rename(columns={"iou_0.25": "iou25"})
    pp = np.load(SEG / "gate" / "palm_pose_cam.npz")
    tot = same = lost = 0
    for b, fb in F.groupby("bag"):
        dz = dict(np.load(A / "dets" / f"{b}_dets.npz"))
        for r in fb.itertuples():
            js = [j for j in np.flatnonzero(dz["frame"] == r.i) if dz["keep25"][j]]
            dets = [(int(dz["cls"][j]), float(dz["conf"][j]), np.unpackbits(dz["mask"][j])[:n].astype(bool).reshape(720, 1280)) for j in js]
            t = "orange" if r.cup == "shaker" else "blue"
            p, R = pp[f"{b}_{r.cup}_p"][r.i], pp[f"{b}_{r.cup}_R"][r.i]
            _, _, q = G.pick_cup(dets, t, p, R, np.eye(4), pp["K"], gate_px=np.inf)
            ref = next((x for x, j in enumerate(js) if dz["cls"][j] == G.CLASSES.index(t)), -1)
            tot += 1; same += int(q == ref)
            if r.good and r.stage != "place" and r.iou25 >= 0.3:
                lost += int(G.pick_cup(dets, t, p, R, np.eye(4), pp["K"])[2] == -1)
    assert tot > 2000 and same == tot, f"{same}/{tot}"
    assert lost == 0, f"80 px 문턱이 good 프레임 맞는 컵 {lost} 개를 버림"
