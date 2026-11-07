"""11.08 place_head_look.look_at — hdgp 20795489 head.py HeadModel.look_at 과 같은 시작 q · 대상 점에서 대조."""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import time

import numpy as np
import pytest

from policy_control import hdgp_pin
from policy_control import place_head_look as L
from policy_control import place_head_pose as P

PLACE_COMMIT = "2079548988b6277de9e40dc511350a946166abaf"


def _hdgp_head(tmp_path_factory):
    tree = hdgp_pin.hdgp_tree()
    if not (tree / ".git").exists():
        pytest.skip(f"hdgp 저장소 없음: {tree}")
    r = subprocess.run(["git", "-C", str(tree), "show", f"{PLACE_COMMIT}:source/openarm/openarm/distillation/student/head.py"],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"hdgp head.py 없음: {r.stderr.strip()}")
    p = tmp_path_factory.mktemp("look_ref") / "head_look_ref.py"
    p.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location("head_look_ref", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["head_look_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cases(hc):
    """(시작 q, 대상 점) 24 개: 탁자 위 무작위 16 + 한계 근처 · 한계 밖 8."""
    rng = np.random.default_rng(3)
    T0 = hc.cam_pose_from_encoder(*hc.q0_enc_deg)
    out = []
    for _ in range(16):
        q = np.array([rng.uniform(-40, 25), rng.uniform(5, 80)])
        out.append((q, np.array([rng.uniform(0.15, 0.55), rng.uniform(-0.35, 0.3), rng.uniform(0.0, 0.25)])))
    out += [
        (np.array(hc.q0_enc_deg), np.array([0.40, -0.60, 0.05])),     # 오른쪽 멀리: pan 상한/하한 쪽
        (np.array(hc.q0_enc_deg), np.array([0.40, 0.70, 0.05])),
        (np.array([29.9, 60.0]), np.array([0.30, -0.80, 0.10])),       # 이미 한계에 붙은 채 바깥으로
        (np.array([-44.9, 60.0]), np.array([0.30, 0.80, 0.10])),
        (np.array([0.0, 81.9]), np.array([0.05, 0.0, -0.30])),          # tilt 상한 밖(아주 아래)
        (np.array([0.0, -1.9]), np.array([1.50, 0.0, 1.40])),           # tilt 하한 밖(위)
        (np.array([-30.0, 80.0]), T0[:3, 3] + T0[:3, 2] * 0.6),         # 시작 자세 광학축 위 점
        (np.array([10.0, 30.0]), np.array([0.20, -0.45, 0.02])),
    ]
    return out


def test_look_at_matches_hdgp(tmp_path_factory):
    torch = pytest.importorskip("torch")
    head = _hdgp_head(tmp_path_factory)
    import yaml

    hc = P.HeadCamPose()
    d = yaml.safe_load(P.EXTRINSICS.read_text())["camera"]
    cfg = head.HeadCfg(cam_pos=tuple(d["position"]), cam_quat_wxyz=tuple(d["orientation_wxyz"]), calib_enc_deg=hc.q0_enc_deg,
                       home_enc_deg=hc.q0_enc_deg, sag=True, sag_table=str(P.SAG_TABLE), randomize=False, yaw_deg=0.0)
    assert (tuple(cfg.pan_lim), tuple(cfg.tilt_lim)) == (L.PAN_LIM, L.TILT_LIM)
    hm = head.HeadModel(1, cfg=cfg)
    worst, well, at_lim, ts = 0.0, 0.0, 0, []
    for q, tgt in _cases(hc):
        hm.q = torch.tensor([q], dtype=torch.float32)
        ref = hm.look_at(torch.tensor([tgt], dtype=torch.float32))[0].double().numpy()
        t0 = time.perf_counter()
        got = L.look_at(hc, tgt, q)
        ts.append(time.perf_counter() - t0)
        dq = np.abs(got - ref).max()
        # 한쪽이 한계에 붙고 남은 축이 거의 안 듣는 자리(점이 목 바로 아래)는 float32(hdgp) · float64 차로 각이 갈린다.
        # 그때는 각 대신 화면 잔차가 같은지 본다.
        dr = abs(np.abs(L._proj(hc, got, tgt)).max() - np.abs(L._proj(hc, ref, tgt)).max())
        assert dq < 1e-2 or dr < 1e-4, (q, tgt, got, ref, dr)
        worst = max(worst, dq)
        if dq < 1e-2:
            well = max(well, dq)
        lim = np.isclose(got, [L.PAN_LIM[0], L.TILT_LIM[0]]) | np.isclose(got, [L.PAN_LIM[1], L.TILT_LIM[1]])
        at_lim += int(lim.any())
        if not lim.any():                                               # 한계 안이면 광학축에 올라야 한다
            assert np.abs(L._proj(hc, got, tgt)).max() < 1e-4, (q, tgt, got)
    print(f"look_at {len(ts)} 건(한계에 닿은 것 {at_lim}) 최대 차 {worst:.2e} deg(잘 정해진 곳 {well:.2e}) · 한 번 p50 {np.median(ts) * 1e3:.1f} "
          f"ms p95 {np.percentile(ts, 95) * 1e3:.1f} ms")
    assert at_lim >= 4


def test_look_at_bad_input():
    hc = P.HeadCamPose()
    with pytest.raises(ValueError):
        L.look_at(hc, [0.3, 0.0, math.nan], hc.q0_enc_deg)
    with pytest.raises(ValueError):
        L.look_at(hc, [0.3, 0.0], hc.q0_enc_deg)
