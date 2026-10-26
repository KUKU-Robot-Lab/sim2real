"""10.26 tilt 1차(tilt_traj 궤적, 잔차 0) 계획기 plan_tilt_traj.py — torch 없이 도는 부분: 60 Hz 재표본(rh_tilt env._teach_at 과
같은 식), 홀더 yaml 순서, 자산 핀(ref md5 · hdgp md5 불일치면 종료 코드 3). 궤적 자체의 sim 동등성은 server 에서 확인했다
(ref_q_r/ref_q_l 와 차이 0.0 rad, 10.26)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "deploy" / "policy_control" / "tools" / "plan_tilt_traj.py"
ASSET = REPO / "deploy" / "policy_control" / "config" / "tilt" / "traj_v10"


def _tool():
    spec = importlib.util.spec_from_file_location("plan_tilt_traj", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_resample_matches_env_teach_at():
    T = _tool()
    knots = np.cumsum(np.ones((10, 2)) * 0.1, axis=0)       # knot k = 0.1 (k+1)
    q0 = np.array([-1.0, 2.0])
    out = T.resample(q0, knots, dt_knot=1 / 30, pre_s=0.5, rate=60.0)
    assert len(out) == int(np.floor((0.5 + 9 / 30) * 60 + 1e-9)) + 1
    assert np.allclose(out[0], q0)                           # s = 0 → 실측
    assert np.allclose(out[30], knots[0])                    # s = pre → 첫 knot
    assert np.allclose(out[31], 0.5 * (knots[0] + knots[1]))  # knot 사이 선형
    assert np.allclose(out[-1], knots[-1])
    mid = out[15]                                            # pre 절반 = min-jerk 0.5
    assert np.allclose(mid, q0 + 0.5 * (knots[0] - q0))


def test_load_holders_sorted(tmp_path):
    T = _tool()
    p = tmp_path / "h.yaml"
    p.write_text("holders:\n  cup_holder_1: {position: [0.38, -0.08, 0.235]}\n  cup_holder_0: {position: [0.38, 0.16, 0.235]}\n")
    H = T.load_holders(p)
    assert H.shape == (2, 3) and H[0, 1] == pytest.approx(0.16)


def test_asset_refs_pinned():
    T = _tool()
    cfg = T.H.load_policy(ASSET)
    T.H.verify_refs(ASSET, cfg["ref"])
    meta = json.loads((ASSET / "ref" / "asset_meta.json").read_text(encoding="utf-8"))
    assert set(cfg["hdgp"]["md5"].items()) <= set(meta["hdgp_md5"].items())
    assert meta["cups"]["240"]["knots"] == 638 and meta["cups"]["200"]["knots"] == 598
    for cup in ("240", "200"):
        p = json.loads((ASSET / "ref" / f"params_{cup}.json").read_text(encoding="utf-8"))
        assert p["from_md5"] == meta["cups"][cup]["params_md5"]


def test_hdgp_mismatch_refuses(tmp_path, monkeypatch, capsys):
    T = _tool()
    monkeypatch.setenv("HDGP_DIR", str(tmp_path))            # 빈 폴더 = 핀 파일 없음
    rc = T.main(["--cup", "240", "--q-json", str(tmp_path / "none.json"), "--out", str(tmp_path / "o")])
    assert rc == 3
    assert "hdgp" in capsys.readouterr().out
