"""11.18 shaker 섞기 검사기 plan_mix.py — 합성 npz(붓기 시작 자세 둘레 작은 돌리기) · 가짜 실측 q 로 거부 · 통과를 본다.
FK · 팔 사이 거리는 실제 hdgp ready_carry(고정 커밋 사본)로 잰다 — 핀이 맞지 않는 호스트에서는 건너뛴다."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "deploy" / "policy_control" / "tools"
ASSET = REPO / "deploy" / "policy_control" / "config" / "mix" / "shaker_swirl"
POUR = yaml.safe_load((REPO / "deploy/policy_control/config/homes/rh56f1_pour_start.yaml").read_text(encoding="utf-8"))


def _tool():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("plan_mix", TOOLS / "plan_mix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _tool()


def _pins_hold() -> bool:
    try:
        M.H.verify(M.H.hdgp_dir(), M.H.load_policy(ASSET)["hdgp"]["md5"])
        return True
    except (M.H.PinError, OSError):
        return False


needs_hdgp = pytest.mark.skipif(not _pins_hold(), reason="hdgp 핀이 이 호스트에서 맞지 않는다")


def _asset(tmp_path: Path, md5: str | None = None, amp: float = 0.2, freq: float = 0.5, j7: float | None = None) -> Path:
    """자산 사본 + 합성 ref npz(왼팔 j7 ±amp rad 사인, 2 s, 시작 · 끝 = 붓기 시작 자세 — j7 을 주면 그 값 둘레)."""
    a = tmp_path / "asset"
    (a / "ref").mkdir(parents=True)
    cfg = yaml.safe_load((ASSET / "policy.yaml").read_text(encoding="utf-8"))
    t = np.arange(121) / 60.0
    q = np.repeat(np.asarray([POUR["left"]], float), len(t), 0)
    if j7 is not None:
        q[:, 6] = j7
    q[:, 6] += amp * np.sin(np.pi * t) ** 2 * np.sin(2 * np.pi * freq * t)
    np.savez_compressed(a / "ref" / cfg["ref_npz"], arm_target=q, joint_names=np.array([f"l_aj_{i}" for i in range(1, 8)]),
                        meta_step_dt=np.array(1 / 60.0))
    cfg["ref"] = {cfg["ref_npz"]: md5 or M.H.md5(a / "ref" / cfg["ref_npz"])}
    (a / "policy.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return a


def _run(tmp_path: Path, asset: Path, q: dict) -> int:
    qj = tmp_path / "q.json"
    qj.write_text(json.dumps(q), encoding="utf-8")         # NaN 은 json 의 NaN 리터럴로 들어간다(파이썬 json 이 읽는다)
    return M.main(["--asset", str(asset), "--q-json", str(qj), "--out", str(tmp_path / "out")])


def _fail(tmp_path: Path) -> list:
    return json.loads((tmp_path / "out" / "check.json").read_text(encoding="utf-8"))["fail"]


def test_ref_md5_mismatch_refuses_with_pin_code(tmp_path):
    assert _run(tmp_path, _asset(tmp_path, md5="0" * 32), POUR) == 3
    assert not (tmp_path / "out" / "mix_cmd.npz").exists()


@pytest.mark.skipif(any(v is not None for v in (M.H.load_policy(ASSET).get("ref") or {}).values()),
                    reason="2단계 뒤(ref md5 채움) — 이 시험은 채우기 전 거부만 본다")
def test_unfilled_repo_asset_refuses_until_step_two(tmp_path):
    assert _run(tmp_path, ASSET, POUR) == 3                         # ref md5 null(2단계 전)


def test_q_json_missing_or_short_is_arg_error(tmp_path):
    a = _asset(tmp_path)
    assert M.main(["--asset", str(a), "--q-json", str(tmp_path / "없음.json"), "--out", str(tmp_path / "out")]) == 1
    assert _run(tmp_path, a, {"right": POUR["right"]}) == 1                            # left 키 없음
    assert _run(tmp_path, a, {"right": POUR["right"], "left": POUR["left"][:6]}) == 1  # 6 관절


def test_bad_rate_scale_refuses_with_pin_code(tmp_path):
    a = _asset(tmp_path)
    cfg = yaml.safe_load((a / "policy.yaml").read_text(encoding="utf-8"))
    for k, v in (("rate_scale", 0.0), ("rate_scale", 1.5), ("rate_scale", "0.6"), ("start_gap_rad", 0.5)):
        (a / "policy.yaml").write_text(yaml.safe_dump({**cfg, k: v}, allow_unicode=True), encoding="utf-8")
        assert _run(tmp_path, a, POUR) == 3, (k, v)


def test_broken_npz_refuses_with_pin_code(tmp_path):
    a = _asset(tmp_path)
    cfg = yaml.safe_load((a / "policy.yaml").read_text(encoding="utf-8"))
    (a / "ref" / cfg["ref_npz"]).write_bytes(b"not a zip")
    cfg["ref"] = {cfg["ref_npz"]: M.H.md5(a / "ref" / cfg["ref_npz"])}             # md5 는 맞고 내용만 깨진 npz
    (a / "policy.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    assert _run(tmp_path, a, POUR) == 3


@needs_hdgp
def test_good_start_passes_and_copies_ref(tmp_path):
    a = _asset(tmp_path)
    assert _run(tmp_path, a, POUR) == 0
    chk = json.loads((tmp_path / "out" / "check.json").read_text(encoding="utf-8"))
    assert chk["ok"] and chk["arms_clear"]["ok"] and chk["fail"] == []
    assert chk["rate_scale"] == yaml.safe_load((a / "policy.yaml").read_text(encoding="utf-8"))["rate_scale"]
    assert chk["checks"]["end_gap_rad"] < 1e-6 and chk["checks"]["table_clear_m"] >= M.TABLE_MARGIN_M
    ref = a / "ref" / yaml.safe_load((a / "policy.yaml").read_text(encoding="utf-8"))["ref_npz"]
    assert (tmp_path / "out" / "mix_cmd.npz").read_bytes() == ref.read_bytes()


@needs_hdgp
def test_start_gap_over_limit_refuses(tmp_path):
    q = {"right": POUR["right"], "left": list(np.asarray(POUR["left"]) + [0, 0, 0, 0.06, 0, 0, 0])}
    assert _run(tmp_path, _asset(tmp_path), q) == 2
    chk = json.loads((tmp_path / "out" / "check.json").read_text(encoding="utf-8"))
    assert chk["fail"] == ["start_gap"] and not (tmp_path / "out" / "mix_cmd.npz").exists()


@needs_hdgp
def test_right_arm_too_close_refuses(tmp_path, monkeypatch):
    a = _asset(tmp_path)
    assert _run(tmp_path, a, POUR) == 0
    d = json.loads((tmp_path / "out" / "check.json").read_text(encoding="utf-8"))["arms_clear"]["min_arm_point_m"]
    shutil.rmtree(tmp_path / "out")
    monkeypatch.setattr(sys.modules["plan_carry"], "ARM_POINT_GAP_M", d + 0.01)    # 같은 자세에서 문턱만 넘긴다 = 오른팔이 1 cm 더 가까운 셈
    assert _run(tmp_path, a, POUR) == 2
    chk = json.loads((tmp_path / "out" / "check.json").read_text(encoding="utf-8"))
    assert chk["fail"] == ["arms_clear"] and not (tmp_path / "out" / "mix_cmd.npz").exists()


@needs_hdgp
def test_replan_fail_in_same_folder_removes_old_npz(tmp_path):
    a = _asset(tmp_path)
    assert _run(tmp_path, a, POUR) == 0 and (tmp_path / "out" / "mix_cmd.npz").exists()
    q = {"right": POUR["right"], "left": list(np.asarray(POUR["left"]) + [0, 0, 0, 0.06, 0, 0, 0])}
    assert _run(tmp_path, a, q) == 2 and not (tmp_path / "out" / "mix_cmd.npz").exists()
    assert _run(tmp_path, a, POUR) == 0 and (tmp_path / "out" / "check.json").exists()
    assert _run(tmp_path, a, {"right": POUR["right"]}) == 1 and not (tmp_path / "out" / "check.json").exists()   # 옛 통과 판정이 남지 않는다


@needs_hdgp
def test_nan_q_or_npz_refuses(tmp_path):
    a = _asset(tmp_path)
    assert _run(tmp_path, a, POUR) == 0
    q = {"right": POUR["right"], "left": [float("nan")] + list(POUR["left"][1:])}
    assert _run(tmp_path, a, q) == 1 and not (tmp_path / "out" / "mix_cmd.npz").exists()
    shutil.rmtree(tmp_path / "asset")
    assert _run(tmp_path, _asset(tmp_path, amp=float("nan")), POUR) == 3


@needs_hdgp
def test_fast_npz_refuses_vmax(tmp_path):
    assert _run(tmp_path, _asset(tmp_path, amp=0.3, freq=1.5), POUR) == 2          # 약 2.8 rad/s > VMAX_RAD_S
    assert "vmax" in _fail(tmp_path) and not (tmp_path / "out" / "mix_cmd.npz").exists()


@needs_hdgp
def test_npz_near_joint_limit_refuses_limit_margin(tmp_path):
    RC = M.H.load_ready_carry(M.H.hdgp_dir())
    hi = float(RC.chain.urdf_limits(RC.chain.default_urdf(), M.ARMS["left"])[6, 1])
    j7 = hi - 0.2 - 0.01                                                           # 사인 꼭대기가 한계 1 cm 앞(여유 0.01 < 0.05)
    q = {"right": POUR["right"], "left": list(POUR["left"][:6]) + [j7]}
    assert _run(tmp_path, _asset(tmp_path, j7=j7), q) == 2
    assert "limit_margin" in _fail(tmp_path) and not (tmp_path / "out" / "mix_cmd.npz").exists()
