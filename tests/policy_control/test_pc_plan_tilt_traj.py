"""11.18 plan_tilt_traj.py IK 장치 고르기(--device auto|cpu|cuda) · auto 의 cuda 실패 → cpu 재계획 · 종료 코드. 실제 IK 는 돌리지 않는다 —
hdgp 모듈 자리에 가짜를 넣고 torch.cuda.is_available 을 바꿔 끼워 CUDA 유무와 무관하게 돈다(GPU 연산 없음)."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "deploy" / "policy_control" / "tools" / "plan_tilt_traj.py"
ASSET = REPO / "deploy" / "policy_control" / "config" / "tilt" / "traj_v10"


def _tool():
    spec = importlib.util.spec_from_file_location("plan_tilt_traj", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_auto_picks_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _tool().pick_device("auto") == "cpu"


def test_device_cuda_without_cuda_is_arg_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _tool().main(["--cup", "240", "--out", str(tmp_path / "o"), "--device", "cuda"]) == 1
    assert "CUDA" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_torch_import_failure_is_rc1(tmp_path, monkeypatch, capsys):
    T = _tool()
    monkeypatch.setitem(sys.modules, "torch", None)                   # import torch → ImportError
    assert T.main(["--cup", "240", "--out", str(tmp_path / "o")]) == 1
    assert "torch" in capsys.readouterr().err and not (tmp_path / "o").exists()


class _CudaLike:
    """cuda 텐서 흉내: .numpy() 는 거부하고 .cpu() 를 거쳐야 numpy 가 된다."""
    def __init__(self, t: torch.Tensor):
        self.t = t

    def __getitem__(self, k):
        return _CudaLike(self.t[k])

    def numpy(self):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    def cpu(self):
        return self.t


def _fakes(L: int = 12, fail_on_cuda: bool = False):
    @dataclasses.dataclass
    class TiltGeo:
        dt: float
        src_mouth_z: float
        rcv_mouth_z: float

    @dataclasses.dataclass
    class ClearCfg:
        pass

    devices, setups = [], []

    def from_urdf(path, tip, arms, device, dtype):
        devices.append(device)
        return types.SimpleNamespace(device=device, lo=torch.full((7,), -3.0), hi=torch.full((7,), 3.0),
                                     fk=lambda q: torch.eye(4).expand(len(q), 4, 4))

    def tilt_setup(**kw):
        setups.append(kw)
        return kw

    tb = types.SimpleNamespace(length=torch.tensor([L]), q_r=_CudaLike(torch.zeros(1, L + 3, 7, dtype=torch.float64)),
                               q_l=_CudaLike(torch.zeros(1, L + 3, 7, dtype=torch.float64)),
                               info={"ik_pos_mm": torch.tensor([1.0]), "clr_min": torch.tensor([0.05]), "table_clear": torch.tensor([0.05])},
                               knots=types.SimpleNamespace(edges=_CudaLike(torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.3, 0.3, 0.4]]))))

    def build_batch(setup, *a):
        if fail_on_cuda and setup["chain_r"].device == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB\n(자세한 줄)")
        return tb

    TT = types.SimpleNamespace(TiltGeo=TiltGeo, TiltSetup=tilt_setup, params_tensor=lambda p: p, build_batch=build_batch)
    CH = types.SimpleNamespace(Chain=types.SimpleNamespace(from_urdf=from_urdf))
    return TT, CH, types.SimpleNamespace(ClearCfg=ClearCfg), devices, setups


def _patch(T, monkeypatch, fail_on_cuda=False):
    TT, CH, CL, devices, setups = _fakes(fail_on_cuda=fail_on_cuda)
    monkeypatch.setattr(T, "load_modules", lambda root: (TT, CH))
    monkeypatch.setattr(T, "table_z", lambda cfg: 0.0)
    monkeypatch.setitem(sys.modules, f"{T.PKG}.tilt_traj.clearance", CL)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)    # cuda 길을 타게만 — 실제 GPU 연산 없음
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return devices, setups


def _holders(tmp_path):
    h = tmp_path / "h.yaml"
    h.write_text("holders:\n  h0: {position: [0.4, -0.1, 0.0]}\n", encoding="utf-8")
    return h


def test_cuda_result_comes_down_to_numpy(tmp_path, monkeypatch):
    T = _tool()
    devices, setups = _patch(T, monkeypatch)
    r = T.plan({"pre_s": 0.5}, ASSET, "240", {"right": [0.0] * 7, "left": [0.0] * 7}, _holders(tmp_path), "", Path("."), "auto")
    assert devices == ["cpu", "cpu", "cuda", "cuda"]                  # 후처리 chain 은 CPU, IK chain 만 cuda
    assert len(setups) == 1 and setups[0]["fast"] is True and setups[0]["cuda_graph"] is True
    assert isinstance(r["knots"], np.ndarray) and r["knots"].shape == (12, 14)
    assert r["run"]["device"] == "cuda" and r["run"]["cuda_graph"] is True and r["run"]["fallback"] is None
    assert r["run"]["ik_s"] >= 0 and r["run"]["plan_s"] >= r["run"]["ik_s"]


def test_auto_cuda_failure_replans_on_cpu(tmp_path, monkeypatch, capsys):
    T = _tool()
    devices, setups = _patch(T, monkeypatch, fail_on_cuda=True)
    for k, v in {"load_policy": lambda a: {"hdgp": {"md5": {}}, "pre_s": 0.5}, "hdgp_tree": lambda: Path("."),
                 "verify": lambda *a: None, "verify_refs": lambda *a: None}.items():
        monkeypatch.setattr(T.H, k, v)
    eye = np.eye(4).tolist()
    (tmp_path / "q.json").write_text(json.dumps({"right": [0.0] * 7, "left": [0.0] * 7}), encoding="utf-8")
    (tmp_path / "tin.json").write_text(json.dumps({"T_in_r": eye, "T_in_l": eye}), encoding="utf-8")
    rc = T.main(["--cup", "240", "--holders", str(_holders(tmp_path)), "--q-json", str(tmp_path / "q.json"),
                 "--tin-json", str(tmp_path / "tin.json"), "--out", str(tmp_path / "o")])
    assert rc == 0
    assert [s["chain_r"].device for s in setups] == ["cuda", "cpu"]
    run = json.loads((tmp_path / "o" / "check.json").read_text(encoding="utf-8"))["run"]
    assert run["device"] == "cpu" and run["cuda_graph"] is False
    assert run["fallback"] == "CUDA out of memory. Tried to allocate 2.00 GiB"
    assert "cpu 로 다시" in capsys.readouterr().err


def test_explicit_cuda_does_not_fall_back(tmp_path, monkeypatch):
    T = _tool()
    _patch(T, monkeypatch, fail_on_cuda=True)
    with pytest.raises(torch.OutOfMemoryError):
        T.plan({"pre_s": 0.5}, ASSET, "240", {"right": [0.0] * 7, "left": [0.0] * 7}, _holders(tmp_path), "", Path("."), "cuda")
