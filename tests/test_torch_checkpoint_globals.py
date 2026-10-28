"""torch_checkpoint_globals — torch ≥ 2.6(weights_only 기본)에서 우리 체크포인트를 허용 목록만 잠깐 열어 읽는다.

10.28 5090 첫 FP++: sm120 이미지(torch 2.7.1 · ultralytics 8.0.120)의 묶음 컨테이너가 YOLO 가중치를 못 읽고 죽었다.
weights_only 보호를 끄는 대신(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD — 권한 판정이 막았다), 그 체크포인트가 요구하는 전역 중
허용 접두어 안의 것만 with 블록 동안 허용한다.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch.serialization, "safe_globals"):
    pytest.skip("torch < 2.5 — weights_only 허용 목록 API 없음", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import torch_checkpoint_globals as TG  # noqa: E402


class Gadget:
    """허용 목록 밖 전역의 예 — 체크포인트에 들어 있으면 열지 않아야 한다."""


def _module_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "legacy.pt"
    torch.save({"model": torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.SiLU())}, p)
    return p


def test_split_keeps_only_the_allowed_prefixes_and_names():
    names = ["ultralytics.nn.tasks.SegmentationModel", "torch.nn.modules.conv.Conv2d", "builtins.getattr",
             "os.system", "ultralytics.utils.downloads.attempt_download_asset"]
    ok, refused = TG.split_allowed(names, TG.YOLO_PREFIXES, TG.YOLO_NAMES)
    assert ok == ["builtins.getattr", "torch.nn.modules.conv.Conv2d", "ultralytics.nn.tasks.SegmentationModel"]
    assert refused == ["os.system", "ultralytics.utils.downloads.attempt_download_asset"]


def test_a_legacy_module_checkpoint_loads_only_inside_the_block(tmp_path):
    p = _module_ckpt(tmp_path)
    with pytest.raises(pickle.UnpicklingError):                 # torch ≥ 2.6 기본 weights_only=True 가 막는다
        torch.load(p)
    with TG.allow_checkpoint_globals(p, prefixes=("torch.nn.modules.",), exact=()) as allowed:
        model = torch.load(p)["model"]
    assert isinstance(model, torch.nn.Sequential)
    assert "torch.nn.modules.conv.Conv2d" in allowed
    with pytest.raises(pickle.UnpicklingError):                 # 블록 밖에서는 다시 막힌다(전역 등록이 남지 않는다)
        torch.load(p)


def test_a_checkpoint_needing_a_global_outside_the_list_is_refused(tmp_path):
    p = tmp_path / "gadget.pt"
    torch.save({"g": Gadget()}, p)
    with pytest.raises(TG.CheckpointGlobalsRefused, match="Gadget"):
        with TG.allow_checkpoint_globals(p):
            pass
    with pytest.raises(pickle.UnpicklingError):
        torch.load(p)


def test_a_missing_file_opens_nothing(tmp_path):
    with TG.allow_checkpoint_globals(tmp_path / "yolov8m-seg.pt") as allowed:   # 이름만 주면 ultralytics 가 받는다 — 막지 않는다
        assert allowed == []


def test_the_snapshot_node_builds_the_yolo_detector_inside_the_allowlist():
    """묶음 FP++ 노드(fpp_snapshot_node.py, 컨테이너 안)가 YOLO 를 만드는 줄이 허용 목록 블록 안에 있다."""
    src = (REPO / "scripts/nodes/fpp_snapshot_node.py").read_text(encoding="utf-8")
    block = src.index("allow_checkpoint_globals(")
    assert block < src.index("YoloCupDetector(", block) < block + 400
