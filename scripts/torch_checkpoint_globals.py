"""torch ≥ 2.6 의 torch.load 기본(weights_only=True) 아래에서 우리 체크포인트를 읽는다 — 허용 목록만 잠깐 연다.

★10.28 5090 첫 FP++: sm120 이미지(torch 2.7.1 · ultralytics 8.0.120)의 묶음 컨테이너가 YOLO 가중치를 못 읽고 시작하자마자
죽었다. 옛 ultralytics 는 torch.load 에 weights_only 를 주지 않아서 torch 2.6 부터 모듈 클래스가 든 체크포인트가 거부된다.
weights_only 보호는 끄지 않는다(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD 는 프로세스 전체를 연다). 대신 그 체크포인트가 실제로
요구하는 전역 중 허용 접두어 안의 것만 with 블록 동안 허용한다 — 밖의 것이 하나라도 있으면 열지 않는다.
yolov8m-seg.pt 가 요구하는 것(이미지 실측 19개): torch.nn.modules.* 8 · ultralytics.nn.* 10 · builtins.getattr.
"""
from __future__ import annotations

import contextlib
import importlib
from collections.abc import Generator, Iterable
from pathlib import Path

#: YOLO(ultralytics 8.0.x) 체크포인트 — 신경망 모듈 클래스와 getattr 만.
YOLO_PREFIXES = ("ultralytics.nn.", "torch.nn.modules.")
YOLO_NAMES = ("builtins.getattr",)


class CheckpointGlobalsRefused(RuntimeError):
    """체크포인트가 허용 목록 밖의 전역을 요구한다 — 열지 않는다."""


def split_allowed(names: Iterable[str], prefixes: Iterable[str], exact: Iterable[str]) -> tuple[list[str], list[str]]:
    """(허용, 거부) — 이름순."""
    pre, ex = tuple(prefixes), set(exact)
    ok, refused = [], []
    for n in sorted(set(names)):
        (ok if n in ex or n.startswith(pre) else refused).append(n)
    return ok, refused


def _resolve(name: str):
    module, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module), attr)


@contextlib.contextmanager
def allow_checkpoint_globals(path: str | Path, prefixes: Iterable[str] = YOLO_PREFIXES,
                             exact: Iterable[str] = YOLO_NAMES) -> Generator[list[str], None, None]:
    """이 체크포인트가 요구하는 전역(허용 목록 안의 것만)을 with 블록 동안 torch.load 에 허용하고, 허용한 이름을 낸다.

    파일이 없거나(이름만 주면 ultralytics 가 받는다) torch 에 이 API 가 없으면(< 2.5 — 그때 기본은 weights_only=False)
    아무것도 열지 않는다."""
    import torch

    ser = torch.serialization
    if not Path(path).is_file() or not (hasattr(ser, "get_unsafe_globals_in_checkpoint") and hasattr(ser, "safe_globals")):
        yield []
        return
    ok, refused = split_allowed(ser.get_unsafe_globals_in_checkpoint(str(path)), prefixes, exact)
    if refused:
        raise CheckpointGlobalsRefused(f"{path}: 허용 목록 밖의 전역 {refused}")
    with ser.safe_globals([_resolve(n) for n in ok]):
        yield ok
