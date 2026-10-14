"""시리얼 포트 단독 소유 잠금 — 한 포트를 두 프로세스가 열지 못하게 한다.

목 버스(/dev/ttyUSB0)를 읽는 프로세스와 쓰는 프로세스가 같이 열면 "Incorrect status packet" 이 난다(2026-10-14 실측).
포트를 여는 어댑터(`DynamixelPositionController`)가 열 때 이 잠금을 먼저 잡는다. 이미 누가 잡았으면 `PortBusy` 로
바로 실패하므로, 목은 `head_joint_publisher.py`(단독 소유자)만 열 수 있고 다른 도구는 `head_move.py` 로 요청한다.
커널 flock 이라 프로세스가 죽으면 저절로 풀린다(낡은 잠금이 남지 않는다).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

LOCK_DIR = Path("/tmp")


class PortBusy(RuntimeError):
    """다른 프로세스가 이 포트를 이미 단독으로 열고 있다."""


def lock_path(port_name: str) -> Path:
    return LOCK_DIR / f"dxl_port_{Path(port_name).name}.lock"


def acquire(port_name: str) -> int:
    """포트 잠금을 잡고 fd 를 돌려준다. 이미 잡혀 있으면 `PortBusy`(소유 PID 포함)."""
    path = lock_path(port_name)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner = os.pread(fd, 32, 0).decode(errors="ignore").strip() or "?"
        os.close(fd)
        raise PortBusy(f"{port_name} 은 PID {owner} 가 단독으로 열고 있다 — 목은 head_joint_publisher 만 연다. "
                       "이동은 scripts/head_move.py 로 요청할 것") from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, str(os.getpid()).encode(), 0)
    return fd


def release(fd: int | None) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def is_held(port_name: str) -> bool:
    """누가 잡고 있는지 확인만 한다(잡지 않는다)."""
    path = lock_path(port_name)
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
