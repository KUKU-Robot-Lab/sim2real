"""시리얼 포트 단독 소유 잠금."""
import multiprocessing as mp

import pytest

import port_lock


def _try(port, queue):
    try:
        port_lock.release(port_lock.acquire(port))
        queue.put("ok")
    except port_lock.PortBusy as exc:
        queue.put(f"busy:{exc}")


def test_second_opener_is_refused_while_first_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(port_lock, "LOCK_DIR", tmp_path)
    fd = port_lock.acquire("/dev/ttyUSB9")
    assert port_lock.is_held("/dev/ttyUSB9")
    queue = mp.get_context("fork").Queue()
    proc = mp.get_context("fork").Process(target=_try, args=("/dev/ttyUSB9", queue))
    proc.start()
    proc.join(5)
    assert queue.get(timeout=2).startswith("busy:")
    port_lock.release(fd)
    assert not port_lock.is_held("/dev/ttyUSB9")
    port_lock.release(port_lock.acquire("/dev/ttyUSB9"))


def test_other_ports_are_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(port_lock, "LOCK_DIR", tmp_path)
    a, b = port_lock.acquire("/dev/ttyUSB1"), port_lock.acquire("/dev/ttyUSB2")
    port_lock.release(a)
    port_lock.release(b)
