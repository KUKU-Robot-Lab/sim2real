"""Fixtures for the policy_control package tests.

``sim2real/policy_control`` is put on ``sys.path`` so ``import policy_control``
works without a colcon install; ``policy_control._paths`` then exposes the
sibling trees. The ``ros`` fixture creates a **private rclpy Context on an
isolated domain** (``PC_TEST_DOMAIN``, default 99) so a test can never talk to a
real robot's DDS graph on the same host.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
PKG_DIR = SIM2REAL / "policy_control"
FIXTURES = SIM2REAL / "tests" / "fixtures" / "policy_control"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import policy_control._paths  # noqa: E402,F401  (side effect: sibling trees on sys.path)


@pytest.fixture(autouse=True)
def _cuda_off_without_gpu_marker(request, monkeypatch):
    """``-m "not gpu"`` 가 실제로 GPU 를 건드리지 못하게 한다.

    GPU 를 마커가 아니라 런타임 가드(``torch.cuda.is_available()``)로만 막던 테스트가 있어
    ``-m "not gpu"`` 로 돌려도 학습 중인 GPU 에 fabrics_sim 이 올라간 적이 있다(09.21,
    ``test_pour_fabric.py`` parity). 마커가 없는 테스트에서는 가용 장치를 숨겨
    그 가드들이 원래대로 skip 하게 만든다 — torch 가 아직 import 되지 않았어도
    ``CUDA_VISIBLE_DEVICES`` 가 warp/fabrics_sim 까지 함께 막는다.
    """
    if request.node.get_closest_marker("gpu"):
        return
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def sim2real_dir() -> Path:
    return SIM2REAL


@pytest.fixture
def ros():
    """Private rclpy context on the test domain; torn down after the test."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context

    domain = int(os.environ.get("PC_TEST_DOMAIN", "99"))
    context = Context()
    rclpy.init(context=context, domain_id=domain)
    try:
        yield context
    finally:
        rclpy.shutdown(context=context)
