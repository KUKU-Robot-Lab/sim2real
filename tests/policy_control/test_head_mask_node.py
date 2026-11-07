"""11.07 head_mask_node 의 rclpy · ultralytics 없는 부분: 인스턴스 고르기 · DDS 프로파일 검사 · color 변환."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
NODE = Path(__file__).resolve().parents[2] / "scripts/nodes/head_mask_node.py"


@pytest.fixture(scope="module")
def hm():
    spec = importlib.util.spec_from_file_location("head_mask_node", NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_keep_instances_takes_top1_per_class_and_drops_cross_class_overlaps(hm):
    m = torch.zeros(4, 4, 10, dtype=torch.bool)
    m[0, :, :5] = True                 # blue  0.9
    m[1, :, 6:] = True                 # blue  0.4 (탁자 밖 오검출 꼴)
    m[2, :, :5] = True                 # pink  0.6 — blue 0 과 같은 자리(IoU 1)
    m[3, :, 5:6] = True                # orange 0.5
    cls, conf = torch.tensor([0, 0, 1, 2]), torch.tensor([0.9, 0.4, 0.6, 0.5])
    assert hm.keep_instances(cls, conf, m, True, 0.5) == [0, 3]
    assert hm.keep_instances(cls, conf, m, False, 0.5) == [0, 3, 1]
    assert hm.keep_instances(cls, conf, m, True, 0.0) == [0, 2, 3]


def test_dds_profile_is_required_on_the_lan_path(hm, tmp_path):
    def xml(size):
        p = tmp_path / f"p{size}.xml"
        p.write_text(f'<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"><transport_descriptors>'
                     f'<transport_descriptor><transport_id>s</transport_id><type>SHM</type><segment_size>{size}</segment_size>'
                     f'</transport_descriptor></transport_descriptors></profiles>')
        return str(p)

    assert "lan_profile" in hm.dds_profile_problem({})[0]
    assert hm.dds_profile_problem({"FASTRTPS_DEFAULT_PROFILES_FILE": xml(67108864)}) == (None, None)
    assert "segment_size" in hm.dds_profile_problem({"FASTRTPS_DEFAULT_PROFILES_FILE": xml(524288)})[0]
    bad, warn = hm.dds_profile_problem({"ROS_LOCALHOST_ONLY": "1"})
    assert bad is None and "시험 전용" in warn
    assert "격리" in hm.dds_profile_problem({"ROS_LOCALHOST_ONLY": "1", "FASTRTPS_DEFAULT_PROFILES_FILE": xml(67108864)})[0]


def test_color_to_bgr(hm):
    rgb = np.arange(720 * 1280 * 3, dtype=np.uint64).astype(np.uint8)
    out = hm.color_to_bgr(720, 1280, "rgb8", 3840, rgb.tobytes())
    assert out.shape == (720, 1280, 3) and (out[..., 0] == rgb.reshape(720, 1280, 3)[..., 2]).all()
    with pytest.raises(ValueError):
        hm.color_to_bgr(720, 1280, "mono8", 1280, bytes(720 * 1280))


def test_worker_ends_node_after_consecutive_failures(hm, monkeypatch):
    """11.07 재리뷰: 추론 · 발행 예외가 이어지면 워커가 조용히 죽지 않고 이유를 남기고 rclpy 를 내린다."""
    import sys
    import threading
    import time
    import types

    down = []
    monkeypatch.setitem(sys.modules, "rclpy", types.SimpleNamespace(try_shutdown=lambda: down.append(1)))
    monkeypatch.setattr(hm, "WORKER_FAIL_MAX", 3)
    log = types.SimpleNamespace(error=lambda *a, **k: None, fatal=lambda *a, **k: None)
    h = object.__new__(hm.HeadMask)
    h.log, h.fails, h.dead, h._latest = log, 0, "", None
    h._lock, h._ready, h._stop = threading.Lock(), threading.Event(), threading.Event()

    def boom(msg):
        raise RuntimeError("publish 실패")
    h._process = boom
    w = threading.Thread(target=h._run, daemon=True)
    w.start()
    for _ in range(10):
        if not w.is_alive():
            break
        with h._lock:
            h._latest = object()
        h._ready.set()
        time.sleep(0.05)
    w.join(timeout=2.0)
    assert not w.is_alive() and "3 프레임 연속" in h.dead and down == [1]
