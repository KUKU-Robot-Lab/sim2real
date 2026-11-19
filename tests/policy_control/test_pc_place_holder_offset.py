"""★11.18 holder_offset_base_m — yaml → job_params(실수 배열) → SetParameters DOUBLE_ARRAY → 노드가 정책 목표 홀더만 옮긴다."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policy_control.episode_spec import EpisodeSpecError, load as load_episode

ROOT = Path(__file__).resolve().parents[2]
EP = ROOT / "config/episodes/cocktail_e2e.yaml"


def _ep(tmp_path, value):
    t = EP.read_text()
    assert t.count("holder_offset_base_m: [0.0, 0.0, 0.0]}") == 1
    p = tmp_path / "x.yaml"
    p.write_text(t.replace("holder_offset_base_m: [0.0, 0.0, 0.0]}", f"holder_offset_base_m: {value}}}"))
    return load_episode(p)


def test_spec_and_job_params_send_floats(tmp_path):
    from policy_control.episode_ros import job_params
    from policy_control.episode_runner import EpisodeManager
    ep = _ep(tmp_path, "[-0.01, 0, 0]")                  # 정수 0 이 섞여도
    assert ep.policies["rh_place_r"].holder_offset_base_m == (-0.01, 0.0, 0.0)
    assert ep.policies["rh_place_l"].holder_offset_base_m is None
    seen = {}
    for n in ep.nodes:
        for plan in EpisodeManager._plans(SimpleNamespace(ep=ep), n, SimpleNamespace(objects={})):
            if plan.kind == "place":
                seen[plan.role] = job_params(ep, plan, {"right": "a.json", "left": "b.json"})["holder_offset_base_m"]
    assert seen["rh_place_r"] == [-0.01, 0.0, 0.0] and seen["rh_place_l"] == [0.0, 0.0, 0.0]
    assert all(type(v) is float for vs in seen.values() for v in vs)


@pytest.mark.parametrize("bad", ["[0.2, 0, 0]", "[0.01, 0]", "abc"])
def test_spec_refuses_bad_offset(tmp_path, bad):
    with pytest.raises(EpisodeSpecError):
        _ep(tmp_path, bad)


def test_set_parameters_uses_double_array():
    R = pytest.importorskip("policy_control.episode_ros")
    from rcl_interfaces.msg import ParameterType
    sent = []

    class _Cli:
        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, req):
            sent.extend(req.parameters)
            return SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(results=[SimpleNamespace(successful=True)] * len(req.parameters)))

    me = SimpleNamespace(node=SimpleNamespace(create_client=lambda *a, **k: _Cli(), destroy_client=lambda c: None),
                         _cbg=None)
    ok, why = R.RosExecutor._set_params(me, "/rh_place_node_right", {"holder_offset_base_m": [-0.01, 0, 0], "holder": 1})
    assert ok, why
    got = {p.name: p.value for p in sent}
    assert got["holder_offset_base_m"].type == ParameterType.PARAMETER_DOUBLE_ARRAY
    assert list(got["holder_offset_base_m"].double_array_value) == [-0.01, 0.0, 0.0]
    assert got["holder"].type == ParameterType.PARAMETER_INTEGER


def test_node_moves_only_the_policy_holder(monkeypatch):
    N = pytest.importorskip("policy_control.rh_place_node")
    node = N.PlaceNode.__new__(N.PlaceNode)
    params = {"object": "", "joint_free_g": 0.0, "release_open_min_rad": 0.5, "tin_subtract_offset": False,
              "holder_offset_base_m": [-0.01, 0, 0], "allow_measured_start": False}
    node.get_parameter = lambda n: SimpleNamespace(value=params[n])
    got = {}
    node.chain = SimpleNamespace(rel=[np.zeros(3)], rcarry=None, attach_info={},
                                 reset=lambda meas, holder, held, allow_measured, attach: got.update(holder=holder),
                                 as_dict=lambda: {"seat": None})
    node.contract = SimpleNamespace(holder_range=None, side=lambda: SimpleNamespace(role="right", side="right"))
    node.fk, node._held = {"right": None}, (lambda meas: None)
    node._holder = real = np.array([0.3832, -0.0601, 0.295])
    node._latch_est = node._rc_chain = node._bundle = None
    node._tin, node._attach_info = False, (lambda: {})
    node.get_logger = lambda: SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
    node._reset_chain({})
    assert np.allclose(got["holder"], [0.3732, -0.0601, 0.295])
    assert np.allclose(node._holder, real)               # 판정 · 거부 · cup_in_holder 기준은 실제 홀더 그대로
    assert node.chain.attach_info["holder_offset_base_m"] == [-0.01, 0.0, 0.0]
    params["holder_offset_base_m"] = [0.3, 0, 0]         # 한도 밖은 reset 거부
    with pytest.raises(N.PourFjNodeError):
        node._reset_chain({})
