"""★11.16 실험 토글 — 실행기 파라미터 toggles 가 에피소드(놓기 tin_subtract_offset · tilt rate_scale · tilt_comp)에 닿는지."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from policy_control import _paths
from policy_control import episode_spec as S
from policy_control.episode_ros import RosExecutor, write_run_meta

EP = _paths.SIM2REAL / "config" / "episodes" / "cocktail_1c_n07b.yaml"


def test_parse_defaults_and_roundtrip():
    assert S.parse_toggles("") == S.TOGGLES
    t = S.parse_toggles("tilt_comp=1,rate_scale=0.5")
    assert t["tilt_comp"] is True and t["rate_scale"] == 0.5 and t["thumb_support_g"] is True and t["place_r"] == "right_n07"
    assert S.parse_toggles(S.format_toggles(t)) == t


@pytest.mark.parametrize("bad", ["x=1", "tilt_comp=2", "rate_scale=0.7", "rate_scale=a"])
def test_parse_refuses_bad(bad):
    with pytest.raises(S.EpisodeSpecError):
        S.parse_toggles(bad)


def test_apply_reaches_place_and_tilt():
    ep = S.load(EP)
    out = S.apply_toggles(ep, S.parse_toggles("tin_subtract_offset=1,rate_scale=0.5"))
    places = [b for b in out.policies.values() if b.kind == "place"]
    assert places and all(b.tin_subtract_offset for b in places)
    tilts = [t for t in out.trajectories.values() if t.kind == "tilt"]
    assert tilts and all(t.rate_scale == 0.5 for t in tilts)
    assert all(t.rate_scale == ep.trajectories[k].rate_scale for k, t in out.trajectories.items() if t.kind != "tilt")
    assert ep.toggles is None and not any(b.tin_subtract_offset for b in ep.policies.values())   # 원본 그대로
    S.validate(out)


def test_defaults_keep_the_episode():
    ep = S.load(EP)
    out = S.apply_toggles(ep, S.TOGGLES)
    assert out.policies == ep.policies and out.trajectories == ep.trajectories


def test_tilt_comp_toggle_beats_asset_yaml():
    ep = S.load(EP)
    f = RosExecutor._tilt_comp
    assert f(SimpleNamespace(ep=ep), {"tilt_comp": True}) is True            # 토글 없음 = 자산 yaml
    on, off = (S.apply_toggles(ep, S.parse_toggles(f"tilt_comp={v}")) for v in (1, 0))
    assert f(SimpleNamespace(ep=off), {"tilt_comp": True}) is False
    assert f(SimpleNamespace(ep=on), {}) is True


def test_meta_records_toggles(tmp_path):
    import json
    ep = S.apply_toggles(S.load(EP), S.parse_toggles("rate_scale=0.5"))
    write_run_meta(tmp_path, ep, "rh56f1")
    assert json.loads((tmp_path / "meta.json").read_text())["toggles"]["rate_scale"] == 0.5


def test_tilt_choice_swaps_only_the_tilt_asset():
    """★11.20 cocktail_e2e 카드 tilt 고르기 — v12 = yaml 그대로, v14 = tilt/traj_v14r1l1_240. 다른 궤적 · 정책은 그대로."""
    ep = S.load(_paths.SIM2REAL / "config" / "episodes" / "cocktail_e2e.yaml")
    v12, v14 = (S.apply_toggles(ep, S.parse_toggles(f"tilt={v}")) for v in ("v12", "v14"))
    assert v12.trajectories == ep.trajectories
    assert v14.trajectories["tilt"].asset == "tilt/traj_v14r1l1_240"
    assert {k: t for k, t in v14.trajectories.items() if k != "tilt"} == {k: t for k, t in ep.trajectories.items() if k != "tilt"}
    assert v14.policies == ep.policies
    S.validate(v14)
    with pytest.raises(S.EpisodeSpecError):
        S.parse_toggles("tilt=v13")
