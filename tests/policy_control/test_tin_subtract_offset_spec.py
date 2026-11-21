from pathlib import Path

import pytest

from policy_control.episode_spec import EpisodeSpecError, load as load_episode

ROOT = Path(__file__).resolve().parents[2]


def test_n07b_t_turns_on_both_hands():
    ep = load_episode(ROOT / "config/episodes/cocktail_1c_n07b_t.yaml")
    assert ep.policies["rh_place_r"].tin_subtract_offset is True
    assert ep.policies["rh_place_l"].tin_subtract_offset is True


def test_requires_tin_json(tmp_path):
    t = (ROOT / "config/episodes/cocktail_1c_n07b_t.yaml").read_text()
    p = tmp_path / "x.yaml"
    p.write_text(t.replace("attach_source: tin_json, tin_subtract_offset", "tin_subtract_offset"))
    with pytest.raises(EpisodeSpecError):
        load_episode(p)


def test_e4000_t_turns_on_both_hands():
    ep = load_episode(ROOT / "config/episodes/cocktail_1c_e4000_t.yaml")
    assert ep.name == "cocktail_1c_e4000_t"
    assert all(ep.policies[r].tin_subtract_offset and ep.policies[r].attach_source == "tin_json" for r in ("rh_place_r", "rh_place_l"))
