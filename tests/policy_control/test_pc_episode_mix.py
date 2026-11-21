"""★11.18 shaker 섞기(trajectory kind mix) — 에피소드 파싱 · 왼팔만 · plan_mix 실패면 재생하지 않음."""
from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import yaml

from policy_control import episode_spec as S

REPO = Path(__file__).resolve().parents[2]
EPISODE = REPO / "config" / "episodes" / "cocktail_e2e.yaml"


def _raw(**traj):
    d = yaml.safe_load(EPISODE.read_text(encoding="utf-8"))
    d["trajectories"]["mix_swirl"] = {**d["trajectories"]["mix_swirl"], **traj}
    return d


def test_mix_node_parses_left_only_after_right_home_before_place_shaker():
    ep = S.load(EPISODE)
    t = ep.trajectories["mix_swirl"]
    assert (t.kind, t.sides, t.asset, t.cups, t.rate_scale) == ("mix", ("left",), "mix/shaker_swirl", True, 0.6)
    ids = [n.id for n in ep.nodes]
    assert ids[ids.index("mix_shaker") - 1: ids.index("mix_shaker") + 2] == ["home_r_after_pink", "mix_shaker", "place_shaker"]


@pytest.mark.parametrize("bad", [{"sides": ["right"]}, {"sides": ["right", "left"]}, {"cups": False},
                                 {"asset": "tilt/base_arc6"}, {"rate_scale": 0.9}])
def test_bad_mix_trajectories_are_refused(bad):
    with pytest.raises(S.EpisodeSpecError):
        S.parse(_raw(**bad), path=str(EPISODE))


def test_mix_rate_scale_may_only_be_slower_than_the_asset():
    assert S.parse(_raw(rate_scale=0.3), path=str(EPISODE)).trajectories["mix_swirl"].rate_scale == 0.3


def _go(tmp_path, rcs: list[int]):
    from policy_control.episode_ros import RosExecutor
    calls = []

    def run(argv):
        calls.append(argv)
        return rcs[len(calls) - 1], "tail"
    me = SimpleNamespace(_tin_rec=None, ep=SimpleNamespace(tin_fallback=""), _wait_pd_tracking=lambda side: True,
                         _wait_arm_still=lambda sides: {"arm_still": None}, _pd_view=lambda side: {}, run_dir=tmp_path,
                         _run=run, cancel=threading.Event())
    world = SimpleNamespace(hand=lambda side: {"right": "EMPTY", "left": "SHAKER"}[side])
    traj = S.load(EPISODE).trajectories["mix_swirl"]
    return RosExecutor._run_planned(me, SimpleNamespace(id="mix_shaker"), traj, world), calls


def test_plan_mix_check_failure_refuses_and_never_replays(tmp_path):
    r, calls = _go(tmp_path, [2])
    assert r.status == "refused" and len(calls) == 1 and Path(calls[0][1]).name == "plan_mix.py"
    assert "--q-json" not in calls[0]                       # 실측에서 계획


def test_plan_mix_pass_replays_left_arm_only_at_the_asset_rate(tmp_path):
    r, calls = _go(tmp_path, [0, 0])
    assert r.status == "completed" and len(calls) == 2
    rp = calls[1]
    assert Path(rp[1]).name == "replay_to_pd.py" and rp[rp.index("--npz") + 1].endswith("mix_shaker/mix_cmd.npz")
    assert rp[rp.index("--joints") + 1] == ",".join(f"l_aj_{i}" for i in range(1, 8))
    assert float(rp[rp.index("--rate-scale") + 1]) == 0.6 and float(rp[rp.index("--max-start-gap") + 1]) == 0.05
    assert "--execute" in rp and "--tilt-comp" not in rp
