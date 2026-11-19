"""★11.16 cocktail_1c_n07b 카드 실험 토글 — 단계 명령에 어떻게 들어가는지 · 회차 CSV 에 남는지."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from mission_core import load_mission
from mission_stages import commands_for, load_runbook

from s2r_console import eval_log as EV

REPO = Path(__file__).resolve().parents[2]
RAW = yaml.safe_load((REPO / "config" / "mission_rh56f1_control.yaml").read_text(encoding="utf-8"))
MISSION = load_mission(RAW)
BOOK = load_runbook(RAW.get("run"), MISSION)
STAGE = "episode_cocktail_1c_n07b"
DEFAULT = {"tilt_comp": False, "hand_unfreeze_on_grasp": True, "thumb_support_g": True, "thumb_open_floor": True,
           "tin_subtract_offset": False, "rate_scale": 1.0,
           "grasp_r": "rcg04", "grasp_l": "rcg04", "place_r": "right_n07", "place_l": "left_n09_e3800"}
REAL = False if "fake" in str(RAW.get("id", "")) else True


def _cmds():
    return commands_for(BOOK, MISSION, STAGE, repo=REPO, execute=True)


def _param(cmds, script, name):
    argv = next(c.argv for c in cmds if any(script in a for a in c.argv))
    return [a.split(":=", 1)[1] for a in argv if a.startswith(f"{name}:=")]


def test_default_toggles_only_add_the_runner_param():
    before, text = _cmds(), EV.stage_toggles(STAGE, DEFAULT)
    after = EV.toggle_commands(before, text)
    assert text == ("tilt_comp=0,hand_unfreeze_on_grasp=1,thumb_support_g=1,thumb_open_floor=1,tin_subtract_offset=0,"
                    "rate_scale=1.0,grasp_r=rcg04,grasp_l=rcg04,place_r=right_n07,place_l=left_n09_e3800,tilt=v12")
    changed = [(a.argv, b.argv) for a, b in zip(before, after) if a != b]
    assert len(changed) == 2 and "episode_runner_node.py" in " ".join(changed[1][0])   # 시작 검사 --toggles + 실행기
    assert changed[0][1] == (*changed[0][0], "--toggles", text)
    assert changed[1][1] == (*changed[1][0], "-p", f"toggles:={text}")


@pytest.mark.parametrize("key,param,off", [("hand_unfreeze_on_grasp", "hand_unfreeze_on_grasp", "false"),
                                           ("thumb_support_g", "thumb_support_g", "0.0"),
                                           ("thumb_open_floor", "thumb_open_floor_rad", "0.0")])
def test_each_hand_assist_toggle_turns_off_only_its_param(key, param, off):
    """★11.18 사용자 "abc 모두 제거" — 생성기가 손 보조 인자를 안 넣으므로(노드 기본 = 끔) 토글을 꺼도 넣을 인자가 없다."""
    after = EV.toggle_commands(_cmds(), EV.stage_toggles(STAGE, {**DEFAULT, key: False}))
    for p in ("hand_unfreeze_on_grasp", "thumb_support_g", "thumb_open_floor_rad"):
        assert _param(after, "rh_aglt_node.py", p) == [], p


def _regen(t):
    from policy_control import episode_spec as S
    ep = S.load(REPO / "config" / "episodes" / "cocktail_1c_n07b.yaml")
    return EV._regen(REPO, MISSION, BOOK, STAGE, REAL, ep if t is None else S.apply_toggles(ep, t))


def test_default_regen_equals_the_mission_stage():
    assert [c.argv for c in _regen(None)] == [c.argv for c in _cmds()]


def test_picking_policies_rebuilds_the_stage():
    t = {**DEFAULT, "grasp_r": "student1105mA", "place_r": "right_n10_e4000", "place_l": "left_student_place"}
    text = EV.stage_toggles(STAGE, t)
    after = " ".join(" ".join(c.argv) for c in EV.toggle_commands(_cmds(), text, regen=_regen))
    assert "right_student1105mA" in after and "right_n10_e4000" in after and "left_student_place" in after
    assert "left_rcg04" in after and "right_rcg04" not in after       # 왼손 잡기는 그대로


def test_policy_pick_needs_regen_and_a_known_name():
    with pytest.raises(EV.EvalError):
        EV.toggle_commands(_cmds(), EV.stage_toggles(STAGE, {**DEFAULT, "place_r": "right_n10_e4000"}))
    with pytest.raises(EV.EvalError):
        EV.stage_toggles(STAGE, {**DEFAULT, "place_r": "right_nope"})


@pytest.mark.parametrize("raw", [{"tilt_comp": "yes"}, {"rate_scale": 0.3}, {"nope": True}, ["tilt_comp"]])
def test_bad_toggles_are_refused(raw):
    with pytest.raises(EV.EvalError):
        EV.stage_toggles(STAGE, raw)


def test_e2e_card_takes_only_the_tilt_choice():
    """★11.20 사용자 "cocktail_e2e 에서는 tilt 에서 v12 또는 v14 모델 고를 수 있고" — 다른 토글은 거부, tilt_comp 는 자산 yaml 값."""
    with pytest.raises(EV.EvalError):
        EV.stage_toggles("episode_cocktail_e2e", DEFAULT)
    with pytest.raises(EV.EvalError):
        EV.stage_toggles("episode_cocktail_e2e", {"tilt": "v13"})
    text = EV.stage_toggles("episode_cocktail_e2e", {"tilt": "v14"})
    assert text.endswith(",tilt=v14") and text.startswith("tilt_comp=1,")      # v14 policy.yaml tilt_comp: true


def test_e2e_tilt_v14_only_touches_the_runner_and_start_check():
    stage = "episode_cocktail_e2e"
    before = commands_for(BOOK, MISSION, stage, repo=REPO, execute=True)
    after = EV.toggle_commands(before, EV.stage_toggles(stage, {"tilt": "v14"}))
    changed = [b.argv for a, b in zip(before, after) if a != b]
    assert len(changed) == 2 and all(any(a.endswith(",tilt=v14") for a in argv) for argv in changed)


def test_t_cards_left_the_tab():
    assert "cocktail_1c_n07b_t" not in EV.EVAL_TAB and "cocktail_1c_e4000_t" not in EV.EVAL_TAB
    assert "cocktail_1c_n07b_t" not in EV.EVAL_EPISODES


def test_round_csv_keeps_toggles_and_migrates_old_header(tmp_path):
    (tmp_path / "config" / "episodes").mkdir(parents=True)
    src = REPO / "config" / "episodes" / "cocktail_1c_n07b.yaml"
    (tmp_path / "config" / "episodes" / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    from datetime import datetime
    now = datetime(2026, 11, 16, 12, 0, 0)
    old = EV.eval_dir(tmp_path) / "2026-11-16.csv"
    old.parent.mkdir(parents=True)
    old.write_text(",".join(EV.COLUMNS[:-1]) + "\n1,11:00:00,cocktail_1c_n07b,a,b,성공,,,\n", encoding="utf-8")
    row = EV.append_round(tmp_path, {"episode": "cocktail_1c_n07b", "result": "성공"}, now=now, toggles="tilt_comp=1")
    rows = EV.read_rows(tmp_path, "2026-11-16")
    assert row["round"] == "2" and [r["toggles"] for r in rows] == ["", "tilt_comp=1"]


def test_shutdown_stops_every_background_of_a_regenerated_stage():
    """정책을 바꿔 배경 명령이 늘어난 단계(7 → 9)도 shutdown 이 남김없이 내린다 — yaml 의 번호 목록만 믿지 않는다."""
    from types import SimpleNamespace

    from s2r_console.runner import StageRunner
    cmds = _regen({**DEFAULT, "grasp_r": "student1105mA"})
    assert sum(c.background for c in cmds) > sum(c.background for c in _cmds())
    up = {f"{STAGE}#{i}" for i, c in enumerate(cmds) if c.background}

    class Sup:
        def alive(self):
            return [SimpleNamespace(key=k) for k in sorted(up)]

        def is_alive(self, k):
            return k in up

        def stop(self, keys):
            up.difference_update(keys)
    r = StageRunner("shutdown", commands_for(BOOK, MISSION, "shutdown", repo=REPO, execute=True), Sup(), on_done=None)
    for step in r.steps:
        if step.targets:
            assert r._do_stop(step) is None
    assert not up


def test_shutdown_stops_backgrounds_of_a_stage_with_none_in_the_yaml():
    """기본 생성 때 배경 0개였던 단계가 토글로 배경을 띄워도 shutdown 이 pd 보다 먼저 내린다(머리 드라이버는 head_home 정지와 함께)."""
    from types import SimpleNamespace

    from s2r_console.runner import StageRunner
    book = commands_for(BOOK, MISSION, "shutdown", repo=REPO, execute=True)
    up = {"episode_new#0": ("python3", "x_policy.py"), "episode_new#1": ("python3", "head_joint_publisher.py")}
    order = []

    class Sup:
        def alive(self):
            return [SimpleNamespace(key=k, argv=a) for k, a in sorted(up.items())]

        def is_alive(self, k):
            return k in up

        def stop(self, keys):
            order.extend(keys)
            for k in keys:
                up.pop(k, None)
    r = StageRunner("shutdown", book, Sup(), on_done=None)
    for step in r.steps:
        if step.targets:
            assert r._do_stop(step) is None
    assert not up


@pytest.mark.parametrize("mission_file", ["mission_rh56f1_control.yaml", "mission_rh56f1_control_cam5090.yaml"])
def test_shutdown_orders_real_toggled_stage_backgrounds(mission_file):
    """실제 control 미션: yaml 에 없는 단계로 다시 만든 에피소드의 배경 전부가 내려가고, 정책은 pd 정지 전, 목 드라이버는 head_home#0 정지에서 내려간다."""
    from types import SimpleNamespace

    from s2r_console.runner import HEAD_DRIVER, StageRunner
    raw = yaml.safe_load((REPO / "config" / mission_file).read_text(encoding="utf-8"))
    mission = load_mission(raw)
    book = load_runbook(raw.get("run"), mission)
    cmds = [c for c in _regen(DEFAULT) if c.background]
    up = {f"episode_toggled#{i}": tuple(c.argv) for i, c in enumerate(cmds)}
    # 목 드라이버 argv 는 이 미션 head_home 단계 그대로 — 토글 단계(head_sweep 류)가 다시 띄운 것처럼 넣는다
    drv = [c for c in commands_for(book, mission, "head_home", repo=REPO, execute=True) if c.background]
    up.update({f"episode_toggled#{len(cmds) + i}": tuple(c.argv) for i, c in enumerate(drv)})
    heads = {k for k, a in up.items() if any(HEAD_DRIVER in x for x in a)}
    assert heads and len(heads) < len(up)
    stopped_at = {}

    class Sup:
        def alive(self):
            return [SimpleNamespace(key=k, argv=a) for k, a in sorted(up.items())]

        def is_alive(self, k):
            return k in up

        def stop(self, keys):
            for k in keys:
                stopped_at.setdefault(k, cur)
                up.pop(k, None)
    r = StageRunner("shutdown", commands_for(book, mission, "shutdown", repo=REPO, execute=True), Sup(), on_done=None)
    for idx, step in enumerate(r.steps):
        cur = (idx, tuple(step.targets))
        if step.targets:
            assert r._do_stop(step) is None
    assert not up
    first_pd = min(i for i, s in enumerate(r.steps) if any(t.startswith("pd_") for t in s.targets))
    for k, (idx, targets) in stopped_at.items():
        if k in heads:
            assert "head_home#0" in targets
        else:
            assert idx < first_pd


def test_head_driver_script_exists():
    from s2r_console.runner import HEAD_DRIVER
    assert (REPO / "scripts" / "nodes" / HEAD_DRIVER).is_file()


def test_head_driver_constant_matches_mission_and_file():
    """HEAD_DRIVER 상수가 실제 스크립트이고, 실 미션 head_home 단계가 그 이름으로 드라이버를 띄운다(이름 바뀌면 여기서 걸림)."""
    from s2r_console.runner import HEAD_DRIVER
    assert (REPO / "scripts" / "nodes" / HEAD_DRIVER).is_file()
    for f in ("mission_rh56f1_control.yaml", "mission_rh56f1_control_cam5090.yaml"):
        raw = yaml.safe_load((REPO / "config" / f).read_text(encoding="utf-8"))
        m = load_mission(raw)
        cs = commands_for(load_runbook(raw.get("run"), m), m, "head_home", repo=REPO, execute=True)
        assert any(c.background and any(HEAD_DRIVER in a for a in c.argv) for c in cs), f
