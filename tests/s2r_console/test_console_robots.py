"""로봇 모듈 · 첫 화면 정책 고르기 — 09.29 사용자.

"처음 창을 키면 robot 및 실행 가능한 policy 선택창 · robot setting 및 state · (앞으로도 추가될 예정이므로 모듈화)".

여기서 잠그는 것:
  ① 로봇 하나 = robots/<id>.yaml 하나 — 깨진 파일은 숨기지 않고 사유와 함께 낸다
  ② 정책은 계약의 asset 이 로봇 자산과 같을 때만 고를 수 있다(계약 없음 · hold · 쪽 다름은 이유와 함께 보인다)
  ③ 고른 정책은 **그 run 의** 미션 산출물만 바꾼다 — yaml 은 그대로, 정책마다 다른 실행 인자는 정책 카드(deploy)에서
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from s2r_console import robots as RB
from s2r_console.errors import ProfileError

SIM2REAL = Path(__file__).resolve().parents[2]
ROBOTS = SIM2REAL / "deploy" / "s2r_console" / "robots"


def _info(pid, *, side="left", asset="asset_a", contract="joint_contract.json", status="candidate", task="open-a_l",
          issues=()):
    return RB.PolicyInfo(id=pid, status=status, side=side, task=task, contract=contract, asset=asset, issues=tuple(issues))


def _robot(tmp_path, **over):
    raw = {"schema": RB.SCHEMA, "id": "bot", "title": "봇", "asset": "asset_a", "task_prefixes": ["open-a_"],
           "profiles": ["t_fake"], "slots": {"left": "joint_left"}, **over}
    return RB.parse(raw, path=tmp_path / "bot.yaml")


def test_the_repo_robot_modules_parse():
    good, bad = RB.scan(ROBOTS)
    assert not bad, bad
    ids = {r.id: r for r in good}
    assert {"openarm_dg5f_m_short", "openarm_rh56f1"} <= set(ids)
    dg = ids["openarm_dg5f_m_short"]
    assert dg.asset == "openarm_dg5f-m-short_bi_rl" and dg.slots == {"right": "joint_right", "left": "joint_left"}
    assert {"dg5f_m_real", "dg5f_m_fake"} <= set(dg.profiles) and dg.host == "local5090"
    rh = ids["openarm_rh56f1"]
    assert rh.asset == "openarm_rh56f1_bi_rl" and rh.host == "arm4090" and "open-rh_" in rh.task_prefixes


def test_every_profile_a_robot_names_exists():
    from s2r_console.profiles import scan  # noqa: PLC0415
    good, _ = scan(SIM2REAL / "deploy" / "s2r_console" / "profiles", repo=SIM2REAL)
    have = {p.id for p in good}
    for r in RB.scan(ROBOTS)[0]:
        assert set(r.profiles) <= have, (r.id, set(r.profiles) - have)


def test_a_broken_module_is_reported_not_hidden(tmp_path):
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump({"schema": RB.SCHEMA, "id": "ok", "title": "t", "asset": "a"}))
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"schema": RB.SCHEMA, "id": "other", "title": "t", "asset": "a"}))
    (tmp_path / "slot.yaml").write_text(yaml.safe_dump({"schema": RB.SCHEMA, "id": "slot", "title": "t", "asset": "a",
                                                        "slots": {"left": "robot_left"}}))
    good, bad = RB.scan(tmp_path)
    assert [r.id for r in good] == ["ok"]
    assert "파일 이름" in bad["bad.yaml"] and "접두어" in bad["slot.yaml"]


def test_only_policies_of_this_robot_are_listed_with_reasons(tmp_path):
    r = _robot(tmp_path)
    policies = [_info("mine"), _info("other_asset", asset="asset_b"), _info("no_contract", contract="", asset=""),
                _info("foreign_task", contract="", asset="", task="open-b_l"), _info("held", status="hold"),
                _info("right_one", side="right"), _info("pour", contract="pour_contract.json")]
    rows = {p["id"]: p for p in RB.choices(r, policies)}
    assert set(rows) == {"mine", "no_contract", "held", "right_one", "pour"}         # 다른 로봇 것은 안 보인다
    assert rows["mine"]["why"] == []
    assert any("계약이 없다" in w for w in rows["no_contract"]["why"])
    assert any("hold" in w for w in rows["held"]["why"])
    assert any("자리가 없다" in w for w in rows["right_one"]["why"])
    assert any("받는 계약" in w for w in rows["pour"]["why"])


def test_overrides_map_the_pick_to_the_slot_artifact(tmp_path):
    repo = tmp_path
    (repo / "deploy/policies/mine").mkdir(parents=True)
    r = _robot(tmp_path)
    out = RB.overrides(r, {"left": "mine"}, {"mine": _info("mine")}, repo / "deploy/policies", repo)
    assert out == {"joint_left": "deploy/policies/mine/joint_contract.json"}
    with pytest.raises(ProfileError, match="hold"):
        RB.overrides(r, {"left": "held"}, {"held": _info("held", status="hold")}, repo / "deploy/policies", repo)
    with pytest.raises(ProfileError, match="없다"):
        RB.overrides(r, {"left": "nope"}, {}, repo / "deploy/policies", repo)


MISSION = """
name: 정책 고르기
artifacts:
  joint_left: deploy/policies/pol_a/joint_contract.json
stages:
  - id: policy_left
    title: 정책
    artifacts: [joint_left]
run:
  policy_left:
    - note: 정책 노드
      argv: [echo, "contract:={artifact:joint_left}", "success_tol_m:={policy:joint_left.success_tol_m}"]
"""


def _policy(repo: Path, pid: str, tol: float, *, asset="asset_a") -> None:
    d = repo / "deploy" / "policies" / pid
    (d / "nn").mkdir(parents=True)
    (d / "params").mkdir()
    (d / "nn" / f"{pid}.pth").write_bytes(pid.encode())
    for f in ("env.yaml", "agent.yaml"):
        (d / "params" / f).write_text("{}\n")
    import hashlib  # noqa: PLC0415
    md5 = hashlib.md5(pid.encode()).hexdigest()
    (d / "joint_contract.json").write_text(json.dumps({"schema": "policy_control/joint_contract/v1", "asset": asset,
                                                       "side": "left", "checkpoint": f"{pid}.pth", "checkpoint_md5": md5}))
    (d / "policy.yaml").write_text(yaml.safe_dump({"id": pid, "status": "candidate", "task": "open-a_l_x", "side": "left",
                                                   "checkpoint": f"{pid}.pth", "deploy": {"success_tol_m": tol}}))


@pytest.fixture()
def picker(tiny_repo, tmp_path, monkeypatch):
    from s2r_console import console as mod  # noqa: PLC0415
    (tiny_repo / "mission.yaml").write_text(textwrap.dedent(MISSION))
    _policy(tiny_repo, "pol_a", 0.03)
    _policy(tiny_repo, "pol_b", 0.11)
    _policy(tiny_repo, "pol_other", 0.5, asset="asset_b")
    robots = tmp_path / "robots"
    robots.mkdir()
    (robots / "bot.yaml").write_text(yaml.safe_dump({"schema": RB.SCHEMA, "id": "bot", "title": "봇", "host": "h",
                                                     "asset": "asset_a", "profiles": ["t_fake"],
                                                     "slots": {"left": "joint_left"},
                                                     "settings": [{"label": "손", "value": "포트 둘"}]}))
    noop = tmp_path / "noop_trigger.py"
    noop.write_text("import sys; sys.exit(0)\n")
    monkeypatch.setattr(mod, "_TRIGGER", noop)
    c = mod.Console(repo=tiny_repo, profiles_dir=tiny_repo / "profiles", bridge=False, robots_dir=robots)
    yield c
    c.shutdown()


def test_the_landing_lists_robots_with_their_policies(picker):
    snap = picker.snapshot()
    (bot,) = snap["robots"]
    assert [p["id"] for p in bot["profiles"]] == ["t_fake"] and bot["defaults"] == {"left": "pol_a"}
    rows = {p["id"]: p for p in bot["policies"]}
    assert set(rows) == {"pol_a", "pol_b"} and not rows["pol_b"]["why"]                 # 다른 자산은 안 보인다
    assert bot["settings"] == [{"label": "손", "value": "포트 둘"}]


def test_a_picked_policy_replaces_the_artifact_for_this_run_only(picker, tiny_repo):
    s = picker.open("t_fake", operator="pytest", policies={"left": "pol_b"})
    assert s.mission.artifacts["joint_left"] == "deploy/policies/pol_b/joint_contract.json"
    from mission_stages import commands_for  # noqa: PLC0415
    (cmd,) = commands_for(s.runbook, s.mission, "policy_left", repo=tiny_repo, execute=True)
    assert "success_tol_m:=0.11" in cmd.argv and any("pol_b/joint_contract.json" in a for a in cmd.argv)
    run = json.loads((s.run_dir / "run.json").read_text())
    assert run["robot"] == "bot" and run["policies"] == {"left": "pol_b"}
    assert "pol_a" in (tiny_repo / "mission.yaml").read_text()                            # yaml 은 그대로
    view = picker.snapshot()["session"]["robot_module"]
    assert view["slots_now"] == {"left": "pol_b"} and view["picked"] == {"left": "pol_b"}


def test_no_pick_keeps_the_mission_default(picker):
    s = picker.open("t_fake", operator="pytest")
    assert s.mission.artifacts["joint_left"] == "deploy/policies/pol_a/joint_contract.json"
    assert picker.snapshot()["session"]["robot_module"]["slots_now"] == {"left": "pol_a"}


def test_a_policy_of_another_robot_is_refused(picker):
    from s2r_console.console import ConsoleError  # noqa: PLC0415
    with pytest.raises(ConsoleError) as exc:
        picker.open("t_fake", operator="pytest", policies={"left": "pol_other"})
    assert any("asset_b" in r for r in exc.value.reasons)
    assert picker.session is None
