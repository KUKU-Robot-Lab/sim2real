"""deploy/policies/INDEX.md — 정책 설명 · 계열별 학습 조건 차이 · 개별 실행법 (10.04 사용자: 비교 분석용 개별 실행)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from policy_control import policy_index as X
from policy_control import policy_registry as R

SIM2REAL = Path(__file__).resolve().parents[2]
AGLT_ENV = {"profile_name": "rh56f1_right", "cup_scale": 0.65, "tol_start": 0.1, "tol_floor": 0.02, "hold_steps": 10,
            "reward_code_path": "/home/oem/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_10/compute_reward.py"}


def mk(root: Path, pid: str, task: str, side: str, env: dict, *, ckpt: str = "last_x_ep_2600_rew_1.0.pth",
       **card) -> Path:
    d = root / "deploy" / "policies" / pid
    (d / "nn").mkdir(parents=True)
    (d / "params").mkdir()
    (d / "nn" / ckpt).write_bytes(b"w")
    (d / "params" / "env.yaml").write_text(yaml.safe_dump(env))
    (d / "params" / "agent.yaml").write_text("x: 1\n")
    (d / R.CARD).write_text(yaml.safe_dump({"id": pid, "status": "candidate", "task": task, "side": side,
                                            "checkpoint": ckpt, **card}, allow_unicode=True))
    return d


@pytest.mark.parametrize("task, family", [
    ("open-rh_r_aglt-lstm", "rh_aglt"), ("open-rh_l_aglt-lstm", "rh_aglt"), ("open-rh_l_place-lstm", "rh_place"),
    ("open-rh_b_pour_fj-lstm", "pour_fj"), ("open-short_l_cup_pick-lstm", "dg5f_joint"),
    ("open-short_l_cup_grasp-lstm", "dg5f_joint"), ("open-short_r_grasp_fj_t2r_rand-lstm", "dg5f_joint"),
    ("open-short_b_pour_fab", "pour_fab"), ("open-other_x", "other")])
def test_family_is_read_from_the_task(task, family):
    assert X.family_of(task).key == family


@pytest.mark.parametrize("name, epoch", [
    ("last_open-rh_r_aglt-lstm_ep_2600_rew_3509.6997.pth", "ep2600"), ("cg_l_i01_e5802.pth", "e5802"),
    ("place_r_i09_ep4000.pth", "ep4000"), ("fj_rand_i01_best_ep5000.pth", "ep5000"), ("open-short_b_pour_fab.pth", "best"),
    ("aglt_l_i05_ep3800_to_r.pth", "ep3800")])
def test_epoch_is_read_from_the_checkpoint_name(name, epoch):
    assert X.epoch_of(name) == epoch


def test_play_task_is_the_registered_play_variant():
    assert X.play_task("open-rh_r_aglt-lstm") == "open-rh_r_aglt-play-lstm"
    assert X.play_task("open-short_b_pour_fab") == "open-short_b_pour_fab-play"


def test_rh_aglt_columns_spell_out_what_the_policy_was_trained_with(tmp_path):
    new = mk(tmp_path, "r_new", "open-rh_r_aglt-lstm", "right",
             {**AGLT_ENV, "object_name": "cyl60", "fpp_enable": True, "fpp_attach_enable": True, "tol_start": 0.02,
              "arm_cmd_delay_steps": [0, 0], "hand_cmd_delay_steps": [0, 0], "adr_start_increment": 20})
    mk(tmp_path, "r_delay", "open-rh_r_aglt-lstm", "right",
       {**AGLT_ENV, "arm_cmd_delay_steps": [9, 12], "hand_cmd_delay_steps": [3, 5]})
    old = mk(tmp_path, "r_old", "open-rh_r_aglt-lstm", "right", dict(AGLT_ENV))
    rows = {e.id: X.columns(e) for e in R.scan(new.parent, deep=False)}
    assert rows["r_new"]["학습 물체"].startswith("cyl60") and rows["r_old"]["학습 물체"].startswith("shaker×0.65")
    assert rows["r_new"]["FP++ (학습)"] == "지각 + 부착" and rows["r_old"]["FP++ (학습)"] == "없음"
    assert rows["r_delay"]["명령 지연"] == "팔 9–12 · 손 3–5 스텝"
    assert rows["r_new"]["명령 지연"] == "없음" and rows["r_old"]["명령 지연"] == "없음 (키 전 런)"
    assert rows["r_new"]["보상"] == "rh_aglt_r/iter_10" and rows["r_new"]["공차"] == "0.02"
    assert rows["r_old"]["공차"] == "0.1→0.02" and rows["r_new"]["시작"] == "홈 · hold 10 (ADR 20 부터)"
    assert old.name == "r_old"


def test_place_policies_say_they_start_from_the_handoff(tmp_path):
    d = mk(tmp_path, "r_place", "open-rh_r_place-lstm", "right",
           {**AGLT_ENV, "object_name": "cyl60", "hold_steps": 0, "start_bank_path": "~/b.npz", "target_holders": [1, 2]})
    row = X.columns(R.check(d, deep=False))
    assert row["시작"] == "인계 뱅크 (hold 0)" and row["목표"] == "홀더 1 · 2"


def test_mission_defaults_are_read_from_the_mission_files(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "mission_rh56f1_control.yaml").write_text(
        "artifacts:\n  aglt_right: deploy/policies/r_new/rh_aglt_contract.json\n")
    (tmp_path / "config" / "mission_rh56f1_fake.yaml").write_text("x: deploy/policies/r_new/a.json\n")
    (tmp_path / "config" / "other.yaml").write_text("x: deploy/policies/r_old/a.json\n")
    assert X.mission_defaults(tmp_path) == {"r_new": ("rh56f1_control", "rh56f1_fake")}


def test_render_groups_by_family_and_keeps_every_policy_with_how_to_run_it(tmp_path):
    mk(tmp_path, "r_new", "open-rh_r_aglt-lstm", "right", {**AGLT_ENV, "object_name": "cyl60"},
       summary="새 cyl60 파지", eval="성공 1.77/ep", status="candidate")
    mk(tmp_path, "b_fab", "open-short_b_pour_fab", "both", {"reward_code_path": "/x/pour_bi/iter_18/c.py"},
       ckpt="open-short_b_pour_fab.pth", status="hold", note="기울기 155°")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "mission_rh56f1_control.yaml").write_text("a: deploy/policies/r_new/x.json\n")
    text = X.render(R.scan(tmp_path / "deploy" / "policies", deep=False), repo=tmp_path)
    assert text.index("RH56F1 한 팔 파지") < text.index("DG-5F short 양팔 붓기")
    assert "새 cyl60 파지" in text and "성공 1.77/ep" in text and "기울기 155°" in text
    assert "--task open-rh_r_aglt-play-lstm" in text and "deploy/policies/r_new/nn/last_x_ep_2600_rew_1.0.pth" in text
    assert "rh56f1_control" in text
    assert all(f"`{s}`" in text for s in R.STATUSES)


def test_the_committed_index_is_regenerated():
    root = SIM2REAL / "deploy" / "policies"
    entries = R.scan(root, deep=False)
    if not entries or any(e.issues for e in entries):
        pytest.skip("가중치가 없는 호스트 — 점검 문제가 있으면 INDEX 가 달라진다")
    assert (root / R.INDEX).read_text() == X.render(entries, repo=SIM2REAL), \
        "deploy/policies/INDEX.md 가 카드 · params 와 다르다 — policies.py --write-index 로 다시 만들 것"
