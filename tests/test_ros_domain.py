"""실기 배포 ROS 도메인은 config/ros_domain.env 한 곳 — 10.30 사용자: motion_acq(126 · 7)와 분리."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for sub in ("deploy/policy_control", "deploy/s2r_console", "scripts/fakes"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))
from policy_control import ros_domain  # noqa: E402

#: motion_acq 가 쓰는 도메인 — nova2 글러브 launch 126 · macq 콘솔 --real 7(4090 ~/.bashrc). 배포가 겹치면 5090 에서 /tf 가 보인다.
MOTION_ACQ_DOMAINS = (126, 7)
#: 옛 실기 숫자가 코드에 다시 박히지 않게 보는 꼴(주석 · 문서는 보지 않는다)
PINNED = re.compile(r'ROS_DOMAIN_ID=126\b|REAL_DOMAIN = 126\b|REAL_DOMAIN = "126"|in 0\|126\)|\(0, 126\)|= "126"|:-126\}')


def test_env_file_is_the_single_source(tmp_path: Path):
    assert ros_domain.read_real_domain() == ros_domain.REAL_DOMAIN
    assert ros_domain.REAL_DOMAIN_STR == str(ros_domain.REAL_DOMAIN)
    good = tmp_path / "ok.env"
    good.write_text("# 주석\n\nOTHER=1\nS2R_REAL_DOMAIN=\"77\"\n", encoding="utf-8")
    assert ros_domain.read_real_domain(good) == 77
    for text in ("OTHER=1\n", "S2R_REAL_DOMAIN=0\n", "S2R_REAL_DOMAIN=233\n"):
        bad = tmp_path / "bad.env"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            ros_domain.read_real_domain(bad)


def test_deploy_domain_is_separate_from_motion_acq_and_zero():
    assert ros_domain.REAL_DOMAIN not in (0, *MOTION_ACQ_DOMAINS)


def test_console_profiles_follow_the_env_file():
    import s2r_console._paths  # noqa: F401
    from s2r_console import profiles

    assert profiles.REAL_DOMAIN == ros_domain.REAL_DOMAIN
    real = []
    for path in sorted((REPO / "deploy/s2r_console/profiles").glob("*.yaml")):
        dom = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("domain") or {}
        if dom.get("class") == "real":
            real.append(path.name)
            assert dom.get("id") == ros_domain.REAL_DOMAIN, path.name
    assert real, "real 프로파일이 하나도 없다"


def test_fake_hand_and_runner_refuse_the_real_domain():
    import fake_rh56f1_hand as fh

    assert fh.REAL_DOMAIN == ros_domain.REAL_DOMAIN
    runner = (REPO / "deploy/policy_control/policy_control/episode_runner_node.py").read_text(encoding="utf-8")
    assert "ros_domain.REAL_DOMAIN_STR" in runner


@pytest.mark.parametrize("rel", ["scripts/vision/common.sh", "scripts/ops/restart_perception_nodes.sh",
                                 "scripts/setup/check_cup_pose_link.sh",
                                 "deploy/policy_control/tools/fake_replay_rh_run.sh"])
def test_shell_scripts_source_the_env_file(rel: str):
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "config/ros_domain.env" in text and "S2R_REAL_DOMAIN" in text, rel


def test_no_code_pins_the_old_real_domain():
    hits = []
    for top in ("scripts", "deploy/policy_control", "deploy/s2r_console", "config"):
        for path in (REPO / top).rglob("*"):
            if path.suffix not in (".py", ".sh", ".yaml", ".yml") or "__pycache__" in path.parts:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if PINNED.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{i}: {line.strip()[:80]}")
    assert not hits, "\n".join(hits)
