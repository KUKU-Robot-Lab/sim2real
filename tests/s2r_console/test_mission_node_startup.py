"""★11.18 미션 정책 노드 실제 시동 시험.

argv 문자열만 보는 시험은 ROS 파라미터 형 충돌(inhand_est 맨 off → bool)이나 노드 안 조합 검사
(cup_dropout_stop_s 는 cup_hold_last 와 같이)를 못 잡아 실기 콘솔에서 rc=1 이 두 번 났다. 여기서는 cam5090 미션의
.venv 정책 노드(rh_aglt · rh_place · pour_fj)를 로봇과 떨어진 ROS 도메인(localhost 전용)에서 실제로 띄워
'... up' 줄이 나오는지 본다. 로봇에 명령은 가지 않는다(토픽이 다른 도메인).
"""
from __future__ import annotations

import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MISSION = REPO / "config" / "mission_rh56f1_control_cam5090.yaml"
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
NODES = ("rh_aglt_node.py", "rh_place_node.py", "pour_fj_node.py")
ISOLATED_DOMAIN = "97"
UP_TIMEOUT_S = 40
MISSION_DOC = yaml.safe_load(MISSION.read_text(encoding="utf-8"))
ARTIFACTS: dict[str, str] = MISSION_DOC.get("artifacts") or {}
PLACEHOLDER_OTHER = re.compile(r"\{(?!repo\}|artifact:)")
UP_RE = re.compile(r"_node_(right|left)\S* up ")


def _node_argvs() -> dict[tuple[str, ...], str]:
    run = MISSION_DOC["run"]
    out: dict[tuple[str, ...], str] = {}

    def walk(x, sid):
        if isinstance(x, dict):
            av = x.get("argv")
            if isinstance(av, list) and str(av[0]) == "{repo}/.venv/bin/python" \
                    and any(str(a).endswith(NODES) for a in av) \
                    and not any(PLACEHOLDER_OTHER.search(str(a)) for a in av):   # {policy:…} 은 콘솔 런타임이 푼다
                out.setdefault(tuple(str(a) for a in av), sid)
            for v in x.values():
                walk(v, sid)
        elif isinstance(x, list):
            for v in x:
                walk(v, sid)

    for sid, stage in run.items():
        walk(stage, sid)
    return out


def _resolve(arg: str) -> str:
    arg = arg.replace("{repo}", str(REPO))
    for m in re.finditer(r"\{artifact:([A-Za-z0-9_]+)\}", arg):
        rel = ARTIFACTS.get(m.group(1))
        assert rel, f"미션 artifacts 에 없는 키: {m.group(1)}"
        arg = arg.replace(m.group(0), str(REPO / rel))
    assert "{" not in arg, f"풀지 못한 자리표시: {arg}"
    return arg


def _start(argv: tuple[str, ...]) -> tuple[bool, str]:
    cmd = shlex.join(_resolve(a) for a in argv)
    env = {**os.environ, "ROS_DOMAIN_ID": ISOLATED_DOMAIN, "ROS_LOCALHOST_ONLY": "1"}
    p = subprocess.Popen(["bash", "-c", f"source {ROS_SETUP} && exec {cmd}"], env=env, cwd=REPO,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    lines: list[str] = []
    ok = False
    assert p.stdout is not None
    try:
        sel = selectors.DefaultSelector()
        sel.register(p.stdout, selectors.EVENT_READ)
        end = time.monotonic() + UP_TIMEOUT_S
        while time.monotonic() < end:
            if not sel.select(timeout=0.5):
                if p.poll() is not None:
                    break
                continue
            line = p.stdout.readline()
            if not line:
                break
            lines.append(line)
            if UP_RE.search(line):
                ok = True
                break
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    return ok, "".join(lines[-15:])


@pytest.mark.skipif(not ROS_SETUP.exists() or not (REPO / ".venv/bin/python").exists(),
                    reason="ROS humble 또는 .venv 없음")
def test_every_mission_policy_node_starts():
    every = _node_argvs()
    # 정책 파일은 git 밖에서 따로 배포되는 것도 있다(pick_place cyl60g) — 이 PC 에 없는 계약은 시동 시험에서 빼고 이름을 남긴다
    argvs = {av: sid for av, sid in every.items() if all(Path(_resolve(a.split(":=", 1)[1])).exists()
                                                          for a in av if a.startswith("contract:="))}
    missing = sorted(set(every.values()) - set(argvs.values()))
    if missing:
        warnings.warn(f"정책 계약 파일이 없어 시동 시험에서 뺀 단계: {missing}")
    assert any("episode_grasp_f_right" == s for s in argvs.values())
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(_start, argvs))
    failed = [(sid, tail) for (sid, (ok, tail)) in zip(argvs.values(), results) if not ok]
    assert not failed, "\n\n".join(f"[{sid}]\n{tail}" for sid, tail in failed)


if shutil.which("bash") is None:  # pragma: no cover
    pytest.skip("bash 없음", allow_module_level=True)
