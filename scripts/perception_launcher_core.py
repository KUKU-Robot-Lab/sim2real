#!/usr/bin/env python3
"""perception_launcher_node 의 순수 로직 — 명령 파싱·액션 계획·상태 집계.

ROS·ssh 없이 import 된다. test_perception_launcher_core.py 대상.
"""
from __future__ import annotations

import json
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import CONTAINER_PREFIX, container_name  # noqa: E402

_OPS = ("start", "stop", "viewer")


@dataclass(frozen=True)
class Command:
    op: str
    objects: tuple[str, ...]
    viewer: bool | None
    camera: bool


@dataclass(frozen=True)
class RemoteState:
    camera_up: bool
    containers: dict[str, str]
    viewer_up: bool
    #: FP++ 자세 → 로봇 PC UDP 송신기(09.26). 영상 · FP++ 는 vision-3090 안에서만 돌아 이것 없이는 자세가 안 온다
    pose_tx_up: bool = False
    #: 컨테이너 이름 → 추적 노드가 죽은 마지막 오류 줄(09.29: 컨테이너 Up · 노드 CUDA OOM · 자세 0)
    crashes: dict[str, str] = field(default_factory=dict)
    #: vision-3090 GPU {"used_mib","total_mib"} — 학습과 같이 쓴다. 옛 status.sh 는 None
    gpu: dict | None = None


def parse_command(text: str, registry) -> Command:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"command is not JSON: {err}") from err
    if not isinstance(raw, dict) or raw.get("op") not in _OPS:
        raise ValueError(f"command 'op' must be one of {_OPS}: {text!r}")
    op = raw["op"]
    objects: tuple[str, ...] = ()
    if op == "start":
        names = raw.get("objects")
        if not isinstance(names, list) or not names:
            raise ValueError("start needs a non-empty 'objects' list")
        seen: list[str] = []
        for name in names:
            canon = registry.resolve(str(name))
            if canon not in seen:
                seen.append(canon)
        objects = tuple(seen)
    viewer = raw.get("viewer") if op == "start" else raw.get("on")
    if viewer is not None and not isinstance(viewer, bool):
        raise ValueError("'viewer'/'on' must be a boolean")
    if op == "viewer" and viewer is None:
        raise ValueError("viewer needs 'on': true|false")
    return Command(op=op, objects=objects, viewer=viewer, camera=bool(raw.get("camera", False)))


def parse_remote_status(text: str) -> RemoteState:
    try:
        raw = json.loads(text.strip().splitlines()[-1]) if text.strip() else None
    except json.JSONDecodeError:
        raw = None
    if not isinstance(raw, dict) or "containers" not in raw:
        raise ValueError(f"remote status is not the expected JSON: {text[:200]!r}")
    gpu = raw.get("gpu")
    gpu = {"used_mib": int(gpu["used_mib"]), "total_mib": int(gpu["total_mib"])} if isinstance(gpu, dict) else None
    return RemoteState(camera_up=bool(raw.get("camera_up")),
                       containers={str(k): str(v) for k, v in raw["containers"].items()},
                       viewer_up=bool(raw.get("viewer_up")), pose_tx_up=bool(raw.get("pose_tx_up")),
                       crashes={str(k): str(v) for k, v in (raw.get("crashes") or {}).items() if v},
                       gpu=gpu)


_LAUNCH_PREFIX = re.compile(r"^\[[^\]]+\]\s?")
_ERROR_LINE = re.compile(r"^[\w.]*(Error|Exception)\b.*")


def last_crash(log: str, limit: int = 160) -> str | None:
    """컨테이너 로그에서 마지막 Traceback 의 오류 줄(없으면 None). vision-3090 의 status.sh 가 부른다.
    ros2 launch 가 붙이는 '[노드-1] ' 머리는 뗀다."""
    lines = [_LAUNCH_PREFIX.sub("", ln) for ln in log.splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last)")]
    if not starts:
        return None
    for ln in lines[starts[-1] + 1:]:
        if _ERROR_LINE.match(ln.strip()):
            return ln.strip()[:limit]
    return "Traceback (오류 줄 없음)"


def _container(name: str, group_of: Callable[[str], str | None] | None) -> str:
    group = group_of(name) if group_of else None
    return container_name(group) if group else container_name(name)


def plan_actions(cmd: Command, state: RemoteState,
                 group_of: Callable[[str], str | None] | None = None) -> list[tuple[str, ...]]:
    """group_of(물체) → 묶음 이름(objects.yaml fpp.group) — 묶음 물체는 컨테이너 fpp_<묶음> 하나(10.08 한 번 찍기)."""
    actions: list[tuple[str, ...]] = []
    if cmd.op == "start":
        if not state.camera_up:
            actions.append(("camera_up",))
        wanted = {_container(n, group_of) for n in cmd.objects}
        for cname in sorted(state.containers):
            if cname.startswith(CONTAINER_PREFIX) and cname not in wanted:
                actions.append(("fpp_down", cname))
        started: set[str] = set()
        for name in cmd.objects:
            cname = _container(name, group_of)
            if cname in started or state.containers.get(cname, "").startswith("Up"):
                continue
            started.add(cname)
            group = group_of(name) if group_of else None
            actions.append(("fpp_group_up", group) if group else ("fpp_up", name))
        if not state.pose_tx_up:
            actions.append(("pose_tx_up",))
        if cmd.viewer is True and not state.viewer_up:
            actions.append(("viewer_up",))
        if cmd.viewer is False and state.viewer_up:
            actions.append(("viewer_down",))
        return actions
    if cmd.op == "stop":
        if state.viewer_up:
            actions.append(("viewer_down",))
        for cname in sorted(state.containers):
            if cname.startswith(CONTAINER_PREFIX):
                actions.append(("fpp_down", cname))
        if state.pose_tx_up:
            actions.append(("pose_tx_down",))
        if cmd.camera and state.camera_up:
            actions.append(("camera_down",))
        return actions
    if cmd.viewer and not state.viewer_up:
        actions.append(("viewer_up",))
    if cmd.viewer is False and state.viewer_up:
        actions.append(("viewer_down",))
    return actions


def build_status(state: RemoteState | None, camera_hz: float, pose_ages: dict[str, float | None],
                 busy: bool, error: str | None, group_of: Callable[[str], str | None] | None = None) -> dict:
    objects = {}
    for name, age in pose_ages.items():
        cname = _container(name, group_of)
        cont = state.containers.get(cname) if state else None
        crash = state.crashes.get(cname) if state else None
        objects[name] = {"container": cont, "pose_age_s": age, "crash": crash}
    return {
        "gpu": state.gpu if state else None,
        "camera_up": bool(state.camera_up) if state else None,
        "camera_hz": round(float(camera_hz), 2),
        "objects": objects,
        "viewer": bool(state.viewer_up) if state else None,
        "busy": bool(busy),
        "error": error,
    }


LOCAL_HOSTS = ("local", "localhost")


def shell_argv(host: str, command: str) -> list[str]:
    """인지 PC 에서 명령을 돌릴 argv — local(10.01 arm4090: 로봇 PC 가 인지도 돌린다)이면 이 PC 의 bash, 아니면 ssh.
    둘 다 홈에서 시작한다고 보고 부른다(스크립트 경로가 홈 기준 상대 — PC 마다 사용자 이름이 다르다)."""
    if host in LOCAL_HOSTS:
        return ["bash", "-c", command]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command]

