"""연결 그림의 **자동 생성** — 미션의 명령(argv) + 계약 json + robot yaml → 상자와 전선. 순수(파일 읽기만).

정책을 바꾼다는 것은 미션이 다른 계약·다른 launch 를 가리킨다는 뜻이다. 그림은 그 셋을 읽어 만든다 —
프로파일에 손으로 적지 않으므로 정책이 바뀌면 노드와 연결이 따라 바뀐다.

무엇을 어디서 읽는가 (콘솔이 새로 지어내는 이름은 없다):
  · 어떤 노드가 뜨는가      ← 미션 명령이 부르는 launch 파일 · 스크립트 (`policy_chain` · `pour_chain` · `pd_controller` …)
  · 노드가 몇 개인가        ← 계약의 `control_only` · `sides` (launch 의 `chain_nodes` 와 같은 규칙)
  · 센서 토픽               ← robot yaml `sources.<역할>[_<팔>].topic`
  · 구동 토픽               ← robot yaml `groups.<이름>` + `pd_backends.forward_topic`
  · 컵 토픽(pour)           ← 미션 명령의 `src_cup_topic:=` / `rcv_cup_topic:=`
  · 인지(FPP) 토픽          ← `scripts/object_registry.py` 의 이름 규칙
  · 체인 내부 토픽          ← `/policy_control/*` — 노드 소스의 고정 문자열 (`TOPIC` 아래 표)

신호는 왼쪽에서 오른쪽으로만 그린다. 되먹임(joint_target → obs 의 decoder_target, action → obs)은 그리지 않는다.
모르는 명령은 버리지 않고 전선 없는 상자로 남긴다 — 그림에서 조용히 사라지는 것이 없다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from . import _paths  # noqa: F401 — scripts/ · policy_control/ 를 올린다
from .diagram_spec import Diagram, parse_diagram
from .units import UnitCmd

from object_registry import INPUT_NS, OUTPUT_NS, input_topic  # noqa: E402
from policy_control.pd_backends import FORWARD_KINDS, forward_topic  # noqa: E402
from policy_control.sources import split_role  # noqa: E402

NS = "/policy_control"
TOPIC = {"obs": f"{NS}/obs", "action": f"{NS}/action", "target": f"{NS}/joint_target", "episode": f"{NS}/episode",
         "palm_cmd": f"{NS}/palm_cmd", "hand_cmd": f"{NS}/hand_cmd", "fill": f"{NS}/pour/fill_level"}
#: `object_registry.render_fpp_yaml` 이 FPP 에 넘기는 카메라 토픽 — 테스트가 그 출력과 맞는지 잠근다
CAMERA_TOPICS = ("/camera/camera/color/image_raw", "/camera/camera/aligned_depth_to_color/image_raw",
                 "/camera/camera/color/camera_info")
SIDE_ORDER = ("right", "left")                       # launch 의 양팔 규약: 우 먼저
SIDE_KO = {"right": "오른", "left": "왼"}
STALE_MS = 500.0
#: `fake_plant.launch.py` 의 OBJECT_TOPIC — launch 모듈은 콘솔 venv 에서 import 되지 않아 글자로 둔다(테스트가 잠근다)
FAKE_PLANT_OBJECT_TOPIC = "/objects/cup_big_s100/pose"
#: 열(왼쪽부터). 쓰인 것만 남겨 0,1,2… 로 다시 매긴다
L_CAMERA, L_TRACK, L_SENSE, L_OBS, L_POLICY, L_FABRIC, L_PD, L_DRIVE = range(8)
L_UNKNOWN = 99                                   # 모르는 명령은 체인 사이에 끼우지 않고 맨 오른쪽에 모은다
L_LAUNCH = -1                                    # 인지 런처는 카메라보다 왼쪽 — 그 PC 를 켜는 것이 먼저다
#: 카메라와 FP++ 컨테이너가 도는 PC. 런처가 tailscale ssh 로 `scripts/vision/*.sh` 를 부른다.
VISION_HOST = "vision-3090"
PERCEPTION_STATUS = "/perception/status"
HEAD_TOPIC = "/head/joint_states"                # scripts/nodes/head_joint_publisher.py 의 DEFAULT_TOPIC
_POUR_INPUT = {"arm": "arm", "ee": "hand", "tip_force": "force"}       # robot yaml 역할 → pour_node 가 status 로 부르는 이름


@dataclass(frozen=True)
class Cmd:
    name: str                       # launch 파일 · 스크립트의 파일 이름
    args: Mapping[str, str]


def parse_cmd(argv: Sequence[str]) -> Cmd:
    """`ros2 launch [<pkg>] <file> k:=v …` 또는 `python3 <script> --flag v -p k:=v …`."""
    tokens = [str(a) for a in argv]
    args = dict(t.split(":=", 1) for t in tokens if ":=" in t)
    if tokens[:2] == ["ros2", "launch"]:
        name = next((Path(t).name for t in tokens[2:] if t.endswith(".launch.py")), "")
        return Cmd(name=name, args=args)
    name = next((Path(t).name for t in tokens if t.endswith(".py")), Path(tokens[0]).name if tokens else "")
    flags = {a[2:]: b for a, b in zip(tokens, tokens[1:]) if a.startswith("--") and not b.startswith("-")}
    return Cmd(name=name, args={**flags, **args})


@dataclass
class _Graph:
    """만드는 동안만 쓰는 장부 — 끝나면 불변 `Diagram` 으로 굳힌다."""

    status_nodes: tuple[str, ...]
    boxes: dict[str, dict] = field(default_factory=dict)
    wires: list[dict] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)          # 토픽 → 그것을 내는 상자
    lazy: dict[str, dict] = field(default_factory=dict)              # 토픽 → 누가 받을 때만 그리는 상자(fake 플랜트의 컵)
    needs: list[tuple] = field(default_factory=list)                 # (topic, role, side, dst, inputs, stale_ms)

    def box(self, box_id: str, title: str, layer: int, **extra) -> str:
        status = extra.pop("status", None)
        entry = {"id": box_id, "title": title, "col": layer, **{k: v for k, v in extra.items() if v not in (None, "", (), [])}}
        if status in self.status_nodes:                              # 브리지가 듣지 않는 status 는 주장하지 않는다
            entry["status"] = status
        old, unit = self.boxes.get(box_id), entry.get("unit")
        if old is not None and unit is not None and old.get("unit") not in (None, unit):
            box_id = f"{box_id}@{unit}"                              # 같은 노드를 두 번 띄우는 미션(좌·우 따로) — 스위치도 둘이어야 한다
            entry["id"] = box_id
        self.boxes.setdefault(box_id, entry)
        return box_id

    def wire(self, src: str, dst: str, topic: str, **extra) -> None:
        self.wires.append({"from": src, "to": dst, "topic": topic, **{k: v for k, v in extra.items() if v not in (None, "", (), [])}})

    def need(self, topic: str, role: str, side: str, dst: str, inputs: Sequence[str], stale_ms: float) -> None:
        self.needs.append((topic, role, side, dst, tuple(inputs), stale_ms))

    def mark(self) -> tuple:
        """명령 하나를 그리기 직전의 장부. 그리다 실패하면 여기로 되돌린다 — 반쪽짜리 체인을 남기지 않는다."""
        return ({k: dict(v) for k, v in self.boxes.items()}, [dict(w) for w in self.wires],
                dict(self.providers), {k: dict(v) for k, v in self.lazy.items()}, list(self.needs))

    def rollback(self, mark: tuple) -> None:
        self.boxes, self.wires, self.providers, self.lazy, self.needs = mark


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str, repo: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else repo / p


def _robot(value: str, repo: Path) -> dict:
    """launch 의 `resolve_robot` 과 같은 규칙: 이름이면 config/robots/<이름>.yaml, 아니면 경로."""
    path = (_paths.POLICY_CONTROL / "config" / "robots" / f"{value}.yaml"
            if "/" not in value and not value.endswith(".yaml") else _resolve(value, repo))
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sources(robot: Mapping, side: str | None) -> list[tuple[str, str, Mapping]]:
    """(역할, 팔, 설정). `side` 를 주면 그 팔 것과 팔 무관 소스만 — `sources.select_side` 와 같은 뜻."""
    out = []
    for name, cfg in (robot.get("sources") or {}).items():
        role, own = split_role(str(name))
        if side is None or own in ("", side):
            out.append((role, own, cfg))
    return out


def _robot_sides(robot: Mapping) -> tuple[str, ...]:
    found = {own for _, own, _ in _sources(robot, None) if own} or \
            {str(g.get("side")) for g in (robot.get("groups") or {}).values() if g.get("side")}
    return tuple(s for s in SIDE_ORDER if s in found)


def _stale_ms(cfg: Mapping) -> float:
    return float(cfg.get("stale_sec", STALE_MS / 1e3)) * 1e3


# ── 체인 ────────────────────────────────────────────────────────────────
def _policy_chain(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    contract, robot = _read_json(_resolve(cmd.args["contract"], repo)), _robot(cmd.args["robot"], repo)
    sides_all = [s for s in SIDE_ORDER if s in (contract.get("sides") or {})]
    asked = cmd.args.get("side", "").strip().lower()
    sides = sides_all if asked in ("both", "all") else [asked or str(contract.get("primary_side") or (sides_all or [""])[0])]
    policy = contract.get("policy") or {}
    fabric_ids = ["fabric"] if len(sides) == 1 else [f"fabric_{s}" for s in sides]
    if contract.get("control_only"):
        master = g.box("episode_master", "episode_master · 에피소드", L_OBS, status="episode_master", ros=["/episode_master"],
                       unit=key, note="정책 없는 계약 — fabric 은 direct 모드로 palm_cmd 를 받는다")
        g.box("operator", "운영자 입력", L_SENSE, note="palm_cmd · hand_cmd 도구가 필요할 때 낸다")
    else:
        master = g.box("obs", "obs_node · 관측", L_OBS, status="obs", ros=["/obs_node"], unit=key,
                       stages=[f"센서 → obs {policy.get('obs_dim', '?')}"])
        g.box("policy", "policy_node · 정책", L_POLICY, status="policy", ros=["/policy_node"], unit=key,
              stages=[f"obs {policy.get('obs_dim', '?')} → action {policy.get('action_dim', '?')}"])
        g.wire("obs", "policy", TOPIC["obs"], stale_ms=STALE_MS, episodic=True)
        g.wire("obs", "policy", TOPIC["episode"], meter=False)
        for role, own, cfg in _sources(robot, sides[0]):
            if role != "decoder_target":                                  # 되먹임 — 그리지 않는다
                g.need(str(cfg["topic"]), role, own or sides[0], "obs", [role], _stale_ms(cfg))
    for side, fid in zip(sides, fabric_ids):
        node = "/fabric_node" if len(sides) == 1 else f"/fabric_node_{side}"
        g.box(fid, f"fabric_node · 역기구학 ({SIDE_KO.get(side, side)}팔)", L_FABRIC, status="fabric", ros=[node], unit=key,
              stages=["decoder", "fabric IK"])
        g.wire(master, fid, TOPIC["episode"], meter=False)
        if contract.get("control_only"):
            g.wire("operator", fid, TOPIC["palm_cmd"], on_demand=True)
            g.wire("operator", fid, TOPIC["hand_cmd"], on_demand=True)
        else:
            g.wire("obs", fid, TOPIC["obs"], stale_ms=STALE_MS, episodic=True)
            g.wire("policy", fid, TOPIC["action"], stale_ms=STALE_MS, episodic=True)
        for role, own, cfg in _sources(robot, side):
            if role in ("arm", "ee", "object"):                           # fabric_node.SOURCE_ROLES
                g.need(str(cfg["topic"]), role, own or side, fid, [], _stale_ms(cfg))


def _pour_chain(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    contract, robot = _read_json(_resolve(cmd.args["contract"], repo)), _robot(cmd.args["robot"], repo)
    fabric = "fabric IK ×2" + (" (꺼짐)" if cmd.args.get("use_fabric", "true").lower() == "false" else "")
    g.box("pour_node", "pour_node · 정책 체인", L_OBS, status="pour_node", ros=["/pour_node"], unit=key,
          stages=["입력함", f"obs {contract.get('obs_dim', '?')}", f"policy → {contract.get('action_dim', '?')}", "decoder", fabric])
    g.box("operator", "운영자 입력 · fill_level", L_SENSE, note="에피소드 전에 한 번 넣는다 (실기에서는 사람이 넣는 값)")
    g.wire("operator", "pour_node", TOPIC["fill"], inputs=["fill"], on_demand=True)
    for entry in contract.get("sides") or ():
        role_name, side = str(entry["role"]), str(entry["side"])
        for role, own, cfg in _sources(robot, side):
            if role in _POUR_INPUT:
                g.need(str(cfg["topic"]), role, own or side, "pour_node", [f"{role_name}:{_POUR_INPUT[role]}"], _stale_ms(cfg))
        topic = cmd.args.get(f"{role_name}_cup_topic")
        if topic:
            g.need(topic, "object", "", "pour_node", [f"{role_name}:cup"], STALE_MS)


def _pour_guard(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    g.box("pour_guard", "pour_guard · 안전 가드", L_POLICY, status="pour_guard", ros=["/pour_guard"], unit=key,
          note="위반이면 episode/abort 를 부른다")
    for name in ("src_cup_topic", "rcv_cup_topic"):
        if cmd.args.get(name):
            g.need(cmd.args[name], "object", "", "pour_guard", [], STALE_MS)


def _pd(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    contract, robot = _read_json(_resolve(cmd.args["contract"], repo)), _robot(cmd.args["robot"], repo)
    asked = [s.strip() for s in cmd.args.get("sides", "").split(",") if s.strip()]
    if asked in (["both"], ["all"]):
        asked = list(SIDE_ORDER)
    sides = [s for s in SIDE_ORDER if s in (asked or _robot_sides(robot)) and s in (contract.get("sides") or {})]
    execute = cmd.args.get("execute", "false").lower() in ("true", "1", "yes")
    muted = "" if execute else "pd 가 execute:=false 로 떠 있다 — 구동 토픽을 내지 않는다(무발행)"
    pd = g.box("pd", "pd_node · PD 제어", L_PD, status="pd", ros=["/pd_node"], unit=key,
               stages=["PD 법칙", "컨트롤러 교대", "발행" if execute else "무발행"])
    drives = 0
    for side in sides:
        for role, own, cfg in _sources(robot, side):
            if role in ("arm", "ee"):                                     # ArmUnit.joint_topics
                g.need(str(cfg["topic"]), role, own or side, pd, [f"{side}:{role}"], _stale_ms(cfg))
        for name in (contract["sides"][side].get("pd_groups") or ()):
            group = (robot.get("groups") or {}).get(name)
            if group is not None:
                _drive(g, pd, side, group, muted)
                drives += 1
    if not drives:                                                        # 실기 pd_node 는 이 상태로 뜨지 않는다 — 그림도 그렇게 말한다
        g.boxes[pd]["note"] = "계약의 pd_groups 가 robot yaml 의 groups 에 없다 — pd_node 는 이대로면 기동하지 않는다"


def _drive(g: _Graph, pd: str, side: str, group: Mapping, muted: str) -> None:
    backend = str(group.get("backend"))
    if backend == "arm_forward":
        g.box("arm_drive", "팔 구동 (forward 컨트롤러)", L_DRIVE, manager="/controller_manager")
        for kind in FORWARD_KINDS:
            topic = forward_topic(side, kind)
            g.wire(pd, "arm_drive", topic, meter=kind == "position", stale_ms=STALE_MS if kind == "position" else None,
                   muted=muted,
                   heard_by=[topic.rsplit("/", 1)[0]])                     # ros2_control: 컨트롤러는 제 이름의 노드로 구독한다
    elif group.get("topic"):
        gripper = backend == "jtc_single_point"                          # 그리퍼는 팔 bringup 의 controller_manager 아래에 있다
        title = "그리퍼 구동" if gripper else f"{SIDE_KO.get(side, side)}손 구동 (JTC)"
        topic = str(group["topic"])
        ns = str(group.get("namespace") or "").strip("/")
        manager = "/controller_manager" if gripper else f"/{ns}/controller_manager" if ns else None
        g.box(f"hand_{side}_drive", title, L_DRIVE, manager=manager)
        if gripper:
            g.providers.setdefault(f"__hand_unit_{side}__", g.providers.get("__arm_unit__"))
        g.wire(pd, f"hand_{side}_drive", topic, meter=False, muted=muted, heard_by=[topic.rsplit("/", 1)[0]])


# ── 내는 쪽 ─────────────────────────────────────────────────────────────
def _fake_plant(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    robot = _robot(cmd.args["robot"], repo) if cmd.args.get("robot") else {}
    for role, own, cfg in _sources(robot, None):
        topic = str(cfg["topic"])
        if role == "arm":
            g.providers.setdefault(topic, g.box("arm_state", "팔 상태 (MockArm)", L_SENSE, ros=["/fake_arm_bridge"], unit=key))
        elif role in ("ee", "tip_force") and own:
            g.providers.setdefault(topic, g.box(f"hand_{own}_state", f"{SIDE_KO[own]}손 · 관절 + 손끝 힘 (fake)", L_SENSE, unit=key))
    g.lazy[FAKE_PLANT_OBJECT_TOPIC] = {"box_id": _object_id(FAKE_PLANT_OBJECT_TOPIC), "layer": L_SENSE, "unit": key,
                                       "title": f"{_object_name(FAKE_PLANT_OBJECT_TOPIC)} 포즈 (fake)"}
    g.providers["__fake_plant__"] = key


def _fake_cup(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    topic = cmd.args.get("topic")
    if topic:
        g.providers[topic] = g.box(_object_id(topic), f"{_object_name(topic)} 포즈 (fake)", L_SENSE, unit=key)


def _bringup(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    g.providers["__arm_unit__"] = key


def _hand_driver(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    side = next((s for s in SIDE_ORDER if f"_{s}_" in cmd.name), "")
    if side:
        g.providers[f"__hand_unit_{side}__"] = key


def _object_name(topic: str) -> str:
    parts = topic.strip("/").split("/")
    return parts[1] if len(parts) == 3 and f"/{parts[0]}" == OUTPUT_NS else topic


def _object_id(topic: str) -> str:
    return "obj_" + "".join(c if c.isalnum() else "_" for c in _object_name(topic)).strip("_")


def _provider(g: _Graph, topic: str, role: str, side: str) -> str:
    """이 토픽을 내는 상자 — 미션이 띄우는 것이 없으면 바깥(robot_control · 인지)의 상자를 만든다."""
    if topic in g.providers:
        return g.providers[topic]
    if topic in g.lazy:
        spec = g.lazy[topic]
        return g.providers.setdefault(topic, g.box(spec["box_id"], spec["title"], spec["layer"], unit=spec["unit"]))
    if role == "object":
        return g.providers.setdefault(topic, _perception(g, topic))
    if role == "head":
        return g.providers.setdefault(topic, g.box("head", "목 관절 (선택 입력)", L_SENSE, ros=["/head_joint_publisher"],
                                                   note="끊겨도 체인은 돈다"))
    if role in ("ee", "tip_force") and side:
        box = g.box(f"hand_{side}_state", f"{SIDE_KO[side]}손 · 관절 + 손끝 힘", L_SENSE, unit=g.providers.get(f"__hand_unit_{side}__"))
        return g.providers.setdefault(topic, box)
    box = g.box("arm_state", "팔 상태 (robot_control)", L_SENSE, unit=g.providers.get("__arm_unit__"),
                note="joint_state_broadcaster — robot_control bringup 이 띄운다")
    return g.providers.setdefault(topic, box)


def _perception(g: _Graph, topic: str) -> str:
    """카메라 → FPP 추적기 → object_pose_node. 앞의 둘은 **vision-3090** 에서 돈다 — 켜는 것은 인지 런처다."""
    name = _object_name(topic)
    if name == topic:                                                    # /objects/<이름>/pose 꼴이 아니다 — 아는 것만 그린다
        return g.box(_object_id(topic), f"{topic} (바깥)", L_SENSE)
    g.box("camera", "카메라 (RealSense)", L_CAMERA, host=VISION_HOST, note="영상은 세지 않는다(연결만 본다)")
    tracker = g.box(f"fpp_{name}", f"FPP 추적 · {name}", L_TRACK, host=VISION_HOST, note=f"docker fpp_{name} ({INPUT_NS})")
    for cam in CAMERA_TOPICS:
        g.wire("camera", tracker, cam, meter=False)
    g.box("object_pose", "object_pose_node · 카메라 → base_link", L_SENSE, ros=["/object_pose_node"])
    g.wire(tracker, "object_pose", input_topic(name), stale_ms=STALE_MS)
    return "object_pose"


def _perception_launcher(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """인지 런처 — 이 PC 에서 돌지만 하는 일은 **vision-3090** 의 카메라·FP++ 컨테이너를 ssh 로 켜고 끄는 것이다.

    그림에서 인지 사슬의 유일한 스위치다(카메라·컨테이너 자체는 저 PC 에 있어 콘솔이 직접 못 켠다).
    """
    host = cmd.args.get("host") or VISION_HOST
    g.box("perception", f"인지 런처 · {host}", L_LAUNCH, ros=["/perception_launcher"], unit=key, host=host,
          note=f"{PERCEPTION_STATUS} 로 저 PC 의 카메라·컨테이너 상태를 말한다 (ssh 로 켜고 끈다)")


_HANDLERS = {"policy_chain.launch.py": _policy_chain, "pour_chain.launch.py": _pour_chain, "pour_guard_node.py": _pour_guard,
             "pd_controller.launch.py": _pd, "perception_launcher_node.py": _perception_launcher}
def _head_publisher(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """목 상태 퍼블리셔(읽기 전용) — 받는 노드가 없어도 상자로 둔다. 콘솔에서 head 까지 관리한다."""
    topic = cmd.args.get("topic") or HEAD_TOPIC
    box = g.box("head", "목 상태 (head)", L_SENSE, ros=["/head_joint_publisher"], unit=key,
                note="읽기 전용 — 토크·게인·목표를 건드리지 않는다")
    g.providers.setdefault(topic, box)


_PROVIDERS = {"fake_plant.launch.py": _fake_plant, "fake_cup_pose_pub.py": _fake_cup, "openarm.bimanual.launch.py": _bringup,
              "dg5f_right_driver.launch.py": _hand_driver, "dg5f_left_driver.launch.py": _hand_driver,
              "head_joint_publisher.py": _head_publisher}


def _shell_step(u: UnitCmd) -> bool:
    """전원 확인·sudo CAN·NIC 설정 같은 수동 셸 단계 — 노드가 아니라 사람이 하는 일이다.
    미션 패널이 메모와 함께 보여 주므로 그림에는 그리지 않는다(모르는 ROS·python 명령은 계속 상자로 남긴다)."""
    return u.kind == "manual" and bool(u.argv) and Path(str(u.argv[0])).name in ("bash", "sh")


def _unknown(g: _Graph, key: str, cmd: Cmd, why: str = "", note: str = "") -> None:
    """그림이 모르는 명령 — 버리지 않는다. 이름은 미션이 붙인 설명을 쓴다("bash" 는 아무 말도 하지 않는다)."""
    safe = "".join(c if c.isalnum() else "_" for c in key)
    said = (note or "").strip().lstrip("★").strip()
    hint = why or "그림이 아직 모르는 명령 — 연결은 아래 '그림 밖' 목록에서 본다"
    g.box(f"unit_{safe}", cmd.name or key, L_UNKNOWN, unit=key,
          note=f"{said} · {hint}" if said and said != key else hint)


def _finish(g: _Graph) -> None:
    """모은 입력을 내는 상자에 잇고, 체인이 내는 목표를 pd·가짜 손에 잇는다."""
    merged: dict[tuple[str, str], dict] = {}
    for topic, role, side, dst, inputs, stale in g.needs:
        src = _provider(g, topic, role, side)
        entry = merged.setdefault((topic, dst), {"src": src, "inputs": [], "stale": stale})
        entry["inputs"] += [i for i in inputs if i not in entry["inputs"]]
    for (topic, dst), e in merged.items():
        g.wire(e["src"], dst, topic, inputs=e["inputs"], stale_ms=e["stale"])
    makers = [b for b in g.boxes if b == "pour_node" or b.startswith("fabric")]
    master = next((b for b in ("pour_node", "obs", "episode_master") if b in g.boxes), None)
    for pd in [b for b in g.boxes if b == "pd" or b.startswith("pd@")]:   # 미션이 pd 를 둘 띄우면 둘 다 잇는다
        for maker in makers:
            g.wire(maker, pd, TOPIC["target"], stale_ms=STALE_MS, episodic=True)
        if master:
            g.wire(master, pd, TOPIC["episode"], meter=False)
    if master == "pour_node" and "pour_guard" in g.boxes:
        g.wire("pour_node", "pour_guard", TOPIC["episode"], meter=False)
    plant = g.providers.get("__fake_plant__")
    if plant:                                                            # fake 손은 joint_target 을 반사한다
        for box_id, box in list(g.boxes.items()):
            if box_id == "arm_drive":
                box.update(unit=plant, ros=["/fake_arm_bridge"])
                for w in g.wires:
                    if w["to"] == "arm_drive":
                        w["heard_by"] = ["/fake_arm_bridge"]
            elif box_id.startswith("hand_") and box_id.endswith("_drive"):
                # `/fake_hand_state_pub` 은 좌·우가 같은 이름이다 — 한쪽이 죽어도 이름이 남는다.
                # 같은 프로세스가 만드는 `/dg5f_<side>/dg5f_<side>_controller` 만 그 손을 가리킨다.
                node = next((w["topic"].rsplit("/", 1)[0] for w in g.wires
                             if w["to"] == box_id and w["topic"].endswith("joint_trajectory")), None)
                box.update(unit=plant, note="fake 손은 joint_target 을 반사한다 — JTC 토픽은 받지 않는다")
                box.pop("manager", None)                                 # fake 에는 손 controller_manager 가 없다
                box["ros"] = [node] if node else []
                for w in g.wires:
                    if w["to"] == box_id:
                        w.pop("heard_by", None)
                        w.setdefault("muted", "fake 손은 JTC 를 받지 않는다 — joint_target 을 그대로 반사한다")
                for maker in makers:
                    g.wire(maker, box_id, TOPIC["target"], stale_ms=STALE_MS, episodic=True)
    else:
        for box_id, box in g.boxes.items():
            if box_id == "arm_drive":
                box.setdefault("unit", g.providers.get("__arm_unit__"))
            elif box_id.startswith("hand_") and box_id.endswith("_drive"):
                box.setdefault("unit", g.providers.get(f"__hand_unit_{box_id.split('_')[1]}__"))
    for box in g.boxes.values():
        if box.get("unit") is None:
            box.pop("unit", None)


def generate(units: Mapping[str, UnitCmd], *, repo: Path, status_nodes: Sequence[str]) -> Diagram:
    """미션의 단위(배경·수동 명령) 전부 → 그림. 같은 입력이면 같은 그림이다."""
    g = _Graph(status_nodes=tuple(status_nodes))
    parsed = [(key, parse_cmd(u.argv)) for key, u in units.items()]
    for table in (_PROVIDERS, _HANDLERS):                                # 내는 쪽을 먼저 — 받는 쪽이 누가 내는지 알아야 한다
        for key, cmd in parsed:
            if cmd.name in table:
                mark = g.mark()
                try:
                    table[cmd.name](g, key, cmd, repo)
                except Exception as exc:                             # noqa: BLE001 — 그림 하나 때문에 콘솔이 안 뜨면 안 된다
                    g.rollback(mark)                                 # 반쪽만 그린 체인이 "다 이어졌다" 로 보이면 안 된다
                    _unknown(g, key, cmd, f"읽지 못했다: {exc}", units[key].note)
    known = set(_PROVIDERS) | set(_HANDLERS)
    for key, cmd in parsed:
        if cmd.name not in known and not _shell_step(units[key]):
            _unknown(g, key, cmd, note=units[key].note)
    _finish(g)
    used = sorted({b["col"] for b in g.boxes.values()})
    boxes = [{**b, "col": used.index(b["col"])} for b in sorted(g.boxes.values(), key=lambda b: b["col"])]
    col = {b["id"]: b["col"] for b in boxes}
    wires = [w for w in g.wires if col[w["from"]] < col[w["to"]]]
    return parse_diagram({"boxes": boxes, "wires": wires}, path=Path("<generated>"), status_nodes=status_nodes)
