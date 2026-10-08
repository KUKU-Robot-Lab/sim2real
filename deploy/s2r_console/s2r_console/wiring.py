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
  · RH56F1 · 정책 노드 이름 ← 각 노드 소스의 고정 문자열 (아래 상수 — 테스트가 소스와 맞는지 잠근다)

상자 하나 = ROS 노드 하나. 같은 노드를 여러 명령이 띄우면(pd 무발행 → 발행, 단독 정책 단계 → 에피소드) 상자는 하나이고
명령마다 스위치가 붙는다 — 이름이 같은 노드는 동시에 둘일 수 없다. 한 명령만 쓰는 전선에는 그 명령을 적는다(`units`).
신호는 왼쪽에서 오른쪽으로만 그린다. 되먹임(joint_target → obs 의 decoder_target, action → obs)은 그리지 않는다.
모르는 명령은 버리지 않고 전선 없는 상자로 남긴다 — 그림에서 조용히 사라지는 것이 없다.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from . import _paths  # noqa: F401 — scripts/ · policy_control/ 를 올린다
from .diagram import _short as short_topic
from .diagram_spec import Diagram, parse_diagram
from .units import UnitCmd

from object_registry import INPUT_NS, OUTPUT_NS, input_topic, output_topic  # noqa: E402
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

# ── 노드 소스의 고정 이름 (테스트가 소스 문자열과 맞는지 잠근다) ─────────────────
#: RH56F1 손 드라이버 — 손 하나에 노드 하나(EtherCAT · fake). angle_set 을 받는다
HAND_DRIVER_NODE = {"rh56f1_driver.py": "/rh56f1_ecat_{side}", "fake_rh56f1_hand.py": "/fake_rh56f1_{side}"}
#: rh56f1_state_node — 드라이버 레지스터 → rad · N · g
HAND_STATE_NODE = "/rh56f1_state_{side}"
HAND_STATE_TOPICS = ("/hand_{side}/joint_states", "/hand_{side}/tip_forces", "/hand_{side}/joint_forces")
FPP_RX_NODE = "/fpp_pose_rx"                     # scripts/nodes/fpp_pose_rx.py — FP++ 자세를 UDP 로 받아 ROS 로
OBJECT_POSE_NODE = "/object_pose_node"
HOLDER_NODE = "/cup_holder_pose_node"
HOLDER_CFG = "config/cup_holders.yaml"           # scripts/calib/cup_holder_pose.py 의 DEFAULT_CFG
HOLDER_TOPIC = "/objects/cup_holder_{}/pose"     # rh_place_node · episode_ros 가 쓰는 홀더 자세
RUNNER_NODE = "/episode_runner"                  # episode_runner_node.NAME
#: 센서를 받는 폴링 노드(raw_poll.make_poll_node) — 구독은 본 노드가 아니라 이 이름으로 그래프에 보인다(10.04 CPU)
PD_POLL_NODE = "/pd_node_poll_{sides}"           # pd_node: f"{NODE_NAME}_poll_{'_'.join(self.sides)}"
POLICY_POLL_NODE = "/{node}_poll"                # pour_fj_node 계열: f"{self.node_name}_poll"
OBJECT_RELAY = "/episode/objects/{}/pose"        # episode_ros.OBJECT_RELAY — snapshot 으로 기록한 정지 자세를 다시 낸다
#: pour_fj_node 계열 정책 노드 — 스크립트 → (기본 노드 이름, 하는 일, 컵 파라미터와 기본 토픽). pour_fj_node.FAMILIES 와 같다
POLICY_NODES = {
    "rh_aglt_node.py": ("rh_aglt_node", "집기 정책", (("cup_topic", "/objects/aglt_cup_s065/pose"),)),
    "rh_place_node.py": ("rh_place_node", "놓기 정책", (("cup_topic", "/objects/cyl60/pose"),)),
    "pour_fj_node.py": ("pour_fj_node", "붓기 정책", (("cup_src_topic", "/objects/cup_src/pose"),
                                                    ("cup_rcv_topic", "/objects/cup_rcv/pose"))),
}
_POLICY_INPUTS = ("arm", "ee", "tip_force", "joint_force")     # pour_fj_node 가 robot yaml 에서 폴링하는 소스 역할
#: 실기 상자 제목 — fake 대역도 같은 상자 · 같은 제목으로 그린다(10.05 사용자: 연결창은 실기 기준, fake 노드는 따로 안 그린다)
ARM_STATE_TITLE = "팔 상태 (robot_control)"
OBJECT_POSE_TITLE = "object_pose_node · 카메라 → base_link"
HOLDER_TITLE = "cup_holder_pose_node · 홀더 자세(마커)"
#: 노드가 아니라 사람이 한 번 내는 도구(수동 명령) — 연결 상태가 아니므로 그리지 않는다. 조작판에는 그대로 있다
_TOOLS = ("aglt_goal.py",
          # ★10.08 프로세스별 CPU 기록 — ROS 노드가 아니다(/proc 만 읽는다). 그림에 상자를 두지 않는다
          "proc_cpu_record.py")


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


def _merged_title(old: str, new: str) -> str:
    """같은 노드를 띄우는 명령마다 제목의 괄호 안이 다르면(오른팔 · 왼팔) 하나로 모은다."""
    head, _, rest = old.partition(" (")
    nhead, _, nrest = new.partition(" (")
    if new == old or head != nhead or not rest.endswith(")") or not nrest.endswith(")"):
        return old
    parts = rest[:-1].split(" | ")
    return old if nrest[:-1] in parts else f"{head} ({' | '.join([*parts, nrest[:-1]])})"


def _merge_wire(old: dict, new: dict) -> None:
    """같은 (내는 쪽, 받는 쪽, 토픽) 전선을 둘 그리지 않는다 — 명령이 여럿이면 속성을 합친다."""
    if "units" in old and "units" in new:
        old["units"] = list(dict.fromkeys([*old["units"], *new["units"]]))
    else:
        old.pop("units", None)                                           # 한쪽이라도 '언제나' 면 언제나
    muted = [w for w in (old, new) if w.get("muted")]
    if muted:
        old["muted"] = muted[0]["muted"]
        if len(muted) == 2 and all(w.get("muted_by") for w in muted) or len(muted) == 1 and muted[0].get("muted_by"):
            old["muted_by"] = list(dict.fromkeys(u for w in muted for u in w.get("muted_by") or ()))
        else:
            old.pop("muted_by", None)                                    # 명령을 가리지 않는 선언이 섞였다 — 언제나
    for key in ("inputs", "heard_by"):
        merged = list(dict.fromkeys([*(old.get(key) or ()), *(new.get(key) or ())]))
        if merged:
            old[key] = merged
    if new.get("meter", True) and not old.get("meter", True):
        old["meter"] = True
    if new.get("episodic"):
        old["episodic"] = True
    if not new.get("on_demand"):
        old.pop("on_demand", None)
    if "stale_ms" in new:
        old["stale_ms"] = min(old.get("stale_ms", new["stale_ms"]), new["stale_ms"])


@dataclass
class _Graph:
    """만드는 동안만 쓰는 장부 — 끝나면 불변 `Diagram` 으로 굳힌다."""

    status_nodes: tuple[str, ...]
    boxes: dict[str, dict] = field(default_factory=dict)
    wires: list[dict] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)          # 토픽 → 그것을 내는 상자
    #: (단계, 토픽) → 그 단계 안에서 그것을 내는 상자. 에피소드 단계의 홀더 · 목표는 같은 단계의 실행기가 낸다
    stage_providers: dict[tuple[str, str], str] = field(default_factory=dict)
    lazy: dict[str, dict] = field(default_factory=dict)              # 토픽 → 누가 받을 때만 그리는 상자(fake 플랜트의 컵)
    needs: list[tuple] = field(default_factory=list)                 # (topic, role, side, dst, inputs, stale_ms, unit, how)
    makers: list[tuple] = field(default_factory=list)                # (정책 상자, 팔들, 명령, episode 토픽) → 그 팔 pd
    hand_drivers: dict[str, tuple[str, str]] = field(default_factory=dict)   # 팔 → (명령, 드라이버 노드)
    percept_host: str = ""                                           # 인지 런처가 카메라 · FP++ 를 켜는 PC
    #: 상자 → 짧은 이름. 한 상자가 같은 토픽을 둘에게서 받을 때 포트에 내는 쪽을 앞에 붙인다(좁은 상자에서도 안 잘린다)
    tags: dict[str, str] = field(default_factory=dict)
    #: 프로파일 status_nodes 밖인데 상자가 제 status 를 읽게 하는 노드(정책 · 실행기) — 콘솔 브리지가 같이 구독한다
    own_status: set[str] = field(default_factory=set)

    def box(self, box_id: str, title: str, layer: int, **extra) -> str:
        """상자 하나 = 노드 하나. 같은 노드(이름이 같은 상자 · 같은 ROS 이름)를 또 띄우는 명령은 그 상자에 스위치를 더한다."""
        status, unit, own = extra.pop("status", None), extra.pop("unit", None), extra.pop("own_status", False)
        entry = {"id": box_id, "title": title, "col": layer, **{k: v for k, v in extra.items() if v not in (None, "", (), [])}}
        if status in self.status_nodes or (own and status):          # 브리지가 듣지 않는 status 는 주장하지 않는다
            entry["status"] = status
            if own and status not in self.status_nodes:
                self.own_status.add(status)
        ros = list(entry.get("ros") or ())
        for bid, old in self.boxes.items():
            mine = list(old.get("ros") or ())
            if bid.split("@")[0] != box_id or (ros and mine and ros != mine):
                continue
            if ros and not mine:
                old["ros"] = ros
            if unit and unit not in old.setdefault("units", []):
                old["units"].append(unit)
                old["title"] = _merged_title(old["title"], title)
            return bid
        if box_id in self.boxes:                                     # 이름은 같은데 다른 노드(좌·우 pd 를 따로 띄운다)
            box_id = f"{box_id}@{unit or len(self.boxes)}"
            entry["id"] = box_id
        if unit:
            entry["units"] = [unit]
        self.boxes[box_id] = entry
        return box_id

    def wire(self, src: str, dst: str, topic: str, **extra) -> None:
        new = {"from": src, "to": dst, "topic": topic, **{k: v for k, v in extra.items() if v not in (None, "", (), [])}}
        old = next((w for w in self.wires if (w["from"], w["to"], w["topic"]) == (src, dst, topic)), None)
        if old is None:
            self.wires.append(new)
        else:
            _merge_wire(old, new)

    def need(self, topic: str, role: str, side: str, dst: str, inputs: Sequence[str], stale_ms: float,
             unit: str | None = None, how: str = "meter") -> None:
        """받는 쪽이 이 토픽을 읽는다 — 내는 상자는 다 그린 뒤에 찾는다. `how`: meter(주기를 잰다) · link(연결만) · on_demand."""
        self.needs.append((topic, role, side, dst, tuple(inputs), stale_ms, unit, how))

    def mark(self) -> tuple:
        """명령 하나를 그리기 직전의 장부. 그리다 실패하면 여기로 되돌린다 — 반쪽짜리 체인을 남기지 않는다."""
        return copy.deepcopy((self.boxes, self.wires, self.providers, self.stage_providers, self.lazy, self.needs,
                              self.makers, self.hand_drivers, self.percept_host, self.tags, self.own_status))

    def rollback(self, mark: tuple) -> None:
        (self.boxes, self.wires, self.providers, self.stage_providers, self.lazy, self.needs,
         self.makers, self.hand_drivers, self.percept_host, self.tags, self.own_status) = mark


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


def _side(cmd: Cmd) -> str:
    side = str(cmd.args.get("side", "")).strip().lower()
    if side not in SIDE_ORDER:
        raise ValueError(f"--side 가 right · left 가 아니다: {side!r}")
    return side


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
                g.need(str(cfg["topic"]), role, own or sides[0], "obs", [role], _stale_ms(cfg), unit=key)
    for side, fid in zip(sides, fabric_ids):
        node = "/fabric_node" if len(sides) == 1 else f"/fabric_node_{side}"
        fid = g.box(fid, f"fabric_node · 역기구학 ({SIDE_KO.get(side, side)}팔)", L_FABRIC, status="fabric", ros=[node],
                    unit=key, stages=["decoder", "fabric IK"])
        g.wire(master, fid, TOPIC["episode"], meter=False)
        if contract.get("control_only"):
            g.wire("operator", fid, TOPIC["palm_cmd"], on_demand=True)
            g.wire("operator", fid, TOPIC["hand_cmd"], on_demand=True)
        else:
            g.wire("obs", fid, TOPIC["obs"], stale_ms=STALE_MS, episodic=True)
            g.wire("policy", fid, TOPIC["action"], stale_ms=STALE_MS, episodic=True)
        for role, own, cfg in _sources(robot, side):
            if role in ("arm", "ee", "object"):                           # fabric_node.SOURCE_ROLES
                g.need(str(cfg["topic"]), role, own or side, fid, [], _stale_ms(cfg), unit=key)


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
                g.need(str(cfg["topic"]), role, own or side, "pour_node", [f"{role_name}:{_POUR_INPUT[role]}"], _stale_ms(cfg),
                       unit=key)
        topic = cmd.args.get(f"{role_name}_cup_topic")
        if topic:
            g.need(topic, "object", "", "pour_node", [f"{role_name}:cup"], STALE_MS, unit=key)


def _pour_guard(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    g.box("pour_guard", "pour_guard · 안전 가드", L_POLICY, status="pour_guard", ros=["/pour_guard"], unit=key,
          note="위반이면 episode/abort 를 부른다")
    for name in ("src_cup_topic", "rcv_cup_topic"):
        if cmd.args.get(name):
            g.need(cmd.args[name], "object", "", "pour_guard", [], STALE_MS, unit=key)


def _policy_node(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """pour_fj_node 계열(rh_aglt · rh_place · pour_fj) — 노드 하나가 관측 · 정책 · 디코더를 다 한다.

    입력: 계약의 팔마다 robot yaml 소스(팔 · 손 · 촉각 · 관절 힘) + 컵(파라미터) + 홀더(놓기) + 목표(집기).
    출력: joint_target 과 episode 를 그 팔의 pd 로(`_finish`). 노드 이름은 `-r __node:=…`, 서비스 · 토픽은 `ns`.
    """
    default, what, cups = POLICY_NODES[cmd.name]
    contract, robot = _read_json(_resolve(cmd.args["contract"], repo)), _robot(cmd.args["robot"], repo)
    node = str(cmd.args.get("__node") or default).lstrip("/")
    ns = str(cmd.args.get("ns", "")).strip("/")
    base = f"{NS}/{ns}" if ns else NS
    roles = {str(r): str(v["side"]) for r, v in (contract.get("sides") or {}).items()}
    sides = [s for s in SIDE_ORDER if s in roles.values()]
    arms = " · ".join(f"{SIDE_KO.get(s, s)}팔" for s in sides)
    box = g.box(node, f"{node} · {what} ({arms})", L_POLICY, status=node, ros=[f"/{node}", POLICY_POLL_NODE.format(node=node)],
                unit=key, own_status=True,
                stages=[f"obs {contract.get('obs_dim', '?')}", f"policy → {contract.get('action_dim', '?')}", "decoder"])
    g.tags[box] = what.split()[0]                                         # 집기 · 놓기 · 붓기
    for role, side in roles.items():
        for r, own, cfg in _sources(robot, side):
            if r in _POLICY_INPUTS:
                g.need(str(cfg["topic"]), r, own or side, box, [f"{role}:{r}"], _stale_ms(cfg), unit=key)
    for param, topic in cups:                     # 컵은 reset 때 한 번 붙잡는다(cup_latch) — 주기는 인지 쪽 전선이 잰다
        g.need(str(cmd.args.get(param) or topic), "object", "", box, [], STALE_MS, unit=key, how="link")
    if cmd.name == "rh_place_node.py":
        holder = str(cmd.args.get("holder", "-1"))
        holder = holder if holder.lstrip("-").isdigit() and int(holder) >= 0 else str((contract.get("target_holders") or [0])[0])
        g.need(str(cmd.args.get("holder_topic") or HOLDER_TOPIC.format(holder)), "holder", "", box, [], STALE_MS,
               unit=key, how="link")
    if cmd.name == "rh_aglt_node.py":
        g.need(f"{base}/goal", "goal", "", box, [], STALE_MS, unit=key, how="on_demand")
    g.makers.append((box, tuple(sides), key, str(cmd.args.get("episode_topic") or f"{base}/episode")))


def _episode_runner(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """에피소드 실행기 — 정책 노드의 episode 서비스를 순서대로 부른다(서비스는 그리지 않는다).

    토픽으로는: FP++ 컵을 창 동안 모아 정지 자세를 기록 → 다시 낸다(relay) · 고정 홀더 자세 · aglt 목표.
    홀더와 목표는 다른 단계의 노드도 내므로 **이 단계 안에서만** 실행기가 내는 쪽이다.
    """
    episode = yaml.safe_load(_resolve(cmd.args["episode"], repo).read_text(encoding="utf-8")) or {}
    box = g.box(RUNNER_NODE.lstrip("/"), "episode_runner · 에피소드 실행기", L_OBS, status=RUNNER_NODE.lstrip("/"),
                ros=[RUNNER_NODE], unit=key, own_status=True, stages=["snapshot", "정책 순서", "복구"],
                note="정책 노드의 episode 서비스를 부른다 — 그림에는 토픽만 그린다")
    g.tags[box] = "실행기"
    stage = key.split("#")[0]
    for name, spec in (episode.get("objects") or {}).items():
        g.need(str(spec["topic"]), "object", "", box, [], STALE_MS, unit=key, how="link")
        relay = OBJECT_RELAY.format(name)
        g.providers.setdefault(relay, box)
        g.stage_providers[(stage, relay)] = box
    for holder in (episode.get("holders") or {}).values():
        g.stage_providers[(stage, HOLDER_TOPIC.format(holder))] = box
    for spec in (episode.get("policies") or {}).values():
        if spec.get("side") in SIDE_ORDER:
            g.stage_providers[(stage, f"{NS}/{spec['side']}/goal")] = box


def _pd(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    contract, robot = _read_json(_resolve(cmd.args["contract"], repo)), _robot(cmd.args["robot"], repo)
    asked = [s.strip() for s in cmd.args.get("sides", "").split(",") if s.strip()]
    if asked in (["both"], ["all"]):
        asked = list(SIDE_ORDER)
    sides = [s for s in SIDE_ORDER if s in (asked or _robot_sides(robot)) and s in (contract.get("sides") or {})]
    execute = cmd.args.get("execute", "false").lower() in ("true", "1", "yes")
    muted = "" if execute else f"무발행 pd({key}) — execute:=false 라 구동 토픽을 내지 않는다"
    label = " · ".join(f"{SIDE_KO.get(sd, sd)}팔" for sd in sides)
    # 이름은 팔마다 갈린다(09.23) — 한 팔이면 `pd_node_<side>` · status `pd_<side>`.
    node = f"/pd_node_{sides[0]}" if len(sides) == 1 else "/pd_node"
    status = f"pd_{sides[0]}" if len(sides) == 1 else "pd"
    chip = "발행" if execute else "무발행"
    pd = g.box("pd", f"{node.lstrip('/')} · PD 제어 ({label})" if label else f"{node.lstrip('/')} · PD 제어",
               L_PD, status=status, ros=[node, PD_POLL_NODE.format(sides="_".join(sides))], unit=key,
               stages=["PD 법칙", "컨트롤러 교대", chip])
    stages = g.boxes[pd]["stages"]
    if chip not in stages[-1].split(" → "):          # 같은 팔을 무발행으로 띄웠다가 발행으로 다시 띄운다 — 상자는 하나
        stages[-1] = f"{stages[-1]} → {chip}"
    drives = 0
    for side in sides:
        for role, own, cfg in _sources(robot, side):
            if role in ("arm", "ee"):                                     # ArmUnit.joint_topics
                g.need(str(cfg["topic"]), role, own or side, pd, [f"{side}:{role}"], _stale_ms(cfg), unit=key)
        for name in (contract["sides"][side].get("pd_groups") or ()):
            group = (robot.get("groups") or {}).get(name)
            if group is not None:
                _drive(g, pd, side, group, muted, key)
                drives += 1
    if not drives:                                                        # 실기 pd_node 는 이 상태로 뜨지 않는다 — 그림도 그렇게 말한다
        g.boxes[pd]["note"] = "계약의 pd_groups 가 robot yaml 의 groups 에 없다 — pd_node 는 이대로면 기동하지 않는다"


def _drive(g: _Graph, pd: str, side: str, group: Mapping, muted: str, key: str) -> None:
    backend = str(group.get("backend"))
    muted_by = [key] if muted else None                                   # 무발행은 그 명령이 떠 있을 때만이다
    if backend == "arm_forward":
        g.box("arm_drive", "팔 구동 (forward 컨트롤러)", L_DRIVE, manager="/controller_manager")
        for kind in FORWARD_KINDS:
            topic = forward_topic(side, kind)
            g.wire(pd, "arm_drive", topic, meter=kind == "position", stale_ms=STALE_MS if kind == "position" else None,
                   muted=muted, muted_by=muted_by,
                   heard_by=[topic.rsplit("/", 1)[0]])                     # ros2_control: 컨트롤러는 제 이름의 노드로 구독한다
    elif backend == "rh56f1_angle" and group.get("topic"):
        # RH56F1: 손 하나에 드라이버 노드 하나(EtherCAT · fake)가 angle_set 을 받는다 — controller_manager 가 없다
        driver = g.hand_drivers.get(side)
        ears = [driver[1]] if driver else None
        box = g.box(f"hand_{side}_drive", f"{SIDE_KO.get(side, side)}손 구동 (RH56F1 드라이버)", L_DRIVE, ros=ears,
                    note="드라이버가 angle_set 을 받아 손에 쓰고 각도 · 촉각 레지스터를 낸다")
        g.wire(pd, box, str(group["topic"]), meter=False, muted=muted, muted_by=muted_by, heard_by=ears)
    elif group.get("topic"):
        gripper = backend == "jtc_single_point"                          # 그리퍼는 팔 bringup 의 controller_manager 아래에 있다
        title = "그리퍼 구동" if gripper else f"{SIDE_KO.get(side, side)}손 구동 (JTC)"
        topic = str(group["topic"])
        ns = str(group.get("namespace") or "").strip("/")
        manager = "/controller_manager" if gripper else f"/{ns}/controller_manager" if ns else None
        g.box(f"hand_{side}_drive", title, L_DRIVE, manager=manager)
        if gripper:
            g.providers.setdefault(f"__hand_unit_{side}__", g.providers.get("__arm_unit__"))
        g.wire(pd, f"hand_{side}_drive", topic, meter=False, muted=muted, muted_by=muted_by,
               heard_by=[topic.rsplit("/", 1)[0]])


# ── 내는 쪽 ─────────────────────────────────────────────────────────────
def _fake_plant(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    robot = _robot(cmd.args["robot"], repo) if cmd.args.get("robot") else {}
    hands = str(cmd.args.get("hands") or "all") != "none"                # hands:=none — 손 드라이버를 팔마다 따로 띄운다
    for role, own, cfg in _sources(robot, None):
        topic = str(cfg["topic"])
        if role == "arm":
            g.providers.setdefault(topic, g.box("arm_state", ARM_STATE_TITLE, L_SENSE, ros=["/fake_arm_bridge"], unit=key,
                                                note="fake 대역 — fake_plant 의 MockArm 이 낸다"))
        elif role in ("ee", "tip_force") and own and hands:
            g.providers.setdefault(topic, g.box(f"hand_{own}_state", f"{SIDE_KO[own]}손 · 관절 + 손끝 힘", L_SENSE, unit=key,
                                                note="fake 대역 — fake_plant 가 낸다"))
    g.lazy[FAKE_PLANT_OBJECT_TOPIC] = {"box_id": "object_pose", "layer": L_SENSE, "unit": key, "title": OBJECT_POSE_TITLE}
    g.providers["__fake_plant__"] = key


def _fake_cup(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """fake 컵 · 홀더 자세 — 실기에서 그 토픽을 내는 상자(물체 자세 · 홀더 노드)의 대역으로 그 상자에 붙인다."""
    topic = cmd.args.get("topic")
    if not topic:
        return
    holder = _object_name(topic).startswith("cup_holder")
    box = g.box("cup_holders" if holder else "object_pose", HOLDER_TITLE if holder else OBJECT_POSE_TITLE, L_SENSE, unit=key,
                note="fake 대역 — 고정 자세를 낸다")
    g.tags[box] = "마커" if holder else "FP++"
    g.providers[topic] = box


def _bringup(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    g.providers["__arm_unit__"] = key


def _hand_driver(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    side = next((s for s in SIDE_ORDER if f"_{s}_" in cmd.name), "")
    if side:
        g.providers[f"__hand_unit_{side}__"] = key


def _rh56f1_driver(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """RH56F1 손 드라이버(EtherCAT 노드 · fake 손) — 구동 상자는 pd 가 그 손 group 을 그릴 때 만든다(`_drive`)."""
    side = _side(cmd)
    g.hand_drivers[side] = (key, HAND_DRIVER_NODE[cmd.name].format(side=side))
    g.providers[f"__hand_unit_{side}__"] = key


def _rh56f1_state(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    side = _side(cmd)
    box = g.box(f"hand_{side}_state", f"{SIDE_KO[side]}손 · 관절 + 손끝 힘", L_SENSE, ros=[HAND_STATE_NODE.format(side=side)],
                unit=key, note="드라이버의 각도 · 촉각 · 관절 힘 레지스터 → rad · N · g")
    for topic in HAND_STATE_TOPICS:
        g.providers.setdefault(topic.format(side=side), box)


def _object_name(topic: str) -> str:
    parts = topic.strip("/").split("/")
    return parts[1] if len(parts) == 3 and f"/{parts[0]}" == OUTPUT_NS else topic


def _object_id(topic: str) -> str:
    return "obj_" + "".join(c if c.isalnum() else "_" for c in _object_name(topic)).strip("_")


def _provider(g: _Graph, topic: str, role: str, side: str, unit: str | None = None) -> str | None:
    """이 토픽을 내는 상자 — 같은 단계의 것이 먼저다. 미션이 띄우는 것이 없으면 바깥(robot_control · 인지)의 상자를 만든다.
    낼 것이 없는 목표(사람 · 실행기가 가끔 주는 값)는 None — 그리지 않는다."""
    stage = unit.split("#")[0] if unit else ""
    if (stage, topic) in g.stage_providers:
        return g.stage_providers[(stage, topic)]
    if topic in g.providers:
        return g.providers[topic]
    if topic in g.lazy:
        spec = g.lazy[topic]
        return g.providers.setdefault(topic, g.box(spec["box_id"], spec["title"], spec["layer"], unit=spec["unit"]))
    if role == "goal":
        return None
    if role == "holder":
        return g.providers.setdefault(topic, g.box(_object_id(topic), f"{topic} (바깥)", L_SENSE))
    if role == "object":
        return g.providers.setdefault(topic, _perception(g, topic))
    if role == "head":
        return g.providers.setdefault(topic, g.box("head", "목 관절 (선택 입력)", L_SENSE, ros=["/head_joint_publisher"],
                                                   note="끊겨도 체인은 돈다"))
    if role in ("ee", "tip_force", "joint_force") and side:
        box = g.box(f"hand_{side}_state", f"{SIDE_KO[side]}손 · 관절 + 손끝 힘", L_SENSE, unit=g.providers.get(f"__hand_unit_{side}__"))
        return g.providers.setdefault(topic, box)
    box = g.box("arm_state", ARM_STATE_TITLE, L_SENSE, unit=g.providers.get("__arm_unit__"),
                note="joint_state_broadcaster — robot_control bringup 이 띄운다")
    return g.providers.setdefault(topic, box)


def _camera(g: _Graph) -> str:
    return g.box("camera", "카메라 (RealSense)", L_CAMERA, host=g.percept_host or VISION_HOST, note="영상은 세지 않는다(연결만 본다)")


def _perception_chain(g: _Graph, name: str, unit: str | None = None) -> str:
    """카메라 → FPP 추적 → object_pose_node. 앞의 둘은 인지 런처가 켠다(vision-3090 · 이 PC).
    FP++ 자세를 UDP 로 받아 ROS 로 내는 수신기(fpp_pose_rx)가 미션에 있으면 FPP 상자의 스위치다."""
    host = g.percept_host or VISION_HOST
    rx = g.providers.get("__fpp_rx__")
    tracker = g.box(f"fpp_{name}", f"FPP 추적 · {name}", L_TRACK, host=host, unit=rx, ros=[FPP_RX_NODE] if rx else None,
                    note=f"docker fpp_{name} ({INPUT_NS})" + (" · 자세는 fpp_pose_rx 가 UDP 로 받아 낸다" if rx else ""))
    camera = _camera(g)
    for cam in CAMERA_TOPICS:
        g.wire(camera, tracker, cam, meter=False)
    pose = g.box("object_pose", OBJECT_POSE_TITLE, L_SENSE, ros=[OBJECT_POSE_NODE], unit=unit)
    g.tags[pose] = "FP++"
    g.wire(tracker, pose, input_topic(name), stale_ms=STALE_MS)
    g.providers.setdefault(output_topic(name), pose)
    return pose


def _perception(g: _Graph, topic: str) -> str:
    """누가 /objects/<이름>/pose 를 받는데 미션에 내는 것이 없다 — 인지 사슬을 그린다(아는 이름 꼴만)."""
    name = _object_name(topic)
    if name == topic:                                                    # /objects/<이름>/pose 꼴이 아니다 — 아는 것만 그린다
        return g.box(_object_id(topic), f"{topic} (바깥)", L_SENSE)
    return _perception_chain(g, name)


def _perception_launcher(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """인지 런처 — 카메라·FP++ 컨테이너를 `--host` 의 PC 에서 켜고 끈다(vision-3090 은 ssh, local 은 이 PC).

    그림에서 인지 사슬의 유일한 스위치다(카메라·컨테이너 자체는 콘솔이 직접 못 켠다).
    """
    host = cmd.args.get("host") or VISION_HOST
    g.percept_host = host
    g.box("perception", f"인지 런처 · {host}", L_LAUNCH, ros=["/perception_launcher"], unit=key, host=host,
          note=f"{PERCEPTION_STATUS} 로 저 PC 의 카메라·컨테이너 상태를 말한다")


def _fpp_rx(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    g.providers["__fpp_rx__"] = key
    for bid, box in g.boxes.items():                                     # 수신기가 늦게 적혀도 이미 그린 FPP 상자에 붙인다
        if bid.startswith("fpp_") and key not in box.setdefault("units", []):
            box["units"].append(key)
            box["ros"] = [FPP_RX_NODE]


def _object_pose(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    names = [n for n in str(cmd.args.get("objects", "")).split(",") if n.strip()]
    for name in names:
        _perception_chain(g, name.strip(), unit=key)
    if not names:
        g.box("object_pose", OBJECT_POSE_TITLE, L_SENSE, ros=[OBJECT_POSE_NODE], unit=key)


def _cup_holders(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """홀더 마커(ArUco) → 홀더 자세. 영상 · 카메라 정보를 받고 설정의 홀더마다 /objects/<홀더>/pose 를 낸다."""
    cfg = yaml.safe_load(_resolve(cmd.args.get("cfg") or HOLDER_CFG, repo).read_text(encoding="utf-8")) or {}
    box = g.box("cup_holders", HOLDER_TITLE, L_SENSE, ros=[HOLDER_NODE], unit=key,
                note="마커로 홀더 자세를 잰다 — 에피소드는 이것이 쓴 고정 홀더 파일을 실행기가 다시 낸다")
    g.tags[box] = "마커"
    camera = _camera(g)
    for cam in (CAMERA_TOPICS[0], CAMERA_TOPICS[2]):                      # 컬러 영상 · 카메라 정보(깊이는 안 쓴다)
        g.wire(camera, box, cam, meter=False)
    for holder in cfg.get("holders") or ():
        g.providers.setdefault(output_topic(str(holder["name"])), box)


def _head_publisher(g: _Graph, key: str, cmd: Cmd, repo: Path) -> None:
    """목 상태 퍼블리셔(읽기 전용) — 받는 노드가 없어도 상자로 둔다. 콘솔에서 head 까지 관리한다."""
    topic = cmd.args.get("topic") or HEAD_TOPIC
    box = g.box("head", "목 상태 (head)", L_SENSE, ros=["/head_joint_publisher"], unit=key,
                note="읽기 전용 — 토크·게인·목표를 건드리지 않는다")
    g.providers.setdefault(topic, box)


#: 내는 쪽 — 받는 쪽이 누가 내는지 알아야 하므로 먼저, 미션 순서대로 그린다(인지 런처 → 수신기 → 물체 자세)
_PROVIDERS = {"fake_plant.launch.py": _fake_plant, "fake_cup_pose_pub.py": _fake_cup, "openarm.bimanual.launch.py": _bringup,
              "dg5f_right_driver.launch.py": _hand_driver, "dg5f_left_driver.launch.py": _hand_driver,
              "head_joint_publisher.py": _head_publisher, "perception_launcher_node.py": _perception_launcher,
              "fpp_pose_rx.py": _fpp_rx, "object_pose_node.py": _object_pose, "cup_holder_pose_node.py": _cup_holders,
              "rh56f1_driver.py": _rh56f1_driver, "fake_rh56f1_hand.py": _rh56f1_driver,
              "rh56f1_state_node.py": _rh56f1_state, "episode_runner_node.py": _episode_runner}
_HANDLERS = {"policy_chain.launch.py": _policy_chain, "pour_chain.launch.py": _pour_chain, "pour_guard_node.py": _pour_guard,
             "pd_controller.launch.py": _pd, **{name: _policy_node for name in POLICY_NODES}}


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


def _pd_sides(box: Mapping) -> set[str]:
    """pd 상자가 맡는 팔 — 본 노드가 `/pd_node_<팔>` 이면 그 팔, `/pd_node` 면 양팔(뒤의 폴링 노드 이름은 보지 않는다)."""
    main = str((box.get("ros") or [""])[0])
    side = next((s for s in SIDE_ORDER if main.endswith(f"_{s}")), None)
    return {side} if side else set(SIDE_ORDER)


def _adopt(box: dict, unit: str | None) -> None:
    if unit and not box.get("units"):
        box["units"] = [unit]


def _finish(g: _Graph) -> None:
    """모은 입력을 내는 상자에 잇고, 체인 · 정책 노드가 내는 목표를 pd·가짜 손에 잇는다."""
    for topic, role, side, dst, inputs, stale, unit, how in g.needs:
        src = _provider(g, topic, role, side, unit)
        if src is None:
            continue
        g.wire(src, dst, topic, inputs=list(inputs), units=[unit] if unit else None,
               stale_ms=stale if how == "meter" else None, meter=False if how != "meter" else None,
               on_demand=True if how == "on_demand" else None)
    makers = [b for b in g.boxes if b == "pour_node" or b.startswith("fabric")]
    master = next((b for b in ("pour_node", "obs", "episode_master") if b in g.boxes), None)
    for pd in [b for b in g.boxes if b == "pd" or b.startswith("pd@")]:   # 미션이 pd 를 둘 띄우면 둘 다 잇는다
        for maker in makers:
            g.wire(maker, pd, TOPIC["target"], stale_ms=STALE_MS, episodic=True)
        if master:
            g.wire(master, pd, TOPIC["episode"], meter=False)
        for box, sides, unit, episode in g.makers:                       # 정책 노드는 제 팔의 pd 로만
            if _pd_sides(g.boxes[pd]) & set(sides):
                g.wire(box, pd, TOPIC["target"], stale_ms=STALE_MS, episodic=True, units=[unit])
                g.wire(box, pd, episode, meter=False, units=[unit])
    if master == "pour_node" and "pour_guard" in g.boxes:
        g.wire("pour_node", "pour_guard", TOPIC["episode"], meter=False)
    plant = g.providers.get("__fake_plant__")
    for box_id, box in list(g.boxes.items()):
        if box_id == "arm_drive":
            if plant:                                                    # fake 팔은 joint_target 을 반사한다
                box.update(units=[plant], ros=["/fake_arm_bridge"])
                for w in g.wires:
                    if w["to"] == "arm_drive":
                        w["heard_by"] = ["/fake_arm_bridge"]
            else:
                _adopt(box, g.providers.get("__arm_unit__"))
        elif box_id.startswith("hand_") and box_id.endswith("_drive"):
            side = box_id.split("_")[1]
            if plant and side not in g.hand_drivers:
                _fake_hand(g, box_id, box, plant)
            else:
                _adopt(box, g.providers.get(f"__hand_unit_{side}__"))
    for box in g.boxes.values():
        if not box.get("units"):
            box.pop("units", None)


def _fake_hand(g: _Graph, box_id: str, box: dict, plant: str) -> None:
    """fake 플랜트의 손은 joint_target 을 반사한다 — JTC 토픽은 받지 않는다."""
    # `/fake_hand_state_pub` 은 좌·우가 같은 이름이다 — 한쪽이 죽어도 이름이 남는다.
    # 같은 프로세스가 만드는 `/dg5f_<side>/dg5f_<side>_controller` 만 그 손을 가리킨다.
    node = next((w["topic"].rsplit("/", 1)[0] for w in g.wires
                 if w["to"] == box_id and w["topic"].endswith("joint_trajectory")), None)
    box.update(units=[plant], note="fake 손은 joint_target 을 반사한다 — JTC 토픽은 받지 않는다")
    box.pop("manager", None)                                             # fake 에는 손 controller_manager 가 없다
    box["ros"] = [node] if node else []
    for w in g.wires:
        if w["to"] == box_id:
            w.pop("heard_by", None)
            w.setdefault("muted", "fake 손은 JTC 를 받지 않는다 — joint_target 을 그대로 반사한다")
            w.pop("muted_by", None)
    makers = [b for b in g.boxes if b == "pour_node" or b.startswith("fabric")]
    for maker in makers:
        g.wire(maker, box_id, TOPIC["target"], stale_ms=STALE_MS, episodic=True)


def _settle(g: _Graph) -> None:
    """전선에 적은 명령이 그 상자를 띄우는 명령 전부면 '언제나' 다 — 적을 필요가 없다.

    한 상자가 같은 토픽을 두 상자에게서 받으면(두 정책의 joint_target · 단독과 에피소드의 목표) 포트 이름이 같아
    중복처럼 보인다 — 이름에 내는 쪽을 붙인다.
    """
    for w in g.wires:
        src, dst = set(g.boxes[w["from"]].get("units") or ()), set(g.boxes[w["to"]].get("units") or ())
        mine = set(w.get("units") or ())
        if mine and ((src and mine >= src) or (dst and mine >= dst)):
            w.pop("units")
        if w.get("muted_by") and src and set(w["muted_by"]) >= src:
            w.pop("muted_by")
    twice = Counter((w["to"], w["topic"]) for w in g.wires)
    for w in g.wires:
        if twice[(w["to"], w["topic"])] > 1 and not w.get("label"):
            tag = g.tags.get(w["from"]) or g.boxes[w["from"]]["title"].split(" · ")[0].replace("_node", "")
            w["label"] = f"{tag} ▸ {short_topic(w['topic'])}"


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
    known = set(_PROVIDERS) | set(_HANDLERS) | set(_TOOLS)
    for key, cmd in parsed:
        if cmd.name not in known and not _shell_step(units[key]):
            _unknown(g, key, cmd, note=units[key].note)
    _finish(g)
    _settle(g)
    used = sorted({b["col"] for b in g.boxes.values()})
    boxes = [{**b, "col": used.index(b["col"])} for b in sorted(g.boxes.values(), key=lambda b: b["col"])]
    col = {b["id"]: b["col"] for b in boxes}
    wires = [w for w in g.wires if col[w["from"]] < col[w["to"]]]
    return parse_diagram({"boxes": boxes, "wires": wires}, path=Path("<generated>"),
                         status_nodes=(*status_nodes, *sorted(g.own_status)))
