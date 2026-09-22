#!/usr/bin/env python3
"""차렷 -> 홈 경로 계획용 충돌 세계(MuJoCo, ROS·Isaac 없음). plan_home_path.py 가 쓴다.

세계 = 자산 URDF 의 양팔 로봇(링크 하나 = MuJoCo body 하나) + env.yaml 테이블 USD(usda) 의 충돌 메쉬를
연결 성분별 AABB 상자로 바꾼 것 + (선택) 스폰 중심의 컵 상자.

충돌 형상 규칙(보수적 = 실제보다 크게):
  * URDF collision 메쉬 -> 볼록 껍질(MuJoCo 가 mesh geom 을 볼록 껍질로 충돌 처리한다).
    껍질 부피가 메쉬의 SLAB_HULL_RATIO 배를 넘고 길이가 SLAB_MIN_EXTENT 보다 긴 메쉬(몸통 기둥)는
    가장 긴 축을 SLAB_THICKNESS 두께로 잘라 조각마다 껍질을 만든다(몸통 껍질 하나가 어깨 주변 팔을 삼키지 않게).
  * 테이블 = usda `Collision` 메쉬의 연결 성분 중 세 축 모두 두께가 있는 것의 AABB 상자(다리 원기둥도 상자로).

거리: MuJoCo 2.3.0 에는 mj_geomDistance 가 없다. 모든 geom 에 margin 을 주고 mj_collision 의 contact.dist
(margin 안의 쌍만 나온다)를 쓴다. 상자-메쉬는 정확, 메쉬-메쉬는 MPR 근사다(검증 결과는 plan_home_path 보고서).
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIM2REAL = Path(__file__).resolve().parents[3]
RL_WS = SIM2REAL.parent
URDF_DEFAULT = RL_WS / "hdgp/assets/robot/openarm_dg5f-m-short_bi_rl/openarm_dg5f-m-short_bi_rl.urdf"
ENV_YAML_DEFAULT = SIM2REAL / "deploy/policies/right_aglt/params/env.yaml"
CONTRACT_DEFAULT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m-short_bi_rl/deploy_contract.json"
PROFILE_DEFAULT = RL_WS / "robot_control/src/robot_control/profiles/openarm_tesollo.yaml"

SLAB_HULL_RATIO = 2.0      # 껍질/메쉬 부피비가 이보다 크면 조각낸다
SLAB_MIN_EXTENT = 0.30     # [m] 이보다 긴 메쉬만
SLAB_THICKNESS = 0.05      # [m]
MIN_BOX_THICKNESS = 1e-4   # [m] 이보다 얇은 성분(판 윗면 조각 등)은 버린다 — 판 상자 안에 들어 있다

# cup_big_rl.usd(crate) 의 collisions 메쉬 AABB, scale 1.0(= cup_big_s100, 실물 빨간 컵).
# /usr/bin/python3 pxr BBoxCache 로 09.22 측정(원점 기준). venv 에 pxr 가 없어 값으로 둔다.
CUP_S100_AABB = ((-0.04635, -0.04600, -0.07729), (0.04365, 0.04400, 0.10033))


def sha1_of(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def local_rl_path(p: str) -> Path:
    """env.yaml 은 학습 호스트 경로(/home/oem/rl_ws/...)다 — rl_ws 아래 상대 경로로 바꿔 이 호스트에서 연다."""
    s = str(p).replace("file://", "")
    if "/rl_ws/" in s:
        return RL_WS / s.split("/rl_ws/", 1)[1]
    return Path(s)


def load_env_yaml(path: Path) -> dict:
    import yaml

    class _L(yaml.SafeLoader):
        pass

    def _py(loader, suffix, node):          # !!python/tuple, !!python/object/apply:builtins.slice
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    _L.add_multi_constructor("tag:yaml.org,2002:python/", _py)
    with open(path) as f:
        return yaml.load(f, Loader=_L)


def load_profile_limits(path: Path) -> dict[str, tuple[float, float]]:
    import yaml

    with open(path) as f:
        prof = yaml.safe_load(f)
    return {j["canonical"]: (float(j["lower"]), float(j["upper"]))
            for j in prof["joints"] if "lower" in j and "upper" in j}


# ---------------------------------------------------------------- 테이블 (usda)

def _usda_array(text: str, name: str) -> str:
    m = re.search(re.escape(name) + r"\s*=\s*\[(.*?)\]", text, re.S)
    if not m:
        raise ValueError(f"usda 에 {name} 가 없다")
    return m.group(1)


def table_boxes_from_usda(usda: Path, offset=(0.0, 0.0, 0.0)) -> list[dict]:
    """`def Mesh "Collision"` 의 연결 성분 AABB -> [{name, lo, hi}] (월드 = 테이블 init_state.pos 더함)."""
    import trimesh

    text = Path(usda).read_text()
    i = text.find('def Mesh "Collision"')
    if i < 0:
        raise ValueError(f"{usda}: Collision 메쉬가 없다")
    body = text[i:]
    if 'metersPerUnit = 1' not in text[:2000]:
        raise ValueError(f"{usda}: metersPerUnit 1 이 아니다 — 단위 변환 미구현")
    pts = np.array([float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?",
                                                 _usda_array(body, "point3f[] points"))]).reshape(-1, 3)
    counts = np.array([int(x) for x in _usda_array(body, "int[] faceVertexCounts").split(",")])
    if not (counts == 3).all():
        raise ValueError("삼각형이 아닌 면 — 미구현")
    faces = np.array([int(x) for x in _usda_array(body, "int[] faceVertexIndices").split(",")]).reshape(-1, 3)
    mesh = trimesh.Trimesh(pts, faces, process=True)
    boxes = []
    for part in mesh.split(only_watertight=False):
        lo, hi = part.bounds
        if (hi - lo).min() < MIN_BOX_THICKNESS:
            continue
        boxes.append({"lo": lo + np.asarray(offset), "hi": hi + np.asarray(offset), "faces": len(part.faces)})
    boxes.sort(key=lambda b: (round(float(b["lo"][2]), 4), round(float(b["lo"][0]), 4), round(float(b["lo"][1]), 4)))
    for k, b in enumerate(boxes):
        b["name"] = f"table_{k}"
    return boxes


# ---------------------------------------------------------------- URDF

def _rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = _rpy_to_R([float(v) for v in o.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


def _R_to_quat(R) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(-1))
    return q


@dataclass
class UJoint:
    name: str
    jtype: str
    parent: str
    child: str
    T: np.ndarray
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0


@dataclass
class ULink:
    name: str
    collisions: list = field(default_factory=list)   # (T, kind, data)


def parse_urdf(path: Path) -> tuple[dict[str, ULink], list[UJoint], str]:
    root = ET.parse(path).getroot()
    links = {}
    for l in root.findall("link"):
        ul = ULink(l.get("name"))
        for c in l.findall("collision"):
            g = c.find("geometry")
            T = _origin(c)
            m = g.find("mesh")
            if m is not None:
                scale = [float(v) for v in m.get("scale", "1 1 1").split()]
                ul.collisions.append((T, "mesh", (local_rl_path(m.get("filename")), scale)))
            elif g.find("box") is not None:
                ul.collisions.append((T, "box", [float(v) / 2 for v in g.find("box").get("size").split()]))
            elif g.find("sphere") is not None:
                ul.collisions.append((T, "sphere", [float(g.find("sphere").get("radius"))]))
            elif g.find("cylinder") is not None:
                cy = g.find("cylinder")
                ul.collisions.append((T, "cylinder", [float(cy.get("radius")), float(cy.get("length")) / 2]))
        links[ul.name] = ul
    joints = []
    for j in root.findall("joint"):
        ax = j.find("axis")
        lim = j.find("limit")
        joints.append(UJoint(
            j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"), _origin(j),
            np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
            float(lim.get("lower", 0)) if lim is not None else 0.0,
            float(lim.get("upper", 0)) if lim is not None else 0.0))
    children = {j.child for j in joints}
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF 루트가 하나가 아니다: {roots}")
    return links, joints, roots[0]


_MESH_CACHE: dict = {}


def mesh_hull_pieces(path: Path, scale) -> tuple[list[np.ndarray], str]:
    """메쉬 -> 볼록 조각들의 꼭짓점 목록과 처리 방식 문자열."""
    key = (str(path), tuple(scale))
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    import trimesh

    m = trimesh.load(str(path), force="mesh")
    m.apply_scale(scale)
    hull = m.convex_hull
    ext = m.extents
    ratio = hull.volume / max(abs(m.volume), 1e-12) if m.is_watertight else 1.0
    pieces, how = [np.asarray(hull.vertices)], "hull"
    if m.is_watertight and ratio > SLAB_HULL_RATIO and ext.max() > SLAB_MIN_EXTENT:
        ax = int(np.argmax(ext))
        n = np.zeros(3)
        n[ax] = 1.0
        lo, hi = m.bounds[0][ax], m.bounds[1][ax]
        cuts = np.arange(lo, hi, SLAB_THICKNESS).tolist() + [hi]
        pieces = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            s = m.slice_plane(n * a, n, cap=False)
            if s is None or len(s.vertices) == 0:
                continue
            s = s.slice_plane(n * b, -n, cap=False)
            if s is None or len(s.vertices) < 4:
                continue
            v = np.asarray(s.vertices)
            if np.linalg.matrix_rank(v - v.mean(0), tol=1e-6) < 3:
                continue
            pieces.append(v)
        how = f"slab{len(pieces)}(hull/mesh {ratio:.1f}x)"
    _MESH_CACHE[key] = (pieces, how)
    return pieces, how


# ---------------------------------------------------------------- MJCF

def _fmt(v) -> str:
    return " ".join(f"{float(x):.7g}" for x in np.asarray(v).reshape(-1))


@dataclass
class WorldSpec:
    urdf: Path = URDF_DEFAULT
    env_yaml: Path = ENV_YAML_DEFAULT
    side: str = "right"
    with_cup: bool = False
    detect_margin: float = 0.08     # contact 를 보고받을 거리 상한 [m]


@dataclass
class World:
    model: object
    data: object
    moving_joints: list[str]
    moving_qadr: np.ndarray
    qadr: dict[str, int]
    body_group: dict[str, int]      # body -> 움직이는 팔 관절 번호(0 = 고정)
    geom_body: list[str]
    table_boxes: list[dict]
    cup_box: dict | None
    mesh_notes: dict[str, str]
    excluded_pairs: set = field(default_factory=set)   # 시작 자세에서 이미 겹친 자기충돌 쌍(보고 후 제외)

    def set_q(self, q: dict[str, float]) -> None:
        for n, v in q.items():
            if n in self.qadr:
                self.data.qpos[self.qadr[n]] = v

    def pair_distances(self, q_moving: np.ndarray) -> dict[tuple[str, str], float]:
        """움직이는 팔 관절값 -> {(body_a, body_b): 최소 dist} (margin 안의 쌍만, 제외 쌍 빼고)."""
        import mujoco
        m, d = self.model, self.data
        d.qpos[self.moving_qadr] = q_moving
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        out: dict = {}
        for c in d.contact[: d.ncon]:
            a, b = self.geom_body[c.geom1], self.geom_body[c.geom2]
            key = (a, b) if a < b else (b, a)
            if key in self.excluded_pairs:
                continue
            if c.dist < out.get(key, np.inf):
                out[key] = float(c.dist)
        return out

    def is_world(self, body: str) -> bool:
        return body.startswith("table_") or body == "cup"


def build_world(spec: WorldSpec) -> World:
    import mujoco

    env = load_env_yaml(spec.env_yaml)
    links, joints, root = parse_urdf(spec.urdf)
    pre = "r_" if spec.side == "right" else "l_"
    moving = [f"{pre}aj_{i}" for i in range(1, 8)]
    by_parent: dict[str, list[UJoint]] = {}
    for j in joints:
        by_parent.setdefault(j.parent, []).append(j)

    # 로봇 베이스 자세 = env.yaml robot_cfg.init_state (테이블도 같은 방식)
    rpos = np.array(env["robot_cfg"]["init_state"]["pos"], dtype=float)
    rquat = np.array(env["robot_cfg"]["init_state"]["rot"], dtype=float)     # w x y z
    tpos = np.array(env["table_cfg"]["init_state"]["pos"], dtype=float)
    trot = np.array(env["table_cfg"]["init_state"]["rot"], dtype=float)
    if not np.allclose(trot, [1, 0, 0, 0]):
        raise ValueError(f"테이블 회전 {trot} — 항등이 아닌 경우 미구현")
    usda = local_rl_path(env["table_cfg"]["spawn"]["usd_path"])
    boxes = table_boxes_from_usda(usda, tpos)

    assets, mesh_notes = [], {}
    body_group: dict[str, int] = {}

    def geoms_xml(link: ULink) -> str:
        out = []
        for k, (T, kind, data) in enumerate(link.collisions):
            pos, quat = T[:3, 3], _R_to_quat(T[:3, :3])
            if kind == "mesh":
                path, scale = data
                pieces, how = mesh_hull_pieces(path, scale)
                mesh_notes[f"{link.name}:{path.name}"] = how
                for p, verts in enumerate(pieces):
                    mname = f"m_{link.name}_{k}_{p}"
                    assets.append(f'<mesh name="{mname}" vertex="{_fmt(verts)}"/>')
                    out.append(f'<geom type="mesh" mesh="{mname}" pos="{_fmt(pos)}" quat="{_fmt(quat)}"/>')
            else:
                out.append(f'<geom type="{kind}" size="{_fmt(data)}" pos="{_fmt(pos)}" quat="{_fmt(quat)}"/>')
        return "\n".join(out)

    def body_xml(name: str, T: np.ndarray, joint: UJoint | None, group: int) -> str:
        body_group[name] = group
        jx = ""
        if joint is not None and joint.jtype in ("revolute", "continuous"):
            rng = f' range="{joint.lower:.7g} {joint.upper:.7g}"' if joint.jtype == "revolute" else ""
            jx = f'<joint name="{joint.name}" type="hinge" axis="{_fmt(joint.axis)}"{rng} limited="{"true" if rng else "false"}"/>'
        elif joint is not None and joint.jtype == "prismatic":
            raise ValueError(f"prismatic 관절 {joint.name} 미구현")
        kids = []
        for cj in by_parent.get(name, []):
            g = moving.index(cj.name) + 1 if cj.name in moving else group
            kids.append(body_xml(cj.child, cj.T, cj, g))
        return (f'<body name="{name}" pos="{_fmt(T[:3, 3])}" quat="{_fmt(_R_to_quat(T[:3, :3]))}">\n{jx}\n'
                f'{geoms_xml(links[name])}\n' + "\n".join(kids) + "\n</body>")

    Troot = np.eye(4)
    Troot[:3, 3] = rpos
    mjq = rquat / np.linalg.norm(rquat)
    Rr = np.zeros(9)
    mujoco.mju_quat2Mat(Rr, mjq)
    Troot[:3, :3] = Rr.reshape(3, 3)
    robot = body_xml(root, Troot, None, 0)

    world_geoms = []
    for b in boxes:
        c, h = (b["lo"] + b["hi"]) / 2, (b["hi"] - b["lo"]) / 2
        world_geoms.append(f'<body name="{b["name"]}" pos="{_fmt(c)}"><geom type="box" size="{_fmt(h)}"/></body>')
    cup_box = None
    if spec.with_cup:
        cx, cy = [float(v) for v in env["object_spawn_center_override"]]
        zs = float(env["table_surface_z"])
        lo, hi = np.array(CUP_S100_AABB[0]), np.array(CUP_S100_AABB[1])
        origin = np.array([cx, cy, zs - lo[2]])      # 뷰어와 같은 규칙: 원점 z = 상판 + 원점 아래 길이
        cup_box = {"name": "cup", "lo": origin + lo, "hi": origin + hi}
        c, h = (cup_box["lo"] + cup_box["hi"]) / 2, (cup_box["hi"] - cup_box["lo"]) / 2
        world_geoms.append(f'<body name="cup" pos="{_fmt(c)}"><geom type="box" size="{_fmt(h)}"/></body>')

    # 충돌 비트: 움직이는 쪽(group>0) contype 1 / conaffinity 3, 나머지 contype 2 / conaffinity 1
    #  -> 고정-고정 쌍은 아예 계산하지 않는다.
    xml = (f'<mujoco model="home_path_world"><compiler angle="radian"/>'
           f'<option><flag gravity="disable"/></option>'
           f'<size nconmax="4000"/>'
           f'<default><geom contype="2" conaffinity="1" margin="{spec.detect_margin}" gap="0"/></default>'
           f'<asset>{"".join(assets)}</asset><worldbody>{robot}{"".join(world_geoms)}</worldbody></mujoco>')

    # 제외: URDF 부모-자식(고정 관절 포함) + 같은 그룹(서로 상대 자세가 변하지 않음, 0 은 이미 비트로 빠짐)
    excl = set()
    for j in joints:
        excl.add(tuple(sorted((j.parent, j.child))))
    by_group: dict[int, list[str]] = {}
    for b, g in body_group.items():
        by_group.setdefault(g, []).append(b)
    for g, bs in by_group.items():
        if g == 0:
            continue
        for i in range(len(bs)):
            for k in range(i + 1, len(bs)):
                excl.add(tuple(sorted((bs[i], bs[k]))))
    spec_x = "".join(f'<exclude body1="{a}" body2="{b}"/>' for a, b in sorted(excl))
    xml = xml.replace("</worldbody>", f"</worldbody><contact>{spec_x}</contact>")

    model = mujoco.MjModel.from_xml_string(xml)
    for gi in range(model.ngeom):
        bname = model.body(model.geom_bodyid[gi]).name
        if body_group.get(bname, 0) > 0:
            model.geom_contype[gi], model.geom_conaffinity[gi] = 1, 3
    data = mujoco.MjData(model)
    qadr = {model.joint(i).name: int(model.jnt_qposadr[i]) for i in range(model.njnt)}
    geom_body = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
    return World(model, data, moving, np.array([qadr[n] for n in moving]), qadr, body_group,
                 geom_body, boxes, cup_box, mesh_notes)
