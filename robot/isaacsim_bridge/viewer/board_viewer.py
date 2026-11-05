"""보드 뷰어 — 조작 패널(board_jog)이 UDP 로 보내는 상태를 실기와 같은 장면(로봇·탁자·홀더)에 그린다. 읽기 전용.

    패널 --UDP JSON 127.0.0.1:47813 30 Hz--> board_viewer.py (Isaac Lab 2.x Kit, ROS 없음)

계약(board_packet.py 가 정본·검사):
    {"t": float, "names": [r_aj_1..7 + 손 구동 6], "q_show": [..], "q_cmd": [..] | null,
     "boards": {"charuco": {"T_base_board": 4x4}}, "target_xyz_base": [3] | null, "tip_xyz_base": [3],
     "contact": bool, "mode": "dry" | "pub"}
    좌표 = 로봇 base(body_root) 기준 m. 손 종속(mimic)은 여기서 자산 URDF <mimic> 으로 펼친다.

그리는 것: q_show 로봇(불투명) · q_cmd 파랑 반투명 고스트 로봇(null 이면 숨김) · ChArUco 7x5(30 mm) 판(흰 판 + 검은 칸 18) ·
목표점 구(r 3 mm, 자홍) · 검지 끝 구(r 5 mm, 주황 · 접촉이면 초록).
장면은 hdgp rh_aglt 와 같은 USD(openarm_rh56f1_bi_rl · env_v1 탁자 · cup_holder_rl)만 담은 최소 InteractiveScene 이다
(태스크 env 를 쓰지 않는다). 홀더 자리 = config/cup_holder_poses_5090_v1.yaml(실측). 로봇 루트 = 원점 = base.
물리는 진행하지 않는다: sim.reset 뒤 관절 상태 쓰기 → sim.forward(운동학만) → sim.render 만 돈다(hdgp sim_video/kit_render 와 같은 방식).

    ~/rl_ws/IsaacLab/isaaclab.sh -p board_viewer.py                    # GUI(DISPLAY), 마우스로 시점 돌리기
    ... board_viewer.py --shot out.png --max_seconds 40                # 첫 패킷 3 s 뒤 뷰포트 한 장
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
RL_WS = HERE.parents[3]
S2R = RL_WS / "sim2real"
ASSETS = RL_WS / "hdgp" / "assets"
ROBOT_USD = ASSETS / "robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.usd"
ROBOT_URDF = ROBOT_USD.with_suffix(".urdf")
TABLE_USD = ASSETS / "simulation_setting/env_v1/usd/env_v1.usda"
HOLDER_USD = ASSETS / "cup_holder/cup_holder_rl.usd"
HOLDER_POSES = S2R / "config/cup_holder_poses_5090_v1.yaml"
CONTRACT = S2R / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
GROUND_Z = -0.10                       # rh_aglt ground_plane_z
sys.path.insert(0, str(HERE))
import board_packet  # noqa: E402

parser = argparse.ArgumentParser(description="조작 패널 상태 → Isaac 장면(읽기 전용)")
parser.add_argument("--port", type=int, default=board_packet.DEFAULT_PORT)
parser.add_argument("--eye", default="1.05,-0.75,0.85", help="카메라 위치(base, m)")
parser.add_argument("--lookat", default="0.30,-0.25,0.25", help="바라볼 점(base, m)")
parser.add_argument("--ghost_opacity", type=float, default=0.35, help="고스트(q_cmd) 불투명도 0~1")
parser.add_argument("--ghost_color", default="0.2,0.45,1.0", help="고스트 색 r,g,b(0~1)")
parser.add_argument("--max_hz", type=float, default=60.0)
parser.add_argument("--shot", default=None, help="PNG — 첫 패킷 3 s 뒤(없으면 20 s 뒤) 뷰포트를 한 장 저장")
parser.add_argument("--max_seconds", type=float, default=0.0, help=">0 이면 이 시간 뒤 종료")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless and args.shot:
    args.enable_cameras = True          # 헤드리스엔 뷰포트 창이 없어 --shot 은 replicator 렌더로 찍는다
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402


def _log(msg: str) -> None:
    print(f"[board_viewer] {msg}", flush=True)


def _xyz(s: str) -> tuple:
    v = tuple(float(x) for x in s.split(","))
    if len(v) != 3:
        raise ValueError(f"'x,y,z' 여야 한다: {s}")
    return v


def _hdgp_mimic():
    """hdgp rh_aglt_r/mimic.py(표준 라이브러리만) — 패키지 init 을 거치지 않게 파일로 읽는다."""
    path = RL_WS / "hdgp/source/openarm/openarm/agnostic/tasks/rh_aglt_r/mimic.py"
    spec = importlib.util.spec_from_file_location("_rh_mimic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {**mod.parse_mimic_pairs(str(ROBOT_URDF), "r"), **mod.parse_mimic_pairs(str(ROBOT_URDF), "l")}


def _robot_cfg(name: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),   # 로봇·고스트가 겹쳐도 밀지 않게
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        soft_joint_pos_limit_factor=1.0)


def _scene_cfg() -> InteractiveSceneCfg:
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    cfg.robot = _robot_cfg("Robot")
    cfg.ghost = _robot_cfg("Ghost")
    cfg.table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=sim_utils.UsdFileCfg(usd_path=str(TABLE_USD)))
    cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)))
    cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)))
    holders = yaml.safe_load(HOLDER_POSES.read_text())["holders"]
    for name, h in sorted(holders.items()):
        yaw = float(h["yaw_rad"])
        setattr(cfg, name, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(HOLDER_USD),
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in h["position"]),
                                                    rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))))
    _log(f"홀더 {len(holders)} 개 ← {HOLDER_POSES}")
    return cfg


# ---------------------------------------------------------------- 보드(board_jog BOARD_CONVENTION)
SX, SY, SQ, PLATE_T, INK_T = 7, 5, 0.030, 0.003, 0.0006
# 판 z = 인쇄면에서 보드 속. 흰 판은 z ∈ [0, PLATE_T], 검은 칸은 인쇄면 위 z ∈ [−INK_T, 0]. 마커 칸((r+c) 홀수)은 흰색.
BOARD_LOCAL = [((SX * SQ / 2, SY * SQ / 2, PLATE_T / 2), 0)] + [
    (((c + 0.5) * SQ, (r + 0.5) * SQ, -INK_T / 2), 1) for r in range(SY) for c in range(SX) if (r + c) % 2 == 0]


BOARD_PATH = "/Visuals/charuco_board"


def _board_prims():
    """보드 = 평범한 USD 큐브 19 개를 한 Xform 아래 둔다(자세는 그 Xform 하나만 바꾼다).
    VisualizationMarkers(PointInstancer)로 그리면 고스트용 RTX 설정(translucency + fractionalCutoutOpacity)이 켜졌을 때
    판이 통째로 안 그려진다(11.05 재현: 두 설정 다 켜면 안 보이고, 하나라도 끄면 보인다)."""
    import omni.usd
    UsdGeom.Xform.Define(omni.usd.get_context().get_stage(), BOARD_PATH)
    kinds = [((SX * SQ, SY * SQ, PLATE_T), (0.95, 0.95, 0.95)), ((SQ, SQ, INK_T), (0.02, 0.02, 0.02))]
    for k, (_, rgb) in enumerate(kinds):                # 재질은 종류별 한 장
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.5)
        mat.func(f"{BOARD_PATH}/Looks/m{k}", mat)
    for i, (p, k) in enumerate(BOARD_LOCAL):
        cfg = sim_utils.CuboidCfg(size=kinds[k][0])
        cfg.func(f"{BOARD_PATH}/c{i:02d}", cfg, translation=p)
        sim_utils.bind_visual_material(f"{BOARD_PATH}/c{i:02d}/geometry/mesh", f"{BOARD_PATH}/Looks/m{k}")
    return UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(BOARD_PATH)).MakeMatrixXform()


def _marker_sets():
    def mat(rgb):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.5)

    board = _board_prims()
    points = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/points", markers={
        "target": sim_utils.SphereCfg(radius=0.003, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.8), emissive_color=(0.6, 0.0, 0.5))),
        "tip": sim_utils.SphereCfg(radius=0.005, visual_material=mat((1.0, 0.5, 0.0))),
        "tip_contact": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.6, 0.0)))}))
    return board, points


class Mirror:
    """canonical 이름 → 관절 위치(속도 0) 쓰기. 종속 관절은 리더 × 배율로 펼치고, 한계 밖은 표시용 clamp."""

    def __init__(self, art, mimic: dict) -> None:
        self.art, self.mimic = art, mimic
        self.index = {n: i for i, n in enumerate(art.joint_names)}
        lim = art.data.soft_joint_pos_limits[0]
        self.lo, self.hi = lim[:, 0].clone(), lim[:, 1].clone()
        self.ignored: set = set()

    def apply(self, q: dict) -> None:
        q = dict(q)
        for dep, (lead, mult) in self.mimic.items():
            if lead in q:
                q[dep] = mult * q[lead]
        names = [n for n in q if n in self.index]
        for n in set(q) - set(names) - self.ignored:
            self.ignored.add(n)
            _log(f"자산에 없는 관절 무시: {n}")
        if not names:
            return
        dev = self.art.device
        ids = torch.tensor([self.index[n] for n in names], device=dev, dtype=torch.long)
        v = torch.clamp(torch.tensor([q[n] for n in names], device=dev), self.lo[ids], self.hi[ids]).unsqueeze(0)
        self.art.write_joint_state_to_sim(v, torch.zeros_like(v), joint_ids=ids)


def _home() -> dict:
    sides = json.loads(CONTRACT.read_text())["sides"]
    out = {}
    for side, p in (("right", "r"), ("left", "l")):
        out |= {f"{p}_aj_{i + 1}": float(v) for i, v in enumerate(sides[side]["home_arm"])} | sides[side]["home_hand"]
    return out


def _draw_board(board, T, origin) -> None:
    """board = 보드 Xform 의 transform op. USD 는 행 벡터 규약이라 T(열 벡터)를 전치해 넣는다."""
    M = [list(r) for r in T]
    for i in range(3):
        M[i][3] += float(origin[i])
    board.Set(Gf.Matrix4d([[M[r][c] for r in range(4)] for c in range(4)]))


def _draw_points(points, pkt, origin, dev) -> None:
    pos = torch.tensor([pkt.target or (0.0, 0.0, 0.0), pkt.tip], device=dev) + origin
    scale = torch.ones(2, 3, device=dev)
    if pkt.target is None:
        scale[0] = 0.0
    points.visualize(pos, scales=scale, marker_indices=[0, 2 if pkt.contact else 1])


def _set_visible(prim_path: str, visible: bool) -> None:
    import omni.usd
    img = UsdGeom.Imageable(omni.usd.get_context().get_stage().GetPrimAtPath(prim_path))
    img.MakeVisible() if visible else img.MakeInvisible()


def _ghost_look(prim_path: str, rgb: tuple, opacity: float) -> None:
    """고스트 메시 전부에 반투명 OmniPBR 한 장을 강한 바인딩으로 덮는다(원래 재질 무시)."""
    import omni.usd
    # RTX 실시간에서 opacity < 1 을 컷아웃이 아닌 반투명으로 그리려면 둘 다 켠다(Isaac Lab 렌더 모드 기본은 translucency 끔)
    carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)
    carb.settings.get_settings().set("/rtx/translucency/enabled", True)
    stage = omni.usd.get_context().get_stage()
    cfg = sim_utils.MdlFileCfg(mdl_path="OmniPBR.mdl")
    cfg.func("/World/Looks/ghost", cfg)
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/ghost/Shader"))
    for name, typ, val in (("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, rgb),
                           ("enable_opacity", Sdf.ValueTypeNames.Bool, True),
                           ("opacity_constant", Sdf.ValueTypeNames.Float, opacity),
                           ("opacity_threshold", Sdf.ValueTypeNames.Float, 0.0),
                           ("enable_emission", Sdf.ValueTypeNames.Bool, True),
                           ("emissive_color", Sdf.ValueTypeNames.Color3f, rgb),
                           ("emissive_intensity", Sdf.ValueTypeNames.Float, 300.0)):
        shader.CreateInput(name, typ).Set(val)
    mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/ghost"))
    root = stage.GetPrimAtPath(prim_path)
    # 인스턴스 안 메시에는 바깥 바인딩이 RTX 에 닿지 않는다 → 고스트만 인스턴싱을 풀고 메시마다 직접 바인딩
    while inst := [p for p in Usd.PrimRange(root) if p.IsInstance()]:
        for p in inst:
            p.SetInstanceable(False)
    # 메시·서브셋과 이미 바인딩을 가진 조상(손 링크는 조상이 strongerThanDescendants 로 묶음)을 전부 고스트 재질로 바꾼다
    gprims = [p for p in Usd.PrimRange(root)
              if p.IsA(UsdGeom.Gprim) or p.IsA(UsdGeom.Subset) or p.HasRelationship("material:binding")]
    for p in gprims:
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
    _log(f"고스트 재질: OmniPBR rgb {rgb} opacity {opacity} → 프림 {len(gprims)} 개")


def _legend() -> None:
    """GUI 뷰포트 왼쪽 위 범례. Kit 기본 글꼴에 한글이 없어 영문으로 쓴다. 헤드리스 캡처에는 안 찍힌다."""
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport_window
    win = get_active_viewport_window()
    if win is None:
        _log("뷰포트 창 없음(헤드리스) — 범례 생략")
        return
    with win.get_frame("board_viewer_legend"):
        with ui.VStack(height=0):
            ui.Label("  BLUE translucent = command (q_cmd)   |   solid = measured (q_show)",
                     height=24, style={"color": 0xFFFFFFFF, "font_size": 18, "background_color": 0xAA000000})


def _capture(path: str, sim) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not args.headless:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        capture_viewport_to_file(get_active_viewport(), os.path.abspath(path))
        return
    import numpy as np
    import omni.replicator.core as rep
    from PIL import Image
    ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    ann.attach([rep.create.render_product("/OmniverseKit_Persp", (1280, 720))])
    for _ in range(60):                                 # 예열 동안은 빈 배열 · 몇 프레임 쌓여야 RTX 가 수렴
        sim.render()
    rgb = np.asarray(ann.get_data())
    if not rgb.size:
        raise RuntimeError("헤드리스 캡처가 비었다")
    Image.fromarray(rgb[..., :3]).save(path)


def main() -> None:
    for p in (ROBOT_USD, ROBOT_URDF, TABLE_USD, HOLDER_USD, HOLDER_POSES, CONTRACT):
        if not p.is_file():
            raise FileNotFoundError(f"없음: {p}")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device))
    scene = InteractiveScene(_scene_cfg())
    board, points = _marker_sets()
    _ghost_look("/World/envs/env_0/Ghost", _xyz(args.ghost_color), args.ghost_opacity)
    if not args.headless:
        _legend()
    sim.reset()                                         # 물리 뷰 만들기만. 이후 sim.step 없음
    dev = sim.device
    origin = scene.env_origins[0]
    robot = Mirror(scene["robot"], _hdgp_mimic())
    ghost = Mirror(scene["ghost"], robot.mimic)
    home = _home()
    robot.apply(home)
    ghost.apply(home)
    _set_visible("/World/envs/env_0/Ghost", False)
    _set_visible(BOARD_PATH, False)
    board_on = False
    points.set_visibility(False)
    eye, look = _xyz(args.eye), _xyz(args.lookat)
    o = origin.tolist()
    sim.set_camera_view([e + b for e, b in zip(eye, o)], [e + b for e, b in zip(look, o)])
    sim.forward()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)
    _log(f"준비 — 관절 {len(robot.index)} 개 · 종속 {len(robot.mimic)} · env 원점 {o} · udp://127.0.0.1:{args.port} 대기 · 물리 진행 없음")

    period = 1.0 / max(args.max_hz, 1.0)
    t0 = last_status = time.monotonic()
    got = bad = 0
    first_at = last = None
    ghost_on = None
    shot_done = False
    while simulation_app.is_running():
        tick = time.monotonic()
        pkt = None
        while True:                                     # 쌓인 것 중 마지막 유효 패킷만
            try:
                data, _ = sock.recvfrom(board_packet.MAX_BYTES + 1)
            except BlockingIOError:
                break
            try:
                pkt = board_packet.decode(data)
                got += 1
            except ValueError as exc:
                bad += 1
                if bad <= 5:
                    _log(f"패킷 버림: {exc}")
        if pkt is not None:
            if first_at is None:
                first_at = tick
                _log(f"첫 패킷: mode {pkt.mode} · {len(pkt.names)} 관절 · 고스트 {'있음' if pkt.q_cmd else '없음'} · "
                     f"보드 {'있음' if pkt.T_base_board else '없음'}")
                points.set_visibility(True)
            last = pkt
            robot.apply(dict(zip(pkt.names, pkt.q_show)))
            if (pkt.q_cmd is not None) != ghost_on:
                ghost_on = pkt.q_cmd is not None
                _set_visible("/World/envs/env_0/Ghost", ghost_on)
            if ghost_on:
                ghost.apply(dict(zip(pkt.names, pkt.q_cmd)))
            if (pkt.T_base_board is not None) != board_on:
                board_on = pkt.T_base_board is not None
                _set_visible(BOARD_PATH, board_on)
            if board_on:
                _draw_board(board, pkt.T_base_board, origin.tolist())
            _draw_points(points, pkt, origin, dev)
            sim.forward()                               # 운동학 갱신만(물리 스텝 아님)
        sim.render()

        if args.shot and not shot_done and ((first_at and tick - first_at > 3.0) or tick - t0 > 20.0):
            _capture(args.shot, sim)
            for _ in range(10):                         # 캡처는 다음 몇 프레임에 쓰인다
                sim.render()
            shot_done = True
            _log(f"SHOT {args.shot} packets={got}")
        if tick - last_status >= 5.0:
            last_status = tick
            age = f"{time.time() - last.t:.2f}s" if last else "-"
            _log(f"packets={got} rejected={bad} mode={last.mode if last else '-'} contact={last.contact if last else '-'} last_age={age}")
        if args.max_seconds > 0 and tick - t0 > args.max_seconds:
            _log("max_seconds 도달 — 종료")
            break
        rest = period - (time.monotonic() - tick)
        if rest > 0:
            time.sleep(rest)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:                        # noqa: BLE001 — Kit 종료가 rc 를 덮지 않게(isaac_viewer 와 같다)
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            simulation_app.close()
            raise
        import traceback
        traceback.print_exc()
        print(f"VIEWER_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        os._exit(1)
    else:
        simulation_app.close()
