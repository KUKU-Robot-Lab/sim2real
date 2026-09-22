"""읽기 전용 Isaac Sim 뷰어 — right_aglt 정책의 학습 장면에 실기 관절 상태를 비춘다.

hdgp 의 태스크 env 를 **그대로**(num_envs=1) 만들고, 정책 번들의 `params/env.yaml` 을 hdgp play.py 와
같은 복원기(`hdgp/scripts/tools/run_cfg_restore.py`, 읽기만)로 덮는다. 그 뒤 env.step 은 **한 번도 부르지
않는다** — 정책·액션·물리 스텝 없음. 매 프레임 UDP 로 받은 관절 값을 articulation 에 쓰고(속도 0)
`sim.render()` 만 돈다(render 가 `update_articulations_kinematic` 로 링크 자세를 갱신한다).
ROS 는 이 프로세스에 없다(relay 가 UDP 로 넘긴다).

    /home/user/rl_ws/IsaacLab/isaaclab.sh -p isaac_viewer.py            # GUI
    ... isaac_viewer.py --headless --shot /tmp/x.png --exit_after_shot   # 검증용 한 장
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
RL_WS = HERE.parents[3]
HDGP = RL_WS / "hdgp"
DEFAULT_POLICY_DIR = RL_WS / "sim2real" / "deploy" / "policies" / "right_aglt"
sys.path.insert(0, str(HERE))
import packet  # noqa: E402

parser = argparse.ArgumentParser(description="읽기 전용 실기 미러 뷰어(정책 학습 장면)")
parser.add_argument("--policy_dir", default=str(DEFAULT_POLICY_DIR), help="policy.yaml + params/env.yaml 이 있는 번들")
parser.add_argument("--port", type=int, default=packet.DEFAULT_PORT, help="UDP 수신 포트(127.0.0.1)")
parser.add_argument("--cup", default="cup_big_s100",
                    help="장면에 둘 컵 종(env.yaml 의 cup_family 8종 중 하나). 실물 빨간 컵 = cup_big_s100(config/objects.yaml)")
parser.add_argument("--max_hz", type=float, default=60.0, help="렌더 루프 상한 Hz")
parser.add_argument("--shot", default=None, help="PNG 경로 — 첫 패킷 적용 후(또는 대기 만료 후) 한 장 저장")
parser.add_argument("--shot_wait_s", type=float, default=20.0, help="--shot 전 첫 패킷 대기 상한")
parser.add_argument("--exit_after_shot", action="store_true")
parser.add_argument("--max_seconds", type=float, default=0.0, help=">0 이면 이 시간 뒤 종료(자동 시험용)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.shot:
    args.enable_cameras = True              # rgb_array 렌더(뷰포트 annotator)에 필요
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: E402,F401
# tasks/__init__ 가 등록 ImportError 를 삼키므로 명시 import 해 드러낸다(hdgp probe 들과 같은 방식).
import openarm.agnostic.tasks.grasp_fj_t2r.config  # noqa: E402,F401
from openarm.agnostic.modules import object_bank as _object_bank  # noqa: E402

sys.path.insert(0, str(HDGP / "scripts" / "tools"))
from run_cfg_restore import apply_logged_env_cfg, load_run_yaml, rebase_logged_paths  # noqa: E402


def _log(msg: str) -> None:
    print(f"[isaac_viewer] {msg}", flush=True)


def _load_policy(policy_dir: Path) -> tuple[str, dict]:
    meta = yaml.safe_load((policy_dir / "policy.yaml").read_text())
    task = str(meta["task"])
    env_yaml = policy_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        raise FileNotFoundError(f"정책 env.yaml 없음: {env_yaml}")
    logged = rebase_logged_paths(load_run_yaml(str(env_yaml)), workspace_root=str(RL_WS))
    return task, logged


def _require_file(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} 자산이 없다(경로 리베이스 실패?): {path}")
    return path


def build_cfg(task: str, logged: dict, cup: str, device: str):
    """등록 cfg -> env.yaml 복원 -> 뷰어 전용 최소 변경(num_envs=1·컵 종 1개·지령 마커 끔)."""
    cfg = parse_env_cfg(task, device=device, num_envs=1)
    apply_logged_env_cfg(cfg, logged)
    cfg.scene.num_envs = 1                              # env.yaml 은 4096
    cfg.enable_cmd_markers = False                      # 스텝을 안 하므로 지령 마커는 리셋 목표에 멈춰 오해를 준다
    # 카메라: env.yaml 의 gui_camera_eye/target(env-local) 을 뷰어 시작 시점으로.
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = tuple(cfg.gui_camera_eye)
    cfg.viewer.lookat = tuple(cfg.gui_camera_target)

    bank = _object_bank.get(cfg.object_bank)
    ids = [s.id for s in bank.specs]
    if cup not in ids:
        raise SystemExit(f"--cup {cup!r} 는 {cfg.object_bank} 에 없다: {ids}")
    k = ids.index(cup)
    spec = bank.specs[k]
    assets = list(cfg.object_cfg.spawn.assets_cfg)
    if len(assets) != len(ids):
        raise RuntimeError(f"env.yaml 컵 변형 {len(assets)} 개 != 뱅크 {len(ids)} 종")
    chosen = assets[k]
    if tuple(float(v) for v in (chosen.scale or (1.0, 1.0, 1.0))) != tuple(float(v) for v in spec.scale):
        raise RuntimeError(f"env.yaml 변형 {k} scale {chosen.scale} != 뱅크 {cup} scale {spec.scale}")
    cfg.object_cfg.spawn.assets_cfg = [chosen]          # env.yaml 의 그 항목 그대로(경로·스케일·질량)
    _require_file(cfg.robot_cfg.spawn.usd_path, "로봇")
    _require_file(cfg.table_cfg.spawn.usd_path, "테이블")
    _require_file(chosen.usd_path, "컵")
    return cfg, spec, chosen


def cup_default_pose(cfg, spec) -> tuple[float, ...]:
    """학습 리셋이 컵을 놓는 자리의 **평균**: 소환 중심(랜덤 ± spawn_range 의 중앙) · 정착고.

    env `_reset_idx`: xy = 소환 중심 + U(±spawn_range), z(정착) = table_surface_z + 종별 원점 오프셋, 회전 항등.
    """
    cx, cy = (float(v) for v in cfg.object_spawn_center_override)
    z = float(cfg.table_surface_z) + float(spec.origin_offset_z)
    return (cx, cy, z, 1.0, 0.0, 0.0, 0.0)


def print_value_table(cfg, spec, chosen, cup_pose, policy_dir: Path) -> None:
    rows = [
        ("task", "policy.yaml task", cfg.__class__.__name__),
        ("robot usd", "env.yaml robot_cfg.spawn.usd_path", cfg.robot_cfg.spawn.usd_path),
        ("robot base pos/rot", "env.yaml robot_cfg.init_state", f"{tuple(cfg.robot_cfg.init_state.pos)} {tuple(cfg.robot_cfg.init_state.rot)}"),
        ("table usd", "env.yaml table_cfg.spawn.usd_path", cfg.table_cfg.spawn.usd_path),
        ("table pos/rot", "env.yaml table_cfg.init_state", f"{tuple(cfg.table_cfg.init_state.pos)} {tuple(cfg.table_cfg.init_state.rot)}"),
        ("table_surface_z", "env.yaml table_surface_z", cfg.table_surface_z),
        ("cup species", "--cup / object_bank", f"{spec.id} (bank {cfg.object_bank})"),
        ("cup usd/scale/mass", "env.yaml object_cfg.spawn.assets_cfg[k]",
         f"{chosen.usd_path} {tuple(chosen.scale)} {chosen.mass_props.mass if chosen.mass_props else None}"),
        ("cup xy", "env.yaml object_spawn_center_override (spawn_range 중앙)", f"{cup_pose[:2]} (학습 랜덤 ±{cfg.spawn_range} 대신 중앙 고정)"),
        ("cup z", "env.yaml table_surface_z + 뱅크 origin_offset_z", f"{cup_pose[2]:.5f} (= {cfg.table_surface_z} + {spec.origin_offset_z:.5f})"),
        ("sim dt / render_interval", "env.yaml sim", f"{cfg.sim.dt} / {cfg.sim.render_interval} (스텝 안 함)"),
    ]
    _log(f"장면 값 출처 (정책 번들 {policy_dir}):")
    for name, src, val in rows:
        print(f"    {name:<22} | {src:<52} | {val}", flush=True)


class UdpLatest:
    """논블로킹 UDP 수신 — 한 프레임에 쌓인 것 중 **마지막 유효 패킷**만 쓴다."""

    def __init__(self, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.received = 0
        self.rejected = 0

    def poll(self):
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(packet.MAX_DATAGRAM_BYTES + 1)
            except BlockingIOError:
                return latest
            try:
                latest = packet.decode(data)
                self.received += 1
            except ValueError as exc:
                self.rejected += 1
                if self.rejected <= 5:
                    _log(f"패킷 버림: {exc}")


class RobotMirror:
    """UDP 관절 값 -> articulation 관절 상태(속도 0). 자산에 없는 이름은 무시하고 한 번만 알린다."""

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.robot
        self.index = {n: i for i, n in enumerate(self.robot.joint_names)}
        lim = self.robot.data.soft_joint_pos_limits[0]
        self.lo, self.hi = lim[:, 0].clone(), lim[:, 1].clone()
        self.ignored: set[str] = set()
        self.clamped: set[str] = set()
        self.applied_names: tuple[str, ...] = ()

    def apply(self, pkt) -> int:
        names, vals = [], []
        for n, v in zip(pkt.names, pkt.positions):
            if n in self.index:
                names.append(n)
                vals.append(v)
            elif n not in self.ignored:
                self.ignored.add(n)
                _log(f"자산에 없는 관절 무시: {n}")
        if not names:
            return 0
        dev = self.env.device
        ids = torch.tensor([self.index[n] for n in names], device=dev, dtype=torch.long)
        q = torch.tensor(vals, device=dev, dtype=torch.float32)
        qc = torch.clamp(q, self.lo[ids], self.hi[ids])
        over = (qc - q).abs() > 1e-4
        for j in torch.nonzero(over).flatten().tolist():
            if names[j] not in self.clamped:
                self.clamped.add(names[j])
                _log(f"한계 밖 값 표시용 clamp: {names[j]} {vals[j]:+.4f} -> [{float(self.lo[ids[j]]):+.4f}, "
                     f"{float(self.hi[ids[j]]):+.4f}]")
        self.robot.write_joint_state_to_sim(qc.unsqueeze(0), torch.zeros_like(qc).unsqueeze(0), joint_ids=ids)
        self.applied_names = tuple(names)
        return len(names)


def pin_cup(env, cup_pose) -> None:
    root = torch.zeros(1, 13, device=env.device)
    root[0, :7] = torch.tensor(cup_pose, device=env.device)
    root[0, :3] += env.scene.env_origins[0]
    env.object.write_root_state_to_sim(root)


def _save_png(img, path: str) -> str:
    import numpy as np
    arr = np.asarray(img)[..., :3]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import cv2
        cv2.imwrite(path, arr[..., ::-1])
    return path


def main() -> None:
    policy_dir = Path(args.policy_dir).resolve()
    task, logged = _load_policy(policy_dir)
    cfg, spec, chosen = build_cfg(task, logged, args.cup, args.device)
    cup_pose = cup_default_pose(cfg, spec)
    print_value_table(cfg, spec, chosen, cup_pose, policy_dir)

    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.shot else None).unwrapped
    env.reset()                                          # 학습과 같은 리셋(로봇 시작 자세 등) — 이후 step 없음
    pin_cup(env, cup_pose)
    mirror = RobotMirror(env)
    udp = UdpLatest(args.port)
    _log(f"준비 — task {task} · 관절 {len(mirror.index)} 개 · udp://127.0.0.1:{args.port} 대기 · 물리 스텝 없음")

    palm_idx = int(env.palm_idx)
    period = 1.0 / max(args.max_hz, 1.0)
    t0 = last_status = time.monotonic()
    last_pkt_t, applied, shot_done = None, 0, False
    first_applied_at = None
    while simulation_app.is_running():
        tick = time.monotonic()
        pkt = udp.poll()
        if pkt is not None:
            applied = mirror.apply(pkt)
            last_pkt_t = pkt.t
            if applied and first_applied_at is None:
                first_applied_at = tick
                _log(f"첫 패킷 적용: {applied} 관절 {list(mirror.applied_names)}")
        env.sim.render()

        if args.shot and not shot_done:
            waited = tick - t0
            if (first_applied_at is not None and tick - first_applied_at > 1.0) or waited > args.shot_wait_s:
                img = None
                for _ in range(6):                        # annotator 반영에 몇 프레임 필요
                    img = env.render()
                path = _save_png(img, args.shot)
                q = env.robot.data.joint_pos[0]
                link = env.robot.root_physx_view.get_link_transforms()[0, palm_idx, :3] - env.scene.env_origins[0]
                cup = env.object.root_physx_view.get_transforms()[0, :3] - env.scene.env_origins[0]
                summary = {
                    "shot": path, "task": task, "cup": spec.id, "packets": udp.received,
                    "applied_joints": list(mirror.applied_names),
                    "joint_pos_sim": {n: round(float(q[i]), 4) for n, i in mirror.index.items()
                                      if n in mirror.applied_names},
                    "palm_link_pos_env": [round(float(v), 4) for v in link],
                    "cup_pos_env": [round(float(v), 4) for v in cup],
                    "cup_pose_target": [round(v, 5) for v in cup_pose],
                    "ignored": sorted(mirror.ignored), "clamped": sorted(mirror.clamped),
                }
                print("VIEWER_SHOT " + json.dumps(summary, ensure_ascii=False), flush=True)
                shot_done = True
                if args.exit_after_shot:
                    break

        if tick - last_status >= 5.0:
            last_status = tick
            age = f"{time.time() - last_pkt_t:.2f}s" if last_pkt_t else "-"
            _log(f"packets={udp.received} rejected={udp.rejected} applied={applied} last_age={age}")
        if args.max_seconds > 0 and tick - t0 > args.max_seconds:
            _log("max_seconds 도달 — 종료")
            break
        rest = period - (time.monotonic() - tick)
        if rest > 0:
            time.sleep(rest)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
