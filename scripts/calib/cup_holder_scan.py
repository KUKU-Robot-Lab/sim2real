#!/usr/bin/env python3
"""컵홀더 세 개를 목 스캔으로 찾아 홀더 자세 파일(place 에피소드의 holder_poses)을 쓴다.

10.20 사용자: "컵홀더는 다시 카메라 모터를 사용해서 테이블 전체 스캐닝해서 위치 뽑아 내야함."
옛 미션 cup_holders 단계는 고정 머리 자세 한 장(cup_holder_pose_node)이었다. 여기서는 머리를 여러 자세로 돌린다.

  1. 자세: 보정 프로필(기본 v9_1014_anchored_z)의 각도 표에서만 고른다(10.15 사용자: 표에 없는 각도는 좌표가 틀린다).
     테이블 위 홀더 후보 위치 격자의 마커가 자세마다 영상 안에 드는지 보고, 모든 칸이 MIN_VIEWS 번 보일 때까지
     탐욕으로 고른 뒤 pan 순서로 한 번 훑는다.
  2. 자세마다: head_move.py 로 이동(드라이버가 정착 확인) -> 실측 각도가 표 안인지(밖이면 그 자세는 버림)
     -> vision-3090 컬러 영상 -> cup_holder_pose.estimate(ArUco, 무늬 추적, 무늬 전체 탐색) -> 홀더별 x, y, z, yaw.
  3. 보정은 object_pose_node(PoseConverter.convert)와 같은 순서: 실측 각도 FK + 프로필 hand-eye -> + 각도별 bias
     -> anchor -> snap_z. yaw 는 anchor rot_deg 만큼 돌린다. 결과는 FP++ 컵 자세와 같은 프레임(pan 0 / tilt -20 정적 외부 파라미터).
  4. 홀더별 중앙값(yaw 는 원형 중앙값). 중앙값에서 먼 관측은 버리고, 세 홀더 모두 서로 다른 자세 MIN_POSES 곳 이상에서
     일관되게 보여야 파일을 쓴다. 하나라도 모자라면 쓰지 않는다(옛 파일은 그대로).
  5. 끝나면(실패해도) 기준 자세(pan 0 / tilt -20)로 돌아간다.

    python3 scripts/calib/cup_holder_scan.py              # 계획만 출력(목 안 움직임)
    python3 scripts/calib/cup_holder_scan.py --execute    # 목이 움직인다. 사용자 허락 뒤에만(허락은 한 번에만 유효)

전제: 목 드라이버(head_joint_publisher)와 vision-3090 카메라 노드(scripts/vision/camera_up.sh)가 떠 있어야 한다.
목·카메라 포트는 perception_plus_plus head_scan.adapters(Sim2RealHead, SshCamera)를 그대로 쓴다.
종료 코드: 0 기록함, 1 스캔은 했지만 기록 안 함(관측 부족·일관성 실패), 2 실행 전·중 거절(자세·목·카메라).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
SIM2REAL = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import calib_profile as cp  # noqa: E402
import head_motion as hm  # noqa: E402
from cup_holder_pose import (AXES, DEFAULT_CFG, SHARED_FAIL_PX, SHARED_WARN_SPREAD, HolderCfg, estimate,  # noqa: E402
                             load_cfg, wrap, write_poses, yaw_quat_wxyz)
from head_camera_pose import HOME_PAN_DEG, HOME_TILT_DEG, base_cam_pose  # noqa: E402
from marker_template_fit import CELLS  # noqa: E402
from table_cad_extrinsics import TABLE_TOP_Z, TABLE_X, TABLE_Y, T_from, project  # noqa: E402

PERCEPTION_PP = SIM2REAL.parent / "perception_plus_plus"
DEFAULT_PROFILE = "v9_1014_anchored_z"
#: 미션 생성기 STATIONS["5090"]["holder_poses"] 와 같은 파일(.gitignore, 자리마다 다르다)
DEFAULT_OUT = SIM2REAL / "config/cup_holder_poses_5090_v1.yaml"
DEFAULT_HOST = "vision-3090"
#: 보정이 가장 촘촘한 줄(각도마다 3~4 장, 볼트 30~47 점). -25·-30 줄은 1~2 장, 6~20 점
DEFAULT_TILT_DEG = -20.0
#: 계획용 공칭 카메라. 실행 때는 영상마다 camera_info K 를 쓴다
CAMERA_SPEC = SIM2REAL / "config/head_camera_sim.json"
HOME = (HOME_PAN_DEG, HOME_TILT_DEG)

MIN_VIEWS = 2                 # 계획: 격자 칸마다 이만큼의 자세에서 보이게(닿을 수 있는 만큼)
MIN_POSES = 2                 # 기록: 홀더마다 서로 다른 자세 이만큼에서 일관되게 보여야
XY_TOL_M = 0.006              # 중앙값에서 이보다 먼 관측은 버린다(각도별 bias 폭 ~3 mm 의 두 배)
YAW_TOL_RAD = math.radians(5.0)
EDGE_PX = 20                  # 마커 네 모서리가 영상 가장자리에서 이만큼 안쪽이어야 보인다고 본다
GRID_STEP_M = 0.04
#: 클램프 안쪽면(STL x +90 mm)이 상판 끝에 닿는다(config/cup_holders.yaml) -> 원점 x 는 상판 끝 - 0.09 까지
CLAMP_INNER_X_M = 0.090
HOLDER_X = (0.10, TABLE_X[1] - CLAMP_INNER_X_M)
HOLDER_Y_MARGIN_M = 0.05      # 링 반지름 40 mm + 여유
#: 마커 한 칸(테두리 포함 8x8 중 하나)이 이보다 작으면 ArUco 는 못 읽고 무늬 맞춤도 검증한 범위 밖이다
#  (10.02 arm4090 1280x720 무늬 맞춤 성공 때 ~5 x 2.3 px)
MIN_CELL_PX = 2.0
NICE = 10                     # 무늬 맞춤 계산이 팔·손 제어 프로세스에 밀리게
#: 시간 추정(실측 아님): 자세당 이동·정착(head_move 시작 + 정착 창 1 s 이상), 영상 한 장(정지 대기 1 s + ssh + scp)
MOVE_EST_S = 3.0
CAPTURE_EST_S = 4.0


# ── 자세 계획 ────────────────────────────────────────────────────────────
def nominal_camera(path: Path = CAMERA_SPEC) -> tuple[np.ndarray, tuple[int, int]]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(spec["intrinsic_matrix"], float).reshape(3, 3), (int(spec["width"]), int(spec["height"]))


def T_base_cam(profile: cp.CalibProfile, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """실측 인코더 각 -> base 기준 카메라(optical). object_pose_node 의 extrinsics_at_head 와 같은 사슬(프로필 hand-eye)."""
    pos, quat = base_cam_pose(pan_deg, tilt_deg, profile.neck_to_camera)
    return T_from(pos, quat)


def fk_table_z(profile: cp.CalibProfile) -> float:
    """FK(anchor 전) 프레임의 상판 높이. anchor 가 z 를 dz 만큼 옮기므로 미리 빼 두면 보정 뒤 상판이 known_z 에 온다."""
    top = profile.known_z.table_top_m if profile.known_z is not None else TABLE_TOP_Z
    return top - (profile.anchor.dz_m if profile.anchor is not None else 0.0)


def candidate_poses(profile: cp.CalibProfile, tilt_deg: float | None) -> list[tuple[float, float]]:
    """각도 표 -> 명령 각도(보정 때처럼 정수 deg). tilt_deg 를 주면 그 줄(+-1 deg)만. 반올림한 명령도 표 안이어야 한다."""
    out = {(float(round(e.pan_deg)), float(round(e.tilt_deg))) for e in profile.entries
           if tilt_deg is None or abs(e.tilt_deg - tilt_deg) <= 1.0}
    return sorted(c for c in out if cp.lookup(profile, *c) is not None)


def holder_grid(step: float = GRID_STEP_M) -> np.ndarray:
    """홀더 원점 후보 (N,2): 상판 전체에서 클램프가 닿을 수 있는 x 까지."""
    xs = np.arange(HOLDER_X[0], HOLDER_X[1] + 1e-9, step)
    ys = np.arange(TABLE_Y[0] + HOLDER_Y_MARGIN_M, TABLE_Y[1] - HOLDER_Y_MARGIN_M + 1e-9, step)
    return np.array([(x, y) for x in xs for y in ys])


def marker_corners_px(cfg: HolderCfg, K: np.ndarray, T_bc: np.ndarray, xy: np.ndarray,
                      z_origin: float) -> tuple[np.ndarray, np.ndarray]:
    """yaw 0(마커가 로봇 쪽 -x 를 봄)으로 선 홀더들의 마커 네 모서리 px (N,4,2) 와 깊이 (N,4)."""
    X = np.concatenate([cfg.corners_stl + np.array([x, y, z_origin]) for x, y in np.asarray(xy, float)])
    uv, depth = project(T_bc, K, X)
    return uv.reshape(-1, 4, 2), depth.reshape(-1, 4)


def visible(uv: np.ndarray, depth: np.ndarray, size: tuple[int, int], edge_px: float = EDGE_PX) -> np.ndarray:
    w, h = size
    inside = (uv[..., 0] >= edge_px) & (uv[..., 0] <= w - edge_px) & (uv[..., 1] >= edge_px) & (uv[..., 1] <= h - edge_px)
    return (inside & (depth > 0)).all(axis=1)


def cell_px(uv: np.ndarray) -> tuple[float, float]:
    """마커 한 칸의 가로(TL-TR)·세로(TL-BL) px, 주어진 마커들 중 가장 작은 값."""
    w = np.linalg.norm(uv[:, 1] - uv[:, 0], axis=1) / CELLS
    h = np.linalg.norm(uv[:, 3] - uv[:, 0], axis=1) / CELLS
    return float(w.min()), float(h.min())


def clamp_row(xy: np.ndarray) -> np.ndarray:
    """격자 중 클램프가 상판 끝에 닿는 줄(x = HOLDER_X[1]). 홀더가 실제로 놓이는 곳이라 마커 크기는 여기서 본다."""
    return np.asarray(xy)[:, 0] >= HOLDER_X[1] - 1e-6


def visibility(profile: cp.CalibProfile, cfg: HolderCfg, K: np.ndarray, size: tuple[int, int],
               poses: list[tuple[float, float]], edge_px: float = EDGE_PX) -> tuple[np.ndarray, dict]:
    """자세 x 격자 칸 보임(bool) 과 자세별 클램프 줄의 가장 작은 마커 칸 px(그 줄이 보일 때만)."""
    xy = holder_grid()
    row = clamp_row(xy)
    z = fk_table_z(profile) - cfg.base_bottom_m
    rows, cells = [], {}
    for pose in poses:
        uv, depth = marker_corners_px(cfg, K, T_base_cam(profile, *pose), xy, z)
        v = visible(uv, depth, size, edge_px)
        rows.append(v)
        if (v & row).any():
            cells[pose] = cell_px(uv[v & row])
    return np.array(rows, bool).reshape(len(poses), len(xy)), cells


@dataclass(frozen=True)
class Plan:
    poses: tuple[tuple[float, float], ...]   # 명령 각도, pan 오름차순
    views: np.ndarray                        # 격자 칸마다 고른 자세 중 보이는 수
    reachable: np.ndarray                    # 후보 전부를 써도 보이는 수(목표 상한)
    cells: dict                              # 자세 -> 가장 작은 마커 칸 (가로, 세로) px


def plan_scan(profile: cp.CalibProfile, cfg: HolderCfg, K: np.ndarray, size: tuple[int, int], *,
              tilt_deg: float | None = DEFAULT_TILT_DEG, min_views: int = MIN_VIEWS, edge_px: float = EDGE_PX) -> Plan:
    """탐욕 다중 덮기: 칸마다 min(닿을 수 있는 수, min_views) 번 보일 때까지. 같은 이득이면 볼트가 많은(보정이 단단한) 각도."""
    cands = candidate_poses(profile, tilt_deg)
    if not cands:
        raise ValueError(f"프로필 {profile.name} 에 tilt {tilt_deg} 줄 각도가 없다")
    V, cells = visibility(profile, cfg, K, size, cands, edge_px)
    goal = np.minimum(V.sum(0), min_views)
    bolts = [cp.lookup(profile, *c).n_bolts for c in cands]
    count = np.zeros(V.shape[1], int)
    chosen: list[int] = []
    while True:
        gain = [0 if i in chosen else int((np.minimum(count + V[i], goal) - np.minimum(count, goal)).sum())
                for i in range(len(cands))]
        best = max(range(len(cands)), key=lambda i: (gain[i], bolts[i]))
        if gain[best] == 0:
            break
        chosen.append(best)
        count = count + V[best]
    poses = tuple(sorted(cands[i] for i in chosen))
    return Plan(poses=poses, views=count, reachable=V.sum(0), cells={p: cells[p] for p in poses})


def check_poses(profile: cp.CalibProfile, poses: list[tuple[float, float]]) -> None:
    """움직이기 전 거부: 비었거나, tilt 가 중력 쪽 한계(head_motion.MAX_TILT_DEG) 위거나, 보정 표에 없는 각도."""
    if not poses:
        raise ValueError("자세가 없다")
    for pan, tilt in poses:
        if not (math.isfinite(pan) and math.isfinite(tilt)):
            raise ValueError(f"유한하지 않은 각도: {(pan, tilt)}")
        if tilt > hm.MAX_TILT_DEG:
            raise ValueError(f"tilt {tilt} deg 는 한계 {hm.MAX_TILT_DEG} deg 위다")
        if cp.lookup(profile, pan, tilt) is None:
            raise cp.UncalibratedAngle(f"자세 {pan:+.1f}/{tilt:+.1f} deg 는 프로필 {profile.name} 표에 없다")


def travel_deg(poses, home=HOME) -> tuple[float, float]:
    """기준 자세에서 출발해 차례로 돌고 기준 자세로 돌아오는 pan·tilt 이동 합(deg)."""
    path = [tuple(home), *poses, tuple(home)]
    return (float(sum(abs(b[0] - a[0]) for a, b in zip(path, path[1:]))),
            float(sum(abs(b[1] - a[1]) for a, b in zip(path, path[1:]))))


# ── 보정 · 모으기 ────────────────────────────────────────────────────────
def require_calibrated(profile: cp.CalibProfile, pan_deg: float, tilt_deg: float) -> cp.AngleEntry:
    """strict 가 아닌 프로필이라도 표 밖 각도는 쓰지 않는다(10.15 사용자)."""
    entry = cp.lookup(profile, pan_deg, tilt_deg)
    if entry is None:
        raise cp.UncalibratedAngle(f"목 실측 {pan_deg:+.2f}/{tilt_deg:+.2f} deg 는 프로필 {profile.name} 표"
                                   f"(+-{profile.angle_tolerance_deg} deg) 밖")
    return entry


def debias(profile: cp.CalibProfile, pose_fk, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """FK 프레임 홀더 자세(x, y, z, yaw) + 그 각도의 bias. 홀더 사이 공유 축은 이 프레임에서 맞춘다(estimate 와 같은 프레임)."""
    require_calibrated(profile, pan_deg, tilt_deg)
    p = np.asarray(pose_fk, float)
    return np.r_[p[:3] + cp.bias_for(profile, pan_deg, tilt_deg), p[3]]


def to_fpp_frame(profile: cp.CalibProfile, cfg: HolderCfg, pose) -> np.ndarray:
    """bias 뒤 자세 -> anchor -> snap_z(서 있는 물체). yaw 는 anchor 회전만큼."""
    p = np.asarray(pose, float)
    pos = cp.anchor_position(profile, p[:3])
    pos[2] = cp.snap_z(profile, pos[2], -cfg.base_bottom_m)
    rot = math.radians(profile.anchor.rot_deg) if profile.anchor is not None else 0.0
    return np.r_[pos, wrap(float(p[3]) + rot)]


def correct(profile: cp.CalibProfile, cfg: HolderCfg, pose_fk, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """object_pose_node PoseConverter.convert 와 같은 순서: + bias -> anchor -> snap_z."""
    return to_fpp_frame(profile, cfg, debias(profile, pose_fk, pan_deg, tilt_deg))


def circular_median(angles) -> float:
    """원 위 중앙값: 원형 평균 둘레로 펼쳐 중앙값을 낸다(관측이 한쪽에 몰려 있을 때 맞다). +-pi 경계를 넘어도 된다."""
    a = np.asarray(angles, float)
    ref = math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))
    d = (a - ref + math.pi) % (2 * math.pi) - math.pi
    return wrap(ref + float(np.median(d)))


@dataclass(frozen=True)
class Obs:
    marker_id: int
    shot: int              # 몇 번째 목 자세
    pose: np.ndarray       # bias 뒤 FK 프레임 x, y, z, yaw
    src: str               # aruco | track | template
    rms_px: float


@dataclass(frozen=True)
class HolderResult:
    marker_id: int
    pose: np.ndarray | None   # bias 뒤 FK 프레임. None 이면 못 씀
    n_obs: int
    n_used: int
    n_shots: int
    spread_xy_m: float = float("nan")
    spread_yaw_rad: float = float("nan")
    reason: str = ""


def _center(P: np.ndarray) -> np.ndarray:
    return np.r_[np.median(P[:, :3], axis=0), circular_median(P[:, 3])]


def aggregate(obs, marker_id: int, *, min_poses: int = MIN_POSES, xy_tol: float = XY_TOL_M,
              yaw_tol: float = YAW_TOL_RAD) -> HolderResult:
    """한 홀더의 관측 -> 중앙값. 중앙값에서 먼 관측은 버리고, 남은 관측이 서로 다른 자세 min_poses 곳 미만이면 못 씀."""
    mine = [o for o in obs if o.marker_id == marker_id]
    if not mine:
        return HolderResult(marker_id, None, 0, 0, 0, reason="한 번도 못 봤다")
    P = np.array([o.pose for o in mine])
    c = _center(P)
    keep = ((np.linalg.norm(P[:, :2] - c[:2], axis=1) <= xy_tol)
            & (np.abs([wrap(a - c[3]) for a in P[:, 3]]) <= yaw_tol))
    used = [o for o, k in zip(mine, keep) if k]
    n_shots = len({o.shot for o in used})
    if n_shots < min_poses:
        return HolderResult(marker_id, None, len(mine), len(used), n_shots,
                            reason=f"일관된 관측이 자세 {n_shots} 곳뿐(필요 {min_poses})")
    Q = np.array([o.pose for o in used])
    out = _center(Q)
    return HolderResult(marker_id, out, len(mine), len(used), n_shots,
                        spread_xy_m=float(np.max(np.linalg.norm(Q[:, :2] - out[:2], axis=1))),
                        spread_yaw_rad=float(np.max(np.abs([wrap(a - out[3]) for a in Q[:, 3]]))))


def apply_shared(poses: dict[int, np.ndarray], shared) -> tuple[dict[int, np.ndarray], dict[str, float], list[str]]:
    """공유 축(cfg.shared, 예 x: 세 클램프가 같은 상판 끝)을 홀더 사이 중앙값으로 맞춘 새 dict, 축별 폭, 거절 사유.

    폭이 SHARED_WARN_SPREAD 를 넘으면 공유 가정이 실물과 맞지 않는 것이라 거절한다(맞춰 쓰면 틀린 좌표가 된다)."""
    out = {i: np.array(p, float) for i, p in poses.items()}
    spreads, errs = {}, []
    if len(poses) < 2:
        return out, spreads, errs
    for axis in shared:
        j = AXES.index(axis)
        vals = np.array([p[j] for p in poses.values()])
        if axis == "yaw":
            v = circular_median(vals)
            spread = float(np.ptp([wrap(a - v) for a in vals]))
        else:
            v = float(np.median(vals))
            spread = float(np.ptp(vals))
        spreads[axis] = spread
        if spread > SHARED_WARN_SPREAD[axis]:
            errs.append(f"공유 축 {axis} 의 홀더 사이 폭 {spread:.4f} > {SHARED_WARN_SPREAD[axis]:.4f}: "
                        "config/cup_holders.yaml constraints.shared 가정이 실물 배치와 다르다")
        for p in out.values():
            p[j] = v
    return out, spreads, errs


# ── 스캔 ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScanLog:
    obs: tuple[Obs, ...]
    skipped: tuple[str, ...]
    frame_size: tuple[int, int] | None
    min_cell: tuple[float, float] | None    # 실제 K 로 본 클램프 줄의 가장 작은 마커 칸 px


def scan(poses, head, camera, profile: cp.CalibProfile, cfg: HolderCfg, *, n_frames: int = 1,
         use_template: bool = True, log=print) -> ScanLog:
    """자세마다 이동 -> 실측 각도 확인 -> 촬영 -> 추정 -> bias. 목·카메라 예외는 그대로 올린다(호출자가 기준 자세로 돌린다)."""
    import cv2

    obs, skipped = [], []
    hits: dict = {}
    size, cell = None, None
    table_z = fk_table_z(profile)
    grid = holder_grid()
    grid = grid[clamp_row(grid)]
    for shot, (pan, tilt) in enumerate(poses):
        reading = head.move(pan, tilt)
        pm, tm = float(reading.pan_deg), float(reading.tilt_deg)
        if cp.lookup(profile, pm, tm) is None:
            skipped.append(f"자세 {pan:+.0f}/{tilt:+.0f}: 실측 {pm:+.2f}/{tm:+.2f} deg 는 보정 표 밖, 버림")
            log(f"  [{shot + 1}/{len(poses)}] {skipped[-1]}")
            continue
        T_bc = T_base_cam(profile, pm, tm)
        for frame in camera.capture(n_frames):
            gray = cv2.cvtColor(np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2GRAY)
            K = np.asarray(frame.k, float)
            size = (gray.shape[1], gray.shape[0])
            uv, depth = marker_corners_px(cfg, K, T_bc, grid, table_z - cfg.base_bottom_m)
            v = visible(uv, depth, size)
            if v.any():
                c = cell_px(uv[v])
                cell = c if cell is None else (min(cell[0], c[0]), min(cell[1], c[1]))
            est = estimate(gray, K, T_bc, cfg, table_z=table_z, use_template=use_template, prev_hits=dict(hits) or None)
            hits = {**hits, **est.hits}
            head_txt = f"  [{shot + 1}/{len(poses)}] 목 {pm:+.2f}/{tm:+.2f}"
            if est.flat:      # ★10.20 상판 위 보정 보드의 같은 id — 누운 마커라 버렸다
                head_txt += " · 보드(누운) 마커 버림 " + ",".join(f"id{i}" for i in est.flat)
            if est.poses and est.worst_rms > SHARED_FAIL_PX:
                skipped.append(f"{head_txt.strip()}: 공유 제약 뒤 rms {est.worst_rms:.1f} px > {SHARED_FAIL_PX}, 이 장 버림")
                log(head_txt + f" 버림(rms {est.worst_rms:.1f} px)")
                continue
            found = []
            for i in sorted(est.poses):
                o = Obs(i, shot, debias(profile, est.poses[i], pm, tm), est.src[i], float(est.rms[i]))
                obs.append(o)
                q = to_fpp_frame(profile, cfg, o.pose)
                found.append(f"id{i} {o.src} ({q[0]:+.4f}, {q[1]:+.4f}) yaw {math.degrees(q[3]):+.1f} rms {o.rms_px:.1f}")
            log(head_txt + " · " + (" · ".join(found) if found else "마커 없음"))
    return ScanLog(tuple(obs), tuple(skipped), size, cell)


def summarize(log: ScanLog, profile: cp.CalibProfile, cfg: HolderCfg, min_poses: int):
    """관측 -> 홀더별 결과, 기록할 자세(FP++ 프레임, 못 쓰면 None), 거절 사유."""
    results = {i: aggregate(log.obs, i, min_poses=min_poses) for i in cfg.ids}
    errs = [f"{name}(id{i}): {results[i].reason}" for name, i in zip(cfg.names, cfg.ids) if results[i].pose is None]
    if errs:
        return results, None, errs
    shared, spreads, shared_errs = apply_shared({i: r.pose for i, r in results.items()}, cfg.shared)
    if shared_errs:
        return results, None, shared_errs
    out = {i: to_fpp_frame(profile, cfg, p) for i, p in shared.items()}
    order_err = left_to_right_problem(cfg, out)
    return results, (None if order_err else out), ([order_err] if order_err else [])


def left_to_right_problem(cfg: HolderCfg, poses: dict) -> str:
    """★10.20 홀더 번호(이름 끝 숫자)는 로봇 왼쪽(+y)부터 0·1·2 — 에피소드 LEFT_HOLDER 0 · RIGHT_HOLDER 2.
    실물을 옮겨 마커 짝이 자리와 어긋나면 사유를, 맞으면 '' 를 돌려준다."""
    by_number = [i for _, i in sorted(zip(cfg.names, cfg.ids), key=lambda t: int(t[0].rsplit("_", 1)[-1]))]
    ys = [float(poses[i][1]) for i in by_number]
    if all(a > b for a, b in zip(ys, ys[1:])):
        return ""
    real = sorted(poses, key=lambda i: -float(poses[i][1]))
    pairs = " · ".join(f"cup_holder_{n}←마커 {i}" for n, i in enumerate(real))
    return (f"홀더 번호가 로봇 왼쪽부터 순서가 아니다: 실측 왼쪽부터 마커 {'·'.join(map(str, real))} "
            f"(y {', '.join(f'{float(poses[i][1]):+.3f}' for i in real)}) → config/cup_holders.yaml 을 {pairs} 로 고친 뒤 다시 스캔")


# ── CLI ──────────────────────────────────────────────────────────────────
def head_scan_adapters():
    """perception_plus_plus 의 실기 목·카메라 포트. 실행 때만 불러온다."""
    if str(PERCEPTION_PP) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_PP))
    from perception_plus_plus_core.head_scan import adapters
    return adapters


def _print_plan(profile, poses, V, cells, n_frames: int) -> None:
    print(f"프로필 {profile.name} · 자세 {len(poses)} 개(pan 순서, 명령 각도):")
    for k, pose in enumerate(poses):
        e = cp.lookup(profile, *pose)
        c = cells.get(pose)
        ctxt = f"클램프 줄 마커 칸 최소 {c[0]:.1f} x {c[1]:.1f} px" if c else "클램프 줄 안 보임"
        print(f"  {k + 1}. pan {pose[0]:+5.0f} tilt {pose[1]:+5.0f} · 표 {e.pan_deg:+.2f}/{e.tilt_deg:+.2f} "
              f"(볼트 {e.n_bolts}, 잔차 {e.bolt_resid_mm:.1f} mm) · 격자 칸 {int(V[k].sum())} · {ctxt}")
    n = V.sum(0)
    print(f"격자 {V.shape[1]} 칸(x {HOLDER_X[0]:.2f}~{HOLDER_X[1]:.2f}, y +-{TABLE_Y[1] - HOLDER_Y_MARGIN_M:.2f} m): "
          f"{MIN_VIEWS} 번 이상 {int((n >= MIN_VIEWS).sum())} · 1 번 {int((n == 1).sum())} · 안 보임 {int((n == 0).sum())}")
    pan_t, tilt_t = travel_deg(poses)
    est_s = len(poses) * (MOVE_EST_S + n_frames * CAPTURE_EST_S) + MOVE_EST_S
    print(f"목 이동 합(기준 자세에서 출발·복귀): pan {pan_t:.0f} deg · tilt {tilt_t:.0f} deg · "
          f"예상 {est_s:.0f} s 이상(추정, 무늬 맞춤 계산 별도)")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true", help="목을 움직여 스캔하고 기록한다(사용자 허락 뒤에만)")
    ap.add_argument("--calib-profile", default=DEFAULT_PROFILE, help="config/calib_profiles 이름 또는 경로")
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tilt", type=float, default=DEFAULT_TILT_DEG, help="계획할 각도 표의 tilt 줄")
    ap.add_argument("--poses", default=None, help="계획 대신 이 자세만 'pan,tilt;pan,tilt' (표에 있어야 한다)")
    ap.add_argument("--frames", type=int, default=1, help="자세마다 찍을 장 수")
    ap.add_argument("--min-poses", type=int, default=MIN_POSES)
    ap.add_argument("--host", default=DEFAULT_HOST, help="카메라 호스트")
    ap.add_argument("--settle-s", type=float, default=1.0, help="정착 뒤 찍기 전 대기")
    ap.add_argument("--save-dir", type=Path, default=None, help="찍은 npz 사본(진단용)")
    ap.add_argument("--no-template", action="store_true", help="ArUco 가 못 찾은 마커에 무늬 맞춤을 쓰지 않는다")
    ap.add_argument("--nice", type=int, default=NICE, help="실행 때 이 프로세스 nice 증가량(0 이면 그대로)")
    return ap.parse_args(argv)


def _parse_pose_list(text: str) -> list[tuple[float, float]]:
    out = []
    for item in text.split(";"):
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"자세는 'pan,tilt' 형식이어야 한다: {item!r}")
        out.append((float(parts[0]), float(parts[1])))
    return out


def _print_results(results, poses_out, cfg: HolderCfg) -> None:
    for name, i in zip(cfg.names, cfg.ids):
        r = results[i]
        if r.pose is None:
            print(f"  {name}(id{i}): 못 씀 · 관측 {r.n_obs} · {r.reason}")
            continue
        p = poses_out[i] if poses_out else None
        where = (f"({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:.4f}) yaw {math.degrees(p[3]):+.2f} deg "
                 f"quat_wxyz {tuple(round(v, 4) for v in yaw_quat_wxyz(p[3]))}") if p is not None else "기록 안 함"
        print(f"  {name}(id{i}): {where} · 관측 {r.n_used}/{r.n_obs} · 자세 {r.n_shots} 곳 · "
              f"폭 {r.spread_xy_m * 1000:.1f} mm / {math.degrees(r.spread_yaw_rad):.1f} deg")


def main(argv=None, head=None, camera=None) -> int:
    args = parse_args(argv)
    profile = cp.load_profile(args.calib_profile)
    cfg = load_cfg(args.cfg)
    K, size = nominal_camera()
    try:
        if args.poses:
            poses = _parse_pose_list(args.poses)
        else:
            poses = list(plan_scan(profile, cfg, K, size, tilt_deg=args.tilt).poses)
        check_poses(profile, poses)
    except ValueError as exc:
        print(f"거절: {exc}", file=sys.stderr)
        return 2
    V, cells = visibility(profile, cfg, K, size, poses)
    print(f"계획(공칭 K {size[0]}x{size[1]}, {CAMERA_SPEC.name}):")
    _print_plan(profile, poses, V, cells, args.frames)
    if not args.execute:
        print("--execute 없이 실행함: 목을 움직이지 않았다")
        return 0

    if head is None or camera is None:
        adapters = head_scan_adapters()
        busy = adapters.busy_head_commanders()
        if busy:
            print("거절: 목을 명령하는 다른 프로세스가 있다\n  " + "\n  ".join(busy), file=sys.stderr)
            return 2
        head = head or adapters.Sim2RealHead(SIM2REAL)
        camera = camera or adapters.SshCamera(args.host, args.settle_s, args.save_dir)
    if args.nice:
        os.nice(args.nice)
    try:
        log = scan(poses, head, camera, profile, cfg, n_frames=args.frames, use_template=not args.no_template)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"거절: 스캔 중단, 기록 안 함 · {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            home = head.go_home()
            print(f"기준 자세 복귀: 실측 {home.pan_deg:+.2f}/{home.tilt_deg:+.2f} deg")
        except (RuntimeError, OSError) as exc:
            print(f"경고: 기준 자세 복귀 실패 · {exc}", file=sys.stderr)

    if log.frame_size is not None:
        c = log.min_cell
        print(f"영상 {log.frame_size[0]}x{log.frame_size[1]}"
              + (f" · 클램프 줄 마커 칸 최소 {c[0]:.1f} x {c[1]:.1f} px" if c else ""))
    for s in log.skipped:
        print(f"  버림: {s}")
    results, poses_out, errs = summarize(log, profile, cfg, args.min_poses)
    print("[결과] FP++ 컵 자세와 같은 프레임")
    _print_results(results, poses_out, cfg)
    if errs:
        print("기록 안 함:\n  " + "\n  ".join(errs))
        if log.min_cell is not None and log.min_cell[1] < MIN_CELL_PX:
            print(f"  참고: 마커 칸 세로 {log.min_cell[1]:.1f} px < {MIN_CELL_PX} px. ArUco 는 못 읽고 무늬 맞춤도 검증 범위 밖이다."
                  " 카메라 컬러 해상도(rgb_camera.color_profile)를 올려야 할 수 있다")
        return 1
    meta = (f"cup_holder_scan {time.strftime('%Y-%m-%d %H:%M')} · 프로필 {profile.name} · 자세 "
            + ";".join(f"{p:+.0f},{t:+.0f}" for p, t in poses)
            + f" · shared={','.join(cfg.shared) or '없음'} · 관측 "
            + ",".join(f"{i}:{results[i].n_used}/{results[i].n_shots}자세" for i in cfg.ids))
    write_poses(args.out, cfg, poses_out, meta)
    print(f"기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
