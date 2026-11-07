"""11.08 놓기 학생 실기: 움직이는 머리의 카메라 자세(base ← camera_color_optical_frame)를 /head/joint_states 로 매 프레임 구한다.

학습(hdgp distillation/student/head.py HeadModel.cam_pose)과 같은 식이다:
  T_neck_cam = T_base_neck(q0)^-1 · T_base_cam0   (q0 = 외부 파라미터를 잰 머리 자세, r20u5 yaml head_pose 틱)
  T_base_cam(q) = T_base_neck(q) · T_neck_cam
목 FK 는 scripts/head_fk_chain.py(URDF 체인)를 그대로 쓴다. /head/joint_states 는 URDF 규약 rad 이고 pan 은 인코더와 부호가 반대다
(head_fk_chain.encoder_from_urdf 로 건넌다).
처짐(11.08 place_lab: 쓴다)은 hdgp HeadModel._eff(ideal) 와 같다: tilt_eff = tilt + sag(q) − sag(q0), 표는
config/head_sag_arm4090_1030.json(hdgp student/data 사본, md5 91ef9d699334d02340277f896264cbb4). 오르내림 hyst·yaw 는
학습에서 에피소드마다 무작위라 실기는 0(가운데)으로 둔다.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from . import _paths

import head_fk_chain as fk  # noqa: E402  (scripts/, _paths 가 sys.path 에 넣는다)

EXTRINSICS = _paths.SIM2REAL / "config" / "global_camera_extrinsics_arm4090_r20u5.yaml"
# 틱 → 인코더 deg 는 드라이버(head_position_hold_node.tick_to_deg: tick/4095·360 − 180)를 쓴다. /head/joint_states 가 이 식으로 나온다.
# hdgp head.py tick_to_enc 는 (tick − 2048)·360/4096 이라 camset 자세에서 pan 0.04° 다르다(외부 파라미터 기준 자세만 바뀌는 차).
TICK_MAX = 4095
SAG_TABLE = _paths.SIM2REAL / "config" / "head_sag_arm4090_1030.json"
SAG_TABLE_MD5 = "91ef9d699334d02340277f896264cbb4"   # hdgp student/data/head_sag_arm4090_1030.json(학습 meta sag_table_md5 와 대조)
HOLDER_TOP_Z = 0.030   # hdgp rh_place_r place_geom.HOLDER_TOP_Z(홀더 원점 → 링 윗면), place_head.target 이 쓴다


def tick_to_enc_deg(tick: int) -> float:
    return int(tick) / TICK_MAX * 360.0 - 180.0


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SagTable:
    """처짐 표 양선형 보간(인코더 deg, 범위 밖은 끝값) — hdgp head.SagTable 과 같은 식."""

    def __init__(self, path: str | Path = SAG_TABLE, md5: str | None = SAG_TABLE_MD5) -> None:
        raw = Path(path).read_bytes()
        self.md5 = hashlib.md5(raw).hexdigest()
        if md5 is not None and self.md5 != md5:
            raise ValueError(f"{path}: 처짐 표 md5 {self.md5} ≠ 학습 {md5}")
        d = json.loads(raw)
        self.pan, self.tilt = np.asarray(d["pan_deg"], np.float64), np.asarray(d["tilt_deg"], np.float64)
        self.sag = np.asarray(d["sag_deg"], np.float64)
        if self.sag.shape != (len(self.pan), len(self.tilt)):
            raise ValueError(f"{path}: 처짐 표 모양 {self.sag.shape} ≠ ({len(self.pan)}, {len(self.tilt)})")

    @staticmethod
    def _idx(grid: np.ndarray, v: float) -> tuple[int, float]:
        v = min(max(v, grid[0]), grid[-1])
        i = int(np.clip(np.searchsorted(grid, v, side="right") - 1, 0, len(grid) - 2))
        return i, (v - grid[i]) / (grid[i + 1] - grid[i])

    def __call__(self, pan_enc_deg: float, tilt_enc_deg: float) -> float:
        (i, wi), (j, wj), t = self._idx(self.pan, pan_enc_deg), self._idx(self.tilt, tilt_enc_deg), self.sag
        return ((1 - wi) * ((1 - wj) * t[i, j] + wj * t[i, j + 1])
                + wi * ((1 - wj) * t[i + 1, j] + wj * t[i + 1, j + 1]))


_RAYS: dict = {}   # (K, h, w) → (h, w, 3) 광선(광학 z 1 m 당), K · 해상도가 같으면 한 번만 만든다


def floor_cut(depth, T_base_cam, K, z_m: float) -> np.ndarray:
    """움직이는 머리 바닥 컷(hdgp real_pipe.set_pose · _front 와 같은 식): base z = cam_z + d·(rays · R[2]) < z_m 인 화소 = 0.
    depth (H, W) 깊이[m] · T_base_cam 4x4(이 프레임 cam_pose) · K 3x3 OpenCV(그 해상도) · z_m = 탁자 상판 − FLOOR_MARGIN_M.
    반환은 새 배열(같은 dtype). 광선은 float32(hdgp 와 같은 정밀도)로 캐시하고 프레임마다 회전 z 행만 곱한다."""
    d = np.asarray(depth)
    h, w = d.shape
    kk = np.asarray(K, dtype=np.float64).reshape(3, 3)
    key = (kk.tobytes(), h, w)
    if key not in _RAYS:
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        _RAYS[key] = np.stack([(u - kk[0, 2]) / kk[0, 0], (v - kk[1, 2]) / kk[1, 1], np.ones_like(u)], -1).astype(np.float32)
    t = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    rz = _RAYS[key] @ t[2, :3].astype(np.float32)
    return np.where(np.float32(t[2, 3]) + d.astype(np.float32) * rz < np.float32(z_m), np.zeros_like(d), d)


def look_target(cup_center_base, holder_origin_base) -> np.ndarray:
    """머리 응시점(base) = (컵 중심 + 홀더 원점 + (0, 0, HOLDER_TOP_Z)) / 2 — hdgp place_head.PlaceHead.target(20795489, look_dz 0).
    컵 중심 = 손바닥 FK + R_palm·c(place_cup_gate.C_DEFAULT = place_head.CUP_C). 학습의 컵 점 잡음 ±CUP_NOISE_M 은 학습 판만이라 넣지 않는다."""
    return 0.5 * (np.asarray(cup_center_base, np.float64) + np.asarray(holder_origin_base, np.float64) + (0.0, 0.0, HOLDER_TOP_Z))


class HeadCamPose:
    def __init__(self, extrinsics: str | Path = EXTRINSICS, sag: bool = True, sag_table: str | Path = SAG_TABLE) -> None:
        d = yaml.safe_load(Path(extrinsics).read_text())
        cam, hp = d.get("camera"), d.get("head_pose")
        if not cam or not hp:
            raise ValueError(f"{extrinsics}: camera · head_pose 블록이 둘 다 있어야 한다(움직이는 머리는 잰 자세가 필요)")
        self.T_base_cam0 = np.eye(4)
        self.T_base_cam0[:3, :3] = quat_wxyz_to_mat(cam["orientation_wxyz"])
        self.T_base_cam0[:3, 3] = cam["position"]
        self.q0_enc_deg = (tick_to_enc_deg(hp["pan_tick"]), tick_to_enc_deg(hp["tilt_tick"]))
        self.T_neck_cam = np.linalg.inv(fk.t_base_neck_from_encoder(*self.q0_enc_deg)) @ self.T_base_cam0
        self.sag = SagTable(sag_table) if sag else None
        self.sag0 = self.sag(*self.q0_enc_deg) if self.sag is not None else 0.0

    def cam_pose_from_encoder(self, pan_enc_deg: float, tilt_enc_deg: float) -> np.ndarray:
        if self.sag is not None:   # 외부 파라미터를 잰 자세의 처짐은 이미 yaml 에 들어 있어 차이만 더한다
            tilt_enc_deg = tilt_enc_deg + self.sag(pan_enc_deg, tilt_enc_deg) - self.sag0
        return fk.t_base_neck_from_encoder(pan_enc_deg, tilt_enc_deg) @ self.T_neck_cam

    def cam_pose_from_joint_state(self, pan_rad: float, tilt_rad: float) -> np.ndarray:
        """/head/joint_states position(head_j_pan, head_j_tilt, URDF rad) → T_base_cam 4x4."""
        if not (math.isfinite(pan_rad) and math.isfinite(tilt_rad)):
            raise ValueError(f"머리 관절각이 유한하지 않다: pan {pan_rad} tilt {tilt_rad}")
        return self.cam_pose_from_encoder(*fk.encoder_from_urdf(math.degrees(pan_rad), math.degrees(tilt_rad)))
