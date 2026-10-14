"""목 카메라 보정 프로필 — hand-eye 와 목 각도별 보정을 한 파일로 묶어 고르게 한다. 순수부(ROS·하드웨어 무관).

프로필은 데이터 파일(`config/calib_profiles/<이름>.yaml`)이고, 코드는 이 모듈 하나만 읽는다. 쓰는 쪽(object_pose_node 등)은
`--calib-profile <이름|경로>` 로 고른다. 프로필을 바꿔도 코드는 안 바뀐다.

내용:
  hand_eye.neck_to_camera.matrix   `head_extrinsics.yaml` 과 같은 형식의 T_neck_cam(4x4). 이 프로필이 쓰는 hand-eye.
  angle_tolerance_deg              목 실측 각도가 표의 각도와 이만큼 안이면 그 항목을 쓴다(pan·tilt 각각).
  strict                           표에 없는 각도면 거부한다(틀린 좌표는 없느니만 못하다). false 면 보정 0 으로 통과.
  frame_anchor                     (선택) 출력 좌표계를 기존에 검증된 좌표계에 붙이는 변환: rot_deg(z 둘레), trans_xy_m, dz_m.
                                   각도별 보정까지 끝난 base 위치에 적용한다(회전 → 이동, z 는 더하기).
  angles[]                         보정한 각도: pan_deg, tilt_deg(실측 평균), bias_xy_m(그 각도에서 base xy 에 더할 값),
                                   n_frames · n_bolts · bolt_resid_mm(만든 근거).
bias 는 상판 높이(CAD 볼트)에서 잰 평행이동이다. 서 있는 물체에도 같이 더한다(높이 차이로 인한 2차 효과는 무시).
프로필은 `perception_plus_plus/scripts/make_calib_profile.py` 가 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PROFILE_DIR = Path(__file__).resolve().parents[1] / "config" / "calib_profiles"
#: 이보다 큰 xy 보정은 보정이 아니라 캘리브 오류다.
MAX_BIAS_M = 0.02
#: 검증된 기존 좌표계와의 차이는 mm~cm 이다. 이보다 크면 앵커가 아니라 잘못된 맞춤이다.
MAX_ANCHOR_ROT_DEG, MAX_ANCHOR_TRANS_M, MAX_ANCHOR_DZ_M = 3.0, 0.05, 0.02
DEFAULT_TOLERANCE_DEG = 1.0


class UncalibratedAngle(ValueError):
    """strict 프로필인데 현재 목 각도가 보정한 각도 표에 없다."""


@dataclass(frozen=True)
class AngleEntry:
    pan_deg: float
    tilt_deg: float
    bias_xy_m: tuple[float, float]
    n_frames: int = 0
    n_bolts: int = 0
    bolt_resid_mm: float = float("nan")


@dataclass(frozen=True)
class FrameAnchor:
    rot_deg: float
    trans_xy_m: tuple[float, float]
    dz_m: float
    source: str = ""


@dataclass(frozen=True)
class CalibProfile:
    name: str
    neck_to_camera: np.ndarray
    entries: tuple[AngleEntry, ...]
    angle_tolerance_deg: float = DEFAULT_TOLERANCE_DEG
    strict: bool = True
    notes: str = ""
    anchor: FrameAnchor | None = None


def _resolve(path_or_name: str | Path) -> Path:
    text = str(path_or_name)
    if "/" in text or text.endswith((".yaml", ".yml")):
        return Path(text)
    return PROFILE_DIR / f"{text}.yaml"


def load_profile(path_or_name: str | Path) -> CalibProfile:
    path = _resolve(path_or_name)
    if not path.is_file():
        raise FileNotFoundError(f"보정 프로필이 없다: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        matrix = np.array(raw["hand_eye"]["neck_to_camera"]["matrix"], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: hand_eye.neck_to_camera.matrix 가 없다") from exc
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: hand_eye 행렬은 4x4 여야 한다 — {matrix.shape}")
    entries = []
    for item in raw.get("angles") or []:
        bias = tuple(float(v) for v in item["bias_xy_m"])
        if len(bias) != 2 or max(abs(v) for v in bias) > MAX_BIAS_M:
            raise ValueError(f"{path}: bias_xy_m {bias} 가 ±{MAX_BIAS_M} m 밖이거나 길이가 2 가 아니다 — 보정이 아니라 캘리브 오류다")
        entries.append(AngleEntry(float(item["pan_deg"]), float(item["tilt_deg"]), bias, int(item.get("n_frames", 0)),
                                  int(item.get("n_bolts", 0)), float(item.get("bolt_resid_mm", float("nan")))))
    if not entries:
        raise ValueError(f"{path}: angles 가 비었다 — 각도별 보정이 없는 프로필은 쓸 수 없다")
    anchor = None
    if raw.get("frame_anchor") is not None:
        a = raw["frame_anchor"]
        anchor = FrameAnchor(float(a["rot_deg"]), tuple(float(v) for v in a["trans_xy_m"]), float(a["dz_m"]), str(a.get("source", "")))
        if (abs(anchor.rot_deg) > MAX_ANCHOR_ROT_DEG or max(abs(v) for v in anchor.trans_xy_m) > MAX_ANCHOR_TRANS_M
                or abs(anchor.dz_m) > MAX_ANCHOR_DZ_M):
            raise ValueError(f"{path}: frame_anchor {anchor} 가 한도 밖이다 — anchor 가 아니라 잘못된 맞춤")
    meta = raw.get("profile") or {}
    return CalibProfile(name=str(meta.get("name", path.stem)), neck_to_camera=matrix, entries=tuple(entries),
                        angle_tolerance_deg=float(raw.get("angle_tolerance_deg", DEFAULT_TOLERANCE_DEG)),
                        strict=bool(raw.get("strict", True)), notes=str(meta.get("notes", "")), anchor=anchor)


def lookup(profile: CalibProfile, pan_deg: float, tilt_deg: float) -> AngleEntry | None:
    """pan·tilt 가 모두 허용 오차 안인 항목 중 가장 가까운 것. 없으면 None."""
    tol = profile.angle_tolerance_deg
    near = [e for e in profile.entries if abs(e.pan_deg - pan_deg) <= tol and abs(e.tilt_deg - tilt_deg) <= tol]
    return min(near, key=lambda e: np.hypot(e.pan_deg - pan_deg, e.tilt_deg - tilt_deg)) if near else None


def bias_for(profile: CalibProfile, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """그 목 각도에서 base 위치에 더할 (dx, dy, 0). strict 인데 표에 없으면 `UncalibratedAngle`."""
    entry = lookup(profile, pan_deg, tilt_deg)
    if entry is None:
        if profile.strict:
            raise UncalibratedAngle(f"프로필 {profile.name}: 목 {pan_deg:+.2f}/{tilt_deg:+.2f}° 는 보정한 각도가 아니다 "
                                    f"(±{profile.angle_tolerance_deg}° 안: " + ", ".join(f"{e.pan_deg:+.1f}/{e.tilt_deg:+.1f}" for e in profile.entries) + ")")
        return np.zeros(3)
    return np.array([entry.bias_xy_m[0], entry.bias_xy_m[1], 0.0])


def anchor_position(profile: CalibProfile, pos: np.ndarray) -> np.ndarray:
    """각도별 보정이 끝난 base 위치에 프레임 앵커를 적용한다. 앵커가 없으면 그대로."""
    if profile.anchor is None:
        return np.asarray(pos, float)
    a = profile.anchor
    c, s = np.cos(np.radians(a.rot_deg)), np.sin(np.radians(a.rot_deg))
    out = np.asarray(pos, float).copy()
    out[0], out[1] = c * pos[0] - s * pos[1] + a.trans_xy_m[0], s * pos[0] + c * pos[1] + a.trans_xy_m[1]
    out[2] = pos[2] + a.dz_m
    return out


def save_profile(profile: CalibProfile, path: Path | str) -> None:
    body = {
        "profile": {"name": profile.name, "notes": profile.notes},
        "hand_eye": {"neck_to_camera": {"matrix": [[float(v) for v in row] for row in profile.neck_to_camera]}},
        "angle_tolerance_deg": profile.angle_tolerance_deg,
        "strict": profile.strict,
        **({"frame_anchor": {"rot_deg": profile.anchor.rot_deg, "trans_xy_m": list(profile.anchor.trans_xy_m), "dz_m": profile.anchor.dz_m,
                             "source": profile.anchor.source}} if profile.anchor else {}),
        "angles": [{"pan_deg": round(e.pan_deg, 3), "tilt_deg": round(e.tilt_deg, 3), "bias_xy_m": [round(v, 6) for v in e.bias_xy_m],
                    "n_frames": e.n_frames, "n_bolts": e.n_bolts, "bolt_resid_mm": round(e.bolt_resid_mm, 3)} for e in profile.entries],
    }
    Path(path).write_text(yaml.safe_dump(body, allow_unicode=True, sort_keys=False), encoding="utf-8")
