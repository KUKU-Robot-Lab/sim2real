"""★11.08 들기 가드 G40H8 — 오른팔 · 손 충돌 메쉬와 탁자 위 분홍 컵 원기둥 사이 틈(sim lg_hook.py 그대로 옮김).

틈 = LINKS 15개의 URDF collision STL 꼭짓점(링크마다 ~1500 개로 솎음)을 관절값 FK 로 옮겨, 원기둥(반지름 R_CUP,
반높이 HALF_H, 중심 z = TABLE_Z + HALF_H + 0.002)까지 부호 있는 거리(안쪽 음수)를 잰 최솟값.
정의 출처: vision-3090 rh_aglt_runs/eval/1108_lift_guard/code/lg_hook.py(G40H8_h120) · 실기 분석
logs/analysis/lift_sweep_1108/mesh_clear.py(signed_clear · load_meshes 같은 식).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policy_control import _paths

LINKS = ("r_al_4", "r_al_5", "r_al_6", "r_al_7", "r_hl_base", "r_hl_palm_1", "r_hl_palm_2", "r_hl_palm_sensor",
         "r_hl_thumb_1", "r_hl_thumb_2", "r_hl_thumb_3", "r_hl_index_1", "r_hl_middle_1", "r_hl_ring_1", "r_hl_pinky_1")
R_CUP = 0.030           # lg_hook R30 — source240_real 반지름
HALF_H = 0.085          # 계약 source240_real half_height
TABLE_Z = 0.205         # sim 탁자 윗면(base). 실기 scene 분홍 z 0.2899 − HALF_H = 0.2049 와 맞다
CUP_Z = TABLE_Z + HALF_H + 0.002
MAX_PTS_PER_LINK = 1500


class LiftGuardError(ValueError):
    pass


def mesh_path(filename: str) -> Path:
    """URDF mesh filename → 파일. 'file:///<어느 홈>/rl_ws/…' 는 이 호스트 rl_ws 루트(_paths.RL_WS) 기준으로 푼다."""
    s = filename.replace("file://", "")
    return _paths.RL_WS / s.split("/rl_ws/", 1)[1] if "/rl_ws/" in s else Path(s)


def read_stl(path: Path) -> np.ndarray:
    b = path.read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:512]:
        v = [list(map(float, ln.split()[1:4])) for ln in b.decode(errors="ignore").splitlines() if ln.strip().startswith("vertex")]
        return np.unique(np.asarray(v), axis=0)
    n = int(np.frombuffer(b[80:84], np.uint32)[0])
    rec = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(rec["v"].reshape(-1, 3).astype(float), axis=0)


def load_meshes(urdf: Path, links=LINKS) -> dict:
    """{링크: (P,3) 링크 프레임 꼭짓점} — lg_hook.load_meshes 와 같다. 빠진 링크 · 회전 있는 collision 은 오류."""
    out = {}
    for ln in ET.parse(urdf).getroot().findall("link"):
        if ln.get("name") not in links:
            continue
        pts = []
        for c in ln.findall("collision"):
            g = c.find("geometry")[0]
            if g.tag != "mesh":
                continue
            v = read_stl(mesh_path(g.get("filename"))) * np.asarray([float(s) for s in (g.get("scale") or "1 1 1").split()])
            o = c.find("origin")
            xyz = np.asarray([float(s) for s in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = [float(s) for s in (o.get("rpy") if o is not None else "0 0 0").split()]
            if not np.allclose(rpy, 0):
                raise LiftGuardError(f"{ln.get('name')} collision rpy {rpy} — 회전 미지원")
            pts.append(v + xyz)
        if pts:
            P = np.concatenate(pts)
            out[ln.get("name")] = P[:: max(1, len(P) // MAX_PTS_PER_LINK)]
    miss = [k for k in links if k not in out]
    if miss:
        raise LiftGuardError(f"충돌 메쉬 없는 링크 {miss} ({urdf})")
    return out


def signed_clear(p: np.ndarray, c: np.ndarray, r: float = R_CUP) -> np.ndarray:
    """p (P,3) · 원기둥 중심 c (3,) → (P,) 표면까지 거리(안쪽 음수). mesh_clear.signed_clear 와 같다."""
    h = np.linalg.norm(p[:, :2] - c[:2], axis=1)
    top, bot, z = c[2] + HALF_H, c[2] - HALF_H, p[:, 2]
    outside = np.hypot(np.maximum(h - r, 0.0), np.maximum(np.maximum(z - top, bot - z), 0.0))
    inside = -np.minimum(np.minimum(r - h, top - z), z - bot)
    return np.where((h < r) & (z < top) & (z > bot), inside, outside)


@dataclass(frozen=True)
class GuardCfg:
    """노드 파라미터 lift_guard_mm · lift_hstop_m · lift_guard_x · lift_guard_y. 둘 다 0 = 끔."""
    guard_m: float = 0.0
    hstop_m: float = 0.0
    x: float = float("nan")
    y: float = float("nan")

    def __post_init__(self) -> None:
        if not (0.0 <= self.guard_m < 0.2 and 0.0 <= self.hstop_m < 0.2):
            raise LiftGuardError(f"lift_guard_mm 0..200 · lift_hstop_m 0..0.2: {self.guard_m * 1e3} mm {self.hstop_m} m")
        if self.guard_m > 0 and not (np.isfinite(self.x) and np.isfinite(self.y)):
            raise LiftGuardError(f"lift_guard: 분홍 컵 자리 x,y 가 없다({self.x}, {self.y}) — scene snapshot 이 못 찾았다")

    @property
    def on(self) -> bool:
        return self.guard_m > 0 or self.hstop_m > 0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, CUP_Z])


class Clearance:
    """메쉬는 한 번 읽어 두고 틱마다 FK(UrdfChainFK) 로 옮겨 최소 틈 · 가장 가까운 링크를 낸다.
    유한한 값끼리만 비교한다 — 한 점이라도 비유한(관절 nan 등)이면 (nan, 그 링크)를 내고 노드가 멈춘다."""

    def __init__(self, fk, urdf: Path, links=LINKS):
        self.fk = fk
        meshes = load_meshes(Path(urdf), links)
        self.links = tuple(links)
        self.pts = [meshes[k] for k in self.links]

    def __call__(self, arm_q, hand_q, center: np.ndarray) -> tuple[float, str]:
        T = self.fk.tree.transforms(self.links, self.fk._q(arm_q, hand_q))
        best, arg = float("inf"), ""
        for k, P in zip(self.links, self.pts):
            v = signed_clear(P @ T[k][:3, :3].T + T[k][:3, 3], center)
            if not np.all(np.isfinite(v)):
                return float("nan"), k
            d = float(v.min())
            if d < best:
                best, arg = d, k
        return best, arg


if __name__ == "__main__":      # 자기 검사: 원기둥 안 · 위 · 옆 점의 부호 있는 거리
    c = np.array([0.3, -0.34, CUP_Z])
    p = np.array([[0.3, -0.34, CUP_Z], [0.3, -0.34, CUP_Z + HALF_H + 0.01], [0.3 + R_CUP + 0.02, -0.34, CUP_Z]])
    d = signed_clear(p, c)
    assert np.allclose(d, [-R_CUP, 0.01, 0.02]), d
    assert not GuardCfg().on and GuardCfg(0.04, 0.08, 0.3, -0.34).on
    try:
        GuardCfg(0.04, 0.0)
        raise AssertionError("자리 없는 가드가 통과했다")
    except LiftGuardError:
        pass
    print("lift_guard self-check ok")
