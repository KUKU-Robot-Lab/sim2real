"""1-B 접근 · 들기 경로의 충돌 · 관절 검사 세계 — 홈 경로 계획기와 같은 세계(home_path_world) · 같은 검사기(plan_home_path.Checker).

10.20 사용자 "1-b 세팅은 ready-carry 에 좀더 조건이 많아야함. 테이블, 자기 충돌, 관절 제약 등을 고려해야함".
  세계: RH56F1 1판 URDF 충돌 메쉬(볼록 조각) · 테이블(env USD 충돌 상자) · 뒤 고정 박스 · 옆 벽 · 몸통 · 머리 ·
        반대팔(실측 자세, 손 = 계약 home) · FP++ 컵 원기둥(바닥 → 위, 레지스트리 반지름).
  여유: 세계 · 몸통 · 머리 · 반대팔 2 cm(홈 경로와 같다) · 같은 팔 사슬 관절 2 개 이내 쌍은 관통만 금지 ·
        컵은 구간마다 다르게(이송 1 cm · 들어가기 관통 금지 — ReachChecker.cup_margin).
  관절: URDF 사슬 한계와 robot_control 프로필 한계 중 좁은 쪽에서 여유를 잰다.
움직이는 팔의 손은 계약 home_hand(편 손 — 에피소드 open_hand 가 둔 자세)로 둔다.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import home_path_world as W  # noqa: E402
import plan_home_path as PH  # noqa: E402

SIM2REAL = Path(__file__).resolve().parents[3]
RL_WS = SIM2REAL.parent
URDF_REL = "assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"      # hdgp 기준(policy.yaml 핀과 같은 파일)
ENV_YAML = SIM2REAL / "deploy/policy_control/config/home_path/rh56f1_world_env.yaml"
CONTRACT = SIM2REAL / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
PROFILE = RL_WS / "robot_control/src/robot_control/profiles/openarm_rh56f1.yaml"
WORLD_MARGIN_M = 0.02


class ReachChecker(PH.Checker):
    """컵 쌍만 여유를 따로(cup_margin). 나머지는 홈 경로 검사기 규칙 그대로.

    ★10.20 세계 쌍(테이블 · 벽 · 뒤 박스 상자, 컵 원기둥) 거리는 MuJoCo 접촉 거리 대신 움직이는 몸의 볼록 조각 꼭짓점에서 다시 잰다
    (상자 · 원기둥 밖이면 꼭짓점 최소 거리, 안이면 −가장 깊은 꼭짓점 깊이). MuJoCo 는 detect margin(8 cm) 안의 볼록-상자 쌍을 MPR 로
    재는데, 얇은 상판(두께 1 cm)에는 수평 방향 '관통'을 낸다 — 왼손 shaker 사전 파지에서 새끼 손끝 꼭짓점 최저가 상판 13.6 mm
    위인데 l_hl_pinky_sensor↔table_8 −253.7 mm 였다. MuJoCo 는 가까운 쌍을 고르는 데만 쓴다."""

    cup_margin = 0.02

    def required(self, k) -> float:
        if any(str(b).startswith("cup") for b in k):
            return self.cup_margin
        return super().required(k)

    def raw(self, q: np.ndarray) -> dict:
        out = super().raw(q)                     # 마지막 장면의 기구학이 data 에 남는다(장면은 하나)
        for k in [k for k in out if self._touches_world(k)]:
            wb, body = (k[0], k[1]) if self.w.is_world(k[0]) else (k[1], k[0])
            v = self._verts(body)
            if v is None:
                continue
            out[k] = self._cyl_dist(v, wb) if wb.startswith("cup") else self._box_dist(v, wb)
        return out

    def _verts(self, body: str) -> np.ndarray | None:
        import mujoco
        m, d = self.w.model, self.w.data
        pts = []
        for g in np.nonzero(m.geom_bodyid == m.body(body).id)[0]:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                pts.append(m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.vstack(pts) if pts else None

    def _box_dist(self, v: np.ndarray, box: str) -> float:
        b = next(x for x in self.w.table_boxes if x["name"] == box)
        lo, hi = np.asarray(b["lo"], float), np.asarray(b["hi"], float)
        out = np.maximum(lo - v, 0.0) + np.maximum(v - hi, 0.0)
        d_out = np.linalg.norm(np.maximum(lo - v, 0.0) + np.maximum(v - hi, 0.0), axis=1)
        inside = ~np.any(out > 0, axis=1)
        if inside.any():
            depth = np.minimum(v[inside] - lo, hi - v[inside]).min(axis=1)
            return -float(depth.max())
        return float(d_out.min())

    def _cyl_dist(self, v: np.ndarray, name: str) -> float:
        m = self.w.model
        g = int(np.nonzero(m.geom_bodyid == m.body(name).id)[0][0])
        c, (r, hh) = self.w.data.geom_xpos[g], m.geom_size[g][:2]
        rr = np.linalg.norm(v[:, :2] - c[:2], axis=1) - r
        dz = np.abs(v[:, 2] - c[2]) - hh
        inside = (rr < 0) & (dz < 0)
        if inside.any():
            return -float(np.minimum(-rr[inside], -dz[inside]).max())
        return float(np.sqrt(np.maximum(rr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2).min())


class ReachWorld:
    def __init__(self, side: str, hdgp: Path, q_now, q_other, cup: dict | None = None):
        other = "left" if side == "right" else "right"
        objs = ()
        if cup is not None:
            objs = ({"name": "cup", "center": ((cup["x"], cup["y"], (cup["z_lo"] + cup["z_hi"]) / 2)),
                     "radius": cup["radius"], "half_height": (cup["z_hi"] - cup["z_lo"]) / 2},)
        self.world = W.build_world(W.WorldSpec(urdf=Path(hdgp) / URDF_REL, env_yaml=ENV_YAML, side=side,
                                               detect_margin=0.08, objects=objs))
        c = json.loads(CONTRACT.read_text())
        sd, od = c["sides"][side], c["sides"][other]
        scene = {**od["home_hand"], **dict(zip(od["arm_joints"], [float(v) for v in q_other])), **sd["home_hand"],
                 "head_j_pan": 0.0, "head_j_tilt": 0.0}
        lim = W.load_profile_limits(PROFILE)
        self.lo = np.array([lim[j][0] for j in self.world.moving_joints])
        self.hi = np.array([lim[j][1] for j in self.world.moving_joints])
        self.chk = ReachChecker(self.world, [scene], WORLD_MARGIN_M, np.asarray(q_now, float), (self.lo, self.hi))

    def check(self, path: np.ndarray, cup_margin: float) -> dict:
        """경로(60 Hz 프레임 또는 꼭짓점)를 0.005 rad 간격으로 순서대로 검사 → 요약(ok · 최소 여유 · 걸린 쌍)."""
        self.chk.cup_margin = float(cup_margin)
        rep = self.chk.check_path(np.asarray(path, float))
        out = {"ok": bool(rep["ok"]), "min_clear_mm": round(float(rep["min_clear"]) * 1000, 1)
               if np.isfinite(rep["min_clear"]) else None,
               "min_clear_pair": PH.fmt_pair(rep["min_clear_pair"]), "cup_margin_mm": round(cup_margin * 1000, 1)}
        if not rep["ok"]:
            f0 = min(rep["fails"], key=lambda f: f["slack"])
            out["worst"] = {"pair": PH.fmt_pair(f0["pair"]), "dist_mm": round(float(f0["dist"]) * 1000, 1),
                            "need_mm": round((float(f0["dist"]) - float(f0["slack"])) * 1000, 1)}
        return out

    def profile_margin_deg(self, path: np.ndarray) -> np.ndarray:
        """프로필 한계 관절 여유 [deg] (N,7)."""
        q = np.asarray(path, float)
        return np.degrees(np.minimum(q - self.lo, self.hi - q))

    def edge_ok(self, a, b, cup_margin: float) -> bool:
        self.chk.cup_margin = float(cup_margin)
        return PH.edge_ok(self.chk, np.asarray(a, float), np.asarray(b, float))

    def rrt(self, start, goal, lo, hi, cup_margin: float, seed: int = 0, iters: int = 3000) -> np.ndarray | None:
        """관절 공간 RRT-Connect + 지름길(홈 경로 계획기 그대로). lo · hi = 표본 범위(관절 여유만큼 안쪽)."""
        self.chk.cup_margin = float(cup_margin)
        rng = np.random.default_rng(seed)
        path = PH.rrt_connect(self.chk, np.asarray(start, float), np.asarray(goal, float), lo, hi, rng, iters,
                              edge_step=0.02, log=lambda *_: None)
        if path is None:
            return None
        return PH.shortcut(self.chk, path, rng, edge_step=0.02)
