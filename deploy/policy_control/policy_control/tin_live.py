"""★11.07 tilt 직전 정지 창에서 그 에피소드의 손 안 컵 자세 T_in(손바닥 FK → 컵)을 잰다 — tilt 뒤 place 의 붙인 자세.

계산은 logs/analysis/tilt_tin_1107/tin_measure.measure 와 같다(시험 test_pc_tin_live 가 bag fixture 로 1 mm 안을 본다):
  · FP++ live_pose 표본 중 header stamp 가 앞 표본보다 새것이고 값이 바뀐 것만 '갱신'으로 센다(멈춘 값 반복 · stale 제외).
  · 팔 관절은 표본 수신 시각에 선형 보간(같은 시계끼리) → 손바닥 FK. T_in_i = FK_i⁻¹ · 컵_i, 위치 평균 + 회전 행렬 평균의 SO(3) 투영.
  · 규약: 위치 m, 쿼터니언 wxyz. 왼손 shaker 는 FP++ 원점 + 0.02·z(plan_tilt_traj rcv_z_off — 분석과 같은 틀, 기록만).
판정: 갱신 MIN_FRESH 개 이상 · 갱신 비율 ≥ 수신의 FRESH_RATIO · 창 안 흔들림(위치 JIT_POS_M · 각도 JIT_ANG_DEG) · 손바닥–컵 NEAR_M 안
· (오른손) 고정 평균과 FIX_POS_M · FIX_ANG_DEG 안. 실행기는 창 시작을 FPP_LATENCY_S 만큼 늦춰 자른다(멈추기 전 영상 제외).
하나라도 어기면 fallback(고정 평균 파일, tin_all · build_v12 산출물)으로 대신하고 그 이유를 json 에 남긴다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

FPP_LATENCY_S = 0.6      # FP++ 영상 지연(episode_ros PLACE_JUDGE_WAIT_S 근거 0.4~0.5 s) — 창 시작 + 이만큼부터 센다
WINDOW_S = FPP_LATENCY_S + 2.0   # 실행기 측정 대기(팔 정지 창 시작 ~ tilt 재생 직전, 모자라면 멈춘 채 기다린다) — 실제 창 2.0 s
MIN_FRESH = 8            # 2 s 창 · FP++ 갱신 약 6 Hz(11.06 bag 51/8.7 s) — bag 151437 정지 창에서 2.0 s 창 최소 10개
FRESH_RATIO = 0.3        # 갱신 / 수신 하한 — bag 151437 은 약 0.5(같은 프레임 재발행이 절반)
ARM_HOLD_S = 0.2         # 마지막 팔 표본 뒤 이 안의 FP++ 프레임은 끝값 유지, 더 늦은 것만 버린다
FIX_POS_M = 0.020        # 오른손 실측이 고정 평균과 이보다 멀면(위치 · 각도) 실측으로 인정하지 않는다
FIX_ANG_DEG = 15.0
FIX_CHECK_SIDES = ("right",)
JIT_POS_M = 0.005
JIT_ANG_DEG = 3.0
NEAR_M = 0.15            # hold_windows NEAR — 이보다 멀면 쥔 컵이 아닌 것을 본다
Z_OFF = {"right": 0.0, "left": 0.02}   # 분석 tin_measure.pose_T(왼손 shaker = 원점 + rcv_z_off·z)
FALLBACK_KEY = {"right": "T_in_r", "left": "T_in_l"}
ROBOT_ASSET = "openarm_rh56f1_bi_rl"    # 분석 FK 와 같은 URDF(손바닥 = <s>_hl_palm_sensor)


class TinError(ValueError):
    pass


def quat_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R: np.ndarray) -> np.ndarray:
    from policy_control.rh_place import _matrix_to_quat
    return _matrix_to_quat(np.asarray(R, float))


def mean_T(Ts: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(0))
    T = np.eye(4)
    T[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    T[:3, 3] = Ts[:, :3, 3].mean(0)
    return T


def tilt_deg(q: Sequence[float]) -> float:
    """컵 z 축과 base z 의 각도(°) — 쿼터니언 wxyz."""
    return float(np.degrees(np.arccos(np.clip(quat_to_R(q)[2, 2], -1.0, 1.0))))


def ang_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def fresh_index(hs: np.ndarray, X: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """sel 안에서 앞 표본(창 밖이어도)보다 stamp 가 크고 값이 바뀐 표본."""
    return np.array([i for i in sel if i > 0 and hs[i] > hs[i - 1] and np.abs(X[i] - X[i - 1]).max() > 0], int)


def measure(t: np.ndarray, hs: np.ndarray, X: np.ndarray, arm_t: np.ndarray, arm_q: np.ndarray,
            fk: Callable[[np.ndarray], np.ndarray], t0: float, t1: float, *, z_off: float = 0.0,
            min_fresh: int = MIN_FRESH) -> dict:
    """t · hs · X(N×7 pos+quat wxyz) = FP++ 수신 시각 · header stamp · 자세, arm_t · arm_q(M×7) = 팔 수신 시각 · 관절,
    fk(q7) → 4×4 손바닥(base). 창 [t0, t1]. 결과 dict — ok 면 T_in(4×4)."""
    t, hs, X = np.asarray(t, float), np.asarray(hs, float), np.asarray(X, float)
    arm_t, arm_q = np.asarray(arm_t, float), np.asarray(arm_q, float)
    sel = np.where((t >= t0) & (t <= t1))[0]
    fresh = fresh_index(hs, X, sel)
    late = 0
    if len(arm_t) >= 2:                      # 팔 표본이 FP++ 끝보다 조금 앞서면 끝값 유지(interp), ARM_HOLD_S 넘게 늦은 프레임만 버린다
        keep = t[fresh] <= arm_t[-1] + ARM_HOLD_S
        late, fresh = int((~keep).sum()), fresh[keep]
    res = {"ok": False, "window_s": round(float(t1 - t0), 2), "n_msgs": int(len(sel)), "n_fresh": int(len(fresh)),
           **({"n_late_dropped": late} if late else {})}
    bad = []
    if len(fresh) < max(min_fresh, FRESH_RATIO * len(sel)):
        bad.append(f"FP++ 갱신 {len(fresh)}/{len(sel)} (< {min_fresh} 또는 수신의 {FRESH_RATIO:g})")
    if len(fresh) and (len(arm_t) < 2 or t[fresh].min() < arm_t[0]):
        bad.append("팔 관절 표본이 FP++ 표본 시각을 덮지 않는다")
    if len(fresh) < 2 or bad:
        return {**res, "reason": " · ".join(bad) or "FP++ 갱신 2개 미만"}
    q = np.stack([np.interp(t[fresh], arm_t, arm_q[:, j]) for j in range(arm_q.shape[1])], 1)
    Ts = []
    for k, i in enumerate(fresh):
        C = np.eye(4)
        C[:3, :3] = quat_to_R(X[i, 3:])
        C[:3, 3] = X[i, :3] + z_off * C[:3, 2]
        Ts.append(np.linalg.inv(fk(q[k])) @ C)
    Ts = np.stack(Ts)
    Tm = mean_T(Ts)
    dpos = np.linalg.norm(Ts[:, :3, 3] - Tm[:3, 3], axis=1)
    dang = np.array([ang_deg(Tm[:3, :3], T[:3, :3]) for T in Ts])
    near = float(np.median(np.linalg.norm(Ts[:, :3, 3], axis=1)))
    res.update({"T_in": Tm.tolist(), "pos_mm": np.round(Tm[:3, 3] * 1000, 1).tolist(),
                "pos_dev_mm_max": round(float(dpos.max()) * 1000, 1), "ang_dev_deg_max": round(float(dang.max()), 1),
                "palm_dist_mm": round(near * 1000, 1)})
    if dpos.max() > JIT_POS_M:
        bad.append(f"위치 흔들림 {dpos.max() * 1000:.1f} mm > {JIT_POS_M * 1000:.0f}")
    if dang.max() > JIT_ANG_DEG:
        bad.append(f"각도 흔들림 {dang.max():.1f}° > {JIT_ANG_DEG:.0f}")
    if near > NEAR_M:
        bad.append(f"손바닥–컵 {near * 1000:.0f} mm > {NEAR_M * 1000:.0f}")
    return {**res, "ok": not bad, **({"reason": " · ".join(bad)} if bad else {})}


def load_fallback(path: Path, side: str) -> np.ndarray:
    """고정 평균 파일(tin_fixed_1107.json) → 4×4. 전체 실측 회전(T_meas_fullrot)이 있으면 그것 — 에피소드 실측(measure)과 같은
    FP++ 회전 틀이다(T 는 컵 축 둘레를 tilt 기본값으로 바꾼 tilt 계획기용)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    h = d.get(FALLBACK_KEY[side])
    if not isinstance(h, dict):
        raise TinError(f"{path}: {FALLBACK_KEY[side]} 가 없다")
    T = np.asarray(h.get("T_meas_fullrot", h.get("T")), float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-3:
        raise TinError(f"{path}: {FALLBACK_KEY[side]} 가 4×4 강체 변환이 아니다")
    return T


def fixed_gap(T: np.ndarray, F: np.ndarray) -> str:
    """실측 T 와 고정 평균 F 의 차이가 FIX_POS_M · FIX_ANG_DEG 를 넘으면 그 사유, 아니면 ""."""
    dp, da = float(np.linalg.norm(T[:3, 3] - F[:3, 3])), ang_deg(T[:3, :3], F[:3, :3])
    if dp > FIX_POS_M or da > FIX_ANG_DEG:
        return f"고정 평균과 차이 {dp * 1000:.1f} mm / {da:.1f}° (> {FIX_POS_M * 1000:.0f} mm 또는 {FIX_ANG_DEG:.0f}°)"
    return ""


def record(meas: dict, side: str, fallback: Path, **meta) -> dict:
    """측정 결과 + (실패면) fallback → 런 폴더 json 한 개의 내용. fallback 도 못 읽으면 TinError.
    오른손 실측은 고정 평균과 FIX_POS_M · FIX_ANG_DEG 안일 때만 인정한다(넘으면 fallback, 이유에 차이)."""
    if meas.get("ok") and side in FIX_CHECK_SIDES:
        why = fixed_gap(np.asarray(meas["T_in"], float), load_fallback(fallback, side))
        if why:
            meas = {**meas, "ok": False, "reason": why}
    if meas.get("ok"):
        T, src = np.asarray(meas["T_in"], float), "measured"
    else:
        T, src = load_fallback(fallback, side), "fallback"
    out = {"source": src, "side": side, "pos_m": [round(float(v), 6) for v in T[:3, 3]],
           "quat_wxyz": [round(float(v), 6) for v in R_to_quat(T[:3, :3])], "T_in": np.round(T, 6).tolist(),
           "convention": "T_in = 손바닥(FK palm_body) → 컵 FP++ 원점" + (" + 0.02·z" if Z_OFF[side] else "") + ", 위치 m, 쿼터니언 wxyz",
           "fallback_file": str(fallback), "measured": {k: v for k, v in meas.items() if k != "T_in"},
           "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
    if src == "fallback":
        out["fallback_reason"] = meas.get("reason", "")
    return out


def load_tin(path: str | Path, side: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """place 노드: 런 폴더 json → (rel_p, rel_q wxyz, 기록). 팔이 다르거나 값이 이상하면 TinError."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("side") != side:
        raise TinError(f"{path}: side {d.get('side')!r} ≠ {side!r}")
    if Z_OFF.get(side):          # shortcut: 왼손 T_in 은 shaker 원점 + 0.02·z 틀(분석과 같게, 기록용) — place 에 쓰려면 원점 틀로 되돌린다
        raise TinError(f"{path}: {side} T_in 은 FP++ 원점 + {Z_OFF[side]}·z 틀이라 붙인 자세로 쓰지 않는다")
    p, q = np.asarray(d.get("pos_m"), float).reshape(3), np.asarray(d.get("quat_wxyz"), float).reshape(4)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))) or abs(np.linalg.norm(q) - 1) > 1e-3 or np.linalg.norm(p) > NEAR_M:
        raise TinError(f"{path}: T_in 값이 이상하다 pos {p.tolist()} quat {q.tolist()}")
    return p, q / np.linalg.norm(q), d


def palm_fk(side: str, asset: str = ROBOT_ASSET) -> Callable[[np.ndarray], np.ndarray]:
    """팔 관절 7개(<s>_aj_1..7) → 4×4 손바닥(base) — 분석 · 시험과 같은 FK."""
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK
    s = side[0]
    f = UrdfChainFK(ASSETS[asset].urdf, [f"{s}_aj_{i}" for i in range(1, 8)], [], f"{s}_hl_palm_sensor", [])

    def fk(q):
        p = f.palm_pose(np.asarray(q, float), np.zeros(0))
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = p.extra["palm_rot"], p.palm_pos
        return M
    return fk
