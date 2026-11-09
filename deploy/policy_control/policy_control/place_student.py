"""11.08 놓기 학생(hdgp 20795489 · 7ffc4f8f, DISTILL_PMASK=1 · DISTILL_PHEAD=1, 과제 rh_place · rh_place_l) 실기 경로의 순수 부분.

학생 입력(hdgp mask_place.PlaceMask.post · rh_place.RhPlaceAdapter.student_obs 와 같은 순서 · 정규화):
  깊이 = 원시 aligned depth 1280x720 16UC1 → m → 이 프레임 카메라 자세의 바닥 컷(place_head_pose.floor_cut) → 블록 중앙값 160x90.
    카메라 자세 = 깊이 stamp 를 사이에 둔 /head/joint_states 두 개의 선형 보간(없으면 가장 가까운 것, |Δt| ≤ HEAD_PAIR_S).
  img2 = [dn x 홀더 마스크, dn x 컵 마스크(안 보이면 마지막 유지), 컵 보임 면], dn = d/d_max (d_min ≤ d ≤ d_max, 밖은 0).
    홀더 = CAD 투영(원해상도) − 투영 깊이 > 원시 측정 깊이 + HOLDER_OCC_M 화소(hdgp 도 바닥 컷 전 렌더 깊이로 가림을 본다) → 블록.
    컵 = /head_mask/image 목표 클래스 비트(160x90, top1_per_class) 중 손바닥 FK + R·c 기대 화소에서 GATE_PX 안인 것.
  prop = 교사 관측 앞 7 칸 · goal_pos(= goal_rel_palm + palm_pos) · prev_action · carry_on · settle_on · head_pan · head_tilt.
    머리 2 칸 = to_action(지금 목 인코더 각, pan −45~30 · tilt −2~82 → −1~1), 에피소드 첫 틱은 시작 자세 q0.
머리 규칙 = 매 틱 look_at(look_target(컵 점, 목표 홀더 원점)) 을 지금 목 각에서 풀어 /head/target_stream 으로, settle 뒤 마지막 명령 유지.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import hdgp_pin
from . import place_cup_gate as CG
from . import place_head_look as HL
from . import place_head_pose as HP
from . import place_holder_proj as HPJ
from . import rh_aglt as A
from . import rh_place as P
from . import student_mask as SM

TASKS = {"right": "rh_place", "left": "rh_place_l"}
PARTS = ("arm_q", "arm_qd", "arm_q_target", "hand_q", "hand_err", "palm_pos", "palm_rot")   # hdgp rh_place PART_NAMES
P_IMG2 = ("holder_mask_depth", "cup_mask_depth", "cup_visible_plane")                      # hdgp mask_place.P_IMG2
HEAD_JOINTS = ("head_j_pan", "head_j_tilt")
HEAD_RULE = "look_at_mid"
HEAD_VEL_DPS, HEAD_ACC_DPS2 = 60.0, 300.0   # = head_motion.STREAM_MAX_VEL_DPS · STREAM_MAX_ACC_DPS2(시험이 대조) = ckpt 평가 판
# /head/joint_states 는 head_motion 스트림 루프 20 Hz(50 ms). 실행 중 네 주기 넘게 없으면 노드가 머리 명령을 멈추고 바로 abort(gap 예산 없음)
HEAD_MAX_AGE_S = 0.2
# 깊이 stamp 와 짝 지을 목 상태의 최대 거리 · 보간할 두 상태의 최대 간격. 20 Hz 에서 상태 하나가 빠져도(간격 0.1 s) 보간한다.
# HEAD_MAX_AGE_S · student_policy.MAX_DEPTH_AGE_S(둘 다 0.2) 보다 작아야 한다(시험이 대조)
HEAD_PAIR_S = 0.15
# 이보다 미래인 stamp(목 상태 · 깊이)는 시계 어긋남 — 불량으로 버린다. arm_v2 는 카메라(realsense2_camera) · 목 드라이버 · 이 노드가
# 같은 5090 이라 시계가 하나다. 카메라 · 목 드라이버가 다른 호스트면 시계 동기(chrony) 오프셋이 50 ms 를 넘지 않아야 한다
MIN_AGE_S = -0.05
# reset 때 실제 목 각과 ckpt 시작 자세 q0 의 허용 차(인코더 deg, 축마다). 10.28 camset 실측: 목표 pan 1806 → 멈춘 자리 1825
# (19 틱 ≈ 1.7°) — 위치 제어가 이만큼 덜 가므로 2° 는 빠듯하다
HEAD_START_TOL_DEG = 3.0
TABLE_Z = 0.205       # head_depth_node.TABLE_Z(탁자 상판 base z, CAD) — ckpt floor_cut_z_m 대조용


class PlaceInputStale(A.RhAgltError):
    """깊이 · 마스크 · 목 상태가 없거나 낡음 — 노드 틱 오류(gap 예산)로 센다."""


def age_refusal(age_s: float, max_s: float, what: str) -> str | None:
    """stamp 나이 검사 — 미래(< MIN_AGE_S) · 상한 초과 · NaN 이면 사유."""
    if age_s < MIN_AGE_S:
        return f"{what} stamp 가 {-age_s:.3f} s 미래 — 시계 어긋남?"
    if not age_s <= max_s:
        return f"{what} 나이 {age_s:.3f} s > {max_s} s"
    return None


def start_refusal(q_enc, q0_enc, tol_deg: float = HEAD_START_TOL_DEG) -> str | None:
    """reset 검사 — 실제 목 각이 ckpt 시작 자세에서 tol 안인가(목은 움직이지 않는다)."""
    d = np.abs(np.asarray(q_enc, np.float64) - np.asarray(q0_enc, np.float64))
    if not np.all(d <= tol_deg):
        return (f"목 각 {np.round(q_enc, 2).tolist()} 가 학생 시작 자세 {np.round(q0_enc, 2).tolist()} 에서 "
                f"{float(np.max(d)):.2f} deg 벗어났다(> {tol_deg}) — 목을 시작 자세로 옮긴 뒤 reset")
    return None


def prop_cols(c) -> list:
    lay = dict(A.actor_layout(c))
    return ([[n, lay[n]] for n in PARTS] + [["goal_pos", 3], ["prev_action", lay["prev_action"]], ["carry_on", 1],
                                            ["settle_on", 1]] + [[n, 1] for n in ("head_pan", "head_tilt")])


def _slices(c) -> dict:
    out, i = {}, 0
    for name, w in A.actor_layout(c):
        out[name] = slice(i, i + w)
        i += w
    return out


def prop_from_obs(c, obs, carry_on: bool, settle_on: bool, head_act) -> np.ndarray:
    """교사 관측(A.build_obs) → 학생 prop. goal_pos = goal_rel_palm + palm_pos(sim env.goals.goal_pos, 체인 자리와 같다)."""
    s, o = _slices(c), np.asarray(obs, np.float64)
    goal = o[s["goal_rel_palm"]] + o[s["palm_pos"]]
    return np.concatenate([o[s[n]] for n in PARTS] + [goal, o[s["prev_action"]], [float(carry_on), float(settle_on)],
                                                       np.asarray(head_act, np.float64)]).astype(np.float32)


def palm_from_obs(c, obs) -> tuple[np.ndarray, np.ndarray]:
    s, o = _slices(c), np.asarray(obs, np.float64)
    x, y = o[s["palm_rot"]][:3], o[s["palm_rot"]][3:]
    return o[s["palm_pos"]], np.stack([x, y, np.cross(x, y)], 1)


def to_action(enc_deg) -> np.ndarray:
    """hdgp HeadModel.to_action: 안전 범위 → −1~1."""
    lo, hi = np.array([HL.PAN_LIM[0], HL.TILT_LIM[0]]), np.array([HL.PAN_LIM[1], HL.TILT_LIM[1]])
    return np.clip(2.0 * (np.asarray(enc_deg, np.float64) - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def block_k(k, b: int) -> np.ndarray:
    """원해상도 K → 블록 화소 K(블록 j 의 중심 = 원화소 b·j + (b−1)/2)."""
    k = np.asarray(k, np.float64).copy()
    k[:2, :2] /= b
    k[:2, 2] = (k[:2, 2] + 0.5) / b - 0.5
    return k


def cup_shape_of(key: str) -> str | None:
    """ckpt head.cup_c_m 키(source240 · source240_real · shaker_c_real …) → C_DEFAULT 형상 이름. 모르면 None."""
    hits = [s for s in CG.C_DEFAULT if key == s or key.startswith(s + "_")]
    return hits[0] if len(hits) == 1 else None


def check_meta(d: dict, c, extr_md5: str, z_cut_want: float) -> None:
    """ckpt 머리글 ↔ 이 배포가 옮긴 규칙. 하나라도 다르면 ValueError."""
    side = c.side().side
    want = {"kind": "depth", "task": TASKS[side], "arch": "v3", "delta": False,
            "n_obs": sum(w for _, w in prop_cols(c)), "n_act": c.action_dim}
    bad = {k: d.get(k) for k, v in want.items() if d.get(k) != v}
    if [list(x) for x in d.get("prop_cols") or ()] != prop_cols(c):
        bad["prop_cols"] = d.get("prop_cols")
    m = d.get("mask") or {}
    for k, v in (("mode", "P"), ("img2", list(P_IMG2)), ("vis_min_px", SM.VIS_MIN_PX), ("holder_occ_m", HPJ.HOLDER_OCC_M),
                 ("holder_mesh_md5", hdgp_pin.md5(HPJ.HOLDER_MESH))):
        if m.get(k) != v:
            bad[f"mask.{k}"] = m.get(k)
    h = d.get("head") if isinstance(d.get("head"), dict) else {}
    cup_c = {k: list(v) for k, v in CG.C_DEFAULT.items()}
    for k, v in (("rule", HEAD_RULE), ("look_dz", 0.0), ("holder_top_z", HP.HOLDER_TOP_Z), ("pan_lim", list(HL.PAN_LIM)),
                 ("tilt_lim", list(HL.TILT_LIM)), ("sag", True), ("sag_table_md5", HP.SAG_TABLE_MD5), ("extr_md5", extr_md5),
                 ("rate_eval_dps", HEAD_VEL_DPS), ("accel_eval_dps2", HEAD_ACC_DPS2), ("obs", ["head_pan", "head_tilt"])):
        if h.get(k) != v:
            bad[f"head.{k}"] = h.get(k)
    hc = h.get("cup_c_m") or {}
    # ★11.09 손마다 학습한 컵만 적힌다(오른손 {"source240_real": ..}) — 키는 컵 형상 이름이 앞에 붙은 policy_object 이름도 받는다
    shape = {k: cup_shape_of(k) for k in hc}
    if not hc or None in shape.values() or not all(np.allclose(hc[k], cup_c[shape[k] or ""], atol=1e-6) for k in hc):
        bad["head.cup_c_m"] = hc
    dp = d.get("depth_pre") or {}
    if "dynamic_pose" not in dp or dp.get("extrinsics_md5") != extr_md5 or dp.get("block") != HPJ.BLOCK:
        bad["depth_pre"] = {k: dp.get(k) for k in ("dynamic_pose", "extrinsics_md5", "block")}
    if dp.get("floor_cut_z_m") is None or abs(float(dp["floor_cut_z_m"]) - z_cut_want) > 1e-6:
        bad["depth_pre.floor_cut_z_m"] = dp.get("floor_cut_z_m")
    if bad:
        raise ValueError(f"놓기 학생 ckpt 형식 어긋남 {bad}")


@dataclass(frozen=True)
class Frame:
    stamp_ns: int
    depth: np.ndarray       # (90, 160) float32 m, 0 = 구멍 · 바닥
    holder: np.ndarray      # (90, 160) bool
    T_base_cam: np.ndarray  # 4x4


class HeadState:
    """/head/joint_states(URDF rad) → 인코더 deg 기록. latest(now) · at(stamp) 는 낡거나 없으면 PlaceInputStale.
    on_state(ROS 콜백)와 at(깊이 일꾼 스레드)이 다른 스레드라 buf 는 lock 안에서만 만진다. 비유한 · 미래 stamp 는 불량으로 센다."""

    def __init__(self, max_age_s: float = HEAD_MAX_AGE_S, keep: int = 64) -> None:
        self.max_age_s, self.buf = float(max_age_s), deque(maxlen=keep)
        self._lock = threading.Lock()
        self.bad, self.bad_why = 0, ""
        self.dt_ms: float | None = None          # 마지막 at() 에서 고른 상태와 깊이 stamp 의 시각 차(보간이면 먼 쪽)

    def _bad(self, why: str) -> None:
        self.bad += 1
        self.bad_why = why

    def on_state(self, stamp_ns: int, names, pos, now_ns: int | None = None) -> None:
        d = dict(zip(names, pos))
        if not all(j in d for j in HEAD_JOINTS):
            return
        pan, tilt = (float(d[j]) for j in HEAD_JOINTS)
        if not (math.isfinite(pan) and math.isfinite(tilt)):
            return self._bad(f"유한하지 않다: {pan} {tilt}")
        age = ((time.time_ns() if now_ns is None else now_ns) - stamp_ns) * 1e-9
        if age < MIN_AGE_S:                                # 미래 stamp 를 받으면 뒤의 정상 상태가 모두 순서 역전으로 버려진다
            return self._bad(f"stamp 가 {-age:.3f} s 미래")
        self.bad = 0
        q = np.array(HP.fk.encoder_from_urdf(math.degrees(pan), math.degrees(tilt)), np.float64)
        with self._lock:
            if self.buf and stamp_ns <= self.buf[-1][0]:
                return                                     # 순서가 뒤집힌 것은 버린다(보간 가정)
            self.buf.append((int(stamp_ns), q))

    def latest(self, now_ns: int) -> np.ndarray:
        with self._lock:
            last = self.buf[-1] if self.buf else None
        if last is None:
            raise PlaceInputStale("/head/joint_states 를 아직 못 받았다 — head_motion 스트림이 도는가")
        age = (now_ns - last[0]) * 1e-9
        if age > self.max_age_s:
            raise PlaceInputStale(f"/head/joint_states {age:.2f} s 전(> {self.max_age_s} s) — 목 드라이버 단절?")
        return last[1]

    def at(self, stamp_ns: int) -> np.ndarray:
        """stamp 를 사이에 둔 두 상태(간격 ≤ HEAD_PAIR_S)의 선형 보간, 한쪽뿐이면 HEAD_PAIR_S 안의 가장 가까운 것."""
        with self._lock:
            buf = list(self.buf)
        if not buf:
            raise PlaceInputStale("/head/joint_states 없음 — 깊이 프레임 자세를 못 정한다")
        for (t0, q0), (t1, q1) in zip(buf[:-1], buf[1:]):
            if t0 <= stamp_ns <= t1:
                if (t1 - t0) * 1e-9 > HEAD_PAIR_S:
                    raise PlaceInputStale(f"깊이 stamp 를 사이에 둔 목 상태 간격 {(t1 - t0) * 1e-9:.3f} s(> {HEAD_PAIR_S}) — 끊김")
                self.dt_ms = max(stamp_ns - t0, t1 - stamp_ns) * 1e-6
                w = (stamp_ns - t0) / max(t1 - t0, 1)
                return q0 + w * (q1 - q0)
        t, q = min(buf, key=lambda x: abs(x[0] - stamp_ns))   # 깊이가 마지막 상태보다 앞서거나 첫 상태보다 이를 때
        if abs(t - stamp_ns) * 1e-9 > HEAD_PAIR_S:
            raise PlaceInputStale(f"깊이 stamp 와 가장 가까운 목 상태가 {abs(t - stamp_ns) * 1e-9:.3f} s 떨어졌다(> {HEAD_PAIR_S})")
        self.dt_ms = abs(t - stamp_ns) * 1e-6
        return q


class HeadLook:
    """머리 규칙 한 틱 → /head/target_stream JSON. hold(settle 뒤)면 마지막 명령을 다시 보낸다(연속 모드 유지)."""

    def __init__(self, hc: HP.HeadCamPose) -> None:
        self.hc, self.cmd, self.seq = hc, None, 0

    def reset(self) -> None:
        self.cmd = None

    def command(self, q_now_enc, cup_center, holder_origin, hold: bool) -> str:
        if not hold or self.cmd is None:
            self.cmd = HL.look_at(self.hc, HP.look_target(cup_center, holder_origin), q_now_enc)
        self.seq += 1
        return json.dumps({"seq": self.seq, "pan_deg": round(float(self.cmd[0]), 3), "tilt_deg": round(float(self.cmd[1]), 3)})


class PlaceImage:
    """원시 깊이 프레임 → Frame(깊이 · 홀더), 틱 → img2. 상태 = 마지막 컵 마스크 · 보임 · 첫 틱."""

    def __init__(self, rd, depth_pre: dict, d_min: float, d_max: float, target: str) -> None:
        self.rd, self.b = rd, int(depth_pre["block"])
        self.min_valid, self.z_cut = float(depth_pre["min_valid"]), float(depth_pre["floor_cut_z_m"])
        self.d_min, self.d_max = float(d_min), float(d_max)
        self.K = np.asarray(rd.D435I_K_1103, np.float64)
        self.kb = block_k(self.K, self.b)
        self.w, self.h = (int(v) for v in rd.D435I_WH)
        self.verts, self.faces = HPJ.load_obj()
        self.target, self.ci = target, SM.class_index(target)
        self.c_cup = np.asarray(CG.C_DEFAULT[CG.CLASS_CUP[target]], np.float64)
        self.reset()

    def reset(self) -> None:
        self.last = np.zeros((self.h // self.b, self.w // self.b), np.float32)
        self.vis, self.first, self.dist_px = 0.0, True, float("inf")

    def frame(self, d_mm: np.ndarray, stamp_ns: int, T_base_cam, holder_pos, holder_yaw: float) -> Frame:
        import torch

        if d_mm.shape != (self.h, self.w) or d_mm.dtype != np.uint16:
            raise ValueError(f"깊이 {d_mm.shape} {d_mm.dtype} ≠ ({self.h}, {self.w}) uint16")
        m = d_mm.astype(np.float32) / np.float32(1000.0)        # head_depth_node.to_small 과 같은 나눗셈
        cut = HP.floor_cut(m, T_base_cam, self.K, self.z_cut)
        small = self.rd.block_median(torch.from_numpy(cut), self.b, self.min_valid).numpy()
        hol, _ = HPJ.holder_mask(self.verts, self.faces, HPJ.holder_T(holder_pos, holder_yaw), T_base_cam, self.K,
                                 self.w, self.h, depth_m=m, block=self.b)
        return Frame(int(stamp_ns), small.astype(np.float32), hol, np.asarray(T_base_cam, np.float64))

    def img2(self, fr: Frame, cup_bits, palm_p, palm_R) -> tuple[np.ndarray, np.ndarray]:
        """(깊이 (h,w), img2 (h,w,3)) — 첫 틱은 둘 다 0(hdgp 학습기가 리셋 틱 프레임을 비운다), 컵 유지 상태는 갱신한다."""
        cup = None
        if cup_bits is not None:
            # shortcut: 거리 문턱을 블록 화소 무게중심으로 본다(양자화 오차 ≲ 4 원화소 ≪ 80). 같은 클래스 두 개면 top1 만 보인다 —
            #   실기에서 놓친 기록이 쌓이면 head_mask_node 에 클래스별 전 인스턴스 발행 옵션을 단다.
            cup, self.dist_px, _ = CG.pick_cup([(self.ci, 1.0, np.asarray(cup_bits, bool))], self.target, palm_p, palm_R,
                                               fr.T_base_cam, self.kb, c=self.c_cup, gate_px=CG.GATE_PX / self.b)
            self.dist_px *= self.b
        seen = cup is not None and int(cup.sum()) >= SM.VIS_MIN_PX
        if seen:
            self.last = cup.astype(np.float32)
        self.vis = float(seen)
        d = fr.depth
        dn = np.where((d >= self.d_min) & (d <= self.d_max), d / self.d_max, 0.0).astype(np.float32)
        out = np.stack([dn * fr.holder, dn * self.last, np.full_like(dn, self.vis)], -1)
        if self.first:
            self.first = False
            return np.zeros_like(d), np.zeros_like(out)
        return d, out


class StudentPlaceChain(P.PlaceChain):
    """sim 처럼 학생이 이송 중에도 돌고(LSTM 진행, 출력은 버림) 이송 끝에 RNN 을 비우지 않는다(hdgp env 는 에피소드 리셋에서만)."""

    def _policy_reset(self) -> None:
        if getattr(self, "_ep_start", False):
            self._ep_start = False
            super()._policy_reset()

    def reset(self, *a, **k):
        self._ep_start = True
        self.carry_stale_n, self.carry_stale_run, self.carry_input_err = 0, 0, ""
        return super().reset(*a, **k)

    def step(self, meas):
        if not self.carrying:      # 낡은 입력이면 베이스 타이머(settle_t · latch_t) · est.step 이 행동 없이 한 틱 가지 않게 먼저 실패
            self.policy.fresh()    # shortcut: latch_s2r 경로 기준(정책이 settle 중에도 돈다) — 배포 place 계약은 모두 latch_s2r
        return super().step(meas)

    def _carry_tick(self, meas):
        obs, a, targets = super()._carry_tick(meas)
        try:                                           # 이송은 정책과 무관한 스크립트 — 입력이 한 틱 낡아도 이송은 잇는다
            self.policy.forward(obs, carry_on=True)
            self.carry_stale_run = 0
        except PlaceInputStale as exc:
            self.carry_stale_n += 1
            self.carry_stale_run += 1
            self.carry_input_err = str(exc)
        return obs, a, targets


class PlaceStudentPolicy:
    """PlaceChain 의 policy 자리(forward(obs) → 행동). 입력원(frame · masks · head)은 노드가 채운다.
    forward 는 최신 Frame · 그 stamp 의 마스크 비트 · 지금 목 각으로 학생 한 틱을 돈다. 낡으면 PlaceInputStale."""

    def __init__(self, policy_dir, c, target: str = "", device: str = "cpu", num_threads: int = 4,
                 max_depth_age_s: float | None = None) -> None:
        import torch

        from .student_policy import MAX_DEPTH_AGE_S, REAL_DEPTH, load_model_module, load_pinned, load_student_dir

        cfg = load_student_dir(policy_dir)
        got = hdgp_pin.md5(cfg["ckpt"]["path"])
        if got != cfg["ckpt"]["md5"]:
            raise ValueError(f"학생 ckpt md5 {got} ≠ student.yaml {cfg['ckpt']['md5']}")
        self.rd, _ = load_pinned(REAL_DEPTH, "hdgp_real_depth")
        torch.set_num_threads(int(num_threads))
        d = torch.load(cfg["ckpt"]["path"], map_location=device, weights_only=True)
        self.extr = cfg["extrinsics"]["path"]
        check_meta(d, c, cfg["extrinsics"]["md5"], TABLE_Z - float(self.rd.FLOOR_MARGIN_M))
        self.c, self.torch, self.device, self.md5 = c, torch, device, got
        self.mod = load_model_module()
        self.model = self.mod.make_depth_student(int(d["n_obs"]), int(d["n_act"]), hid=int(d["hid"]), n_aux=int(d["n_aux"]),
                                                 pretrained=False, arch=d["arch"], n_vis=int(d.get("n_vis") or 0)).to(device)
        self.model.load_state_dict(d["model"])
        self.model.eval()
        self.meta = {k: v for k, v in d.items() if k != "model"}
        self.img_hw = tuple(int(x) for x in d["img_hw"])
        self.target = target or str((cfg.get("mask") or {}).get("target_class") or "")
        shapes = {cup_shape_of(k) for k in (d.get("head") or {}).get("cup_c_m") or {}}
        if CG.CLASS_CUP.get(self.target) not in shapes:
            raise ValueError(f"목표 {self.target!r}(컵 {CG.CLASS_CUP.get(self.target)}) 는 이 ckpt 가 배운 컵 {sorted(map(str, shapes))} 가 아니다")
        self.hc = HP.HeadCamPose(self.extr, sag=True)
        # ★11.09 시작 자세는 틱으로 대조한다. ckpt start_enc_deg 는 hdgp 규약(·360/4096), 실기 /head/joint_states 와 hc.q0_enc_deg 는
        #   드라이버 규약(·360/4095)이라 같은 틱도 0.04° 남짓 다르다. 틱이 같으면 q0 는 실기 q 와 같은 드라이버 규약 값을 쓴다.
        tick = d["head"].get("start_tick")
        if tick is not None:
            if tuple(int(t) for t in tick) != self.hc.q0_tick:
                raise ValueError(f"ckpt 머리 시작 틱 {list(tick)} ≠ 외부 파라미터 head_pose 틱 {list(self.hc.q0_tick)}")
        elif not np.allclose(d["head"]["start_enc_deg"], self.hc.q0_enc_deg, atol=1e-6):
            raise ValueError(f"ckpt 머리 시작 자세 {d['head']['start_enc_deg']} ≠ 외부 파라미터 head_pose {self.hc.q0_enc_deg}")
        self.q0 = np.asarray(self.hc.q0_enc_deg, np.float64)
        self.image = PlaceImage(self.rd, d["depth_pre"], float(d["d_min"]), float(d["d_max"]), self.target)
        self.masks = SM.MaskStream(self.img_hw)
        self.head = HeadState()
        self.look = HeadLook(self.hc)
        self.cfg, self.frame, self.chain, self.last_palm = cfg, None, None, None
        self.max_depth_age_s = float(MAX_DEPTH_AGE_S if max_depth_age_s is None else max_depth_age_s)
        self.now_ns = None                      # 노드가 틱마다 벽시계(ns)를 넣는다(stamp 와 같은 기준)
        self.reset()

    def reset(self) -> None:
        self.hc_state = self.model.zero_state(1, self.device)
        self.image.reset()
        self.masks.reset()
        self.look.reset()
        self.last_palm, self.ok_ns, self._fresh = None, None, None
        self.last_in: dict = {}

    def fresh(self) -> tuple:
        """(Frame, 깊이 나이, 마스크 비트, 지금 목 각). 하나라도 없거나 낡으면 PlaceInputStale.
        한 틱(같은 now_ns)에 체인 사전 검사와 forward 가 두 번 부른다. 성공한 결과는 now_ns 로 묶어 두고 다시 돌려준다 —
        그 사이 일꾼이 새 Frame 을 넣어도 사전 검사가 통과시킨 입력으로 정책을 돈다(마스크 짝 · 놓침 셈도 한 번)."""
        if self._fresh is not None and self._fresh[0] == self.now_ns:
            return self._fresh[1]
        fr, now = self.frame, self.now_ns
        if fr is None:
            raise PlaceInputStale("놓기 깊이 프레임이 아직 없다")
        age = (now - fr.stamp_ns) * 1e-9
        why = age_refusal(age, self.max_depth_age_s, "놓기 깊이")
        if why:
            raise PlaceInputStale(why)
        try:
            bits = self.masks.bits_for(fr.stamp_ns, self.image.ci)
        except SM.MaskStale as exc:
            raise PlaceInputStale(str(exc)) from exc
        out = (fr, age, bits, self.head.latest(now))
        self._fresh = (now, out)
        return out

    def inputs(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(깊이, img2, 머리 2 칸). 깊이 · 마스크 · 목 상태가 낡으면 PlaceInputStale."""
        fr, age, bits, q = self.fresh()
        palm_p, palm_R = palm_from_obs(self.c, obs)
        self.last_palm = (palm_p, palm_R)
        first = self.image.first                       # 첫 틱 표시는 PlaceImage 하나 — img2 가 끈다
        depth, img2 = self.image.img2(fr, SM.decode(bits, self.image.ci)[0], palm_p, palm_R)
        head = to_action(self.q0 if first else q)
        self.last_in = {"depth_age_ms": round(age * 1e3, 1), "cup_vis": self.image.vis, "cup_dist_px": self.image.dist_px,
                        "holder_px": int(fr.holder.sum()), "head_enc_deg": [round(float(v), 2) for v in q]}
        return depth, img2, head

    def forward(self, obs, carry_on: bool = False) -> np.ndarray:
        t = self.torch
        depth, img2, head = self.inputs(obs)
        settle = bool(self.chain is not None and self.chain.settling)
        prop = prop_from_obs(self.c, obs, carry_on, settle, head)
        if prop.shape != (self.model.n_prop,):
            raise ValueError(f"놓기 학생 prop {prop.shape} ≠ ({self.model.n_prop},)")
        with t.inference_mode():
            img = self.mod.depth_input(t.from_numpy(depth).to(self.device)[None], self.image.d_min, self.image.d_max)
            img = t.cat([img, t.from_numpy(img2).to(self.device).permute(2, 0, 1)[None]], 1)
            a, _, self.hc_state = self.model(img, t.from_numpy(prop).to(self.device)[None], self.hc_state, train_cnn=False)
        self.ok_ns = self.now_ns                       # 이 틱 학생 입력이 신선했다(노드 머리 규칙이 본다)
        return np.clip(a[0].float().cpu().numpy(), -1.0, 1.0)
