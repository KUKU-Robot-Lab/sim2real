"""rh_cap_student_node: rh_cap 깊이 학생 depth_h01(캡 찾기 → 접근 → 잡기 → 이송 → 끼우기, 움직이는 머리) 실기 노드(11.18).
rh_cap_node(CapNode) 위에 체인 · 깊이 · 머리 · 구간 넘김만 얹는다. 정책 코어 = cap_student.CapStudent.

  in   CapNode 입력 가운데 팔 · 손 · /policy_control/joint_target(pd 가 붙잡은 시작 목표)
       캡 · 리시버 자세(하강 · 이송 스크립트만 쓴다, cap_scripts): cap_pose_topic(기본 비움 — FP++ 에 뚜껑이 없다) ·
         rcv_pose_topic(기본 /objects/shaker_c_orange/live_pose, 쉐이커 메쉬 원점 → env 리시버 원점 환산). 받은 지 CUP_STALE_S 안이면
         FP++, 아니면 고정 자리(cap_fixed_* · rcv_fixed_*). start 순간 한 번 골라 그 에피소드 내내 쓴다.
         리시버가 고정 자리로 떨어지면 start 거부(rcv_allow_fixed:=true 면 WARN 뒤 고정 자리)
       /head_depth/image 32FC1 (90, 160) [m](구멍 0) · /head/joint_states(URDF rad → 인코더 deg, place_student.HeadState)
  out  CapNode 출력(joint_target · obs 62 · action 13 = 실행 행동 · policy_action = 학생 출력 · episode · status)
       /head/target_stream(String JSON {"seq","pan_deg","tilt_deg"} 절대 인코더 deg) — 학생 행동 끝 2 칸(cap_student.head_deg)을
       그대로, 성공한 틱마다. 외부 파라미터 환산 · 영점 보정 없음(11.18 사용자)
  srv  /policy_control/<ns>/episode/{reset,start,stop,abort} · /policy_control/<ns>/stage/next(Trigger — 사람이 확인한 "다음 구간")

    .venv-student/bin/python deploy/policy_control/policy_control/rh_cap_student_node.py --ros-args -r __node:=rh_cap_student_right \\
        -p ns:=right -p contract:=<rh_cap_contract.json> -p robot:=… -p student_dir:=deploy/policies/rh56f1/cap/right_student_h01

구간(hdgp db1160bc rh_cap_env.py, depth_h01 = chain_search · chain_capa direct · chain_carry_start success). 단계 one-hot 은 학생이
정하지 않는 입력이다(_stage_onehot). sim 은 전환을 참값(캡 위치 · 접촉)으로 판정하므로 그 전환은 stage/next 로 받는다.
  search    학생 팔 그대로 · 손 폄(_pre_physics_step a_search)   끝 = 캡 보임 5 스텝(_head_perceive, 참값)          → stage/next
  approach  학생 팔 그대로 · 손 폄(a_capa, 상자 없음)            끝 = 접근 성공 래치(_capa_end_mask, 참값)          → stage/next
  descend   IK 하강 스크립트(_descend_actions) · 단계 칸 grasp    끝 = 도달 · 닿음(손끝 촉각) · 150 스텝            → 자동
  grasp     학생 팔(상자 ± phase_arm_box_rad) · 손 절대           끝 = _grasp_ok 120 스텝(참값)                     → stage/next
  carry     ready_carry Mover(_carry_actions) · 손 = 잡은 목표    끝 = Mover 끝 뒤 캡 정지 또는 60 스텝(chain_advance) → 자동(60 틱)
  seat      학생 팔(상자) · 손 = 넘긴 목표 + 잔차                  끝 = _seat_ok 30 스텝(참값)                       → stage/next = 에피소드 stop
stage/next 는 학생 구간에서만, 그 구간에 성공한 틱이 있고 직전 틱이 오류가 아닐 때 받는다(스크립트 구간은 자동만).
  예외: 이송 Mover 가 도착 없이 끝나면(실기 전용, sim 은 그대로 넘긴다) 목표를 유지하고 자동 전환 대신 stage/next 를 받는다.
★ sim 은 하강에 참 캡 자세, 이송에 참 손 안 캡 자세를 쓴다. 실기는 공급 자세와 "잡기로 넘어가는 순간 손바닥에 붙인 캡"으로 근사한다(cap_scripts).
LSTM 은 에피소드 reset 에서만 0(learner.on_step reset_state(new_ep)) — 구간이 바뀌어도, 스크립트 구간에도 매 틱 학생을 돌린다.
학생 예외(StaleDepth · ValueError · FloatingPointError)는 틱 오류다: 그 틱은 팔 · 손 · 머리 명령을 내지 않고 max_gap_ticks 넘게
이어지면 abort(pd 가 붙든다) — rh_place_student_node 와 같다. /head/joint_states 가 끊기면 바로 abort.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import cap_scripts as CSC  # noqa: E402
from policy_control import cap_student as CS  # noqa: E402
from policy_control import place_student as PS  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402
from policy_control import rh_cap as C  # noqa: E402
from policy_control.pour_fj_node import CUP_STALE_S, Family, PourFjNodeError, main  # noqa: E402
from policy_control.rh_aglt_student_node import CONFIG_TOPIC, DEPTH_TOPIC, image_array, stamp_ns  # noqa: E402
from policy_control.rh_cap_node import CapNode  # noqa: E402
from policy_control.rh_place_student_node import HEAD_TOPIC, STREAM_TOPIC  # noqa: E402

PHASES = ("search", "approach", "descend", "grasp", "carry", "seat")
DONE = "done"
#: 구간 → 학생 관측 단계 칸(cap_student.STAGES 이름). 하강은 sim 에서 잡기 단계 안의 스크립트다
STAGE_OF = {"search": "search", "approach": "approach", "descend": "grasp", "grasp": "grasp", "carry": "carry", "seat": "seat"}
SCRIPTED = ("descend", "carry")          # 학생이 팔을 조종하지 않는 구간
# 아래 셋은 학습 런 env.yaml(server wt_cap/runs/cap_split/depth_h01/s31/run/params) 값 — 묶음 student.yaml 에는 아직 없다
CARRY_WAIT_TICKS = 60                    # chain_carry_wait_steps: 이송기 끝난 뒤 넘김까지(캡 정지를 못 보므로 상한을 쓴다)
EPISODE_S = 95.0                         # episode_length_s
LIVE_ARM_START_Q = (-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646)   # live_arm_start_q = 에피소드 시작 팔 자세
DEPTH_START_WAIT_S = 1.0                 # reset 때 새 깊이 프레임을 기다리는 상한 = rh_place_student_node 11.10 값(그쪽은 아직 미커밋이라 가져오지 않는다)
HAND_COMMAND = "position"                # 증류 spec real.hand_adm_enable false
STUDENT_ERRORS = (CS.StaleDepth, FloatingPointError)   # 계열 오류(ValueError 포함)에 더해 틱 오류로 받는 학생 예외
# env 는 리셋 때 손이 편 상태다. 손 목표가 start 에서 편 값으로 뛰므로 실측이 편 손에서 멀면 거부한다.
#   값 = episode_ros.HAND_OPEN_TOL_RAD(손 펴짐 확인 · pd 홈 손 경로 검사와 같은 값). 여기는 구동 6 관절 모두 본다.
HAND_OPEN_TOL_RAD = 0.15
#: 깊이 노드 /head_depth/config 기대값 = head_depth_node --free-head. 학생은 바닥 컷 · 목 게이트 없이 0.01~10 m 밖 = 0 인 깊이로 배웠다
#:   (hdgp db1160bc rh_cap_env_cfg head_depth_clip). 머리 고정 모드 config 에는 floor_cut · head_gate · clip_m 키가 없다.
DEPTH_CFG_EXPECT = {"floor_cut": False, "head_gate": False, "clip_m": [0.01, CS.DEPTH_MAX_M],
                    "out_wh": [CS.IMG_HW[1], CS.IMG_HW[0]]}


def depth_config_refusals(cfg: dict | None) -> list:
    """/head_depth/config 하나 → 거부 사유(빈 목록 = 통과). bool 은 0 · 1 과 섞이지 않게 is 로 본다."""
    if cfg is None:
        return [f"{CONFIG_TOPIC} 깊이 노드 config 없음 — head_depth_node --free-head 를 먼저 띄운다"]
    if not isinstance(cfg, dict):
        return [f"{CONFIG_TOPIC} 가 JSON 객체가 아니다: {cfg!r:.80}"]
    return [f"{CONFIG_TOPIC} {k} = {cfg.get(k, '(키 없음)')!r}, 기대 {v!r}(--free-head)" for k, v in DEPTH_CFG_EXPECT.items()
            if not (cfg.get(k) is v if isinstance(v, bool) else cfg.get(k) == v)]


def depth_configs_refusals(texts) -> list:
    """받은 config 본문 전부 → 거부 사유. latched 토픽은 살아 있는 발행자마다 한 통씩 오므로, 하나라도 어긋나면
    다른 모드의 깊이 노드가 같은 /head_depth/image 에 섞여 낼 수 있다고 보고 거부한다(마지막 한 통만 보면 도착 순서에 달림)."""
    if not texts:
        return depth_config_refusals(None)
    out = []
    for t in texts:
        try:
            out += depth_config_refusals(json.loads(t))
        except ValueError as exc:
            out.append(f"{CONFIG_TOPIC} JSON 아님: {exc}")
    return out


class Stages:
    """구간 기계. next() = 사람이 확인한 전환, auto() = 참값 없이 판정되는 전환(틱 끝마다)."""

    def __init__(self, carry_wait: int = CARRY_WAIT_TICKS) -> None:
        self.carry_wait = int(carry_wait)
        self.reset()

    def reset(self) -> None:
        self.phase, self.wait = PHASES[0], 0

    def next(self) -> str:
        """다음 구간으로 — seat 뒤는 DONE."""
        self.phase, self.wait = (*PHASES, DONE)[PHASES.index(self.phase) + 1], 0
        return self.phase

    def auto(self, script_done: bool) -> bool:
        """하강 스크립트가 끝나면 잡기로, 이송 스크립트가 끝난 뒤 carry_wait 틱이면 끼우기로(hdgp chain_advance). 넘겼으면 True.
        이송의 script_done 은 도착으로 끝났을 때만 온다 — 미도달(시간 초과 등)은 자동 전환 없이 사람 stage/next 를 기다린다."""
        self.wait = self.wait + 1 if self.phase == "carry" and script_done else 0
        if (self.phase == "descend" and script_done) or (self.phase == "carry" and self.wait >= self.carry_wait):
            self.next()
            return True
        return False


def head_stream_body(seq: int, deg) -> str:
    """머리 목표 인코더 deg (pan, tilt) → /head/target_stream 본문(place_student.HeadLook 과 같은 형식). 변환 없음."""
    pan, tilt = (float(v) for v in deg)
    return json.dumps({"seq": int(seq), "pan_deg": round(pan, 3), "tilt_deg": round(tilt, 3)})


def start_refusals(arm_q, tol: float) -> list:
    """에피소드 시작 팔 자세 = 학습 live_arm_start_q(± tol rad) — 아니면 거부."""
    err = float(np.abs(np.asarray(arm_q, float) - LIVE_ARM_START_Q).max())
    return [] if err <= tol else [f"팔이 학생 시작 자세에서 {err:.3f} rad(> {tol}) — 시작 자세 {list(LIVE_ARM_START_Q)} 로 옮긴 뒤 reset"]


def hand_open_refusals(hand_q, hand_open) -> list:
    """손 실측(구동 6)이 편 손(계약 hand_open)에서 HAND_OPEN_TOL_RAD 밖이면 거부 — start 에서 손 목표가 편 값으로 뛴다."""
    err = float(np.abs(np.asarray(hand_q, float) - np.asarray(hand_open, float)).max())
    return [] if err <= HAND_OPEN_TOL_RAD else [
        f"손 실측이 편 손 {list(hand_open)} 에서 {err:.3f} rad(> {HAND_OPEN_TOL_RAD}) — 손을 편 뒤 reset · start"]


def next_refusal(phase: str, phase_steps: int, gap: int, carry_missed: bool = False) -> list:
    """stage/next 를 받을 수 없는 사유 — 스크립트 구간(자동 전환) · 이 구간에 성공한 틱 없음 · 직전 틱 오류.
    예외: 이송이 미도달로 끝났으면(carry_missed) 자동 전환이 없으므로 사람이 보고 넘긴다."""
    if phase in SCRIPTED and not (phase == "carry" and carry_missed):
        return [f"stage/next: {phase} 는 스크립트 구간 — 끝나면 자동으로 넘어간다"]
    if phase_steps < 1:
        return [f"stage/next: {phase} 에서 성공한 틱이 아직 없다"]
    return [f"stage/next: 직전 {gap} 틱이 오류다"] if gap > 0 else []


def contract_mismatch(c, card: dict) -> list:
    """학생 묶음(student.yaml side · train)과 디코더 계약이 어긋나는 칸. 비어야 노드가 뜬다(묶음에 값이 없어도 어긋남)."""
    t = card.get("train") or {}
    # 묶음 pt 를 바꾸고 계약을 다시 안 만든 경우 · chain 아닌 계약을 잡는다
    want = {"side": card.get("side"), "profile": t.get("profile_name"), "hand_command": HAND_COMMAND,
            "cap_phase": C.CHAIN, "checkpoint_md5": (card.get("ckpt") or {}).get("md5"),
            "hand_block": t.get("hand_block_enable"),
            **{k: t.get(k) for k in ("policy_hz", "k_arm", "arm_ema", "hand_ema", "hand_full_range_s", "joint_err_norm",
                                     "phase_arm_box_rad")}}
    got = {"side": c.side().side, "hand_block": bool(c.hand_block), **{k: getattr(c, k) for k in want if k not in ("side", "hand_block")}}
    return [f"{k}: 계약 {got[k]!r} ≠ 학생 {want[k]!r}" for k in want if got[k] != want[k]]


class CapStudentChain:
    """한 틱: 관측 62(단계 칸 · 머리 각) → 학생 → 구간별 실행 행동 13 → rh_aglt 디코더.
    학생 호출이 실패하면 디코더 · 직전 행동 · 구간은 그대로다(학생이 먼저 돈다)."""

    def __init__(self, c: C.CapContract, policy) -> None:
        self.c, self.policy = c, policy
        self.dec = A.RaDecoder(c)
        self.lo, self.hi = self.dec.law.bounds()
        self.a_open = self._hand_action(c.side().hand_open)            # hdgp _a_open
        self.stages = Stages()
        self.scripts = None             # cap_scripts.CapScripts(노드가 넣는다) — 하강 · 이송
        self._clear()

    def _clear(self) -> None:
        self.stages.reset()
        self.prev = np.zeros(CS.N_ARM_HAND)
        self.center = self.hand_base = self.head_cmd = self.policy_a = None
        self.poses = None               # {"cap": (pos, quat), "rcv": (pos, quat)} — start 순간 공급 자세(노드가 넣는다)
        self.cap_rel = None             # 하강 끝 순간 손바닥 틀의 캡 원점(cap_scripts 손 안 캡 근사와 같은 순간)
        self.step_i = self.phase_steps = 0
        if self.scripts is not None:
            self.scripts.reset()

    def _hand_action(self, target, a=(0.0,) * 6) -> np.ndarray:
        """손 목표(+ 잔차 a) → 절대 법칙 행동(hdgp hand_residual_to_abs)."""
        return CS.seat_hand_action(a, target, self.lo, self.hi)

    def reset(self, meas, held: tuple | None = None) -> None:
        """held = pd 가 붙잡은 마지막 (팔 7, 손 6) 목표 — 없으면 팔 실측. 손은 편 손에서 시작한다(sim 리셋)."""
        self.dec.reset(arm_start=meas["arm"].arm_q if held is None else held[0], hand_start=self.c.side().hand_open)
        self._clear()
        self.policy.reset()                                            # LSTM 0 은 여기서만

    def _enter(self) -> None:
        """구간에 들어선 순간의 넘김 값(hdgp _descend_actions 끝 · _chain_step g_evt · _chain_handoff)."""
        st, ph = self.dec.state["arm"], self.stages.phase
        self.phase_steps = 0
        if ph in ("grasp", "seat"):
            self.center = st.arm_target.copy()                         # 팔 상자 중심 = 넘긴 순간 팔 목표
        if ph in ("carry", "seat"):
            self.hand_base = st.hand_target.copy()                     # 이송 = 이 손 목표 유지 · 끼우기 = 이 목표 + 잔차

    def next(self) -> str:
        ph = self.stages.next()
        self._enter()
        return ph

    def _script(self, ph: str, m, hand_q) -> tuple[np.ndarray | None, bool]:
        """스크립트 구간 팔 행동과 끝 신호(cap_scripts). 학생 구간은 (None, False)."""
        if ph not in SCRIPTED:
            return None, False
        if self.scripts is None or self.poses is None:
            raise CSC.ScriptError(f"{ph}: 스크립트 · 공급 자세가 없다(start 전)")
        tgt = self.dec.state["arm"].arm_target
        if ph == "descend":
            return self.scripts.descend_step(m, hand_q, tgt, self.poses["cap"])
        return self.scripts.carry_step(m, tgt, self.poses["rcv"])     # 도착 없이 끝나면 끝 False 유지 — 자동 seat 없음(stage/next)

    def _executed(self, ph: str, a: np.ndarray, a_script: np.ndarray | None) -> np.ndarray:
        if ph in ("search", "approach"):
            return np.concatenate([a[:7], self.a_open])
        if ph == "descend":                                            # 손은 편 채(env _descend_actions)
            return np.concatenate([a_script, self.a_open])
        if ph == "carry":                                              # 손 = 잡기 끝 순간 목표 유지
            return np.concatenate([a_script, self._hand_action(self.hand_base)])
        a_arm = C.box_arm_action(self.c, a[:7], self.dec.state["arm"].arm_target, self.center)
        return np.concatenate([a_arm, a[7:] if ph == "grasp" else self._hand_action(self.hand_base, a[7:])])

    def step(self, meas) -> tuple[np.ndarray, np.ndarray, dict]:
        m, st, ph = meas["arm"], self.dec.state["arm"], self.stages.phase
        hand_q = np.array([m.hand_q[j] for j in self.c.side().hand_joints], float)
        obs = CS.build_obs(m.arm_q, m.arm_qd, st.arm_target, hand_q, st.hand_target, m.palm_pos, m.palm_R, self.prev,
                           CS.STAGES.index(STAGE_OF[ph]), meas["head"])
        depth, age = meas["depth"]
        a, head = self.policy.step(depth, obs, age)
        a_script, done = self._script(ph, m, hand_q)
        a_out = self._executed(ph, a, a_script)
        cap = self.cap_pos(m)
        targets = self.dec.step(a_out, active=True, tactile_n=m.tactile_n, hand_q=hand_q, joint_force=m.joint_force,
                                obj_palm_m=float(np.linalg.norm(cap - np.asarray(m.palm_pos, float))))
        self.prev, self.policy_a, self.head_cmd = a_out, a, head       # 직전 행동 = 실행한 행동(hdgp prev_actions)
        self.step_i += 1
        self.phase_steps += 1
        if self.stages.auto(done):                                     # 하강 끝 → 잡기 · 이송 끝 + 60 틱 → 끼우기
            if ph == "descend":                                        # 이 뒤 캡 = 손바닥에 붙인 자세
                self.cap_rel = np.asarray(m.palm_R, float).T @ (cap - np.asarray(m.palm_pos, float))
            self._enter()
        return obs, a_out, targets

    def cap_pos(self, m) -> np.ndarray:
        """grasp_near_m 의 물체 점 = 캡 원점(hdgp db1160bc rh_cap self.cup · rh_aglt _cup_pos ↔ _palm_pos 1139-1141).
        하강 끝 전 = start 공급 자세, 뒤 = 그 순간 손바닥에 붙인 캡. 자세가 없거나 유한하지 않으면 틱 오류."""
        if self.poses is None:
            raise CSC.ScriptError("캡 자세가 없다(start 전) — 디코더 grasp_near_m 거리를 못 잰다")
        p = self.poses["cap"][0] if self.cap_rel is None else np.asarray(m.palm_pos, float) + np.asarray(m.palm_R, float) @ self.cap_rel
        p = np.asarray(p, float)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(np.asarray(m.palm_pos, float)))):
            raise CSC.ScriptError(f"캡 · 손바닥 위치에 NaN/inf — 캡 {p.tolist()} · 손바닥 {np.asarray(m.palm_pos).tolist()}")
        return p

    @property
    def carry_missed(self) -> bool:
        return self.stages.phase == "carry" and self.scripts is not None and self.scripts.carry_missed

    def as_dict(self) -> dict:
        ph = self.stages.phase
        return {"phase": ph, "stage": STAGE_OF.get(ph), "step": self.step_i, "phase_steps": self.phase_steps,
                "scripted": ph in SCRIPTED, "carry_wait": self.stages.wait,
                "note": "carry 미도달 — 목표 유지, stage/next 로 seat" if self.carry_missed else None,
                "scripts": None if self.scripts is None else self.scripts.as_dict(),
                "head_cmd": None if self.head_cmd is None else [round(float(v), 2) for v in self.head_cmd]}


class CapStudentNode(CapNode):
    PHASES = (C.CHAIN,)

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        import yaml
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_srvs.srv import Trigger
        from policy_control.raw_poll import poll_subscription

        d = str(self.declare_parameter("student_dir", "").value).strip()
        self._head_on = bool(self.declare_parameter("head_stream", True).value)
        if not d:
            raise PourFjNodeError("student_dir:=<학생 묶음 폴더(student.yaml)> 가 필요하다")
        # torch 스레드 = 1: main 의 keep_policy 가 풀이 생긴 뒤 프로세스 전체를 정책 cpu 하나에 묶는다. 4 스레드가 한 코어에서
        #   서로 기다려 학생 한 스텝 1.65 s · 이송 계획 6.2 s(11.18 오프라인 실측), 1 스레드면 11 ms · 26 ms.
        nt = int(self.declare_parameter("num_threads", 1).value)
        self._policy = CS.CapStudent.from_dir(d, device=self.device, num_threads=nt)   # 묶음 · md5 · 스냅숏 커밋 검사(PinError)
        card = yaml.safe_load((Path(d).expanduser() / CS.STUDENT_CARD).read_text(encoding="utf-8")) or {}
        bad = contract_mismatch(self.contract, card)
        if bad:
            raise PourFjNodeError(f"계약이 학생 묶음 {d} 과 어긋난다 — {bad}")
        self.fam = Family(**{**self.fam.__dict__, "chain": CapStudentChain, "errors": (*self.fam.errors, *STUDENT_ERRORS),
                             "refusals": lambda _c, meas, tol: self._start_refusals(meas, tol)})
        self.attach, self.gates = {}, {}                               # 학생 관측에 캡 자세가 없다(_cup_for) — 스크립트만 쓴다
        # 캡 · 리시버 공급(cap_scripts): 토픽이 비면 고정 자리만. 기본 고정 자리 = 학습 소환 중심 · 리시버 뱅크 중앙값(로봇 루트 틀)
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self._pose_topic = {"cap": str(p("cap_pose_topic", "")).strip(),
                            "rcv": str(p("rcv_pose_topic", "/objects/shaker_c_orange/live_pose")).strip()}
        self._fixed = {"cap": (CSC.vec(p("cap_fixed_xyz", list(CSC.CAP_FIXED_POS)), 3, "cap_fixed_xyz"),
                               CSC.vec(p("cap_fixed_quat", list(CSC.CAP_FIXED_QUAT)), 4, "cap_fixed_quat")),
                       "rcv": (CSC.vec(p("rcv_fixed_xyz", list(CSC.RCV_FIXED_POS)), 3, "rcv_fixed_xyz"),
                               CSC.vec(p("rcv_fixed_quat", list(CSC.RCV_FIXED_QUAT)), 4, "rcv_fixed_quat"))}
        self._poll_pose = {k: self._make_cup_poll(t) for k, t in self._pose_topic.items() if t}
        self._start_poses: dict | None = None
        from policy_control.place_ready_carry import load_chain
        s, h = self.contract.side(), float(p("descend_h_m", CSC.DESCEND_H))
        if not CSC.DESCEND_H_RANGE[0] <= h <= CSC.DESCEND_H_RANGE[1]:
            raise PourFjNodeError(f"descend_h_m {h} 가 학습 범위 {CSC.DESCEND_H_RANGE} 밖")
        xy_max, len_max = float(p("descend_xy_max_m", CSC.DESCEND_XY_MAX_M)), float(p("descend_len_max_m", CSC.DESCEND_LEN_MAX_M))
        if not (0.0 < xy_max and 0.0 < len_max):
            raise PourFjNodeError(f"descend_xy_max_m {xy_max} · descend_len_max_m {len_max} 는 양수여야 한다")
        self._rcv_allow_fixed = bool(p("rcv_allow_fixed", False))
        RC, mch = load_chain(s.palm_body, list(s.arm_joints))         # 배포 핀 hdgp ready_carry 사본(PinError 면 노드가 안 뜬다)
        self._scripts = CSC.CapScripts(self.contract, CSC.Carry(RC, mch, k_arm=self.contract.k_arm, arm_ema=self.contract.arm_ema), h,
                                       xy_max=xy_max, len_max=len_max)
        if float(self.get_parameter("max_episode_s").value) < 0:
            self.max_s = EPISODE_S
        self._head = PS.HeadState()
        self._depth: tuple[np.ndarray, int] | None = None
        self._Image, self._head_seq, self._phase_log = Image, 0, None
        self._sub_depth = poll_subscription(self._poll, Image, DEPTH_TOPIC, qos_profile_sensor_data)
        self._sub_head = poll_subscription(self._poll, self._JointState, HEAD_TOPIC, QoSProfile(depth=50))
        # 깊이 노드 config(transient_local) — 발행자마다 한 통. depth 1 이면 두 발행자 것이 한꺼번에 올 때 하나가 덮이므로 10
        self._depth_cfgs: dict[str, None] = {}                         # 받은 본문(순서 유지 · 같은 본문 한 번)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(self._String, CONFIG_TOPIC, lambda m: self._depth_cfgs.setdefault(m.data, None), latched)
        self._pub_head = self.create_publisher(self._String, STREAM_TOPIC, 10)
        self.create_service(Trigger, f"{self.base}/stage/next", self._srv_next)
        self.get_logger().info(f"rh_cap 학생 {d} md5 {self._policy.md5} · 머리 스트림 {'켬' if self._head_on else '끔'} · "
                               f"에피소드 상한 {self.max_s:.1f} s · 구간 넘김 {self.base}/stage/next · 하강 h {h} m · "
                               f"캡 {self._pose_topic['cap'] or '(토픽 없음)'} / 고정 {self._fixed['cap'][0]} · "
                               f"리시버 {self._pose_topic['rcv'] or '(토픽 없음)'} / 고정 {self._fixed['rcv'][0]}"
                               f"(고정 시작 {'허락' if self._rcv_allow_fixed else '거부'}) · 하강 시작 xy ≤ {xy_max} m · L ≤ {len_max} m")

    # ---------------------------------------------------------------- inputs
    def _cup_for(self, role: str, advance: bool):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])             # 캡 자세 없음 — 측정 틀(RaMeas)만 채운다

    def _drain(self) -> None:
        super()._drain()
        from rclpy.serialization import deserialize_message
        from policy_control.raw_poll import take_all, take_last
        for raw, _t in take_all(self._sub_head):
            msg = deserialize_message(raw, self._JointState)
            self._head.on_state(stamp_ns(msg), msg.name, msg.position)
        for k, sub in self._poll_pose.items():                         # 캡 · 리시버 FP++ → self.cups["pose_<k>"](받은 시각 그대로)
            got = take_last(sub)
            if got is not None:
                self._apply_cup(f"pose_{k}", deserialize_message(got[0], self._Pose), got[1])
        got = take_last(self._sub_depth)
        if got is not None:
            try:
                msg = deserialize_message(got[0], self._Image)
                self._depth = (image_array(msg, CS.IMG_HW, "32FC1"), stamp_ns(msg))
                self._errors.pop("head_depth", None)
            except CS.StaleDepth as exc:
                self._errors["head_depth"] = str(exc)

    def _frame(self) -> tuple[np.ndarray, float]:
        """(깊이 m, 나이 s). 나이 상한은 CapStudent.step 이 본다."""
        if self._depth is None:
            raise CS.StaleDepth(f"{DEPTH_TOPIC} 를 아직 못 받았다 {self._errors.get('head_depth', '')}")
        return self._depth[0], (time.time_ns() - self._depth[1]) * 1e-9

    def _poses(self) -> dict:
        """지금 공급 자세 {"cap", "rcv"} — 수신 · stamp 나이 둘 다 CUP_STALE_S 안이면 FP++(리시버는 쉐이커 → env 원점 환산), 아니면 고정 자리."""
        now, wall = time.monotonic(), time.time()
        return {k: CSC.pick_pose(self.cups.get(f"pose_{k}"), self._fixed[k], now, CUP_STALE_S,
                                 CSC.rcv_from_shaker if k == "rcv" else None, wall=wall) for k in ("cap", "rcv")}

    def _start_refusals(self, meas: dict, tol: float) -> list:
        m, s = meas["arm"], self.contract.side()
        self._start_poses = poses = self._poses()                      # start 가 받아들여지면 이 값을 에피소드 내내 쓴다
        return [*start_refusals(m.arm_q, tol), *hand_open_refusals([m.hand_q[j] for j in s.hand_joints], s.hand_open),
                *CSC.cap_refusals(poses["cap"]), *CSC.rcv_refusals(poses["rcv"]),
                *CSC.rcv_fixed_refusals(poses["rcv"], self._rcv_allow_fixed, CUP_STALE_S),
                *depth_configs_refusals(list(self._depth_cfgs))]          # reset 뒤 사람 확인 동안 올라온 깊이 노드도 잡는다

    def _measure(self, advance: bool = False) -> dict:
        meas = super()._measure(advance)
        return {**meas, "depth": self._frame(), "head": self._head.latest(time.time_ns())} if advance else meas

    # ---------------------------------------------------------------- episode
    def _wait_depth(self) -> None:
        """reset 때 깊이가 낡았으면 DEPTH_START_WAIT_S 까지 새 프레임을 기다린다(rh_place_student_node 11.10). 못 받으면 StaleDepth."""
        t0 = time.monotonic()
        while True:
            try:
                age = self._frame()[1]
                if 0.0 <= age <= self._policy.max_depth_age_s:
                    return
                why = f"{DEPTH_TOPIC} 나이 {age:.3f} s"
            except CS.StaleDepth as exc:
                why = str(exc)
            if time.monotonic() - t0 >= DEPTH_START_WAIT_S:
                raise CS.StaleDepth(f"reset: {why}")
            time.sleep(0.01)
            self._drain()

    def _reset_chain(self, meas: dict) -> None:
        held = self._held(meas)
        if held is None and not bool(self.get_parameter("allow_measured_start").value):
            raise PourFjNodeError("pd 가 붙잡은 마지막 joint_target 이 없다 — 시작 자세 이동 뒤에 띄울 것(실측에서 시작하려면 allow_measured_start)")
        bad = start_refusals(meas["arm"].arm_q, self.reset_tol)
        if bad:
            raise PourFjNodeError(bad[0])
        bad = depth_configs_refusals(list(self._depth_cfgs))
        if bad:   # 깊이 노드를 바꿨는데 옛 config 가 남아 거부되면 학생 노드를 다시 띄운다(죽은 발행자 것도 받은 것으로 친다)
            raise PourFjNodeError(" · ".join(bad))
        self._wait_depth()
        CS.head_obs(self._head.latest(time.time_ns()))                 # 목 각이 없거나 학습 한계 밖이면 reset 거부
        self._scripts.fk = self.fk["arm"]
        self.chain.scripts = self._scripts
        self.chain.reset(meas, held=held)
        self._phase_log = None

    def _srv_start(self, req, res):
        self._start_poses = None
        res = super()._srv_start(req, res)
        if res.success and self.chain is not None:
            p = self._start_poses
            self.chain.poses = {k: CSC.as_pair(v) for k, v in p.items()}
            self.get_logger().info("start 자세 · " + " · ".join(f"{k} {json.dumps(CSC.pose_view(v))}" for k, v in p.items()))
            if p["rcv"]["src"] == "fixed":                             # rcv_allow_fixed 로만 여기 온다
                self.get_logger().warning(f"리시버 FP++ 없이 고정 자리로 시작(rcv_allow_fixed) · {json.dumps(CSC.pose_view(p['rcv']))}")
        return res

    def _srv_next(self, _req, res):
        if self.book.phase != "running" or self.chain is None:
            return self._reply(res, False, [f"stage/next: 실행 중이 아니다(phase {self.book.phase})"])
        bad = next_refusal(self.chain.stages.phase, self.chain.phase_steps, self._gap, self.chain.carry_missed)
        if bad:
            return self._reply(res, False, bad)
        was, now = self.chain.stages.phase, self.chain.next()
        if now == DONE:
            self._end("stop", "seat 끝 — 사람이 끼움을 확인했다")       # pd 가 마지막 목표로 붙든다
        return self._reply(res, True, [f"{was} → {now}"])

    def _after_step(self) -> bool:
        ph = self.chain.stages.phase
        if ph != self._phase_log:
            self.get_logger().info(f"구간 {self._phase_log} → {ph} · 단계 칸 {STAGE_OF[ph]} · 스텝 {self.chain.step_i}")
            self._phase_log = ph
        return False

    def _on_tick(self) -> None:
        running = self.book.phase == "running" and self.chain is not None
        if running:
            self._drain()
            try:
                self._head.latest(time.time_ns())
            except PS.PlaceInputStale as exc:                          # 목 상태가 끊기면 gap 예산 없이 멈춘다(놓기 학생과 같다)
                self._end("abort", f"머리 명령 멈춤 · {exc}")
                return
        n0 = self.chain.step_i if running else None
        super()._on_tick()
        # 이 틱 학생이 성공했을 때만 · 무발행(publish_target false) 점검에서는 목도 움직이지 않는다
        if (self._head_on and self._publish and self._lg_stop is None and self.book.phase == "running"
                and n0 not in (None, self.chain.step_i)):
            self._head_seq += 1
            self._pub_head.publish(self._String(data=head_stream_body(self._head_seq, self.chain.head_cmd)))

    def _status_extra(self) -> dict:
        poses = {k: CSC.pose_view(v) for k, v in self._poses().items()}     # 지금 공급 쪽(fpp / fixed) · 마지막 FP++ 나이
        start = None if self._start_poses is None else {k: v["src"] for k, v in self._start_poses.items()}
        out = {"poses": poses, "poses_start": start}
        return {**out, "cap_student": self.chain.as_dict(), "head_bad": self._head.bad} if self.chain is not None else out


if __name__ == "__main__":
    raise SystemExit(main(family="rh_cap", node_cls=CapStudentNode))
