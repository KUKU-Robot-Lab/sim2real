"""11.05 깊이 학생(rh_aglt 잡기, 1104e) 한 스텝 — pour_fj_node.AgltChain 과 같은 자리(reset · step), 노드는 rh_aglt_student_node.

sim(hdgp 0bd455a8, aglt_start=ap · student_adapters/rh_aglt.py)에서 학생은 홈부터 접근 · 잡기 · 들기를 한 정책으로 운전한다:
  - 디코더 하나가 에피소드 내내 같은 행동 법칙(팔 증분 · 손 grip)으로 푼다. env 의 법칙은 hold_steps 만 끄고(_ctl_hold_mask)
    접근 · 정지 동안에도 켜져 있다(rh_aglt_env.py 1095-1137). 그래서 계약 = 홈 출발(arm_start home) 계약(ap05 와 rcg04 는
    arm_start · ckpt 해시만 다르다), 디코더는 홈에서 reset 한 그대로 넘김 뒤에도 이어 간다(경로 D 처럼 바꾸지 않는다).
  - 실행 행동: hold_steps 와 넘김 정지(stop_s) = (팔 0, 손 폄)(_a_open_rdy, env 1106-1113), 그 밖 = 학생 출력.
    직전 행동 관측 칸 = 실행 행동(env 1729), 리셋 직후 0(env 2034).
  - 단계 one-hot: hold · 접근 = approach, 넘긴 틱부터 grasp(adapter stage_onehot: state <= MOVE). 넘김 = StudentHandover.
  - 팔 들림 래치 · lift_hold 기준 높이 = 정책이 처음 움직이는 스텝(접근 · 정지가 끝난 첫 잡기 틱)의 관측 컵 z
    (env 1567-1570 _fz_z0, _hold_mask 에 aps.busy() 포함). 그래서 그 전에는 디코더에 cup_z = nan 을 주고(z0 미설정 · 래치 없음),
    목표 장부(GoalBook)는 첫 잡기 틱의 컵으로 다시 연다(경로 D handover 와 같은 방식). 판정(goals.step)도 잡기 틱에만.
FP++ 컵(m.cup_pos)은 관측에 넣지 않는다(student_obs 가 컵 칸을 뺀다) — 래치 · lift_hold · 엄지 받침 거리 · 쥠 보조에만 쓴다.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import rh_aglt as A
from . import rh_aglt_goals as G
from .cup_attach import grasp_signal
from .pour_fj_node import AgltChain
from .student_handover import StudentHandover
from .student_obs import build_student_obs
from .student_policy import STUDENT_CARD, StaleDepth

#: 접근 상한 초과 = 실패(배포 approach_max_s abort, hdgp patch_ap_rule 의 rule.fail 과 같다)
ON_MAX = "fail"


def spec_limits(policy_cfg: dict) -> tuple[float, float]:
    """학생 묶음 student.yaml source.spec(평가 spec.json) phys → (접근 상한 s, 에피소드 길이 s). sim 평가 조건을 그대로 쓴다."""
    p = Path(policy_cfg["dir"]) / str(policy_cfg["source"]["spec"])
    phys = json.loads(p.read_text(encoding="utf-8"))["phys"]
    hz = 60.0     # ap_max_steps 는 정책 스텝(60 Hz) — 계약 policy_hz 와 다르면 체인이 거부한다
    ap_s, ep_s = int(phys["ap_max_steps"]) / hz, float(phys["episode_length_s"])
    if phys.get("aglt_start") != "ap" or not (0 < ap_s < ep_s):
        raise A.RhAgltError(f"{p}: aglt_start {phys.get('aglt_start')!r}(ap 여야) · 접근 상한 {ap_s} s · 에피소드 {ep_s} s")
    return ap_s, ep_s


def bundle_limits(student_id: str, root: Path | None = None) -> tuple[float, float]:
    """등록부 상대 id(예 rh56f1/aglt/right_student1104e) → spec_limits. 실행기 · 미션 생성기가 노드와 같은 한도를 쓴다."""
    import yaml

    from . import _paths

    d = (root or _paths.SIM2REAL / "deploy" / "policies") / student_id
    cfg = yaml.safe_load((d / STUDENT_CARD).read_text(encoding="utf-8")) or {}
    return spec_limits({**cfg, "dir": d})


class StudentChain(AgltChain):
    """관측(컵 칸 없음) → 학생(깊이 · RGB · LSTM) → 디코더. 깊이는 meas["depth"] = (depth_m, rgb | None, age_s)."""

    def __init__(self, c: A.RaContract, policy) -> None:
        if c.arm_start != "home":
            raise A.RhAgltError(f"학생 계약은 홈 출발(arm_start home, 예 *_approach_ap05)이어야 한다: {c.arm_start}")
        if c.policy_hz != 60.0:
            raise A.RhAgltError(f"학생은 60 Hz 로 학습했다: 계약 policy_hz {c.policy_hz}")
        super().__init__(c, policy)
        ap_s, _ = spec_limits(policy.policy_cfg)
        self.hand = StudentHandover(hz=c.policy_hz, max_s=ap_s, on_max=ON_MAX, **policy.handover)
        self.end: tuple[str, str] | None = None
        self.grasp_started = False

    @property
    def phase(self) -> str:
        """pour_fj_node 의 '접근 단계 쥔 신호 버림'이 읽는다 — 넘김 전 = approach."""
        return self.hand.stage

    def reset(self, meas: dict | None = None) -> None:
        if self.ag is not None or self.inhand is not None:
            raise A.RhAgltError("학생 체인은 approach_grasp · 손 안 추정기(inhand_est)를 쓰지 않는다 — 접근도 학생이 운전한다")
        super().reset(meas)
        self.hand.reset()
        self.end, self.grasp_started = None, False

    def handover(self, *_a, **_k) -> None:
        raise A.RhAgltError("학생 체인에는 경로 D 넘김이 없다")

    def step(self, meas: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        frame = meas.get("depth")
        if frame is None:
            raise StaleDepth("깊이 프레임이 없다")
        depth_m, rgb, age = frame
        self.policy.check_inputs(depth_m, age, rgb)     # 넘김 · 디코더 상태를 바꾸기 전에 — 낡은 틱을 접근 · 정지 시간으로 세지 않는다
        m = self._obs_meas(meas)["arm"]
        hold_a = A.ready_hold_action(self.c)
        active = self.step_i >= self.c.hold_steps
        stage, stopping = "approach", False
        if active and self.end is None:
            stage, stopping, failed = self.hand.step(m.palm_pos)
            if failed:
                self.end = ("abort", f"학생 접근 실패 — {self.hand.why}")
        driving = active and self.end is None and not stopping
        grasping = driving and stage == "grasp"
        if grasping and not self.grasp_started:     # 첫 잡기 틱 = sim _fz_z0 스텝 — 목표 · lift_hold 기준을 이 컵으로
            if not np.all(np.isfinite(np.asarray(m.cup_pos, float))):
                raise A.RhAgltError(f"첫 잡기 틱 컵 위치가 유한값이 아니다({m.cup_pos}) — FP++ 컵을 본다")
            old = self.goals
            self.goals = G.GoalBook.start(self.c, m.cup_pos, m.cup_quat, freeze=self.freeze)
            if old is not None and old.target is not None:     # 잡기 전에 받은 목표 요청 — 줄은 새 컵 기준으로 다시 짠다
                bad = self.goals.request(old.target)
                if bad:
                    raise A.RhAgltError(f"잡기 전에 받은 목표를 새 컵 기준으로 옮기지 못했다: {bad}")
            self.grasp_started = True
        obs = build_student_obs(self.c, m, self.dec, self.prev, stage)
        a = self.policy.step(depth_m, obs, age, rgb)     # 정지 · 대기 틱에도 돌린다(LSTM · 깊이 지연이 sim 처럼 이어진다)
        ex = a if driving else hold_a
        hand_q = np.array([m.hand_q[j] for j in self.c.side().hand_joints], float)
        near = float(np.linalg.norm(np.asarray(m.cup_pos, float) - np.asarray(m.palm_pos, float)))
        # 디코더 z0 = 첫 잡기 틱 관측 컵 z. sim 은 첫 정책 스텝 행동 뒤 관측(한 틱 뒤)이라 mm 이하 차이
        cup_z = float(np.asarray(m.cup_pos, float)[2]) if grasping else float("nan")
        targets = self.dec.step(ex, active=active, tactile_n=m.tactile_n, hand_q=hand_q, joint_force=m.joint_force,
                                obj_palm_m=near, cup_z=cup_z)
        if grasping:                                # AgltChain.step 과 같은 쥔 판정
            g = self.dec.grasped()
            if g is None and self.grasp_cfg is not None:
                g = grasp_signal(m.tactile_n, m.joint_force, replace(self.grasp_cfg, force_n=self.c.grasp_threshold_n))
            if not g and self.lift_grasped(m):
                g = True
            self.goals.step(m.cup_pos, m.cup_quat, m.tactile_n, is_grasped=g, fresh=self.cup_fresh)
        self.prev = np.asarray(ex, float).copy()
        self.step_i += 1
        return obs, self.prev.copy(), targets

    def view(self) -> dict:
        h = self.hand
        return {"stage": h.stage, "approach_s": round(h.n / self.c.policy_hz, 2),
                "palm_v_mps": None if not np.isfinite(h.v) else round(h.v, 4), "handover": h.why,
                "depth_empty": bool(getattr(self.policy, "depth_empty", False)), "end": self.end}
