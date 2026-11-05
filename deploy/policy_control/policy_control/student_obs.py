"""11.04 FP++ 없는 깊이 학생(rh_aglt 잡기)의 고유수용 관측 60 칸.

학생 관측 = 교사 actor 칸에서 컵 · 목표 칸(FP++ 지각)을 뺀 것 + 단계 one-hot 2 칸(hdgp scripts/tools/student_adapters/rh_aglt.py
student_obs · keep_index · STAGES, origin/main ed11c791). 칸 값은 교사 배포 관측(rh_aglt.build_obs)을 그대로 불러 그 칸만 잘라 쓴다 —
계산을 따로 두지 않아 교사와 어긋날 곳이 없다.
sim 학생 관측은 actor 클립(obs_clip)을 거치지 않는다(env._actor_parts 에 nan_to_num 만) → 클립을 끈 계약으로 부른다.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import rh_aglt as A

#: hdgp rh_aglt.CUP_PARTS — 학생 관측에서 빼는 actor 칸
CUP_PARTS = ("cup_rel_palm", "cup_up", "tips_rel_cup", "kp_rel_goal", "goal_rel_palm")
STAGES = ("approach", "grasp")
#: ckpt prop_cols 가 이것과 다르면 거부한다(student_policy). 촉각 없는 교사(rcg04) · 물체 3 종 기준.
PROP_COLS = (("arm_q", 7), ("arm_qd", 7), ("arm_q_target", 7), ("hand_q", 6), ("hand_err", 6), ("palm_pos", 3),
             ("palm_rot", 6), ("prev_action", 13), ("object_id", 3), ("stage_approach", 1), ("stage_grasp", 1))
_NO_CUP = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])


def student_layout(c: A.RaContract) -> tuple:
    """계약의 actor 칸에서 컵 칸을 빼고 단계 칸을 붙인 (이름, 폭) — PROP_COLS 와 같아야 한다."""
    return tuple(x for x in A.actor_layout(c) if x[0] not in CUP_PARTS) + tuple((f"stage_{s}", 1) for s in STAGES)


def build_student_obs(c: A.RaContract, m: A.RaMeas, dec: A.RaDecoder, prev_action, stage: str) -> np.ndarray:
    """RaMeas + 디코더 상태(직전 목표) + 물체(c.active_object) + 단계 → (60,) float64. prev_action = 직전 틱 실행 행동."""
    lay = student_layout(c)
    if lay != PROP_COLS:
        raise A.RhAgltError(f"학생 관측 칸 {lay} ≠ {PROP_COLS} — 계약(촉각 · 물체 · extra)이 학생 학습 env 와 다르다")
    if stage not in STAGES:
        raise A.RhAgltError(f"단계 {stage!r} ∉ {STAGES}")
    # build_obs 는 nan_to_num 으로 NaN 을 0 으로 바꾸므로 정책 입력 칸은 여기서 거부한다(컵 칸은 아래에서 버리니 NaN 허용)
    bad = [k for k, v in (("arm_q", m.arm_q), ("arm_qd", m.arm_qd), ("hand_q", list(m.hand_q.values())),
                          ("palm_pos", m.palm_pos), ("palm_R", m.palm_R)) if not np.isfinite(np.asarray(v, float)).all()]
    if bad:
        raise A.RhAgltError(f"학생 관측 입력이 유한하지 않다: {bad}")
    # 컵 칸은 잘라 버리므로 FP++ 값이 없어도(NaN · 옛 값) 결과가 같게 고정값을 넣는다
    m0 = replace(m, cup_pos=_NO_CUP[0], cup_quat=_NO_CUP[1])
    full = A.build_obs(replace(c, obs_clip=np.inf), m0, dec, A.Goal(*_NO_CUP), prev_action)
    out, i = [], 0
    for name, w in A.actor_layout(c):
        if name not in CUP_PARTS:
            out.append(full[i:i + w])
        i += w
    out.append(np.array([stage == s for s in STAGES], float))
    return np.concatenate(out)
