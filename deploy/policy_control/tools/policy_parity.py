#!/usr/bin/env python3
"""정책 재현 시험: bag 에 기록된 obs 를 배포 로더(JointPolicy)로 ckpt 에 다시 넣어 기록된 행동과 대조한다(읽기 전용).

배포와 같게 노드 창 첫 틱에 은닉 상태를 0 으로 두고 틱마다 이어 간다. prev_action 은 기록된 obs 안에 이미 있다.
행동 = clip(forward(obs), ±action_clip) — pour_fj_node(aglt /action) · rh_place(policy_action) 과 같은 식.
obs 와 행동은 seq 로 짝짓는다(place 는 정책을 안 돌린 틱(settling)에 policy_action 이 없다).
sim_real_compare.py 가 노드마다 부른다. 단독 실행은 없다.
"""
from __future__ import annotations

import numpy as np

import run_bag as RB  # noqa: F401  (sys.path 에 policy_control 을 넣는다)
from policy_control import codec

TOL = 1e-4      # 같은 ckpt · 같은 obs 면 CPU/GPU 차이도 이 안이다(float32)
_POL: dict = {}


def load_policy(contract, device: str = "cpu"):
    """배포 노드와 같은 로더. 계약 경로마다 한 번만 읽는다."""
    from policy_control.joint_policy import JointPolicy
    key = (str(contract.checkpoint), device)
    if key not in _POL:
        _POL[key] = JointPolicy(contract, device)
    return _POL[key]


def pair(obs_msgs: list, act_msgs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, Float64MultiArray) 목록 둘 → seq 가 같은 틱만 (seq, O, U). seq 순."""
    O = {s.seq: s.data for s in (codec.decode_float_array(m) for _, m in obs_msgs)}
    U = {s.seq: s.data for s in (codec.decode_float_array(m) for _, m in act_msgs)}
    seq = np.array(sorted(set(O) & set(U)), int)
    if not len(seq):
        return seq, np.zeros((0, 0)), np.zeros((0, 0))
    return seq, np.array([O[k] for k in seq], float), np.array([U[k] for k in seq], float)


def replay(policy, O: np.ndarray, action_clip: float) -> np.ndarray:
    policy.reset()
    return np.array([np.clip(np.asarray(policy.forward(o), float), -action_clip, action_clip) for o in O])


def compare(U_hat: np.ndarray, U: np.ndarray, tol: float = TOL) -> dict:
    d = np.abs(U_hat - U).max(1)
    bad = np.flatnonzero(d > tol)
    return dict(n=int(len(d)), max_abs=float(d.max()) if len(d) else None,
                first_mismatch=int(bad[0]) if len(bad) else None, tol=tol)


def parity(contract, obs_msgs: list, act_msgs: list, device: str = "cpu") -> dict:
    seq, O, U = pair(obs_msgs, act_msgs)
    if not len(seq):
        return dict(n=0, max_abs=None, first_mismatch=None, note="obs·행동 seq 짝 없음")
    if O.shape[1] != contract.obs_dim or U.shape[1] != contract.action_dim:
        return dict(n=len(seq), max_abs=None, first_mismatch=None, note=f"폭 obs {O.shape[1]} act {U.shape[1]} ≠ 계약")
    gaps = int((np.diff(seq) != 1).sum())
    r = compare(replay(load_policy(contract, device), O, contract.action_clip), U)
    r.update(seq0=int(seq[0]), seq_gaps=gaps)     # seq 가 끊기면(틱 오류) 배포 은닉 상태와 어긋날 수 있다
    return r


def fmt(r: dict | None) -> str:
    if not r:
        return ""
    if r.get("max_abs") is None:
        return r.get("note", "")
    fm = "없음" if r["first_mismatch"] is None else str(r["first_mismatch"])
    return f"{r['max_abs']:.1e} 첫불일치 {fm}/{r['n']}" + (f" seq끊김 {r['seq_gaps']}" if r.get("seq_gaps") else "")

