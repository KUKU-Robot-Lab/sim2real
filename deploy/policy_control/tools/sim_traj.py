#!/usr/bin/env python3
"""10.19 sim 궤적 실행 — 실기 컵 자리에서 sim 정책을 돌려 뽑은 관절 궤적을 pd 로 재생한다(정책 노드 없이).

사용자 10.19: "sim 에서 뽑은 trajectory 를 실행 — FP++ 를 바탕으로 구현된 sim 환경에서 그대로 따라가게"(+ 노드 끊김 처리 둘 다).

  plan   실기 컵 자리(FP++ live 를 모은 중앙값, 또는 --cup x,y) · 정책 폴더(deploy/policies/<id>)로 server sim 을 trials 번
         돌린다(코드 고정본 our_source/s2r_simtraj/hdgp_51094780 = neo03 학습 커밋 · rh_aglt_grid_eval hook · 실기 규격 REAL ·
         결정론 정책, 시행마다 다른 건 물리 DR · FP++ 모델 잡음). 그 물체로 목표(컵 10 cm 위)에 성공한 시행 중 손바닥 경로가
         가장 가운데인 것을 고른다 → plan.npz(팔 = sim 실제 관절, 손 = sim 손 PD 목표, 60 Hz, 성공 프레임까지) + 영상.
         sim 좌표 = 로봇 base(URDF FK 손바닥 = sim 손바닥, 10.19 확인) — FP++ 컵 xy 를 그대로 넣는다.
  run    plan.npz 를 pd 로 재생. ★실기 동작 — 사용자 승인 뒤 --execute(없으면 검사만).
         검사: 정책 노드가 달리지 않음 · pd TRACKING · 손 입력 position · 팔이 첫 프레임 0.15 rad 안 · 컵(FP++ live)이 계획
         자리 1 cm 안. 손가락이 닫히기 시작하는 프레임에서 팔이 그 자세(손바닥 1 cm · 관절 0.05 rad) 안에 들어올 때까지
         기다린다 — 실기 10.19 손목 j7 이 정책 목표보다 0.15~0.20 rad 늦어 손바닥이 4~5 cm 어긋났다. 3 s 안에 안 들어오면
         멈춤(pd 가 지금 자세로 붙든다). 끝에 episode stop — pd 가 마지막 자세 · 손을 붙든다.

sim 열린 루프 재생(10.19, 원본 성공 궤적의 PD 목표를 컵 자리만 옮겨 재생 · 7 cm 들고 쥔 채 0.5 s): 0 cm 4/6 · 1 cm 27/48 ·
2 cm 33/96 · 3 cm 20/144(shaker 0 cm 2/2 · 1 cm 13/16 · 2 cm 8/32) — 컵 자리 오차가 1 cm 를 넘으면 잘 안 된다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
from policy_control import _paths  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402

SERVER = "server"
HDGP = "~/rl_ws/our_source/s2r_simtraj/hdgp_51094780"      # neo03 학습 커밋 고정본(README 참고)
JOBS = "/tmp/s2r_simtraj"
OUT = _paths.SIM2REAL / "logs" / "sim_traj"
SIDE_KEY = {"left": "l", "right": "r"}
CLOSE_RISE_RAD = 0.08        # 손가락(검지~새끼) 목표가 접근 중 값보다 이만큼 굽기 시작하면 닫기 시작
CUP_TOL_M = 0.01             # sim 열린 루프 재생: 1 cm 13/16 · 2 cm 8/32(shaker)
SETTLE_PALM_M = 0.01
SETTLE_Q_RAD = 0.05
SETTLE_TIMEOUT_S = 3.0
MAX_START_GAP_RAD = 0.15
MAX_ARM_SPEED = 1.5          # rad/s(재생 배속 반영) — 넘는 계획은 거부


class SimTrajError(RuntimeError):
    pass


# ============================================================================ 순수
def sim_cup_name(env_object_names, real_name: str) -> str:
    """배포 물체 이름(source240_real · shaker_c_real …) → 학습 env 물체 이름(_cvx 등)."""
    names = [str(n) for n in env_object_names]
    hit = [n for n in names if A.SIM_COLLIDER_ALIAS.get(n, n) == real_name]
    if len(hit) != 1:
        raise SimTrajError(f"물체 {real_name!r} — 학습 물체 {names}(→ {[A.SIM_COLLIDER_ALIAS.get(n, n) for n in names]})")
    return hit[0]


def close_start(hand_pd: np.ndarray, grasp: int, rise: float = CLOSE_RISE_RAD) -> int:
    """손가락(검지~새끼, 열 2:) 목표가 접근 중(쥠 1.5~0.5 s 전 중앙값)보다 rise 넘게 굽기 시작한 프레임.
    sim neo03 shaker 8 시행: 쥠 0.15~0.27 s 전 · 손바닥은 그 사이 1.0~1.8 cm 더 간다."""
    h = np.asarray(hand_pd, float)[:, 2:]
    lo, hi = max(0, grasp - 90), max(1, grasp - 30)
    base = np.median(h[lo:hi], axis=0)
    up = (h - base).max(axis=1)
    k = int(grasp)
    while k > 0 and up[k] > rise:
        k -= 1
    return k + 1 if k < grasp else int(grasp)


def resample(path: np.ndarray, n: int = 100) -> np.ndarray:
    p = np.asarray(path, float)
    s = np.linspace(0.0, 1.0, len(p))
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, s, p[:, i]) for i in range(p.shape[1])], axis=1)


def medoid(paths: list) -> int:
    """정규화 시간으로 맞춘 손바닥 경로들 중 다른 것들과의 평균 거리가 가장 작은 것."""
    r = [resample(p) for p in paths]
    cost = [sum(float(np.linalg.norm(a - b, axis=1).mean()) for b in r) for a in r]
    return int(np.argmin(cost))


def first(mask) -> int | None:
    i = np.flatnonzero(np.asarray(mask))
    return int(i[0]) if i.size else None


def choose(trace, rows: list, spec: dict, target: str) -> dict:
    """trace.npz · result rows · spec → 고른 시행(env, 성공 프레임, 쥔 프레임, 닫기 시작)과 후보 요약."""
    envs = [i for i, e in enumerate(spec["envs"]) if e["cup"] == target]
    by_env = {int(r["env"]): r for r in rows}
    ok = []
    for e in envs:
        r = by_env.get(e, {})
        s = first(trace["success"][:, e])
        g = first(trace["grasped"][:, e])
        if r.get("end") == "success" and s is not None and g is not None and g < s:
            ok.append((e, s, g))
    if not ok:
        ends = {e: by_env.get(e, {}).get("end") for e in envs}
        raise SimTrajError(f"{target}: sim {len(envs)} 시행 중 성공 0 — 끝 {ends}")
    k = medoid([trace["palm"][: s + 1, e] for e, s, _ in ok])
    e, s, g = ok[k]
    c = close_start(trace["hand_pd"][:, e], g)
    spread = np.std([trace["palm"][close_start(trace["hand_pd"][:, e2], g2), e2] for e2, _, g2 in ok], axis=0)
    return {"env": e, "success": s, "grasp": g, "close": c, "n_ok": len(ok), "n": len(envs),
            "palm_close_std_cm": [round(float(v) * 100, 2) for v in spread], "row": by_env.get(e, {})}


def plan_arrays(trace, pick: dict, side: A.RaSide) -> dict:
    e, s = pick["env"], pick["success"]
    arm = np.asarray(trace["arm_q"][: s + 1, e], float)
    hand = np.asarray(trace["hand_pd"][: s + 1, e], float)
    return {"names": np.array(list(side.arm_joints) + list(side.hand_joints)), "q": np.concatenate([arm, hand], axis=1),
            "dt": float(trace["dt"]), "close": int(pick["close"]), "grasp": int(pick["grasp"]), "success": int(s),
            "palm": np.asarray(trace["palm"][: s + 1, e], float), "cup": np.asarray(trace["cup"][: s + 1, e], float)}


def check_plan(plan: dict, side: A.RaSide, fk=None, rate_scale: float = 0.5) -> list[str]:
    """한계 · 속도 · (fk 를 주면) URDF FK 손바닥 = sim 손바닥."""
    bad = []
    q = plan["q"]
    arm, hand = q[:, :7], q[:, 7:]
    lo, hi = np.asarray(side.arm_lo), np.asarray(side.arm_hi)
    if (arm < lo - 1e-3).any() or (arm > hi + 1e-3).any():
        bad.append(f"팔 관절 한계 밖 {np.flatnonzero(((arm < lo - 1e-3) | (arm > hi + 1e-3)).any(0)).tolist()}")
    hlo, hhi = np.asarray(side.hand_lim_lo), np.asarray(side.hand_lim_hi)
    if (hand < hlo - 1e-3).any() or (hand > hhi + 1e-3).any():
        bad.append("손 관절 한계 밖")
    v = np.abs(np.diff(arm, axis=0)).max() / plan["dt"] * rate_scale
    if v > MAX_ARM_SPEED:
        bad.append(f"팔 속도 {v:.2f} rad/s > {MAX_ARM_SPEED} (배속 {rate_scale})")
    if not np.isfinite(q).all():
        bad.append("비유한 값")
    if fk is not None:
        d = max(float(np.linalg.norm(fk.palm_pose(arm[k], hand[k]).palm_pos - plan["palm"][k]))
                for k in range(0, len(q), 10))
        if d > 0.005:
            bad.append(f"URDF FK 손바닥이 sim 손바닥과 {d * 100:.1f} cm 다르다(좌표 · 관절 순서 확인)")
    return bad


def run_refusals(plan: dict, meta: dict, arm_now, cup_now, pd_arm: dict | None, policy_running: bool,
                 cup_tol: float = CUP_TOL_M, max_gap: float = MAX_START_GAP_RAD) -> list[str]:
    """run 전 거부 사유(빈 목록이면 실행 가능)."""
    why = []
    if policy_running:
        why.append("정책 노드가 달리는 중 — 에피소드를 끝낸 뒤")
    if pd_arm is None:
        why.append("pd 상태 없음")
    else:
        if pd_arm.get("phase") != "TRACKING":
            why.append(f"pd phase {pd_arm.get('phase')}")
        if pd_arm.get("hand_command", "position") != "position":
            why.append(f"pd 손 입력 {pd_arm.get('hand_command')} — position 이어야(sim 손 PD 목표를 위치로 보낸다)")
    if arm_now is None:
        why.append("/joint_states 팔 실측 없음")
    else:
        gap = float(np.abs(np.asarray(arm_now) - plan["q"][0, :7]).max())
        if gap > max_gap:
            why.append(f"팔이 첫 프레임에서 {gap:.3f} rad(한계 {max_gap}) — 홈으로 먼저")
    if cup_now is None:
        why.append("FP++ live 컵 없음")
    else:
        d = float(np.linalg.norm(np.asarray(cup_now)[:2] - np.asarray(meta["cup_xy"])))
        if d > cup_tol:
            why.append(f"컵이 계획 자리에서 {d * 100:.1f} cm(한계 {cup_tol * 100:.0f} cm) — 다시 plan")
    return why


def settle_error(fk, arm_meas, target: np.ndarray, hand_q) -> tuple[float, float]:
    """(관절 최대 오차 rad, 손바닥 위치 오차 m) — 손 관절은 계획값으로 FK."""
    q = np.asarray(arm_meas, float)
    eq = float(np.abs(q - target[:7]).max())
    ep = float(np.linalg.norm(fk.palm_pose(q, hand_q).palm_pos - fk.palm_pose(target[:7], hand_q).palm_pos))
    return eq, ep


def schedule(plan: dict) -> list[tuple[str, int]]:
    """재생 순서: 닫기 직전까지 → 정착 대기 → 끝까지."""
    c, n = int(plan["close"]), len(plan["q"])
    return [("frame", k) for k in range(c)] + [("settle", c)] + [("frame", k) for k in range(c, n)]


# ============================================================================ 정책 · 서버
def policy_dir(pid: str) -> Path:
    d = Path(pid)
    if not d.is_dir():
        d = _paths.SIM2REAL / "deploy" / "policies" / pid
    if not (d / "rh_aglt_contract.json").is_file():
        raise SimTrajError(f"{d}: rh_aglt_contract.json 없음(rh_aglt 정책 폴더)")
    return d


def load_policy(pid: str) -> tuple[Path, A.RaContract, Path, dict]:
    d = policy_dir(pid)
    c = A.load_contract(d / "rh_aglt_contract.json")
    ck = d / "nn" / Path(c.checkpoint).name
    if not ck.is_file():
        raise SimTrajError(f"체크포인트 없음 {ck}")
    md5 = hashlib.md5(ck.read_bytes()).hexdigest()
    if md5 != c.checkpoint_md5:
        raise SimTrajError(f"{ck.name} md5 {md5} ≠ 계약 {c.checkpoint_md5}")
    env = A.read_env(d / "params" / "env.yaml")
    return d, c, ck, env


def _sh(cmd: list[str], timeout: float, check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SimTrajError(f"{' '.join(cmd[:3])}… rc={r.returncode}: {(r.stderr or r.stdout)[-800:]}")
    return r.stdout


def ssh(cmd: str, timeout: float = 120, check: bool = True) -> str:
    return _sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", SERVER, cmd], timeout, check)


PREP_GRID = 2                # prep 격자 최소(1 은 0 나누기) — respec 이 덮어쓴다
CKPT_ROOTS = ("~/rl_ws/hdgp/log/rl_games", "~/rl_ws/our_source")   # server 에 이미 있는 학습 런 — 같은 md5 면 복사로 대신


def _remote_md5(path: str) -> str | None:
    got = ssh(f"md5sum {path} 2>/dev/null | cut -d' ' -f1", 120, check=False).strip()
    return got or None


def upload_ckpt(ck: Path, md5: str, dest_dir: str) -> str:
    """체크포인트를 server {dest_dir}/{ck.name} 에 둔다. 반환 = 어떻게 했는지.

    10.19: 98 MB 를 scp 로 올리다 300 s 에 끊겨 잘린 파일(md5 다름)이 남았다. 그래서
    ① 이미 같은 md5 면 그대로 ② server 학습 런에 같은 이름·md5 가 있으면 server 안에서 복사
    ③ 없을 때만 .part 로 올리고 md5 를 맞춰 본 뒤 이름을 바꾼다 — 잘린 파일이 남지 않는다."""
    dest = f"{dest_dir}/{ck.name}"
    if _remote_md5(dest) == md5:
        return "server 에 이미 있음"
    roots = " ".join(CKPT_ROOTS)
    found = ssh(f"for r in {roots}; do [ -d \"$r\" ] && find \"$r\" -name '{ck.name}' -type f 2>/dev/null; done | head -20",
                300, check=False).split()
    for src in found:
        if _remote_md5(src) == md5:
            ssh(f"cp '{src}' '{dest}.part' && mv '{dest}.part' '{dest}'", 300)
            if _remote_md5(dest) != md5:
                raise SimTrajError(f"server 복사 뒤 md5 가 다르다: {dest}")
            return f"server 복사 ← {src}"
    _sh(["scp", "-q", str(ck), f"{SERVER}:{dest}.part"], 1800)
    if _remote_md5(f"{dest}.part") != md5:
        ssh(f"rm -f '{dest}.part'", 60, check=False)
        raise SimTrajError(f"올린 {ck.name} md5 가 다르다(잘렸을 수 있음) — 지웠다")
    ssh(f"mv '{dest}.part' '{dest}'", 60)
    return "scp 업로드"


JOB = """#!/usr/bin/env bash
# s2r_1st sim_traj: 실기 컵 자리 sim 궤적. 출력은 {job} 만.
cd {hdgp} || exit 2
source ~/miniforge3/etc/profile.d/conda.sh; conda activate proj-hdgp-py311
source /home/oem/isaacsim/5.1.0/setup_conda_env.sh
export PYTHONPATH="$PWD/source/openarm:${{PYTHONPATH:-}}" CUDA_VISIBLE_DEVICES={gpu} TERM=xterm PYTHONUNBUFFERED=1 RH_GRID_SPEC={job}/eval/spec.json
timeout {timeout} python scripts/tools/rh_aglt_grid_eval.py hook --task {task} --checkpoint {ckpt} --num_envs {n} --headless \\
  --adr_level 0 --seed 42{cams}
echo "RC=$?"
"""


def respec(spec: dict, x: float, y: float, trials: int) -> dict:
    """prep 의 격자 칸을 실기 컵 자리 한 칸으로 — env i 의 물체 = object_names[i % K] 규칙은 그대로."""
    objs = list(spec["objects"])
    envs = [{"cup": o, "cell": 0, "x": x, "y": y, "trial": t} for t in range(trials) for o in objs]
    return {**spec, "xs": [x], "ys": [y], "cells": [[x, y]], "dropped": [], "trials": trials, "envs": envs}


def server_plan(a, pdir: Path, c: A.RaContract, ck: Path, x: float, y: float, out: Path) -> Path:
    """server 에 정책을 올리고 prep → spec 고침 → hook(백그라운드) → 끝나면 받아 온다. 반환 = 받은 폴더."""
    job = f"{JOBS}/{out.name}"
    side = SIDE_KEY[c.sides["arm"].side]
    # --ckpt 를 주면 grid eval 은 --ep 를 쓰지 않는다(필수 인자라 받기만 함). 10.19 이름 바꾼 approach_r.pth 처럼
    # _ep_<n>_ 이 없는 이름은 0 을 넘긴다.
    ep = re.search(r"_ep_(\d+)_", ck.name)
    ep = ep.group(1) if ep else "0"
    ssh(f"mkdir -p {job}/src/params {job}/src/nn && ls -d {a.hdgp}/scripts/tools", 60)
    _sh(["scp", "-q", str(pdir / "params" / "env.yaml"), str(pdir / "params" / "agent.yaml"), f"{SERVER}:{job}/src/params/"], 120)
    print(f"[sim_traj] 체크포인트: {upload_ckpt(ck, c.checkpoint_md5, f'{job}/src/nn')}", flush=True)
    # 격자 2x2: prep 의 _grid 가 (nx-1) 로 나눈다(1 이면 0 나누기). 칸·env 는 respec 이 실기 컵 한 칸으로 다 바꾼다.
    prep = (f"cd {a.hdgp} && source ~/miniforge3/etc/profile.d/conda.sh && conda activate proj-hdgp-py311 && "
            f"python scripts/tools/rh_aglt_grid_eval.py prep --side {side} --run {job}/src --ep {ep} "
            f"--ckpt {job}/src/nn/{ck.name} --out {job}/eval --trials {a.trials} --nx {PREP_GRID} --ny {PREP_GRID} --forensic"
            + (" --video" if a.video else ""))
    got = ssh(prep, 300)
    m = re.search(r"--task (\S+)", got)
    if "PREP_OK" not in got or m is None:
        raise SimTrajError(f"prep 실패: {got[-800:]}")
    spec = json.loads(ssh(f"cat {job}/eval/spec.json", 60))
    spec = respec(spec, x, y, a.trials)
    (out / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    _sh(["scp", "-q", str(out / "spec.json"), f"{SERVER}:{job}/eval/spec.json"], 60)
    (out / "job.sh").write_text(JOB.format(job=job, hdgp=a.hdgp, gpu=a.gpu, timeout=int(a.timeout), task=m.group(1),
                                           ckpt=f"{job}/eval/run/nn/{ck.name}", n=len(spec["envs"]),
                                           cams=" --enable_cameras" if a.video else ""))
    _sh(["scp", "-q", str(out / "job.sh"), f"{SERVER}:{job}/job.sh"], 60)
    ssh(f"nohup bash {job}/job.sh > {job}/job.log 2>&1 < /dev/null &", 30)
    print(f"[sim_traj] server sim 시작 · GPU{a.gpu} · env {len(spec['envs'])} · {job}/job.log", flush=True)
    t0, seen = time.time(), 0
    while True:
        time.sleep(10)
        log = ssh(f"grep -a '\\[grid\\]\\|RC=\\|Traceback\\|Error' {job}/job.log 2>/dev/null; true", 60, check=False)
        lines = log.splitlines()
        for ln in lines[seen:]:
            print(f"  {ln[:160]}", flush=True)
        seen = len(lines)
        rc = re.search(r"RC=(\d+)", log)
        if rc:
            if rc.group(1) != "0":
                raise SimTrajError(f"sim rc={rc.group(1)} — {job}/job.log")
            break
        if time.time() - t0 > a.timeout + 120:
            raise SimTrajError(f"sim 시간 초과 — {job}/job.log")
    for f in ("trace.npz", "result.json"):
        _sh(["scp", "-q", f"{SERVER}:{job}/eval/{f}", str(out / f)], 300)
    return out


# ============================================================================ ROS
def _interrupt(sig, _frame):
    raise KeyboardInterrupt(signal.Signals(sig).name)


def _ros_node(name: str):
    import rclpy
    from rclpy.node import Node
    if not rclpy.ok():
        rclpy.init()
    return rclpy, Node(name)


def read_cup(topic: str, seconds: float = 1.5, min_frames: int = 5):
    """FP++ live 를 모아 (중앙값 xyz, 퍼짐 m, 프레임 수) — 새 영상(stamp)만 센다."""
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    rclpy, node = _ros_node("sim_traj_cup")
    got: dict = {}
    node.create_subscription(PoseStamped, topic, lambda m: got.setdefault(
        (m.header.stamp.sec, m.header.stamp.nanosec), (m.pose.position.x, m.pose.position.y, m.pose.position.z)),
        qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    if len(got) < min_frames:
        return None, None, len(got)
    p = np.array(list(got.values()))
    med = np.median(p, axis=0)
    return med, float(np.linalg.norm(p[:, :2] - med[:2], axis=1).max()), len(got)


# ============================================================================ 명령
def cmd_plan(a) -> int:
    pdir, c, ck, env = load_policy(a.policy)
    s = c.sides["arm"]
    target = sim_cup_name(env.get("object_names") or [], a.object)
    if a.cup:
        x, y = (float(v) for v in a.cup.split(","))
        z, spread, n = None, None, 0
    else:
        if not a.cup_topic:
            raise SimTrajError("--cup x,y 또는 --cup-topic(FP++ live) 이 필요하다")
        med, spread, n = read_cup(a.cup_topic)
        if med is None:
            raise SimTrajError(f"{a.cup_topic}: FP++ 새 프레임 {n} 개 — 컵이 보이는지 · live 추적이 도는지")
        if spread > 0.005:
            raise SimTrajError(f"FP++ 컵이 1.5 s 동안 {spread * 100:.1f} cm 흔들렸다 — 컵이 서 있는지 · 손이 가리는지")
        x, y, z = (float(v) for v in med)
    tag = time.strftime("%m%d_%H%M%S") + f"_{s.side}_{a.object}"
    out = Path(a.out) if a.out else Path(a.out_root or OUT) / tag     # 콘솔은 --out-root(카드 폴더) — 승인한 명령이 시각에 따라 안 바뀐다
    out.mkdir(parents=True, exist_ok=True)
    print(f"[sim_traj] {pdir.name} · 물체 {a.object}(sim {target}) · 컵 ({x:.3f}, {y:.3f})"
          + (f" z {z:.3f} · FP++ {n} 프레임 퍼짐 {spread * 1000:.1f} mm" if z is not None else " (--cup)"), flush=True)
    server_plan(a, pdir, c, ck, x, y, out)
    trace = np.load(out / "trace.npz")
    res = json.loads((out / "result.json").read_text())
    spec = json.loads((out / "spec.json").read_text())
    pick = choose(trace, res["rows"], spec, target)
    plan = plan_arrays(trace, pick, s)
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK
    fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    bad = check_plan(plan, s, fk, a.rate_scale)
    p = plan["palm"]
    lift = p[-1] - p[plan["close"]]
    meta = {"policy": str(pdir.relative_to(_paths.SIM2REAL / "deploy" / "policies")) if pdir.is_relative_to(
                _paths.SIM2REAL / "deploy" / "policies") else str(pdir),
            "checkpoint_md5": c.checkpoint_md5, "side": s.side, "object": a.object, "sim_object": target,
            "cup_xy": [x, y], "cup_z_fpp": z, "cup_topic": a.cup_topic, "sim_cup0": [float(v) for v in plan["cup"][0]],
            "env": pick["env"], "n_ok": pick["n_ok"], "n": pick["n"], "palm_close_std_cm": pick["palm_close_std_cm"],
            "server_job": f"{JOBS}/{out.name}", "hdgp": a.hdgp, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "problems": bad}
    np.savez(out / "plan.npz", **{k: v for k, v in plan.items()}, meta=json.dumps(meta, ensure_ascii=False))
    vid = None
    if a.video:
        r = pick["row"]
        name = ssh(f"ls {JOBS}/{out.name}/eval/video 2>/dev/null | grep -a '_t{r.get('trial', 0)}\\.mp4$' | "
                   f"grep -a '{target}' | head -1; true", 60, check=False).strip()
        if name:
            _sh(["scp", "-q", f"{SERVER}:{JOBS}/{out.name}/eval/video/{name}", str(out / name)], 120)
            vid = out / name
    dt = plan["dt"]
    print(f"[sim_traj] 성공 {pick['n_ok']}/{pick['n']} 시행 · 고른 env {pick['env']} · "
          f"닫기 시작 {plan['close'] * dt:.2f} s · 쥠 {plan['grasp'] * dt:.2f} s · 성공(끝) {plan['success'] * dt:.2f} s")
    print(f"           닫기 시작 손바닥 퍼짐(시행 간 표준편차) {pick['palm_close_std_cm']} cm · "
          f"닫기→끝 손바닥 이동 x {lift[0] * 100:+.1f} y {lift[1] * 100:+.1f} z {lift[2] * 100:+.1f} cm")
    if z is not None and abs(z - plan["cup"][0][2]) > 0.015:
        print(f"  ※ FP++ 컵 z {z:.3f} 가 sim 컵 z {plan['cup'][0][2]:.3f} 와 {abs(z - plan['cup'][0][2]) * 100:.1f} cm 다르다")
    print(f"  계획 {out / 'plan.npz'}" + (f" · 영상 {vid}" if vid else ""))
    if bad:
        print("  ★문제: " + " · ".join(bad))
        return 1
    print(f"  다음: python3 {HERE.relative_to(_paths.SIM2REAL)} run {out / 'plan.npz'}   (검사만 — 실기 동작은 --execute)")
    return 0


def load_plan(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    plan = {k: z[k] for k in z.files if k != "meta"}
    plan["dt"], plan["close"] = float(plan["dt"]), int(plan["close"])
    return plan, json.loads(str(z["meta"]))


def cmd_run(a) -> int:
    import replay_to_pd as RP
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from shadow_replay_core import PARK_SPEED_RAD_PER_SEC, approach_ramp

    plan, meta = load_plan(Path(a.plan))
    pdir, c, _ck, _env = load_policy(meta["policy"])
    s = c.sides["arm"]
    if list(plan["names"]) != list(s.arm_joints) + list(s.hand_joints):
        raise SimTrajError(f"계획 관절 {list(plan['names'])} ≠ 계약 {list(s.arm_joints) + list(s.hand_joints)}")
    fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    bad = check_plan(plan, s, fk, a.rate_scale)
    if bad:
        raise SimTrajError("계획 문제: " + " · ".join(bad))
    side = s.side
    rclpy, node = _ros_node("sim_traj_run")
    # ★10.19 콘솔 정지는 SIGTERM — 손대지 않으면 rclpy 가 context 를 닫아 episode stop 을 못 보내고(pd 는 워치독으로 간다) 기록도
    #   안 남는다(실측: SIGTERM 뒤 publish 실패). Ctrl-C 와 같게 KeyboardInterrupt 로 바꿔 stop() 이 돈다.
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    st: dict = {"q": {}, "pd": None, "pd_t": 0.0, "policy": None, "applied": None}
    node.create_subscription(JointState, RP.STATE_TOPIC, lambda m: st["q"].update(zip(m.name, m.position)),
                             qos_profile_sensor_data)

    def on_pd(m):
        try:
            st["pd"], st["pd_t"] = json.loads(m.data), time.time()
        except ValueError:
            pass
    node.create_subscription(String, RP.PD_STATUS.format(side=side), on_pd, 10)
    node.create_subscription(String, f"/policy_control/status/rh_aglt_node_{side}",
                             lambda m: st.update(policy=json.loads(m.data).get("phase")), 10)
    node.create_subscription(JointState, f"/policy_control/pd_{side}/applied",
                             lambda m: st.update(applied=dict(zip(m.name, m.position))), qos_profile_sensor_data)
    pub = node.create_publisher(JointState, RP.TOPIC, 10)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ep_pub = node.create_publisher(String, RP.EPISODE_TOPIC_SIDE.format(side=side), latched)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        rclpy.spin_once(node, timeout_sec=0.05)
    arm_names = [RP.measured_name(j) for j in s.arm_joints]
    arm_now = np.array([st["q"][n] for n in arm_names]) if all(n in st["q"] for n in arm_names) else None
    pd_arm = None if st["pd"] is None else (st["pd"].get("arms") or {}).get(side)
    if pd_arm is not None:
        pd_arm = {**pd_arm, "phase": st["pd"].get("phase") if not st["pd"].get("estop") else "ESTOP"}
    cup_now = None
    if not a.no_cup_check:
        topic = a.cup_topic or meta.get("cup_topic")
        if not topic:
            raise SimTrajError("컵 확인할 FP++ 토픽이 없다(--cup-topic, 또는 --no-cup-check)")
        cup_now, spread, n = read_cup(topic, 1.0, 3)
    why = run_refusals(plan, meta, arm_now, cup_now if not a.no_cup_check else meta["cup_xy"], pd_arm,
                       st["policy"] == "running", a.cup_tol)
    dt = plan["dt"]
    pub_dt = dt / a.rate_scale
    n = len(plan["q"])
    print(f"[sim_traj] {Path(a.plan).parent.name} · {meta['policy']} · {side} · {n} 프레임 × {pub_dt * 1000:.0f} ms "
          f"({a.rate_scale:g}×, {n * pub_dt:.1f} s + 정착 대기) · 닫기 시작 프레임 {plan['close']}")
    if cup_now is not None:
        print(f"  컵 지금 ({cup_now[0]:.3f}, {cup_now[1]:.3f}) · 계획 ({meta['cup_xy'][0]:.3f}, {meta['cup_xy'][1]:.3f}) · "
              f"{np.linalg.norm(np.asarray(cup_now[:2]) - meta['cup_xy']) * 100:.1f} cm")
    if arm_now is not None:
        print(f"  팔 ↔ 첫 프레임 {np.abs(arm_now - plan['q'][0, :7]).max():.3f} rad")
    if why:
        print("  거부: " + " · ".join(why))
        return 2
    if not a.execute:
        print("  검사 통과 — 실기 동작은 --execute(사용자 승인 뒤)")
        return 0

    hand_now = plan["q"][0, 7:]
    if st["applied"] and all(j in st["applied"] for j in s.hand_joints):
        hand_now = np.array([st["applied"][j] for j in s.hand_joints])
    start = np.concatenate([arm_now, hand_now])
    ramp = approach_ramp(start, plan["q"][0], speed=PARK_SPEED_RAD_PER_SEC, dt=pub_dt)
    ep_id = RP.replay_episode_id()
    names = list(plan["names"])
    log: dict = {"t": [], "tgt": [], "arm": [], "phase": []}

    def send(k, q, qd, phase):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = f"{ep_id}:{k}"
        msg.name, msg.position, msg.velocity, msg.effort = names, q.tolist(), qd.tolist(), [0.0] * len(q)
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        log["t"].append(time.time()), log["tgt"].append(q.copy()), log["phase"].append(phase)
        log["arm"].append([st["q"].get(nm, np.nan) for nm in arm_names])

    def stop(reason: str) -> None:
        body = json.dumps({"episode": ep_id, "event": "stop", "object_anchor": None, "home_q": {},
                           "reasons": [f"sim_traj: {reason}"], "t_ns": time.time_ns()})
        ep_pub.publish(String(data=body))
        t1 = time.time()
        while time.time() - t1 < RP.STOP_WAIT_S and (st["pd"] or {}).get("target") != "internal":
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(pub_dt)
        out = Path(a.plan).parent / f"run_{time.strftime('%H%M%S')}.npz"
        np.savez(out, t=np.array(log["t"]), tgt=np.array(log["tgt"]), arm=np.array(log["arm"], float),
                 phase=np.array(log["phase"]), names=np.array(names), reason=reason, rate_scale=a.rate_scale)
        print(f"[sim_traj] 끝 — {reason} · pd target {(st['pd'] or {}).get('target')} · 기록 {out}")

    def pd_bad() -> str | None:
        if time.time() - st["pd_t"] > 0.5:
            return "pd 상태 0.5 s 끊김"
        if (st["pd"] or {}).get("estop"):
            return "pd estop"
        if (st["pd"] or {}).get("phase") not in ("TRACKING", None):
            return f"pd phase {(st['pd'] or {}).get('phase')}"
        return None

    k = 0
    try:
        for q in ramp:
            send(k, q, np.zeros_like(q), "ramp")
            k += 1
            time.sleep(pub_dt)
        q_prev = plan["q"][0]
        for kind, i in schedule(plan):
            q = plan["q"][i]
            if kind == "settle":
                t_s = time.time()
                while True:
                    send(k, q, np.zeros_like(q), "settle")
                    k += 1
                    arm = np.array([st["q"].get(nm, np.nan) for nm in arm_names])
                    eq, ep = settle_error(fk, arm, q, q[7:])
                    if eq <= a.settle_q and ep <= a.settle_palm:
                        print(f"  정착 {time.time() - t_s:.2f} s · 관절 {eq:.3f} rad · 손바닥 {ep * 100:.1f} cm")
                        break
                    if time.time() - t_s > a.settle_timeout:
                        stop(f"닫기 전 정착 실패 {a.settle_timeout:g} s — 관절 {eq:.3f} rad · 손바닥 {ep * 100:.1f} cm")
                        return 3
                    if (why := pd_bad()) is not None:
                        stop(why)
                        return 3
                    time.sleep(pub_dt)
                q_prev = q
                continue
            qd = np.zeros_like(q)
            qd[:7] = (q[:7] - q_prev[:7]) / pub_dt
            send(k, q, qd, "frame")
            k += 1
            q_prev = q
            if (why := pd_bad()) is not None:
                stop(why)
                return 3
            time.sleep(pub_dt)
    except KeyboardInterrupt as exc:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)       # stop() 도중 두 번째 신호로 끊기지 않게
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop(f"중단 신호 {exc or 'SIGINT'}(콘솔 정지 · Ctrl-C)")
        return 130
    stop("궤적 끝(sim 목표 성공 프레임)")
    node.destroy_node()
    rclpy.shutdown()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="실기 컵 자리 sim 궤적을 뽑는다(server GPU)")
    p.add_argument("--policy", required=True, help="정책 id(rh56f1/aglt/left_neo03) 또는 폴더")
    p.add_argument("--object", required=True, help="배포 물체 이름(source240_real · source200_real · shaker_c_real)")
    p.add_argument("--cup", default="", help="x,y(m, 로봇 base) — 없으면 --cup-topic FP++ live 1.5 s 중앙값")
    p.add_argument("--cup-topic", default="", help="FP++ live 토픽(/objects/<물체>/live_pose)")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--hdgp", default=HDGP)
    p.add_argument("--timeout", type=float, default=1500)
    p.add_argument("--no-video", dest="video", action="store_false")
    p.add_argument("--rate-scale", type=float, default=0.5, help="계획 속도 검사용 재생 배속")
    p.add_argument("--out", default="")
    p.add_argument("--out-root", default="", help="계획을 <이 폴더>/<시각_팔_물체> 에 쓴다(--out 이 없을 때, 기본 logs/sim_traj)")
    r = sub.add_parser("run", help="plan.npz 를 pd 로 재생(★--execute 는 사용자 승인 뒤)")
    r.add_argument("plan")
    r.add_argument("--rate-scale", type=float, default=0.5)
    r.add_argument("--cup-topic", default="")
    r.add_argument("--cup-tol", type=float, default=CUP_TOL_M)
    r.add_argument("--no-cup-check", action="store_true")
    r.add_argument("--settle-q", type=float, default=SETTLE_Q_RAD)
    r.add_argument("--settle-palm", type=float, default=SETTLE_PALM_M)
    r.add_argument("--settle-timeout", type=float, default=SETTLE_TIMEOUT_S)
    r.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    if not 0.0 < a.rate_scale <= 1.0:
        raise SystemExit("--rate-scale 는 (0, 1]")
    try:
        return {"plan": cmd_plan, "run": cmd_run}[a.cmd](a)
    except SimTrajError as exc:
        print(f"[sim_traj] 거부: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
