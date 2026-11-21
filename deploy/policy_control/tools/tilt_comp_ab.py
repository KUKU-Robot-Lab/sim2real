#!/usr/bin/env python3
"""pd 보상 조건 반복 비교(자동 실험) — 양팔 차렷 → 홈 → tilt 시작 자세 → tilt 경로 재생 → 홈 → 차렷 을 조건을 바꿔 N 회.
★11.17 사용자: pd tilt_comp(왼팔 payload ff + j4 적분)를 켜고 끄며 추종 오차를 기록 · 측정(처음 A/B). ★11.17 일반화: 조건은 yaml 한 파일
(--conds, 기본 config/pd_comp_ab/tilt_comp.yaml = 예전 A/B 와 같은 결과). 조건 = pd yaml 덮어쓰기(점 경로) + 재생 인자(replay_to_pd).
경로 · rate · 게인은 조건끼리 같다. 손은 빈손(hand_load "empty") — 차렷 손 = 주먹(pd hand_path_pose). 판정은 손끝 FK 오차(분석기).

조건 yaml:
  order: AB                       # 사이클 순서(--order 가 덮는다) — 한 글자 이름이면 "ABAB", 아니면 쉼표 "base,grav"
  conditions:
    A: {}                         # 아무것도 안 바꿈
    B:
      pd: {gravity.payload.left: [0.30, 0.014, 0.008, 0.133]}   # 떠 있던 pd yaml 의 점 경로 → 값(없는 키 · 길이 다른 목록 거부)
      replay_args: [--tilt-comp]  # 재생(replay_to_pd)에 붙일 인자
      replay_on: all              # 그 인자를 붙일 재생 구간 — 기본 all = home · tilt_start · tilt · back · rest 전부
pd 덮어쓰기가 있는 조건은 사이클 앞에서 그 yaml 로 pd 를 다시 띄운다(팔마다 pd release → 정지 → 새 yaml 로 발행 모드 기동, 바뀔 때만).
끝까지 돌면 원래 yaml 로 되돌린다. 중간에 멈추면 pd 는 그 조건 yaml 로 붙든 채 남는다(메타 · 화면에 적는다).
다시 띄우다 끊겨 pd 가 없는 팔은 종료 때 원래 yaml 로 한 번 띄운다(IDLE). 다시 띄운 pd 는 콘솔 유닛이 아니라서(콘솔 /api/unit 은
실기 pd 유닛을 켜지 않고 pd_config 를 받지 않는다) 한 번이라도 바꿨으면 끝난 뒤 콘솔에서 pd_arm_<팔> 단계를 rewind 해야 한다
(기본 teardown 이 해 준다. --keep-bringup 이면 직접).
종료 코드: 0 끝 · 2 중단/payload 복원 실패 · 3 시작 전 거부 · 4 브링업 실패 · 5 pd 없는 팔이 남음.

빈손이라 payload ff 가 0.117 kg 이면 없는 하중만큼 팔을 민다 → 실험 시작 때 떠 있는 왼팔 pd 의 tilt_comp.left payload kg 를
--payload-kg(기본 0, 무게중심 그대로)로 바꾸고(토픽 pd_left/tilt_comp_payload, pd 재시작 없음 · pd 를 다시 띄우면 다시 보낸다), 끝 · 중단 ·
Ctrl-C · 예외 때 원래 값으로 되돌린다. 적용 · 복원은 pd status 의 tilt_comp_payload 로 확인하고 메타에 적는다. pd 가 TRACKING 이면 pd 가 거부한다.

--bringup(기본 켬): 실험 전에 콘솔 미션 단계(preflight · drivers · 손 · 손 점검 · pd 무발행 → 발행, 양팔)를 콘솔 HTTP API 로
돌린다(console_bringup.py). 이미 떠 있는 단계는 건너뛰고, ★모터 전원 · 손 전원 · 랜 같은 손 확인은 터미널에서 Enter 를 받는다.
teardown(기본, 11.17 사용자 "실험 종료하면 자동으로 bringup까지 종료"): payload 복원 뒤 두 팔 pd release(IDLE 확인) → 이 도구가
띄운 pd(프로세스 그룹 SIGINT) → 콘솔에 살아 있는 손 · pd 유닛 → drivers 를 끄고 pd_arm · pd_load · 손 · drivers 단계를 rewind 한다
(다음 실행이 처음부터 다시 오른다. --bringup 을 안 했어도 콘솔에 살아 있는 유닛은 끈다). ★drivers 는 끝까지 돌았고 두 팔이
IDLE 로 풀렸을 때만(차렷 자세, 사용자 허락) — 중단 · 예외 · Ctrl-C · release 실패면 팔을 놓은 쪽 pd 와 손까지만 내리고 drivers 는
남긴다(이유는 화면 · meta.json teardown). --keep-bringup 이면 아무것도 내리지 않는다.

사이클(기존 도구를 그대로 부른다 — 미션 home_* · return_* · tilt_now 와 같은 단계):
  engage(--skip-engaged) → 손 주먹(pd_hand_path) → 정착 → [팔마다 check_path_start · 저장 홈 경로 재생 · goto_home] → 정착
  → [오른팔 → 왼팔: tilt_now 계획 · 검사 → 이송 경로 재생(rate 0.5)] → 정지 2.5 s → tilt 재생(--traj) → 정착
  → 이송 경로 역재생(왼팔 → 오른팔) → goto_home → [팔마다 check_path_start --at end · 홈 경로 역재생] → 정착
끝나면(중단이어도) 분석기 tilt_comp_ab_report.py 를 돌려 report.md · report.json 을 만든다.
사이클마다 bag(rh56f1_record.sh, EXTRA = 팔별 applied · pd status)과 cycle_XX.json(조건 · 단계 시각 · 재생 발행 통계 · git · 경로 md5).
단계 실패 · pd HOLD · 정착 시간 초과면 그 자리에서 멈춘다(이후 사이클 없음, pd 는 붙든 채). Ctrl-C = 팔마다 stop 사건 → 기록 정리.
끝까지 성공하면 차렷에서 pd release(return_* 와 같음).

    ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/tilt_comp_ab.py                 # 계획만(브링업 · 조건 · 단계 · 예상 시간)
    ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/tilt_comp_ab.py --conds <yaml> --cycles 6 --execute   # 실행
분석: tilt_comp_ab_report.py <출력 폴더>.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import console_bringup as CB  # noqa: E402
from tilt_now import ARM_JOINTS, SIDES  # noqa: E402
from tilt_pour import RECORD, SIM2REAL, TOOLS, run_paths, write_meta  # noqa: E402

PY = sys.executable
CONTRACT = SIM2REAL / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
HOME_NPZ = {s: TOOLS.parent / "paths" / f"home_rh56f1_{s}.npz" for s in SIDES}
TRAJ = SIM2REAL / "logs/runs/261116/episode/150115_cocktail_1c_n07b/tilt_tilt_blue/tilt_cmd.npz"
PAYLOAD_TIMEOUT = 5.0     # payload 변경 · 복원 확인(status) 대기
CARRY_RATE = 0.5          # 150115 에피소드 to_tilt_start 와 같은 빠르기
START_GAP = 0.05          # 자산 policy.yaml start_gap_rad
PRE_HOLD_S = 2.5          # tilt 직전 정지 구간(분석은 앞 2 s)
STILL_RAD, STILL_S, STILL_TIMEOUT = 0.003, 0.5, 20.0   # 0.5 s 동안 관절별 변동 < 0.003 rad 면 정착
HOLD_GRACE_S = 1.0        # 워치독만인 HOLD 는 이만큼 풀리기를 기다린다(fake 부하에서 재생 시작 0.254 s 끊김 실측)
TRACK_TIMEOUT = 5.0       # 단계 뒤 두 팔 engage · ok 대기(상태 토픽 지연)
RECORD_TOPICS = [f"/policy_control/{k}" for s in SIDES for k in (f"pd_{s}/applied", f"status/pd_{s}")]
EPISODE_TOPIC_SIDE = "/policy_control/{side}/episode"
CONDS = TOOLS.parent / "config" / "pd_comp_ab" / "tilt_comp.yaml"
PD_LAUNCH = TOOLS.parent / "launch" / "pd_controller.launch.py"
PD_DEFAULT = {"real": TOOLS.parent / "config" / "pd_rh56f1_exec.yaml", "fake": TOOLS.parent / "config" / "pd_rh56f1_fake.yaml"}
SEGMENTS = ("home", "tilt_start", "tilt", "back", "rest", "pose")   # 궤적 재생 구간 — 조건 replay_args 를 붙일 곳(pose = --pose 의 창 · 이동)
#: 조건이 덮으면 안 되는 재생 인자(경로 · 방향 · 발행은 이 도구가 정한다)
REPLAY_ARGS_OWNED = {"--npz", "--joints", "--rate-scale", "--reverse", "--max-start-gap", "--execute"}
PD_UP_TIMEOUT, PD_STOP_TIMEOUT = 40.0, 20.0
TAIL_LINES = 40            # 단계마다 하위 명령 출력 끝 줄 수(cycle_XX.json steps[].extra.tail)
#: replay_to_pd 끝 줄 — "[replay] 발행 N 프레임 · 늦음(> X ms) n · 최대 지연 W ms · 간격 p95 P ms"
REPLAY_LINE = re.compile(r"\[replay\] 발행 (\d+) 프레임 · 늦음\(> (\d+) ms\) (\d+) · 최대 지연 ([\d.]+) ms · 간격 p95 ([\w.]+) ms")


@dataclass(frozen=True)
class Step:
    id: str
    argv: tuple = ()               # 빈 튜플 = 내부 단계(kind)
    kind: str = "cmd"              # cmd · still · hold · enter(argv[0] 안내 뒤 Enter 대기, --pose object)
    est_s: float = 0.0             # 움직임 예상 시간(계획 · 서비스 대기 제외)


@dataclass
class Result:
    id: str
    t0: float
    t1: float
    rc: int
    note: str = ""
    extra: dict = field(default_factory=dict)


def conditions(cycles: int | None, order: str, names=("A", "B")) -> list[str]:
    """사이클마다 조건 이름. order = 한 글자 이름 줄("ABAB") 또는 쉼표 목록("base,grav") 또는 이름 하나("base") — 대소문자 무시,
    사이클 수만큼 되풀이(None = 순서 한 바퀴)."""
    canon = {str(n).upper(): n for n in names}
    toks = [t.strip() for t in order.split(",")] if "," in order or order.strip().upper() in canon else list(order.strip())
    seq = [canon.get(t.upper()) for t in toks]
    cycles = len(seq) if cycles is None else cycles
    if cycles < 1 or not toks or None in seq:
        raise ValueError(f"--cycles ≥ 1 · --order 는 {' · '.join(map(str, names))} 로만(받음 {cycles}, {order!r})")
    return [seq[i % len(seq)] for i in range(cycles)]


def load_conds(path: Path) -> dict:
    """조건 yaml → {"path", "order", "conditions": {이름: {note, pd, replay_args, replay_on}}}. 잘못되면 ValueError(움직이기 전)."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = d.get("conditions")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: conditions 가 비었다")
    if len({str(n).upper() for n in raw}) != len(raw):
        raise ValueError(f"{path}: 대소문자만 다른 조건 이름이 있다 {list(raw)}")
    conds = {}
    for name, c in raw.items():
        c = c or {}
        bad = set(c) - {"note", "pd", "replay_args", "replay_on"}
        pd, args, on = c.get("pd") or {}, c.get("replay_args") or [], c.get("replay_on", "all")
        on = list(SEGMENTS) if on == "all" else on
        why = ("이름은 영문 · 숫자 · _ . - 만(폴더 이름이 된다)" if not re.fullmatch(r"[\w.-]+", str(name)) else
               f"모르는 키 {sorted(bad)}" if bad else
               "pd 는 {점 경로: 값}" if not isinstance(pd, dict) or not all(isinstance(k, str) for k in pd) else
               "replay_args 는 문자열 목록" if not isinstance(args, list) or not all(isinstance(x, str) for x in args) else
               f"replay_args 에 이 도구가 정하는 인자 {sorted(REPLAY_ARGS_OWNED & set(args))}" if REPLAY_ARGS_OWNED & set(args) else
               f"replay_on 은 all 또는 {list(SEGMENTS)} 의 목록" if not isinstance(on, list) or set(on) - set(SEGMENTS) else
               "tilt_comp.left.payload 는 --payload-kg 가 실험 내내 정한다(토픽)" if any(k.startswith("tilt_comp.left.payload") for k in pd)
               else "")
        if why:
            raise ValueError(f"{path}: 조건 {name}: {why}")
        conds[str(name)] = {"note": str(c.get("note", "")), "pd": dict(pd), "replay_args": list(args), "replay_on": list(on)}
    return {"path": str(path), "order": None if d.get("order") is None else str(d["order"]), "conditions": conds}


def apply_overrides(cfg: dict, ov: dict) -> dict:
    """pd yaml dict 사본에 점 경로 덮어쓰기. 없는 키 · 목록 길이가 다른 값은 KeyError — 오타가 조용히 새 키가 되지 않게."""
    out = copy.deepcopy(cfg)
    for key, val in ov.items():
        *head, last = key.split(".")
        node = out
        for k in head:
            node = node.get(k) if isinstance(node, dict) else None
        if not isinstance(node, dict) or last not in node:
            raise KeyError(f"pd config 에 {key} 가 없다")
        if isinstance(node[last], list):
            n = len(node[last])
            ok = lambda v: isinstance(v, list) and len(v) == n            # noqa: E731
            # 팔별 매핑 {left: [...], right: [...]} 도 받는다(gravity.scale 처럼 pd 가 팔별 값을 받는 키). pd 로더가 뒤에서 다시 검사
            sided = isinstance(val, dict) and val and set(val) <= {"left", "right"} and all(ok(v) for v in val.values())
            if not (ok(val) or sided):
                raise KeyError(f"{key}: 길이 {n} 목록 또는 팔별 {{left,right}} 목록이어야 한다(받음 {val!r})")
        node[last] = val
    return out


def cond_pd_yaml(base: Path, cond: str, ov: dict, out_dir: Path) -> Path:
    """조건의 pd yaml — 덮어쓰기가 없으면 원본 그대로, 있으면 out_dir/pd_<조건>_<원본 이름> 사본(pd 엄격 로더로 검사)."""
    if not ov:
        return base
    if str(TOOLS.parent) not in sys.path:
        sys.path.insert(0, str(TOOLS.parent))
    from policy_control.pd_law import load_pd_config
    cfg = apply_overrides(yaml.safe_load(base.read_text(encoding="utf-8")), ov)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pd_{cond}_{base.name}"
    path.write_text(f"# tilt_comp_ab 조건 {cond}: {base} + 덮어쓰기 {json.dumps(ov, ensure_ascii=False)}\n"
                    + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    load_pd_config(path)              # 키 · 형식이 pd 가 받는 것인지 — 움직이기 전에 PdConfigError
    return path


def parse_replay(lines) -> list[dict]:
    """재생 로그 줄 → [{n, late_thr_ms, late, max_lag_ms, p95_ms}] (재생 하나에 한 줄, tilt_now 처럼 둘이면 둘)."""
    out = []
    for line in lines:
        m = REPLAY_LINE.search(line)
        if m:
            out.append({"n": int(m[1]), "late_thr_ms": int(m[2]), "late": int(m[3]), "max_lag_ms": float(m[4]), "p95_ms": float(m[5])})
    return out


def npz_seconds(path: Path, rate: float) -> float:
    with np.load(path) as d:
        dt = float(d["meta_step_dt"]) if "meta_step_dt" in d else 0.02
        return len(d["arm_target"]) * dt / rate


def ctl(side: str, stage: str, *extra: str) -> tuple:
    return (PY, str(TOOLS / "episode_ctl.py"), "--side", side, "--only", stage, *extra, "--execute", "--approve", stage)


def replay(npz, joints: list[str], rate: float, *extra: str) -> tuple:
    return (PY, str(TOOLS / "replay_to_pd.py"), "--npz", str(npz), "--joints", ",".join(joints), "--rate-scale", repr(rate),
            *extra, "--execute")


def cycle_steps(cond: dict, cdir: Path, traj: Path, rate: float, fake: bool = False) -> list[Step]:
    """한 사이클 단계. cond = load_conds 의 조건 하나 — replay_args 를 replay_on 구간의 재생에 붙인다.
    이송 경로는 tilt_now(계획 · 검사만)가 만든 뒤에야 알므로 argv 에 <carry_팔> 자리 표시(실행 때 cdir 에서 찾는다).
    fake: fake 플랜트는 차렷(0)으로 정확히 돌아와 check_path_start 의 엔코더 0 가드에 걸린다 — 그 가드만 끈다."""
    zero = ("--allow-exact-zero",) if fake else ()
    on, args = set(cond.get("replay_on", SEGMENTS)), tuple(cond.get("replay_args", ()))
    x = lambda seg: args if seg in on else ()  # noqa: E731
    home_s = {s: npz_seconds(HOME_NPZ[s], 1.0) for s in SIDES}
    st = [Step(f"engage_{s}", ctl(s, "pd_engage", "--skip-engaged", "--hold-s", "2")) for s in SIDES]
    st += [Step(f"fist_{s}", ctl(s, "pd_hand_path", "--service-timeout", "15")) for s in SIDES]
    st.append(Step("still_rest", kind="still"))
    for s in SIDES:
        st += [Step(f"check_home_{s}", (PY, str(TOOLS / "check_path_start.py"), "--npz", str(HOME_NPZ[s]),
                                        "--contract", str(CONTRACT), "--tol", "0.15", *zero)),
               Step(f"to_home_{s}", replay(HOME_NPZ[s], ARM_JOINTS[s], 1.0, *x("home")), est_s=home_s[s]),
               Step(f"settle_home_{s}", ctl(s, "pd_goto_home", "--service-timeout", "45"))]
    st.append(Step("still_home", kind="still"))
    for s in SIDES:                      # tilt_now --execute 와 같은 두 걸음(오른팔 → 왼팔) — 재생에 조건 인자를 붙이려고 나눴다
        st += [Step(f"plan_tilt_start_{s}", (PY, str(TOOLS / "tilt_now.py"), "--sides", s, "--out-root", str(cdir),
                                             "--rate-scale", repr(CARRY_RATE))),
               Step(f"to_tilt_start_{s}", replay(f"<carry_{s}>", ARM_JOINTS[s], CARRY_RATE, "--max-start-gap", repr(START_GAP),
                                                 *x("tilt_start")), est_s=3.0)]
    st.append(Step("pre_tilt_hold", kind="hold"))
    st.append(Step("tilt", replay(traj, ARM_JOINTS["right"] + ARM_JOINTS["left"], rate, "--max-start-gap", repr(START_GAP),
                                  *x("tilt")), est_s=npz_seconds(traj, rate)))
    st.append(Step("still_after_tilt", kind="still"))
    for s in ("left", "right"):          # 갈 때 오른팔 → 왼팔로 계획했으니 되돌아올 때는 거꾸로
        st.append(Step(f"back_{s}", replay(f"<carry_{s}>", ARM_JOINTS[s], CARRY_RATE, "--reverse", "--max-start-gap",
                                           repr(START_GAP), *x("back")), est_s=3.0))
    st += [Step(f"settle_back_{s}", ctl(s, "pd_goto_home", "--service-timeout", "45")) for s in SIDES]
    for s in SIDES:
        st += [Step(f"check_rest_{s}", (PY, str(TOOLS / "check_path_start.py"), "--npz", str(HOME_NPZ[s]),
                                        "--contract", str(CONTRACT), "--at", "end", *zero)),
               Step(f"to_rest_{s}", replay(HOME_NPZ[s], ARM_JOINTS[s], 1.0, "--reverse", *x("rest")), est_s=home_s[s])]
    st.append(Step("still_end", kind="still"))
    return st


def resolve_carry(argv: tuple, cdir: Path) -> tuple:
    """<carry_팔> → cdir 에서 tilt_now 가 만든 그 팔의 가장 새 carry_cmd.npz. 없으면 FileNotFoundError."""
    out = []
    for a in argv:
        if a.startswith("<carry_") and a.endswith(">"):
            hits = sorted(cdir.glob(f"*_setting_*/carry_{a[7:-1]}/carry_cmd.npz"))
            if not hits:
                raise FileNotFoundError(f"{cdir}/*_setting_*/carry_{a[7:-1]}/carry_cmd.npz 없음")
            a = str(hits[-1])
        out.append(a)
    return tuple(out)


def run_cycle(steps: list[Step], do) -> list[Result]:
    """do(step) → Result. 첫 실패(rc ≠ 0)에서 멈춘다."""
    out = []
    for s in steps:
        r = do(s)
        out.append(r)
        if r.rc != 0:
            break
    return out


def pd_proc(side: str = "left") -> tuple[int, dict] | None:
    """이 호스트에 떠 있는 pd 노드(pd_node_<팔>, 양팔 한 노드면 pd_node) — (pid, 파라미터 dict). 프로세스 명령줄에서 읽는다.
    pd 노드는 파라미터 서비스를 끈다(lean_node_kwargs) → `ros2 param get` 이 늘 시간 초과라 쓸 수 없다."""
    found = {}
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    for cmd in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = cmd.read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        node = next((a.split(":=", 1)[1] for a in argv if a.startswith("__node:=")), None)
        if node not in (f"pd_node_{side}", "pd_node"):
            continue
        try:   # 다른 도메인의 pd(같은 PC 의 실기 · 다른 fake)는 남의 것 — 보지도 건드리지도 않는다
            if proc_env(int(cmd.parent.name)).get("ROS_DOMAIN_ID", "") != domain:
                continue
        except OSError:
            continue
        params: dict = {}
        if "--params-file" in argv:  # launch_ros 가 파라미터를 임시 yaml 로 넘긴 경우
            try:
                pf = yaml.safe_load(Path(argv[argv.index("--params-file") + 1]).read_text(encoding="utf-8")) or {}
                params = next((v.get("ros__parameters", {}) for v in pf.values() if isinstance(v, dict)), {})
            except (OSError, IndexError, yaml.YAMLError, AttributeError):
                params = {}                     # 못 읽어도 pd 는 떠 있는 것으로 센다(내려간 것으로 착각하지 않게)
        params.update(a.split(":=", 1) for a in argv if ":=" in a and not a.startswith("__"))
        found[node] = (int(cmd.parent.name), params)
    return found.get(f"pd_node_{side}") or found.get("pd_node")


def running_pd_config(side: str = "left") -> Path | None:
    """떠 있는 pd 노드의 pd_config(파라미터를 못 읽었으면 None)."""
    got = pd_proc(side)
    return None if got is None or not got[1].get("pd_config") else Path(got[1]["pd_config"])


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def git_head() -> dict:
    g = lambda *a: subprocess.run(["git", "-C", str(SIM2REAL), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"git": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--short", "--untracked-files=no"))}


class Watch:
    """구독만 — 팔 실측(정착 판정)과 pd 상태(HOLD 감시). 백그라운드 스레드에서 돈다. stop 사건 발행자도 여기."""

    def __init__(self) -> None:
        import rclpy
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        self.String = String
        from rclpy.signals import SignalHandlerOptions
        # rclpy 기본 핸들러는 SIGINT · SIGTERM 에 컨텍스트를 먼저 내린다 — 그러면 finally 의 stop 사건 · payload 복원이
        # RCLError 로 죽는다(11.17 fake 재시작 중 SIGTERM 실측). 신호는 이 도구가 받아 KeyboardInterrupt 로 정리한다.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node("tilt_comp_ab")
        self.q: deque = deque(maxlen=4000)
        self.status: dict = {s: None for s in SIDES}
        self.hold_since: dict = {s: None for s in SIDES}
        self.blips: list[str] = []         # 스스로 풀린 워치독 HOLD(기록만)
        names = [f"openarm_{s}_joint{i}" for s in SIDES for i in range(1, 8)]

        def on_js(m):
            d = dict(zip(m.name, m.position))
            if all(n in d for n in names):
                self.q.append((time.time(), np.array([d[n] for n in names])))
        self.node.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)
        for s in SIDES:
            self.node.create_subscription(String, f"/policy_control/status/pd_{s}",
                                          lambda m, s=s: self._on_status(s, m.data), 10)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ep_pub = {s: self.node.create_publisher(String, EPISODE_TOPIC_SIDE.format(side=s), latched) for s in SIDES}
        from std_msgs.msg import Float64
        self.Float64 = Float64
        self.payload_pub = {s: self.node.create_publisher(Float64, f"/policy_control/pd_{s}/tilt_comp_payload", 10)
                            for s in SIDES}
        self._spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin.start()

    def _on_status(self, side: str, text: str) -> None:
        st = _loads(text)
        held = bool(st and st.get("phase") == "HOLD")
        if held and self.hold_since[side] is None:
            self.hold_since[side] = time.time()
        elif not held and self.hold_since[side] is not None:
            self.blips.append(f"{side} HOLD {time.time() - self.hold_since[side]:.2f} s 뒤 풀림")
            self.hold_since[side] = None
        self.status[side] = st

    def hold_sides(self) -> list[str]:
        """멈출 HOLD 인 팔. 워치독(목표 끊김)만이 사유인 HOLD 는 목표가 다시 오면 pd 가 스스로 푼다
        (pd_state._target_fresh) — HOLD_GRACE_S 안에 풀리면 넘기고 기록만 한다. 다른 사유는 바로."""
        now, out = time.time(), []
        for s, st in self.status.items():
            if not (st and st.get("phase") == "HOLD"):
                continue
            kinds = {str(r).split(":", 1)[0].strip() for r in st.get("reasons", [])}
            since = self.hold_since[s] or now
            if kinds != {"watchdog"} or now - since > HOLD_GRACE_S:
                out.append(s)
        return out

    def tracking(self) -> list[str]:
        """engage 상태(RAMPING · TRACKING) · ok 가 아닌 팔. 목표를 아직 못 받은 팔은 RAMPING 에 머문다(정상 —
        왼팔은 to_home_left 첫 목표가 올 때까지 RAMPING, chain.py ramp_done 은 목표 도달 때만)."""
        return [s for s, st in self.status.items()
                if not (st and st.get("phase") in ("RAMPING", "TRACKING") and st.get("ok", False))]

    def wait_tracking(self, timeout: float = TRACK_TIMEOUT) -> list[str]:
        """두 팔이 engage · ok 가 될 때까지 기다린다(상태 토픽 지연 대비). HOLD 면 바로 돌려준다."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.hold_sides():
                return self.hold_sides()
            if not self.tracking():
                return []
            time.sleep(0.1)
        return self.tracking()

    def still(self) -> bool:
        now = time.time()
        w = [q for t, q in list(self.q) if t >= now - STILL_S]
        return len(w) > 10 and self.q[0][0] <= now - STILL_S and float(np.ptp(np.array(w), axis=0).max()) < STILL_RAD

    def wait_still(self, timeout: float = STILL_TIMEOUT) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.hold_sides():
                return False
            if self.still():
                return True
            time.sleep(0.05)
        return False

    def payload(self, side: str = "left") -> list | None:
        """pd status 가 말하는 지금 tilt_comp payload (kg, x, y, z). pd 에 tilt_comp 블록이 없거나 status 가 없으면 None."""
        st = self.status.get(side) or {}
        return ((st.get("arms") or {}).get(side) or {}).get("tilt_comp_payload")

    def set_payload(self, kg: float, side: str = "left", timeout: float = PAYLOAD_TIMEOUT) -> str | None:
        """pd 에 payload kg 를 보내고 status 로 확인한다. None = 적용됨, 아니면 이유(pd 거부는 last_error)."""
        t0 = time.time()
        while self.payload_pub[side].get_subscription_count() == 0 and time.time() - t0 < timeout:
            time.sleep(0.05)
        while time.time() - t0 < timeout:
            cur = self.payload(side)
            if cur is not None and abs(cur[0] - kg) < 1e-9:
                return None
            self.payload_pub[side].publish(self.Float64(data=float(kg)))
            time.sleep(0.3)
        err = str((self.status.get(side) or {}).get("last_error") or "")
        return f"{timeout:g} s 안에 status payload 가 {kg:g} kg 가 아니다(지금 {self.payload(side)})" + (
            f" — pd: {err}" if "tilt_comp payload" in err else "")

    def graph(self) -> dict:
        """console_bringup.already_up 용 — 발행자가 있는 토픽 · 팔별 pd status. 발견을 1.5 s 기다린다."""
        time.sleep(1.5)
        topics = {n for n, _ in self.node.get_topic_names_and_types() if self.node.count_publishers(n) > 0}
        return {"topics": topics, "pd": dict(self.status)}

    def stop_event(self, why: str) -> None:
        ep = 900_000_000 + (time.time_ns() // 1_000_000) % 99_000_000
        msg = self.String(data=json.dumps({"episode": ep, "event": "stop", "object_anchor": None, "home_q": {},
                                           "reasons": [why], "t_ns": time.time_ns()}))
        for p in self.ep_pub.values():
            p.publish(msg)
        time.sleep(0.5)

    def close(self) -> None:
        import rclpy
        rclpy.try_shutdown()            # spin 스레드를 먼저 끝낸다 — 도는 중 destroy_node 면 종료 때 abort(core)
        self._spin.join(timeout=2.0)
        self.node.destroy_node()


def _loads(text: str) -> dict | None:
    try:
        return json.loads(text)
    except ValueError:
        return None


def exec_step(step: Step, w: Watch, engaged: bool) -> Result:
    """한 단계 실행 — 도는 동안 pd HOLD 면 자식을 끊고 실패. 끝나면 (engage 뒤에는) 두 팔 engage(RAMPING · TRACKING) · ok 를 요구한다."""
    t0 = time.time()
    if step.kind == "still":
        ok = w.wait_still()
        return Result(step.id, t0, time.time(), 0 if ok else 1, "" if ok else f"정착 못 함({STILL_TIMEOUT:g} s · HOLD {w.hold_sides()})")
    if step.kind == "hold":
        ok = w.wait_still()
        time.sleep(PRE_HOLD_S)
        ok = ok and w.still() and not w.hold_sides()
        return Result(step.id, t0, time.time(), 0 if ok else 1, "" if ok else "tilt 직전 정지 구간에 움직임 · HOLD")
    if step.kind == "enter":
        try:
            input(f"[tilt_comp_ab] ★{step.argv[0]} Enter (Ctrl-C = 중단) ")
        except EOFError:
            return Result(step.id, t0, time.time(), 1, "입력 없음(EOF)")
        bad = w.hold_sides()
        return Result(step.id, t0, time.time(), 1 if bad else 0, f"기다리는 동안 HOLD {bad}" if bad else "")
    print("$", " ".join(step.argv), flush=True)
    n_blip = len(w.blips)
    p = subprocess.Popen(step.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines: list[str] = []          # 화면에 그대로 내면서 재생 끝 줄(발행 · 늦음 · 지연)을 모은다

    def pump() -> None:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
    pumper = threading.Thread(target=pump, daemon=True)
    pumper.start()
    try:
        r = _wait_child(step, w, engaged, p, t0)
    except BaseException:               # Ctrl-C · SIGTERM(→ KeyboardInterrupt) · 예외 — 재생 자식을 남기지 않는다
        _end_child(p)
        raise
    pumper.join(timeout=2.0)
    rp = parse_replay(lines)
    r.extra = {"replay": rp} if rp else {}
    r.extra["tail"] = [x.rstrip("\n") for x in lines[-TAIL_LINES:]]   # 실패 원인을 cycle_XX.json 에 남긴다(11.18 012707 은 화면에만 있었다)
    warn = ""
    if pumper.is_alive():
        warn = "★자식 출력 끝을 다 못 읽었다(재생 통계가 빠졌을 수 있다)"
    elif r.rc == 0 and step.argv[1:2] and step.argv[1].endswith("replay_to_pd.py") and not rp:
        warn = "★재생 끝 줄([replay] 발행 …)이 없다 — 늦음 · 지연 통계 없음"
    if warn:
        print(f"[tilt_comp_ab] {step.id} {warn}", flush=True)
    r.note = "; ".join(x for x in (r.note, "; ".join(w.blips[n_blip:]), warn) if x)
    return r


def _end_child(p, grace: float = 5.0) -> None:
    """SIGINT → grace 초 기다림 → 안 끝나면 kill. 이미 끝났으면 아무것도 안 한다."""
    if p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    except ProcessLookupError:
        pass


def _wait_child(step: Step, w: Watch, engaged: bool, p: subprocess.Popen, t0: float) -> Result:
    """자식이 끝날 때까지 — 도는 동안 pd HOLD 면 끊고 실패. 끝나면 (engage 뒤에는) 두 팔 engage · ok 를 요구한다."""
    while p.poll() is None:
        held = w.hold_sides() if engaged else []
        if held:
            why = {s: (w.status[s] or {}).get("reasons") for s in held}
            _end_child(p, grace=10.0)
            return Result(step.id, t0, time.time(), 1, f"pd HOLD {why} — 단계를 끊었다")
        time.sleep(0.05)
    if p.returncode != 0:
        return Result(step.id, t0, time.time(), p.returncode, "명령 실패")
    if step.id.startswith("engage_"):
        return Result(step.id, t0, time.time(), 0)
    bad = w.wait_tracking()
    return Result(step.id, t0, time.time(), 1 if bad else 0, f"pd 가 engage · ok 가 아님 {bad}" if bad else "")


class PdSwitchError(RuntimeError):
    pass


def proc_env(pid: int) -> dict:
    """그 프로세스의 환경 — 콘솔이 띄운 pd 와 같은 ROS · DDS 환경으로 다시 띄우려고."""
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return dict(kv.decode(errors="replace").split("=", 1) for kv in raw if b"=" in kv)


def launch_args(pid: int) -> dict:
    """pd 노드를 띄운 `ros2 launch` 의 `키:=값` 인자(부모 프로세스 명령줄) — fake · use_source · params_file 은 노드 인자에 없다."""
    ppid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    argv = [a.decode(errors="replace") for a in Path(f"/proc/{ppid}/cmdline").read_bytes().split(b"\0") if a]
    if not any(a.endswith("pd_controller.launch.py") for a in argv):
        raise PdSwitchError(f"pd(PID {pid})의 부모가 pd_controller.launch.py 가 아니다 — launch 인자를 모른다")
    return dict(a.split(":=", 1) for a in argv if ":=" in a)


REWIND = ("콘솔 rewind 필요 — 이 도구가 띄운 pd 는 콘솔 유닛이 아니다(콘솔 /api/unit 은 실기 pd 유닛을 켜지 않고 "
          "argv 도 미션 yaml 에서만 온다). 먼저 이 도구가 띄운 pd 를 내린다(pd release 로 IDLE 확인 → `kill -INT -<PGID>`, "
          "남겨 두면 rewind 가 같은 이름 pd 를 하나 더 띄운다). 그다음 콘솔 화면에서 pd_arm_<팔> 단계로 되돌려(rewind) "
          "다시 실행해야 콘솔 pd 로 돌아간다")


def _pgid(pid: int):
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _same(a, b) -> bool:
    return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()


class _Adopted:
    """지난 실행이 setsid 로 띄운 pd 프로세스 그룹 — Popen 처럼 pid(=PGID) · poll() 만 쓴다."""

    def __init__(self, pgid: int) -> None:
        self.pid = pgid

    def poll(self):
        try:
            os.killpg(self.pid, 0)
            return None
        except ProcessLookupError:
            return 0


def adopt_ours(side: str, pid: int, running_cfg: Path):
    """pd(pid) 의 PGID 가 지난 tilt_comp_ab 실행 meta 의 ours_pgid[side] 에 있으면 (_Adopted, 원래 yaml), 아니면 None.
    원래 yaml: meta 의 pd_switch.orig[side], 없으면 조건 사본 이름 pd_<조건>_<원본> 에서 원본(config/)을 되찾는다."""
    pgid = _pgid(pid)
    if pgid is None:
        return None
    for m in sorted((SIM2REAL / "logs/runs").glob("*/tilt_comp_ab/*/meta.json"), reverse=True):
        try:
            pdst = json.loads(m.read_text(encoding="utf-8")).get("pd_switch") or {}
        except (OSError, ValueError):
            continue
        if (pdst.get("ours_pgid") or {}).get(side) != pgid:
            continue
        orig = (pdst.get("orig") or {}).get(side)
        if orig is None:
            name = running_cfg.name
            if running_cfg.parent.name != "pd_yaml" or not name.startswith("pd_") or name.count("_") < 2:
                return _Adopted(pgid), running_cfg
            orig = SIM2REAL / "deploy/policy_control/config" / name.split("_", 2)[2]
        orig = Path(orig)
        return (_Adopted(pgid), orig) if orig.is_file() else None
    return None


class PdSwitch:
    """조건마다 pd yaml 바꾸기 — 팔마다 pd release → 정지 → 그 yaml 로 발행 모드 다시 기동(원래 pd 의 launch 인자 · 환경 그대로).
    정지: 이 도구가 띄운 pd 는 그 프로세스 그룹에 SIGINT, 처음 pd 는 콘솔 유닛(pd_arm_<팔>#…)을 콘솔 API 로 끈다.
    처음 pd 가 콘솔 유닛이 아니면(손으로 띄운 pd · 지난 실행이 남긴 pd) 만들 때 거부한다 — 남의 프로세스는 내리지 않는다.
    다시 띄우기는 콘솔을 거치지 않는다(`REWIND` 의 이유) — 한 번이라도 띄웠으면 끝난 뒤 콘솔 rewind 가 필요하다."""

    def __init__(self, spec: dict, out: Path, w, api=None) -> None:
        self.w, self.out, self.api = w, out, api or CB.Api()
        self.orig, self.launch, self.ours, self.fake = {}, {}, {}, {}
        self.touched = False                   # 콘솔 pd 를 한 번이라도 내렸나 → 콘솔 rewind 필요
        prof = (self._state().get("session") or {}).get("profile") or {}
        if str(prof.get("domain")) != os.environ.get("ROS_DOMAIN_ID", ""):
            raise PdSwitchError(f"콘솔({getattr(self.api, 'url', '?')}) 프로파일 {prof.get('id')} 도메인 {prof.get('domain')} ≠ 이 셸 ROS_DOMAIN_ID "
                                f"{os.environ.get('ROS_DOMAIN_ID')} — 다른 콘솔의 pd 유닛을 끌 수 있어 거부(--console-url)")
        for s in SIDES:
            got = pd_proc(s)
            if got is None or str(got[1].get("execute")).lower() != "true" or got[1].get("sides") != s \
                    or not got[1].get("pd_config"):
                raise PdSwitchError(f"{s} 팔 하나짜리 발행 모드 pd(pd_node_{s})가 없다 — 다시 띄울 수 없다({got})")
            la = launch_args(got[0])
            if str(la.get("use_source", "false")).lower() == "true" or la.get("params_file"):
                raise PdSwitchError(f"{s} pd 가 use_source · params_file 로 떠 있다({la}) — 같은 모양으로 다시 띄우지 못한다")
            orig = Path(got[1]["pd_config"])
            if not self._console_units(s):
                adopted = adopt_ours(s, got[0], orig)     # ★11.18 사용자: 지난 실행이 띄운 pd 를 이어받아 연달아 실험
                if adopted is None:
                    raise PdSwitchError(f"{s} pd(PID {got[0]} · PGID {_pgid(got[0])})가 콘솔 유닛도, 지난 tilt_comp_ab 실행이 띄운 pd 도 아니다 "
                                        f"— 이 도구가 내릴 수 없다. 지난 실행이 다시 띄운 pd 면: {REWIND}")
                self.ours[s], orig = adopted
                self.touched = True
                print(f"[tilt_comp_ab] {s} pd 이어받음 — 지난 실행이 띄운 PGID {self.ours[s].pid}, 원래 yaml {orig}", flush=True)
            self.orig[s] = orig
            self.launch[s] = (got[1], proc_env(got[0]))
            self.fake[s] = str(la.get("fake", "false")).lower() == "true"
        self.want = {c: {s: cond_pd_yaml(self.orig[s], c, d["pd"], out / "pd_yaml") for s in SIDES}
                     for c, d in spec["conditions"].items()}

    def _state(self) -> dict:
        try:
            return self.api.get("/api/state")
        except (OSError, CB.BringupError) as exc:
            raise PdSwitchError(f"콘솔 API 응답 없음({exc}) — pd 를 건드리지 않는다") from exc

    def _console_units(self, s: str) -> list[str]:
        units = (self._state().get("session") or {}).get("units") or {}
        return [k for k, v in units.items() if k.split("#")[0] == f"pd_arm_{s}" and v.get("alive")]

    def ensure(self, cond: str) -> list[str]:
        """그 조건의 yaml 로 — 다시 띄운 팔 목록."""
        return self._to(self.want[cond])

    def restore(self) -> list[str]:
        return self._to(self.orig)

    def _to(self, cfg: dict) -> list[str]:
        need = [s for s in SIDES if not _same(running_pd_config(s), cfg[s])]
        if need:
            self._free_all()
        for s in need:
            self._restart(s, cfg[s])
        return need

    def _free_all(self) -> None:
        release_both(self.w)

    def missing(self) -> list[str]:
        return [s for s in SIDES if pd_proc(s) is None]

    def left_as(self) -> dict:
        """원래 yaml 이 아닌 채 떠 있는 팔 · pd 가 없는 팔 → 설명(중단 때 알림용)."""
        out = {}
        for s in SIDES:
            if pd_proc(s) is None:
                out[s] = "pd 없음 — 콘솔에서 다시 띄울 것"
            elif not _same(c := running_pd_config(s), self.orig[s]):
                out[s] = str(c)
        return out

    def revive(self) -> dict:
        """다시 띄우다 끊겨 pd 가 없는 팔에 원래 yaml 로 한 번 띄운다(IDLE 로 뜬다 — 팔은 JTC 가 잡는다). 팔 → 결과 문구."""
        res = {}
        for s in self.missing():
            try:
                self._start(s, self.orig[s])
                res[s] = "원래 yaml 로 다시 띄움"
            except (PdSwitchError, OSError) as exc:
                res[s] = f"★다시 띄우기 실패: {exc}"
            print(f"[tilt_comp_ab] {s} pd 없음 → {res[s]}", flush=True)
        return res

    def _restart(self, s: str, cfg: Path) -> None:
        print(f"[tilt_comp_ab] {s} pd 다시 띄움 — {cfg}", flush=True)
        phase = (self.w.status[s] or {}).get("phase")
        if phase == "HOLD":
            raise PdSwitchError(f"{s} pd HOLD — 다시 띄우지 않는다")
        try:
            if phase != "IDLE" and subprocess.run(ctl(s, "pd_release")).returncode != 0:
                raise PdSwitchError(f"{s} pd release 실패")
            self._stop(s)
            self._start(s, cfg)
        except (OSError, CB.BringupError) as exc:        # ProcessLookupError 는 OSError
            raise PdSwitchError(f"{s} pd 다시 띄우기 중 오류: {exc!r}") from exc

    def _stop(self, s: str) -> None:
        p = self.ours.get(s)
        if p is not None:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass                                       # 그룹이 이미 없다 — 아래에서 pd 가 내려갔는지 본다
        else:
            keys = self._console_units(s)
            if not keys:
                raise PdSwitchError(f"{s} pd 콘솔 유닛이 없다 — 내리지 않는다")
            self.touched = True
            self.api.lease()
            try:
                for k in keys:
                    self._unit_off(k)
            finally:
                self.api.release()
        t0 = time.time()
        while pd_proc(s) is not None:
            if time.time() - t0 > PD_STOP_TIMEOUT:
                raise PdSwitchError(f"{s} pd 가 {PD_STOP_TIMEOUT:g} s 안에 내려가지 않았다 — 더 손대지 않는다")
            time.sleep(0.2)
        self.ours.pop(s, None)                             # 내려간 것을 본 뒤에만 잊는다(teardown 이 다시 시도할 수 있게)
        self.w.status[s] = None

    def _unit_off(self, key: str) -> None:
        CB.unit_off(self.api, key, PD_STOP_TIMEOUT)

    def _start(self, s: str, cfg: Path) -> None:
        params, env = self.launch[s]
        argv = ["ros2", "launch", str(PD_LAUNCH), f"contract:={params['contract']}", f"robot:={params['robot']}",
                f"pd_config:={cfg}", f"sides:={s}", "execute:=true", f"stage:={params.get('stage', 'full')}",
                *(["fake:=true"] if self.fake[s] else [])]
        log = self.out / "pd_logs" / f"pd_{s}_{time.strftime('%H%M%S')}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print("$", " ".join(argv), f"> {log}", flush=True)
        with open(log, "w", encoding="utf-8") as fh:   # setsid: Ctrl-C 가 pd 를 같이 죽이지 않게(콘솔이 띄운 pd 와 같다)
            p = self.ours[s] = subprocess.Popen(argv, env=env, cwd=SIM2REAL, stdout=fh, stderr=subprocess.STDOUT,
                                                start_new_session=True)
        t0 = time.time()
        try:
            while time.time() - t0 < PD_UP_TIMEOUT:
                if p.poll() is not None:
                    raise PdSwitchError(f"{s} pd 기동 실패 rc {p.returncode} · {log}")
                if _same(running_pd_config(s), cfg) and (self.w.status[s] or {}).get("phase") == "IDLE":
                    return
                time.sleep(0.2)
        except (KeyboardInterrupt, SystemExit):
            # 뜨는 중에 끊김 — 반쯤 뜬 launch 를 남기면 revive 가 같은 팔 pd 를 하나 더 띄운다. 내리고 '없음' 으로 둔다
            self._kill_launch(s, p)
            raise
        raise PdSwitchError(f"{s} pd 가 {PD_UP_TIMEOUT:g} s 안에 IDLE 로 뜨지 않았다(PID {p.pid}) · {log}")

    def _kill_launch(self, s: str, p) -> None:
        """이 도구가 띄운 launch 그룹에 SIGINT → 10 s → SIGTERM. SIGKILL 은 쓰지 않는다(사람 확인 몫)."""
        for sig, wait_s in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            t0 = time.time()
            while p.poll() is None and time.time() - t0 < wait_s:
                time.sleep(0.1)
            if p.poll() is not None:
                break
        if p.poll() is not None:
            self.ours.pop(s, None)
        print(f"[tilt_comp_ab] {s} 뜨던 pd launch(PID {p.pid}) 내림 — {'끝남' if p.poll() is not None else '★안 끝남'}", flush=True)

    def stop_ours(self) -> list[str]:
        """teardown: 이 도구가 띄운 pd 중 IDLE 인 것만 내린다. 돌려주는 것 = 못 내린 팔."""
        bad = []
        for s in [s for s in self.ours if (self.w.status[s] or {}).get("phase") == "IDLE"]:
            try:
                self._stop(s)
                print(f"[teardown] ▼ {s} pd(이 도구가 띄운 것)", flush=True)
            except (PdSwitchError, OSError) as exc:
                print(f"[teardown] ★{s} pd 내리기 실패: {exc}", flush=True)
                bad.append(s)
        return bad


def release_both(w) -> None:
    """두 팔 다 release → IDLE 확인. 콘솔은 한 팔이라도 잡고 있으면 어느 pd 유닛도 끄지 않는다(가장 무거운 phase,
    11.17 fake 실측: 오른팔만 release 하면 'pd 가 TRACKING 다'). HOLD 인 팔이 있으면 아무것도 건드리지 않는다."""
    hold = [s for s in SIDES if (w.status[s] or {}).get("phase") == "HOLD"]
    if hold:
        raise PdSwitchError(f"{'·'.join(hold)} pd HOLD — 손대지 않는다")
    for s in SIDES:
        if (w.status[s] or {}).get("phase") != "IDLE" and subprocess.run(ctl(s, "pd_release")).returncode != 0:
            raise PdSwitchError(f"{s} pd release 실패")
    t0 = time.time()
    while busy := [s for s in SIDES if (w.status[s] or {}).get("phase") != "IDLE"]:
        if time.time() - t0 > PD_STOP_TIMEOUT:
            raise PdSwitchError(f"{busy} pd 가 {PD_STOP_TIMEOUT:g} s 안에 IDLE 이 아니다")
        time.sleep(0.2)


def teardown(a, w, sw, finished: bool) -> dict:
    """11.17 사용자 "실험 종료하면 자동으로 bringup까지 종료" — 두 팔 pd release(IDLE 확인) → 이 도구가 띄운 pd → 콘솔 손 · pd 유닛
    → drivers → 단계 rewind(console_bringup.teardown). drivers 는 끝까지 돌았고(finished) 두 팔이 IDLE 로 풀렸을 때만 내린다.
    중단이면 release 하지 않는다(붙든 채 멈추는 설계 그대로) — 이미 팔을 놓은 쪽 pd 와 손만 내리고 drivers 는 남긴다."""
    keep = None if finished else "실험이 끝까지 가지 않았다(중단 · 예외 · Ctrl-C) — pd 가 팔을 붙든 채일 수 있다"
    if keep is None and (none := [s for s in SIDES if w.status.get(s) is None]):
        keep = f"pd 상태가 없는 팔 {none}"
    if keep is None:
        try:
            release_both(w)
        except (PdSwitchError, OSError) as exc:
            keep = f"pd release 실패: {exc}"
    if sw is not None and (bad := sw.stop_ours()) and keep is None:
        keep = f"이 도구가 띄운 pd 를 못 내렸다 {bad}"
    free = [s for s in SIDES if (w.status.get(s) or {}).get("phase") in (None, "IDLE")]   # None = 이 도구가 내렸거나 pd 없음
    stages = ["hand_left", "hand_right", *(f"{p}_{s}" for s in free for p in ("pd_arm", "pd_load"))]
    return CB.teardown(CB.Api(getattr(a, "console_url", CB.URL)), stages, keep_drivers=keep,
                       retry_s=PD_STOP_TIMEOUT if keep is None else 0.0)


def record(cmd: str, name: str, out_root: Path | None = None) -> int:
    env = {**os.environ, "EXTRA": " ".join(RECORD_TOPICS)}
    if out_root is not None:
        env["RH56F1_RECORD_OUT"] = str(out_root)
    return subprocess.run(["bash", str(RECORD), cmd, name], env=env).returncode


def pd_switches(conds: list[str], spec: dict) -> int:
    """사이클 순서에서 pd yaml 이 바뀌는 횟수(처음 = 원래 yaml, 끝에 원래로 되돌리기 포함)."""
    keys = ["{}"] + [json.dumps(spec["conditions"][c]["pd"], sort_keys=True) for c in conds] + ["{}"]
    return sum(a != b for a, b in zip(keys, keys[1:]))


POSE: dict | None = None    # --pose 면 {"spec", "grip"} — 사이클 단계를 자세 유지 측정으로 바꾼다(pose_hold.pose_steps)


def steps_of(cond: dict, cdir: Path, traj: Path, rate: float, fake: bool = False) -> list[Step]:
    if POSE is None:
        return cycle_steps(cond, cdir, traj, rate, fake)
    import pose_hold
    return pose_hold.pose_steps(sys.modules[__name__], POSE["spec"], POSE["grip"], cond, cdir, traj, rate, fake)


def print_plan(conds: list[str], out: Path, traj: Path, rate: float, payload_kg: float, spec: dict) -> None:
    steps = steps_of(spec["conditions"][conds[0]], out / f"cycle_01_{conds[0]}", traj, rate)
    motion = sum(s.est_s for s in steps)
    print(f"[tilt_comp_ab] 계획만 · 사이클 {len(conds)} ({' '.join(conds)}) · tilt {traj} · rate {rate:g} · 빈손 · payload {payload_kg:g} kg")
    print(f"  출력 {out}/  (cycle_XX_<조건>/ bag · cycle_XX.json · report)")
    print(f"  실험 시작 때 왼팔 pd tilt_comp.left payload → {payload_kg:g} kg(무게중심 그대로, pd 재시작 없음) · 끝 · 중단 때 원래 값으로")
    print(f"  조건 {spec['path']}:")
    for name in dict.fromkeys(conds):
        c = spec["conditions"][name]
        args = " ".join(c["replay_args"])
        print(f"   {name}: pd 덮어쓰기 {json.dumps(c['pd'], ensure_ascii=False) if c['pd'] else '없음'} · 재생 인자 "
              + (f"{args} @ {','.join(c['replay_on'])}" if args else "없음") + (f" — {c['note']}" if c["note"] else ""))
    n_sw = pd_switches(conds, spec)
    if n_sw:
        print(f"  pd 다시 띄우기 {n_sw} 번(조건 yaml 이 바뀌는 사이클 앞 · 끝에 원래로 — 팔마다 release → 정지 → 기동, 번당 ~30 s 짐작)")
    print(f"  한 사이클 단계(조건 {conds[0]} — 다른 조건은 재생 인자 · pd yaml 만 다르다):")
    for s in steps:
        what = f"Enter 대기: {s.argv[0]}" if s.kind == "enter" else " ".join(s.argv[1:]) if s.argv else {"still": f"정착 대기(0.5 s 변동 < {STILL_RAD} rad)",
                                                     "hold": f"정착 뒤 {PRE_HOLD_S} s 정지(tilt 직전 구간)"}[s.kind]
        print(f"   {s.id:21s} {f'~{s.est_s:.0f} s' if s.est_s else '':7s} {what.replace(str(SIM2REAL) + '/', '')}")
    print(f"  예상 움직임 시간 사이클당 ~{motion:.0f} s × {len(conds)} (계획 · 서비스 · 정착 대기 별도, 사이클당 +30~60 s 짐작)")
    print("[tilt_comp_ab] 계획만 — --execute 로 실행")


def dry_bringup(profile: str, url: str = CB.URL) -> None:
    """계획만 — 콘솔이 떠 있으면 그 상태(끝난 단계)로, ROS 도메인이 있으면 그래프로 건너뛰기를 판단한다(둘 다 읽기만)."""
    stages = CB.plan(CB.profile_mission(profile))
    api, done, note = CB.Api(url), {}, f"꺼짐 — 실행 때 프로파일 {profile} 로 띄운다(setsid, 실험 뒤에도 남는다)"
    if api.up():
        st = api.get("/api/state")
        sess = st.get("session")
        note = (f"떠 있음 · run 없음 — 실행 때 {profile} 를 연다" if sess is None else
                f"떠 있음 · {sess['profile']['id']} run {sess.get('run_id')}"
                + ("" if sess["profile"]["id"] == profile else " ★다른 프로파일 — 실행하면 멈춘다(--profile)"))
        done = {k: r.get("done") for k, r in CB.mission_rows(st).items()}
    graph = None
    if os.environ.get("ROS_DOMAIN_ID", "") not in ("", "0"):
        w = Watch()
        try:
            graph = w.graph()
        finally:
            w.close()
    CB.print_plan(stages, done, graph, note)


def check_pd_overrides(spec: dict, profile_kind: str) -> None:
    """계획만 — 떠 있는 pd yaml(없으면 프로파일 기본 exec yaml)에 덮어쓰기를 얹어 pd 엄격 로더로 검사(임시 폴더, 출력 폴더는 안 만든다)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for s in SIDES:
            base = running_pd_config(s) or PD_DEFAULT[profile_kind]
            for name, c in spec["conditions"].items():
                cond_pd_yaml(base, name, c["pd"], Path(tmp))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--conds", type=Path, default=CONDS, help="조건 yaml(모듈 설명 참고) — 기본은 예전 tilt_comp A/B")
    ap.add_argument("--cycles", type=int, default=None, help="기본 6(tilt) · --pose 면 순서 한 바퀴")
    ap.add_argument("--order", default=None, help="사이클 순서(조건 이름, 예 ABAB · base,grav) — 없으면 yaml order, 그것도 없으면 이름 순")
    ap.add_argument("--rate-scale", type=float, default=1.0, help="tilt 재생 빠르기(조건끼리 같음)")
    ap.add_argument("--traj", type=Path, default=TRAJ, help="tilt_cmd.npz(14 관절)")
    ap.add_argument("--payload-kg", type=float, default=0.0,
                    help="실험 동안 왼팔 pd tilt_comp payload kg(빈손 = 0, 적분만). 끝 · 중단 때 원래 값으로")
    ap.add_argument("--bringup", action=argparse.BooleanOptionalAction, default=True,
                    help="콘솔 미션으로 드라이버 · 손 · pd(양팔 발행 모드)까지 준비 — --no-bringup 이면 떠 있다고 본다")
    ap.add_argument("--console-url", default=os.environ.get("S2R_CONSOLE_URL", CB.URL),
                    help=f"콘솔 주소(기본 {CB.URL}, 환경변수 S2R_CONSOLE_URL). 기본이 아니면 그 콘솔이 먼저 떠 있어야 한다 — "
                         "이 도구가 콘솔을 띄우면 늘 기본 포트로 뜬다(console_bringup.start_console)")
    ap.add_argument("--profile", default=None, help=f"콘솔 프로파일(기본 {CB.PROFILE_REAL}, --fake 면 {CB.PROFILE_FAKE})")
    ap.add_argument("--keep-bringup", action="store_true",
                    help="끝난 뒤 아무것도 내리지 않는다. 기본은 pd · 손 · drivers 를 내리고 콘솔 단계를 rewind"
                         "(drivers 는 끝까지 돌고 두 팔이 IDLE 일 때만, 아니면 남김)")
    ap.add_argument("--out", type=Path, default=None, help="없으면 logs/runs/<YYMMDD>/tilt_comp_ab/<HHMMSS>")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--fake", action="store_true", help="fake 플랜트 전용 — check_path_start 에 --allow-exact-zero")
    ap.add_argument("--pose", type=Path, default=None,
                    help="자세 유지 측정(중력 · 마찰 동정) yaml — 사이클 단계를 tilt 비교 대신 이것으로(tools/pose_hold.py). "
                         "--conds 를 안 주면 이 yaml 의 conditions(기본 하나 = 떠 있던 pd yaml)")
    ap.add_argument("--grip", choices=("open", "closed", "object", "held"), default="open",
                    help="--pose 손: open 편 손 · closed 빈손 쥠 · object 물체 넣고(Enter) 쥠 · held 이미 쥔 채 홈에서(손 지령 없음, "
                         "브링업 · teardown · 끝 release 없음 — 놓기는 사람이) — closed · object · held 는 grip_poses 만")
    a = ap.parse_args(argv)
    if a.grip == "held" and a.pose is None:
        print("✗ --grip held 는 --pose 와 함께만")
        return 1
    global POSE
    try:
        if a.pose is not None:
            import pose_hold
            POSE = {"spec": pose_hold.load_pose(a.pose.resolve()), "grip": a.grip}
            if a.grip == "held":   # 브링업은 손 점검 단계를 다시 돌릴 수 있고 teardown 은 손 유닛을 끈다 — 쥔 물체가 떨어진다
                a.bringup, a.keep_bringup = False, True
            a.conds = a.pose.resolve() if a.conds == CONDS else a.conds
        a.spec = load_conds(a.conds)
        names = list(a.spec["conditions"])
        order = a.order or a.spec["order"] or ("".join(names) if all(len(n) == 1 for n in names) else ",".join(names))
        conds = conditions(a.cycles if a.cycles or POSE is not None else 6, order, names)
        a.cycles = len(conds)
        if not a.execute:
            check_pd_overrides(a.spec, "fake" if a.fake else "real")
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"✗ 조건 · 순서: {exc}")
        return 1
    if not 0.1 <= a.rate_scale <= 1.0 or not 0.0 <= a.payload_kg <= TC_PAYLOAD_KG_MAX:
        print(f"✗ --rate-scale 0.1~1.0 · --payload-kg 0~{TC_PAYLOAD_KG_MAX}")
        return 1
    traj = a.traj.resolve()
    profile = a.profile or (CB.PROFILE_FAKE if a.fake else CB.PROFILE_REAL)
    out = Path(a.out or run_paths.run_dir("tilt_comp_ab", "", mkdir=False)).resolve()
    if not a.execute:
        if a.bringup:
            dry_bringup(profile, a.console_url)
        print_plan(conds, out, traj, a.rate_scale, a.payload_kg, a.spec)
        return 0
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if domain in ("", "0"):
        print("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")
        return 1
    signal.signal(signal.SIGTERM, _sigterm)            # kill 에도 finally(payload 복원 · 기록 정리)를 밟는다
    out.mkdir(parents=True, exist_ok=True)
    if POSE is not None:                               # 창 궤적 · 충돌 검사 — 브링업 전, 하나라도 실패면 시작 전 거부
        prep = [PY, str(TOOLS / "pose_hold.py"), "--pose", POSE["spec"]["path"], "--grip", a.grip, "--out", str(out / "pose"),
                "--traj", str(traj), *(("--fake",) if a.fake else ())]
        print("$", " ".join(prep), flush=True)
        if subprocess.run(prep).returncode != 0:
            print("✗ 자세 준비 · 충돌 검사 실패 — 시작 전 거부", flush=True)
            return 3
    w, rec = Watch(), None
    try:
        if a.bringup:
            api = CB.Api(a.console_url)
            if a.console_url.rstrip("/") != CB.URL and not api.up():
                print(f"✗ 콘솔 {a.console_url} 이 꺼져 있다 — 이 도구가 띄우면 {CB.URL} 로 뜬다. 그 주소로 콘솔을 먼저 띄울 것", flush=True)
                return 4
            rec = CB.bringup(api, profile, w.graph, int(domain), log=out / "console.log")
            (out / "bringup.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            if not rec["ok"]:
                print(f"✗ 브링업 실패 — 실험에 들어가지 않는다\n  {rec['error']}", flush=True)
                return 4
        return _experiment(a, conds, traj, out, w, rec)
    except KeyboardInterrupt:
        print(f"[tilt_comp_ab] ★중단(Ctrl-C) · {out}", flush=True)
        return 2
    finally:
        w.close()


TC_PAYLOAD_KG_MAX = 0.5    # policy_control.tilt_comp.PAYLOAD_KG_MAX 와 같게(pd 도 이 범위 밖이면 거부한다)


def _sigterm(signum, frame):  # noqa: ANN001, ARG001
    raise KeyboardInterrupt


def _experiment(a, conds: list[str], traj: Path, out: Path, w: Watch, bring: dict | None) -> int:
    running = running_pd_config("left")
    t0 = time.time()
    while w.payload("left") is None and time.time() - t0 < PAYLOAD_TIMEOUT:   # 첫 status 를 기다린다
        time.sleep(0.1)
    before = w.payload("left")
    if running is None or before is None:
        print(f"✗ 왼팔 pd 가 없거나(pd_config {running}) status 에 tilt_comp_payload 가 없다({before}) — "
              "pd yaml 에 tilt_comp.left 블록 · 11.17 이후 pd 코드로 떠 있어야 한다. 움직이지 않았다.")
        return 3
    spec = getattr(a, "spec", None)
    if POSE is not None and POSE["grip"] == "held":   # goto_home 이 손을 계약 홈(편 손)으로 보내면 쥔 물체가 떨어진다
        bad = [f"{s} {running_pd_config(s)}" for s in SIDES
               if (yaml.safe_load(Path(running_pd_config(s) or "/dev/null").read_text(encoding="utf-8")) or {}).get("home_hand") != "keep"]
        bad += [f"조건 {c}" for c, d in ((spec or {}).get("conditions") or {}).items() if "home_hand" in d["pd"]]
        if bad:
            print(f"✗ --grip held: pd yaml home_hand 가 keep 이 아니거나 조건이 덮는다 {bad} — 움직이지 않았다")
            return 3
    a.pd_switch = None
    if spec and any(c["pd"] for c in spec["conditions"].values()):
        try:
            a.pd_switch = PdSwitch(spec, out, w, api=CB.Api(getattr(a, "console_url", CB.URL)))
        except (PdSwitchError, OSError, ValueError, KeyError) as exc:     # PdConfigError 는 ValueError
            print(f"✗ 조건 pd yaml 준비 실패 — 움직이지 않았다: {exc}")
            return 3
    pay = {"before": before, "target_kg": a.payload_kg}
    err = w.set_payload(a.payload_kg)
    if err:
        print(f"✗ payload {a.payload_kg:g} kg 적용 실패 — 움직이지 않았다: {err}")
        return 3
    pay["applied"] = w.payload("left")
    print(f"[tilt_comp_ab] 왼팔 pd tilt_comp payload {before} → {pay['applied']}", flush=True)
    (out / "pd_config_left.yaml").write_bytes(running.read_bytes())
    base = {**git_head(), "rate_scale": a.rate_scale, "traj": str(traj), "traj_md5": md5(traj), "payload_kg": a.payload_kg,
            "hand_load": "empty", "pd_config_left": str(running), "pd_config_left_md5": md5(running)}
    run_meta = dict(tool="tilt_comp_ab", cycles=a.cycles, order=conds, bringup=bring, conds=spec,
                    **{k: v for k, v in base.items() if k not in ("git", "dirty")})
    write_meta(out, **run_meta, payload=pay)
    stopped, sw, pdst, td = "예외", a.pd_switch, {}, None     # _cycles 가 돌려주지 못하면(예외) 끝까지 간 것이 아니다
    try:
        stopped = _cycles(a, conds, traj, out, w, base)
    finally:
        if sw is not None:                   # 다시 띄우다 끊긴 팔 — 원래 yaml 로 한 번 되살린다
            pdst = {"revived": sw.revive(), "left_as": sw.left_as(), "missing": sw.missing(),
                    "rewind_needed": sw.touched, "rewind": REWIND if sw.touched else ""}
        if sw is not None:
            pdst["ours_pgid"] = {k: p.pid for k, p in sw.ours.items() if p.poll() is None}   # setsid 라 pid = PGID
            pdst["orig"] = {k: str(v) for k, v in sw.orig.items()}   # 다음 실행이 이어받을 때 되돌릴 yaml
        if sw is not None and "left" in pdst["missing"]:
            err = "왼팔 pd 없음 — payload 복원 건너뜀"
        elif (w.status.get("left") or {}).get("phase") in ("RAMPING", "TRACKING") \
                and (w.payload("left") or [None])[0] != before[0]:     # 실패 중단 · held 끝은 붙든 채 — pd 가 거부한다(같으면 보낼 것 없음)
            err = f"왼팔 pd 가 붙든 채(TRACKING) — pd release 뒤 payload {before[0]:g} kg 로 되돌릴 것"
        else:
            err = w.set_payload(before[0], timeout=2 * PAYLOAD_TIMEOUT)   # 끝 · 중단 · Ctrl-C · 예외 모두 — stop 뒤 HOLD 면 pd 가 받는다
        pay.update(after=w.payload("left"), restored=err is None, restore_error=err)
        write_meta(out, **run_meta, payload=pay, pd_switch=pdst)
        print(f"[tilt_comp_ab] payload 복원 {'✓ ' + str(pay['after']) if err is None else '★실패: ' + err}", flush=True)
        td = {"skipped": "--keep-bringup"} if a.keep_bringup else teardown(a, w, sw, finished=stopped is None)   # payload 복원 뒤
        write_meta(out, **run_meta, payload=pay, pd_switch=pdst, teardown=td)
    if pdst.get("rewind_needed") and not {f"pd_arm_{s}" for s in SIDES} <= set(td.get("rewound", ())):
        print(f"[tilt_comp_ab] ★{REWIND}" + (f" · 이 도구가 띄운 pd PGID {pdst['ours_pgid']}" if pdst.get("ours_pgid") else ""),
              flush=True)
    if stopped is not None and pdst.get("revived"):      # _cycles 의 '남음' 문구는 되살리기 전 — 끝 상태를 덧붙인다
        stopped += f" → 끝에 되살림 {pdst['revived']} · 남은 조건 yaml {pdst.get('left_as') or '없음'}"
    print(f"[tilt_comp_ab] {'끝 — ' + str(len(conds)) + ' 사이클' if stopped is None else '★중단: ' + stopped} · {out}\n"
          f"  분석: python3 {TOOLS / 'tilt_comp_ab_report.py'} {out}", flush=True)
    if any(out.glob("cycle_*.json")):      # 보고까지 한 번에(중단이어도 끝난 회차로) — 실패해도 실험 결과(bag)는 그대로
        subprocess.run([PY, str(TOOLS / "tilt_comp_ab_report.py"), str(out)])
    if pdst.get("missing"):
        print(f"[tilt_comp_ab] ★pd 없는 팔 {pdst['missing']} — 콘솔에서 다시 띄울 것", flush=True)
        return 5
    return 0 if stopped is None and pay["restored"] else 2


def _cycles(a, conds: list[str], traj: Path, out: Path, w: Watch, base: dict) -> str | None:
    """사이클 전부. 돌려주는 것 = 멈춘 이유(None = 끝까지)."""
    stopped, sw = None, getattr(a, "pd_switch", None)
    try:
        for i, cond in enumerate(conds, 1):
            cs = a.spec["conditions"][cond]
            if sw is not None:                 # 사이클 밖(bag 앞) — 조건 yaml 로 pd 다시 띄우기, 띄웠으면 payload 다시
                try:
                    if "left" in sw.ensure(cond) and (err := w.set_payload(a.payload_kg)):
                        raise PdSwitchError(f"payload {a.payload_kg:g} kg 다시 적용 실패: {err}")
                except PdSwitchError as exc:
                    stopped = f"사이클 {i}({cond}) 앞 pd 다시 띄우기 실패: {exc}"
                    break
            cdir = out / f"cycle_{i:02d}_{cond}"
            cdir.mkdir()
            meta = {"cycle": i, "condition": cond, "cond": cs,
                    "tilt_comp": "--tilt-comp" in cs["replay_args"] and "tilt" in cs["replay_on"], **base,
                    "start": time.time(), "steps": [], "pd_config": {s: str(running_pd_config(s)) for s in SIDES},
                    "pd_payload": w.payload("left")}      # 이 회차 시작 때 pd 가 말한 실제 값
            name = f"tilt_comp_ab_c{i:02d}"
            if record("start", name, cdir) != 0:
                stopped = f"사이클 {i} bag 기록 시작 실패 — 기록 없이 진행하지 않는다"
                break
            meta["bag"] = str(next(iter(sorted(cdir.glob(f"*_{name}"))), ""))
            try:
                ok = run_one(cs, cdir, traj, a.rate_scale, w, meta, a.fake)
            except KeyboardInterrupt:
                w.stop_event("tilt_comp_ab Ctrl-C")          # 기록을 닫기 전에 — stop 사건이 bag 에 남는다
                raise
            finally:
                meta["end"] = time.time()
                record("stop", name)
                (out / f"cycle_{i:02d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            if not ok:
                stopped = f"사이클 {i}({cond}) {meta['steps'][-1]['id']} 실패: {meta['steps'][-1]['note']}"
                break
            print(f"[tilt_comp_ab] 사이클 {i}/{len(conds)} ({cond}) 끝", flush=True)
    except KeyboardInterrupt:
        stopped = "Ctrl-C"
        w.stop_event("tilt_comp_ab Ctrl-C")                  # 사이클 밖(기록 시작 중 등)에서 눌렀을 때도
    held = POSE is not None and POSE["grip"] == "held"
    if stopped is None and not held:         # held 는 홈에서 쥔 채 붙든다 — release 하면 처진 뒤 JTC 가 잡는다(10.09)
        for s in SIDES:                      # 차렷에서 JTC 로(return_* 와 같음)
            subprocess.run(ctl(s, "pd_release"))
        if sw is not None:
            try:
                sw.restore()
            except PdSwitchError as exc:
                stopped = f"원래 pd yaml 로 되돌리기 실패: {exc}"
    if sw is not None and (left := sw.left_as()) and held and stopped is None:
        print(f"[tilt_comp_ab] --grip held: pd 를 조건 yaml {left} 로 홈에서 붙든 채 둔다(끝 release · 되돌리기 없음) — "
              f"놓은 뒤 {REWIND}", flush=True)
    elif sw is not None and left:
        print(f"[tilt_comp_ab] ★pd 가 원래 yaml 이 아니다 {left} — 되돌리지 않았다(붙든 채 멈춤)", flush=True)
        stopped = (stopped or "") + f" · pd 조건 yaml 로 남음 {left}"
    return stopped


def run_one(cond: dict, cdir: Path, traj: Path, rate: float, w: Watch, meta: dict, fake: bool = False) -> bool:
    """한 사이클. 재생 argv 의 <carry_팔> 은 그 단계 직전에 tilt_now 가 만든 이송 경로로 바꾼다."""
    engaged = {"v": w.tracking() == []}

    def do(step: Step) -> Result:
        try:
            step = replace(step, argv=resolve_carry(step.argv, cdir))
        except FileNotFoundError as exc:
            r = Result(step.id, time.time(), time.time(), 1, str(exc))
        else:
            if step.id.startswith("to_tilt_start_"):
                meta.setdefault("carry", {})[step.id[len("to_tilt_start_"):]] = step.argv[step.argv.index("--npz") + 1]
            r = exec_step(step, w, engaged["v"])
        engaged["v"] = engaged["v"] or (step.id == "engage_left" and r.rc == 0)
        meta["steps"].append(r.__dict__)
        rp = " · ".join(f"발행 {x['n']} 늦음 {x['late']}{'★' if x['late'] else ''} 최대 지연 {x['max_lag_ms']:g} ms"
                        for x in r.extra.get("replay", []))
        print(f"[tilt_comp_ab] {step.id} rc {r.rc} {r.t1 - r.t0:.1f} s {r.note} {rp}".rstrip(), flush=True)
        return r

    if POSE is not None:
        meta["pose"] = str(cdir.parent / "pose" / "windows.json")
    return run_cycle(steps_of(cond, cdir, traj, rate, fake), do)[-1].rc == 0


if __name__ == "__main__":
    raise SystemExit(main())
