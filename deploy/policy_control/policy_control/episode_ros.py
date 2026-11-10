"""에피소드 실행기의 ROS 쪽 — 정책 노드 · pd · 인지에 이미 있는 길만 쓴다(새로 관절 명령을 내지 않는다).

  정책   /policy_control/<팔>/episode/{reset,start,stop} (rh_aglt_node · rh_place_node, ns = 팔) · 이벤트 /policy_control/<팔>/episode
         aglt 는 reset 뒤 /policy_control/<팔>/goal 에 SETTING 을 주고(노드 stop_on_target 이 도달 때 끝낸다), place 는 그대로.
  궤적   rehome = plan_rehome(실측 → RRT, 로봇 세계) → check_path_start → replay_to_pd --execute → pd goto_home 정착 → 손끝 영점(11.04).
         물체를 든 손이 있는 팔은 거부한다(계획기가 든 컵을 모른다).
         carry · tilt(10.17) = plan_<kind>.py(실측 → 계획 · 검사, 움직이지 않음) → replay_to_pd(그 팔 관절) — 손 상태(cups)가 다르면 거부.
  손     hand open = pd hand_release(빈손에서만) · hand grasp = tools/hand_grasp.py(팔 제자리, 닿을 때까지 닫고 쥔 판정) — 10.17 경로 B.
  배치   snapshot = 물체 토픽(FP++)을 창(1.5 s) 동안 모아 정지 · 신선이면 중앙값을 기록 → /episode/objects/<이름>/pose 로
         30 Hz 다시 낸다(집기 전 정지 컵 — 10.04 사용자 "처음에 FP++ 로 각 컵 배치를 한 번에 기록"). 집힌 물체는 그만 낸다.
  홀더   고정(10.04 사용자) — holder_poses 파일(cup_holder_pose_node --write)을 /objects/cup_holder_<id>/pose 로 latched 발행.
  안전   pd 상태(HOLD · estop)를 보고 있다가 정책 중이면 바로 끝낸다. safe_stop = 떠 있는 정책 노드 episode/stop(pd 가 붙든다) ·
         돌던 재생 프로세스 SIGTERM.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from policy_control import _paths
import run_paths  # noqa: E402 — _paths 가 scripts/ 를 경로에 넣은 뒤
from policy_control.episode_failure import NodeResult
from policy_control.episode_spec import EMPTY, PLANNED_TRAJ, TRAJ_ASSETS, Episode, Node, episode_sides, hand_contract_file
from policy_control.rh_place import HOLDER_FLOOR_Z, HOLDER_TOL_M

NS = "/policy_control"
OBJECT_RELAY = "/episode/objects/{}/pose"
#: ★10.19 팔마다 띄운 실행기(episode_runner_<팔>)는 자기 이름 아래로 다시 낸다 — 두 팔 에피소드가 같은 물체 이름(CUP)을 써도
#  /episode/objects/CUP/pose 에 두 실행기가 섞여 내지 않게. 기본 실행기(두 팔 tilt 등)는 그대로.
DEFAULT_RUNNER = "episode_runner"


def relay_topic(runner: str, name: str) -> str:
    """snapshot 재발행 토픽 — 기본 실행기는 /episode/objects/<이름>/pose, 그 밖은 /episode/<실행기 꼬리>/objects/<이름>/pose."""
    if runner in ("", DEFAULT_RUNNER):
        return OBJECT_RELAY.format(name)
    tail = runner[len(DEFAULT_RUNNER) + 1:] if runner.startswith(DEFAULT_RUNNER + "_") else runner
    return f"/episode/{tail}/objects/{name}/pose"


def live_topic(topic: str) -> str | None:
    """FP++ 물체 snapshot 토픽 …/pose → 같은 물체의 live 추적 토픽 …/live_pose(object_pose_node 규약). 아니면 None."""
    return topic[: -len("/pose")] + "/live_pose" if topic.endswith("/pose") else None


SNAPSHOT_WINDOW_S = 1.5
SNAPSHOT_STILL_M = 0.005          # 창 안 위치 퍼짐 — 넘으면 컵이 움직이는 중
PLACE_JUDGE_WAIT_S = 1.0          # ★11.02 놓기 판정은 정책 stop 뒤 이 시간 이후 수신한 FP++ 프레임만 — 지연 0.4~0.5 s + 창 0.5 s 가
                                  #   놓기 전(손에 든) 영상으로 PLACE_FAILED 를 내던 것(sim_e2e 재현 dz 65~73 mm). sim 은 nodes.py 0.7 s
SNAPSHOT_MIN_FRAMES = 5
LIFT_RISE_MIN_M = 0.07            # ★10.19 1-B lift 판정 — 손바닥 10 cm 중 컵이 7 cm 넘게(결정 10 의 7 cm 위와 같은 선)
#: ★11.08 들기 가드(guard · hstop)로 멈춘 잡기의 판정(사용자 승인 제안안) — 고정 구간 FP++ live 컵 상승(snapshot 정지 컵 z 기준)이
#  GUARD_HOLD_RISE_M 이상인 채로 GUARD_HOLD_S(60 틱 @60 Hz) 이어져야 한다. 프레임 사이 GUARD_HOLD_GAP_S 넘게 끊기면 처음부터.
#  기다림은 GUARD_HOLD_WAIT_S 까지(FP++ live 재등록 끊김 2.8~3.5 s 를 한 번 넘길 만큼). lift_hold 판정은 그대로다.
GUARD_HOLD_RISE_M = 0.05
GUARD_HOLD_S = 1.0
#  ★11.08 리뷰: FP++ live 는 약 11 Hz 로 오지만 같은 값을 두 번씩 보내 값은 약 5.6 Hz(0.18~0.25 s)로 바뀐다(bag 113604 · 151437,
#  logs/analysis/lift_sweep_1108/live_rate.txt). 같은 값이 반복되는 프레임은 세지 않고, 끊김은 그 두 배인 0.5 s 로 본다.
GUARD_HOLD_GAP_S = 0.5
GUARD_HOLD_WAIT_S = 6.0
#: 손 안 판정 — live 컵 ↔ 손바닥 FK(노드 status lift_guard.palm) 거리 상한. 같은 bag 들린 뒤 실측 0.064~0.084 m
#  (노드 lift_grasp_palm_m 0.08 은 오프셋 보정 컵 기준이라 원값 live 에는 좁다) + 여유 → 0.12 m. palm 은 0.5 s 안 status 만
GUARD_HOLD_PALM_M = 0.12
GUARD_HOLD_PALM_AGE_S = 0.5
#: 이보다 큰 상승(m)은 FP++ 튐으로 버린다(들기 가드 hstop 0.08 · lift_hold 0.07 의 두 배 넘게)
GUARD_HOLD_RISE_MAX_M = 0.20
#: reset 직후 status 로 가드가 켜졌는지 확인 — 기다림 · 자리 허용(plan x · y 와)
LG_CHECK_WAIT_S = 1.0
LG_XY_TOL_M = 0.001
#: 가드 판정 실패 문구 — episode_failure 가 lift_guard_failed(언제나 safe stop)로 나눈다
GUARD_FAIL_TEXT = "lift_guard hold failed"
GUARD_STOPS = ("guard:", "hstop:")
POLICY_MARGIN_S = 10.0            # 정책 에피소드 시간 + 이만큼 안에 이벤트가 안 오면 stop → timeout
#: ★10.14 실기 183230: FP++ live 추적이 품질 검사(MASK_TOO_SMALL) 3번 실패마다 다시 등록하며 2.8~3.5 s 끊긴다(5~6 s 마다).
#  정책 reset 은 0.5 s 안에 받은 live 컵만 받아서, 그 사이에 부르면 거부되고 에피소드가 안전 정지했다.
#  reset 은 로봇을 움직이지 않는다 — 이 거부만 이 시간 안에서 다시 부른다(다른 거부는 그대로 판정으로 간다).
RESET_CUP_STALE_TEXT = "FP++ live 컵 자세가"
RESET_CUP_RETRY_S = 6.0
RESET_CUP_RETRY_STEP_S = 0.2
#: ★10.28 사람이 놓은 물체(snapshot objects) — FP++ 묶음 컨테이너가 그 물체를 지금 카메라로 다시 찍게 하고(다시 찍는 동안
#  옛 좌표를 내지 않는다) 그 뒤에 온 프레임만 기록한다. 컨테이너는 못 찾은 물체를 3 s 마다 스스로 찾으므로, 사람이 놓는
#  도중(손이 컵을 가린 채) 찍힌 자리가 고정돼 있을 수 있다. 미션 cups 단계와 같은 도구 · 물어본 물체를 다 찾아야 통과.
RESCAN_TOOL = _paths.SIM2REAL / "scripts" / "ops" / "fpp_rescan.py"
RESCAN_WAIT_S = 60


def rescan_argv(ep: Episode, names) -> list[str]:
    """그 물체들을 FP++ 로 다시 찍는 명령 — 토픽이 /objects/<실물>/pose 가 아니면 ValueError(다시 찍을 실물 이름이 없다)."""
    real = [real_object_name(str((ep.objects.get(n) or {}).get("topic", ""))) for n in names]
    if not all(real):
        raise ValueError(f"FP++ 실물 이름을 모른다 — {[n for n, r in zip(names, real) if not r]} 의 토픽이 /objects/<실물>/pose 가 아니다")
    return [sys.executable, str(RESCAN_TOOL), *real, "--wait", str(RESCAN_WAIT_S), "--require-all"]


def reset_until_cup_fresh(call, *, retry_s: float = RESET_CUP_RETRY_S, step_s: float = RESET_CUP_RETRY_STEP_S,
                          sleep=time.sleep, clock=time.monotonic, cancelled=lambda: False) -> tuple[bool, list]:
    """reset 을 부르고, live 컵이 끊겨 거부되면 retry_s 안에서 다시 부른다. 마지막 답을 그대로 돌려준다."""
    deadline = clock() + retry_s
    while True:
        ok, why = call()
        stale = not ok and any(RESET_CUP_STALE_TEXT in str(w) for w in why)
        if not stale or cancelled() or clock() >= deadline:
            return ok, why
        sleep(step_s)
SEAT_DZ_CYL60 = HOLDER_FLOOR_Z + 0.085
HAND_OPEN_WAIT_S = 3.0            # 손 펴기 — RH56F1 전 행정 약 1 s(벤더 speedSet 기본) + 여유. 그 안에 실측이 펴져야 한다
HAND_OPEN_TOL_RAD = 0.15          # 펴짐 = 엄지 굽힘 · 네 손가락이 0(편 끝)에서 이 안(엄지 회전은 보지 않는다). pd 홈 손 경로 검사와 같은 값
                                  # (pd 는 손 목표를 한계 안쪽 0.05 로 물린다 — fake 실측 0.07~0.09 에서 섰다)
HAND_OPEN_AXES = ("thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
HAND_STATE = "/hand_{side}/joint_states"   # rh56f1_state_node — canonical r_hj_* · l_hj_*
ARM_STATE = "/joint_states"                 # 팔 실측(openarm_<side>_joint<i>) — replay_to_pd 가 시작 간격을 재는 토픽
#: ★10.30 계획 전 팔 정지 대기(사용자 1번 결정). 잡기 정책이 끝난 직후 팔은 마지막 목표로 아직 움직인다(oneshot 64 env:
#  분홍 잡기 끝 6번 관절이 0.4~0.8 s 동안 0.05~0.07 rad 더 감). 그 자세로 계획하면 계획하는 동안(실기 약 3 s) 팔이 계획 첫
#  프레임에서 멀어져 replay 시작 검사(0.05 rad)가 거부한다(to_tilt_start_r2 10/31). 창 안 관절 변화가 문턱 아래가 되면 계획한다.
ARM_STILL_WINDOW_S = 0.3
ARM_STILL_RAD = 0.003              # 창 안 관절마다 (max − min) — 0.01 rad/s 수준
ARM_STILL_WAIT_S = 2.0             # 이 안에 안 멈추면 그대로 계획한다(시작 검사가 여전히 막는다)
# ★11.10 tilt 직전 T_in 실측 실패 → FP++ 재잡기 + 정지 창 다시. 회차 = 재잡기(fpp_rescan 대기 RESCAN_WAIT_S, 도구 상한 +30 s)
# + 팔 정지 대기 ARM_STILL_WAIT_S + 창 tin_live.WINDOW_S ≈ 최대 95 s. 2 회면 컵을 든 채 3 분 남짓이라 그 위로는 늘리지 않는다.
TIN_RETRY_N = 2
TIN_RETRY_S = 200.0                # 재시도를 새로 시작하는 시간 상한(첫 측정부터)
TOOLS = Path(__file__).resolve().parents[1] / "tools"
#: ★11.08 리뷰 — 도구 대기 상한(s). 넘으면 끊고 controller_error(안전 정지). 재생은 rate-scale 0.25 로 2분 궤적 + 여유
TOOL_TIMEOUT_S = {"plan_carry.py": 180.0, "plan_tilt_traj.py": 180.0, "plan_rehome.py": 120.0, "plan_reach.py": 180.0,
                  "plan_lift.py": 180.0, "plan_tilt.py": 180.0, "check_path_start.py": 60.0, "replay_to_pd.py": 600.0, "episode_ctl.py": 120.0,
                  "trigger.py": 150.0, "hand_grasp.py": 60.0, "fpp_rescan.py": RESCAN_WAIT_S + 30.0}
TOOL_TIMEOUT_DEFAULT_S = 300.0
REPLAY_TIMEOUT_MARGIN_S = 120.0    # 재생 상한 = 궤적 길이 / rate-scale + 이 여유(진입 램프 ≤ 0.5 s · 기동 · 정착)


def _replay_limit(argv: list) -> float:
    """★11.08 재검토 — replay_to_pd 상한을 npz 길이 / rate-scale 로 잰다(고정 600 s 는 rate 0.1 이면 60 s 궤적도 끊는다).
    npz 를 못 읽으면 고정값으로 간다(재생 도구가 같은 파일로 따로 거부한다)."""
    base = TOOL_TIMEOUT_S["replay_to_pd.py"]
    try:
        npz = Path(argv[argv.index("--npz") + 1])
        rate = float(argv[argv.index("--rate-scale") + 1]) if "--rate-scale" in argv else 0.25
        with np.load(npz) as d:
            key = next(k for k in ("fabric_q", "arm_target", "arm_q_cmd") if k in d)   # replay_to_pd.NPZ_JOINT_KEYS
            n = int(np.asarray(d[key]).shape[0])
            dt = float(d["meta_step_dt"].item()) if "meta_step_dt" in d else (
                float(d["meta_fabric_dt"].item()) if "meta_fabric_dt" in d else 0.02)
        return max(base, n * dt / max(rate, 1e-3) + REPLAY_TIMEOUT_MARGIN_S)
    except (ValueError, IndexError, OSError, KeyError, StopIteration):
        return base


def _killpg(proc, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass
#: ★11.04 사용자 "go_home_start 가 너무 오래 걸린다 · 보통 control 의 rehome 으로 고쳐 두고 시작한다": 팔이 이미 홈(계약 home_arm)
#  이 안이고 멈춰 있으면 계획 · 재생 · goto_home 을 건너뛴다(실기 12:58 두 팔 차례로 26 s). 홈 밖이면 지금처럼 계획해서 간다.
HOME_SKIP_RAD = 0.03
SIDES = ("right", "left")
NODES = {"aglt": "rh_aglt_node", "place": "rh_place_node"}


def service_ns(kind: str, side: str) -> str:
    """정책 노드 서비스 ns — 한 팔에 둘이 같이 떠서 aglt = <팔>, place = <팔>_place(미션 에피소드 단계와 같다).
    이벤트는 둘 다 /policy_control/<팔>/episode 에 node 이름을 실어 낸다(pd 가 그 팔에 적용)."""
    return side if kind == "aglt" else f"{side}_{kind}"


def _place_cup_obs(policy: str | None) -> str:
    """등록부 놓기 정책 계약의 cup_obs(★11.02 live | attached). 계약 파일 이름은 등록부(policy_registry.check)가 정한다 —
    미션 생성기와 같은 규칙. 계약이 있는데 못 읽으면 크게 실패한다(조용히 넘기면 놓기가 옛 /pose 에 남는다 · 11.03 리뷰)."""
    from policy_control import policy_registry as R
    d = _paths.SIM2REAL / "deploy" / "policies" / str(policy)
    e = R.check(d, deep=False)
    if not e.contract:
        return ""
    doc = json.loads((d / e.contract).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{d / e.contract}: 계약이 JSON 객체가 아니다")
    return str(doc.get("cup_obs", "attached"))


def job_params(ep: Episode, p, tin: Mapping | None = None) -> dict:
    """★10.28 정책 노드에 reset 전에 넣는 파라미터 — 노드는 첫 job 값으로 뜨고 reset 마다 이것을 다시 읽는다.

    object = 다물체 정책의 물체 이름(10.12, YAML 에 적힌 단계만) · aglt cup_topic = 잡을 물체의 FP++ live 토픽(미션 생성기
    _live 와 같은 규칙) · place holder = 실물 홀더 번호. 같은 팔이 두 물체를 잡고 두 홀더에 놓는 칵테일(cocktail_1 · 1b)이
    쓴다 — 전에는 노드가 컵 · 홀더를 시작 때만 읽어 분홍 잡기가 파랑을 쫓고 분홍 놓기가 홀더 1 을 노렸다.
    ★11.03 place cup_topic = 계약 cup_obs live 일 때만 놓을 물체(source_object)의 live 토픽(미션 생성기와 같은 규칙, attached 계약은 안 넣는다).
    ★10.29 place holder_tol_m = 카드가 그 job 에 넓힌 홀더 허용, 없으면 HOLDER_TOL_M. 놓기 job 마다 늘 넣는다 — 앞 job 의
    넓힌 값이 노드에 남아 다음 홀더 검사를 느슨하게 하지 않게.
    ★11.07 place attach_source tin_json = 붙인 자세를 그 팔의 tilt 직전 T_in json(tin: 팔 → 경로, 없으면 "")으로.
    place job 마다 늘 넣는다(바인딩 없으면 "" = 계약 그대로) — 노드가 앞 에피소드 · job 의 tin_json 을 들고 있지 않게."""
    out: dict = {}
    if p.kind in ("aglt", "place") and getattr(p, "policy_object", None):
        out["object"] = str(p.policy_object)
    if p.kind == "aglt" and p.target_object:
        topic = live_topic(str((ep.objects.get(p.target_object) or {}).get("topic", "")))
        if topic:
            out["cup_topic"] = topic
    if p.kind == "place" and p.source_object and _place_cup_obs(p.policy) == "live":   # 파랑 → 분홍 놓기가 파랑 live 에 머물지 않게
        topic = live_topic(str((ep.objects.get(p.source_object) or {}).get("topic", "")))
        if topic:
            out["cup_topic"] = topic
    if p.kind == "place" and p.holder_id is not None:
        out["holder"] = int(p.holder_id)
        tol = getattr(p, "holder_tol_m", None)
        out["holder_tol_m"] = float(HOLDER_TOL_M if tol is None else tol)
    if p.kind == "place":
        b = ep.policies.get(p.role)
        on = b is not None and b.attach_source == "tin_json"
        out["attach_source"] = "tin_json" if on else ""
        out["tin_json"] = str((tin or {}).get(p.side, "")) if on else ""
    # ★11.08 들기 가드 — 오른손 잡기(aglt) job 이면 에피소드와 상관없이 늘 넣는다(끄는 job 은 0 · nan 을 명시).
    #   노드는 reset 에서 읽자마자 끔으로 되돌리므로 값은 그 job 한 번만 쓰인다
    if p.kind == "aglt" and p.side == "right":
        g, h, x, y = getattr(p, "lift_guard", None) or (0.0, 0.0, float("nan"), float("nan"))
        out.update(lift_guard_mm=float(g), lift_hstop_m=float(h), lift_guard_x=float(x), lift_guard_y=float(y))
    return out


def _q(msg) -> tuple:
    p, o = msg.pose.position, msg.pose.orientation
    return (p.x, p.y, p.z), (o.w, o.x, o.y, o.z)


def real_object_name(topic: str) -> str:
    """/objects/<실물>/pose → 실물 이름(config/objects 등록 이름). 아니면 ""."""
    parts = [x for x in str(topic).split("/") if x]
    return parts[1] if len(parts) == 3 and parts[0] == "objects" else ""


#: ★11.12 rehome 계획에 쓸 live 자세의 최대 나이[s](monotonic) — 더 낡으면 "자세 없음"(리뷰 H2: 옮겨진 컵의 옛 자리로 계획하지 않게)
REHOME_LIVE_MAX_AGE_S = 5.0


class RehomeCupUnknown(Exception):
    """놓기 실패 컵(unknown)의 자리를 snapshot · 신선한 live 어디서도 못 얻었다 — 컵 위치를 모르는 채 팔을 움직이지 않는다."""


def rehome_table_cups(ep: Episode, world, relay: Mapping, live: Mapping,
                      now: float | None = None) -> tuple[list[dict], list[str]]:
    """★11.11 rehome 계획 세계에 넣을 탁자 위 컵(plan_home_path --table-cups) + 로그 문구.
    손에 쥔 컵 · 홀더에 꽂힌 컵(홀더 원기둥이 덮는다) · 아직 안 놓은 컵(away)은 뺀다. 자리 = snapshot 고정값
    (/episode/objects/<이름>) → 없으면 live 마지막 값(나이 ≤ REHOME_LIVE_MAX_AGE_S) → table 컵만 에피소드 region 가운데
    (넓히지 않음). unknown 컵이 자리를 못 얻으면 RehomeCupUnknown(★11.12 리뷰 H1 — 조용히 빼지 않는다)."""
    from policy_control.episode_spec import region_box
    now = time.monotonic() if now is None else now
    cups, notes, lost = [], [], []
    for name, o in ep.objects.items():
        at = str((world.objects.get(name) or {}).get("at", "table"))
        if at not in ("table", "unknown"):
            continue
        real = real_object_name(str(o.get("topic", "")))
        if not real:                                   # 리뷰 M3: 실물 토픽이 아니면 메쉬를 모른다 — 남긴다
            notes.append(f"{name}({at}) 실물 토픽 아님({o.get('topic')!r}) — 계획 세계에서 뺌")
            if at == "unknown":
                lost.append(f"{name}(실물 토픽 아님)")
            continue
        loc = world.objects.get(name) or {}
        pos, src = (loc.get("pos"), "snapshot") if at == "table" else (None, "")
        if pos is None and at == "table" and name in relay:
            pos, src = relay[name][0], "snapshot"
        if pos is None and live.get(name):
            f = live[name][-1]
            age = now - float(f[2])
            if age <= REHOME_LIVE_MAX_AGE_S:
                pos, src = f[0], f"live({age:.1f} s)"
            else:
                notes.append(f"{name} live 낡음 {age:.1f} s > {REHOME_LIVE_MAX_AGE_S} s — 자세 없음으로 봄")
        item = {"name": name, "object": real}
        if pos is not None:
            cups.append(item | {"pos": [round(float(v), 4) for v in pos]})
        elif at == "table" and o.get("region") is not None and set(region_box(o["region"])) == {"x", "y"}:
            (x0, x1), (y0, y1) = (region_box(o["region"])[k] for k in ("x", "y"))
            # spread 0: 반대각선(28 mm)만큼 넓히면 홈 목표 손가락이 16 mm 로 들어와 계획이 거부된다(184028 6/6, 11.11)
            cups.append(item | {"pos": [round((x0 + x1) / 2, 4), round((y0 + y1) / 2, 4)]})
            src = "region 기본값"
        else:
            notes.append(f"{name}({at}) 자리 모름")
            if at == "unknown":
                lost.append(name)
            continue
        notes.append(f"{name} {src} {cups[-1]['pos']}")
    if lost:
        raise RehomeCupUnknown(f"놓기 실패 컵 자리 모름 {lost} — 컵 위치를 모르는 채 팔을 움직이지 않는다 ({'; '.join(notes)})")
    return cups, notes


#: 홀더 번호 = 이름 끝 숫자(테이블 왼쪽부터, config/cup_holders.yaml) — 에피소드 holders · /objects/cup_holder_N/pose 와 같은 번호.
HOLDER_NAME = re.compile(r"cup_holder_(\d+)")
ARM_JOINT_NAME = re.compile(r"openarm_(right|left)_joint([1-7])")   # /joint_states 이름 → canonical r_aj_i · l_aj_i


def load_holder_poses(path: Path) -> dict[int, tuple]:
    """holder_poses 파일 → {홀더 번호: (pos, yaw)}. 없으면 빈 dict.
    ★10.20 번호는 이름에서 읽는다 — marker_id 가 아니다(사용자 "순서를 다시 0,1,2로", 마커 짝은 config/cup_holders.yaml)."""
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[int, tuple] = {}
    for name, h in (raw.get("holders") or {}).items():
        m = HOLDER_NAME.fullmatch(str(name))
        if m is None:
            raise ValueError(f"{path}: 홀더 이름 {name!r} 은 cup_holder_<번호> 꼴이어야 한다")
        out[int(m.group(1))] = (tuple(float(v) for v in h["position"]), float(h.get("yaw_rad", 0.0)))
    return out


def snapshot_of(frames: list) -> tuple | None:
    """창 안 프레임들 → (pos 중앙값, 마지막 quat) — 프레임이 모자라거나 움직이면 None."""
    if len(frames) < SNAPSHOT_MIN_FRAMES:
        return None
    pos = np.array([f[0] for f in frames], float)
    if float(np.max(np.linalg.norm(pos - np.median(pos, axis=0), axis=1))) > SNAPSHOT_STILL_M:
        return None
    return tuple(float(v) for v in np.median(pos, axis=0)), tuple(frames[-1][1])


def pick_scene_pose(live: list, static: list, rescanned: bool, waited_s: float) -> tuple:
    """★11.06 scene 자리 고르기 → (snap, 출처, 사유). snap None 이면 아직 못 정함.
    다시 찍은 직후엔 …/pose 가 정본. 아니면 live 가 정본: live 가 멈춰 있으면 그 자리(z 는 known_z 보정된 …/pose 것),
    live 가 움직이면 기다린다(옛 자리로 가지 않음). live 가 없거나 모자랄 때만 …/pose — 모자라면 창 하나만큼 기다린 뒤."""
    s_snap, l_snap = snapshot_of(static), snapshot_of(live)
    why = f"live {len(live)} · pose {len(static)} 프레임"
    if rescanned:
        return (s_snap, "pose", why) if s_snap else (None, None, why + " · 다시 찍은 자리 없음/움직임")
    if l_snap:
        (x, y, z), q = l_snap
        return ((x, y, s_snap[0][2] if s_snap else z), q), "live", why
    if len(live) >= SNAPSHOT_MIN_FRAMES:
        return None, None, why + " · live 움직임"
    if live and waited_s < SNAPSHOT_WINDOW_S:
        return None, None, why + " · live 모자람"
    return (s_snap, "pose", why) if s_snap else (None, None, why + (" · pose 움직임" if len(static) >= SNAPSHOT_MIN_FRAMES else ""))


def arm_still(frames: list, joints: list, window_s: float = ARM_STILL_WINDOW_S,
              tol_rad: float = ARM_STILL_RAD) -> tuple[bool, float | None]:
    """팔 실측 프레임 [(시각, {이름: rad}), ...] → (멈췄나, 창 안 관절별 변화의 최댓값). 창을 덮을 만큼 프레임이 없으면 (False, None)."""
    if not frames or frames[-1][0] - frames[0][0] < window_s:
        return False, None
    t_end = frames[-1][0]
    win = [f for f in frames if t_end - f[0] <= window_s]
    if len(win) < 2 or any(j not in f[1] for f in win for j in joints):
        return False, None
    q = np.array([[f[1][j] for j in joints] for f in win], float)
    span = float(np.max(q.max(axis=0) - q.min(axis=0)))
    return span < tol_rad, span


def home_arm(robot: str, side: str) -> list[float] | None:
    """plan_rehome 이 목표로 쓰는 계약의 그 팔 home_arm(r_aj_1..7 순). 계약이 없으면 None(건너뛰기 안 함)."""
    sys.path.insert(0, str(TOOLS))
    try:
        from plan_rehome import ROBOTS
        argv = ROBOTS.get(robot, [])
        path = Path(argv[argv.index("--contract") + 1]) if "--contract" in argv else None
        return [float(v) for v in json.loads(path.read_text())["sides"][side]["home_arm"]] if path else None
    except (ImportError, OSError, ValueError, KeyError, IndexError):
        return None


def at_home(frames: list, side: str, home: list[float] | None, tol_rad: float = HOME_SKIP_RAD) -> tuple[bool, float | None]:
    """팔 실측 마지막 프레임이 홈 tol 안이고 멈춰 있나 → (그렇다, 관절 최대 차). 실측 · 홈이 없으면 (False, None)."""
    joints = [f"{side[0]}_aj_{i}" for i in range(1, 8)]
    if not home or not frames or any(j not in frames[-1][1] for j in joints):
        return False, None
    err = float(np.max(np.abs(np.array([frames[-1][1][j] for j in joints]) - np.array(home))))
    still, _ = arm_still(frames, joints)
    return err <= tol_rad and still, err


def cup_in_holder(cup_pos, holder_pos, seat_dz: float, xy_tol: float = 0.035, z_tol: float = 0.03) -> bool:
    c, h = np.asarray(cup_pos, float), np.asarray(holder_pos, float)
    return bool(np.linalg.norm(c[:2] - h[:2]) < xy_tol and abs(c[2] - (h[2] + seat_dz)) < z_tol)


def write_run_meta(run_dir: Path, ep: Episode, robot: str) -> None:
    """★11.09 사용자 "tilt v12 등 자료가 뭔지 구분" — 런 폴더에 meta.json(git HEAD · 에피소드 · carry/tilt 자산 버전).
    같은 에피소드 이름이라도 자산이 바뀌면(cocktail_1c_e4000: 11.08 v11 → v12) 여기서 갈린다. 실패해도 런은 계속한다."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(_paths.SIM2REAL), *args], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    assets = {name: t.asset for name, t in ep.trajectories.items() if t.asset}
    meta = {"git": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--short", "--untracked-files=no")),
            "argv": sys.argv, "host": os.uname().nodename, "start": time.strftime("%F %T"), "episode": ep.name,
            "episode_version": ep.version, "episode_path": ep.path, "robot": robot, "assets": assets}
    try:
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


class RosExecutor:
    """EpisodeManager 의 실행기 — rclpy 노드(콜백은 다른 스레드의 executor 가 돌린다)를 받아 쓴다."""

    def __init__(self, node, ep: Episode, *, robot: str = "rh56f1", run_dir: Path | None = None,
                 contract: str = "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json",
                 runner: str = DEFAULT_RUNNER, fpp_rescan: bool = True) -> None:
        from geometry_msgs.msg import Point, PoseStamped
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.node, self.ep, self.robot, self.runner = node, ep, robot, runner
        #: ★10.28 snapshot objects 를 FP++ 로 다시 찍는가 — fake 도메인은 FP++ 가 없다(fake 컵이 고정 자세를 낸다)
        self.fpp_rescan = fpp_rescan
        # ★10.19 이 에피소드가 움직이는 팔 — safe_stop 은 이 팔의 정책만 멈춘다(다른 팔 실행기가 따로 도는 정책을 건드리지 않는다).
        #   못 찾으면 두 팔(옛 동작).
        self.sides = tuple(episode_sides(ep)) or SIDES
        self.run_dir = run_dir or run_paths.run_dir("episode", ep.name)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_run_meta(self.run_dir, ep, robot)
        self.contract = str(_paths.SIM2REAL / contract)
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._events: dict[str, dict] = {}
        self._status: dict[str, dict] = {}
        self._pd: dict[str, dict] = {}
        self._frames: dict[str, list] = {name: [] for name in ep.objects}
        self._live: dict[str, list] = {name: [] for name in ep.objects}   # ★10.20 lift 판정용 FP++ live 추적
        self._hand: dict[str, dict] = {s: {} for s in SIDES}   # ★10.17 손 실측(손 펴짐 확인) — 이름 → (rad, 시각)
        self._arm: list = []                          # ★10.30 팔 실측 [(시각, {canonical 이름: rad})] — 계획 전 정지 대기
        self._tin_rec: dict | None = None             # ★11.07 tilt 직전 T_in 기록 중 {"live": {물체: [(시각, stamp, x7)]}, "arm": [(시각, q)]}
        self._tin_json: dict[str, str] = {}           # 팔 → 이 에피소드 마지막 T_in json(place attach_source tin_json)
        self._tin_why: dict[str, str] = {}            # 팔 → T_in 을 못 낸 사유(place 거부 문구)
        self._tin_note: list = []                     # 이 tilt 노드의 T_in 출처 한 줄씩(노드 결과 reason 에 붙인다)
        self._goal_result: dict[str, dict] = {}
        self._relay: dict[str, tuple] = {}            # 물체 → (pos, quat) — snapshot 뒤 집힐 때까지
        #: ★11.08 lanes — 두 줄이 도구(계획 · replay_to_pd)를 동시에 띄운다. safe_stop 이 다 끊게 하나가 아니라 모음
        self._procs: set = set()
        self._tool_seq = 0
        self._Point, self._Trigger, self._Pose = Point, Trigger, PoseStamped
        #: 서비스 응답은 다른 콜백(예: 노드의 stop 서비스) 안에서도 받아야 한다 — 서로 막지 않는 그룹
        self._cbg = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for side in SIDES:
            node.create_subscription(String, f"{NS}/{side}/episode", self._on_event, latched)
            node.create_subscription(String, f"{NS}/{side}/goal_result", self._on_json(self._goal_result, side), latched)
            node.create_subscription(String, f"{NS}/status/pd_{side}", self._on_json(self._pd, side), QoSProfile(depth=10))
            for kind in ("rh_aglt_node", "rh_place_node"):
                name = f"{kind}_{side}"
                node.create_subscription(String, f"{NS}/status/{name}", self._on_json(self._status, name), QoSProfile(depth=10))
        snap_objs = set(ep.objects) if any(n.type == "snapshot" for n in ep.nodes) else set()   # 경로 B 는 FP++ 를 안 받는다
        for name in snap_objs:
            node.create_subscription(PoseStamped, ep.objects[name]["topic"], self._on_frame(name), qos_profile_sensor_data)
        #: ★10.20 실기 1-B: 들림 판정은 live 추적으로 본다. …/pose 는 locate 때 한 번 찍은 자리를 계속 다시 내서
        #  032511 shaker 가 live 로 9.6 cm 들렸는데 before = after 로 0.0 cm 판정 → 들고 멈춤.
        #: ★11.04 놓기 판정도 live — …/pose 는 cups 때 찍은 테이블 자리를 계속 내서 제대로 놓아도 cup_in_holder False 가 된다
        place_src = {j.source_object for n in ep.nodes for j in getattr(n, "jobs", ()) or () if j.source_object}
        #: ★11.06 scene snapshot 도 live 먼저 — …/pose 는 다시 찍기 전까지 옛 자리라 컵을 옮겨도 region 거부가 풀리지 않았다
        for name in {t.object for t in ep.trajectories.values() if t.kind == "lift" and t.object} | place_src | snap_objs:
            topic = live_topic(ep.objects.get(name, {}).get("topic", ""))
            if topic:
                node.create_subscription(PoseStamped, topic, self._on_frame(name, self._live), qos_profile_sensor_data)
        if ep.trajectories:                     # ★11.04 rehome 도 — 이미 홈이면 건너뛴다(at_home)
            from sensor_msgs.msg import JointState
            node.create_subscription(JointState, ARM_STATE, self._on_arm, qos_profile_sensor_data)
        if any(n.type == "hand" for n in ep.nodes):
            from sensor_msgs.msg import JointState
            for side in SIDES:
                node.create_subscription(JointState, HAND_STATE.format(side=side), self._on_hand(side), qos_profile_sensor_data)
        self._goal_pub = {s: node.create_publisher(Point, f"{NS}/{s}/goal", QoSProfile(depth=10)) for s in SIDES}
        self._relay_pub = {name: node.create_publisher(PoseStamped, relay_topic(runner, name), 10) for name in ep.objects}
        self._holder_pub = {}
        self.holders = load_holder_poses(_paths.SIM2REAL / ep.holder_poses) if ep.holder_poses else {}
        for hid in sorted(set(ep.holders.values())):
            self._holder_pub[hid] = node.create_publisher(PoseStamped, f"/objects/cup_holder_{hid}/pose", latched)
        self.publish_holders()
        node.create_timer(1.0 / 30.0, self._relay_tick)

    # ---------------------------------------------------------------- 입력
    def _on_event(self, msg) -> None:
        """팔 이벤트 토픽 — 정책 노드 둘이 같이 내므로 본문 node 로 가른다."""
        try:
            body = json.loads(msg.data)
        except ValueError:
            return
        with self._lock:
            self._events[str(body.get("node", ""))] = {**body, "_t": time.monotonic()}

    def _on_json(self, store: dict, key: str):
        def cb(msg) -> None:
            try:
                body = json.loads(msg.data)
            except ValueError:
                return
            with self._lock:
                store[key] = {**body, "_t": time.monotonic()}
        return cb

    def _on_hand(self, side: str):
        def cb(msg) -> None:
            now = time.monotonic()
            with self._lock:
                self._hand[side].update({n: (float(p), now) for n, p in zip(msg.name, msg.position)})
        return cb

    def _on_arm(self, msg) -> None:
        now = time.monotonic()
        q = {}
        for n, p in zip(msg.name, msg.position):
            m = ARM_JOINT_NAME.fullmatch(n)
            if m:
                q[f"{m.group(1)[0]}_aj_{m.group(2)}"] = float(p)
        if not q:
            return
        with self._lock:
            self._arm.append((now, q))
            if self._tin_rec is not None:
                self._tin_rec["arm"].append((now, q))
            cut = now - 2 * ARM_STILL_WINDOW_S
            while self._arm and self._arm[0][0] < cut:
                self._arm.pop(0)

    def _wait_arm_still(self, sides) -> dict:
        """★10.30 계획 전: 그 팔 관절이 ARM_STILL_WINDOW_S 동안 ARM_STILL_RAD 안에 머물 때까지(최대 ARM_STILL_WAIT_S) 기다린다.
        팔 실측이 한 번도 안 왔으면(실측 없는 fake 등) 기다리지 않는다. 결과는 기록용(안 멈춰도 계획은 한다)."""
        joints = [f"{s[0]}_aj_{i}" for s in sides for i in range(1, 8)]
        t0 = time.monotonic()
        span = None
        while not self.cancel.is_set():
            with self._lock:
                fr = list(self._arm)
            if not fr and time.monotonic() - t0 > ARM_STILL_WINDOW_S:
                return {"arm_still": None, "arm_wait_s": 0.0}
            ok, span = arm_still(fr, joints)
            if ok or time.monotonic() - t0 >= ARM_STILL_WAIT_S:
                return {"arm_still": ok, "arm_wait_s": round(time.monotonic() - t0, 2),
                        "arm_span_rad": None if span is None else round(span, 4)}
            time.sleep(0.02)
        return {"arm_still": None, "arm_wait_s": round(time.monotonic() - t0, 2)}

    def _hand_open_err(self, side: str) -> float | None:
        """펴짐 오차 max |q| [rad](엄지 굽힘 · 네 손가락) — 실측이 0.5 s 안에 없으면 None."""
        names = [f"{side[0]}_hj_{a}" for a in HAND_OPEN_AXES]
        with self._lock:
            got = [self._hand[side].get(n) for n in names]
        if any(g is None or time.monotonic() - g[1] > 0.5 for g in got):
            return None
        return max(abs(g[0]) for g in got)

    def _on_frame(self, name: str, store: dict | None = None):
        store = self._frames if store is None else store

        def cb(msg) -> None:
            with self._lock:
                fr = store[name]
                pq, now = _q(msg), time.monotonic()
                fr.append((*pq, now))
                if self._tin_rec is not None and store is self._live:
                    st = msg.header.stamp
                    self._tin_rec["live"].setdefault(name, []).append((now, st.sec + st.nanosec * 1e-9, (*pq[0], *pq[1])))
                cutoff = time.monotonic() - SNAPSHOT_WINDOW_S
                while fr and fr[0][2] < cutoff:
                    fr.pop(0)
        return cb

    def _pose_msg(self, pos, quat, frame: str = "base_link"):
        m = self._Pose()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (float(v) for v in quat)
        return m

    def publish_holders(self) -> list[str]:
        missing = []
        for hid, pub in self._holder_pub.items():
            if hid not in self.holders:
                missing.append(f"holder {hid}")
                continue
            pos, yaw = self.holders[hid]
            pub.publish(self._pose_msg(pos, (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))))
        return missing

    def _relay_tick(self) -> None:
        with self._lock:
            relay = dict(self._relay)
        for name, (pos, quat) in relay.items():
            self._relay_pub[name].publish(self._pose_msg(pos, quat))

    # ---------------------------------------------------------------- 서비스
    def _call(self, path: str, timeout: float = 5.0, wait: float = 2.0) -> tuple[bool, list]:
        cli = self.node.create_client(self._Trigger, path, callback_group=self._cbg)
        try:
            if not cli.wait_for_service(timeout_sec=wait):
                return False, [f"{path}: 서비스가 없다(노드가 떠 있는가)"]
            fut = cli.call_async(self._Trigger.Request())
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < timeout:
                time.sleep(0.02)
            if not fut.done():
                return False, [f"{path}: 응답 없음({timeout:.0f} s)"]
            res = fut.result()
            try:
                body = json.loads(res.message)
                return bool(body.get("ok", res.success)), list(body.get("reasons", []))
            except ValueError:
                return bool(res.success), [res.message] if res.message else []
        finally:
            self.node.destroy_client(cli)

    def _set_params(self, node: str, params: Mapping, timeout: float = 5.0, wait: float = 2.0) -> tuple[bool, list]:
        """정책 노드 파라미터(job_params — object · cup_topic · holder · holder_tol_m)를 SetParameters 한 번으로 — 다음 reset 이 읽는다."""
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        path = f"{node}/set_parameters"
        cli = self.node.create_client(SetParameters, path, callback_group=self._cbg)
        try:
            if not cli.wait_for_service(timeout_sec=wait):
                return False, [f"{path}: 서비스가 없다(노드가 떠 있는가)"]
            req = SetParameters.Request()
            for name, v in params.items():               # 노드가 선언한 타입 그대로(holder 는 정수, holder_tol_m 은 실수, 나머지는 문자열)
                req.parameters.append(Parameter(name=name, value=(
                    ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=v) if isinstance(v, int)
                    else ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v) if isinstance(v, float)
                    else ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=str(v)))))
            fut = cli.call_async(req)
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < timeout:
                time.sleep(0.02)
            if not fut.done():
                return False, [f"{path}: 응답 없음({timeout:.0f} s)"]
            bad = [r.reason for r in fut.result().results if not r.successful]
            return (False, [f"{path} {dict(params)}: {bad}"]) if bad else (True, [])
        finally:
            self.node.destroy_client(cli)

    def _pd_fault(self, sides) -> str | None:
        with self._lock:
            for s in sides:
                st = self._pd.get(s) or {}
                if st.get("estop") or st.get("phase") == "HOLD":
                    return f"pd_{s} {'estop' if st.get('estop') else 'HOLD'} {st.get('reasons', '')}"
        return None

    # ---------------------------------------------------------------- 실행기 계약
    def snapshot(self, node: Node, world) -> tuple[NodeResult, dict]:
        since = 0.0                                       # 이 시각(monotonic) 뒤에 온 프레임만
        if node.objects:                                  # ★10.28 사람이 방금 놓은 물체만 — FP++ 로 다시 찍은 뒤의 자리
            names = list(node.objects)
            if self.fpp_rescan:
                try:
                    argv = rescan_argv(self.ep, names)
                except ValueError as exc:
                    return NodeResult("refused", f"cup pose missing — {exc}"), {}
                rc, tail = self._run(argv)
                if rc != 0:
                    return NodeResult("refused", f"cup pose missing — FP++ 다시 찍기 rc {rc}: {tail}"), {}
                since = time.monotonic()
        else:
            names = [n for n in self.ep.objects if (world.objects.get(n) or {}).get("at", "table") == "table"]
        rescanned = since > 0.0
        t0 = time.monotonic()
        deadline = t0 + SNAPSHOT_WINDOW_S + 3.0
        got, why, src, gap = {}, {}, {}, {}
        while time.monotonic() < deadline and len(got) < len(names) and not self.cancel.is_set():
            time.sleep(0.1)
            with self._lock:
                live = {n: list(self._live.get(n, ())) for n in names}
                frames = {n: list(self._frames[n]) for n in names}
            now = time.monotonic()
            for n in names:
                if n in got:
                    continue
                lv, st = ([f for f in fr if now - f[2] < SNAPSHOT_WINDOW_S and f[2] >= since] for fr in (live[n], frames[n]))
                snap, kind, why[n] = pick_scene_pose(lv, st, rescanned, now - t0)
                if snap is not None:
                    got[n], src[n] = snap, kind
                    other = snapshot_of(st if kind == "live" else lv)
                    if other is not None:                 # live 와 …/pose 가 어긋난 정도 — scene.json 에 남김
                        gap[n] = round(float(np.linalg.norm(np.subtract(snap[0][:2], other[0][:2]))), 4)
        missing = [n for n in names if n not in got]
        if missing:
            return NodeResult("refused", "cup pose missing — " + ", ".join(f"{n}({why.get(n, '없음')})" for n in missing)), {}
        with self._lock:
            self._relay.update(got)
        (self.run_dir / "scene.json").write_text(json.dumps({"at": time.strftime("%F %T"), "objects": got, "source": src,
                                                           "live_static_gap_m": gap}, indent=2))
        return NodeResult("completed", "", signals={"objects": sorted(got)}), got

    def run_policies(self, node: Node, plans: list, world) -> list:
        out: list = [None] * len(plans)

        def one(i, p):
            try:
                out[i] = self._policy(p)
            except Exception as exc:                  # 스레드 안 예외는 결과로
                out[i] = NodeResult("error", f"{type(exc).__name__}: {exc}", role=p.role)
        threads = [threading.Thread(target=one, args=(i, p), daemon=True) for i, p in enumerate(plans)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for p, res in zip(plans, out):               # 집힌 물체는 더 다시 내지 않는다(이제 손바닥 FK)
            if p.kind == "aglt" and res is not None and res.status == "completed" and p.target_object:
                with self._lock:
                    self._relay.pop(p.target_object, None)
        return out

    def _policy(self, p) -> NodeResult:
        side = p.side
        node_name = NODES.get(p.kind)
        if node_name is None:
            return NodeResult("refused", f"{p.kind} 정책 노드가 아직 없다", role=p.role)
        status_key = f"{node_name}_{side}"
        ns = service_ns(p.kind, side)
        self._wait_node_ready(status_key)                 # 연속 실행: snapshot 직후 재발행 컵이 노드에 닿을 시간
        t0 = time.monotonic()
        with self._lock:
            last_ep = int((self._events.get(status_key) or {}).get("episode", 0))
        params = job_params(self.ep, p, self._tin_json)   # ★10.12 물체 one-hot · ★10.28 컵 · 홀더 — 노드 reset 이 다시 읽는다
        if params.get("attach_source") == "tin_json" and not params["tin_json"]:
            why = self._tin_why.get(side, "이 에피소드에서 그 팔 tilt 가 돌지 않았다(tilt 가 먼저 돌아야 한다)")
            return NodeResult("refused", f"place tin_json: {side} 팔 tilt 직전 T_in 없음 — {why}", role=p.role)
        if params:
            ok, why = self._set_params(f"/{node_name}_{side}", params)
            if not ok:
                return NodeResult("refused", "; ".join(why), role=p.role)
        t_reset = time.monotonic()
        ok, why = reset_until_cup_fresh(lambda: self._call(f"{NS}/{ns}/episode/reset"), cancelled=self.cancel.is_set)
        if self.cancel.is_set():                          # ★11.08 정지로 끊긴 reset 은 거부가 아니라 정지
            return NodeResult("aborted", "episode runner stop", time.monotonic() - t0, p.role)
        if not ok:
            return NodeResult("refused", "; ".join(why), role=p.role)
        if getattr(p, "lift_guard", None) is not None:    # ★11.08 가드를 요청했으면 노드가 실제로 켰는지 start 전에 본다
            why = self._lift_guard_on(status_key, p.lift_guard, t_reset)
            if self.cancel.is_set():                      # ★11.08 정지로 끊긴 가드 확인은 lift_guard 실패가 아니다
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0, p.role)
            if why:
                return NodeResult("refused", f"lift_guard: {why}", role=p.role)
        t0 = time.monotonic()                             # 정책 시간 한도는 reset 이 받아들여진 뒤부터 센다
        if p.kind == "aglt" and p.setting is not None:
            with self._lock:
                self._goal_result.pop(side, None)
            x, y, z = p.setting
            self._goal_pub[side].publish(self._Point(x=float(x), y=float(y), z=float(z)))
            deadline = time.monotonic() + 3.0
            res = None
            while time.monotonic() < deadline:
                with self._lock:
                    res = self._goal_result.get(side)
                if res and [round(v, 4) for v in res.get("request", [])] == [round(float(v), 4) for v in p.setting]:
                    break
                time.sleep(0.05)
            if not res or not res.get("ok"):
                return NodeResult("refused", f"goal {p.setting} refused: {(res or {}).get('reasons', 'no answer')}", role=p.role)
        if self.cancel.is_set():                          # ★11.08 정지가 reset · 목표 사이에 왔다 — 정책을 시작하지 않는다
            return NodeResult("aborted", "episode runner stop", time.monotonic() - t0, p.role)
        ok, why = self._call(f"{NS}/{ns}/episode/start")
        if not ok:
            return NodeResult("refused", "; ".join(why), role=p.role)
        limit = float(getattr(p, "max_s", 15.0)) + POLICY_MARGIN_S     # ★10.17 경로 C 는 노드 한도가 길다
        while True:
            time.sleep(0.05)
            if self.cancel.is_set():
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0, p.role)
            fault = self._pd_fault([side])
            if fault:
                self._call(f"{NS}/{ns}/episode/stop")
                return NodeResult("completed", fault, time.monotonic() - t0, p.role, {"pd_fault": fault})
            with self._lock:
                ev = dict(self._events.get(status_key) or {})
                st = dict(self._status.get(status_key) or {})
            if int(ev.get("episode", 0)) > last_ep and ev.get("event") in ("stop", "abort"):
                break
            if time.monotonic() - t0 > limit:
                self._call(f"{NS}/{ns}/episode/stop")
                return NodeResult("timeout", f"no stop event in {limit:.0f} s", time.monotonic() - t0, p.role)
        reasons = "; ".join(ev.get("reasons", []))
        dur = time.monotonic() - t0
        if ev.get("event") == "abort":                # ★11.02 abort(경로 D 시간 초과)도 신호를 남긴다 — 못 쥔 건지 들다 만 건지 구분
            return NodeResult("aborted", reasons, dur, p.role, self._signals(p, st))
        if "user stop" in reasons:                    # 사람이 정지 바로 멈췄다 — 복구하지 않고 실행기도 멈춘다
            return NodeResult("aborted", f"operator stopped the policy node ({reasons})", dur, p.role)
        if "episode time" in reasons:
            return NodeResult("timeout", reasons, dur, p.role, self._signals(p, st))
        sig = self._signals(p, st)
        if p.kind == "aglt" and any(str(r).startswith(GUARD_STOPS) for r in ev.get("reasons", [])) and not sig.get("dropped"):
            ok, why, hold = self._guard_hold(p, status_key)   # 떨어뜨렸으면(dropped) 판정 없이 episode_failure 가 OBJECT_DROPPED 로
            sig = {**sig, **hold, "guard_hold_ok": ok}
            if hold.get("guard_hold_cancelled"):      # ★11.08 리뷰 3 — 판정 도중 정지: 판정 실패가 아니다
                return NodeResult("aborted", f"episode runner stop — 가드 고정 판정 중 정지 ({reasons})",
                                  time.monotonic() - t0, p.role, sig)
            if not ok:                                # GRASP_FAILED(되돌아가 손 펴기)가 아니라 lift_guard_failed — 언제나 safe stop
                return NodeResult("aborted", f"{GUARD_FAIL_TEXT}: {why} ({reasons})", time.monotonic() - t0, p.role, sig)
        return NodeResult("completed", reasons, dur, p.role, sig)

    def _lift_guard_on(self, status_key: str, want: tuple, since: float) -> str:
        """★11.08 reset 뒤 노드 status lift_guard 가 요청(plan.lift_guard)과 같은지 — 다르면 거부 사유(빈 문자열 = 켜짐)."""
        g, h, x, y = (float(v) for v in want)
        deadline, lg = time.monotonic() + LG_CHECK_WAIT_S, None
        while time.monotonic() < deadline and not self.cancel.is_set():
            with self._lock:
                st = dict(self._status.get(status_key) or {})
            if float(st.get("_t", 0.0)) > since and st.get("phase") == "armed":
                lg = st.get("lift_guard") or {}
                try:
                    if (lg.get("on") is True and abs(float(lg["x"]) - x) <= LG_XY_TOL_M and abs(float(lg["y"]) - y) <= LG_XY_TOL_M
                            and abs(float(lg["guard_mm"]) - g) < 0.05 and abs(float(lg["hstop_m"]) - h) < 1e-6):
                        return ""
                except (KeyError, TypeError, ValueError):
                    pass
            time.sleep(0.05)
        return (f"reset 뒤 {LG_CHECK_WAIT_S:.0f} s 안에 노드 status 에서 요청한 가드({g:.0f} mm · {h} m · ({x:.4f}, {y:.4f}))가 "
                f"켜진 것을 못 봤다 — status lift_guard {lg}")

    def _guard_palm(self, status_key: str):
        with self._lock:
            st = dict(self._status.get(status_key) or {})
        palm = (st.get("lift_guard") or {}).get("palm")
        if palm is None or time.monotonic() - float(st.get("_t", 0.0)) > GUARD_HOLD_PALM_AGE_S:
            return None
        return np.asarray(palm, float)

    def _guard_hold(self, p, status_key: str = "") -> tuple[bool, str, dict]:
        """★11.08 가드 · 높이 정지 뒤 고정 구간 — FP++ live 컵이 손 안(손바닥 FK 와 GUARD_HOLD_PALM_M 안)에서 상승
        GUARD_HOLD_RISE_M 이상인 채 GUARD_HOLD_S 이어지면 성공. 같은 값 반복 프레임은 세지 않고, 상승이 GUARD_HOLD_RISE_MAX_M
        넘는 프레임은 튐으로 버린다. 값이 바뀐 프레임 사이 GUARD_HOLD_GAP_S 넘게 끊기면 처음부터."""
        name = p.target_object
        z0 = None if p.object_pose is None else float(p.object_pose[0][2])
        if not name or name not in self._live or z0 is None:
            return False, f"FP++ live {name!r} 또는 reset 컵 z 없음", {"guard_hold_rise_min_m": None, "guard_hold_s": 0.0}
        t_stop = time.monotonic()
        seen: set = set()
        prev = run0 = last = None
        best, rise_min, n_rep, n_out, n_far = 0.0, None, 0, 0, 0
        rises: list = []
        while time.monotonic() - t_stop < GUARD_HOLD_WAIT_S and not self.cancel.is_set():
            with self._lock:
                fr = [f for f in self._live[name] if f[2] >= t_stop and f[2] not in seen]
            palm = self._guard_palm(status_key)
            for f in sorted(fr, key=lambda f: f[2]):
                seen.add(f[2])
                pos = tuple(float(v) for v in f[0])
                if pos == prev:                       # FP++ 가 같은 값을 다시 보냈다 — 새 관측이 아니다
                    n_rep += 1
                    continue
                prev = pos
                rise = pos[2] - z0
                if not np.isfinite(rise) or rise > GUARD_HOLD_RISE_MAX_M:
                    n_out += 1
                    continue
                rises.append(rise)
                near = palm is not None and float(np.linalg.norm(np.asarray(pos) - palm[:3])) < GUARD_HOLD_PALM_M
                n_far += not near
                good = rise >= GUARD_HOLD_RISE_M and near
                if not good or (last is not None and f[2] - last > GUARD_HOLD_GAP_S):
                    run0, rise_min = (f[2], rise) if good else (None, None)
                elif run0 is None:
                    run0, rise_min = f[2], rise
                else:
                    rise_min = min(rise_min, rise)
                last = f[2]
                if run0 is not None:
                    best = max(best, f[2] - run0)
                    if best >= GUARD_HOLD_S:
                        return True, "", {"guard_hold_rise_min_m": round(rise_min, 4), "guard_hold_s": round(best, 2)}
            time.sleep(0.05)
        if self.cancel.is_set():                      # ★11.08 리뷰 3 — 정지로 끊겼다(판정 실패와 구분)
            return False, "episode runner stop — 판정 중 정지", {"guard_hold_rise_min_m": None if not rises else round(min(rises), 4),
                                                              "guard_hold_s": round(best, 2), "guard_hold_cancelled": True}
        why = (f"FP++ 새 프레임 {len(rises)}개(반복 {n_rep} · 튐 {n_out} · 손바닥 {GUARD_HOLD_PALM_M * 100:.0f} cm 밖/모름 {n_far}) · "
               f"상승 ≥ {GUARD_HOLD_RISE_M * 100:.0f} cm 최장 {best:.2f} s < {GUARD_HOLD_S} s"
               + (f" · 상승 {min(rises) * 100:.1f}~{max(rises) * 100:.1f} cm" if rises else ""))
        return False, why, {"guard_hold_rise_min_m": None if not rises else round(min(rises), 4), "guard_hold_s": round(best, 2)}

    def _wait_node_ready(self, status_key: str, timeout: float = 5.0) -> None:
        """정책 노드 status 가 ok(측정 · 컵이 다 들어옴)일 때까지 잠깐 기다린다 — 안 되면 그대로 reset(거부 사유가 판정한다)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.cancel.is_set():
            with self._lock:
                st = dict(self._status.get(status_key) or {})
            if st.get("ok") and time.monotonic() - float(st.get("_t", 0)) < 1.0:
                return
            time.sleep(0.1)

    def _wait_pd_tracking(self, side: str, timeout: float = 5.0) -> bool:
        """정책이 끝난 직후 pd 는 마지막 목표가 0.25 s 넘게 늙어 잠깐 watchdog HOLD 로 갔다가 정지 이벤트의 붙들기로 풀린다
        (10.04 fake 연속 실행). 손 · 궤적 명령 전에 TRACKING 으로 돌아오기를 기다린다 — 안 돌아오면 False(진짜 HOLD)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.cancel.is_set():
            with self._lock:
                st = dict(self._pd.get(side) or {})
            if st.get("phase") == "TRACKING" and not st.get("estop") and time.monotonic() - float(st.get("_t", 0)) < 1.0:
                return True
            time.sleep(0.1)
        return False

    def _pd_view(self, side: str) -> dict:
        with self._lock:
            st = dict(self._pd.get(side) or {})
        return {"pd_phase": st.get("phase"), "pd_reasons": st.get("reasons")}

    def _signals(self, p, st: Mapping) -> dict:
        return {**self._signals_of(p, st), **self._pd_view(p.side)}

    def _signals_of(self, p, st: Mapping) -> dict:
        if p.kind == "aglt":
            cup = (st.get("cup") or {}).get("arm") or {}
            # ★10.14 붙이기를 끈 노드(cup_attach switch False · 위치제어)는 '안 붙음'이 쥠 실패가 아니다 — 모름(None)
            return {"attached": cup.get("source") == "attached" if cup and cup.get("switch", True) else None,
                    "dropped": bool(cup.get("releases", 0)), "setting_err_m": st.get("kp_dist"), "reached": st.get("reached"),
                    "live_off_m": cup.get("check_max_m")}         # ★10.13 붙은 컵 ↔ FP++ 실측 최대 거리(episode_failure)
        if p.kind == "place":
            pl = st.get("place") or {}
            # ★11.10 학습 시간 끝(outcome episode_end)으로 멈춘 놓기는 손이 비었는지 노드가 모른다(래치 없이 정책이 연 것) — None,
            #   판정은 아래 cup_in_holder 만으로.
            empty = True if pl.get("phase") == "done" else (None if pl.get("outcome") == "episode_end" else False)
            sig = {"hand_empty": empty, "opened": pl.get("opened"), "tact_max": pl.get("tact_max"), "jf_max": pl.get("jf_max"),
                   "phase": pl.get("phase"), "ready_carry": pl.get("ready_carry"),      # ★10.25 이송 결과(없으면 None)
                   "outcome": pl.get("outcome")}       # ★11.02 빼기 결과 clear | withdraw_timeout(episode_failure 가 본다)
            hold = self.holders.get(p.holder_id) if p.holder_id is not None else None
            # ★11.04 놓은 자리 = FP++ live(stop + 1.0 s 뒤 프레임) 와 손을 연 순간의 묶은 컵(손바닥 FK ⊗ 묶음, latch_cup).
            #   …/pose(cups 때 찍은 테이블 자리)는 쓰지 않는다. live 는 쥔 컵을 놓치면 엉뚱한 값을 내므로(131817) 둘 중 하나라도
            #   홀더 안이면 들어간 것, 둘 다 밖일 때만 실패. 둘 다 없으면 판정 보류(None).
            pos = self._fresh_object(p.source_object, after=time.monotonic() + PLACE_JUDGE_WAIT_S, store=self._live)
            fk = pl.get("latch_cup")
            if hold is not None:
                votes = {}
                if pos is not None:
                    votes["live"] = cup_in_holder(pos, hold[0], SEAT_DZ_CYL60)
                    sig["cup_pos"] = [round(v, 4) for v in pos]
                if fk is not None and len(fk) == 3:
                    votes["fk"] = cup_in_holder(fk, hold[0], SEAT_DZ_CYL60)
                    sig["cup_pos_fk"] = [round(float(v), 4) for v in fk]
                if votes:
                    sig["cup_in_holder"] = any(votes.values())
                    sig["cup_in_holder_by"] = votes
            return sig
        return {}

    def _fresh_object(self, name: str | None, wait_s: float = 2.0, store: dict | None = None, after: float | None = None):
        """놓은 뒤 물체 자리 — 그 물체의 FP++ 토픽 새 프레임(창 안 중앙값). 없으면 None(판정 보류).
        after: 이 monotonic 시각 이후 수신한 프레임만 쓴다(놓기 판정 — 지연된 놓기 전 영상 배제). 최대 대기 wait_s 는 지금부터."""
        store = self._frames if store is None else store
        if not name or name not in store:
            return None
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            with self._lock:
                fr = [f for f in store[name] if time.monotonic() - f[2] < 0.5 and (after is None or f[2] >= after)]
            if len(fr) >= 3:
                return tuple(float(v) for v in np.median(np.array([f[0] for f in fr]), axis=0))
            time.sleep(0.1)
        return None

    def _live_object(self, name: str | None, wait_s: float = 2.0):
        """lift 판정용 물체 자리 — FP++ live 추적(…/live_pose) 새 프레임 중앙값. 없으면 None(판정 보류)."""
        return self._fresh_object(name, wait_s, self._live)

    def run_trajectory(self, node: Node, traj, world) -> NodeResult:
        if traj.kind == "noop":
            return NodeResult("completed", "noop")
        if traj.kind in PLANNED_TRAJ:
            return self._run_planned(node, traj, world)
        t0 = time.monotonic()
        skipped = []
        for side in traj.sides:
            if not self._wait_pd_tracking(side):
                pd = self._pd_view(side)
                return NodeResult("error", f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                  signals={"pd_fault": pd["pd_phase"] or "unknown"})
            if world.hand(side) != EMPTY:
                return NodeResult("refused", f"{side} hand holds {world.hand(side)} — rehome 계획기가 든 물체를 모른다")
            with self._lock:
                fr = list(self._arm)
            home, err = at_home(fr, side, home_arm(self.robot, side))
            if home:                                       # ★11.04 이미 홈 · 정지 — 움직이지 않는다
                self._log_line(f"[rehome] {side} 이미 홈(관절 최대 차 {err:.4f} rad ≤ {HOME_SKIP_RAD}) — 계획 · 재생 건너뜀")
                skipped.append(side)
                continue
            out = self.run_dir / f"rehome_{side}_{node.id}.npz"
            joints = ",".join(f"{side[0]}_aj_{i}" for i in range(1, 8))
            with self._lock:                               # ★11.11 탁자 위 컵도 계획 세계에
                relay, live = dict(self._relay), {k: list(v) for k, v in self._live.items()}
            try:
                cups, notes = rehome_table_cups(self.ep, world, relay, live)
            except RehomeCupUnknown as e:                  # ★11.12 리뷰 H1 — 움직이지 않았다
                self._log_line(f"[rehome] {side} 거부: {e}")
                return NodeResult("refused", f"{side} rehome: {e}", time.monotonic() - t0)
            self._log_line(f"[rehome] {side} 탁자 위 컵: " + ("; ".join(notes) or "없음"))
            # 리뷰 M2: 홀더 원기둥 = 이 에피소드가 발행한 홀더 파일(없으면 홀더 없이 — 기본 파일로 대신하지 않는다)
            hold = (["--holders", str(_paths.SIM2REAL / self.ep.holder_poses)] if self.ep.holder_poses else ["--no-holders"])
            steps = [
                [sys.executable, str(TOOLS / "plan_rehome.py"), "--side", side, "--robot", self.robot, "--out", str(out),
                 *hold, *(["--table-cups", json.dumps(cups)] if cups else [])],
                [sys.executable, str(TOOLS / "check_path_start.py"), "--npz", str(out), "--contract", self.contract],
                [sys.executable, str(TOOLS / "replay_to_pd.py"), "--npz", str(out), "--joints", joints, "--rate-scale", "1.0",
                 "--execute"],
                [sys.executable, str(TOOLS / "episode_ctl.py"), "--side", side, "--only", "pd_goto_home",
                 "--service-timeout", "45", "--execute", "--approve", "pd_goto_home"],
            ]
            for k, argv in enumerate(steps):
                rc, tail = self._run(argv)
                if self.cancel.is_set():
                    return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
                if rc != 0:                                # 계획 · 시작점 검사 실패 = 안전한 경로 없음(움직이지 않았다)
                    status = "refused" if k < 2 else "error"
                    return NodeResult(status, f"{'no safe path — ' if k < 2 else ''}{Path(argv[1]).name} rc {rc}: {tail}",
                                      time.monotonic() - t0)
        # ★11.04 사용자: 빈손 · 홈에서 손끝 촉각 bias 를 잡아 기록(tools.log + logs/runs/<YYMMDD>/rh56f1_tip_zero). 거부돼도 노드는 계속.
        #   두 팔을 한 번에(각자 프로세스) — 차례로 부르면 파이썬 기동만 팔마다 1~2 s
        self._run_many([[sys.executable, str(TOOLS / "trigger.py"), "hand/tip_zero", "--side", side, "--service-timeout", "5",
                         "--execute"] for side in traj.sides])
        return NodeResult("completed", f"이미 홈: {','.join(skipped)}" if skipped else "", time.monotonic() - t0,
                          signals={"home_skipped": skipped} if skipped else {})

    def _run_planned(self, node: Node, traj, world) -> NodeResult:
        self._tin_note = []
        try:
            res = RosExecutor._run_planned_inner(self, node, traj, world)
            if self._tin_note:                       # ★11.07 T_in 출처(measured | fallback)를 에피소드 로그(exit reason)에
                res = dataclasses.replace(res, reason="; ".join([*([res.reason] if res.reason else []), *self._tin_note]))
            return res
        finally:
            if self._tin_rec is not None:
                with self._lock:
                    self._tin_rec = None              # ★11.07 T_in 기록은 이 tilt 노드 안에서만

    def _run_planned_inner(self, node: Node, traj, world) -> NodeResult:
        """10.17 carry(ready_carry → 붓기 시작 자세, 한 팔씩 또는 두 팔) · tilt(D4 기본 궤적 + 복귀, 두 팔): 실측에서 계획 · 검사
        (움직이지 않음) → 그 팔 관절을 한 시각표로 재생. 계획 · 검사 실패(rc ≠ 0)면 움직이지 않고 refused. 손 목표는 그대로
        (팔 관절만 보낸다). cups = 그 팔 손이 컵을 쥐었을 때만(False = 빈손일 때만 — 경로 B 는 컵 없이 시작 자세로 간다)."""
        t0 = time.monotonic()
        sides = [s for s in SIDES if s in traj.sides]
        for side in sides:
            if not self._wait_pd_tracking(side):
                pd = self._pd_view(side)
                return NodeResult("error", f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                  signals={"pd_fault": pd["pd_phase"] or "unknown"})
            held = world.hand(side)
            if traj.cups and held == EMPTY:
                return NodeResult("refused", f"{traj.kind}: {side} 손이 비었다 — 컵을 쥔 상태에서만(손 안 컵 자리 기준)")
            if not traj.cups and held != EMPTY:
                return NodeResult("refused", f"{traj.kind}(빈손): {side} 손에 {held} — 빈손 이송은 손이 빈 때만")
        tin_objs = ({s: world.hand(s) for s in sides if world.hand(s) != EMPTY}
                    if traj.kind == "tilt" and self.ep.tin_fallback else {})   # ★11.07 tilt 직전 손 안 컵 T_in 을 잴 팔 → 쥔 물체
        if tin_objs:
            for s in tin_objs:
                self._tin_json.pop(s, None)
                self._tin_why[s] = f"[{node.id}] tilt 가 T_in 측정 전에 끝났다(정지 대기 · 계획 · 검사 실패나 정지)"
            with self._lock:
                self._tin_rec = {"live": {}, "arm": []}
        still = self._wait_arm_still(sides)          # ★10.30 실측에서 계획하기 전에 팔이 멈추기를 기다린다
        tin_t0 = time.monotonic() - (ARM_STILL_WINDOW_S if still.get("arm_still") else 0.0)   # 정지 창 시작
        if self.cancel.is_set():
            return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
        if still.get("arm_still") is False:
            self.node.get_logger().warning(f"[{node.id}] 팔이 {ARM_STILL_WAIT_S:.0f} s 안에 안 멈췄다 {still} — 그대로 계획")
        asset = TRAJ_ASSETS / traj.asset
        cfg = yaml.safe_load((asset / "policy.yaml").read_text(encoding="utf-8")) or {}
        out = self.run_dir / f"{traj.kind}_{node.id}"
        joints = ",".join(f"{s[0]}_aj_{i}" for s in sides for i in range(1, 8))
        side_arg = ["--sides", ",".join(sides)] if traj.kind == "carry" else []
        before = None
        if traj.kind == "reach":                      # ★10.19 1-B: 지금 FP++ 컵 자세(창 안 중앙값, 멈춰 있어야)로 계획
            snap = self._still_object(traj.object)
            if snap is None:
                return NodeResult("refused", f"reach: {traj.object} 컵 자세가 없거나 움직인다({SNAPSHOT_WINDOW_S} s 창, "
                                             f"{SNAPSHOT_STILL_M * 1000:.0f} mm) — FP++ 확인", time.monotonic() - t0)
            real = real_object_name(self.ep.objects[traj.object].get("topic", ""))
            if not real:
                return NodeResult("refused", f"reach: {traj.object} 토픽에서 실물 이름을 못 읽었다", time.monotonic() - t0)
            side_arg = ["--side", sides[0], "--object", real, "--cup", ",".join(f"{v:.5f}" for v in snap[0]),
                        "--quat", ",".join(f"{v:.6f}" for v in snap[1]), "--stop", traj.stop]   # ★10.20 pregrasp = 경로 D
        elif traj.kind == "tilt" and cfg.get("kind") == "tilt_traj":   # ★10.27 traj_v10: 컵별 궤적 · 실측 홀더로 여유 검사
            side_arg = ["--cup", traj.cup]
            if self.ep.holder_poses:
                side_arg += ["--holders", str(_paths.SIM2REAL / self.ep.holder_poses)]
        elif traj.kind == "lift":
            side_arg = ["--side", sides[0]]
            if traj.object:
                before = self._live_object(traj.object)
        steps = [
            [sys.executable, str(TOOLS / ("plan_tilt_traj.py" if cfg.get("kind") == "tilt_traj" else f"plan_{traj.kind}.py")), "--asset", str(asset), "--out", str(out), *side_arg],
            [sys.executable, str(TOOLS / "replay_to_pd.py"), "--npz", str(out / f"{traj.kind}_cmd.npz"), "--joints", joints,
             "--rate-scale", repr(float(traj.rate_scale)), "--max-start-gap", repr(float(cfg.get("start_gap_rad", 0.05))),
             "--execute"],
        ]
        tin_plan = traj.kind == "tilt" and cfg.get("kind") == "tilt_traj"
        # T_in 이 없으면 멈출 팔: tilt_traj 계획은 잰 팔마다 이 런 T_in 을 쓴다(_planner_tin) · 그 밖은 place tin_json 팔만
        need = (set(tin_objs) if tin_plan else {b.side for b in self.ep.policies.values() if b.attach_source == "tin_json"}) if tin_objs else set()
        for k, argv in enumerate(steps):
            if k == (0 if tin_plan else 1) and tin_objs:   # ★11.09 tilt_traj 는 계획 전에 잰다(계획이 이 런의 쥔 자리를 쓴다) · 그 밖은 재생 직전
                why = self._measure_tin(node, tin_objs, tin_t0, still, required=need)
                if self.cancel.is_set():
                    self._drop_tin(node, set(tin_objs), "정지로 tilt 가 돌지 않았다")
                    return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
                if why:
                    return NodeResult("refused", why, time.monotonic() - t0)
                if tin_plan:
                    why = self._planner_tin(node, tin_objs, asset / "ref" / f"setup_{traj.cup}.npz", argv)
                    if why:
                        self._drop_tin(node, set(tin_objs), "tilt 계획 실패로 tilt 가 돌지 않았다")
                        return NodeResult("refused", why, time.monotonic() - t0)
            rc, tail = self._run(argv)
            if self.cancel.is_set():
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
            if rc != 0:                                # 계획 · 검사 실패 = 움직이지 않았다
                if tin_plan and tin_objs and k == 0:   # 이번에 잰 T_in 은 tilt 가 안 돌았으니 place 가 쓰지 않게
                    self._drop_tin(node, set(tin_objs), f"{Path(argv[1]).name} rc {rc} 로 tilt 가 돌지 않았다")
                status = "refused" if k == 0 else "error"
                return NodeResult(status, f"{Path(argv[1]).name} rc {rc}: {tail}", time.monotonic() - t0)
        if traj.kind == "lift" and traj.object:       # ★10.19 1-B: 컵이 손과 같이 올라갔는가(유지 끝 FP++ — 없으면 판정 보류)
            after = self._live_object(traj.object)
            if before is None or after is None:
                return NodeResult("completed", f"들기 끝 · FP++ 컵 {'전' if before is None else '뒤'} 프레임 없음 — 들림 판정 보류",
                                  time.monotonic() - t0, signals={"cup_rise_m": None})
            rise = float(after[2] - before[2])
            sig = {"cup_rise_m": round(rise, 4), "cup_before": [round(v, 4) for v in before], "cup_after": [round(v, 4) for v in after]}
            if rise < LIFT_RISE_MIN_M:
                return NodeResult("aborted", f"컵이 {rise * 100:.1f} cm 만 올라갔다(< {LIFT_RISE_MIN_M * 100:.0f} cm) — 못 쥐었거나 놓쳤다",
                                  time.monotonic() - t0, signals=sig)
            return NodeResult("completed", f"컵 {rise * 100:.1f} cm 들림 · 유지 끝", time.monotonic() - t0, signals=sig)
        return NodeResult("completed", "", time.monotonic() - t0)

    def _still_object(self, name: str, wait_s: float = 3.0):
        """reach 계획용 컵 자세 — 창(SNAPSHOT_WINDOW_S) 안 프레임이 멈춰 있으면 (pos 중앙값, 마지막 quat wxyz), 아니면 None."""
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not self.cancel.is_set():
            with self._lock:
                fr = [f for f in self._frames.get(name, []) if time.monotonic() - f[2] < SNAPSHOT_WINDOW_S]
            snap = snapshot_of(fr)
            if snap is not None:
                return snap
            time.sleep(0.1)
        return None

    def run_hand(self, node: Node, world) -> NodeResult:
        """★10.17 경로 B: open = 손을 계약 홈 손(편 손)으로(pd hand_release, 팔은 pd 가 그 자리에 붙든 채) — 빈손에서만.
        grasp = 한 손을 닿을 때까지 닫고 쥔 판정 · 3 s 유지(tools/hand_grasp.py, 팔 제자리). 못 쥐면 aborted(not grasped)."""
        t0 = time.monotonic()
        for side in node.sides:
            if not self._wait_pd_tracking(side):
                pd = self._pd_view(side)
                return NodeResult("error", f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                  signals={"pd_fault": pd["pd_phase"] or "unknown"})
            if world.hand(side) != EMPTY:
                return NodeResult("refused", f"hand {node.action}: {side} 손에 {world.hand(side)} — 빈손에서만(펴면 떨어진다)")
        if node.action == "open":
            for side in node.sides:
                rc, tail = self._run([sys.executable, str(TOOLS / "trigger.py"), "pd/hand_release", "--side", side, "--execute"])
                if self.cancel.is_set():
                    return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
                if rc != 0:
                    pd = self._pd_view(side)
                    return NodeResult("error", f"hand_release {side} 거부 · pd {pd['pd_phase']} {pd['pd_reasons']}: {tail}",
                                      time.monotonic() - t0)
            deadline = time.monotonic() + HAND_OPEN_WAIT_S        # 리뷰: 대기만 하지 않고 실측이 펴졌는지 본다
            errs: dict = {}
            while time.monotonic() < deadline and not self.cancel.is_set():
                errs = {side: self._hand_open_err(side) for side in node.sides}
                if all(e is not None and e <= HAND_OPEN_TOL_RAD for e in errs.values()):
                    return NodeResult("completed", f"손 펴짐 {errs}", time.monotonic() - t0)
                time.sleep(0.1)
            if self.cancel.is_set():
                return NodeResult("aborted", "episode runner stop", time.monotonic() - t0)
            return NodeResult("error", f"손이 {HAND_OPEN_WAIT_S:.0f} s 안에 안 펴졌다 — max|q| {errs}(허용 {HAND_OPEN_TOL_RAD} rad, "
                                       f"None = 손 실측 없음)", time.monotonic() - t0)
        side = node.sides[0]
        contract = hand_contract_file(node.hand_contract, _paths.SIM2REAL / "deploy" / "policies")
        if not contract.is_file():
            return NodeResult("refused", f"hand grasp: {contract} 가 없다(hand_contract {node.hand_contract})")
        rc, tail = self._run([sys.executable, str(TOOLS / "hand_grasp.py"), "--side", side, "--contract", str(contract),
                              "--cfg", json.dumps(dict(node.cfg)), "--out", str(self.run_dir / f"hand_{node.id}"), "--execute"])
        dur = time.monotonic() - t0
        if self.cancel.is_set():
            return NodeResult("aborted", "episode runner stop", dur)
        if rc == 0:
            return NodeResult("completed", tail, dur)
        if rc == 2:                                    # 못 쥠 — 들지 않았다(손은 그 자리, pd 가 붙든다)
            return NodeResult("aborted", f"hand_grasp rc 2: {tail}", dur)
        if rc == 3:
            return NodeResult("error", f"hand_grasp rc 3: {tail}", dur, signals={"pd_fault": self._pd_view(side)["pd_phase"] or "pd"})
        return NodeResult("error", f"hand_grasp rc {rc}: {tail}", dur)

    def _log_line(self, text: str) -> None:
        with open(self.run_dir / "tools.log", "a") as fh:
            fh.write(f"\n{text}\n")

    def _run_many(self, argvs: list) -> list[int]:
        """기록용 명령 여럿을 동시에 — 출력은 명령마다 tools.log 에 차례로 붙인다(섞이지 않게)."""
        procs = [(argv, subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                         env={**os.environ, "PYTHONUNBUFFERED": "1"})) for argv in argvs]
        rcs = []
        with open(self.run_dir / "tools.log", "a") as fh:
            for argv, p in procs:
                try:
                    out, _ = p.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    out, _ = p.communicate()
                fh.write(f"\n$ {' '.join(argv)}\n{out}")
                rcs.append(p.returncode)
        return rcs

    def _planner_tin(self, node: Node, objs: Mapping, setup: Path, argv: list) -> str:
        """★11.09 문제 1 — 방금 잰 팔마다 T_in(런 json, measured | fallback)을 tilt 계획기 규약으로(tin_live.planner_tin) 런 폴더에
        쓰고 argv 에 --tin-json 을 붙인다. 반환 = 못 만든 사유(없으면 "") — 기본값으로 몰래 계획하지 않는다."""
        from policy_control import tin_live as TL
        try:
            runs = {s: json.loads(Path(p).read_text(encoding="utf-8")) for s, p in self._tin_json.items() if s in objs}
            T_by = {s: np.asarray(d["T_in"], float) for s, d in runs.items()}
            out = TL.planner_tin(T_by, setup, sources={s: str(d.get("source", "?")) for s, d in runs.items()}, node=node.id, episode=self.ep.name, tin_json=dict(self._tin_json))
        except (TL.TinError, OSError, KeyError, ValueError) as exc:
            msg = f"[{node.id}] tilt 계획용 T_in 을 못 만들었다({exc!r}) — 기본 T_in 으로 계획하지 않는다"
            self.node.get_logger().error(msg)
            return msg
        path = self.run_dir / f"tilt_tin_{node.id}.json"
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        argv += ["--tin-json", str(path)]
        self.node.get_logger().info(f"[{node.id}] tilt 계획 T_in = {out['sides']} → {path.name}")
        return ""

    def _measure_tin(self, node: Node, objs: Mapping, t0: float, still: Mapping, required=None) -> str:
        """★11.07 tilt 재생 직전 — 정지 창 [t0 + FP++ 지연, 지금](t0 + TL.WINDOW_S 까지 멈춘 채 기다린다)의 FP++ live 새 프레임 ×
        손바닥 FK → 팔마다 T_in json(런 폴더, measured). ★11.10 실측이 안 되면 그 컵을 FP++ 로 다시 잡고(fpp_rescan) 정지 자세에서
        창을 다시 연다(TIN_RETRY_N 회 · TIN_RETRY_S 안). 그래도 안 되면 실기(fpp_rescan 켬)는 고정값으로 가지 않는다
        (사유 = 회차마다 어긴 판정 · 재잡기 결과, 런 폴더 tin_<노드>_<팔>_failed.json). fake(재잡기 없음)만 옛 fallback 을 쓴다.
        required = T_in 이 없으면 tilt 를 멈출 팔(None = objs 전부). 그 밖의 팔은 사유만 남기고 진행한다.
        반환 = tilt 를 시작하지 않을 사유(없으면 ""). 정지(cancel)면 json 을 쓰지 않고 바로 돌아간다(호출부가 aborted)."""
        from policy_control import tin_live as TL
        required = set(objs) if required is None else set(required)
        left, tries, last, final, wrote = dict(objs), {s: [] for s in objs}, {}, set(), []
        deadline = time.monotonic() + TIN_RETRY_S
        for attempt in range(TIN_RETRY_N + 1 if self.fpp_rescan else 1):   # fake 는 다시 잡을 FP++ 가 없다
            rescan = None
            if attempt:
                redo = [s for s in left if s not in final]   # 기록을 못 만든 팔(final)은 다시 잡아도 같다
                if self.cancel.is_set() or not redo or time.monotonic() > deadline:
                    break
                try:
                    rc, tail = self._run(rescan_argv(self.ep, [left[s] for s in redo]))
                    rescan = f"FP++ 재잡기 rc {rc}" + (f": {tail}" if rc else "")
                except ValueError as exc:
                    rescan = f"FP++ 재잡기 못 함: {exc}"
                self.node.get_logger().warning(f"[{node.id}] T_in {sorted(redo)} {attempt}차 재시도 — {rescan}")
                if self.cancel.is_set():
                    break
                with self._lock:
                    self._tin_rec = {"live": {}, "arm": []}
                still = self._wait_arm_still(redo)
                if self.cancel.is_set():
                    break
                t0 = time.monotonic()
            wait = t0 + TL.WINDOW_S - time.monotonic()
            if (wait > 0 and self.cancel.wait(wait)) or self.cancel.is_set():
                break
            t1 = time.monotonic()
            with self._lock:
                rec = self._tin_rec or {"live": {}, "arm": []}
                self._tin_rec = None
            for side, obj in list(left.items()):
                if side in final:
                    continue
                names = [f"{side[0]}_aj_{i}" for i in range(1, 8)]
                arm = [(t, q) for t, q in rec["arm"] if all(n in q for n in names)]
                fr = rec["live"].get(obj, [])
                if still.get("arm_still") is not True:
                    meas = {"ok": False, "reason": f"팔이 멈추지 않아 측정 안 함 {dict(still)}"}
                else:
                    try:
                        meas = TL.measure(np.array([f[0] for f in fr]), np.array([f[1] for f in fr]),
                                          np.array([f[2] for f in fr]).reshape(-1, 7), np.array([a[0] for a in arm]),
                                          np.array([[a[1][n] for n in names] for a in arm]).reshape(-1, 7),
                                          TL.palm_fk(side), t0 + TL.FPP_LATENCY_S, t1, z_off=TL.Z_OFF[side],
                                          near_m=TL.NEAR_M_SIDE[side])
                    except Exception as exc:          # noqa: BLE001 — 측정 코드가 무엇을 던지든 실패 사유로(json 에)
                        meas = {"ok": False, "reason": f"측정 실패({exc!r})"}
                out = None
                if meas.get("ok"):                    # ★11.10 고정값 대조 거부 없음 — record 는 차이(vs_fixed)만 남긴다
                    try:
                        out = TL.record(meas, side, _paths.SIM2REAL / self.ep.tin_fallback, fallback_file=self.ep.tin_fallback,
                                        node=node.id, object=obj, episode=self.ep.name)
                    except (TL.TinError, OSError, ValueError) as exc:
                        meas = {**meas, "ok": False, "reason": f"T_in 기록 못 만듦: {exc}"}
                        final.add(side)               # 같은 값으로 다시 해도 같다 — 재잡기로 안 풀린다
                if not meas.get("ok"):
                    last[side] = meas
                    tries[side].append({"attempt": attempt, "reason": meas.get("reason", ""),
                                        **({"rescan": rescan} if rescan else {}),
                                        **{k: meas.get(k) for k in ("n_fresh", "n_msgs") if k in meas}})
                    self.node.get_logger().warning(f"[{node.id}] T_in {side}({obj}) {attempt + 1}회차 실측 실패 — {meas.get('reason', '')}")
                    continue
                left.pop(side)
                self._write_tin(node, side, out, meas, tries[side])
                wrote.append(side)
            if not left:
                return ""
        if self.cancel.is_set():                       # 정지 — 실측 실패가 아니다(failed.json · 거부 사유 없음)
            self._drop_tin(node, wrote, "정지로 tilt 가 돌지 않았다")
            return "episode runner stop"
        stop = []
        for side, obj in left.items():
            if self.fpp_rescan:
                msg = (f"[{node.id}] T_in {side}({obj}) 실측 {len(tries[side])}회 실패 — 고정값으로 진행하지 않는다: "
                       + " / ".join(f"{t['attempt'] + 1}회 {t['reason']}" + (f"({t['rescan']})" if t.get("rescan") else "")
                                    for t in tries[side]))
                path = self.run_dir / f"tin_{node.id}_{side}_failed.json"
                try:
                    path.write_text(json.dumps({"source": "failed", "side": side, "object": obj, "node": node.id, "episode": self.ep.name,
                                                "attempts": tries[side], "retry_n": TIN_RETRY_N, "retry_s": TIN_RETRY_S,
                                                "t": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1, ensure_ascii=False), encoding="utf-8")
                    self.node.get_logger().error(f"{msg} → {path.name}")
                except OSError as exc:
                    self.node.get_logger().error(f"{msg} ({path.name} 못 씀: {exc})")
                self._tin_why[side] = msg
            else:
                try:                                   # fake(fpp_rescan 끔)만: 옛 고정 fallback
                    out = TL.record(last.get(side, {"ok": False, "reason": "측정 없음"}), side, _paths.SIM2REAL / self.ep.tin_fallback,
                                    fallback_file=self.ep.tin_fallback, node=node.id, object=obj, episode=self.ep.name)
                except (TL.TinError, OSError, ValueError) as exc:
                    msg = f"[{node.id}] T_in {side} 측정 실패({last.get(side, {}).get('reason', '')}) · fallback {self.ep.tin_fallback} 못 읽음: {exc}"
                    self.node.get_logger().error(msg)
                    self._tin_why[side] = msg
                else:
                    self._write_tin(node, side, out, last.get(side, {}), tries[side])
                    wrote.append(side)
                    continue
            if side in required:
                stop.append(msg)
            else:                                      # tilt 계획도 place 도 이 팔 T_in 을 안 쓴다 — 사유만 남기고 진행
                self._tin_note.append(f"T_in {side} 없음")
        if stop:
            self._drop_tin(node, wrote, "다른 팔 T_in 실패로 tilt 가 돌지 않았다")
            return " · ".join(stop) + " — tilt 를 시작하지 않는다"
        return ""

    def _drop_tin(self, node: Node, sides, why: str) -> None:
        """tilt 가 안 돈 이번 노드의 T_in 은 place 가 쓰지 않게 잊는다(json 파일은 기록으로 남긴다)."""
        for side in sides:
            self._tin_json.pop(side, None)
            self._tin_why[side] = f"[{node.id}] {why}"
        self._tin_note = [n for n in self._tin_note if not any(n.startswith(f"T_in {s} ") for s in sides)]

    def _write_tin(self, node: Node, side: str, out: dict, meas: Mapping, tries: list) -> None:
        """런 폴더 tin_<노드>_<팔>.json(place tin_json · tilt 계획이 읽는다) + 로그 · 노드 결과 한 줄."""
        out = {**out, "failed_attempts": tries} if tries else out
        path = self.run_dir / f"tin_{node.id}_{out['side']}.json"
        try:                                           # 못 써도 경로는 넘긴다 — 계획 · place 가 파일 없음으로 멈춘다(기본값으로 몰래 가지 않게)
            path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self.node.get_logger().error(f"[{node.id}] T_in {side} json {path.name} 못 씀: {exc}")
        self._tin_json[side] = str(path)
        self._tin_why.pop(side, None)
        vs = out.get("vs_fixed") or {}               # ★11.10 고정값과 차이(기록만) — 노드 결과 reason(콘솔) · 로그에 같이 보인다
        vs_txt = (vs["error"] if "error" in vs else f"고정값과 {vs['pos_mm']} mm/{vs['ang_deg']}° (dx {vs['dx_mm']} · dy {vs['dy_mm']} · dz {vs['dz_mm']} mm)"
                  ) if vs else ""
        self._tin_note.append(f"T_in {side} measured" + (f"({len(tries) + 1}회차)" if tries else "") + (f" · {vs_txt}" if vs_txt else "")
                              if out["source"] == "measured" else f"T_in {side} fallback(실측 아님: {out.get('fallback_reason', '')})")
        line = (f"[{node.id}] T_in {side}({out.get('object')}) = {out['source']} pos {np.round(np.asarray(out['pos_m']) * 1000, 1).tolist()} mm · "
                f"갱신 {meas.get('n_fresh')}/{meas.get('n_msgs')} · 흔들림 {meas.get('pos_dev_mm_max')} mm/{meas.get('ang_dev_deg_max')}°"
                + (f" · {vs_txt}" if vs_txt else "") + f" → {path.name}")
        if out["source"] == "fallback":
            self.node.get_logger().warning(f"{line} — 실측 아님, 고정값 사용: {out.get('fallback_reason', '')}")
        else:
            self.node.get_logger().info(line)

    def _run(self, argv: list) -> tuple[int, str]:
        """도구 하나 — ★11.08 리뷰: 출력은 도구마다 tools/<번호>_<이름>.log(두 팔 줄이 동시에 띄워도 끝줄이 섞이지 않게),
        tools.log 에는 명령과 그 파일만. 대기 상한(TOOL_TIMEOUT_S)을 넘으면 끊고 rc 124 · 'tool timeout'(→ controller_error,
        안전 정지). 정지(cancel) 뒤에는 띄우지 않는다."""
        name = Path(argv[1]).name if len(argv) > 1 else str(argv[0])
        if self.cancel.is_set():
            return 130, "episode runner stop — 도구를 띄우기 전 정지"
        with self._lock:
            self._tool_seq += 1
            log = self.run_dir / "tools" / f"{self._tool_seq:03d}_{Path(name).stem}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        limit = _replay_limit(argv) if name == "replay_to_pd.py" else TOOL_TIMEOUT_S.get(name, TOOL_TIMEOUT_DEFAULT_S)
        self._log_line(f"$ {' '.join(argv)}\n  → {log.relative_to(self.run_dir)} (상한 {limit:.0f} s)")
        timed_out = False
        with open(log, "w") as fh:
            fh.write(f"$ {' '.join(argv)}\n")
            fh.flush()
            proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            with self._lock:
                self._procs.add(proc)
            if self.cancel.is_set():                   # safe_stop 이 add 바로 앞에서 지나갔다 — 이 도구도 끊는다
                _killpg(proc)
            try:
                try:
                    rc = proc.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _killpg(proc)
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        _killpg(proc, signal.SIGKILL)
                        proc.wait()
                    rc = 124
            finally:
                with self._lock:
                    self._procs.discard(proc)
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
        if timed_out:
            return rc, f"tool timeout {name} {limit:.0f} s 초과 — 끊음 · {tail[0][-200:]}"
        return rc, tail[0][-300:]

    def prepare(self, step: str, plans: list, world) -> NodeResult:
        if step == "open_hand":
            for side in sorted({p.side for p in plans if p.side in SIDES}):
                if not self._wait_pd_tracking(side):
                    pd = self._pd_view(side)
                    return NodeResult("error", f"pd_{side} 가 TRACKING 으로 안 돌아온다 · {pd['pd_phase']} {pd['pd_reasons']}",
                                      signals={"pd_fault": pd["pd_phase"] or "unknown"})
                rc, tail = self._run([sys.executable, str(TOOLS / "trigger.py"), "pd/hand_release", "--side", side, "--execute"])
                if rc != 0:
                    pd = self._pd_view(side)
                    return NodeResult("error", f"hand_release {side} 거부 · pd {pd['pd_phase']} {pd['pd_reasons']}: {tail}",
                                      signals={"pd_fault": pd["pd_phase"]} if pd["pd_phase"] == "HOLD" else {})
            time.sleep(1.0)
            return NodeResult("completed", "")
        if step == "refresh_objects":
            names = [p.target_object or p.source_object for p in plans if (p.target_object or p.source_object)]
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self._lock:
                    fresh = all(n in self._relay or any(time.monotonic() - f[2] < 0.5 for f in self._frames.get(n, []))
                                for n in names)
                if fresh:
                    time.sleep(1.0)                        # 정책 노드가 받을 시간
                    return NodeResult("completed", "")
                time.sleep(0.1)
            return NodeResult("refused", f"cup pose missing — {names} 프레임이 5 s 동안 없다")
        if step == "refresh_holders":
            missing = self.publish_holders()
            if missing:
                self.holders = load_holder_poses(_paths.SIM2REAL / self.ep.holder_poses)
                missing = self.publish_holders()
            if missing:
                return NodeResult("refused", f"holder pose missing: {missing} ({self.ep.holder_poses})")
            time.sleep(1.0)
            return NodeResult("completed", "")
        return NodeResult("refused", f"모르는 준비 {step}")

    def safe_stop(self, reason: str) -> None:
        self.cancel.set()
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                _killpg(proc)
        for side in self.sides:                       # ★10.19 이 에피소드의 팔만 — 다른 팔 실행기의 정책은 그대로
            for kind in NODES:
                self._call(f"{NS}/{service_ns(kind, side)}/episode/stop", timeout=2.0, wait=0.3)

    def joints(self) -> dict:
        return {}

    def clear_cancel(self) -> None:
        self.cancel.clear()
        self._tin_json, self._tin_why = {}, {}       # ★11.07 새 에피소드 — 실행기는 에피소드를 넘어 살아 있다, 앞 T_in 을 쓰지 않는다
