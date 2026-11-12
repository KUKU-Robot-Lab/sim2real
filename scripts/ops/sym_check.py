#!/usr/bin/env python3
"""양팔 비대칭(엔코더 영점 의심) 확인 — 단계형 CLI. 홈 · 탁자 대칭 자세에서 관절값 · 목 틱 · 목 카메라 RGB-D 를 남긴다.

sim 홈은 좌우 거울(오른쪽 = 왼쪽 부호 반대, j4 만 같음)이다. 실기에서 관절값은 거울인데 눈으로 비대칭이면 영점이 의심된다.
사용자가 단계마다 정면 사진을 찍고, 이 스크립트는 관절값(1 s 평균 · 표준편차)과 카메라 color · depth 를 저장한다
(나중에 FK 와 point cloud ICP 로 비교).

단계(Enter 로 진행, q = 중단 → 팔 홈 복귀 → 목 복귀 → disable):
  0 사전 점검(CAN · 목 · 카메라를 쓰는 노드가 떠 있으면 PID 를 보여 주고 끝냄, kill 하지 않음) · 카메라 열기 · 팔 읽기(enable 없음)
  1 양팔 enable → 지금 자세에서 홈 → snap home
  2 목 정면 · 아래 보기 → snap home_head0                         (--skip-head 면 건너뜀)
  3 홈 → 경유점 → 탁자 대칭 자세 → snap table
  4 탁자 → 경유점 → 홈 → snap home_back → 목 시작 틱으로 → 양팔 disable

제어는 logs/runs/261111/left_zero/j3_release_v2.py 를 양팔로 넓힌 것이다(CAN 직결 MIT, 같은 KP · KD · 중력 보상 · ERR_MAX · VEL_MAX 검사).
사진을 기다리는 동안에도 같은 검사를 하며 자세를 붙든다.

인터프리터: openarm_can 은 motion_acq venv 에만 있다. 목·카메라는 기본 arm4090-lan 의 sim2real/.venv python 도우미(SSH JSON)로 연다. --head-host/--cam-host local 로 로컬 선택.
  ~/rl_ws/motion_acq/.venv/bin/python scripts/ops/sym_check.py --dry          # 로봇 · 목 · 카메라를 열지 않음
  ~/rl_ws/motion_acq/.venv/bin/python scripts/ops/sym_check.py                # 실기(사용자 승인 뒤에만)
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import base64
import os
import shlex
import threading
import json
import re
import select
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

S2R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2R / "logs/runs/261110/j3_zero/gravity"))  # fk.py: j3_release_v2 와 같은 URDF FK · 중력 모델
sys.path.insert(0, str(S2R / "scripts"))                            # head_home(목 설정 · 순서 규약)
import fk  # noqa: E402

CAM_PY = S2R / ".venv/bin/python"                 # --cam-host local 인터프리터
CAM_SERIAL = "348122071637"                       # 머리 D435i rs_serial(deploy/policy_control/config/sites/arm_v2.yaml camera_rs_serial)
HEAD_CFG = S2R / "config/head_home_rh56f1.yaml"   # pan 2033 · tilt 2864 틱 = 5090 홈 화면에 맞춘 캘리브 자세(아래 HEAD_FRONT)

# ---- 자세 ----
# deploy/policies/rh56f1/aglt/left_rcg04/rh_aglt_contract.json sides.arm.arm_home (right_rcg04 = 거울)
HOME_L = np.array([1.2127, -0.2026, -0.6538, 1.7608, -0.3791, -0.5785, -0.6646])
MIRROR = np.array([-1, -1, -1, 1, -1, -1, -1.0])  # 오른쪽 = 왼쪽 거울(j4 만 같은 부호), 계약 두 파일과 같다
# 탁자 대칭 자세(왼팔). 11.12 FK 탐색(벌점 Nelder-Mead, 아래 table_report 가 매번 다시 검사): 손바닥 x 0.347 · y +0.146/−0.149 ·
#   탁자 위 0.139 m · 법선 −z 와 18°, j4 51.8°(홈 100.9°). 손끼리 8 cm 조건이 묶여 y 는 0.15 근처까지만 좁아진다(팔뚝끼리 0.083 m)
TABLE_L = np.array([-0.4815, -0.3215, 1.5208, 0.9048, 1.0233, 0.5326, 1.2951])
VIA_L = np.array([-0.8827, -0.8169, -0.0019, 1.5762, 0.0121, -0.0335, -0.1151])   # 홈 → 탁자 경유점(왼팔), plan_mid 참고

# ---- 환경(근사) ----
TABLE_Z = 0.205        # 탁자 상판 z(base 프레임). config/cup_holders.yaml "바닥이 상판(z 0.205) 위" · cup_holder_poses_5090_v1 z 0.235 = 0.205 + 0.030
TABLE_X_NEAR = 0.15    # 이 x 보다 앞의 점만 상판 검사. 실기 상판 앞(몸 쪽) 끝은 기록 없음 — 확인 필요
TABLE_MARGIN = 0.03
# 몸통 = body_link 충돌 메쉬(body_link0_symp_cut_nohousing_top730.stl) AABB 를 둘로: 기둥(6 x 6 cm) · 바닥판
BODY_BOXES = [(np.array([-0.03, -0.03, 0.05]), np.array([0.03, 0.03, 0.73])),
              (np.array([-0.153, -0.085, 0.0]), np.array([0.03, 0.085, 0.05]))]
ARM_R, HAND_R = 0.045, 0.02   # 링크 중심선 점의 반지름 근사 — 거리는 '점 − 반지름'(표면 근사)
BODY_MARGIN, HAND_GAP = 0.05, 0.08
PALM_DOWN_MIN = np.cos(np.deg2rad(20))   # 손바닥 법선(palm_sensor +z, 손가락이 굽는 쪽)과 −z 의 cos 하한

# ---- 제어(j3_release_v2 그대로) ----
TYPES_NAMES = ["DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310"]
IDS, MIDS = [1, 2, 3, 4, 5, 6, 7], [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
KP = np.array([70, 70, 70, 60, 20, 20, 20.0])
KD = np.array([2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5])
FEEDBACK_MAX_AGE = 0.050  # s, per-motor valid state reception deadline
DT = 0.01
ERR_MAX, VEL_MAX = np.array([0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.08]), 2.0
CAN = {"right": "can0", "left": "can1"}
SIDES = ("left", "right")
SPEED_MAX = 0.5        # rad/s, --speed 상한
START_TOL = np.deg2rad(2)   # 시작 자세가 관절 한계 밖으로 이만큼까지는 읽기 오차로 본다

M = fk.Model()


def mirror(q):
    return np.asarray(q, float) * MIRROR


def pair(q_left):
    """왼팔 값 → (왼, 오른) 거울 쌍."""
    return np.asarray(q_left, float), mirror(q_left)


def limits():
    out = {}
    for j in ET.parse(fk.URDF).getroot().iter("joint"):
        n, lim = j.get("name", ""), j.find("limit")
        if re.fullmatch(r"[lr]_aj_[1-7]", n) and lim is not None:
            out[n] = (float(lim.get("lower")), float(lim.get("upper")))
    return {s: np.array([out[f"{s[0]}_aj_{i}"] for i in range(1, 8)]) for s in SIDES}


LIM = limits()


def in_limits(side, q, tol=0.0):
    lo, hi = LIM[side][:, 0] - tol, LIM[side][:, 1] + tol
    return bool(np.all((q >= lo) & (q <= hi)))


def fk_both(qL, qR):
    T, _ = M.fk({**fk.arm_q("left", list(qL)), **fk.arm_q("right", list(qR))})
    return T


def points(T, s):
    """한 팔의 근사 점 (N,3) 와 반지름 (N,). 어깨 l_al_0·1 은 몸통에 붙어 있어 뺀다."""
    chain = [T[f"{s}_al_{i}"][:3, 3] for i in range(2, 8)] + [T[f"{s}_hl_palm_sensor"][:3, 3]]
    arm = [a + (b - a) * t for a, b in zip(chain, chain[1:]) for t in np.linspace(0, 1, 5)]
    hand = [T[n][:3, 3] for n in T if n.startswith(f"{s}_hl_")]
    return np.array(arm + hand), np.array([ARM_R] * len(arm) + [HAND_R] * len(hand))


def box_dist(p, lo, hi):
    return np.linalg.norm(np.maximum(np.maximum(lo - p, 0), p - hi), axis=-1)


def clearance(qL, qR):
    """표면 근사 거리[m]: body(몸통 상자) · hands(왼팔 점 ↔ 오른팔 점) · table(상판 위 여유, 앞쪽 점만)."""
    T = fk_both(qL, qR)
    (pl, rl), (pr, rr) = points(T, "l"), points(T, "r")
    p, r = np.vstack([pl, pr]), np.concatenate([rl, rr])
    body = min(float((box_dist(p, lo, hi) - r).min()) for lo, hi in BODY_BOXES)
    hands = float((np.linalg.norm(pl[:, None] - pr[None], axis=-1) - rl[:, None] - rr[None]).min())
    front = p[:, 0] > TABLE_X_NEAR
    table = float((p[front, 2] - r[front] - TABLE_Z).min()) if front.any() else float("inf")
    return {"body": body, "hands": hands, "table": table}


def palms(qL, qR):
    T = fk_both(qL, qR)
    return {s: {"xyz": T[f"{s[0]}_hl_palm_sensor"][:3, 3].round(4).tolist(),
                "normal": T[f"{s[0]}_hl_palm_sensor"][:3, 2].round(4).tolist()} for s in SIDES}


def lerp(a, b, n):
    return [a + (b - a) * t for t in np.linspace(0, 1, n)]


def check_path(Qs, relax=None, tol=0.0):
    """Qs: [(qL, qR)]. 실패면 (False, 이유). relax = 출발 자세 clearance — 이미 가까운 출발(차렷)은 그보다 더 가까워질 때만 실패."""
    thr = {"body": BODY_MARGIN, "hands": HAND_GAP, "table": TABLE_MARGIN}
    if relax:
        thr = {k: min(v, relax[k] - 1e-3) for k, v in thr.items()}
    worst = {k: np.inf for k in thr}
    for i, (qL, qR) in enumerate(Qs):
        for s, q in (("left", qL), ("right", qR)):
            if not in_limits(s, q, tol):   # 출발 → 홈 경로는 START_TOL(차렷 자세 읽기 오차) 허용
                return False, f"점 {i}: {s} 관절 한계 밖 {np.round(np.rad2deg(q), 1).tolist()}", worst
        c = clearance(qL, qR)
        worst = {k: min(worst[k], c[k]) for k in worst}
        bad = [k for k in thr if c[k] < thr[k]]
        if bad:
            return False, f"점 {i}: {bad} 여유 부족 {({k: round(c[k], 3) for k in bad})} < {thr}", worst
    return True, "", worst


def densify(ways, step=np.deg2rad(2)):
    """관절공간 선형 경유 (qL, qR) 목록 → 2° 간격 점."""
    out = []
    for (aL, aR), (bL, bR) in zip(ways, ways[1:]):
        n = max(2, int(np.ceil(max(np.abs(bL - aL).max(), np.abs(bR - aR).max()) / step)) + 1)
        out += list(zip(lerp(aL, bL, n), lerp(aR, bR, n)))
    return out


def plan_mid(home_L, table_L):
    """홈 → 경유점 → 탁자 경로가 검사를 통과하는 첫 경유점(왼팔 값). 후보: FK 탐색 경유점 · 가운데 · 어깨 먼저 · 팔꿈치 먼저."""
    cands = {"via_fk": VIA_L,   # 11.12 FK 탐색: 경로 최소 여유를 최대화한 경유점(Nelder-Mead), 위 TABLE_L 과 짝
             "mid": (home_L + table_L) / 2,
             "shoulder_first": np.r_[table_L[:3], home_L[3:]],
             "elbow_first": np.r_[home_L[:3], table_L[3], home_L[4:]]}
    for name, w in cands.items():
        ok, why, worst = check_path(densify([pair(home_L), pair(w), pair(table_L)]))
        if ok:
            return name, w, worst
    raise SystemExit(f"홈 → 탁자 경유점을 못 찾음(마지막 이유: {why})")


def table_report(table_L=TABLE_L):
    """탁자 자세 조건 검사. 어기면 AssertionError(자체 검사 — dry 와 실기 둘 다 먼저 돈다)."""
    qL, qR = pair(table_L)
    pa, c = palms(qL, qR), clearance(qL, qR)
    for s, sg in (("left", 1), ("right", -1)):
        x, y, z = pa[s]["xyz"]
        assert 0.30 <= x <= 0.40, f"{s} 손바닥 x {x}"
        assert 0.12 <= sg * y <= 0.15, f"{s} 손바닥 y {y}"
        assert 0.10 <= z - TABLE_Z <= 0.15, f"{s} 손바닥 탁자 위 {z - TABLE_Z:.3f}"
        assert -pa[s]["normal"][2] >= PALM_DOWN_MIN, f"{s} 손바닥 법선 {pa[s]['normal']}"
        assert in_limits(s, qL if s == "left" else qR), f"{s} 관절 한계"
    assert abs(table_L[3] - HOME_L[3]) >= np.deg2rad(30), "j4 가 홈과 30° 넘게 달라야 한다"
    assert c["body"] >= BODY_MARGIN and c["hands"] >= HAND_GAP and c["table"] >= TABLE_MARGIN, c
    pT = fk_both(qL, qR)
    palm_gap = float(np.linalg.norm(pT["l_hl_palm_sensor"][:3, 3] - pT["r_hl_palm_sensor"][:3, 3]))
    return {"palms": pa, "palm_gap_m": round(palm_gap, 4), "clearance_m": {k: round(v, 4) for k, v in c.items()}}


def plan():
    """dry 와 실기가 같이 쓰는 계획(출발 → 홈 구간은 실기에서 출발 자세를 읽은 뒤 따로 검사)."""
    rep = table_report()
    mid_name, mid_L, worst = plan_mid(HOME_L, TABLE_L)
    home, mid, table = pair(HOME_L), pair(mid_L), pair(TABLE_L)
    return {"home": home, "mid": mid, "table": table, "mid_name": mid_name,
            "path_worst_m": {k: round(v, 4) for k, v in worst.items()}, "table_report": rep,
            "home_palms": palms(*home), "home_clearance_m": {k: round(v, 4) for k, v in clearance(*home).items()}}


def deg(q):
    return np.round(np.rad2deg(q), 2).tolist()


def jsonable(p):
    return {k: ({"left_deg": deg(v[0]), "right_deg": deg(v[1]), "left_rad": np.round(v[0], 5).tolist(),
                 "right_rad": np.round(v[1], 5).tolist()} if isinstance(v, tuple) else v) for k, v in p.items()}


def print_plan(p):
    for k in ("home", "mid", "table"):
        print(f"  {k:5s} 왼 {deg(p[k][0])}\n        오 {deg(p[k][1])}")
    print(f"  경유점 = {p['mid_name']} · 경로 최소 여유 {p['path_worst_m']} (기준 body {BODY_MARGIN} · hands {HAND_GAP} · table {TABLE_MARGIN})")
    r = p["table_report"]
    for s in SIDES:
        xyz = r["palms"][s]["xyz"]
        print(f"  탁자 {s:5s} 손바닥 xyz {xyz} · 탁자 위 {xyz[2] - TABLE_Z:.3f} m · 법선 {r['palms'][s]['normal']}")
    print(f"  탁자 손바닥 간격 {r['palm_gap_m']} m · 여유 {r['clearance_m']}")
    print(f"  홈 손바닥 {[p['home_palms'][s]['xyz'] for s in SIDES]} · 홈 여유 {p['home_clearance_m']}")


# =========================== 실기 ===========================
SSH_OPTIONS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=2"]
REMOTE_ROOT = "~/rl_ws/sim2real"


def ssh_command(host, command):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host):
        raise ValueError(f"잘못된 SSH host: {host!r}")
    return ["ssh", *SSH_OPTIONS, host, command]


def helper_command(host, helper, *args):
    if host == "local":
        python = str(CAM_PY) if helper == "sym_cam.py" else sys.executable
        return [python, str(S2R / "scripts/ops" / helper), *args]
    command = "cd " + REMOTE_ROOT + " && exec timeout -k 10 --signal=TERM 7200 .venv/bin/python -u "
    command += shlex.join(["scripts/ops/" + helper, *args])
    return ssh_command(host, command)


def precheck(a):
    checks = {"local": [r"ros2_control_node|controller_manager|openarm"]}
    if not a.skip_head:
        checks.setdefault(a.head_host, []).append(r"head_joint_publisher|head_position_hold")
    if not a.no_cam:
        checks.setdefault(a.cam_host, []).append(r"realsense2_camera|realsense.*node|rs_launch")
    busy = []
    for host, patterns in checks.items():
        cmd = ["ps", "-eo", "pid,args"] if host == "local" else ssh_command(host, "timeout 10 ps -eo pid,args")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
        for line in result.stdout.splitlines()[1:]:
            # ssh wrappers / shells containing a command are not device owners.
            fields = line.strip().split(None, 1)
            if len(fields) != 2:
                continue
            pid, command = fields
            executable = Path(command.split()[0]).name
            if executable in ("ssh", "timeout", "bash", "sh") or "sym_check.py" in command:
                continue
            if re.search("|".join(patterns), command):
                busy.append(f"{host}: PID {pid}")
    if busy:
        print("장치 사용 중 — PID만 출력하고 종료(kill 없음):\n" + "\n".join(busy))
        raise SystemExit(2)
    for c in CAN.values():
        if not Path(f"/sys/class/net/{c}").exists():
            raise SystemExit(f"{c} 없음")
    if not a.skip_head and a.head_host == "local":
        import yaml
        port = yaml.safe_load(HEAD_CFG.read_text())["port"]
        if not Path(port).exists():
            raise SystemExit(f"목 포트 없음: {port}")


class Helper:
    """Bounded JSON RPC; keep the arms controlled while SSH/device I/O is pending."""
    def __init__(self, host, helper, out, *args):
        self.pump: Callable[[], object] | None = None
        self.buffer = bytearray()
        self.error = None
        self.close_error = None
        self.stopping = threading.Event()
        self.log = (out / (Path(helper).stem + "_stderr.log")).open("ab", buffering=0)
        try:
            self.p = subprocess.Popen(helper_command(host, helper, *args), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
            assert self.p.stdin is not None and self.p.stdout is not None
            os.set_blocking(self.p.stdin.fileno(), False)
            threading.Thread(target=self._heartbeat, daemon=True).start()
            self.info = self.reply()
        except BaseException:
            self.close()
            raise

    def _heartbeat(self):
        while not self.stopping.wait(1):
            try:
                self.send({"cmd": "ping"})
            except Exception as e:
                self.error = e
                return

    def send(self, request):
        assert self.p.stdin is not None
        data = (json.dumps(request) + "\n").encode()
        # All requests fit in PIPE_BUF, so heartbeat and RPC cannot interleave.
        if len(data) > 4096 or os.write(self.p.stdin.fileno(), data) != len(data):
            raise RuntimeError("도우미 명령 전송 실패")

    def check(self):
        if self.error is not None or self.p.poll() is not None:
            raise RuntimeError(f"도우미 연결 종료: rc={self.p.poll()}, {self.error!r}")
        assert self.p.stdout is not None
        if select.select([self.p.stdout], [], [], 0)[0]:
            # Outside RPC no response is expected: EOF or an asynchronous error.
            raise RuntimeError("도우미 EOF 또는 원격 오류")

    def reply(self):
        assert self.p.stdout is not None
        deadline = time.monotonic() + 45
        while b"\n" not in self.buffer:
            if self.error is not None:
                raise RuntimeError(f"도우미 통신 실패: {self.error!r}")
            if time.monotonic() > deadline:
                raise TimeoutError("도우미 응답 45 s 초과")
            if self.pump:
                self.pump()
            if select.select([self.p.stdout], [], [], 0.01)[0]:
                chunk = os.read(self.p.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"도우미 EOF(rc={self.p.poll()})")
                self.buffer.extend(chunk)
        line, _, remaining = self.buffer.partition(b"\n")
        self.buffer = bytearray(remaining)
        result = json.loads(line)
        if "error" in result:
            if result.get("torque_held"):
                print("\n!!! 목 복원 안 됨, 토크 유지 중 !!!\n", flush=True)
            raise RuntimeError(f"도우미: {result['error']}")
        return result

    def request(self, cmd, **kwargs):
        self.in_rpc = True
        try:
            self.send({"cmd": cmd, **kwargs})
            return self.reply()
        finally:
            self.in_rpc = False

    def close(self):
        self.stopping.set()
        pump_error = None
        close_buffer = self.buffer

        def tick():
            nonlocal pump_error, close_buffer
            if self.pump:
                try:
                    self.pump()
                except Exception as e:
                    if pump_error is None:
                        pump_error = e
            if self.p.stdout and select.select([self.p.stdout], [], [], 0)[0]:
                close_buffer.extend(os.read(self.p.stdout.fileno(), 65536))
            time.sleep(DT)

        def wait_for(seconds, stop_on_pump_error=False):
            deadline = time.monotonic() + seconds
            while self.p.poll() is None and time.monotonic() < deadline:
                tick()
                if stop_on_pump_error and pump_error is not None:
                    break  # escalate immediately; do not wait 35 s after a control fault
            return self.p.poll() is not None

        try:
            if hasattr(self, "p"):
                if self.p.stdin and not self.p.stdin.closed:
                    self.p.stdin.close()
                # Never block in wait(): pump through graceful exit, TERM and KILL.
                if not wait_for(35, stop_on_pump_error=True):
                    self.p.terminate()  # local SSH; remote timeout/lease handles remote cleanup
                    if not wait_for(5):
                        self.p.kill()
                        if not wait_for(5):
                            self.close_error = "도우미 kill 후에도 종료 안 됨"
                tick()  # collect final JSON emitted during remote cleanup
                for line in close_buffer.splitlines():
                    try:
                        result = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if result.get("error"):
                        self.close_error = result["error"]
                    if result.get("torque_held"):
                        print("\n!!! 목 복원 안 됨, 토크 유지 중 !!!\n", flush=True)
                if self.p.poll() not in (None, 0) and not self.close_error:
                    self.close_error = f"도우미 비정상 종료 rc={self.p.poll()} (목 토크 상태 확인 필요)"
        finally:
            if hasattr(self, "p") and self.p.stdout:
                self.p.stdout.close()
            self.log.close()
        if pump_error is not None:
            raise pump_error
        if self.close_error:
            raise RuntimeError(self.close_error)


class Camera(Helper):
    def __init__(self, host, out):
        super().__init__(host, "sym_cam.py", out, "--serial", CAM_SERIAL)

    def snap(self, prefix):
        result = self.request("snap")
        for kind, suffix in (("color", "_color.png"), ("depth", "_depth.npy")):
            path = Path(str(prefix) + suffix)
            path.write_bytes(base64.b64decode(result.pop(kind + "_b64"), validate=True))
            result[kind] = str(path)
        return result


class Head(Helper):
    def __init__(self, host, out):
        super().__init__(host, "sym_head.py", out)
        self.start = self.info["start"]
        self.targets = self.info["targets"]

    def ticks(self):
        return self.request("read")["ticks"]

    def goto_deg(self, targets):
        return self.request("goto", targets=targets)

    def back_to_start(self):
        return self.request("restore")


class Abort(Exception):
    pass


class Arms:
    def __init__(self):
        import openarm_can as oa
        self.oa = oa
        types = [getattr(oa.MotorType, t) for t in TYPES_NAMES]
        self.o, self.arm, self.rx, self.last_rx = {}, {}, {}, {}
        for s in SIDES:
            self.rx[s] = oa.CANSocket(CAN[s], True)
            o = oa.OpenArm(CAN[s], True)
            o.init_arm_motors(types, IDS, MIDS)
            o.set_callback_mode_all(oa.CallbackMode.STATE)
            self.o[s], self.arm[s] = o, o.get_arm()
            self.last_rx[s] = {i: None for i in MIDS}
        self.names = {s: [f"{s[0]}_aj_{i}" for i in range(1, 8)] for s in SIDES}
        self.enabled = False
        self.helpers = []
        self.hold_target = None
        self.feedback_started = time.monotonic()

    def read_disabled(self, n=20):
        """Refresh BOTH arms each round; neither arm's samples age while reading the other."""
        samples = {s: [] for s in SIDES}
        self.feedback_started = time.monotonic()
        for _ in range(n):
            for s in SIDES:
                self.arm[s].refresh_all()
            time.sleep(0.005)
            for s in SIDES:
                self.receive(s)
                samples[s].append([m.get_position() for m in self.arm[s].get_motors()])
        self.check_feedback()
        out = {}
        for s in SIDES:
            sm = np.array(samples[s])
            if np.all(sm == 0) or not np.all(np.isfinite(sm)):
                raise Abort(f"{s} 팔 응답 없음(전원 · CAN 확인)")
            out[s] = sm.mean(0)
        return out

    def receive(self, side):
        # openarm_can 1.2.9: OpenArm::recv_all is void; Motor has no timestamp/counter.
        # Source: robot_control/ros_ws/src/openarm_can/{src/openarm/can/socket/openarm.cpp,
        # src/openarm/damiao_motor/dm_motor_device.cpp, python/src/openarm_can.cpp}.
        # Use the bound read_canfd_frame bool and the SAME decoder + collection dispatch
        # on a receive-only SDK socket. A valid, identical stationary frame renews age;
        # cached getter equality cannot distinguish a stopped motor from a broken bus.
        # Dispatch here (instead of recv_all) couples the age to actual SDK state updates.
        motors = {m.get_recv_can_id(): m for m in self.arm[side].get_motors()}
        rx = self.rx[side]
        for _ in range(256):  # bound receive work even under a continuously flooded bus
            if not select.select([rx.get_socket_fd()], [], [], 0)[0]:
                return
            frame = self.oa.CanFdFrame()
            if not rx.read_canfd_frame(frame):
                return
            motor = motors.get(frame.can_id)
            if motor is not None and self.oa.CanPacketDecoder.parse_motor_state_data(motor, list(frame.data)).valid:
                self.o[side].get_master_can_device_collection().dispatch_frame_callback(frame)
                self.last_rx[side][frame.can_id] = time.monotonic()
        raise Abort(f"{side} CAN 수신 과다")

    def check_feedback(self):
        now = time.monotonic()
        for side, stamps in self.last_rx.items():
            for can_id, stamp in stamps.items():
                age = now - (stamp if stamp is not None else self.feedback_started)
                if age > FEEDBACK_MAX_AGE:
                    raise Abort(f"{side} motor 0x{can_id:x} stale feedback {age * 1000:.1f} ms")

    def read(self, check_feedback=True):
        if check_feedback and self.enabled:
            self.check_feedback()  # scheduler stalls must not make queued old frames look fresh
        q, dq = {}, {}
        for s in SIDES:
            self.receive(s)
            ms = self.arm[s].get_motors()
            q[s] = np.array([m.get_position() for m in ms]); dq[s] = np.array([m.get_velocity() for m in ms])
        if check_feedback:
            self.check_feedback()
        return q, dq

    def grav(self, s, q):
        return M.gravity_hold(fk.arm_q(s, list(q)), self.names[s])   # 손가락 0(펼침) 가정 — 손은 이 스크립트가 다루지 않는다

    def send(self, qt, kp=KP, kd=KD, gscale=1.0):
        q, dq = self.read()
        P = self.oa.MITParam
        for s in SIDES:
            t = self.grav(s, q[s]) * gscale
            self.arm[s].mit_control_all([P(float(kp[i]), float(kd[i]), float(qt[s][i]), 0.0, float(t[i])) for i in range(7)])
        return q, dq

    def check(self, q, dq, qt):
        for s in SIDES:
            e = np.abs(q[s] - qt[s]); v = np.abs(dq[s])
            if (e > ERR_MAX).any() or v.max() > VEL_MAX:
                j = int(np.argmax(e / ERR_MAX))   # 원인 구분용: 목표·실측·부호 있는 오차(rad, 4자리)를 그대로 남긴다
                raise Abort(f"{s} 오차 {np.round(e, 3).tolist()} 속도 {np.round(v, 2).tolist()} · 최악 j{j + 1}: "
                            f"목표 {qt[s][j]:.4f} 실측 {q[s][j]:.4f} 오차 {q[s][j] - qt[s][j]:+.4f} 속도 {dq[s][j]:+.3f}")

    def step(self, qt):
        self.hold_target = qt
        self.check_helpers()
        q, dq = self.send(qt); self.check(q, dq, qt); time.sleep(DT)
        return q

    def check_helpers(self):
        for helper in self.helpers:
            if not getattr(helper, "in_rpc", False):
                helper.check()

    def move(self, a, b, speed, hold=0.5):
        n = max(1, int(max(np.abs(b[s] - a[s]).max() for s in SIDES) / speed / DT))
        for i in range(n + 1):
            t = min(1.0, i / n)
            self.step({s: a[s] + (b[s] - a[s]) * t for s in SIDES})
        t0 = time.time()
        while time.time() - t0 < hold:
            self.step(b)
        return b

    def hold_for(self, qt, sec):
        t0, Q = time.time(), []
        while time.time() - t0 < sec:
            Q.append(self.step(qt))
        return Q

    def enable(self):
        self.read_disabled(n=5)  # bounded refresh before any enable writes
        self.hold_target, _ = self.read()
        self.enabled = True  # partial enable failure must still enter cleanup
        for s in SIDES:
            self.o[s].enable_all()

    def damp_down(self):
        """Try 1 s hold, then always try 3 s damping on BOTH arms, even with stale RX."""
        try:
            q, _ = self.read()
            self.hold_for(q, 1.0)
        except Exception as e:
            print(f"감쇠: 유지 불가 {e!r} — 중력 보상 없이 수동 감쇠", flush=True)
        t0, errors = time.monotonic(), {}
        while time.monotonic() - t0 < 3.0:
            scale = max(0.0, 1.0 - (time.monotonic() - t0) / 3.0)
            for s in SIDES:
                try:
                    self.receive(s)
                    ms = self.arm[s].get_motors()
                    q = np.array([m.get_position() for m in ms])
                    fresh = all(t is not None and time.monotonic() - t <= FEEDBACK_MAX_AGE for t in self.last_rx[s].values())
                    g = self.grav(s, q) * scale if fresh else np.zeros(7)
                    self.arm[s].mit_control_all([self.oa.MITParam(0.0, float(KD[i] * 3), float(q[i]), 0.0, float(g[i])) for i in range(7)])
                except Exception as e:
                    errors[s] = repr(e)
            time.sleep(DT)
        if errors:
            print(f"!!! 감쇠 전송 실패 {errors} — 비상정지 준비 !!!", flush=True)

    def disable(self):
        failed = set(SIDES)
        for attempt in range(3):  # first attempt + two retries, independent per arm
            for s in tuple(failed):
                try:
                    self.o[s].disable_all()
                except Exception as e:
                    print(f"{s} disable {attempt + 1}/3 실패: {e!r}", flush=True)
                else:
                    failed.remove(s)
            if failed and attempt < 2:
                time.sleep(DT)
        self.enabled = bool(failed)
        for s in sorted(failed):
            print(f"\n{'!' * 64}\n{s} 팔 enable 상태로 남음 — 비상정지\n{'!' * 64}\n", flush=True)
        return sorted(failed)


def wait_key(arms, qt, prompt):
    """자세를 붙든 채(검사 포함) 줄 입력을 기다린다. 'q' 면 Abort."""
    print(prompt, flush=True)
    while True:
        if arms is not None:
            arms.check_helpers()
        if arms is not None and arms.enabled:
            arms.step(qt)
        else:
            time.sleep(DT)
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line.strip().lower() == "q" or line == "":
                raise Abort("사용자 q")
            return


def run_dir():
    d = S2R / "logs/runs" / time.strftime("%y%m%d") / "sym_check" / time.strftime("%H%M%S")
    d.mkdir(parents=True, exist_ok=False)
    return d


def snap(name, arms, head, cam, qt, out, meta):
    arms.hold_for(qt, 0.5)
    Q = np.array([[q[s] for s in SIDES] for q in arms.hold_for(qt, 1.0)])   # (N, 2, 7)
    rec = {"t": time.strftime("%F %T"), "target_deg": {s: deg(qt[s]) for s in SIDES},
           "q_mean_rad": {s: Q[:, k].mean(0).round(6).tolist() for k, s in enumerate(SIDES)},
           "q_std_deg": {s: np.rad2deg(Q[:, k].std(0)).round(4).tolist() for k, s in enumerate(SIDES)},
           "n": len(Q), "head_tick": head.ticks() if head else None}
    rec["q_mean_deg"] = {s: deg(np.array(rec["q_mean_rad"][s])) for s in SIDES}
    rec["fk_palms_measured"] = palms(np.array(rec["q_mean_rad"]["left"]), np.array(rec["q_mean_rad"]["right"]))
    rec["camera"] = cam.snap(out / name) if cam else None
    meta["snaps"][name] = rec
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"  snap {name}: 왼 {rec['q_mean_deg']['left']}\n              오 {rec['q_mean_deg']['right']} · 목 {rec['head_tick']}")
    wait_key(arms, qt, "  ▶ 지금 정면 사진을 찍고 Enter (q = 중단)")


def live(a, p, out, meta):
    precheck(a)
    cam = head = arms = q0 = None
    side = "home"
    q_home = {"left": p["home"][0], "right": p["home"][1]}
    q_mid = {"left": p["mid"][0], "right": p["mid"][1]}
    q_tab = {"left": p["table"][0], "right": p["table"][1]}
    try:
        if not a.no_cam:
            cam = Camera(a.cam_host, out); meta["camera"] = cam.info
            print(f"0) 카메라 열림 {cam.info}")
        if not a.skip_head:
            head = Head(a.head_host, out); meta["head_start_tick"] = head.start
            print(f"   목 시작 틱 {head.start}")
        arms = Arms()
        arms.helpers = [h for h in (head, cam) if h]
        for h in arms.helpers:
            h.pump = lambda: arms.step(arms.hold_target) if arms.enabled else None
        q0 = arms.read_disabled()
        meta["q_start_deg"] = {s: deg(q0[s]) for s in SIDES}
        print(f"   팔 시작(enable 전) 왼 {deg(q0['left'])}\n                      오 {deg(q0['right'])}")
        for s in SIDES:
            if not in_limits(s, q0[s], START_TOL):
                raise SystemExit(f"{s} 시작 자세가 관절 한계 밖 — 중단")
        relax = clearance(q0["left"], q0["right"])
        # 11.12 실측: 차렷에서 홈으로 곧장 가면 손이 탁자 앞 모서리에 3 cm 까지 붙는다 → j1(어깨 앞뒤)만 먼저 홈 값으로 올린 뒤 나머지
        q_j1 = {s: np.r_[p["home"][i][0], q0[s][1:]] for i, s in enumerate(SIDES)}
        ok, why, worst = check_path(densify([(q0["left"], q0["right"]), (q_j1["left"], q_j1["right"]), p["home"]]),
                                    relax=relax, tol=START_TOL)
        meta["start_to_home"] = {"ok": ok, "why": why, "worst_m": worst, "start_clearance_m": relax}
        print(f"   출발 → 홈 경로 검사 {'통과' if ok else '실패: ' + why} · 최소 여유 {({k: round(v, 3) for k, v in worst.items()})}")
        if not ok:
            raise SystemExit("출발 → 홈 경로가 검사를 못 넘는다 — 팔을 손으로 차렷 쪽에 두고 다시")

        wait_key(arms, None, f"1) 양팔 enable → 홈으로 {a.speed} rad/s 이동. Enter (q = 중단)")
        arms.enable()
        q, _ = arms.read()
        cur = arms.move(q, q, a.speed, hold=0.3)        # 읽은 자리 붙들기 먼저(enable 직후 처짐 방지)
        cur = arms.move(cur, q_j1, a.speed, hold=0.2)
        cur = arms.move(cur, q_home, a.speed)
        snap("home", arms, head, cam, cur, out, meta)

        if head:
            print("2) 목 정면 · 아래 보기(head_home_rh56f1 캘리브 자세 pan 2033 · tilt 2864 틱, '정면' 은 확인 필요)")
            wait_key(arms, cur, "   Enter (q = 중단)")
            head.goto_deg(head.targets)
            snap("home_head0", arms, head, cam, cur, out, meta)

        wait_key(arms, cur, f"3) 홈 → 경유점({p['mid_name']}) → 탁자 대칭 자세. Enter (q = 중단)")
        side = "table"   # 여기부터 중단되면 경유점을 거쳐 홈으로
        cur = arms.move(cur, q_mid, a.speed, hold=0.2)
        cur = arms.move(cur, q_tab, a.speed)
        snap("table", arms, head, cam, cur, out, meta)

        print("4) 탁자 → 경유점 → 홈 → snap → 목 시작 틱 → 출발 자세 → disable")
        cur = arms.move(cur, q_mid, a.speed, hold=0.2)
        cur = arms.move(cur, q_home, a.speed)
        side = "home"
        snap("home_back", arms, head, cam, cur, out, meta)
        meta["done"] = True
    except (Exception, KeyboardInterrupt) as e:
        protect_cleanup_signals()
        meta["abort"] = f"{type(e).__name__}: {e}"
        print(f"\n중단: {meta['abort']} — 팔 복귀 → 감쇠 → disable", flush=True)
    finally:
        protect_cleanup_signals()
        cleanup_run(arms, head, cam, q0, q_home, q_mid, side, a.speed, out, meta)


def protect_cleanup_signals():
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)


def interrupt_run(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def cleanup_run(arms, head, cam, q0, q_home, q_mid, side, speed, out, meta):
    """Only exit through damping + independent disable, including return/pump errors."""
    protect_cleanup_signals()
    if arms:
        arms.helpers = []
    pump = (lambda: arms.step(arms.hold_target) if arms.enabled else None) if arms else None
    try:
        if arms and arms.enabled:
            try:
                q, _ = arms.read()
                cur = arms.move(q, q, speed, hold=0.3)
                for w in ([q_mid, q_home] if side == "table" else [q_home]):
                    cur = arms.move(cur, w, speed, hold=0.0)
                if q0 is not None:
                    arms.move(cur, q0, speed, hold=0.5)
            except Exception as e:
                meta["return_error"] = repr(e)
                print(f"복귀 실패: {e!r} — 감쇠 후 disable", flush=True)
        for name, helper in (("head", head), ("camera", cam)):
            if helper:
                helper.pump = pump
                try:
                    # EOF close restores the head and returns its final status as JSON.
                    helper.close()
                except Exception as e:
                    meta[name + "_close_error"] = repr(e)
                    if name == "head":
                        print("\n!!! 목 복원 안 됨, 토크 유지 중 (원격 로그 확인) !!!\n", flush=True)
    finally:
        try:
            if arms and arms.enabled:
                try:
                    arms.damp_down()
                except Exception as e:
                    meta["damp_error"] = repr(e)
                    print(f"!!! 감쇠 실패: {e!r} — 비상정지 준비 !!!", flush=True)
                finally:
                    meta["disable_failed"] = arms.disable()
        finally:
            (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=float))


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt_run)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="CAN · 목 · 카메라를 열지 않고 계획 · FK · 여유만 출력하고 meta.json 저장")
    ap.add_argument("--skip-head", action="store_true", help="목을 열지도 움직이지도 않는다(2단계 건너뜀)")
    ap.add_argument("--head-host", default="arm4090-lan", help="목 SSH 호스트 또는 local")
    ap.add_argument("--cam-host", default="arm4090-lan", help="카메라 SSH 호스트 또는 local")
    ap.add_argument("--no-cam", action="store_true", help="카메라 없이 관절만 기록")
    ap.add_argument("--speed", type=float, default=0.25, help=f"관절 최대 속도 rad/s(기본 0.25 = j3_release_v2, 상한 {SPEED_MAX})")
    a = ap.parse_args()
    if not 0 < a.speed <= SPEED_MAX:
        raise SystemExit(f"--speed 는 0~{SPEED_MAX} rad/s")
    p = plan()
    print("계획(왼 / 오, deg):"); print_plan(p)
    out = run_dir()
    meta = {"head_host": a.head_host, "cam_host": a.cam_host, "no_cam": a.no_cam, "skip_head": a.skip_head,
            "dry": a.dry, "t": time.strftime("%F %T"), "speed_rad_s": a.speed, "plan": jsonable(p), "snaps": {},
            "consts": {"TABLE_Z": TABLE_Z, "TABLE_X_NEAR": TABLE_X_NEAR, "BODY_MARGIN": BODY_MARGIN, "HAND_GAP": HAND_GAP,
                       "TABLE_MARGIN": TABLE_MARGIN, "ARM_R": ARM_R, "HAND_R": HAND_R, "CAM_SERIAL": CAM_SERIAL,
                       "HEAD_CFG": str(HEAD_CFG), "urdf": str(fk.URDF)}}
    if a.dry:
        (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        print(f"dry — 저장 {out / 'meta.json'}")
        return
    live(a, p, out, meta)
    print(f"저장 {out}")


if __name__ == "__main__":
    main()
