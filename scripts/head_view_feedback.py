#!/usr/bin/env python3
"""sim = real 머리 시야 피드백 — IMU 중력과 /head/joint_states 를 보며 tilt 를 맞춘다(--execute 일 때만 움직인다).

10.30 사용자: 최종 목표는 정책이 원하는 시야(sim 카메라)와 실기 시야를 같게 하는 것이고, 그 피드백 제어를 IMU 와 joint state 로 한다.
11.03 사용자: "FP++ 를 받는 것도 중요하지만 sim 과 real 의 카메라 시야가 같아야 의미가 있다" — 기준을 camset 으로 옮겼다.

  기준  외부 파라미터 config/global_camera_extrinsics_arm4090_m1117.yaml(camset, 미션 extrinsics, 11.17 테이블 마커 판). 10.28 실기
        시야를 sim 시야에 맞춘 자세(옛 r20u5 멈춘 자리 1825/2815 = 새 영점 1825/2833)에서 잰 값이고 FP++ base 좌표가 이 값으로만 맞다.
        sim 카메라 출처 = hdgp scripts/tools/e2e_chain/cocktail/headcam.py --static global_camera_extrinsics_arm4090.yaml(10.27 home)
        --static_rel -20,-5(URDF 목 FK 근사) → videos/cocktail_1028_b/headcam_static/headcam_1028.json 의 static_dpan-20_dtilt-5,
        숙임 67.67°. r20u5 숙임과의 차(10.28 볼트 맞춤 잔차)는 설정 sim_pitch_deg 로 같이 찍는다.
  측정  카메라 숙임 = D435i 가속도(중력)로 잰 광축의 아래 각. 처짐 · 조립 유격 · 손으로 건드린 것까지 보인다(인코더는 못 본다).
        yaw(pan 방향 어긋남)는 중력으로 못 잰다. 인코더 각 = /head/joint_states(URDF 규약 → 인코더 deg 로 되돌린다).
  검사  --check(미션 head_view_check 단계, 움직이지 않음): ★11.17 "외부 파라미터를 잰 그때와 머리 · 카메라가 같은가"를 잰다.
        기준 파일 config/head_view_ref_arm4090_h2048.yaml(그때의 보정 전 IMU 위쪽 벡터 · 목 틱 · 외부 파라미터 md5)과
        지금의 보정 전 IMU 숙임 · 옆 기울기 · 틱을 직접 비교한다(허용치 설정 ref_tol_*). bias 는 쓰지 않는다 — 11.15 bias 는
        "base z = 중력"이라 놓고 구한 값이라 검사에 쓰면 순환이고, 옆 기울기와 틱도 안 봤다. md5 가 다르면 실패(--record-ref 필요).
        기준이 실측(measured true)이 아니면 판정을 찍고도 실패한다. 기준의 접근 방향(approach, 기본 down = 숙이며 도착)으로
        도착했을 때만 유효하다 — 같은 인코더 자리도 오르내림에 따라 숙임이 평균 0.5° · 최대 1.13° 다르다.
        --record-ref: 외부 파라미터를 막 잰 자세에서 기준 파일을 새로 쓴다(measured true, --approach). 틱이 head_pose ± tol_tick
        밖 · σ · 샘플 부족이면 거부, (외부 파라미터 각 + 설정 bias) 와 1° 넘게 다르면 --force 없이는 거부. 옛 파일 .bak-<시각>.
  마커  ★11.17 테이블 영구 고정 ArUco(DICT_6X6_250 id 70 왼쪽 끝 · 71 오른쪽 끝 · 73, 검은 부분 실측 79.62 mm, 설정 anchor_*)로 IMU 가
        못 보는 좌우(yaw) · 위치를 본다. --record-ref 가 컬러 영상 2 x ANCHOR_FRAMES 장의 코너 평균 · PnP 를 기준 anchors 로
        함께 쓰고(마커가 하나라도 안 보이거나 코너 σ > ANCHOR_MAX_SD_PX 면 거부 — σ 는 --force 로만, 붙이기 전이면 --no-anchors),
        두 묶음끼리 비교한 값을 self_noise 로 남긴다. --check 는 기준에 anchors 가 있으면 마커별 코너 px · 마커 중심 이동 mm ·
        카메라 회전 ° 도 허용치(ref_tol_anchor_*) 안이어야 통과한다. 없으면 IMU · 틱만 보고 그렇다고 찍는다.
        마커를 쓰는 모드는 ROS 를 띄운 뒤 어떤 이동보다 먼저 영상을 한 번 받아 본다(안 오면 움직이지 않고 1).
  목표  --target extrinsics(기본): 기준 파일의 보정 전 숙임(imu_up_cam)을 보정 전 IMU 로 맞춘다. 기준이 없거나 md5 가 다르거나
        measured false 면 --execute 는 이유를 찍고 멈춘다(--allow-bias-target 이면 옛 방식: IMU 0점 bias 로 외부 파라미터 숙임).
        tilt 는 기준의 접근 방향으로만 들어오고, 목표를 넘어서면 APPROACH_BACKOFF_DEG 물렸다가 다시 들어온다.
        기준에 anchors 가 있으면 tilt 뒤 pan 을 마커 가로 이동(pan_shift_px)으로 맞추고(approach_pan 방향으로만, 시작 pan ±
        --pan-window), tilt 를 한 번 더 확인한 뒤 --check 와 같은 판정을 찍는다.
        --target <deg>: 숙임 각을 직접(0 = 정면 수평, 상한 찾기용).
  정렬  --align --execute(★11.17 시작 루틴, 목이 움직인다): --camset(head_camset_rh56f1_h2048.yaml) 자세로 접근 방향(기준
        approach · approach_pan, 반대쪽이면 APPROACH_BACKOFF_DEG 물렸다가)으로 보낸 뒤, 마커 70 · 71 · 73 의 코너 px 가 기준
        anchors 에서 설정 align_tol_px 안에 들 때까지 tilt(세로 이동) · pan(가로 이동)을 번갈아 고친다(_fit_axis, 한 축 규칙은
        --execute 의 pan 맞춤과 같다, 걸음 lim.max_iters · 시간 ALIGN_TIMEOUT_S). 그 자세에서 마커 코너(2 x ANCHOR_FRAMES 장
        평균)와 정답 base 좌표(설정 anchor_truth: 70 · 71 xy, 73 은 x 만 · y 자유, 상판 z)로 외부 파라미터를 RGB 만으로 풀어
        --extrinsics 와 extr_tol_mm · extr_tol_deg 로 비교한다. 밖이면 1, --write-extrinsics 면 camera 위치 · 회전만 새로 쓴다
        (옛 파일 .bak-<시각>, 뒤에 --record-ref). 73 이 안 보이면 경고하고 70 · 71 로, 70 · 71 이 안 보이면 1. IMU 가 기준과
        imu_warn_deg 넘게 다르면 경고만. --out 에 JSON 한 줄을 덧붙인다.
  제어  남은 숙임 오차를 인코더 이동으로 바꿔 head_move 요청을 보낸다. 비율(Δ숙임/Δ인코더)은 매 단계 다시 잰다.
        한 번에 --max-step° 이하, tilt 는 [--min-tilt, --max-tilt] 안, 비율이 RATIO_OK 밖이면 멈춘다.
  IMU 0점 imu_pitch_bias = (시야를 맞춘 자세의 IMU 숙임) − (외부 파라미터 숙임). D435i IMU 는 공장 보정이 없어 0점이 치우친다.
        절대 중력(테이블 수평 대비) 숙임을 볼 때(--sweep · --grid · --target <deg>)만 쓴다. --calib-bias 는 지금 자세가 기준 시야와
        맞았다고 보고 다시 잰다.

  python3 scripts/head_view_feedback.py --check                 # 시야 검사만(미션 단계): 기준 파일과 숙임 · 옆 · 틱 차 · 판정
  python3 scripts/head_view_feedback.py --record-ref            # 기준 다시 잡기(외부 파라미터를 막 잰 자세에서, 움직이지 않음)
  python3 scripts/head_view_feedback.py --record-ref --no-anchors   # 고정 마커를 붙이기 전(IMU · 틱만)
  python3 scripts/head_view_feedback.py                         # 재기만: 지금 숙임 · 목표 · 남은 오차 · 다음 tilt
  python3 scripts/head_view_feedback.py --execute               # 기준 시야로 맞춘다(목이 움직인다 — 사용자 허락 뒤)
  python3 scripts/head_view_feedback.py --execute --target 0 --min-tilt -8   # 정면 수평 찾기(상한)
  python3 scripts/head_view_feedback.py --calib-bias            # IMU 0점 다시 재기
  python3 scripts/head_view_feedback.py --align --execute --out <기록.jsonl>   # 시작 루틴(목이 움직인다 — 허락 뒤)

ROS: ROS_DOMAIN_ID 는 드라이버 · 카메라와 같아야 한다. 목 드라이버(head_joint_publisher)와 realsense(IMU 켬,
scripts/vision/camera_up.sh)가 떠 있어야 한다. 마커를 볼 때는 컬러 /camera/camera/color/image_raw · camera_info 도 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import ArrayLike

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

REPO = _HERE.parent
DEFAULT_EXTRINSICS = REPO / "config" / "global_camera_extrinsics_arm4090_m1117.yaml"   # camset = 미션 extrinsics
DEFAULT_CONFIG = REPO / "config" / "head_view_feedback_arm4090.yaml"
DEFAULT_REF = REPO / "config" / "head_view_ref_arm4090_h2048.yaml"   # ★11.17 --check 기준(외부 파라미터를 잰 그때의 IMU · 틱)
#: ★11.17 --auto-extrinsics 산출물(호스트마다 다른 런타임 파일 · .gitignore). m1117 은 학생 · head_depth_node 가 md5 로 고정해 쓰지 않는다
DEFAULT_LIVE = REPO / "config" / "global_camera_extrinsics_arm4090_live.yaml"
DEFAULT_LIVE_REF = REPO / "config" / "head_view_ref_arm4090_live.yaml"   # live 를 쓴 자세의 기준(재기준) — 같으면 live 를 다시 안 쓴다
#: 정지 IMU 1샘플 σ 0.012 m/s²(10.30). 그 4배를 넘으면 머리가 움직이는 중이라 재지 않는다.
REF_MAX_SD = 0.05
#: Sensors.read 가 평균 내는 IMU 샘플 수. 검사 · 기준 기록은 이 절반도 못 받으면 쓰지 않는다.
N_SAMPLES = 400
#: tilt 틱 증가 = 숙임(scripts/head_turns.py 10.28 확인, Limits.min_tilt). 마지막 이동이 이 부호여야 기준과 같은 백래시 쪽이다.
APPROACH_SIGN = {"down": 1, "up": -1}
#: 오르내림 숙임 차 최대 1.13°(10.30 처짐 지도) 보다 크게 물렸다가 접근 방향으로 다시 들어온다.
APPROACH_BACKOFF_DEG = 1.5
#: --record-ref 가드: 새 보정 전 IMU 각과 (외부 파라미터 각 + 설정 bias) 가 이보다 다르면 --force 없이는 안 쓴다.
REF_GUARD_DEG = 1.0
IMU_TOPIC = "/camera/camera/imu"
JOINT_TOPIC = "/head/joint_states"
#: 10.30 처짐 지도: 36~61° 구간은 처짐이 1.5 → 3.8° 로 바뀌어 구간 비율이 0.84~1.22 로 정상이다. 그 밖은 1.00 ± 0.03.
RATIO_OK = (0.7, 1.3)
#: ★11.17 사용자: 테이블 양 끝 영구 고정 마커(id · 크기는 설정 anchor_ids · anchor_size_m). IMU 가 못 보는 좌우 · 위치를 본다.
ANCHOR_DICT = "DICT_6X6_250"
ANCHOR_FRAMES = 5   # 코너를 평균 낼 영상 장수
ANCHOR_MAX_SD_PX = 0.5   # --record-ref 코너 σ 상한(정지 실측 마커 중심 흔들림 0.2~0.6 mm). 넘으면 --force 로만
PAN_PROBE_DEG = 1.0      # _fit_pan 첫 이동 상한 — 이득 부호 · 크기를 실제로 확인하기 전에는 크게 돌리지 않는다
COLOR_TOPIC = "/camera/camera/color/image_raw"     # scripts/vision/cup_view.py 와 같은 토픽 · rgb8/bgr8 처리
INFO_TOPIC = "/camera/camera/color/camera_info"
#: pan 틱 증가 = 왼쪽으로 돈다(config/head_camset_rh56f1.yaml: home 2033 에서 "오른쪽 208 틱" = 1825,
#: head_fk_chain.PAN_ENCODER_TO_URDF = −1 · URDF pan 축 (0,0,−1)). 기본 접근 = 틱이 커지며(left) 도착.
PAN_APPROACH_SIGN = {"left": 1, "right": -1}
DEFAULT_CAMSET = REPO / "config" / "head_camset_rh56f1_h2048.yaml"   # --align 이 처음 보내는 목 자세(미션 head_home 과 같은 파일)
TICK_DEG = 360 / 4096          # XC330 한 틱. 남은 보정이 반 틱보다 작으면 더 움직여도 못 줄인다
ALIGN_TIMEOUT_S = 180.0        # --align 전체 시간 상한(이동 한 번은 head_move.DONE_TIMEOUT_S 40 s)
ALIGN_MAX_RMS_PX = 2.0         # --align 외부 파라미터 풀이 재투영 rms 상한(11.17 verify 0.94 px). 넘으면 판정 · 쓰기 안 함


# ---------- 순수부(ROS · 하드웨어 무관) ----------

def gravity_angles(ax: float, ay: float, az: float) -> tuple[float, float]:
    """광학 프레임(x 오른쪽 · y 아래 · z 앞) 가속도 → (숙임 deg, 옆 기울기 deg). 정지 때 가속도 = 위쪽 단위벡터 × g."""
    g = math.sqrt(ax * ax + ay * ay + az * az)
    if g < 5.0:
        raise ValueError(f"가속도 크기 {g:.2f} m/s² — 중력이 아니다(IMU 이상)")
    return math.degrees(math.asin(max(-1.0, min(1.0, -az / g)))), math.degrees(math.atan2(ax, -ay))


def up_to_angles(up: ArrayLike) -> tuple[float, float]:
    """위쪽 벡터(광학 프레임, 크기 무관) → (숙임, 옆 기울기) deg."""
    up = np.asarray(up, float)
    return gravity_angles(*(up / np.linalg.norm(up) * 9.81))


def angles_to_up(pitch: float, roll: float) -> np.ndarray:
    """gravity_angles 의 역: (숙임, 옆 기울기) deg → 광학 프레임 위쪽 단위벡터."""
    p, r = math.radians(pitch), math.radians(roll)
    return np.array([math.cos(p) * math.sin(r), -math.cos(p) * math.cos(r), -math.sin(p)])


def deg_to_tick(deg: float) -> int:
    """인코더 deg → XC330 틱(영점 2048 · offset 0)."""
    return round(deg * 4096 / 360) + 2048


def compare_to_ref(up_now: ArrayLike, ref_up: ArrayLike, now_ticks: tuple[int, int] | None, ref_ticks: tuple[int, int],
                   tol_pitch: float, tol_roll: float, tol_tick: float) -> dict:
    """지금 보정 전 IMU 위쪽 · 틱(pan, tilt) 을 기준과 비교. 틱이 None(못 읽음)이면 틱 항목 실패."""
    (pn, rn), (pr, rr) = up_to_angles(up_now), up_to_angles(ref_up)
    a, b = (np.asarray(v, float) / np.linalg.norm(v) for v in (up_now, ref_up))
    d_tick = None if now_ticks is None else (now_ticks[0] - ref_ticks[0], now_ticks[1] - ref_ticks[1])
    r = dict(pitch_now=pn, pitch_ref=pr, d_pitch=pn - pr, roll_now=rn, roll_ref=rr, d_roll=rn - rr,
             angle_deg=math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b))))),
             ticks_now=now_ticks, ticks_ref=tuple(ref_ticks), d_tick=d_tick, tol_pitch=tol_pitch, tol_roll=tol_roll, tol_tick=tol_tick,
             ok_pitch=abs(pn - pr) <= tol_pitch, ok_roll=abs(rn - rr) <= tol_roll,
             ok_tick=d_tick is not None and max(abs(d) for d in d_tick) <= tol_tick)
    r["ok"] = r["ok_pitch"] and r["ok_roll"] and r["ok_tick"]
    return r


def ref_report(r: dict, ref_name: str) -> str:
    """--check 출력: 한 줄 요약 + 항목별 줄."""
    mark = {True: "✓", False: "❌"}
    if r["ok"]:
        head = f"✓ 머리 · 카메라가 외부 파라미터를 잰 때와 같다(기준 {ref_name}, 위쪽 벡터 차 {r['angle_deg']:.2f}°, yaw 는 IMU 로 못 잰다)"
    else:
        bad = [n for n, k in (("숙임", "ok_pitch"), ("옆", "ok_roll"), ("틱", "ok_tick")) if not r[k]]
        head = (f"❌ 머리 · 카메라가 외부 파라미터를 잰 때와 다르다({' · '.join(bad)}, 위쪽 벡터 차 {r['angle_deg']:.2f}°) — "
                "카메라 · 머리를 건드렸는지 보고, 맞춘 뒤 외부 파라미터를 다시 재고 --record-ref(--help 끝)")
    t = r["ticks_now"]
    tick = ("못 읽음" if t is None else
            f"pan {r['d_tick'][0]:+d} · tilt {r['d_tick'][1]:+d}(지금 {t[0]}/{t[1]} − 기준 {r['ticks_ref'][0]}/{r['ticks_ref'][1]})")
    return "\n".join([
        head,
        f"  숙임 차 {r['d_pitch']:+.2f}°(지금 {r['pitch_now']:.2f} − 기준 {r['pitch_ref']:.2f}, 보정 전) 허용 ±{r['tol_pitch']}° {mark[r['ok_pitch']]}",
        f"  옆 차   {r['d_roll']:+.2f}°(지금 {r['roll_now']:+.2f} − 기준 {r['roll_ref']:+.2f}, 보정 전) 허용 ±{r['tol_roll']}° {mark[r['ok_roll']]}",
        f"  틱 차   {tick} 허용 ±{r['tol_tick']} {mark[r['ok_tick']]}"])


REF_HEADER = """# head_view_feedback.py --check 기준 — 외부 파라미터를 잰 그때의 머리 · 카메라 상태(arm4090 머리 D435i · 목 영점 2048).
#
# extrinsics     기준을 잡은 외부 파라미터 파일 이름과 md5. --extrinsics 파일의 md5 가 다르면 --check 는 실패한다.
# imu_up_cam     보정 전 IMU 가속도 평균의 단위벡터. 광학 프레임(x 오른쪽 · y 아래 · z 앞)이고 정지 때 가속도 = 위쪽 × g.
# pitch_raw_deg  위 벡터의 숙임 · 옆 기울기(gravity_angles). 읽기용이고 bias 를 빼지 않은 값이다.
# roll_raw_deg
# pan_tick       그때 목 틱(/head/joint_states → 인코더 deg → 틱).
# tilt_tick
# n · acc_sd     평균 낸 IMU 샘플 수와 가속도 σ(m/s²). time · source 는 언제 · 어떻게 잡았는지.
# measured       true = --record-ref 로 잰 값. false(이주 · 손으로 만든 값)면 --check 는 판정을 찍고도 실패하고 --execute 는 안 쓴다.
# approach       그 자세에 도착한 마지막 tilt 이동 방향. down = 틱이 커지며(숙이며) 도착, up = 틱이 작아지며(들며) 도착.
#                같은 인코더 자리도 오르내림에 따라 숙임이 평균 0.5° · 최대 1.13° 달라서(10.30 처짐 지도) 이 방향으로 도착했을
#                때만 기준이 유효하다. --execute 는 이 방향으로만 들어온다(넘어서면 물렸다가 다시).
# approach_pan   --execute 가 pan 을 마커로 맞출 때 들어오는 방향. left = pan 틱이 커지며(기본), right = 작아지며.
# anchors        ★11.17 테이블 양 끝 영구 고정 ArUco(DICT_6X6_250, 설정 anchor_ids · anchor_size_m)를 그때 본 값.
#                markers.<id>: corners = 네 코너 px(ANCHOR_FRAMES 장 평균, 서브픽셀), rvec · tvec = T_cam_marker(m, IPPE_SQUARE).
#                n_frames · corner_sd_px = 평균 장수와 장 사이 코너 σ(ddof 1, 상한 ANCHOR_MAX_SD_PX).
#                self_noise = 연속 두 묶음(ANCHOR_FRAMES 장씩)을 서로 비교한 px · mm · ° (허용치 ref_tol_anchor_* 의 근거).
#                size_m · dict · image_wh · fx = 그때 설정 · 영상. --check 는 설정 · 지금 영상과 다르면 실패한다.
#                있으면 --check 가 IMU · 틱에 더해 마커(좌우 · 위치)도 보고, --execute 가 pan 을 마커 가로 이동으로 맞춘다.
#                없으면(--record-ref --no-anchors, 마커 붙이기 전) IMU · 틱만 본다.
# 다시 잴 때: 외부 파라미터를 막 잰 자세 그대로(머리 · 카메라를 건드리기 전) 목 드라이버와 카메라(IMU 켬)를 띄우고
#   python3 scripts/head_view_feedback.py --record-ref --extrinsics <그 외부 파라미터 파일> --approach <down|up>
#   (설정 anchor_ids 마커가 하나라도 안 보이면 거부. 마커를 붙이기 전이면 --no-anchors)
#   틱이 그 파일 head_pose 에서 tol_tick 넘게 벗어나거나 σ · 샘플 수가 모자라면 쓰지 않는다. (외부 파라미터 각 + 설정 bias) 와
#   1° 넘게 다르면 --force 없이는 쓰지 않는다. 옛 파일은 .bak-<시각> 으로 남고 이 주석은 코드가 다시 붙인다.
"""


def ref_text(data: dict) -> str:
    return REF_HEADER + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def ref_problem(ref: object, ext_md5: str, cfg: dict | None = None) -> str | None:
    """기준 파일을 이 외부 파라미터로 쓸 수 없는 이유(없으면 None). measured 는 따로 본다.
    cfg 를 주면 기준 anchors 가 설정(anchor_ids 전부 · anchor_size_m) · ANCHOR_DICT 와 같은지도 본다."""
    if ref is None:
        return "기준 파일이 없다"
    if not isinstance(ref, dict) or not isinstance(ref.get("extrinsics"), dict) or any(
            ref.get(k) is None for k in ("imu_up_cam", "pan_tick", "tilt_tick")):
        return "기준 파일이 깨졌다(extrinsics · imu_up_cam · pan_tick · tilt_tick)"
    if ref.get("approach", "down") not in APPROACH_SIGN:
        return f"기준 approach {ref.get('approach')!r} 가 down · up 이 아니다"
    if ref.get("approach_pan", "left") not in PAN_APPROACH_SIGN:
        return f"기준 approach_pan {ref.get('approach_pan')!r} 가 left · right 가 아니다"
    anc = ref.get("anchors")
    if anc is not None and not (isinstance(anc, dict) and isinstance(anc.get("markers"), dict) and anc["markers"]):
        return "기준 anchors 가 깨졌다(markers 없음)"
    if anc is not None and cfg is not None:
        want, have = sorted(int(i) for i in cfg.get("anchor_ids", [])), sorted(int(k) for k in anc["markers"])
        if want != have:
            return f"기준 마커 {have} 가 설정 anchor_ids {want} 와 다르다"
        if anc.get("size_m") is None or abs(float(anc["size_m"]) - float(cfg.get("anchor_size_m", -1))) > 1e-6:
            return f"기준 마커 크기 {anc.get('size_m')} m 가 설정 anchor_size_m {cfg.get('anchor_size_m')} 와 다르다"
        if anc.get("dict") != ANCHOR_DICT:
            return f"기준 마커 사전 {anc.get('dict')} 가 {ANCHOR_DICT} 가 아니다"
    if ref["extrinsics"].get("md5") != ext_md5:
        return f"기준이 다른 외부 파라미터로 잡혔다(md5 {ref['extrinsics'].get('md5')} ≠ {ext_md5})"
    return None


def run_check(ref: object, up_now: ArrayLike, ticks_now: tuple[int, int] | None, n: int, sd: float, ext_md5: str,
              tol: tuple[float, float, float], ref_name: str = "기준 파일", anchors_now: dict | None = None,
              anchor_tol: tuple[float, float, float] | None = None) -> tuple[int, str]:
    """--check 판정 → (종료 코드, 출력). 실측(measured true)이 아닌 기준은 판정 줄을 찍고도 1.
    기준에 anchors 가 있으면 고정 마커 비교(anchors_now, anchor_tol = px · mm · deg)도 통과해야 0. 없으면 IMU · 틱만."""
    why = ref_problem(ref, ext_md5)
    if why:
        return 1, f"❌ {why} — 외부 파라미터를 잰 자세에서 --record-ref 필요"
    if sd > REF_MAX_SD:
        return 1, f"❌ IMU 흔들림 σ {sd:.3f} > {REF_MAX_SD} m/s² — 머리가 움직이는 중이다. 멈춘 뒤 다시"
    if n < N_SAMPLES // 2:
        return 1, f"❌ IMU 샘플 {n} 개 < {N_SAMPLES // 2} — 카메라 IMU 가 느리거나 끊긴다"
    r = compare_to_ref(up_now, ref["imu_up_cam"], ticks_now, (ref["pan_tick"], ref["tilt_tick"]), *tol)
    lines = [ref_report(r, ref_name),
             f"  기준 접근 방향 {ref.get('approach', 'down')} — 이 방향으로 도착했을 때만 유효"]
    ok = r["ok"]
    if ref.get("anchors") is None:
        lines.append("  마커 기준 없음 — 좌우 · 위치는 안 봤다(IMU · 틱만)")
    elif anchors_now is None or anchor_tol is None:
        lines.append("❌ 기준에 고정 마커가 있는데 지금 영상 · 허용치가 없다")
        ok = False
    else:
        a = compare_anchors(anchors_now, ref["anchors"]["markers"], *anchor_tol)
        lines += anchor_report(a)
        ok = ok and a["ok"]
    if ref.get("measured") is not True:
        lines.append("❌ 기준이 실측이 아님 — --record-ref 필요")
        return 1, "\n".join(lines)
    return (0 if ok else 1), "\n".join(lines)


def record_ref_guard(up: ArrayLike, ticks: tuple[int, int], n: int, sd: float, head_pose: dict,
                     ext_angles: tuple[float, float], bias: tuple[float, float], force: bool) -> tuple[bool, str]:
    """--record-ref 를 써도 되나 → (되나, 출력). σ · 샘플 수 · head_pose 틱은 늘 막고, (외부 각 + bias) 와 1° 넘는 차는 force 로만."""
    if sd > REF_MAX_SD:
        return False, f"❌ IMU 흔들림 σ {sd:.3f} > {REF_MAX_SD} m/s² — 머리가 움직이는 중이다. 멈춘 뒤 다시"
    if n < N_SAMPLES // 2:
        return False, f"❌ IMU 샘플 {n} 개 < {N_SAMPLES // 2} — 카메라 IMU 가 느리거나 끊긴다"
    d = max(abs(ticks[0] - head_pose["pan_tick"]), abs(ticks[1] - head_pose["tilt_tick"]))
    if d > head_pose["tol_tick"]:
        return False, (f"❌ 틱 {ticks[0]}/{ticks[1]} 이 외부 파라미터 head_pose {head_pose['pan_tick']}/{head_pose['tilt_tick']} 와 "
                       f"{d} 틱 차(> {head_pose['tol_tick']}) — 외부 파라미터를 잰 자세가 아니라 기준을 쓰지 않는다")
    p, r = up_to_angles(up)
    ep, er = ext_angles[0] + bias[0], ext_angles[1] + bias[1]
    text = (f"새 보정 전 IMU 숙임 {p:.2f}° 옆 {r:+.2f}° − (외부 파라미터 + 설정 bias) {ep:.2f}° {er:+.2f}° "
            f"= 숙임 {p - ep:+.2f}° 옆 {r - er:+.2f}°")
    if max(abs(p - ep), abs(r - er)) > REF_GUARD_DEG and not force:
        return False, (text + f"\n❌ {REF_GUARD_DEG}° 넘게 다르다 — 카메라 · 머리를 건드렸거나 다른 자세다. "
                       "외부 파라미터를 이 자세에서 막 쟀고 bias 가 낡은 것이 확실하면 --force")
    return True, text


def build_ref(up: ArrayLike, ticks: tuple[int, int], n: int, sd: float, ext_name: str, ext_md5: str, approach: str,
              time_str: str, source: str, approach_pan: str = "left", anchors: dict | None = None) -> dict:
    """--record-ref 가 쓰는 기준(measured true). anchors = anchors_block(…) 또는 None(--no-anchors)."""
    u = np.asarray(up, float) / np.linalg.norm(up)
    p, r = up_to_angles(u)
    out = {"extrinsics": {"file": ext_name, "md5": ext_md5}, "measured": True, "approach": approach,
           "approach_pan": approach_pan,
           "imu_up_cam": [round(float(v), 6) for v in u], "pitch_raw_deg": round(p, 3), "roll_raw_deg": round(r, 3),
           "pan_tick": int(ticks[0]), "tilt_tick": int(ticks[1]), "n": int(n), "acc_sd": round(float(sd), 4),
           "time": time_str, "source": source}
    if anchors is not None:
        out["anchors"] = anchors
    return out


def write_ref(path: Path, data: dict) -> None:
    replace_file(path, ref_text(data))


def replace_file(path: Path, text: str) -> Path | None:
    """옛 파일은 .bak-<시각> 으로 두고, 임시 파일에 쓴 뒤 바꿔 끼운다(쓰다 끊겨도 반쯤 남지 않게) → 백업 경로."""
    bak = None
    if path.exists():
        bak = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return bak


def approach_cmd(tilt: float, cmd: float, approach: str, lim: Limits, arrived_wrong: bool = False) -> tuple[float, bool]:
    """다음 tilt 목표를 접근 방향으로만 들어오게 → (명령, 물림인가). 반대로 가야 하면(넘어섬) 또는 반대 방향으로 도착했으면
    목표보다 APPROACH_BACKOFF_DEG 더 반대쪽으로 물린다(한 번에 max_step · tilt 한계 안). 다음 스텝이 접근 방향으로 들어온다."""
    return approach_axis(tilt, cmd, APPROACH_SIGN[approach], lim.min_tilt, lim.max_tilt, lim.max_step, arrived_wrong)


def approach_axis(cur: float, cmd: float, s: int, lo: float, hi: float, max_step: float,
                  arrived_wrong: bool = False) -> tuple[float, bool]:
    """approach_cmd 를 한 축(tilt · pan)으로: s = 접근 방향 부호(+1 = 각이 커지며 도착), [lo, hi] = 한계."""
    if (cmd - cur) * s >= 0 and not arrived_wrong:
        return cmd, False
    back = cur + max(-max_step, min(max_step, cmd - s * APPROACH_BACKOFF_DEG - cur))
    return max(lo, min(hi, back)), True


# ---------- 고정 마커(테이블 양 끝 ArUco) ----------

def _pnp(corners: ArrayLike, K: ArrayLike, dist: ArrayLike, size_m: float) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    h = size_m / 2   # SOLVEPNP_IPPE_SQUARE 코너 순서(왼위 · 오른위 · 오른아래 · 왼아래 = aruco 검출 순서)
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float32).reshape(4, 1, 2), np.asarray(K, float),
                                  np.asarray(dist, float), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("마커 solvePnP 실패")
    return rvec.ravel(), tvec.ravel()


def detect_anchors(gray: np.ndarray, K: ArrayLike, dist: ArrayLike, ids: set[int], size_m: float) -> dict[int, dict]:
    """한 장에서 고정 마커(ids)를 찾는다 → {id: {"corners": 4x2 px, "rvec", "tvec"(m, T_cam_marker)}}.
    코너 서브픽셀 정밀화, PnP SOLVEPNP_IPPE_SQUARE. 같은 id 가 두 번 보이면 어느 쪽이 고정 마커인지 몰라 ValueError."""
    import cv2
    ar = cv2.aruco
    prm = ar.DetectorParameters()
    prm.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
    det = ar.ArucoDetector(ar.getPredefinedDictionary(getattr(ar, ANCHOR_DICT)), prm)
    corners, found, _ = det.detectMarkers(gray)
    out: dict[int, dict] = {}
    for c, i in zip(corners, [] if found is None else found.ravel()):
        i = int(i)
        if i not in ids:
            continue
        if i in out:
            raise ValueError(f"마커 id {i} 가 두 번 보인다")
        c = c.reshape(4, 2).astype(float)
        rvec, tvec = _pnp(c, K, dist, size_m)
        out[i] = {"corners": c, "rvec": rvec, "tvec": tvec}
    return out


def anchors_from_frames(grays: list[np.ndarray], K: ArrayLike, dist: ArrayLike, ids: set[int],
                        size_m: float) -> tuple[dict[int, dict], float]:
    """여러 장의 코너를 평균해 PnP 를 다시 푼다 → (마커, 코너 σ px 최대, ddof 1 · 한 장이면 inf).
    한 장이라도 빠진 마커는 뺀다(가려짐 · 흔들림)."""
    dets = [detect_anchors(g, K, dist, ids, size_m) for g in grays]
    out, sd = {}, 0.0 if len(grays) > 1 else math.inf
    for i in sorted(ids):
        if not dets or any(i not in d for d in dets):
            continue
        cs = np.stack([d[i]["corners"] for d in dets])
        if len(cs) > 1:
            sd = max(sd, float(cs.std(0, ddof=1).max()))
        rvec, tvec = _pnp(cs.mean(0), K, dist, size_m)
        out[i] = {"corners": cs.mean(0), "rvec": rvec, "tvec": tvec}
    return out, sd


def _pose(rvec: ArrayLike, tvec: ArrayLike) -> np.ndarray:
    import cv2
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def compare_anchors(now: dict, ref: dict, tol_px: float, tol_mm: float, tol_deg: float) -> dict:
    """기준 마커(ref = 기준 파일 anchors.markers)마다 지금과 비교 → {"markers": {id: {...}}, "pan_shift_px", "ok"}.
    판정: px = 네 코너 이동 거리 평균, mm = 마커 중심 이동(카메라 프레임에서 본 PnP tvec 차, 마커별), deg = 카메라 회전
    (ΔT = T_cam_marker_ref · inv(T_cam_marker_now) 의 회전). cam_mm = ΔT 평행이동(카메라 위치 추정)은 마커 방향 잡음이
    거리만큼 커져 정보로만 찍고 판정에 쓰지 않는다. 기준에 있던 마커가 안 보이면 그 마커는 실패(가려짐 · 떨어짐).
    pan_shift_px = 보인 마커들의 코너 가로(u) 이동 평균. +면 장면이 영상 오른쪽으로 갔다 = 카메라가 기준보다 왼쪽(pan 틱 +)으로
    돌았다 → pan 을 줄여야(오른쪽) 맞는다. tilt_shift_px = 세로(v) 이동 평균(+ = 장면이 아래로). 보인 마커가 없으면 None."""
    res, shifts = {}, []
    for i, r in sorted((int(k), v) for k, v in ref.items()):
        n = now.get(i)
        if n is None:
            res[i] = {"seen": False, "ok": False}
            continue
        d = np.asarray(n["corners"], float) - np.asarray(r["corners"], float)
        dT = _pose(r["rvec"], r["tvec"]) @ np.linalg.inv(_pose(n["rvec"], n["tvec"]))
        deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dT[:3, :3]) - 1) / 2))))
        px = float(np.linalg.norm(d, axis=1).mean())
        mm = float(np.linalg.norm(np.asarray(n["tvec"], float) - np.asarray(r["tvec"], float)) * 1000)
        shifts.append(d.mean(0))
        res[i] = {"seen": True, "px": px, "mm": mm, "deg": deg, "cam_mm": float(np.linalg.norm(dT[:3, 3]) * 1000),
                  "ok": px <= tol_px and mm <= tol_mm and deg <= tol_deg}
    sh = np.mean(shifts, 0) if shifts else (None, None)
    return {"markers": res, "pan_shift_px": None if sh[0] is None else float(sh[0]),
            "tilt_shift_px": None if sh[1] is None else float(sh[1]),
            "ok": bool(res) and all(m["ok"] for m in res.values()), "tol": (tol_px, tol_mm, tol_deg)}


def anchor_report(a: dict) -> list[str]:
    mark = {True: "✓", False: "❌"}
    px, mm, deg = a["tol"]
    lines = []
    for i, m in a["markers"].items():
        if not m["seen"]:
            lines.append(f"  마커 {i} 안 보임(가려짐 · 떨어짐?) ❌")
        else:
            lines.append(f"  마커 {i} 코너 {m['px']:.2f} px · 마커 중심 이동 {m['mm']:.1f} mm · 카메라 회전 {m['deg']:.2f}° "
                         f"허용 {px} px · {mm} mm · {deg}° {mark[m['ok']]} (카메라 위치 추정 {m['cam_mm']:.1f} mm, 판정 안 씀)")
    if a["pan_shift_px"] is not None:
        lines.append(f"  마커 가로 이동 {a['pan_shift_px']:+.2f} px(+ = 카메라가 기준보다 왼쪽)")
    return lines


def pan_px_per_deg(corners: ArrayLike, K: ArrayLike, pitch_deg: float) -> float:
    """pan(세로축 회전) 1° 에 코너가 가로로 움직이는 px 평균(+ = pan 틱 + 일 때 오른쪽).
    숙임 p 인 카메라가 세로축으로 δ 돌면 정규화 좌표 (x, y) 의 가로 이동 ≈ δ(cos p − sin p·y + cos p·x²) 라
    화면 가운데에서도 fx·δ·cos p(68° 숙임이면 0.37 배)이고 아래쪽 마커는 더 작다 — atan(px/fx) 만 쓰면 2~3 배 모자란다."""
    K = np.asarray(K, float)
    c = np.asarray(corners, float).reshape(-1, 2)
    x, y = (c[:, 0] - K[0, 2]) / K[0, 0], (c[:, 1] - K[1, 2]) / K[1, 1]
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    return float(np.mean(K[0, 0] * (cp - sp * y + cp * x * x))) * math.pi / 180


def tilt_px_per_deg(corners: ArrayLike, K: ArrayLike) -> float:
    """tilt 1° 에 코너가 세로로 움직이는 px 평균(+ = tilt 틱 + 일 때 아래). 숙임 = 카메라 x 축 회전이라 정규화 y 이동 = −(1 + y²)δ:
    숙이면 장면이 위로 가서 음수다."""
    K = np.asarray(K, float)
    y = (np.asarray(corners, float).reshape(-1, 2)[:, 1] - K[1, 2]) / K[1, 1]
    return -float(np.mean(K[1, 1] * (1 + y * y))) * math.pi / 180


def anchors_block(markers: dict, n_frames: int, sd: float, size_m: float, image_wh: list[int], fx: float,
                  self_noise: dict | None) -> dict:
    """--record-ref 가 기준 파일에 쓰는 anchors. self_noise = compare_anchors(묶음1, 묶음2) 결과(없으면 None)."""
    r = lambda v, k: [round(float(x), k) for x in np.asarray(v).ravel()]   # noqa: E731
    noise = None if self_noise is None else {
        "pan_shift_px": round(float(self_noise["pan_shift_px"]), 3),
        "markers": {i: {k: round(float(m[k]), 3) for k in ("px", "mm", "deg", "cam_mm")}
                    for i, m in self_noise["markers"].items()}}
    return {"size_m": size_m, "dict": ANCHOR_DICT, "image_wh": [int(v) for v in image_wh], "fx": round(float(fx), 3),
            "n_frames": n_frames, "corner_sd_px": round(sd, 4), "self_noise": noise,
            "markers": {i: {"corners": [r(c, 3) for c in m["corners"]], "rvec": r(m["rvec"], 6), "tvec": r(m["tvec"], 6)}
                        for i, m in sorted(markers.items())}}


def anchor_meta_problem(anc: dict, K: ArrayLike, image_wh: list[int]) -> str | None:
    """기준 마커를 잰 영상과 지금 영상의 크기 · fx 가 다르면 이유(해상도 · 프로필이 바뀌면 px 비교가 무의미)."""
    if list(anc.get("image_wh") or []) != list(image_wh):
        return f"영상 크기 {list(image_wh)} 가 기준 {anc.get('image_wh')} 와 다르다"
    fx = float(np.asarray(K, float)[0, 0])
    if anc.get("fx") is None or abs(fx - float(anc["fx"])) > 0.5:
        return f"fx {fx:.1f} 가 기준 {anc.get('fx')} 와 다르다"
    return None


def quat_to_matrix(wxyz: ArrayLike) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def extrinsics_angles(wxyz: ArrayLike) -> tuple[float, float]:
    """base ← 광학 프레임 회전 → IMU 와 같은 정의의 (숙임, 옆 기울기). base z 가 중력 위쪽이라고 본다."""
    up = quat_to_matrix(wxyz)[2, :]  # base 위쪽을 카메라 프레임으로 = R^T e_z
    return up_to_angles(up)


def load_extrinsics_pitch(path: Path = DEFAULT_EXTRINSICS) -> tuple[float, float]:
    cam = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["camera"]
    return extrinsics_angles(cam["orientation_wxyz"])


def view_error(raw_pitch: float, bias: float, ref_pitch: float, tol: float) -> tuple[float, bool]:
    """(IMU 숙임 − 0점) − 기준 숙임, 그리고 |차| ≤ tol 인가. +면 기준보다 숙였다."""
    err = raw_pitch - bias - ref_pitch
    return err, abs(err) <= tol


# ---------- ★11.17 고정 마커로 외부 파라미터 풀기(--align) ----------

def load_extrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """외부 파라미터 파일 → (R base←광학, t base m)."""
    cam = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["camera"]
    return quat_to_matrix(cam["orientation_wxyz"]), np.asarray(cam["position"], float)


def _corner_offsets(corners: ArrayLike, K: ArrayLike, dist: ArrayLike, size_m: float, R0: np.ndarray,
                    t0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """코너 4개의 base 정답 오프셋(±size/2) · 저장 외부 파라미터로 옮긴 base 중심 xy. 영상 코너가 마커의 어느 모서리인지는 저장 외부
    파라미터로 base 에 옮겨 중심 대비 부호로 정한다(arm4090 logs/runs/261117/verify/vcommon.corner_signs 와 같다)."""
    h = size_m / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    T = _pose(*_pnp(corners, K, dist, size_m))
    pb = (T[:3, :3] @ obj.T + T[:3, 3:]).T @ R0.T + t0
    c = pb[:, :2].mean(0)
    sg = np.sign(pb[:, :2] - c)
    if not np.all(sg) or len({tuple(v) for v in sg}) != 4:
        raise ValueError("마커 코너가 base 의 어느 모서리인지 못 정한다(마커가 base 축과 45° 가까이 돌았거나 저장 외부 파라미터가 크게 틀렸다)")
    return sg * h, c


def solve_extrinsics(corners: dict, K: ArrayLike, dist: ArrayLike, size_m: float, truth: dict, z: float,
                     R0: np.ndarray, t0: np.ndarray) -> dict:
    """고정 마커 코너 px(여러 장 평균) → base←광학 R · t(RGB 만). truth = {id: (x, y 또는 None)} m, 마커 면 z = 상판 z.
    y 가 None 인 마커는 x · z 만 정답이고 y 는 같이 푼다. 초기값 = 저장 외부 파라미터(arm4090 logs/runs/261117/verify/
    s1_extrinsics.py 의 solve 와 같은 방식). y 까지 정답인 마커가 둘 이상 있어야 푼다."""
    import cv2
    from scipy.optimize import least_squares
    K, dist = np.asarray(K, float), np.asarray(dist, float)
    ids = sorted(int(i) for i in corners if int(i) in truth)
    if sum(truth[i][1] is not None for i in ids) < 2:
        raise ValueError(f"위치를 다 아는 마커가 둘 이상 보여야 외부 파라미터를 푼다(보인 정답 마커 {ids})")
    off, ctr = {}, {}
    for i in ids:
        off[i], ctr[i] = _corner_offsets(corners[i], K, dist, size_m, R0, t0)
    free = [i for i in ids if truth[i][1] is None]

    def pts(i: int, p: np.ndarray) -> np.ndarray:
        x, y = truth[i]
        y = p[6 + free.index(i)] if y is None else y
        return np.c_[x + off[i][:, 0], y + off[i][:, 1], np.full(4, z)]

    def res(p: np.ndarray) -> np.ndarray:
        return np.concatenate([(cv2.projectPoints(pts(i, p), p[:3], p[3:6], K, dist)[0].reshape(-1, 2)
                                - np.asarray(corners[i], float)).ravel() for i in ids])

    s = least_squares(res, np.r_[cv2.Rodrigues(R0.T)[0].ravel(), -R0.T @ t0, [ctr[i][1] for i in free]], method="lm")
    R = cv2.Rodrigues(s.x[:3])[0].T
    e = np.linalg.norm(s.fun.reshape(-1, 2), axis=1)
    return {"R": R, "t": -R @ s.x[3:6], "rms_px": float(np.sqrt((e ** 2).mean())), "max_px": float(e.max()), "ids": ids,
            "free_y_m": {i: float(s.x[6 + k]) for k, i in enumerate(free)}}


def extrinsics_diff(R: np.ndarray, t: ArrayLike, R0: np.ndarray, t0: ArrayLike) -> tuple[float, float]:
    """(카메라 위치 차 mm, 회전 차 deg)."""
    c = (np.trace(R @ R0.T) - 1) / 2
    return float(np.linalg.norm(np.asarray(t, float) - t0) * 1000), math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _camera_values(R: np.ndarray, t: ArrayLike) -> tuple[list[float], list[float]]:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = [w, x, y, z] if w >= 0 else [-w, -x, -y, -z]
    return [round(float(v), 6) for v in t], [round(float(v), 6) for v in q]


def write_extrinsics(path: Path, R: np.ndarray, t: ArrayLike, note: str) -> Path | None:
    """외부 파라미터 파일에서 camera 의 position · orientation_wxyz 두 줄만 바꾸고 camera: 위에 note 주석을 단다(다른 값 · 주석은
    그대로). 다시 읽어 그 두 값 말고 바뀐 것이 있으면 쓰지 않고 ValueError. 옛 파일은 .bak-<시각> → 백업 경로."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    pos, quat = _camera_values(R, t)
    lines = text.splitlines(keepends=True)
    i0 = next((k for k, s in enumerate(lines) if s.rstrip() == "camera:"), None)
    if i0 is None:
        raise ValueError(f"{path} 에 camera: 블록이 없다")
    done = set()
    for k in range(i0 + 1, len(lines)):
        s = lines[k]
        if s.strip() and not s.startswith((" ", "\t", "#")):
            break   # camera 블록 끝
        key, ind = s.strip().split(":")[0], s[:len(s) - len(s.lstrip())]
        if key in ("position", "orientation_wxyz") and key not in done:
            lines[k] = f"{ind}{key}: [{', '.join(f'{v:.6f}' for v in (pos if key == 'position' else quat))}]\n"
            done.add(key)
    if done != {"position", "orientation_wxyz"}:
        raise ValueError(f"{path} camera 블록에 position · orientation_wxyz 가 없다")
    new = "".join(lines[:i0] + [f"# {note}\n"] + lines[i0:])
    a, b = yaml.safe_load(text), yaml.safe_load(new)
    if {**a, "camera": {**a["camera"], "position": pos, "orientation_wxyz": quat}} != b:
        raise ValueError(f"{path} 를 고치면 camera 위치 · 회전 말고 다른 값도 바뀐다 — 쓰지 않는다")
    return replace_file(path, new)


RECALIB_EPILOG = """머리를 --execute 로 옮겼으면 그 자세에서 외부 파라미터를 다시 재야 FP++ 좌표가 맞는다(옮긴 자리는 head_pose ±4 틱 밖일 수
있고, 그러면 object_pose_node 가 HeadOffCalibPose 로 발행을 멈춘다). 로봇을 움직이지 않고, 목 · 카메라가 붙은 PC 에서:
  1. 목 드라이버(head_joint_publisher)를 내린다 — --head-config 가 포트를 직접 읽어 틱을 head_pose 에 적는다(포트 단독 소유).
  2. m1117 은 --write 로 덮지 않는다(11.17 마커 값 · 학생 카드 · head_depth_node 가 md5 고정). 새 날짜 파일로 복사한 뒤:
     python3 scripts/calib/table_cad_extrinsics.py --init config/global_camera_extrinsics_arm4090_m1117.yaml \\
             --out config/global_camera_extrinsics_arm4090_<MMDD>.yaml --write --head-config config/head_home_rh56f1_h2048.yaml
     그 파일로 바꿀 때는 미션 생성기 CAMERA_EXTRINSICS · 학생 카드(deploy/policies/rh56f1/*/*/student.yaml) camera md5 ·
     head_depth_node EXTRINSICS_MD5 를 같이 바꾼다.
  3. 새 head_pose 틱에 맞춰 config/head_camset_rh56f1_h2048.yaml 의 deg 를 고친다(tests/s2r_console/test_rh56f1_mission.py 가 비교).
  4. 드라이버를 다시 띄우고 그 자세에서 --record-ref 로 시야 기준을 다시 잡은 뒤 --check 가 ✓ 인지 본다
     (★11.17 --check 는 기준 파일과 직접 비교한다. 절대 중력이 필요하면 --calib-bias 로 IMU 0점도 다시 잰다).
자세한 절차: docs/HOST_arm4090.md "시야 검사"."""


@dataclass(frozen=True)
class Limits:
    min_tilt: float = 60.0     # 인코더 deg — 작을수록 머리가 든다
    max_tilt: float = 74.8
    max_step: float = 10.0
    tol: float = 0.2           # 숙임 deg
    max_iters: int = 12


def next_tilt(enc: float, pitch_err: float, ratio: float, lim: Limits) -> float:
    """남은 숙임 오차(측정 − 목표, +면 너무 숙임) → 다음 인코더 목표. 큰 오차는 max_step 으로 끊고 가까우면 90 % 만 간다."""
    step = pitch_err / ratio
    step = max(-lim.max_step, min(lim.max_step, step if abs(step) > 2.0 else 0.9 * step))
    return max(lim.min_tilt, min(lim.max_tilt, enc - step))


def ratio_from(prev: tuple[float, float], cur: tuple[float, float], fallback: float) -> float:
    """(인코더, 숙임) 두 점 → Δ숙임/Δ인코더. 이동이 0.5° 미만이면 이전 값을 쓴다."""
    d_enc = cur[0] - prev[0]
    return (cur[1] - prev[1]) / d_enc if abs(d_enc) >= 0.5 else fallback


@dataclass
class Step:
    tilt_cmd: float
    enc_pan: float
    enc_tilt: float
    pitch: float       # 제어에 쓴 숙임(기준 파일이 맞으면 보정 전, 아니면 IMU 0점 보정 뒤)
    roll: float
    err: float
    ratio: float
    backoff: bool = False   # 접근 방향 반대로 물린 스텝


# ---------- ROS 부 ----------

class Sensors:
    def __init__(self, node, rclpy) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu, JointState
        from head_fk_chain import encoder_from_urdf
        self._node, self._rclpy, self._e2u = node, rclpy, encoder_from_urdf
        self._acc: list[tuple[float, float, float]] = []
        self.joint: tuple[float, float] | None = None
        node.create_subscription(Imu, IMU_TOPIC, self._on_imu, qos_profile_sensor_data)
        node.create_subscription(JointState, JOINT_TOPIC, self._on_joint, 10)

    def _on_imu(self, m) -> None:
        a = m.linear_acceleration
        self._acc.append((a.x, a.y, a.z))

    def _on_joint(self, m) -> None:
        pos = dict(zip(m.name, m.position))
        self.joint = self._e2u(math.degrees(pos["head_j_pan"]), math.degrees(pos["head_j_tilt"]))

    def read(self, n: int = N_SAMPLES, settle_s: float = 0.5, timeout_s: float = 8.0) -> tuple[float, float, float, float, float]:
        """(숙임, 옆, IMU 흔들림 σ m/s², 인코더 pan, 인코더 tilt) — 멈춘 뒤 n 개 평균."""
        end = time.monotonic() + settle_s
        while time.monotonic() < end:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        self._acc.clear()
        end = time.monotonic() + timeout_s
        while (len(self._acc) < n or self.joint is None) and time.monotonic() < end:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        if len(self._acc) < n // 4:
            raise RuntimeError(f"IMU {IMU_TOPIC} 가 안 들어온다({len(self._acc)} 개) — realsense 를 IMU 켜고 띄울 것")
        if self.joint is None:
            raise RuntimeError(f"{JOINT_TOPIC} 가 안 들어온다 — 목 드라이버를 띄울 것")
        cols = list(zip(*self._acc))
        mean = [statistics.fmean(c) for c in cols]
        sd = max(statistics.pstdev(c) for c in cols)
        pitch, roll = gravity_angles(*mean)
        return pitch, roll, sd, self.joint[0], self.joint[1]


class AnchorCam:
    """컬러 영상 · 내부 파라미터(scripts/vision/cup_view.py 와 같은 토픽 · rgb8/bgr8). grab 이 부를 때만 영상을 모은다
    (Sensors.read 가 같은 노드를 돌리는 동안 1280x720 영상이 쌓이지 않게)."""
    color_topic, info_topic = COLOR_TOPIC, INFO_TOPIC   # __init__ 이 바꾼다(시험은 __init__ 없이 만든다)

    def __init__(self, node, rclpy, color_topic: str = COLOR_TOPIC, info_topic: str = INFO_TOPIC) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        self._node, self._rclpy = node, rclpy
        self.color_topic, self.info_topic = color_topic, info_topic   # ★11.18 글로벌 카메라(scripts/calib/global_cam_extrinsics.py)
        self.K: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self._grays: list[np.ndarray] = []
        self._want, self._bad = 0, None
        node.create_subscription(Image, color_topic, self._on_img, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)

    def _on_info(self, m) -> None:
        self.K, self.dist = np.array(m.k, float).reshape(3, 3), np.array(m.d, float)

    def _on_img(self, m) -> None:
        import cv2
        if len(self._grays) >= self._want:
            return
        if m.encoding not in ("rgb8", "bgr8"):
            self._bad = m.encoding
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        self._grays.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if m.encoding == "rgb8" else cv2.COLOR_BGR2GRAY))

    def grab(self, n: int = ANCHOR_FRAMES, timeout_s: float = 5.0) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        self._grays, self._want, self._bad = [], n, None
        end = time.monotonic() + timeout_s
        while (len(self._grays) < n or self.K is None) and time.monotonic() < end:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        self._want = 0
        if self._bad:
            raise RuntimeError(f"{self.color_topic} 인코딩 {self._bad} — rgb8/bgr8 만 읽는다")
        if self.K is None or len(self._grays) < n:
            raise RuntimeError(f"{self.color_topic} · {self.info_topic} 가 안 들어온다({len(self._grays)}/{n} 장, K "
                               f"{'있음' if self.K is not None else '없음'}) — camera_up.sh 로 카메라를 띄웠는지, 이 셸에서 "
                               "`source scripts/vision/common.sh`(FASTRTPS LAN 프로필 · ROS_DOMAIN_ID)를 했는지 볼 것"
                               "(안 하면 영상이 안 온다)")
        return self._grays, self.K, self.dist


def _anchor_tol(cfg: dict) -> tuple[float, float, float]:
    return float(cfg["ref_tol_anchor_px"]), float(cfg["ref_tol_anchor_mm"]), float(cfg["ref_tol_anchor_deg"])


def _read_anchors(cam: AnchorCam, cfg: dict, n: int = ANCHOR_FRAMES) -> tuple[dict, float, np.ndarray, list[int]]:
    """지금 고정 마커(n 장 평균) → (마커, 코너 σ px, K, 영상 [w, h]). 왜곡 계수는 cam.dist."""
    grays, K, dist = cam.grab(n)
    m, sd = anchors_from_frames(grays, K, dist, {int(i) for i in cfg["anchor_ids"]}, float(cfg["anchor_size_m"]))
    return m, sd, K, list(grays[0].shape[::-1])


def _cam_problem(cam: AnchorCam, cfg: dict, ref_anchors: dict | None, need_markers: bool = True) -> str | None:
    """이동 · 기록 전에 영상을 한 번 받는다: K · 영상 · 설정 마커 전부가 보이나(need_markers, --align 은 camset 에 가기 전이라
    안 본다), 기준과 영상 크기 · fx 가 같나. 영상이 안 오면 grab 이 RuntimeError(main 이 잡아 1)."""
    m, _, K, wh = _read_anchors(cam, cfg)
    missing = [int(i) for i in cfg["anchor_ids"] if int(i) not in m]
    if missing and need_markers:
        return f"❌ 고정 마커 {missing} 가 {ANCHOR_FRAMES} 장 모두에서 보이지는 않는다(가려짐 · 시야 밖?) — 움직이지 않고 멈춘다"
    why = None if ref_anchors is None else anchor_meta_problem(ref_anchors, K, wh)
    return f"❌ {why} — 기준을 잰 영상과 달라 마커 비교를 못 한다. --record-ref 로 다시" if why else None


def _check(ref: dict, up: ArrayLike, ticks: tuple[int, int], n: int, sd: float, ext_md5: str, cfg: dict, ref_name: str,
           cam: AnchorCam | None) -> tuple[int, str]:
    """--check 와 --execute 끝 판정: IMU · 틱 + (기준에 anchors 가 있으면) 고정 마커(영상 크기 · fx 가 기준과 같을 때만)."""
    anchors_now = anchor_tol = None
    if ref.get("anchors") is not None and cam is not None:
        m, _, K, wh = _read_anchors(cam, cfg)
        why = anchor_meta_problem(ref["anchors"], K, wh)
        if why:
            return 1, f"❌ {why} — 기준을 잰 영상과 달라 마커 비교를 못 한다. --record-ref 로 다시"
        anchors_now, anchor_tol = m, _anchor_tol(cfg)
    return run_check(ref, up, ticks, n, sd, ext_md5,
                     (float(cfg["ref_tol_pitch_deg"]), float(cfg["ref_tol_roll_deg"]), float(cfg["ref_tol_tick"])),
                     ref_name, anchors_now, anchor_tol)


def _record_anchors(cam: AnchorCam | None, cfg: dict, force: bool) -> tuple[dict | None, str | None]:
    """--record-ref 의 anchors → (anchors_block 또는 None, 거부 이유). cam None = --no-anchors.
    연속 2 x ANCHOR_FRAMES 장: 전체 평균을 기준으로, 앞 · 뒤 묶음끼리 비교한 값을 self_noise 로. 코너 σ > ANCHOR_MAX_SD_PX 면
    force 없이는 거부."""
    if cam is None:
        return None, None
    grays, K, dist = cam.grab(2 * ANCHOR_FRAMES)
    ids, size = {int(i) for i in cfg["anchor_ids"]}, float(cfg["anchor_size_m"])
    m, sd = anchors_from_frames(grays, K, dist, ids, size)
    missing = sorted(ids - set(m))
    if missing:
        return None, (f"❌ 고정 마커 {missing} 가 {len(grays)} 장 모두에서 보이지는 않는다 — 가려졌는지 보거나, "
                      "마커를 붙이기 전이면 --no-anchors")
    if not sd <= ANCHOR_MAX_SD_PX and not force:
        return None, (f"❌ 코너 σ {sd:.3f} px > {ANCHOR_MAX_SD_PX} px — 카메라 · 머리가 흔들린다. 멈춘 뒤 다시, "
                      "흔들림이 아닌 게 확실하면 --force")
    half = [anchors_from_frames(g, K, dist, ids, size)[0] for g in (grays[:ANCHOR_FRAMES], grays[ANCHOR_FRAMES:])]
    noise = compare_anchors(half[1], half[0], math.inf, math.inf, math.inf)
    return anchors_block(m, len(grays), sd, size, list(grays[0].shape[::-1]), K[0][0], noise), None


def _fit_tilt(sens, head_move, lim: Limits, pan: float, tilt: float, pitch: float, target: float, ratio: float,
              approach: str, ctl_bias: float, steps: list, last_dir: int | None = None) -> tuple:
    """숙임을 target 으로(tilt 만, 접근 방향으로만 들어온다) → (코드, pan, tilt, 숙임, 비율, 마지막 방향, 마지막 tilt 명령)."""
    code, sign, last_cmd = 1, APPROACH_SIGN[approach], None
    for _ in range(lim.max_iters):
        err = pitch - target
        done = abs(err) <= lim.tol
        if done and last_dir != -sign:
            code = 0
            break
        cmd, back = approach_cmd(tilt, tilt if done else next_tilt(tilt, err, ratio, lim), approach, lim, arrived_wrong=done)
        if abs(cmd - tilt) < 0.05:
            print(f"❌ tilt 한계 [{lim.min_tilt}, {lim.max_tilt}] 에 막혔다(남은 오차 {err:+.2f}°)")
            break
        rc, _ = head_move.request_move(pan, cmd, max_tilt=lim.max_tilt)
        if rc == 2:
            print("❌ 드라이버가 요청을 거절했다")
            break
        last_dir, last_cmd = (1 if cmd > tilt else -1), cmd
        raw2, roll, sd, pan2, tilt2 = sens.read()
        pitch2 = raw2 - ctl_bias
        ratio = ratio_from((tilt, pitch), (tilt2, pitch2), ratio)
        steps.append(Step(cmd, pan2, tilt2, pitch2, roll, pitch2 - target, ratio, back))
        print(f"  tilt 목표 {cmd:+.2f}°{'(물림)' if back else ''} 실제 {tilt2:+.2f}° | 숙임 {pitch2:+.2f}° 오차 {pitch2 - target:+.2f}° "
              f"옆 {roll:+.2f}° | 비율 {ratio:.3f}", flush=True)
        pan, tilt, pitch = pan2, tilt2, pitch2  # 멈춰도 최종 보고는 실제로 선 자리
        if not RATIO_OK[0] <= ratio <= RATIO_OK[1]:
            print(f"❌ 비율 {ratio:.3f} — 카메라가 인코더만큼 안 움직인다(풀림 · 미끄러짐?), 그 자리에서 멈춘다")
            break
    return code, pan, tilt, pitch, ratio, last_dir, last_cmd


AXES = ("pan", "tilt")


def _move(head_move, k: int, cmd: float, hold: float, lim: Limits) -> int:
    """axis k(0 pan · 1 tilt) 만 cmd 로, 다른 축은 hold(마지막 명령)를 다시 보낸다 — head_motion 요청은 pan · tilt 를 같이 줘야
    해서(하나만 None 이면 parse_request 가 거절) 한 축만 보내는 길이 없다."""
    return head_move.request_move(*((cmd, hold) if k == 0 else (hold, cmd)), max_tilt=lim.max_tilt)[0]


def _axis_return(sens, head_move, k: int, start: float, cur: float, hold: float, sign: int, lo: float, hi: float,
                 lim: Limits) -> float:
    """시작 각으로 되돌린다 — 접근 방향 규칙대로(넘어선 쪽이면 물렸다가 들어온다) → 실제 각."""
    c = cur
    for _ in range(3):
        c, back = approach_axis(c, start, sign, lo, hi, lim.max_step)
        rc = _move(head_move, k, c, hold, lim)
        if rc != 0:
            print(f"❌ 시작 {AXES[k]} 으로 되돌리기 요청 실패(rc {rc}) — 그 자리에서 멈춘다")
            break
        if not back:
            break
    cur = sens.read()[3 + k]
    print(f"  시작 {AXES[k]} {start:+.2f}° 로 되돌림 → 실제 {cur:+.2f}°")
    return cur


def _fit_axis(sens, cam: AnchorCam, head_move, cfg: dict, ref_markers: dict, lim: Limits, k: int, cur: float, hold: float,
              pitch: float, window: float, tol_px: float, sign: int, center: float | None = None,
              need: list[int] | None = None, deadline: float = math.inf) -> tuple[int, float, float, int]:
    """기준 마커의 평균 이동(k = 0 pan: 가로 u, k = 1 tilt: 세로 v)을 tol_px 안으로: 그 축만 움직인다
    → (코드, 그 축 실제 각, 그 축 마지막 명령, 이동 수). cur = 그 축 실제 각, hold = 다른 축 마지막 명령.
    sign 쪽으로만 들어오고(넘어서면 물렸다가 다시), 목표가 center(기본 cur) ± window(tilt 는 [min_tilt, max_tilt] 안) 밖이면 1.
    이득(px/°)을 실제 이동(0.5° 이상)으로 재기 전에는 한 번에 PAN_PROBE_DEG 까지만, 잰 뒤에는 lim.max_step 까지 간다.
    이득이 기하값(pan_px_per_deg · tilt_px_per_deg)의 0.33~3 배 밖이거나 부호가 반대면 시작 각으로 되돌리고 1.
    need(기본 기준 마커 전부)가 하나라도 안 보이거나 이동 요청이 rc ≠ 0(1 = 실패 · 시간 초과, 2 = 거절)이면 그 자리에서 멈추고 1.
    남은 보정이 반 틱(TICK_DEG / 2)보다 작으면 맞은 것으로 본다(더 움직여도 못 줄인다)."""
    name = AXES[k]
    center = cur if center is None else center
    lo, hi = center - window, center + window
    if k == 1:
        lo, hi = max(lo, lim.min_tilt), min(hi, lim.max_tilt)
    need = sorted(int(i) for i in ref_markers) if need is None else need
    start, last_cmd, moves = cur, cur, 0
    prev, gain, verified, last_dir = None, None, False, None
    for _ in range(lim.max_iters):
        if time.monotonic() > deadline:
            print(f"❌ 시간 상한 — {name} 을 맞추다 멈춘다")
            return 1, cur, last_cmd, moves
        now, _, K, _ = _read_anchors(cam, cfg)
        a = compare_anchors(now, ref_markers, math.inf, math.inf, math.inf)
        unseen = [i for i in need if not a["markers"].get(i, {}).get("seen")]
        if unseen:
            print(f"❌ 기준 마커 {unseen} 가 안 보인다 — {name} 을 맞출 수 없어 그 자리에서 멈춘다")
            return 1, cur, last_cmd, moves
        shift = a["pan_shift_px" if k == 0 else "tilt_shift_px"]
        seen = np.concatenate([np.asarray(now[i]["corners"], float) for i in sorted(now) if i in ref_markers])
        geo = pan_px_per_deg(seen, K, pitch) if k == 0 else tilt_px_per_deg(seen, K)
        if prev is not None and abs(cur - prev[0]) >= 0.5:   # 실제로 돈 만큼으로 이득을 잰다(부호 포함)
            gain, verified = (shift - prev[1]) / (cur - prev[0]), True
        elif gain is None:
            gain = geo
        if geo * (1 if k == 0 else -1) < 1.0 or not 0.33 <= gain / geo <= 3.0:   # 마커가 이 축에 거의 안 움직이는 자리거나 기구 · 부호가 틀렸다
            print(f"❌ {name} 이득 {gain:.2f} px/° (기하 {geo:.2f}) — 기구 · 마커 이상 의심, 멈춘다")
            if last_dir is not None:
                cur = _axis_return(sens, head_move, k, start, cur, hold, sign, lo, hi, lim)
            return 1, cur, last_cmd, moves
        done = abs(shift) <= tol_px or abs(shift / gain) < TICK_DEG / 2
        if done and last_dir != -sign:
            print(f"  {name} 맞음: 마커 {'가로' if k == 0 else '세로'} 이동 {shift:+.2f} px(허용 ±{tol_px:g}) · 접근 방향으로 도착")
            return 0, cur, last_cmd, moves
        goal = cur if done else cur - shift / gain
        if not lo <= goal <= hi:
            print(f"❌ {name} 목표 {goal:+.2f}° 가 창 [{lo:+.2f}, {hi:+.2f}] 밖(마커 이동 {shift:+.2f} px) — 멈춘다")
            return 1, cur, last_cmd, moves
        step = lim.max_step if verified else PAN_PROBE_DEG
        cmd, back = approach_axis(cur, cur + max(-step, min(step, goal - cur)), sign, lo, hi, step, arrived_wrong=done)
        if abs(cmd - cur) < 0.02:
            print(f"❌ {name} 창 [{lo:+.2f}, {hi:+.2f}] 에 막혔다(마커 이동 {shift:+.2f} px)")
            return 1, cur, last_cmd, moves
        rc = _move(head_move, k, cmd, hold, lim)
        if rc != 0:
            print(f"❌ {name} 이동 요청 실패(rc {rc}: 1 = 이동 실패 · 시간 초과, 2 = 거절) — 그 자리에서 멈춘다")
            return 1, cur, last_cmd, moves
        last_dir, prev, last_cmd, moves = (1 if cmd > cur else -1), (cur, shift), cmd, moves + 1
        cur = sens.read()[3 + k]
        print(f"  {name} 목표 {cmd:+.2f}°{'(물림)' if back else ''} 실제 {cur:+.2f}° | 마커 이동(이동 전) {shift:+.2f} px "
              f"이득 {gain:.2f} px/°{'' if verified else '(기하값, 확인 전)'}", flush=True)
    print(f"❌ {name} 반복 상한")
    return 1, cur, last_cmd, moves


def _fit_pan(sens, cam: AnchorCam, head_move, cfg: dict, ref: dict, lim: Limits, pan: float, tilt_hold: float,
             pitch: float, window: float) -> tuple[int, float]:
    """--execute: 기준 마커의 pan_shift_px 를 허용치(ref_tol_anchor_px) 안으로 pan 만(_fit_axis k = 0) → (코드, pan).
    기준 approach_pan 쪽으로만 들어오고 시작 pan ± window 안에서만 움직인다. tilt 는 tilt_hold(마지막 tilt 명령)를 다시 보낸다."""
    code, pan, _, _ = _fit_axis(sens, cam, head_move, cfg, ref["anchors"]["markers"], lim, 0, pan, tilt_hold, pitch, window,
                                _anchor_tol(cfg)[0], PAN_APPROACH_SIGN[ref.get("approach_pan", "left")])
    return code, pan


def _goto(sens, head_move, lim: Limits, goal: tuple[float, float], signs: tuple[int, int]) -> tuple[int, list[float]]:
    """camset (pan, tilt) 로 — 두 축 모두 접근 방향으로 들어온다. 접근 쪽으로 APPROACH_BACKOFF_DEG / 2 넘게 떨어져 있지 않으면
    (반대쪽 · 이미 그 자리라 도착 방향을 모름) 먼저 APPROACH_BACKOFF_DEG 물렸다가 들어온다 → (코드, 마지막 명령 [pan, tilt])."""
    c, done = list(sens.read()[3:5]), [False, False]
    lims = ((-math.inf, math.inf), (lim.min_tilt, lim.max_tilt))
    for _ in range(3):
        for k in (0, 1):
            if not done[k]:
                c[k], back = approach_axis(c[k], goal[k], signs[k], *lims[k], 360.0,
                                           arrived_wrong=(goal[k] - c[k]) * signs[k] < APPROACH_BACKOFF_DEG / 2)
                done[k] = not back
        rc, _ = head_move.request_move(c[0], c[1], max_tilt=lim.max_tilt)
        print(f"  camset 으로: 명령 pan {c[0]:+.2f}° tilt {c[1]:+.2f}°{'' if all(done) else '(물림)'}", flush=True)
        if rc != 0:
            print(f"❌ camset 이동 요청 실패(rc {rc}: 1 = 이동 실패 · 시간 초과, 2 = 거절) — 그 자리에서 멈춘다")
            return 1, c
        if all(done):
            return 0, c
    print("❌ camset 에 접근 방향으로 못 들어왔다(tilt 한계에 막힘?)")
    return 1, c


def _align_extrinsics(cam: AnchorCam, cfg: dict, truth: dict, ext_path: Path, write: bool) -> tuple[int, dict]:
    """--align c: 지금 자세에서 2 x ANCHOR_FRAMES 장 코너 평균 → 외부 파라미터 풀이 → 저장 값과 비교(→ 코드, 기록).
    허용치 밖이면 차이를 찍고 1, write 면 옛 파일을 .bak-<시각> 으로 남기고 camera 위치 · 회전만 새로 쓴다(코드 0)."""
    size, z = float(cfg["anchor_size_m"]), float(cfg["anchor_truth"]["table_z_m"])
    tol_mm, tol_deg = float(cfg.get("extr_tol_mm", 3.0)), float(cfg.get("extr_tol_deg", 0.3))
    m, sd, K, _ = _read_anchors(cam, cfg, 2 * ANCHOR_FRAMES)
    dist = cam.dist
    if not sd <= ANCHOR_MAX_SD_PX:
        print(f"❌ 코너 σ {sd:.3f} px > {ANCHOR_MAX_SD_PX} px — 카메라 · 머리가 흔들린다. 외부 파라미터를 판정하지 않는다")
        return 1, {"corner_sd_px": round(sd, 3)}
    R0, t0 = load_extrinsics(ext_path)
    s = solve_extrinsics({i: v["corners"] for i, v in m.items()}, K, dist, size, truth, z, R0, t0)
    dmm, ddeg = extrinsics_diff(s["R"], s["t"], R0, t0)
    pos, quat = _camera_values(s["R"], s["t"])
    e = {"ids": s["ids"], "rms_px": round(s["rms_px"], 3), "max_px": round(s["max_px"], 3), "corner_sd_px": round(sd, 3),
         "d_pos_mm": round(dmm, 2), "d_xyz_mm": [round(float(v) * 1000, 2) for v in s["t"] - t0], "d_rot_deg": round(ddeg, 3),
         "tol": [tol_mm, tol_deg], "position": pos, "orientation_wxyz": quat,
         "free_y_mm": {str(i): round(v * 1000, 1) for i, v in s["free_y_m"].items()}, "ok": dmm <= tol_mm and ddeg <= tol_deg,
         "written": None}
    print(f"  외부 파라미터(마커 {s['ids']}, 재투영 rms {s['rms_px']:.2f} px): 저장 값과 위치 {dmm:.1f} mm {e['d_xyz_mm']} · "
          f"회전 {ddeg:.3f}° (허용 {tol_mm} mm · {tol_deg}°) {'✓' if e['ok'] else '❌'}")
    if s["rms_px"] > ALIGN_MAX_RMS_PX:
        print(f"❌ 재투영 rms {s['rms_px']:.2f} px > {ALIGN_MAX_RMS_PX} — 마커 정답(anchor_truth) · 크기 · 코너가 맞지 않는다. 판정 · 쓰기 안 함")
        return 1, e
    if e["ok"]:
        return 0, e
    if not write:
        print(f"❌ 외부 파라미터가 {ext_path.name} 와 허용치 넘게 다르다 — 새 값 position {pos} · orientation_wxyz {quat}. "
              "덮어쓰려면 --write-extrinsics")
        return 1, e
    bak = write_extrinsics(ext_path, s["R"], s["t"], f"★{time.strftime('%Y-%m-%d %H:%M')} head_view_feedback.py --align "
                           f"--write-extrinsics: 마커 {s['ids']} 로 camera 를 다시 풀었다(rms {s['rms_px']:.2f} px, 옛 값과 "
                           f"{dmm:.1f} mm · {ddeg:.3f}°)")
    e["written"] = None if bak is None else bak.name
    print(f"  {ext_path} 를 고쳤다(옛 파일 {e['written']}). md5 가 바뀌어 --check 는 실패한다 — 이 자세에서 --record-ref 로 기준을 다시 잡을 것")
    return 0, e


def _align(sens, cam: AnchorCam, head_move, cfg: dict, ref: dict, lim: Limits, goal: tuple[float, float], window: float,
           ext_path: Path, write_ext: bool) -> tuple[int, dict]:
    """--align: a camset 으로 → b 마커 px 를 기준에 맞춘다(tilt 세로 · pan 가로 번갈아, 접근 방향으로만) → c 외부 파라미터 풀이 ·
    비교 → d IMU 경고(실패 아님) → (코드, --out 기록). y 까지 정답인 마커(70 · 71)는 꼭 보여야 하고, x 만 정답인 마커(73)는
    안 보이면 경고하고 뺀다."""
    deadline = time.monotonic() + ALIGN_TIMEOUT_S
    markers = {int(i): v for i, v in ref["anchors"]["markers"].items()}
    truth = {int(i): (float(v[0]), None if v[1] is None else float(v[1])) for i, v in cfg["anchor_truth"]["markers"].items()}
    need = sorted(i for i in markers if i in truth and truth[i][1] is not None)
    tol = float(cfg.get("align_tol_px", 1.0))
    signs = (PAN_APPROACH_SIGN[ref.get("approach_pan", "left")], APPROACH_SIGN[ref.get("approach", "down")])
    rec: dict = {"mode": "align", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "camset_deg": list(goal), "align_tol_px": tol}
    code, cmd = _goto(sens, head_move, lim, goal, signs)
    raw, _, _, pan, tilt = sens.read()
    a = None
    for _ in range(lim.max_iters):   # 라운드: tilt(세로) → pan(가로) — 한 축이 다른 축 이동을 조금 흔든다
        if code:
            break
        if time.monotonic() > deadline:
            print(f"❌ 시간 상한 {ALIGN_TIMEOUT_S:.0f} s")
            code = 1
            break
        code, tilt, cmd[1], n1 = _fit_axis(sens, cam, head_move, cfg, markers, lim, 1, tilt, cmd[0], raw, window, tol / 2,
                                           signs[1], goal[1], need, deadline)
        n0 = 0
        if code == 0:
            code, pan, cmd[0], n0 = _fit_axis(sens, cam, head_move, cfg, markers, lim, 0, sens.read()[3], cmd[1], raw, window,
                                              tol / 2, signs[0], goal[0], need, deadline)
        if code:
            break
        a = compare_anchors(_read_anchors(cam, cfg)[0], markers, tol, math.inf, math.inf)
        if all(v["ok"] for v in a["markers"].values() if v["seen"]):
            break
        if n0 + n1 == 0:
            print(f"❌ 마커 px 가 {tol} px 안에 안 들어오는데 pan · tilt 로 더 못 줄인다 — 남은 차는 카메라 위치 · 옆 기울기"
                  "(마운트 · 처짐)")
            code = 1
            break
    else:
        code = 1
        print("❌ 라운드 상한")
    if a is not None:
        rec["marker_px"] = {str(i): round(v["px"], 3) if v["seen"] else None for i, v in a["markers"].items()}
        rec["missing"] = [i for i, v in a["markers"].items() if not v["seen"]]
        print(f"  마커 px 오차 {rec['marker_px']} (허용 {tol} px) {'✓' if code == 0 else '❌'}")
        if rec["missing"]:
            print(f"⚠ 마커 {rec['missing']} 가 안 보여 {sorted(set(markers) - set(rec['missing']))} 로만 맞췄다(x 만 정답인 마커라 판정은 계속)")
    rec["px_ok"] = code == 0
    if code == 0:
        try:
            code, rec["extrinsics"] = _align_extrinsics(cam, cfg, truth, ext_path, write_ext)
        except (RuntimeError, ValueError) as e:   # 영상 끊김 · 정답 마커 가려짐 · 풀이 · 쓰기 거부 — 기록은 남긴다
            print(f"❌ 외부 파라미터: {e}")
            code, rec["extrinsics"] = 1, {"error": str(e)}
    raw2, roll2, _, pan, tilt = sens.read()
    pr, rr = up_to_angles(ref["imu_up_cam"])
    warn_deg = float(cfg.get("imu_warn_deg", 1.0))
    rec["imu"] = {"d_pitch_deg": round(raw2 - pr, 3), "d_roll_deg": round(roll2 - rr, 3), "warn_deg": warn_deg,
                  "warn": max(abs(raw2 - pr), abs(roll2 - rr)) > warn_deg}
    if rec["imu"]["warn"]:
        print(f"⚠ IMU 가 기준과 숙임 {raw2 - pr:+.2f}° · 옆 {roll2 - rr:+.2f}° 다르다(> {warn_deg}°) — 카메라 · 마운트를 건드렸는지 "
              "볼 것(경고만, 판정은 마커로 한다)")
    ticks = (deg_to_tick(pan), deg_to_tick(tilt))
    hp = yaml.safe_load(Path(ext_path).read_text(encoding="utf-8")).get("head_pose") or {}
    if "pan_tick" in hp:   # object_pose_node 는 head_pose ± tol_tick 밖이면 발행을 멈춘다(HeadOffCalibPose)
        rec["head_pose_d_tick"] = [ticks[0] - int(hp["pan_tick"]), ticks[1] - int(hp["tilt_tick"])]
        if max(map(abs, rec["head_pose_d_tick"])) > int(hp.get("tol_tick", 4)):
            print(f"⚠ 틱 {ticks[0]}/{ticks[1]} 이 외부 파라미터 head_pose {hp['pan_tick']}/{hp['tilt_tick']} ± {hp.get('tol_tick', 4)} 밖 "
                  "— FP++ 노드가 HeadOffCalibPose 로 멈출 수 있다")
    rec.update(final_ticks=list(ticks), final_deg=[round(pan, 3), round(tilt, 3)], last_cmd_deg=[round(v, 3) for v in cmd],
               ok=code == 0, rc=code)
    print(("✓" if code == 0 else "❌") + f" --align 끝: 틱 {ticks[0]}/{ticks[1]} · 접근 tilt {ref.get('approach', 'down')} · "
          f"pan {ref.get('approach_pan', 'left')}")
    return code, rec


# ---------- ★11.17 --auto-extrinsics: camset 에서 마커로 카메라 자세를 다시 잰다(목 피드백 맞춤 없음) ----------

LIVE_HEADER = ("# 생성됨 — scripts/head_view_feedback.py --execute --auto-extrinsics. 손으로 고치지 않는다(시작마다 다시 확인 · 풂).\n"
               "# camera = camset 자세에서 테이블 고정 마커로 푼 값(provenance.base 와 extr_tol 안이면 base 값 그대로),\n"
               "# head_pose = 그때 /head/joint_states 틱. 나머지 키는 base 와 같다. provenance = 출처 · 풀이 · 명령 틱 · IMU.\n")


def _atomic_write(path: Path, data: str | bytes) -> None:
    """임시 파일 → os.replace(쓰다 끊겨도 반쯤 남지 않게). 런타임 파일이라 .bak 는 남기지 않는다."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    os.replace(tmp, path)


def _take_live(live: Path, live_ref: Path) -> dict:
    """★11.17 리뷰: 시작하자마자 live · 재기준을 지운다(신호 · SIGKILL 로 죽어도 옛 값이 남지 않게). kept 판정용 이전 바이트를 돌려준다."""
    prev = {}
    for k, f in (("live", live), ("ref", live_ref)):
        if f.exists():
            prev[k] = f.read_bytes()
            f.unlink()
    return prev


def _stamp(t: object) -> float | None:
    try:
        return time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S")) if isinstance(t, str) else None
    except ValueError:
        return None


def live_problem(live: str | Path, base_md5: str | None = None, live_ref: str | Path | None = None,
                 max_age_h: float | None = None, now: float | None = None) -> str | None:
    """live 외부 파라미터를 FP++ 가 써도 되나 → None, 아니면 이유. 비지 않은 yaml · camera · provenance 가 있고,
    provenance.base_md5 가 base(기본 = 같은 폴더의 provenance.base 파일) md5 와 같고, 마지막 확인(provenance.time 과 재기준
    verified_at 중 늦은 쪽, 재기준은 extrinsics.md5 가 이 live 와 맞을 때만)이 max_age_h(기본 설정 live_max_age_h, 12) 안."""
    live = Path(live)
    if not live.exists() or live.stat().st_size == 0:
        return f"{live.name} 가 없거나 비었다"
    try:
        d = yaml.safe_load(live.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        return f"{live.name} 를 못 읽는다: {e}"
    prov = d.get("provenance") if isinstance(d, dict) else None
    if not isinstance(prov, dict) or not isinstance(d.get("camera"), dict):
        return f"{live.name} 에 camera · provenance 가 없다(--auto-extrinsics 가 쓴 파일이 아니다)"
    if base_md5 is None:
        base = live.with_name(str(prov.get("base")))
        if not base.is_file():
            return f"base {base.name} 가 없다"
        base_md5 = file_md5(base)
    if prov.get("base_md5") != base_md5:
        return f"{live.name} 가 다른 base 로 만들어졌다(provenance {prov.get('base')} {prov.get('base_md5')} ≠ {base_md5})"
    if max_age_h is None:
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) if DEFAULT_CONFIG.exists() else {}
        max_age_h = float((cfg or {}).get("live_max_age_h", 12.0))
    stamps = [_stamp(prov.get("time"))]
    ref = Path(live_ref) if live_ref is not None else live.with_name(DEFAULT_LIVE_REF.name)
    if ref.is_file():
        try:
            lr = yaml.safe_load(ref.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, UnicodeDecodeError):
            lr = {}
        if isinstance(lr, dict) and (lr.get("extrinsics") or {}).get("md5") == file_md5(live):
            stamps.append(_stamp(lr.get("verified_at")))
    stamps = [t for t in stamps if t is not None]
    if not stamps:
        return f"{live.name} 의 확인 시각이 없다"
    age_h = ((time.time() if now is None else now) - max(stamps)) / 3600
    if age_h > max_age_h:
        return f"{live.name} 의 마지막 확인이 {age_h:.1f} h 전이다(허용 live_max_age_h {max_age_h:g} h)"
    return None


def require_fresh_live(path: str | Path | None) -> None:
    """FP++ 런처 · 물체 자세 노드 시작 검사: *_live.yaml 이면 live_problem 이 있을 때 시작을 거부한다(base 를 직접 주면 검사 없음)."""
    if path is None or not str(path).endswith("_live.yaml"):
        return
    why = live_problem(path)
    if why:
        raise SystemExit(f"❌ {why} — 미션의 head_view_check(시야 검사 · 외부 파라미터 자동 갱신)를 다시 돌려라")


def live_cli(argv: list[str]) -> int:
    """원격 인지 미션이 가져온 live 를 mv 하기 전 검사: [live.tmp, live_ref.tmp, 이 PC 의 base] → 0 / 1(이유 출력)."""
    live, ref, base = argv
    why = live_problem(live, file_md5(Path(base)), ref)
    if why:
        print(f"❌ 가져온 live 를 쓰지 않는다 — {why}")
    return 1 if why else 0


def live_text(base: Path, camera: dict | None, ticks: tuple[int, int], prov: dict) -> str:
    """base 외부 파라미터 → live 내용: camera(None = base 값 그대로) · head_pose 틱만 바꾸고 provenance 를 붙인다(같은 형식)."""
    d = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    if camera is not None:
        d["camera"] = {**d["camera"], **camera}
    d["head_pose"] = {**(d.get("head_pose") or {}), "pan_tick": int(ticks[0]), "tilt_tick": int(ticks[1])}
    d["provenance"] = prov
    return LIVE_HEADER + yaml.safe_dump(d, allow_unicode=True, sort_keys=False)


def view_problem(m: dict, need: list[int], ref_m: dict, wh: list[int], margin: float) -> tuple[list[int], list[int]]:
    """need 마커 중 (잘림, 가림). 잘림 = 보이는데 코너가 가장자리 margin px 안쪽이 아니거나, 안 보이는데 기준 px 를 보인 마커의
    평균 이동만큼 옮긴 자리가 그 밖. 가림 = 안 보이는데 그 자리가 화면 안쪽(팔 · 손 · 물건). ref_m = 기준 anchors.markers(int 키)."""
    lo, hi = np.array([margin, margin]), np.array([wh[0] - margin, wh[1] - margin])
    seen = [i for i in ref_m if i in m]
    shift = (np.mean([np.mean(np.asarray(m[i]["corners"], float) - np.asarray(ref_m[i]["corners"], float), 0) for i in seen], 0)
             if seen else np.zeros(2))
    cut, hidden = [], []
    for i in need:
        c = np.asarray(m[i]["corners"], float) if i in m else np.asarray(ref_m[i]["corners"], float) + shift
        if not np.all((c >= lo) & (c <= hi)):
            cut.append(i)
        elif i not in m:
            hidden.append(i)
    return cut, hidden


def _live_kept(prev: dict, base: Path, cfg: dict, s: dict, ticks: tuple[int, int]) -> str | None:
    """live 를 다시 쓰지 않아도 되나 → None, 아니면 이유. 재기준이 live md5 · 설정 마커와 맞고(실측 · 마커 기준), live 가 같은 base 로
    만들어졌고, 지금 푼 값이 live 와 extr_tol 안, 틱이 재기준과 ref_tol_tick 안이어야 한다."""
    if "live" not in prev or "ref" not in prev:
        return "live · 재기준이 없다"
    lr = yaml.safe_load(prev["ref"].decode("utf-8"))
    why = ref_problem(lr, hashlib.md5(prev["live"]).hexdigest(), cfg) or (None if lr.get("measured") is True and lr.get("anchors") else
                                                   "재기준이 실측 · 마커 기준이 아니다")
    if why:
        return why
    d = yaml.safe_load(prev["live"].decode("utf-8"))
    prov = d.get("provenance") or {}
    if prov.get("base_md5") != file_md5(base):
        return f"live 가 다른 base 로 만들어졌다({prov.get('base')} {prov.get('base_md5')})"
    dmm, ddeg = extrinsics_diff(s["R"], s["t"], quat_to_matrix(d["camera"]["orientation_wxyz"]),
                                np.asarray(d["camera"]["position"], float))
    tol_mm, tol_deg = float(cfg.get("extr_tol_mm", 3.0)), float(cfg.get("extr_tol_deg", 0.3))
    if dmm > tol_mm or ddeg > tol_deg:
        return f"지금 푼 값이 live 와 {dmm:.1f} mm · {ddeg:.3f}° 다르다(허용 {tol_mm} mm · {tol_deg}°)"
    dt = max(abs(ticks[0] - int(lr["pan_tick"])), abs(ticks[1] - int(lr["tilt_tick"])))
    return f"틱이 재기준과 {dt} 다르다(허용 {cfg.get('ref_tol_tick', 4)})" if dt > int(cfg.get("ref_tol_tick", 4)) else None


def _auto_extrinsics(sens, cam: AnchorCam, head_move: object | None, cfg: dict, ref: dict, lim: Limits, goal: tuple[float, float],
                     base: Path, live: Path, live_ref: Path, prev: dict | None = None) -> tuple[int, dict]:
    """--auto-extrinsics(★11.17 사용자: 마커는 테이블에 고정된 정답이고 움직이는 건 카메라다 — 목을 기준 화면으로 되돌리지 않고
    지금 화면에서 카메라 자세를 다시 잰다): camset 명령을 보내고(접근 방향 규칙 _goto, head_move None = --execute 없음 → 목을
    움직이지 않고 지금 자세에서 잰다 · head_pose 는 어느 쪽이든 실제 틱) auto_settle_s 기다린 뒤,
    설정 anchor_ids 가 2 x ANCHOR_FRAMES 장 모두 가장자리 auto_edge_margin_px 안쪽에 보이면 그 코너로 외부 파라미터를 푼다.
    잘림이면 바로 1(목을 움직이지 않는다), 가림이면 auto_occlusion_wait_s 동안 다시 잡는다. 코너 σ ≤ ANCHOR_MAX_SD_PX ·
    rms ≤ ALIGN_MAX_RMS_PX · base 대비 auto_max_mm · auto_max_deg 안일 때만 live 를 쓰고(base 와 extr_tol 안이면 base camera 그대로),
    같은 코너 · IMU · 틱으로 live_ref(재기준)를 쓴다. 재기준과 같으면(_live_kept) live 를 다시 쓰지 않는다(md5 안정).
    base(m1117) · ref 는 읽기만 한다. live · live_ref 는 시작하자마자 지우고(prev = main 이 이미 지우며 읽은 바이트) 성공일 때만
    새로 쓴다 — kept 면 같은 바이트(md5 불변)에 재기준 verified_at 만 갱신. 실패하면 1."""
    if prev is None:
        prev = _take_live(live, live_ref)
    size, z = float(cfg["anchor_size_m"]), float(cfg["anchor_truth"]["table_z_m"])
    margin, wait = float(cfg.get("auto_edge_margin_px", 5.0)), float(cfg.get("auto_occlusion_wait_s", 10.0))
    amm, adeg = float(cfg.get("auto_max_mm", 20.0)), float(cfg.get("auto_max_deg", 2.0))
    tol_mm, tol_deg = float(cfg.get("extr_tol_mm", 3.0)), float(cfg.get("extr_tol_deg", 0.3))
    need = sorted(int(i) for i in cfg["anchor_ids"])
    truth = {int(i): (float(v[0]), None if v[1] is None else float(v[1])) for i, v in cfg["anchor_truth"]["markers"].items()}
    ref_m = {int(i): v for i, v in ref["anchors"]["markers"].items()}
    signs = (PAN_APPROACH_SIGN[ref.get("approach_pan", "left")], APPROACH_SIGN[ref.get("approach", "down")])
    rec: dict = {"mode": "auto_extrinsics", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "camset_deg": list(goal),
                 "ok": False, "rc": 1}
    if head_move is None:
        print("  --execute 없음 — 목을 움직이지 않고 지금 자세에서 잰다", flush=True)
        cmd = None
    else:
        code, cmd = _goto(sens, head_move, lim, goal, signs)
        if code:
            return 1, rec
        time.sleep(float(cfg.get("auto_settle_s", 1.0)))
    end, told = time.monotonic() + wait, False
    while True:
        m, sd, K, wh = _read_anchors(cam, cfg, 2 * ANCHOR_FRAMES)
        cut, hidden = view_problem(m, need, ref_m, wh, margin)
        if cut or not hidden or time.monotonic() > end:
            break
        if not told:
            print(f"  마커 {hidden} 가 {2 * ANCHOR_FRAMES} 장 모두에 보이지는 않는다(가림?) — {wait:g} s 까지 다시 잡는다", flush=True)
            told = True
    rec.update(cut=cut, hidden=hidden, corner_sd_px=round(sd, 3))
    if cut:
        print(f"❌ camset 자세에서 마커 {cut} 가 화면 밖(가장자리 {margin:g} px 여유 밖) — 목을 움직이지 않는다. "
              "카메라 마운트 · camset 을 볼 것")
        return 1, rec
    if hidden:
        print(f"❌ 마커 {hidden} 가 {wait:g} s 동안 가려졌다(화면 안인데 검출 안 됨 — 팔 · 손 · 물건?)")
        return 1, rec
    if not sd <= ANCHOR_MAX_SD_PX:
        print(f"❌ 코너 σ {sd:.3f} px > {ANCHOR_MAX_SD_PX} px — 카메라 · 머리가 흔들린다")
        return 1, rec
    R0, t0 = load_extrinsics(base)
    try:
        s = solve_extrinsics({i: v["corners"] for i, v in m.items()}, K, cam.dist, size, truth, z, R0, t0)
    except ValueError as err:   # 코너 모서리를 못 정함 · 정답 마커 부족
        print(f"❌ 외부 파라미터 풀이: {err}")
        return 1, rec
    dmm, ddeg = extrinsics_diff(s["R"], s["t"], R0, t0)
    pos, quat = _camera_values(s["R"], s["t"])
    raw, roll, acc_sd, pan, tilt = sens.read()
    ticks = (deg_to_tick(pan), deg_to_tick(tilt))
    e = {"ids": s["ids"], "rms_px": round(s["rms_px"], 3), "max_px": round(s["max_px"], 3), "corner_sd_px": round(sd, 3),
         "d_pos_mm": round(dmm, 2), "d_xyz_mm": [round(float(v) * 1000, 2) for v in s["t"] - t0], "d_rot_deg": round(ddeg, 3),
         "free_y_mm": {str(i): round(v * 1000, 1) for i, v in s["free_y_m"].items()}}
    rec.update(extrinsics=e, cmd_deg=None if cmd is None else [round(v, 3) for v in cmd],
               cmd_ticks=None if cmd is None else [deg_to_tick(v) for v in cmd], ticks=list(ticks))
    print(f"  외부 파라미터(마커 {s['ids']}, rms {s['rms_px']:.2f} px, σ {sd:.3f} px): {base.name} 대비 {dmm:.1f} mm "
          f"{e['d_xyz_mm']} · {ddeg:.3f}° | 틱 {ticks[0]}/{ticks[1]}(명령 {rec['cmd_ticks']})")
    if s["rms_px"] > ALIGN_MAX_RMS_PX:
        print(f"❌ 재투영 rms {s['rms_px']:.2f} px > {ALIGN_MAX_RMS_PX} — 마커 정답(anchor_truth) · 크기 · 코너가 맞지 않는다")
        return 1, rec
    if dmm > amm or ddeg > adeg:
        print(f"❌ {base.name} 와 {dmm:.1f} mm · {ddeg:.3f}° — 자동 상한 auto_max {amm:g} mm · {adeg:g}° 를 넘는다"
              "(카메라가 크게 움직였거나 마커가 옮겨졌다). live 를 쓰지 않는다")
        return 1, rec
    why = _live_kept(prev, base, cfg, s, ticks)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if why is None:
        lr = yaml.safe_load(prev["ref"].decode("utf-8"))
        lr["verified_at"] = now
        _atomic_write(live, prev["live"])
        _atomic_write(live_ref, ref_text(lr))
        rec.update(live="kept", live_md5=file_md5(live), ok=True, rc=0)
        print(f"✓ 재기준과 같다 — {live.name} 그대로(md5 {rec['live_md5'][:8]}) · 확인 시각 {now}")
        return 0, rec
    as_base = dmm <= tol_mm and ddeg <= tol_deg
    prov = {"base": base.name, "base_md5": file_md5(base), "time": now, "camera": "base" if as_base else "solved",
            "marker_ids": s["ids"], "rms_px": e["rms_px"], "corner_sd_px": e["corner_sd_px"], "d_pos_mm": e["d_pos_mm"],
            "d_xyz_mm": e["d_xyz_mm"], "d_rot_deg": e["d_rot_deg"], "cmd_deg": rec["cmd_deg"], "cmd_ticks": rec["cmd_ticks"],
            "ticks": list(ticks), "imu_pitch_raw_deg": round(raw, 3), "imu_roll_raw_deg": round(roll, 3)}
    _atomic_write(live, live_text(base, None if as_base else {"position": pos, "orientation_wxyz": quat}, ticks, prov))
    anchors = anchors_block(m, 2 * ANCHOR_FRAMES, sd, size, wh, K[0][0], None)
    new_ref = build_ref(angles_to_up(raw, roll), ticks, len(sens._acc), acc_sd, live.name, file_md5(live),
                        ref.get("approach", "down"), now, "--auto-extrinsics(live 를 쓴 자세의 마커 · IMU · 틱)",
                        ref.get("approach_pan", "left"), anchors)
    _atomic_write(live_ref, ref_text({**new_ref, "verified_at": now}))
    rec.update(live="written", live_md5=file_md5(live), camera=prov["camera"], ok=True, rc=0)
    print(f"✓ {live.name} 를 썼다(camera {prov['camera']}, 이유: {why}) · 재기준 {live_ref.name}")
    return 0, rec


def sweep_targets(start: float, step: float, lim: Limits) -> list[float]:
    """start 에서 step 씩 한계까지(한계 자체 포함)."""
    if step == 0:
        raise ValueError("step 이 0 이다")
    out, t = [], start
    while True:
        t = max(lim.min_tilt, min(lim.max_tilt, t + step))
        if out and abs(t - out[-1]) < 1e-9 or abs(t - start) < 1e-9 and not out:
            return out
        out.append(t)


def grid_plan(pans: list[float], tilts: list[float]) -> list[tuple[float, float, str]]:
    """pan 마다 tilts 순서대로(up) 갔다가 거꾸로(down). 같은 자리 두 번 재기는 뺀다. (pan, tilt, 방향) 목록."""
    plan = []
    for pan in pans:
        plan += [(pan, t, "up") for t in tilts]
        plan += [(pan, t, "down") for t in tilts[-2::-1]]
    return plan


def _grid(args, lim: Limits, sens, head_move, bias: float) -> int:
    pans = [float(v) for v in args.grid_pans.split(",")]
    tilts = [float(v) for v in args.grid_tilts.split(",")]
    for t in tilts:
        if not lim.min_tilt <= t <= lim.max_tilt:
            raise SystemExit(f"tilt {t} 가 [{lim.min_tilt}, {lim.max_tilt}] 밖")
    plan = grid_plan(pans, tilts)
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    prev = None
    print(f"{len(plan)} 자세 · pan {pans} · tilt {tilts[0]}→{tilts[-1]} 양방향", flush=True)
    for i, (pan, tilt, d) in enumerate(plan):
        rc, _ = head_move.request_move(pan, tilt, max_tilt=lim.max_tilt)
        if rc == 2:
            print("❌ 드라이버가 요청을 거절했다"); return 1
        raw, roll, sd, ep, et = sens.read()
        row = {"i": i, "pan_cmd": pan, "tilt_cmd": tilt, "dir": d, "enc_pan": round(ep, 3), "enc_tilt": round(et, 3),
               "pitch": round(raw - bias, 3), "roll": round(roll, 3), "sd": round(sd, 4), "sag": round(raw - bias - et, 3)}
        if out:
            out.write(json.dumps(row) + "\n"); out.flush()
        print(f"[{i + 1}/{len(plan)}] pan {ep:+6.2f} tilt {et:+6.2f} {d:4} 숙임 {row['pitch']:+6.2f} 처짐 {row['sag']:+5.2f} 옆 {roll:+5.2f}",
              flush=True)
        if prev and prev["pan_cmd"] == pan and abs(et - prev["enc_tilt"]) >= 2:
            r = (row["pitch"] - prev["pitch"]) / (et - prev["enc_tilt"])
            if not 0.5 <= r <= 1.5:
                print(f"❌ 비율 {r:.3f} — 기구 이상 의심, 멈춘다"); return 1
        prev = row
    return 0


def _sweep(args, lim: Limits, sens, head_move, pan: float, tilt: float, pitch: float, bias: float) -> int:
    rows = [{"tilt_cmd": None, "enc_tilt": tilt, "pitch": pitch}]
    print(f"{'목표':>8} {'인코더':>8} {'숙임':>8} {'구간 비율':>8} {'누적 비율':>8}")
    print(f"{'-':>8} {tilt:+8.2f} {pitch:+8.2f}")
    code = 0
    for cmd in sweep_targets(tilt, args.sweep, lim):
        rc, _ = head_move.request_move(pan, cmd, max_tilt=lim.max_tilt)
        if rc == 2:
            print("❌ 드라이버가 요청을 거절했다"); code = 1; break
        raw, roll, sd, pan2, tilt2 = sens.read()
        p2 = raw - bias
        r = ratio_from((rows[-1]["enc_tilt"], rows[-1]["pitch"]), (tilt2, p2), float("nan"))
        r0 = ratio_from((rows[0]["enc_tilt"], rows[0]["pitch"]), (tilt2, p2), float("nan"))
        rows.append({"tilt_cmd": cmd, "enc_tilt": tilt2, "enc_pan": pan2, "pitch": p2, "roll": roll, "sd": sd, "ratio": r})
        print(f"{cmd:+8.2f} {tilt2:+8.2f} {p2:+8.2f} {r:8.3f} {r0:8.3f}  옆 {roll:+.2f}", flush=True)
        if not 0.5 <= r <= 1.5:
            print(f"❌ 구간 비율 {r:.3f} — 기구 이상 의심, 멈춘다"); code = 1; break
        if p2 <= args.sweep_stop_pitch:
            print(f"✓ 숙임 {p2:+.2f}° ≤ {args.sweep_stop_pitch}° — 정면을 지났다"); break
    if args.out:
        Path(args.out).write_text(json.dumps({"bias": bias, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return code


def _die(signum: int, _frame: object) -> None:
    raise SystemExit(1)   # SIGTERM · SIGHUP(콘솔 중지 · ssh 끊김)도 finally 를 타서 live 를 지우게


def main(argv: list[str] | None = None) -> int:
    """★11.17 --auto-extrinsics: 시작하자마자 live · 재기준을 지우고(이전 바이트는 메모리로 — kept 판정용) 성공일 때만 새로 쓴다.
    어디서든(인자 · 기준 · 영상 · 풀이 · 신호) 실패하면 다시 지운다 — FP++ 가 이번에 확인하지 못한 값을 쓰지 않게(실패 닫힘)."""
    auto = "--auto-extrinsics" in (sys.argv[1:] if argv is None else argv)
    if not auto:
        return _main(argv)
    prev = _take_live(DEFAULT_LIVE, DEFAULT_LIVE_REF)
    old = {sig: signal.signal(sig, _die) for sig in (signal.SIGTERM, signal.SIGHUP)}
    rc = 1
    try:
        rc = _main(argv, prev)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        raise
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
        if rc != 0:
            for f in (DEFAULT_LIVE, DEFAULT_LIVE_REF):
                if f.exists():
                    f.unlink()
            print(f"  {DEFAULT_LIVE.name} · {DEFAULT_LIVE_REF.name} 없음(이번 확인 실패 — 이 파일을 읽는 FP++ · 물체 자세 노드는 "
                  "다음 성공까지 못 뜬다)")
    return rc


def _main(argv: list[str] | None = None, prev: dict | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0], epilog=RECALIB_EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("--target", default="extrinsics", help="'extrinsics'(sim 시야) 또는 숙임 deg")
    p.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="imu_pitch_bias 를 읽고(--calib-bias 면 쓰는) 파일")
    p.add_argument("--ref", default=str(DEFAULT_REF), help="--check · --record-ref 기준 파일")
    p.add_argument("--check", action="store_true",
                   help="움직이지 않고 시야만 검사 — 기준 파일의 보정 전 IMU 숙임 · 옆 기울기 · 목 틱과 비교해 설정 ref_tol_* 를 "
                        "넘거나 기준이 다른 외부 파라미터(md5)로 잡혔으면 종료 코드 1. bias 는 쓰지 않는다")
    p.add_argument("--record-ref", action="store_true",
                   help="움직이지 않고 지금 IMU 평균 · 틱으로 기준 파일을 새로 쓴다(외부 파라미터를 막 잰 자세에서만, measured true)")
    p.add_argument("--approach", choices=sorted(APPROACH_SIGN), default="down",
                   help="--record-ref: 그 자세에 마지막으로 도착한 tilt 방향(down = 숙이며, 기본). --execute 는 기준의 이 방향으로만 들어온다")
    p.add_argument("--force", action="store_true",
                   help="--record-ref: 새 IMU 각이 (외부 파라미터 각 + 설정 bias) 와 1° 넘게 달라도, 마커 코너 σ 가 "
                        f"{ANCHOR_MAX_SD_PX} px 를 넘어도 쓴다")
    p.add_argument("--no-anchors", action="store_true",
                   help="--record-ref: 고정 마커(설정 anchor_ids) 없이 IMU · 틱만 기록(마커를 붙이기 전). 기본은 마커가 다 보여야 쓴다")
    p.add_argument("--approach-pan", choices=sorted(PAN_APPROACH_SIGN), default="left",
                   help="--record-ref: --execute 가 pan 을 마커로 맞출 때 들어오는 방향(left = pan 틱이 커지며, 기본)")
    p.add_argument("--pan-window", type=float, default=3.0,
                   help="--execute: 마커로 pan 을 맞출 때 시작 pan ± 이 범위(deg, 0 < w ≤ 5) 안에서만 움직인다")
    p.add_argument("--execute", action="store_true", help="목을 움직인다(사용자 허락 뒤에만)")
    p.add_argument("--align", action="store_true",
                   help="★11.17 시작 루틴(--execute 와 같이, 목이 움직인다): --camset 자세로 → 고정 마커 px 가 기준(--ref anchors)에서 "
                        "설정 align_tol_px 안에 올 때까지 tilt · pan 을 고친다 → 마커로 외부 파라미터를 풀어 --extrinsics 와 "
                        "extr_tol_mm · extr_tol_deg 로 비교 → IMU 가 기준과 imu_warn_deg 넘게 다르면 경고만. 하나라도 실패면 종료 코드 1")
    p.add_argument("--camset", default=str(DEFAULT_CAMSET), help="--align 이 처음 보내는 목 자세(motors.pan · tilt deg)")
    p.add_argument("--auto-extrinsics", action="store_true",
                   help="★11.17 시작 루틴(--execute 와 같이, 목이 --camset 명령값으로 한 번 간다 — 마커 px 피드백 맞춤 없음): 설정 "
                        "auto_settle_s 기다린 뒤 고정 마커(anchor_ids)로 외부 파라미터를 풀어 rms · σ · base(--extrinsics) 대비 "
                        "auto_max_mm · auto_max_deg 안이면 live 파일(config/global_camera_extrinsics_arm4090_live.yaml)과 재기준"
                        "(config/head_view_ref_arm4090_live.yaml)을 쓴다. 재기준과 같으면 다시 안 쓴다. 마커가 화면 밖이면 목을 "
                        "움직이지 않고 1, 가림은 auto_occlusion_wait_s 기다린다. 실패하면 live · 재기준을 지운다. --extrinsics 는 안 쓴다")
    p.add_argument("--write-extrinsics", action="store_true",
                   help="--align: 푼 외부 파라미터가 허용치 밖이면 --extrinsics 의 camera 위치 · 회전을 새 값으로 쓴다(옛 파일 .bak-<시각>)")
    p.add_argument("--allow-bias-target", action="store_true",
                   help="--execute --target extrinsics 인데 기준 파일을 못 쓸 때(없음 · md5 · measured false) 옛 방식(IMU 0점 bias)으로 맞춘다")
    p.add_argument("--calib-bias", action="store_true", help="지금 자세 = sim 시야라 보고 IMU 0점을 다시 재서 설정에 쓴다. bias 는 절대 중력(테이블 수평 대비)용이고 "
                        "--check 는 쓰지 않는다(검사는 기준 파일의 보정 전 IMU 와 직접 비교)")
    p.add_argument("--min-tilt", type=float, default=Limits.min_tilt)
    p.add_argument("--max-tilt", type=float, default=Limits.max_tilt)
    p.add_argument("--max-step", type=float, default=Limits.max_step)
    p.add_argument("--tol", type=float, default=Limits.tol)
    p.add_argument("--out", help="기록 JSON 경로")
    p.add_argument("--sweep", type=float, metavar="STEP",
                   help="처짐 지도: 지금 tilt 에서 STEP°씩(음수 = 머리를 든다) --min-tilt/--max-tilt 까지 가며 인코더 · 숙임을 기록. "
                        "숙임이 --sweep-stop-pitch 를 넘거나 비율이 0.5~1.5 밖이면 멈춘다")
    p.add_argument("--grid", action="store_true",
                   help="보정표 측정: --grid-pans 각 pan 에서 --grid-tilts 를 순서대로(든다) 갔다가 거꾸로(숙인다) 돌아오며 기록. "
                        "--out 에 한 줄씩(jsonl) 바로 쓴다 — 중간에 끊겨도 남는다")
    p.add_argument("--grid-pans", default="-30,-20,-10,0,10,20,30")
    p.add_argument("--grid-tilts", default="74.8,70,65,60,55,50,45,40,35,30,25,20,15,10,5,0,-2")
    p.add_argument("--sweep-stop-pitch", type=float, default=-2.0, help="sweep 이 멈추는 숙임(보정 뒤 deg, 기본 정면 −2°)")
    args = p.parse_args(argv)
    if not 0 < args.pan_window <= 5:
        p.error(f"--pan-window {args.pan_window} 는 0 < w ≤ 5 여야 한다")
    if args.align and (not args.execute or args.check or args.record_ref or args.calib_bias or args.grid or args.sweep):
        p.error("--align 은 목을 움직인다 — --execute 와 같이, --check · --record-ref · --calib-bias · --grid · --sweep 없이")
    if args.auto_extrinsics and (args.align or args.check or args.record_ref or args.calib_bias or args.grid or args.sweep
                                 or args.write_extrinsics):
        p.error("--auto-extrinsics 는 혼자 쓴다(--execute 만 같이) — --align · --check · --record-ref · --calib-bias · --grid · "
                "--sweep · --write-extrinsics 없이")
    if args.auto_extrinsics and Path(args.extrinsics).resolve() in (DEFAULT_LIVE.resolve(), DEFAULT_LIVE_REF.resolve()):
        p.error("--auto-extrinsics 의 --extrinsics 는 base(m1117)다 — live 파일을 base 로 주지 않는다")
    if args.write_extrinsics and not args.align:
        p.error("--write-extrinsics 는 --align 에서만 쓴다")

    ext_pitch, ext_roll = load_extrinsics_pitch(Path(args.extrinsics))
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    bias = float(cfg.get("imu_pitch_bias", 0.0))
    ref_path = Path(args.ref)
    ext_md5 = file_md5(Path(args.extrinsics))
    align = args.align or args.auto_extrinsics   # 기준 · 마커 · camset 준비가 같다
    feedback = args.execute and not (args.check or args.record_ref or args.calib_bias or args.grid or args.sweep or align)
    ref = None   # 기준 파일은 --check · --execute · --align 에서만 읽는다(깨진 파일이어도 --record-ref 는 돈다)
    if args.check or feedback or align:
        ref = yaml.safe_load(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    if args.check:
        missing = [k for k in ("ref_tol_pitch_deg", "ref_tol_roll_deg", "ref_tol_tick") if k not in cfg]
        if missing:
            raise SystemExit(f"❌ {cfg_path} 에 {missing} 가 없다 — 검사 허용치를 모른다")
        why = ref_problem(ref, ext_md5, cfg)
        if why:   # ROS 를 띄우기 전에 끝낸다(run_check 도 같은 판정)
            print(f"❌ {ref_path}: {why} — 외부 파라미터를 잰 자세에서 --record-ref 필요")
            return 1
    if align:   # ROS 를 띄우기 전에 끝낸다
        why = (ref_problem(ref, ext_md5, cfg) or (None if ref.get("measured") is True else "기준이 실측이 아니다(measured false)")
               or (None if ref.get("anchors") is not None else "기준에 고정 마커 px 가 없다(--no-anchors 로 잡은 기준)"))
        if why:
            print(f"❌ --align 기준 {ref_path} 를 못 쓴다: {why} — 외부 파라미터를 잰 자세에서 마커가 다 보일 때 --record-ref")
            return 1
        truth = (cfg.get("anchor_truth") or {}).get("markers") or {}
        if "table_z_m" not in (cfg.get("anchor_truth") or {}) or sum(v[1] is not None for v in truth.values()) < 2:
            raise SystemExit(f"❌ {cfg_path} 의 anchor_truth(table_z_m · 위치를 다 아는 마커 둘 이상)가 없다")
        camset = yaml.safe_load(Path(args.camset).read_text(encoding="utf-8"))["motors"]
        goal = (float(camset["pan"]["deg"]), float(camset["tilt"]["deg"]))
    use_ref = False
    ref_anchors = isinstance(ref, dict) and ref.get("anchors") is not None
    need_cam = ((args.check or feedback or align) and ref_anchors) or (args.record_ref and not args.no_anchors)
    if need_cam:
        keys = ["anchor_ids", "anchor_size_m"] + (["ref_tol_anchor_px", "ref_tol_anchor_mm", "ref_tol_anchor_deg"] if ref_anchors else [])
        missing = [k for k in keys if k not in cfg]
        if missing:
            raise SystemExit(f"❌ {cfg_path} 에 {missing} 가 없다 — 고정 마커 id · 크기 · 허용치를 모른다")
    if feedback and args.target == "extrinsics":
        why = ref_problem(ref, ext_md5, cfg) or (None if ref.get("measured") is True else "기준이 실측이 아니다(measured false)")
        if why and not args.allow_bias_target:
            print(f"❌ --target extrinsics 기준 {ref_path} 를 못 쓴다: {why}. 외부 파라미터를 잰 자세에서 --record-ref 하거나, "
                  "옛 방식(IMU 0점 bias 로 외부 파라미터 숙임)으로 맞추려면 --allow-bias-target")
            return 1
        use_ref = why is None   # 보정 전 숙임끼리 맞춘다(bias 안 뺌)
    ref_anchors = ref_anchors and (use_ref or not feedback)   # bias 방식으로 내려가면 깨진 기준의 마커는 안 쓴다
    need_cam = need_cam and (ref_anchors or args.record_ref)
    approach = ref.get("approach", "down") if isinstance(ref, dict) and use_ref else "down"
    ctl_bias = 0.0 if use_ref else bias
    if args.target != "extrinsics":
        target = float(args.target)
    else:
        target = up_to_angles(ref["imu_up_cam"])[0] if use_ref else ext_pitch
    lim = Limits(args.min_tilt, args.max_tilt, args.max_step, args.tol)

    import rclpy
    from rclpy.node import Node
    import head_move
    rclpy.init()
    node = Node("head_view_feedback")
    try:
        sens = Sensors(node, rclpy)
        cam = AnchorCam(node, rclpy) if need_cam else None
        raw_pitch, roll, sd, pan, tilt = sens.read()
        n = len(sens._acc)
        print(f"지금: 인코더 pan {pan:+.2f}° tilt {tilt:+.2f}° | IMU 숙임 {raw_pitch:+.2f}°(보정 뒤 {raw_pitch - bias:+.2f}°) "
              f"옆 {roll:+.2f}° σ {sd:.3f} n {n} | 외부 파라미터 숙임 {ext_pitch:+.2f}° 옆 {ext_roll:+.2f}° | "
              f"목표 {target:+.2f}°({'기준 파일, 보정 전' if use_ref else '0점 보정 뒤'})")
        if cam is not None:   # ★영상 확인은 어떤 이동 · 기록보다 먼저(안 오면 머리를 움직이지 않는다)
            why = _cam_problem(cam, cfg, ref["anchors"] if ref_anchors else None, need_markers=not align)
            if why:
                print(why)
                return 1
        if args.auto_extrinsics:
            rc, rec = _auto_extrinsics(sens, cam, head_move if args.execute else None, cfg, ref, lim, goal, Path(args.extrinsics),
                                       DEFAULT_LIVE, DEFAULT_LIVE_REF, prev)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rc
        if args.align:
            rc, rec = _align(sens, cam, head_move, cfg, ref, lim, goal, args.pan_window, Path(args.extrinsics), args.write_extrinsics)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rc
        ticks = (deg_to_tick(pan), deg_to_tick(tilt))
        up = angles_to_up(raw_pitch, roll)
        if args.check:
            rc, text = _check(ref, up, ticks, n, sd, ext_md5, cfg, ref_path.name, cam)
            print(text)
            return rc
        if args.record_ref:
            hp = yaml.safe_load(Path(args.extrinsics).read_text(encoding="utf-8"))["head_pose"]
            ok, text = record_ref_guard(up, ticks, n, sd, hp, (ext_pitch, ext_roll),
                                        (bias, float(cfg.get("imu_roll_bias", 0.0))), args.force)
            print(text)
            if not ok:
                return 1
            anchors, why = _record_anchors(cam, cfg, args.force)
            if why:
                print(why)
                return 1
            write_ref(ref_path, build_ref(up, ticks, n, sd, Path(args.extrinsics).name, ext_md5, args.approach,
                                          time.strftime("%Y-%m-%d %H:%M"),
                                          "--record-ref(외부 파라미터를 잰 자세에서 IMU 평균 · 틱" + (" · 고정 마커" if anchors else "")
                                          + ")" + (" --force" if args.force else ""), args.approach_pan, anchors))
            print(f"기준 {ref_path} 를 썼다: 숙임 {raw_pitch:.3f}° 옆 {roll:+.3f}°(보정 전) 틱 {ticks[0]}/{ticks[1]} "
                  f"접근 {args.approach} · pan 접근 {args.approach_pan} · measured true · "
                  + (f"마커 {sorted(anchors['markers'])} 코너 σ {anchors['corner_sd_px']} px" if anchors else "마커 없음(--no-anchors)"))
            return 0
        if args.calib_bias:
            new = raw_pitch - ext_pitch
            cfg.update(imu_pitch_bias=round(new, 3), imu_roll_bias=round(roll - ext_roll, 3),
                       bias_measured_at={"pan_deg": round(pan, 3), "tilt_deg": round(tilt, 3),
                                         "time": time.strftime("%Y-%m-%d %H:%M"), "extrinsics": Path(args.extrinsics).name})
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"IMU 0점 {bias:+.3f} → {new:+.3f}° 를 {cfg_path} 에 썼다")
            return 0
        pitch = raw_pitch - ctl_bias
        steps: list[Step] = []
        ratio = float(cfg.get("ratio_guess", 0.98))
        if args.grid and args.execute:
            return _grid(args, lim, sens, head_move, bias)
        if args.sweep and args.execute:
            return _sweep(args, lim, sens, head_move, pan, tilt, raw_pitch - bias, bias)   # 처짐 지도는 절대 중력(bias)
        if not args.execute:
            print(f"남은 오차 {pitch - target:+.2f}° → 다음 tilt {next_tilt(tilt, pitch - target, ratio, lim):+.2f}°(--execute 없음, 움직이지 않는다)")
            return 0
        sign = APPROACH_SIGN[approach]
        # last_dir = 마지막 명령 방향(+1 숙임 · −1 듦). 인코더 실제 이동은 백래시로 작을 수 있어 명령으로 본다
        code, pan, tilt, pitch, ratio, last_dir, tilt_cmd = _fit_tilt(sens, head_move, lim, pan, tilt, pitch, target, ratio,
                                                                      approach, ctl_bias, steps)
        pan_fit = use_ref and ref_anchors and args.target == "extrinsics"
        if code == 0 and pan_fit:   # ★11.17 IMU 가 못 보는 좌우를 고정 마커 가로 이동으로
            code, pan = _fit_pan(sens, cam, head_move, cfg, ref, lim, pan, tilt if tilt_cmd is None else tilt_cmd,
                                 pitch + ctl_bias, args.pan_window)
            if code == 0:   # pan 이 tilt 를 흔들 수 있어 한 번 더(같은 접근 방향 상태에서)
                raw2, _, _, pan, tilt = sens.read()
                code, pan, tilt, pitch, ratio, last_dir, _ = _fit_tilt(sens, head_move, lim, pan, tilt, raw2 - ctl_bias, target,
                                                                       ratio, approach, ctl_bias, steps, last_dir)
        tick = deg_to_tick(tilt)
        approach_ok = None if last_dir is None else last_dir == sign
        arrived = {None: "움직이지 않음(도착 방향 모름)", True: f"{approach} 로 도착", False: f"{approach} 반대로 도착"}[approach_ok]
        print(("✓" if code == 0 else "❌") + f" 끝: 인코더 tilt {tilt:+.2f}°(틱 {tick}) 숙임 {pitch:+.2f}° 목표 {target:+.2f}° · {arrived}")
        if args.out:
            Path(args.out).write_text(json.dumps({"target": target, "bias": ctl_bias, "uses_ref": use_ref, "approach": approach,
                                                  "approach_ok": approach_ok, "ok": code == 0, "final_tilt_deg": tilt,
                                                  "final_tick": tick, "steps": [asdict(s) for s in steps]},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
        if use_ref and args.target == "extrinsics":   # --check 와 같은 판정(움직이지 않음)
            raw2, roll2, sd2, pan2, tilt2 = sens.read()
            rc, text = _check(ref, angles_to_up(raw2, roll2), (deg_to_tick(pan2), deg_to_tick(tilt2)), len(sens._acc), sd2,
                              ext_md5, cfg, ref_path.name, cam)
            print("판정(--check 와 같음):\n" + text)
            code = code or rc
        return code
    except (RuntimeError, ValueError) as e:   # 토픽 없음 · 같은 마커 id 두 번 · 범위 밖 요청 등
        print(f"❌ {e}")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
