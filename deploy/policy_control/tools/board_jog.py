#!/usr/bin/env python3
"""보드 조그 기록 — 창(PyQt5)에서 한 팔을 조그해 검지 끝을 탁자 위 보드 목표점(ChArUco 교차점 · ArUco GridBoard 마커 모서리)에 대고, 접촉한 채 관절각을 기록한다.

11.05 사용자: 이 기록으로 관절 영점 · 손끝 오프셋 · 보드 자세를 함께 푼다(풀이는 따로). 기록 형식 = 아래 JSON 계약 board_jog_v1.

정밀 모드(--control precise): 120 Hz, 수동 명목 경로 최대 0.30 rad/s · 0.60 rad/s²의 정지 시작/종료 5차 경로.
기본은 발행 시작 때 현재 PD 목표와 일치시킨 보정값 c를 고정한다. 경로 완료 후 발행 목표도 고정된다.
--continuous_hold 실험 모드에서만 실측 손 FK 기준으로 연속 적분한다(K=2/s, 변화율 0.03 rad/s · 크기 0.12 rad).
위 가속도 제한은 명목 경로에만 해당한다. 실제 발행 cmd+c에는 보정이 더해지므로 같은 가속도 제한을 보장하지 않는다.
접촉 · 측정/힘 끊김 · 하강 때는 보정을 얼린다. 기존 모터 PD·중력보상은 바꾸지 않는다. 실기 성능 미검증.
손 POINT는 기존 변환표의 명령 레지스터 범위를 반영한다(오른손 index 0 요청 → 약 0.064 rad).
보드 초기 z=CAD205+표시오프셋12 mm는 이번 보정 결과나 독립 실측 높이가 아니다.
발행 길은 teach_place · touch_probe 와 같다: pd 가 그 팔을 TRACKING 으로 붙든 채 /policy_control/joint_target 에 팔 7 + 손 6 을 60 Hz 로 낸다.

    deploy/policy_control/tools/board_jog.sh --side right --out logs/analysis/fk_board_jog_1105/rec/right_1.json            # dry(발행 0)
    deploy/policy_control/tools/board_jog.sh --side right --grip 1.57,0,0,1.4,1.4,1.4 \\
        --out logs/analysis/fk_board_jog_1105/rec/right_1.json --execute       # ★실기 — 사용자 승인 뒤, 창의 '실기 발행' 체크도 따로 켠다
    deploy/policy_control/tools/board_jog.sh --side right --boards charuco,grid50,grid50b --board_pose boards.yaml \\
        --out logs/analysis/fk_board_jog_1105/rec/right_2.json                 # 보드 여럿(boards.yaml = {grid50: {x, y, yaw_deg}, …})
    python3 deploy/policy_control/tools/board_jog.py --side right --demo        # ROS 없이 가짜 팔 · 힘으로 창만
    .venv/bin/python deploy/policy_control/tools/board_jog.py --selftest        # ROS · 화면 없이(마지막 줄 'selftest ok', cv2 4.14 로 보드 대조)

발행 안전: --execute 없이는 발행 0(조그는 가상 명령만 바꾸고 회색 골격으로 그린다). --execute 여도 창의 '실기 발행' 체크를 켜기 전까지 0.
체크를 켜는 순간 시작 검사 통과 · pd TRACKING(pd_fault) · 측정 신선 · pd 세트포인트(arm_hold_target) · 정지 처짐 e0(start_e0)를 본다.
명령 cmd = 조그 · IK 가 정한 관절, 뜻한 관절 q_int = cmd + e0(실측 공간). 발행 = cmd + c(유지 보정, 11.05 사용자 "입력한 관절 상태를 유지").
유지 보정은 기본적으로 발행 시작의 c를 고정한다. --continuous_hold에서만 팔 7 관절 c를 계속 적분한다.
--no_hold는 초기 c도 쓰지 않는 기존 e0 방식이다. 고정 c 모드는 자세에 따른 정상상태 추종 오차를 없애지 않는다.
유지 켬이면 e0 = 0(발행을 켤 때 잰 처짐은 c 첫값 −e0 로 흡수)이라 q_int = cmd = 명령 6D 손끝 자세의 IK 해다.
연속 적분 모드에서만 자세마다 다른 처짐을 c로 보정한다. 이동 중 · 끝난 뒤 HOLD_SETTLE_S · 접촉(+ 뒤 REC_WIN) · 힘 끊김 ·
하강 전 정지 · 하강 · 자동 approach · hover · descend · settle · 발행 꺼짐(dry) · 측정 끊김 · c 가 발행 손끝을 z 하한 아래로 더 내릴 때는
c 를 그 자리에 얼린다(0 으로 되돌리지 않는다). 자동 확인 대기(위 10 mm 정지)에서는 돈다 — 상태 띠 '손끝 실측−명령' 이 작아진 뒤 Enter.
발행 중 매 틱: 팔 측정 0.5 s 끊김 · pd 이상 · 추종 오차 |실측 − q_int| > TRACK_TOL_MOVE 면 명령 = 실측 − e0(발행 = 그 + c)로 해제하고
episode stop(touch_probe.release_and_stop) 뒤 체크를 끈다. 창 닫기 · Ctrl-C · SIGTERM · SIGHUP 도 같은 절차로 끝낸다.
다른 팔에는 아무것도 보내지 않는다(이름 · 에피소드 토픽 모두 --side 쪽만).

창(board_jog_gui.py): 이동 패드(전진 +x · 후진 −x · 왼쪽 +y · 오른쪽 −y · 위 +z · 아래 −z, 방향키 · PageUp/Down), 회전 패드, 상태 띠, 목표 · 기록.
sim 창(Isaac)에는 --sim_port(기본 47813) 로 30 Hz UDP JSON(sim_packet)을 보낸다 — 송신 실패는 제어에 영향 없다.
조그: 버튼 한 번 = 목표 자세 한 단계(누르고 있으면 이동이 끝날 때마다 다음 단계, 쌓지 않는다). 손 관절은 고정하고 팔 7 관절을 수치 자코비안 + 감쇠 최소제곱(DLS)으로 푼다(ik). 해가 관절 한계
(− LIMIT_MARGIN_RAD) 밖 · 수렴 실패 · 한 단계 관절 변화 IK_JUMP_RAD 초과면 거부. 명령은 관절당 0.15 rad/s 로 옮기고, 이동 중 증분은 받지 않는다.
접촉 표시등(검지 관절 힘 · 촉각, touch_probe.Detector)이 켜졌거나 힘이 끊겼으면 손끝을 base −z 로 내리는 증분은 거부한다(위로는 허용).
이동 중 접촉이 새로 켜지거나, 접촉한 채 시작한 이동에서 힘이 시작값보다 FORCE_G · TIP_N 넘게 커지면 멈추고 명령 = 실측 − e0.
손끝 z 하한(H1): 표시 보드면 z − 8 mm. 조그 · 하강 · 복귀 · 손 바꾸기 모두 보간 경로 FK 샘플이 하한 아래로 가면 거부(이미 아래면 더 내려가는 것만).
시작 자세 복귀는 접촉 중이면 거부한다(먼저 위로 빼기).
천천히 하강: 0.5 s 정지(e0_local · 힘 기준선) → base −z 3 mm/s, 접촉 즉시 명령 = 실측 − e0 로 누름을 푼다. 40 mm 또는 하한에서 멈춘다.
기록: 접촉 표시등 + 최근 0.5 s 접촉 비율 ≥ 90 % + 팔 속도 < 0.01 rad/s 일 때만. 기록마다 --out JSON 을 원자적으로 덮어쓴다(임시 → rename).
모든 틱은 10 분마다, 그리고 종료 때 같은 이름 _raw_partNN.npz 조각으로 내보낸다(메모리에서 비움).

보드: --boards charuco,grid50,grid50b … (BOARDS). charuco = 7x5 ChArUco 교차점 24 개, gridNN[b] = 2x2 ArUco GridBoard
(hand_arm_aruco_A4.pdf 4쪽, ids 50-53 오른팔 · 54-57 왼팔, 같은 것 두 장이라 b 붙은 것이 둘째 장) 마커 네 모서리 16 개.

4 모서리 → 자동(11.05 사용자 "최외각 직사각형 4점만 내가 맞춰주고, 나머진 점들은 자동화"): 보드마다 바깥 4 모서리(BOARDS[*]["outer"],
charuco = bc00 · bc05 · bc18 · bc23)를 사람이 조그로 기록하면 FK 손끝 4 점으로 보드 자세를 강체 맞춤(board_auto.fit_board, Kabsch ·
반사 금지)한다. 관문(board_auto): 대각선 · 네 변 길이 차 > 4 mm 경고 · > 10 mm 거부, 잔차 최대 > 3 mm 거부, 기울기 > 5° 거부,
표시 자세(카메라 · 입력 추정) yaw 와 > 90° 다르면 거부(180° 뒤집힌 순서). 맞춤에는 aligned_by human_corner · human 점만 쓴다(auto 점은
기록 · 분석용). 사람 점이 8 개면 전체로 다시 맞추되 같은 잔차 관문을 걸고, 4 모서리 대각선 · 변 검사도 계속 건다. 받아들인 맞춤이 창 · sim 의
보드 표시 자세가 되고, 손끝 z 하한을 max(기존, 맞춤 면 최저 − 8 mm) 로 올린다(내리지 않는다).
[자동 다음 점](키 N): 창에서 고른 보드 안에서만(보드를 바꾸면 해제). 바깥 모서리는 자동이 다시 대지 않는다. 기록 없는 목표 중 손끝에서
가까운 것 → 순수 +z 로 올림(보드 면 최고점 + 30 mm, 이미 높으면 그대로) → 그 높이에서 수평 이동 · 손목 회전 → 위 10 mm 까지 10 mm/s 이하로
내림 → 0.3 s 정지 · 힘 기준선(이미 닿은 값이면 해제) → 사람 확인 대기(조그로 xy 보정, Enter(auto_descend) — 대기 때만 받는다, 건너뛰기 X
— s 는 조그 −x 키라 X) → 기존 천천히 하강 → 접촉 0.5 s 뒤 기존 기록 조건이면 기록 → 위로 20 mm → 다음. N 한 번에 6 점까지.
이동은 조각마다 Jogger.plan(조그와 같은 IK · 경로 하한 · 접촉 거부 · 속도). space · 접촉 이상 · 거부 · 힘 끊김 · 창 포커스 잃음 · 보드 바뀜 ·
맞춤 바뀜 = 자동 해제(space 말고는 '확인 없이' 도 끔). '확인 없이 자동 하강' 을 켜도 N 뒤 첫 점은 확인을 받는다.
점마다 aligned_by = human_corner(바깥 모서리 수동) · human(사람 확인) · auto.

--seed(11.05 사용자 "다시 실행하면 이 4 점으로 보드 자세를 잡고 나머지 교차점 전부를 자동으로, 점마다 사진"): 이전 기록의 산 점을
새 --out 에 seeded_from 으로 복사하고(원본은 읽기만, side · urdf_md5 다르면 거부) 처음 맞춤에 쓴다(aligned_by human 인 모서리도 모서리로).
보드 표시 자세는 --board_pose · --board_npz 가 없으면 seed 기록의 것. 이때 N = 남은 점 전체 자동(N 한 번 6 점 상한 없음): 위 10 mm 정지 ·
힘 기준선 뒤 사람 Enter 대신 유지 보정 수렴(손끝 실측−명령 ≤ 1 mm · 0.5° 가 0.5 s, 8 s 넘으면 그 점 건너뜀 · JSON auto_skipped) →
처음 --auto_confirm_first 점(기본 2)만 Enter 확인(human), 그 뒤 확인 없이(auto). 거부 · 이상 · 힘 끊김 · 해제는 그 자리 정지(다음 점 안 감).
기록마다 그 전 맞춤 잔차 fit_res_mm(3 mm 넘으면 표시만), 끝나면 요약(창 · stdout · JSON auto_summary). 사진(board_photo):
기록 창 안 원본 컬러 한 장 → <out>_photos/<순번>_<보드>_<목표>.jpg · _overlay.jpg · 점의 photo_check(카메라 없으면 이유만, 기록은 계속).

좌표:
  base      URDF 루트(로봇 base). 조그 'base' 프레임 = base 축, 회전은 손끝점을 중심으로 base 축 둘레.
  손끝(tip) URDF 링크 <p>_hl_index_tip. 조그 'tip' 프레임 = 그 링크 축(이동 · 회전 모두).
  보드      BOARD_CONVENTION · GRID_CONVENTION(아래). GUI 로 놓는 보드는 탁자면에 엎어 둔 것이 아니라 인쇄면이 위 — 보드 z 는 탁자 속(base −z)이다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import place_teach as P  # noqa: E402
from policy_control.fk_numpy import UrdfTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import board_auto as A  # noqa: E402
import board_photo as PH  # noqa: E402
from hand_grasp import HAND_STATE, MEAS_WAIT_S, STALE_S, arm_hold_target  # noqa: E402
from replay_to_pd import EPISODE_TOPIC_SIDE, PD_STATUS, STATE_TOPIC, TOPIC, replay_episode_id  # noqa: E402
from teach_place import MOVING, arm_limits, pd_fault  # noqa: E402
from touch_probe import (BASE_S, E0_SETTLE_MAX_S, E0_WIN_S, FORCE_G, INDEX_JF, INDEX_TIP, JF_TOPIC, PD_VEL_RAD_S, PRE_JF_G,  # noqa: E402
                         SPIN_MAX, START_TOL_RAD, TIP_N, TIP_TOPIC, TRACK_TOL_MOVE, FORCE_STALE_S, Detector, clean_force,
                         release_and_stop, settled_e0, start_e0)

S2R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(S2R / "scripts"))
import run_paths  # noqa: E402
URDF_DEFAULT = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
CONTRACT = S2R / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"   # home_arm(plan_touch 와 같은 홈)
BOARD_DIR = S2R / "logs/analysis/fk_touch_probe_1105"                           # plan_board.board_pose(--board_npz 때만 import)
POINT = (1.57, 0.0, 0.0, 1.4, 1.4, 1.4)        # 검지만 편 손(holes.POINT · plan_touch 손)
FK_DZ_MM = {"right": 12.0, "left": -12.0}       # 상판이 FK 로 읽히는 높이 − 205(11.04 측정, PROCEDURE.md) — 표시용
TABLE_Z = 0.205                                 # 상판 CAD(plan_touch.TABLE_Z)
DT = 1.0 / P.POLICY_HZ
JOG_VEL = 0.15                                  # rad/s 관절당 명령 이동 상한(pd reduced 0.25 아래)
HAND_VEL = 1.0                                  # rad/s 손(place_teach.HAND_VEL_RAD_S, pd hand.max_vel 과 같다)
DESC_V, DESC_MAX = 0.003, 0.040                 # 천천히 하강 m/s · 최대 거리 m
LIFT_M = 0.020
IK_LAMBDA = 0.01
# DLS 감쇠: 홈 자세 자코비안 특이값 위치 0.12~0.19 m/rad · 회전 1.4~1.8(11.05 실제 URDF) ≫ λ 라 평소엔 가우스-뉴턴과 거의 같고,
# 특이 자세(σ→0)에서만 반복당 |dq| ≤ |e|/(2λ) = 50|e| 로 묶인다(1 mm 증분이면 ≤ 0.05 rad).
IK_ITERS = 60
IK_POS_TOL, IK_ANG_TOL = 2e-4, np.radians(0.1)
IK_JUMP_RAD = 0.3                               # 한 단계 해의 관절 변화 상한 — 넘으면 특이점 · 다른 가지로 튄 것
IK_H = 1e-6
# 손끝 z 가 이보다 더 내려가면 '아래로 미는' 증분. IK 위치 허용(IK_POS_TOL 0.2 mm)보다 커야 수평 · 회전 증분이 IK 잔차만으로
# '아래로' 판정되지 않는다. 0.3 mm 안의 내려감은 이동 중 접촉 · 힘 상승 정지와 z 하한이 받친다.
DOWN_EPS = 3e-4
FLOOR_BELOW_M = 0.008                           # 손끝 명령 z 하한 = 표시 보드면 − 이 값(H1)
PATH_DQ = 0.005                                 # rad — 보간 경로 FK 샘플 간격(가장 먼 관절 기준)
STILL_V = 0.01                                  # rad/s 기록 조건 정지 속도
REC_WIN = 0.5
REC_CONTACT_FRAC = 0.9                          # 기록 창 안 접촉 틱 비율 하한
REC_XY_MAX_MM = 40.0                            # 기록 때 손끝 FK xy ↔ 고른 목표 표시 xy 상한(표시 자세 ~15 + FK ~10 mm 여유) — 넘으면 이름 잘못 붙음
REC_XY_MAX_CORNER_MM = 80.0                     # 맞춤 전 바깥 모서리(표시 자세를 바로잡는 기록이라 넓게)
REC_DUP_MM, REC_DUP_S = 3.0, 3.0                # 직전과 같은 목표를 손끝 이동 < 3 mm · 3 s 안에 다시 기록 = 겹친 창 중복
RAW_PART_S, RAW_KEEP_S = 600.0, 5.0             # 원본 틱 조각 주기 · 메모리에 남길 최근 구간(창 계산용, E0_WIN · BASE · REC_WIN 보다 길게)
DIR_DISTINCT_DEG = 10.0                         # 같은 교차점에서 손끝 z 축이 이보다 다르면 다른 손목 방향으로 센다
TRANS_STEPS_MM, ROT_STEPS_DEG = (1, 2, 5, 10), (0.5, 1, 2, 5)
AXES = ("x", "y", "z", "roll", "pitch", "yaw")
# 글자 키(창에서는 방향키 · PageUp/Down 도). base 축 회전은 손끝점 중심이라 pitch + = 손목이 +x 로(앞으로 기울이기),
# roll + = 손목이 −y 로(오른쪽으로 기울이기), yaw + = 위에서 봐 반시계(⟲). u/o 는 키보드 왼쪽 · 오른쪽에 맞춰 11.05 에 뒤집었다.
KEYS = {"w": ("x", 1), "s": ("x", -1), "a": ("y", 1), "d": ("y", -1), "r": ("z", 1), "f": ("z", -1),
        "u": ("roll", -1), "o": ("roll", 1), "i": ("pitch", 1), "k": ("pitch", -1), "j": ("yaw", 1), "l": ("yaw", -1)}
AUTO_CLEAR_M, AUTO_HOVER_M, AUTO_LIFT_M = 0.030, 0.010, 0.020   # 자동: 수평 이동 높이(보드 면 최고점 위) · 확인 대기 높이(목표 위) · 기록 뒤 올림
AUTO_STEP_M, AUTO_ROT_DEG = 0.010, 5.0          # 자동 이동 한 조각(조그 한 단계처럼 Jogger.plan 한 번)
AUTO_APPROACH_V, AUTO_APPROACH_STEP_M = 0.010, 0.002   # 위 10 mm 로 내려가는 손끝 속도 상한 m/s · 그 조각 m(Jogger.plan tip_v 로 관절 속도를 낮춤)
# 조각을 V·DT(0.17 mm)로 줄이는 길은 IK 위치 허용 0.2 mm 안이라 IK 가 움직이지 않는다 — 조각은 2 mm, 관절 속도를 그 조각 손끝 속도 ≤ V 로 낮춘다.
AUTO_HOVER_S = BASE_S                           # 확인 대기 높이에서 정지 뒤 힘 기준선(창 BASE_S 0.3 s 가 정지 뒤 틱만 담게)
AUTO_MAX_PTS = 6                                # N 한 번에 도는 목표 수 상한(건너뛴 것 포함)
AUTO_FIT_BY = ("human_corner", "human")         # 맞춤에 쓰는 aligned_by(auto 점은 기록 · 분석용)
AUTO_TOL_M, AUTO_TOL_DEG = 5e-4, 0.5            # 조각 목표 도착 판정(IK 허용 0.2 mm · 0.1° 보다 크게)
AUTO_REC_WAIT_S = 2.0                           # 접촉 정지 뒤 기록 조건을 이만큼까지 기다린다(넘으면 자동 해제)
AUTO_PH = {"rise": "올림(+z)", "travel": "수평 이동", "approach": f"위 {AUTO_HOVER_M * 1000:.0f} mm 로", "hover": "정지 · 힘 기준선",
           "wait": "확인 대기", "descend": "하강", "settle": "기록 대기", "retreat": "위로 빼기", "converge": "유지 수렴 대기"}
AUTO_MOVES = ("rise", "travel", "approach", "retreat")
# 유지 보정(관절 유지 바깥 적분, 11.05): 실측 처짐 차 0.05~0.12 rad 를 수 초에(τ = 1/K = 2 s, 0.08 → 0.01 약 4 s). 상한 = 손목 처짐 실측 범위
HOLD_K, HOLD_MAX_RAD, HOLD_RATE = 0.5, 0.12, 0.03   # /s · rad · rad/s(관절당)
HOLD_SETTLE_S = 0.5   # 이동 중 · 끝난 뒤 이만큼은 얼림 — 추종 지연을 처짐으로 알고 쌓지 않게(적분 와인드업 방지)
# 자동 이 단계에선 c 를 얼린다(보드에 다가가 · 힘 기준선 · 하강 · 기록). 확인 대기(wait, 위 10 mm 정지)에선 하강 전에 실측을 명령에 맞춘다
HOLD_FREEZE_AUTO = ("approach", "hover", "descend", "settle")
# 전체 자동(--seed, 11.05 사용자): 위 10 mm 에서 사람 Enter 대신 유지 보정 수렴을 기다린다 — 손끝 실측 − 명령 ≤ 1 mm · 0.5° 가 0.5 s,
# 8 s 안에 안 되면 그 점은 건너뛴다(이유는 JSON auto_skipped). 처음 --auto_confirm_first 점은 그 뒤 Enter 확인.
AUTO_CONV_MM, AUTO_CONV_DEG, AUTO_CONV_HOLD_S, AUTO_CONV_MAX_S = 1.0, 0.5, 0.5, 8.0
PHOTO_WIN_S = REC_WIN - 0.1   # 사진 = 기록 창 안에 받은 영상(앞 0.1 s 는 영상 수신 지연 몫 — 닿기 전 영상을 쓰지 않게)
SIM_HOST, SIM_PORT, SIM_HZ ="127.0.0.1", 47813, 30.0   # sim 창(Isaac) 으로 보내는 상태 — 계약은 sim_packet
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# 보드(scripts/charuco_calib.py:19-26): 7x5 칸, 30 mm, 마커 22 mm, DICT_6X6_250. 안쪽 교차점 6x4 = 24 개.
SX, SY, SQUARE, MARKER = 7, 5, 0.030, 0.022
# board_cam_fit.CORNERS(= cv2 CharucoBoard.getChessboardCorners(), id 순)와 같은 값 — cv2 없이 쓰려고 식으로 옮겼다(selftest 가 cv2 와 대조).
CORNERS = np.array([[(i % (SX - 1) + 1) * SQUARE, (i // (SX - 1) + 1) * SQUARE, 0.0] for i in range((SX - 1) * (SY - 1))])
MARKERS = [((c + 0.5) * SQUARE, (r + 0.5) * SQUARE) for r in range(SY) for c in range(SX) if (r + c) % 2]   # 마커 id 순 중심
BOARD_CONVENTION = ("OpenCV CharucoBoard((7,5),0.030,0.022,DICT_6X6_250) 보드 좌표 m(board_cam_fit.CORNERS 와 같음). "
                    "원점 = 마커 0 왼쪽 검은 칸(0행 0열)의 바깥 모서리, x = 마커 0→1→2 방향(7 칸), y = 마커 0 줄에서 마커 3 줄 방향(5 칸), "
                    "z = x × y = 인쇄면에서 보드 속(위에서 보면 탁자 아래). 교차점 id i → (x, y) = ((i % 6 + 1)·0.03, (i // 6 + 1)·0.03).")

# ArUco GridBoard(hand_arm_aruco_A4.pdf 4쪽): 2x2, 마커 25 mm, 간격 5 mm, 보드 55 mm, DICT_6X6_250. 같은 ids 보드가 두 장씩.
GRID_MARKER, GRID_GAP = 0.025, 0.005
GRID_FIRST = {"grid50": 50, "grid50b": 50, "grid54": 54, "grid54b": 54}   # 50-53 오른팔 · 54-57 왼팔, b = 둘째 장
# cv2 4.14 GridBoard((2,2),0.025,0.005,dict,ids).getObjPoints() 와 같은 값(selftest 가 대조): 마커 k(ids 순) 모서리 c0..c3.
GRID_CORNERS = np.array([[(k % 2) * (GRID_MARKER + GRID_GAP) + dx, (k // 2) * (GRID_MARKER + GRID_GAP) + dy, 0.0] for k in range(4)
                         for dx, dy in ((0, 0), (GRID_MARKER, 0), (GRID_MARKER, GRID_MARKER), (0, GRID_MARKER))])
GRID_CONVENTION = ("OpenCV aruco GridBoard((2,2),0.025,0.005,DICT_6X6_250,ids) getObjPoints() 보드 좌표 m(cv2 4.14). "
                   "원점 = 첫 마커(ids[0]) 검은 사각의 바깥 모서리, x = ids[0]→ids[1] 방향, y = ids[0]→ids[2] 방향, "
                   "z = x × y = 인쇄면에서 보드 속(ChArUco 와 같은 규약). 마커 k(0..3, ids 순)의 모서리 c → "
                   "(x, y) = ((k % 2)·0.03 + [0, .025, .025, 0][c], (k // 2)·0.03 + [0, 0, .025, .025][c]) — c0 원점 쪽, c1 +x, c2 +x+y, c3 +y. "
                   "target_id = m{마커 id}_c{c}.")


def _grid(bid: str) -> dict:
    ids = list(range(GRID_FIRST[bid], GRID_FIRST[bid] + 4))
    return {"type": "aruco_grid", "targets": GRID_CORNERS, "outer": (0, 5, 11, 14),
            "names": [f"m{i}_c{c}" for i in ids for c in range(4)],
            "marks": [(GRID_CORNERS[4 * k:4 * k + 4, :2].mean(0), str(i)) for k, i in enumerate(ids)],
            "geometry": {"markers": [2, 2], "marker_m": GRID_MARKER, "gap_m": GRID_GAP, "size_m": 2 * GRID_MARKER + GRID_GAP, "ids": ids,
                         "dict": "DICT_6X6_250", "corner_convention": GRID_CONVENTION}}


# board_id → 목표 점(보드 좌표) · 이름 · 그림용 표지 · JSON 에 적을 기하 설명
# outer = 바깥 4 모서리 목표 index [원점, +x, +y, +x+y](charuco bc00 · bc05 · bc18 · bc23, grid m{첫}_c0 · m{+1}_c1 · m{+2}_c3 · m{+3}_c2)
BOARDS = {"charuco": {"type": "charuco", "targets": CORNERS, "outer": (0, 5, 18, 23), "names": [f"bc{i:02d}" for i in range(len(CORNERS))],
                      "marks": [(m, f"m{i}") for i, m in enumerate(MARKERS)],
                      "geometry": {"squares": [SX, SY], "square_m": SQUARE, "marker_m": MARKER, "dict": "DICT_6X6_250",
                                   "corner_convention": BOARD_CONVENTION}},
          **{b: _grid(b) for b in GRID_FIRST}}


# ------------------------------------------------------------------ 운동학(순수)
class Kin:
    """한 팔 FK. tip(q, h) = 손끝 4x4(base), skeleton = 팔 링크 원점 · 손가락 링크 원점(그리기)."""

    def __init__(self, urdf: Path, side: str, hand: np.ndarray):
        self.tree, self.side, self.hand = UrdfTree(Path(urdf)), side, np.asarray(hand, float)
        self.arm, self.hands, p = list(P.arm_names(side)), list(P.hand_names(side)), side[0]
        self.tip_link = f"{p}_hl_index_tip"
        self.arm_links = [f"{p}_al_{i}" for i in range(8)] + [f"{p}_hl_base"]
        self.finger_links = [[f"{p}_hl_base"] + [f"{p}_hl_{f}_{k}" for k in ((1, 2, 3, 4) if f == "thumb" else (1, 2))] + [f"{p}_hl_{f}_tip"]
                             for f in FINGERS]
        self.all_links = list(dict.fromkeys(self.arm_links + sum(self.finger_links, []) + [self.tip_link]))

    def _q(self, q, h) -> dict:
        return self.tree.expand_mimic(dict(zip(self.arm, q)) | dict(zip(self.hands, self.hand if h is None else h)))

    def tip(self, q, h=None) -> np.ndarray:
        return self.tree.transforms([self.tip_link], self._q(q, h))[self.tip_link]

    def skeleton(self, q, h=None) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        T = self.tree.transforms(self.all_links, self._q(q, h))
        return (np.array([T[n][:3, 3] for n in self.arm_links]), [np.array([T[n][:3, 3] for n in ch]) for ch in self.finger_links],
                T[self.tip_link])


def rotvec(R: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(R).as_rotvec()


def pose_err(Tg: np.ndarray, T: np.ndarray) -> np.ndarray:
    """[위치 차(base), 회전 차 rotvec(base)] — 자코비안 열과 같은 표현."""
    return np.r_[Tg[:3, 3] - T[:3, 3], rotvec(Tg[:3, :3] @ T[:3, :3].T)]


def ik(kin: Kin, q0: np.ndarray, Tg: np.ndarray, lo: np.ndarray, hi: np.ndarray, h=None,
       bounded: bool = False) -> tuple[np.ndarray | None, str]:
    q, why = _ik_dls(kin, q0, Tg, lo, hi, h)
    if q is None and bounded:
        from bounded_jog_ik import solve
        return solve(kin, q0, Tg, lo, hi, h)
    return q, why


def _ik_dls(kin: Kin, q0: np.ndarray, Tg: np.ndarray, lo: np.ndarray, hi: np.ndarray, h=None, bounded: bool = False) -> tuple[np.ndarray | None, str]:
    """목표 손끝 자세 Tg 에 가는 팔 관절. (q, "") 또는 (None, 이유). lo · hi 는 이미 여유를 뺀 한계. 순수."""
    q = np.asarray(q0, float).copy()
    for _ in range(IK_ITERS):
        T = kin.tip(q, h)
        e = pose_err(Tg, T)
        if np.linalg.norm(e[:3]) < IK_POS_TOL and np.linalg.norm(e[3:]) < IK_ANG_TOL:
            break
        J = np.empty((6, 7))
        for j in range(7):
            qj = q.copy()
            qj[j] += IK_H
            J[:, j] = pose_err(kin.tip(qj, h), T) / IK_H
        q = q + J.T @ np.linalg.solve(J @ J.T + IK_LAMBDA ** 2 * np.eye(6), e)
    else:
        e = pose_err(Tg, kin.tip(q, h))
        return None, f"IK 수렴 실패(위치 {np.linalg.norm(e[:3]) * 1000:.2f} mm · 각 {np.degrees(np.linalg.norm(e[3:])):.2f}°)"
    bad = [f"j{j + 1} {q[j]:+.3f} ∉ [{lo[j]:+.3f}, {hi[j]:+.3f}]" for j in range(7) if not lo[j] <= q[j] <= hi[j]]
    if bad:
        return None, "관절 한계(여유 포함) 밖: " + ", ".join(bad)
    jump = float(np.abs(q - q0).max())
    if jump > IK_JUMP_RAD:
        return None, f"한 단계 관절 변화 {jump:.3f} rad > {IK_JUMP_RAD} — 특이 자세 근처"
    return q, ""


def jog_goal(T: np.ndarray, axis: str, sign: int, step: float, frame: str) -> np.ndarray:
    """T 에서 한 단계 간 목표. step: 이동 m · 회전 rad. frame 'tip' = 손끝 축, 'base' = base 축(회전 중심 = 손끝점). 순수."""
    k = AXES.index(axis)
    d = np.zeros(3)
    d[k % 3] = sign * step
    Tg = T.copy()
    if k < 3:
        Tg[:3, 3] = T[:3, 3] + (T[:3, :3] @ d if frame == "tip" else d)
    else:
        dR = Rot.from_rotvec(d).as_matrix()
        Tg[:3, :3] = T[:3, :3] @ dR if frame == "tip" else dR @ T[:3, :3]
    return Tg


def refuse_down(z_from: float, z_to: float, contact: bool, stale: bool) -> str | None:
    """접촉 중(또는 힘 끊김) 손끝을 base −z 로 더 내리는 이동이면 이유. 순수."""
    if z_to < z_from - DOWN_EPS and (contact or stale):
        return ("접촉 중" if contact else "힘 끊김 —") + f" 아래로 미는 증분 거부(Δz {(z_to - z_from) * 1000:+.2f} mm) — 위로 빼기는 된다"
    return None


def path_min_z(kin: Kin, q0, q1, h0, h1) -> float:
    """명령이 실제로 지나는 길(step_toward = 관절 공간 직선, 손도 직선 근사) 위 손끝 FK z 최솟값. 순수."""
    q0, q1, h0, h1 = (np.asarray(v, float) for v in (q0, q1, h0, h1))
    n = int(np.ceil(max(np.abs(q1 - q0).max(), np.abs(h1 - h0).max()) / PATH_DQ)) + 1
    return min(float(kin.tip(q0 + (q1 - q0) * s, h0 + (h1 - h0) * s)[2, 3]) for s in np.linspace(0.0, 1.0, max(n, 2)))


def refuse_floor(z_min: float, z_now: float, floor: float) -> str | None:
    """경로 최저 z 가 하한 아래면 이유. 이미 하한 아래에 있으면 지금보다 더 내려가는 것만 거부(빼기는 된다). 순수."""
    lim = floor if z_now >= floor else z_now - DOWN_EPS
    if z_min < lim:
        return f"손끝 z 하한 {floor * 1000:.1f} mm 아래로 가는 경로(최저 {z_min * 1000:.1f} mm) 거부"
    return None


def step_toward(q: np.ndarray, goal: np.ndarray, vmax: float, dt: float) -> np.ndarray:
    """goal 쪽으로 관절당 vmax·dt 를 넘지 않게(관절 공간 직선, 가장 먼 관절이 vmax). 순수."""
    d = np.asarray(goal, float) - q
    m = float(np.abs(d).max()) if d.size else 0.0
    lim = vmax * dt
    return np.asarray(goal, float).copy() if m <= lim else q + d * (lim / m)


def arm_speed(ts, qs) -> float:
    """창의 관절 속도 최대(앞 · 뒤 절반 중앙값 차 / 반 창, touch_probe.settled_e0 와 같은 잼). 창 부족이면 inf. 순수."""
    ts, Q = np.asarray(ts, float), np.asarray(qs, float)
    span = float(ts[-1] - ts[0]) if len(ts) else 0.0
    if len(ts) < 5 or span < 0.9 * REC_WIN or not np.all(np.isfinite(Q)):
        return float("inf")
    h = len(ts) // 2
    return float(np.abs(np.median(Q[h:], 0) - np.median(Q[:h], 0)).max() / (span / 2))


def contact_frac(flags) -> float:
    return float(np.mean(flags)) if len(flags) else 0.0


def can_record(contact: bool, ts, qs, flags) -> tuple[bool, str]:
    """flags = 창 안 틱별 접촉 표시등. 순수."""
    if not contact:
        return False, "접촉 표시등 꺼짐"
    if (fr := contact_frac(flags)) < REC_CONTACT_FRAC:
        return False, f"창 안 접촉 비율 {fr:.0%} < {REC_CONTACT_FRAC:.0%}"
    v = arm_speed(ts, qs)
    return (True, "") if v < STILL_V else (False, f"팔이 아직 움직인다({v:.4f} rad/s ≥ {STILL_V})")


def reachable_grip(grip: np.ndarray, side: str) -> np.ndarray:
    """Existing command-range projection; never redefine the measured joint zero."""
    from policy_control.rh56f1_map import load
    hand_map = load(S2R / "deploy/policy_control/config/rh56f1_hand_map.yaml")
    return hand_map.to_rad(hand_map.to_register(grip, side=side), side=side)


def dir_count(dirs: list) -> int:
    """손끝 z 축 방향 목록에서 서로 DIR_DISTINCT_DEG 넘게 다른 방향 수(앞에서부터 묶음). 순수."""
    reps: list = []
    for d in dirs:
        if all(np.degrees(np.arccos(np.clip(np.dot(d, r), -1, 1))) > DIR_DISTINCT_DEG for r in reps):
            reps.append(np.asarray(d, float))
    return len(reps)


def dir_deg(d) -> tuple[float, float]:
    """방향 → (방위 atan2(y,x), 고각 asin(z)) °."""
    d = np.asarray(d, float)
    return float(np.degrees(np.arctan2(d[1], d[0]))), float(np.degrees(np.arcsin(np.clip(d[2], -1, 1))))


# ------------------------------------------------------------------ 보드
def board_T(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    """인쇄면이 위인 평평한 보드: 원점 (x,y,z), 보드 x 축이 base x 에서 yaw. 보드 z 는 아래(BOARD_CONVENTION)."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = x, y, z
    return T


def board_xyyaw(T: np.ndarray) -> tuple[float, float, float, float]:
    return float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def rec_xy_check(tip_xy, name: str, tgt_xy, all_targets: dict, lim_mm: float = REC_XY_MAX_MM) -> tuple[bool, str]:
    """기록 전 손끝 xy(m)가 고른 목표 name 의 표시 xy(m) 에서 lim_mm 안인가. 아니면 all_targets {표시 이름: xy} 중 가장 가까운 점을 알린다. 순수."""
    tip = np.asarray(tip_xy, float)[:2]
    d = float(np.linalg.norm(tip - np.asarray(tgt_xy, float)[:2])) * 1000
    if d <= lim_mm:
        return True, ""
    n2, d2 = min(((n, float(np.linalg.norm(tip - np.asarray(x, float)[:2])) * 1000) for n, x in all_targets.items()), key=lambda t: t[1])
    return False, f"손끝이 목표 {name} 에서 {d:.0f} mm 떨어짐(한계 {lim_mm:.0f}). 가장 가까운 점 {n2} ({d2:.0f} mm)"


def to_base(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]   # plan_board.corners_base 와 같다


def corners_base(T: np.ndarray) -> np.ndarray:
    return to_base(CORNERS, T)


def _pose(d, where: str, zc: float) -> tuple[np.ndarray, bool]:
    """{T_base_board: 4x4} 또는 {x, y[, z], yaw_deg} → (T, GUI 로 옮길 수 있나)."""
    if not isinstance(d, dict):
        raise ValueError(f"{where}: 사전이 아니다")
    if "T_base_board" in d:
        T = np.asarray(d["T_base_board"], float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)) or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-3:
            raise ValueError(f"{where}: T_base_board 가 4x4 강체 변환이 아니다")
        return T, False
    v = [float(d["x"]), float(d["y"]), float(d.get("z", zc)), float(d.get("yaw_deg", 0.0))]
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{where}: 비유한 값")
    return board_T(*v), True


def load_board_poses(path: Path, zc: float, bids: list) -> dict:
    """yaml/json → {board_id: (T, 옮길 수 있나)}. 키가 board_id 인 여러 보드 형식({charuco: {...}, grid50: {...}}),
    또는 한 보드 형식({T_base_board: 4x4} 또는 {x, y[, z], yaw_deg}, m · °) = --boards 첫 보드. z 없으면 탁자면 zc."""
    import yaml

    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError(f"{path}: 사전이 아니다")
    if not set(d) & set(BOARDS):
        return {bids[0]: _pose(d, str(path), zc)}
    if bad := sorted(set(d) - set(BOARDS)):
        raise ValueError(f"{path}: 모르는 보드 키 {bad} — {list(BOARDS)}")
    return {b: _pose(v, f"{path}:{b}", zc) for b, v in d.items() if b in bids}


def board_from_npz(npz: Path, extr: Path, zc: float) -> tuple[np.ndarray, dict]:
    """plan_board.board_pose(카메라 한 장 · 외부 파라미터)로 대략 자세. 표시 높이는 plan_board 계획처럼 기대 접촉 FK 높이 zc 로 옮긴다."""
    sys.path.insert(0, str(BOARD_DIR))
    import plan_board  # noqa: PLC0415 — cv2 4.7+ 가 필요하다(sim2real/.venv)

    T, det, warn, _img, _K = plan_board.board_pose(Path(npz), Path(extr))
    Td = T.copy()
    Td[2, 3] += zc - float(corners_base(T)[:, 2].mean())   # charuco 전용(plan_board 는 ChArUco 만 찾는다)
    return Td, {"npz": str(npz), "extrinsics": str(extr), "T_base_board_est": T.tolist(), "reproj_px": float(det["reproj"]), "warn": warn}


def board_fit(bid: str, points: list, yaw_ref: float | None = None) -> dict | None:
    """기록(points, JSON 꼴)에서 보드 자세. 사람이 맞춘 점(aligned_by ∈ AUTO_FIT_BY, 키 없으면 사람)만 쓴다. 바깥 4 모서리(모서리마다
    마지막 기록)가 다 없으면 None. 4 모서리 맞춤(대각선 · 변 · 잔차 · 기울기 · yaw 관문)이 거부면 그대로 거부, 통과하고 사람 점이
    A.REFIT_MIN 개 이상이면 전체로(같은 잔차 · 기울기 · yaw 관문, 거부면 거부). R_wrist = 그 점들 손끝 자세 평균(자동 이동 손목). 순수."""
    spec = BOARDS[bid]
    idx = {n: i for i, n in enumerate(spec["names"])}
    pts = [p for p in points if not p["deleted"] and p["board_id"] == bid and p.get("aligned_by", "human") in AUTO_FIT_BY]
    if not pts:
        return None

    def pos(ps: list) -> np.ndarray:
        return np.array([np.asarray(p["T_tip_fk"], float)[:3, 3] for p in ps])

    extra = {"R_wrist": Rot.from_matrix([np.asarray(p["T_tip_fk"], float)[:3, :3] for p in pts]).mean().as_matrix()}
    last = {p["target_id"]: p for p in pts}
    corner = [last.get(spec["names"][i]) for i in spec["outer"]]
    if any(c is None for c in corner):
        return None
    fc = A.fit_board(spec["targets"][list(spec["outer"])], pos(corner), corners=True, yaw_ref=yaw_ref) | extra | {"kind": "4점"}
    if not fc["ok"] or len(pts) < A.REFIT_MIN:
        return fc
    kind = f"전체 {len(pts)}점"
    f = A.fit_board(spec["targets"][[idx[p["target_id"]] for p in pts]], pos(pts), yaw_ref=yaw_ref)
    if not f["ok"]:
        return f | extra | {"kind": kind, "why": f"{kind}: {f['why']} — 가장 큰 잔차 {pts[int(f['res_mm'].argmax())]['target_id']}"}
    return f | extra | {"kind": kind, "diag_mm": fc["diag_mm"], "edge_mm": fc["edge_mm"], "warn": fc["warn"]}


def auto_queue(points: list, fits: dict, skip, start, bid: str) -> list:
    """자동 방문 차례 [(board_id, 목표 index, base xyz)]: 보드 bid(맞춤 받아들임)의 산 기록 없는 목표(바깥 모서리 · skip 제외), 손끝 start 에서
    가까운 차례. 바깥 모서리는 사람이 조그로 댄다(지워졌으면 맞춤도 없어 자동이 서지 않는다).
    shortcut: 목표당 손목 1 방향(이 코드에 손목 방향 목록이 없다) — 방향 목록이 생기면 (목표, 방향) 단위로 넣는다."""
    done = {(p["board_id"], p["target_id"]) for p in points if not p["deleted"]}
    f, sp = fits.get(bid), BOARDS[bid]
    C = [] if f is None or not f["ok"] else [
        (bid, i, x) for i, (n, x) in enumerate(zip(sp["names"], to_base(sp["targets"], f["T"])))
        if i not in sp["outer"] and (bid, n) not in done and (bid, i) not in skip]
    return [C[k] for k in A.visit_order(start, [c[2] for c in C])] if C else []


def auto_wait_text(name: str) -> str:
    return (f"목표 {name} 위 {AUTO_HOVER_M * 1000:.0f} mm — 위치 보고 조그로 xy 보정 후 [하강]/Enter, 건너뛰기 [X]"
            "(s 는 조그 −x)")


# ------------------------------------------------------------------ 기록 파일
def say(msg: str, err: bool = False) -> None:
    """정리 경로의 출력 — 터미널이 닫혀(SIGHUP) print 가 실패해도 해제 · 저장은 이어 간다."""
    try:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)
    except Exception:   # noqa: BLE001, S110
        pass


def check_out(out: Path) -> None:
    for f in [out, *out.parent.glob(f"{out.stem}_raw*.npz"), out.with_name(f"{out.stem}_photos")]:
        if f.exists():
            raise FileExistsError(f"{f} 가 이미 있다 — 덮어쓰지 않는다")
    if out.suffix != ".json":
        raise ValueError(f"--out 은 .json: {out}")


def load_seed(paths: list, side: str, urdf_md5: str, bids: list, zc: float) -> tuple[list, list, dict]:
    """--seed: 이전 기록들의 산(deleted 아닌) 점 → (점 복사본[seeded_from 표시], 출처 [{path, md5, n}], {보드: (표시 자세 T, 출처)}).
    side · urdf_md5 가 지금과 다르면 · --boards 밖 보드 · 모르는 목표면 거부. 원본 파일은 읽기만 한다."""
    pts, info, disp = [], [], {}
    for path in map(Path, paths):
        raw = path.read_bytes()
        d = json.loads(raw)
        if not isinstance(d, dict) or d.get("schema") != "board_jog_v1":
            raise ValueError(f"--seed {path}: board_jog_v1 기록이 아니다")
        if d.get("side") != side or d.get("urdf_md5") != urdf_md5:
            raise ValueError(f"--seed {path}: side {d.get('side')} · urdf_md5 {d.get('urdf_md5')} ≠ 지금 {side} · {urdf_md5} — 다른 팔 · URDF 기록은 섞지 않는다")
        live = [(i, p) for i, p in enumerate(d.get("points", [])) if not p.get("deleted")]
        for i, p in live:
            if p.get("board_id") not in bids or p.get("target_id") not in BOARDS[p["board_id"]]["names"]:
                raise ValueError(f"--seed {path}#{i}: 보드 {p.get('board_id')} · 목표 {p.get('target_id')} 가 --boards {bids} 에 없다")
            if np.asarray(p.get("T_tip_fk"), float).shape != (4, 4):
                raise ValueError(f"--seed {path}#{i}: T_tip_fk 가 4x4 가 아니다")
        pts += [{**copy.deepcopy(p), "seeded_from": f"{path}#{i}"} for i, p in live]
        info.append({"path": str(path), "md5": hashlib.md5(raw).hexdigest(), "n": len(live)})
        for b, v in d.get("boards", {}).items():   # 그 기록의 표시 자세(카메라 · 입력 추정) — 맞춤 yaw 검사(180°) 기준
            if b in bids and b not in disp and "T_base_board" in (v.get("pose_display") or {}):
                disp[b] = (_pose({"T_base_board": v["pose_display"]["T_base_board"]}, f"--seed {path}:{b}", zc)[0], str(path))
    return pts, info, disp


def raw_part(out: Path, n: int) -> Path:
    return out.with_name(f"{out.stem}_raw_part{n:02d}.npz")


def save_json_atomic(path: Path, doc: dict) -> None:
    """임시 파일에 다 쓰고 fsync 뒤 rename — 창이 죽어도 직전 기록이 온전히 남는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def new_doc(kin: Kin, urdf: Path, grip: np.ndarray, fk_dz_mm: float, poses: dict, z_floor: float) -> dict:
    """poses = {board_id: pose_display}. board = 옛 charuco 키(charuco 를 쓸 때만, 호환)."""
    doc = {"schema": "board_jog_v1", "side": kin.side, "urdf": str(urdf), "urdf_md5": hashlib.md5(Path(urdf).read_bytes()).hexdigest(),
           "tip_link": kin.tip_link, "arm_names": kin.arm, "hand_names": kin.hands, "hand_q": np.round(grip, 4).tolist(),
           "boards": {b: {"type": BOARDS[b]["type"], "geometry": BOARDS[b]["geometry"], "pose_display": p} for b, p in poses.items()},
           "fk_dz_mm": fk_dz_mm, "tip_z_floor_m": round(z_floor, 5), "start_wall_ns": time.time_ns(),
           "notes": "q_arm = 접촉 정지 0.5 s 팔 실측 중앙값(rad, arm_names 순). T_tip_fk = 이 URDF 로 (q_arm, q_hand 실측) 의 tip_link 4x4(base). "
                    "tip_dir_base = T_tip_fk 의 z 축. 점의 목표 = (board_id, target_id), target_xy_board_m 은 그 보드 좌표(boards.*.geometry). "
                    "charuco 점은 옛 corner_id · corner_xy_board_m 도 함께 적는다. contact_frac = 기록 창 안 접촉 틱 비율. "
                    "보드 표시 자세는 표시용 추정이다(풀이 입력 아님). deleted = 창에서 지운 행(남겨 둔다). 원본 틱 = <이름>_raw_partNN.npz.",
           "points": []}
    if "charuco" in poses:
        doc["board"] = {**BOARDS["charuco"]["geometry"], "pose_display": poses["charuco"]}
    return doc


def save_raw(path: Path, rows: list, side: str) -> int:
    cols = list(zip(*rows)) if rows else [[]] * 17
    names = ("t", "t_wall_ns", "tick", "q_meas", "q_cmd", "hand_meas", "hand_cmd", "jf_zeroed", "tip_zeroed", "contact", "publishing",
             "hold_c", "hold_state", "loop_dt", "T_tip_reference", "T_tip_measured", "q_sent")   # 팔 발행 = q_cmd + hold_c(관절 한계로 자른 값)
    arrs = {n: np.asarray(c) for n, c in zip(names, cols)}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, side=np.array(side), arm_names=np.array(P.arm_names(side)), hand_names=np.array(P.hand_names(side)), **arrs)
    os.replace(tmp, path)
    return len(rows)


# ------------------------------------------------------------------ 명령 상태기계(순수)
class Jogger:
    """가상 · 실기 공통 팔 명령. mode: idle · move(goal 로 JOG_VEL) · dwell(하강 전 정지 — Ctl 이 끝낸다) · descend."""

    def __init__(self, kin: Kin, lo: np.ndarray, hi: np.ndarray, cmd: np.ndarray, hand: np.ndarray, z_floor: float = -np.inf, precise: bool = False):
        self.kin, self.lo, self.hi = kin, lo + P.LIMIT_MARGIN_RAD, hi - P.LIMIT_MARGIN_RAD
        self.cmd, self.hand, self.hand_goal = np.asarray(cmd, float).copy(), np.asarray(hand, float).copy(), np.asarray(hand, float).copy()
        self.e0, self.z_floor = np.zeros(7), float(z_floor)
        self.precise, self.path, self.path_elapsed = precise, None, 0.0
        self.c = np.zeros(7)   # 유지 보정 — 발행 = cmd + c(Ctl._hold_step 이 바꾼다)
        self.mode, self.goal, self.down, self.t_mode, self.vmax = "idle", None, False, 0.0, JOG_VEL
        self.T0: np.ndarray | None = None
        self.c_prev, self.f_ref = False, None   # 이동 중 접촉 새로 켜짐 · 힘 상승 판정

    def pub(self) -> np.ndarray:
        """팔 발행값 = cmd + c(관절 한계 − 여유로 자름)."""
        return np.clip(self.cmd + self.c, self.lo, self.hi)

    def refuse_path(self, q1: np.ndarray, h1: np.ndarray, contact: bool, stale: bool) -> str | None:
        """cmd → q1(손 → h1) 경로의 접촉 중 내림 · z 하한 검사를 명령과 c 를 더한 발행값(c ≠ 0 일 때) 모두에 건다."""
        for d in [np.zeros(7)] + ([self.c] if self.c.any() else []):
            z0, z1 = float(self.kin.tip(self.cmd + d, self.hand)[2, 3]), float(self.kin.tip(q1 + d, h1)[2, 3])
            if why := refuse_down(z0, z1, contact, stale) or refuse_floor(path_min_z(self.kin, self.cmd + d, q1 + d, self.hand, h1), z0,
                                                                          self.z_floor):
                return why
        return None

    def hold(self, meas: np.ndarray | None, t: float = 0.0, contact: bool = False, emergency: bool = False) -> None:
        """명령 = 실측 − e0(누름 없음, 발행 = 그 + c). 실측이 없으면 지금 명령 그대로. c 는 그대로 둔다."""
        if self.precise and not emergency:
            if self.mode == "brake":
                return
            self.mode, self.goal, self.t_mode = "idle", None, t
            if meas is not None:
                target = np.clip(np.asarray(meas, float) - self.e0,
                                 np.maximum(self.lo, self.lo - self.c), np.minimum(self.hi, self.hi - self.c))
                if np.all(np.isfinite(target)) and self.refuse_path(target, self.hand, True, False) is None:
                    from precise_jog import QuinticPath
                    self.path = QuinticPath(self.cmd, target, max_vel=0.15, max_acc=0.60)
                    self.path_elapsed, self.goal, self.mode = 0.0, target.copy(), "brake"
                    self.f_ref, self.c_prev = None, contact
            return
        if meas is not None:
            c = np.clip(np.asarray(meas, float) - self.e0, self.lo, self.hi)
            if np.all(np.isfinite(c)):
                self.cmd = c
        self.mode, self.goal, self.t_mode = "idle", None, t

    def z_now(self) -> float:
        return float(self.kin.tip(self.cmd, self.hand)[2, 3])

    def plan(self, Tg: np.ndarray, contact: bool, stale: bool, tip_v: float | None = None, retarget: bool = False) -> str | None:
        """목표 손끝 자세로 이동 시작. 거부면 이유. tip_v(m/s) = 이 이동의 손끝 속도 상한(관절 속도를 JOG_VEL 아래로만 낮춘다)."""
        if self.mode != "idle" and not (self.precise and retarget and self.mode == "move"):
            return f"{self.mode} 중 — 끝난 뒤에 받는다"
        lower = np.maximum(self.lo, self.lo - self.c) if self.precise else self.lo
        upper = np.minimum(self.hi, self.hi - self.c) if self.precise else self.hi
        q, why = ik(self.kin, self.cmd, Tg, lower, upper, self.hand, bounded=self.precise)
        if q is None:
            return why
        vmax = JOG_VEL
        if tip_v is not None:   # 관절 공간 직선 · 짧은 조각이라 손끝 속도 ≈ 손끝 거리 × 관절 속도 / 가장 먼 관절 변화
            L, m = float(np.linalg.norm(Tg[:3, 3] - self.kin.tip(self.cmd, self.hand)[:3, 3])), float(np.abs(q - self.cmd).max())
            if L > 0 and m > 0:
                vmax = min(JOG_VEL, tip_v * m / L)
        return self.go(q, contact, stale, vmax=vmax, precise_fast=tip_v is None, retarget=retarget)

    def go(self, q: np.ndarray, contact: bool, stale: bool, t: float = 0.0, vmax: float = JOG_VEL, precise_fast: bool = True, retarget: bool = False) -> str | None:
        if self.mode != "idle" and not (self.precise and retarget and self.mode == "move"):
            return f"{self.mode} 중 — 끝난 뒤에 받는다"
        if any(np.any(v < self.lo) or np.any(v > self.hi) for v in (q, q + self.c)):
            return "목표가 관절 한계(여유 포함) 밖" + ("(유지 보정 c 포함)" if self.c.any() else "")
        if why := self.refuse_path(q, self.hand, contact, stale):
            return why
        path = None
        if self.precise:
            from precise_jog import RetargetablePath
            cap = 0.30 if precise_fast else min(float(vmax), 0.30)
            v, acc = (np.zeros(7), np.zeros(7)) if self.mode != "move" else self.path.sample_state(self.path_elapsed)[1:]
            try:
                path = RetargetablePath(self.cmd, q, velocity=v, acceleration=acc, max_vel=cap, max_acc=0.60)
            except ValueError as error:
                return f"연속 경로 거부: {error}"
            low, high = path.position_bounds()
            if np.any(low + np.minimum(self.c, 0) < self.lo - 1e-10) or np.any(high + np.maximum(self.c, 0) > self.hi + 1e-10):
                return "연속 경로 제동 여유 부족(관절 한계)"
            # Bound distance between actual curved-path samples by PATH_DQ.
            count = max(2, int(np.ceil(path.duration * cap / PATH_DQ)) + 1)
            offsets = [np.zeros(7)] + ([self.c] if self.c.any() else [])
            initial_z = [float(self.kin.tip(self.cmd + d, self.hand)[2, 3]) for d in offsets]
            for elapsed in np.linspace(0, path.duration, count):
                point = path.sample(elapsed)[0]
                for d, z0 in zip(offsets, initial_z):
                    z = float(self.kin.tip(point + d, self.hand)[2, 3])
                    if why := refuse_down(z0, z, contact, stale) or refuse_floor(z, z0, self.z_floor):
                        return why
        z0, z1 = self.z_now(), self.kin.tip(q, self.hand)[2, 3]
        self.goal, self.down, self.mode, self.t_mode = np.asarray(q, float).copy(), z1 < z0 - DOWN_EPS, "move", t
        self.vmax = min(float(vmax), JOG_VEL)
        if self.precise:
            self.path, self.path_elapsed = path, 0.0
        self.c_prev, self.f_ref = contact, None
        return None

    def start_descent(self, t: float) -> None:
        self.mode, self.t_mode, self.T0 = "descend", t, self.kin.tip(self.cmd, self.hand)

    def tick(self, t: float, dt: float, meas: np.ndarray | None, contact: bool, stale: bool, force=None) -> str | None:
        """한 틱. force = (|검지 관절 Δ| g, 촉각 Δ N)(Detector 기준선 대비) 또는 None. → 알림 문구(멈춘 이유 등) 또는 None."""
        self.hand = step_toward(self.hand, self.hand_goal, HAND_VEL, dt)
        if self.mode == "brake":
            if contact and not self.c_prev:
                self.mode, self.goal = "idle", None
                return "제동 중 새 접촉 — 현재 명령 유지"
            self.c_prev = contact
            if contact and force is not None:
                if self.f_ref is not None and (force[0] - self.f_ref[0] > FORCE_G or force[1] - self.f_ref[1] > TIP_N):
                    self.mode, self.goal = "idle", None
                    return "접촉 힘 상승 — 제동 경로 중단, 현재 명령 유지"
                if self.f_ref is None:
                    self.f_ref = force
            if stale:
                self.mode, self.goal = "idle", None
                return "힘 끊김 — 정지 유지"
            self.path_elapsed += dt
            self.cmd = self.path.sample(self.path_elapsed)[0]
            if self.path_elapsed >= self.path.duration:
                self.mode, self.goal = "idle", None
            return None
        if self.mode == "move":
            if self.down and stale:
                self.hold(meas, t, contact=contact)
                return "힘 끊김 — 내려가던 이동 멈춤"
            if contact and not self.c_prev:   # 방향과 상관없이(M2)
                self.hold(meas, t, contact=contact)
                return "이동 중 접촉 — 멈추고 명령 = 실측"
            self.c_prev = contact
            if contact and force is not None:
                if self.f_ref is None:
                    self.f_ref = force
                elif force[0] - self.f_ref[0] > FORCE_G or force[1] - self.f_ref[1] > TIP_N:
                    self.hold(meas, t, contact=contact)
                    return (f"접촉 중 힘 상승(검지 {force[0] - self.f_ref[0]:+.0f} g · 촉각 {force[1] - self.f_ref[1]:+.2f} N) — "
                            "멈추고 명령 = 실측")
            if self.precise:
                self.path_elapsed += dt
                self.cmd = self.path.sample(self.path_elapsed)[0]
            else:
                self.cmd = step_toward(self.cmd, self.goal, self.vmax, dt)
            if np.array_equal(self.cmd, self.goal):
                self.mode, self.goal = "idle", None
        elif self.mode == "descend":
            if contact or stale:
                self.hold(meas, t, contact=contact)
                return "접촉 — 하강 멈춤, 명령 = 실측(누름 풂)" if contact else "힘 끊김 — 하강 멈춤"
            s = min(DESC_V * (t - self.t_mode), DESC_MAX)
            Tg = self.T0.copy()
            Tg[2, 3] -= s
            if Tg[2, 3] < self.z_floor:
                self.hold(meas, t, contact=contact)
                return f"하강 멈춤 — 손끝 z 하한 {self.z_floor * 1000:.1f} mm(무접촉, 보드 자세를 확인한다)"
            lower = np.maximum(self.lo, self.lo - self.c) if self.precise else self.lo
            upper = np.minimum(self.hi, self.hi - self.c) if self.precise else self.hi
            q, why = ik(self.kin, self.cmd, Tg, lower, upper, self.hand)
            if q is None:
                self.hold(meas, t, contact=contact)
                return f"하강 멈춤 — {why}"
            if self.c.any() and self.kin.tip(q + self.c, self.hand)[2, 3] < self.z_floor:   # 발행값(+ 얼린 c)도 하한 위
                self.hold(meas, t, contact=contact)
                return f"하강 멈춤 — 발행값(+ 유지 보정 c) 손끝 z 하한 {self.z_floor * 1000:.1f} mm"
            self.cmd = step_toward(self.cmd, q, JOG_VEL, dt)
            if s >= DESC_MAX:
                self.mode = "idle"
                return f"하강 {DESC_MAX * 1000:.0f} mm — 무접촉, 멈춤"
        return None


# ------------------------------------------------------------------ 입출력: ROS · 가짜(demo)
class RosIO:
    """teach_place.run · touch_probe.run 과 같은 구독 · 발행. 발행자는 실기 발행을 처음 켤 때 만든다."""

    def __init__(self, side: str, photo_topic: str | None = None):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String

        self.rclpy, self.JointState, self.String, self.side = rclpy, JointState, String, side
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # 시그널은 이 도구가 받는다(release_and_stop)
        self.node = rclpy.create_node(f"board_jog_{side}")
        self.arm_src = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.hand_names = list(P.hand_names(side))
        self.arm, self.hand, self.pd, self.force, self.cnt = {}, {}, {}, {}, [0]
        self.pub = self.ep_pub = None
        self.ep_id = None

        def on_js(store):
            def cb(m) -> None:
                self.cnt[0] += 1
                store.update({n: (p, time.monotonic()) for n, p in zip(m.name, m.position)})
            return cb

        def on_pd(m) -> None:
            self.cnt[0] += 1
            try:
                d = json.loads(m.data)
            except (ValueError, TypeError):
                return
            if isinstance(d, dict):
                self.pd.clear()
                self.pd.update(d, _t=time.monotonic())

        def on_force(key: str):
            def cb(m) -> None:   # Float64MultiArray 에는 header 가 없다 — 받은 시각으로 끊김을 본다
                self.cnt[0] += 1
                self.force[key] = (np.asarray(m.data, float), time.monotonic())
            return cb

        n = self.node
        n.create_subscription(JointState, STATE_TOPIC, on_js(self.arm), qos_profile_sensor_data)
        n.create_subscription(JointState, HAND_STATE.format(side=side), on_js(self.hand), qos_profile_sensor_data)
        n.create_subscription(String, PD_STATUS.format(side=side), on_pd, 10)
        n.create_subscription(Float64MultiArray, JF_TOPIC.format(side=side), on_force("jf"), 10)
        n.create_subscription(Float64MultiArray, TIP_TOPIC.format(side=side), on_force("tip"), 10)
        self.img, self.info, self.img_t, self.n_img, self.cam_ex = None, None, deque(maxlen=90), [0], None
        if photo_topic:   # 사진: 원본 컬러를 별 노드 · 별 스레드 실행기로 — 받기(역직렬화)가 60 Hz 제어 spin 을 막지 않게. 콜백은 참조만 둔다
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import CameraInfo, Image

            def on_img(m) -> None:
                t = time.monotonic()
                self.img, self.n_img[0] = (m, t), self.n_img[0] + 1
                self.img_t.append(t)

            def on_info(m) -> None:
                self.info = np.asarray(m.k, float).reshape(3, 3)

            cn = self.cam_node = rclpy.create_node(f"board_jog_{side}_photo")
            cn.create_subscription(Image, photo_topic, on_img, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            cn.create_subscription(CameraInfo, photo_topic.rsplit("/", 1)[0] + "/camera_info", on_info, qos_profile_sensor_data)
            self.cam_ex = SingleThreadedExecutor()
            self.cam_ex.add_node(cn)
            threading.Thread(target=self.cam_ex.spin, daemon=True).start()

    def spin(self) -> None:
        for _ in range(SPIN_MAX):   # touch_probe.spin_all
            n0 = self.cnt[0]
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
            if self.cnt[0] == n0:
                break

    @staticmethod
    def _fresh(store: dict, names: list) -> np.ndarray | None:
        now = time.monotonic()
        if not all(n in store and now - store[n][1] < STALE_S for n in names):
            return None
        v = np.array([store[n][0] for n in names], float)
        return v if np.all(np.isfinite(v)) else None

    def _fresh_force(self, key: str, index: int) -> np.ndarray | None:   # touch_probe.run 의 fresh_force(:693)
        v = self.force.get(key)
        return None if v is None or time.monotonic() - v[1] > FORCE_STALE_S else clean_force(v[0], index)

    def read(self, now: float) -> dict:
        return {"meas": self._fresh(self.arm, self.arm_src), "hmeas": self._fresh(self.hand, self.hand_names), "pd": self.pd,
                "jf": self._fresh_force("jf", INDEX_JF), "tip": self._fresh_force("tip", INDEX_TIP)}

    def open_pub(self) -> int:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        if self.pub is None:
            self.pub = self.node.create_publisher(self.JointState, TOPIC, 10)
            latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.ep_pub = self.node.create_publisher(self.String, EPISODE_TOPIC_SIDE.format(side=self.side), latched)
        self.ep_id = replay_episode_id()
        return self.ep_id

    def event(self, name: str, **kw) -> None:
        self.ep_pub.publish(self.String(data=json.dumps({"episode": self.ep_id, "event": name, "t_ns": time.time_ns(), **kw})))

    def send(self, k: int, qa: np.ndarray, qh: np.ndarray) -> None:
        m = self.JointState()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = f"{self.ep_id}:{k}"
        m.name = list(P.arm_names(self.side)) + self.hand_names
        m.position = [float(v) for v in np.concatenate([qa, qh])]
        m.velocity = [0.0] * len(m.name)
        m.effort = [0.0] * len(m.name)
        self.pub.publish(m)

    def pd_target(self):
        return self.pd.get("target")

    def image(self):
        """(마지막 원본 컬러 msg, 받은 monotonic) 또는 None — 처리는 사진 Worker 가."""
        return self.img

    def cam_info(self):
        return self.info

    def cam_hz(self) -> float:
        now = time.monotonic()
        return sum(now - t < 2.0 for t in list(self.img_t)) / 2.0

    def close(self) -> None:
        if self.cam_ex is not None:
            self.cam_ex.shutdown(timeout_sec=1.0)
            self.cam_node.destroy_node()
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


class FakeIO:
    """--demo · selftest: 실측 = 발행(명령 + c) + 처짐(sag(q), 없으면 E0), 손끝(명령 FK)이 보드면 zc 아래로 1 mm 당 검지 힘 400 g · 촉각 0.5 N. 발행 없음."""
    E0 = np.array([0.0, 0.004, 0.0, -0.003, 0.0, 0.01, 0.0])

    def __init__(self, side: str, kin: Kin, home: np.ndarray, zc: float):
        self.side, self.kin, self.home, self.zc, self.jog = side, kin, home, zc, None
        self.stale, self.dz = False, 0.0   # 시험 주입: 힘 토픽 끊김 · 실제 면 높이 어긋남(m, 맞춤 · 표시와 다르게)
        self.sag = None   # 시험 주입: 자세 의존 처짐 q_pub → rad(7)
        self.img, self.info = None, None   # 시험 주입: (가짜 Image, 받은 시각) · K 3x3

    def spin(self) -> None:
        pass

    def read(self, now: float) -> dict:
        cmd = self.home if self.jog is None else self.jog.cmd
        qp = self.home if self.jog is None else self.jog.pub()
        hand = np.asarray(POINT) if self.jog is None else self.jog.hand
        depth = max(0.0, self.zc + self.dz - float(self.kin.tip(cmd, hand)[2, 3])) * 1000
        jf, tip = np.zeros(6), np.zeros(5)
        jf[INDEX_JF], tip[INDEX_TIP] = -400.0 * depth, 0.5 * depth
        return {"meas": qp + (self.E0 if self.sag is None else self.sag(qp)), "hmeas": hand.copy(), "jf": None if self.stale else jf, "tip": None if self.stale else tip,
                "pd": {"phase": "TRACKING", "target": "internal", "_t": now, "arms": {self.side: {"q_set": self.home.tolist()}}}}

    def pd_target(self):
        return "internal"

    def image(self):
        return self.img

    def cam_info(self):
        return self.info

    def cam_hz(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ 제어 스레드(60 Hz) — 창 그리기와 떨어져 돈다
class Ctl:
    def __init__(self, io, kin: Kin, lo, hi, home: np.ndarray, grip: np.ndarray, execute: bool, demo: bool,
                 out: Path | None, doc: dict | None, clock=time.monotonic, z_floor: float = -np.inf, hold: bool = True,
                 auto_all: bool = False, confirm_first: int = 2, photo_dir: Path | None = None, extrinsics: Path | None = None, precise: bool = False,
                 continuous_hold: bool = True):
        self.io, self.kin, self.home, self.grip, self.execute, self.demo, self.out, self.doc = io, kin, home, grip, execute, demo, out, doc
        self.precise, self.dt = precise, 1.0 / 120.0 if precise else DT
        self.hold_on, self.hold_state, self.t_contact, self.t_move = bool(hold), "dry", -np.inf, -np.inf   # 유지 보정 켬 · 상태 · 마지막 접촉 · 이동
        self.continuous_hold = bool(continuous_hold)
        self.clock, self.lo, self.hi, self.z_floor = clock, lo, hi, float(z_floor)
        self.jog = Jogger(kin, lo, hi, home, grip, self.z_floor, precise=self.precise)
        self.det = Detector()
        self.rows: list = []
        self.req: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.snap: dict = {}
        self.msg, self.start_fail, self.start_info = "시작 검사 중…", ["시작 검사 전"], {}
        self.pub, self.k, self.sp_est, self.last, self.q_start = False, 0, None, None, home.copy()
        self.sig, self.r = None, {}
        self.records_ver, self.alive, self.quit, self.fatal = 0, True, threading.Event(), None
        self.n_put, self.n_done = 0, 0   # 넣은 · 처리한 창 요청 수 — 누르고 있기 반복이 이동이 끝난 뒤에만 다음을 넣게
        self.view: dict = {"boards": {}, "target": None}   # 창이 고른 보드 자세 · 목표(base) — sim 송신용
        self.t_prev, self.t_check, self.published = None, None, False
        self.t_flush, self.n_part, self.saver = None, 0, None
        self.fits, self.fit_ver = {}, 0   # 기록으로 맞춘 보드 자세(board_fit) · 판 번호(창이 바뀐 것만 다시 그린다)
        self.last_rec = None   # (board_id, target_id, 손끝 xyz, 시각) — 겹친 창 중복 기록 막기
        self.auto, self.auto_free, self.auto_skip = None, False, set()   # 자동 접근 상태 · 확인 없이 하강 · 건너뛴 (보드, 목표 index)
        bd = (doc or {}).get("boards", {})
        self.sel_bid = next(iter(bd), None)   # 창에서 고른 보드(자동은 이 보드 안에서만)
        self.est_yaw = {b: A.yaw_deg(np.asarray(v["pose_display"]["T_base_board"], float)) for b, v in bd.items()
                        if "T_base_board" in v.get("pose_display", {})}   # 맞춤 전 표시 자세 yaw(L2 180° 순서 거부 기준)
        self.auto_all, self.confirm_first, self.all_run, self.sum_bid = bool(auto_all), int(confirm_first), False, None
        self.photo_dir, self.photo = photo_dir, (PH.Worker(extrinsics) if photo_dir is not None else None)
        self.summary, self.photo_note, self.cam = None, "", None   # 마지막 자동 요약 · 사진 알림 · 카메라 관리(PH.Camera, main 이 넣는다)
        if doc is not None and doc["points"]:   # --seed: 불러온 점으로 처음 맞춤(4 모서리 Kabsch) · 그 보드를 고른다
            self.sel_bid = next((p["board_id"] for p in doc["points"] if "seeded_from" in p), self.sel_bid)
            self._commit(doc)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # ---------------- 시작
    def start_check(self, t_wait: float = MEAS_WAIT_S) -> None:
        """측정 · pd · 영점 힘이 올 때까지(MEAS_WAIT_S) 모으고 touch_probe dry 처럼 출력. 실패해도 창은 dry 로 간다."""
        t0 = self.clock()
        while True:
            self.io.spin()
            now = self.clock()
            r = self.io.read(now)
            self._row(now, r)
            ok = all(r[k] is not None for k in ("meas", "hmeas", "jf", "tip")) and r["pd"]
            if (ok and now - t0 >= max(E0_WIN_S, BASE_S) + 0.1) or now - t0 > t_wait or self.quit.is_set():
                break
            time.sleep(0.01)
        meas, hmeas = r["meas"], r["hmeas"]
        base = self._force_base(now)
        self.det.begin(now, *(base or (0.0, 0.0)), 0.0)
        q_set = arm_hold_target(r["pd"], self.kin.side, meas)[0] if meas is not None else None
        cmd = q_set if q_set is not None else (meas if meas is not None else self.home)
        self.jog = Jogger(self.kin, self.lo, self.hi, cmd, hmeas if hmeas is not None else self.grip, self.z_floor, precise=self.precise)
        self.q_start = cmd.copy()
        if isinstance(self.io, FakeIO):
            self.io.jog = self.jog
        self._recheck(now, r)
        self.start_info["force_base"] = base
        print(json.dumps({"phase": "start_check", **self.start_info}, ensure_ascii=False), flush=True)

    def _recheck(self, now: float, r: dict) -> None:
        """시작 검사 판정(발행 꺼져 있을 때 1 s 마다 다시). 홈 검사는 첫 발행 전까지만 — 그 뒤 q_start 는 첫 발행 자세."""
        first, self.t_check = self.t_check is None, now
        meas, hmeas = r["meas"], r["hmeas"]
        fail = []
        if meas is None or hmeas is None:
            fail.append(f"측정 없음 — {STATE_TOPIC} 또는 {HAND_STATE.format(side=self.kin.side)}")
        if r["jf"] is None or r["tip"] is None:
            fail.append(f"{JF_TOPIC.format(side=self.kin.side)} · {TIP_TOPIC.format(side=self.kin.side)} 가 안 온다 — 홈 tip_zero 먼저")
        home_err = None if meas is None else float(np.abs(meas - self.home).max())
        if home_err is not None and home_err > START_TOL_RAD and not self.published:
            fail.append(f"팔이 홈이 아니다(최대 {home_err:.3f} rad > {START_TOL_RAD})")
        if why := pd_fault(r["pd"], now):
            fail.append(why)
        q_set, why = arm_hold_target(r["pd"], self.kin.side, meas) if meas is not None else (None, "")
        if meas is not None and q_set is None:
            fail.append(why)
        if not self.execute:
            fail.append("--execute 없이 시작(dry)")
        changed = first or bool(fail) != bool(self.start_fail)   # 통과 ↔ 막힘이 바뀔 때만 상태줄을 덮는다
        self.start_fail = fail
        self.start_info = {"side": self.kin.side, "q_meas": None if meas is None else np.round(meas, 4).tolist(),
                           "hand_meas": None if hmeas is None else np.round(hmeas, 3).tolist(), "home": np.round(self.home, 4).tolist(),
                           "home_err_rad": None if home_err is None else round(home_err, 4),
                           "jf_zeroed": None if r["jf"] is None else np.round(r["jf"], 1).tolist(),
                           "tip_zeroed": None if r["tip"] is None else np.round(r["tip"], 3).tolist(),
                           "pd_phase": (r["pd"] or {}).get("phase"), "refuse_publish": fail}
        if changed:
            self.msg = "시작 검사 통과 — '실기 발행' 을 켤 수 있다" if not fail else "실기 발행 막힘: " + " / ".join(fail)

    # ---------------- 루프
    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        try:
            self.start_check()
            t_next = self.clock()
            while not self.quit.is_set():
                self.step(self.clock())
                t_next = max(t_next + self.dt, self.clock())
                time.sleep(max(0.0, t_next - self.clock()))
        except BaseException as error:   # noqa: BLE001 — 어떤 오류든 누름을 풀고 stop 으로
            self.fatal = f"{type(error).__name__}: {error}"
            say(f"★ 제어 스레드 오류: {self.fatal}\n{traceback.format_exc()}", err=True)
        finally:
            self._finish(self.fatal or "창 닫기 · 정지 신호")
            self.alive = False

    def _finish(self, why: str) -> None:
        if self.pub:
            self._release(f"끝: {why}")
        if self.saver is not None:
            self.saver.join()
        try:
            if self.photo is not None:
                self.photo.close()
                self._drain_photos()
            if self.sum_bid is not None:   # 마지막 사진까지 넣은 요약
                self._summarize("종료")
        except Exception as error:   # noqa: BLE001 — 원본 저장 · io 닫기는 이어 간다
            say(f"★ 끝 사진 · 요약 저장 실패: {type(error).__name__}: {error}", err=True)
        self._flush_rows(self.clock(), final=True)
        try:
            self.io.close()
        except Exception as error:   # noqa: BLE001
            say(f"io 닫기 실패: {error}", err=True)

    def _flush_rows(self, now: float, final: bool = False) -> None:
        """원본 틱 조각 내보내기 — 최근 RAW_KEEP_S 만 메모리에 남긴다(final 이면 전부, 이 스레드에서 바로 저장)."""
        self.t_flush = now
        cut = len(self.rows) if final else next((i for i, row in enumerate(self.rows) if row[0] >= now - RAW_KEEP_S), len(self.rows))
        old, self.rows = self.rows[:cut], self.rows[cut:]
        if self.out is None or not old:
            return
        path, self.n_part = raw_part(self.out, self.n_part), self.n_part + 1
        if final:
            self._save_part(path, old)
            return
        if self.saver is not None:
            self.saver.join()
        self.saver = threading.Thread(target=self._save_part, args=(path, old), daemon=True)   # 저장이 60 Hz 발행을 막지 않게
        self.saver.start()

    def _save_part(self, path: Path, rows: list) -> None:
        try:
            n = save_raw(path, rows, self.kin.side)
            say(f"원본 {n} 틱 → {path}")
        except Exception as error:   # noqa: BLE001
            say(f"★ 원본 조각 저장 실패 {path}: {type(error).__name__}: {error}", err=True)

    def stop(self, timeout: float = 15.0) -> bool:
        """→ 제어 스레드가 끝났나(해제 · stop · 원본 저장까지)."""
        self.quit.set()
        if self.thread.is_alive():
            self.thread.join(timeout)
        return not self.thread.is_alive()

    def call(self, name: str, *args) -> None:
        self.n_put += 1
        self.req.put((name, args))

    def _row(self, now: float, r: dict) -> None:
        nan = np.full
        self.rows.append((now, time.time_ns(), self.k, nan(7, np.nan) if r["meas"] is None else r["meas"], self.jog.cmd.copy(),
                          nan(6, np.nan) if r.get("hmeas") is None else r["hmeas"], self.jog.hand.copy(),
                          nan(6, np.nan) if r["jf"] is None or len(r["jf"]) != 6 else r["jf"],
                          nan(5, np.nan) if r["tip"] is None or len(r["tip"]) != 5 else r["tip"],
                          self.sig in ("index_force", "tip_touch"), self.pub, self.jog.c.copy(),
                          self.hold_state, getattr(self, "actual_dt", self.dt),
                          self.kin.tip(self.jog.cmd, self.jog.hand),
                          nan((4, 4), np.nan) if r["meas"] is None or r.get("hmeas") is None else self.kin.tip(r["meas"], r["hmeas"]),
                          self.jog.pub().copy() if self.pub else nan(7, np.nan)))

    def _window(self, now: float, w: float) -> list:
        out = []
        for row in reversed(self.rows):
            if row[0] < now - w:
                break
            out.append(row)
        return out[::-1]

    def _force_base(self, now: float) -> tuple[float, float] | None:
        W = [r for r in self._window(now, BASE_S) if np.isfinite(r[7][INDEX_JF]) and np.isfinite(r[8][INDEX_TIP])]
        return None if len(W) < 3 else (float(np.mean([r[7][INDEX_JF] for r in W])), float(np.mean([r[8][INDEX_TIP] for r in W])))

    def step(self, now: float) -> None:
        dt = 0.0 if self.t_prev is None else min(max(now - self.t_prev, 0.0), 0.1)
        self.actual_dt = self.dt if self.t_prev is None else now - self.t_prev
        self.t_prev = now
        self.io.spin()
        r = self.r = self.io.read(now)
        meas = r["meas"]
        if meas is not None:
            self.last = meas
        jf = None if r["jf"] is None else float(r["jf"][INDEX_JF])
        tip = None if r["tip"] is None else float(r["tip"][INDEX_TIP])
        self.sig = self.det.touch(now, jf, tip)
        contact, stale = self.sig in ("index_force", "tip_touch"), self.sig == "force_stale"
        while not self.req.empty():
            name, args = self.req.get_nowait()
            try:
                self.msg = getattr(self, "_req_" + name)(now, contact, stale, *args) or self.msg
            except Exception as error:   # noqa: BLE001
                self.msg = f"{name} 실패: {type(error).__name__}: {error}"
            finally:
                self.n_done += 1
        if self.jog.mode == "dwell":
            self._dwell(now)
        force = None if jf is None or tip is None else (abs(jf - self.det.jf0), tip - self.det.tip0)
        note = self.jog.tick(now, dt, self.last, contact, stale, force)
        if note:
            self.msg = note
        self._auto_step(now, contact, stale, note)
        if self.all_run and self.auto is None:   # 전체 자동이 어떤 이유로든 끝남(남은 점 없음 · 정지 · 이상 · 해제) → 요약
            self.all_run = False
            self._summarize(self.msg)
        if self.photo is not None:
            self._drain_photos()
        if self.pub:
            fault = None
            if meas is None:
                fault = f"팔 측정 끊김({STALE_S} s)"
            elif f := pd_fault(r["pd"], now, MOVING):
                fault = f
            else:
                err = np.abs(meas - self.jog.cmd - self.jog.e0)   # |실측 − q_int|(유지 켬이면 e0 = 0 이라 |실측 − 명령|)
                if np.any(err > TRACK_TOL_MOVE):
                    j = int(np.argmax(err - TRACK_TOL_MOVE))
                    fault = f"추종 오차 j{j + 1} {err[j]:.3f} rad > {TRACK_TOL_MOVE[j]}"
            if fault:
                self._release(fault)   # c 는 그 자리(이 틱 적분 없이)
            else:
                self._hold_step(now, dt, meas, contact, stale)
                self._send()
        else:
            self._hold_step(now, dt, meas, contact, stale)   # 발행 안 할 때는 demo 만 적분(dry 는 얼림)
        if not self.pub and (self.t_check is None or now - self.t_check >= 1.0):
            self._recheck(now, r)
        self._row(now, r)
        if self.t_flush is None:
            self.t_flush = now
        elif now - self.t_flush >= RAW_PART_S:
            self._flush_rows(now)
        self._publish_snap(now, contact, stale)

    def _hold_step(self, now: float, dt: float, meas: np.ndarray | None, contact: bool, stale: bool) -> None:
        """유지 보정 한 틱: c += HOLD_K·(q_int − 실측)·dt, 관절당 변화율 · 크기 · 관절 한계 상한. 얼림 = c 를 그 자리에(hold_state = 이유)."""
        jg, a = self.jog, self.auto
        if contact:
            self.t_contact = now
        if jg.mode == "move":
            self.t_move = now
        why = ("끔(--no_hold)" if not self.hold_on else "dry" if not (self.pub or self.demo) else "측정 끊김" if meas is None
               else "힘 끊김" if stale else "접촉" if now - self.t_contact < REC_WIN   # 접촉 뒤 기록 창 길이까지
               else jg.mode if jg.mode in ("dwell", "descend", "brake") else "이동 중" if not self.precise and now - self.t_move < HOLD_SETTLE_S else f"자동 {a['phase']}" if a and a.get("phase") in HOLD_FREEZE_AUTO
               else None)
        self.hold_state = why or "동작"
        if why or dt <= 0:
            return
        if not self.continuous_hold:
            # Preserve the bias captured at publication start. Never chase
            # stationary friction by integrating feedback into the motor goal.
            self.hold_state = "보정 고정"
            return
        if self.precise:
            from precise_jog import bias_step
            hand = self.r.get("hmeas")
            if hand is None:
                self.hold_state = "손 측정 끊김"
                return
            # Keep the nominal tip goal fixed; solve its joint reference with the REAL hand.
            target = self.kin.tip(jg.cmd, jg.hand)
            reference, reason = ik(self.kin, jg.cmd, target, jg.lo, jg.hi, hand)
            if reference is None:
                self.hold_state = f"손 FK 보정 거부: {reason}"
                return
            c = bias_step(jg.c, reference - meas, dt)
        else:
            dc = np.clip(HOLD_K * (jg.cmd + jg.e0 - meas) * dt, -HOLD_RATE * dt, HOLD_RATE * dt)
            c = np.clip(jg.c + dc, -HOLD_MAX_RAD, HOLD_MAX_RAD)
        c = np.where((jg.cmd + c < jg.lo) | (jg.cmd + c > jg.hi), jg.c, c)   # 관절 한계로 가는 관절은 그 자리
        if not np.all(np.isfinite(c)):
            return
        hands = [jg.hand]
        if self.precise and self.r.get("hmeas") is not None:
            hands.append(self.r["hmeas"])
        for hand in hands:
            z0, z1 = float(self.kin.tip(jg.cmd + jg.c, hand)[2, 3]), float(self.kin.tip(jg.cmd + c, hand)[2, 3])
            if z1 < jg.z_floor and z1 < z0:
                self.hold_state = "z 하한"
                return
        jg.c = c

    def _send(self) -> None:
        cmd, hand = self.jog.pub(), self.jog.hand
        if not (np.all(np.isfinite(cmd)) and np.all(np.isfinite(hand))):   # touch_probe Probe.outgoing — codec 이 NaN 목표를 거부한다
            self.msg = "비유한 명령 — 보내지 않음, 지난 명령으로"
            self.jog.cmd = self.sp_est - self.jog.c if self.sp_est is not None else self.jog.cmd   # sp_est 는 발행 공간
            return
        self.io.send(self.k, cmd, hand)
        self.k += 1
        # Prefer the actual PD setpoint over a guessed reduced-stage ramp at high speed.
        measured_set, _ = arm_hold_target(self.r.get("pd") or {}, self.kin.side, self.last) if self.precise else (None, None)
        self.sp_est = measured_set if measured_set is not None else (cmd.copy() if self.sp_est is None else self.sp_est + np.clip(
            cmd - self.sp_est, -PD_VEL_RAD_S * self.dt, PD_VEL_RAD_S * self.dt))

    def _release(self, why: str) -> None:
        """명령 = 실측 − e0, 발행 = 그 + c(얼린 c — 빼면 팔이 c 만큼 처져 누른다) → release_and_stop(해제 송신 → episode stop →
        pd internal 확인). 발행 끝."""
        self.jog.hold(self.last, emergency=True)
        self.jog.mode, self.auto, self.auto_free = "idle", None, False
        ns = SimpleNamespace(cmd=self.jog.pub(), sp_est=self.sp_est, dt=self.dt)

        def send() -> None:
            self.io.send(self.k, self.jog.pub(), self.jog.hand)
            self.k += 1

        def stop_event() -> None:
            self.io.event("stop", object_anchor=None, home_q={}, reasons=[f"board_jog 끝: {why}"])

        try:
            T, ok = release_and_stop(ns, send, stop_event, self.io.spin, self.io.pd_target)
        except Exception as error:   # noqa: BLE001
            T, ok = 0.0, False
            say(f"★★★ 해제 · stop 실패: {type(error).__name__}: {error} — pd 상태를 확인한다 ★★★", err=True)
        self.pub = False
        self.msg = f"실기 발행 끔 — {why} (해제 {T:.2f} s, stop {'확인' if ok else '★확인 실패 — pd 상태를 본다'})"
        say(self.msg)

    def _dwell(self, now: float) -> None:
        """하강 전 정지: e0_local(발행 · demo) · 힘 기준선을 잰 뒤 하강 시작. E0_SETTLE_MAX_S 안에 못 재면 취소."""
        if now - self.jog.t_mode > E0_SETTLE_MAX_S:
            self.jog.mode, self.msg = "idle", f"하강 취소 — {E0_SETTLE_MAX_S:g} s 안에 팔이 멈추지 않았다"
            return
        if now - self.jog.t_mode < E0_WIN_S:
            return
        if self.pub or self.demo:   # 팔이 명령을 따를 때만 처짐을 잰다(dry 가상 명령은 실측과 무관)
            W = [r for r in self._window(now, E0_WIN_S) if np.all(np.isfinite(r[3]))]
            e0, _why = settled_e0([r[0] for r in W], [r[3] for r in W], self.jog.cmd)
            if e0 is None:
                return
            if not self.hold_on:   # 유지 켬: 명령 = 실측이 갈 곳(e0 = 0), 남은 차는 c 가 맡는다 — 멈춘 것만 확인
                self.jog.e0 = e0
        base = self._force_base(now)
        if base is None:
            self.jog.mode, self.msg = "idle", "하강 취소 — 힘 기준선을 못 잼(힘 토픽)"
            return
        if abs(base[0]) > PRE_JF_G or base[1] >= TIP_N:
            self.jog.mode, self.msg = "idle", f"하강 취소 — 이미 닿은 값(검지 {base[0]:.0f} g · 촉각 {base[1]:.2f} N, 홈 영점 대비)"
            return
        self.det.begin(now, base[0], base[1], 0.0)
        self.jog.start_descent(now)
        self.msg = f"하강 {DESC_V * 1000:.0f} mm/s — 기준선 {base[0]:.0f} g · {base[1]:.2f} N, e0 {np.round(self.jog.e0, 3).tolist()}"

    # ---------------- 자동 접근(목표 위 → 사람 확인 → 기존 하강 · 기록). 이동은 조각마다 Jogger.plan — 조그와 같은 검사 · 속도
    def _auto_off(self, why: str, user: bool = False) -> str | None:
        """자동 해제. 사용자가 끈 것이 아니면 '확인 없이 자동 하강' 도 끈다(다음 N 은 확인부터)."""
        if self.auto is None:
            return None
        self.auto = None
        if not user:
            self.auto_free = False
        if self.jog.mode != "idle":
            self.jog.hold(self.last, self.t_prev or 0.0)
        self.msg = f"자동 해제 — {why}"
        return self.msg

    def _tip_meas(self) -> np.ndarray:
        """명령 + e0 의 손끝 FK = 팔이 명령을 따를 때 실측 손끝 추정(기록 · 맞춤은 실측 FK 라 목표도 이 공간)."""
        return self.kin.tip(self.jog.cmd + self.jog.e0, self.jog.hand)

    def _chunk(self, pos, R, contact: bool, stale: bool, step: float = AUTO_STEP_M, tip_v: float | None = None) -> tuple[bool, str | None]:
        """실측 손끝을 pos · 손목 R 쪽으로 한 조각(≤ step · AUTO_ROT_DEG) Jogger.plan. → (도착, 거부 이유). 이동 중이면 (False, None)."""
        jg = self.jog
        if jg.mode != "idle":   # 조각이 끝나기 전에는 도착도 다음 조각도 없다
            return False, None
        Tc = self.kin.tip(jg.cmd, jg.hand)
        goal = np.asarray(pos, float) - (self._tip_meas()[:3, 3] - Tc[:3, 3])   # 처짐 e0 만큼 명령 목표를 옮긴다
        dp, rv = goal - Tc[:3, 3], rotvec(np.asarray(R) @ Tc[:3, :3].T)
        dn, rn = float(np.linalg.norm(dp)), float(np.linalg.norm(rv))
        if dn < AUTO_TOL_M and np.degrees(rn) < AUTO_TOL_DEG:
            return True, None
        Tg = Tc.copy()
        Tg[:3, 3] += dp * min(1.0, step / max(dn, 1e-12))
        Tg[:3, :3] = Rot.from_rotvec(rv * min(1.0, np.radians(AUTO_ROT_DEG) / max(rn, 1e-12))).as_matrix() @ Tc[:3, :3]
        return False, jg.plan(Tg, contact, stale, tip_v)

    def _auto_enter(self, ph: str, now: float) -> None:
        a, cur = self.auto, self._tip_meas()[:3, 3]
        x, y, z = a["pos"]
        zs = max(cur[2], a["top"] + AUTO_CLEAR_M)   # 보드 면 최고점 + 30 mm 보다 낮으면 그만큼 올리고, 높으면 그 높이로 수평 이동
        if ph == "rise":   # 올림은 순수 +z — 손목은 지금 명령 그대로(회전은 travel 에서)
            a["Rr"] = self.kin.tip(self.jog.cmd, self.jog.hand)[:3, :3]
        a["phase"], a["t"] = ph, now
        a["goal"] = {"rise": np.r_[cur[:2], zs], "travel": np.r_[x, y, zs], "approach": np.r_[x, y, z + AUTO_HOVER_M]}.get(ph)
        self.msg = f"자동 {a['name']}: {AUTO_PH[ph]}"

    def _auto_next(self, now: float) -> None:
        """고른 보드의 남은 목표 중 손끝에서 가장 가까운 것으로(맞춤은 사람 기록마다 새로 — 사람 점 8 개부터 전체 맞춤).
        n = 이번 N 에서 몇 번째 목표, fv = 받아들인 맞춤 판(자동 밖에서 바뀌면 해제)."""
        prev, b = self.auto or {}, self.sel_bid
        n, fv, f = prev.get("n", 0), prev.get("fv", self.fit_ver), self.fits.get(b)
        q = [] if b is None else auto_queue(self.doc["points"] if self.doc else [], self.fits, self.auto_skip, self._tip_meas()[:3, 3], b)
        self.auto = prev
        if f is None or not f["ok"]:
            self._auto_off(f"{b} 맞춤 없음 · 거부" + (f"({f['why']})" if f else ""))
        elif not q:
            self._auto_off("남은 목표 없음" + (f"(건너뛴 {len(self.auto_skip)} 개)" if self.auto_skip else ""))
        elif n >= AUTO_MAX_PTS and not self.auto_all:
            self._auto_off(f"N 한 번에 {AUTO_MAX_PTS} 점 — 기록을 보고 N 을 다시")
        else:
            _, i, x = q[0]
            self.auto = {"bid": b, "ti": i, "name": BOARDS[b]["names"][i], "pos": np.asarray(x, float), "R": f["R_wrist"],
                         "top": float(to_base(BOARDS[b]["targets"], f["T"])[:, 2].max()), "free": False, "n": n + 1, "fv": fv}
            self._auto_enter("rise", now)

    def _auto_step(self, now: float, contact: bool, stale: bool, note: str | None) -> None:
        a = self.auto
        if a is None:
            return
        ph = a["phase"]
        if a["bid"] != self.sel_bid:
            self._auto_off(f"보드 바뀜({a['bid']} → {self.sel_bid}) — 보드를 고르고 N 다시")
        elif self.fit_ver != a["fv"]:   # 자동 밖에서(지움 · 사람 기록) 맞춤이 바뀌었다 — 목표 자리를 다시 본다
            self._auto_off("보드 맞춤이 바뀜 — 확인하고 N 다시")
        elif stale:
            self._auto_off("힘 끊김")
        elif note and ph in AUTO_MOVES:   # 이동 중 새 접촉 · 접촉 중 힘 상승 정지(Jogger 가 이미 명령 = 실측)
            self._auto_off(note)
        elif contact and ph in ("travel", "approach", "hover", "wait", "converge"):
            self._auto_off("목표 위에서 접촉 이상 — 보드 자세 · 높이를 확인한다")
        elif ph in ("rise", "travel", "approach"):
            done, why = self._chunk(a["goal"], a["Rr"] if ph == "rise" else a["R"], contact, stale,
                                    *((AUTO_APPROACH_STEP_M, AUTO_APPROACH_V) if ph == "approach" else (AUTO_STEP_M, None)))
            if why:
                self._auto_off(f"거부: {why}")
            elif done and ph != "approach":
                self._auto_enter("travel" if ph == "rise" else "approach", now)
            elif done:
                a["phase"], a["t"] = "hover", now
                self.msg = f"자동 {a['name']}: {AUTO_PH['hover']}"
        elif ph == "hover":
            if now - a["t"] < AUTO_HOVER_S:
                return
            base = self._force_base(now)
            if base is None:
                self._auto_off("목표 위 힘 기준선을 못 잼(힘 토픽)")
                return
            if abs(base[0]) > PRE_JF_G or base[1] >= TIP_N:
                self._auto_off(f"목표 위 {AUTO_HOVER_M * 1000:.0f} mm 에서 이미 닿은 값(검지 {base[0]:.0f} g · 촉각 {base[1]:.2f} N) — 보드 높이를 확인한다")
                return
            self.det.begin(now, base[0], base[1], 0.0)
            if self.auto_all:   # 전체 자동: Enter 대신 유지 보정 수렴(hold 가 이 단계에선 돈다)
                a["phase"], a["t"], a["t_ok"] = "converge", now, None
                self.msg = f"자동 {a['name']}: {AUTO_PH['converge']}"
                return
            a["phase"], a["free"] = "wait", self.auto_free and a["n"] > 1   # 확인 없이여도 N 뒤 첫 점은 확인
            self.msg = auto_wait_text(a["name"])
            self._auto_down(now, contact, stale)
        elif ph == "converge":
            ok, txt = self._converged()
            a["t_ok"] = (a["t_ok"] if a["t_ok"] is not None else now) if ok else None
            if a["t_ok"] is not None and now - a["t_ok"] >= AUTO_CONV_HOLD_S:
                a["phase"], a["free"] = "wait", a["n"] > self.confirm_first   # 처음 confirm_first 점은 Enter 확인(human)
                self.msg = auto_wait_text(a["name"]) + f" · 유지 수렴 {txt}" + ("" if a["free"] else f" · 확인 {a['n']}/{self.confirm_first}")
                self._auto_down(now, contact, stale)
            elif now - a["t"] > AUTO_CONV_MAX_S:
                self._auto_skip_point(now, f"유지 수렴 {AUTO_CONV_MAX_S:g} s 넘음(손끝 실측−명령 {txt}, 기준 ≤ {AUTO_CONV_MM:g} mm · {AUTO_CONV_DEG:g}°)")
            else:
                self.msg = f"자동 {a['name']}: {AUTO_PH['converge']} — 손끝 실측−명령 {txt}(≤ {AUTO_CONV_MM:g} mm · {AUTO_CONV_DEG:g}° 를 {AUTO_CONV_HOLD_S:g} s)"
        elif ph == "wait":
            if self.jog.mode in ("dwell", "descend"):   # [천천히 하강] · Enter
                a["phase"] = "descend"
        elif ph == "descend":
            if self.jog.mode in ("dwell", "descend", "brake"):
                return
            if contact:
                a["phase"], a["t"] = "settle", now
                self.msg = f"접촉 — {REC_WIN:g} s 뒤 기록 조건이면 자동 기록"
            else:
                self._auto_off(f"하강이 접촉 없이 끝남({self.msg})")
        elif ph == "settle" and now - a["t"] >= REC_WIN:
            f = self.fits.get(a["bid"])
            if f is None or not f["ok"]:
                self._auto_off("기록 전 맞춤이 거부 · 없음 — 기록하지 않는다")
                return
            try:
                m = self._req_record(now, contact, stale, a["bid"], a["ti"], f["T"].copy(), A.fit_pose(f), "auto" if a["free"] else "human")
            except Exception as error:   # noqa: BLE001 — 저장 실패(기록 · 파일 그대로, _commit) · 맞춤 사라짐: 자동만 끈다
                self._auto_off(f"기록 실패: {type(error).__name__}: {error}")
                return
            if not m.startswith("기록 #"):
                if now - a["t"] > AUTO_REC_WAIT_S:
                    self._auto_off(m)
                return
            a["fv"] = self.fit_ver   # 자기 기록으로 바뀐 맞춤은 받는다(다음 목표가 새 맞춤으로)
            Tg = self.kin.tip(self.jog.cmd, self.jog.hand)
            Tg[2, 3] += AUTO_LIFT_M
            why = self.jog.plan(Tg, contact, stale)
            if why:
                self._auto_off(f"{m} · 위로 빼기 거부: {why} — 손끝 닿은 채 정지, 조그 +z(PgUp)로 빼 주세요")
                return
            a["phase"], self.msg = "retreat", f"{m} → 위로 {AUTO_LIFT_M * 1000:.0f} mm"
        elif ph == "retreat" and self.jog.mode == "idle":
            self._auto_next(now)

    def _auto_down(self, now, contact, stale) -> None:
        """확인 없이(a['free']): 사람이 [하강] 누른 것과 같은 길."""
        a = self.auto
        if not a["free"]:
            return
        m = self._req_descend(now, contact, stale)
        if self.jog.mode == "dwell":
            a["phase"], self.msg = "descend", m
        else:
            self._auto_off(m)

    def _converged(self) -> tuple[bool, str]:
        """손끝 실측 FK − 명령(q_int = 명령 + e0) 6D 가 AUTO_CONV 안인가. dry(발행 · demo 아님)는 실측이 명령을 따르지 않아 판정 없음."""
        meas = self.r.get("meas")
        if not (self.pub or self.demo):
            return True, "dry — 판정 없음"
        if meas is None:
            return False, "측정 없음"
        hand = self.r.get("hmeas")
        if hand is None:
            return False, "손 측정 없음"
        e6 = pose_err(self.kin.tip(self.jog.cmd + self.jog.e0, self.jog.hand), self.kin.tip(meas, hand))
        mm, deg = float(np.linalg.norm(e6[:3]) * 1000), float(np.degrees(np.linalg.norm(e6[3:])))
        return mm <= AUTO_CONV_MM and deg <= AUTO_CONV_DEG, f"{mm:.1f} mm {deg:.2f}°"

    def _auto_skip_point(self, now, why: str) -> None:
        """그 점만 건너뛰고(이유는 JSON auto_skipped) 다음 점으로. 기록 실패면 자동을 끈다."""
        a = self.auto
        self.auto_skip.add((a["bid"], a["ti"]))
        if self.doc is not None:
            new = copy.deepcopy(self.doc)
            new.setdefault("auto_skipped", []).append({"board_id": a["bid"], "target_id": a["name"], "why": why, "t_wall_ns": time.time_ns()})
            try:
                self._commit(new, refit=False)
            except Exception as error:   # noqa: BLE001
                self._auto_off(f"건너뜀 기록 실패: {type(error).__name__}: {error}")
                return
        m = f"{a['name']} 건너뜀 — {why}"
        say(m)
        self._auto_next(now)
        self.msg = f"{m} · {self.msg}"

    def _summarize(self, why: str) -> None:
        """이번 실행 점의 맞춤 잔차 · photo_check 요약 → 창(msg) · stdout · JSON auto_summary."""
        if self.doc is None or self.sum_bid is None:
            return
        s = PH.summary(self.doc["points"], self.sum_bid, A.RES_MAX_MM)
        self.summary = PH.summary_text(s)
        say(json.dumps({"phase": "auto_summary", "end": why, **s}, ensure_ascii=False))
        self.msg = f"{why} ‖ 요약: {self.summary}"
        new = copy.deepcopy(self.doc)
        new.setdefault("auto_summary", []).append({**s, "end": why, "t_wall_ns": time.time_ns()})
        try:
            self._commit(new, refit=False)
        except Exception as error:   # noqa: BLE001 — 요약은 창 · stdout 에 이미 있다
            self.msg += f" · ★요약 저장 실패 {type(error).__name__}: {error}"

    def _photo_job(self, now, seq: int, bid: str, name: str, ti: int, tip, tgt) -> tuple[dict, dict | None]:
        """기록 창 안 영상이 있으면 (pending, Worker 일감), 없으면 (이유, None) — 경고만, 기록은 계속."""
        got, K = self.io.image(), self.io.cam_info()
        age = None if got is None else now - got[1]
        why = ("카메라 영상 없음" if got is None else "camera_info 없음" if K is None
               else f"기록 창 안 영상 없음(마지막 영상 {age:.1f} s 전)" if age > PHOTO_WIN_S else None)
        if why:
            return {"status": "none", "why": why}, None
        path = self.photo_dir / f"{seq:03d}_{bid}_{name}.jpg"
        return {"status": "pending", "photo": path.name}, {"i": seq, "msg": got[0], "K": np.asarray(K, float), "tip": np.asarray(tip, float),
                                                           "tgt": np.asarray(tgt, float), "ti": ti, "path": path, "age": round(age, 3),
                                                           "charuco": BOARDS[bid]["type"] == "charuco"}

    def _drain_photos(self) -> None:
        """Worker 가 끝낸 photo_check 를 점에 넣는다(맞춤 · fit_ver 는 그대로 — 자동이 '맞춤 바뀜' 으로 서지 않게)."""
        got = []
        while True:
            try:
                got.append(self.photo.done.get_nowait())
            except queue.Empty:
                break
        if not got or self.doc is None:
            return
        new = copy.deepcopy(self.doc)
        for i, pc in got:
            new["points"][i]["photo_check"] = pc
        try:
            self._commit(new, refit=False)
        except Exception as error:   # noqa: BLE001
            self.photo_note = f"★ 사진 결과 저장 실패: {type(error).__name__}: {error}"
            return
        i, pc = got[-1]
        p = new["points"][i]
        self.photo_note = (f"사진 #{i} {p['target_id']}: " + (f"목표 검출 ↔ 손끝 {pc['px_err']:.1f} px ≈ {pc['mm_err_approx']:.1f} mm"
                           if pc.get("px_err") is not None else f"목표 검출 없음({pc.get('why')})")
                           + (f" · PnP {pc['px_err_pnp']:.1f} px ≈ {pc['mm_err_pnp_approx']:.1f} mm" if pc.get("px_err_pnp") is not None else ""))

    def _req_auto(self, now, contact, stale, on: bool, why: str = "사용자"):
        if not on:
            return self._auto_off(why, user=why == "사용자")
        if self.auto is not None:
            return None
        if self.jog.mode != "idle" or stale:
            return f"자동 거부 — {'힘 끊김' if stale else self.jog.mode + ' 중'}"
        b = self.sel_bid
        f = self.fits.get(b)
        if b is None or f is None or not f["ok"]:
            need = "" if b is None else "·".join(BOARDS[b]["names"][i] for i in BOARDS[b]["outer"])
            return f"자동 거부 — {b}: " + (f["why"] if f is not None else f"바깥 4 모서리({need})를 먼저 사람이 기록한다")
        self.auto_skip.clear()   # L4: 건너뛰기는 N 한 번 안에서만
        self._auto_next(now)
        if self.auto is not None and self.auto_all:
            self.all_run, self.sum_bid = True, b
        return self.msg

    def _req_auto_descend(self, now, contact, stale):
        """Enter: 자동 확인 대기에서만 [천천히 하강]. 그 밖에는 무시(상태 띠에 알림)."""
        a = self.auto
        if a is None or a["phase"] != "wait":
            return f"Enter 무시 — 자동 확인 대기에서만 하강({'자동 꺼짐' if a is None else AUTO_PH[a['phase']]})"
        return self._req_descend(now, contact, stale)

    def _req_board(self, now, contact, stale, bid: str):
        self.sel_bid = bid   # 자동이 다른 보드면 다음 틱 _auto_step 이 해제

    def _req_est(self, now, contact, stale, bid: str, yaw: float):
        self.est_yaw[bid] = float(yaw)   # 창에서 옮긴 표시 자세 yaw — 다음 맞춤의 180° 검사 기준

    def _req_auto_free(self, now, contact, stale, on: bool):
        self.auto_free = bool(on)
        return f"확인 없이 자동 하강 {'켬' if on else '끔'}"

    def _req_skip(self, now, contact, stale):
        a = self.auto
        if a is None or a["phase"] != "wait" or self.jog.mode != "idle":
            return f"건너뛰기는 자동 확인 대기(위 {AUTO_HOVER_M * 1000:.0f} mm, 정지 상태)에서만"
        self.auto_skip.add((a["bid"], a["ti"]))
        self._auto_next(now)
        return f"{a['name']} 건너뜀 · " + self.msg

    def _req_demo_corners(self, now, contact, stale, bid: str):
        """demo · 스모크 전용: 표시 자세에서 (+6, −4 mm, yaw +3°) 옮긴 '실제' 보드의 바깥 4 모서리를 기록한 것처럼 넣는다(demo_synthetic)."""
        if not self.demo or self.doc is None:
            return "demo · --out 전용"
        spec, T0 = BOARDS[bid], np.asarray(self.view["boards"][bid], float)
        c, Rz = to_base(spec["targets"], T0).mean(0), Rot.from_euler("z", 3.0, degrees=True).as_matrix()
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = Rz, c - Rz @ c + [0.006, -0.004, 0.0]
        X, Tt = to_base(spec["targets"], D @ T0), self.kin.tip(self.jog.cmd, self.jog.hand)
        new = copy.deepcopy(self.doc)
        for p in new["points"]:   # demo: 그 보드의 산 점(--seed 포함)은 지운 것으로 — 사람이 틀린 모서리를 지우고 다시 대는 것과 같은 길
            if p["board_id"] == bid and not p["deleted"]:
                p["deleted"], Tt = True, np.r_[np.c_[np.asarray(p["T_tip_fk"], float)[:3, :3], Tt[:3, 3]], [[0, 0, 0, 1]]]   # 손목 = 실제로 댄 방향
                if "q_arm" in p and self.jog.mode == "idle":   # 가상 팔도 그 기록 자세로(홈에서는 IK 가지가 달라 보드에 못 간다 — 실기는 조그로 옮긴다)
                    self.jog.cmd = np.clip(np.asarray(p["q_arm"], float), self.jog.lo, self.jog.hi)
        for i in spec["outer"]:
            T = Tt.copy()
            T[:3, 3] = X[i]
            new["points"].append({"board_id": bid, "target_id": spec["names"][i], "target_xy_board_m": spec["targets"][i, :2].tolist(),
                                  "T_tip_fk": np.round(T, 6).tolist(), "tip_dir_base": np.round(T[:3, 2], 6).tolist(),
                                  "tip_dir_deg": np.round(dir_deg(T[:3, 2]), 2).tolist(), "force": {"index_jf_delta_g": None},
                                  "deleted": False, "aligned_by": "human_corner", "demo_synthetic": True})
        self._commit(new)
        return f"demo: {bid} 바깥 4 모서리 가짜 기록(표시 자세 +6 · −4 mm · +3°)"

    # ---------------- 창 요청(제어 스레드 안에서 실행). → 상태줄 문구
    def _req_jog(self, now, contact, stale, axis: str, sign: int, step: float, frame: str):
        if self.auto is not None and self.auto["phase"] != "wait":
            return f"거부: 자동 이동 중 — 조그는 위 {AUTO_HOVER_M * 1000:.0f} mm 확인 대기에서만(정지 = space)"
        base = self.jog.cmd
        retarget = self.precise and self.jog.mode == "move" and self.auto is None
        if retarget:
            base = self.jog.goal
            remaining = pose_err(self.kin.tip(base, self.jog.hand), self.kin.tip(self.jog.cmd, self.jog.hand))
            if np.linalg.norm(remaining[:3]) > 0.002 or np.linalg.norm(remaining[3:]) > np.radians(.5):
                return "조그 진행 중 — 남은 경로를 먼저 따라감"
        Tg = jog_goal(self.kin.tip(base, self.jog.hand), axis, sign, step, frame)
        why = self.jog.plan(Tg, contact, stale, retarget=retarget)
        unit = f"{step * 1000:g} mm" if AXES.index(axis) < 3 else f"{np.degrees(step):g}°"
        return f"거부: {why}" if why else f"조그 {axis}{'+' if sign > 0 else '−'} {unit} ({frame})"

    def _req_stop(self, now, contact, stale):
        was = self.auto is not None
        self.auto = None   # 사용자 정지 — '확인 없이' 는 그대로
        self.jog.hold(self.last, now, contact=contact)
        return "정지 — 명령 = 실측" + (" · 자동 해제" if was else "")

    def _req_lift(self, now, contact, stale):
        self._auto_off("위로 빼기", user=True)
        if self.jog.mode != "idle":
            self.jog.hold(self.last, now, contact=contact)
        Tg = self.kin.tip(self.jog.cmd, self.jog.hand)
        Tg[2, 3] += LIFT_M
        why = self.jog.plan(Tg, contact, stale)
        return f"위로 빼기 거부: {why}" if why else f"위로 빼기 {LIFT_M * 1000:.0f} mm"

    def _req_descend(self, now, contact, stale):
        if self.auto is not None and self.auto["phase"] != "wait":
            return "하강 거부 — 자동 진행 중"
        if self.jog.mode != "idle":
            return f"{self.jog.mode} 중 — 하강 거부"
        if contact or stale:
            return "하강 거부 — " + ("접촉 중" if contact else "힘 끊김")
        self.jog.mode, self.jog.t_mode = "dwell", now
        return "하강 준비 — 0.5 s 정지 · 기준선"

    def _req_home(self, now, contact, stale):
        self._auto_off("시작 자세 복귀", user=True)
        if contact:
            return "시작 자세 복귀 거부: 접촉 중 — 먼저 위로 빼기"
        why = self.jog.go(self.q_start, contact, stale, now)
        return f"시작 자세 복귀 거부: {why}" if why else "시작 자세로 복귀"

    def _req_point(self, now, contact, stale):
        self._auto_off("손 POINT", user=True)
        jg = self.jog
        if why := jg.refuse_path(jg.cmd, self.grip, contact, stale):   # 명령 · 발행(+ c) 모두
            return f"손 POINT 거부: {why}"
        jg.hand_goal = self.grip.copy()
        return f"손 → {np.round(self.grip, 2).tolist()} ({HAND_VEL} rad/s)"

    def _req_rebase(self, now, contact, stale):
        base = self._force_base(now)
        if base is None:
            return "기준선 못 잡음 — 힘 토픽"
        if abs(base[0]) > PRE_JF_G or base[1] >= TIP_N:
            return f"기준선 거부 — 검지가 닿은 값(검지 {base[0]:.0f} g · 촉각 {base[1]:.2f} N, 홈 영점 대비)"
        self.det.begin(now, base[0], base[1], 0.0)
        return f"기준선 {base[0]:.1f} g · {base[1]:.3f} N"

    def _req_publish(self, now, contact, stale, on: bool):
        if not on:
            if self.pub:
                self._release("사용자가 끔")
            return None
        if self.pub:
            return None
        if not self.execute or self.demo:
            return "--execute 로 시작하지 않아 발행할 수 없다"
        if self.start_fail:
            return "실기 발행 막힘(시작 검사): " + " / ".join(self.start_fail)
        r, meas = self.r, self.r.get("meas")
        why = pd_fault(r["pd"], now) or (None if meas is not None and r["hmeas"] is not None else "팔 · 손 측정 끊김")
        if why:
            return f"실기 발행 거부: {why}"
        q_set, why = arm_hold_target(r["pd"], self.kin.side, meas)
        if q_set is None:
            return f"실기 발행 거부: {why}"
        W = [x for x in self._window(now, E0_WIN_S) if np.all(np.isfinite(x[3]))]
        e0, why = start_e0([x[0] for x in W], [x[3] for x in W], q_set)
        if e0 is None:
            return f"실기 발행 거부: {why}"
        self._auto_off("실기 발행 켬")
        self.io.open_pub()
        self.io.event("hand_mode", hand_command="position", reasons=["board_jog 손 위치 명령"])
        if self.hold_on:   # 처짐 e0 를 c 첫값으로 흡수: 명령 = 지금 실측(q_int), 발행 = 명령 + c = q_set(첫 발행 = pd 세트포인트, 튐 없음)
            self.jog = Jogger(self.kin, self.lo, self.hi, q_set + e0, r["hmeas"], self.z_floor, precise=self.precise)
            self.jog.c = -e0   # |e0| ≤ E0_START_MAX < HOLD_MAX_RAD. 재발행이면 해제 때 얼린 c 와 거의 같다(pd 가 그 발행값을 붙들고 있었다)
        else:
            self.jog = Jogger(self.kin, self.lo, self.hi, q_set, r["hmeas"], self.z_floor, precise=self.precise)
            self.jog.e0 = e0
        self.sp_est, self.pub = q_set.copy(), True
        if not self.published:   # 시작 자세 = 첫 발행 자세만(재발행 때 덮지 않는다, 명령 공간)
            self.q_start, self.published = self.jog.cmd.copy(), True
        return f"★ 실기 발행 켬 — 에피소드 {self.io.ep_id}, e0 {np.round(e0, 3).tolist()}"

    def _req_record(self, now, contact, stale, bid: str, ti: int, T_board: np.ndarray, pose: dict, aligned_by: str | None = None):
        if self.jog.mode == "brake":
            return "기록 거부 — 접촉 정지 경로가 끝날 때까지 대기"
        Wa = self._window(now, REC_WIN)
        W = [r for r in Wa if np.all(np.isfinite(r[3]))]
        flags = [r[9] for r in Wa]
        ok, why = can_record(contact, [r[0] for r in W], [r[3] for r in W], flags)
        if not ok:
            return f"기록 거부 — {why}"
        if self.doc is None:
            return "기록 거부 — --out 없음(demo)"
        spec = BOARDS[bid]
        name, xyz = spec["names"][ti], to_base(spec["targets"], T_board)[ti]
        q = np.median([r[3] for r in W], 0)
        H = [r[5] for r in W if np.all(np.isfinite(r[5]))]
        qh = np.median(H, 0) if H else self.jog.hand.copy()
        F = [r for r in W if np.isfinite(r[7][INDEX_JF]) and np.isfinite(r[8][INDEX_TIP])]
        jfm = np.median([r[7] for r in F], 0) if F else np.full(6, np.nan)
        tipm = np.median([r[8] for r in F], 0) if F else np.full(5, np.nan)
        T = self.kin.tip(q, qh)
        corner = aligned_by is None and ti in spec["outer"] and not (self.fits.get(bid) or {}).get("ok")   # 맞춤 전 모서리(human_corner)
        boards = {**self.view.get("boards", {}), bid: T_board}   # 창에 보이는 모든 보드(고른 보드는 이번 자세로)
        allt = {(n if b == bid else f"{b} {n}"): x for b, Tb in boards.items() for n, x in zip(BOARDS[b]["names"], to_base(BOARDS[b]["targets"], Tb))}
        ok, why = rec_xy_check(T[:3, 3], name, xyz, allt, REC_XY_MAX_CORNER_MM if corner else REC_XY_MAX_MM)
        if not ok:
            return f"기록 거부 — {why}"
        lr = self.last_rec
        if lr and lr[:2] == (bid, name) and np.linalg.norm(T[:3, 3] - lr[2]) * 1000 < REC_DUP_MM and now - lr[3] < REC_DUP_S:
            return f"기록 거부 — 이미 기록함({bid} {name}, {now - lr[3]:.1f} s 전 · 손끝 이동 < {REC_DUP_MM:g} mm)"
        nn = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in np.ravel(a)]   # noqa: E731 — JSON 에 NaN 금지
        pt = {"board_id": bid, "target_id": name, "target_xy_board_m": np.round(spec["targets"][ti, :2], 6).tolist(),
              "q_arm": np.round(q, 6).tolist(), "q_hand": np.round(qh, 5).tolist(), "T_tip_fk": np.round(T, 6).tolist(),
              "tip_dir_base": np.round(T[:3, 2], 6).tolist(), "tip_dir_deg": np.round(dir_deg(T[:3, 2]), 2).tolist(),
              "force": {"jf_zeroed_g": nn(jfm), "tip_zeroed_n": nn(tipm), "jf_base_g": round(self.det.jf0, 2),
                        "tip_base_n": round(self.det.tip0, 4), "index_jf_delta_g": nn(jfm[INDEX_JF] - self.det.jf0)[0],
                        "index_tip_delta_n": nn(tipm[INDEX_TIP] - self.det.tip0)[0], "signal": self.sig,
                        "index_jf": INDEX_JF, "index_tip": INDEX_TIP},
              "contact_frac": round(contact_frac(flags), 3),
              "q_cmd": np.round(self.jog.cmd, 6).tolist(), "e0": np.round(self.jog.e0, 5).tolist(), "publishing": self.pub,
              "hold_c": np.round(self.jog.c, 5).tolist(), "hold_state": self.hold_state,   # 발행 = q_cmd + hold_c(유지 보정)
              "t_wall_ns": time.time_ns(), "frame_id": f"{self.io.ep_id}:{self.k - 1}" if self.pub else None, "tick": self.k,
              "window_s": REC_WIN, "n_window": len(W), "board_T_display": np.round(T_board, 6).tolist(),
              "target_xyz_base_display": np.round(xyz, 6).tolist(), "deleted": False,
              "aligned_by": aligned_by or ("human_corner" if ti in spec["outer"] else "human")}   # xy 를 누가 맞췄나(분석의 xy 신뢰도)
        if spec["type"] == "charuco":   # 옛 키(호환)
            pt.update(corner_id=int(ti), corner_xy_board_m=pt["target_xy_board_m"], corner_xyz_base_display=pt["target_xyz_base_display"])
        f = self.fits.get(bid)   # 기록 전 맞춤(이 점은 아직 안 들어감)으로 본 잔차 — 3 mm 넘으면 표시만
        res = None if f is None or not f["ok"] else float(np.linalg.norm(T[:3, 3] - to_base(spec["targets"], f["T"])[ti]) * 1000)
        pt.update(fit_res_mm=None if res is None else round(res, 2), fit_kind=None if res is None else f["kind"])
        job = None
        if self.photo is not None:
            pt["photo_check"], job = self._photo_job(now, len(self.doc["points"]), bid, name, ti, T[:3, 3], xyz)
        new = copy.deepcopy(self.doc)
        new["points"].append(pt)
        new["boards"][bid]["pose_display"] = pose
        if bid == "charuco" and "board" in new:
            new["board"]["pose_display"] = pose
        self._commit(new)
        self.last_rec = (bid, name, T[:3, 3].copy(), now)
        if job is not None:
            self.photo.put(job)
        d = (T[:3, 3] - xyz) * 1000
        rs = "" if res is None else f" · 맞춤 잔차 {res:.1f} mm" + (f" ★> {A.RES_MAX_MM:g} mm" if res > A.RES_MAX_MM else "")
        ph = "" if self.photo is None else (" · 사진 처리 중" if job else f" · ★사진 없음({pt['photo_check']['why']})")
        return f"기록 #{len(new['points']) - 1} {bid} {name} — 표시 보드와 차 {np.round(d, 1).tolist()} mm{rs}{ph}"

    def _req_delete(self, now, contact, stale, i: int):
        if self.doc is None or not 0 <= i < len(self.doc["points"]):
            return "지울 행 없음"
        new = copy.deepcopy(self.doc)
        new["points"][i]["deleted"] = not new["points"][i]["deleted"]
        self._commit(new)
        return f"행 {i} {'지움' if new['points'][i]['deleted'] else '되살림'}"

    def _commit(self, new: dict, refit: bool = True) -> None:
        """복사본을 먼저 저장하고 성공한 뒤에만 바꾼다 — 저장이 실패하면 메모리 · 파일 모두 그대로(M3).
        refit=False = 점 · 맞춤이 그대로인 덧붙임(사진 결과 · 건너뜀 · 요약) — fit_ver 를 올리지 않는다. --out 없으면(demo) 메모리만."""
        fits, zf = self.fits, self.z_floor
        if refit:
            fits = {b: f for b in new["boards"] if (f := board_fit(b, new["points"], self.est_yaw.get(b))) is not None}
            face = [float(to_base(BOARDS[b]["targets"], f["T"])[:, 2].min()) for b, f in fits.items() if f["ok"]]
            zf = max(self.z_floor, min(face) - FLOOR_BELOW_M) if face else self.z_floor   # M1: 맞춤 면 최저 − 8 mm 로 올리기만
            new["tip_z_floor_m"] = round(zf, 5)
        if self.out is not None:
            save_json_atomic(self.out, new)
        with self.lock:
            self.doc, self.fits = new, fits
        self.z_floor, self.jog.z_floor = zf, max(self.jog.z_floor, zf)
        self.records_ver += 1
        self.fit_ver += int(refit)

    def _publish_snap(self, now: float, contact: bool, stale: bool) -> None:
        r = self.r
        meas = r.get("meas")
        snap = {"meas": None if meas is None else meas.copy(), "hmeas": r.get("hmeas"), "cmd": self.jog.cmd.copy(),
                "hand": self.jog.hand.copy(), "jf": r.get("jf"), "tip": r.get("tip"), "jf0": self.det.jf0, "tip0": self.det.tip0,
                "contact": contact, "stale": stale, "sig": self.sig, "pub": self.pub, "mode": self.jog.mode, "msg": self.msg,
                "start_fail": list(self.start_fail), "e0": self.jog.e0.copy(), "hold_c": self.jog.c.copy(), "hold_state": self.hold_state, "control": "precise" if self.precise else "legacy", "control_hz": 1 / self.dt, "loop_dt": getattr(self, "actual_dt", self.dt),
                "goal": self.jog.cmd.copy() if self.jog.goal is None else self.jog.goal.copy(),
                "pd": {k: (r.get("pd") or {}).get(k) for k in ("phase", "target", "estop", "reasons")},
                "track": None if meas is None else float(np.abs(meas - self.jog.cmd - self.jog.e0).max()),
                "home_err": None if meas is None else float(np.abs(meas - self.home).max()),
                "records_ver": self.records_ver, "alive": self.alive, "fatal": self.fatal, "n_done": self.n_done,
                "auto": None if self.auto is None else {k: self.auto[k] for k in ("phase", "bid", "ti", "name")},
                "auto_free": self.auto_free, "auto_skip": sorted(self.auto_skip), "fit_ver": self.fit_ver,
                "summary": self.summary, "photo_note": self.photo_note}
        Wa = self._window(now, REC_WIN)
        W = [x for x in Wa if np.all(np.isfinite(x[3]))]
        snap["can_record"], snap["rec_why"] = can_record(contact, [x[0] for x in W], [x[3] for x in W], [x[9] for x in Wa])
        if self.jog.mode == "brake":
            snap["can_record"], snap["rec_why"] = False, "접촉 정지 경로 중"
        with self.lock:
            self.snap = snap

    def get_snap(self) -> dict:
        with self.lock:
            return dict(self.snap)

    def get_fits(self) -> dict:
        with self.lock:
            return dict(self.fits)

    def records(self) -> list:
        with self.lock:
            return [] if self.doc is None else [dict(p) for p in self.doc["points"]]


# ------------------------------------------------------------------ sim 창으로 상태 보내기(UDP, 창 · 제어와 떨어진 스레드)
def sim_packet(t: float, side: str, snap: dict, kin: Kin, view: dict) -> dict:
    """sim 직원과 같은 계약(한 줄 JSON). q_show = 발행 여부와 무관하게 신선한 실측 우선, 없으면 virtual 명시. q_cmd = 조그 목표(유지 보정 제외), 발행 중일 때만. tip 변환은 축 표시용. 순수."""
    pub = bool(snap["pub"])
    measured = snap["meas"] is not None and snap["hmeas"] is not None
    arm = snap["meas"] if measured else snap["cmd"]
    hand = snap["hmeas"] if measured else snap["hand"]
    tgt = view.get("target")
    goal = snap.get("goal", snap["cmd"]) if snap.get("control") == "precise" else snap["cmd"]
    return {"t": float(t), "state_source": "measured" if measured else "virtual", "names": list(P.arm_names(side) + P.hand_names(side)),
            "q_show": np.r_[arm, hand].astype(float).tolist(),
            "q_cmd": np.r_[snap["cmd"], snap["hand"]].astype(float).tolist() if pub else None,   # 조그 목표(유지 보정은 모터 발행에만)
            "boards": {b: {"T_base_board": np.asarray(T, float).tolist()} for b, T in view.get("boards", {}).items()},
            "target_xyz_base": None if tgt is None else np.asarray(tgt, float).tolist(),
            "T_tip_goal": kin.tip(goal, snap["hand"]).tolist(),
            "T_tip_meas": kin.tip(arm, hand).tolist(),
            "tip_xyz_base": kin.tip(arm, hand)[:3, 3].tolist(), "contact": bool(snap["contact"]), "mode": "pub" if pub else "dry"}


def sim_sender(ctl: Ctl, kin: Kin, port: int, host: str = SIM_HOST) -> None:
    """SIM_HZ 로 ctl 스냅숏을 보낸다. 어떤 실패도 이 스레드 안에서 삼키고(처음 한 번만 알림) 제어 스레드는 모른다."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)   # 받는 쪽이 없어도 · 버퍼가 차도 멈추지 않는다
    warned = False
    try:
        while not ctl.quit.wait(1.0 / SIM_HZ):
            try:
                s = ctl.get_snap()
                if s:
                    data = json.dumps(sim_packet(time.time(), kin.side, s, kin, ctl.view), allow_nan=False, separators=(",", ":"))
                    sock.sendto(data.encode(), (host, port))
            except Exception as error:   # noqa: BLE001 — sim 표시용, 제어와 무관
                if not warned:
                    say(f"sim 송신 실패({host}:{port}, 제어와 무관 · 계속 시도): {type(error).__name__}: {error}", err=True)
                    warned = True
    finally:
        sock.close()


# ------------------------------------------------------------------ 자체 검사(ROS · 화면 없음)
def selftest() -> int:
    import tempfile

    side = "right"
    kin = Kin(URDF_DEFAULT, side, np.asarray(POINT))
    lo, hi = arm_limits(side)
    lo_m, hi_m = lo + P.LIMIT_MARGIN_RAD, hi - P.LIMIT_MARGIN_RAD
    home = np.array(json.loads(CONTRACT.read_text())["sides"][side]["home_arm"], float)
    # (보드) 교차점 식 = cv2 CharucoBoard
    try:
        sys.path.insert(0, str(S2R / "scripts"))
        import charuco_calib as CC

        b = CC.build_board()
        if hasattr(b, "getChessboardCorners"):
            assert np.allclose(np.asarray(b.getChessboardCorners()).reshape(-1, 3), CORNERS, atol=1e-6), "CORNERS ≠ cv2"
            M = np.array([np.asarray(c).reshape(-1, 3).mean(0)[:2] for c in b.getObjPoints()])
            assert np.allclose(M, MARKERS, atol=1e-6), "MARKERS ≠ cv2"
            print("보드 교차점 · 마커 = cv2 CharucoBoard")
    except (ImportError, AttributeError):   # 시스템 cv2 4.5 에는 CharucoBoard 가 없다
        print("cv2 CharucoBoard 없음 — 보드 대조 건너뜀")
    # (보드) GridBoard 모서리 상수 = cv2 GridBoard((2,2),0.025,0.005,DICT_6X6_250,ids).getObjPoints()
    try:
        import cv2

        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        for b in GRID_FIRST:
            ids = BOARDS[b]["geometry"]["ids"]
            g = cv2.aruco.GridBoard((2, 2), GRID_MARKER, GRID_GAP, dic, np.array(ids))
            assert list(np.asarray(g.getIds()).ravel()) == ids, (b, g.getIds())
            assert np.allclose(np.concatenate([np.asarray(o).reshape(-1, 3) for o in g.getObjPoints()]), GRID_CORNERS, atol=1e-6), b
        print(f"GridBoard 모서리 16 점 = cv2 {cv2.__version__} GridBoard ({', '.join(GRID_FIRST)})")
    except (ImportError, AttributeError, TypeError):   # 시스템 cv2 4.5 는 GridBoard 생성자가 다르다
        print("cv2 GridBoard((2,2),…) 없음 — GridBoard 대조 건너뜀(sim2real/.venv 로 돌린다)")
    assert BOARDS["grid54b"]["names"][6] == "m55_c2" and np.allclose(GRID_CORNERS[6], [0.055, 0.025, 0.0])
    # (a) DLS 6D 증분(실제 URDF), 두 프레임 × 6 축 × ±
    worst = [0.0, 0.0]
    T0 = kin.tip(home)
    for frame in ("base", "tip"):
        for axis in AXES:
            for sign in (1, -1):
                step = 0.001 if AXES.index(axis) < 3 else np.radians(0.5)
                Tg = jog_goal(T0, axis, sign, step, frame)
                q, why = ik(kin, home, Tg, lo_m, hi_m)
                assert q is not None, f"{frame} {axis}{sign}: {why}"
                e = pose_err(Tg, kin.tip(q))
                worst = [max(worst[0], np.linalg.norm(e[:3]) * 1000), max(worst[1], np.degrees(np.linalg.norm(e[3:])))]
                moved = np.linalg.norm(Tg[:3, 3] - T0[:3, 3]) * 1000
                assert np.linalg.norm(e[:3]) * 1000 <= 0.2 and np.degrees(np.linalg.norm(e[3:])) <= 0.1, (frame, axis, e)
                assert (moved > 0.99) == (AXES.index(axis) < 3), (frame, axis, moved)   # 회전은 손끝점 고정
    print(f"(a) DLS 24 증분 최대 오차 {worst[0]:.4f} mm · {worst[1]:.4f}°")
    q, why = ik(kin, home, jog_goal(T0, "z", 1, 0.001, "base"), home - 1e-4, home + 1e-4)
    assert q is None and "한계" in why, why
    q, why = ik(kin, home, jog_goal(T0, "z", 1, 0.5, "base"), lo_m, hi_m)
    assert q is None, "50 cm 한 단계는 거부(한계 · 수렴 · 점프)"
    print(f"(a) 한계 밖 거부: {ik(kin, home, jog_goal(T0, 'z', 1, 0.001, 'base'), home - 1e-4, home + 1e-4)[1][:40]}…")
    # (b) 접촉 중 −z 거부 · +z 허용
    jg = Jogger(kin, lo, hi, home, np.asarray(POINT))
    assert jg.plan(jog_goal(T0, "z", -1, 0.002, "base"), contact=True, stale=False) is not None
    assert jg.plan(jog_goal(T0, "z", -1, 0.002, "base"), contact=False, stale=True) is not None
    assert jg.mode == "idle"
    assert jg.plan(jog_goal(T0, "z", 1, 0.002, "base"), contact=True, stale=False) is None and jg.mode == "move"
    assert jg.plan(jog_goal(T0, "z", 1, 0.002, "base"), contact=True, stale=False) is not None   # 이동 중 증분은 받지 않는다
    jg2 = Jogger(kin, lo, hi, home, np.asarray(POINT))
    assert jg2.plan(jog_goal(T0, "z", -1, 0.002, "base"), contact=False, stale=False) is None and jg2.down
    jg2.tick(0.0, DT, home + 0.001, contact=True, stale=False)
    assert jg2.mode == "idle" and np.allclose(jg2.cmd, home + 0.001), "내려가던 이동 중 접촉 → 명령 = 실측"
    print("(b) 접촉 · 힘 끊김 중 −z 거부, +z 허용, 이동 중 증분 거부, 내려가던 이동 중 접촉 = 명령 실측")
    # (b2) M2: 옆 이동 중 접촉 새로 켜짐 → 정지 · 접촉한 채 시작한 옆 이동에서 힘 상승 → 정지
    jg = Jogger(kin, lo, hi, home, np.asarray(POINT))
    assert jg.plan(jog_goal(T0, "x", 1, 0.002, "base"), contact=False, stale=False) is None and not jg.down
    assert jg.tick(0.0, DT, None, False, False) is None and jg.mode == "move"
    assert "이동 중 접촉" in (jg.tick(DT, DT, home, True, False, (25.0, 0.3)) or "") and jg.mode == "idle"
    jg = Jogger(kin, lo, hi, home, np.asarray(POINT))
    assert jg.plan(jog_goal(T0, "y", 1, 0.002, "base"), contact=True, stale=False) is None   # 옆으로는 접촉 중에도 된다
    assert jg.tick(0.0, DT, None, True, False, (25.0, 0.3)) is None and jg.tick(DT, DT, None, True, False, (40.0, 0.4)) is None
    assert "힘 상승" in (jg.tick(2 * DT, DT, home, True, False, (25.0 + FORCE_G + 1, 0.3)) or "") and jg.mode == "idle"
    jg = Jogger(kin, lo, hi, home, np.asarray(POINT))
    jg.plan(jog_goal(T0, "y", 1, 0.002, "base"), contact=True, stale=False)
    jg.tick(0.0, DT, None, True, False, (25.0, 0.3))
    assert "힘 상승" in (jg.tick(DT, DT, home, True, False, (25.0, 0.3 + TIP_N + 0.01)) or "")
    print("(b2) 옆 이동 중 새 접촉 정지 · 접촉 중 이동의 힘 상승(검지 · 촉각) 정지")
    # (b3) H1: 손끝 z 하한 — 조그 · 하강 · 복귀(경로 FK 샘플) 거부, 하한 아래에서는 빼기만
    z_home = float(T0[2, 3])
    jg = Jogger(kin, lo, hi, home, np.asarray(POINT), z_floor=z_home - 0.004)
    assert jg.plan(jog_goal(T0, "z", -1, 0.002, "base"), False, False) is None   # 하한 위 2 mm — 된다
    jg.mode = "idle"
    why = jg.plan(jog_goal(T0, "z", -1, 0.005, "base"), False, False)
    assert why and "하한" in why and jg.mode == "idle", why
    jg.start_descent(0.0)
    t, note = 0.0, None
    while note is None and t < 20:
        t += DT
        note = jg.tick(t, DT, jg.cmd.copy(), False, False)
    assert note and "하한" in note and jg.mode == "idle", note
    assert jg.z_now() >= jg.z_floor - 1e-4, (jg.z_now(), jg.z_floor)
    q_up, _ = ik(kin, home, jog_goal(T0, "z", 1, 0.010, "base"), lo_m, hi_m)
    jg = Jogger(kin, lo, hi, q_up, np.asarray(POINT), z_floor=z_home + 0.005)   # 위 10 mm 에서 홈(하한 아래 5 mm)으로 복귀
    why = jg.go(home, False, False)
    assert why and "하한" in why, why
    assert path_min_z(kin, q_up, home, POINT, POINT) < z_home + 1e-4
    jg = Jogger(kin, lo, hi, home, np.asarray(POINT), z_floor=z_home + 0.005)   # 이미 하한 아래: 위로는 되고 더 아래는 거부
    assert jg.plan(jog_goal(T0, "z", 1, 0.002, "base"), False, False) is None
    jg.mode = "idle"
    assert "하한" in (jg.plan(jog_goal(T0, "z", -1, 0.002, "base"), False, False) or "")
    print("(b3) 하한: 조그 −5 mm 거부 · 하강이 하한에서 멈춤 · 하한 아래 지나는 복귀 거부 · 하한 아래에선 위로만")
    # (c) 기록 활성 조건
    ts = np.arange(0, 0.5, 1 / 250)
    still = np.tile(home, (len(ts), 1)) + np.random.default_rng(0).normal(0, 2e-4, (len(ts), 7))
    moving = still + np.outer(ts, np.r_[0.03, np.zeros(6)])
    on = np.ones(len(ts), bool)
    assert can_record(True, ts, still, on)[0] and not can_record(False, ts, still, on)[0] and not can_record(True, ts, moving, on)[0]
    assert not can_record(True, ts[:3], still[:3], on[:3])[0]
    part = on.copy()
    part[: int(0.15 * len(ts))] = False   # 창의 15 % 무접촉 → 85 %
    ok, why = can_record(True, ts, still, part)
    assert not ok and "접촉 비율" in why, why
    part[: int(0.15 * len(ts))] = True
    part[: int(0.05 * len(ts))] = False   # 95 %
    assert can_record(True, ts, still, part)[0]
    print("(c) 기록 조건: 접촉 + 정지 + 접촉 비율 ≥ 90 % 참, 무접촉 · 움직임 · 창 부족 · 비율 85 % 거짓")
    # (d) JSON 원자 저장 · 스키마 키 · 덮어쓰기 거부 / (f) 가짜 io 로 하강 → 접촉 → 기록
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rec" / "right_1.json"
        check_out(out)
        tip0 = T0[:3, 3]
        zc = float(tip0[2]) - 0.010   # 가짜 면 = 홈 손끝 10 mm 아래(하강 40 mm 안)
        Tb = board_T(tip0[0] - 0.105, tip0[1] + 0.075, zc, 0.0)   # 교차점 id 9(가운데 근처)가 손끝 바로 아래
        Tg = board_T(tip0[0] - 0.04, tip0[1] - 0.02, zc, 30.0)   # 목표 5(m51_c1)가 손끝에서 ~11 mm(기록 xy 관문 안)
        poses = {"charuco": {"source": "selftest", "T_base_board": Tb.tolist()}, "grid50": {"source": "selftest", "T_base_board": Tg.tolist()}}
        z_floor = zc - FLOOR_BELOW_M
        doc = new_doc(kin, URDF_DEFAULT, np.asarray(POINT), FK_DZ_MM[side], poses, z_floor)
        io = FakeIO(side, kin, home, zc)
        clk = [100.0]
        ctl = Ctl(io, kin, lo, hi, home, np.asarray(POINT), execute=False, demo=True, out=out, doc=doc, clock=lambda: clk[0],
                  z_floor=z_floor, hold=False)   # 옛 e0 방식 경로 그대로(유지 보정은 selftest_hold)

        def run_for(sec: float) -> None:
            for _ in range(int(sec / DT)):
                clk[0] += DT
                ctl.step(clk[0])

        orig_sleep = time.sleep
        time.sleep = lambda s: clk.__setitem__(0, clk[0] + max(s, 0.01))
        try:
            ctl.start_check(1.0)
        finally:
            time.sleep = orig_sleep
        assert ctl.start_fail == ["--execute 없이 시작(dry)"], ctl.start_fail
        run_for(0.6)
        ctl.call("record", "charuco", 9, Tb, poses["charuco"])
        run_for(0.1)
        assert "기록 거부" in ctl.msg and not ctl.doc["points"], ctl.msg
        ctl.call("descend")
        z_start = kin.tip(ctl.jog.cmd)[2, 3]
        run_for(1.0)
        assert ctl.jog.mode == "descend", (ctl.jog.mode, ctl.msg)
        for _ in range(60 * 30):
            run_for(DT)
            if ctl.jog.mode == "idle":
                break
        assert "접촉" in ctl.msg, ctl.msg
        z_stop = kin.tip(ctl.jog.cmd)[2, 3]
        assert zc - 0.002 < z_stop < zc + 1e-3 and z_stop < z_start, (z_start, z_stop, zc)
        assert np.allclose(ctl.jog.e0, FakeIO.E0, atol=1e-6), ctl.jog.e0   # 하강 전 정지에서 처짐을 쟀다
        ctl.call("jog", "z", -1, 0.001, "base")
        run_for(0.6)
        assert "거부" in ctl.msg, ctl.msg
        ctl.call("home")   # M2: 접촉 중 시작 자세 복귀 거부
        run_for(0.05)
        assert "접촉 중" in ctl.msg and ctl.jog.mode == "idle", ctl.msg
        ctl.call("record", "charuco", 0, Tb, poses["charuco"])   # 손끝은 bc08 · bc09 근처, bc00 은 ~87 mm(맞춤 전 모서리 한계 80 도 넘음)
        run_for(0.05)
        assert "목표 bc00 에서" in ctl.msg and "가장 가까운 점 bc08" in ctl.msg and not ctl.doc["points"], ctl.msg
        ctl.call("record", "charuco", 9, Tb, poses["charuco"])
        run_for(0.05)
        ctl.call("record", "charuco", 9, Tb, poses["charuco"])   # 겹친 창으로 같은 점 두 번(11.05 right_3 bc15)
        run_for(0.05)
        assert "이미 기록함" in ctl.msg and len(ctl.doc["points"]) == 1, ctl.msg
        ctl.call("record", "grid50", 5, Tg, poses["grid50"])
        run_for(0.05)
        assert len(ctl.doc["points"]) == 2, ctl.msg
        # M3: 저장 실패 → 메모리 · 파일 그대로
        g = globals()
        save_ok, before = g["save_json_atomic"], ctl.doc

        def boom(*_a) -> None:
            raise OSError("디스크 가득(시험)")

        g["save_json_atomic"] = boom
        try:
            ctl.call("record", "charuco", 8, Tb, poses["charuco"])   # 손끝에 가장 가까운 bc08(기록 xy 관문 안)
            run_for(0.05)
            assert "실패" in ctl.msg and ctl.doc is before and len(ctl.doc["points"]) == 2, ctl.msg
            ctl.call("delete", 0)
            run_for(0.05)
            assert "실패" in ctl.msg and ctl.doc["points"][0]["deleted"] is False, ctl.msg
        finally:
            g["save_json_atomic"] = save_ok
        d = json.loads(out.read_text())
        assert len(d["points"]) == 2 and d["points"][0]["deleted"] is False
        assert not list(out.parent.glob("*.tmp")), "임시 파일이 남았다"
        assert {"schema", "side", "urdf", "urdf_md5", "tip_link", "hand_q", "board", "boards", "fk_dz_mm", "tip_z_floor_m", "points"} <= set(d)
        assert d["schema"] == "board_jog_v1" and d["board"]["squares"] == [7, 5] and d["board"]["square_m"] == 0.03
        assert {"squares", "square_m", "pose_display", "corner_convention"} <= set(d["board"])
        assert set(d["boards"]) == {"charuco", "grid50"} and d["boards"]["grid50"]["type"] == "aruco_grid"
        assert d["boards"]["grid50"]["geometry"]["ids"] == [50, 51, 52, 53] and "getObjPoints" in d["boards"]["grid50"]["geometry"]["corner_convention"]
        assert {"type", "geometry", "pose_display"} <= set(d["boards"]["charuco"])
        p, pg = d["points"]
        assert {"board_id", "target_id", "target_xy_board_m", "corner_id", "corner_xy_board_m", "q_arm", "q_hand", "T_tip_fk",
                "tip_dir_base", "force", "contact_frac", "t_wall_ns", "deleted"} <= set(p)
        assert len(p["q_arm"]) == 7 and len(p["q_hand"]) == 6 and np.asarray(p["T_tip_fk"]).shape == (4, 4) and len(p["tip_dir_base"]) == 3
        assert p["board_id"] == "charuco" and p["target_id"] == "bc09" and p["corner_id"] == 9
        assert p["corner_xy_board_m"] == p["target_xy_board_m"] == [0.12, 0.06] and p["deleted"] is False and p["contact_frac"] >= 0.9
        assert pg["board_id"] == "grid50" and pg["target_id"] == "m51_c1" and pg["target_xy_board_m"] == [0.055, 0.0], pg["target_id"]
        assert "corner_id" not in pg and len(pg["target_xyz_base_display"]) == 3 and pg["contact_frac"] >= 0.9
        ctl.call("delete", 0)
        run_for(0.05)
        assert json.loads(out.read_text())["points"][0]["deleted"] is True
        ctl.call("lift")
        run_for(0.05)
        assert ctl.jog.mode == "move", ctl.msg
        # LOW6: 원본 조각 — 10 분 조각(별 스레드) + 종료 조각, 메모리엔 최근 RAW_KEEP_S 만
        n_all = len(ctl.rows)
        ctl._flush_rows(clk[0])
        assert len(ctl.rows) <= RAW_KEEP_S / DT + 2 < n_all, (len(ctl.rows), n_all)
        n_kept = len(ctl.rows)
        ctl._finish("selftest")
        parts = sorted(out.parent.glob("right_1_raw_part*.npz"))
        assert [x.name for x in parts] == ["right_1_raw_part00.npz", "right_1_raw_part01.npz"], parts
        Z = [np.load(x) for x in parts]
        assert Z[0]["q_meas"].shape[1] == 7 and len(Z[0]["t"]) + len(Z[1]["t"]) == n_all and len(Z[1]["t"]) == n_kept
        assert Z[0]["t"][-1] < Z[1]["t"][0]
        try:
            check_out(out)
            raise AssertionError("있는 --out 을 받았다")
        except FileExistsError:
            pass
        (out.parent / "right_2_raw_part00.npz").touch()
        try:
            check_out(out.parent / "right_2.json")
            raise AssertionError("원본 조각이 있는 --out 을 받았다")
        except FileExistsError:
            pass
    print(f"(d) JSON 원자 저장 · 새 키(boards · board_id · target_id · contact_frac) · charuco 옛 키 · 저장 실패 시 불변 · "
          f"접촉 중 복귀 거부 · 덮어쓰기 거부 · 지움 표시 · _raw_partNN.npz / 가짜 하강 접촉 z {z_stop * 1000:.1f} mm(면 {zc * 1000:.0f})")
    # (e) 보간 속도 ≤ 0.15 rad/s
    rng = np.random.default_rng(1)
    jg = Jogger(kin, lo, hi, home, np.asarray(POINT))
    vmax = 0.0
    for _ in range(20):
        jg.mode, jg.goal, jg.down = "move", np.clip(home + rng.uniform(-0.3, 0.3, 7), jg.lo, jg.hi), False
        t = 0.0
        while jg.mode == "move":
            prev = jg.cmd.copy()
            dt = DT * rng.uniform(0.5, 1.5)
            t += dt
            jg.tick(t, dt, None, False, False)
            vmax = max(vmax, float(np.abs(jg.cmd - prev).max() / dt))
    assert vmax <= JOG_VEL + 1e-9, vmax
    print(f"(e) 보간 최대 관절 속도 {vmax:.4f} rad/s ≤ {JOG_VEL}")
    # (g) sim 송신 패킷: canonical 이름 13 개 · 길이 · JSON(NaN 없음) · dry/pub 의 q_show · UDP 한 번(받은 그대로)
    import socket

    s = ctl.get_snap()
    view = {"boards": {"charuco": Tb, "grid50": Tg}, "target": to_base(CORNERS, Tb)[9]}
    names = list(P.arm_names(side) + P.hand_names(side))
    assert names == [f"r_aj_{i}" for i in range(1, 8)] + [f"r_hj_{a}" for a in P.HAND_AXES], names
    assert set(names) <= set(kin.tree.joints), sorted(set(names) - set(kin.tree.joints))
    pk = sim_packet(1.0, side, s, kin, view)
    assert set(pk) == {"t", "names", "q_show", "q_cmd", "boards", "target_xyz_base", "tip_xyz_base", "contact", "mode", "T_tip_goal", "T_tip_meas", "state_source"}, set(pk)
    assert pk["names"] == names and len(pk["q_show"]) == 13 and pk["q_cmd"] is None and pk["mode"] == "dry"
    assert np.allclose(pk["q_show"], np.r_[s["meas"], s["hmeas"]]) and pk["state_source"] == "measured"   # dry = 가상 명령
    assert np.allclose(pk["tip_xyz_base"], kin.tip(s["meas"], s["hmeas"])[:3, 3]) and len(pk["target_xyz_base"]) == 3
    assert np.asarray(pk["boards"]["grid50"]["T_base_board"]).shape == (4, 4) and isinstance(pk["contact"], bool)
    pp = sim_packet(1.0, side, dict(s, pub=True), kin, view)
    assert pp["mode"] == "pub" and np.allclose(pp["q_show"][:7], s["meas"]) and len(pp["q_cmd"]) == 13   # 발행 중 = 실측
    corrected = sim_packet(1.0, side, dict(s, pub=True, hold_c=np.full(7, 0.04)), kin, view)
    assert np.allclose(corrected["q_cmd"][:7], s["cmd"]), "ghost must show jog goal, not compensated motor target"
    assert np.allclose(corrected["T_tip_goal"], kin.tip(s["cmd"], s["hand"]))
    assert np.allclose(corrected["T_tip_meas"], kin.tip(s["meas"], s["hmeas"]))
    assert json.loads(json.dumps(pk, allow_nan=False)) == pk and json.loads(json.dumps(pp, allow_nan=False)) == pp
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    ctl.view = view
    ctl.quit.clear()
    th = threading.Thread(target=sim_sender, args=(ctl, kin, rx.getsockname()[1]), daemon=True)
    th.start()
    try:
        got = json.loads(rx.recv(65535))
    finally:
        ctl.quit.set()
        th.join(2.0)
        rx.close()
    assert got["names"] == names and got["mode"] == "dry" and np.allclose(got["q_show"], pk["q_show"]) and not th.is_alive()
    print(f"(g) sim 패킷: 이름 13 개(r_aj_1..7 + r_hj_{{{','.join(P.HAND_AXES)}}}) · dry = 가상 명령 · pub = 실측 · UDP {len(json.dumps(got))} B 받음")
    selftest_auto(kin, lo, hi, home)
    # 유지 켬: 관절 한계가 실측이 갈 관절(명령)에 걸린다. 이 시험 모서리는 옛 판에서 실측 j6 가 여유 한계를 0.01(E0) 넘었다 — 시험 한계만 0.02 넓힌다
    selftest_auto(kin, lo - 0.02, hi + 0.02, home, hold=True)
    selftest_rec_xy()
    selftest_hold(kin, lo, hi, home)
    PH.selftest()
    selftest_autoall(kin, lo - 0.02, hi + 0.02, home)
    print("selftest ok")


def selftest_hold(kin: Kin, lo: np.ndarray, hi: np.ndarray, home: np.ndarray) -> None:
    """(n) 유지 보정(FakeIO 자세 의존 처짐 = E0 + A·sin(2(q2 − 홈 q2)) 을 j2 · j6 에): (a) 켬 → 실측 → 명령(관절 · 손끝 6D), 끔 → 처짐 그대로
    (b) 접촉 · 힘 끊김 · 하강에서 c 불변 (c) |c| · 변화율 상한 (d) 추종 오차 해제 = 명령 실측 − e0, 발행 = 그 + 얼린 c(끔이면 옛 값과 같다)."""
    w = np.array([0.0, 1, 0, 0, 0, 1, 0])

    def sag_of(amp: float):
        return lambda q: FakeIO.E0 + amp * np.sin(2 * (q[1] - home[1])) * w

    def make(hold: bool):
        io = FakeIO(kin.side, kin, home, -1.0)   # 면은 멀리(무접촉, 주입은 io.dz)
        io.sag, clk = sag_of(0.08), [100.0]
        ctl = Ctl(io, kin, lo, hi, home, np.asarray(POINT), execute=False, demo=True, out=None, doc=None, clock=lambda: clk[0], hold=hold)
        orig = time.sleep
        time.sleep = lambda s: clk.__setitem__(0, clk[0] + max(s, 0.01))
        try:
            ctl.start_check(1.0)
        finally:
            time.sleep = orig
        e0 = io.sag(home)   # = start_e0(실측 − pd 세트포인트 홈)
        if hold:   # _req_publish 유지 켬과 같은 흡수
            ctl.jog.cmd, ctl.jog.c = home + e0, -e0
        else:
            ctl.jog.e0 = e0
        return ctl, io, clk

    dc_max = [0.0]

    def run(ctl, clk, sec: float, until=None) -> None:
        for _ in range(int(round(sec / DT))):
            c0 = ctl.jog.c.copy()
            clk[0] += DT
            ctl.step(clk[0])
            dc_max[0] = max(dc_max[0], float(np.abs(ctl.jog.c - c0).max()) / DT)
            assert np.abs(ctl.jog.c).max() <= HOLD_MAX_RAD + 1e-12, ctl.jog.c
            if until is not None and until():
                return

    goal = home + np.r_[0.0, 0.4, 0, 0, 0, 0, 0]   # j2 +0.4 rad → 처짐 차 0.08·sin 0.8 = 0.057 rad(j2 · j6)
    res = {}
    for hold in (False, True):
        ctl, io, clk = make(hold)
        assert np.allclose(ctl.jog.pub(), home), "첫 발행 = pd 세트포인트(튐 없음)"
        assert ctl.jog.go(goal, False, False, clk[0]) is None
        run(ctl, clk, 3.0, lambda: ctl.jog.mode == "idle")   # 이동 2.7 s
        t_end, hit = clk[0], {}
        for _ in range(int(9.0 / DT)):   # 멈춘 뒤 9 s: 관절 0.01 rad · 손끝 1 mm 안으로 든 시각
            run(ctl, clk, DT)
            meas, qi = ctl.r["meas"], ctl.jog.cmd + ctl.jog.e0
            e6 = pose_err(kin.tip(qi, ctl.jog.hand), kin.tip(meas, ctl.jog.hand))
            v = (float(np.abs(meas - qi).max()), np.linalg.norm(e6[:3]) * 1000, np.degrees(np.linalg.norm(e6[3:])))
            res.setdefault((hold, "t0"), v)
            for key, ok in (("q", v[0] < 0.01), ("mm", v[1] < 1.0)):
                if ok:
                    hit.setdefault(key, clk[0] - t_end)
        res[hold] = (*v, ctl.hold_state, hit)
    assert res[True][0] < 0.01 and res[True][1] < 0.5 and res[True][3] == "동작", res[True]
    assert res[False][0] > 0.04 and not res[False][4], res[False]   # 끔: 시작 e0 뒤 자세 차 처짐 그대로
    t0 = res[(True, "t0")]
    print(f"(n·a) j2 +0.4 rad 이동 끝: 처짐 차 |실측 − 명령| {t0[0]:.4f} rad · 손끝 6D {t0[1]:.1f} mm {t0[2]:.2f}° → 유지 켬 "
          f"0.01 rad 안 {res[True][4]['q']:.1f} s · 1 mm 안 {res[True][4]['mm']:.1f} s · 9 s 뒤 {res[True][0]:.4f} rad · {res[True][1]:.2f} mm "
          f"{res[True][2]:.3f}° / 끔 9 s 뒤 {res[False][0]:.4f} rad · {res[False][1]:.1f} mm {res[False][2]:.2f}°")
    # (c) 처짐 차가 상한보다 크면 c 는 HOLD_MAX_RAD 에서 멈춘다
    io.sag = sag_of(0.25)   # 0.25·0.717 = 0.18 rad 차
    run(ctl, clk, 10.0)
    assert abs(abs(ctl.jog.c[1]) - HOLD_MAX_RAD) < 1e-9, ctl.jog.c
    io.sag = sag_of(0.08)
    run(ctl, clk, 8.0)
    # (b) 접촉 · 힘 끊김 · 하강 전 정지 · 하강에서 c 불변
    io.sag = sag_of(0.12)   # 맞출 차를 새로 만든다
    io.dz = 2.0             # 면을 손끝 위로 — 접촉
    run(ctl, clk, 1.0, lambda: ctl.sig in ("index_force", "tip_touch"))
    c0, n0 = ctl.jog.c.copy(), 0
    for _ in range(int(1.5 / DT)):
        run(ctl, clk, DT)
        assert np.array_equal(ctl.jog.c, c0) and ctl.hold_state == "접촉", ctl.hold_state
        n0 += 1
    io.dz = 0.0
    run(ctl, clk, REC_WIN - 2 * DT)   # 접촉이 꺼진 뒤 기록 창 길이까지 얼림(판정 지연 포함)
    assert np.array_equal(ctl.jog.c, c0), "접촉 뒤 기록 창 안에서 바뀜"
    io.stale = True
    run(ctl, clk, 1.0, lambda: ctl.hold_state == "힘 끊김")
    c0 = ctl.jog.c.copy()
    run(ctl, clk, 0.5)
    assert np.array_equal(ctl.jog.c, c0) and ctl.hold_state == "힘 끊김", ctl.hold_state
    io.stale = False
    run(ctl, clk, 1.0)
    assert not np.array_equal(ctl.jog.c, c0) and ctl.hold_state == "동작", ctl.hold_state
    ctl.call("descend")
    run(ctl, clk, DT)
    c1, modes = ctl.jog.c.copy(), set()
    for _ in range(int(16 / DT)):
        run(ctl, clk, DT)
        if ctl.jog.mode == "idle":
            break
        modes.add(ctl.jog.mode)
        assert np.array_equal(ctl.jog.c, c1), ctl.hold_state
    assert modes == {"dwell", "descend"} and "무접촉" in ctl.msg, (modes, ctl.msg)
    assert dc_max[0] <= HOLD_RATE + 1e-9, dc_max[0]
    print(f"(n·b·c) 접촉 {n0} 틱 · 접촉 뒤 {REC_WIN:g} s · 힘 끊김 · 하강 전 정지 · 하강 {DESC_MAX * 1000:.0f} mm 동안 c 불변 / "
          f"처짐 차 0.18 rad → |c| = {HOLD_MAX_RAD} 에서 멈춤 · 최대 변화율 {dc_max[0]:.4f} ≤ {HOLD_RATE} rad/s")
    # (d) 발행 중 추종 오차 해제: 명령 = 실측 − e0, 발행 = 그 + c(그 틱 적분 없이 얼린 값), 끔이면 옛 값(실측 − e0)
    for hold in (True, False):
        ctl, io, clk = make(hold)
        sent: list = []
        io.send = lambda k, qa, qh: sent.append(np.asarray(qa, float).copy())
        io.event = lambda *a, **kw: None
        ctl.pub, ctl.sp_est = True, ctl.jog.pub()
        run(ctl, clk, 3.0)
        assert ctl.pub and np.allclose(sent[-1], ctl.jog.pub())   # 발행 = 명령 + c
        c0, n = ctl.jog.c.copy(), len(sent)
        base = io.sag
        io.sag = lambda q: base(q) + np.r_[0.3, np.zeros(6)]   # j1 0.3 rad 어긋남 > TRACK_TOL_MOVE
        clk[0] += DT
        ctl.step(clk[0])
        meas = ctl.r["meas"]
        assert not ctl.pub and "추종 오차 j1" in ctl.msg, ctl.msg
        assert np.array_equal(ctl.jog.c, c0), "해제 틱에 c 가 바뀜"
        want = np.clip(meas - ctl.jog.e0, ctl.jog.lo, ctl.jog.hi)
        assert np.allclose(ctl.jog.cmd, want) and len(sent) > n
        assert all(np.allclose(x, np.clip(want + c0, ctl.jog.lo, ctl.jog.hi)) for x in sent[n:]), "해제 송신 ≠ 실측 − e0 + c"
        if not hold:
            assert not c0.any() and all(np.allclose(x, want) for x in sent[n:])   # 끔 = 옛 해제 경로 그대로
    print("(n·d) 추종 오차 해제: 명령 = 실측 − e0, 송신 = 그 + 얼린 c(유지 끔이면 옛 값 그대로) · 해제 틱에 c 불변")


def selftest_rec_xy() -> None:
    """기록 xy 관문(11.05 bc15 를 고른 채 bc18 자리를 기록한 사고): 먼 점 거부 + 가장 가까운 이름, 가까운 점 통과, 모서리 넓은 한계."""
    allt = {"bc15": (0.0, 0.0), "bc18": (0.1, 0.0), "grid50 m50_c0": (0.0, 0.2)}
    ok, why = rec_xy_check((0.1, 0.002), "bc15", (0.0, 0.0), allt)
    assert not ok and "bc15 에서 100 mm" in why and "가장 가까운 점 bc18 (2 mm)" in why, why
    assert rec_xy_check((0.01, 0.0), "bc15", (0.0, 0.0), allt) == (True, "")
    assert rec_xy_check((0.06, 0.0), "bc18", (0.0, 0.0), allt, REC_XY_MAX_CORNER_MM)[0]
    assert not rec_xy_check((0.09, 0.0), "bc18", (0.0, 0.0), allt, REC_XY_MAX_CORNER_MM)[0]
    assert not rec_xy_check((0.06, 0.0), "bc15", (0.0, 0.0), allt)[0]   # 모서리 아니면 40 mm
    return 0


def selftest_auto(kin: Kin, lo: np.ndarray, hi: np.ndarray, home: np.ndarray, hold: bool = False) -> None:
    """(h)~(m) 4 모서리 맞춤 관문 · 자동 접근 · 자동 해제 경로(FakeIO, 가짜 시계). hold = 유지 보정 켬(e0 = 0, 처짐은 c 첫값)."""
    import tempfile

    for b, sp in BOARDS.items():   # 바깥 4 모서리 = [원점, +x, +y, +x+y] 이고 목표 범위의 꼭짓점
        Q = sp["targets"][list(sp["outer"])]
        lo_, hi_ = sp["targets"].min(0), sp["targets"].max(0)
        assert np.allclose(Q[:, :2], [lo_[:2], [hi_[0], lo_[1]], [lo_[0], hi_[1]], hi_[:2]]), b
    assert [BOARDS["charuco"]["names"][i] for i in BOARDS["charuco"]["outer"]] == ["bc00", "bc05", "bc18", "bc23"]
    # (h) Kabsch: 아는 보드 자세(조금 기울임) + 4 모서리 점마다 1 mm(무작위 방향) 잡음
    Tt = board_T(0.31, -0.12, 0.21, 23.0)
    Tt[:3, :3] = Rot.from_euler("xy", [1.0, -0.7], degrees=True).as_matrix() @ Tt[:3, :3]
    Pb = CORNERS[list(BOARDS["charuco"]["outer"])]
    Pw = to_base(Pb, Tt)
    f = A.fit_board(Pb, Pw, corners=True)
    assert f["ok"] and np.allclose(f["T"], Tt, atol=1e-9) and f["res_mm"].max() < 1e-6
    rng = np.random.default_rng(5)
    E = []
    for _ in range(300):
        n = rng.normal(size=(4, 3))
        f = A.fit_board(Pb, Pw + n * (0.001 / np.linalg.norm(n, axis=1)[:, None]), corners=True)
        rv = rotvec(f["T"][:3, :3] @ Tt[:3, :3].T)
        E.append((np.degrees(np.linalg.norm(rv)), np.degrees(abs(rv @ Tt[:3, 2])), np.linalg.norm(f["T"][:3, 3] - Tt[:3, 3]) * 1000, f["ok"]))
    E = np.array(E)
    assert E[0, 0] <= 0.5 and E[0, 2] <= 2.0, E[0]   # 첫 표본(시드 5)
    assert E[:, 3].all() and E[:, 2].max() <= 2.0 and np.percentile(E[:, 1], 95) <= 0.5, E.max(0)
    print(f"(h) Kabsch 4 모서리 + 1 mm 잡음 300 회: 회전 중앙 {np.median(E[:, 0]):.2f}° · 95 % {np.percentile(E[:, 0], 95):.2f}° "
          f"(면 둘레 yaw 95 % {np.percentile(E[:, 1], 95):.2f}°, 기울기가 대부분) · 원점 이동 최대 {E[:, 2].max():.2f} mm · "
          f"0.5° 넘는 비율 {np.mean(E[:, 0] > 0.5):.0%}")
    # (i) 모서리를 잘못 댐: bc05 대신 bc04 → 길이 차로 거부 · 거울 순서(x 뒤집음) → 기울기로 거부 · 한 모서리 z +15 mm → 잔차로 거부 ·
    #     180° 뒤집힌 순서 → yaw 로 거부 · 8 점 → 전체 맞춤 · auto 점은 맞춤에서 빠짐 · 전체 맞춤도 잔차 관문
    Pbad = Pw.copy()
    Pbad[1] = to_base(CORNERS, Tt)[4]
    f = A.fit_board(Pb, Pbad, corners=True)
    assert not f["ok"] and "길이 차" in f["why"] and max(f["diag_mm"].max(), f["edge_mm"].max()) > A.DIAG_REJECT_MM, f["why"]
    f2 = A.fit_board(Pb, Pw[[1, 0, 3, 2]], corners=True)
    assert not f2["ok"] and "기울" in f2["why"], f2["why"]
    Pz = Pw.copy()
    Pz[2, 2] += 0.015
    fz = A.fit_board(Pb, Pz, corners=True)
    assert max(fz["diag_mm"].max(), fz["edge_mm"].max()) < A.DIAG_WARN_MM, fz   # 길이 검사로는 안 보이는 z 이상점
    assert not fz["ok"] and "잔차" in fz["why"] and fz["res_mm"].max() > A.RES_MAX_MM, fz["why"]
    yaw_t = A.yaw_deg(Tt)
    f3 = A.fit_board(Pb, Pw[[3, 2, 1, 0]], corners=True)
    assert f3["ok"], f3["why"]   # 강체 맞춤은 180° 순서를 그대로 받는다 — 그래서 yaw 검사가 있다
    f3 = A.fit_board(Pb, Pw[[3, 2, 1, 0]], corners=True, yaw_ref=yaw_t + 20.0)
    assert not f3["ok"] and "180°" in f3["why"], f3["why"]
    assert A.fit_board(Pb, Pw, corners=True, yaw_ref=yaw_t + 20.0)["ok"]
    hc = lambda i, by, dz=0.0: {"board_id": "charuco", "target_id": f"bc{i:02d}", "deleted": False, "aligned_by": by,   # noqa: E731
                                "T_tip_fk": np.r_[np.c_[np.eye(3), to_base(CORNERS, Tt)[i] + [0, 0, dz]], [[0, 0, 0, 1]]].tolist()}
    fake = [hc(i, "human_corner") for i in (0, 5, 18, 23)] + [hc(i, "human") for i in (7, 9, 14, 16)]
    assert board_fit("charuco", fake[:3]) is None and board_fit("charuco", fake[:4])["kind"] == "4점"
    fa = board_fit("charuco", fake, yaw_t)
    assert fa["kind"] == "전체 8점" and np.allclose(fa["T"], Tt, atol=1e-6) and fa["edge_mm"] is not None
    fu = board_fit("charuco", fake[:4] + [hc(i, "auto", 0.02) for i in (7, 9, 14, 16)], yaw_t)   # 20 mm 어긋난 auto 점 넷
    assert fu["ok"] and fu["kind"] == "4점" and np.allclose(fu["T"], Tt, atol=1e-6), fu["kind"]
    fo = board_fit("charuco", fake[:7] + [hc(16, "human", 0.015)], yaw_t)
    assert not fo["ok"] and fo["kind"] == "전체 8점" and "bc16" in fo["why"] and "잔차" in fo["why"], fo["why"]
    assert not board_fit("charuco", fake[:4], yaw_t + 180.0)["ok"]
    print(f"(i) bc05 자리에 bc04 → 거부({f['why'][:22]}…) · 거울 순서 → 거부({f2['why'][:14]}…) · 모서리 z +15 mm → 거부"
          f"(잔차 {fz['res_mm'].max():.1f} mm, 기울기 {fz['tilt_deg']:.1f}°) · 180° 순서 → 거부 · 8 점 → 전체 맞춤 · auto 점 4 개(20 mm) 빠짐 · "
          "전체 8점 중 bc16 z +15 mm → 거부")
    # (j)~(l) FakeIO: 사람이 4 모서리(조그 조각 → 하강 → 기록) → 자동
    side, T0 = kin.side, kin.tip(home)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "auto.json"
        tip0 = T0[:3, 3]
        zc = float(tip0[2]) - 0.010
        Tb = board_T(tip0[0] - 0.065, tip0[1] + 0.075, zc, 0.0)   # 보드 가운데가 홈 손끝 40 mm 앞(홈 −75 mm 쪽 bc00 은 j1 한계)
        pose = {"source": "selftest", "editable": True, "T_base_board": Tb.tolist()}
        doc = new_doc(kin, URDF_DEFAULT, np.asarray(POINT), FK_DZ_MM[side], {"charuco": pose}, zc - FLOOR_BELOW_M)
        clk = [100.0]
        ctl = Ctl(FakeIO(side, kin, home, zc), kin, lo, hi, home, np.asarray(POINT), execute=False, demo=True, out=out, doc=doc,
                  clock=lambda: clk[0], z_floor=zc - FLOOR_BELOW_M, hold=hold)
        ctl.view = {"boards": {"charuco": Tb}, "target": None}
        orig_sleep = time.sleep
        time.sleep = lambda s: clk.__setitem__(0, clk[0] + max(s, 0.01))
        try:
            ctl.start_check(1.0)
        finally:
            time.sleep = orig_sleep
        if hold:   # _req_publish 유지 켬처럼 처짐을 c 첫값으로
            ctl.jog.cmd, ctl.jog.c = ctl.jog.cmd + FakeIO.E0, -FakeIO.E0
        else:
            ctl.jog.e0 = FakeIO.E0.copy()   # 가짜 처짐을 첫 하강 전부터 안다(하강 전 정지가 같은 값을 다시 잰다)
        ph: list = []
        trace: list = []   # (자동 단계, 손끝, 보드 면 최고점) 틱마다

        def run_until(cond, sec: float) -> None:
            for _ in range(int(sec / DT)):
                clk[0] += DT
                ctl.step(clk[0])
                if ctl.auto is not None and (not ph or ph[-1] != ctl.auto["phase"]):
                    ph.append(ctl.auto["phase"])
                if ctl.auto is not None:
                    trace.append((ctl.auto["phase"], ctl._tip_meas()[:3, 3].copy(), ctl.auto["top"]))
                if cond():
                    return
            raise AssertionError(f"{sec} s 안에 안 됨: {ctl.msg} · {ph}")

        R0 = kin.tip(ctl.jog.cmd, ctl.jog.hand)[:3, :3]
        for k, i in enumerate(BOARDS["charuco"]["outer"]):   # 사람: 조그로 모서리 위 3 mm → 천천히 하강 → 기록 → (마지막 빼고) 위로
            p = to_base(CORNERS, Tb)[i] + [0, 0, 0.003]
            for _ in range(3000):
                done, why = ctl._chunk(p, R0, False, False)
                assert why is None, why
                if done:
                    break
                run_until(lambda: True, DT)
            ctl.call("descend")
            run_until(lambda: ctl.jog.mode == "idle" and ctl.n_done == ctl.n_put, 10)
            assert "접촉" in ctl.msg, ctl.msg
            for _ in range(int(0.6 / DT)):   # 기록 창(0.5 s) 채우기
                clk[0] += DT
                ctl.step(clk[0])
            ctl.call("record", "charuco", i, Tb, pose)
            run_until(lambda: ctl.n_done == ctl.n_put, 1)
            assert ctl.msg.startswith("기록 #"), ctl.msg
            if k < 3:
                ctl.call("lift")
                run_until(lambda: ctl.jog.mode == "idle" and ctl.n_done == ctl.n_put, 10)
            assert ("charuco" in ctl.fits) == (k == 3)
        fc = ctl.fits["charuco"]
        assert fc["ok"] and fc["kind"] == "4점" and fc["res_mm"].max() < 1.5 and fc["diag_mm"].max() < A.DIAG_WARN_MM, A.fit_pose(fc)
        assert [p["aligned_by"] for p in ctl.doc["points"]] == ["human_corner"] * 4
        assert ctl.get_snap()["fit_ver"] == ctl.fit_ver == 4
        face_min = float(to_base(CORNERS, fc["T"])[:, 2].min())
        assert ctl.z_floor == ctl.jog.z_floor >= face_min - FLOOR_BELOW_M - 1e-9 and ctl.doc["tip_z_floor_m"] == round(ctl.z_floor, 5)
        # (j) 자동: 접촉한 채(마지막 모서리) 시작 → 올림 → 수평 → 위 5 mm → 대기 → 건너뛰기 → … → 대기 → Enter → 하강 → 기록 → 위로 → 다음
        ctl.call("auto", True)
        run_until(lambda: ctl.auto is not None and ctl.auto["phase"] == "wait", 30)
        assert ph == ["rise", "travel", "approach", "hover", "wait"], ph
        a0 = dict(ctl.auto)
        assert f"위 {AUTO_HOVER_M * 1000:.0f} mm" in ctl.msg and a0["name"] not in {"bc00", "bc05", "bc18", "bc23"}, ctl.msg
        zt = [p[2] - top for k, p, top in trace if k == "travel"]
        assert min(zt) >= AUTO_CLEAR_M - 1e-3, min(zt)   # 수평 이동 = 보드 면 최고점 + 30 mm 이상
        ap = np.array([p for k, p, _ in trace if k == "approach"])
        v_ap = float(np.linalg.norm(np.diff(ap, axis=0), axis=1).max() / DT)
        assert v_ap <= AUTO_APPROACH_V * 1.05, v_ap
        tgt = to_base(CORNERS, fc["T"])[a0["ti"]]
        dm = (ctl._tip_meas()[:3, 3] - tgt) * 1000
        assert np.abs(dm[:2]).max() < 1.0 and abs(dm[2] - AUTO_HOVER_M * 1000) < 1.0, dm
        ctl.call("jog", "x", 1, 0.001, "base")   # 대기 중 조그 보정은 된다
        run_until(lambda: ctl.jog.mode == "idle" and ctl.n_done == ctl.n_put, 5)
        assert ctl.msg.startswith("조그 ") and ctl.auto["phase"] == "wait", ctl.msg
        ctl.call("skip")
        run_until(lambda: ctl.auto["phase"] == "travel", 10)
        assert ("charuco", a0["ti"]) in ctl.auto_skip and ctl.auto["name"] != a0["name"]
        ctl.call("jog", "x", 1, 0.001, "base")   # 이동 중 조그 거부
        run_until(lambda: ctl.n_done == ctl.n_put, 1)
        assert "자동 이동 중" in ctl.msg and ctl.auto is not None, ctl.msg
        run_until(lambda: ctl.auto["phase"] == "wait", 30)
        a1, ph[:] = dict(ctl.auto), ["wait"]
        ctl.call("auto_descend")   # = Enter
        run_until(lambda: ctl.auto is not None and ctl.auto["name"] != a1["name"], 30)
        assert ph == ["wait", "descend", "settle", "retreat", "rise"], ph
        last = ctl.doc["points"][-1]
        assert last["target_id"] == a1["name"] and last["aligned_by"] == "human" and last["board_T_display"] == np.round(fc["T"], 6).tolist()
        assert last["contact_frac"] >= REC_CONTACT_FRAC and json.loads(out.read_text())["points"][-1]["aligned_by"] == "human"
        print(f"(j) 4 모서리(잔차 최대 {fc['res_mm'].max():.2f} mm · 대각선 차 {fc['diag_mm'].max():.2f} mm) → 자동 {a0['name']} "
              f"위 {AUTO_HOVER_M * 1000:.0f} mm(xy {np.abs(dm[:2]).max():.2f} mm, 수평 이동 면 위 ≥ {min(zt) * 1000:.1f} mm, "
              f"접근 최대 {v_ap * 1000:.1f} mm/s) → 건너뜀 → {a1['name']} 대기 → Enter → 접촉 → 기록(human) → 위로 → "
              f"{ctl.auto['name']} · 이동 중 조그 거부")
        # (k) space = 정지 → 자동 해제
        run_until(lambda: ctl.auto["phase"] == "travel" and ctl.jog.mode == "move", 10)
        ctl.call("stop")
        run_until(lambda: ctl.n_done == ctl.n_put, 1)
        assert ctl.auto is None and ctl.jog.mode == "idle" and "자동 해제" in ctl.msg, ctl.msg
        n_pts = len(ctl.doc["points"])
        print(f"(k) space → {ctl.msg}")
        # (l) 확인 없이 자동 하강: N 뒤 첫 점은 확인 대기(human), 다음부터 대기 없이(auto), N 한 번에 6 점에서 해제 · '확인 없이' 도 꺼짐
        ctl.call("auto_free", True)
        ctl.call("auto", True)
        ph.clear()
        run_until(lambda: ctl.auto is not None and ctl.auto["phase"] == "wait", 30)
        assert ctl.auto["n"] == 1 and not ctl.auto["free"], ctl.auto
        ctl.call("auto_descend")
        run_until(lambda: ctl.auto is None, 200)
        new = ctl.doc["points"][n_pts:]
        assert f"{AUTO_MAX_PTS} 점" in ctl.msg and len(new) == AUTO_MAX_PTS, (ctl.msg, len(new))
        assert [p["aligned_by"] for p in new] == ["human"] + ["auto"] * (AUTO_MAX_PTS - 1) and ph.count("wait") == 1, ph
        assert not ctl.auto_free and ctl.fits["charuco"]["kind"] == "4점"   # auto 점은 맞춤에 안 들어간다(사람 점 6 < 8)
        print(f"(l) 확인 없이: 첫 점 wait(human) 뒤 {AUTO_MAX_PTS - 1} 점 auto · {ctl.msg} · '확인 없이' 꺼짐 · 맞춤 {ctl.fits['charuco']['kind']}")
        selftest_release(ctl, run_until, clk)
        ctl._finish("selftest")


def selftest_autoall(kin: Kin, lo: np.ndarray, hi: np.ndarray, home: np.ndarray) -> None:
    """(q) --seed · 전체 자동: seed 거부(side · md5) · 불러온 4 모서리(bc18 은 human)로 맞춤 · 원본 불변 / 유지 수렴 → 처음 2 점 Enter →
    그 뒤 확인 없이 / 사진(가짜 영상) · 사진 없어도 기록 / 수렴 시간 초과 = 건너뜀(이유 JSON) / 이상(접촉) = 그 자리 정지 · 요약."""
    import tempfile
    from types import SimpleNamespace

    side, T0 = kin.side, kin.tip(home)
    tip0 = T0[:3, 3]
    zc = float(tip0[2]) - 0.010
    Tb = board_T(tip0[0] - 0.065, tip0[1] + 0.075, zc, 0.0)
    md5 = hashlib.md5(URDF_DEFAULT.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sd = new_doc(kin, URDF_DEFAULT, np.asarray(POINT), FK_DZ_MM[side], {"charuco": {"source": "t", "T_base_board": Tb.tolist()}}, zc - 0.008)
        X = to_base(CORNERS, Tb)
        for i, by, dl in ((0, "human_corner", False), (15, "human", True), (5, "human_corner", False), (23, "human_corner", False), (18, "human", False)):
            T = T0.copy()
            T[:3, 3] = X[i]
            sd["points"].append({"board_id": "charuco", "target_id": f"bc{i:02d}", "T_tip_fk": T.tolist(), "deleted": dl, "aligned_by": by,
                                 "tip_dir_base": T[:3, 2].tolist(), "tip_dir_deg": [0, -90], "force": {"index_jf_delta_g": None}})
        sp = td / "seed.json"
        save_json_atomic(sp, sd)
        h0 = hashlib.md5(sp.read_bytes()).hexdigest()
        for bad, key in (({**sd, "side": "left"}, "side"), ({**sd, "urdf_md5": "0" * 32}, "urdf_md5")):
            save_json_atomic(td / "bad.json", bad)
            try:
                load_seed([td / "bad.json"], side, md5, ["charuco"], zc)
                raise AssertionError(f"{key} 다른 seed 를 받았다")
            except ValueError as error:
                assert "섞지 않는다" in str(error), error
        pts, info, disp = load_seed([sp], side, md5, ["charuco"], zc)
        assert [p["target_id"] for p in pts] == ["bc00", "bc05", "bc23", "bc18"] and all(p["seeded_from"].startswith(str(sp)) for p in pts)
        assert info[0]["n"] == 4 and np.allclose(disp["charuco"][0], Tb)
        out = td / "rec" / "all.json"
        doc = new_doc(kin, URDF_DEFAULT, np.asarray(POINT), FK_DZ_MM[side], {"charuco": {"source": "seed", "T_base_board": Tb.tolist()}}, zc - 0.008)
        doc["points"], doc["seeded_from"] = pts, info
        io, clk = FakeIO(side, kin, home, zc), [100.0]
        ctl = Ctl(io, kin, lo, hi, home, np.asarray(POINT), execute=False, demo=True, out=out, doc=doc, clock=lambda: clk[0],
                  z_floor=zc - FLOOR_BELOW_M, hold=True, auto_all=True, confirm_first=2, photo_dir=td / "rec" / "all_photos",
                  extrinsics=S2R / "config/global_camera_extrinsics_arm4090_r20u5.yaml")
        f = ctl.fits["charuco"]
        assert f["ok"] and f["kind"] == "4점" and np.allclose(f["T"], Tb, atol=1e-6) and ctl.sel_bid == "charuco"
        assert hashlib.md5(sp.read_bytes()).hexdigest() == h0 and len(json.loads(out.read_text())["points"]) == 4   # 원본 그대로 · 새 출력에 복사
        ctl.view = {"boards": {"charuco": Tb}, "target": None}
        orig = time.sleep
        time.sleep = lambda s_: clk.__setitem__(0, clk[0] + max(s_, 0.01))
        try:
            ctl.start_check(1.0)
        finally:
            time.sleep = orig
        ctl.jog.cmd, ctl.jog.c = ctl.jog.cmd + FakeIO.E0, -FakeIO.E0
        io.info = np.array([[600.0, 0, 80], [0, 600.0, 60], [0, 0, 1]])
        frame = SimpleNamespace(encoding="bgr8", height=120, width=160, step=480, data=np.full((120, 160, 3), 170, np.uint8).tobytes())
        cam, ph = [True], []

        def run_until(cond, sec: float) -> None:
            for _ in range(int(sec / DT)):
                clk[0] += DT
                io.img = (frame, clk[0]) if cam[0] else io.img   # 60 Hz 가짜 카메라(끄면 마지막 영상이 늙는다)
                ctl.step(clk[0])
                if ctl.auto is not None and (not ph or ph[-1] != (ctl.auto["name"], ctl.auto["phase"])):
                    ph.append((ctl.auto["name"], ctl.auto["phase"]))
                if cond():
                    return
            raise AssertionError(f"{sec} s 안에 안 됨: {ctl.msg} · {ph[-6:]}")

        ctl.call("auto", True)
        for k in range(2):   # 처음 2 점: 수렴 → Enter 확인(human)
            run_until(lambda: ctl.auto is not None and ctl.auto["phase"] == "wait", 40)
            assert ph[-2][1] == "converge" and "확인" in ctl.msg, (ph[-3:], ctl.msg)
            ctl.call("auto_descend")
            n0 = len(ctl.doc["points"])
            run_until(lambda: len(ctl.doc["points"]) > n0, 30)
        cam[0] = False   # 셋째 점: 확인 없이 · 카메라 끊김 → 기록은 계속, 사진 이유
        n0 = len(ctl.doc["points"])
        run_until(lambda: len(ctl.doc["points"]) > n0, 40)
        names = [n for n, p_ in ph if p_ == "settle"]
        assert [p_ for n, p_ in ph if n == names[2]] == ["rise", "travel", "approach", "hover", "converge", "descend", "settle", "retreat"], ph
        new = ctl.doc["points"][4:]
        assert [p["aligned_by"] for p in new] == ["human", "human", "auto"] and all(p["fit_res_mm"] < 1.0 for p in new), new
        assert new[2]["photo_check"]["status"] == "none" and "영상" in new[2]["photo_check"]["why"] and "사진 없음" in ctl.msg
        cam[0] = True
        ctl._converged = lambda: (False, "시험 9.9 mm")   # 넷째 점: 수렴 안 됨 → 8 s 뒤 건너뜀(이유 JSON) → 다음 점
        run_until(lambda: bool(ctl.doc.get("auto_skipped")), 40)
        sk = ctl.doc["auto_skipped"][0]
        assert "유지 수렴" in sk["why"] and ctl.auto is not None and ctl.auto["name"] != sk["target_id"] and ("charuco", int(sk["target_id"][2:])) in ctl.auto_skip
        del ctl._converged
        run_until(lambda: ctl.auto["phase"] == "travel" and ctl.jog.mode == "move", 20)
        n0 = len(ctl.doc["points"])
        io.dz = 0.3   # 이상(수평 이동 중 접촉) → 그 자리 정지, 다음 점으로 안 감 · 요약
        run_until(lambda: ctl.auto is None, 5)
        io.dz = 0.0
        for _ in range(int(1.0 / DT)):
            clk[0] += DT
            ctl.step(clk[0])
        assert ctl.auto is None and not ctl.all_run and len(ctl.doc["points"]) == n0 and "요약" in ctl.msg, ctl.msg
        for _ in range(500):   # 사진 Worker(실제 스레드) 결과를 기다려 넣는다
            ctl._drain_photos()
            if all(p["photo_check"]["status"] != "pending" for p in ctl.doc["points"][4:]):
                break
            time.sleep(0.01)
        ctl._finish("selftest")
        d = json.loads(out.read_text())
        pc = [p["photo_check"] for p in d["points"][4:]]
        assert [c["status"] for c in pc] == ["done", "done", "none"] and pc[0]["why"].startswith("보드 검출 실패"), pc
        assert (td / "rec" / "all_photos" / pc[0]["photo"]).exists() and (td / "rec" / "all_photos" / pc[0]["overlay"]).exists()
        sm = d["auto_summary"][-1]
        assert sm["n_points"] == 3 and sm["fit_res_mm"]["max"] < 1.0 and sm["photo"]["n_photo"] == 2 and not sm["over"], sm
        assert sum("seeded_from" in p for p in d["points"]) == 4 and d["seeded_from"][0]["n"] == 4
    print(f"(q) seed: side · md5 다르면 거부 · 4 모서리(bc18 human) 맞춤 · 원본 불변 / 전체 자동 {' → '.join(names[:3])}: 수렴 → 처음 2 점 Enter "
          f"(human) → 셋째 확인 없이(auto) · 카메라 끊겨도 기록(사진 이유) / 수렴 안 됨 {sk['target_id']} 8 s 뒤 건너뜀 · 다음 점 / "
          f"수평 이동 중 접촉 → 그 자리 정지 · 요약 {sm['n_points']} 점 잔차 최대 {sm['fit_res_mm']['max']:.2f} mm · 사진 {sm['photo']['n_photo']}")


def selftest_release(ctl: Ctl, run_until, clk: list) -> None:
    """(m) 자동 해제 경로마다: auto None · jog idle · 다음 틱 명령 불변. Enter 는 확인 대기에서만."""
    io, done = ctl.io, []

    def start() -> None:
        ctl.call("auto", True)
        run_until(lambda: ctl.n_done == ctl.n_put, 1)
        assert ctl.auto is not None, ctl.msg

    def at(p: str, move: bool = False):
        return lambda: ctl.auto is not None and ctl.auto["phase"] == p and (not move or ctl.jog.mode == "move")

    def released(tag: str, sub: str, sec: float) -> None:
        run_until(lambda: ctl.auto is None, sec)
        assert ctl.jog.mode == "idle" and sub in ctl.msg, (tag, ctl.jog.mode, ctl.msg)
        c = ctl.jog.cmd.copy()
        clk[0] += DT
        ctl.step(clk[0])
        assert ctl.auto is None and ctl.jog.mode == "idle" and np.array_equal(c, ctl.jog.cmd), tag
        done.append(tag)

    def settle_io(sec: float = 0.5) -> None:   # 주입을 되돌린 뒤 힘 · 접촉 판정이 따라오게
        for _ in range(int(sec / DT)):
            clk[0] += DT
            ctl.step(clk[0])

    ctl.call("auto_descend")   # 자동 없음 — Enter 무시
    run_until(lambda: ctl.n_done == ctl.n_put, 1)
    assert "Enter 무시" in ctl.msg and ctl.jog.mode == "idle", ctl.msg
    start()
    run_until(at("travel", True), 10)
    ctl.call("auto_descend")   # 이동 중 — Enter 무시, 자동은 계속
    run_until(lambda: ctl.n_done == ctl.n_put, 1)
    assert "Enter 무시" in ctl.msg and ctl.auto is not None and ctl.jog.mode not in ("dwell", "descend"), ctl.msg
    io.stale = True   # 힘 끊김
    released("힘 끊김", "힘 끊김", 2)
    io.stale = False
    settle_io()
    for ph_ in ("travel", "approach"):   # 이동 중 접촉(면 높이 어긋남 주입)
        start()
        run_until(at(ph_, True), 15)
        io.dz = 0.3
        released(f"{ph_} 중 접촉", "접촉", 2)
        io.dz = 0.0
        settle_io()
    start()   # plan 거부(관절 한계 주입)
    run_until(at("approach", True), 15)
    lim = ctl.jog.lo.copy(), ctl.jog.hi.copy()
    ctl.jog.lo, ctl.jog.hi = ctl.jog.cmd - 1e-5, ctl.jog.cmd + 1e-5
    released("plan 거부", "거부:", 3)
    ctl.jog.lo, ctl.jog.hi = lim
    start()   # 무접촉 하강 끝(실제 면이 50 mm 아래)
    run_until(at("wait"), 30)
    io.dz = -0.05
    ctl.call("auto_descend")
    released("무접촉 하강 끝", "접촉 없이", 30)
    ctl.call("lift")
    run_until(lambda: ctl.jog.mode == "idle" and ctl.n_done == ctl.n_put, 10)
    io.dz = 0.0
    start()   # settle 시간 초과(접촉 뒤 면이 4 mm 내려감 → 기록 조건 안 됨)
    run_until(at("wait"), 30)
    ctl.call("auto_descend")
    run_until(at("settle"), 30)
    n0 = len(ctl.doc["points"])
    io.dz = -0.004
    released("settle 시간 초과", "기록 거부", AUTO_REC_WAIT_S + 1)
    assert len(ctl.doc["points"]) == n0
    io.dz = 0.0
    settle_io()
    assert ctl.sig in ("index_force", "tip_touch")
    T0 = ctl.kin.tip(ctl.jog.cmd, ctl.jog.hand)
    start()   # 접촉 중 N → 순수 +z 로 먼저 빼기
    run_until(at("travel"), 15)
    T1 = ctl.kin.tip(ctl.jog.cmd, ctl.jog.hand)
    dxy, dth = np.linalg.norm(T1[:2, 3] - T0[:2, 3]) * 1000, np.degrees(np.linalg.norm(rotvec(T1[:3, :3] @ T0[:3, :3].T)))
    assert dxy < 0.3 and dth < 0.05 and T1[2, 3] - T0[2, 3] > 0.02, (dxy, dth, T1[2, 3] - T0[2, 3])
    run_until(at("wait"), 30)   # 위로 빼기 거부(관절 한계 주입) — 기록은 남고 상태 띠에 PgUp 안내
    ctl.call("auto_descend")
    run_until(at("settle"), 30)
    n0 = len(ctl.doc["points"])
    lim = ctl.jog.lo.copy(), ctl.jog.hi.copy()
    ctl.jog.lo, ctl.jog.hi = ctl.jog.cmd - 1e-5, ctl.jog.cmd + 1e-5
    released("위로 빼기 거부", "PgUp", 3)
    ctl.jog.lo, ctl.jog.hi = lim
    assert len(ctl.doc["points"]) == n0 + 1
    ctl.call("auto_free", True)   # 포커스 잃음 → 자동 · '확인 없이' 모두 끔
    start()
    run_until(at("travel", True), 15)
    ctl.call("auto", False, "창 포커스 잃음")
    released("포커스 잃음", "포커스", 1)
    assert not ctl.auto_free
    start()   # 보드 바뀜
    run_until(at("travel", True), 15)
    ctl.call("board", "grid50")
    released("보드 바뀜", "보드 바뀜", 1)
    ctl.call("auto", True)
    run_until(lambda: ctl.n_done == ctl.n_put, 1)
    assert ctl.auto is None and "자동 거부 — grid50" in ctl.msg, ctl.msg
    ctl.call("board", "charuco")
    start()   # 자동 밖에서 맞춤 판이 바뀜(사람 기록 지움)
    run_until(at("travel", True), 15)
    k = next(i for i, p in enumerate(ctl.doc["points"]) if p["aligned_by"] == "human" and not p["deleted"])
    ctl.call("delete", k)
    released("fit_ver 바뀜", "맞춤이 바뀜", 1)
    ctl.call("delete", k)
    run_until(lambda: ctl.n_done == ctl.n_put, 1)
    print(f"(m) 자동 해제 {len(done)} 경로(auto None · jog idle · 다음 틱 명령 불변): {' · '.join(done)} / "
          f"Enter 는 대기 아닐 때 무시 · 접촉 중 N = 순수 +z(xy {dxy:.2f} mm · {dth:.3f}°) · 다른 보드 맞춤 없으면 N 거부")


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"))
    ap.add_argument("--out", type=Path, help="기록 JSON(있으면 시작 거부). 모든 틱 = 같은 이름 _raw_partNN.npz(10 분마다 · 종료 때)")
    ap.add_argument("--urdf", type=Path, default=URDF_DEFAULT)
    ap.add_argument("--grip", help=f"손 POINT 자세 rad, {','.join(P.HAND_AXES)} (기본 {','.join(map(str, POINT))})")
    ap.add_argument("--fk_dz_mm", type=float, help="표시 보드 높이 = 205 + 이 값(기본 오른팔 +12 · 왼팔 −12, 11.04 측정)")
    ap.add_argument("--boards", default="charuco", help=f"쓸 보드(쉼표) — {', '.join(BOARDS)} (기본 charuco)")
    ap.add_argument("--board_pose", type=Path, help="보드 표시 자세 yaml/json: {board_id: {x, y[, z], yaw_deg} 또는 {T_base_board: 4x4}} "
                                                    "또는 한 보드 형식(= --boards 첫 보드). m · °")
    ap.add_argument("--board_npz", type=Path, help="grab_rgbd 캡처 — plan_board.board_pose 로 charuco 대략 자세(sim2real/.venv 의 cv2 4.14)")
    ap.add_argument("--extrinsics", type=Path, default=S2R / "config/global_camera_extrinsics_arm4090_r20u5.yaml")
    ap.add_argument("--execute", action="store_true", help="실기 발행 체크를 켤 수 있게(켜기 전까지 발행 0)")
    ap.add_argument("--demo", action="store_true", help="ROS 없이 가짜 팔 · 힘(발행 없음)")
    ap.add_argument("--control", choices=("legacy", "precise"), default="legacy", help="precise: 120Hz · 가속도 제한 · 이동 중 실측 손 FK 추종 보정(실기 미검증)")
    ap.add_argument("--no_hold", action="store_true", help="유지 보정(c) 끔 — 옛 방식: 실측 ≈ 명령 + e0(고정 처짐)")
    ap.add_argument("--continuous_hold", action="store_true", help="실험용 연속 유지 적분 — 기본은 발행 시작 보정값 고정(정지 중 목표 변동 방지)")
    ap.add_argument("--sim_port", type=int, default=SIM_PORT, help=f"sim 창으로 {SIM_HOST}:<port> {SIM_HZ:g} Hz UDP JSON(0 = 끔)")
    ap.add_argument("--seed", type=Path, nargs="+", help="이전 기록 JSON(들) — 산 점을 새 --out 에 seeded_from 으로 복사하고 처음 맞춤에 쓴다 "
                                                        "(원본은 읽기만, side · urdf_md5 가 다르면 거부). 주면 N = 남은 점 전체 자동(연속)")
    ap.add_argument("--auto_confirm_first", type=int, default=2, help="전체 자동에서 하강 전에 Enter 확인할 처음 점 수(기본 2, 0 = 확인 없음)")
    ap.add_argument("--photo_topic", default=PH.COLOR_TOPIC, help=f"점마다 사진 원본 컬러 토픽(기본 {PH.COLOR_TOPIC}, camera_info 는 같은 "
                                                                   "namespace). '' = 사진 끔. 압축 토픽은 쓰지 않는다")
    ap.add_argument("--camera", choices=("auto", "off"), default="auto", help="auto = 영상이 1.5 s 안에 없고 카메라 노드 프로세스도 없으면 "
                                                                              "우리가 띄운다(끝낼 때 우리 것만 SIGTERM). off = 띄우지 않는다")
    ap.add_argument("--smoke", help="demo 를 몇 초 돌려 화면을 이 png 로 저장하고 끝낸다(QT_QPA_PLATFORM=offscreen 시험)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.no_hold and a.continuous_hold:
        ap.error("--no_hold 와 --continuous_hold 는 함께 사용할 수 없다")
    if a.selftest:
        return selftest()
    if a.side is None:
        ap.error("--side 가 필요하다")
    if not 0 <= a.sim_port <= 65535:
        ap.error(f"--sim_port {a.sim_port}: 0(끔) 또는 1-65535")
    if a.demo and a.execute:
        ap.error("--demo 와 --execute 는 같이 쓰지 않는다")
    if a.out is None and not a.demo:
        ap.error("--out 이 필요하다(--demo 만 없이 된다)")
    if a.auto_confirm_first < 0:
        ap.error("--auto_confirm_first 는 0 이상")
    bids = [b.strip() for b in a.boards.split(",") if b.strip()]
    if not bids or len(set(bids)) != len(bids) or not set(bids) <= set(BOARDS):
        ap.error(f"--boards {a.boards!r}: 겹치지 않는 {', '.join(BOARDS)} 중에서")
    if a.board_npz and "charuco" not in bids:
        ap.error("--board_npz 는 charuco 자세만 찾는다 — --boards 에 charuco 를 넣는다")
    try:
        if a.out is not None:
            check_out(a.out)
        grip = P.parse_pose(a.grip, POINT, "--grip")
        requested_grip = grip.copy()
        if a.control == "precise":
            grip = reachable_grip(grip, a.side)
            print(f"손 명령 도달범위 반영: 요청 {requested_grip.tolist()} → {grip.tolist()}", flush=True)
        kin = Kin(a.urdf, a.side, grip)
        lo, hi = arm_limits(a.side)
        home = np.array(json.loads(CONTRACT.read_text())["sides"][a.side]["home_arm"], float)
        fk_dz = FK_DZ_MM[a.side] if a.fk_dz_mm is None else a.fk_dz_mm
        zc = TABLE_Z + fk_dz / 1000
        if a.demo:   # 가짜 힘이 나는 면 = 표시 보드면, 홈 손끝 15 mm 아래(하강 40 mm 안)
            zc = float(kin.tip(home)[2, 3]) - 0.015
        seeds, seed_info, seed_disp = load_seed(a.seed, a.side, hashlib.md5(a.urdf.read_bytes()).hexdigest(), bids, zc) if a.seed else ([], [], {})
        boards, srcs = {}, {}
        if a.board_pose:
            for b, (T, ed) in load_board_poses(a.board_pose, zc, bids).items():
                boards[b], srcs[b] = T, {"source": f"file {a.board_pose}", "editable": ed}
        if a.board_npz and "charuco" not in boards:
            boards["charuco"], info = board_from_npz(a.board_npz, a.extrinsics, zc)
            srcs["charuco"] = {"source": "npz", "editable": False, **info}
            for w in info["warn"]:
                print(f"보드 경고: {w}", file=sys.stderr)
        for b, (T, src) in seed_disp.items():
            if b not in boards:
                boards[b], srcs[b] = T, {"source": f"seed {src}", "editable": False}
        t = kin.tip(home)[:3, 3]   # 자세 없는 보드: 첫 보드 가운데가 홈 손끝 아래, 다음 보드는 바깥쪽 y 로 0.13 m 씩
        for i, b in enumerate(bids):
            if b not in boards:
                c = BOARDS[b]["targets"][:, :2].mean(0)
                dy = 0.13 * i * (-1 if a.side == "right" else 1)
                boards[b], srcs[b] = board_T(t[0] - c[0], t[1] + c[1] + dy, zc, 0.0), {"source": "gui", "editable": True}
        boards = {b: boards[b] for b in bids}
        for b in bids:
            srcs[b]["T_base_board"] = boards[b].tolist()
        # H1: 명령 손끝 z 하한 = 가장 높은 표시 보드면 − 8 mm(보드가 하나도 탁자면을 못 주면 zc 가 그 자리)
        z_floor = max(float(to_base(BOARDS[b]["targets"], boards[b])[:, 2].mean()) for b in bids) - FLOOR_BELOW_M
        print(f"손끝 z 하한 {z_floor * 1000:.1f} mm (표시 보드면 − {FLOOR_BELOW_M * 1000:.0f} mm)", flush=True)
        doc = new_doc(kin, a.urdf, grip, fk_dz, srcs, z_floor) if a.out is not None or seeds else None   # seed 는 demo 에서도 메모리 문서
        if doc is not None:
            doc["control"] = {"mode": a.control, "hz": 120 if a.control == "precise" else 60,
                              "requested_grip": requested_grip.tolist(), "reachable_grip": grip.tolist(),
                              "board_z_note": "표시 추정값; FK 보정이나 실측 높이가 아님"}
        if seeds:
            doc["points"], doc["seeded_from"] = seeds, seed_info
            print(f"seed {len(seeds)} 점 ← " + " · ".join(f"{i['path']} {i['n']} 점" for i in seed_info), flush=True)
        if a.demo:   # 가짜 힘 면 = 첫 보드 표시 면(seed 표시 자세여도 그 높이에서 닿는다)
            zc = float(to_base(BOARDS[bids[0]]["targets"], boards[bids[0]])[:, 2].mean())
        io = FakeIO(a.side, kin, home, zc) if a.demo else RosIO(a.side, a.photo_topic or None)
    except (P.TeachError, ValueError, OSError, KeyError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    photos = a.out.with_name(f"{a.out.stem}_photos") if a.out is not None and a.photo_topic else None
    ctl = Ctl(io, kin, lo, hi, home, grip, a.execute, a.demo, a.out, doc, z_floor=z_floor, hold=not a.no_hold,
              auto_all=bool(seeds), confirm_first=a.auto_confirm_first, photo_dir=photos, extrinsics=a.extrinsics, precise=a.control == "precise",
              continuous_hold=a.continuous_hold)
    if seeds:
        f = ctl.fits.get(ctl.sel_bid)
        print(f"seed 맞춤 {ctl.sel_bid}: " + ("없음(바깥 4 모서리가 다 없다)" if f is None else f"{f['kind']} " + (
              f"받아들임 · 잔차 최대 {f['res_mm'].max():.1f} mm" if f["ok"] else f"★거부 — {f['why']}")), flush=True)
    ctl.view = {"boards": dict(boards), "target": None}
    if isinstance(io, RosIO) and a.photo_topic:   # 사진용 카메라: 상태 · 켜기 · 끄기는 창, 시작 분기는 --camera
        ctl.cam = PH.Camera(lambda: io.n_img[0], io.cam_hz, run_paths.day_dir("perception") / "realsense_board_jog.log")
        if a.camera == "auto":
            threading.Thread(target=lambda: say(f"카메라: {ctl.cam.auto()}"), daemon=True).start()
    ctl.start()
    if a.sim_port:
        threading.Thread(target=sim_sender, args=(ctl, kin, a.sim_port), daemon=True).start()
    released = False
    try:
        from board_jog_gui import run_gui   # Qt · matplotlib 은 창을 띄울 때만

        rc = run_gui(ctl, kin, lo, hi, boards, srcs, smoke=a.smoke)
    finally:
        released = ctl.stop()
        if ctl.cam is not None and ctl.cam.mine():   # 우리가 띄운 카메라만 SIGTERM(남의 것은 그대로)
            say(f"카메라: {ctl.cam.stop(block=True)}")
        if not released:   # M1 — 해제 · episode stop 이 끝났다고 말할 수 없다
            say("\n★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★\n"
                "★ 해제 미완료 — 제어 스레드가 15 s 안에 끝나지 않았다. pd 상태(TRACKING · target)를 직접 확인한다 ★\n"
                "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★", err=True)
    n = 0 if ctl.doc is None else sum(not p["deleted"] for p in ctl.doc["points"])
    say(json.dumps({"phase": "stopped", "out": None if a.out is None else str(a.out), "points": n, "fatal": ctl.fatal,
                    "released": released}, ensure_ascii=False))
    return rc if released else max(rc, 2)


if __name__ == "__main__":
    sys.modules.setdefault("board_jog", sys.modules[__name__])   # board_jog_gui 의 import board_jog 가 이 모듈을 다시 싣지 않게
    raise SystemExit(main())
