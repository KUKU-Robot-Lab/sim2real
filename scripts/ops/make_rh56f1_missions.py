#!/usr/bin/env python3
"""RH56F1 로봇 미션(로봇 PC = ROBOT_PC, 10.13 부터 5090 + 인지 vision-3090) — 실기 · fake 두 벌을 **한 정의에서** 만든다.

    python3 scripts/ops/make_rh56f1_missions.py            # config/mission_rh56f1_{control,fake}.yaml 을 다시 쓴다
    python3 scripts/ops/make_rh56f1_missions.py --check    # 커밋된 두 파일이 이 정의에서 나온 것인지(테스트가 부른다)

09.29 사용자: "sim2real 과 robot_control 쪽에서 rh56f1 제어 part 연결" · 손마다 개별 포트 · USB RS485 / CANFD 를
상황에 따라 바꿔 쓴다 · 정책은 pour_fj(양팔) 다음 rh_aglt — 곧 나온다.
범위 = 손 연결 · 점검 · 한 축 방향 확인 · 팔 pd(무발행 → 발행) · 홈 · 두 컵 · 양팔 pour_fj 정책 · 정리.
홈 = hdgp rh_aglt 시작 자세(09.29 사용자 "aglt 보면 home 자세를 수정했어"). 차렷 → 홈은 저장 경로(paths/home_rh56f1_*.npz,
plan_home_path RRT · RH56F1 자산 충돌 검사 · 편 손/접은 손 둘 다)를 pd 로 재생한다 — 실기 · fake 같은 경로.
★10.01 인지는 arm4090 안에서(docker FP++ · RealSense · 런처 --host local). 실기 컵은 cyl60 원통 둘(10.08 사용자: 오른쪽 노랑
cyl60 · 왼쪽 파랑 cyl60_blue — FP++ 가 색으로 가른다. 10.04~10.07 은 노랑 하나, 그 전 aglt_cup_s065) — 한 팔 rh_aglt 는
/objects/<REAL_CUPS[팔]>/pose 를 읽는다. 양팔 pour_fj 는 붓기 보류(사용자 확인 전)라 막아 둔다.
카메라 좌표는 고정 외부 파라미터(5090 홈 화면) — 머리를 head_home_rh56f1(5090 과 같은 화면)에 둔 뒤에만 맞다.

실기와 fake 가 다른 것(그 밖은 같다):
  · 팔 드라이버: CAN + openarm bringup ↔ fake 플랜트(양팔 MockArm rate, hands:=none)
  · 손 드라이버: tools/rh56f1_driver.py(포트 설정 rh56f1_ports.yaml) ↔ scripts/fakes/fake_rh56f1_hand.py
  · robot yaml · pd 설정 → *_fake, ros2 launch 에 fake:=true, touches_real 을 모두 뗀다(승인 없음, 단계 · 순서는 같다)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))   # object_registry — 묶음 물체 목록
OUT = {"real": REPO / "config" / "mission_rh56f1_control.yaml", "fake": REPO / "config" / "mission_rh56f1_fake.yaml"}
SIDES = ("right", "left")
PC = "{repo}/deploy/policy_control"
#: 방향 확인 한 축씩 — 네 손가락은 굽힘 쪽(+), 엄지 굽힘 +, 엄지 회전은 지금에서 − (편 손 1.57 에서 grip 1.20 쪽)
PROBES = (("index_1", "0.3"), ("middle_1", "0.3"), ("ring_1", "0.3"), ("pinky_1", "0.3"),
          ("thumb_2", "0.15"), ("thumb_1", "-0.3"))
#: 실기 컵 — config/fpp_active.yaml(scripts/ops/fpp_object.py activate)이 정한다: 묶음(컨테이너 fpp_<묶음> 하나가 한 번 찍기) ·
#: 팔별 물체. fake 는 학습 배치의 두 컵(cup_src · cup_rcv). 10.04 cyl60 노랑 → 10.08 왼쪽 파랑 추가 → 10.09 파일로(물체를 자주 바꾼다)
ACTIVE_FPP = REPO / "config" / "fpp_active.yaml"


def _active() -> tuple[list[str], dict[str, str]]:
    """(묶음들, 팔별 물체) — 팔마다 다른 묶음이어도 된다(10.09 오른손 source200 병 · 왼손 shaker_c). 묶음은 팔 물체에서 푼다."""
    from object_registry import load_registry
    raw = yaml.safe_load(ACTIVE_FPP.read_text(encoding="utf-8")) or {}
    sides = {str(k): str(v) for k, v in (raw.get("sides") or {}).items()}
    if "right" not in sides:
        raise SystemExit(f"{ACTIVE_FPP}: sides.right 가 필요하다(fpp_object.py activate)")
    reg = load_registry()
    groups = []
    for s_ in SIDES:
        if s_ in sides:
            g = reg.get(sides[s_]).fpp.get("group")
            if not g:
                raise SystemExit(f"{ACTIVE_FPP}: {sides[s_]} 는 묶음(fpp.group)이 없다 — fpp_object.py add 로 등록한 물체여야 한다")
            groups.append(g)
    return list(dict.fromkeys(groups)), sides


REAL_GROUPS, REAL_CUPS = _active()
REAL_CUP = REAL_CUPS["right"]
#: 놓기 목표 홀더 — 좌우 학습 목표(우 1 · 2, 좌 0 · 1)에 모두 드는 가운데 홀더 1(10.04)
PLACE_HOLDER = {"right": 1, "left": 1}
#: fake 홀더 y — hdgp rh_place env holder_ys(0.153, −0.002, −0.161)
FAKE_HOLDER_Y = {0: 0.153, 1: -0.002, 2: -0.161}
#: 상황판에서 굴리는 에피소드(10.04 사용자: 구분 실행 → 연속 실행) — config/episodes/<이름>.yaml, 실기에 쓸 수 있는 것만
# ★10.20 사용자 "경로 a는 폐지, -> 경로d가 대체함": grasp_rg3_<팔> · tilt_base_a 를 빼고 grasp_d_<팔> · tilt_base_d
EPISODES = ("pick_place_right", "pick_place_left", "grasp_c_right", "grasp_c_left",
            # ★10.17 tilt 기본 궤적 — B(수동 파지, 확인 두 번) · C(ap05 + 스크립트 파지) · ★10.20 D(1-B 접근 → rcg03, 옛 A 대체)
            "tilt_base_b", "tilt_base_c", "tilt_base_d",
            # ★10.19 1-B — 정책 없이 제어 모듈만(ready_carry 접근 → [확인] 닫기 → 10 cm 들기 · 3 s), 팔마다 따로
            "grasp_b_right", "grasp_b_left",
            # ★10.20 경로 D — 1-B 접근(ready_carry, 사전 파지 자리에서 멈춤) → rcg03 이 쥐고 들기(사용자 "grasp_b 는 그대로 두고,
            #   grasp_d 를 1-b 접근에 rcg03 으로 교체" — 옛 D 의 ap05 넘김은 실기 넘김 방위가 rcg03 학습 시작과 15~25° 달랐다)
            "grasp_d_right", "grasp_d_left",
            # ★10.20 밤 경로 E — ap05 접근 → 사전 파지 자리에서 같은 노드가 rcg03 으로 넘겨 쥐고 들기(사용자 "Ap+rg3는 기존 방법이
            #   된다고 보고 이전처럼 끊어서 진행이 아니라 정책2개가 연달아 진행되는 세트를 만들어 두기 · 내일 실험1")
            "grasp_e_right", "grasp_e_left",
            # ★10.21 경로 F — 경로 E 와 같은 넘김, 넘겨받는 정책만 rcg04(사용자 "ap05 + rcg_04 또한 준비 필요")
            "grasp_f_right", "grasp_f_left",
            # ★10.22 사용자 "Place 세팅 진행" — 경로 F 잡고 들기 → 놓기(붙인 자세 = 잡기의 grasp_state 넘김 묶음)
            #   ★10.27 결정 1: 놓기 newton_05 → 오른손 n07 · 왼손 n06(ready_carry 이송 뒤 놓기, sim A1 확인)
            "pick_place_f_right", "pick_place_f_left",
            # ★10.27 실기 2단계 grasp2tilt(sim_e2e 실기 계획) — 경로 F 양손 잡기 → ready_carry 로 tilt 시작 자세 → tilt_traj(컵 240).
            #   컵 200(grasp2tilt_f_200)은 source200 FP++ 묶음이 새로 들어가 콘솔 그림(fake 짝 · 노드 상한)을 먼저 정해야 해 뒤로 미룬다
            "grasp2tilt_f_240")
#: ★10.20 사용자 "영상은 필요 없음. 일단 기본 off" — 카메라 영상 · FP++ 실시간 좌표 기록(vision-3090)과 끝난 뒤 보고서 영상
#   만들기(episode_media.py)를 기본으로 끈다. 10.20 실기: 보고서 영상 정리가 옛 회까지 다시 가져오며 6 분 넘게 팔 작업을 붙잡아
#   rehome · tilt_now 승인이 막혔다. 켜려면 True 로 바꾸고 이 생성기를 다시 돌린다(미션 YAML 이 바뀐다).
VISION_RECORD = False
#: ★10.19 사용자 "작성할 보고서 사진/영상도 필요함" — VISION_RECORD 일 때 이 에피소드들은 카메라 영상을 보고서용 빈도로 남긴다(기본 5 Hz)
REPORT_VIDEO_HZ = {"grasp_b_right": 15.0, "grasp_b_left": 15.0}
#: 에피소드 snapshot 이 다시 내는 정지 물체 자세(episode_ros.OBJECT_RELAY) — aglt 는 이것을 컵으로 본다
EPISODE_RELAY = "/episode/objects/{}/pose"        # ★10.19 한 팔 실행기는 episode_ros.relay_topic(실행기 이름 아래)
# 제자리 들기(grasp_rg3) 정책 노드의 lift_hold 멈춤 = 컵 7 cm 위 + 쥠 연속 틱(60 Hz).
# ★10.14 사용자(1안): 오른손은 7 cm 에서 0.5 s 만에 멈추면 정책이 sim 목표(10 cm, 2 s 유지)까지 못 가 약지 · 새끼가 감기기
#   전에 끝난다(실기 190421 · 190758 엄지-검지만 쥠) — 목표 성공에 끝나게 늦춘다. sim hp3 ep1450 trace 32 에피소드: 7 cm →
#   목표 성공 2.0~4.3 s. 5 s 는 그 뒤의 받침이다(정책 시간 15 s 를 넘기면 POLICY_TIMEOUT 복구가 손을 펴 컵을 떨군다).
# ★10.17 사용자: 양손 2 s(120 틱) — 오른손 5 s 는 hp3 처럼 들고 나서 정책이 손을 열 시간을 주고, 0.5 s 는 약지 · 새끼가
#   감기기 전에 끝난다(10.14). sim hp3 의 7 cm → 목표 성공 최소 2.0 s 에 맞춘다. 첫 목표 성공도 멈춤(goal_max:=1).
LIFT_HOLD_TICKS = {"right": 120, "left": 120}
# ★10.12 사용자(A안): 실기를 sim 에 맞춘다 — aglt 정책은 쥘 때까지 FP++ live 추적(/objects/<물체>/live_pose, 약 10 Hz)을 받고
#   손이 가려 끊기면 마지막 값(sim FP++ ZOH), 쥔 뒤는 sim PalmAttach 처럼 파지 + 100 ms 뒤 찍힌 프레임과 그 시각 손바닥 FK 로
#   붙인다. 가려서 프레임이 안 오면 0.7 s(sim 이 붙는 시간 범위 위)에 파지 시작 FK 로. 10.04 의 정지 컵(cup_latch · cup_static) 대신.
AGLT_LIVE_ARGS = ("-p", "cup_static:=false", "-p", "cup_latch:=false", "-p", "cup_hold_last:=true",
                  "-p", "attach_fallback_s:=0.7",
                  # ★10.13 실기 112100: 스친 오판 부착 뒤 FP++ 컵과 FK 추정이 9 cm 벌어졌다 — 4 cm 넘는 새 프레임 3 번 연속이면 떼고 live 로
                  "-p", "attach_guard_m:=0.04", "-p", "attach_guard_frames:=3",
                  # ★10.13 A안(fpp_gate.py): 쥐기 전 손바닥 20 cm 안에서 2 cm 넘게 튀거나 테이블 아래(−2 cm)로 간 live 프레임은 버린다.
                  #   sim(vision-3090 rg3s ep4700): 실기 같은 튐을 넣으면 진짜 성공 0.25, 거르면 0.83(튐 없을 때 0.89 · 거르기만 0.89).
                  "-p", "fpp_gate_jump_m:=0.02", "-p", "fpp_gate_dz_m:=0.02", "-p", "fpp_gate_radius_m:=0.2",
                  "-p", "fpp_gate_max_hold_s:=2.5",
                  # ★10.19 사용자 승인(cup_dropout.py): live 컵이 끊기면 — 잡기 전 1 s(손바닥 20 cm 안은 학습 가림 DR 상한 2.5 s)
                  #   넘으면 멈춤 · 쥔 상태면 손바닥 FK 로 따라감 · 쥠이 풀린 뒤 0.5 s 면 멈춤. 실기 19:09 오른손: 떨어진 컵의 마지막
                  #   값(공중)을 쫓다 팔이 비틀렸다(j2 +2.2 rad). 오늘 실기 잡기 전 최장 끊김 0.18 s.
                  "-p", "cup_dropout_stop_s:=1.0", "-p", "cup_dropout_near_m:=0.2", "-p", "cup_dropout_near_max_s:=2.5",
                  # ★10.13 사용자 "a, b 둘 다 — 제어가 정책을 충실히 따라가게"(rh_aglt.GripAssist). bag 164845 · 165303 재생:
                  #   A 쥔 뒤 막힘 고정 풀기 — 오른손 검지 목표 0.52 → 0.72 rad · 왼손 0.66 → 1.08(정책 목표)
                  #   B 엄지 받침(엄지 굽힘 힘 > 80 g) + 들림(> 7 cm · 손바닥 8 cm 안) — 왼손 쥔 판정 0 틱 → 2.93 s, lift_hold 멈춤 4.59 s
                  "-p", "hand_unfreeze_on_grasp:=true", "-p", "thumb_support_g:=80.0", "-p", "lift_grasp_palm_m:=0.08",
                  # ★10.13 C(bag 223839 오른손): 컵에 밀린 엄지 굽힘(0.038 < 편 끝 0.049)으로 쥔 판정 유지가 손을 다 펴도 안 풀려
                  #   다시 쥔 컵이 안 붙음 → 실행기 쥠 실패 → 든 컵을 놓음. 쥔 자세 엄지 굽힘을 0.06 rad 보다 펴진 쪽으로 기록 안 함
                  "-p", "thumb_open_floor_rad:=0.06")


def _live(topic: str) -> str:
    """물체 pose 토픽 → 그 물체의 live 추적 토픽(snapshot 노드 · object_pose_node 규약: …/pose 옆 …/live_pose)."""
    if not topic.endswith("/pose"):
        raise ValueError(f"{topic!r} 는 …/pose 가 아니다")
    return topic[: -len("/pose")] + "/live_pose"
REAL_CUP_ORIGIN_Z = 0.085                 # 원점 높이(바닥 위) — 컵 자세 확인 문구
#: 실기 로봇 PC 구성 — 로봇(팔 CAN · 손 EtherCAT · 머리)이 붙은 PC 와 카메라 · FP++ 를 돌리는 PC.
#  ★10.13 사용자: "5090 로봇 pc" · "인지 pc vision3090 이미 카메라 연결" — 실기 미션은 ROBOT_PC 하나로 만든다.
#  arm4090 로 돌아가면 ROBOT_PC = "arm4090" 으로 바꾸고 다시 생성한다(--check 가 커밋 파일과 맞는지 본다).
STATIONS = {
    # ★10.27 사용자 "2번 일부 진행"(vision + 정책 + 로봇 제어 모두 5090) — D435i 를 5090 USB 3(Bus 02)에 꽂았다.
    #   카메라 · FP++ · 홀더 노드를 같은 PC 에서 돈다(arm4090 과 같은 local 경로).
    #   머리 · 외부 파라미터는 이 PC 에 붙은 RH56F1 머리(FT763P8T)의 것을 쓴다(10.27 수정): head_home_rh56f1.yaml ·
    #   global_camera_extrinsics_arm4090.yaml(10.27 이 머리 pan 2033 · tilt 2864 틱에서 잼). KITECH 머리 파일을 쓰면 머리가 tilt
    #   약 92° 다른 자세로 가고, 홀더 노드가 그 값으로 홀더 파일을 덮어쓴다.
    #   FP++ 이미지는 5090 용(sm_120) 재빌드 perception-plus-plus:humble-cup-sm120 을 humble-cup 으로 tag 해서 쓴다.
    #   남은 것(sudo): ros-humble-realsense2-camera 설치.
    "5090": {
        "perception_host": "local",
        "extrinsics": "config/global_camera_extrinsics_arm4090.yaml",
        "head_config": "config/head_home_rh56f1.yaml",
        "head_tool": "head_home",
        "pose_dest": "",                             # 같은 PC — 런처가 127.0.0.1 로 받는다
        "holder_poses": "config/cup_holder_poses_5090_v1.yaml",
    },
    # 10.13~10.26: 로봇 5090 + 인지 vision-3090(원격, 자세 UDP). 돌아가면 ROBOT_PC = "5090_vision3090".
    "5090_vision3090": {
        "perception_host": "vision-3090",            # 카메라 D435 · FP++ 컨테이너는 vision-3090, 자세만 UDP 로 이 PC 에
        # 5090 머리 카메라 hand-eye(목 pan 0 · tilt −20 정적 스냅샷, b100d85 · head_v1 v2) — head_pose 블록이 없어
        # head_pose_check 대신 head_home.py 가 그 자세로 맞추고 검증한다(DG-5F 5090 미션과 같은 방법)
        "extrinsics": "config/global_camera_extrinsics.yaml",
        "head_config": "config/head_home.yaml",
        "head_tool": "head_home",
        # ★10.13 FP++ 자세 UDP 는 이더넷 허브로(5090 eno1 169.254.186.1 ↔ vision-3090 eno2 169.254.186.100, 왕복 0.4 ms).
        #   ssh 는 tailscale(wifi, 왕복 3~130 ms)이라 받을 주소를 $SSH_CLIENT 로 정하면 자세가 wifi 로 온다.
        #   이 주소는 NetworkManager 프로필 hand-right(eno1)가 준다 — 그 프로필을 끄면 자세가 안 온다.
        "pose_dest": "169.254.186.1",
        # ★10.13 사용자: 홀더를 이 자리에 다시 놓고 새로 잰다("5090holder_v1"). 홀더 자세 파일은 .gitignore(PC · 배치마다 다르다)
        "holder_poses": "config/cup_holder_poses_5090_v1.yaml",
    },
    "arm4090": {
        "perception_host": "local",                  # 10.01 arm4090 은 카메라 · FP++ 를 같은 PC 에서
        "extrinsics": "config/global_camera_extrinsics_arm4090.yaml",   # 테이블 CAD 캘리브(10.01 기본 방법)
        "head_config": "config/head_home_rh56f1.yaml",
        "head_tool": "head_pose_check",
        "pose_dest": "",                             # 같은 PC — 런처가 127.0.0.1 로 받는다
        "holder_poses": "config/cup_holder_poses_arm4090.yaml",
    },
}
ROBOT_PC = "5090"
STATION = STATIONS[ROBOT_PC]
PERCEPTION_HOST = STATION["perception_host"]
REMOTE_PERCEPTION = PERCEPTION_HOST != "local"
CAMERA_EXTRINSICS = STATION["extrinsics"]
HOLDER_POSES = STATION["holder_poses"]


def _cups() -> list[str]:
    """실기 FP++ 물체 — 팔별(오른쪽 · 왼쪽) 다음 묶음의 나머지, 중복 없이. 묶음 컨테이너는 묶음 전부를 찍는다."""
    from object_registry import group_members, load_registry
    reg = load_registry()
    members = [n for g in REAL_GROUPS for n in group_members(reg, g)]
    side = [REAL_CUPS[s] for s in SIDES if s in REAL_CUPS]
    return list(dict.fromkeys([*side, *members]))


def _containers_txt() -> str:
    return " · ".join(f"fpp_{g}" for g in REAL_GROUPS)


def _side_cup(s: str) -> str:
    return REAL_CUPS.get(s, REAL_CUP)


def _policy_object(s: str) -> str:
    """그 팔 FP++ 물체 → 다물체 aglt(rg3) 정책 물체 이름(노드 파라미터 object). 묶음 → 이름 표는 콘솔 workspace.MODELS 하나."""
    from object_registry import load_registry
    sys.path.insert(0, str(REPO / "deploy" / "s2r_console"))
    from s2r_console.workspace import MODELS
    group = load_registry().get(_side_cup(s)).fpp.get("group")
    if group not in MODELS:
        raise SystemExit(f"{_side_cup(s)}: 묶음 {group!r} 의 정책 물체 이름이 없다 — s2r_console/workspace.py MODELS 에 더한다")
    return MODELS[group][1]


def _cups_txt() -> str:
    side = {v: k for k, v in REAL_CUPS.items()}
    return " · ".join(f"{c}{'=' + ('오른팔' if side[c] == 'right' else '왼팔') if c in side else ''}" for c in _cups())
#: 머리 설정(포트 · 게인 · 모터 id) — head_home · head_pose_check 가 같이 쓴다
HEAD_CONFIG = STATION["head_config"]
#: ★10.04 사용자 "fpp 진행 전에 자동으로 각도 확인하고 세팅을 제대로 맞춘 다음에 진행" — 외부 파라미터를 잰 머리 자세
#  (CAMERA_EXTRINSICS 의 head_pose)와 지금 자세를 비교하고, 실기 실행이면 그 자세로 맞춘다(pan 은 목표보다 18 틱 앞에서 멈춘다)
# 포트는 head_joint_publisher 하나가 소유한다. 인자 없는 head_move 는
# 드라이버가 읽은 station config 의 기준 자세로 이동하고 실측 정착을 확인한다.
HEAD_CHECK_ARGV = ["python3", "{repo}/scripts/head_move.py"]
HEAD_HOME_ARGV = HEAD_CHECK_ARGV
_head_config = yaml.safe_load((REPO / HEAD_CONFIG).read_text())
HEAD_DRIVER_ARGV = ["python3", "{repo}/scripts/nodes/head_joint_publisher.py", "--config", "{repo}/" + HEAD_CONFIG,
                    "--port", str(_head_config['port']), "--baud", str(_head_config['baud']),
                    "--ids", ','.join(str(_head_config['motors'][axis]['id']) for axis in ('pan', 'tilt'))]
RECALIB_HINT = ("scripts/calib/table_cad_extrinsics.py --init " + CAMERA_EXTRINSICS + " --write --head-config " + HEAD_CONFIG)
#: 원격 인지 PC 에서 도는 명령 — 홈에서 시작하는 ssh(사용자 이름이 PC 마다 달라 상대 경로, perception_launcher 와 같은 규약)
REMOTE_SIM2REAL = "rl_ws/sim2real"
#: 인지 PC 의 sim2real 전용 파이썬 — ★10.13 vision-3090 시스템 python3 는 ~/.local numpy 2.2 가 scipy 1.8 을 깨서
#  홀더 노드(scipy.optimize) · 영상 기록기(scipy.spatial)가 시작하자마자 죽는다. scripts/vision/venv_setup.sh 가 만든다
REMOTE_PY = ".venv/bin/python"
#: ★10.13 컵 좌표 다시 찍기 — FP++ 상태 토픽이 인지 PC 의 localhost DDS 에만 있어 상태 단계를 그 PC 에서(실기 첫 회 cups 막힘)
RESCAN_HOST = ["--status-host", PERCEPTION_HOST] if REMOTE_PERCEPTION else []


def _on_perception_pc(script: str) -> list:
    """인지 PC 에서 bash 한 줄을 돈다. 원격이면 ssh -tt(콘솔이 이 ssh 를 끝내면 저쪽 셸이 HUP 를 받는다)."""
    if not REMOTE_PERCEPTION:
        return ["bash", "-lc", script]
    return ["ssh", "-tt", "-o", "BatchMode=yes", PERCEPTION_HOST,
            f"cd {REMOTE_SIM2REAL} && source scripts/vision/common.sh && {script}"]


HEADER = {
    "real": f"# RH56F1 로봇({ROBOT_PC}, 인지 {PERCEPTION_HOST if REMOTE_PERCEPTION else '같은 PC'}) 실기 미션 — scripts/ops/make_rh56f1_missions.py 가 만든다. 손으로 고치지 말 것(--check 가 잡는다).\n",
    "fake": "# RH56F1 로봇 fake 미션(도메인 97) — scripts/ops/make_rh56f1_missions.py 가 실기 미션과 같은 정의에서 만든다.\n",
}


def _cmd(note: str, argv=None, **kw) -> dict:
    out = {"note": note}
    if argv is not None:
        out["argv"] = argv
    out.update(kw)
    return out


def _stages(kind: str) -> list[dict]:
    real = kind == "real"
    st = [
        {"id": "preflight", "group": "check", "lane": "rig",
         "title": "테스트 · RH56F1 제어 전용 계약 재생성(양팔 pour_fj 홈 · 편 손) (읽기 전용)", "artifacts": ["rh56f1_map"]},
        {"id": "head_home", "group": "motion", "lane": "rig", "needs": ["preflight"], "skippable": True, "touches_real": real,
         "title": ("머리를 카메라 캘리브 자세(외부 파라미터 head_pose)로 + I 게인 — 카메라 좌표가 이 자세에서만 맞다(머리가 조금 움직인다)"
                   if real else "fake — 머리 없음(실기 순서를 맞추려고 둔 자리)")},
        {"id": "head_control", "group": "motion", "lane": "rig", "skippable": True, "touches_real": real,
         "title": "목 pan · tilt 변경 — 승인한 보정 프로필과 자세로 이동 후 실측 정착 확인"},
        {"id": "cups", "group": "connect", "lane": "both", "needs": ["head_home"], "skippable": True,
         "touches_real": real,
         "title": (f"컵 자세({PERCEPTION_HOST if REMOTE_PERCEPTION else ROBOT_PC} FP++) — RealSense · FP++ 묶음 컨테이너({_containers_txt()}, 묶음마다 하나 · 색 · 모양으로 가르기 · 한 번 찍기: {_cups_txt()}) → /objects/<물체>/pose (base). GPU VRAM 컨테이너마다 약 2 GB" if real else
                   "fake 컵 두 개 — 학습 배치 중심(0.38, ∓0.16), 테이블 위에 선 채")},
        {"id": "cup_holders", "group": "connect", "lane": "both", "needs": ["head_home"], "skippable": True,
         "touches_real": real,
         "title": ("컵홀더 자세(마커 ID 0·1·2) — RealSense 컬러 → /objects/cup_holder_{0,1,2}/pose (base) · /cup_holders/status. "
                   "로봇은 움직이지 않는다. CPU 만" if real else "fake — 카메라 없음(실기 순서를 맞추려고 둔 자리)")},
        # ★10.13 사용자 실기: "다음부턴 개별로 하고, 둘중 하나라도 인지되게 해야함. 개별로 다시 할수 있게 해야해" —
        #   팔마다 그 팔의 컵 하나만 다시 인지하는 단계(선택 묶음, 차례에 끼지 않는다). cups 가 띄운 노드 위에서 돈다
        *({"id": f"cups_{s}", "group": "recognize", "lane": f"arm_{s}", "needs": ["cups"], "skippable": True,
           "touches_real": real,
           "title": (f"[{s}] 컵 하나만 다시 인지 — {REAL_CUPS[s]} (FP++ 컨테이너가 내려가 있으면 그것만 다시 띄우고, 이 컵의 "
                     f"묶음만 지금 카메라로 다시 찍는다)" if real else f"[{s}] fake — FP++ 없음(fake 컵은 cups 단계가 낸다)")}
          for s in SIDES if s in REAL_CUPS),
        {"id": "drivers", "group": "connect", "lane": "rig", "needs": ["preflight"],
         "title": ("모터 전원 → CAN ×2 → 팔 브링업 → CAN 응답 확인 (토크는 들어가지만 팔은 제자리)" if real else
                   "fake 플랜트 — 양팔 MockArm(rate) · controller_manager 스텁 (손은 손 창에서 따로)"),
         "needs_why": "점검 전에 띄우면 오배선 상태로 모터가 돈다", "touches_real": real},
    ]
    for s in SIDES:
        st += [
            {"id": f"hand_{s}", "group": "connect", "lane": f"arm_{s}", "needs": ["preflight"], "skippable": True,
             "touches_real": real, "artifacts": ["rh56f1_map"] + (["rh56f1_ports"] if real else []),
             "title": f"[{s}] RH56F1 손 드라이버({'포트 설정 rh56f1_ports.yaml · RS485 / CANFD' if real else 'fake 손'}) + 상태 노드(0.1° → rad)"},
            {"id": f"hand_check_{s}", "group": "prepare", "lane": f"arm_{s}", "needs": [f"hand_{s}"], "skippable": True,
             "title": f"[{s}] 손 점검 — 주기 · 슬롯 이름 · 레지스터 → rad · 한계 (읽기 전용)"},
            {"id": f"probe_{s}", "group": "diagnose", "lane": f"arm_{s}", "needs": [f"hand_check_{s}"], "skippable": True,
             "touches_real": real, "artifacts": ["rh56f1_map"],
             "title": f"[{s}] 한 축씩 방향 확인 — 한 손가락만 작게 움직였다 되돌린다(엄지 두 축은 변환표 verified 전)"},
            {"id": f"pd_load_{s}", "group": "prepare", "lane": f"arm_{s}", "needs": ["drivers", f"hand_check_{s}"],
             "skippable": True, "touches_real": real, "artifacts": ["contract", f"robot_{s}", "pd"],
             "needs_why": "pd 는 팔(브링업)과 그 팔의 손 상태가 둘 다 있어야 입력이 들어온다",
             "title": f"[{s}] pd 무발행 기동 (execute:=false) — 게인 · 입력 신선도 확인"},
            {"id": f"pd_arm_{s}", "group": "prepare", "lane": f"arm_{s}", "needs": [f"pd_load_{s}"], "skippable": True,
             "touches_real": real, "artifacts": ["contract", f"robot_{s}", "pd_exec"],
             "needs_why": "무발행 상태에서 게인 · 입력이 맞는지 본 뒤에만 발행 권한을 준다",
             "title": f"[{s}] pd 발행 모드 전환 — engage 는 하지 않는다(팔은 JTC 가 잡는다)"},
            {"id": f"home_{s}", "group": "motion", "lane": f"arm_{s}", "needs": [f"pd_arm_{s}"], "skippable": True,
             "touches_real": real, "artifacts": ["contract", f"path_{s}"],
             "title": f"[{s}] (한 번 승인) engage → 차렷 손(주먹) → 저장 경로로 차렷 → 홈(rh_aglt 시작 자세) → 정착 → 손 초기 자세"},
            {"id": f"return_{s}", "group": "finish", "lane": f"arm_{s}", "needs": [f"home_{s}"], "skippable": True,
             "touches_real": real, "artifacts": ["contract", f"path_{s}"], "undoes": [f"home_{s}"],
             "title": f"[{s}] 홈 → 차렷 — 홈 정착 → 차렷 손(주먹) → 저장 경로 역재생 → pd 해제(한 번 승인, 홈 근처에서만)"},
            {"id": f"release_{s}", "group": "finish", "lane": f"arm_{s}", "needs": [f"pd_arm_{s}"], "skippable": True,
             "touches_real": real, "undoes": [f"pd_arm_{s}", f"home_{s}"],
             "title": f"[{s}] pd 해제(역블렌드 → JTC) → 이 팔의 pd 정지"},
        ]
    st.append({"id": "policy_pourfj", "group": "policy", "lane": "both", "skippable": True, "touches_real": real,
               **({"blocked": "tilt 정책은 보류(10.07 사용자: 성공 전 정책은 직접 확인 뒤 · 10.17 기본 궤적은 에피소드 tilt_base_*) "
                             "— 실기 인지는 10.08 부터 색으로 두 컵"
                             f"({_cups_txt()})을 낸다"} if real else {}),
               "needs": ["home_right", "home_left", "cups", "hand_check_right", "hand_check_left"],
               "needs_why": "정책은 양팔이 pour_fj 시작 자세 · 손이 편 채 · 두 컵이 선 채로만 출발해 봤다. ★09.29 홈이 rh_aglt 시작 자세로 "
                            "바뀌어 pour_fj 시작 자세와 다르다 — 정책 노드가 start 를 거부한다(0.15 rad). 홈 → pour_fj 시작 경로가 필요",
               "artifacts": ["pourfj_both", "robot_bi", "contract"],
               # 10.17 사용자 결정 3: 화면 · 에피소드 · 보고서는 tilt(코드 모듈 pour_fj 이름은 그대로)
               "title": "[양팔] tilt 정책(pour_fj, 첫 화면에서 고른 것) — 정책 노드 → reset → start → 관찰 → stop (시간이 되면 스스로 끝난다)"})
    for s in SIDES:
        st.append({"id": f"policy_aglt_{s}", "group": "policy", "lane": f"arm_{s}", "skippable": True, "touches_real": real,
                   "needs": [f"home_{s}", "cups", f"hand_check_{s}"],
                   "needs_why": "정책은 팔이 rh_aglt 시작 자세(= 홈) · 손이 편 채 · 컵이 학습 배치에 선 채로만 출발해 봤다",
                   "artifacts": [f"aglt_{s}", f"robot_{s}", "contract"],
                   "title": f"[{s}] rh_aglt 정책(첫 화면에서 고른 것) — 컵에 접근 · 쥐기 · 들기 · 목표(컵 위 14 cm, "
                            "/policy_control/<팔>/goal 로 바꿀 수 있다)로 이송. 정책 노드 → reset → start → 관찰 → stop"})
    for s in SIDES:
        st.append({"id": f"policy_place_{s}", "group": "policy", "lane": f"arm_{s}", "skippable": True, "touches_real": real,
                   "needs": [f"policy_aglt_{s}", "cup_holders"],
                   "needs_why": "놓기 정책은 aglt 가 컵을 쥐고 인계 자리(0.25, ∓0.12, 0.41)에 멈춘 상태에서만 출발해 봤다 — "
                                "팔 · 손 목표는 pd 가 붙잡은 aglt 마지막 joint_target, 목표는 홀더 자세",
                   "artifacts": [f"place_{s}", f"robot_{s}", "contract"],
                   "title": f"[{s}] rh_place 정책 — 쥔 컵을 컵홀더 자리에 내려놓고 손을 편 뒤 팔을 시작 관절로(놓음 5 스텝 → "
                            "스크립트 45 스텝 → 스스로 끝). 정책 노드 → reset → start → 관찰"})
    for s in SIDES:
        st.append({"id": f"rehome_{s}", "group": "policy", "lane": f"arm_{s}", "skippable": True, "touches_real": real,
                   "needs": [f"pd_arm_{s}"],
                   "needs_why": "pd 가 떠 있어야 경로를 재생한다(IDLE 이면 지금 자리에서 engage)",
                   "artifacts": ["contract", f"robot_{s}"],
                   "title": f"[{s}] 정책 시작 자세(홈)로 되돌아오기 — 손 먼저 펴기 → 실측에서 경로 계획(직선이 막히면 RRT) → 재생 → 정착. "
                            "정책이 멈춘 자리에서 다시 정책 · return · 에피소드로(10.04 사용자: DG-5F 처럼 rehome)"})
    # ★10.19 사용자: "tilt_now 는 틸트 초기 자세(세팅 자세)로 가라고 하는 것(rehome 처럼)" — 상황판 control 탭 버튼(actions.py).
    #   한 팔씩(오른팔 → 왼팔) 계획 · 검사 → 재생. 손은 건드리지 않는다. pd 가 IDLE 이면 그 자리에서 engage(rehome 과 같다).
    st.append({"id": "tilt_now", "group": "policy", "lane": "both", "skippable": True, "touches_real": real,
               "needs": ["pd_arm_right", "pd_arm_left"],
               "needs_why": "pd 가 떠 있어야 경로를 재생한다(IDLE 이면 지금 자리에서 engage)",
               "title": "tilt 시작 자세(세팅 자세)로 — 두 팔 한 팔씩(오른팔 → 왼팔): 실측에서 계획 · 검사(걸리면 그 팔부터 움직이지 "
                        "않음) → 재생. 손은 그대로(rehome 처럼)"})
    # ★10.19 사용자: "내가 미리 세팅 자세를 만들어 둘 테니 그 상태에서 바로 틸팅" — 컵을 쥔 채라 engage 하지 않는다(TRACKING 이어야).
    st.append({"id": "tilt_pour", "group": "policy", "lane": "both", "skippable": True, "touches_real": real,
               "needs": ["pd_arm_right", "pd_arm_left"],
               "needs_why": "두 팔 pd 가 지금 자세를 붙들고 있어야(TRACKING) 재생한다 — 컵을 쥔 채라 여기서 engage 하지 않는다",
               "title": "tilt 붓기 — 지금 두 팔 자세(세팅 자세, 컵을 쥔 채)에서 바로: 실측 → D4 기본 궤적 계획 · 검사(걸리면 움직이지 "
                        "않음) → 두 팔 재생 → 시작 자세 복귀"})
    for name in EPISODES:
        ep = _episode(name)
        sides = sorted(_episode_sides(ep))
        perceives = _perceives(ep)
        st.append({"id": f"episode_{name}", "group": "policy", "lane": f"arm_{sides[0]}", "skippable": True,
                   "touches_real": real,
                   # 10.12 홀더가 없는 에피소드(파지 전용)는 홀더 자세 단계를 기다리지 않는다
                   # ★10.17 배치 기록(snapshot)이 없는 에피소드(경로 B, 사람이 컵을 댄다)는 FP++ 컵 단계를 기다리지 않는다
                   "needs": [*(f"home_{s}" for s in sides), *(["cups"] if perceives else []),
                             *(["cup_holders"] if ep.holders else [])],
                   "needs_why": "에피소드는 홈(rh_aglt 시작 자세)" + (" · FP++ 컵" if perceives else "")
                                + (" · 고정 홀더 자세 파일" if ep.holders else "") + " 위에서 노드를 잇는다",
                   "title": f"[에피소드] {name} — 노드를 띄우고 상황판 에피소드 패널에서 [다음](구분 실행) · [연속 실행]으로 "
                            f"진행: {' → '.join(n.id for n in ep.nodes if n.type != 'terminal')}"})
    st.append({"id": "shutdown", "group": "finish", "lane": "rig", "touches_real": real,
               "title": "안전 종료 — 팔 받침 확인 → 정책 → 인지 → 목 → pd → 손 → 팔 (★목 · 팔 토크가 풀린다)"})
    if not real:
        for x in st:
            x.pop("touches_real", None)
    return st


def _run(kind: str) -> dict:
    real = kind == "real"
    fake_arg = [] if real else ["fake:=true"]
    run = {
        "preflight": [
            *([_cmd("CPU — 실시간 한도(EtherCAT 마스터 FIFO 80 · controller_manager 50) · 코어 배치(이 PC 의 코어를 읽어 노드가 "
                    "스스로 정한다). 한도가 안 열렸으면 MISS — 운영자가 sudo bash scripts/setup/rt_setup.sh 한 번 → 재부팅(10.03)",
                    ["python3", "{repo}/scripts/setup/check_host.py", "--robot", "rh56f1", "--only", "cpu"])] if real else []),
            _cmd("RH56F1 변환표 · 백엔드 · 상태 노드 · 계약 · 미션 테스트(수 초)",
                 ["python3", "-m", "pytest", "-q", "-m", "not gpu", "-p", "no:cacheprovider",
                  "{repo}/tests/policy_control/test_pc_rh56f1.py", "{repo}/tests/s2r_console/test_rh56f1_mission.py"]),
            _cmd("제어 전용 계약 재생성 — 자산 openarm_rh56f1_bi_rl, 홈 = hdgp rh_aglt 시작 자세(양팔 거울, config/homes/rh56f1_aglt.yaml), "
                 "손 = 편 손. 홈이 바뀌면 저장 경로의 계약 해시가 어긋나 home 단계가 멈춘다 — 경로를 다시 계획할 것",
                 ["python3", f"{PC}/tools/build_deploy_contract.py", "--asset", "openarm_rh56f1_bi_rl",
                  "--home", "arms:config/homes/rh56f1_aglt.yaml", "--out", "{artifact:contract}"]),
        ],
        "drivers": ([
            _cmd("★모터 전원(양팔) ON — 물리 스위치", ["bash", "-lc", "true"], manual=True),
            # 10.15 사용자: "can ip up 을 sudo 로 맨날 내가 했다 — 자동으로". 이미 FD 1M/5M 로 올라와 있으면 건드리지 않는다.
            # 비밀번호 없는 허용은 scripts/setup/sudoers-s2r-can(운영자가 한 번 설치) — 없으면 이 줄이 실패하고 설치 명령을 낸다.
            _cmd(f"CAN 자동 설정(우 can0 · 좌 can1, FD 1M/5M) — 꺼져 있거나 설정이 다르거나 BUS-OFF 면 다시 올린다. "
                 f"{ROBOT_PC} 의 CAN 이름이 다르면 scripts/setup/can_up.py 와 sudoers 를 같이 고칠 것",
                 ["python3", "{repo}/scripts/setup/can_up.py", "can0", "can1", "--execute"]),
            _cmd("팔 브링업(robot_control) — 콘솔이 띄우고 감독한다",
                 ["ros2", "launch", "openarm_bringup", "openarm.bimanual.launch.py", "use_fake_hardware:=false",
                  "right_can_interface:=can0", "left_can_interface:=can1", "use_rviz:=false"], background=True),
            _cmd("★모터 응답 확인 — can0 · can1 수신 패킷이 늘어나는가", ["python3", "{repo}/scripts/setup/check_can_rx.py", "can0", "can1"]),
        ] if real else [
            _cmd("fake 플랜트 — 양팔 MockArm(rate), 손 없음(손 창이 fake 손을 띄운다)",
                 ["ros2", "launch", f"{PC}/launch/fake_plant.launch.py", "side:=both", "robot:={artifact:robot_bi}",
                  "contract:={artifact:contract}", "hands:=none", "plant_model:=rate",
                  # 차렷(0)에서 시작하면 j4 가 하한 0 에 붙어 pd 가 '한계 밖 목표'로 HOLD 한다(09.29 fake) — j4 만 0.02 안쪽
                  # (저장 경로 시작점 검사 허용 0.05 안)
                  "arm_start:=right=0,0,0,0.02,0,0,0;left=0,0,0,0.02,0,0,0"], background=True),
        ]),
        "shutdown": [
            _cmd("★양팔을 받침 위 · 안전 자세에 두었는가. 두 팔 모두 pd 해제를 끝냈는가. 다음 스텝부터 토크가 풀린다",
                 ["bash", "-lc", "true"], manual=True),
            # 실기 aglt 단계는 맨 앞에 기록(bag · CPU) 두 줄이 붙어 정책 노드가 #3, CPU 기록이 #2(10.08)
            _cmd("남은 정책 노드 정지", stop=["policy_pourfj#1"] + [f"policy_aglt_{s}#{AGLT_NODE_IDX[real]}" for s in SIDES]
                 + ([f"policy_aglt_{s}#2" for s in SIDES] if real else [])),
            # 정책 · 에피소드 단계가 도중에 실패하면 bag 기록기가 남는다(setsid) — 기록 중이 아니면 그렇다고만 찍는다(10.08)
            *([_cmd("남은 bag 기록 마무리(rh56f1_record.sh stop)", ["bash", "{repo}/deploy/policy_control/tools/rh56f1_record.sh", "stop"])]
              if real else []),
            *([_cmd("카메라 · FP++ 내리기 — 런처가 없으면 이 PC 에서 직접 내린다",
                    ["python3", "{repo}/scripts/ops/perception_ctl.py", "stop", "--camera", "--host", PERCEPTION_HOST, "--wait", "60"]),
               _cmd("인지 런처 · 자세 수신기 · 물체 자세 노드 · 컵홀더 노드 정지",
                    stop=["cups#1", "cups#2", "cups#3", "cup_holders#3"])] if real else []),
            *([_cmd("목 드라이버 정지 — 단독 소유한 시리얼 포트를 닫는다", stop=["head_home#0"]),
               _cmd("목 토크 해제 — 받침 확인, pan · tilt 가 중력으로 처질 수 있다",
                    ["python3", "{repo}/scripts/head_home.py", "--config", "{repo}/" + HEAD_CONFIG, "--release"],
                    execute_args=["--execute"])] if real else []),
            _cmd("남은 pd 정지 — 정책과 인지를 정리한 뒤",
                 stop=[f"{p}_{s}#{i}" for s in SIDES for p, i in (("pd_load", 0), ("pd_arm", 1))]),
            _cmd("손 상태 노드 · 손 드라이버 정지 — 손가락은 마지막 자세에서 멈춘다(벤더 펌웨어가 잡는다)",
                 stop=[f"hand_{s}#{i}" for s in SIDES for i in ((2, 1) if real else (1, 0))]),
            _cmd("★팔 브링업 정지 — 모든 팔 모터가 꺼진다(받침으로 내려앉는다)" if real else "fake 플랜트 정지",
                 stop=["drivers#2" if real else "drivers#0"]),
        ],
    }
    run["cup_holders"] = [
        _cmd("★홀더 세 개가 상판 위에 서 있고 -x 면 마커가 손 · 컵에 가리지 않는가(머리 자세는 다음 스텝이 캘리브 자세로 맞춘다)",
             ["bash", "-lc", "true"], manual=True),
        _cmd("머리 자세 확인 · 맞춤 — 카메라 외부 파라미터를 잰 자세(head_pose)와 다르면 그 자세로 맞춘다(머리가 조금 움직인다)", HEAD_CHECK_ARGV, execute_args=["--execute"]),
        _cmd("카메라(RealSense) 켜기 — 이미 떠 있으면 그대로(cups 단계와 같이 써도 된다)",
             ["bash", "{repo}/scripts/vision/camera_up.sh"]),
        _cmd("컵홀더 자세 노드 — ArUco → 직전 자세 추적(0.2 s) → 놓친 id 만 무늬 전체 탐색(첫 장 ~6 s). "
             f"x 공유 · 상판 z 고정 · 5장 중앙값, 세 홀더가 안정되면 {HOLDER_POSES} 을 갱신",
             ["python3", "{repo}/scripts/nodes/cup_holder_pose_node.py", "--write", "--out", "{repo}/" + HOLDER_POSES,
              "--extrinsics", "{repo}/" + CAMERA_EXTRINSICS], background=True),
        _cmd("★컵홀더 확인 — ok: true · 세 홀더 x ≈ 같은 값(0.38 부근) · y ≈ l +0.16 / r −0.06 / r 바깥 −0.28(간격 0.22 · 10.27 A1 배치) · stable ✓ 인가 "
             f"(어긋나면 {RECALIB_HINT} · 겹친 영상은 scripts/calib/cup_holder_pose.py --png)",
             ["bash", "-lc", "timeout 15 ros2 topic echo --once /cup_holders/status std_msgs/msg/String"], manual=True),
    ] if real and not REMOTE_PERCEPTION else [
        # ★10.13 원격 인지: 홀더 노드는 카메라 영상(localhost DDS)을 직접 읽어야 해 카메라 PC 에서 돌고, 고정 자세 파일만 이 PC 로 가져온다.
        #   놓기 에피소드는 그 파일(base 좌표)을 읽는다 — 홀더 토픽은 이 PC 로 넘어오지 않는다
        _cmd("★홀더 세 개가 상판 위에 서 있고 -x 면 마커가 손 · 컵에 가리지 않는가(머리 자세는 다음 스텝이 캘리브 자세로 맞춘다)",
             ["bash", "-lc", "true"], manual=True),
        _cmd("머리 자세 확인 · 맞춤 — 카메라 외부 파라미터를 잰 자세와 다르면 그 자세로 맞춘다(머리가 조금 움직인다)",
             HEAD_CHECK_ARGV, execute_args=["--execute"]),
        _cmd(f"카메라(RealSense, {PERCEPTION_HOST}) 켜기 — 이미 떠 있으면 그대로(cups 단계와 같이 써도 된다)",
             _on_perception_pc("bash scripts/vision/camera_up.sh")),
        _cmd(f"컵홀더 자세 노드({PERCEPTION_HOST} 에서, 영상을 그 PC 안에서 읽는다) — ArUco → 직전 자세 추적 → 놓친 id 만 무늬 전체 "
             f"탐색(첫 장 ~6 s). x 공유 · 상판 z 고정 · 5장 중앙값, 세 홀더가 안정되면 저 PC 의 {HOLDER_POSES} 을 갱신. "
             f"파이썬은 저 PC 의 {REMOTE_PY}(없으면 scripts/vision/venv_setup.sh)",
             _on_perception_pc(f"{REMOTE_PY} scripts/nodes/cup_holder_pose_node.py --write --out {HOLDER_POSES} "
                               f"--extrinsics {CAMERA_EXTRINSICS} > /tmp/perception/cup_holders.log 2>&1 & P=$!; "
                               "trap 'kill -TERM $P; wait $P' HUP TERM INT; wait $P"), background=True),
        _cmd("★컵홀더 확인 — ok: true · 세 홀더 x ≈ 같은 값(0.38 부근) · y ≈ l +0.16 / r −0.06 / r 바깥 −0.28(간격 0.22 · 10.27 A1 배치) · stable ✓ 인가 "
             f"(어긋나면 {RECALIB_HINT})",
             _on_perception_pc("timeout 15 ros2 topic echo --once /cup_holders/status std_msgs/msg/String"), manual=True),
        _cmd(f"홀더 자세 파일을 이 PC 로 — {PERCEPTION_HOST}:{HOLDER_POSES} → {HOLDER_POSES}(놓기 에피소드가 읽는다). "
             "파일이 아직 없으면 세 홀더가 안정되지 않은 것이다 — 확인 스텝부터 다시",
             ["scp", "-o", "BatchMode=yes", f"{PERCEPTION_HOST}:{REMOTE_SIM2REAL}/{HOLDER_POSES}", "{repo}/" + HOLDER_POSES]),
    ] if real else [_cmd("fake — 카메라 없음", ["bash", "-lc", "true"])]
    run["head_home"] = [
        _cmd("목 드라이버 — 포트 단독 소유 · 실측 관절 발행 · 이동 요청 처리",
             HEAD_DRIVER_ARGV, background=True),
        # ★10.04: head_home.py 는 설정 목표(pan 2015)로 검증하는데 이 머리 pan 은 목표보다 18 틱 앞에서 멈춰 늘 ✗(19:35 실기 단계
        #   실패). 카메라가 맞는 자세는 그 목표가 아니라 캘리브 때 멈춘 자리(외부 파라미터 head_pose) — 거기로 맞추고 거기로 검증한다.
        _cmd("머리를 카메라 캘리브 자세로 — 토크 · 모드 · I 게인(RAM, 전원을 끄면 사라진다)이 다르면 먼저 적용하고, 외부 파라미터 "
             "head_pose 로 목표를 고쳐 가며 맞춘 뒤 ±4 틱으로 검증(머리가 조금 움직인다)",
             HEAD_HOME_ARGV, execute_args=["--execute"]),
    ] if real else [_cmd("fake — 머리 없음", ["bash", "-lc", "true"])]
    run["cups"] = [
        _cmd(f"★컵이 테이블에 똑바로 서 있고 손이 가리지 않는가 · {PERCEPTION_HOST if REMOTE_PERCEPTION else ROBOT_PC} GPU 여유가 있는가"
             "(nvidia-smi — 학습이 돌면 VRAM 이 모자랄 수 있다). 머리 자세는 FP++ 를 켜기 전에 이 단계가 캘리브 자세로 맞춘다",
             (["ssh", "-o", "BatchMode=yes", PERCEPTION_HOST] if REMOTE_PERCEPTION else ["bash", "-lc"])
             + ["nvidia-smi --query-compute-apps=pid,used_memory --format=csv; "
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"], manual=True),
        _cmd((f"인지 런처(이 PC) — {PERCEPTION_HOST} 의 카메라 · FP++ 컨테이너를 ssh 로 켜고 끈다. 스스로는 아무것도 켜지 않는다"
              if REMOTE_PERCEPTION else
              "인지 런처(이 PC) — 카메라 · FP++ 컨테이너를 같은 PC 의 스크립트로 켜고 끈다. 스스로는 아무것도 켜지 않는다"),
             ["python3", "{repo}/scripts/nodes/perception_launcher_node.py", "--host", PERCEPTION_HOST,
              *(["--pose-dest", STATION["pose_dest"]] if STATION["pose_dest"] else []),
              "--camera-extrinsics", "{repo}/" + CAMERA_EXTRINSICS], background=True),
        _cmd(("FP++ 자세 수신기 — 영상 · FP++ 는 " + (f"{PERCEPTION_HOST} 안" if REMOTE_PERCEPTION else "localhost 전용 DDS")
              + f"에서 돌고 자세만 UDP({'저 PC → 이 PC ' + (STATION['pose_dest'] or '(ssh 주소)') if REMOTE_PERCEPTION else '127.0.0.1'})로 넘어온다"),
             ["python3", "{repo}/scripts/nodes/fpp_pose_rx.py"], background=True),
        _cmd(f"물체 자세 → base — {ROBOT_PC} 카메라 외부 파라미터({CAMERA_EXTRINSICS}, + depth z 보정). /objects/<물체>/pose(레지스트리의 모든 물체)",
             ["python3", "{repo}/scripts/nodes/object_pose_node.py",       # 레지스트리 전체 — 묶음을 바꿔도(activate) 다시 띄우지 않는다
              "--camera-extrinsics", "{repo}/" + CAMERA_EXTRINSICS,
              "--head-joint-topic", "/head/joint_states"], background=True),
        _cmd("FP++ 전 머리 자세 확인 · 맞춤 — 카메라 외부 파라미터를 잰 자세(head_pose)와 다르면 그 자세로 맞춘다(머리가 조금 움직인다)", HEAD_CHECK_ARGV, execute_args=["--execute"]),
        _cmd(f"카메라 + FP++ 켜기 — 묶음 컨테이너({_containers_txt()})가 {_cups_txt()} 를 차례로 한 번 찍는다. 이미 떠 있으면 그대로. "
             "최대 150 s. ★10.13 묶음 하나가 못 뜨면 경고만(나머지는 계속), 하나도 못 띄우면 실패",
             ["python3", "{repo}/scripts/ops/perception_ctl.py", "start", *_cups(), "--viewer", "--wait", "150"]),
        _cmd("컵 좌표 추출 — 처음엔 FP++ 가 켜지며 찍는 회차를 기다리고, 이 단계를 다시 실행하면(컵을 옮긴 뒤 '↶ 여기서 다시') "
             "컨테이너를 끄지 않고 지금 카메라로 다시 찍는다(10.08 사용자). 컵마다 base 좌표 · 판정을 찍는다. ★10.13 하나라도 찾으면 통과 · 못 찾은 컵은 이유와 함께 경고(그 컵은 개별 인지 단계로 다시)",
             ["python3", "{repo}/scripts/ops/fpp_rescan.py", *(REAL_CUPS[s] for s in SIDES if s in REAL_CUPS), "--wait", "120",
              *RESCAN_HOST]),
        _cmd(f"★컵 자세 확인(바로 위 추출 결과) — 테이블 위 컵 원점 z ≈ {0.205 + REAL_CUP_ORIGIN_Z:.3f}(상판 0.205 + 원점 {REAL_CUP_ORIGIN_Z}, ±8 mm) · 기울기 < 3° · "
             f"x 0.1~0.4 · |y| 0.1~0.3 · 오른팔 {_side_cup('right')}(y < 0) · 왼팔 {_side_cup('left')}(y > 0) 인가"
             f"(카메라를 건드렸으면 {RECALIB_HINT})",
             ["bash", "-lc", "true"], manual=True),
    ] if real else [
        _cmd(f"fake 컵({role}) — {t}", ["python3", "{repo}/scripts/fakes/fake_cup_pose_pub.py", "--x", "0.38",
                                         "--y", y, "--z", "0.264865", "--topic", t],
             background=True)
        for role, y in (("src", "-0.16"), ("rcv", "0.16"))
        for t in (f"/objects/cup_{role}/pose", _live(f"/objects/cup_{role}/pose"))]   # live = aglt 정책 입력(10.12 A안)
    for s in SIDES:
        if s not in REAL_CUPS:
            continue
        # ★10.13 개별 인지 — perception_ctl start 는 넘긴 물체 밖의 컨테이너를 내리므로 컵 전부를 넘긴다(떠 있는 것은 그대로,
        #   내려간 것만 다시 띄운다). 다시 찍기는 이 컵 하나만 — 이 컵을 못 찾으면 이 단계만 실패한다
        run[f"cups_{s}"] = [
            _cmd(f"★[{s}] {REAL_CUPS[s]} 가 테이블에 똑바로 서 있고 손이 가리지 않는가", ["bash", "-lc", "true"], manual=True),
            _cmd(f"[{s}] FP++ 컨테이너 확인 — 내려간 묶음만 다시 띄운다({_containers_txt()}). 최대 150 s",
                 ["python3", "{repo}/scripts/ops/perception_ctl.py", "start", *_cups(), "--wait", "150"]),
            _cmd(f"[{s}] {REAL_CUPS[s]} 만 다시 찍기 — base 좌표 · 판정을 찍고, 못 찾으면 이유와 함께 실패",
                 ["python3", "{repo}/scripts/ops/fpp_rescan.py", REAL_CUPS[s], "--wait", "120", *RESCAN_HOST]),
        ] if real else [_cmd(f"[{s}] fake — FP++ 없음", ["bash", "-lc", "true"])]
    run["policy_pourfj"] = [
        _cmd("★두 컵이 학습 배치(로봇 앞 x ≈ 0.38, y ≈ ∓0.16)에 서 있고 콘솔에 두 컵 자세가 들어오는가. 양팔 · 양손 주변이 비어 있는가. "
             "정책은 팔을 스스로 움직이고 손가락을 쥔다 — 빈 컵만" if real else "fake — 확인만", ["bash", "-lc", "true"], manual=True),
        _cmd("양팔 정책 노드(pour_fj_node, LSTM · CPU) — start 전에는 아무것도 보내지 않는다. 처음 30 스텝은 팔을 시작 자세에 둔다(학습 hold)",
             ["{repo}/.venv/bin/python", f"{PC}/policy_control/pour_fj_node.py", "--ros-args",
              "-p", "contract:={artifact:pourfj_both}", "-p", "robot:={artifact:robot_bi}", "-p", "device:=cpu",
              "-p", "max_episode_s:={policy:pourfj_both.max_episode_s}"], background=True),
        _cmd("episode reset — 두 컵 · 양팔 측정이 모두 있어야 받는다", ["python3", f"{PC}/tools/trigger.py", "episode/reset"],
             execute_args=["--execute"]),
        _cmd("★episode start — 양팔이 시작 자세 0.15 rad 안 · 두 컵이 서 있어야 받는다", ["python3", f"{PC}/tools/trigger.py", "episode/start"],
             execute_args=["--execute"]),
        _cmd("★관찰 — 이상하면 정지 바의 '에피소드 정지'", ["bash", "-lc", "true"], manual=True),
        _cmd("episode stop — pd 가 그 자세 · 손 쥠을 붙잡는다", ["python3", f"{PC}/tools/trigger.py", "episode/stop"],
             execute_args=["--execute"]),
        _cmd("정책 노드 정지", stop=["policy_pourfj#1"]),
    ]
    for s, cup in (("right", "src"), ("left", "rcv")):
        cup_t = f"/objects/{_side_cup(s)}/pose" if real else f"/objects/cup_{cup}/pose"
        rec = _record_start(f"aglt_{s}", [s], [cup_t, _live(cup_t)], [_side_cup(s)]) if real else []
        assert len(rec) + 1 == AGLT_NODE_IDX[real]
        run[f"policy_aglt_{s}"] = [
            _cmd(f"★[{s}] 컵이 학습 배치(로봇 앞 x ≈ 0.25, y ≈ {'−' if s == 'right' else '+'}0.20 ± 0.1)에 서 있고 콘솔에 컵 자세"
                 f"(/objects/{_side_cup(s)}/pose)가 들어오는가. 이 팔 · 손 주변과 컵 위 20 cm 가 비어 있는가. 정책은 팔을 스스로 움직이고 "
                 "컵을 쥐어 든다 — 빈 컵만" if real else "fake — 확인만", ["bash", "-lc", "true"], manual=True),
            *rec,
            _cmd(f"[{s}] rh_aglt 정책 노드(LSTM · CPU) — start 전에는 아무것도 보내지 않는다. 처음 10 스텝은 팔을 시작 자세 · 손을 편 채(학습 hold)",
                 ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_aglt_node.py", "--ros-args",
                  # 팔마다 이름 · 에피소드를 가른다 — 양팔 정책을 한 세션에서 동시에 띄워도 서로의 reset · stop 이 섞이지 않는다(09.30)
                  "-r", f"__node:=rh_aglt_node_{s}", "-p", f"ns:={s}",
                  "-p", f"contract:={{artifact:aglt_{s}}}", "-p", f"robot:={{artifact:robot_{s}}}", "-p", "device:=cpu",
                  "-p", f"cup_topic:={_live(cup_t)}",
                  "-p", f"object:={_policy_object(s)}",   # 다물체 정책 — 물체를 모르면 reset 이 거부한다
                  *AGLT_LIVE_ARGS,                       # 10.12 A안 — 쥘 때까지 live(끊기면 마지막 값), 쥔 뒤 프레임 시각 FK
                  "-p", f"max_episode_s:={{policy:aglt_{s}.max_episode_s}}"],
                 background=True),
            _cmd(f"[{s}] episode reset — 컵 · 팔 측정이 있어야 받는다. 목표 = 지금 컵 + 계약 goal_offset",
                 ["python3", f"{PC}/tools/trigger.py", "episode/reset", "--episode-ns", s], execute_args=["--execute"]),
            _cmd(f"★[{s}] episode start — 팔이 시작 자세 0.15 rad 안 · 컵이 서 있어야 받는다",
                 ["python3", f"{PC}/tools/trigger.py", "episode/start", "--episode-ns", s], execute_args=["--execute"]),
            _cmd("★관찰 — 이상하면 정지 바의 '에피소드 정지'", ["bash", "-lc", "true"], manual=True),
            _cmd("episode stop — pd 가 그 자세 · 손 쥠을 붙잡는다", ["python3", f"{PC}/tools/trigger.py", "episode/stop", "--episode-ns", s],
                 execute_args=["--execute"]),
            _cmd("정책 노드 정지", stop=[f"policy_aglt_{s}#{AGLT_NODE_IDX[real]}"]),
            *(_record_stop(f"policy_aglt_{s}", 2, f"aglt_{s}") if real else []),
        ]
        npz = f"{{repo}}/logs/policy_control/rehome_{s}.npz"
        p = s[0]
        run[f"rehome_{s}"] = [
            _cmd(f"★[{s}] 손에 컵이 있으면 사람이 받아 든다(다음 스텝이 손을 편다) · 이 팔 · 손 주변과 경로(테이블 앞 · 몸통 옆)가 "
                 "비어 있는가. 정책이 멈춘 자리 → 정책 시작 자세(홈)로 간다(최대 0.1 rad/s, 약 20~40 s)" if real else "fake — 확인만",
                 ["bash", "-lc", "true"], manual=True),
            _cmd(f"[{s}] pd 가 IDLE 이면 지금 자리에서 engage — 이미 붙들고 있으면(정책 뒤 TRACKING) 건너뜀",
                 ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_engage", "--skip-engaged", "--hold-s", "2"],
                 execute_args=["--execute", "--approve", "pd_engage"]),
            _cmd(f"[{s}] 손을 편다(pd/hand_release — 팔은 제자리) — 쥔 컵을 놓는다",
                 ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_hand_release", "--service-timeout", "15"],
                 execute_args=["--execute", "--approve", "pd_hand_release"]),
            _cmd(f"[{s}] 지금 자세 → 정책 시작 자세 경로를 실측에서 계획 — 편 손 · 테이블 · 몸통 · 반대 팔(여유 2 cm), 직선이 막히면 RRT. "
                 "곧장 goto_home 은 손끝이 상판을 지날 수 있다(09.28)",
                 ["python3", f"{PC}/tools/plan_rehome.py", "--side", s, "--robot", "rh56f1", "--out", npz]),
            _cmd(f"[{s}] 경로 시작점 = 지금 자세인가(0.05 rad) · 경로가 지금 계약으로 만든 것인가",
                 ["python3", f"{PC}/tools/check_path_start.py", "--npz", npz, "--contract", "{artifact:contract}"]),
            _cmd(f"[{s}] 경로 재생 → 정책 시작 자세 — 끝나면 episode stop 으로 pd 가 그 자세를 붙든다",
                 ["python3", f"{PC}/tools/replay_to_pd.py", "--npz", npz,
                  "--joints", ",".join(f"{p}_aj_{i}" for i in range(1, 8)), "--rate-scale", "1.0"],
                 execute_args=["--execute"]),
            _cmd(f"[{s}] 정책 시작 자세에서 정착 — 이제 policy_aglt_{s} · return_{s} · 에피소드를 다시 돌릴 수 있다",
                 ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_goto_home", "--service-timeout", "45"],
                 execute_args=["--execute", "--approve", "pd_goto_home"]),
        ]
        hid = PLACE_HOLDER[s]
        run[f"policy_place_{s}"] = ([] if real else [
            _cmd(f"fake 홀더 {hid} — /objects/cup_holder_{hid}/pose (latched, 학습 배치 x 0.38)",
                 ["python3", "{repo}/scripts/fakes/fake_cup_pose_pub.py", "--latched", "--rate", "2", "--x", "0.38",
                  "--y", str(FAKE_HOLDER_Y[hid]), "--z", "0.235", "--topic", f"/objects/cup_holder_{hid}/pose"],
                 background=True)]) + [
            _cmd(f"★[{s}] 직전 aglt 에피소드가 컵을 쥔 채 stop 했는가(pd 가 그 자세 · 손을 붙잡고 있다). 인계 자리로 옮기려면 "
                 f"aglt 를 다시 start 하고 aglt_goal.py --side {s} --handoff --execute 로 목표를 준 뒤 도착하면 stop. "
                 f"홀더 {hid} 위 20 cm 가 비어 있는가 — 정책이 컵을 홀더에 내려놓고 손을 편다" if real else "fake — 확인만",
                 ["python3", f"{PC}/tools/aglt_goal.py", "--side", s, "--handoff"], manual=True),
            _cmd(f"[{s}] rh_place 정책 노드(LSTM · CPU) — start 전에는 아무것도 보내지 않는다. 컵은 reset 때 FP++ 한 장으로 "
                 "손바닥에 붙이고 그 뒤는 손바닥 FK 로만 본다(학습 attached)",
                 ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_place_node.py", "--ros-args",
                  "-r", f"__node:=rh_place_node_{s}", "-p", f"ns:={s}",
                  "-p", f"contract:={{artifact:place_{s}}}", "-p", f"robot:={{artifact:robot_{s}}}", "-p", "device:=cpu",
                  "-p", f"cup_topic:=/objects/{_side_cup(s) if real else 'cup_' + ('src' if s == 'right' else 'rcv')}/pose",
                  "-p", f"holder:={hid}", *([] if real else ["-p", "require_grasp:=false"])],
                 background=True),
            _cmd(f"[{s}] episode reset — 홀더 {hid} 자세 · pd 가 붙잡은 aglt 마지막 목표(팔 실측 0.15 rad 안) · 컵 자세가 있어야 받는다",
                 ["python3", f"{PC}/tools/trigger.py", "episode/reset", "--episode-ns", s], execute_args=["--execute"]),
            _cmd(f"★[{s}] episode start — 컵을 쥐고 있어야(엄지 AND 다른 손가락 촉각 > 1 N) 받는다",
                 ["python3", f"{PC}/tools/trigger.py", "episode/start", "--episode-ns", s], execute_args=["--execute"]),
            _cmd("★관찰 — 놓음(손끝 · 관절 힘이 비고 5 스텝) 뒤 손을 펴고 팔이 돌아오면 스스로 끝난다. 이상하면 정지 바의 '에피소드 정지'",
                 ["bash", "-lc", "true"], manual=True),
            _cmd("episode stop(이미 끝났으면 그대로) — pd 가 그 자세를 붙잡는다",
                 ["python3", f"{PC}/tools/trigger.py", "episode/stop", "--episode-ns", s], execute_args=["--execute"]),
            _cmd("정책 노드 정지", stop=[f"policy_place_{s}#{1 if real else 2}"]),
        ] + ([] if real else [_cmd("fake 홀더 정지", stop=[f"policy_place_{s}#0"])])
    for s in SIDES:
        hand = [_cmd(f"★[{s}] 손 EtherCAT 확인 — 손 전원 · 랜 케이블(손 하나 = NIC 하나, 두 손 모두 USB 3.0 랜(오른손 RTL8153 · 왼손 RTL8156, 10.09) — "
                     "deploy/policy_control/config/rh56f1_ports.yaml). 링크가 up 이고 마스터에 setcap 이 붙어 있는가",
                     ["bash", "-lc", "ip -br link; getcap {repo}/tools/ethercat/rh56f1_ecat_master; "
                                     "cat {repo}/deploy/policy_control/config/rh56f1_ports.yaml"], manual=True),
                _cmd(f"[{s}] RH56F1 EtherCAT 드라이버 — 1 kHz 마스터 + ROS 노드(벤더와 같은 토픽). OP 로 올라가지만 첫 각도 "
                     "명령 전에는 손이 제자리(목표 = 지금 각도 · ENABLE 0)",
                     ["python3", f"{PC}/tools/rh56f1_driver.py", "--side", s], background=True)] if real else [
                _cmd(f"[{s}] fake RH56F1 손 — 벤더 드라이버와 같은 토픽 · 단위(편 손에서 시작)",
                     ["python3", "{repo}/scripts/fakes/fake_rh56f1_hand.py", "--side", s], background=True)]
        hand.append(_cmd(f"[{s}] 상태 노드 — /hand_{s}/angle_actual(0.1°) → /hand_{s}/joint_states(rad) · 촉각 → tip_forces",
                         ["python3", f"{PC}/policy_control/rh56f1_state_node.py", "--side", s, "--map", "{artifact:rh56f1_map}"],
                         background=True))
        probe = [_cmd(f"★[{s}] 손 주변이 비어 있는가 — 한 손가락씩 작게(최대 0.5 rad) 움직였다 되돌린다. 어느 손가락이 어느 쪽으로 "
                      "움직이는지 눈으로 본다", ["bash", "-lc", "true"], manual=True)]
        for axis, by in PROBES:
            probe += [
                _cmd(f"[{s}] {axis} {by} rad → 되돌림", ["python3", f"{PC}/tools/rh56f1_axis_probe.py", "--side", s,
                                                        "--axis", axis, "--by", by, "--back", "--map", "{artifact:rh56f1_map}"],
                     execute_args=["--execute"]),
                _cmd(f"★[{s}] {axis} 가 맞는 손가락이 맞는 쪽으로 움직였는가(엄지면 변환표 verified 를 여기서 판단)",
                     ["bash", "-lc", "true"], manual=True),
            ]
        run.update({
            f"hand_{s}": hand,
            f"hand_check_{s}": [_cmd(f"[{s}] 손 점검(읽기만) — 드라이버 주기 ≥ 20 Hz · 슬롯 이름 · 변환 · 한계",
                                     ["python3", f"{PC}/tools/rh56f1_hand_check.py", "--side", s, "--map", "{artifact:rh56f1_map}"])],
            f"probe_{s}": probe,
            f"pd_load_{s}": [_cmd(f"[{s}] pd 무발행 기동 — status 의 입력 신선도를 본다",
                                  ["ros2", "launch", f"{PC}/launch/pd_controller.launch.py", "contract:={artifact:contract}",
                                   f"robot:={{artifact:robot_{s}}}", "pd_config:={artifact:pd}", f"sides:={s}",
                                   "execute:=false", *fake_arg], background=True)],
            f"pd_arm_{s}": [
                _cmd(f"[{s}] 무발행 pd 정지", stop=[f"pd_load_{s}#0"]),
                _cmd(f"[{s}] pd 발행 모드 기동 (execute:=true · stage:=full) — IDLE 로 뜬다. engage 전까지 팔은 JTC 가 잡는다",
                     ["ros2", "launch", f"{PC}/launch/pd_controller.launch.py", "contract:={artifact:contract}",
                      f"robot:={{artifact:robot_{s}}}", "pd_config:={artifact:pd_exec}", f"sides:={s}",
                      "execute:=true", "stage:=full", *fake_arg], background=True)],
            f"home_{s}": _home(s),
            f"return_{s}": _return(s),
            f"release_{s}": [
                _cmd(f"[{s}] pd release — 역블렌드 → 0 송출 → JTC 복귀",
                     ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_release"], execute_args=["--execute"]),
                _cmd(f"[{s}] 이 팔의 pd 정지 — IDLE 이라 토크는 JTC 가 잡는다", stop=[f"pd_arm_{s}#1"])],
        })
    run["tilt_now"] = [
        _cmd("★두 팔 주변과 경로(테이블 앞 · 몸통 앞 · 두 팔 사이)가 비었는가. 오른팔 → 왼팔 순서로 tilt 시작 자세(세팅 자세)로 간다 "
             "— 손은 그대로" if real else "fake — 확인만", ["bash", "-lc", "true"], manual=True),
        *(_cmd(f"[{s}] pd 가 IDLE 이면 지금 자리에서 engage — 이미 붙들고 있으면(TRACKING) 건너뜀",
               ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_engage", "--skip-engaged", "--hold-s", "2"],
               execute_args=["--execute", "--approve", "pd_engage"]) for s in ("right", "left")),
        _cmd("tilt 시작 자세로 — 팔마다 계획 · 검사(plan_carry, 걸리면 그 팔부터 움직이지 않음) → 재생(replay_to_pd) · "
             "기록 logs/tilt/<시각>_setting",
             ["python3", f"{PC}/tools/tilt_now.py"], execute_args=["--execute"]),
    ]
    run["tilt_pour"] = [
        _cmd("★두 팔이 세팅 자세(붓기 시작 자세)에서 컵을 쥐고 있는가 — 오른손 병(입구 위) · 왼손 shaker · 두 팔 주변과 붓는 길"
             "(테이블 위 · 몸통 앞)이 비었는가. 지금 자세에서 계획 · 검사하고 통과하면 재생한 뒤 시작 자세로 돌아온다" if real
             else "fake — 확인만", ["bash", "-lc", "true"], manual=True),
        _cmd("tilt 붓기 — 지금 자세에서 계획 · 검사(plan_tilt, 걸리면 움직이지 않음) → 두 팔 재생(replay_to_pd) · 기록 logs/tilt/<시각>_pour"
             + (" · bag tilt_pour" if real else ""),
             ["python3", f"{PC}/tools/tilt_pour.py", *([] if real else ["--no-record"])], execute_args=["--execute"]),
    ]
    for name in EPISODES:
        run[f"episode_{name}"] = _episode_run(name, real)
    # 에피소드 단계가 도중에 실패하면 CPU 기록기가 남는다 — shutdown 의 정지 목록에 더한다(10.08 리뷰)
    cpu = [f"episode_{name}#{i}" for name in EPISODES for i, c in enumerate(run[f"episode_{name}"])
           if "proc_cpu_record.py" in " ".join(map(str, c.get("argv") or ()))]
    if cpu:
        stop = next(c for c in run["shutdown"] if c.get("stop") and "policy_pourfj#1" in c["stop"])
        stop["stop"] = [*stop["stop"], *cpu]
    # 부분 연결 / 중간 실패도 정리한다. 명령 번호를 중복 관리하지 않고 생성된 배경 작업에서 모은다.
    policy_stop = next(c for c in run["shutdown"] if c.get("stop") and "policy_pourfj#1" in c["stop"])
    policy_stop["stop"] = list(dict.fromkeys([*policy_stop["stop"],
        *(f"{stage}#{i}" for stage, cmds in run.items() for i, c in enumerate(cmds)
          if c.get("background") and (stage.startswith("policy_") or stage.startswith("episode_")))]))
    perception_ids = [f"{stage}#{i}" for stage in ("cups", "cup_holders")
                      for i, c in enumerate(run.get(stage, [])) if c.get("background")]
    if real:
        perception_stop = next(c for c in run["shutdown"] if "cups#1" in c.get("stop", []))
        perception_stop["stop"] = list(dict.fromkeys([*perception_stop["stop"], *perception_ids]))
    elif perception_ids:
        pd_index = next(i for i, c in enumerate(run["shutdown"]) if "pd_load_right#0" in c.get("stop", []))
        run["shutdown"].insert(pd_index, _cmd("fake 인지 물체 발행 정지", stop=perception_ids))
    run["head_control"] = [
        _cmd("목 pan · tilt 변경 후 실측 정착 확인", ["python3", "{repo}/scripts/head_move.py"],
             execute_args=["--execute"])
    ] if real else [_cmd("fake — 목 없음(승인한 자세 설정만 보관)", ["bash", "-lc", "true"])]
    return run


def _episode_sides(ep) -> list[str]:
    from policy_control import episode_spec as S
    return S.episode_sides(ep)


def _perceives(ep) -> bool:
    """FP++ 배치 기록(snapshot)을 쓰는 에피소드인가 — 경로 B(사람이 컵을 댄다)는 아니다."""
    return any(n.type == "snapshot" for n in ep.nodes)


def _episode(name: str):
    sys.path.insert(0, str(REPO / "deploy" / "policy_control"))
    from policy_control import episode_spec as S
    return S.load(REPO / "config" / "episodes" / f"{name}.yaml")


def _place_attach(contract: Path) -> str:
    """rh_place 계약의 붙인 자세 출처(★10.22 newton_05 = bundle) — 못 읽으면 ""."""
    try:
        return str(json.loads(contract.read_text()).get("attach_source", "fpp"))
    except (OSError, ValueError):
        return ""


def _episode_run(name: str, real: bool) -> list:
    """에피소드 단계: (fake 컵 · 홀더) → 정책 노드들 → episode_runner_node → 상황판에서 진행(수동 확인) → 내린다."""
    from policy_control import policy_registry as R
    ep = _episode(name)
    entries = {e.id: e for e in R.scan(REPO / "deploy" / "policies", deep=False)}
    cmds, bg = [], []
    perceives = _perceives(ep)
    if not real and perceives:
        for oname, obj in ep.objects.items():
            side = next((ep.policies[j.role].side for n in ep.nodes for j in n.jobs if j.target_object == oname),
                        # ★10.19 1-B(정책 없음) — 그 물체로 다가가는 궤적(reach)의 팔
                        next((tr.sides[0] for tr in ep.trajectories.values() if tr.object == oname), "right"))
            # ★10.20 1-B: 궤적으로 다가가 쥐는 물체는 fake 손 '손 안 컵'(/fake/hand_<팔>/cup)이 켜지면 손바닥을 따라간다 — 들기 판정 시험
            follow = ["--follow-side", side] if any(tr.object == oname for tr in ep.trajectories.values()) else []
            for t in (obj["topic"], _live(obj["topic"])):           # live = aglt 정책 입력(10.12 A안)
                cmds.append(_cmd(f"fake 물체 — {t}(aglt 학습 배치 x 0.25)",
                                 ["python3", "{repo}/scripts/fakes/fake_cup_pose_pub.py", "--x", "0.25",
                                  "--y", "0.20" if side == "left" else "-0.20", "--z", "0.29", "--topic", t, *follow],
                                 background=True))
        # ★10.22 홀더 자세 파일이 있으면 fake 홀더도 그 자세 — 실행기도 같은 파일을 같은 토픽에 낸다(두 발행이 어긋나지 않게)
        from policy_control.episode_ros import load_holder_poses
        real_h = load_holder_poses(REPO / ep.holder_poses) if ep.holder_poses else {}
        for hid in sorted(set(ep.holders.values())):
            hx, hy, hz = ((float(real_h[hid][0][0]), float(real_h[hid][0][1]), float(real_h[hid][0][2])) if hid in real_h
                          else (0.38, FAKE_HOLDER_Y[hid], 0.235))
            cmds.append(_cmd(f"fake 홀더 {hid} — /objects/cup_holder_{hid}/pose (latched"
                             + (f", {ep.holder_poses} 자세)" if hid in real_h else ")"),
                             ["python3", "{repo}/scripts/fakes/fake_cup_pose_pub.py", "--latched", "--rate", "2", "--x", f"{hx:.4f}",
                              "--y", f"{hy:.4f}", "--z", f"{hz:.4f}", "--topic", f"/objects/cup_holder_{hid}/pose"],
                             background=True))
    first_obj = next(iter(ep.objects))
    sides = sorted(_episode_sides(ep))
    from policy_control.episode_ros import relay_topic
    # ★10.19 사용자 "좌우 동시 실행": 한 팔 에피소드는 그 팔 이름의 실행기(episode_runner_<팔>) — 서비스 · 상태 · 승인 · snapshot
    #   재발행이 팔마다 따로라 오른팔 · 왼팔 에피소드가 같이 돈다. 두 팔 에피소드(tilt 등)는 기본 실행기 하나.
    runner = f"episode_runner_{sides[0]}" if len(sides) == 1 else "episode_runner"
    left = "|".join([*(f"(rh_aglt_node|rh_place_node)_{s}" for s in sorted({b.side for b in ep.policies.values()})),
                     runner])
    cmds.append(_cmd("시작 검사 — 정책이 등록부에서 쓸 수 있고" + (f" 고정 홀더 자세({ep.holder_poses})에 쓰는 홀더가 다 있는가"
                     if real else "") + " (읽기만)",
                     ["python3", "{repo}/deploy/policy_control/tools/episode_run.py", "--episode", f"config/episodes/{name}.yaml",
                      "--plan", *(["--require-holders"] if real else [])]))
    cmds.append(_cmd("시작 검사 — 같은 이름의 정책 노드 · 실행기가 남아 있지 않은가(다른 정책 단계가 띄운 것) (읽기만)",
                     ["bash", "-lc", f"! ros2 node list 2>/dev/null | grep -E '^/({left})$'"]))
    if perceives:
        check = (f"★[에피소드 {name}] 물체({', '.join(ep.objects)})가 학습 배치에 서 있고 FP++ 가 잡고 있는가"
                 + (f" · 홀더 {', '.join(f'{k}={v}' for k, v in ep.holders.items())} 위가 비었는가 · {ep.holder_poses} 가 있는가"
                    "(cup_holders 단계 --write)" if ep.holders else ""))
    else:                                            # ★10.17 경로 B — 사람이 컵을 손에 댄다(FP++ · 정책 없음)
        check = (f"★[에피소드 {name}] 물체({', '.join(ep.objects)})와 비드를 손 닿는 곳에 준비했는가. 로봇이 컵 없이 시작 자세로 간 뒤 "
                 "확인 노드마다 멈춘다 — 컵은 확인한 뒤에만 손에 댄다")
    cmds.append(_cmd(check + ". 노드를 띄운 뒤에는 상황판 에피소드 패널에서 진행한다" if real else "fake — 확인만",
                     ["bash", "-lc", "true"], manual=True))
    for role, b in ep.policies.items():
        e = entries.get(b.policy)
        # Missing/partial optional bundles stay unavailable at the episode --plan gate.
        # Always use a typed file path, never the policy directory as contract:=.
        if e is None or not e.contract:
            filename = {'aglt': 'rh_aglt_contract.json', 'place': 'rh_place_contract.json'}[b.kind]
        else:
            filename = e.contract
        contract = f"{{repo}}/deploy/policies/{b.policy}/{filename}"
        robot = f"{{artifact:robot_{b.side}}}"
        if b.kind == "aglt":
            obj = next((j.target_object for n in ep.nodes for j in n.jobs if j.role == role and j.target_object), first_obj)
            # 10.12 A안: 컵 = FP++ live(끊기면 마지막 값) → 쥐면 파지 + 100 ms 뒤 찍힌 프레임 시각의 손바닥 FK(가리면 0.7 s 대체)
            # ★10.13 방안 A: SETTING 이 없는 에피소드(제자리 들기)는 sim 처럼 첫 목표 성공에 끝낸다(goal_max:=1)
            lift_only = b.side not in ep.setting
            # ★10.13 grasping lift_hold(사용자 승인): 컵이 리셋보다 7 cm 위 + 쥠 N 틱 연속이어도 끝낸다(성공과 OR)
            hold = LIFT_HOLD_TICKS[b.side]
            end_args, end_note = ((["goal_max:=1", "-p", "freeze_rule:=lift_hold", "-p", "freeze_lift_m:=0.07",
                                    "-p", f"freeze_hold_steps:={hold}"],
                                   f"첫 목표(컵 + 10 cm) 성공 또는 7 cm 위 {hold / 60:g} s 쥠 유지에 스스로 끝남") if lift_only
                                 else (["stop_on_target:=true"], "SETTING 도달에 스스로 끝남"))
            max_s = "15.0"
            if b.approach_grasp is not None:            # ★10.17 경로 C — 접근 정책 뒤 스크립트 닫기 · 쥠 확인 · 들기 · 유지
                from policy_control.approach_grasp import ApproachGraspCfg, episode_s
                end_args = [f"approach_grasp:={ApproachGraspCfg.to_param(b.approach_grasp)}"]
                ag = ApproachGraspCfg.parse(ApproachGraspCfg.to_param(b.approach_grasp))
                end_note = ("경로 C: 접근 정책 → 정책 끄고 손 닫기 → 쥠 확인(네 손가락 2 개 · 8 cm) → 10 cm 들기 → 3 s 유지, 못 쥐면 들지 않고 멈춤"
                            if ag is not None and ag.judge else      # ★10.18 사용자: 기본 강제 파지(판정 없음)
                            "경로 C: 접근 정책 → 정책 끄고 손 닫기(닿은 손가락은 실측에 고정 · 안 닿은 손가락 앞섬 "
                            f"{(ag or ApproachGraspCfg()).close_lead_rad:g} rad) → 판정 없이 10 cm 들기 → 3 s 유지, 들기 미달만 멈춤")
                max_s = f"{episode_s(b.approach_grasp):.1f}"   # 최악 경로 + 3 s(실행기는 PolicyPlan.max_s + 10 s 를 기다린다)
            if b.handoff:                               # ★10.20 옛 경로 D v1 · 10.20 밤부터 경로 E(grasp_e_<팔>, 사용자 "정책2개가 연달아 진행되는 세트")
                from policy_control.approach_grasp import ApproachGraspCfg, episode_s
                ag = ApproachGraspCfg.parse(ApproachGraspCfg.to_param(b.approach_grasp))
                hold_ = LIFT_HOLD_TICKS[b.side]
                end_args = [f"approach_grasp:={ApproachGraspCfg.to_param(b.approach_grasp)}",
                            "-p", f"handoff_contract:={{repo}}/deploy/policies/{b.handoff}/rh_aglt_contract.json",
                            "-p", "goal_max:=1", "-p", "freeze_rule:=lift_hold", "-p", "freeze_lift_m:=0.07",
                            "-p", f"freeze_hold_steps:={hold_}"]
                route = "경로 F" if b.handoff.endswith("_rcg04") else "경로 E"   # ★10.21 경로 F(ap05 → rcg04) — 같은 넘김 · ★10.22 pick_place_f 도
                end_note = (f"{route}: 접근 정책 → 사전 파지 자리(2 cm)에 {ag.still_s:g} s 멈추면 같은 노드에서 {b.handoff} 로 넘김 → "
                            f"첫 목표(컵 + 10 cm) 성공 또는 7 cm 위 {hold_ / 60:g} s 쥠 유지에 끝남 · 접근 {ag.approach_max_s:g} s 안에 못 오면 멈춤")
                max_s = f"{episode_s(b.approach_grasp):.1f}"   # 접근 최악 + 넘겨받은 정책 15 s + 3 s(실행기 _node_max_s 와 같은 식)
            # ★10.19 사용자: "sim 은 세게 쥐면 물리가 터져 적당히 쥐게 학습했다 — 실기는 제어로 더 세게"(rh_aglt.GripAssist D).
            #   정책이 쥐는 에피소드(제자리 들기 · 경로 D)만. 경로 C 는 닫기가 스크립트라 끈다.
            squeeze = lift_only and (b.approach_grasp is None or bool(b.handoff))
            if squeeze:
                end_note += " · 쥠 보조(정책이 반 넘게 닫으면 네 손가락 끝까지 · 안 풂)"
            cmds.append(_cmd(f"[{b.side}] {role} 노드({b.policy}) — {end_note} · 컵 = FP++ live"
                             " · 쥐면 손바닥 FK 로 손에 붙임",
                             ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_aglt_node.py", "--ros-args",
                              "-r", f"__node:=rh_aglt_node_{b.side}", "-p", f"ns:={b.side}", "-p", f"contract:={contract}",
                              "-p", f"robot:={robot}", "-p", "device:=cpu", "-p", *end_args,
                              "-p", f"cup_topic:={_live(ep.objects[obj]['topic'])}", *AGLT_LIVE_ARGS,
                              *(["-p", "hand_squeeze:=true"] if squeeze else []),
                              "-p", f"max_episode_s:={max_s}"], background=True))
        elif b.kind == "place":
            src = next((j for n in ep.nodes for j in n.jobs if j.role == role), None)
            obj = src.source_object if src and src.source_object else first_obj
            hid = ep.holders.get(src.target_holder) if src and src.target_holder else PLACE_HOLDER[b.side]
            bundle_note = (" · 붙인 자세 = 잡기 노드의 grasp_state 넘김 묶음(bundle:=auto)" if e is not None and e.contract
                           and _place_attach(REPO / "deploy" / "policies" / b.policy / e.contract) == "bundle" else "")
            cmds.append(_cmd(f"[{b.side}] {role} 노드({b.policy}) — 홀더 {hid} · 놓은 뒤 스스로 끝남{bundle_note}",
                             ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_place_node.py", "--ros-args",
                              # 한 팔에 aglt 와 같이 뜬다 — 서비스는 <팔>_place, 이벤트는 그 팔 토픽(pd 가 reset · stop 적용)
                              "-r", f"__node:=rh_place_node_{b.side}", "-p", f"ns:={b.side}_place",
                              "-p", f"episode_topic:=/policy_control/{b.side}/episode", "-p", f"contract:={contract}",
                              "-p", f"robot:={robot}", "-p", "device:=cpu", "-p", f"cup_topic:={ep.objects[obj]['topic']}",
                              "-p", f"holder:={hid}", *([] if real else ["-p", "require_grasp:=false"])], background=True))
    if real:                                         # 첫 실기는 놓음 문턱(손끝 · 관절 힘 · 손 목표)을 bag 으로 정한다(PLACE 10.04)
        extra = [f"/policy_control/{x}" for x in ("joint_target", *(f"status/pd_{s}" for s in sides),
                                                   *(f"status/rh_aglt_node_{s}" for s in sides),
                                                   *(f"status/rh_place_node_{s}" for s in sides), f"status/{runner}",
                                                   *(f"{s}/episode" for s in sides))]
        extra += [relay_topic(runner, o) for o in ep.objects] + [o["topic"] for o in ep.objects.values()]
        extra += [_live(o["topic"]) for o in ep.objects.values()]          # 10.12 A안 — aglt 가 받은 live 컵
        extra += [f"/objects/cup_holder_{h}/pose" for h in sorted(set(ep.holders.values()))]
        extra += [f"/policy_control/pd_{s}/applied" for s in sides]       # 10.08 팔 지연 도구(arm_latency_report)가 쓴다
        # ★10.12 실기: 관측 · 행동이 bag 에 없어 실패 원인(어느 입력이 정책을 sim 과 갈랐나)을 역산 · 재구성으로만 쟀다
        extra += [f"/policy_control/{s}/{x}" for s in sides for x in ("obs", "action")]
        hands = f"HAND_SIDES='{' '.join(sides)}' " if len(sides) == 1 else ""      # 한 팔 에피소드는 그 손만(다른 팔 기록과 겹치지 않게)
        cmds.append(_cmd("기록 시작 — 팔 · 손 bag(rh56f1_record.sh) + 정책 · 실행기 상태 · 컵 · 홀더",
                         ["bash", "-lc", f"{hands}EXTRA='{' '.join(extra)}' bash {{repo}}/deploy/policy_control/tools/rh56f1_record.sh "
                                         f"start episode_{name}"]))
        cmds.append(_cpu_record(f"episode_{name}"))
        if VISION_RECORD and name in REPORT_VIDEO_HZ:   # ★10.19 보고서 사진 · 영상 — 카메라 PC 에서 따로(tools/episode_media.py 가 묶는다)
            cmds.append(_vision_record(f"episode_{name}", [real_object(o["topic"]) for o in ep.objects.values()],
                                       REPORT_VIDEO_HZ[name]))
    cmds.append(_cmd(f"에피소드 실행기 {runner} — {name} · 승인은 상황판(노드마다 이름 · 연속 실행은 episode:{name})",
                     ["{repo}/.venv/bin/python", f"{PC}/policy_control/episode_runner_node.py", "--ros-args",
                      *(["-r", f"__node:={runner}"] if runner != "episode_runner" else []),
                      "-p", f"episode:=config/episodes/{name}.yaml", "-p", "robot:=rh56f1"], background=True))
    cmds.append(_cmd("★에피소드 진행 — 상황판 에피소드 패널의 [다음](구분 실행) · [연속 실행]. 성공 · 정지로 끝나면 확인 "
                     "(이상하면 '에피소드 실행기 정지' · 정지 바)", ["bash", "-lc", "true"], manual=True))
    bg = [i for i, c in enumerate(cmds) if c.get("background")]
    cmds.append(_cmd("에피소드 노드들 정지", stop=[f"episode_{name}#{i}" for i in reversed(bg)]))
    if real:
        cmds.append(_cmd("기록 끝 — 이 에피소드의 bag 두 개 마무리(SIGINT)",
                         ["bash", "{repo}/deploy/policy_control/tools/rh56f1_record.sh", "stop", f"episode_{name}"]))
        if VISION_RECORD and name in REPORT_VIDEO_HZ:   # ★10.19 보고서 사진 · 영상 — 이 단계에서 돈 회마다(아직 안 만든 것)
            cmds.append(_cmd(f"보고서 영상 · 장면 사진 — 카메라 영상({PERCEPTION_HOST}) + 노드 시각 → logs/report_media/<회>"
                             "(video.mp4 · 시작 · 다가간 뒤 · 쥔 뒤 · 들어 올린 뒤 · sheet.jpg)",
                             ["python3", f"{PC}/tools/episode_media.py", "--pending", name,
                              "--host", PERCEPTION_HOST if REMOTE_PERCEPTION else "local"]))
    return cmds


#: 단독 aglt 단계에서 정책 노드 명령 번호 — 실기는 앞에 기록 시작(bag) · CPU 기록이 붙는다
AGLT_NODE_IDX = {True: 4 if VISION_RECORD else 3, False: 1}     # 실기: 0 확인 · 1 bag · 2 CPU · (3 영상) · 정책 노드


def real_object(topic: str) -> str:
    """/objects/<실물>/pose → 실물 이름."""
    from policy_control.episode_ros import real_object_name
    return real_object_name(topic)


def _cpu_record(name: str) -> dict:
    """프로세스별 CPU 1 Hz CSV(tools/proc_cpu_record.py) — 10.03 은 정책이 15 s 에 멈춰 추론 부하를 못 쟀다(10.08 실기 전 세팅)."""
    return _cmd("CPU 기록 — 프로세스별 코어 1 Hz → logs/cpu(끝날 때 요약)",
                ["python3", f"{PC}/tools/proc_cpu_record.py", "--out", f"{{repo}}/logs/cpu/{name}.csv"], background=True)


def _record_start(name: str, sides: list, topics: list, objs: list | None = None) -> list[dict]:
    """단독 정책 단계 기록(실기) — bag(팔 · 손) + CPU. 팔 지연 도구가 쓰는 pd applied · 목표 · status 를 같이 싣는다(10.08)."""
    extra = ["/policy_control/joint_target", *(f"/policy_control/{x}" for s in sides for x in
                                                (f"status/pd_{s}", f"status/rh_aglt_node_{s}", f"{s}/episode", f"pd_{s}/applied",
                                                 f"{s}/obs", f"{s}/action")),      # 10.12 관측 · 행동도(실패 원인 재구성)
             *topics]
    # ★10.09 실기: 양팔 동시 — 이름마다 따로 기록(한 팔의 시작 · 끝이 다른 팔 기록을 끄지 않는다), 손 bag 은 그 팔 손만
    return [_cmd(f"기록 시작 — 팔 · 손 bag(rh56f1_record.sh {name}) · 팔 지연은 tools/arm_latency_report.py <bag>/arm",
                 ["bash", "-lc", f"EXTRA='{' '.join(extra)}' HAND_SIDES='{' '.join(sides)}' "
                                 f"bash {{repo}}/deploy/policy_control/tools/rh56f1_record.sh start {name}"]),
            _cpu_record(name),
            # ★10.09 사용자: 카메라 영상 5 Hz · FP++ 실시간 좌표(정책 입력 아님) — localhost DDS 라 따로 띄운다(10.20 기본 끔)
            *([_vision_record(name, objs or [])] if VISION_RECORD else [])]


def _vision_record(name: str, objs: list, hz: float = 5.0) -> dict:
    """영상 · FP++ 실시간 좌표 기록 — 카메라가 있는 PC 의 localhost DDS 에서 돈다(10.09 사용자)."""
    rate = [] if hz == 5.0 else ["--hz", f"{hz:g}"]
    if not REMOTE_PERCEPTION:
        return _cmd(f"영상 · FP++ 실시간 좌표 기록({hz:g} Hz JPEG · live_pose CSV) → logs/vision_runs",
                    ["env", "ROS_LOCALHOST_ONLY=1", "python3", "{repo}/scripts/nodes/vision_recorder.py", "--name", name,
                     "--camera-extrinsics", "{repo}/" + CAMERA_EXTRINSICS, *rate, "--objects", *objs], background=True)
    # 원격: ssh -tt 를 콘솔이 끝내면 저쪽 셸이 HUP 를 받는다 → 기록기에 TERM(마무리 · CSV 닫기) 뒤 기다린다.
    # 기록은 저 PC 의 sim2real/logs/vision_runs 에 남는다(로그는 /tmp/perception/vrec_<이름>.log).
    rec = (f"{REMOTE_PY} scripts/nodes/vision_recorder.py --name {name} --camera-extrinsics {CAMERA_EXTRINSICS} {' '.join(rate)} "
           f"--objects {' '.join(objs)} > /tmp/perception/vrec_{name}.log 2>&1 & P=$!; trap 'kill -TERM $P; wait $P' HUP TERM INT; wait $P")
    return _cmd(f"영상 · FP++ 실시간 좌표 기록({PERCEPTION_HOST}, {hz:g} Hz JPEG · live_pose CSV) → 저 PC 의 logs/vision_runs",
                _on_perception_pc(rec), background=True)


def _record_stop(stage: str, cpu_index: int, name: str) -> list[dict]:
    return [_cmd("CPU · 영상 기록 끝(요약은 그 단계 로그)" if VISION_RECORD else "CPU 기록 끝(요약은 그 단계 로그)",
                 stop=[*([f"{stage}#{cpu_index + 1}"] if VISION_RECORD else []), f"{stage}#{cpu_index}"]),
            _cmd(f"기록 끝 — {name} bag 두 개 마무리(SIGINT, 이 단계 기록만)",
                 ["bash", "{repo}/deploy/policy_control/tools/rh56f1_record.sh", "stop", name])]


def _home(s: str) -> list[dict]:
    """차렷 → 홈, 한 번 승인으로 끝까지(10.01 사용자: "home 자세 진행하면 팔-손 한번에"). 사람 확인은 맨 앞 하나 —
    그 뒤는 도구가 실패하면 그 자리에서 멈춘다(engage 는 10 s HOLD 감시, 재생 전 시작점 검사, hand_home 은 pd 가 정착했을 때만 받는다)."""
    joints = ",".join(f"{s[0]}_aj_{i}" for i in range(1, 8))
    ctl = lambda only, *extra: ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", only, *extra]  # noqa: E731
    return [
        _cmd(f"★[{s}] 경로 주변(로봇 옆 · 테이블 앞 가장자리 · 몸통)이 비어 있는가. 약 15 s 동안 최대 0.3 rad/s 로 움직인다 — "
             "engage → 차렷 손(주먹) → 경로 재생 → 홈 정착 → 손을 홈 손 자세로, 끊지 않고 이어 간다", ["bash", "-lc", "true"], manual=True),
        _cmd(f"[{s}] engage → 제자리 10 s. pd 가 HOLD 로 가면 실패하고 pd 를 해제한다", ctl("pd_engage", "--hold-s", "10"),
             execute_args=["--execute", "--approve", "pd_engage"]),
        _cmd(f"[{s}] 손을 차렷 손(주먹)으로(pd/hand_path) — 네 손가락 1.45 · 엄지 대향 0.8 · 굽힘 0.3. 홈 경로는 이 손으로 계획했다",
             ctl("pd_hand_path", "--service-timeout", "15"), execute_args=["--execute", "--approve", "pd_hand_path"]),
        _cmd(f"[{s}] 저장 경로를 재생해도 되는가 — 팔이 차렷(경로 시작점 0.05 rad 안) · 경로가 지금 계약으로 만든 것 · 관절 상태가 살아 있음",
             ["python3", f"{PC}/tools/check_path_start.py", "--npz", f"{{artifact:path_{s}}}", "--contract", "{artifact:contract}"]),
        _cmd(f"[{s}] 저장 경로를 pd 로 재생(약 15 s, 최대 0.3 rad/s) — 끝나면 이 팔의 episode stop 으로 pd 가 마지막 자세를 붙든다",
             ["python3", f"{PC}/tools/replay_to_pd.py", "--npz", f"{{artifact:path_{s}}}", "--joints", joints,
              "--rate-scale", "1.0"], execute_args=["--execute"]),
        _cmd(f"[{s}] 홈에서 정착(이미 도착 — 남은 오차만)", ctl("pd_goto_home", "--service-timeout", "45"),
             execute_args=["--execute", "--approve", "pd_goto_home"]),
        _cmd(f"[{s}] 손을 계약 홈 손 자세로(pd/hand_home — 팔이 홈에 정착했을 때만 받는다)", ctl("pd_hand_home", "--service-timeout", "15"),
             execute_args=["--execute", "--approve", "pd_hand_home"]),
    ]


def _return(s: str) -> list[dict]:
    """홈 → 차렷, 한 번 승인(10.01 사용자 · 4090:s2r 실기 순서). 홈 근처에서만 — 정책이 멈춘 먼 자리에서 goto_home 직선은 테이블을
    모른다(DG-5F 09.28). 홈 정착(굳은 hold 도 풀린다) → 주먹 → 경로 끝 검사 → 같은 경로 역재생 → pd 해제(JTC 가 차렷을 잡는다)."""
    joints = ",".join(f"{s[0]}_aj_{i}" for i in range(1, 8))
    ctl = lambda only, *extra: ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", only, *extra]  # noqa: E731
    return [
        _cmd(f"★[{s}] 팔이 홈 근처에 있고(정책 뒤면 먼저 rehome_{s} 단계로 홈에) 손에 컵이 없는가 · 경로 주변이 비어 있는가. "
             "홈 정착 → 주먹 → 저장 경로 역재생(약 15 s, 최대 0.3 rad/s) → pd 해제, 끊지 않고 이어 간다", ["bash", "-lc", "true"], manual=True),
        _cmd(f"[{s}] 홈에서 정착 — 남은 오차만(굳은 hold 도 여기서 풀린다)", ctl("pd_goto_home", "--service-timeout", "45"),
             execute_args=["--execute", "--approve", "pd_goto_home"]),
        _cmd(f"[{s}] 손을 차렷 손(주먹)으로(pd/hand_path) — 홈 경로는 이 손으로 계획했다", ctl("pd_hand_path", "--service-timeout", "15"),
             execute_args=["--execute", "--approve", "pd_hand_path"]),
        _cmd(f"[{s}] 되짚어도 되는가 — 팔이 경로 끝(홈) 0.05 rad 안 · 경로가 지금 계약으로 만든 것 · 관절 상태가 살아 있음",
             ["python3", f"{PC}/tools/check_path_start.py", "--npz", f"{{artifact:path_{s}}}", "--contract", "{artifact:contract}",
              "--at", "end"]),
        _cmd(f"[{s}] 같은 경로를 거꾸로 재생 → 차렷", ["python3", f"{PC}/tools/replay_to_pd.py", "--npz", f"{{artifact:path_{s}}}",
                                                   "--reverse", "--joints", joints, "--rate-scale", "1.0"], execute_args=["--execute"]),
        _cmd(f"[{s}] pd release — 역블렌드 → 0 송출 → JTC 가 차렷을 잡는다(pd 프로세스는 남는다 — 끝낼 때 release_{s})",
             ctl("pd_release"), execute_args=["--execute"]),
    ]


def mission(kind: str) -> dict:
    real = kind == "real"
    robot = "real" if real else "fake"
    arts = {
        "contract": "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json",
        "robot_right": f"deploy/policy_control/config/robots/rh56f1_right_{robot}.yaml",
        "robot_left": f"deploy/policy_control/config/robots/rh56f1_left_{robot}.yaml",
        "pd": f"deploy/policy_control/config/pd_rh56f1{'' if real else '_fake'}.yaml",
        "pd_exec": f"deploy/policy_control/config/pd_rh56f1{'_exec' if real else '_fake'}.yaml",
        "rh56f1_map": "deploy/policy_control/config/rh56f1_hand_map.yaml",
        # 차렷 → 홈(rh_aglt 시작 자세) 저장 경로 — plan_home_path.py --side <s> --goal contract --other-arm both --hand-start both
        #   --urdf <RH56F1 자산> --contract <이 미션 contract> --profile openarm_rh56f1 --env-yaml <aglt 런> --pd-config pd_rh56f1
        #   ★plan_home_path 기본값(--urdf · --contract · --profile · --env-yaml)은 DG5F 다 — 빼먹으면 DG5F 세계로 검사한다(10.01)
        #   10.01 재계획: --hand-start pd(주먹) --ramp-time 0.5 --max-speed 0.3 --seed 0 --env-yaml <side>_rh_aglt_i10d — 각 약 14~15 s
        "path_right": "deploy/policy_control/paths/home_rh56f1_right.npz",
        "path_left": "deploy/policy_control/paths/home_rh56f1_left.npz",
        # 양팔 붓기 정책(첫 화면의 '양팔' 자리가 바꾼다) · 양팔 robot yaml
        "pourfj_both": "deploy/policies/rh56f1/pour_fj/both_f01/pour_fj_contract.json",
        "robot_bi": f"deploy/policy_control/config/robots/rh56f1_bi_{robot}.yaml",
        # 한 팔 rh_aglt 정책(첫 화면의 '오른팔 · 왼팔' 자리가 바꾼다) — 09.30
        # ★10.18 사용자: 옛 정책 폴더(i03 ~ src21c · g5362 · cyl60g · env17 · rg3s · rg3mft 등 24개) 삭제 → 기본 = rg3wo 좌우
        #   (10.17 grasping 파지 배포, 다물체 · 위치 제어). 노드에 그 팔 FP++ 물체의 정책 물체(object)를 넘긴다(_policy_object).
        #   옛 기본 이력은 git log 에 있다(10.09 right_src21c · left_g5362).
        # ★10.20 사용자: 전달받은 정책만 남김(옛 9개 → deploy/policies_retired). 기본 = 접근 ap05 — 홈에서 출발하는 정책은 이것뿐이다.
        #   rcg03(파지 · 들기, 10.20 rcg02 대체)는 사전 파지 자세에서 시작해야 해서 이 단계(홈 출발)에 고르지 않는다.
        "aglt_right": "deploy/policies/rh56f1/aglt/right_approach_ap05/rh_aglt_contract.json",
        "aglt_left": "deploy/policies/rh56f1/aglt/left_approach_ap05/rh_aglt_contract.json",
        # 한 팔 컵 홀더 놓기(10.04 PLACE 세션, aglt cyl60 인계) — 콘솔 자리는 아직 없다(팔마다 한 자리 = aglt)
        "place_right": "deploy/policies/rh56f1/place/right_i09/rh_place_contract.json",
        "place_left": "deploy/policies/rh56f1/place/left_i01/rh_place_contract.json",
    }
    if real:
        arts["rh56f1_ports"] = "deploy/policy_control/config/rh56f1_ports.yaml"
    return {
        "name": f"RH56F1 양팔 {'실기' if real else 'fake'} 손 연결 · pd 점검",
        "artifacts": arts,
        "checkpoints": {},
        "groups": [{"id": "check", "title": "점검"}, {"id": "connect", "title": "연결"}, {"id": "prepare", "title": "준비"},
                   {"id": "motion", "title": "자세 이동", "motion": True}, {"id": "policy", "title": "정책 동작", "motion": True},
                   {"id": "finish", "title": "정리"},
                   {"id": "recognize", "title": "개별 인지 (선택)", "optional": True},    # ★10.13 컵 하나씩 다시 인지
                   {"id": "diagnose", "title": "진단 (선택)", "motion": True, "optional": True}],
        "lanes": [{"id": "rig", "title": "드라이버"},
                  {"id": "arm_right", "title": "오른팔 · 오른손", "side": "right", "focus": "policy_aglt_right"},
                  {"id": "arm_left", "title": "왼팔 · 왼손", "side": "left", "focus": "policy_aglt_left"},
                  {"id": "both", "title": "양팔 · 컵 · 정책", "focus": "policy_pourfj"}],
        "stages": _stages(kind),
        "run": _run(kind),
    }


def render(kind: str) -> str:
    return HEADER[kind] + yaml.safe_dump(mission(kind), allow_unicode=True, sort_keys=False, width=140)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="쓰지 않고 커밋된 파일과 비교만")
    args = ap.parse_args(argv)
    bad = []
    for kind, path in OUT.items():
        text = render(kind)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                bad.append(path.name)
            continue
        path.write_text(text, encoding="utf-8")
        print(f"썼다: {path.relative_to(REPO)}")
    if args.check:
        print("일치" if not bad else f"다르다: {bad} — make_rh56f1_missions.py 로 다시 만들 것")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
