#!/usr/bin/env python3
"""RH56F1 로봇(arm4090) 미션 — 실기 · fake 두 벌을 **한 정의에서** 만든다.

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
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = {"real": REPO / "config" / "mission_rh56f1_control.yaml", "fake": REPO / "config" / "mission_rh56f1_fake.yaml"}
SIDES = ("right", "left")
PC = "{repo}/deploy/policy_control"
#: 방향 확인 한 축씩 — 네 손가락은 굽힘 쪽(+), 엄지 굽힘 +, 엄지 회전은 지금에서 − (편 손 1.57 에서 grip 1.20 쪽)
PROBES = (("index_1", "0.3"), ("middle_1", "0.3"), ("ring_1", "0.3"), ("pinky_1", "0.3"),
          ("thumb_2", "0.15"), ("thumb_1", "-0.3"))
#: 실기 컵 — FP++ 물체 하나. fake 는 학습 배치의 두 컵(cup_src · cup_rcv)
#: ★10.04 cyl60(⌀60 × 170 mm 노란 원통, 원점 = 중심) — rh_aglt cyl60g 정책으로 s2r(T2R Grasping, 사용자 승인). 전: aglt_cup_s065
REAL_CUP = "cyl60"
#: ★10.08 사용자: 왼쪽 파란 컵 · 오른쪽 노란 컵 — 팔마다 FP++ 물체 하나(색으로 가른다: 노랑 bright · 파랑 blue). 모양은 같은 cyl60
REAL_CUPS = {"right": REAL_CUP, "left": "cyl60_blue"}
#: 같은 묶음(objects.yaml fpp.group cups)의 나머지 컵 — 한 컨테이너가 같이 찍는다(10.08 사용자: 파랑 · 핑크 · 노랑)
REAL_EXTRA_CUPS = ("cyl60_pink",)
#: 놓기 목표 홀더 — 좌우 학습 목표(우 1 · 2, 좌 0 · 1)에 모두 드는 가운데 홀더 1(10.04)
PLACE_HOLDER = {"right": 1, "left": 1}
#: fake 홀더 y — hdgp rh_place env holder_ys(0.153, −0.002, −0.161)
FAKE_HOLDER_Y = {0: 0.153, 1: -0.002, 2: -0.161}
#: 상황판에서 굴리는 에피소드(10.04 사용자: 구분 실행 → 연속 실행) — config/episodes/<이름>.yaml, 실기에 쓸 수 있는 것만
EPISODES = ("pick_place_right", "pick_place_left")
#: 에피소드 snapshot 이 다시 내는 정지 물체 자세(episode_ros.OBJECT_RELAY) — aglt 는 이것을 컵으로 본다
EPISODE_RELAY = "/episode/objects/{}/pose"
REAL_CUP_ORIGIN_Z = 0.085                 # 원점 높이(바닥 위) — 컵 자세 확인 문구
#: arm4090 머리 카메라 외부 파라미터 — 테이블 CAD 캘리브(scripts/calib/table_cad_extrinsics.py, 10.01 기본 방법)
CAMERA_EXTRINSICS = "config/global_camera_extrinsics_arm4090.yaml"


def _cups() -> list[str]:
    """실기 FP++ 컵 — 오른쪽 · 왼쪽 · 나머지 순, 중복 없이."""
    return list(dict.fromkeys([*(REAL_CUPS[s] for s in SIDES), *REAL_EXTRA_CUPS]))


def _cups_txt() -> str:
    return " · ".join([*(f"{REAL_CUPS[s]}={'노랑 오른쪽' if s == 'right' else '파랑 왼쪽'}" for s in SIDES),
                       *(f"{c}=핑크" for c in REAL_EXTRA_CUPS)])
#: 머리 설정(포트 · 게인 · 모터 id) — head_home · head_pose_check 가 같이 쓴다
HEAD_CONFIG = "config/head_home_rh56f1.yaml"
#: ★10.04 사용자 "fpp 진행 전에 자동으로 각도 확인하고 세팅을 제대로 맞춘 다음에 진행" — 외부 파라미터를 잰 머리 자세
#  (CAMERA_EXTRINSICS 의 head_pose)와 지금 자세를 비교하고, 실기 실행이면 그 자세로 맞춘다(pan 은 목표보다 18 틱 앞에서 멈춘다)
HEAD_CHECK_ARGV = ["python3", "{repo}/scripts/head_pose_check.py", "--config", "{repo}/" + HEAD_CONFIG,
                   "--extrinsics", "{repo}/" + CAMERA_EXTRINSICS]
RECALIB_HINT = ("scripts/calib/table_cad_extrinsics.py --init " + CAMERA_EXTRINSICS + " --write --head-config " + HEAD_CONFIG)
HEADER = {
    "real": "# RH56F1 로봇(arm4090) 실기 미션 — scripts/ops/make_rh56f1_missions.py 가 만든다. 손으로 고치지 말 것(--check 가 잡는다).\n",
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
        {"id": "cups", "group": "connect", "lane": "both", "needs": ["head_home"], "skippable": True,
         "touches_real": real,
         "title": (f"컵 자세(arm4090 FP++) — RealSense · FP++ 컨테이너 하나(fpp_cups, 색으로 가르기 · 한 번 찍기: {_cups_txt()}) → /objects/<컵>/pose (base). GPU VRAM 약 4 GB" if real else
                   "fake 컵 두 개 — 학습 배치 중심(0.38, ∓0.16), 테이블 위에 선 채")},
        {"id": "cup_holders", "group": "connect", "lane": "both", "needs": ["head_home"], "skippable": True,
         "touches_real": real,
         "title": ("컵홀더 자세(마커 ID 0·1·2) — RealSense 컬러 → /objects/cup_holder_{0,1,2}/pose (base) · /cup_holders/status. "
                   "로봇은 움직이지 않는다. CPU 만" if real else "fake — 카메라 없음(실기 순서를 맞추려고 둔 자리)")},
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
               **({"blocked": "붓기는 보류(10.07 사용자: 성공 전 정책은 직접 확인 뒤) — 실기 인지는 10.08 부터 색으로 두 컵"
                             f"({_cups_txt()})을 낸다"} if real else {}),
               "needs": ["home_right", "home_left", "cups", "hand_check_right", "hand_check_left"],
               "needs_why": "정책은 양팔이 pour_fj 시작 자세 · 손이 편 채 · 두 컵이 선 채로만 출발해 봤다. ★09.29 홈이 rh_aglt 시작 자세로 "
                            "바뀌어 pour_fj 시작 자세와 다르다 — 정책 노드가 start 를 거부한다(0.15 rad). 홈 → pour_fj 시작 경로가 필요",
               "artifacts": ["pourfj_both", "robot_bi", "contract"],
               "title": "[양팔] pour_fj 정책(첫 화면에서 고른 것) — 정책 노드 → reset → start → 관찰 → stop (시간이 되면 스스로 끝난다)"})
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
    for name in EPISODES:
        ep = _episode(name)
        sides = sorted({b.side for b in ep.policies.values()})
        st.append({"id": f"episode_{name}", "group": "policy", "lane": f"arm_{sides[0]}", "skippable": True,
                   "touches_real": real,
                   "needs": [*(f"home_{s}" for s in sides), "cups", "cup_holders"],
                   "needs_why": "에피소드는 홈(rh_aglt 시작 자세) · FP++ 컵 · 고정 홀더 자세 파일 위에서 정책을 잇는다",
                   "title": f"[에피소드] {name} — 노드를 띄우고 상황판 에피소드 패널에서 [다음](구분 실행) · [연속 실행]으로 "
                            f"진행: {' → '.join(n.id for n in ep.nodes if n.type != 'terminal')}"})
    st.append({"id": "shutdown", "group": "finish", "lane": "rig", "needs": ["drivers"], "touches_real": real,
               "title": "안전 종료 — 팔 받침 확인 → 남은 pd → 손 상태 · 드라이버 → 팔 브링업 (★팔 토크가 풀린다)"})
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
            _cmd("★CAN 설정(우 can0) — sudo, 운영자 셸에서. arm4090 의 CAN 이름이 다르면 그 이름으로",
                 ["bash", "-lc", "sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can0 up"],
                 manual=True),
            _cmd("★CAN 설정(좌 can1)",
                 ["bash", "-lc", "sudo ip link set can1 down && sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can1 up"],
                 manual=True),
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
            _cmd("남은 정책 노드 · pd 정지", stop=["policy_pourfj#1"] + [f"policy_aglt_{s}#{AGLT_NODE_IDX[real]}" for s in SIDES]
                 + ([f"policy_aglt_{s}#2" for s in SIDES] if real else [])
                 + [f"{p}_{s}#{i}" for s in SIDES for p, i in (("pd_load", 0), ("pd_arm", 1))]),
            # 정책 · 에피소드 단계가 도중에 실패하면 bag 기록기가 남는다(setsid) — 기록 중이 아니면 그렇다고만 찍는다(10.08)
            *([_cmd("남은 bag 기록 마무리(rh56f1_record.sh stop)", ["bash", "{repo}/deploy/policy_control/tools/rh56f1_record.sh", "stop"])]
              if real else []),
            *([_cmd("카메라 · FP++ 내리기 — 런처가 없으면 이 PC 에서 직접 내린다",
                    ["python3", "{repo}/scripts/ops/perception_ctl.py", "stop", "--camera", "--host", "local", "--wait", "60"]),
               _cmd("인지 런처 · 자세 수신기 · 물체 자세 노드 · 컵홀더 노드 정지",
                    stop=["cups#1", "cups#2", "cups#3", "cup_holders#3"])] if real else []),
            _cmd("손 상태 노드 · 손 드라이버 정지 — 손가락은 마지막 자세에서 멈춘다(벤더 펌웨어가 잡는다)",
                 stop=[f"hand_{s}#{i}" for s in SIDES for i in ((2, 1) if real else (1, 0))]),
            _cmd("★팔 브링업 정지 — 모든 팔 모터가 꺼진다(받침으로 내려앉는다)" if real else "fake 플랜트 정지",
                 stop=["drivers#3" if real else "drivers#0"]),
        ],
    }
    run["cup_holders"] = [
        _cmd("★홀더 세 개가 상판 위에 서 있고 -x 면 마커가 손 · 컵에 가리지 않는가(머리 자세는 다음 스텝이 캘리브 자세로 맞춘다)",
             ["bash", "-lc", "true"], manual=True),
        _cmd("머리 자세 확인 · 맞춤 — 카메라 외부 파라미터를 잰 자세(head_pose)와 다르면 그 자세로 맞춘다(머리가 조금 움직인다)", HEAD_CHECK_ARGV, execute_args=["--execute"]),
        _cmd("카메라(RealSense) 켜기 — 이미 떠 있으면 그대로(cups 단계와 같이 써도 된다)",
             ["bash", "{repo}/scripts/vision/camera_up.sh"]),
        _cmd("컵홀더 자세 노드 — ArUco → 직전 자세 추적(0.2 s) → 놓친 id 만 무늬 전체 탐색(첫 장 ~6 s). "
             "x 공유 · 상판 z 고정 · 5장 중앙값, 세 홀더가 안정되면 config/cup_holder_poses_arm4090.yaml 을 갱신",
             ["python3", "{repo}/scripts/nodes/cup_holder_pose_node.py", "--write"], background=True),
        _cmd("★컵홀더 확인 — ok: true · 세 홀더 x ≈ 같은 값(0.39 부근) · y 간격 ~0.12 · stable ✓ 인가 "
             f"(어긋나면 {RECALIB_HINT} · 겹친 영상은 scripts/calib/cup_holder_pose.py --png)",
             ["bash", "-lc", "timeout 15 ros2 topic echo --once /cup_holders/status std_msgs/msg/String"], manual=True),
    ] if real else [_cmd("fake — 카메라 없음", ["bash", "-lc", "true"])]
    run["head_home"] = [
        # ★10.04: head_home.py 는 설정 목표(pan 2015)로 검증하는데 이 머리 pan 은 목표보다 18 틱 앞에서 멈춰 늘 ✗(19:35 실기 단계
        #   실패). 카메라가 맞는 자세는 그 목표가 아니라 캘리브 때 멈춘 자리(외부 파라미터 head_pose) — 거기로 맞추고 거기로 검증한다.
        _cmd("머리를 카메라 캘리브 자세로 — 토크 · 모드 · I 게인(RAM, 전원을 끄면 사라진다)이 다르면 먼저 적용하고, 외부 파라미터 "
             "head_pose 로 목표를 고쳐 가며 맞춘 뒤 ±4 틱으로 검증(머리가 조금 움직인다)",
             HEAD_CHECK_ARGV + ["--home"], execute_args=["--execute"]),
    ] if real else [_cmd("fake — 머리 없음", ["bash", "-lc", "true"])]
    run["cups"] = [
        _cmd("★컵이 테이블에 똑바로 서 있고 손이 가리지 않는가 · arm4090 GPU 여유가 있는가(nvidia-smi — 학습이 돌면 VRAM 이 "
             "모자랄 수 있다). 머리 자세는 FP++ 를 켜기 전에 이 단계가 캘리브 자세로 맞춘다",
             ["bash", "-lc", "nvidia-smi --query-compute-apps=pid,used_memory --format=csv; "
                             "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"], manual=True),
        _cmd("인지 런처(이 PC) — 카메라 · FP++ 컨테이너를 같은 PC 의 스크립트로 켜고 끈다. 스스로는 아무것도 켜지 않는다",
             ["python3", "{repo}/scripts/nodes/perception_launcher_node.py", "--host", "local"], background=True),
        _cmd("FP++ 자세 수신기 — 영상 · FP++ 는 localhost 전용 DDS 에서 돌고 자세만 UDP(127.0.0.1)로 넘어온다",
             ["python3", "{repo}/scripts/nodes/fpp_pose_rx.py"], background=True),
        _cmd(f"물체 자세 → base — arm4090 테이블 CAD 캘리브 외부 파라미터(+ depth z 보정). /objects/<컵>/pose({_cups_txt()})",
             ["python3", "{repo}/scripts/nodes/object_pose_node.py", "--objects", *_cups(),
              "--camera-extrinsics", "{repo}/" + CAMERA_EXTRINSICS], background=True),
        _cmd("FP++ 전 머리 자세 확인 · 맞춤 — 카메라 외부 파라미터를 잰 자세(head_pose)와 다르면 그 자세로 맞춘다(머리가 조금 움직인다)", HEAD_CHECK_ARGV, execute_args=["--execute"]),
        _cmd(f"카메라 + FP++ 켜기 — 컨테이너 fpp_cups 하나가 {_cups_txt()} 를 차례로 한 번 찍는다. 이미 떠 있으면 그대로. "
             "최대 150 s, 실패하면 이 단계도 실패",
             ["python3", "{repo}/scripts/ops/perception_ctl.py", "start", *_cups(), "--wait", "150"]),
        _cmd("컵 좌표 추출 — 처음엔 FP++ 가 켜지며 찍는 회차를 기다리고, 이 단계를 다시 실행하면(컵을 옮긴 뒤 '↶ 여기서 다시') "
             "컨테이너를 끄지 않고 지금 카메라로 다시 찍는다(10.08 사용자). 컵마다 base 좌표 · 판정을 찍고, 정책이 읽는 컵을 못 찾으면 실패",
             ["python3", "{repo}/scripts/ops/fpp_rescan.py", *(REAL_CUPS[s] for s in SIDES), "--wait", "120"]),
        _cmd(f"★컵 자세 확인(바로 위 추출 결과) — 테이블 위 컵 원점 z ≈ {0.205 + REAL_CUP_ORIGIN_Z:.3f}(상판 0.205 + 원점 {REAL_CUP_ORIGIN_Z}, ±8 mm) · 기울기 < 3° · "
             f"x 0.1~0.4 · |y| 0.1~0.3 · 왼쪽(y > 0) = 파랑 {REAL_CUPS['left']} · 오른쪽(y < 0) = 노랑 {REAL_CUPS['right']} 인가"
             f"(카메라를 건드렸으면 {RECALIB_HINT})",
             ["bash", "-lc", "true"], manual=True),
    ] if real else [
        _cmd(f"fake 컵({role}) — /objects/cup_{role}/pose", ["python3", "{repo}/scripts/fakes/fake_cup_pose_pub.py", "--x", "0.38",
                                                             "--y", y, "--z", "0.264865", "--topic", f"/objects/cup_{role}/pose"],
             background=True) for role, y in (("src", "-0.16"), ("rcv", "0.16"))]
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
        rec = _record_start(f"aglt_{s}", [s], [f"/objects/{REAL_CUPS[s]}/pose"]) if real else []
        assert len(rec) + 1 == AGLT_NODE_IDX[real]
        run[f"policy_aglt_{s}"] = [
            _cmd(f"★[{s}] 컵이 학습 배치(로봇 앞 x ≈ 0.25, y ≈ {'−' if s == 'right' else '+'}0.20 ± 0.1)에 서 있고 콘솔에 컵 자세"
                 f"(/objects/{REAL_CUPS[s]}/pose, {'노랑' if s == 'right' else '파랑'})가 들어오는가. 이 팔 · 손 주변과 컵 위 20 cm 가 비어 있는가. 정책은 팔을 스스로 움직이고 "
                 "컵을 쥐어 든다 — 빈 컵만" if real else "fake — 확인만", ["bash", "-lc", "true"], manual=True),
            *rec,
            _cmd(f"[{s}] rh_aglt 정책 노드(LSTM · CPU) — start 전에는 아무것도 보내지 않는다. 처음 10 스텝은 팔을 시작 자세 · 손을 편 채(학습 hold)",
                 ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_aglt_node.py", "--ros-args",
                  # 팔마다 이름 · 에피소드를 가른다 — 양팔 정책을 한 세션에서 동시에 띄워도 서로의 reset · stop 이 섞이지 않는다(09.30)
                  "-r", f"__node:=rh_aglt_node_{s}", "-p", f"ns:={s}",
                  "-p", f"contract:={{artifact:aglt_{s}}}", "-p", f"robot:={{artifact:robot_{s}}}", "-p", "device:=cpu",
                  "-p", f"cup_topic:=/objects/{REAL_CUPS[s] if real else 'cup_' + cup}/pose",
                  # ★10.04 실기: 손이 다가가자 FP++ 가 1.5~2.6 s 끊겨 집기 전에 멈췄다 — reset 때 컵을 잡아 두고(정지 컵) 쥐면 FK
                  "-p", "cup_latch:=true", "-p", "cup_static:=true",
                  "-p", f"max_episode_s:={{policy:aglt_{s}.max_episode_s}}"],
                 background=True),
            _cmd(f"[{s}] episode reset — 컵 · 팔 측정이 있어야 받는다. 목표 = 지금 컵 + (0, 0, 0.14)",
                 ["python3", f"{PC}/tools/trigger.py", "episode/reset", "--episode-ns", s], execute_args=["--execute"]),
            _cmd(f"★[{s}] episode start — 팔이 시작 자세 0.15 rad 안 · 컵이 서 있어야 받는다",
                 ["python3", f"{PC}/tools/trigger.py", "episode/start", "--episode-ns", s], execute_args=["--execute"]),
            _cmd("★관찰 — 이상하면 정지 바의 '에피소드 정지'", ["bash", "-lc", "true"], manual=True),
            _cmd("episode stop — pd 가 그 자세 · 손 쥠을 붙잡는다", ["python3", f"{PC}/tools/trigger.py", "episode/stop", "--episode-ns", s],
                 execute_args=["--execute"]),
            _cmd("정책 노드 정지", stop=[f"policy_aglt_{s}#{AGLT_NODE_IDX[real]}"]),
            *(_record_stop(f"policy_aglt_{s}", 2) if real else []),
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
                  "-p", f"cup_topic:=/objects/{REAL_CUPS[s] if real else 'cup_' + ('src' if s == 'right' else 'rcv')}/pose",
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
        hand = [_cmd(f"★[{s}] 손 EtherCAT 확인 — 손 전원 · 랜 케이블(손 하나 = NIC 하나, 오른손 USB-C 랜 · 왼손 내장 랜 — "
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
    for name in EPISODES:
        run[f"episode_{name}"] = _episode_run(name, real)
    # 에피소드 단계가 도중에 실패하면 CPU 기록기가 남는다 — shutdown 의 정지 목록에 더한다(10.08 리뷰)
    cpu = [f"episode_{name}#{i}" for name in EPISODES for i, c in enumerate(run[f"episode_{name}"])
           if "proc_cpu_record.py" in " ".join(map(str, c.get("argv") or ()))]
    if cpu:
        stop = next(c for c in run["shutdown"] if c.get("stop") and "policy_pourfj#1" in c["stop"])
        stop["stop"] = [*stop["stop"], *cpu]
    return run


def _episode(name: str):
    sys.path.insert(0, str(REPO / "deploy" / "policy_control"))
    from policy_control import episode_spec as S
    return S.load(REPO / "config" / "episodes" / f"{name}.yaml")


def _episode_run(name: str, real: bool) -> list:
    """에피소드 단계: (fake 컵 · 홀더) → 정책 노드들 → episode_runner_node → 상황판에서 진행(수동 확인) → 내린다."""
    from policy_control import policy_registry as R
    ep = _episode(name)
    entries = {e.id: e for e in R.scan(REPO / "deploy" / "policies", deep=False)}
    cmds, bg = [], []
    if not real:
        for oname, obj in ep.objects.items():
            side = next((ep.policies[j.role].side for n in ep.nodes for j in n.jobs if j.target_object == oname), "right")
            cmds.append(_cmd(f"fake 물체 — {obj['topic']}(aglt 학습 배치 x 0.25)",
                             ["python3", "{repo}/scripts/fakes/fake_cup_pose_pub.py", "--x", "0.25",
                              "--y", "0.20" if side == "left" else "-0.20", "--z", "0.29", "--topic", obj["topic"]],
                             background=True))
        for hid in sorted(set(ep.holders.values())):
            cmds.append(_cmd(f"fake 홀더 {hid} — /objects/cup_holder_{hid}/pose (latched)",
                             ["python3", "{repo}/scripts/fakes/fake_cup_pose_pub.py", "--latched", "--rate", "2", "--x", "0.38",
                              "--y", str(FAKE_HOLDER_Y[hid]), "--z", "0.235", "--topic", f"/objects/cup_holder_{hid}/pose"],
                             background=True))
    first_obj = next(iter(ep.objects))
    sides = sorted({b.side for b in ep.policies.values()})
    left = "|".join(f"(rh_aglt_node|rh_place_node)_{s}" for s in sides)
    cmds.append(_cmd("시작 검사 — 정책이 등록부에서 쓸 수 있고" + (f" 고정 홀더 자세({ep.holder_poses})에 쓰는 홀더가 다 있는가"
                     if real else "") + " (읽기만)",
                     ["python3", "{repo}/deploy/policy_control/tools/episode_run.py", "--episode", f"config/episodes/{name}.yaml",
                      "--plan", *(["--require-holders"] if real else [])]))
    cmds.append(_cmd("시작 검사 — 같은 이름의 정책 노드 · 실행기가 남아 있지 않은가(다른 정책 단계가 띄운 것) (읽기만)",
                     ["bash", "-lc", f"! ros2 node list 2>/dev/null | grep -E '^/({left}|episode_runner)$'"]))
    cmds.append(_cmd(f"★[에피소드 {name}] 물체({', '.join(ep.objects)})가 학습 배치에 서 있고 FP++ 가 잡고 있는가 · 홀더 "
                     f"{', '.join(f'{k}={v}' for k, v in ep.holders.items())} 위가 비었는가 · {ep.holder_poses} 가 있는가"
                     "(cup_holders 단계 --write). 노드를 띄운 뒤에는 상황판 에피소드 패널에서 진행한다" if real else "fake — 확인만",
                     ["bash", "-lc", "true"], manual=True))
    for role, b in ep.policies.items():
        e = entries.get(b.policy)
        contract = f"{{repo}}/deploy/policies/{b.policy}/{e.contract if e else 'missing.json'}"
        robot = f"{{artifact:robot_{b.side}}}"
        if b.kind == "aglt":
            obj = next((j.target_object for n in ep.nodes for j in n.jobs if j.role == role and j.target_object), first_obj)
            # 컵 = snapshot 정지 기록 재발행 → 붙이기는 파지 시작 시각의 손바닥 FK 로(cup_static, cup_attach static)
            cmds.append(_cmd(f"[{b.side}] {role} 노드({b.policy}) — SETTING 도달에 스스로 끝남 · 컵 = snapshot 재발행"
                             " · 쥐면 파지 시작 자세로 손에 붙임",
                             ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_aglt_node.py", "--ros-args",
                              "-r", f"__node:=rh_aglt_node_{b.side}", "-p", f"ns:={b.side}", "-p", f"contract:={contract}",
                              "-p", f"robot:={robot}", "-p", "device:=cpu", "-p", "stop_on_target:=true",
                              "-p", f"cup_topic:={EPISODE_RELAY.format(obj)}", "-p", "cup_static:=true", "-p", "cup_latch:=true",
                              "-p", "max_episode_s:=15.0"], background=True))
        elif b.kind == "place":
            src = next((j for n in ep.nodes for j in n.jobs if j.role == role), None)
            obj = src.source_object if src and src.source_object else first_obj
            hid = ep.holders.get(src.target_holder) if src and src.target_holder else PLACE_HOLDER[b.side]
            cmds.append(_cmd(f"[{b.side}] {role} 노드({b.policy}) — 홀더 {hid} · 놓은 뒤 스스로 끝남",
                             ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_place_node.py", "--ros-args",
                              # 한 팔에 aglt 와 같이 뜬다 — 서비스는 <팔>_place, 이벤트는 그 팔 토픽(pd 가 reset · stop 적용)
                              "-r", f"__node:=rh_place_node_{b.side}", "-p", f"ns:={b.side}_place",
                              "-p", f"episode_topic:=/policy_control/{b.side}/episode", "-p", f"contract:={contract}",
                              "-p", f"robot:={robot}", "-p", "device:=cpu", "-p", f"cup_topic:={ep.objects[obj]['topic']}",
                              "-p", f"holder:={hid}", *([] if real else ["-p", "require_grasp:=false"])], background=True))
    if real:                                         # 첫 실기는 놓음 문턱(손끝 · 관절 힘 · 손 목표)을 bag 으로 정한다(PLACE 10.04)
        extra = [f"/policy_control/{x}" for x in ("joint_target", *(f"status/pd_{s}" for s in sides),
                                                   *(f"status/rh_aglt_node_{s}" for s in sides),
                                                   *(f"status/rh_place_node_{s}" for s in sides), "status/episode_runner",
                                                   *(f"{s}/episode" for s in sides))]
        extra += [EPISODE_RELAY.format(o) for o in ep.objects] + [o["topic"] for o in ep.objects.values()]
        extra += [f"/objects/cup_holder_{h}/pose" for h in sorted(set(ep.holders.values()))]
        extra += [f"/policy_control/pd_{s}/applied" for s in sides]       # 10.08 팔 지연 도구(arm_latency_report)가 쓴다
        cmds.append(_cmd("기록 시작 — 팔 · 손 bag(rh56f1_record.sh) + 정책 · 실행기 상태 · 컵 · 홀더",
                         ["bash", "-lc", f"EXTRA='{' '.join(extra)}' bash {{repo}}/deploy/policy_control/tools/rh56f1_record.sh "
                                         f"start episode_{name}"]))
        cmds.append(_cpu_record(f"episode_{name}"))
    cmds.append(_cmd(f"에피소드 실행기 — {name} · 승인은 상황판(노드마다 이름 · 연속 실행은 episode:{name})",
                     ["{repo}/.venv/bin/python", f"{PC}/policy_control/episode_runner_node.py", "--ros-args",
                      "-p", f"episode:=config/episodes/{name}.yaml", "-p", "robot:=rh56f1"], background=True))
    cmds.append(_cmd("★에피소드 진행 — 상황판 에피소드 패널의 [다음](구분 실행) · [연속 실행]. 성공 · 정지로 끝나면 확인 "
                     "(이상하면 '에피소드 실행기 정지' · 정지 바)", ["bash", "-lc", "true"], manual=True))
    bg = [i for i, c in enumerate(cmds) if c.get("background")]
    cmds.append(_cmd("에피소드 노드들 정지", stop=[f"episode_{name}#{i}" for i in reversed(bg)]))
    if real:
        cmds.append(_cmd("기록 끝 — bag 두 개 마무리(SIGINT)",
                         ["bash", "{repo}/deploy/policy_control/tools/rh56f1_record.sh", "stop"]))
    return cmds


#: 단독 aglt 단계에서 정책 노드 명령 번호 — 실기는 앞에 기록 시작(bag) · CPU 기록이 붙는다
AGLT_NODE_IDX = {True: 3, False: 1}


def _cpu_record(name: str) -> dict:
    """프로세스별 CPU 1 Hz CSV(tools/proc_cpu_record.py) — 10.03 은 정책이 15 s 에 멈춰 추론 부하를 못 쟀다(10.08 실기 전 세팅)."""
    return _cmd("CPU 기록 — 프로세스별 코어 1 Hz → logs/cpu(끝날 때 요약)",
                ["python3", f"{PC}/tools/proc_cpu_record.py", "--out", f"{{repo}}/logs/cpu/{name}.csv"], background=True)


def _record_start(name: str, sides: list, topics: list) -> list[dict]:
    """단독 정책 단계 기록(실기) — bag(팔 · 손) + CPU. 팔 지연 도구가 쓰는 pd applied · 목표 · status 를 같이 싣는다(10.08)."""
    extra = ["/policy_control/joint_target", *(f"/policy_control/{x}" for s in sides for x in
                                                (f"status/pd_{s}", f"status/rh_aglt_node_{s}", f"{s}/episode", f"pd_{s}/applied")),
             *topics]
    return [_cmd(f"기록 시작 — 팔 · 손 bag(rh56f1_record.sh {name}) · 팔 지연은 tools/arm_latency_report.py <bag>/arm",
                 ["bash", "-lc", f"EXTRA='{' '.join(extra)}' bash {{repo}}/deploy/policy_control/tools/rh56f1_record.sh start {name}"]),
            _cpu_record(name)]


def _record_stop(stage: str, cpu_index: int) -> list[dict]:
    return [_cmd("CPU 기록 끝(요약은 그 단계 로그)", stop=[f"{stage}#{cpu_index}"]),
            _cmd("기록 끝 — bag 두 개 마무리(SIGINT)", ["bash", "{repo}/deploy/policy_control/tools/rh56f1_record.sh", "stop"])]


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
        # 10.01 사용자: 기본 = iter_10 ②(실측 지연 적응) i10d 좌우. 이전 기본은 우 mirror_l5 · 좌 i05(09.30).
        # 10.04 cyl60g(FP++ 지각 · 파지 후 부착으로 학습) — 계약은 i10d 와 체크포인트 외 같아 홈 · 저장 경로는 그대로
        # ★10.06 env17(T2R Grasping 새 s2r 후보, 보상 iter_17 · 쥔 높이 컵 중심 위 +2.0 cm, cyl60g +4.3 cm) — 계약은 cyl60g 와 체크포인트 외 같다.
        #   단독 aglt 점검 자리만 바꾼다. 에피소드(config/episodes/*.yaml)는 cyl60g 그대로 — 놓기 i09 · i01 시작 뱅크(our_source/place_bank/
        #   bank_{r,l}_cyl60_keep.npz)의 쥔 높이가 p5~p95 4.3~5.3 cm(우) · 3.1~5.5 cm(좌), 3 cm 아래 0 %(우) · 4 %(좌)다(10.06 실측)
        #   ★10.08 기본 = 손 어드민턴스 다지 파지 최종(T2R Grasping, 사용자 "이 정책으로 실기 테스트") — 계약 hand_command admittance 로
        #   pd 가 손 목표를 /hand_<s>/angle_target 으로 보낸다. env17 · env17f(위치 제어)는 첫 화면에서 고를 수 있다.
        "aglt_right": "deploy/policies/rh56f1/aglt/right_g5362b/rh_aglt_contract.json",
        #   10.06 왼팔은 env17 거울(left_env17mir) 대신 그 거울을 왼팔 env 에서 200 epoch 이어 학습한 left_env17f(T2R Grasping)
        "aglt_left": "deploy/policies/rh56f1/aglt/left_g5362/rh_aglt_contract.json",
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
