#!/usr/bin/env python3
"""RH56F1 로봇(arm4090) 미션 — 실기 · fake 두 벌을 **한 정의에서** 만든다.

    python3 scripts/ops/make_rh56f1_missions.py            # config/mission_rh56f1_{control,fake}.yaml 을 다시 쓴다
    python3 scripts/ops/make_rh56f1_missions.py --check    # 커밋된 두 파일이 이 정의에서 나온 것인지(테스트가 부른다)

09.29 사용자: "sim2real 과 robot_control 쪽에서 rh56f1 제어 part 연결" · 손마다 개별 포트 · USB RS485 / CANFD 를
상황에 따라 바꿔 쓴다 · 정책은 pour_fj(양팔) 다음 rh_aglt — 곧 나온다.
범위 = 손 연결 · 점검 · 한 축 방향 확인 · 팔 pd(무발행 → 발행) · 홈 · 두 컵 · 양팔 pour_fj 정책 · 정리.
홈 = hdgp rh_aglt 시작 자세(09.29 사용자 "aglt 보면 home 자세를 수정했어"). 차렷 → 홈은 저장 경로(paths/home_rh56f1_*.npz,
plan_home_path RRT · RH56F1 자산 충돌 검사 · 편 손/접은 손 둘 다)를 pd 로 재생한다 — 실기 · fake 같은 경로.
★10.01 인지는 arm4090 안에서(docker FP++ · RealSense · 런처 --host local). 실기 컵은 하나(10.04 부터 cyl60 노란 원통 — 그 전 aglt_cup_s065, rh_aglt 학습 컵
shaker_closed_thick × 0.65 의 흰 출력물, 10.01 사용자) — 한 팔 rh_aglt 는 /objects/<REAL_CUP>/pose 를 읽고, 두 컵이 필요한
양팔 pour_fj 는 막아 둔다(같은 컵 둘을 FP++ 가 못 가른다 — 색이 다른 두 컵이면 레지스트리 두 항목으로 풀린다).
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
REAL_CUP_ORIGIN_Z = 0.085                 # 원점 높이(바닥 위) — 컵 자세 확인 문구
#: arm4090 머리 카메라 외부 파라미터 — 테이블 CAD 캘리브(scripts/calib/table_cad_extrinsics.py, 10.01 기본 방법)
CAMERA_EXTRINSICS = "config/global_camera_extrinsics_arm4090.yaml"
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
         "title": ("머리 기준자세(head_home_rh56f1 = 5090 홈과 같은 화면) + I 게인 — 카메라 좌표가 이 자세에서만 맞다(머리가 조금 움직인다)"
                   if real else "fake — 머리 없음(실기 순서를 맞추려고 둔 자리)")},
        {"id": "cups", "group": "connect", "lane": "both", "needs": ["head_home"], "skippable": True,
         "touches_real": real,
         "title": (f"컵 자세(arm4090 FP++) — RealSense · FP++ 컨테이너({REAL_CUP}) → /objects/{REAL_CUP}/pose (base). GPU VRAM 수 GB" if real else
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
               **({"blocked": f"두 컵 구분 전 — 실기 인지는 컵 하나({REAL_CUP})만 낸다. 같은 컵 둘은 FP++ 가 못 가른다"
                             "(색이 다른 컵 · 자리로 가르기 중 하나가 필요)"} if real else {}),
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
            _cmd("남은 정책 노드 · pd 정지", stop=["policy_pourfj#1"] + [f"policy_aglt_{s}#1" for s in SIDES]
                 + [f"{p}_{s}#{i}" for s in SIDES for p, i in (("pd_load", 0), ("pd_arm", 1))]),
            *([_cmd("카메라 · FP++ 내리기 — 런처가 없으면 이 PC 에서 직접 내린다",
                    ["python3", "{repo}/scripts/ops/perception_ctl.py", "stop", "--camera", "--host", "local", "--wait", "60"]),
               _cmd("인지 런처 · 자세 수신기 · 물체 자세 노드 · 컵홀더 노드 정지",
                    stop=["cups#1", "cups#2", "cups#3", "cup_holders#2"])] if real else []),
            _cmd("손 상태 노드 · 손 드라이버 정지 — 손가락은 마지막 자세에서 멈춘다(벤더 펌웨어가 잡는다)",
                 stop=[f"hand_{s}#{i}" for s in SIDES for i in ((2, 1) if real else (1, 0))]),
            _cmd("★팔 브링업 정지 — 모든 팔 모터가 꺼진다(받침으로 내려앉는다)" if real else "fake 플랜트 정지",
                 stop=["drivers#3" if real else "drivers#0"]),
        ],
    }
    run["cup_holders"] = [
        _cmd("★머리가 기준자세인가(head_home 을 했는가 — 외부 파라미터가 그 자세에서만 맞다) · 홀더 세 개가 상판 위에 서 있고 "
             "-x 면 마커가 손 · 컵에 가리지 않는가", ["bash", "-lc", "true"], manual=True),
        _cmd("카메라(RealSense) 켜기 — 이미 떠 있으면 그대로(cups 단계와 같이 써도 된다)",
             ["bash", "{repo}/scripts/vision/camera_up.sh"]),
        _cmd("컵홀더 자세 노드 — ArUco → 직전 자세 추적(0.2 s) → 놓친 id 만 무늬 전체 탐색(첫 장 ~6 s). "
             "x 공유 · 상판 z 고정 · 5장 중앙값, 세 홀더가 안정되면 config/cup_holder_poses_arm4090.yaml 을 갱신",
             ["python3", "{repo}/scripts/nodes/cup_holder_pose_node.py", "--write"], background=True),
        _cmd("★컵홀더 확인 — ok: true · 세 홀더 x ≈ 같은 값(0.39 부근) · y 간격 ~0.12 · stable ✓ 인가 "
             "(어긋나면 head_home 뒤 scripts/calib/table_cad_extrinsics.py · 겹친 영상은 scripts/calib/cup_holder_pose.py --png)",
             ["bash", "-lc", "timeout 15 ros2 topic echo --once /cup_holders/status std_msgs/msg/String"], manual=True),
    ] if real else [_cmd("fake — 카메라 없음", ["bash", "-lc", "true"])]
    run["head_home"] = [
        _cmd("머리 기준자세 + I 게인(RAM — 전원을 끄면 사라진다) — arm4090 머리는 5090 과 숫자가 다르다(config/head_home_rh56f1.yaml)",
             ["python3", "{repo}/scripts/head_home.py", "--config", "{repo}/config/head_home_rh56f1.yaml"],
             execute_args=["--execute"])] if real else [_cmd("fake — 머리 없음", ["bash", "-lc", "true"])]
    run["cups"] = [
        _cmd("★머리가 기준자세인가(head_home 을 했는가) · 컵이 테이블에 똑바로 서 있고 손이 가리지 않는가 · arm4090 GPU 여유가 "
             "있는가(nvidia-smi — 학습이 돌면 VRAM 이 모자랄 수 있다)",
             ["bash", "-lc", "nvidia-smi --query-compute-apps=pid,used_memory --format=csv; "
                             "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"], manual=True),
        _cmd("인지 런처(이 PC) — 카메라 · FP++ 컨테이너를 같은 PC 의 스크립트로 켜고 끈다. 스스로는 아무것도 켜지 않는다",
             ["python3", "{repo}/scripts/nodes/perception_launcher_node.py", "--host", "local"], background=True),
        _cmd("FP++ 자세 수신기 — 영상 · FP++ 는 localhost 전용 DDS 에서 돌고 자세만 UDP(127.0.0.1)로 넘어온다",
             ["python3", "{repo}/scripts/nodes/fpp_pose_rx.py"], background=True),
        _cmd(f"물체 자세 → base — arm4090 테이블 CAD 캘리브 외부 파라미터(+ depth z 보정). /objects/{REAL_CUP}/pose",
             ["python3", "{repo}/scripts/nodes/object_pose_node.py", "--objects", REAL_CUP,
              "--camera-extrinsics", "{repo}/" + CAMERA_EXTRINSICS], background=True),
        _cmd(f"카메라 + FP++({REAL_CUP}) 켜기 — 런처가 끝낼 때까지 최대 150 s, 실패하면 이 단계도 실패",
             ["python3", "{repo}/scripts/ops/perception_ctl.py", "start", REAL_CUP, "--wait", "150"]),
        _cmd(f"★컵 자세 확인 — 테이블 위 컵 원점 z ≈ {0.205 + REAL_CUP_ORIGIN_Z:.3f}(상판 0.205 + 원점 {REAL_CUP_ORIGIN_Z}, ±8 mm) · 기울기 < 3° · "
             f"x 0.1~0.4 · |y| 0.1~0.3 인가(어긋나면 head_home 뒤 scripts/calib/table_cad_extrinsics.py)",
             ["bash", "-lc", f"timeout 5 ros2 topic echo --once /objects/{REAL_CUP}/pose geometry_msgs/msg/PoseStamped"],
             manual=True),
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
        run[f"policy_aglt_{s}"] = [
            _cmd(f"★[{s}] 컵이 학습 배치(로봇 앞 x ≈ 0.25, y ≈ {'−' if s == 'right' else '+'}0.20 ± 0.1)에 서 있고 콘솔에 컵 자세"
                 f"(/objects/{REAL_CUP}/pose)가 들어오는가. 이 팔 · 손 주변과 컵 위 20 cm 가 비어 있는가. 정책은 팔을 스스로 움직이고 "
                 "컵을 쥐어 든다 — 빈 컵만" if real else "fake — 확인만", ["bash", "-lc", "true"], manual=True),
            _cmd(f"[{s}] rh_aglt 정책 노드(LSTM · CPU) — start 전에는 아무것도 보내지 않는다. 처음 10 스텝은 팔을 시작 자세 · 손을 편 채(학습 hold)",
                 ["{repo}/.venv/bin/python", f"{PC}/policy_control/rh_aglt_node.py", "--ros-args",
                  # 팔마다 이름 · 에피소드를 가른다 — 양팔 정책을 한 세션에서 동시에 띄워도 서로의 reset · stop 이 섞이지 않는다(09.30)
                  "-r", f"__node:=rh_aglt_node_{s}", "-p", f"ns:={s}",
                  "-p", f"contract:={{artifact:aglt_{s}}}", "-p", f"robot:={{artifact:robot_{s}}}", "-p", "device:=cpu",
                  "-p", f"cup_topic:=/objects/{REAL_CUP if real else 'cup_' + cup}/pose",
                  "-p", f"max_episode_s:={{policy:aglt_{s}.max_episode_s}}"],
                 background=True),
            _cmd(f"[{s}] episode reset — 컵 · 팔 측정이 있어야 받는다. 목표 = 지금 컵 + (0, 0, 0.14)",
                 ["python3", f"{PC}/tools/trigger.py", "episode/reset", "--episode-ns", s], execute_args=["--execute"]),
            _cmd(f"★[{s}] episode start — 팔이 시작 자세 0.15 rad 안 · 컵이 서 있어야 받는다",
                 ["python3", f"{PC}/tools/trigger.py", "episode/start", "--episode-ns", s], execute_args=["--execute"]),
            _cmd("★관찰 — 이상하면 정지 바의 '에피소드 정지'", ["bash", "-lc", "true"], manual=True),
            _cmd("episode stop — pd 가 그 자세 · 손 쥠을 붙잡는다", ["python3", f"{PC}/tools/trigger.py", "episode/stop", "--episode-ns", s],
                 execute_args=["--execute"]),
            _cmd("정책 노드 정지", stop=[f"policy_aglt_{s}#1"]),
        ]
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
    return run


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
        _cmd(f"★[{s}] 팔이 홈 근처에 있고(정책이 멀리 끌고 갔으면 먼저 홈으로) 손에 컵이 없는가 · 경로 주변이 비어 있는가. "
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
        "pourfj_both": "deploy/policies/both_rh_pourfj_f01/pour_fj_contract.json",
        "robot_bi": f"deploy/policy_control/config/robots/rh56f1_bi_{robot}.yaml",
        # 한 팔 rh_aglt 정책(첫 화면의 '오른팔 · 왼팔' 자리가 바꾼다) — 09.30
        # 10.01 사용자: 기본 = iter_10 ②(실측 지연 적응) i10d 좌우. 이전 기본은 우 mirror_l5 · 좌 i05(09.30).
        # ★10.04 cyl60g(FP++ 지각 · 파지 후 부착으로 학습) — 계약은 i10d 와 체크포인트 외 같아 홈 · 저장 경로는 그대로
        "aglt_right": "deploy/policies/right_rh_aglt_cyl60g/rh_aglt_contract.json",
        "aglt_left": "deploy/policies/left_rh_aglt_cyl60gmir/rh_aglt_contract.json",
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
