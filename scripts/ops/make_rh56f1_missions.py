#!/usr/bin/env python3
"""RH56F1 로봇(arm4090) 미션 — 실기 · fake 두 벌을 **한 정의에서** 만든다.

    python3 scripts/ops/make_rh56f1_missions.py            # config/mission_rh56f1_{control,fake}.yaml 을 다시 쓴다
    python3 scripts/ops/make_rh56f1_missions.py --check    # 커밋된 두 파일이 이 정의에서 나온 것인지(테스트가 부른다)

09.29 사용자: "sim2real 과 robot_control 쪽에서 rh56f1 제어 part 연결" · 손마다 개별 포트 · USB RS485 / CANFD 를
상황에 따라 바꿔 쓴다 · 정책은 pour_fj(양팔) 다음 rh_aglt — 곧 나온다.
범위 = 손 연결 · 점검 · 한 축 방향 확인 · 팔 pd(무발행 → 발행) · 홈 · 두 컵 · 양팔 pour_fj 정책 · 정리.
★실기에서 막아 둔 것(blocked — 이유가 화면에 보인다): 홈(RH56F1 손 기하로 홈 경로를 계획하기 전 — DG-5F 저장 경로를 쓰지 않고,
  검사 없는 goto_home 직선도 쓰지 않는다) · 두 컵 자세(arm4090 인지 연결 전). fake 는 goto_home · fake 컵으로 끝까지 돈다.

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
        {"id": "cups", "group": "connect", "lane": "both", "needs": ["preflight"], "skippable": True,
         "title": ("두 컵 자세 — 붓는 컵 /objects/cup_src/pose · 받는 컵 /objects/cup_rcv/pose (base)" if real else
                   "fake 컵 두 개 — 학습 배치 중심(0.38, ∓0.16), 테이블 위에 선 채"),
         **({"blocked": "arm4090 인지(카메라 · FP++ 두 물체) 연결 전 — 컵 두 개의 자세를 /objects/cup_src · cup_rcv 로 내는 단계가 없다"}
            if real else {})},
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
             "touches_real": real, "artifacts": ["contract"],
             "title": f"[{s}] engage → pour_fj 시작 자세(홈) — 손은 편 손",
             **({"blocked": "RH56F1 홈 경로 계획 전 — plan_home_path 를 RH56F1 손 기하로 돌려 저장 경로를 만든 뒤(검사 없는 "
                            "goto_home 직선으로 차렷에서 홈까지 가지 않는다)"} if real else {})},
            {"id": f"release_{s}", "group": "finish", "lane": f"arm_{s}", "needs": [f"pd_arm_{s}"], "skippable": True,
             "touches_real": real, "undoes": [f"pd_arm_{s}", f"home_{s}"],
             "title": f"[{s}] pd 해제(역블렌드 → JTC) → 이 팔의 pd 정지"},
        ]
    st.append({"id": "policy_pourfj", "group": "policy", "lane": "both", "skippable": True, "touches_real": real,
               "needs": ["home_right", "home_left", "cups", "hand_check_right", "hand_check_left"],
               "needs_why": "정책은 양팔이 pour_fj 시작 자세 · 손이 편 채 · 두 컵이 선 채로만 출발해 봤다",
               "artifacts": ["pourfj_both", "robot_bi", "contract"],
               "title": "[양팔] pour_fj 정책(첫 화면에서 고른 것) — 정책 노드 → reset → start → 관찰 → stop (시간이 되면 스스로 끝난다)"})
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
            _cmd("RH56F1 변환표 · 백엔드 · 상태 노드 · 계약 · 미션 테스트(수 초)",
                 ["python3", "-m", "pytest", "-q", "-m", "not gpu", "-p", "no:cacheprovider",
                  "{repo}/tests/policy_control/test_pc_rh56f1.py", "{repo}/tests/s2r_console/test_rh56f1_mission.py"]),
            _cmd("제어 전용 계약 재생성 — 자산 openarm_rh56f1_bi_rl, 홈 = hdgp pour_fj 리셋 홈(양팔, config/homes), 손 = 편 손",
                 ["python3", f"{PC}/tools/build_deploy_contract.py", "--asset", "openarm_rh56f1_bi_rl",
                  "--home", "arms:config/homes/rh56f1_pour_fj.yaml", "--out", "{artifact:contract}"]),
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
                  # 차렷(0)에서 시작하면 j4 가 하한 0 에 붙어 pd 가 '한계 밖 목표'로 HOLD 한다(09.29 fake) — j4 만 0.1 안쪽
                  "arm_start:=right=0,0,0,0.1,0,0,0;left=0,0,0,0.1,0,0,0"], background=True),
        ]),
        "shutdown": [
            _cmd("★양팔을 받침 위 · 안전 자세에 두었는가. 두 팔 모두 pd 해제를 끝냈는가. 다음 스텝부터 토크가 풀린다",
                 ["bash", "-lc", "true"], manual=True),
            _cmd("남은 정책 노드 · pd 정지", stop=["policy_pourfj#1"] + [f"{p}_{s}#{i}" for s in SIDES for p, i in (("pd_load", 0), ("pd_arm", 1))]),
            _cmd("손 상태 노드 · 손 드라이버 정지 — 손가락은 마지막 자세에서 멈춘다(벤더 펌웨어가 잡는다)",
                 stop=[f"hand_{s}#{i}" for s in SIDES for i in ((2, 1) if real else (1, 0))]),
            _cmd("★팔 브링업 정지 — 모든 팔 모터가 꺼진다(받침으로 내려앉는다)" if real else "fake 플랜트 정지",
                 stop=["drivers#3" if real else "drivers#0"]),
        ],
    }
    run["cups"] = [] if real else [
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
    for s in SIDES:
        hand = [_cmd(f"★[{s}] 손 포트 확인 — ls -l /dev/serial/by-id · 설정 deploy/policy_control/config/rh56f1_ports.yaml "
                     "(transport rs485 | canfd · port · hand_id). 손 전원이 켜져 있는가. 드라이버는 시작할 때 쓰기를 하지 않는다",
                     ["bash", "-lc", "ls -l /dev/serial/by-id 2>/dev/null; cat {repo}/deploy/policy_control/config/rh56f1_ports.yaml"],
                     manual=True),
                _cmd(f"[{s}] RH56F1 드라이버 — 읽기만 도는 벤더 노드(명령 토픽이 오기 전에는 쓰지 않는다)",
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
            f"home_{s}": [] if real else [
                _cmd(f"[{s}] pd engage", ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_engage"],
                     execute_args=["--execute", "--approve", "pd_engage"]),
                _cmd(f"[{s}] pd goto_home — pour_fj 시작 자세(fake 에서만 직선)",
                     ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_goto_home", "--service-timeout", "45"],
                     execute_args=["--execute", "--approve", "pd_goto_home"])],
            f"release_{s}": [
                _cmd(f"[{s}] pd release — 역블렌드 → 0 송출 → JTC 복귀",
                     ["python3", f"{PC}/tools/episode_ctl.py", "--side", s, "--only", "pd_release"], execute_args=["--execute"]),
                _cmd(f"[{s}] 이 팔의 pd 정지 — IDLE 이라 토크는 JTC 가 잡는다", stop=[f"pd_arm_{s}#1"])],
        })
    return run


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
        # 양팔 붓기 정책(첫 화면의 '양팔' 자리가 바꾼다) · 양팔 robot yaml
        "pourfj_both": "deploy/policies/both_rh_pourfj_f01/pour_fj_contract.json",
        "robot_bi": f"deploy/policy_control/config/robots/rh56f1_bi_{robot}.yaml",
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
        "lanes": [{"id": "rig", "title": "드라이버"}, {"id": "arm_right", "title": "오른팔 · 오른손", "side": "right"},
                  {"id": "arm_left", "title": "왼팔 · 왼손", "side": "left"},
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
