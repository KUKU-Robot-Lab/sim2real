# 로봇별 실물 ↔ Isaac Sim 연동 가이드

세 로봇을 실제 하드웨어와 Isaac Sim 사이에서 연결하는 방법을 로봇별로 정리한다.
이 문서는 **실물 ↔ sim 실시간 연동**(sim2real 방향)을 다룬다. sim 안에서 정책을
학습·재생하는 것은 `hdgp` 쪽 `play.py` / task 등록이 담당하며 여기서 다루지 않는다.

## 데이터 흐름

```
정책 / Isaac Sim  →  /isaacsim/* 명령 토픽  →  isaacsim_bridge  →  ros2_control  →  실제 하드웨어
                                                     ↑
실제 하드웨어 상태  →  /joint_states, /dg5f_right/joint_states  →  병합  →  /isaacsim/joint_states
```

핵심 노드는 `isaacsim_bridge/isaacsim_bridge/bridge_node.py` 하나다. 이 노드가
`/isaacsim/*_cmd` 다섯 토픽을 구독해 각 하드웨어 컨트롤러로 전달한다.

| 구독 (입력) | 타입 | 전달 대상 (출력) |
|---|---|---|
| `/isaacsim/left_arm_cmd` | `Float64MultiArray[7]` | `/left_joint_trajectory_controller/joint_trajectory` |
| `/isaacsim/right_arm_cmd` | `Float64MultiArray[7]` | `/right_joint_trajectory_controller/joint_trajectory` |
| `/isaacsim/right_hand_cmd` | `Float64MultiArray[20]` | `/dg5f_right/dg5f_right_controller/joint_trajectory` |
| `/isaacsim/left_gripper_cmd` | `Float64` | `/left_gripper_controller/gripper_cmd` |
| `/isaacsim/emergency_stop` | `Bool` | (전 채널 정지) |

브리지는 위치 목표를 `trajectory_time_sec`(기본 0.2s) 짜리 `JointTrajectory`로 보간해
내보낸다. 이 보간은 실시간 배포에는 적절하지만, actuator 파라미터 식별에는 부적절하다
(그 경우는 `hdgp/scripts/r2s_autotune/README.md` 참조).

---

## 지원 현황 (2026-07-11 기준)

| 로봇 | 실물 드라이버 | 브리지 연동 | sim task |
|---|---|---|---|
| OpenArm (양팔 7-DOF) | 있음 (`openarm_hardware`, CAN/MIT) | 있음 | 있음 |
| Tesollo DG-5F (오른손 20-DOF) | 있음 (`delto_m_ros2/dg5f_driver`) | 있음 (`dg5f_right`) | 있음 |
| RH56F1 (양손) | **없음** | **없음** | 있음 (sim 전용) |

`isaacsim_bridge`는 OpenArm + Tesollo 오른손 + 왼쪽 gripper만 다룬다. RH56F1은 실물
제어 스택 자체가 없어 실물 연동이 불가능하다 (§3 참조).

---

## 1. OpenArm (팔)

### 하드웨어

`openarm_hardware/src/v10_simple_hardware.cpp`가 관절을 MIT 모드 PD로 구동한다.

```cpp
arm_params.push_back({kp_[i], kd_[i], pos_commands_[i], vel_commands_[i], tau_commands_[i]});
openarm_->get_arm().mit_control_all(arm_params);   // τ = kp·(q*−q) + kd·(q̇*−q̇)
```

게인은 `openarm_description/config/arm/v10/control_gains.yaml`에 있다.

| | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
|---|---|---|---|---|---|---|---|
| kp | 70 | 70 | 70 | 60 | 10 | 10 | 10 |
| kd | 2.75 | 2.5 | 2.0 | 2.0 | 0.7 | 0.6 | 0.5 |

상태: position / velocity / effort 모두 발행 (`/joint_states`).

### 브링업

```bash
cd /home/user/rl_ws/teleopration_openarm_tesollo
source /opt/ros/humble/setup.bash && source install/setup.bash

# 기본값: right_can_interface=can0, left_can_interface=can1
ros2 launch openarm_bringup openarm.bimanual.launch.py \
    robot_controller:=joint_trajectory_controller
```

`joint_trajectory_controller`가 브리지의 팔 출력 토픽과 맞물린다.
raw 위치를 직접 넣으려면 `robot_controller:=forward_position_controller`로 띄우고
`/right_forward_position_controller/commands`(`Float64MultiArray[7]`,
순서 `openarm_right_joint1..7`)로 발행한다.

### 관절 이름

`openarm_left_joint1..7`, `openarm_right_joint1..7`. `Float64MultiArray`에는 이름이
없으므로 순서를 지켜야 한다. 컨트롤러 yaml의 `joints` 목록이 기준이다.

---

## 2. Tesollo DG-5F (오른손)

### 하드웨어

`delto_m_ros2/dg5f_driver`가 20-DOF 손을 구동한다. 두 가지 명령 인터페이스가 있다.

- `/dg5f_right/dg5f_right_controller/joint_trajectory` — ros2_control JTC (브리지 출력)
- `/dg5f_right/rj_dg_pospid/reference` — 드라이버 내부 PID의 raw 레퍼런스
  (`control_msgs/MultiDOFCommand`, 100 Hz, `dof_names` 포함)

실시간 정책 배포는 위(JTC)를 쓴다. actuator 식별은 아래(raw)를 쓴다.

### 관절 순서 (20-DOF)

```
rj_dg_1_1 rj_dg_1_2 rj_dg_1_3 rj_dg_1_4   (엄지)
rj_dg_2_1 rj_dg_2_2 rj_dg_2_3 rj_dg_2_4   (검지)
rj_dg_3_1 rj_dg_3_2 rj_dg_3_3 rj_dg_3_4   (중지)
rj_dg_4_1 rj_dg_4_2 rj_dg_4_3 rj_dg_4_4   (약지)
rj_dg_5_1 rj_dg_5_2 rj_dg_5_3 rj_dg_5_4   (소지)
```

`/isaacsim/right_hand_cmd`(`Float64MultiArray[20]`)도 이 순서다.

### 브링업

목적에 따라 launch 파일이 나뉜다.

```bash
# 드라이버 + JTC (브리지의 right_hand 출력과 맞물림)
ros2 launch dg5f_driver dg5f_right_driver.launch.py

# 또는 raw PID 레퍼런스(rj_dg_pospid/reference)를 받는 컨트롤러
ros2 launch dg5f_driver dg5f_right_pid_controller.launch.py
```

상태는 `/dg5f_right/joint_states`, 힘센서는 `/tesollo/right/sensor`.

### 연결 검증 (스모크)

```bash
# 손끝을 조금 굽혔다 펴는 raw 명령 (드라이버 직접, 브리지 우회)
python3 teleopration_openarm_tesollo/src/delto_m_ros2/dg5f_driver/script/dg5f_right_pid_test.py
```

---

## 3. RH56F1 — 실물 연동 미구현

RH56F1은 **Isaac Sim task(`open-rh56f1_r_grasp_v1/v2` 등)는 있으나 실물 제어 스택이
전혀 없다.** teleop / sim2real 어디에도 드라이버·컨트롤러·브리지 매핑이 없고 URDF/USD만
있다. 따라서 지금은 sim 안에서만 다룰 수 있고, 실물 ↔ sim 연동은 불가능하다.

실물 연동을 하려면 Tesollo(`dg5f_driver`)에 대응하는 다음이 필요하다.

1. **하드웨어 드라이버** — RH56F1 통신(시리얼/CAN)을 ros2_control `SystemInterface`로 감싸는
   패키지. `delto_m_ros2/dg5f_driver`가 참고 모델.
2. **컨트롤러 설정** — 관절 순서를 정의하는 `joint_trajectory_controller` yaml.
   RH56F1은 underactuated이므로 drive/mimic 관계를 컨트롤러 또는 드라이버에서 처리해야 한다.
   sim 쪽 mimic 그룹은 `rh56f1/right/grasp_v2/grasp_right_env_cfg.py` 참조.
3. **브리지 매핑** — `bridge_node.py`에 `/isaacsim/right_hand_cmd`를 RH56F1 컨트롤러
   토픽으로 보내는 파라미터 추가. 단, RH56F1의 손 명령 차원(sim의 6-DOF drive)과 Tesollo의
   20-DOF는 다르므로 그대로 재사용할 수 없다.

canonical 관절 이름(`r_hj_*`)과 drive/mimic 그룹 정의는 이미 있으므로
(`urdf/generated/rl/openarm_bi_rh56f1_rl_manifest.yaml`), 드라이버가 생기면 그 이름 계약을
따르면 된다.

---

## 4. dry-run (하드웨어 없이 검증)

실물 없이 브리지 배선과 정책 출력을 확인하려면 fake hardware + RViz를 쓴다.
`SIM2REAL_INFERENCE.md`의 4-터미널 절차가 `5g_grasp_right_v7` 정책 기준의 완결된 예다.

```bash
# 터미널 1: fake hardware + RViz
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=true

# 터미널 2: isaacsim_bridge
ros2 run isaacsim_bridge bridge_node

# 터미널 3: dry-run 노드 (하드웨어 대신 RViz로 확인)
python3 sim2real/scripts/sim2real_dryrun.py
```

로봇별 정책·체크포인트를 바꿔가며 이 골격을 재사용한다.

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| `isaacsim_bridge/README.md` | 브리지 패키지 상세, 튜닝 리포트 |
| `isaacsim_bridge/ISAACSIM_ACTION_GRAPH.md` | Isaac Sim 쪽 Action Graph로 5토픽 발행 |
| `isaacsim_bridge/ISAACSIM_POLICY_WIRING.md` | Action Graph에 정책 출력 연결 |
| `SIM2REAL_INFERENCE.md` | OpenArm+Tesollo `5g_grasp_right_v7` 배포 전체 절차 |
| `hdgp/scripts/r2s_autotune/README.md` | 실물 응답으로 sim actuator 보정 (반대 방향) |
