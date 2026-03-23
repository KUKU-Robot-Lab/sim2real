# 새 환경에서 OpenArm/Tesollo 좌우 개별 EEF 제어 세팅 및 사용법

이 문서는 새 PC 또는 새 Ubuntu 환경에서 `sim2real` 저장소를 기준으로, 최종적으로 좌우 팔을 개별 노드로 EEF 제어하기 위해 필요한 설치 항목과 사용 순서를 정리합니다.

기준 제어 방식:

- 모델 기준: `/home/user/rl_ws/sim2real/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf`
- 오른팔 노드: [right_arm_eef_controller.cpp](/home/user/rl_ws/sim2real/openarm_eef_control/src/right_arm_eef_controller.cpp)
- 왼팔 노드: [left_arm_eef_controller.cpp](/home/user/rl_ws/sim2real/openarm_eef_control/src/left_arm_eef_controller.cpp)
- 오른팔 입력: `/openarm/right_arm/eef_target` (`geometry_msgs/msg/PoseStamped`)
- 왼팔 입력: `/openarm/left_arm/eef_target` (`geometry_msgs/msg/PoseStamped`)
- 오른팔 출력: `/right_joint_trajectory_controller/joint_trajectory` (`trajectory_msgs/msg/JointTrajectory`)
- 왼팔 출력: `/left_joint_trajectory_controller/joint_trajectory` (`trajectory_msgs/msg/JointTrajectory`)
- 오른팔 tip link: `palm_ee`
- 왼팔 tip link: `openarm_left_hand_tcp`

참고한 문서:

- [Install pyKDL_5.1.0.md](/home/user/rl_ws/sim2real/reference%20files/Install%20pyKDL_5.1.0.md)
- [OpenArm_Tesollo 기반 EEF control 적용 방법.md](/home/user/rl_ws/sim2real/reference%20files/OpenArm_Tesollo%20기반%20EEF%20control%20적용%20방법.md)

## 1. 먼저 이해할 점

이 저장소의 현재 방식은 `ur16e` 예제 방식이 아닙니다.

- UR 계열 6축 조인트 이름을 쓰지 않음
- `JointState -> joint_command` 직접 발행 방식이 아님
- OpenArm 7축 좌우 팔 조인트를 사용함
- IK 결과를 각 팔의 `JointTrajectoryController` 토픽으로 보냄

즉, 흐름은 아래입니다.

```text
목표 EEF Pose
-> IK 계산
-> openarm_right_joint1~7 값 생성
-> /right_joint_trajectory_controller/joint_trajectory publish
```

## 2. 새 환경에서 필요한 것

### 필수

- Ubuntu
- ROS 2 Humble
- `colcon`
- C++ 빌드 도구 (`build-essential`, `cmake`)
- OpenArm / Tesollo 제어에 필요한 ROS 2 패키지
- 현재 저장소의 vendor 패키지들

### Isaac Sim 연동 또는 Isaac Python 안에서 KDL을 쓸 경우 추가

- Isaac Sim 5.1.0
- Isaac Sim 내부 Python 3.11
- PyKDL

중요:

- ROS 2 실기 제어 노드 자체는 C++ `orocos_kdl`을 사용하므로, Isaac Sim용 `PyKDL`은 필수는 아닙니다.
- 다만 Isaac Sim 내부 Python에서 KDL 기반 처리를 할 계획이면 `PyKDL` 설치가 필요합니다.

## 3. ROS 2 기본 설치

ROS 2 Humble이 없으면 먼저 설치합니다.

설치 후 최소 확인:

```bash
source /opt/ros/humble/setup.bash
printenv ROS_DISTRO
```

정상이라면 `humble`이 출력되어야 합니다.

## 4. OS 패키지 설치

새 환경에서는 아래 패키지들을 먼저 넣는 편이 안전합니다.

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  ros-humble-rclcpp \
  ros-humble-sensor-msgs \
  ros-humble-geometry-msgs \
  ros-humble-trajectory-msgs \
  ros-humble-kdl-parser \
  ros-humble-orocos-kdl \
  ros-humble-urdf \
  ros-humble-xacro \
  ros-humble-robot-state-publisher \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-position-controllers \
  ros-humble-control-msgs \
  ros-humble-rviz2
```

필요하면 `rosdep`도 초기화합니다.

```bash
sudo rosdep init
rosdep update
```

## 5. 워크스페이스 준비

저장소를 원하는 경로에 둡니다.

예:

```bash
mkdir -p ~/rl_ws
cd ~/rl_ws
git clone <sim2real-repo-url> sim2real
cd sim2real
```

현재 문서 기준 워크스페이스 루트는 아래입니다.

```bash
/home/user/rl_ws/sim2real
```

## 6. 의존 패키지 확인

이 저장소는 루트 아래에 vendor 패키지를 포함하고 있습니다.

주요 포함 패키지:

- `vendor/openarm/openarm_description`
- `vendor/openarm/openarm_bringup`
- `vendor/openarm/openarm_hardware`
- `vendor/tesollo/dg5f_driver`
- `vendor/tesollo/dg_description`
- `vendor/tesollo/dg_msgs`

추가로 `openarm_eef_control` 패키지가 있어야 합니다.

- [openarm_eef_control](/home/user/rl_ws/sim2real/openarm_eef_control)

## 7. 빌드

워크스페이스에서 빌드:

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
colcon build --packages-select openarm_eef_control
source install/setup.bash
```

전체 실기 관련 패키지를 함께 빌드하려면 상황에 따라 전체 빌드를 사용할 수 있습니다.

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## 8. 실기 제어 실행 순서

### 8-1. 실기 OpenArm/Tesollo bringup 실행

오른팔 제어 토픽이 실제로 살아 있어야 하므로, 먼저 실기 런치를 올립니다.

예:

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch /home/user/rl_ws/sim2real/integrated_control/launch/openarm_left_gripper_right_dg5_real.launch.py \
  left_can_interface:=can1 \
  right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

최소 확인:

```bash
ros2 topic list | grep right_joint_trajectory_controller
ros2 topic echo /joint_states
ros2 control list_controllers
```

### 8-2. 오른팔 EEF 제어 노드 실행

다른 터미널에서:

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch openarm_eef_control right_arm_eef_control.launch.py
```

현재 기본 파라미터:

- `urdf_path`: `/home/user/rl_ws/sim2real/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf`
- `root_link`: `openarm_right_link0`
- `tip_link`: `palm_ee`
- `joint_state_topic`: `/joint_states`
- `target_pose_topic`: `/openarm/right_arm/eef_target`
- `trajectory_topic`: `/right_joint_trajectory_controller/joint_trajectory`

초기화 로그에서 아래가 보이면 정상입니다.

```text
Loaded KDL chain with 7 joints
Right arm EEF controller ready.
```

### 8-3. 목표 EEF pose 보내기

예시:

```bash
ros2 topic pub --once /openarm/right_arm/eef_target geometry_msgs/msg/PoseStamped '{
  header: {frame_id: world},
  pose: {
    position: {x: 0.45, y: -0.20, z: 0.65},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

이 명령을 보내면 노드는 현재 `/joint_states`를 seed로 사용해서 IK를 계산하고, 결과를 `/right_joint_trajectory_controller/joint_trajectory`로 publish 합니다.

### 8-4. 왼팔 EEF 제어 노드 실행

다른 터미널에서:

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch openarm_eef_control left_arm_eef_control.launch.py
```

현재 기본 파라미터:

- `urdf_path`: `/home/user/rl_ws/sim2real/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf`
- `root_link`: `openarm_left_link0`
- `tip_link`: `openarm_left_hand_tcp`
- `joint_state_topic`: `/joint_states`
- `target_pose_topic`: `/openarm/left_arm/eef_target`
- `trajectory_topic`: `/left_joint_trajectory_controller/joint_trajectory`

목표 pose 예시:

```bash
ros2 topic pub --once /openarm/left_arm/eef_target geometry_msgs/msg/PoseStamped '{
  header: {frame_id: world},
  pose: {
    position: {x: 0.45, y: 0.20, z: 0.65},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

### 8-5. GUI로 좌우 팔/그리퍼 제어

`test_gui`를 사용하면 좌우 팔과 그리퍼를 GUI에서 직접 제어할 수 있습니다.

현재 연결:

- 왼쪽 `ARM` 패널
  - EEF -> `/openarm/left_arm/eef_target`
  - Gripper -> `/left_gripper_controller/gripper_cmd`
- 오른쪽 `ARM2` 패널
  - EEF -> `/openarm/right_arm/eef_target`
  - Tesollo hand -> `/dg5f_right/dg5f_right_controller/joint_trajectory`

즉, GUI를 쓰려면 아래 4개가 올라와 있어야 합니다.

1. 실기 bringup
2. 왼팔 EEF IK 노드
3. 오른팔 EEF IK 노드
4. `test_gui`

실행:

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch test_gui gui.launch.py
```

GUI 사용 기준:

- `ARM` 패널: 왼팔/왼그리퍼
- `ARM2` 패널: 오른팔/오른 Tesollo hand
- `Target move`: 현재 pose 입력값을 실제 제어 토픽으로 publish
- `Init position`: 기본 시작 pose를 입력하고 즉시 publish
- `Up/Down` 버튼: 축별로 소량 증분 이동

## 9. 확인 포인트

실행 중 아래를 확인하면 됩니다.

```bash
ros2 topic echo /right_joint_trajectory_controller/joint_trajectory
ros2 topic echo /joint_states
```

필요하면 RViz 또는 별도 모니터 노드로 오른팔 움직임을 같이 확인합니다.

## 10. Isaac Sim에서 같이 쓸 경우

Isaac Sim 5.1.0 내부 Python에서 KDL을 직접 써야 한다면 `PyKDL`을 별도로 설치합니다.

자세한 절차는 아래 문서를 기준으로 진행합니다.

- [Install pyKDL_5.1.0.md](/home/user/rl_ws/sim2real/reference%20files/Install%20pyKDL_5.1.0.md)

핵심 요약:

1. `ISAAC_SIM`과 `ISAAC_PY` 설정
2. Isaac Python 3.11에 `pybind11>=2.10.4` 설치
3. `orocos_kinematics_dynamics` 다운로드
4. `orocos_kdl` C++ 설치
5. `python_orocos_kdl` 빌드
6. `PyKDL` import 확인

예:

```bash
export ISAAC_SIM=~/isaacsim-5.1.0
export ISAAC_PY=$ISAAC_SIM/python.sh

$ISAAC_PY -m pip install -U "pybind11>=2.10.4"
```

주의:

- Isaac Sim용 `PyKDL` 설치와 ROS 2 쪽 C++ `orocos_kdl` 사용은 별개입니다.
- Isaac Python에 `PyKDL`을 깔았다고 해서 ROS 2 C++ 패키지 의존성이 자동으로 해결되지는 않습니다.

## 11. 새 환경에서 꼭 설치해야 하는 항목 요약

### ROS 2 실기 EEF 제어만 할 때

- ROS 2 Humble
- `python3-colcon-common-extensions`
- `python3-rosdep`
- `build-essential`, `cmake`, `git`
- `ros-humble-kdl-parser`
- `ros-humble-orocos-kdl`
- `ros-humble-rclcpp`
- `ros-humble-geometry-msgs`
- `ros-humble-sensor-msgs`
- `ros-humble-trajectory-msgs`
- `ros-humble-urdf`
- 컨트롤러 관련 ROS 2 패키지
- 이 저장소와 vendor 패키지

### Isaac Sim Python에서도 KDL이 필요할 때 추가

- Isaac Sim 5.1.0
- Isaac Python용 `pybind11`
- `PyKDL`

## 12. 주의 사항

- 현재 좌우 노드 모두 입력 pose를 절대 목표 pose로 해석합니다.
- TF 변환 처리는 아직 넣지 않았습니다.
- 목표 pose가 작업공간 밖이면 IK 실패가 날 수 있습니다.
- 실기에서는 반드시 작은 이동 범위로 먼저 검증해야 합니다.
- 오른팔 기본 tip은 `palm_ee`입니다.
- 왼팔 기본 tip은 `openarm_left_hand_tcp`입니다.
- URDF 기준이 바뀌면 `root_link`, `tip_link`, joint limits를 다시 확인해야 합니다.

## 13. 가장 짧은 실행 절차

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
colcon build --packages-select openarm_eef_control
source install/setup.bash
```

실기 bringup:

```bash
ros2 launch /home/user/rl_ws/sim2real/integrated_control/launch/openarm_left_gripper_right_dg5_real.launch.py \
  left_can_interface:=can1 \
  right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

EEF 제어 노드:

```bash
ros2 launch openarm_eef_control right_arm_eef_control.launch.py
ros2 launch openarm_eef_control left_arm_eef_control.launch.py
```

목표 pose 전송:

```bash
ros2 topic pub --once /openarm/right_arm/eef_target geometry_msgs/msg/PoseStamped '{
  header: {frame_id: world},
  pose: {
    position: {x: 0.45, y: -0.20, z: 0.65},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```
