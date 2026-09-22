# OpenArm_Tesollo 기반 EEF control 적용 방법

이 문서는 [ROS2 기반 ur16e EEF control 방법](/home/user/rl_ws/sim2real/reference%20files/ROS2%20기반%20ur16e%20EEF%20control%20방법.md) 을 현재 저장소의 실제 로봇 구성에 맞게 바꿔서 적용할 때 기준으로 사용합니다.

핵심은 다음입니다.

- `ur16e` 전용 조인트 이름을 그대로 쓰면 안 됩니다.
- `sensor_msgs/msg/JointState`를 `joint_command`로 바로 보내는 방식도 현재 저장소 구조와 맞지 않습니다.
- 현재 실기 OpenArm 팔은 `JointTrajectoryController` 기반으로 제어해야 합니다.
- 오른손 Tesollo(DG5F)도 별도 `JointTrajectoryController` 토픽으로 제어합니다.

## 1. 현재 저장소의 실제 제어 구조

실기 통합 런치:

- [openarm_left_gripper_right_dg5_real.launch.py](/home/user/rl_ws/sim2real/integrated_control/launch/openarm_left_gripper_right_dg5_real.launch.py)

OpenArm 컨트롤러 설정:

- [openarm_left_gripper_bimanual_controllers.yaml](/home/user/rl_ws/sim2real/integrated_control/config/openarm_left_gripper_bimanual_controllers.yaml)

실제 팔 제어 토픽:

- 왼팔: `/left_joint_trajectory_controller/joint_trajectory`
- 오른팔: `/right_joint_trajectory_controller/joint_trajectory`

실제 손 제어 토픽:

- 오른손 DG5F: `/dg5f_right/dg5f_right_controller/joint_trajectory`

상태 토픽:

- OpenArm: `/joint_states`
- DG5F right: `/dg5f_right/joint_states`

## 2. UR16e 문서에서 바꿔야 하는 부분

### 조인트 이름

UR16e 문서의 조인트 이름:

```text
shoulder_pan_joint
shoulder_lift_joint
elbow_joint
wrist_1_joint
wrist_2_joint
wrist_3_joint
```

현재 OpenArm 조인트 이름:

```text
openarm_left_joint1
openarm_left_joint2
openarm_left_joint3
openarm_left_joint4
openarm_left_joint5
openarm_left_joint6
openarm_left_joint7
```

오른팔을 제어할 경우:

```text
openarm_right_joint1
openarm_right_joint2
openarm_right_joint3
openarm_right_joint4
openarm_right_joint5
openarm_right_joint6
openarm_right_joint7
```

즉, 기존 6축 UR 기준 코드는 그대로 재사용할 수 없고, 최소한 7축 OpenArm 기준으로 수정해야 합니다.

### 출력 메시지 타입

UR16e 문서 예제는 아래처럼 `sensor_msgs/msg/JointState`를 직접 발행합니다.

```cpp
ur_jointarget_pub = this->create_publisher<sensor_msgs::msg::JointState>("joint_command", 10);
```

현재 저장소의 실기 제어는 이 방식이 아니라 `trajectory_msgs/msg/JointTrajectory` 기반입니다.

따라서 팔 제어 퍼블리셔는 아래처럼 바꾸는 것이 맞습니다.

```cpp
auto left_arm_pub = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
  "/left_joint_trajectory_controller/joint_trajectory", 10);
```

오른팔이면:

```cpp
auto right_arm_pub = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
  "/right_joint_trajectory_controller/joint_trajectory", 10);
```

## 3. EEF 제어 노드의 권장 구조

현재 로봇용 EEF 제어 노드는 아래 구조가 맞습니다.

1. `/joint_states`를 subscribe 해서 현재 팔 관절값을 읽음
2. 목표 EEF pose 또는 delta pose를 subscribe 함
3. OpenArm용 IK를 계산함
4. 결과 joint 값을 `JointTrajectory` 한 점(point)으로 만들어 컨트롤러 토픽에 publish 함

즉, `EEF -> IK -> JointTrajectoryController` 흐름으로 가야 합니다.

## 4. 최소 변경 예시

### 의존성

기존 문서에서 아래 방향만 유지하면 됩니다.

- `rclcpp`
- `sensor_msgs`
- `geometry_msgs`
- `trajectory_msgs`
- `control_msgs` 필요 시 추가
- `orocos_kdl`, `trac_ik_lib` 또는 MoveIt IK 사용

반대로 아래는 현재 구조상 핵심 의존성은 아닙니다.

- `joint_state_broadcaster`를 제어 노드 패키지 의존성으로 직접 둘 필요는 낮음
- `joint_trajectory_controller` 패키지를 find_package 하는 것보다, 메시지 타입 `trajectory_msgs` 기준으로 퍼블리시하는 편이 더 직접적임

### 퍼블리셔 예시

```cpp
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

auto arm_pub = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
  "/left_joint_trajectory_controller/joint_trajectory", 10);
```

### 메시지 작성 예시

```cpp
trajectory_msgs::msg::JointTrajectory msg;
msg.joint_names = {
  "openarm_left_joint1",
  "openarm_left_joint2",
  "openarm_left_joint3",
  "openarm_left_joint4",
  "openarm_left_joint5",
  "openarm_left_joint6",
  "openarm_left_joint7",
};

trajectory_msgs::msg::JointTrajectoryPoint point;
point.positions = target_joints;   // size 7
point.time_from_start.sec = 0;
point.time_from_start.nanosec = 200000000;  // 0.2 sec

msg.points.push_back(point);
arm_pub->publish(msg);
```

## 5. IK 쪽에서 바꿔야 할 점

UR16e 문서는 사실상 UR16e 체인과 UR 조인트 순서를 가정합니다. 현재 로봇은 다음 점이 다릅니다.

- 7 DoF arm
- 조인트 순서 다름
- 링크 이름 다름
- 말단 EE 프레임 이름 다름

따라서 `TRAC-IK` 또는 `KDL`을 계속 쓸 수는 있지만, 아래를 전부 OpenArm 기준으로 다시 맞춰야 합니다.

- base link 이름
- tip link 이름
- joint limit
- joint ordering
- seed state

현재 저장소에서 손 쪽 wrapper에 `palm_ee` 보조 프레임이 있습니다.

- [tesollo_left_wrapper.xacro](/home/user/rl_ws/sim2real/urdf/eef/tesollo_left_wrapper.xacro)
- [tesollo_right_wrapper.xacro](/home/user/rl_ws/sim2real/urdf/eef/tesollo_right_wrapper.xacro)

손 기준 EE를 쓸 경우 `left_palm_ee` 또는 prefix가 붙은 palm frame을 tip 후보로 검토하는 게 맞습니다.

반면 OpenArm 자체 hand/tcp 기준 링크는 vendor xacro 쪽 정의도 함께 확인해야 합니다.

## 6. 현재 저장소 기준으로 더 쉬운 방법

현재 저장소에는 이미 `JointTrajectory` 퍼블리시 패턴이 구현돼 있습니다.

- [bridge_node.py](/home/user/rl_ws/sim2real/isaacsim_bridge/isaacsim_bridge/bridge_node.py)

여기서 확인 가능한 점:

- 왼팔 조인트 순서가 이미 정의돼 있음
- 오른팔 조인트 순서가 이미 정의돼 있음
- `JointTrajectory` 한 점을 publish 하는 패턴이 이미 있음

따라서 새 EEF 제어 노드를 만들 때는 다음이 가장 안전합니다.

1. `bridge_node.py`의 조인트 순서를 그대로 사용
2. IK만 별도 노드에서 계산
3. 계산 결과를 `/left_joint_trajectory_controller/joint_trajectory` 또는 `/right_joint_trajectory_controller/joint_trajectory`로 publish

## 7. 결론

`ROS2 기반 ur16e EEF control 방법.md`에서 재사용 가능한 것은 아래뿐입니다.

- ROS 2 패키지 생성 방식
- EEF 목표를 subscribe 하고 joint로 변환하는 전체 구조
- IK solver를 노드 안에서 사용하는 아이디어

그대로 쓰면 안 되는 것은 아래입니다.

- UR16e 조인트 이름
- 6축 가정
- `joint_command` + `JointState` 직접 발행 구조
- UR 전용 base/tip 링크 설정

현재 로봇에 맞는 올바른 방향은 아래입니다.

- OpenArm 7축 조인트 이름 사용
- `/joint_states` 입력 사용
- IK 계산 결과를 `JointTrajectory`로 변환
- `/left_joint_trajectory_controller/joint_trajectory` 또는 `/right_joint_trajectory_controller/joint_trajectory`로 publish

## 8. 바로 구현하려면

새 패키지 예시 이름:

```bash
ros2 pkg create --build-type ament_cmake openarm_eef_control --dependencies rclcpp sensor_msgs geometry_msgs trajectory_msgs
```

그 다음 최소 구현은 아래 3개입니다.

1. 현재 joint state subscribe
2. 목표 EEF pose 또는 twist subscribe
3. IK 결과를 OpenArm 7축 `JointTrajectory`로 publish

원하면 다음 단계로 바로 진행할 수 있습니다.

- `openarm_eef_control` 패키지 뼈대 생성
- 왼팔 기준 EEF target subscriber 추가
- `/left_joint_trajectory_controller/joint_trajectory` 퍼블리셔까지 실제 코드 작성
