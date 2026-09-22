# openarm_eef_control

좌우 팔을 개별 노드로 제어하는 EEF 제어 패키지입니다.

오른팔 기본 설정:

- URDF: `/home/user/rl_ws/sim2real/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf`
- root link: `openarm_right_link0`
- tip link: `palm_ee`
- 입력 pose topic: `/openarm/right_arm/eef_target`
- 출력 trajectory topic: `/right_joint_trajectory_controller/joint_trajectory`

왼팔 기본 설정:

- URDF: `/home/user/rl_ws/sim2real/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf`
- root link: `openarm_left_link0`
- tip link: `openarm_left_hand_tcp`
- 입력 pose topic: `/openarm/left_arm/eef_target`
- 출력 trajectory topic: `/left_joint_trajectory_controller/joint_trajectory`

빌드:

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
colcon build --packages-select openarm_eef_control
source install/setup.bash
```

오른팔 실행:

```bash
ros2 launch openarm_eef_control right_arm_eef_control.launch.py
```

왼팔 실행:

```bash
ros2 launch openarm_eef_control left_arm_eef_control.launch.py
```

오른팔 목표 pose publish 예시:

```bash
ros2 topic pub --once /openarm/right_arm/eef_target geometry_msgs/msg/PoseStamped '{
  header: {frame_id: world},
  pose: {
    position: {x: 0.45, y: -0.20, z: 0.65},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

왼팔 목표 pose publish 예시:

```bash
ros2 topic pub --once /openarm/left_arm/eef_target geometry_msgs/msg/PoseStamped '{
  header: {frame_id: world},
  pose: {
    position: {x: 0.45, y: 0.20, z: 0.65},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

주의:

- 현재 두 노드 모두 입력 pose를 URDF 기준 절대 목표로 해석합니다.
- frame transform 처리는 아직 넣지 않았습니다.
- 실기 제어 전에 작은 이동 범위로 먼저 검증하는 것이 안전합니다.
