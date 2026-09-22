# legacy — 더는 빌드하지 않는 ROS 패키지

`COLCON_IGNORE` 가 있어 저장소 루트에서 `colcon build` 를 해도 여기는 빌드되지 않는다.
지우지 않은 이유는 배선을 손으로 확인할 때 아직 쓸 수 있고, 경위가 남아 있기 때문이다.

| 패키지 | 무엇 | 왜 옮겼나 | 대신 쓰는 것 |
|---|---|---|---|
| `ros_pkgs/openarm_control/` | 왼손 그리퍼 + 오른팔만 쓰는 옛 OpenArm 런치 래퍼 | 2026-03 하드웨어 구성. colcon 패키지도 아니다(`package.xml` 없음) | `robot_control` 의 `openarm_bringup openarm.bimanual.launch.py` |
| `ros_pkgs/tesollo_control/` | DG-5F 오른손 단독 런치 래퍼 | 같은 시기. `package.xml` 없음 | `robot_control` 의 `dg5f_<side>_driver.launch.py` |
| `ros_pkgs/openarm_eef_control/` | EEF 목표 → IK → JTC | 빌드 스크립트가 빌드하지 않는다. 배포 경로는 fabric IK(policy_control) | `policy_control` 의 `fabric_node` |
| `ros_pkgs/test_gui/` | upstream 예제 포크 GUI | **게이트 없이 실손 JTC 로 발행**한다 — 배포 경로에서 배제 | `s2r_console` |

## 다시 쓰려면

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths legacy/ros_pkgs/test_gui legacy/ros_pkgs/openarm_eef_control
source install/setup.bash
```

옛 런치 래퍼(openarm_control · tesollo_control)는 빌드 없이 경로로 부른다:
`ros2 launch legacy/ros_pkgs/openarm_control/launch/openarm_left_gripper_bimanual_real.launch.py`
