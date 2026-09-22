# legacy — 더는 쓰지 않는 것

두 갈래다: `ros_pkgs/`(빌드하지 않는 ROS 패키지) · `scripts/`(참조 0 으로 확인된 스크립트).

## ros_pkgs/ — 더는 빌드하지 않는 ROS 패키지

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

## scripts/ — 참조 0 으로 확인된 스크립트

2026-09-22 감사(4관점 → 반증 검증)에서 **참조가 0** 인 것을 확인한 뒤 옮겼다. 다시 쓰려면 원래 자리로 되돌린다.

**왜 `scripts/` 안이 아닌가** — `tests/conftest.py` 와 `deploy/policy_control/policy_control/_paths.py` 는
`scripts/` 의 **모든 하위 디렉터리를 sys.path 에 얹는다**(`_SKIP` 은 `__pycache__` 계열뿐). 죽은 코드를
`scripts/deprecated/` 에 두면 모든 테스트와 policy_control import 가 그것을 import 경로에 달고 다닌다.
그리고 `legacy/` 에는 `COLCON_IGNORE` 가 있어 colcon 도 여기를 보지 않는다.

| 경로 | 무엇 | 대체된 것 |
|---|---|---|
| `scripts/deprecated/` | `sim2real_inference.py` · `sim2real_dryrun.py` | policy_control 체인 + `scripts/ops/mission_run.py` |
| `scripts/probes/` | `probe_head_push_response` · `probe_policy_on_sim_states` · `probe_s2r_right_record` | — (`probe_policy_on_sim_states` 는 parity 하네스였지만 테스트가 없어 계약 변경에 조용히 썩었다) |
| `scripts/analysis/` | `hand_status_monitor` · `monitor_left_side` · `monitor_right_side` | `deploy/policy_control/tools/status_board.py` · `deploy/s2r_console` |
| `scripts/calib/` | `cup_touch_survey` · `touch_probe_left` | — |
| `scripts/nodes/` | `npz_udp_feed` · `ros2_teleop_device` · `shaker_centroid_node` | shaker 과제 자체가 폐기 |
| `scripts/vision/` | `grab_frame` · `pose_stats_recorder` | `scripts/vision/grab_rgbd.py` · 인지 런처 |

(처음엔 최상위 `archive/` 에 두었다가 2026-09-22 계층 정리에서 여기로 합쳤다.)
