# sim2real_control

OpenArm + Tesollo DG5 + Isaac Sim 연동을 위한 최소 워크스페이스입니다.

이 저장소는 다음 용도를 기준으로 정리되어 있습니다.

1. 실제 OpenArm 제어
2. 실제 Tesollo DG5 제어
3. OpenArm + Tesollo 통합 제어
4. Isaac Sim ROS 2 브리지
5. 실기-시뮬레이터 상태 비교 및 튜닝

## 디렉토리 구성

- `openarm_control/`: OpenArm 전용 런치 래퍼
- `tesollo_control/`: Tesollo DG5 전용 런치 래퍼
- `integrated_control/`: OpenArm + Tesollo 통합 런치
- `isaacsim_bridge/`: Isaac Sim ROS 2 브리지, 상태 로거, 튜닝 도구
- `urdf/`: OpenArm/Tesollo 조합용 xacro, urdf, usd
- `vendor/`: 이 저장소에 포함된 upstream 의존 패키지
- `scripts/`: 빌드 및 보조 스크립트

## 외부 의존성

이 저장소 밖에서 필요한 것은 아래뿐입니다.

- ROS 2 Humble: `/opt/ros/humble`
- Isaac Sim 설치본: 경로는 자유, 이 저장소에는 포함되지 않음
- 실제 하드웨어 연결:
  - OpenArm CAN 인터페이스 (`can0`, `can1` 등)
  - Tesollo DG5 Ethernet IP/Port

소스 코드 기준으로는 추가 외부 레포가 필요하지 않습니다. OpenArm/Tesollo 관련 ROS 2 패키지는 `vendor/` 아래에 포함되어 있습니다.

## 0. 기본 설치 방법

### 사전 조건

- Ubuntu + ROS 2 Humble
- `colcon` 설치
- Isaac Sim을 사용할 경우 Isaac Sim의 ROS 2 bridge 사용 가능 환경

### 빌드

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
cd "${REPO_DIR}"
./scripts/build_vendor_pkgs.sh
source "${REPO_DIR}/install/setup.bash"
```

브리지 패키지만 다시 빌드할 때:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
cd "${REPO_DIR}"
./scripts/build_vendor_pkgs.sh --bridge-only
source "${REPO_DIR}/install/setup.bash"
```

### 빌드 산출물

- `build/`
- `install/`
- `log/`

이 3개는 로컬 빌드 산출물이며, 다른 PC에서 다시 생성됩니다.

## 1. OpenArm, Tesollo 연결 및 ROS 2 사용법

### OpenArm만 실행

현재 구성:

- 왼쪽 OpenArm + 왼쪽 OpenArm 그리퍼 사용
- 오른쪽 OpenArm은 암만 사용
- 오른쪽 OpenArm 그리퍼는 사용 안 함

실기:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/openarm_control/launch/openarm_left_gripper_bimanual_real.launch.py" \
  left_can_interface:=can1 \
  right_can_interface:=can0
```

가짜 하드웨어:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/openarm_control/launch/openarm_left_gripper_bimanual_real.launch.py" \
  use_fake_hardware:=true
```

### Tesollo DG5 오른손만 실행

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/tesollo_control/launch/dg5f_right_real.launch.py" \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

### OpenArm + Tesollo 통합 실행

현재 통합 스택:

- 왼쪽 OpenArm 암 + 왼쪽 OpenArm 그리퍼
- 오른쪽 OpenArm 암
- 오른쪽 Tesollo DG5 핸드

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch "${REPO_DIR}/integrated_control/launch/openarm_left_gripper_right_dg5_real.launch.py" \
  left_can_interface:=can1 \
  right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

### 자주 보는 ROS 2 토픽

- `/joint_states`
- `/dg5f_right/joint_states`
- `/left_joint_trajectory_controller/joint_trajectory`
- `/right_joint_trajectory_controller/joint_trajectory`
- `/dg5f_right/dg5f_right_controller/joint_trajectory`

예시:

```bash
ros2 topic list
ros2 topic echo /joint_states
ros2 control list_controllers
```

## 2. Isaac Sim 연결 및 ROS 2 노드 사용법

### 사용 중인 USD / URDF

현재 Isaac Sim에서 사용하는 파일:

- USD: `urdf/openarm_modular_dual/openarm_modular_dual.usd`
- URDF 원본: `urdf/openarm_modular_dual.urdf`
- xacro 원본: `urdf/openarm_modular_dual.xacro`

즉, `urdf/openarm_modular_dual.urdf`를 기반으로 USD를 만든 구성입니다.

### 경로 수정 사항

다른 PC에서 바로 쓰기 위해 아래를 수정했습니다.

- `urdf/openarm_modular_dual.urdf`
- `urdf/openarm_tesollo_bi.urdf`

수정 내용:

- 절대 경로 `file:///home/user/...` 제거
- Tesollo mesh 경로를 `../vendor/tesollo/dg_description/meshes/...` 로 변경
- OpenArm mesh 경로를 `../vendor/openarm/openarm_description/meshes/...` 로 변경
- Tesollo xacro 원본(`vendor/tesollo/dg_description/urdf/*.xacro`)도 `file://$(find ...)` 대신 `package://dg_description/...` 로 변경

따라서 URDF 파일은 저장소 루트 기준 상대경로로 mesh를 찾습니다.

### Isaac Sim 브리지 빌드 및 실행

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
cd "${REPO_DIR}"
./scripts/build_vendor_pkgs.sh --bridge-only
source "${REPO_DIR}/install/setup.bash"
```

브리지만 실행:

```bash
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py
```

브리지와 실기 동시 실행:

```bash
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py \
  with_hardware:=true \
  left_can_interface:=can1 \
  right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

### Isaac Sim에서 사용하는 입력 토픽

- `/isaacsim/left_arm_cmd`
- `/isaacsim/right_arm_cmd`
- `/isaacsim/left_gripper_cmd`
- `/isaacsim/right_hand_cmd`
- `/isaacsim/emergency_stop`

### 브리지 출력 대상

- `/left_joint_trajectory_controller/joint_trajectory`
- `/right_joint_trajectory_controller/joint_trajectory`
- `/left_gripper_controller/gripper_cmd`
- `/dg5f_right/dg5f_right_controller/joint_trajectory`

### 실기 상태 병합 토픽

- 실기 OpenArm + Tesollo 상태를 모아서 `/isaacsim/joint_states` 로 재발행

### Isaac Sim Action Graph 스크립트

세부 내용은 `isaacsim_bridge/README.md` 참고.

- 명령 입력 그래프 생성: `isaacsim_bridge/scripts/create_action_graph.py`
- Sim shadow joint state 퍼블리시 그래프 생성: `isaacsim_bridge/scripts/create_sim_joint_state_publish_graph.py`
- 기본 강한 drive 세팅: `isaacsim_bridge/scripts/tune_shadow_joint_drives.py`
- 생성된 drive config 적용: `isaacsim_bridge/scripts/apply_joint_drive_config.py`

Isaac Sim Script Editor에서는 `exec(open(...).read())` 대신, 해당 스크립트 파일 내용을 직접 열어서 붙여넣는 방식이 가장 이식성이 좋습니다.

## 3. 튜닝 방법

### 목적

실기 joint 상태와 Isaac Sim shadow robot joint 상태가 최대한 비슷하게 움직이도록 맞춥니다.

우선순위:

1. zero offset / sign / 스케일 / 조인트 순서
2. stiffness / damping
3. 필요 시 실제 저수준 gain (`kp`, `kd`, Tesollo PID)

### 현재 제공되는 도구

#### 1) 실기-시뮬레이터 오차 기록

실기와 Sim shadow를 비교하여 CSV 저장:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 run isaacsim_bridge joint_error_recorder \
  --ros-args \
  -p real_joint_states_topic:=/isaacsim/joint_states \
  -p sim_joint_states_topic:=/isaacsim/sim_joint_states \
  -p output_path:=/tmp/isaacsim_joint_error.csv
```

또는 런치에서 같이 실행:

```bash
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py \
  with_hardware:=true \
  with_recorder:=true \
  recorder_output_path:=/tmp/isaacsim_joint_error.csv
```

#### 2) 오차 리포트 생성

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 run isaacsim_bridge joint_tuning_report -- \
  --input /tmp/isaacsim_joint_error.csv \
  --output /tmp/isaacsim_joint_tuning_report.json
```

이 도구는 joint별로 아래를 계산합니다.

- 평균 위치 오차
- 위치 RMSE
- 속도 RMSE
- 최대 위치 오차

그리고 offset 우선 확인, stiffness 증가, damping 증가 같은 휴리스틱 추천을 출력합니다.

#### 3) 자동 1회 튜닝 사이클

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 run isaacsim_bridge joint_tuning_cycle -- \
  --input-csv /tmp/isaacsim_joint_error.csv \
  --output-report /tmp/isaacsim_joint_tuning_report.json \
  --output-drive-config /tmp/isaacsim_next_joint_drive_config.json
```

생성물:

- `/tmp/isaacsim_joint_tuning_report.json`
- `/tmp/isaacsim_next_joint_drive_config.json`

`isaacsim_next_joint_drive_config.json` 에는 joint별 다음 값이 들어갑니다.

- `stiffness`
- `damping`
- `recommended_offset_delta`

#### 4) Isaac Sim에 새 drive 값 적용

`isaacsim_bridge/scripts/apply_joint_drive_config.py` 를 Isaac Sim Script Editor에서 실행합니다.

이 스크립트는 기본적으로 `/tmp/isaacsim_next_joint_drive_config.json` 을 읽습니다.

### 권장 튜닝 루프

1. Isaac Sim과 실기를 연결한 뒤 대표적인 움직임을 실행합니다.
2. `joint_error_recorder` 로 오차를 기록합니다.
3. `joint_tuning_cycle` 로 다음 drive config를 생성합니다.
4. Isaac Sim에서 `apply_joint_drive_config.py` 로 새 stiffness/damping 을 적용합니다.
5. 다시 측정해서 RMSE가 안정될 때까지 반복합니다.

### 주의 사항

- `recommended_offset_delta` 는 현재 자동으로 URDF나 브리지에 직접 적용되지는 않습니다.
- 즉, offset은 사람이 확인해서 zero calibration 또는 매핑 계층에 반영해야 합니다.
- OpenArm 그리퍼 매핑은 아직 근사치가 포함되어 있으므로, 큰 오차가 계속 나면 gain보다 매핑을 먼저 의심해야 합니다.

## 외부 디렉토리 사용 여부

다른 PC로 옮길 때 기준으로, 이 저장소 밖을 직접 참조하는 것은 아래입니다.

- `/opt/ros/humble`
  - ROS 2 환경 로드용
- Isaac Sim 설치 디렉토리
  - 저장소에는 포함되지 않음

그 외 소스 코드/URDF/mesh는 현재 기준으로 모두 이 저장소 내부 상대경로로 정리했습니다.

다만 런타임 리소스는 환경에 따라 아래를 사용합니다.

- CAN 인터페이스 이름 (`can0`, `can1`)
- Tesollo 장비 IP 주소 (`169.254.186.72` 등)
- 임시 출력 경로 (`/tmp/*.csv`, `/tmp/*.json`)

## 관련 문서

- `openarm_control/README.md`
- `tesollo_control/README.md`
- `integrated_control/README.md`
- `isaacsim_bridge/README.md`
