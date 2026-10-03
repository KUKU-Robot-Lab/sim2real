# INSTALL — 새 PC 세팅 가이드 (step by step)

깨끗한 Ubuntu 22.04 PC에서 이 레포만으로 실물 로봇 제어·비전·정책 추론
환경을 구축하는 절차. 각 Step 끝의 **확인** 커맨드가 통과해야 다음으로 넘어간다.

```bash
# 지금 이 PC에 뭐가 준비됐는지부터 확인 (설치는 안 함) — 로봇 PC
python3 scripts/setup/check_host.py --robot dg5f     # 또는 --robot rh56f1
./scripts/setup/setup_check.sh vision                # 비전 PC(옛 점검 스크립트)
```

## 어떤 PC에 어떤 Step이 필요한가

PC 역할은 자유롭게 합칠 수 있다 (한 대에 전부도 가능). DDS(같은
`ROS_DOMAIN_ID`)로 통신하므로 역할별로 나눠도 코드는 동일하다.

| 역할 | 담당 | 필요한 Step |
|---|---|---|
| **로봇 PC**(local5090 = DG-5F-M short · arm4090 = RH56F1) | 드라이버 · pd · 정책 · 콘솔 | 1~5 |
| **vision**(vision-3090) | 카메라 + FP++ → 물체 자세 UDP | 1~2, 6~8 (docs/USAGE_DEPLOY.md §6) |
| sim | Isaac Sim (sim-shadow 검증용, 선택) | 1~3 + Isaac Sim 설치본 |

---

## Step 1. OS·GPU 전제 확인

- Ubuntu 22.04 LTS, x86_64 (Isaac ROS 공식 지원 조합)
- vision/policy 역할이면 NVIDIA GPU + 드라이버 (Blackwell 계열은 드라이버 570+)

```bash
# NVIDIA 드라이버 (vision/policy PC만)
ubuntu-drivers devices                 # GPU와 후보 드라이버 확인
sudo apt install -y nvidia-driver-580  # 이 프로젝트 표준: 580 계열로 통일
sudo reboot
```

> 버전 지침: Isaac ROS(FoundationPose)는 드라이버 ≥535면 되지만, 최신
> CUDA 컨테이너 호환을 위해 **580 계열로 전 머신 통일**을 권장한다.
> 호스트에 CUDA 툴킷은 설치하지 말 것 — 컨테이너가 자체 포함하며,
> 호스트 요건은 드라이버 + nvidia-container-toolkit(Step 6)뿐이다.

**확인**
```bash
grep VERSION_ID /etc/os-release   # "22.04"
uname -m                          # x86_64
nvidia-smi                        # GPU 이름·드라이버 버전 표시 (vision/policy)
```

## Step 2. ROS2 Humble + colcon

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install -y \
    ros-humble-desktop \
    ros-humble-vision-msgs \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-xacro \
    python3-colcon-common-extensions

# 모든 PC에서 동일한 도메인 ID (이 프로젝트 관례: 126)
echo 'export ROS_DOMAIN_ID=126' >> ~/.bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

**확인**
```bash
ros2 doctor --report | head -5
echo $ROS_DOMAIN_ID               # 126
```

## Step 3. 저장소 네 개 + 빌드

sim2real 혼자로는 돌지 않는다. **네 저장소를 `~/rl_ws` 에 나란히** 둔다 — 코드가 이 상대 위치를 전제한다
(`sim2real/../robot_control` 등).

| 저장소 | 브랜치 | sim2real 이 쓰는 것 |
|---|---|---|
| `KUKU-Robot-Lab/sim2real` | main | 이 저장소 — 콘솔 · 미션 · policy_control · 정책 계약 · 홈 경로 |
| `KUKU-Robot-Lab/robot_control` | humble | 드라이버(openarm bringup · dg5f_ros2 · dg_hardware · inspire_rh56f1) · 관절 프로필 |
| `KUKU-Robot-Lab/hdgp` | main | 자산(URDF/USD · manifest · 테이블 env_v1) · pour_fj 프로필 모듈 · FABRICS |
| `KUKU-Robot-Lab/urdf` | main | 팔 PD 게인 `vendor/openarm_description/config/arm/v10/control_gains.yaml` |

```bash
mkdir -p ~/rl_ws && cd ~/rl_ws
git clone https://github.com/KUKU-Robot-Lab/sim2real.git
git clone -b humble https://github.com/KUKU-Robot-Lab/robot_control.git
git clone https://github.com/KUKU-Robot-Lab/hdgp.git
git clone https://github.com/KUKU-Robot-Lab/urdf.git
# 이미 있으면 네 곳 모두 git pull
```

> hdgp 는 비공개라 https 로는 `could not read Username` 에서 멈춘다. GitHub ssh 키가 있는 PC 는 ssh 주소로 받는다:
> `git -C ~/rl_ws/hdgp fetch git@github.com:KUKU-Robot-Lab/hdgp.git main:refs/remotes/origin/main && git -C ~/rl_ws/hdgp merge --ff-only origin/main`

### Step 3-A. robot_control 드라이버 빌드

```bash
cd ~/rl_ws/robot_control/ros_ws
# 빌드 의존성 — 5090 에 깔린 것과 같게(09.29 arm4090 에서 빠져 있던 목록). openarm_can 은 CLI11 이 없으면 cmake 에서 멈춘다.
sudo apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-dev libyaml-cpp-dev \
    libcli11-dev libspdlog-dev python3.10-venv \
    ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-controller-manager ros-humble-control-msgs \
    ros-humble-hardware-interface ros-humble-ros2-control-test-assets ros-humble-xacro \
    ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui \
    ros-humble-moveit-configs-utils ros-humble-moveit-kinematics ros-humble-moveit-planners \
    ros-humble-moveit-ros-move-group ros-humble-moveit-ros-visualization ros-humble-moveit-setup-assistant \
    ros-humble-moveit-simple-controller-manager ros-humble-ros-gz ros-humble-ign-ros2-control \
    ros-humble-realsense2-description
# dg_sdk_ros2_bridge 가 링크하는 libDGSDK.so 는 .gitignore 라 clone 에 없다 — 버전 파일 171 을 이름만 바꿔 둔다(5090 과 md5 같음)
cp src/delto_m_ros2/dg_sdk_ros2_bridge/libs/libDGSDK_171.so src/delto_m_ros2/dg_sdk_ros2_bridge/libs/libDGSDK.so
./build.sh                                   # colcon --symlink-install, install/ 에 openarm_* · dg5f_* · rh56f1_*
```

### Step 3-B. sim2real policy_control 빌드 (★symlink 필수)

```bash
cd ~/rl_ws/sim2real
source /opt/ros/humble/setup.bash && source ../robot_control/ros_ws/install/setup.bash
PYTHONNOUSERSITE=1 colcon build --packages-select policy_control --base-paths deploy --symlink-install
readlink -f build/policy_control/policy_control     # → …/sim2real/deploy/policy_control/policy_control 이어야 한다
```

> ★`PYTHONNOUSERSITE=1`: `~/.local` 에 setuptools 80 이상이 있으면(arm4090 은 84) `--symlink-install` 을 줘도 조용히
> 복사 설치가 된다(09.30). 시스템 setuptools(59.6)로 빌드하게 `~/.local` 을 끈다.
> ★`--symlink-install` 없이 빌드하면(복사 설치) `_paths` 가 저장소 루트를 못 찾아 pd_node · joint_node 가
> import 에서 죽는다(09.28 실기). 잘못 빌드했으면 `rm -rf build/policy_control install/policy_control` 뒤 다시.
> 콘솔이 떠 있는 동안에는 다시 빌드하지 않는다(떠 있는 콘솔은 옛 환경을 들고 있다).

**확인**
```bash
ros2 pkg list | grep -E "openarm_bringup|dg5f_driver|rh56f1_driver|policy_control"
```

## Step 4. sim2real/.venv — 정책 추론 · 경로 계획 · 테스트

콘솔(`deploy/s2r_console/tools/console.sh`)과 미션 명령의 `python3` 는 이 venv 다. **시스템 패키지를 보게** 만들어야
ROS(rclpy)가 보인다. 버전은 local5090 기준(2026-09-29) — 다르면 `check_host.py` 가 WARN 을 낸다.

```bash
cd ~/rl_ws/sim2real
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # Blackwell=cu128+
.venv/bin/pip install rl-games==1.6.1 mujoco==2.3.0 numpy==1.26.4 scipy==1.13.1 trimesh pyyaml pytest shapely==2.1.2 h5py==3.14.0
# FABRICS(DG-5F fabric 단계): hdgp 안의 소스를 venv 에 연결
echo "$HOME/rl_ws/hdgp/source/FABRICS/src" > .venv/lib/python3.10/site-packages/fabrics_sim.pth
```

> `~/.local` 에 pydantic · flask 같은 사용자 패키지가 있으면 venv 로 새어 들어온다(include-system-site-packages).
> venv 에는 위 목록 밖의 패키지를 설치하지 않는다 — rclpy · torch · fabrics 가 같이 사는 유일한 곳이다.

### Step 4-B. 정책 가중치 — git 에 없다

`deploy/policies/*/nn/*.pth` 는 `.gitignore` 로 빠진다(용량). 5090 에서 복사한다 — 계약의 md5 와 같아야 한다.

```bash
rsync -av <5090 PC>:~/rl_ws/sim2real/deploy/policies/ ~/rl_ws/sim2real/deploy/policies/ \
      --include='*/' --include='nn/*.pth' --include='trace.npz' --exclude='*'     # trace.npz: 골든 대조(both_pour_i24)
```

### Step 4-C. 이 PC 에만 있는 값

| 파일 | 무엇 |
|---|---|
| `deploy/policy_control/config/rh56f1_ports.yaml` | RH56F1 손마다 포트 · transport(rs485/canfd) · Hand_ID (`ls -l /dev/serial/by-id`) |
| CAN 이름 | 미션 drivers 단계가 `can0`(우) · `can1`(좌) 를 쓴다 |
| DG-5F 손 네트워크 | 미션 hand_<side> 단계의 `hand_net_dual.sh --apply`(최초 1회) |
| `ROS_DOMAIN_ID` | 실기 126 (fake 는 콘솔이 97 로 띄운다) |
| 실시간 한도(로봇 PC) | `sudo bash scripts/setup/rt_setup.sh` 한 번 → 재부팅. EtherCAT 마스터(SCHED_FIFO 80) · controller_manager(50) 가 실시간 우선순위를 받는다. `--performance` 를 붙이면 부팅 때 CPU governor 도 performance(선택) |
| 코어 배치 | 손으로 정하지 않는다 — 노드가 그 PC 의 코어(sysfs)를 읽어 EtherCAT 마스터를 제 코어에, 나머지 노드를 그 밖에 둔다(`python3 -m policy_control.cpu_plan` 으로 보기). 끄기 `S2R_CPU_PIN=0` |

## Step 5. 점검 · 회귀 테스트 게이트

```bash
cd ~/rl_ws/sim2real && source /opt/ros/humble/setup.bash
python3 scripts/setup/check_host.py --robot dg5f       # 또는 --robot rh56f1 (arm4090) · --fetch 로 원격 비교
python3 scripts/setup/check_host.py --robot rh56f1 --only cpu   # CPU 만 — 실기 미션 preflight 가 매번 부른다
.venv/bin/python -m pytest tests -q -m "not gpu"        # 실패 0 이어야 실기
```

`check_host.py` 는 읽기만 한다 — 저장소 배치 · 원격과의 차이 · venv 버전 · policy_control symlink 빌드 ·
robot_control 패키지 · 가중치 md5(계약과) · 실기 미션이 가리키는 파일 · 홈 경로와 계약의 짝 · PC 별 값.
MISS 가 하나라도 있으면 rc 1 이고, 고칠 방법(이 문서의 Step)을 같이 낸다.

> 테스트는 `tests/` 에 있다. GPU 를 쓰는 것은 `-m "not gpu"` 로 뺀다 — 학습이 도는 GPU 를 건드리지 않기 위해서다.
> hdgp 자산 · 모듈을 읽는 테스트는 hdgp 가 옆에 있어야 돈다(Step 3).

## Step 6. Docker + nvidia-container-toolkit (vision PC만)

```bash
# Docker
sudo apt install -y docker.io
sudo usermod -aG docker $USER    # 재로그인 필요

# NVIDIA container toolkit (GPU 컨테이너 실행용)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**확인**
```bash
docker run --rm --gpus all ubuntu nvidia-smi   # 컨테이너 안에서 GPU 표시
```

## Step 7. Isaac ROS FoundationPose (vision PC만)

> Isaac ROS는 **Isaac Sim과 무관한 실물용 ROS2 지각 패키지**다.
> FoundationPose 신경망 추론이 여기서 돌고, 우리 레포의
> `scripts/cup_pose_relay.py`가 그 출력을 `/cup_pose`로 변환한다.

설치·모델 다운로드·실행 상세는 **`robot/USAGE_ISAACSIM_ROS2.md` §7-1** 참조. 요약:

```bash
mkdir -p ~/workspaces/isaac_ros-dev/src && cd ~/workspaces/isaac_ros-dev/src
git clone -b main https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b main https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git
cd isaac_ros_common && ./scripts/run_dev.sh        # dev 컨테이너 진입
# [컨테이너] sudo apt install ros-humble-isaac-ros-foundationpose \
#                             ros-humble-isaac-ros-examples ros-humble-realsense2-camera
```

준비물: **컵 textured CAD 메시**(.obj + texture). CAD 원점/축이 sim body
프레임(원점=바닥 중심, +z=위)과 다르면 Step 8에서 보정.

**확인**
```bash
# 컨테이너 안에서
ros2 pkg list | grep foundationpose
```

## Step 8. 캘리브레이션 (vision PC, 실기 1회)

`config/global_camera_extrinsics.yaml`의 두 변환을 실측으로 교체:

1. **`camera`** — robot base ← `camera_color_optical_frame` (글로벌 D435i 장착 후 hand-eye/타깃 보드 캘리브)
2. **`cad_to_body`** — 컵 CAD 원점/축 ↔ sim body 프레임 정합

⚠ 둘 다 기본값이 PLACEHOLDER(identity)다. **교체 전 실기 구동 금지.**
`setup_check.sh vision`이 PLACEHOLDER 상태를 감지해 경고한다.

## Step 9. 실행

실기 운영은 콘솔 하나로 한다 — 로봇(DG-5F-M short · RH56F1) → 정책 → 실기/fake 를 첫 화면에서 고른다.

```bash
deploy/s2r_console/tools/console.sh --port 8091          # 브라우저 http://127.0.0.1:8091 (원격은 ssh -L 8091:127.0.0.1:8091)
deploy/s2r_console/tools/console.sh --window             # 전용 창
```

자세한 배포 절차는 [`docs/USAGE_DEPLOY.md`](docs/USAGE_DEPLOY.md). 옛 역할별 브링업 문서는 아래 표:

| 하고 싶은 것 | 문서 |
|---|---|
| 로봇별 제어 → sim 연결 → test | USAGE §1(OpenArm) §2(Tesollo) §3(RH56F1) |
| 팔+손 통합 | USAGE §4 |
| 하드웨어 없이 배선 확인 | USAGE §5 (dry-run) |
| 비전 노드 (`/cup_pose`) | USAGE §7 |
| 정책 배포(현행) | [`docs/USAGE_DEPLOY.md`](docs/USAGE_DEPLOY.md) — 등록·계약·미션·콘솔 |

---

## 부록. 멀티 PC 상호 접속 — Tailscale (선택)

여러 PC(로컬 GPU·서버·비전 PC)가 기관 WiFi ↔ 핫스팟을 오가면 DHCP IP가
계속 바뀐다. Tailscale을 깔면 머신마다 **네트워크와 무관한 고정 가상
IP(100.x)와 고정 이름**이 생겨, 어느 WiFi에 있든(서로 다른 망이어도)
`ssh <머신이름>` 하나로 접속된다.

```bash
# [각 PC에서 1회]
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                        # URL 열어 같은 계정으로 로그인
sudo tailscale set --hostname=<이름>     # 예: pc5090, server

# 확인
tailscale status                         # 머신 목록·가상 IP
tailscale ping <이름>                    # 터널 연결 확인
```

`~/.ssh/config`에 등록하면 한 단어로 접속:

```
Host server
    HostName server        # MagicDNS 이름 (또는 100.x 가상 IP)
    User <서버 계정>
    ServerAliveInterval 30
```

```bash
ssh-copy-id server         # 최초 1회 (비밀번호 입력) → 이후 무비밀번호
ssh server
```

> ⚠ ROS2 DDS는 별개다 — Tailscale 위에서는 멀티캐스트가 안 돼서 PC 간
> DDS 통신은 같은 LAN(같은 WiFi + `ROS_DOMAIN_ID`)을 쓰거나 discovery
> server/unicast peer 설정이 따로 필요하다. SSH·scp·rsync·모니터링은 바로 된다.

---

## 트러블슈팅

| 증상 | 확인 |
|---|---|
| `check_host.py` · `setup_check.sh` MISS | 표기된 Step으로 이동 |
| pd_node 가 import 에서 죽음 | policy_control 이 복사 설치 — Step 3-B symlink 로 다시 |
| 정책 단계가 "checkpoint md5" 로 멈춤 | 가중치가 계약과 다르다 — Step 4-B 로 5090 에서 다시 복사 |
| PC끼리 토픽 안 보임 | 모든 PC `ROS_DOMAIN_ID` 동일 + 같은 서브넷 + 방화벽(UDP 멀티캐스트) |
| `docker: unknown runtime nvidia` | Step 6의 `nvidia-ctk runtime configure` + docker 재시작 |
| torch가 GPU 커널 에러 (Blackwell) | cu128 이상 빌드로 재설치 (Step 4-B) |
| 회귀 테스트 실패 | 학습 코드(env_cfg/preset)가 바뀐 것 — sim2real 포팅 상수 재정합 필요 |
