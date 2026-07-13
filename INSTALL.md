# INSTALL — 새 PC 세팅 가이드 (step by step)

깨끗한 Ubuntu 22.04 PC에서 이 레포만으로 실물 로봇 제어·비전·정책 추론
환경을 구축하는 절차. 각 Step 끝의 **확인** 커맨드가 통과해야 다음으로 넘어간다.

```bash
# 지금 이 PC에 뭐가 준비됐는지부터 확인 (설치는 안 함)
./scripts/setup_check.sh            # 전체
./scripts/setup_check.sh vision     # 역할별: control | vision | policy
```

## 어떤 PC에 어떤 Step이 필요한가

PC 역할은 자유롭게 합칠 수 있다 (한 대에 전부도 가능). DDS(같은
`ROS_DOMAIN_ID`)로 통신하므로 역할별로 나눠도 코드는 동일하다.

| 역할 | 담당 | 필요한 Step |
|---|---|---|
| **control** | OpenArm / Tesollo / RH56F1 드라이버 | 1~3, 5 |
| **vision** | D435i + FoundationPose → `/cup_pose` | 1~3, 5, **6~8** |
| **policy** | pour 정책 추론 (GPU) | 1~5 (Step 4-B 포함) |
| sim | Isaac Sim (sim-shadow 검증용, 선택) | 1~3 + Isaac Sim 설치본 |

---

## Step 1. OS·GPU 전제 확인

- Ubuntu 22.04 LTS, x86_64 (Isaac ROS 공식 지원 조합)
- vision/policy 역할이면 NVIDIA GPU + 드라이버 (Blackwell 계열은 드라이버 570+)

```bash
# NVIDIA 드라이버 (vision/policy PC만)
sudo ubuntu-drivers install   # 또는 원하는 버전 명시
```

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

## Step 3. 레포 클론 + 빌드

```bash
git clone https://github.com/divingyoon/sim2real.git
cd sim2real
./scripts/build_vendor_pkgs.sh    # isaacsim_bridge + OpenArm/Tesollo vendor 전체
source install/setup.bash
```

### Step 3-B. RH56F1 손을 쓸 경우 (inspire_ws 별도 빌드)

```bash
cd vendor/inspire_ws
colcon build
source install/setup.bash
cd ../..
```

**확인**
```bash
ros2 pkg list | grep -E "isaacsim_bridge|openarm|dg5f"
# RH56F1 사용 시: ros2 pkg list | grep inspire
```

## Step 4. Python 의존성

```bash
# 공통 (릴레이·테스트)
pip install numpy pyyaml pytest
```

### Step 4-B. policy PC만 — 정책 추론 스택

```bash
# torch: 본인 GPU에 맞는 CUDA 빌드 선택 (Blackwell=cu128+)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install rl-games

# FABRICS: 레포 형제 디렉토리에 배치 (pour_inference.py가 자동 탐색)
#   <workspace>/sim2real          ← 이 레포
#   <workspace>/repo/FABRICS      ← 여기 또는 <workspace>/hdgp/source/FABRICS
```

**확인**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"   # True
python3 -c "import rl_games; print('rl_games OK')"
```

## Step 5. 회귀 테스트 게이트

sim 학습 코드와의 정합(drift-guard 포함)을 확인하는 순수 로직 테스트.
**하나라도 실패하면 실기 구동 금지.**

```bash
cd scripts
python3 -m pytest test_pour_obs_geometry.py test_pour_obs_builder.py \
    test_palm_fk.py test_pour_action_decoder.py test_cup_pose_relay.py -q
# → 46 passed
```

> drift-guard 테스트 일부는 형제 디렉토리 `hdgp/`(학습 레포)가 있으면 학습
> 코드와 직접 대조하고, 없으면 skip된다. 학습 코드를 변경한 PC에서는 hdgp를
> 옆에 두고 돌리는 것을 권장.

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

설치·모델 다운로드·실행 상세는 **`USAGE_ISAACSIM_ROS2.md` §7-1** 참조. 요약:

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

역할별 브링업·테스트 절차는 **`USAGE_ISAACSIM_ROS2.md`** 를 따른다:

| 하고 싶은 것 | 문서 |
|---|---|
| 로봇별 제어 → sim 연결 → test | USAGE §1(OpenArm) §2(Tesollo) §3(RH56F1) |
| 팔+손 통합 | USAGE §4 |
| 하드웨어 없이 배선 확인 | USAGE §5 (dry-run) |
| 비전 노드 (`/cup_pose`) | USAGE §7 |
| pour 정책 추론 | `scripts/pour_inference.py --help`, `SIM2REAL_INFERENCE.md` |

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
| `setup_check.sh` MISS | 표기된 Step으로 이동 |
| PC끼리 토픽 안 보임 | 모든 PC `ROS_DOMAIN_ID` 동일 + 같은 서브넷 + 방화벽(UDP 멀티캐스트) |
| `docker: unknown runtime nvidia` | Step 6의 `nvidia-ctk runtime configure` + docker 재시작 |
| torch가 GPU 커널 에러 (Blackwell) | cu128 이상 빌드로 재설치 (Step 4-B) |
| 회귀 테스트 실패 | 학습 코드(env_cfg/preset)가 바뀐 것 — sim2real 포팅 상수 재정합 필요 |
