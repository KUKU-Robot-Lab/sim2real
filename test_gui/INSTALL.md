# test_gui 설치 및 실행

이 문서는 `/home/user/Downloads/test_gui.zip` 파일을 현재 ROS 2 워크스페이스에 배치하고 실행하는 절차를 정리합니다.

## 기준 환경

- OS: Ubuntu
- ROS 2: Humble
- 워크스페이스: `/home/user/rl_ws/sim2real`
- 패키지 경로: `/home/user/rl_ws/sim2real/test_gui`

## 1. 준비 사항

Qt Creator는 이미 설치되어 있다고 가정합니다.

ROS 2 환경을 먼저 불러옵니다.

```bash
source /opt/ros/humble/setup.bash
```

## 2. ZIP 파일 배치

압축 파일:

```bash
/home/user/Downloads/test_gui.zip
```

압축을 풀 때는 `test_gui` 패키지 디렉터리가 워크스페이스 아래에 위치하도록 배치합니다.

예시:

```bash
cd /home/user/rl_ws/sim2real
unzip /home/user/Downloads/test_gui.zip
```

압축 해제 후 최종 구조:

```text
/home/user/rl_ws/sim2real/test_gui
├── CMakeLists.txt
├── package.xml
├── launch/
├── include/
├── src/
└── resources/
```

참고:

- ZIP 안에 `.git/` 디렉터리가 들어있을 수 있으나, 없어도 됩니다.
- 현재 패키지는 워크스페이스에 이미 배치되어 있습니다.

## 3. 빌드

워크스페이스 루트에서 빌드합니다.

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
colcon build --packages-select test_gui
```

빌드가 끝나면 overlay 환경을 불러옵니다.

```bash
source /home/user/rl_ws/sim2real/install/setup.bash
```

## 4. 실행

다음 명령으로 GUI를 실행합니다.

```bash
cd /home/user/rl_ws/sim2real
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch test_gui gui.launch.py
```

## 5. 확인 결과

현재 워크스페이스에서 아래 항목을 확인했습니다.

- `colcon build --packages-select test_gui` 성공
- `ros2 launch test_gui gui.launch.py` 실행 시작 확인
- 런치 파일 경로: `test_gui/launch/gui.launch.py`

## 6. 실행 시 참고

실행 중 아래 경고가 보일 수 있습니다.

```text
QMetaObject::connectSlotsByName: No matching signal for on_timer_count()
libpng warning: iCCP: known incorrect sRGB profile
```

현재 확인 범위에서는 GUI 시작 자체를 막는 치명 오류는 아니었습니다.
