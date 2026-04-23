# Pour-Mimic-V1 ROS2 Demo Recording

이 문서는 `Pour-Mimic-V1` 데모를 **방법 B** 기준으로 수집하는 절차를 정리한다.

방법 B의 핵심은 `ros2_demo_recorder.py`가 HDF5를 직접 임의 포맷으로 쓰는 것이 아니라, IsaacLab 환경을 띄운 뒤 IsaacLab recorder 포맷으로 demo를 export하는 것이다. 그래야 다음 단계의 `annotate_demos.py`와 `generate_dataset.py`가 같은 파일을 읽을 수 있다.

---

## 현재 상태

현재 구현된 것:

- `/isaacsim/right_arm_cmd`, `/isaacsim/right_hand_cmd`, `/isaacsim/left_arm_cmd`, `/isaacsim/left_gripper_cmd` 토픽 계약
- right arm 7D, right hand 20D, left arm 7D 입력을 `Pour-Mimic-V1` 18D action으로 변환하는 코어
- ROS2 없이 import 가능한 dry test path
- `Se3ROS2Device` dry path

아직 완료되어야 하는 것:

- `ros2_demo_recorder.py`가 IsaacLab `env.recorder_manager` 또는 `EpisodeData`/`HDF5DatasetFileHandler` 포맷으로 `initial_state + actions`를 저장하도록 연결
- 실제 ROS2 subscriber loop
- 저장/폐기 UX
- right palm FK provider 연결
- `annotate_demos.py`로 1개 demo가 통과하는 acceptance

따라서 이 문서는 **최종 사용 절차와 acceptance 기준**이다. 현재 코드가 이 절차를 끝까지 만족하지 못하면 Phase 2는 “action 변환 코어 완료, IsaacLab native recorder 연동 미완료”로 봐야 한다.

---

## 왜 native recorder 포맷이 필요한가

IsaacLab Mimic의 `annotate_demos.py`는 입력 HDF5에서 아래 데이터를 읽는다.

```python
initial_state = episode.data["initial_state"]
actions = episode.data["actions"]
env.reset_to(initial_state, None, is_relative=True)
env.step(action)
```

즉 입력 파일에는 최소한 다음이 있어야 한다.

- `/data/demo_*/initial_state`
- `/data/demo_*/actions`
- `/data` group의 `env_args`

단순히 `obs/actions/rewards/dones`만 있는 HDF5는 Mimic annotation 입력으로 부족하다.

---

## 목표 데이터 흐름

```text
실제 로봇 제어 또는 GUI
  -> /isaacsim/right_arm_cmd     Float64MultiArray, 7D
  -> /isaacsim/right_hand_cmd    Float64MultiArray, 20D
  -> /isaacsim/left_arm_cmd      Float64MultiArray, 7D
  -> /isaacsim/left_gripper_cmd  Float64, scalar

ros2_demo_recorder.py
  -> IsaacLab Pour-Mimic-V1-Mimic-v0 env 생성
  -> env.reset()
  -> initial_state 저장
  -> ROS2 command snapshot 수신
  -> 18D action 생성
  -> env.step(action)
  -> IsaacLab recorder 포맷으로 export

/tmp/pour_one_demo.hdf5
  -> annotate_demos.py
  -> /tmp/pour_one_demo_annotated.hdf5
  -> generate_dataset.py
```

---

## 터미널 1: 실기 제어 스택 실행

```bash
source /opt/ros/humble/setup.bash
cd /home/user/rl_ws/sim2real
source install/setup.bash

ros2 launch integrated_control openarm_left_gripper_right_dg5_real.launch.py \
  left_can_interface:=can1 \
  right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

하드웨어 없이 먼저 점검할 때:

```bash
source /opt/ros/humble/setup.bash
cd /home/user/rl_ws/sim2real
source install/setup.bash

ros2 launch integrated_control openarm_left_gripper_right_dg5_real.launch.py \
  use_fake_hardware:=true
```

---

## 터미널 2: isaacsim_bridge 실행

`/isaacsim/*_cmd`를 실제 controller topic으로 넘기는 브리지다.

```bash
source /opt/ros/humble/setup.bash
cd /home/user/rl_ws/sim2real
source install/setup.bash

ros2 launch isaacsim_bridge isaacsim_bridge.launch.py
```

실기까지 같이 붙이는 launch를 쓰는 경우에는 터미널 1과 중복 실행하지 않는다.

---

## 터미널 3: recorder 실행

최종 목표 명령:

```bash
cd /home/user/rl_ws/IsaacLab

TERM=xterm ./isaaclab.sh -p /home/user/rl_ws/sim2real/scripts/ros2_demo_recorder.py \
  --task Pour-Mimic-V1-Mimic-v0 \
  --output_file /tmp/pour_one_demo.hdf5 \
  --num_demos 1 \
  --headless \
  --device cuda:0
```

현재 머신처럼 GPU/driver가 불안정한 경우 CPU smoke:

```bash
cd /home/user/rl_ws/IsaacLab

TERM=xterm ./isaaclab.sh -p /home/user/rl_ws/sim2real/scripts/ros2_demo_recorder.py \
  --task Pour-Mimic-V1-Mimic-v0 \
  --output_file /tmp/pour_one_demo.hdf5 \
  --num_demos 1 \
  --headless \
  --device cpu
```

주의:

- 위 명령이 성공하려면 `ros2_demo_recorder.py`가 IsaacLab native recorder export를 구현해야 한다.
- 현재 action 변환 코어만으로는 Mimic용 HDF5 acceptance를 통과하지 못할 수 있다.
- `Pour-Mimic-V1-Mimic-v0` env reset/step 자체가 실패하면 먼저 IsaacLab scene/CUDA/asset 문제를 해결해야 한다.

---

## 저장/폐기 조작 기준

권장 UX:

- `S`: 현재 episode를 성공 demo로 저장
- `R`: 현재 episode 폐기 후 reset
- `Q` 또는 `Ctrl-C`: 종료

최소 acceptance에서는 키보드 UX 대신 아래처럼 동작해도 된다.

- `--num_demos 1`
- episode horizon 도달
- recorder가 success 여부를 인자로 받아 저장

하지만 최종 Phase 3 수집에서는 실수 episode를 버릴 수 있어야 하므로 `R` 폐기는 필요하다.

---

## right palm FK provider

직접 로봇을 제어하면 Fabrics controller는 필요 없다.

하지만 `Pour-Mimic-V1` action `[0:6]`은 right palm pose delta이므로 recorder는 저장 시점에 FK가 필요하다.

```text
current right arm joints -> current palm pose
target right arm joints  -> target palm pose
target - current         -> 6D palm delta
```

가능한 FK provider:

- IsaacLab env 안의 robot articulation body pose
- URDF/PyKDL 기반 FK
- 기존 sim2real/FABRICS FK 유틸의 FK 부분만 사용

중요한 점:

- 실시간 제어에 Fabrics IK를 쓰지 않아도 된다.
- 데모 저장 action 변환에는 FK가 필요하다.
- FK 결과 pose는 `[x, y, z, qx, qy, qz, qw]` 형식이어야 한다.

---

## 생성 파일 확인

recorder가 성공하면 파일이 생겨야 한다.

```bash
ls -lh /tmp/pour_one_demo.hdf5
```

HDF5 구조 확인:

```bash
python3 - <<'PY'
import h5py

path = "/tmp/pour_one_demo.hdf5"
with h5py.File(path, "r") as f:
    print("root:", list(f.keys()))
    print("data attrs:", dict(f["data"].attrs))
    for demo in f["data"]:
        g = f["data"][demo]
        print("demo:", demo, "attrs:", dict(g.attrs))
        def walk(name, obj):
            if hasattr(obj, "shape"):
                print(" ", name, obj.shape)
            else:
                print(" ", name)
        g.visititems(walk)
PY
```

필수 체크:

- `data.attrs["env_args"]` 안에 `Pour-Mimic-V1-Mimic-v0` 또는 같은 env 이름이 있어야 한다.
- `data/demo_*/initial_state`가 있어야 한다.
- `data/demo_*/actions` shape가 `(T, 18)`이어야 한다.

---

## annotation smoke

1개 demo가 저장된 뒤에 실행한다.

```bash
cd /home/user/rl_ws/IsaacLab

TERM=xterm ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Pour-Mimic-V1-Mimic-v0 \
  --auto \
  --input_file /tmp/pour_one_demo.hdf5 \
  --output_file /tmp/pour_one_demo_annotated.hdf5 \
  --headless \
  --device cuda:0
```

CPU smoke:

```bash
cd /home/user/rl_ws/IsaacLab

TERM=xterm ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Pour-Mimic-V1-Mimic-v0 \
  --auto \
  --input_file /tmp/pour_one_demo.hdf5 \
  --output_file /tmp/pour_one_demo_annotated.hdf5 \
  --headless \
  --device cpu
```

성공 기준:

- `FileNotFoundError` 없음
- `initial_state` KeyError 없음
- env reset/replay 성공
- `Exported 1 (out of 1) annotated episode.`
- `/tmp/pour_one_demo_annotated.hdf5` 생성

---

## generation smoke

annotation이 성공한 뒤에만 실행한다.

```bash
cd /home/user/rl_ws/IsaacLab

TERM=xterm ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Pour-Mimic-V1-Mimic-v0 \
  --device cuda:0 \
  --headless \
  --num_envs 1 \
  --generation_num_trials 2 \
  --input_file /tmp/pour_one_demo_annotated.hdf5 \
  --output_file /tmp/pour_generated_smoke.hdf5
```

성공 기준:

- env 생성
- input annotated dataset 로드
- 최소 1개 이상 generated demo export

---

## Phase 3로 넘어가는 기준

아래가 모두 만족되어야 35개 수집을 시작한다.

- `ros2_demo_recorder.py`가 `/tmp/pour_one_demo.hdf5` 생성
- HDF5에 `initial_state`와 `(T, 18)` actions 존재
- `annotate_demos.py --auto`가 1개 demo 통과
- `generate_dataset.py --generation_num_trials 2` smoke 통과
- 실패 episode를 폐기할 수 있는 조작이 있음

이 기준 전에는 35개를 모아도 Mimic 파이프라인에서 다시 막힐 수 있다.

