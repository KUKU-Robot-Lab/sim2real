# 읽기 전용 Isaac 뷰어 (right_aglt 정책 장면 + 실기 관절 미러)

rviz 대신 Isaac Sim 에서 **정책 학습 장면**(로봇 자산·테이블·컵)을 띄우고, 로봇 관절을 실기 `/joint_states` 로 따라 움직인다.
로봇에 명령을 보내는 경로는 없다.

```
실기 ROS(도메인 126) --구독만--> joint_state_relay.py (py3.10, rclpy) --UDP JSON 127.0.0.1:47811--> isaac_viewer.py (Isaac py3.11, ROS 없음)
```

## 실행

```bash
cd /home/user/rl_ws/sim2real/robot/isaacsim_bridge/viewer
ROS_DOMAIN_ID=126 ./run_viewer.sh                       # 둘 다 띄움(GUI). 창을 닫거나 Ctrl-C 면 relay 도 같이 끝난다
ROS_DOMAIN_ID=126 ./run_viewer.sh --cup cup_big_s115    # 컵 종 바꾸기(기본 cup_big_s100 = 실물 빨간 컵)
```

따로 띄울 때:

```bash
# 터미널 1 — relay
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=126 python3 joint_state_relay.py --port 47811

# 터미널 2 — 뷰어 (ROS 를 source 하지 않은 셸)
/home/user/rl_ws/IsaacLab/isaaclab.sh -p isaac_viewer.py --port 47811
# 검증용 한 장: ... isaac_viewer.py --headless --shot /tmp/view.png --exit_after_shot
```

- `ROS_DOMAIN_ID` 가 비었거나 0 이면 relay·런처 둘 다 거부한다.
- 같은 GPU 에서 학습(python/Isaac 계산 프로세스) 중이면 런처가 띄우지 않는다(`VIEWER_FORCE=1` 로만 무시).
- 부팅은 1~2 분 걸린다(env 생성). 부팅 로그 `[isaac_viewer] 장면 값 출처` 표에 각 값의 출처가 찍힌다.

## 무엇을 하나

- **relay**: `/joint_states`·`/dg5f_right/joint_states`·`/dg5f_left/joint_states`·`/head/joint_states` 를 BEST_EFFORT 로 구독,
  `robot_control/.../profiles/openarm_tesollo.yaml` 의 source->canonical·sign 으로 이름을 바꿔 30 Hz 로 UDP 송신.
  발행자는 rclpy 내부 `/parameter_events` 하나뿐이다(Humble 이 조건 없이 만든다, 부팅 시 점검).
- **뷰어**: hdgp 태스크 `open-short_r_grasp_fj_t2r_rand-lstm` 을 num_envs=1 로 만들고 `deploy/policies/right_aglt/params/env.yaml` 을
  hdgp play.py 와 같은 복원기(`hdgp/scripts/tools/run_cfg_restore.py`)로 덮는다. `env.step` 은 부르지 않는다(정책·액션·물리 스텝 0).
  매 프레임 받은 관절 값을 쓰고(속도 0) `sim.render()` 만 돈다.
- 학습과 다른 점(뷰어 전용): 컵 1종(`--cup`)만 스폰 · 컵 위치는 학습 랜덤(중심 ±0.15) 대신 **중심 (0.25, −0.15) 고정**,
  z = 상판 0.205 + 종별 원점 오프셋 · 지령 마커 끔 · 카메라는 env.yaml `gui_camera_eye/target`.
- 자산이 short-**tl** 이라 `r_hj_thumb_1`(용접)은 sim 에 없다 — 실기 값은 무시된다(로그 1회).
  한계 밖 값은 표시용으로 clamp 하고 로그 1회.

## 시험

```bash
cd /home/user/rl_ws/sim2real && python3 -m pytest tests/isaacsim_viewer -q
```
