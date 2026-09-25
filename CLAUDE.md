# sim2real 작업 규칙

이 저장소의 명령은 실기(OpenArm 양팔, DG-5F 양손, 목)를 움직인다. 코드 작업보다 실기 안전이 먼저다.

## 실기 허락
- 로봇이 움직이거나 힘을 내는 명령은 실행 직전에 매번 "이 명령을 지금 실행할까요?"로 멈추고 답을 기다린다.
- 대상: `--execute`, `execute:=true`, `--approve <id>`, pd engage·goto_home·preset·`replay_to_pd`, `ep_start`, 손 지령 발행, 로봇 명령 토픽에 쓰는 `ros2 topic pub`·`ros2 service call`·`ros2 action send_goal`, 실하드웨어 launch.
- 허락은 그 명령 한 번에만 유효하다. 같은 단계를 다시 돌려도 다시 묻는다.
- 발열·충돌 같은 긴급 상황도 같다. 상태를 즉시 보고하고 조치안을 내되 실행은 허락 뒤에 한다.
- 드라이버·pd·FPP 노드의 기동과 kill, 남은 프레임워크(이전 세션 노드, Isaac) 종료도 허락 뒤에 한다.
- sudo·물리 조작(모터 전원, CAN, 손 네트워크)은 운영자가 한다. Claude 는 실행하지 않는다.
- 허락 없이 해도 되는 것: 상태 토픽 구독, bag·로그 분석, fake 도메인(97/99) 리허설, sim probe, 코드 읽기·수정.

## bringup 점검 순서
단계의 정본은 `config/mission_dg5f_m_control.yaml` 이다. 요지:
1. 드라이버·노드 PID 와 남은 프로세스를 확인한다. 이전 세션 노드가 남아 있으면 보고하고 허락 뒤 내린다.
2. 상태 토픽(`/joint_states`, `/dg5f_*/joint_states`, `/head/joint_states`) 수신과 관절 방향을 실기와 대조한다.
3. pd 는 무발행(`execute:=false`)으로 먼저 띄워 게인·입력을 확인한다. 발행 모드 전환은 허락 뒤.
4. 자세 이동(home, preset, 경로 재생)은 pd 가 한다. 한 팔씩, 단계마다 허락.
5. 제어 점검이 끝나기 전에는 정책 단계로 가지 않는다.
6. 끝낼 때는 받침 확인 → pd 해제 → 손 → 팔 순서로 내린다.
실기 도메인은 `ROS_DOMAIN_ID=126`, fake 는 97/99 다. 도메인을 섞지 않는다.

## 사용자 소유 영역
- 이 저장소, `rl_ws/robot_control/`, `rl_ws/urdf/`, USD 자산은 사용자가 요청한 작업 범위 안에서만 고친다. 범위 밖은 제안만 한다.
