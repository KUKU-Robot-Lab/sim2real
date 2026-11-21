# 제조사 기반 팔 교정 → RH56F1 재장착 확인


> **검증 상태 (11.15)**: 무구동 시험(66개)까지만 통과했다. 실기에서는 새 프로파일이 한 번도 돌지 않았다.
> 11.12 오른팔 실패(j4 정지마찰 해제 직후 0.129 rad/s 로 속도 한계 0.1 중단)의 원인은 확정하지 못했다.
> 11.15 수정: (1) 준비 이동 종료 판정이 제조사 FF 0 처짐(g/Kp)을 고려, 영점 저장 직전은 원래 0.004 rad (2) 사전 경로 검사도 보정 전 오프셋 여유(max_offset) 적용 (3) plan 이 명령별 경로만 검사(`--for measure|apply|verify`).
> 다음 실기는 전체 탐색이 아니라 j4 하나의 제한 응답 확인부터, 실행 전 매번 승인.
2026-11-13 설정: `openarm-can-1.2.9-arm-sequence-v2`.
오른팔(can0, ID 1–7) 다음 왼팔(can1, ID 1–7) 순서다. 목과 stock gripper(ID 8), RH56F1 손가락을 구동하지 않는다.
RH56F1와 ADAPTOR를 함께 분리하고 bare L7을 유지한다.

## 제조사 코드에서 가져온 것

기준: `robot_control/ros_ws/src/openarm_can/setup/openarm-can-zero-position-calibration`.
원본은 수정하지 않으며 실행 모듈은 `sim2real/scripts/ops/arm_zero_auto.py`에 있다.

- 순서: j4 → j2 우 +5°/좌 −5° → j3 → j3 영점 → j2 원복 → j4 +90° → j5 → j5 영점 → j6 → j6 영점 → j7 → j7 영점 → j2 → j2 영점 → j1 → j1 영점 → j4 영점.
- 스톱 기준각: 오른팔 `[-80,-10,90,0,90,45,90]°`, 왼팔 `[80,10,90,0,90,45,90]°`.
- 탐색 게인: Kp 45 / Kd 1.2, j1만 280 / 2.1.
- 이동 게인: 52 / 1.5, j2 스톱 후 복귀는 280 / 2.0.
- 유지 게인: Kp `[300,300,150,150,40,40,30]`, Kd `[2.5,2.5,2.5,2.5,0.8,0.8,0.8]`. FF 토크는 제조사처럼 0이다.

기존의 j4 +0.225rad 준비 이동과 각 탐색 직전 스톱 근처로 보내던 동작을 제거했다.
차렷 자세의 j4가 이미 스톱에 있어도 첫 위치 확인은 허용한다. 이후 세 번은 실제 후퇴·재접근과 0.5° 이내 반복성을 요구한다.

제조사와 의도적으로 다른 부분:

- 원본의 0.2°/5ms 무한 탐색 대신 지령 속도 0.05rad/s, 실제 속도 중단값 0.1rad/s를 유지한다. 원본과 같은 속도 설정이라고 해석하지 않는다.
- 차렷에서 +90°까지 탐색하므로 고정 탐색 상한은 2rad/100초, 전체는 30rad/1800초다. 실패 후 자동 확대하지 않는다.
- 모든 팔 모터에 유지 지령을 주기적으로 보내고, 새 상태·토크·충돌·정지 지속을 확인한다. 스톱은 중력을 뺀 접근 방향 토크로 판정한다.
- 원본의 미구현 정밀 홈 함수와 `set_zero_all()`은 쓰지 않는다. 실제 영점 도달과 disable 후 정지를 검증한 관절 하나만 저장한다.
- 원본 `GRIPPER=0`과 `J1=0`의 enum 별칭 문제를 가져오지 않는다. j1 기준각은 별도 팔 참조표를 사용한다.

## 1. 손·ADAPTOR 제거 상태

```bash
cd ~/rl_ws/sim2real
# 설정 생성/갱신만 한다. 장치 접근 없음.
bash scripts/ops/arm_zero.sh setup --side right --hand detached
bash scripts/ops/arm_zero.sh setup --side left --hand detached

# 오른팔 자동 측정. 차렷 자세, 팔 지지, 이동 공간 확보 후 실행.
bash scripts/ops/arm_zero.sh run --side right --hand detached --execute
```

`detached`의 기본 동작은 `measure`다. 모터가 움직이며 스톱 후보를 생성하지만 **영점은 아직 저장하지 않는다**.
경유점·게인·목 각도·자체 충돌 제외쌍을 입력하지 않는다. 실제 준비 상태 확인과 실행 직전 승인만 받는다.
기존 팔 CAN 제어는 운영자가 정상 종료해야 한다. 이 도구는 별도 ROS 팔 bringup 없이 CAN을 직접 초기화하며 다른 제어 프로세스를 자동 종료하지 않는다.

측정이 완료되면 출력된 후보를 검토하고 관절별로 저장한다. 아래는 j4의 예다.
같은 형식으로 `--joint`를 3, 5, 6, 7, 2, 1로 바꾸어 각각 실행한다. 전 관절 일괄 저장 명령은 제공하지 않는다.

```bash
bash scripts/ops/arm_zero.sh run --side right --action apply --joint 4 --hand detached --execute
```

후보와 누적 저장 이력은 팔별 설정의 최신 분석 경로에서 자동 선택한다. 일부 저장 상태에서도 저장된 관절의 delta는 다시 적용하지 않는다.
관절마다 현재 스톱과 후보가 일치하는지 재확인한다. 중단 시 자동 재시도나 미검증 홈 복귀를 하지 않는다.

운영자가 전원을 재투입하고 새 좌표를 확인한 뒤, 손 제거 상태의 동일 자세 비교 기준을 기록한다.

```bash
bash scripts/ops/arm_zero.sh run --side right --action verify --hand detached --execute
```

검증 목표는 제조사 영점 → j4 +90°(나머지 0°) → 제조사 영점이다. 기존 작업용 홈·탁자 자세를 사용하지 않는다.
왼팔은 위 명령들의 `--side right`를 `--side left`로 바꾸어 같은 순서로 수행한다.

## 2. RH56F1·ADAPTOR 재장착 상태

첫 단계의 영점·후보·누적 이력을 유지하고, 같은 세 목표 관절각을 사용한다.
부착 후 차이를 영점에 더하거나 제거/부착 값을 평균하지 않는다.

```bash
bash scripts/ops/arm_zero.sh run --side right --hand attached --execute
bash scripts/ops/arm_zero.sh run --side left --hand attached --execute
```

`attached`의 기본 동작은 `verify`다. `measure`와 `apply`는 차단한다.
완료된 detached 검증 bag과 같은 목표·영점 이력이 있어야 enable한다.
검증 기준을 기록한 뒤 영점을 변경했다면 detached 검증을 다시 해야 한다.

**현재 부착 단계의 미확인 항목:** 지정 URDF에는 별도 어댑터 모델이 없다(`../urdf/URDF_LAYOUT.md`의 link7 설명).
따라서 팔별 설정 `model.adaptor_payloads`, `attached_payload_audit`, 실제 `finger_positions`가 확인되기 전에는 부착 상태 구동을 차단한다.
어댑터 필드 형식은 [실험 문서](2026-11-12_arm_zero_calibration.md)에 있다. 손가락 지령을 보내거나 미확인 질량을 0으로 채우지 않는다.

## 기록과 검증 상태

상태·지령·단계·영점 이벤트는 기존 rosbag으로 기록하고, 종료 후 `logs/analysis/arm_zero/<기록 이름>/`에 후보·이력·비교표를 만든다.
`run_status.yaml`은 실패 원인을 남긴다. 실패한 런으로 이전 정상 후보를 덮어쓰지 않는다.
`inspect`는 선택 가능한 읽기 전용 보조 기능이며 자동 교정을 대신하지 않는다.

무구동 회귀 검사에는 양팔 전체 순서, 시작 시 j4 접촉, 세 번 반복 접근, 제조사 게인, 단일 관절 저장, 부분 저장 좌표, 오래된 응답, disable 후 이동, 부착 검증 기준 검사가 포함된다.
모의 CAN은 제어 흐름 검사이며 실제 마찰·하중·속도 안정성을 증명하지 않는다.
이 프로파일의 실기 성공과 영점 저장·전원 재투입·실측 대칭은 아직 검증하지 않았다.
