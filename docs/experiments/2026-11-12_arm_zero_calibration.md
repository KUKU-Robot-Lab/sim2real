> 2026-11-13: 제조사 팔 순서·게인 기반 프로파일로 갱신했다. 기본 동작은 detached=measure, attached=verify다. 최신 명령과 차이점은 [운영자 안내](arm_zero_operator.md)를 따른다. 아래는 설계·이력 자료다.

# 양팔 영점 보정과 RH56F1 재장착 검증

직접 실행 진입점과 순서: [운영자 실행 안내](arm_zero_operator.md). `bash scripts/ops/arm_zero.sh setup --side right`로 시작한다.

현재 상태: **이전 프로파일의 실기 측정은 j4 준비 이동 중 중단됐다. 새 제조사 기반 프로파일은 코드·무구동 검사까지 수행했으며, 실기 성공·영점 저장·전원 재투입 검증은 미완료다.**
`plan`의 기본 결과는 `ready: false`(종료 코드 2)다. 실제 준비 상태와 토크·경로를 모르는 값을 임의로 채우지 않는다.

## 범위와 확정 사항

오른팔 → 왼팔 → 양손 재장착 → 실제 대칭 검증 순서다. 한 번에 한 팔만 사용한다.

2026-11-12 운영자 확인:
> RH56F1와 L7 링크들 사이에 있는 ADAPTOR도 같이 분리해서 오픈암의 그리퍼 파트만 제거된 상태로 실험 한다고 보면돼

따라서 양손과 ADAPTOR를 함께 떼고 **bare L7은 유지**한다. 탁자 이동·분리·장착·지지대·전원 조작은 운영자가 한다.
오른팔 j3의 과거 5.34°는 이미 현재 엔코더 좌표에 포함되어 있다. 다시 적용하지 않는다.
이번 복귀 중 j7 오류는 사용자 외력 사건이며, 보정값 추정 자료에서 제외한다.
손 부착 전후 결과를 평균하지 않고 손 하중 차이를 영점에 더하지 않는다.

## 제조사 근거와 변경점

요청에 포함된 [제조사 절차](https://docs.openarm.dev/setup/openarm-setup/motor-config#step-2-zero-position-calibration)는 최초 웹 도구에서 404를 반환했으나 이후 curl로 HTTP 200 본문을 읽었다(공식 페이지 갱신일 2026-11-10). 기준 영점 근처의 물리 시작 자세와 한 팔씩 자동 보정 요구를 확인했다. 상세 순서·임계값은 **로컬 제조사 소스에서 직접 확인**했다.

- `../robot_control/ros_ws/src/openarm_can/setup/openarm-can-zero-position-calibration`
- 저장소 HEAD: `396cf537752641b8d0fc782dd08853d8fa0a2cc4`
- 파일 SHA256: `c6cdda3b1ee3d44637fc08a6c28724d90ce2b45cb0969c8092f31c7baee80423`
- 상태/단일 모터 패킷 근거: 같은 저장소 `python/src/openarm_can.cpp`, `src/openarm/damiao_motor/dm_motor_control.cpp`, `include/openarm/damiao_motor/dm_motor_constants.hpp`.
- 기존 수신·제어 참조: `scripts/ops/sym_check.py`. 이 파일을 import하지 않는다(실험 로그의 FK 코드에 의존함).

제조사의 팔 탐색 순서는 j4 → j3 → j5 → j6 → j7 → j2 → j1이다. 그리퍼 초기화와 구동은 제거한다.
원본은 `move_to_precise_home()`가 미구현이고 마지막에 `set_zero_all()`을 호출한다. 이 부분은 사용하지 않는다.

| 관절 | 오른팔 스톱 (deg) | 왼팔 스톱 (deg) | 접근 방향 |
|---|---:|---:|---|
| j1 | −80 | +80 | 우 − / 좌 + |
| j2 | −10 | +10 | 우 − / 좌 + |
| j3 | +90 | +90 | + |
| j4 | 0 | 0 | − |
| j5 | +90 | +90 | + |
| j6 | +45 | +45 | + |
| j7 | +90 | +90 | + |

제조사 중간 동작은 j4 스톱 이후 j2를 우 +5°/좌 −5° 이동 → j3 탐색 → j3 −90° → j2 원복 → j4 +90°,
이후 j5/j6/j7 탐색마다 각각 −90°/−45°/−90° 후퇴, j2 탐색 뒤 우 +10°/좌 −10°, j1 탐색 뒤 우 +80°/좌 −80°다.
이 상대 동작을 현재 상태와 무관하게 재생하지 않는다. `measure_routes`에 각 탐색 전의 **검토된 절대 모델 좌표 경유점**을 넣고 경로 검사를 통과시킨다. 제조사와 달리 실패 후 홈 복귀는 없다.

원본 스톱 판정은 j1에서 |dq|<0.0125, |tau|>5 Nm, 나머지에서 |dq|<0.1, |tau|>2 Nm다.
이 도구는 손·ADAPTOR 제거 모델의 중력 토크를 빼고 **접근 방향 잔여 토크**로 판정한다. 이 임계와 전체 토크 상한은 운영자 문답에서 입력받지 않는다. 기본 임계는 제조사 값, 전체 상한은 v2 URDF effort 한계를 사용하며 실제 구동 검증은 별도다.
정지 속도 0.0125 rad/s는 DM8009의 ±45 rad/s, 12bit 양자화 최소 절댓값 `45/4095 = 0.01098901098901099`보다 크다.
실제 속도·토크·위치 응답이 이 모델에 맞는지는 실기 전 별도 확인 대상이다.

## 구현과 모델

- `scripts/ops/arm_zero_calibrate.py`: 순수 제어 코어와 네 명령의 CLI.
- `scripts/ops/arm_zero_model.py`: 메모리상 URDF 제거, 양팔 FK·중력, 보수적 충돌 메쉬 AABB 검사.
- `scripts/ops/arm_zero_runtime.py`: 승인된 실행에만 로드하는 SocketCAN/SDK·ROS 어댑터.
- `scripts/ops/arm_zero_bag.py`: 종료한 기존 SQLite rosbag에서 후보·이력·비교표를 만드는 무구동 분석기.
- `config/arm_zero_calibration.yaml`: CAN·기준각·한계·준비 상태·경유점. null은 미확인이지 기본 승인값이 아니다.

URDF 원본:
`../urdf/generated/rl/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf`
SHA256: `c3809752c5dd6013ca9a6266bd904583ff223a728c8e6f9c1f093af8ee141d32`.

제거 모델은 확인된 자산 `../hdgp/assets/robot/openarm_rh56f1_bi_v2_rl/openarm_rh56f1_bi_v2_rl.urdf`에서
`r_hj_mount/l_hj_mount` 자식 subtree만 제외하고 `r_al_7/l_al_7`을 보존한다. 원본 해시
`87009f2c70a28890a616fe972df8569a062b7d2d2ba43a2b984bd09381eb868f`를 설정과 실험 기록에 고정한다.
URDF에서 양쪽 L7 질량이 `0.4659771327380578 kg`, L7 collision mesh가 `link7_flange_cut.stl`임을 확인했다.
사용자가 알려준 대로 이 자산은 ADAPTOR를 제외한 L7이며 RH56F1 base는 별도 fixed child다.
따라서 분리 모델은 사용자가 지정한 물리 상태를 나타낸다. `bare_l7_audit`에는 이 URDF/자산 식별자를 기록했다. 기존 어댑터가 L7에 합쳐져 있다면
맞는 자산을 별도로 준비해야 하며, 원본 URDF에서 임의 질량을 빼지 않는다.

재장착 상태는 모든 독립 손가락 관절의 실제 각도와 양쪽 어댑터의 누락 질량·COM·L7 좌표계 AABB를 요구한다.
`adaptor_payloads`는 `r_al_7`, `l_al_7` 각각 `{mass, com, box_lower, box_upper}`이며 단위는 kg, m다.
원본에 이미 포함된 질량은 중복하지 않는다(0 추가 질량도 근거가 필요). 손은 이 도구에서 구동하지 않는다.

충돌 검사는 메쉬를 포함하는 AABB로 하므로 실제로 충돌하지 않는 자세도 거부할 수 있다.
기본 충돌 제외는 해당 자산 manifest의 `self_collision_filtered_pairs`에서 자동으로 가져온다. `geometry.exclusions`는 별도 검토된 추가 설정만을 위한 항목이며 운영자 문답에서 입력받지 않는다. 손 전체나 팔 전체를 일괄 제외하지 않는다.
탁자·지지대·몸통 주변 장애물은 `geometry.obstacles`에 기록하고, 반대 팔은 지지된 고정 자세를 `other_arm_q`로 명시한다.
케이블·외력·모델 오차는 이 검사로 판정할 수 없다. `geometry.review`와 실제 관찰을 대체하지 않는다.
탐색/저장 직전 재측정은 아직 영점이 미확정이므로 실제 상태의 URDF 한계에 `max_offset`만큼만 허용차를 둔다.
스톱 접촉용 가상 PD 목표와 실제 링크 자세는 분리해서 검사한다. 충돌 검사는 현재 추정 좌표 기준이며 **절대적인 실기 무충돌 보증이 아니다.**

## 명령과 무구동 점검

저장소 루트에서 다음 명령은 장치를 열지 않는다.

```bash
python3 scripts/ops/arm_zero_calibrate.py plan --side right --hand detached
python3 scripts/ops/arm_zero_calibrate.py plan --side left --hand detached
python3 scripts/ops/arm_zero_calibrate.py plan --side right --hand attached
python3 scripts/ops/arm_zero_calibrate.py plan --side left --hand attached
```

`plan`은 `--execute`를 붙여도 장치를 열지 않는다. `measure/apply/verify`도 `--execute` 없이는 같은 무구동 검사만 한다.
기본 설정에서는 경계 모델을 검사하고 빠진 실제 조건을 모두 출력하며 코드 2로 종료한다. 이를 경로 합격으로 해석하지 않는다.

| 명령 | 실행 동작 |
|---|---|
| `measure --side right --execute` | 제조사 순서로 한 팔의 스톱 반복 측정·후보 발행. 영점 쓰기 없음 |
| `apply --side right --joint 4 --candidate … --history … --execute` | 지정 관절의 스톱을 다시 측정해 현재 좌표계와 후보 일치 확인 → 실제 모델 0° 이동·정착 → 해당 모터만 disable·검사·1회 저장 |
| `verify --side right --hand detached --candidate … --history … --execute` | 같은 모델 관절각에서 엔코더·FK 추정 플랜지 기록 |
| `verify --side right --hand attached --candidate … --history … --execute` | 동일 후보·동일 검증 경로로 재장착 상태 기록. 영점 쓰기 없음 |

위 표의 실행 예시는 **아직 승인되지 않았다**. `sim2real/CLAUDE.md`의 “로봇이 움직이거나 힘을 내는 명령은 실행 직전에 매번 … 답을 기다린다”에 따라,
설정·무구동 결과·정확한 명령을 운영자에게 제시하고 그 명령 한 번의 승인을 받은 뒤에만 실행한다.
실기 환경은 명시적인 `ROS_DOMAIN_ID=50`이 필요하다. 이 세션에서는 설정하지도, 실기 명령을 실행하지도 않았다.

## 탐색·저장 규칙

1. 전 관절 새 응답을 받은 뒤에만 enable한다. 현재 읽기가 `start_raw`에서 0.02 rad 이상 다르면 거부한다.
2. 전체 작업 공통 잠금으로 양팔 동시 실행을 막는다. CAN 수신 등록을 확인하고 can0/can1 사용자가 있으면 거부한다. 다른 프로세스를 kill하지 않는다.
3. 탐색 속도 ≤0.05 rad/s. 기본 접근별 최대 이동 0.35 rad, 30초, 실제 누적 전체 이동 12 rad, 전체 900초다. PD 요청 토크와 실제 토크를 모두 제한한다.
4. 최초 locating pass는 버린다. 동일 방향으로 후퇴 0.07 rad 후 3회 접근한다. 실제 방향 이동 ≥0.035 rad, 정지 ≥0.3초, 방향성 잔여 토크 조건을 모두 요구한다.
5. 세 위치의 max−min ≤0.5°만 후보가 된다. `δ = mean(stop_raw) − reference`; 기준 자세는 `q_raw = q_model + δ`다.
6. 스톱 확인 중 가상 목표를 계속 밀어 넣지 않는다. 조기 정지·timeout·범위·수신·온도·유지 오차 실패 시 강도·범위를 늘리지 않는다.
7. 경로 전수 검사 중에도 현재 자세 유지 명령과 감시를 반복한다. 탐색 외 6관절도 매 틱 유지한다. 각 모터의 커널 수신 timestamp와 새 응답 sequence를 확인한다.
8. `apply`는 bag 원본으로 후보/부분 저장 이력을 재검사하고, 선택 관절을 현재 좌표에서 재측정한다. 과거 후보와 0.004 rad 넘게 다르면 저장하지 않는다.
9. 모델 0°의 실제 목표 도달·정착을 확인한 뒤 선택 모터만 disable한다. 0.3초 동안 위치 변화·속도·disabled 상태를 확인한다. 지지대는 필수다.
10. zero write는 지정 ID 하나에 한 번만 보낸다. 저장 시도 이벤트 이후 검증 이벤트가 없으면 **UNKNOWN**이다. 재저장·홈 복귀하지 않는다.
11. 저장 후 새 좌표계를 먼저 적용하고 새 요청 이후 수신한 값으로 0°를 확인한다. 이전 목표·캐시는 폐기한다. 완료된 관절 δ만 0으로 만들고 나머지는 유지한다.
12. 종료/중단에는 허락된 팔의 disable 정리를 시도한다. 이미 enable되어 있던 다른 제어를 발견한 경우에는 건드리지 않는다. disable 전후 안전한 기계적 지지가 있어야 한다.

## 두 단계 실기 절차

### A. 손·ADAPTOR 제거

1. 양손·어댑터 분리, 탁자 이동, 양팔 지지, 케이블 여유, CAN 소유자와 새 상태 읽기, bare L7 모델 근거를 확인한다.
2. 제조사 순서에 맞춘 `measure_routes`, 현재 `start_raw`, 반대 팔 고정각, 검증 경로, 토크/충돌 근거를 설정하고 `plan` 통과를 확인한다.
3. 기존 `deploy/policy_control/tools/rh56f1_record.sh`로 bag 기록을 시작한다(노드 작업 승인은 별도). 이 기록기의 PD bag에 `/arm_zero_calibration/events`가 추가되어 있다.
4. 오른팔 `measure` 한 번을 승인받아 수행한다. 종료한 bag을 분석하고 각 스톱의 케이블·몸통·외력 접촉 여부를 검토한다. 의심 후보는 폐기한다.
5. 후보 분석 YAML의 `reviewed_no_contact: true`는 위 검토를 끝낸 운영자가 표시한다. 측정 수치·히트 배열은 수정하지 않는다.
6. 관절별 `apply`를 각각 승인받는다. 매번 현재 자세·경로를 다시 확인하며 기록한다. apply 자체에 선택 관절 재측정이 포함된다.
7. 각 명령 후 bag을 정상 종료하고 측정 bag **및 모든 후속 apply bag**으로 이력을 다시 만든다. 이전 history를 재사용하지 않는다. UNKNOWN이 있으면 좌표계를 조사하고 진행을 멈춘다.
8. 운영자가 전원을 재투입하고, 새 응답에서 유지 여부를 확인한다. 자동 이벤트의 `power_cycle_verified`는 계속 false이며 소프트웨어가 전원 재투입을 증명하지 않는다.
9. 동일 검증 경로의 여러 자세에서 encoder·플랜지 위치/방향·고정 촬영 위치의 사진을 남긴다. 왼팔에 같은 절차를 적용한다.

### B. RH56F1·ADAPTOR 재장착

1. 운영자가 재장착하고 손가락 실제 자세·어댑터 하중/형상을 반영한다. 새로운 `plan --hand attached`를 통과한다.
2. A의 candidate/history와 **같은 verify_route**로 같은 촬영 위치에서 반복한다. 경로가 다르면 도구가 거부한다.
3. 목표 대비 실제 관절각, 실측 플랜지 위치·방향, 플랜지 대비 손 방향을 별도로 비교한다.
4. 차이가 나도 영점을 다시 쓰지 않는다. 추종 오차, 하중 변형, 장착 차이로 나누어 조사한다.
5. 실제 대칭이 남아 있으면 보정 완료가 아니다. 남은 관절·플랜지·손 장착 차이를 명시한다.

## 기록과 분석

원시 상태(q/dq/torque/수신시각/sequence/enabled), 전 관절 명령(q/중력/KP/KD), 단계, 중단, 영점 시도/확인은 모두
`std_msgs/msg/String`의 `/arm_zero_calibration/events`로 발행한다. 별도 JSONL/NPZ/CSV 원시 기록기는 없다.
기존 rosbag recorder 구독이 없거나 사라지면 실행을 거부/중단한다. DDS ACK는 디스크 영구 저장 증명이 아니므로 정상 bag 종료·원본 확인도 필요하다.

```bash
# 종료한 측정 bag + 모든 후속 저장/검증 bag을 명시한다. 장치 접근 없음.
python3 scripts/ops/arm_zero_bag.py <측정-bag> <apply-bag-1> <apply-bag-2> --out <새-분석-디렉터리>
```

생성물은 `candidate.yaml`, `history.yaml`, `comparison.md`다. 기존 분석 디렉터리는 덮어쓰지 않는다.
측정 후보는 원본 bag과 모델·side·session으로 묶인다. 부분 저장은 `saved_joints`, 중단/확인 실패는 `unknown_joints`에 남는다.
분석은 현재 기록 체계인 SQLite `.db3`를 읽는다. 후보가 완성되지 않은 측정은 부분 후보로 저장에 사용하지 않는다.

비교표의 플랜지는 **encoder+URDF FK 추정**이며 외부 실측값이 아니다. 실측 플랜지·손 상대 자세 필드는 null로 남긴다.
사진만으로 보정되지 않은 mm/각도 정밀도를 주장하지 않는다. 외부 측정 기준/촬영 기하가 검증된 뒤 측정값과 불확도를 별도로 분석한다.

| 실기 합격 항목 | 현재 |
|---|---|
| bare L7 모델/손·어댑터 질량·분리 경계 검토 | 사용자가 지정한 v2 URDF와 링크 질량·충돌 mesh 일치. 재장착 어댑터 실측 질량·COM/AABB 대기 |
| 실제 환경·유지/탐색 토크·경로 점검 | 대기 |
| 스톱 반복성·접촉 배제 | 미실행 |
| 단일 관절 저장 후 새 응답 | 미실행 |
| 전원 재투입 후 유지 | 미실행 |
| 제거/부착 여러 자세 비교·실제 대칭 | 미실행 |

## 소프트웨어 검증

`tests/ops/test_arm_zero_calibrate.py`, `tests/ops/test_arm_zero_bag.py`는 하드웨어 없이 좌우 부호, 반복성, 실제 이동/조기 정지,
미도달, 오래된 응답, disable 후 이동, 저장 실패, 단일 저장, 부분 좌표, 저장 직후 새 좌표 검사, 접촉 목표 한계,
경로 검사 중 유지, 이동 시간 상한, 비선택 모터 disable, bag의 UNKNOWN·후보 생성, 무구동 장치 접근 금지를 검사한다.
기존 recorder 테스트와 함께 실행하며 Python 진단은 pyright·ruff를 사용한다.
실제 CAN/모터·ROS recorder 통합은 아직 검증하지 않았다. 모의 검사를 실기 합격으로 대체하지 않는다.

## 팔 전용 설정 변경

운영자 요청으로 목 각도 입력과 목 자세 준비를 필수 조건에서 제외했다. `head_positions: null`이면 head pan pivot부터 각 collision mesh까지의 고정 이동 길이 합과 mesh 모서리 최대 반경으로 모든 머리 회전 자세를 포함하는 영역을 구성한다. 팔과 이 영역의 충돌 검사는 유지한다. 목 구동은 없다.

## 자동 프로파일 추가

최신 운영자 진입점은 숫자 입력을 없앴다. `setup`은 무구동으로 제조사 기반 프로파일을 저장하고, `run --execute`는 recorder 시작 후 양팔 disabled 상태를 읽어 경로를 생성한다. 상세 절차/상한/검증 상태는 [운영자 안내](arm_zero_operator.md)를 따른다. 실제 구동·영점 쓰기는 이 변경 검증 중 수행하지 않았다.
