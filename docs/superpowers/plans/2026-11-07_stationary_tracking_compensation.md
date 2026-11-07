# RH56F1 stationary tracking 구현 계획 및 검증 원장

기준 사양: [실험 문서](../../experiments/2026-11-07_rh56f1_stationary_tracking.md), 2026-11-07 사용자 승인 구현 계획. 최초 범위는 오른팔이었으며, 후속 요청에 따라 오른팔 측정 자세의 손끝 경로를 좌우 대칭한 왼팔 준비·실행도 추가한다. 세 보상 기능은 기본 OFF다.

이 문서는 구현 인터페이스와 확인 항목을 기록한다. 체크되지 않은 항목은 완료를 의미하지 않는다. 실기 A–G는 아직 실행하지 않았다. 자세 미수집 기록은 초기 계획 시점의 기록이며 아래 2026-11-07 후속 진행에서 갱신한다. 자동 commit/push, 새 branch/worktree를 만들지 않는다. 기존 미커밋 조그·미션 변경은 이 작업 소유가 아니므로 보존한다.

## 2026-11-07 양팔 경로 준비 추가 기록

- 오른팔 6개 자세를 입력으로 사용한다. 01–03은 fit, 04–06은 holdout으로 고정되어 있다. 왼팔 홈과 주변 장면은 `logs/analysis/stationary_tracking_1107/inputs/right_pose_set_v1/scene_complete_attention_left_home.json`에 저장되어 있다. 오른팔은 캡처 당시 attention 자세다.
- `stationary_tracking.py prepare --side left --mirror-right`가 좌우 반사된 손끝 목표를 왼팔 IK로 변환한다. 왼팔 관절각을 단순 부호 반전해 경로로 쓰지 않는다.
- A 계획: `logs/analysis/stationary_tracking_1107/runs/left_mirror_prepare_A_v4/plan.json`; G 계획: `.../left_mirror_prepare_G_v3/plan.json`. 각각 63개 runnable 경로, 63개 충돌 제외가 기록됐다. runnable 경로는 홈 접근·목표 경로·15초 유지·목표 경로와 접근 경로의 역순을 통한 홈 복귀를 포함한다. 복귀는 이미 정적 장면에서 충돌 검사된 경로를 역순으로 사용한다.
- A와 G에서 각각 trial 하나씩 무발행 dry run을 실행했고, 양쪽 모두 `commands_sent=0`이다. 실기 이동·힘 명령은 아직 발행하지 않았다.
- 실행 준비용 `deploy/policy_control/config/pd_rh56f1_stationary_left_exec.yaml`을 추가했다. 왼팔 무발행 stationary 설정과 `execute`만 다르며, 실제 PD 기동/engage는 승인 전까지 수행하지 않는다.
- 실행기는 한 번에 trial 하나만 선택한다. 자동 후보 연속 진행은 하지 않는다. 실제 실행 전 왼팔 PD의 TRACKING/runtime gain 검증, 캡처 홈과 실측 관절 차이 ≤0.002 rad, 주변 양팔·손 장면 일치, 충돌 상태와 명령 발행자 단독 여부가 필요하다. 로봇 이동 명령은 `sim2real/CLAUDE.md`의 명령별 승인 절차를 따른다.
- 오른팔 원시 6개 자세는 기존 캡처를 재사용하므로 재캡처하지 않았다. 왼팔 현재 홈이나 오른팔 attention 자세가 바뀌면 최신 양팔 scene으로 두 계획을 다시 생성한다.

## 인터페이스와 책임

| 파일/경로 | 책임 |
|---|---|
| `deploy/policy_control/tools/stationary_tracking.py` | capture/prepare/run/analyze CLI, 기본 무발행, 실행별 새 산출물 |
| `deploy/policy_control/policy_control/stationary_tracking.py` | ROS와 분리한 이름 정렬·시간 대응·판정·실험 계획 생성 |
| `deploy/policy_control/policy_control/stationary_compensation.py` | 순수 보상 계산, 설정 검증, payload 회귀·보류 검증 |
| `deploy/policy_control/policy_control/pd_gains.py` | 학습 계약 비교와 실험 전용 실행 게인/XML 검증 |
| `deploy/policy_control/policy_control/pd_law.py` | 엄격한 PD 설정 로드, 선택적 보상/실험 키 |
| `deploy/policy_control/policy_control/pd_arm.py` | 실제 Kp/Kd 사용, 기존 tick 안 보상, 상태 전환·동일 tick 진단 |
| `deploy/policy_control/policy_control/pd_node.py` | live robot_description·접촉 입력, 별도 진단 토픽 |
| `../robot_control/ros_ws/src/openarm_ros2/openarm_bringup/launch/openarm.bimanual.launch.py` | 실제 미션이 호출하는 bringup의 선택적 오른팔 override |
| `tests/*stationary*`, `tests/policy_control/*stationary*`, 관련 gains 테스트 | 순수 계산·경계·상태 전환·회귀 검증 |

Ruling: 실제 RH56F1 미션의 bringup은 `robot_control`의 설치 소스이며 `sim2real/robot/vendor/openarm/openarm_bringup` 사본이 아니다. 실제 호출 경로에 override를 연결한다. 잘못된 경로에만 수정하면 YAML이 적용되지 않는 위험이 있다.

Ruling: 학습 계약 게인은 비교용으로 유지하고 모터 PD 추정, blend, settle에 실행 게인을 쓴다. 실험의 계약 차이 허용을 일반 정책으로 확장하지 않는다. 드라이버 실행 XML과 기대 게인이 다르면 실험 engage를 거부한다.

Ruling: 접촉 신호를 받지 못한 상태는 비접촉이 아니다. 접촉 상태가 unknown/stale이면 적분 갱신과 합격 판정을 허용하지 않는다. 센서가 없는 초기 캡처는 기준선 분석은 가능해도 정식 합격의 증거가 불충분할 수 있다.

## 데이터 계약

진단 토픽은 `/policy_control/pd_right/diagnostics`, 메시지는 `std_msgs/String`의 JSON이다. `applied`와 그 effort 의미는 유지한다. 모든 배열은 `joint_names` 순서의 canonical rad/rad·s⁻¹/N·m이며, 미수신 실측 토크·온도는 null이다.

| 구분 | 필드 |
|---|---|
| 시계 | `stamp`, `t_mono`, `arm_source_stamp`, `arm_receive_time` |
| 목표/상태 | `joint_names`, `user_target`, `pd_target`, `motor_position`, `measured`, `velocity`, `motor_velocity`, `hand_names`, `measured_hand` |
| 토크 | `torque_gravity`, `torque_gravity_correction`, `torque_friction`, `torque_integral`, `torque_request`, `torque_feedforward`, `torque_pd`, `torque_estimated_total`, `torque_feedback` |
| 판정 보조 | `temperature`, `contact`, `stale`, `protected`, `target_changed`, `limited`, `phase`, `episode`, `gains` |

진단은 한 제어 tick의 값으로 묶는다. 과거 이벤트는 원본과 수신 timestamp를 보존하고 이름으로 정렬한 뒤 최대 20 ms 이내에서만 대응한다. 누락/대응 실패율을 summary에 남긴다. 시간 공백을 보간해 10초 연속 유지로 만들지 않는다.

`manifest.json`은 코드 버전/변경 상태, URDF·설정 해시, 실행 게인과 보상 계수, 관절 순서, 손 자세, domain, episode, 단계/후보/반복/접근 방향, 명령 발행 여부와 기동 증거를 담는다. `events.jsonl`은 원시 근거이고 `summary.json`, `report.md`, 그래프는 재생성 가능한 분석이다. 기존 run은 덮어쓰지 않는다.

## 실행 게인 구성

실험 PD 설정의 `gains` 블록은 다음 확장 키를 사용한다. 기본 설정에 이 블록의 추가 키가 없으면 기존 동작을 유지한다.

```yaml
gains:
  yaml: urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml
  accept_sim_mismatch: true
  experiment_profile: stationary_tracking
  arm_gains_override: sim2real/logs/analysis/stationary_tracking_1107/<run>/right_gains.yaml
```

이 예시는 실험 전용 사본에만 작성한다. override는 오른팔 joint1–joint7의 kp/kd를 전부 명시한다. 기존 게인 파일 및 왼팔은 수정하지 않는다. 활성 게인은 bringup 시 결정되므로 각 게인 블록은 별도 승인된 재기동이 필요하다. `require_live_description`은 실험 프로파일에서 강제되며 사용자가 끌 수 있는 우회 키가 아니다. 기동 로그와 생성 robot_description을 run에 보관한다.

순수 함수 `verify_runtime_gains(xml, side, expected_kp, expected_kd)`가 일치 검사와 XML 해시/하드웨어 이름/관절 순서/게인 증거를 반환한다. live XML 미수신, 잘못된 관절, 비유한 값, 게인 불일치는 거부한다. 학습 계약과 다름을 허용해도 실제 기동 값과 다름은 허용하지 않는다.

## 보상 계산 계약

`CompensationConfig.from_mapping(mapping, n=7)`는 누락 시 기본 OFF로 만들고 알 수 없는 키·잘못된 shape·비유한 수·`vs <= 0`을 거부한다. 플래그와 모든 계수는 실험 시작 후 고정한다.

```text
StationaryCompensator(config).step(
    error, velocity, base_torque, torque_limit, dt,
    settled=..., contact=..., stale=..., stopped=...,
    target_changed=..., episode_changed=..., gravity_correction=...
)
```

`error`는 PD 목표−실측, `base_torque`는 실제 게인에 의한 모터 PD + 기존 중력 + 요청 토크다. 반환 추가분을 FF에 더한다. 모터 위치 목표를 적분으로 계속 이동시키지 않는다. 결과의 gravity/friction/integral/estimated_total/saturated/updating을 진단한다. payload 보정 regressor는 `payload_regressor(chain,q) @ payload_parameters`이고 질량·1차 모멘트는 체인의 마지막 팔 링크 프레임 기준이다.

| 키 | 기본값 |
|---|---|
| `gravity_enabled`, `friction_enabled`, `integral_enabled` | false |
| `payload_parameters` | `[0, 0, 0, 0]` |
| `fc`, `fv`, `fo`, `ki` | 관절별 0 |
| `vs` | 관절별 0.01 rad/s |
| `integral_limit` | `[1, 1, 1, 1, 0.3, 0.3, 0.3]` N·m |
| `integral_rate` | `[0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05]` N·m/s |
| `deadband_stop_deg`, `deadband_resume_deg` | 0.03, 0.06 |
| `blend_sec` | 기존 PD blend와 일치시키며 초기 1.0 s |

식별은 rank/SVD/조건수 및 보류 데이터 예측을 함께 보고한다. 회귀에 사용한 자세를 보류 검증으로 재사용하지 않는다. 식별이 불충분하거나 검증 오차가 개선되지 않으면 적용 후보로 채택하지 않는다.

## 상태 전환

| 상태/사건 | 적분 | 출력 및 안전 우선순위 |
|---|---|---|
| 기능 OFF | 항상 0 | 기존 출력과 동일 |
| 목표 이동/limiter 작동/정착 전 | 갱신하지 않음 | 기존 PD 경로 유지 |
| TRACKING + 정착 + 신선한 비접촉 상태 | deadband 밖에서 갱신 | 토크 여유·적분 상한·변화율 적용 |
| 오차 ≤0.03° | 값 보존 | 오차 ≥0.06°에서 재개 |
| 포화 | 악화 방향 적분 금지 | 전체 추정 토크 기준 여유 검사 |
| 새 목표/episode | blend 시간 이상, 변화율 한계를 지키며 제거 | 이전 적분을 학습값으로 재사용 금지 |
| 접촉/stale/중단 | 갱신 중지 | 기존 HOLD/estop/release가 우선 |
| release | 기존 blend와 일관되게 제거 | 불연속 제거·자동 드라이버 종료 금지 |
| 조그/home/droop 적분 중복 | 시작 거부 | 동시에 적용하지 않음 |

## 구현 순서와 검증 체크리스트

### 1. 계측·오프라인 분석

- [ ] 구독 전용 기록기 재사용, capture가 제어 토픽을 만들지 않음.
- [ ] prepare는 자세 6개/3+3 분리/관절 한계/충돌/손 자세/경로 속도·가속도를 검사하고 무발행.
- [ ] run 기본 무발행, `--execute`에서만 기존 PD/episode 인터페이스 사용.
- [ ] 같은 tick 진단과 과거 timestamp 대응의 관절 이름 정렬·20 ms 경계를 테스트.
- [ ] 3초 진입+10초 연속 유지, 데이터 공백/접촉/목표 변경/보호의 제외를 테스트.
- [ ] 기존 fixed_target_ab_01을 새 run으로 재분석하고 19.8551439661452 mm 계산과 비교.

### 2. 실행 게인 일관성

- [ ] override 없음의 기존 launch 결과 동일성.
- [ ] 오른팔 override가 왼팔에 영향 없음.
- [ ] actual robot_description과 기대 kp/kd 불일치·미수신에서 실험 engage 거부.
- [ ] 실제 게인으로 blend/settle/토크 추정 계산, 계약 게인은 비교용 유지.
- [ ] 실험 차이 허용이 일반 정책 검증을 완화하지 않음.

### 3. 보상과 전환

- [ ] 비활성 출력의 기존 코드와 동일성.
- [ ] 중력 회귀 rank/조건수/보류 검증 및 잘못된 payload 입력.
- [ ] 마찰 정/역 속도·0 속도·`vs` 검증.
- [ ] 적분 포화·반대 방향 해제·상한·변화율·deadband hysteresis.
- [ ] 목표/episode 변경 시 이전 적분 제거와 제거 중 연속성.
- [ ] 접촉·stale·중단에서 갱신 중지 및 기존 보호 우선.
- [ ] 조그/home/droop 중복 적분 거부.

### 4. fake 검증

- [ ] 일정 부하, 잘못된 payload, 마찰·양자화, 출력 포화, 통신 지연을 각각 주입.
- [ ] 주입 사실과 추정 원인 분류를 구분하고 보상 전/후를 같은 판정기로 비교.
- [ ] domain 97/99만 사용; 실기 DDS와 분리.
- [ ] fake 합격을 실기 안정성·1 mm 정확도 보장으로 쓰지 않음.

### 5. 실기 (입력과 승인 미확보)

- [ ] 안전 비접촉 자세 6개와 손 고정 자세 확보.
- [ ] A–G 각 후보의 설정·원시 기록·판정·중단 이력 저장.
- [ ] 보류 검증 통과한 최종 후보와 원복 절차 기록.

## 검증 명령

다음은 `sim2real/`에서 실행한다. ROS 실기 노드를 기동하지 않는 코드 검증 명령이다. 테스트 파일이 추가되는 동안은 `--collect-only`로 이름을 확인한다. 실행 결과 숫자는 실제 출력 확인 후 별도 기록한다.

```bash
python3 deploy/policy_control/tools/stationary_tracking.py --help
python3 deploy/policy_control/tools/stationary_tracking.py capture --help
python3 deploy/policy_control/tools/stationary_tracking.py prepare --help
python3 deploy/policy_control/tools/stationary_tracking.py run --help
python3 deploy/policy_control/tools/stationary_tracking.py analyze --help
python3 -m pytest -q tests/policy_control/test_pc_pd_law.py tests/policy_control/test_pc_pd_gains.py tests/policy_control/test_pc_pd_settle.py tests/policy_control/test_pc_pd_backends.py tests/policy_control/test_pc_pd_hold_recovery.py
```

새 실험·통합 테스트, ruff/pyright, 수정 파일 py_compile도 함께 수행한다. GPU나 실기 노드가 필요한 체크를 위 명령의 성공에 포함시키지 않는다. 확인한 테스트 수·skip 이유·실패·환경 제약은 최종 검증 원장에 실제 출력 그대로 남긴다.

## 진행 원장

| 항목 | 상태/근거 |
|---|---|
| 사용자 계획 재검토 | 완료: 기본 OFF, 실험 프로파일 한정, 기존 안전 전환 보존 |
| 기존 기준선 파일 | 확인: fixed_target_ab_01/stationary_tracking.json, 19.8551439661452 mm |
| 출처 확인 | FAQ v2 권고와 Teleop/PID 코드 확인. DOI 본문은 접근 실패로 미검증 |
| 계측·실행·분석 | 구현. 기존 로그 분석·capture serializer·무발행 run 검증. 실기 run은 미실행 |
| 게인 override/검증 | 구현. 오른팔 한정·기본 동일·live XML 불일치 거부 테스트 통과 |
| 보상 계산/상태 전환 | 구현. 기본 OFF·rate/deadband/포화/episode/연속 목표 fade/ROS 진단 검증 |
| fake 시험 | CPU 주입 비교 10건 및 실제 PD ROS 진단 1건 완료. 실기 안정성 증명 아님 |
| 실기 A–G/최종 설정 | 미실행; 자세 6개·명령별 승인 필요 |

문서만으로 구현 완료를 선언하지 않는다. 최종 상태는 변경 코드, 테스트 출력, 새 run의 manifest/summary를 함께 검토한다.


## 구현 후 사용 절차와 남은 실기 입력

2026-11-07 사용자 확인: 안전 자세 6개는 **미수집**이다. 실제 A–G, 최종 채택 게인/보상, 물리 공간 정확도 검증은 미실행이다. 다음은 코드 인터페이스 예시이며 실기 이동 승인을 대신하지 않는다.

- 별도 프로파일 `deploy/policy_control/config/pd_rh56f1_stationary.yaml`: 오른팔만, execute=false, 추가 보상 세 기능 OFF, home settle gain=0, 속도 0.05 rad/s. 기존 운영 YAML은 수정하지 않았다.
- `capture --pose-id p1 --split fit --seconds 15`: 구독 전용 기록기를 재사용한다. PD 명령/engage 없이 팔·손 실측으로 snapshot.json을 만든다. snapshot의 scene에는 관측된 반대팔·손도 담는다. 이것이 충돌 안전 인증은 아니다.
- `prepare --poses poses.json --scene scene.json --pd-config <실험 yaml> --contract <RH56F1 계약> --profile <RH56F1 profile> --env-yaml <현장 env> --urdf <RH56F1 URDF> --stage A`: 자세 6개 목록, fit/holdout 각 3개, 공통 hand, 전체 주변 관절 scene을 요구한다. 기존 코사인 경로 생성기와 기하 충돌 검사기를 사용한다. 기존 접촉 자세 탈출 완화는 끈다. 실패 접근은 제외한다.
- poses.json 각 항목: id, split(fit/holdout), q(오른팔 7 rad), hand(고정 손 관절 이름→rad). scene.json은 모든 비제어 가동 관절의 canonical 이름→실측 rad이다. 원시 snapshot은 보존하고 의도한 고정 손 자세는 별도 명시한다.
- 준비 결과 plan.json의 trial id 하나를 골라 `run --plan .../plan.json --trial <id>`로 무발행 점검한다. 발행은 `--execute`만 허용한다. 노드 engage나 시작 자세 이동을 대신하지 않으며, 현재 실측이 해당 접근 시작점에서 0.002 rad를 넘으면 거부한다. 필요한 이동/재기동은 기존 경로와 명령별 승인으로 별도 수행한다.
- 실행 시 `/policy_control/pd_right/contact`의 신선한 Bool=false가 필요하다. 접촉 판정 발행자는 현장 센서/감독 체계에 맞춰 별도로 연결해야 한다. 이 도구는 모르는 접촉을 false로 생성하지 않는다. live description, 모든 관절 온도/실측 토크, 손 자세, 반대팔/손 scene도 확인한다. 다른 joint_target 발행자(조그 포함)가 있으면 거부한다.
- B/C/E/F 후보는 표의 순서대로 **별도 설정·재기동 블록**으로 준비한다. prepare의 단계 검증은 A–D 추가 보상 OFF, B 단일 Kp 변경/Kd 고정, C 두 Kd 방식, E 중력±마찰, F Ki=Kp/Ti를 검사한다. D는 --speed 0.01/0.025/0.05로 각각 준비한다. 자동 재기동/연속 후보 진행 기능은 없다.
- `analyze --events <events.jsonl> --path-end <wall seconds>`는 저장 데이터만 읽는다. 경로 종료를 모르면 판정 제외한다. 과거 내부 HOLD는 당시 status가 20 ms 안에 대응되는 경우에만 motor_position을 승인된 고정 기준으로 사용하며, 계속 발행되던 외부 user_target과 구분한다.
- 원시 robot_description도 이벤트에 저장하고 runtime gains 검증 결과를 manifest에 남긴다. bringup의 `arm_gains_startup` 로그는 운영자가 해당 run에 함께 보관한다. XML은 기동 설정 증거이며 모터 레지스터 readback은 아니다.

Ruling: 새 목표가 매 tick 바뀌어도 적분 fade를 다시 시작하지 않는다. blend_sec와 적분 변화율을 동시에 지키기 위해 제거가 blend_sec보다 길 수 있다. 이 선택은 제거 지연을 허용하는 대신 출력 기울기를 제한한다.
Ruling: 실제 로봇 자세를 자동 선택하지 않는다. 시작점으로의 이동도 별도 승인된 기존 경로가 필요하다. 실험의 자동화 범위가 각 준비된 접근/15초 유지로 제한된다.
Ruling: fresh-context 보조 리뷰는 도구 사용량 제한으로 중단되었다. 최종 검토는 작성자의 diff·경계조건 검토와 테스트로 수행했다. 독립 검토가 완료된 것으로 표현하지 않는다.

## 재현된 기준선

`logs/analysis/stationary_tracking_1107/baseline_stationary_tail_02/`는 전환 후 3초부터의 기존 기록을 새로 분석한 결과다. 관절 중앙값 FK 차이 **19.85514396614522 mm**, z 차이 **18.031556240379388 mm**로 기존 값을 재현했다. 관측 최소 관절 증가량은 움직임이 관측된 관절에서 **0.02185693885446026°**이다. 이는 encoder 사양이 아니라 이 데이터의 최소 관측 간격이다. 0.03° deadband는 약 1.37배, 0.06°는 약 2.75배다. 정지 관절에는 해상도를 추정할 변화가 없어 null을 기록했다.

기존 외부 목표는 HOLD 이후에도 발행되고 있었다. 무시된 외부 목표를 승인된 고정 목표로 사용하면 잘못된 오차가 생기므로 status 대응이 안 되는 샘플은 제외한다. 접촉·필수 상태 결손, 계획 종료 시각 결손, 충분한 연속 유지 관측 부족 때문에 이 기준선은 정식 합격 판정 대상이 아니다.

## fake 검증 범위

`PYTHONPATH=deploy/policy_control python3 -m policy_control.stationary_fake --out <새 경로>`는 일정 부하, payload 모델 편향, 마찰+양자화, 포화, 통신 지연의 OFF/ON 10개 결과를 만든다. 원인 이름은 **주입한 정답**이며 계측만으로 추론한 원인 분류가 아니다. CPU toy plant이며 로봇의 관성·기구학·충돌·센서 지연을 정확히 재현하지 않는다.

`fake_01/comparison.json`: 일정 부하 중앙값 FK 차이 OFF 2.4743582965269675 mm → ON 0.5452531530357062 mm, 마찰/양자화 2.6327172275046937 → 0.6581793068761734 mm. 이 설정은 3초 진입 등 전체 판정을 통과하지 않았다. 포화는 실패, 지연은 평가 제외다. 한계를 자동 확대하지 않았다. 별도로 실제 ArmUnit tick/FSM과 ROS PD 노드의 진단 발행을 fake 입력·execute=false·격리 도메인 99에서 검사했다. 실기 A–G 전체의 성공이나 실기 안정성을 입증한 것은 아니다.


## 최종 코드 검증 기록

- 신규/통합 순수 테스트: `53 passed in 1.33s` (`logs/analysis/stationary_tracking_1107/final_new_tests.txt`).
- 기존 PD law/gains/settle/backend/hold recovery 및 node를 포함한 마지막 ROS 회귀 묶음: `1 failed, 132 passed, 29 skipped in 2.99s`. 실패는 `test_every_control_gains_copy_in_the_workspace_agrees`가 전체 workspace 검색 중 일시적으로 `.aws` 경로를 찾지 못한 건이었다. 같은 테스트를 별도로 재실행하여 `1 passed in 0.57s`를 확인했다. 이전 동일 묶음은 `127 passed, 29 skipped in 3.10s`였다(추가 테스트 전).
- 새 실제 PD ROS 진단 시험: `1 passed in 0.68s`, 도메인 99, execute=false. 초기 시험 executor가 전역 context를 사용하는 오류는 격리 context executor로 수정했다.
- 전체 CPU 회귀: `32 failed, 3525 passed, 308 skipped, 112 deselected, 4 warnings, 10 errors in 422.66s (0:07:02)`. 모든 테스트가 통과했다고 주장하지 않는다. 실패 파일/테스트명은 `logs/analysis/stationary_tracking_1107/regression_failures.md`, 원본 출력은 `regression_suite.txt`. 삭제된 기존 계약, 가중치/환경, 기존 기대값 불일치, sandbox socket 제한이 포함된다. 범위 밖 파일은 고치지 않았다.
- socket 제한으로 실패한 console/video/가짜 EtherCAT 통신 재검증: `30 passed in 3.90s` (`socket_recheck.txt`).
- ruff: 수정한 4개 기존 PD 파일과 새 계산/CLI/테스트 모두 `All checks passed!`. Python compile 및 두 저장소의 수정 경로 `git diff --check` 통과.
- pyright: 런타임 interpreter의 site-packages 자동 발견이 실패하여 읽기 가능한 기존 패키지 경로를 명시한 임시 설정으로 재검사, `0 errors, 0 warnings, 0 informations` (`pyright_explicit_paths.txt`). 저장소/사용자 pyright 설정은 변경하지 않았다.
- 구독 전용 기록기: `selftest ok (no ROS initialized)`.
- 실제 CLI 무발행 시험: `dryrun_01/manifest.json`, `commands_sent=0`; 준비/충돌 인증용 실기 자세가 아닌 스키마 시험 fixture 사용.
- 실제 로봇 명령 0, commit/push 0. 사용자가 미수집이라고 확인한 자세 6개 때문에 실기 A–G와 최종 채택 설정은 남아 있다.

최종 endpoint 판정은 준비된 목표와 계획 종료 시각을 기준으로 한다. 최초 20 ms의 전달 중 이전 목표 표시는 수신 정렬 허용폭으로 처리하되 FK 오차는 준비된 최종 목표로 계산한다. 이후 다른 목표가 수신되면 제외한다. 추가 회귀 테스트로 두 경우를 구분했다.
