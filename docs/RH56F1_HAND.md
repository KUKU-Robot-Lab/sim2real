# RH56F1 손 — 행동 구조 · 센서 · 배포 연결 (2026-09-30)

정책이 내는 손 행동 6 이 실제 손가락 움직임이 되기까지, 그리고 손끝 촉각이 정책 관측으로 돌아오기까지의 한 장 요약.
학습 기준은 hdgp `tasks/rh_aglt_r/hand_action.py`(= pour_fabric_mimic `direct_hand_targets`, 09.29 사용자 "손 액션은 pour_bi_rh
와 동일하게"), 배포 구현은 `deploy/policy_control/policy_control/rh56f1_hand.py` 한 곳이다. 둘의 수치 일치는
`tests/policy_control/test_pc_rh56f1_hand.py`(무작위 300 스텝, 1e-5) 가 잠근다.

## 1. 관절

| 행동 슬롯 | 관절 | 뜻 | 편 손(open) | 쥔 손(grip) | 한계 | 벤더 레지스터 슬롯(0.1°) |
|---|---|---|---|---|---|---|
| 0 | thumb_1 | 엄지 외전(회전) | 1.57 | 1.20 | 0 ~ 2.094 | 5 (≈ 600~1750, verified 전) |
| 1 | thumb_2 | 엄지 굽힘 | 0 | 0.24 | 0 ~ 0.475 | 4 (1100~1350, verified 전) |
| 2 | index_1 | 검지 | 0 | 1.08 | 0 ~ 1.529 | 3 (1740 편 ~ 900 굽힘) |
| 3 | middle_1 | 중지 | 0 | 1.08 | 〃 | 2 |
| 4 | ring_1 | 약지 | 0 | 0.85 | 〃 | 1 |
| 5 | pinky_1 | 새끼 | 0 | 0.85 | 〃 | 0 |

- 종속 6(thumb_3 = 1.1425·thumb_2, thumb_4 = 0.7508·thumb_3, *_2 = 1.1169·*_1)은 행동에 없다. sim 은 PhysX mimic, 실기는 손 기구.
- 좌우 같은 값(좌 URDF 가 엄지 축을 이미 뒤집었다 — 손 거울 부호 +1).
- sim 액추에이터: 강성 30 · 감쇠 0.3 · 최대 토크 1 N·m · 최대 속도 2 rad/s(USD). 60 Hz 정책(dt 1/120 · decimation 2).

## 2. 행동 → 목표 (정책마다 같은 식)

```
[lo, hi] = [min(open, grip), max(open, grip)]            관절별 (pour_fj f00b · f01 만 관절 한계 전 범위)
raw      = lo + ½(clip(a, −1, 1) + 1)(hi − lo)          a = 0 은 반쯤 쥔 손 · thumb_1 만 a = +1 이 편 쪽
ema      = 0.1·raw + 0.9·q*                             EMA
Δ        = clip(ema − q*, ± 한계 전 범위 / (1.0 s · 60))  "전 범위 1 초" 속도 상한
           pour_fj rh5~(env hand_vel_cap_rad_s): ± 2.1/60 rad, thumb_2 만 ± 0.56/60 rad (실기 속도)
lo       = max(lo, 0.065) 네 손가락만                    pour_fj rh5~(env hand_finger_open_floor_rad) — 실기 1740 까지만 펴짐
동결     : 그 손가락 촉각 > 1 N 이고 닫는 방향이면 Δ = 0 (펴기는 허용)
q*       = clip(q* + Δ, lo, hi)
대기     : rh_aglt 처음 10 스텝은 편 손 · pour_fj 처음 30 스텝은 손이 행동을 따른다(팔만 고정)
```

## 3. 목표 → 손가락 (실기)

```
정책 노드 q*(rad, 60 Hz) ─ /policy_control/joint_target ─▶ pd_node rh56f1_angle 백엔드
   속도 상한 max_vel_track 2.1 rad/s(= sim 상한) · 한계 여유 · 측정값과의 거리 제한
   → rh56f1_hand_map.yaml 로 rad → 레지스터(끝점 선형, 엄지 두 축은 verified 전이라 −1 = 안 움직임)
   → 바뀐 것만 · 최대 30 Hz · 1 s 마다 재전송 ─ /hand_<side>/angle_set ─▶ rh56f1_ecat_node ─ 유닉스 소켓 ─▶
   rh56f1_ecat_master(SOEM, 1 kHz PDO) ─ EtherCAT ─▶ 손   (10.02 RS485 에서 바꿈 — 아래 'EtherCAT' 절)
손 자체 설정(pd yaml hand.hw_*, 발행 모드에서 첫 명령 전 · 5 s 마다): speed_set 2000(= 무부하 전 행정 1 s) · force_set 600 g(벤더 기본)
```

## 4. 센서 → 관측

| 신호 | 경로 | 정책에서 |
|---|---|---|
| 손 관절각 | 드라이버 angle_actual(0.1°, 50 Hz) → `rh56f1_state_node` → `/hand_<side>/joint_states`(rad, 종속 포함) | hand_q · hand_err((q* − q)/1.2) |
| 손끝 촉각 | 드라이버 touch_data.finger_forces(새끼부터, 원시) → 상태 노드 ×0.01 N · 엄지부터 → `/hand_<side>/tip_forces` | tanh(clip(F, 0, 10)/3) 관측 5 · 동결 1 N |
| (손 모터 힘 forceAct · 전류) | 드라이버 force_actual | 학습에 없다 — 안전 감시용 |

- 학습 촉각 = 각 손가락 `*_sensor` 링크의 net 접촉력(대상 무관) — 실기 촉각과 같은 뜻. 동결은 학습이 컵만 거른 첫마디 · 손끝 힘을
  쓰고 배포는 손끝 촉각으로 대신한다.
- 촉각 단위 0.01 N 은 벤더 매뉴얼 해상도에서 온 추정이다(`rh56f1_hand_map.yaml touch.unit_verified: false`).

## 5. 정책별

| | pour_fj (`rh56f1/pour_fj/both_f01`) | rh_aglt (`rh56f1/aglt/right_i03` · `rh56f1/aglt/left_i05`) |
|---|---|---|
| 행동 | 26 = [오른팔 7 · 오른손 6][왼팔 7 · 왼손 6] | 13 = 팔 7 · 손 6 |
| 손 범위 · 동결 | f01: 한계 전 범위 · 동결 없음 (f02~: grip · 1 N) | grip · 1 N |
| 손 관측 순서 | PhysX 순 — 10.01 t2r_rh5_f01 ep800 trace 로 실측: 가정(index · middle · pinky · ring · thumb_1 · thumb_2)과 같다 | 프로필 순(이름) — 문제 없음 |
| Isaac 대조 | rh5_f01 ep800 · f02 ep3000: 팔 · 손 q* 한 스텝 재생 오차 < 1e-7(동결 = 직전 스텝 첫마디 OR 손끝 컵 접촉), LSTM 행동 ≤ 1.2e-3(f01, 리셋부터) | 계약 테스트(test_pc_rh_aglt) |
| 대기 | 30 스텝, 손은 따른다 | 10 스텝, 편 손 |
| 노드 | `pour_fj_node.py` | `rh_aglt_node.py` (같은 모듈의 rh_aglt 계열) |
| 컵 | /objects/cup_src · cup_rcv | 오른팔 cup_src · 왼팔 cup_rcv, 첫 목표 = 리셋 때 컵 + 14 cm, 그 뒤 목표 직접 입력(아래) |
| 기본 정책(10.06) | rh56f1/pour_fj/both_f01 | 단독 aglt 점검(미션 policy_aglt_<side>): 오른팔 rh56f1/aglt/right_env17 · 왼팔 rh56f1/aglt/left_env17f(오른팔 거울을 왼팔 env 에서 이어 학습, 거울 left_env17mir 는 hold)(cyl60 · 보상 iter_17 · 붓기 하중으로 이어 학습, T2R Grasping, 실기 점검 전). 쥔 높이(컵 중심 위 손바닥, 컵 축)가 +2.0 cm 로 cyl60g(+4.3 cm)보다 낮다. 놓기 i09 · i01 시작 뱅크의 쥔 높이는 p5~p95 4.3~5.3 cm(우) · 3.1~5.5 cm(좌)이고 3 cm 아래는 0 %(우) · 4 %(좌)라 env17 인계는 놓기 학습 분포 밖(우) · 끝자락(좌)이다. 그래서 에피소드(config/episodes)는 10.04 기본 rh56f1/aglt/right_cyl60g · rh56f1/aglt/left_cyl60gmir(cyl60 · FP++ 지각 · 파지 후 부착) 그대로. 실기 컵 cyl60. 이전 기본 i10d(aglt_cup_s065) · mirror_l5 · i05, i03 은 hold |

**양팔 rh_aglt 를 한 세션에서 동시에(09.30):** 정책 노드를 팔마다 `-r __node:=rh_aglt_node_<side> -p ns:=<side>` 로 띄운다 —
에피소드 서비스 · 토픽 · 관측 · 행동이 `/policy_control/<side>/…` 로 갈리고(`joint_target` 은 공용), pd 는
`/policy_control/<side>/episode` 를 그 팔에만 적용한다. 부르기: `trigger.py episode/start --episode-ns <side>`.
콘솔 정지 바의 에피소드 정지 · 중단은 `--episode-ns '*'` 로 떠 있는 정책 노드 전부를 멈춘다. 미션 `policy_aglt_<side>` 가 이렇게 띄운다.

## 6. 실측으로 정할 것 (튜닝 — 손이 움직인다, 단계마다 승인)

1. 엄지 두 축 방향 · 대응(미션 probe_<side>) → `verified: true`.
2. 손 속도: 편 손 ↔ 반쯤 쥔 손 스텝의 지연 · 도달 시간 vs sim(전 범위 1 s, EMA 0.1) → `hw_speed`.
3. 레지스터 ↔ 각도 비선형(벤더 행정-각도 표) → 변환표 보정.
4. 힘 멈춤 임계(`hw_force`, 지금 벤더 기본 600 g) — sim 은 손끝 1 N 에서 닫기를 멈춘다. 파지 실험으로 조정.
5. 촉각 단위(0.01 N 추정)를 알려진 무게로 확인 → `unit_verified: true`.

**동결 신호 차이(10.01, f02 trace):** 학습의 손 동결은 손가락 **첫마디 OR 손끝**의 컵 접촉력(> 1 N)이다. f02 왼손(쥐는 손)은
첫마디 접촉으로 동결되는 스텝이 많아, 손끝만으로 흉내 내면 1만 스텝 넘게 달라진다. 실기 촉각은 손끝뿐이라 배포는 손끝 촉각으로
대신한다 — 첫마디 쪽은 손 펌웨어 힘 멈춤(forceSet 600 g)에 기댄다. 모터 힘(force_actual)으로 첫마디 접촉을 대신할지는 실측 뒤 정한다.

**rh_aglt 목표 직접 입력(10.01 사용자):** 에피소드 reset 뒤 `/policy_control/<ns>/goal`(geometry_msgs/Point, 로봇 base m)에 최종 목표를 낸다.
결과는 `/policy_control/<ns>/goal_result`(latched JSON: ok · reasons · goal · queue · kp_dist · near_steps · successes). 규칙(`rh_aglt_goals.py`):
- 학습 목표 박스 밖이면 거부 — 오른팔 x 0.10~0.40 · y −0.30~−0.10, 왼팔 y +0.10~+0.30, z 0.345~0.485(= 정착고 0.265 + 0.08~0.22).
- 첫 목표 전: 리셋 때 컵에서 수평 ±0.05 · 위로 0.10~0.18 안이면 첫 목표를 그것으로, 밖이면 첫 목표를 달성한 뒤 거기서부터 잇는다.
- 먼 목표는 직전 달성 목표에서 축마다 ±0.08 m 이내 중간 목표로 나눠 차례로 준다.
- 달성 = 키포인트 최대거리 ≤ 0.02 m(tol_floor) 누적 10 스텝 + 지금 근처 + 쥠(엄지 AND 다른 손가락 촉각 > 1 N). 마지막 목표에는 머문다.
- 예: `ros2 topic pub --once /policy_control/right/goal geometry_msgs/msg/Point "{x: 0.30, y: -0.15, z: 0.44}"` — 팔이 움직이는 입력이라 실기에서는 승인 뒤.


## 7. EtherCAT (10.02 — RS485 대신, 정책 제어 포함)

사용자 결정: RS485(115200 baud · 상태 50 Hz · 명령 최대 30 Hz)는 더 쓰지 않는다. 손 EtherCAT 포트는 하나뿐이라 **손 하나 = NIC 하나**
(일반 스위치로 묶으면 0 slave + 브로드캐스트 폭주, 10.01). arm4090: 오른손 USB-C 랜 `enx00e04c6806e1` · 왼손 내장 랜 `enp6s0`.

```
pd · 정책 ─ /hand_<side>/angle_set · force_set · speed_set ─▶ rh56f1_ecat_node.py (ROS, 벤더와 같은 토픽 · 메시지)
                                                                 │ 유닉스 데이터그램(명령 RHC1 · 상태 RHS1, policy_control/rh56f1_ecat.py)
                                                                 ▼
                         tools/ethercat/rh56f1_ecat_master (C · SOEM v1.4.0 · cap_net_raw · 1 kHz PDO) ─ EtherCAT ─▶ 손
상태: 마스터 ─(state_hz 100)▶ 노드 ─▶ /hand_<side>/angle_actual · force_actual · current_actual · touch_data · ecat_status(JSON 1 Hz)
```

- 두 프로세스로 나눈 이유: setcap 실행 파일은 `LD_LIBRARY_PATH` 를 무시해 ROS 라이브러리를 못 읽는다. 노드가 마스터를 자식으로 띄우고,
  마스터는 `PR_SET_PDEATHSIG` 로 노드와 같이 끝난다.
- PDO(매뉴얼 §2.6 표 50): 입력 76 × INT16(위치 · 각도 · 힘 · 전류 · 오류 · 상태 · 온도 각 6 + 촉각), 출력 19 × INT16(ENABLE_SET · 각도 6 ·
  힘 6 · 속도 6). 각도 단위 · 슬롯 순서는 RS485 레지스터와 같아 변환표(`rh56f1_hand_map.yaml`)를 그대로 쓴다.
- **AL 0x1E 우회**: 펌웨어의 PDO 매핑 항목(0x1601 · 0x1A00)이 표준 UINT32 가 아니라 UINT16(`0x0110`)이다. SOEM complete access 가
  크기를 잘못 계산(144/608 bit)해 SM 길이가 어긋난다 → 마스터가 `ECT_COEDET_SDOCA` 를 끄면 38 / 152 B 로 맞는다.
- 안전: 첫 각도 명령 전 · 노드 하트비트 0.5 s 끊김 · 정지 때 마스터는 매 주기 목표 = 지금 각도 · ENABLE_SET 0(제자리).
  -1 = 그 축 직전 목표 유지. 범위 밖 값은 매뉴얼 범위(네 손가락 900~1740 · 엄지 굽힘 1100~1350 · 엄지 회전 600~1800 · 힘 ≤ 1000 g ·
  속도 ≤ 4000)로 자른다. `--no-op` 는 SAFE_OP 에 머문다(상태만, 손은 출력을 쓰지 않는다).
- 실측(10.02, SAFE_OP 1 kHz 3000 회): 잃음 0 · PDO 왕복 오른손 p50 68 us(USB 랜) · 왼손 41 us · 주기 흔들림 ±25 us.
- ★10.03: 1 kHz 에서는 OP 가 안 된다(손 MCU 가 OP 요청을 못 읽음) → 주기 500 Hz. 두 손 OP · 검지 하나 굽힘/폄 · 되돌림 성공(docs/RH56F1_ECAT_RUNBOOK.md).
- 확인: ENABLE_SET 1 = 동작 허용(10.03). 상태 코드 255 = 전원 뒤 첫 명령 전(두 손 모두 관찰).
- 빌드 · 권한: `bash tools/ethercat/build.sh` → `sudo setcap cap_net_raw,cap_net_admin=ep tools/ethercat/rh56f1_ecat_master`
  (내용이 바뀌어 다시 빌드되면 setcap 도 다시). 점검 도구: `tools/ethercat/ecat_rh56f1 {rtt|safeop} <ifname>`.
