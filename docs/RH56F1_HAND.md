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
동결     : 그 손가락 촉각 > 1 N 이고 닫는 방향이면 Δ = 0 (펴기는 허용)
q*       = clip(q* + Δ, lo, hi)
대기     : rh_aglt 처음 10 스텝은 편 손 · pour_fj 처음 30 스텝은 손이 행동을 따른다(팔만 고정)
```

## 3. 목표 → 손가락 (실기)

```
정책 노드 q*(rad, 60 Hz) ─ /policy_control/joint_target ─▶ pd_node rh56f1_angle 백엔드
   속도 상한 max_vel_track 2.1 rad/s(= sim 상한) · 한계 여유 · 측정값과의 거리 제한
   → rh56f1_hand_map.yaml 로 rad → 레지스터(끝점 선형, 엄지 두 축은 verified 전이라 −1 = 안 움직임)
   → 바뀐 것만 · 최대 30 Hz · 1 s 마다 재전송 ─ /hand_<side>/angle_set ─▶ robot_control rh56f1_driver ─ RS485 ─▶ 손
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

| | pour_fj (`both_rh_pourfj_f01`) | rh_aglt (`right_rh_aglt_i03` · `left_rh_aglt_i05`) |
|---|---|---|
| 행동 | 26 = [오른팔 7 · 오른손 6][왼팔 7 · 왼손 6] | 13 = 팔 7 · 손 6 |
| 손 범위 · 동결 | f01: 한계 전 범위 · 동결 없음 (f02~: grip · 1 N) | grip · 1 N |
| 손 관측 순서 | PhysX 순 — **추정**(assumed), Isaac trace 로 실측 필요 | 프로필 순(이름) — 문제 없음 |
| 대기 | 30 스텝, 손은 따른다 | 10 스텝, 편 손 |
| 노드 | `pour_fj_node.py` | `rh_aglt_node.py` (같은 모듈의 rh_aglt 계열) |
| 컵 | /objects/cup_src · cup_rcv | 오른팔 cup_src · 왼팔 cup_rcv, 목표 = 리셋 때 컵 + 14 cm |

## 6. 실측으로 정할 것 (튜닝 — 손이 움직인다, 단계마다 승인)

1. 엄지 두 축 방향 · 대응(미션 probe_<side>) → `verified: true`.
2. 손 속도: 편 손 ↔ 반쯤 쥔 손 스텝의 지연 · 도달 시간 vs sim(전 범위 1 s, EMA 0.1) → `hw_speed`.
3. 레지스터 ↔ 각도 비선형(벤더 행정-각도 표) → 변환표 보정.
4. 힘 멈춤 임계(`hw_force`, 지금 벤더 기본 600 g) — sim 은 손끝 1 N 에서 닫기를 멈춘다. 파지 실험으로 조정.
5. 촉각 단위(0.01 N 추정)를 알려진 무게로 확인 → `unit_verified: true`.
