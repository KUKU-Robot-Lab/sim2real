# grasp-v1 라이브 sim2real 브링업 — 설계

**날짜:** 2026-07-21
**범위:** tesollo/right grasp-v1 라이브 sim2real의 **①단계**(grasp 잡고 들기)만.
②pour-v1 라이브 · ③순차 오케스트레이션은 별도 spec.

## 1. 목표

FoundationPose로 컵 위치를 인식하고, hdgp `grasp-v1` 정책으로 실기(OpenArm 팔 +
Tesollo dg5f 손)를 제어해 **컵을 안정적으로 잡고 든다**. 그 물리적 "쥔 상태"가
이후 pour(②)로 넘어가는 warmstate(물리 핸드오프)가 된다.

## 2. 확정된 설계 결정

- **warmstate = 물리 핸드오프.** grasp가 컵을 쥔 물리 상태에서 pour가 이어받는다.
  별도 상태 파일 없음.
- **핸드오프 트리거는 수동+자동 둘 다 지원**(①에서는 held/성공 판정 신호까지만 준비).
- **grasp 정책 = hdgp `tesollo/right/grasp_v1`.** 폴더명은 v1이나 내부 설계는 "v7"
  (Fabrics 팔 6D palm + per-finger lerp 5D). **obs 106D(actor) / action 11D**.
  → 기존 `sim2real_inference.py`(obs_dim=106, action_dim=11)와 **계약이 이미 일치**한다.
  즉 이번 작업은 obs/action 재설계가 아니라 **실기 커미셔닝(캘리브·튜닝·검증)**이다.
- **브링업 전략 = 검증 우선(A).** 로봇을 움직이기 전에 obs/캘리브를 다 맞추고,
  각 단계에 검증 게이트를 둔다.

## 3. 아키텍처 (대부분 기존 재사용, 새 아키텍처 없음)

```
FoundationPose ─▶ cup_pose_relay.py ─▶ /cup_pose (geometry_msgs/PoseStamped, robot base)
OpenArm 팔  ─▶ /joint_states (7D)                 ┐
Tesollo dg5f ─▶ /dg5f_right/joint_states (20D)     ├─▶ sim2real_inference.py
             └▶ /dg5f_right/contact_forces (5D N) ┘   106D obs → grasp-v1 정책 → 11D action
                                                       → Fabrics IK(palm) + finger lerp
                                                       → send_right_full(arm7, hand20)
```

**106D obs 구성**(sim2real_inference 기존): arm_pos7 + arm_vel7 + hand_pos20 + hand_vel20
+ palm_center3(FK) + (tips-palm)15(FK) + (cup-palm)3 + (tips-cup)15 + contact5 + last_action11.

**필요 드라이버**(벤더): OpenArm 팔(있음, 확인 필요) · Tesollo dg5f_right(있음, contact_forces
발행) · FoundationPose(perception, 비전 머신). 핵심 드라이버는 sim2real에 존재.

## 4. 검증 우선 브링업 단계 (각 단계 = 게이트)

### Stage 0 — Dry-run obs 검증 (로봇 무동작)
- 드라이버 + FoundationPose + cup_pose_relay 기동, 4개 입력 토픽 흐름 확인.
- **joint 순서/부호 매핑 검증**(sim 관절순서 ↔ dg5f/openarm 드라이버 순서) — 알려진 자세로
  대조. silent 오류 위험이 가장 큼.
- FK(palm_center, fingertip_pos)를 실기 joint로 계산해 로봇 기하와 sanity 대조.
- 106D obs를 로깅해 범위/단위가 sim obs 정규화와 일치하는지 확인.
- **게이트:** 모든 입력 흐름, 매핑 정확, obs 벡터 타당.

### Stage 1 — extrinsics 캘리브 (cup_pose 로봇 프레임 정합)
- ArUco 캘리브(perception `tools/calibrate_extrinsics.py`)로 T_base_cam 산출 →
  `config/global_camera_extrinsics.yaml`(현재 PLACEHOLDER) 갱신.
- 컵을 로봇 기준 아는 위치에 두고 `/cup_pose` 오차 측정.
- **게이트:** cup_pose 오차 수 cm 이내.

### Stage 2 — actuator 캘리브 + 안전 (소동작만)
- r2s_autotune 적용 또는 최소한 명령 추종 검증. rate limit·workspace bound·e-stop 준비.
- `send_right_full`로 소폭 안전 이동 → 추종 확인.
- **게이트:** 명령 안전 추종, runaway 없음.

### Stage 3 — 감독 하 라이브 grasp (반복 튜닝)
- APPROACHING(pregrasp arm + approach hand, settle) → RUNNING(정책 grasp phase → lift phase).
- 사람 감독 + e-stop 상시. 튜닝 대상: pregrasp 위치, settle_time, contact 임계
  (`CONTACT_FORCE_THRESHOLD`), delta 스케일, lift 시작 스텝.
- **게이트:** 컵을 안정적으로 잡고 든다.

### Stage 4 — held/성공 판정 (코드 추가)
- 성공 정의: 컵 lift(팔/컵 높이 임계) + 안정 그립(≥N 손끝 접촉) T초 유지.
- 핸드오프용 신호/서비스 발행(③ 오케스트레이션과 ② pour가 소비).
- **게이트:** held 상태를 신뢰성 있게 감지.

## 5. 코드 변경 (sim2real, 편집 가능; hdgp는 READ-ONLY)

- Stage 0: dry-run obs 검증 스크립트(토픽/매핑/FK/obs 로깅). 신규.
- Stage 4: held/성공 판정 로직 + 핸드오프 신호. sim2real_inference 확장 또는 별도 노드.
- 튜닝 파라미터·안전 설정(config). 코어 `sim2real_inference.py`는 재사용.

## 6. 리스크

- **joint 매핑 silent 오류**(sim↔실기 순서/부호) — Stage 0에서 반드시 검증.
- **extrinsics 정확도** — cup_pose 틀리면 grasp 위치 어긋남.
- **contact 임계 실기 스케일** — FT force[N] 스케일이 sim과 달라 접촉 판정 왜곡 가능.
- **sim2real 정책 전이 실패** — 정책이 실기 dynamics gap으로 전이 안 되면 r2s_autotune
  또는 재튜닝 필요(최악: 정책 재학습, 이 spec 범위 밖).

## 7. 범위 밖 (Out of scope)

- ② pour-v1 라이브(별도 spec) · ③ 순차 오케스트레이션(수동+자동, 별도 spec).
- 정책 재학습(hdgp READ-ONLY).
- 24.04/Jazzy 이식(현재 22.04/Humble 배포 기준; Jazzy는 별도 브랜치 작업).

## 8. 성공 기준 (①단계 완료 정의)

FoundationPose로 인식한 컵을 grasp-v1 정책이 실기에서 **반복적으로 잡고 들어**,
Stage 4의 held/성공 판정이 안정적으로 뜬다(= pour 핸드오프 준비 완료).
