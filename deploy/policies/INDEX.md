# policies — 쓸 정책 목록 · 차이 · 개별 실행

`python3 deploy/policy_control/tools/policies.py --write-index` 가 만든다. 손으로 고치지 않는다 — 설명은 각 `policy.yaml` 의 `summary` · `eval` · `note`, 학습 조건은 `params/env.yaml` 에서 온다.

- **미션 기본** = 콘솔에서 정책을 고르지 않으면 그 미션이 쓰는 정책(`config/mission_*.yaml`).
- **sim 평가** 는 카드에 적힌 결정론 평가 요약이다. 공차 · 조건이 정책마다 달라 숫자끼리 바로 비교하지 않는다.
- 체크포인트 가중치는 git 에 없다(`nn/` .gitignore) — 다른 호스트에서는 `check_host.py` 가 받는 명령을 알려 준다.

## 폴더 (손 / 과제 / 팔_태그)

```
dg5f_m/
  cup_grasp/
    left_i01
    left_i14
  cup_pick/
    left_a00
    left_a01
    right_m15
  grasp_fj_rand/
    left_i00b
    right_i01  (hold)
  pour_fab/
    both_i18
    both_i24  (hold)
rh56f1/
  aglt/
    left_cyl60gmir
    left_env17mir
    left_i05
    left_i09d
    left_i10
    left_i10d
    right_cyl60g
    right_env17
    right_i03  (hold)
    right_i09d
    right_i10
    right_i10d
    right_mirror_l5
  place/
    left_i01
    right_i09
  pour_fj/
    both_f01
```

## RH56F1 한 팔 파지 · 이송 (rh_aglt)

먼 출발 → 컵 쥐기 → 들기 → 목표(리셋 때 컵 + 0.14 m)로 이송. 관측 96 · 행동 13(팔 관절 증분 7 + 손 6) · 60 Hz · LSTM. 계약은 체크포인트 · 컵 치수 말고 모두 같다 — 차이는 아래 학습 조건과 가중치다.

**실기 개별 실행** — 콘솔 → 로봇 `openarm_rh56f1` → 오른/왼 자리에서 고른다(미션 단계 `policy_aglt_<팔>`, 노드 `rh_aglt_node`). ★실물 컵과 FP++ 물체가 정책의 `학습 물체`와 같아야 한다 — 실기 미션은 `REAL_CUP = cyl60`(`scripts/ops/make_rh56f1_missions.py`)이라 shaker 정책을 돌리려면 컵 · 미션을 바꿔야 한다.

| id | 쪽 | status | 체크포인트 | 설명 | sim 평가 | 미션 기본 |
|---|---|---|---|---|---|---|
| `rh56f1/aglt/left_cyl60gmir` | left | candidate | ep2600 | 우 cyl60g ep2600 의 거울(추가 학습 없음) | 같은 조건 성공 1.62/ep · 쥔 기울기 8.4° | rh56f1_control, rh56f1_fake |
| `rh56f1/aglt/left_env17mir` | left | candidate | ep1000 | 우 env17 ep1000 의 거울(추가 학습 없음) | 같은 조건 성공 4.40/5(5개 다 채움 0.83) · 쥔 기울기 8.1° | rh56f1_control, rh56f1_fake |
| `rh56f1/aglt/left_i05` | left | candidate | ep3800 | 왼팔 첫 파지 정책(보상 iter_05) — mirror_l5 의 원본 | 학습 로그 파지 0.80 · 들기 0.75 · 성공 0.9~1.1/ep(tol 0.021) · 엄지 대향 0.04 | – |
| `rh56f1/aglt/left_i09d` | left | candidate | ep3000 | shaker × 0.65 · 보상 iter_09 · 실측 지연 적응 · 다섯 손가락 파지 | 지연 켬 · 컵 든 0.98 · 성공 1.19/ep(tol 0.035) · 0.05(tol 0.02) · 컵 기울기 18° · 어깨 j2 한계 0.31 | – |
| `rh56f1/aglt/left_i10` | left | candidate | ep4600 | 우 i10 ep4600 의 거울(추가 학습 없음) | 지연 0 · 컵 든 1.00 · 성공 2.02/ep(tol 0.0229) · 0.83(tol 0.02) · 실측 지연 넣으면 0.78 | – |
| `rh56f1/aglt/left_i10d` | left | candidate | ep2200 | 우 i10 거울 → 실측 지연으로 이어 2200 epoch | 지연 켬 · 컵 든 0.89 · 성공 2.55/ep(tol 0.052) · 0.06(tol 0.02) · 컵 기울기 13° | – |
| `rh56f1/aglt/right_cyl60g` | right | candidate | ep2600 | cyl60 원통 · FP++ 지각과 파지 후 FK 부착으로 이어 학습한 최신 파지 · 이송 | 결정론 64 env · tol 0.02 · FP++ 조건 성공 1.77/ep(원 cyl60n 1.16) · 쥔 기울기 4.4° · 이송 중 j2 한계 37 % | rh56f1_control, rh56f1_fake |
| `rh56f1/aglt/right_env17` | right | candidate | ep1000 | cyl60 · 보상 iter_17(목표 근처 머묾 수입 제거) · 붓기 하중으로 이어 학습한 최신 파지 · 이송 — 쥔 높이 컵 중심 위 약 +2 cm | 결정론 64 env · tol 0.02 · ADR 0 · 하중 끔: 성공 4.91~4.93/5(5개 다 채움 0.97) · 목표 2 cm 안 16.6 % · 쥔 기울기 4.9° · 관절 한계 ≤ 0.03 | rh56f1_control, rh56f1_fake |
| `rh56f1/aglt/right_i03` | right | hold | ep4600 | 엄지를 컵 입구 안에 넣는 파지 — hold(오른팔 첫 후보였던 것) | 학습 로그 파지 0.73~0.79 · 성공 1.1~1.3/ep · 엄지 끝 컵 안 0.97 | – |
| `rh56f1/aglt/right_i09d` | right | candidate | ep5800 | shaker × 0.65 · 보상 iter_09 · 실측 지연 적응 · 세 손가락 파지 | 지연 켬 · 컵 든 0.91 · 성공 0.84/ep(tol 0.039) · 0.05(tol 0.02) · 손목 j6 한계 0.45 | – |
| `rh56f1/aglt/right_i10` | right | candidate | ep4600 | shaker × 0.65 · 보상 iter_10 · 지연 없이 학습(①) | 지연 0 · 컵 든 1.00 · 성공 2.0~2.3/ep(tol 0.0229) · 0.8~0.9(tol 0.02) · 실측 지연 넣으면 컵 든 0.8 | – |
| `rh56f1/aglt/right_i10d` | right | candidate | ep1400 | shaker × 0.65 · 보상 iter_10 · i10 을 실측 지연(팔 9–12 · 손 3–5 스텝)으로 이어 학습(파일 이름 ep_1400 = 이어 학습 합계 2000) | 지연 켬 · 컵 든 에피소드 0.94 · 성공 2.40/ep(tol 0.059) · 0.16(tol 0.02) · 손목 j6 한계 0.40~0.48 | – |
| `rh56f1/aglt/right_mirror_l5` | right | candidate | ep3800 | 좌 i05 ep3800 의 거울(가중치) — params 는 우 i06 런(보상 칸은 그 런 것) | 우 env 결정론 파지 0.806 · 들기 0.753 · 목표 성공 0.38 · 엄지 대향 0.22 | – |

학습 조건(각 폴더 `params/env.yaml` 에서 읽음):

| id | 보상 | 학습 물체 | 명령 지연 | FP++ (학습) | 공차 | 시작 |
|---|---|---|---|---|---|---|
| `rh56f1/aglt/left_cyl60gmir` | rh_aglt_r/iter_10 | cyl60 (노란 원통 Ø60×170) | 없음 | 지각 + 부착 | 0.02 | 홈 · hold 10 (ADR 20 부터) |
| `rh56f1/aglt/left_env17mir` | rh_aglt_r/iter_17 | cyl60 (노란 원통 Ø60×170) | 없음 | 지각 + 부착 | 0.02 | 홈 · hold 10 (ADR 20 부터) |
| `rh56f1/aglt/left_i05` | rh_aglt_l/iter_05 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 없음 (키 전 런) | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/left_i09d` | rh_aglt_l/iter_09 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 팔 9–12 · 손 3–5 스텝 | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/left_i10` | rh_aglt_l/iter_10 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 없음 | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/left_i10d` | rh_aglt_l/iter_10 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 팔 9–12 · 손 3–5 스텝 | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/right_cyl60g` | rh_aglt_r/iter_10 | cyl60 (노란 원통 Ø60×170) | 없음 | 지각 + 부착 | 0.02 | 홈 · hold 10 (ADR 20 부터) |
| `rh56f1/aglt/right_env17` | rh_aglt_r/iter_17 | cyl60 (노란 원통 Ø60×170) | 없음 | 지각 + 부착 | 0.02 | 홈 · hold 10 (ADR 20 부터) |
| `rh56f1/aglt/right_i03` | rh_aglt_r/iter_03 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 없음 (키 전 런) | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/right_i09d` | rh_aglt_r/iter_09 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 팔 9–12 · 손 3–5 스텝 | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/right_i10` | rh_aglt_r/iter_10 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 없음 | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/right_i10d` | rh_aglt_r/iter_10 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 팔 9–12 · 손 3–5 스텝 | 없음 | 0.1→0.02 | 홈 · hold 10 |
| `rh56f1/aglt/right_mirror_l5` | rh_aglt_r/iter_06 | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 없음 (키 전 런) | 없음 | 0.1→0.02 | 홈 · hold 10 |

## RH56F1 한 팔 컵 홀더 놓기 (rh_place)

rh_aglt 가 cyl60 을 쥐고 (0.25, ∓0.12, +0.12)에 멈춘 상태를 인계받아 컵 홀더 자리에 내려놓는다. 관측 · 행동 차원과 디코더는 rh_aglt 와 같고 목표(홀더 자리) · 시작(인계 뱅크, hold 0) · 놓은 뒤 45 스텝 sim 스크립트가 다르다.

**실기 개별 실행** — 미션 단계 `policy_place_<팔>`(노드 `rh_place_node`, 계약 rh_place_contract.json) — aglt 가 컵을 쥐고 인계 자리(`aglt_goal.py --handoff`)에서 stop 한 뒤. 목표 홀더는 미션 `PLACE_HOLDER`(기본 1). 콘솔 자리는 아직 없다(팔마다 한 자리 = aglt) — 다른 놓기 정책은 미션 산출물 `place_<팔>` 을 바꿔 쓴다.

| id | 쪽 | status | 체크포인트 | 설명 | sim 평가 | 미션 기본 |
|---|---|---|---|---|---|---|
| `rh56f1/place/left_i01` | left | candidate | ep800 | 우 place i09 ep4000 거울 → 보상 iter_01 로 이어 학습 ep800 | 결정론 128 env × 1800 스텝(2511 에피소드) 완벽 0.851 · 부분 0.066 · 실패 0.084 · 같은 조건 거울 원본 0.306 | rh56f1_control, rh56f1_fake |
| `rh56f1/place/right_i09` | right | candidate | ep4000 | cyl60 을 쥔 채 인계받아 홀더 1 · 2 에 내려놓기(보상 iter_09) | 결정론 128 env 완벽 0.907 · 부분 0.067 · 실패 0.026 · 실측 오차 넣으면 완벽 0.825 | rh56f1_control, rh56f1_fake |

학습 조건(각 폴더 `params/env.yaml` 에서 읽음):

| id | 보상 | 학습 물체 | 명령 지연 | FP++ (학습) | 시작 | 목표 |
|---|---|---|---|---|---|---|
| `rh56f1/place/left_i01` | rh_place_l/iter_01 | cyl60 (노란 원통 Ø60×170) | 없음 | 없음 | 인계 뱅크 (hold 0) | 홀더 0 · 1 |
| `rh56f1/place/right_i09` | rh_place_r/iter_09 | cyl60 (노란 원통 Ø60×170) | 없음 | 없음 | 인계 뱅크 (hold 0) | 홀더 1 · 2 |

## RH56F1 양팔 붓기 (pour_fj)

두 팔이 각자 컵을 쥔 채 소스 → 리시버로 붓는다. 관측 165 · 행동 26(팔당 7 + 손 6). 행동 법칙이 런마다 바뀌어 계약이 그 런의 env.yaml 에서 법칙을 읽는다.

**실기 개별 실행** — 콘솔 → `openarm_rh56f1` → 양팔 자리(미션 단계 `policy_pourfj`). 실기 미션에서는 두 컵 구분 전이라 막혀 있다.

| id | 쪽 | status | 체크포인트 | 설명 | sim 평가 | 미션 기본 |
|---|---|---|---|---|---|---|
| `rh56f1/pour_fj/both_f01` | both | candidate | ep2300 | RH56F1 양팔 붓기 첫 full-joint 런 — 배관 · fake 확인용(성능 후보 아님) | – | rh56f1_control, rh56f1_fake |

학습 조건(각 폴더 `params/env.yaml` 에서 읽음):

| id | 보상 | 팔 법칙 | 손 법칙 | 컵 | 시작 |
|---|---|---|---|---|---|
| `rh56f1/pour_fj/both_f01` | pour_bi_rh/iter_00 | increment | limits | shaker×0.65 (흰 출력물 aglt_cup_s065, Ø57) | 홈 · hold 30 |

## DG-5F-M short 한 팔 컵 집기 (joint)

DG-5F-M short 손 · 관절 증분 팔. 관측 133 · 행동 26(팔 7 증분 + 손 19 절대). 계약 joint_contract.json.

**실기 개별 실행** — 콘솔 → 로봇 `openarm_dg5f_m_short` → 오른/왼 자리(joint 계약이 있는 것만). 도착 판정 · 에피소드 길이는 카드 deploy 가 정한다.

| id | 쪽 | status | 체크포인트 | 설명 | sim 평가 | 미션 기본 |
|---|---|---|---|---|---|---|
| `dg5f_m/cup_grasp/left_i01` | left | candidate | e5802 | DG-5F 왼팔 컵 집기 · 들기 · 이송(cup_grasp iter_01) — 09.28 실기 | 끝 100 epoch 파지 0.75 · 들기 0.70 · 손바닥 접촉 0.65 · 4번 관절 과굽힘 손끝 파지 | dg5f_m_control, dg5f_m_fake |
| `dg5f_m/cup_grasp/left_i14` | left | candidate | e5320 | DG-5F 왼팔 인벨롭 그립(iter_14) — 컵 윗부분을 위에서 감싼다 | successes 0.70 · success_ep 0.43(tol 0.1125, 커리큘럼이 안 줄었다) | – |
| `dg5f_m/cup_pick/left_a00` | left | candidate | e3400 | DG-5F 왼팔 컵 접근 e3400(계약 없음) — left_a01(접근 + C자 유지)의 앞 단계 | – | – |
| `dg5f_m/cup_pick/left_a01` | left | candidate | e4280 | DG-5F 왼팔 컵 옆 C자 사전파지에서 멈춤(잡지 않음) — 실기 첫 확인용 | 먼 출발 접근 0.943 · 컵 접촉 0 · 손바닥-컵 간격 0.059 m | – |
| `dg5f_m/cup_pick/right_m15` | right | candidate | ep800 | DG-5F 오른팔 컵 집기 · 들기(cup_pick iter_15) — 09.28 실기 첫 오른팔 | 학습 창 파지 0.967 · 들기 0.959 · 먼 출발 성공 0.827 · 검지 0 · 5지 인벨롭 0 | dg5f_m_control, dg5f_m_fake |
| `dg5f_m/grasp_fj_rand/left_i00b` | left | candidate | e800 | DG-5F 왼팔 grasp_fj_rand i00b e800 — 왼팔 cup_pick 런(left_a00 · left_a01)의 출발 가중치(계약 없음) | – | – |
| `dg5f_m/grasp_fj_rand/right_i01` | right | hold | ep5000 | DG-5F 오른팔 grasp_fj_rand i01 best — 계약 없음 · hold | 먼 출발 성공 0.85 · 공차 0.019 · 컵 기울기 4.3° | – |

학습 조건(각 폴더 `params/env.yaml` 에서 읽음):

| id | 보상 | 목표 이어주기 | 배포 도착 판정 | 에피소드 | 손 관측 순서 |
|---|---|---|---|---|---|
| `dg5f_m/cup_grasp/left_i01` | cup_grasp_l/iter_01 | 0.08 | 0.0318 | 15.0 | 가정 |
| `dg5f_m/cup_grasp/left_i14` | cup_grasp_l/iter_14 | 0.08 | 0.1125 | 15.0 | 가정 |
| `dg5f_m/cup_pick/left_a00` | cup_pick_l_approach/iter_00 | 0.0 | – | – | – |
| `dg5f_m/cup_pick/left_a01` | cup_pick_l_approach/iter_01 | 0.0 | 끔 | 끝없음 | 가정 |
| `dg5f_m/cup_pick/right_m15` | cup_pick_r_manip/iter_15 | 0.08 | 끔 | 끝없음 | 실측 |
| `dg5f_m/grasp_fj_rand/left_i00b` | grasp_fj_rand_left/iter_00 | 0.0 | – | – | – |
| `dg5f_m/grasp_fj_rand/right_i01` | grasp_fj_rand/iter_01 | 0.0 | – | – | – |

## DG-5F short 양팔 붓기 (pour_fab, fabric)

팔당 손바닥 6D 증분(fabric) + grip3 · 관측 223 · 행동 18 · MLP.

**실기 개별 실행** — fake 미션만(`config/mission_pour_fake.yaml` · `mission_pour_i24_fake.yaml`). 실기는 ckpt_gate · pour_guard 를 넘긴 뒤.

| id | 쪽 | status | 체크포인트 | 설명 | sim 평가 | 미션 기본 |
|---|---|---|---|---|---|---|
| `dg5f_m/pour_fab/both_i18` | both | candidate | ep2500 | DG-5F short 양팔 붓기 i18 — 붓기 s2r 기준(09.28), 당장 실기 안 함 | ADR30 64 env 성공 0.828 · in_target 0.907 · 흘림 0.07 · 소스 최대 기울기 155°(ckpt_gate 탈락 3) | – |
| `dg5f_m/pour_fab/both_i24` | both | hold | best | DG-5F short 양팔 붓기 i24 best — 기울기를 116° 로 낮췄지만 hold | ADR30 64 env env 성공 0.719 · in_target 0.925 · 흘림 중앙 0 · 리시버 떨림 소스의 1.5 배 | pour_i24_fake |

학습 조건(각 폴더 `params/env.yaml` 에서 읽음):

| id | 보상 | 팜 slew |
|---|---|---|
| `dg5f_m/pour_fab/both_i18` | pour_bi/iter_18 | 없음 (slew 전) |
| `dg5f_m/pour_fab/both_i24` | pour_bi/iter_24 | 0.03 |

## sim 에서 개별 실행(비교 분석)

hdgp 에서 play 로 돌린다. 체크포인트 옆 `params/` 를 play 가 복원하므로 학습 때 env(물체 · FP++ · 지연)가 그대로 선다. 학습이 도는 GPU 에는 올리지 않는다(nvidia-smi 확인). 결정론 평가는 그 정책을 학습한 호스트에서 한다.

```bash
cd ~/rl_ws/hdgp
python scripts/reinforcement_learning/rl_games/play.py --task open-short_l_cup_pick-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/cup_grasp/left_i01/nn/cg_l_i01_e5802.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_l_cup_grasp-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/cup_grasp/left_i14/nn/cg_l_i14_e5320.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_l_cup_pick-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/cup_pick/left_a00/nn/cup_pick_l_approach_e3400.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_l_cup_pick-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/cup_pick/left_a01/nn/cup_pick_l_approach_hold_e4280.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_r_cup_pick-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/cup_pick/right_m15/nn/last_open-short_r_cup_pick-lstm_ep_800_rew_2313.1377.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_l_grasp_fj_t2r_rand-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/grasp_fj_rand/left_i00b/nn/seed_fj_randL_i00b_e800.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_r_grasp_fj_t2r_rand-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/grasp_fj_rand/right_i01/nn/fj_rand_i01_best_ep5000.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_b_pour_fab-play --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/pour_fab/both_i18/nn/last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-short_b_pour_fab-play --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/dg5f_m/pour_fab/both_i24/nn/open-short_b_pour_fab.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_l_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/left_cyl60gmir/nn/last_open-rh_l_aglt-lstm_ep_2600_rew_0.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_l_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/left_env17mir/nn/last_open-rh_l_aglt-lstm_ep_1000_rew_0.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_l_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/left_i05/nn/last_open-rh_l_aglt-lstm_ep_3800_rew_3346.764.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_l_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/left_i09d/nn/last_open-rh_l_aglt-lstm_ep_3000_rew_2940.748.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_l_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/left_i10/nn/last_open-rh_l_aglt-lstm_ep_4600_mirror_of_r_i10.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_l_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/left_i10d/nn/last_open-rh_l_aglt-lstm_ep_2200_rew_2715.922.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/right_cyl60g/nn/last_open-rh_r_aglt-lstm_ep_2600_rew_3509.6997.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/right_env17/nn/last_open-rh_r_aglt-lstm_ep_1000_rew_1240.1438.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/right_i03/nn/last_open-rh_r_aglt-lstm_ep_4600_rew_3222.941.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/right_i09d/nn/last_open-rh_r_aglt-lstm_ep_5800_rew_1844.392.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/right_i10/nn/last_open-rh_r_aglt-lstm_ep_4600_rew_2678.948.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/right_i10d/nn/last_open-rh_r_aglt-lstm_ep_1400_rew_1483.9092.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_aglt-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/aglt/right_mirror_l5/nn/aglt_l_i05_ep3800_to_r.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_l_place-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/place/left_i01/nn/last_open-rh_l_place-lstm_ep_800_rew_935.48755.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_r_place-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/place/right_i09/nn/place_r_i09_ep4000.pth
python scripts/reinforcement_learning/rl_games/play.py --task open-rh_b_pour_fj-play-lstm --headless --num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/rh56f1/pour_fj/both_f01/nn/last_open-rh_b_pour_fj-lstm_ep_2300_rew_1107.8425.pth
```

## status

- `candidate` — 받아만 뒀다 — 계약·체인 검증 전
- `verified` — 계약이 서고 fake 체인을 통과했다 — 실기 승인 전
- `deployed` — 실기에서 승인받아 돌린 적이 있다
- `hold` — 쓰지 않는다 — 이유는 note 에

## note (카드 원문)

- `dg5f_m/cup_grasp/left_i01` — run cg_l_i01(09.27 arm4090, PPO-LSTM 4096 env, seed 42, fresh). best = e5802 · last_mean_rewards 2597. 영상은 e7200(snap_e7200_0931) 과 best 둘 다 있다. 끝 100 epoch: 파지 0.75 · 들기 0.70 · 손바닥 접촉 0.65. 알려진 결함(사용자 판정): 4번 관절 과굽힘 손끝 파지 · 새끼 미참여 · 급하게 들어 올림. 계약은 dg5f_m/cup_pick/left_a01 과 출처 칸(체크포인트 · params 해시)만 다르다 — 같은 과제 · 자산 · 시작 자세(왼팔 홈 경로 끝) · 목표(컵 + 0.246 m). 학습은 목표에 닿으면 0.08 m 옮긴 다음 목표를 주지만(goal_delta_distance, 최대 5) 배포는 첫 목표에서 멈춘다. 손 관측 순서는 가정(e4280 과 같은 규칙) — Isaac trace 대조는 아직. fake 체인(도메인 97, CPU, 컵 0.25 0.15): 32 s · 59.8 Hz · not-ok 0 · proc p95 9.1 ms · 손을 쥔 뒤 물체 추정 attached_live.
- `dg5f_m/cup_grasp/left_i14` — run cg_l_i14(09.28 local 5090, PPO-LSTM 4096 env, seed 42, fresh, bounds_loss_coef 0.005). best = e5320 · last_mean_rewards 596.6(보상 식이 i01 과 달라 값 비교 불가). e5320 부근: successes_mean 0.70 · success_ep 0.43. tfevents task/tol 은 학습 내내 0.1125(커리큘럼 시작값, 줄지 않음) → 배포 도착 판정도 0.1125 m · 10 스텝. 사용자 영상 판정: 원한 인벨롭 그립, 다만 컵 윗부분을 위에서 감싼다(iter_15 에서 더 아래로). 계약은 dg5f_m/cup_grasp/left_i01 과 출처 칸만 다르다 — 같은 자산 · 시작 자세(왼팔 홈 경로 끝) · 목표(컵 + 0.246 m) · obs/act. critic 만 157 → 178(배포는 actor 만 쓴다). env 에 새로 생긴 real_*_lag_steps · real_obj_dropout 은 전부 0(학습에 지연 없음). 손 관측 순서는 가정(e4280 · i01 과 같은 규칙) — Isaac trace 대조는 아직. fake 체인(도메인 97, CPU, 컵 0.25 0.15): 30 s · not-ok 0 · proc p95 9.0 ms · 팔 이동 1.42 rad · reset/start FP++ 검사 통과.
- `dg5f_m/cup_pick/left_a00` — 런 fj_abL_a00 e3400. 학습 당시 gym id 는 open-short_l_grasp_fj_ab-lstm(지금 open-short_l_cup_pick-lstm 과 같은 cfg, 이름 변경 전). 출발 가중치 fj_randL_i00b e800(= dg5f_m/grasp_fj_rand/left_i00b). 인터페이스 obs 133 / action 26(팔 7 관절 증분 + 손 19 절대, 프로파일 tesollo_left_short_tl). 계약 없음 — 실기 확인은 dg5f_m/cup_pick/left_a01.
- `dg5f_m/cup_pick/left_a01` — run cp_l_a01 e4284(학습 커밋 0fce5124, local 5090). 저장 시 먼 출발 접근 0.943 · 컵 접촉 0 · 컵 기울기 0.03° · 손바닥-컵 간격 0.059 m. 컵을 잡지 않는다 — 컵 옆 C자 사전파지 자세로 가서 멈춘다(실기 첫 확인에 안전한 쪽). 시작 자세 = 오른팔 시작 자세의 거울 = 왼팔 홈 경로 끝(차이 0). 컵 학습 범위 x 0.10~0.40 · y 0.00~0.30. 손 관측 순서는 가정(오른팔 실측과 같은 규칙) — 로컬 5090 이 다른 세션 학습 중이라 Isaac trace 대조는 아직. fake 체인(도메인 97, CPU, 컵 0.25 0.15): 30 s 1754 틱 · not-ok 0 · seq 결손 0 · proc p95 7.7 ms · 팔 이동 1.23 rad. 이상 추종 닫힌 루프(15 s): 손바닥이 컵 원점 0.30 → 0.144 m 에서 3 s 만에 멈추고 유지, 팔 속도 ≤ 0.30 rad/s.
- `dg5f_m/cup_pick/right_m15` — cp_r_m15 ep_800 (md5 4717e325 = policy_zoo cup_pick_r_manip_e800). 보상 iter_15. 학습 창 파지 0.967 · 들기 0.959 · 먼 출발 성공 0.827, 외란 20 N/kg 600 epoch 유지. 약점: 검지 0 · 5지 인벨롭 0(4지+손바닥), 고정 공차 평가 없음, 공차 0.082 까지만.
- `dg5f_m/grasp_fj_rand/left_i00b` — 런 fj_randL_i00b e800. gym id 가 s2r 동결 트랙(grasp_fj_t2r_rand)이다 — 재생만 하고 그 트랙에 쓰지 않는다. 인터페이스 obs 133 / action 26(팔 7 관절 증분 + 손 19 절대, 프로파일 tesollo_left_short_tl). 계약 없음.
- `dg5f_m/grasp_fj_rand/right_i01` — 09.28 hold — 오른팔 첫 실험은 dg5f_m/cup_pick/right_m15(cup_pick 세션 s2r 후보, 사용자 선택)으로 한다. 인터페이스는 같다. s2r 후보. 먼 출발 성공 0.85 · 공차 0.019 · 컵 기울기 4.3°. sha256 앞 16자리 e76f9c6079663b61. 계약 없음 — obs 133 / action 26(팔 7 관절 증분 + 손 19 절대)은 기존 세 family 어디에도 안 맞는다. build_deploy_contract 가 grasp_s2r 로 판정했다가 인터페이스 차이로 거절한다. 새 family 가 필요하다. 같은 한 벌의 fj_rand_i01_final.pth 는 공차 바닥 뒤 열화(성공 0.61)라 쓰지 않는다.
- `dg5f_m/pour_fab/both_i18` — 09.28 사용자 결정: 붓기는 i18 을 쓴다(i24 는 hold), 다만 당장 진행하지 않는다. checkpoint 는 검증된 계약 (logs/policy/pour_i18/pour_contract.json, md5 f50b09a6)과 같은 ep_2500 으로 바꿨다. ★가중치가 계약과 다르다: 여기 nn/open-short_b_pour_fab.pth 는 md5 e473c709…, 검증된 pour_contract.json 은 f50b09a6…(logs/policy/pour_i18/nn/last_..._ep_2500_rew_36408.105.pth)로 만들어졌다. 크기도 4,667,063 vs 4,669,209 로 다르다 — 같은 런의 다른 내보내기다. 이 파일로 쓰려면 계약을 다시 만들어야 한다. hold 인 본래 이유: ckpt_gate 탈락 3건 — 소스컵 최대 기울기 155.1°(한계 120) · 이탈 0.344 m(한계 0.25) · 중앙선 교차 64/64 env. 실기에서는 pour_guard_node 가 같은 두 조건으로 episode/abort 를 건다. 검증된 쪽(계약·trace·actor 재현 2.1e-6)은 logs/policy/pour_i18/ 에 그대로 있다. 다음 후보는 dg5f_m/pour_fab/both_i24(09.28 등록, 붓기 세션 s2r 1순위).
- `dg5f_m/pour_fab/both_i24` — 09.28 hold — 사용자 결정: 붓기 s2r 기준은 i18. 계약 · 재현 · fake 체인은 통과한 채로 둔다. t2r_i24 best(epoch 1557, 학습 커밋 8cc6c6c6). ADR30 결정론 평가 64env: env 성공 0.719 · in_target 0.925 · 흘림 중앙값 0 · 최고 기울기 116°(i18 은 155°). 들기 순간 파지 유지 드리프트 1.08~1.56 cm. 약점: 리시버 팔 떨림이 소스의 약 1.5배(액션 부호반전 0.445 대 0.294) · 작업이 중심선 왼쪽(소스 y +0.16, 리시버 +0.22) · 전이 60 에피소드 중 14 개 env 성공 실패. trace.npz · trace_meta.json 은 그 ADR30 평가(best) 궤적이다. 재생 코드 주의: hdgp main 은 관측 224/294(c6708ec7) — 223 인 이 정책은 d5c80a4c 또는 2d1c1c78 이전 코드로 재생한다.
- `rh56f1/aglt/left_cyl60gmir` — 결정론(5090, 64 env, tol 0.02, 같은 지각): 에피소드당 성공 1.62. 주의: 쥔 동안 컵 기울기 중앙 8.4°(오른팔 4.4°).
- `rh56f1/aglt/left_env17mir` — 결정론(64 env × 1800 스텝, tol 0.02, ADR 0, 하중 끔): 에피소드당 성공 4.40 / 5(5개 다 채움 0.83, 오른팔 4.91~4.93). 주의: 쥔 동안 컵 기울기 8.1°(오른팔 4.9°). params/env.yaml 의 pour_load_enable 은 false(오른팔 true) — 학습하지 않은 거울이라 계약에는 영향 없다(계약 diff 0). 놓기 인계 주의: 놓기 i01 시작 뱅크의 쥔 높이는 p5~p95 3.1~5.5 cm, 오른팔 env17 은 +2.0 cm (이 거울은 따로 재지 않았다) — 에피소드는 아직 cyl60gmir.
- `rh56f1/aglt/left_i05` — 학습 로그 e3785~3964 파지 0.80 · 들기 0.75 · 목표 성공 0.9~1.1/에피소드(허용오차 2.1 cm). 엄지 대향이 후반에 줄었다(0.52 → 0.04). 계약 rh_aglt_contract.json 은 tools/build_rh_aglt_contract.py 로 런의 env.yaml 에서 만든다. 손 관측은 이름(프로필) 순 — pour_fj 와 달리 PhysX 순서 문제가 없다. 목표 = 리셋 때 컵 + (0, 0, 0.14)(학습 첫 목표 분포의 가운데).
- `rh56f1/aglt/left_i09d` — 보상 iter_09 c1 · env 실기 반응(팔 지연 9~12 · 손 3~5 스텝 · 펌웨어 멈춤 · 편 손 하한)으로 학습. 결정론(64 env, 지연 켬) 컵 든 에피소드 0.98 · 낙하 0 · 목표 성공 1.19/에피소드(공차 0.035) — 공차 0.02(배포 달성 판정)에서는 0.05. 다섯 손가락 파지. 주의: 어깨 j2 한계 0.31 · 컵 기울기 중앙 18°.
- `rh56f1/aglt/left_i10` — 좌 env 결정론(arm5080, 64 env) 지연 0: 컵 든 에피소드 1.00 · 목표 성공 2.02/에피소드(0.0229) · 0.83(0.02). 실측 지연: 컵 든 에피소드 0.78. 엄지 · 검지 · 중지 · 새끼 접촉, 약지 안 닿음. 컵 기울기 중앙 8.6°.
- `rh56f1/aglt/left_i10d` — 우 i10 거울(i10mir) → 실측 지연으로 이어 2200 epoch(server). 결정론(64 env, 지연 켬): 컵 든 에피소드 0.89 · 낙하 0.006 · 목표 성공 2.55/에피소드(학습 공차 0.052) · 0.06(0.02). 다섯 손가락 파지 · 컵 기울기 중앙 13°.
- `rh56f1/aglt/right_cyl60g` — aglt_r_cyl60n ep12800 → FP++ 지각(지연 250~400 ms · 10.5 Hz · 광선 잔차 ±4 mm) + 파지 후 FK 부착으로 이어 학습. 결정론(5090, 64 env, tol 0.02, 같은 지각): 에피소드당 성공 1.77(원 정책 1.16). 배포 전제 796b074 · 03cd917 · 104f2b1. 주의: 이송 중 어깨 j2 가 한계 0.05 rad 안에 37 %(출발 · 접근 0, 몸 접촉 0).
- `rh56f1/aglt/right_env17` — aglt_r_env16 ep800 → 보상 iter_17 로 aglt_r_env17(3000 epoch) 이어 학습, ep1000 선택. 학습 env 는 cyl60g 와 같은 FP++ 지각 · 파지 후 부착 · 명령 지연 0 에 붓기 하중(prob 0.7)과 ADR 20→30 을 더했다. 뒤집기(구슬 20 · 실제 회전 ~118°): 놓침 0.14(cyl60g 0.06), 손 안 이동 중앙 0.29 cm. 실기에서 볼 것: 쥔 높이가 cyl60g(+4.3 cm)보다 약 2.3 cm 낮다. 놓기 인계 주의: 놓기 i09 시작 뱅크의 쥔 높이는 p5~p95 4.3~5.3 cm 라 env17(+2.0 cm)은 그 밖이다 — 에피소드는 아직 cyl60g.
- `rh56f1/aglt/right_i03` — 09.30 hold — 사용자 영상 판정: 엄지를 입구 안에 넣는 파지(결정론 probe 엄지 끝 컵 안 0.97). 오른팔은 rh56f1/aglt/right_mirror_l5 를 쓴다. LOOP_STATE 추천 구간 e4400~4800 (성공 200 epoch 평균 최고 1.1~1.3/에피소드, 파지 0.73~0.79). 마지막 가중치 쓰지 말 것. 엄지 대향 0 — 사용자 판정: 모양 결함 알고 쓰는 첫 실기 후보. 계약 rh_aglt_contract.json 은 tools/build_rh_aglt_contract.py 로 런의 env.yaml 에서 만든다. 손 관측은 이름(프로필) 순 — pour_fj 와 달리 PhysX 순서 문제가 없다. 목표 = 리셋 때 컵 + (0, 0, 0.14)(학습 첫 목표 분포의 가운데).
- `rh56f1/aglt/right_i09d` — 보상 iter_09 c1 · env 실기 반응(팔 지연 9~12 · 손 3~5 스텝 · 펌웨어 멈춤 · 편 손 하한)으로 학습. 결정론(5090, 64 env, 지연 켬) 컵 든 에피소드 0.91 · 낙하 스텝 0.016 · 목표 성공 0.84/에피소드(공차 0.039) — 공차 0.02(배포 달성 판정)에서는 0.05. 세 손가락 파지(약지 · 새끼 0). 주의: 손목 j6 한계(여유 < 5 %) 0.45 · j6 출력 |mu|>1 0.58.
- `rh56f1/aglt/right_i10` — 보상 iter_10 · env 9a46c174(지연 키 0 — 지연 없이 학습). 학습 공차 0.0229 m. 결정론(64 env) 지연 0: 컵 든 에피소드 1.00, 목표 성공 2.0~2.3/에피소드(0.0229) · 0.8~0.9(0.02). 실측 지연을 넣으면 컵 든 에피소드 0.8. 엄지 · 검지 · 중지 · 새끼 접촉, 약지 안 닫음. 지연 적응판(②)은 학습 중.
- `rh56f1/aglt/right_i10d` — ① i10 ep4600 → 실측 지연(팔 9~12 · 손 3~5 스텝)으로 이어 학습. 결정론(server, 64 env, 지연 켬): 컵 든 에피소드 0.94 · 낙하 0.006 · 목표 성공 2.40/에피소드(학습 공차 0.059) · 0.16(0.02). 다섯 손가락 파지 · 컵 기울기 중앙 12°. 주의: 손목 j6 한계 0.40~0.48(이송 중).
- `rh56f1/aglt/right_mirror_l5` — 우 env 결정론 probe(서버 GPU0, 64 env × 1800 스텝) 파지 0.806 · 들기 0.753 · 목표 성공 0.38 — 좌 원본(0.805 · 0.755)과 같다. 엄지 입구 안 0.0002(i03 은 0.97 — 사용자 영상 판정상 입구 안 엄지 파지라 좋은 오른팔 정책이 아니라고 grasping 세션이 알림). 알려진 결함: 엄지 대향 낮음(0.22), 네 손가락 손끝 파지(첫마디 0).
- `rh56f1/place/left_i01` — server GPU0 place_l_i01(보상 reward_gen/rh_place_l/iter_01). 평가는 홀더 고정 배치(holder_y_shift_fixed 0 · DR 축소), params/ 는 학습 런 그대로(평가 때 바꾼 홀더 DR 값 아님). 목표 y 구간 −0.06~0 완벽 0.85 · 0~+0.20 완벽 0.851. 배포는 rh56f1/place/right_i09 note 와 같다(rh_place_contract.json · rh_place_node · policy_place_left). FP++ 지각 · 부착 없이 학습, 명령 지연 0. hdgp 51013e96. 놓음 게이트(release_open_min_rad 0.15) sim 검증(PLACE 10.04): 인계 때 쥔 양 중앙 0.53 rad(0.40~0.64), 놓음 때 연 양 중앙 0.334 · 하위 1 % 0.217 — 0.15 미만 0 %. 실기 파지가 덜 쥔 채 넘어오면 '쥔 양 대비 비율' 게이트로 바꾼다.
- `rh56f1/place/right_i09` — server GPU0 place_r_i09(보상 reward_gen/rh_place_r/iter_09) ep4000. 좌 place_l_i01 의 출발점. 학습 코드 hdgp 51013e96. 10.04 배포 연결: rh_place_contract.json(tools/build_rh_aglt_contract.py 가 place 런을 알아본다) · rh_place_node · 미션 단계 policy_place_<팔>. 목표 = 홀더 원점 + 0.060 m, 시작 = pd 가 붙잡은 aglt 마지막 joint_target, 컵은 reset 때 FP++ 한 장으로 손바닥에 붙인다(학습 attached), 놓음 판정 = 손끝 촉각 < 1 N · 관절 힘 < 300 g(임시) 5 스텝. FP++ 지각 · 부착 없이 학습(fpp_enable · fpp_attach_enable false), 명령 지연 0. 홀더 x 0.380 · 목표 홀더 1 · 2. 놓음 게이트(release_open_min_rad 0.15) sim 검증(PLACE 10.04, 128 env 결정론): 인계 때 쥔 양 중앙 0.41 rad(0.38~0.46), 놓음 때 연 양 중앙 0.257 · 하위 1 % 0.209 — 0.15 미만 0.2~0.4 %. 실기 문턱은 0.25 를 넘기지 않는다(쥔 양 상한).
- `rh56f1/pour_fj/both_f01` — t2r_rh4_f01(pour_bi_rh 4번째 사이클 첫 full-joint 런). 손 direct 범위가 관절 한계 전 범위 · 접촉 동결 없음 — 그 뒤 런(de926392 grip 범위 + 1 N 동결, da3e33fb 팔 절대 목표)과 행동 규칙이 다르다. 계약이 런의 env.yaml 에서 그 시대 규칙을 읽는다. 손 관측 순서는 추정(assumed) — verified 전. 실기 성능 후보가 아니라 배관 · fake 확인용.
