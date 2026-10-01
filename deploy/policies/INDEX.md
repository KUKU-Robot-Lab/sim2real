# policies — 쓸 정책 목록

`deploy/policy_control/tools/policies.py --write-index` 가 만든다. 손으로 고치지 않는다 — 고칠 것은 각 `policy.yaml` 이다.

| id | status | task | side | checkpoint | 계약 | 점검 |
|---|---|---|---|---|---|---|
| `both_pour_i18` | candidate | open-short_b_pour_fab | both | last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth | - | ok |
| `both_pour_i24` | hold | open-short_b_pour_fab | both | open-short_b_pour_fab.pth | pour_contract.json | ok |
| `both_rh_pourfj_f01` | candidate | open-rh_b_pour_fj-lstm | both | last_open-rh_b_pour_fj-lstm_ep_2300_rew_1107.8425.pth | pour_fj_contract.json | ok |
| `left_aglt` | candidate | open-short_l_cup_pick-lstm | left | cup_pick_l_approach_hold_e4280.pth | - | ok |
| `left_cg_i01` | candidate | open-short_l_cup_pick-lstm | left | cg_l_i01_e5802.pth | joint_contract.json | ok |
| `left_cg_i14` | candidate | open-short_l_cup_grasp-lstm | left | cg_l_i14_e5320.pth | joint_contract.json | ok |
| `left_cp_e4280` | candidate | open-short_l_cup_pick-lstm | left | cup_pick_l_approach_hold_e4280.pth | joint_contract.json | ok |
| `left_rh_aglt_i05` | candidate | open-rh_l_aglt-lstm | left | last_open-rh_l_aglt-lstm_ep_3800_rew_3346.764.pth | rh_aglt_contract.json | ok |
| `left_rh_aglt_i09d` | candidate | open-rh_l_aglt-lstm | left | last_open-rh_l_aglt-lstm_ep_3000_rew_2940.748.pth | rh_aglt_contract.json | ok |
| `left_rh_aglt_i10` | candidate | open-rh_l_aglt-lstm | left | last_open-rh_l_aglt-lstm_ep_4600_mirror_of_r_i10.pth | rh_aglt_contract.json | ok |
| `right_aglt` | hold | open-short_r_grasp_fj_t2r_rand-lstm | right | fj_rand_i01_best_ep5000.pth | - | ok |
| `right_m15_e800` | candidate | open-short_r_cup_pick-lstm | right | last_open-short_r_cup_pick-lstm_ep_800_rew_2313.1377.pth | joint_contract.json | ok |
| `right_rh_aglt_i03` | hold | open-rh_r_aglt-lstm | right | last_open-rh_r_aglt-lstm_ep_4600_rew_3222.941.pth | rh_aglt_contract.json | ok |
| `right_rh_aglt_i09d` | candidate | open-rh_r_aglt-lstm | right | last_open-rh_r_aglt-lstm_ep_5800_rew_1844.392.pth | rh_aglt_contract.json | ok |
| `right_rh_aglt_i10` | candidate | open-rh_r_aglt-lstm | right | last_open-rh_r_aglt-lstm_ep_4600_rew_2678.948.pth | rh_aglt_contract.json | ok |
| `right_rh_aglt_mirror_l5` | candidate | open-rh_r_aglt-lstm | right | aglt_l_i05_ep3800_to_r.pth | rh_aglt_contract.json | ok |

## status

- `candidate` — 받아만 뒀다 — 계약·체인 검증 전
- `verified` — 계약이 서고 fake 체인을 통과했다 — 실기 승인 전
- `deployed` — 실기에서 승인받아 돌린 적이 있다
- `hold` — 쓰지 않는다 — 이유는 note 에

## note

- `both_pour_i18` — 09.28 사용자 결정: 붓기는 i18 을 쓴다(i24 는 hold), 다만 당장 진행하지 않는다. checkpoint 는 검증된 계약 (logs/policy/pour_i18/pour_contract.json, md5 f50b09a6)과 같은 ep_2500 으로 바꿨다. ★가중치가 계약과 다르다: 여기 nn/open-short_b_pour_fab.pth 는 md5 e473c709…, 검증된 pour_contract.json 은 f50b09a6…(logs/policy/pour_i18/nn/last_..._ep_2500_rew_36408.105.pth)로 만들어졌다. 크기도 4,667,063 vs 4,669,209 로 다르다 — 같은 런의 다른 내보내기다. 이 파일로 쓰려면 계약을 다시 만들어야 한다. hold 인 본래 이유: ckpt_gate 탈락 3건 — 소스컵 최대 기울기 155.1°(한계 120) · 이탈 0.344 m(한계 0.25) · 중앙선 교차 64/64 env. 실기에서는 pour_guard_node 가 같은 두 조건으로 episode/abort 를 건다. 검증된 쪽(계약·trace·actor 재현 2.1e-6)은 logs/policy/pour_i18/ 에 그대로 있다. 다음 후보는 both_pour_i24(09.28 등록, 붓기 세션 s2r 1순위).
- `both_pour_i24` — 09.28 hold — 사용자 결정: 붓기 s2r 기준은 i18. 계약 · 재현 · fake 체인은 통과한 채로 둔다. t2r_i24 best(epoch 1557, 학습 커밋 8cc6c6c6). ADR30 결정론 평가 64env: env 성공 0.719 · in_target 0.925 · 흘림 중앙값 0 · 최고 기울기 116°(i18 은 155°). 들기 순간 파지 유지 드리프트 1.08~1.56 cm. 약점: 리시버 팔 떨림이 소스의 약 1.5배(액션 부호반전 0.445 대 0.294) · 작업이 중심선 왼쪽(소스 y +0.16, 리시버 +0.22) · 전이 60 에피소드 중 14 개 env 성공 실패. trace.npz · trace_meta.json 은 그 ADR30 평가(best) 궤적이다. 재생 코드 주의: hdgp main 은 관측 224/294(c6708ec7) — 223 인 이 정책은 d5c80a4c 또는 2d1c1c78 이전 코드로 재생한다.
- `both_rh_pourfj_f01` — t2r_rh4_f01(pour_bi_rh 4번째 사이클 첫 full-joint 런). 손 direct 범위가 관절 한계 전 범위 · 접촉 동결 없음 — 그 뒤 런(de926392 grip 범위 + 1 N 동결, da3e33fb 팔 절대 목표)과 행동 규칙이 다르다. 계약이 런의 env.yaml 에서 그 시대 규칙을 읽는다. 손 관측 순서는 추정(assumed) — verified 전. 실기 성능 후보가 아니라 배관 · fake 확인용.
- `left_aglt` — 세 체크포인트 한 벌(접근 e3400 · 접근+C자 유지 e4280 · 입력 시드 i00b). 후보는 가장 나중 단계인 e4280 으로 적어 뒀다 — 다른 것을 쓰려면 이 checkpoint: 한 줄만 바꾸면 된다. 인터페이스는 오른팔과 같다(obs 133 / action 26, 팔 7 증분 + 손 19 절대, 프로파일 tesollo_left_short_tl). 계약 없음 — right_aglt 와 같은 이유(새 family 필요). seed_fj_randL_i00b_e800 의 gym id 는 s2r 동결 트랙이다 — 재생만 하고 그 트랙에 쓰지 않는다.
- `left_cg_i01` — run cg_l_i01(09.27 arm4090, PPO-LSTM 4096 env, seed 42, fresh). best = e5802 · last_mean_rewards 2597. 영상은 e7200(snap_e7200_0931) 과 best 둘 다 있다. 끝 100 epoch: 파지 0.75 · 들기 0.70 · 손바닥 접촉 0.65. 알려진 결함(사용자 판정): 4번 관절 과굽힘 손끝 파지 · 새끼 미참여 · 급하게 들어 올림. 계약은 left_cp_e4280 과 출처 칸(체크포인트 · params 해시)만 다르다 — 같은 과제 · 자산 · 시작 자세(왼팔 홈 경로 끝) · 목표(컵 + 0.246 m). 학습은 목표에 닿으면 0.08 m 옮긴 다음 목표를 주지만(goal_delta_distance, 최대 5) 배포는 첫 목표에서 멈춘다. 손 관측 순서는 가정(e4280 과 같은 규칙) — Isaac trace 대조는 아직. fake 체인(도메인 97, CPU, 컵 0.25 0.15): 32 s · 59.8 Hz · not-ok 0 · proc p95 9.1 ms · 손을 쥔 뒤 물체 추정 attached_live.
- `left_cg_i14` — run cg_l_i14(09.28 local 5090, PPO-LSTM 4096 env, seed 42, fresh, bounds_loss_coef 0.005). best = e5320 · last_mean_rewards 596.6(보상 식이 i01 과 달라 값 비교 불가). e5320 부근: successes_mean 0.70 · success_ep 0.43. tfevents task/tol 은 학습 내내 0.1125(커리큘럼 시작값, 줄지 않음) → 배포 도착 판정도 0.1125 m · 10 스텝. 사용자 영상 판정: 원한 인벨롭 그립, 다만 컵 윗부분을 위에서 감싼다(iter_15 에서 더 아래로). 계약은 left_cg_i01 과 출처 칸만 다르다 — 같은 자산 · 시작 자세(왼팔 홈 경로 끝) · 목표(컵 + 0.246 m) · obs/act. critic 만 157 → 178(배포는 actor 만 쓴다). env 에 새로 생긴 real_*_lag_steps · real_obj_dropout 은 전부 0(학습에 지연 없음). 손 관측 순서는 가정(e4280 · i01 과 같은 규칙) — Isaac trace 대조는 아직. fake 체인(도메인 97, CPU, 컵 0.25 0.15): 30 s · not-ok 0 · proc p95 9.0 ms · 팔 이동 1.42 rad · reset/start FP++ 검사 통과.
- `left_cp_e4280` — run cp_l_a01 e4284(학습 커밋 0fce5124, local 5090). 저장 시 먼 출발 접근 0.943 · 컵 접촉 0 · 컵 기울기 0.03° · 손바닥-컵 간격 0.059 m. 컵을 잡지 않는다 — 컵 옆 C자 사전파지 자세로 가서 멈춘다(실기 첫 확인에 안전한 쪽). 시작 자세 = 오른팔 시작 자세의 거울 = 왼팔 홈 경로 끝(차이 0). 컵 학습 범위 x 0.10~0.40 · y 0.00~0.30. 손 관측 순서는 가정(오른팔 실측과 같은 규칙) — 로컬 5090 이 다른 세션 학습 중이라 Isaac trace 대조는 아직. fake 체인(도메인 97, CPU, 컵 0.25 0.15): 30 s 1754 틱 · not-ok 0 · seq 결손 0 · proc p95 7.7 ms · 팔 이동 1.23 rad. 이상 추종 닫힌 루프(15 s): 손바닥이 컵 원점 0.30 → 0.144 m 에서 3 s 만에 멈추고 유지, 팔 속도 ≤ 0.30 rad/s.
- `left_rh_aglt_i05` — 학습 로그 e3785~3964 파지 0.80 · 들기 0.75 · 목표 성공 0.9~1.1/에피소드(허용오차 2.1 cm). 엄지 대향이 후반에 줄었다(0.52 → 0.04). 계약 rh_aglt_contract.json 은 tools/build_rh_aglt_contract.py 로 런의 env.yaml 에서 만든다. 손 관측은 이름(프로필) 순 — pour_fj 와 달리 PhysX 순서 문제가 없다. 목표 = 리셋 때 컵 + (0, 0, 0.14)(학습 첫 목표 분포의 가운데).
- `left_rh_aglt_i09d` — 보상 iter_09 c1 · env 실기 반응(팔 지연 9~12 · 손 3~5 스텝 · 펌웨어 멈춤 · 편 손 하한)으로 학습. 결정론(64 env, 지연 켬) 컵 든 에피소드 0.98 · 낙하 0 · 목표 성공 1.19/에피소드(공차 0.035) — 공차 0.02(배포 달성 판정)에서는 0.05. 다섯 손가락 파지. 주의: 어깨 j2 한계 0.31 · 컵 기울기 중앙 18°.
- `left_rh_aglt_i10` — 좌 env 결정론(arm5080, 64 env) 지연 0: 컵 든 에피소드 1.00 · 목표 성공 2.02/에피소드(0.0229) · 0.83(0.02). 실측 지연: 컵 든 에피소드 0.78. 엄지 · 검지 · 중지 · 새끼 접촉, 약지 안 닿음. 컵 기울기 중앙 8.6°.
- `right_aglt` — 09.28 hold — 오른팔 첫 실험은 right_m15_e800(cup_pick 세션 s2r 후보, 사용자 선택)으로 한다. 인터페이스는 같다. s2r 후보. 먼 출발 성공 0.85 · 공차 0.019 · 컵 기울기 4.3°. sha256 앞 16자리 e76f9c6079663b61. 계약 없음 — obs 133 / action 26(팔 7 관절 증분 + 손 19 절대)은 기존 세 family 어디에도 안 맞는다. build_deploy_contract 가 grasp_s2r 로 판정했다가 인터페이스 차이로 거절한다. 새 family 가 필요하다. 같은 한 벌의 fj_rand_i01_final.pth 는 공차 바닥 뒤 열화(성공 0.61)라 쓰지 않는다.
- `right_m15_e800` — cp_r_m15 ep_800 (md5 4717e325 = policy_zoo cup_pick_r_manip_e800). 보상 iter_15. 학습 창 파지 0.967 · 들기 0.959 · 먼 출발 성공 0.827, 외란 20 N/kg 600 epoch 유지. 약점: 검지 0 · 5지 인벨롭 0(4지+손바닥), 고정 공차 평가 없음, 공차 0.082 까지만.
- `right_rh_aglt_i03` — 09.30 hold — 사용자 영상 판정: 엄지를 입구 안에 넣는 파지(결정론 probe 엄지 끝 컵 안 0.97). 오른팔은 right_rh_aglt_mirror_l5 를 쓴다. LOOP_STATE 추천 구간 e4400~4800 (성공 200 epoch 평균 최고 1.1~1.3/에피소드, 파지 0.73~0.79). 마지막 가중치 쓰지 말 것. 엄지 대향 0 — 사용자 판정: 모양 결함 알고 쓰는 첫 실기 후보. 계약 rh_aglt_contract.json 은 tools/build_rh_aglt_contract.py 로 런의 env.yaml 에서 만든다. 손 관측은 이름(프로필) 순 — pour_fj 와 달리 PhysX 순서 문제가 없다. 목표 = 리셋 때 컵 + (0, 0, 0.14)(학습 첫 목표 분포의 가운데).
- `right_rh_aglt_i09d` — 보상 iter_09 c1 · env 실기 반응(팔 지연 9~12 · 손 3~5 스텝 · 펌웨어 멈춤 · 편 손 하한)으로 학습. 결정론(5090, 64 env, 지연 켬) 컵 든 에피소드 0.91 · 낙하 스텝 0.016 · 목표 성공 0.84/에피소드(공차 0.039) — 공차 0.02(배포 달성 판정)에서는 0.05. 세 손가락 파지(약지 · 새끼 0). 주의: 손목 j6 한계(여유 < 5 %) 0.45 · j6 출력 |mu|>1 0.58.
- `right_rh_aglt_i10` — 보상 iter_10 · env 9a46c174(지연 키 0 — 지연 없이 학습). 학습 공차 0.0229 m. 결정론(64 env) 지연 0: 컵 든 에피소드 1.00, 목표 성공 2.0~2.3/에피소드(0.0229) · 0.8~0.9(0.02). 실측 지연을 넣으면 컵 든 에피소드 0.8. 엄지 · 검지 · 중지 · 새끼 접촉, 약지 안 닫음. 지연 적응판(②)은 학습 중.
- `right_rh_aglt_mirror_l5` — 우 env 결정론 probe(서버 GPU0, 64 env × 1800 스텝) 파지 0.806 · 들기 0.753 · 목표 성공 0.38 — 좌 원본(0.805 · 0.755)과 같다. 엄지 입구 안 0.0002(i03 은 0.97 — 사용자 영상 판정상 입구 안 엄지 파지라 좋은 오른팔 정책이 아니라고 grasping 세션이 알림). 알려진 결함: 엄지 대향 낮음(0.22), 네 손가락 손끝 파지(첫마디 0).
