# policies — 쓸 정책 목록

`deploy/policy_control/tools/policies.py --write-index` 가 만든다. 손으로 고치지 않는다 — 고칠 것은 각 `policy.yaml` 이다.

| id | status | task | side | checkpoint | 계약 | 점검 |
|---|---|---|---|---|---|---|
| `both_pour_i18` | hold | open-short_b_pour_fab | both | open-short_b_pour_fab.pth | - | ok |
| `left_aglt` | candidate | open-short_l_cup_pick-lstm | left | cup_pick_l_approach_hold_e4280.pth | - | ok |
| `right_aglt` | candidate | open-short_r_grasp_fj_t2r_rand-lstm | right | fj_rand_i01_best_ep5000.pth | - | ok |

## status

- `candidate` — 받아만 뒀다 — 계약·체인 검증 전
- `verified` — 계약이 서고 fake 체인을 통과했다 — 실기 승인 전
- `deployed` — 실기에서 승인받아 돌린 적이 있다
- `hold` — 쓰지 않는다 — 이유는 note 에

## note

- `both_pour_i18` — ★가중치가 계약과 다르다: 여기 nn/open-short_b_pour_fab.pth 는 md5 e473c709…, 검증된 pour_contract.json 은 f50b09a6…(logs/policy/pour_i18/nn/last_..._ep_2500_rew_36408.105.pth)로 만들어졌다. 크기도 4,667,063 vs 4,669,209 로 다르다 — 같은 런의 다른 내보내기다. 이 파일로 쓰려면 계약을 다시 만들어야 한다. hold 인 본래 이유: ckpt_gate 탈락 3건 — 소스컵 최대 기울기 155.1°(한계 120) · 이탈 0.344 m(한계 0.25) · 중앙선 교차 64/64 env. 실기에서는 pour_guard_node 가 같은 두 조건으로 episode/abort 를 건다. 검증된 쪽(계약·trace·actor 재현 2.1e-6)은 logs/policy/pour_i18/ 에 그대로 있다. 다음 후보는 t2r_i19.
- `left_aglt` — 세 체크포인트 한 벌(접근 e3400 · 접근+C자 유지 e4280 · 입력 시드 i00b). 후보는 가장 나중 단계인 e4280 으로 적어 뒀다 — 다른 것을 쓰려면 이 checkpoint: 한 줄만 바꾸면 된다. 인터페이스는 오른팔과 같다(obs 133 / action 26, 팔 7 증분 + 손 19 절대, 프로파일 tesollo_left_short_tl). 계약 없음 — right_aglt 와 같은 이유(새 family 필요). seed_fj_randL_i00b_e800 의 gym id 는 s2r 동결 트랙이다 — 재생만 하고 그 트랙에 쓰지 않는다.
- `right_aglt` — s2r 후보. 먼 출발 성공 0.85 · 공차 0.019 · 컵 기울기 4.3°. sha256 앞 16자리 e76f9c6079663b61. 계약 없음 — obs 133 / action 26(팔 7 관절 증분 + 손 19 절대)은 기존 세 family 어디에도 안 맞는다. build_deploy_contract 가 grasp_s2r 로 판정했다가 인터페이스 차이로 거절한다. 새 family 가 필요하다. 같은 한 벌의 fj_rand_i01_final.pth 는 공차 바닥 뒤 열화(성공 0.61)라 쓰지 않는다.
