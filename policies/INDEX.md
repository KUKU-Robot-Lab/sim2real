# policies — 쓸 정책 목록

`policy_control/tools/policies.py --write-index` 가 만든다. 손으로 고치지 않는다 — 고칠 것은 각 `policy.yaml` 이다.

| id | status | task | side | checkpoint | 계약 | 점검 |
|---|---|---|---|---|---|---|
| `grasp_fj_rand_i01` | candidate | open-short_r_grasp_fj_t2r_rand-lstm | right | fj_rand_i01_best_ep5000.pth | - | ok |
| `pour_i18` | hold | open-short_b_pour_fab | both | last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth | pour_contract.json | ok |

## status

- `candidate` — 받아만 뒀다 — 계약·체인 검증 전
- `verified` — 계약이 서고 fake 체인을 통과했다 — 실기 승인 전
- `deployed` — 실기에서 승인받아 돌린 적이 있다
- `hold` — 쓰지 않는다 — 이유는 note 에

## note

- `grasp_fj_rand_i01` — 계약 없음 — 빌더가 아직 없다. obs 133 / act 26 (팔 7 관절 증분 + 손 19 관절 절대, 시너지·fabric 디코더 아님), 자산 tesollo_right_short_tl (thumb_1 용접). grasp_s2r 의 설정 키를 물려받아 그 family 로 판정되지만 인터페이스가 달라 build_deploy_contract 가 거절한다. 새 family(obs 133 세그먼트 + 관절공간 디코더)를 만들어야 verified 로 올릴 수 있다. 학습 커밋은 test_history.md 가 없어 미상.
- `pour_i18` — 계약·재현은 통과: actor 재현 2.1e-6, 디코더 palm/hand 목표 < 1e-5 (test_pour_registered_run.py). hold 인 이유는 ckpt_gate REJECT 2건 — 소스컵 최대 기울기 155.1 deg (한계 120), 소스컵이 중앙선을 넘어간 env 64/64. 실기에서는 pour_guard_node 가 같은 두 조건으로 episode/abort 를 건다. 사용자 영상 판정 대기. 다음 후보는 t2r_i19 (학습 중).
