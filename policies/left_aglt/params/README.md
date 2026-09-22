# 왼팔 정책별 런 설정 (09.22 보강)

각 폴더 = 같은 이름의 `.pth` 를 만든 런이 저장한 `params/` 그대로. play.py 는 이 `env.yaml` 을 복원해
CLI 오버라이드를 덮으므로, 재생·재개할 때 이 파일이 정책과 한 벌이다.

| 폴더 | 런 | gym id | 보상 |
|---|---|---|---|
| `cup_pick_l_approach_e3400` | `fj_abL_a00` | `open-short_l_grasp_fj_ab-lstm` (현 `open-short_l_cup_pick-lstm` 과 같은 cfg, 이름 변경 전) | `reward_gen/cup_pick_l_approach/iter_00` |
| `cup_pick_l_approach_hold_e4280` | `cp_l_a01` | `open-short_l_cup_pick-lstm` | `reward_gen/cup_pick_l_approach/iter_01` |
| `seed_fj_randL_i00b_e800` | `fj_randL_i00b` | `open-short_l_grasp_fj_t2r_rand-lstm` (**s2r 동결 트랙의 gym id**) | `reward_gen/grasp_fj_rand_left/iter_00` |

## 쓸 때 주의

- `reward_code_path` 는 **이 호스트의 절대경로**(`/home/user/rl_ws/hdgp/...`)로 적혀 있고, 세 경로 모두 현재 존재한다(09.22 확인).
  다른 호스트로 옮기면 그 줄을 고쳐야 한다 — 안 고치면 play 가 `FileNotFoundError: reward_code_path` 로 죽는다.
- 불러올 때 vendor `rl_games_sapg` 가 PYTHONPATH 에 있어야 한다(없으면 `KeyError: 'model'`).
- agent 는 셋 다 PPO-LSTM(units 1024 · horizon 16 · minibatch 16384 · seq_length 16 · zero_rnn_on_done).
- env 는 셋 다 컵 소환 x 0.10–0.40 · y 0.00–0.30 uniform · episode 15 s · num_envs 4096 · tol 0.1125 · 가까운 출발 끔.
- `seed_fj_randL_i00b_e800` 의 gym id 는 s2r 동결 트랙(`grasp_fj_t2r_rand`)이다. **재생만** 하고 그 트랙에 쓰지 않는다.

인터페이스(셋 공통): obs 133 / critic 157 / action 26 = 팔 7 관절 증분 + 손 19 관절 절대(시너지 아님).
가동 손 관절 13 개, 잠긴 것 `thumb_2`·`index/middle/ring/pinky_1`·`pinky_2`, `thumb_1` 은 자산에서 용접.
프로파일 `tesollo_left_short_tl`, `hand_side="l"`.
