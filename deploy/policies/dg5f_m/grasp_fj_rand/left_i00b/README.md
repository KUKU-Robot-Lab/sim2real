# dg5f_m/grasp_fj_rand/left_i00b — 왼팔 grasp_fj_rand 시드 e800
- 런 `fj_randL_i00b` e800, 보상 `reward_gen/grasp_fj_rand_left/iter_00`, gym id `open-short_l_grasp_fj_t2r_rand-lstm`(**s2r 동결 트랙** — 재생만 하고 그 트랙에 쓰지 않는다).
- 왼팔 cup_pick 런의 출발 가중치다: `dg5f_m/cup_pick/left_a00`(fj_abL_a00) · `dg5f_m/cup_pick/left_a01`(cp_l_a01).
- `params/` 는 이 런이 저장한 그대로. `reward_code_path` 는 로컬 절대경로라 다른 호스트에서는 그 줄을 고친다. 불러올 때 vendor `rl_games_sapg` 가 PYTHONPATH 에 있어야 한다.
- 인터페이스: obs 133 / critic 157 / action 26 = 팔 7 관절 증분 + 손 19 관절 절대, 가동 손 관절 13, 프로파일 `tesollo_left_short_tl`.
- `nn/seed_fj_randL_i00b_e800.json` = 사이드카. 10.06 옛 `deploy/policies/left_aglt` 한 벌에서 떼어 냈다.
