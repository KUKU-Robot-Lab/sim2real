# dg5f_m/cup_pick/left_a00 — 왼팔 컵 접근 e3400
- 런 `fj_abL_a00` e3400, 보상 `reward_gen/cup_pick_l_approach/iter_00`. 학습 당시 gym id `open-short_l_grasp_fj_ab-lstm`(지금 `open-short_l_cup_pick-lstm` 과 같은 cfg, 이름 변경 전).
- 출발 가중치: `fj_randL_i00b` e800(= `dg5f_m/grasp_fj_rand/left_i00b`). 다음 단계는 `dg5f_m/cup_pick/left_a01`(런 cp_l_a01, 접근 + C자 유지, joint 계약 있음).
- `params/` 는 이 런이 저장한 그대로. play.py 는 이 `env.yaml` 을 복원하므로 CLI 오버라이드가 덮인다. `reward_code_path` 는 로컬 절대경로(`/home/user/rl_ws/hdgp/...`)라 다른 호스트에서는 그 줄을 고친다.
- 불러올 때 vendor `rl_games_sapg` 가 PYTHONPATH 에 있어야 한다(없으면 `KeyError: 'model'`). agent: PPO-LSTM(units 1024 · horizon 16 · minibatch 16384 · seq_length 16).
- 인터페이스: obs 133 / critic 157 / action 26 = 팔 7 관절 증분 + 손 19 관절 절대(시너지 아님), 가동 손 관절 13, 프로파일 `tesollo_left_short_tl`.
- `nn/cup_pick_l_approach_e3400.json` = 사이드카(관절 순서 · 인터페이스). 10.06 옛 `deploy/policies/left_aglt` 한 벌(세 체크포인트)을 태그별 폴더로 나눴다.
