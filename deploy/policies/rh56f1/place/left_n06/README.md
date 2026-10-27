# newton_06_l ep5000 (left 손 컵 홀더 놓기 — 10.27 사용자 결정 1 로 배포 놓기: pick_place_f_left · cocktail_1 · 2)
- 학습: learn:place, server `~/rl_ws/hdgp/log/rl_games/open-rh/left/place/newton_06_l/nn/last_open-rh_l_place-lstm_ep_5000_rew_799.7323.pth`, ckpt md5 2c8b54650f68521fc45462e56d287366.
- 시작 = **이송 끝** 상태(start_bank_path 1023_grasp_bank/carry_end_rcg04_l.npz, carry_mode ready). 들기 끝에서 바로 시작하면 n05 와 반대 방향으로 분포가 어긋난다
  → 배포 계약에 ready_carry 가 있어야 하고(`--ready-carry` 없이 빌드하면 거부), 노드는 정책 전에 Mover 이송(v_max 0.0015 m/스텝 · qd_max 0.012 · carry_dz 0.085 · plan_down 0.04)을 먼저 돈다.
- 배포 계약: `python3 deploy/policy_control/tools/build_rh_aglt_contract.py --run deploy/policies/rh56f1/place/left_n06 --attach bundle --ready-carry 600`
- 래치 GRU: `latch/est_l.pt`(git 밖, md5 a3fde0a9d8d125e8c5af0c3cf8eca5d3) — n05 와 같은 파일.
- sim e2e(learn:e2e-aglt 1024_e2e_chain, gru · est · train): left_n060 %.
- 알려진 위험(10.25): Mover 가 먼 IK 해(관절 이동 1.0~1.2 rad)를 고르면 정책 디코더 증분 상한(0.005 rad/스텝) 때문에 600 틱 안에 못 따라간다
  (옛 홀더 배치 배포 dry-run 도착 l 60/75 · r 70/75, 같은 행 sim 74~75/75 — 차이는 IK 무작위 시드 값이다. sim 도 끝 자세 라이브러리가 빈 채로 계획한다).
  10.25 사용자 결정(learn:e2e-aglt 경유): 해 규칙 margin · 상한 끔 · 배포 Mover 변경 없음 — near · 상한 거르기는 sim e2e 차이 ±4 %p 안.
  실기에서는 놓기 노드 '준비 자세 이송 끝' 로그의 carry_ok · err_mm 를 본다.
