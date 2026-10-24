# newton_07_r ep2000 (right 손 컵 홀더 놓기, 10.25 배포 후보 — 기본은 아직 n05)
- 학습: learn:place, server `~/rl_ws/hdgp/log/rl_games/open-rh/right/place/newton_07_r/nn/last_open-rh_r_place-lstm_ep_2000_rew_776.9702.pth`, ckpt md5 4afccc50cd87d7592689b2e036b42c77.
- 시작 = **이송 끝** 상태(start_bank_path 1023_grasp_bank/carry_end_rcg04_r.npz, carry_mode ready). 들기 끝에서 바로 시작하면 n05 와 반대 방향으로 분포가 어긋난다
  → 배포 계약에 ready_carry 가 있어야 하고(`--ready-carry` 없이 빌드하면 거부), 노드는 정책 전에 Mover 이송(v_max 0.0015 m/스텝 · qd_max 0.012 · carry_dz 0.085 · plan_down 0.04)을 먼저 돈다.
- 배포 계약: `python3 deploy/policy_control/tools/build_rh_aglt_contract.py --run deploy/policies/rh56f1/place/right_n07 --attach bundle --ready-carry 600`
- 래치 GRU: `latch/est_r.pt`(git 밖, md5 fdf46702d8765853e41c638a4bdd2d13) — n05 와 같은 파일.
- sim e2e(learn:e2e-aglt 1024_e2e_chain, gru · est · train): right_n070 %.
- 알려진 문제(10.25): 배포 Mover 는 에피소드마다 끝 자세 라이브러리가 비어 먼 IK 해를 고를 수 있다(실기 홀더 dry-run l 60/75 · r 70/75 도착, sim 75/75 · 74/75). 수정안 A/B 는 사용자 결정 대기.
