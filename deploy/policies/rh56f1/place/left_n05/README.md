# newton_05_l ep3400 (left 손 컵 홀더 놓기, 10.22 배포 등록)
- 학습: learn:place, server `~/rl_ws/hdgp/log/rl_games/open-rh/left/place/newton_05_l/`, Newton(Isaac Lab 3), hdgp origin/main(rh_place_r 마지막 16f25c219). ckpt md5 ec6c591d3ce4b48f43df7eb095adc90d.
- 시작 = 들기 끝 상태(시작 뱅크 bank_l_newton01_keep = rcg_03 들기 끝, 컵 3종 source240 · source200 · shaker). 에피소드 5 s, 60 Hz.
- sim(손마다 360 회): 기준 97.5 % · 완벽 91.7 %. **ev08(FP++ 없이 grasp_state 인계 자세 + 실측 오차, hdgp b846472af) 95.6 %** · 물리 오류 0.
- 배포 계약: `python3 deploy/policy_control/tools/build_rh_aglt_contract.py --run deploy/policies/rh56f1/place/left_n05 --attach bundle`
  - 관측 101 = 91(촉각 없음) + 물체 one-hot 3 + 래치 1 + 이웃 홀더 6(= 0, 학습 내내 0 — 래치 GRU 데이터 통계로 확인).
  - 붙인 자세 = 잡기 노드의 grasp_state 넘김 묶음(logs/inhand, 노드 bundle:=auto). 묶음이 없으면 reset 거부.
  - 래치 = 래치 컵(손바닥 FK ⊗ 래치 GRU 보정)이 자리 위 dz(0.04 · shaker 0.07) · 바닥 중심 xy 0.012 · 기울기 8° 를 3 틱 연속 → 유지.
    래치 뒤 손 = 편 손(팔 정책) → 15 틱 뒤 놓음 → 정책이 팔을 뺌(손 편 채), 손–컵 5 cm 또는 120 틱이면 끝.
- 래치 GRU: `latch/est_l.pt`(git 밖, md5 a3fde0a9d8d125e8c5af0c3cf8eca5d3) ← server `~/logs/place_newton/est/est_l.pt`. 같은 폴더 est_*_np1.pt 는 쓰지 않는다.
- 학습 목표 홀더 = sim 슬롯 1 → 원점 x 0.375 ~ 0.380 · y +0.037 ~ +0.223 m. 10.20 스캔으로는 실물 홀더 0 이 이 범위(x 는 1 cm 남짓 바깥).
- 운용(learn:place): 바깥쪽 이웃 칸에 컵이 있으면 63~84 % — 비워 둔다. 홀더 줄이 바깥으로 1 cm 넘게 어긋나면 79 %.
- 배포 점검(10.22 s2r:depoly): 오프라인 폐루프(배포 PlaceChain · 래치 GRU · 뱅크 bank_l_newton01 시작 60 개, 팔 · 손은 직전 목표를 그대로 따름 — 물리 없음)
  5 s 안 래치 = 실물 홀더 0(0.388, 0.136) shaker 52/60 · source240 50/60 · 범위 안 홀더(0.378, 0.10) shaker 52/60. 손바닥 FK = 뱅크 손바닥(오차 0).
