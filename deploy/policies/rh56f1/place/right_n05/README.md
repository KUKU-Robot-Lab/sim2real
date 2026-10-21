# newton_05_r ep3400 (right 손 컵 홀더 놓기, 10.22 배포 등록)
- 학습: learn:place, server `~/rl_ws/hdgp/log/rl_games/open-rh/right/place/newton_05_r/`, Newton(Isaac Lab 3), hdgp origin/main(rh_place_r 마지막 16f25c219). ckpt md5 b656b78e0120c7e493d8185821ca3c57.
- 시작 = 들기 끝 상태(시작 뱅크 bank_r_newton01_keep = rcg_03 들기 끝, 컵 3종 source240 · source200 · shaker). 에피소드 5 s, 60 Hz.
- sim(손마다 360 회): 기준 98.1 % · 완벽 90.6 %. **ev08(FP++ 없이 grasp_state 인계 자세 + 실측 오차, hdgp b846472af) 96.4 %** · 물리 오류 0.
- 배포 계약: `python3 deploy/policy_control/tools/build_rh_aglt_contract.py --run deploy/policies/rh56f1/place/right_n05 --attach bundle`
  - 관측 101 = 91(촉각 없음) + 물체 one-hot 3 + 래치 1 + 이웃 홀더 6(= 0, 학습 내내 0 — 래치 GRU 데이터 통계로 확인).
  - 붙인 자세 = 잡기 노드의 grasp_state 넘김 묶음(logs/inhand, 노드 bundle:=auto). 묶음이 없으면 reset 거부.
  - 래치 = 래치 컵(손바닥 FK ⊗ 래치 GRU 보정)이 자리 위 dz(0.04 · shaker 0.07) · 바닥 중심 xy 0.012 · 기울기 8° 를 3 틱 연속 → 유지.
    래치 뒤 손 = 편 손(팔 정책) → 15 틱 뒤 놓음 → 정책이 팔을 뺌(손 편 채), 손–컵 5 cm 또는 120 틱이면 끝.
- 래치 GRU: `latch/est_r.pt`(git 밖, md5 fdf46702d8765853e41c638a4bdd2d13) ← server `~/logs/place_newton/est/est_r.pt`. 같은 폴더 est_*_np1.pt 는 쓰지 않는다.
- 학습 목표 홀더 = sim 슬롯 1 → 원점 x 0.375 ~ 0.380 · y −0.223 ~ −0.037 m. 10.20 스캔으로는 실물 홀더 1 이 이 범위(x 는 1 cm 남짓 바깥).
- 운용(learn:place): 바깥쪽 이웃 칸에 컵이 있으면 63~84 % — 비워 둔다. 홀더 줄이 바깥으로 1 cm 넘게 어긋나면 79 %.
- 배포 점검(10.22 s2r:depoly): 오프라인 폐루프(배포 PlaceChain · 래치 GRU · 뱅크 bank_r_newton01 시작 60 개, 팔 · 손은 직전 목표를 그대로 따름 — 물리 없음)
  5 s 안 래치 = 실물 홀더 1(0.391, −0.061) 54/60 · 범위 안 홀더(0.378, −0.061) 58/60 · (0.378, −0.10) 56/60 · shaker 56/60. 손바닥 FK = 뱅크 손바닥(오차 0).
  fake 리허설(pick_place_f_right): 묶음 선택 · 홀더 범위 경고 · reset · start 까지 연결 확인. fake 잡기는 컵이 손을 따라오지 않아 들기 끝 자세가
  학습 밖(손바닥 y −0.53, 뱅크 컵 y ≥ −0.44)이라 놓기 성공 여부는 fake 로 못 본다(5 s 안에 자리 9 cm 앞에서 멈춤).
