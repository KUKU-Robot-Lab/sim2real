# place_l_i01_ep800 (왼손 컵 홀더 놓기)
- 원 정책: 우 place_r_i09 ep4000 의 거울(place_l_mir_i09_ep4000) → server GPU0 place_l_i01(보상 reward_gen/rh_place_l/iter_01) 이어 학습, ep800 선택.
- 결정론 평가(server GPU0, 128 env × 1800 스텝, 홀더 고정 배치 holder_y_shift_fixed 0·DR 축소): 에피소드 2511, 완벽 0.851 · 부분 0.066 · 실패 0.084. 목표 y 구간 -0.06~0 완벽 0.85·홀더 안 0.911, 0~+0.20 완벽 0.851·0.92.
  같은 조건 거울 원본(학습 전) 0.306.
- params/ 는 학습 런 그대로(평가 때 바꾼 홀더 DR 값 아님). profile rh56f1_left, decimation 2, 에피소드 15 s.
- 주의: fpp_enable·fpp_attach_enable 모두 false — FP++ 지연·잡음 없이 학습됨(aglt cyl60g 와 다름).
- 코드: hdgp 51013e96(rh_place_l env, PlaceEnvMixin). md5 22c186e46518fc955b5be8f6299b0a6a
