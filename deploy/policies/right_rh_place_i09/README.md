# place_r_i09_ep4000 (오른손 컵 홀더 놓기)
- 학습: server GPU0 place_r_i09(보상 reward_gen/rh_place_r/iter_09), ep4000 선택. 좌 place_l_i01 의 출발점.
- 결정론 평가: 완벽 0.907 · 부분 0.067 · 실패 0.026(128 env 실물 배치). 실측 오차 넣은 평가(place_det_i09 trace_real_err, 2750 에피소드): 완벽 0.825 · 부분 0.150 · 실패 0.025.
- params/ 는 학습 런 그대로. profile rh56f1_right, decimation 2, 에피소드 15 s, target_holders (1, 2).
- 주의: fpp_enable·fpp_attach_enable 모두 false — FP++ 지연·잡음 없이 학습됨.
- md5 4ee40f18d625b021c1ec795916dbf659
