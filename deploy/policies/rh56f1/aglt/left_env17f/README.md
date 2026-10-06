# aglt_l_env17f_ep200
- 원 정책: 우 aglt_r_env17 ep1000 의 거울(aglt_l_env17mir_ep1000, mirror_rh_aglt_ckpt.py selftest 0) → 왼팔 env 에서 aglt_l_env17f 200 epoch 이어 학습, ep200 선택.
- 학습 env: 보상 rh_aglt_l/iter_17(우 iter_17 과 같은 보상) · cyl60 SDF · FP++ 실측 지각 · 파지 후 프레임 시각 부착 · 붓기 하중 0.7 · 명령 지연 0 · ADR 20.
- 배포 조건: cyl60g · env17 과 같다 — sim2real 796b074(부착)·03cd917(관절 힘 파지 신호)·104f2b1(깊이 광선 보정). 계약은 left_env17mir 와 체크포인트 · 해시 외 차이 0.
- 결정론(5090, 64 env × 1800 스텝, tol 0.02, ADR 0, 하중 끔): 에피소드당 성공 4.79 / 5(5개 다 채움 0.93), 컵 기울기 중앙 5.6° · p90 12.3°, 쥔 높이 0.23 × 입구.
  거울 그대로 4.40 · 8.1° → 개선. 우 env17 ep1000 4.91 · 4.9°. 같은 런 ep400 4.71 · ep600 4.75(7.8°).
- md5 927120f8738cb2596eac382507ac9199
