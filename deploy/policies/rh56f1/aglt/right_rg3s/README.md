# aglt_r_rg3s_ep4700 (10.13 grasping 배포 후보)
- 3물체 한 정책: object_names = (source240_real, source200_real, shaker_c_real) — one-hot 순서 id0/id1/id2
- obs: actor 94 = 91(촉각 없음, actor_tactile_obs=False) + one-hot 3. critic 은 촉각 · 지연 관측 포함
- 손: hand_block 켬(막힘 쥠 멈춤 · 쥔 판정, firmware_stop 끔, lead cap 0.2 rad) · 어드민턴스 힘 DR(배율 0.6~1.6, 오프셋 ±40 g)
- 쥔 판정 유지: grasp_hold_rule=block_follow_open(배포 때 켬 — 학습 덤프에는 없음, policy.yaml note 참고)
- 팔: 오프셋 DR(j2 +50 ms · j5 +100 ms 지연, 중력보상 배율 0.8~1.05, 상수 토크 바이어스, 쿨롱 마찰) — sim 전용
- 지연 ADR 30(실측 1.5배까지) · FP++ 지각 모델(가림 포함) · 보상 reward_gen/rh_aglt_d6/iter_03
- 체크포인트: last_open-rh_r_aglt-lstm_ep_4700_rew_527.14453.pth (md5 896130cb145d8de3c1cc0d4f3fd36400)
- 결정론 평가(server, drfull 최대 난이도, 256 env, grasp-hold 켬): D6(목표 2 cm 안 2 s) 0.74 · RIGID(들고 2 s 손안이동 5 mm 이하) 0.82 · 못쥠 0.10 · 근처못감 0.03 · 유지실패 0.12 · 들림 최대높이 중앙 19.0 cm
