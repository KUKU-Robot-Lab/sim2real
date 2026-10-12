# aglt_l_rg3_ep1900 (10.12 grasping 최종, 사용자 성공 확정)
- 3물체 한 정책: object_names = (source240_real, source200_real, shaker_c_real) — object id one-hot 순서 id0/id1/id2
- obs: actor 99 = 96 + one-hot 3, critic 155 = 152 + 3(지연 관측 포함). per_env_geometry · object_id_obs · critic_delay_obs · hand_adm_enable · fpp_enable true
- 지연: cmd_delay_adr_mix(env마다 배율 U(0, 현재 상한)), ADR 30 = 실측 지연 1.5배까지 학습
- 성공: 컵 위 10 cm 키포인트 3 cm 안 2 s(기울기 포함) · 보상 reward_gen/rh_aglt_d6/iter_02
- 평가(실기 지연 1.0 결정론 256 env): 아래 sim2real 메시지 참조
