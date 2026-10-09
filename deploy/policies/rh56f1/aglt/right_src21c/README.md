# 오른팔 rh_aglt source 병 두 개 겸용(240 · 200) — 10.09 T2R Grasping OBJ_RETRAIN 통과
- 런: aglt_r_src21c_ep1000 · ckpt nn/last_open-rh_r_aglt-lstm_ep_1000_rew_1774.0562.pth · md5 a067df3a6fdfdeb5e4663a55096d74ed
- 물체: source240_pla + source200_pla(env 마다 반반)
- 평가: 결정론 64 env × 1800 · tol 0.02 · ADR 0: 240 병 4.58/5 · 200 병 4.96/5(vision-3090), 5090 재확인 4.66 · 4.66 · 넘어뜨림 ≤ 0.02
- 학습: 보상 iter_21 · 어드민턴스 손(DR 0) · 팔/손 지연 0 · FP++ 끔 · tol 0.02 · 웜스타트 aglt_r_src21 ep400(두 병 섞기) 이어서, ADR 1 부터 — 정본 hdgp reward_gen/rh_aglt_r/OBJ_RETRAIN.md
- 실기 FP++ 물체: sim2real config/objects.d(scripts/ops/fpp_object.py) · 활성 config/fpp_active.yaml
- params/ = 학습 env.yaml · agent.yaml 그대로. nn/*.pth 는 git 에 없다(arm4090 our_source/policy 에서 복사)
