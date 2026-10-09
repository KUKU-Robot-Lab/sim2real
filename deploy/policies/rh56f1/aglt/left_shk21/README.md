# 왼팔 rh_aglt 쉐이커(shaker_c) — 10.09 OBJ_RETRAIN 통과
- 런: aglt_l_shk21_ep1800 · ckpt nn/last_open-rh_l_aglt-lstm_ep_1800_rew_2480.1245.pth · md5 4dc97d39adfe192cbffbbb5a4ae31ace
- 물체: shaker_c(높이 130 · 원점 = 높이 가운데)
- 평가: 결정론 64 env × 1800 · ADR 0: 4.75/5 · 넘어뜨림 0.02(arm4090)
- 학습: 보상 iter_21 · 어드민턴스 손(DR 0) · 팔/손 지연 0 · FP++ 끔 · tol 0.02 · 웜스타트 aglt_l_g5 ep1200 — 정본 hdgp reward_gen/rh_aglt_r/OBJ_RETRAIN.md
- 실기 FP++ 물체: sim2real config/objects.d(scripts/ops/fpp_object.py) · 활성 config/fpp_active.yaml
- params/ = 학습 env.yaml · agent.yaml 그대로. nn/*.pth 는 git 에 없다(arm4090 our_source/policy 에서 복사)
