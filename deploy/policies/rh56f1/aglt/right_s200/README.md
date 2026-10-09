# 오른팔 rh_aglt source200 병 전용 — 10.09 OBJ_RETRAIN 통과
- 런: aglt_r_s200_ep800 · ckpt nn/last_open-rh_r_aglt-lstm_ep_800_rew_1920.0023.pth · md5 f6bc4c5834a8a2e70f25763430c2dd5b
- 물체: source200_pla 단독
- 평가: 결정론 ADR 0: 4.92/5(arm4090) · 5090 재확인 4.84 · 넘어뜨림 0.01
- 학습: 보상 iter_21 · 어드민턴스 손(DR 0) · 팔/손 지연 0 · FP++ 끔 · tol 0.02 · 웜스타트 aglt_r_g5 ep1200 — 정본 hdgp reward_gen/rh_aglt_r/OBJ_RETRAIN.md
- 실기 FP++ 물체: sim2real config/objects.d(scripts/ops/fpp_object.py) · 활성 config/fpp_active.yaml
- params/ = 학습 env.yaml · agent.yaml 그대로. nn/*.pth 는 git 에 없다(arm4090 our_source/policy 에서 복사)
