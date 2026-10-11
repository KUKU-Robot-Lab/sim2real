# 실기 지연: live policy(방안 1) vs sim 궤적 재생(방안 2) — 외부 참고 (2026-10-11)

사용자 질문(10.11): 방안 1 = 지금처럼 live policy 를 if-then 으로 연결(지연에 약함).
방안 2 = FP++ 로 물체 좌표를 받아 sim 에서 정책 체인을 굴려 성공 궤적을 저장하고 실기에서 천천히 재생(지연에 무감).

## 참고 문헌

### 방안 2 — sim 궤적 → 실기 open-loop 재생
- **Lessons from Learning to Spin "Pens"** — Wang, Yuan, Che, Qi, Ma, Malik, X. Wang. CoRL 2024 (PMLR v270).
  https://arxiv.org/abs/2407.18902 · https://corl-2024-dexpen.github.io/
  · sim 에서 특권 정보 oracle 정책(RL)으로 궤적 생성 → 실기에서 그 행동을 **open-loop 재생** → 성공한 것만 저장(물체당 15개, 총 45개)
    → 그 실기 궤적으로 proprioceptive 정책 미세조정(50개 미만).
  · 랜덤 출발에서는 sim 1000 env 중 최고 궤적을 골라 재생. 미세조정 정책이 재생 기준선보다 22~40 %p 높음.
  · 정책 입력: 관절 30 스텝 + 직전 목표 30 스텝(행동 이력).
- **Real-is-Sim: Bridging the Sim-to-Real Gap with a Dynamic Digital Twin** — 2025. https://arxiv.org/abs/2504.03597 · https://real-is-sim.github.io/
  · 정책은 항상 sim 로봇에만 행동, 실기는 sim 로봇 관절값(q, q' 아님)을 따라감. 카메라로 sim 물체 상태를 60 Hz 보정.
  · sim2real 차이를 정책이 아니라 트윈 동기화가 떠안음. 실기가 sim 을 살짝 뒤따르므로 짧은 미래 미리보기가 생김.
- **RialTo: Reconciling Reality through Simulation (Real-to-Sim-to-Real)** — Torne et al. RSS 2024. https://arxiv.org/abs/2403.03949
  · 실제 장면을 스캔해 디지털 트윈 → 그 안에서 RL 로 모방 정책을 강건화(강건성 67 %+ 향상).

### 방안 1 — 지연 하 closed-loop
- **Reinforcement Learning with Random Delays (DCAC)** — Bouteiller, Ramstedt, Beltrame, Pal, Binas. ICLR 2021. https://arxiv.org/abs/2010.02966 · 코드 https://github.com/rmst/rlrd
  · 지연 환경에서 Markov 성질을 지키려면 관측에 **최근 K 개의 보낸 행동**(K = 관측+행동 최대 지연)을 붙여야 한다. 지연 값도 상태에 넣으면 더 좋다.
- **Real-Time Execution of Action Chunking Flow Policies (RTC)** — Black et al. (Physical Intelligence). NeurIPS 2025. https://arxiv.org/abs/2506.07339
  · 현재 chunk 실행 중 다음 chunk 생성, 실행이 확정된 행동은 고정하고 나머지를 inpainting. 300 ms+ 추론 지연에서도 동적 작업 성공.
  · 후속: Training-Time Action Conditioning for Efficient Real-Time Chunking https://arxiv.org/abs/2512.05964
- **DextrAH-G: Pixels-to-Action Dexterous Arm-Hand Grasping with Geometric Fabrics** — Lum et al. CoRL 2024. https://arxiv.org/abs/2407.02274
  · RL 정책은 상위 명령(PCA 11차원)만, 관절 한계·충돌은 하위 geometric fabric 이 담당하는 계층 구조. (정책 Hz 는 미확인)
- (확인 안 한 고전 참고) Tan et al. 2018 "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots"(지연 모델링),
  Hwangbo et al. 2019 "Learning agile and dynamic motor skills"(관절·행동 이력 관측).

## 이 프로젝트 적용 메모(10.11 분석, 가설 — 미실험)
- rh_aglt actor 관측은 `arm_q_target`(최신 q*)·`prev_action`(1 스텝)뿐 → PD 에 실제 들어가는 q*_{t−d} 와 d 를 모름(Markov 깨짐).
  critic extra 에도 `arm_q_applied`·지연 값 없음. 필요한 이력 K ≈ 10 스텝(j6 8.6 스텝).
- 팔 행동 법칙 증분+EMA(α 0.1 @60 Hz) 자체가 시정수 ~160 ms → 지연·PD τ 와 겹쳐 적분기 넘침 → 관절 한계 흘러감·느린 도달(10.07 g52 붕괴, ② 느린 도달과 일치).
- 동료(pouring;rh56f1, 10.11) sim 수치: 지연 없이 학습한 정책 지연 4 스텝 49.4 %·8 스텝 16.3 %, 지연 DR 2~9 로 학습한 궤적+잔차 71.2 %(궤적만 47.6 %).
- 진단 순서 제안: D0 = 기존 g5362b 를 지연 0/실측/2배에서 결정론 평가, D1 = actor q* 이력 10 스텝 + critic arm_q_applied·지연 값.
- 방안 2 기존 부품: `scripts/shadow_replay_core.py`, `deploy/policy_control/tools/replay_to_pd.py`(팔만, rate_scale ≤ 1),
  `scripts/ops/bake_preset_trajectory.py`. 빠진 것: 손 채널 재생, FP++ 스냅샷→sim 초기화+정책 체인 러너, 도달 관절 궤적 기록 키.
