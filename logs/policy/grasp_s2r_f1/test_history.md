# f1_fresh — 실험 기록

## 코드 스냅샷 (학습 시작 시점)
- **라벨**: f1_fresh
- **Date**: 2026-09-11 12:34
- **Task**: open-short_r_grasp_s2r-lstm
- **Commit**: `ccfe4a5a` — fix(grasp_fj): hand_curl 이 엄지 대향 해제를 보상하고 있었다 — 부동 관절 고정 + curl 재정의
- **로봇 자산**: `/home/oem/rl_ws/hdgp/assets/robot/openarm_dg5f-m-short_bi_rl/openarm_dg5f-m-short_bi_rl.usd`
- **전체 설정**: `params/env.yaml` (seed·num_envs·모든 weight) · `params/agent.yaml`

### Uncommitted 변경
(uncommitted 변경 없음)

### Note (가설)
F1 리미터 해제(정책→fabric 다이렉트) + E1 보상 유지. 리미터 0.0/0.0, 질량DR(0.5,2.5), 외란W1. arm_qd 로 실기 브리지 상한(1.0 rad/s) 검증. 기준선 E1 success 0.0013·stable 0.123

## 분석 (rl-diag) — 학습 후 `/rl-diag /home/oem/rl_ws/hdgp/log/rl_games/open-short/right/grasp-s2r/f1_fresh` 가 채움
<!-- rl-diag-analysis -->
