# rcg_04_nr — 실험 기록

## 코드 스냅샷 (학습 시작 시점)
- **라벨**: rcg_04_nr
- **Date**: 2026-10-20 16:49
- **Task**: open-rh_r_aglt-lstm
- **Commit**: `87b96e6d` — docs(skill_chaining): 접근(AP) -> 파지(rcg) 연결 문헌 4편(Sequential Dexterity, T-STAR, ManipGen, DextrAH-G) 원문·코드 저장소·확인 수치 정리, PDF 는 폴더 LFS
- **로봇 자산**: `/home/oem/rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.usd`
- **전체 설정**: `params/env.yaml` (seed·num_envs·모든 weight) · `params/agent.yaml`

### Uncommitted 변경
(uncommitted 변경 없음)

### Note (가설)
RCG (newton) = ready_carry 접근 → 정책 · 2판 자산 rh56f1_v2_right · 보상 neo iter_03 · 팔 들림 래치 0.10 m · 출발 approach_r(mu +0, 예열 50) · ADR 20 · tol 0.10 · 그 밖 neo03 그대로 · 물리만 x(PhysX)/n(Newton) 비교

## 분석 (rl-diag) — 학습 후 `/rl-diag /home/oem/rl_ws/hdgp/log/rl_games/open-rh/right/aglt/rcg_04_nr` 가 채움
<!-- rl-diag-analysis -->
