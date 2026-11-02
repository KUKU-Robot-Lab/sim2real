# newton_10_r ep1000 (right 손 컵 홀더 놓기 — ★11.02 oneshot 세팅 실기 연결, 사용자 지시 sim_e2e 경유)
- 학습: learn:place(10.30 place_lab), 5090 `~/rl_ws/our_source/rh_aglt_runs/eval/1030_place_fresh/params/newton_10_r/last_open-rh_r_place-lstm_ep_1000_rew_793.51764.pth`, ckpt md5 ab26d3736f64939eefe2c62cf352602d.
- 학습 설정이 n07 과 다른 점: 물체 1종 source240_real(object_id_obs false, 관측 one-hot 없음 → obs 92 = 91 + 래치 1) · 래치 dz 0.04 · dxy 0.012 ·
  **cup_obs_mode 'true'**(actor 컵 = 매 스텝 sim 참값, 지연 · 잡음 · 편향 0) · 뱅크가 홀더 자세를 덮음(occ_from_bank, bank_layout_shift 0.02).
- 배포 계약: `python3 deploy/policy_control/tools/build_rh_aglt_contract.py --run deploy/policies/rh56f1/place/right_n10_e1000 --checkpoint nn/last_open-rh_r_place-lstm_ep_1000_rew_793.51764.pth --attach bundle --ready-carry 600`
  - `cup_obs: live`(11.02 rh_place) — 실기는 참값이 없어 FP++ 생방송: 틱마다 FP++ 가 CUP_STALE_S 안이면 그 자세, 아니면 손바닥 FK ⊗ 넘김 묶음(10.21 "추정기 전환은 FP++ 끊김 때만").
    래치 입력도 같은 자세. 준비 이송(ready_carry)은 FK. 노드 상태 `cup_src`(live/fk 틱 수)로 실기에서 얼마나 FK 로 갔는지 본다.
  - 물체 기하: 빌더가 object_geometry = {source240_real: 기하(cup_scale 0.65)} 를 싣는다(10.30 place_lab · oneshot true_place 와 같은 채움). 에피소드 policy_object 는 source240_real.
  - 홀더 범위 = 뱅크 `bank/bank_right_train.npz`(git 밖, md5 a428d42c6a8822a120d0221883cce124)의 목표 홀더 자세 ± 0.02(env holder_ys 계산은 이런 런에서 틀린다 — oneshot true_place.apply_holder_range 와 같음).
- sim e2e(sim_e2e oneshot 10.31): 오른손 ep1000 · 왼손 ep3800 조합 191/256 = 74.6 %. 오른손 ep4000 은 네 시드 판정 대기(사용자 기준 "ADR 20 이상").
- 알려진 위험: (1) FP++ 지연 · 프레임 간 유지 · 홀더 위 손 가림은 학습이 본 적 없다 — attached 로 학습한 n07 보다 흔들릴 수 있다.
  (2) 왼손 래치 0.04 — 셰이커 잡는 자세의 손끝 높이를 실기에서 다시 재야 한다(place_lab). (3) Mover 끝 자세 빔(10.25)은 n07 과 같다.
