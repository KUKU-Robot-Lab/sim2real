# 방안 1 공통 지연 학습 규격 — S2R-DELAY-1011 (2026-10-11)

작성: sim2real 세션. 대상: grasping · place · locking 세션. pouring 은 제외다(10.11 사용자 결정, 아래 트랙별 메모).
근거: `docs/reference/S2R_LATENCY_VS_REPLAY_REFS_2026-10-11.md`, 보고서 `~/rl_ws/report/plan1_latency_eli5.html`.

**한 줄:** 지연은 env 설정값으로 넣고, critic 정답은 critic 입력 끝에 cfg 플래그로 붙인 뒤, 지금 성공한 체크포인트에서 이어서 학습한다. actor 입력과 배포 계약은 그대로 둔다.

**사용자 결정(10.11):** actor 관측에 행동 이력(K≈10)은 넣지 않는다. 4개 트랙 모두 actor 가 LSTM(seq_length 16 ≈ 267 ms)이고 매 스텝 prev_action 을 보므로 LSTM 이 이력을 대신한다. 넣는 것은 아래 A · B · C 세 가지다.

---

## A. 팔 — 관절별 실측 지연

| 키 | 값 |
|---|---|
| `arm_cmd_delay_joint_steps` | `(3.5, 3.2, 3.3, 2.3, 4.1, 8.6, 7.3)` — `ARM_DELAY_JOINT_STEPS_MEASURED`(rh_aglt_r/rh_aglt_env_cfg.py:51), 60 Hz 정책 스텝 |
| `arm_cmd_delay_joint_jitter` | `1` — 에피소드·관절마다 round(nominal) ± 1 스텝 |

- 값의 뜻: 실측 applied → 측정 관절각 시프트 중앙값(arm4090 10.03 home_return 양팔 bag) + pd 경로 8 ms − sim PD 가 이미 내는 지연(kd/kp). 즉 **sim PD 위에 더 얹을 순수 지연**이다. 양팔에 같은 값을 쓴다.
- 전제: sim 팔 PD 게인 = 벤더 값(kp 70/70/70/60/10/10/10, kd 2.75/2.5/2.5/2.0/0.7/0.6/0.5). 게인이 다르면 이 값은 틀린다.
- **금지:** 균일 `arm_cmd_delay_steps (9, 12)`. sim PD 지연을 두 번 센다(memory arm-delay-double-counted-1007).
- 커리큘럼: `cmd_delay_adr=True`, 배율 0 → 1.0(실측) → 1.5(여유, 제안값). 합격 판정은 배율 1.0 에서 한다. 이미 ADR 이 끝난 체크포인트에서 이어서 학습하면 `cmd_delay_adr_from` 으로 지연 배율 구간을 따로 잡는다. 지연선 길이는 배율 끝값에 맞춰 자동으로 늘어난다(rh_aglt_env.py:350-353).

## B. 손 · 눈 — 실측 지연

**손**

| 키 | 값 |
|---|---|
| `hand_cmd_delay_steps` | `(4, 6)` — 67~100 ms |

- 실측: 명령 → 첫 움직임 60~80 ms(RH56F1_TUNING_0930 §2), 10.09 실기 80~90 ms. sftM2 와 같은 값이다.
- ADR 지연 배율은 손에도 같이 걸린다(`_adr_scale_delay`).
- 손 모드는 배포 계약과 같게 둔다. aglt 계약은 `"hand_command": "admittance"` 이므로 `hand_adm_enable=True`. place i09 · i01 계약에는 hand_command 가 없으므로 position 이다.

**눈(FP++)** — 배포에서 컵 pose 를 FP++ 로 받는 구간에만 해당한다.

| 키 | 값 |
|---|---|
| `fpp_enable` | `True`. 나머지는 기본값 그대로: 지연 15~24 스텝(250~400 ms) ±1, 갱신 주기 5~7 스텝(≈10 Hz), 프레임 빠짐 0~10 %, 위치 잡음 σ 0.5~3 mm |
| `fpp_ramp_steps` | 이어서 학습할 때 > 0. 지연·간격을 0 → 실측으로 서서히 올린다 |
| `fpp_attach_enable` | `True`(10.11 사용자 결정: FP++ 288 ms · 10 Hz 로는 실시간 제어를 못 따라가므로, 잡은 뒤에는 손바닥 FK 로 컵을 안다). 배포 rh_aglt_node(= pour_fj_node family rh_aglt)에 이미 같은 규칙의 `CupAttach` 가 있고 기본으로 켜져 있다(`cup_attach` True). 규칙: 파지 시작 100 ms 뒤에 찍힌 FP++ 프레임 시각의 손바닥 FK 로 붙이고, 15 스텝 끊기면 뗀다. 실기 파지 판정은 손끝 촉각 > 1 N 또는 관절 힘 > 300 g(`attach_signal` "either")이다. 300 g 는 임시값이라 실측해서 정해야 한다 |

- 실측: 288 ms(p10 272 · p90 306 · 최대 367), 10.5 Hz.
- 원칙: **sim 컵 관측 경로 = 배포 컵 관측 경로.** place 는 `cup_obs_mode "attached"`(손바닥 FK)라서 FP++ 지연이 해당하지 않는다.

## C. critic 정답 (critic 입력 끝에 추가)

- 이유: critic 은 RNN 없는 MLP(central value)다. 그래서 지연 중에 어떤 명령이 PD 에 들어가는지 기억할 수 없다. critic 은 배포하지 않으므로 배포 계약에 영향이 없다.
- 플래그: `critic_delay_obs: bool = False`. 기본은 꺼짐이므로 옛 체크포인트의 play · resume 은 그대로 된다.
- 켜면 critic extra 끝에 아래 순서로 붙인다(24 칸, 팔 7 · 손 6 기준).

| 이름 | 폭 | 값 |
|---|---|---|
| `arm_q_applied` | 7 | 지금 팔 PD 에 들어가는 목표(지연선 출력) |
| `arm_delay` | 7 | 관절별 현재 지연 스텝 / 15 |
| `hand_applied` | 6 | 지금 손 PD · 어드미턴스에 들어가는 목표 |
| `hand_delay` | 1 | 손 지연 스텝 / 10 |
| `cup_obs_err` | 3 | actor 가 본 컵 위치 − 실제 위치(같은 좌표계). FP++ · 잡음이 꺼져 있으면 0 |

- 체크포인트: critic 입력만 늘리고 새 칸 가중치는 0 으로 둔다. 그러면 이어서 학습을 시작하는 순간 가치 출력이 원본과 같다. 도구는 `hdgp/scripts/tools/widen_obs_ckpt.py` 에 `--critic-only` 옵션을 추가해서 쓴다. 지금은 actor 와 critic 을 같이 늘린다.

## 바꾸지 않는 것

- actor 관측의 폭과 순서. 그래서 배포 obs_dim · 디코더 · 계약도 그대로다.
- 행동 법칙(증분 + EMA, k_arm 0.05, arm_ema 0.1)과 보상.
- 행동 이력(2단계)은 A + B + C 로 학습했는데 배율 1.0 에서 원래 성공률에 못 미칠 때만 다시 논의한다.

## 공통 코드 담당 (충돌 방지)

- **rh_aglt_r 계열**(grasping · place · locking 이 물려받음): `critic_delay_obs` 플래그 · layout · `widen_obs_ckpt.py --critic-only` 는 **grasping;rh56f1 이 한 번 구현하고 커밋**한다. place · locking 은 그 커밋을 받아 쓰고, `rh_aglt_env.py` · `layout.py` 를 각자 고치지 않는다.
- **pour_fabric_mimic: 제외**(10.11 사용자 결정, pouring 세션 회신). 구현하지 않는다.
- 테스트: 플래그 꺼짐 → critic 폭이 기존과 같음. 켜짐 → +24. widen 직후 → actor · critic 출력이 원본과 같음.

## 진행 순서 (트랙마다 같음)

1. **D0 기준 평가(학습 전):** 지금 성공한 체크포인트를 A · B 설정으로 결정론 sim 평가한다. 지연 배율은 0 / 1.0 / 1.5. 배율 1.0 에서 원래 성공률과 비슷하면 그 트랙은 재학습이 필요 없다.
2. `widen_obs_ckpt.py --critic-only --add 24` 로 체크포인트를 준비한다. A · B · C 를 켜고, 지연 배율 램프와 `fpp_ramp_steps` 를 걸어 이어서 학습한다.
3. 같은 조건으로 평가하고 아래 표로 보고한다.

| 체크포인트 | 배율 0 | 배율 1.0 | 배율 1.5 | env 수 · 시드 | 성공 기준 |
|---|---|---|---|---|---|

- 결과 체크포인트의 policy.yaml · README 에 `sim_delay_spec: S2R-DELAY-1011` 과 학습에 쓴 지연 값을 남긴다.

## 트랙별 메모

- **grasping (rh_aglt_l / r):** sftM2 는 균일 (9, 12) 팔 지연 + 손 (4, 6) 으로 학습했고, FP++ 모델은 꺼져 있다. → A 로 교체하고, FP++ 와 attach 를 켠다(10.11 사용자 결정). 공통 코드 커밋: e633cc2a, bcb91f38. 오른손 src21c · s240 · s200 은 지연 0 으로 학습했다. **공통 코드 담당.**
- **place (rh_place_l / r):** i09 · i01 은 지연 0 이다. delay29(ep1000 에서 멈춤)는 이미 팔 A 와 같다(arm_cmd_delay_joint_steps 실측값, jitter 1, curr_delay_scale 0→1). A 와 다른 점은 손 (3, 5)와 1.5 여유 구간이 없다는 것뿐이다(10.11 place 세션 정정). 컵 관측이 attached(FK)라 FP++ 지연은 해당 없고, 홀더 위치는 고정값이다. 손 모드는 position(계약).
- **locking (rh_cap_r):** 아직 성공 체크포인트가 없다. → 재학습이 아니라, 다음 fresh 라운드(단계 0, 6 mm)부터 A · B · C 를 처음부터 켠다. rh_cap 배포 계약 · 디코더는 아직 없다(deploy/policies/rh56f1 에는 aglt · place · pour_fj 만 있다). 그래서 FP++ 는 aglt 와 같은 경로(파지 전 FP++, 잡은 뒤 attach)를 기본값으로 두고, 배포를 만들 때 맞춘다.

## 공유 작업 트리 주의 (10.11)

- `rh_aglt_r/rh_aglt_env.py` · `rh_aglt_env_cfg.py` 에 locking 세션의 커밋하지 않은 수정이 있다(`arm_act_filter`, 기본 꺼짐, 사용자 승인 변경). grasping 이 공통 코드를 커밋할 때는 자기 hunk 만 커밋한다(`git add -p`). locking 의 수정을 섞거나 되돌리지 않는다.
- `arm_act_filter` 는 이 규격의 '행동 법칙 그대로'와 다르다. rh_cap_r 에서만 쓸지는 사용자가 결정한다. 배포하려면 디코더에도 같은 필터가 필요하다.
- **pouring (pour_fabric_mimic): 제외**(10.11 사용자 결정). 실기 붓기는 정책이 아니라 기본 궤적을 재생한다(export_s2r_traj: 잔차 0, 손 파지 고정, D4 방식). 정책을 써도 이득이 없었다(병 D1 궤적만 84.3 %, 정책 83.9 %). 손은 고정이고 컵 관측은 FK 라서 B 도 해당하지 않는다. 그래서 관절별 지연 · critic 칸 구현, s2h4 이어서 학습, D0 평가를 모두 하지 않는다.
