# s2r_init_right — 오른팔 컵 집기 정책 s2r 반입 한 벌 (09.21)

서버 `oem@server` 에서 **읽기만** 해서 받아온 사본. 원본(서버 `our_source/fj_t2r_init`, `log/.../grasp-fj-t2r-rand`)은
s2r 동결 대상이라 건드리지 않는다. 여기 있는 것도 학습 입력으로만 쓰고 덮어쓰지 않는다.

## 가중치 (`nn/`)

| 파일 | 무엇 | 지표 |
|---|---|---|
| `fj_rand_i01_best_ep5000.pth` | **s2r 후보** — 접근·파지·리프트·목표까지 한 정책 | 먼 출발 성공 0.85 · 공차 0.019 · 컵 기울기 4.3° |
| `fj_rand_i01_final.pth` | 같은 런 마지막 가중치 | 공차 바닥(0.015) 뒤 열화 — 성공 0.61(e8700). **쓰지 않는다** |
| `fj_rand_i00_ep800.pth` | 접근·인벨롭 단계만 | 접근 0.91 · 인벨롭 0.59 · 리프트 0.07 |

sha256 (best_ep5000) 앞 16자리 `e76f9c6079663b61` — 서버 원본과 일치 확인(09.21).

## 런 설정 (`params/`)

`fj_rand_i01` 런이 저장한 그대로. play.py 는 이 `env.yaml` 을 복원하므로 CLI 오버라이드가 덮인다.

- `reward_code_path` 가 **서버 절대경로**(`/home/oem/rl_ws/hdgp/...`)다. 로컬에서 재생하려면
  `/home/user/rl_ws/hdgp/reward_gen/grasp_fj_rand/iter_01/compute_reward.py` 로 고쳐야 한다(보상 코드 자체는 git 으로 이미 로컬에 있다).
- `profile_name: tesollo_right_short_tl` · `episode_length_s: 15.0` · `tol_floor: 0.015`
- agent: PPO-LSTM(`units 1024`, `horizon_length 16`, `minibatch_size 16384`, `seq_length 16`, `zero_rnn_on_done: true`),
  task id `open-short_r_grasp_fj_t2r_rand-lstm`. 불러올 때 vendor `rl_games_sapg` 가 PYTHONPATH 에 있어야 한다(없으면 `KeyError: 'model'`).

## 인터페이스 (실기 배선용)

- obs 133 / critic 157 / action 26
- action = 팔 7 관절 **증분** + 손 19 관절 **절대**(시너지 아님). 손 19 인 것은 `-tl` 자산에서 `thumb_1` 이 용접됐기 때문.
- 실제로 움직이는 손 관절은 13 개. 잠긴 것: `thumb_2`, `index/middle/ring/pinky_1`, `pinky_2`.
- env/과제 코드는 `hdgp/source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/`(동결, 읽기 전용).
