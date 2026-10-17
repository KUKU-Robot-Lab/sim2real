# 실기 파지 s2r 확인 → tilt(붓기) 실기 평가 10회

> 작성 2026-10-17 (s2r_1st). 사용자 승인 "권장대로 진행" (2026-10-17 21:20 KST).
> 근거: `~/rl_ws/handoff/2026-10-16_codex_to_claude_rg3_console.md`, 사용자 요청(10.17 /ecc:plan), 경로 C 추가 요청(10.17).

## 목표

보고서용 영상·사진과 평가 지표를 만든다. 파지 정책의 s2r gap 이 풀렸는지 먼저 보고, 풀렸으면 파지 정책으로,
안 풀렸으면 예비 경로로 컵을 쥔 채 tilt 를 10회 평가한다.

## 사용자 결정 (2026-10-17)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 파지 확인 횟수·합격 | 쪽당 5회. 들어 올린 뒤 정책이 멈추고 손이 3 s 유지되면 1회 성공. 4/5 이상이면 경로 A |
| 2 | tilt 실행 방식 | D4 잔차 정책 먼저. 시간이 되면 기본 궤적으로도 10회 → 보고서에서 비교 |
| 3 | 이름 변경 범위 | 콘솔 화면·에피소드·보고서만 tilt. 코드 모듈 `pour_*.py` 와 hdgp 학습 이름은 그대로 |
| 4 | 성공 정의 | 시뮬 판정과 같게. 0-2 에서 원문 확인 후 숫자로 제시 |
| 5 | C 로 넘어가는 기준 | A 가 쪽당 4/5 미만이면 C |
| 6 | 1-C 시뮬 사전 확인 | 한다. local 5090 이 비어 있으면 거기서 |
| 7 | 접근 멈춤 기준 | 손바닥–컵 거리 ≤ 6 cm 이고 팔 정지가 0.5 s 이어지면 멈춤, 그 전에 15 s 면 멈춤. 6 cm 는 1-C 에서 조정 |
| 8 | tilt 순서 (10.17 추가) | 기본 궤적(잔차 0) 먼저 실기 10 회 기준, D4 이식은 이어서 — 결정 2 의 순서를 바꿈 |
| 9 | 복귀 | 정책은 7.0 s 궤적에서만, 복귀는 잔차 0 으로 원호 4 s 되짚기 |
| 10 | 경로 A 멈춤 | 양손 모두 컵 7 cm 위 + 쥠 2 s(120 틱)에 정책 멈춤(첫 목표 성공도 멈춤). 그 뒤 3 s 유지로 성공 판정. 10.14 의 오른손 5 s(약지 · 새끼 감기 기다림)와 0.5 s 사이 값 |
| 11 | 비드 수 | 10 개(D4 sim 전량 82.4 %) |
| 12 | 잡기 뒤 이송 | ready_carry(10.13 규칙) + 끝 = 고정 이정표 붓기 시작 자세(10.12 규칙). hdgp Mover 에 끝 관절 지정(q_goal) 인자 추가 |

## 경로

- **A (파지 정책):** rg3wo 오른손 잡기 → 왼손 잡기 → ready_carry(한 팔씩) → tilt → 시작 자세 복귀 → 정지.
- **C (접근 정책 + 스크립트 파지):** 접근 정책 `rg3wc ep800` 원본(손을 편 채 접근) → 멈춤 조건(결정 7)에서 정책 끔·팔 고정
  → thumb_1 유지, 엄지 굽힘·네 손가락을 정한 속도로 닫음(손가락별 `force_set` 상한, 닿으면 멈춤)
  → 배포 판정 HandBlockOpp(네 손가락 중 2 개 이상 막힘 + 손바닥–컵 < 8 cm). 못 쥐면 들지 않고 멈춤(복구는 사용자 rehome)
  → 손 목표 유지한 채 10 cm 들기 → ready_carry → tilt 이후 A 와 같음.
- **B (수동 파지):** 팔을 tilt 시작 자세로 → 손 펴기 → 사용자가 컵을 손에 댐 → 닿을 때까지 닫고 래치. 확인 두 번(컵 대기 전, 닫기 전).
- 갈림길 순서: A → C → B.

## 단계

### 0. 정보 확보 (로봇 없음)
- 0-1 grasping 에 rg3wo 문의: 체크포인트·md5·env, 칸 수(hp3 94/13), 실기용 학습 조건, 판정 키, `1017d_grid75` 칸별 성공률.
- 0-2 tilt 정책: D4 `rh_ref_s2h4` ep5000 체크포인트·입력 구성, 기본 궤적·복귀 구간, 시뮬 비드 성공 판정 원문. (s2r 2nd method 문의 + server·hdgp 직접 확인)
- 0-3 궤적에서 tilt 중 왼팔이 움직이는지 → 양팔 재생 필요 여부.

### 1. 배포 (로봇 없음)
- `aglt/{right,left}_rg3wo` 등록, 계약, CPU 추론 3 물체. 에피소드 `grasp_rg3wo_{right,left}.yaml`:
  성공(또는 들림 래치) 뒤 정책 멈춤 + 손은 마지막 지령 유지. 성공 조건(10 cm, 2 s)과 정지 조건(7 cm, 5 s)을 하나로 맞춤.
- 경로 C: `aglt/{right,left}_approach_rg3wc` 등록·계약·CPU 추론. 에피소드 형식에 셋 추가 — 정책 단계 조건 종료, 스크립트 파지 단계, 들기·이송 단계.
  가짜 도메인 99 에서 정상 흐름 + 쥠 확인 실패 흐름.
- 1-C (시뮬): hdgp act_override 로 접근 정책 → 스크립트 닫기, 실기와 같은 칸에서 10 회씩.

### 2. 실기 파지 s2r 확인 (사용자 승인 뒤)
- 컵 자리: 0-1 결과에서 성공률이 높은 칸(오른손 src240, 왼손 shaker), 테이프 표시. tilt 평가도 같은 자리.
- 기록: 헤드 카메라 영상, 정면 대각선 휴대폰 영상, bag, 원장, 성공 여부·실패 유형(접근 빗나감, 밀림, 들다 놓침, 유지 중 풀림), 최대 들림 높이·유지 시간, 같은 칸 시뮬 성공률.
- 실기 전에 Codex 인계 문서 갱신.

### 3. tilt 실행 경로 (로봇 없음, 2 와 병렬)
- 3-1 이름 바꾸기(결정 3). 3-2 ready_carry 배포(hdgp main 모듈 그대로, md5 기록, 오프라인 일치 비교).
- 3-3 tilt 실행(두 팔 동시 시작, 끝에 시작 자세 복귀). 3-4 수동 파지 B. 3-5 에피소드 A·B·C.

### 4. 실기 tilt 평가 10 회 (사용자 승인 뒤)
- 회차마다: 같은 자리에 컵·비드 N 개 → 에피소드 → 옮긴·흘린 비드 세기 → 전후 사진 → rehome.
- 기록 칸: 회차, 경로(A/B/C), 파지 결과, 옮긴 비드/전체, 흘린 비드, 성공 여부, 시간, 영상 파일.
  C 이면 접근 멈춘 자리의 손바닥–컵 거리, 닿아서 멈춘 손가락 수.

### 5. 보고서
- `report/pour/pour_tilt_real_1017_eval.html`, `report/grasp/grasp_rg3wo_real_1017_eval.html` + README 색인.
- 지표: 파지 실기 n/N 과 같은 칸 시뮬 성공률, tilt 성공 x/10·평균 옮김 비율·흘림 수.

## 진행 기록

- 10.17 21:2x 0-1(grasping), 0-2(s2r 2nd method) 문의 보냄.
- 10.17 21:2x local 5090 은 PID 259223(open-rh_r_aglt-lstm, rg3wo warmstart, 8192 env, 21 GB)이 쓰는 중 → 결정 6 의 "비어 있으면" 불충족. arm4090 GPU0 은 비어 있음(810 MiB).
  grasping 회신: PID 259223 = aglt_r_rg3wp_hm(사용자 지시 "당장 실험", 약 22 h). 학습 GPU 에 play 를 같이 올리지 않는 규칙.
- 10.17 경로 A 배포: `aglt/{right,left}_rg3wo`(ep10000, md5 624e55dc… · 7cf1bd19…, server hdgp_grasp_w). 3 물체 12 스텝 CPU 추론 ok.
  rh_aglt.py 두 곳 고침 — (1) hdgp 충돌체 사본 `*_cvx` · `*_solid` → 계약 물체 이름 `*_real`(one-hot 순서 그대로),
  (2) hand_block 명목값 덮음 검사에 ADR 종점(`*_end`, adr_env_mix) 범위를 합친다. 에피소드 grasp_rg3_{right,left} 정책 교체 · 미션 재생성.
  멈춤 조건은 10.14 사용자 결정(오른손 7 cm · 5 s = 300 틱, 왼손 30 틱) 그대로 — 하나로 맞추지 않았다(사용자 결정 필요).
- 10.17 경로 C 배포: `aglt/{right,left}_approach_rg3wc`(오른손 원본 md5 fff6c4aa…, 왼손 = 거울 45e24515… — 60 스텝 거울 일치 0.0).
  접근 정책 계약은 rg3wo 와 관측 · 행동 해석이 같다(쥠 판정 키만 다름). `policy_control/approach_grasp.py` + 노드 파라미터
  `approach_grasp`("" 끔 · "on" · "키=값,…"), 에피소드 `grasp_c_{right,left}.yaml`(policies.<역할>.approach_grasp), 미션 max_episode_s 22.
  계획과 다른 점: 손가락 멈춤은 force_set 이 아니라 배포 막힘 신호(hand_block 명목값) — 펌웨어 힘 멈춤은 10.12 이후 증거가 없어 계약이 막는다.
  가짜 플랜트 시험: 접근 2.8 s → 닫기 1.5 s(4 손가락 래치) → 들기 0.091 m → 3 s 유지 → 성공. 빈손 · 먼 컵 · 유지 중 풀림은 들지 않고 abort.
- 10.17 D4 정보(s2r 2nd method): 관측 180 · 행동 26 · 60 Hz LSTM, 시연 궤적(210 매듭 7.0 s) + 잔차, 양팔 모두 움직임(왼팔 최대 13.3 cm,
  잔차 43°). 위상 관측 = t / 궤적 끝 시각(7.0 s) — 복귀 붙은 궤적을 주면 학습 밖. 배포 코드에 teach residual 이 아예 없다.
  sim 성공 = 쓴 비드 전부 받는 컵 · 쏟음 ≤ 0.25 · 1/3 s 유지(사실상 전량 · 흘림 0). D4 전량 성공: 10 개 82.4 % · 15 개 72.1 % · 20 개 56.0 %.
- 10.17 경로 C 리뷰 반영: 값마다 안전 범위(`SAFE_RANGE` — judge_ticks ≥ 1, lift_m ≤ 0.2 m, near_m ≤ 0.12 m, loss_ticks ≤ 60,
  lift_damping ≥ 0.01). 노드 한도 = 최악 경로(15 + 4 + 2 + 1.5 + 3 = 25.5 s) + 3 s = 29 s, 실행기는 그 + 10 s 를 기다린다
  (`PolicyPlan.max_s`, 전에는 15 + 10 s 고정이라 늦게 쥔 성공이 실행기 시간 초과로 잘릴 수 있었다).
  경로 C 진행 중 시간 초과는 성공(stop)이 아니라 abort. `hand_block_*_end` 는 두 값이 아니면 계약을 거부.
  남은 위험(고치지 않음): 쥘 때 손에 가린 컵은 FP++ 마지막 값을 쓴다 — 컵이 밀려나도 거리 조건이 참일 수 있다.
  그때 빈손으로 들면 들기 중 풀림 감지(0.5 s)로 abort 한다.
- 10.17 D4 실기 이식 범위: 배포 pour_fj 는 joint increment 팔 · 촉각 관측을 이미 갖지만, D4 가 켠 아래 기능이 배포에 없다
  (pour_fj.UNPORTED_OFF 가 계약 생성을 막는다): arm_action_lpf 0.3 · hand_hold · rcv_arm_hold · teach 궤적(+잔차 EMA 0.1,
  0.6 rad/s, 출발 합류 1 s) · teach_rcv_follow + rcv 잔차 · 관측 끝 15 칸(위상 1 + 소스 기준 7 + 리시버 기준 7).
  실기 출발 궤적은 hdgp retarget_real_start.py(출발 하나 IK, check.json) 로 만들되 arc6 옵션을 넘겨야 한다.
  D4 README 의 실기 순서도 "잔차 0(궤적만) 기준 → 정책 잔차"다.
- 10.17 경로 A 멈춤 양손 2 s(120 틱)로 통일(결정 10, sim2real 2679811).
- 10.17 tilt 기본 궤적(3-3): hdgp `tilt_real_start.py`(5b4be2371 · c237f04a5) — 실기 출발(측정 팔 q + 손 안 컵 자리)을 1 항목 출발
  뱅크로 바꿔 `traj_retarget_starts` 를 그대로 부른다(옵션 = chain_pour.teach_args). 기준 출발에서 D4 arc6_meta 와 같은 값
  (최대 기울기 128.609°, 4.86 s, 입구 이동 0.1263 m). 복귀 끝 받는 컵이 시연 출발점(탁자 위 2 cm)이라 측정 출발 관절까지
  최소 저크를 붙였다 → 17.4 s, 끝 = 출발 관절. 검사: IK 오차 · 관절 여유 · 60 Hz 속도(병 최대 1.89 rad/s, 한계 2.0) · sim 출발 범위 ·
  탁자 여유(실기 탁자 0.2037 에서 받는 컵 최저점 1.57 cm, 한계 1 cm).
- 10.17 이송(결정 12): hdgp ready_carry `Mover.start(q_goal=)`(68e4620e3, 기존 시험 32 + 1 통과). sim2real `plan_carry.py` 가
  열린 고리로 굴려 60 Hz 목표를 만든다. 들린 자세(컵 (0.25, ∓0.20, 0.36))에서 2.0 s, 끝 관절 오차 0, 최대 0.32 rad/s, 탁자 여유 7.1 cm.
- 10.17 sim2real: 자산 `deploy/policy_control/config/tilt/base_arc6/`(policy.yaml 에 hdgp 파일 13 개 md5 · 기준 파일 md5 고정,
  다르면 계획 거부), `plan_tilt.py` · `plan_carry.py`(움직이지 않음), `replay_to_pd.py` 14 관절 한 스트림 + `--max-start-gap`,
  에피소드 궤적 종류 `carry` · `tilt`(두 손이 컵을 쥐었을 때만, 계획 실패면 refused), 에피소드 `tilt_base_a`, 미션 재생성.
  손 안 컵 자리는 기준 출발(single734)의 값이다 — 실제 쥔 자리와 다르면 컵 경로가 그만큼 어긋난다(실기 확인 항목).
- 10.17 fake 도메인 97 리허설(두 팔 pd 홈 → plan_carry → 14 관절 재생 → plan_tilt → 재생 → shutdown): 이송 계획 2.4 s · 재생 241 프레임
  (홈에서 시작이라 4.0 s 상한), 끝 ↔ 붓기 시작 자세 0.0000 rad · tilt 계획 2.7 s · 재생 1045 프레임 18.7 s, 끝 ↔ 출발 0.0000 rad ·
  두 팔 모두 stop 뒤 pd 내부 유지(internal). 홈 단계(한 팔 재생)도 고친 replay_to_pd 로 통과. fake 는 robot_control/ros_ws/install
  overlay 를 source 해야 RH56F1 가짜 손이 뜬다(첫 시도 실패 원인).
- 10.17 결정 4 성공 정의(숫자): D4 params/env.yaml 판정 = success_fill_ratio 1.0 · success_spill_max 0.25 · success_xy_thresh 0.20 m ·
  success_hold_steps 20(1/3 s). 실기 = tilt(복귀까지) 끝난 뒤 받는 컵(shaker) 비드 = 넣은 10 개 전부. sim 은 에피소드 중 한 순간이라도
  1/3 s 성립하면 성공이고 실기는 끝에 센다(복귀 중 흘림까지 실패 — 보수적). 기록: `tools/eval_record.py tilt|grasp|summary`
  → logs/eval_real/<날짜>/{tilt,grasp}.{jsonl,csv}(세 칸 합이 넣은 수와 다르면 거부, 결정 1 파지 성공 = ok + 3 s 유지).
- 10.17 경로 A 이송을 계획대로 한 팔씩(오른팔 → 왼팔): carry sides [right] · [left], plan_carry `--sides`, 재생은 그 팔 7 관절.
- 10.17 경로 C 에피소드 `tilt_base_c`(grasp_c 두 손 → 한 팔씩 이송 → tilt), 경로 B 에피소드 `tilt_base_b`:
  빈손 이송(carry cups: false) → 손 펴기(hand open = pd hand_release) → 확인 노드 둘(컵 대기 전 · 닫기 전) → hand grasp
  (`policy_control/hand_grasp.py` + `tools/hand_grasp.py`, 경로 C 와 같은 닫기 · 막힘 래치 판정, 거리 조건 없음, 3 s 유지, 팔은 시작
  실측에 고정) → 왼손 같은 순서 → tilt. 확인 노드(confirm)는 연속 실행에서도 멈추고 그 노드 이름의 승인([다음])을 따로 받는다.
  엄지 회전(thumb_1)은 편 손 값 1.57 그대로(경로 C 의 "엄지 회전 고정"과 같은 규칙) — 실기에서 컵 대향이 부족하면 조정 항목.
- 10.17 결정 3 이름: 콘솔 정책 상자 "tilt 정책", 작업 공간 과제 버튼 · 카드 "tilt"(내부 키 pouring 그대로), 미션 pour_fj 단계 제목,
  에피소드 tilt_base_*. 코드 모듈 pour_*.py · hdgp 이름은 그대로.
- 10.17 fake 손 '손 안 컵'(`/fake/hand_<side>/cup` Bool, `--cup-stop`) — 켜면 손가락이 그 각도에서 막혀 막힘 신호가 실기처럼 생긴다.
- 10.17 리뷰(코드 리뷰어, CRITICAL · HIGH 없음) 반영: hand_grasp 도구가 SIGTERM(실행기 정지)에도 stop 을 내고 끝남 · 팔 목표 = pd 세트포인트
  (pd status arms.<팔>.q_set 추가, 실측과 0.1 rad 넘게 다르면 거부) · hand grasp 바로 앞 confirm 두 개 강제 · hand grasp 재시도 상한 0 강제 ·
  손 펴기는 실측이 펴질 때까지(3 s, 0.1 rad) · plan_carry 두 팔 사이 검사(팔 관절 4~7 · 손바닥 점 ≥ 0.10 m, 두 컵 겉면 ≥ 0.03 m —
  경로 A · B 한 팔씩 실측 0.30 m · 0.14 m 이상) · 확인 노드는 그 노드에 온 뒤 쓴 승인만 받음(남은 옛 승인 줄 무시).
- 10.18 발견: 작업 트리에서 지워진 옛 정책 폴더 right_cyl60g 의 env.yaml 을 홈 복귀 계획기(plan_rehome)가 세계로 읽어, 모든 에피소드의
  첫 홈 복귀가 "경로를 못 만들었다"로 멈췄다(fake 경로 B 첫 리허설). 같은 파일(git HEAD, md5 같음)을
  `deploy/policy_control/config/home_path/rh56f1_world_env.yaml` 로 두고 계획기가 그것을 읽게 했다. 지워진 폴더를 쓰는 다른 곳
  (미션 pick_place 에피소드 · policy_aglt 단계의 right_src21c · left_g5362 · 테스트 166 건)은 그대로다(이 작업 범위 밖).
- 10.18 sim 세팅(사용자 "다른 기능 다 구현한 뒤 sim"): hdgp `scripts/tools/pour_bank/tilt_base_sim_eval.py`(D4 env 사본 · 궤적만 ·
  비드 10 · 기준 출발 · 판정 = D4 성공(1/3 s 전부)과 끝 판정(실기와 같음)) · `scripts/tools/rh_aglt_approach_close_eval.py`(1-C —
  접근 정책 → 배포 approach_grasp 단계 기계를 env 마다 torch 로, 막힘 명목값 · 실기 규격 · substep 4, 결과 = 스크립트 성공 · 실제 들림 ·
  거짓 성공 · 접근 끝 거리). 둘 다 GPU 가 필요하다 — 결정 6 의 로컬 5090 은 10.18 00시 다른 세션 프로브 둘(29 GB / 32 GB)이 쓰는 중.

