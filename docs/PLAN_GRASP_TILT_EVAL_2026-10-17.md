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
- 10.18 fake 도메인 97 경로 B(tilt_base_b) 상황판 연속 실행 → SUCCESS: 홈 → 빈손 이송(우 → 좌) → 손 펴기(실측 0.09 · 0.07 rad) →
  확인 4 번(연속 실행에서 각각 멈춤 · 그 노드 이름 승인) → 두 손 쥠(4 손가락 래치 · 3 s 유지) → tilt 재생 22.7 s.
  첫 시도에서 pd 가 붙들던 손 → 도구 목표로 넘어갈 때 손 실측이 약 0.6 s 안 움직여 막힘 판정이 오인 래치 → 손가락마다 0.01 rad 움직인
  뒤에만 래치(hand_grasp.RESPOND_RAD). 상황판 policy 탭에 tilt 평가 묶음(경로 카드 · 지금 에피소드 · 계획 검사 · 회차 기록 · sim 기준).
- 10.18 sim(로컬 5090 이 빈 때, 결정 6) — 결과는 sim2real logs/sim/1018_*:
  · 기본 궤적(잔차 0) 비드 10 · 기준 출발: 유효 63 env 중 62 성공(98.4 %, 받는 컵 평균 99.8 %, 흘림 0, 실패 1 = 병에 1 개 남음,
    무효 1 = 시작 직후 리셋). 출발 DR 뱅크(bank_env17_startdr_x4 무작위 64 행 — 손 안 컵 자리 · 팔 자세가 다른 출발): 60/64(93.8 %,
    받는 컵 평균 96.4 %, 흘림 1 개). D4 README 의 궤적만 47.6 % 는 학습 외란(밀기 · 지연)을 다 건 값이라 조건이 다르다.
    sim 은 복귀 원호 끝까지(실기 마지막 최소 저크 2 s 없음).
  · 1-C(경로 C, 오른손, 2 칸 × 10 회 × 물체 3 = 60): 실기 규격(지연 · FP++ · substep 4) 0/60 — 접근 끝 거리 중앙값 13.5 cm,
    접근 중 컵 넘어짐 20. 학습 조건(런 env.yaml 그대로, substep 1)에서도 0/60 — 접근 정책은 손바닥–컵 8.1~9.8 cm(중앙값 8.4 cm)에서
    멈춰 6 cm 조건이 한 번도 안 맞고(15 s 시간 초과), 쥔 판정 거리 8 cm 보다도 멀다. 멈춤 10 cm · 판정 12 cm 로 바꿔도 0/60 —
    접근이 7.7 s 에 멈추고 닫으면 손가락이 컵을 밀어 60 중 38 이 넘어지거나 물리 오류, 나머지는 래치 0~1 개로 못 쥠.
    → 경로 C 를 지금 설계(rg3wc 접근 끝에서 바로 닫기)로는 sim 에서 못 쥔다. 실기 전에 사용자 결정 필요(예: 닫기 전 손바닥 법선으로
    스크립트 접근 몇 cm, 또는 잡기 정책 rg3wg 를 잇기). 왼손 1-C 는 오른손 결과로 돌리지 않았다.
- 10.18 사용자 "경로 C 를 사용할 만한 approach 정책이 있을 텐데? 컵에 닿기만 하고 파지를 안 하는 정책이 있음" → 후보 rg3wh
  (오른손, 10.15 잡기 단계 런 — 접근 rg3wc ep800 에서 이어 학습했지만 손을 끝까지 안 닫음: 편 손 · 닫힘 0.07 · 컵 축 1.6 반지름까지 깊이).
  rg3wc 는 접근 단계 보상이 '컵에 닿지 말고 2.5 반지름에서 멈춤'(rh_aglt_ap prompt)이라 스크립트 닫기 앞 자리로 맞지 않는다.
  1-C 시뮬(arm5080 RTX 5080 — 사용자 "5090 은 locking", 오른손 2 칸 × 10 회 × 물체 3 = 60, 학습 조건 = 런 env.yaml · substep 1):
  | 접근 정책 | 판정 | 성공(쥠 · 10 cm 들기 · 3 s 유지 · 진짜 들림) | 접근 멈춤 거리 중앙값 | 비고 |
  |---|---|---|---|---|
  | rg3wc ep800 | 배포 래치 | 0/60 | 8.4 cm | 닫으면 손가락이 컵을 민다 |
  | rg3wc ep800 | 접촉(sim 진단) | 0/60 | 8.4 cm | 닫기 끝 접촉 0 개 37 · 넘어짐 22 |
  | rg3wh ep3100 | 배포 래치 | 3/60 · 1/60(재실행) | 5.4 cm | 닫으면 58/60 이 다섯 손가락 모두 닿는데 래치는 엄지 + 손가락 0~1 개 |
  | rg3wh ep3100 | 접촉(sim 진단) | 24/60(병 240 14/20 · 병 200 8/20 · shaker 2/20) | 5.4 cm | 실패 대부분 = 든 뒤 컵 기울어 env 종료(tipped) |
  | rg3wh ep3100 | 실기 규격(지연 · FP++) · 접촉 | 9/60 | 7.2 cm | 접근 중 컵 넘어뜨림 25(rg3wh 는 지연 · FP++ 없이 학습) |
  | rg3wh ep300 | 배포 래치 | 0/60 | 5.4 cm | |
  → 경로 C 는 '깊이 들어가 편 손으로 머무는' 접근이 있어야 물리적으로 쥔다(rg3wh 학습 조건 40 %). 남은 문제 두 가지:
  (1) 배포 쥔 판정(네 손가락 래치 ≥ 2)이 sim 에서는 이 쥠을 못 알아본다 — 실기 래치 동작 확인 또는 판정 변경 필요.
  (2) rg3wh 는 실기 규격(지연 · FP++)에서 접근 중 컵을 넘어뜨린다 — 실기 전 그 DR 로 파인튜닝하거나 실기에서 천천히 확인.
  왼손: rg3wh 왼손은 접촉 없이 반쯤 닫는 정책이라 후보가 아니다. 결과 sim2real logs/sim/1018_1c_right_{wh300,wh3100,wh3100b,wh3100c,wh3100r,wcc}
  (arm5080 같은 경로) · 도구 hdgp 40d717737 · a4dd7421b(손가락 접촉 · 래치 기록, --judge contact).

- 10.18 사용자 "1-C 접근만 하게 하고 스크립트처럼 파지를 하게 우리가 강제하는 거니까 쥠 판정 따위 필요 없음" → approach_grasp 기본
  judge=0(강제 파지): 닫기 끝(모든 축 막힘 · 쥔 자세, 늦어도 close_max_s 4 s) 뒤 판정 없이 든다. 들기 높이 미달만 실패(d4aca04).
  사용자 "스크립트 기준 접촉 판단이 되면 멈추면 됨 · 실기는 핸드에 과도한 토크(과전류)가 걸리지 않게만 조정 · sim 에서 안 되는 건 별로 중요하지 않음"
  → 막힘(래치) 축은 그 순간 실측 + pin_lead_rad(기본 0)에 고정, 안 닿은 축은 지령이 실측을 close_lead_rad 0.08 rad 넘게 앞서지 않음
  (pd HAND_MAX_LEAD 0.2 대신). 펌웨어 전류 한계 800 mA · FORCE_SET 그대로(4035a2a). sim 1-C 도구도 같은 규칙(접촉력으로 멈춤,
  substep 4 · mimic nf0, hdgp 6d459152f).
- 10.18 경로 C 접근 정책 rg3wc ep800 → rg3wg ep300 양손(grasping 1순위, 사용자 승인): `aglt/{right,left}_approach_rg3wg`
  (server hdgp_grasp_w open-rh, md5 04dd459d… · 19ccc867…, 학습 = 지연 · FP++ · ADR 없음 · 손 position · 판정 contact).
  계약 생성기가 contact 판정 위치제어 런의 G(0.0128, 학습 고정)를 거부해 pos 판정 런과 같게 학습 범위 끝으로 붙인다(rh_aglt.hand_block_of)
  — 경로 C 닫기는 close_contract 가 막힘 명목값(0.0218)을 다시 씌우므로 실기 닫기 동작은 같다. 에피소드 grasp_c_{right,left} · tilt_base_c,
  미션 재생성. 경로 B(tilt_base_b)의 hand_contract 는 rg3wc 그대로(막힘 명목값만 쓴다).
- 10.19 이름 바꿈(사용자 지시, aglt_new 전달 — 학습 쪽 hdgp/log/rl_games/open-rh/{right,left}/approach/approach_{r,l}):
  `aglt/{right,left}_approach_rg3wc` → `aglt/{right,left}_approach`(10.06 경로 규칙 <팔>_<태그> 때문에 approach_r 대신, 사용자 선택),
  체크포인트 approach_r.pth · approach_l.pth(md5 그대로 fff6c4aa… · 45e24515…). 계약은 경로 두 칸만 고쳤다. 이 문서 위쪽의
  rg3wc 폴더 이름은 그때 기록이다. rg3wg 두 폴더 이름은 그대로(사용자 결정).
- 10.19 1-C 기본 접근 정책 rg3wg ep300 → `aglt/{right,left}_approach`(rg3wc ep800, 사용자 결정 "접근은 rg3wc ep800 을 일단 써 본다").
  에피소드 grasp_c_{right,left} 와 미션 재생성. 콘솔 접근 카드는 전부터 카드 정책으로 바꿔 돌렸다. tilt_base_c 는 rg3wg 그대로.
- 10.19 콘솔 잡기 카드(경로 A aglt)에 sim 궤적 버튼 둘(사용자 "정책 live 와 sim 궤적 재생 둘 다 고르게"): simplan_<카드> = FP++ live
  컵 자리로 server sim 계획(tools/sim_traj.py plan, 로봇 안 움직임) · simrun_<카드> = 그 카드의 마지막 계획을 pd 로 재생(run --execute,
  정책 노드 없음, 컵이 계획 자리 1 cm 안). 계획은 logs/sim_traj/card_<카드>/. 정지는 SIGTERM → sim_traj 가 episode stop → pd/hold 확인.
- 10.19 실기(22:07~22:19) 결과와 방향 전환:
  · 1-C 접근 카드(approach_r/l = rg3wc ep800): 손바닥이 3 s 에 컵에서 8.7 cm(정책 보상 목표 = 2.5 반지름 + 3 cm 위)까지 온 뒤
    바깥으로 10~15 cm 흘러감 → 6 cm 멈춤 조건이 안 맞아 15 s 시간 초과로 허공에서 닫고 들었다(오른손) · 컵 1 s 끊김으로 멈춤(왼손).
    10.18 1-C sim(0/60, 실기 규격 접근 끝 13.5 cm)이 이미 예측한 결과. 사용자: "1-C 는 보류"(2.5R 이 눈으로 보이지 않아 판단 보류) ·
    "파지도 내가 명령을 내리는 것이었는데 자동이 됐다".
  · 1-A neo03 왼손: FP++ 는 끝까지 잡았고(끊김 처리 안 걸림) 컵을 앞으로 4.7 cm 밀고 쥔 뒤 뒤로 10.7 · 바깥 12 cm 끌며 10 cm 들었다.
    같은 칸 sim 도 바깥 7.4 cm 로 휜다(학습 성공 판정이 높이만 본다).
  · 사용자: "그냥 지금부터는 1-B 처럼 정책이 아니라 제어 모듈들로 모든 걸 진행해야 함 · 작성할 보고서 사진/영상도 필요함".
- 10.19 1-B(정책 없이 제어 모듈만) — 사용자 결정: 쥐는 자리 = tilt 기준 자리(single734 행 0 의 손 안 컵 자리) · 접근 = 앞에서 곧게 ·
  승인 = 닫기 전만 · 왼손 / 오른손 독립 · sim 은 로컬 5090(1-A1, 정책 궤적이라 보류).
  · 에피소드 `config/episodes/grasp_b_{right,left}.yaml`: 홈 → 배치 기록 → 손 펴기 → reach → [확인] 닫기 → hand grasp(경로 B 와 같은
    hand_grasp.py) → lift → 끝. 미션 단계 `episode_grasp_b_<팔>`(팔마다 실행기 episode_runner_<팔>) + bag · CPU · 카메라 15 Hz 기록 →
    끝에 `tools/episode_media.py --pending`(logs/report_media/<회>/video.mp4 · 01_start · 02_reached · 03_grasped · 04_lifted · sheet.jpg).
  · `tools/plan_reach.py`: tilt 기준 자리 = 컵 중심이 손가락 방향 4.1 cm · 손바닥 앞 3.2 cm(깊은 쥠). 편 손(엄지 회전 1.57)은 엄지 끝이
    손바닥 앞 7.7 cm 라 손바닥 법선으로 들어가면 엄지가 막는다 → 손가락 방향 수평으로 곧게(컵이 손가락 면과 엄지 사이로, 틈 src240 2.4 ·
    shaker 1.3 cm). 컵 FP++ 원점 → 바닥 맞춤(shaker_c +2 cm). 쥔 자세 = 어깨 → 컵 방위 ±45°(15°) × 시드 8 의 손바닥 자세 IK, 곧게
    들어가기까지 풀어 경로 전체 관절 여유가 큰 것. standoff 12 → 11 → 10 cm(손끝이 컵 중심보다 6.3 cm 앞; 오른손 오늘 칸은 12 cm 면
    들어가기 시작에서 j7 이 한계 2° 안 → 10 cm 6.2°). 이송 = ready_carry(q_goal). 검사: IK · 여유 ≥ 5° · 직선 3 mm · 2 rad/s ·
    손 점(손바닥 + 손끝) 탁자 + 1.5 cm · 팔 관절 탁자 + 3 cm · 이송 중 컵 겉면 2 cm · 들어가는 동안 컵 안 아님 · 다른 팔(실측) 10 cm.
    오프라인(홈 출발, 오늘 칸): 오른손 src240b (0.244, −0.116) 9.0 s(여유 6.2°, 들어가며 손끝–컵 7.8 mm) · 왼손 shaker (0.259, 0.078)
    10.0 s(여유 15.3°, 3.8 mm — 빠듯).
  · `tools/plan_lift.py`: 지금 자세에서 손바닥 방향 고정 수직 10 cm(최소 저크 3 s) + 3 s 유지. 실행기가 끝에 FP++ 로 컵 상승 ≥ 7 cm 판정
    (없으면 보류). episode_spec: 궤적 reach · lift(한 팔 · object), reach 바로 뒤 hand grasp 는 확인 하나(사람이 컵을 대는 경로 B 는 둘).
  · 카메라 PC vision_recorder: TERM 에서 rclpy 가 먼저 context 를 닫아 예외로 끝나 CSV 마지막 10 s · 요약이 빠졌다 → 예외를 잡고 마무리
    (vision-3090 사본도 같은 파일로 교체, 옛 것은 /tmp/vision_recorder.py.bak_1019).
- 10.20 1-B 접근을 aglt_new ReadyApproach 로 바꿈(사용자 "컵 접근은 ready-carry 세팅을 aglt_new 세션에 물어봐 여기는 금방하던데",
  "가짜 리허설에서 실패하면 안 되는 것 아닌가", "로봇 실기는 local 5090 사용해야지").
  · 옛 계획(쥔 자세 7 방위 × 시드 8 IK + 331 스텝 직선 IK × standoff 12→10 재시도)은 로봇 PC 에서 50 s. 이제: hdgp
    tasks/rh_aglt_r/ready_approach.py(md5 8e122d36 고정)를 그대로 — 사전 파지(컵 축 2.5 R · 원점 높이 · 방위 회귀 · 손 방향 컵 틀 고정)
    6D IK 한 자세 → Mover 경유점(바깥 4 · 위 4 cm) 60 Hz 굴림. 값 = config/ready_approach/rh56f1.yaml(cff27786d cfg 사본, jitter 0).
    그 뒤 tilt 기준 쥔 자리까지 짧게(위치 직선 · 방향 구면 보간 · 손바닥 2 cm/s · 회전 15°/s · 관절 0.38 rad/s 평균 중 느린 쪽 최소 저크).
  · 오늘 칸 홈 출발: 오른손 src240b 계획 0.5 s · 이동 2.7 s + 들어가기 2.6 cm 5.5° 1.3 s · 손끝–컵 16.6 mm · 왼손 shaker 0.9 s ·
    4.0 s + 5.8 cm 7.6° 3.1 s(사전 파지에서 팔꿈치 j4 0.221 rad — 첫 칸에서 j4 0.076 rad 고쳐 잡음, 손바닥 0.45 mm 안) · 탁자 위 19 mm.
    관절 여유 최소: 이동 중 j6 7.8°(오) · 5.9°(왼), 들어가기 j7 18.9° · 12.0°.
  · fake 리허설(도메인 97) 양손 SUCCESS: 홈 → 배치 → 손 펴기 → reach → [확인] → hand_grasp(4 손가락 · 3 s) → lift(컵 10.0 cm).
    fake 컵(fake_cup_pose_pub --follow-side)이 fake 손 '손 안 컵'(/fake/hand_<팔>/cup) 켜진 동안 손바닥을 따라가 들기 판정까지 시험.
    (확인 대기 13 분으로 에피소드 900 s 제한에 걸린 첫 회 · 손 안 컵 없이 래치 0 · 컵이 안 따라와 0.0 cm 는 리허설 쪽 문제였다)
- 10.20 1-B 계획에 조건 추가(사용자 "1-b 세팅은 ready-carry 에 좀더 조건이 많아야함. 테이블, 자기 충돌, 관절 제약 등을 고려해야함").
  · 세계 = 홈 경로 계획기와 같은 것(`tools/reach_world.py` → home_path_world · plan_home_path.Checker): 1판 URDF 충돌 메쉬 · env USD
    테이블 상자 · 뒤 박스 · 옆 벽 · 몸통 · 머리 · 반대팔(실측 자세) 2 cm, 같은 팔 사슬 2 관절 이내는 관통만 금지, FP++ 컵 원기둥
    (이송 1 cm · 들어가기 관통 금지). 관절 여유 = URDF 와 robot_control 프로필 한계 중 좁은 쪽에서 5°. 세계 쌍 거리는 볼록 조각
    꼭짓점으로 다시 잰다(MuJoCo MPR 이 얇은 상판에 수평 '관통' −254 mm 를 냈다, 실제 13.6 mm 위).
  · 이송: aglt_new ReadyApproach(Mover) 경로가 걸리면 → 관절 공간 직선 → RRT-Connect + 지름길, 사전 파지 높이 × 방위 후보
    (0 · ±10 · ±20 · ±30°)를 차례로. 모두 걸리면 후보마다 이유를 남기고 거부. 들기(plan_lift)도 같은 세계 · 관절 검사.
  · 오프라인(홈 출발): 오른손 src240b (0.244, −0.116) ready_carry 그대로 2.5 s · 관절 여유 7.9°(j6) · 세계 13 mm(컵–엄지) ·
    왼손 shaker (0.259, 0.078) joint_line 10.8 s · 10.5°(j2). 중심선 컵(aglt_new 격자에서 j6 한계에 닿은 칸) 오 (0.25, 0 · −0.06) ·
    왼 (0.25, 0 · +0.06) 모두 통과(여유 ≥ 8.9°). 들기: 자기 사슬 45 · 47 mm, 관절 여유 18.9 · 23.0°.
  · fake 리허설(도메인 97) 양손 SUCCESS(reach → [확인] → 쥠 4 손가락 3 s → 컵 10.0 cm). 1-B · 홈 경로 테스트 59 통과.
- 10.20 02:06 · 02:08 실기 1-B reach 가 계획 전에 두 번 안전 정지(팔 안 움직임). 원인 = aglt_new hdgp 225abc056(02:03, begin() 방위 재시도를
  한 배치로 — 속도만)으로 ready_approach.py md5 8e122d36 → bb7b0aa8 → 배포 핀 불일치. 옛 · 새 파일로 같은 컵 11 자리 계획이 arm_target
  차이 0.0 rad 로 같아 핀을 bb7b0aa8 로 옮김(config/ready_approach/rh56f1.yaml). aglt_new 에 "바꾸기 전에 알림" 다시 요청.
- 10.20 사용자 "위치 세팅이 이상함 policy 창서도 보이게 해야지": 상황판 policy 탭 맨 위에 "제어 모듈 에피소드 — 1-B" 칸(작업
  ctl_grasp_b_right · _left, 정책 카드 없음). 실행 = 미션 단계 episode_grasp_b_<팔> 그대로 띄움 → 승인한 연속 실행(닫기 전 확인은
  지금 에피소드 패널 [다음]) → 끝(성공 · 실패 · 정지)이면 pd 붙듦 · 기록 마무리 · 보고 사진 · 영상. 시간 상한 600 s. rg3 카드 실행과
  같은 함수(_drive_episode)를 쓴다. 상황판을 다시 띄워야 보인다(로봇 스택이 상황판 자식이라 안전 종료 뒤).
- 10.20 실기 1-B 첫 회(03:25, 오른손 src240b · 왼손 shaker) 뒤 수정.
  · 사용자: "src240 은 엄지가 닿을 정도로 접근은 했는데 좀 더 깊숙이 들어가야 함. 손끝 쪽에 몰려 있어 파지할 때 컵이 밀려남 · shaker 는
    깊숙이 들어가서 잡힘. 리프트까지 잘 했는데 오류 남 · rehome 이 안 먹혀 제자리로 못 돌아옴(episode/stop rc=1)".
  · 원인: tilt 기준 쥔 자리는 손바닥을 컵 축에서 늘 52 mm 에 둔다 → 반지름 30 mm src240 은 손바닥–겉면 22 mm(손끝 파지), shaker(35.4)는
    15.7 mm. 들기 판정은 locate 때 찍은 …/pose 를 계속 다시 내는 토픽을 봐서 실제 9.6 cm 들림을 0.0 cm 로 판정 → 정지. rehome 은 에피소드
    실행기가 내려간 뒤 episode/stop 서비스가 없어 rc=1 로 멈췄고, pd 가 워치독 HOLD 라 engage 를 거부했다.
  · 고침: `plan_reach.deepen` — 손바닥–컵 겉면 수평 거리가 16 mm(shaker 깊이, GRASP_PALM_GAP_M)를 넘으면 그만큼 컵 축 쪽으로(src240b 오늘 칸
    +6.3 mm, 왼손 자리 +5.1 mm, shaker 0). 들기 판정 = FP++ live_pose. trigger.py: 서비스가 없는 episode/stop 은 멈출 것이 없음 = 성공.
    pd: 워치독 HOLD 에서 pd_engage 를 받아 풀고 붙듦. 구분 실행에서 마지막 노드 뒤 terminal 을 바로 닫는다(가짜 리허설이 done 에서 15 분 대기).
  · 확인: 1-B 시험 36 · 에피소드 139 통과. 가짜(97) 왼손 shaker 1-B(단계 실행) → lift 10 cm → control rehome_left 7 단계 27 s DONE.
- 10.20 정책 정리(사용자 "newton 세션하고 aglt_new 세션에서 전달해준 grasp, approach 정책들로만 policy 노드의 grasping, approach 탭에 존재하게
  해줘 그전 것들은 제거. 이것들을 기반으로 s2r 할 예정임").
  · 등록: rh56f1/aglt/right_approach_ap05(ap_05_r ep2300, bb9a1951) · left_approach_ap05(ap_05_l ep2600, 3be01dfa) · right_rcg02(rcg_02_nr
    ep1300, cc2cf6a4) · left_rcg02(rcg_02_nlm ep300, 7970864e). 관측 94 · 행동 13 · 손 position · 2판 손 USD(프로필 rh56f1_v2_*, hdgp 프로필은 이름 ·
    USD 경로만 다름). 계약 빌더가 rh56f1_v2_left 를 오른팔로 읽던 것 고침(이름 끝 _left). 4 개 모두 배포 체인 12 스텝 로드 확인.
  · 옛 9 개(rg3wo · neo03 · rg3hp3 · approach · approach_rg3wg) → `deploy/policies_retired/`(등록부 · 상황판 밖, 시험 고정본). 1-B 손 닫기
    계약은 `config/hand/rh56f1_<팔>_hand.json`(옛 approach 계약 그대로, md5 같음)으로 떼어 1-B 동작은 그대로.
  · 실행은 아직 막음(policy.yaml deploy.execution false): rcg 는 사전 파지 자세(ReadyApproach 접근 뒤 정착)에서 시작해야 하고, ap05 는 사전
    파지에서 멈춰 그 뒤 들어가기 · 닫기 · 들기가 필요하다. 옛 rg3 · 경로 C 에피소드는 시작 · 이음 자세가 달라 그대로 돌리지 않는다.
    grasp_rg3 · grasp_c · tilt A · C 에피소드는 옛 정책 id 그대로라 '정책 없음'으로 보인다. 미션 단독 정책 단계 기본 = ap05(홈 출발).
