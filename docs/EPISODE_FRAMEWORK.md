# 에피소드 실행기 (Step 1 정상 · Step 2 실패 처리) — 10.04

근거: `.orca/drops/sim2real_episode_implementation_guide.md`(사용자 10.04) · 사용자 결정 "상황판과 연계 · 시험은 구분 실행,
최종은 한 번에(중간 사용자 개입 없이) · 처음에 FP++ 로 컵 배치를 한 번에 기록 · 컵홀더는 고정". VLM(Step 3)은 자리만 둔다.

## 구조 — 기존 계층 위에 한 층

```
config/episodes/<이름>.yaml ── episode_spec(검증 · 역할 → 등록부 정책)
          │
   episode_runner_node (ROS) ── EpisodeManager(episode_runner) ── FailureDetector · RecoveryManager(episode_failure)
          │                                  │                        WorldState(episode_world)
          │                         RosExecutor(episode_ros)
          ├─ 정책  rh_aglt_node(서비스 ns <팔>) · rh_place_node(ns <팔>_place) — 한 팔에 둘이 같이 뜬다.
          │        이벤트는 둘 다 /policy_control/<팔>/episode 에 node 이름을 실어 낸다(pd 가 reset · stop 을 그 팔에 적용)
          ├─ 궤적  plan_rehome(--robot rh56f1, RRT) → check_path_start → replay_to_pd → pd goto_home
          ├─ 배치  물체 FP++ 토픽 1.5 s → /episode/objects/<이름>/pose 재발행 · 고정 홀더 자세 파일 → /objects/cup_holder_<id>/pose
          └─ 안전  pd HOLD · estop 감시 · safe_stop = episode/stop(pd 가 붙든다) + 재생 SIGTERM
상황판(s2r_console) ── 미션 단계 episode_<이름>(노드를 띄운다) · 에피소드 패널 · /api/episode → tools/episode_cmd.py
```

| 가이드 | 여기 |
|---|---|
| Episode YAML · EpisodeManager · StateMachine | `config/episodes/*.yaml` · `policy_control/episode_spec.py` · `episode_runner.py` |
| WorldState | `episode_world.py` — 자세 · 두 손 물체 · 진행 플래그 · 물체 자리(테이블 · 손 · 홀더) |
| PolicyExecutor · ParallelExecutor | `episode_ros.RosExecutor.run_policies`(팔마다 스레드, 같은 팔 동시 실행은 정의에서 거부) |
| TrajectoryExecutor | `run_trajectory` — 직선 goto_home 이 아니라 실측에서 RRT 재계획(09.28 손끝이 상판을 지난 기록) |
| FailureDetector · FailureCode | `episode_failure.py` — 계열(kind)별 규칙, 정책 이름 분기 없음 |
| RecoveryManager · Retry · Checkpoint · SafeStop | 같은 파일 — 아래 표 |
| 로그(가이드 1-10) | `logs/episode/<시각>_<이름>/episode_<id>.jsonl` · `scene.json` · `tools.log` |

## 실행 — 구분 실행 → 연속 실행

1. 상황판에서 미션 `rh56f1_real` 을 열고 홈 · 컵(FP++) · 홀더(`cup_holders --write`) · pd 발행 단계를 끝낸다.
2. 단계 `episode_pick_place_right` 실행 → 정책 노드 둘(aglt `stop_on_target` · 컵 = snapshot 재발행, place) + `episode_runner_node`.
3. 에피소드 패널
   - **[다음: <노드>]** 구분 실행 — 그 노드 이름을 입력해야 한다. 실패하면 다음 명령이 복구 하나(이름 `recover:<노드>:<동작>`).
   - **[연속 실행]** `episode:<이름>` 을 한 번 입력 — 끝까지. 복구도 그 승인 안이다(재시도 상한 · 안전 정지는 그대로).
     실패로 멈추면 승인은 끝난다 — 다시 하려면 [새 에피소드] 뒤 새로 승인.
   - **■ 에피소드 실행기 정지** lease 없이 언제나. 정지 바의 '에피소드 정지 · 중단'도 실행기를 같이 멈춘다
     (정책 노드만 멈추면 연속 실행이 그것을 실패로 보고 복구 동작으로 이어 간다). 사람이 멈춘 정책은 복구하지 않는다.
4. CLI 도 같다: `tools/episode_cmd.py status|next|run|stop|reset --approve <이름> --execute`.
   모의: `tools/episode_run.py --episode … --plan | --dry-run [--inject 노드=코드] [--step]`.
   fake 리허설: `ROS_DOMAIN_ID=97 deploy/s2r_console/tools/run_fake_episode.py --episode pick_place_right --mode step|run`.

승인 규칙: 실행기는 승인 없이 아무것도 움직이지 않는다(승인 없음 = 실행 안 함, 실패 아님). 승인은 `logs/episode/approvals.jsonl`
한 줄 = 한 번, 실행기가 뜬 뒤 쓰인 것만. `auto_approve` 는 fake 도메인에서만(126 이면 노드가 거부).

## 첫 실기 순서(pick_place_right)

| # | 상황판 단계 | 허락 | 볼 것 |
|---|---|---|---|
| 1 | preflight · drivers · hand_right · hand_check_right | drivers 는 운영자(전원 · CAN) | CPU 실시간 한도 · 손 250 Hz |
| 2 | head_home | 단계 승인(머리가 움직인다) | 카메라 화면이 5090 홈과 같은가 |
| 3 | cups | 단계 승인 | /objects/cyl60/pose 가 들어오는가 · 컵은 aglt 배치(x ≈ 0.25, y ≈ −0.20 ± 0.1) |
| 4 | cup_holders | — | `--write` 가 `config/cup_holder_poses_arm4090.yaml` 을 쓴다(★없으면 에피소드 시작 검사가 멈춘다) |
| 5 | pd_load_right → pd_arm_right → home_right | 단계마다 승인 | pd TRACKING · 홈 정착 |
| 6 | episode_pick_place_right | 단계 승인 | 시작 검사(정책 · 홀더 파일 · 남은 노드) → bag 기록 시작 → 노드 · 실행기 |
| 7 | 에피소드 패널 [다음] × 노드 수 | 노드 이름 입력(구분 실행) | 홈 → snapshot → 집기(SETTING 도달) → 놓기(홀더 1) → 홈 |
| 8 | 에피소드 패널 [연속 실행] | `episode:pick_place_right` 한 번 | 7 이 한 번 끝까지 된 뒤에 |

놓음 문턱(손 목표 0.15 rad · 관절 힘 300 g)은 6 의 bag(손 · 정책 · 실행기 상태)으로 정한다.
화면 확인: `node deploy/s2r_console/tools/screenshot_cdp.mjs http://127.0.0.1:8091/ out.png`(chrome --screenshot 는 SSE 상황판을 못 담는다).

## 에피소드

| 파일 | 순서 | 실기 |
|---|---|---|
| pick_place_right / left | 홈 → snapshot → 집기(aglt → SETTING) → 놓기(홀더 1) → 홈 | 가능 |
| pick_place_both | 홈 → snapshot → 양팔 동시 집기 → 놓기 우 → 놓기 좌 → 홈 | 두 컵 구분(색 · FP++ 항목) 뒤 |
| bead_mix_episode | 가이드 1-2 전체 | 모의만 — 붓기 성공 후보 · 뚜껑 · 흔들기 정책 · 색 컵 인지 없음 |

SETTING = aglt 고정 목표 (0.25, ∓0.12, 0.41)(손에 든 컵 원점) — rh_place · pour 인계 뱅크가 이 자세에서 만들어졌다.

## 실패 → 복구(Step 2)

| 실패 | 판정(신호) | 복구 |
|---|---|---|
| TARGET_NOT_FOUND / HOLDER_NOT_FOUND | 거부 사유 "컵 자세가 없다" · "holder pose missing" · snapshot 프레임 없음 | 인지 갱신 후 재시도(물체 = 프레임 다시 기다림 · 홀더 = 자세 파일 다시 읽어 발행) |
| GRASP_FAILED_* | aglt 끝 status cup.source ≠ attached | 손 펴기(pd hand_release) → 빈손 홈 checkpoint 로 RRT → 배치 재기록 → 재시도 |
| POLICY_TIMEOUT | 이벤트 "episode time ≥" · 실행기 시한 | aglt 만 위와 같이, 나머지 정지 |
| POSE_MISMATCH | 거부 "rad from the training start pose" · "held target" · SETTING 오차 > 4 cm | 빈손 홈 checkpoint 가 있으면 되돌아가 재시도, 아니면 정지 |
| OBJECT_DROPPED_* | aglt cup.releases > 0 | 정지(장면 재평가 = Step 3) |
| PLACE_FAILED | 놓은 뒤 FP++ 컵이 홀더 자리 밖(xy 3.5 cm · z 3 cm) · 손이 안 비었다 | 정지(재파지 기술 없음) |
| POUR_* · LOCK_FAILED · SHAKE_ABORTED | 정책이 생기면 신호를 붙인다 | 표대로 |
| CONTROLLER_ERROR · OBSERVATION_ERROR | pd HOLD · estop · 노드 오류 · 결손 | 복구 없이 즉시 정지 |

재시도 상한은 에피소드 `failure_policy`(기본 1, aglt 2, place 0). 컵을 든 채 홈으로 돌아가지 않는다(계획기가 든 컵을 모른다).

## 남은 것

- 첫 실기: 구분 실행으로 pick_place_right(노드마다 승인) → 연속 실행. 놓음 문턱(0.15 rad · 300 g) bag 확인.
- 두 컵 이상: 색이 다른 컵을 FP++ 레지스트리 항목으로 — pick_place_both · bead_mix 의 실기 조건.
- 붓기(rh_pour)는 성공 후보가 나오면 붙인다(10.04 사용자). 뚜껑 · 흔들기 정책 · VLM(Step 3)은 그 뒤.
