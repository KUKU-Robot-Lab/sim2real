# VLM 감독 6단계 러너 설계안 (초안)

2026-11-19 초안. 사용자 승인 전이다. 정본 계획은 `PLAN_VLM.md` 의 "새 러너" 절이다. 구현과 커밋은 승인 뒤에 한다.

## 0. 지금 있는 것

새 러너의 순수 논리는 `vlm_supervisor/runner/` 에 이미 있다(11.17). ROS 몸체, 사람 승인 쓰기, 다음 노드 고르기는 없다.

| 부품 | 상태 | 근거 |
|---|---|---|
| BoundaryRunner (record · propose · auto) | 있음, 순수 논리 | boundary.py:60-96 |
| 사람 승인 읽기 Grants | 있음, 읽기 전용 | grants.py:1-4, 20-63 |
| 접점 FrameSource · Judge · ArmStill | 있음 | protocols.py:25-52 |
| Judge 구현 VlmHttpJudge | 있음. 2단 decide 는 아직 규칙 | vlm_http.py:76-81 |
| 센서 판정 SensorHub · combine | 있음 | signals.py:324-338, 418-450 |
| 시험 | 함수 30개 | tests/test_boundary_runner.py:107-437 |
| ROS 노드(서비스 · 상태 · 카메라 구독) | 없음 | |
| 사람 승인 줄 쓰기 도구 | 없음 | `vlm_grants` 는 grants.py 와 시험에만 나온다 |
| 전제 조건 후보 고르기 | 없음. yaml 순서 고정 | boundary.py:3 |

## 1. 기존 러너와의 경계

기존 러너(episode_runner_node + EpisodeManager)는 고치지 않는다. 새 러너는 EpisodeManager 를 감싸 step() 과 approve 콜백으로 돌린다(boundary.py:1, 81-86).

| 다시 쓰는 것 | 방식 | 근거 |
|---|---|---|
| EpisodeManager.step() | 노드 하나씩. run() 은 부르지 않는다 | episode_runner.py:149-162, boundary.py:148 |
| auto 노드 | propose · auto 는 auto 를 끈 사본으로 모든 노드를 경계로 만든다 | boundary.py:83-85, episode_runner.py:164-173 |
| lanes 노드 | 줄 전체가 경계 하나. 줄 실패는 복구 없이 정지 | episode_runner.py:248-250, 472-473 |
| 복구 대기 | 새 러너는 복구하지 않고 사람에게 넘긴다 | boundary.py:173-175, episode_runner.py:485-490 |
| Executor | RosExecutor · FakeExecutor 를 그대로 넘긴다 | episode_runner.py:80-87, episode_ros.py:411, episode_fake.py:45 |
| 에피소드 로그 | EpisodeManager 의 jsonl log 그대로 | episode_runner_node.py:10, 125 |
| record 모드 승인 | 기존 episode_approvals.jsonl 과 Approvals.take | episode_runner_node.py:41-66, 128-134 |
| 경계 사진 | 머리 카메라 `/camera/camera/color/image_raw/compressed` | interface.py:19, bench/real_frames.py:25 |
| 글로벌 사진(11.21, 있을 때만) | arm4090 D435i `/global_camera/camera/color/image_raw`(압축은 `/compressed`). FrameSource 는 `{"head": jpg, "global": jpg}`, 질문 `view: head\|global\|any`(기본 head) | scripts/vision/camera_up.sh:4 · deploy/policy_control/config/sites/arm_v2.yaml:35 · config/global_camera_extrinsics_arm4090_g1118.yaml |
| 팔 정지 | episode_ros.arm_still(frames, joints) | episode_ros.py:351-362 |

쓰지 않는 것도 있다. 옛 감독기 Interface.perform 은 사람 승인 파일에 `operator='vlm_supervisor'` 줄을 쓴다(interface.py:68, 106-112). 새 러너는 이 경로를 import 하지 않는다.

## 2. 경계 루프

노드마다 같은 순서로 돈다.

1. 승인 확인. propose · auto 는 노드를 돌리기 전에 사람 줄을 hold_limit_s 안에서 기다린다(boundary.py:140-143, 289-300).
2. 앞 사진. 실행 직전 2 s 안에 찍힌 것만 쓴다(boundary.py:31, 145).
3. em.step() 으로 노드 하나를 돌린다(boundary.py:146-150).
4. 러너 신호는 ok · code · reason · results 다(boundary.py:154-156). 센서가 확실한 실패를 내면 ok 를 False 로 내린다(signals.py:324-326, boundary.py:158-164).
5. FAILURE · STOPPED 는 VLM 을 거치지 않고 STOP 이다(boundary.py:169-170).
6. 팔이 멈출 때까지 기다린다. 뒤 사진은 멈춘 뒤 것만 쓴다(boundary.py:184-190, 198).
7. 1단. 노드 질문을 전후 사진과 함께 예/아니오로 묻는다(boundary.py:235, vlm_http.py:44-74). 질문은 judge_short.yaml 의 node_checks 에서 온다.
8. 사진 답과 러너 신호가 다르면 handoff 다(boundary.py:237-239).
9. 2단. 허용 행동 안에서 고른다. 러너 신호가 실패면 허용은 handoff · stop 뿐이다(boundary.py:195, 215-216).
10. look_again 은 경계당 1회다(boundary.py:197-214).

결정은 advance · look_again · handoff · stop 넷이다(protocols.py:7-8).

| 상황 | record(기록만) | propose(VLM 제안 + 노드별 사람 확인) | auto(에피소드 1회 승인) |
|---|---|---|---|
| 노드 시작 승인 | 기존 approvals 파일의 next 줄 | 그 노드 · advance 사람 줄(제안 시각 뒤) | 살아 있는 에피소드 줄 |
| auto 노드 | 기존처럼 이어 간다 | 경계가 되어 사람 줄 필요 | 경계가 되어 판정 뒤 진행 |
| confirm 노드 | 기존 그대로 | 노드 줄 | 노드 줄이 따로 필요(boundary.py:293) |
| VLM advance | 기록만 | 제안을 기록. 사람이 줄을 쓰면 진행 | 진행 |
| VLM handoff · stop | 기록만 | 넘김 · 정지. 승인 끝 | 넘김 · 정지. 에피소드 승인 끝(grants.py:60-63) |
| 판정이 진행을 막나 | 아니다. 스레드에서 기록(boundary.py:264-286) | 막는다 | 막는다 |
| 기록 실패 | 진행 그대로 | advance 하지 않고 handoff(boundary.py:312-316) | 같다 |

record 모드는 step() 으로만 돈다(boundary.py:98-105). 콘솔의 [연속 실행] 대신 [다음]을 쓴다.

## 3. 판정 합치기

러너 규칙과 에피소드 뒤 검토 규칙이 다르다.

| 경우 | 러너(boundary.py:237-243) | 에피소드 뒤 검토(bench/episode_review.py:179-212) |
|---|---|---|
| outcome 노드(grasp_ · place_) 사진 통과, 러너 실패 | handoff | 성공 + 과정 경고 |
| outcome 사진 판정 불가, 러너 성공 | handoff | 성공 |
| state 노드(이동 · 세팅) | 질문이 없으면 handoff(boundary.py:192-193) | 러너만 본다 |
| drop 노드(tilt_) | 어긋나면 handoff | 결과는 러너. 사진이 막으면 실패 |

6단계 첫 판은 러너 규칙(어긋나면 사람)을 그대로 쓴다. 놓치는 것보다 거짓 보류가 싸다.

11.21: 역할은 episode_review.role — state = go_home · scene · to_tilt_start · mix · home_*(★11.18 사용자, 사진으로 판정하지 않음), outcome = grasp_ · place_, drop = tilt_. judge_short.yaml node_checks 는 cocktail_e2e 노드 전부를 키로 갖는다(state 는 빈 목록이거나 시험대 측정용, 덮임 검사 test_episode_review.test_judge_short_covers_every_cocktail_e2e_node). lanes 노드 grasp_to_tilt_start 는 줄 항목 id 로 검사를 갖는다(줄마다 그 팔 손 검사만). 러너는 아직 질문 빌더가 없어 state 경계 · lanes 경계를 묻지 못하면 handoff 다. 질문 빌더는 lanes 노드에서 줄 항목의 검사를 합친다(signals.plan_checks 와 같은 방식, signals.py:378-380). state 노드를 "러너 성공이면 advance" 로 둘지는 사용자가 정한다.

## 4. VLM 이 사람 승인을 쓰지 못하게 하는 구조

| 막이 | 내용 | 근거 |
|---|---|---|
| 파일 분리 | 사람 줄은 vlm_grants.jsonl. 기존 episode_approvals.jsonl 과 다른 파일 | grants.py:1, episode_runner_node.py:41 |
| 쓰는 코드 없음 | 꾸러미 안에 승인 파일을 쓰는 코드가 없음을 시험이 검사 | test_boundary_runner.py:413 |
| 출처 거부 | operator 가 비었거나 "vlm" 으로 시작하면 버린다 | grants.py:38 |
| 범위 | episode_id 일치, 실행 시작 뒤, 마지막 종료 뒤, 미래 5 s 안의 줄만 | grants.py:13, 38-41 |
| 한 번만 | node 줄은 결정(노드 · 동작) 하나에 한 번 | grants.py:52-58 |
| 종료 | handoff · stop · 실패 때 그때까지 줄을 모두 죽인다 | grants.py:60-63, boundary.py:302-321 |
| VLM 위치 | VLM 은 server 에서 점수만 돌려주고 로봇 PC 파일에 닿지 않는다 | vlm_http.py:44-59 |

남는 구멍은 operator 문자열을 쓰는 쪽이 정한다는 점이다. 그래서 사람 줄 쓰기 도구는 deploy 쪽(콘솔 · CLI)에만 두고, 사람이 타이핑으로 확인해야 줄이 써지게 한다.

## 5. 답이 없거나 늦을 때

| 상황 | 기본 동작 | 근거 |
|---|---|---|
| VLM 마감 초과 | handoff | boundary.py:247-261 |
| 앞 경계의 판정 스레드가 아직 돈다 | handoff | boundary.py:201-204 |
| 사진 없음 · 빈 사진 · 시각 이상 | handoff | boundary.py:199-200, protocols.py:27-32 |
| 팔이 마감까지 안 멈춤 | handoff | boundary.py:187-188 |
| 사람 줄이 hold_limit_s 안에 없음 | handoff. 로봇은 그 자리, 에피소드는 READY | boundary.py:29, 297-298 |
| 판정 중 정지 | STOP | boundary.py:185-186, 249-262 |
| 센서 판정 오류 | handoff. VLM 성공만으로 가지 않는다 | boundary.py:176-177 |

시간 값은 후보만 적는다. 사용자가 정한다.

| 값 | 지금 기본 | 후보 |
|---|---|---|
| boundary_budget_s(경계 판정 마감) | 20 s(boundary.py:64) | 5 s(PLAN 5단계 기준), 10 s |
| VLM HTTP timeout | 15 s(vlm_http.py:23) | 질문 수 x 2 s. budget 보다 작게 |
| hold_limit_s(사람 줄 대기) | 30 s(boundary.py:64) | 30 s, 60 s |
| 물체를 든 채 handoff 뒤 상한 | 없음 | 60 s, 120 s |

## 6. 다음 노드 고르기

6단계는 yaml 순서 고정이다(boundary.py:3). 후보 고르기는 7단계에 붙이고, 구조만 지금 정한다.

- 후보는 지금 index 뒤 노드 가운데 전제 조건이 World 와 맞는 것이다.
- 기본 순위는 yaml 순서다. 후보의 첫째가 기본 선택이다.
- VLM 은 후보 목록 안에서만 고른다. 목록 밖 답은 handoff 다.
- 지금 Node 에는 전제 조건 칸이 없다. expect 는 노드가 끝난 뒤 World 에 덧쓰는 값이다(episode_runner.py:455, 564). 스킬 카드의 시작_조건(config/skill_cards.yaml:123 등)은 글이다. 어느 쪽에서 전제 조건을 만들지 정한다.
- em.index 는 EpisodeManager 밖에서 옮기지 않는다. 순서를 바꾸려면 EpisodeManager 에 다음 노드 지정 진입점이 필요하다(8절 7번).

## 7. sim · fake · 실기 공통 접점

세 환경 모두 `BoundaryRunner(ep, executor, mode, judge, frames, arm, questions, sink, grants_path 또는 approve, sensors)` 하나로 돈다(boundary.py:61-67). Executor 계약은 episode_runner.py:80-87 이다.

| 접점 | sim | fake(도메인 97/99) | 실기(도메인 50) |
|---|---|---|---|
| Executor | 없음. hdgp 브리지는 next · stop 과 셸 동사만 받는다(sim_interface.py:14-15) | RosExecutor + 가짜 드라이버, 또는 FakeExecutor | RosExecutor(episode_ros.py:411) |
| FrameSource | 브리지 /frame(sim_interface.py:42-50) 또는 저장된 경계 사진 | 저장된 실기 사진 재생 | 카메라 토픽 구독. 받은 시각을 monotonic 으로(interface.py:202-204) |
| ArmStill | 브리지 상태 | arm_still(pd 상태) | arm_still(/pd_state_broadcaster/joint_states) |
| Judge | VlmHttpJudge(server 사본 8767) | 같음 | VlmHttpJudge(server, 터널) |
| sensors | 없음 | 없음 또는 bag 재생 SensorHub | SensorHub(signals.py:418) |
| sink | jsonl | jsonl | 토픽 → bag(8절 6번) |
| 승인 | grants 파일 | grants 파일(허락 불필요) | grants 파일. 사람 도구만 쓴다 |

sim 은 둘 중 하나를 고른다. 브리지 위에 Executor 를 얹거나, 저장 사진과 FakeExecutor 로 배선만 본다.

## 8. 사용자와 s2r:deploy 가 정할 것

1. 실기 허락 문구. 안: "VLM 감독 에피소드 허락(auto)은 그 실행 안에서 VLM 이 advance 를 낸 노드만 덮는다. 복구와 confirm 은 덮지 않는다. handoff · stop · 실패 · 시간 초과로 끝난다. propose 는 노드마다 한 줄이다."
2. 물체를 든 채 handoff 될 때의 상한과 그 뒤 동작. 안: 상한까지 그 자세로 들고, 넘으면 경고만 낸다. 내려놓기나 홈처럼 로봇을 움직이는 동작은 허락 뒤에 한다. 상한 값은 손 발열을 보고 정한다.
3. 5절의 시간 값 네 개.
4. 복구(재시도 · 되돌아가기)를 auto 에피소드 허락에 넣을지. 지금은 넣지 않는다(boundary.py:173-175).
5. state 노드를 러너 신호만으로 advance 할지(3절).
6. 경계 기록 방식. 실기 기록은 rosbag 만 쓴다. 결정 행을 토픽으로 내서 rh56f1_record.sh 에 더할지, jsonl 을 예외로 승인할지.
7. deploy/policy_control 에서 바뀌는 곳.
   (a) 새 ROS 노드(서비스 next · run · stop · reset, 상태 토픽, 카메라 · pd · 손 구독)를 둘 자리.
   (b) 사람 승인 줄 쓰기 CLI. episode_cmd.py:132 와 같은 꼴로 vlm_grants.jsonl 에 쓴다.
   (c) 7단계용 다음 노드 지정 진입점.
   (d) record 모드의 runner 이름. 기존 이름을 쓰면 콘솔 [다음]이 그대로 붙고, 다르면 승인 줄의 runner 칸이 갈린다(episode_runner_node.py:61).
8. 콘솔에 붙는 위치. 에피소드 버튼(console.py:998-1019) 옆에 노드 승인 · 에피소드 승인 버튼과 VLM 제안 표시를 둔다. 콘솔은 사용자 소유라 제안만 한다.
9. 카메라 경로. 머리 카메라가 4090 에 있으면 5090 이 압축 영상을 랜으로 받는다. 4090 에서 5090 으로 가는 랜이 찼을 때 손 설정이 실패한 기록이 있다. 빈도와 해상도 상한을 정한다.

## 9. 구현 순서와 시험

| 순서 | 일 | 시험 | 실기 허락 |
|---|---|---|---|
| 1 | 질문 빌더(node_checks, lanes 합치기, state 규칙) | cocktail_1c_e4000 의 모든 노드가 질문이나 규칙을 갖는지 보는 단위 시험 | 불필요 |
| 2 | 저장 사진 FrameSource + FakeExecutor 로 배선 확인 | 11.16 실패 장면 재생에서 놓침 0 | 불필요 |
| 3 | ROS 노드 + 사람 승인 CLI | fake 도메인에서 record, propose, auto 순서. 승인 파일의 vlm 줄 0 | 불필요(fake) |
| 4 | hdgp sim 브리지 Executor(정하면) | PLAN 3단계 주입 장면 | 불필요 |
| 5 | 실기 record | 사람이 [다음]으로 승인하는 실행 옆에서 판정만. PLAN 5단계 기준 | 매번 |
| 6 | 실기 propose, 다음 auto | PLAN 6단계 기준 | 매번 |

## 확인하지 않은 가정

- 실기 머리 카메라가 4090 에 있어 5090 이 랜으로 영상을 받는다(메모리 arm_v2 기준, 코드 확인 안 함).
- handoff 로 READY 가 됐을 때 pd 가 마지막 목표를 붙들고 손이 쥔 힘을 유지하는지.
- hdgp sim 브리지에 노드 단위 Executor 를 얹을 수 있는지.
- record 모드에서 콘솔 [연속 실행]을 못 쓴다는 점(코드 읽기로만 판단).
- `sim_interface.py` 의 /frame 시각을 BoundaryRunner 시계로 바꾸는 방식이 FrameSource 계약을 만족하는지.
