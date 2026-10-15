# Claude 인수인계: S2R 콘솔 개편과 rg3 운용 연결

## 먼저 확인할 위치

- **작업 worktree:** `/home/user/rl_ws/wt_console_rg3`
- **브랜치:** `feat/s2r-console-workspace`
- 기준 커밋: `9dd4887`. 최종 상태는 `git status --short`와 `git log -8 --oneline`으로 확인한다.
- **원본 `/home/user/rl_ws/sim2real/main`에는 사용자 작업이 있다.** 이 작업을 원본에 덮어쓰거나 임의로 병합하지 않는다.
- 원본에 이미 있던 변경: `hand_block.py`, `pour_fj_node.py`, `rh_aglt.py`, `rh_place.py`, 별도 정책 로그와 `test_pc_hand_block_judge.py`. 콘솔 개편 범위와 분리한다.
- 운용 설명: `deploy/s2r_console/README.md`.

## 사용자 의도와 유지할 조건

setting / control / policy 탭으로 나누고, 상단 control은 모든 탭에서 접근한다.
기반 연결 → 노드 | 타이밍 | FP++ 영상 한 행 → 양팔·양손 관절 상태 → 관리 탭 순서다.
기존 로봇 정보 패널은 제거하고 FPP 상세는 policy 인지·물체 영역으로 옮겼다.

- setting: preflight, drivers, 측별 손 연결→점검→PD 무발행→발행 준비 묶음, head_home, cups, shutdown.
- control: engage, home, rehome, return, 목 자세와 보정 프로필, 선택 물체 인지 재추출. 완료 단계 순서 대신 현재 연결·PD·실측 상태로 가드한다.
- 승인: 단계 ID 입력 없이 **취소 / 승인 후 실행**. 확인 ID는 run·lease·표시 명령·정책/계약 파일·물체·입력원에 묶인 일회용이다. 기존 API는 호환 유지한다.
- 정책 중 control: 관련 정책 정지 → **각 대상 팔** PD 현재 자세 유지 확인 → 요청 수행. 자동 정책 재개 금지.
- 목 스캔 중 새 명령: 협력적 취소 후 현재 이동과 운용 자세 복귀의 **실측 정착 성공**을 확인해야 한다.
- src200 / src240 / shaker × 하늘색·주황·핑크. 측정 HSV 보존, 추가 조합은 기존 프리셋과 미측정 표시. holder는 마커별 카드, 색 locate 미지원.
- FPP / locate 결과·좌표계·시각·출처·유효 상태 분리. 입력원 자동 대체 금지.
- approaching / grasping / placing / locking / pouring 카드는 등록 파일 선택·배치·이동·제거·로봇별 복원을 지원한다. 파일 선택/복원으로 실행하지 않는다.
- **실행 연결은 rg3만. FPP의 신선한 live_pose 필수. locate만으로 실행 금지. FK 추정을 강제로 켜지 않는다.** 물체를 source200_real / source240_real / shaker_c_real에 정확히 대응시킨다.
- 실기 실행은 하지 않았다. 실기 수락은 별도 승인 후 연결 → 자세 제어 → 영상·인지 → rg3 순서다.

## 변경 파일 지도

| 역할 | 주요 파일 |
|---|---|
| 화면·탭·카드·승인 모달·안정적 영상 DOM | `deploy/s2r_console/s2r_console/web/{index.html,console.js,console.css}` |
| 상태 통합·run 수명·로봇 토픽 | `deploy/s2r_console/s2r_console/console.py` |
| 등록 명령 카탈로그·묶음·우선권·rg3 실행/정리 | `deploy/s2r_console/s2r_console/actions.py` |
| 일회용 승인·변경 감지 | `deploy/s2r_console/s2r_console/confirmations.py` |
| workspace 갱신·prepare/execute 경로 | `deploy/s2r_console/s2r_console/api.py` |
| MJPEG 중계 | `deploy/s2r_console/s2r_console/server.py` |
| 물체 선택·결과·정책 카탈로그·배치 저장 | `deploy/s2r_console/s2r_console/workspace.py` |
| RH56F1 양손 토픽·관절별 수신 시각 | `bridge.py`, `feed.py`, `console_state.py`, `robot_view.py`, `robots.py`, `robots/openarm_rh56f1.yaml` |
| 색 등록 | `config/objects.d/{source200,shaker_c}.yaml` (기존 source240/source240b 조합 재사용) |
| 기존 locate 코어 어댑터·운용 자세 복귀·취소 | `scripts/ops/registered_locate.py` |
| RH56F1 미션 원본 | **`scripts/ops/make_rh56f1_missions.py`** — real/fake YAML 직접 수정 금지 |

배치는 `runs/s2r_console/workspaces/<로봇 해시>.json`에 저장한다.
영상 기본 주소는 `http://127.0.0.1:8080/stream`; 서버 시작 전 `S2R_VIDEO_URL`로 운영자가 변경한다.
locate는 형제 checkout `perception_plus_plus/perception_plus_plus_core/head_scan`을 재사용한다.

## 마지막 리뷰와 보완

독립 리뷰에서 다음 다섯 항목을 발견했고 보완 중이다. **최종 커밋·검증 기록을 아래에서 확인한 뒤 완료 여부를 판단한다.**

1. FPP 재추출 전에 활성화한 모든 물체 그룹을 실제 시작해야 한다. 개별 재추출 시 다른 활성 그룹을 끄지 않는다.
2. 목 이동은 기존 단독 포트 소유자 `head_joint_publisher.py`와 `head_move.py` 요청 규약을 사용해야 한다. FPP 변환에는 `/head/joint_states`와 선택한 보정 프로필을 연결한다. 이동/프로필 변경 후 이전 프레임 좌표는 무효다.
3. scan 취소 후 스레드 종료만으로 진행하면 안 된다. `home_ok` 등 복귀 정착 성공 증거를 확인한다.
4. 정책 정지 대상 팔의 PD가 stale이면 다른 팔의 신선한 상태로 대신하지 않는다.
5. 한쪽만 연결된 상태의 shutdown: PD 프로세스/그래프 생존 판정을 팔별로 분리한다.

추가한 프레임 무효화 훅: `Workspace.invalidate_results(reason, sources=('fpp','locate'))`.
FPP 수신 시각이 아니라 **원본 획득 stamp**를 기준으로 이전 프레임을 배제한다.
목 tilt 상한은 기존 `head_motion.py`의 사용자 한계 **-10°**를 따른다.

## 검증 기록과 재현

- 마지막 리뷰 보완 전 관련 회귀: **653 passed, 2 skipped**.
- 브라우저: fake / no-bridge 상태에서 실제 Chrome CDP로 15개 상호작용 확인. 탭·상단 control·추가/이동/복원/제거·영상 DOM 유지·승인 취소/중복 등을 확인했다.
- 전체 `pytest -q`: **2752 passed, 123 skipped, 37 failed**. 통과로 표현하지 않는다.
- 같은 37개를 원본 checkout에서 재실행: **35 failed, 2 passed**. `rl_games` / `warp` 누락 및 기존 `test_cup_holder_pose.py` 수치 실패가 재현된다. 나머지 2개는 worktree에 없는 원본의 비추적 `logs/policy/t2r_rh5_f01_ep800/trace_meta.json`, `t2r_rh5_f02_ep3000/trace_meta.json`에 의존한다.
- 전체 실패 목록과 로그: `/tmp/s2r-full-suite.log`, 원본 재현 `/tmp/s2r-existing-failures.log`.
- 원본 정책 가중치 등 49개 대형 자산은 worktree에서 심볼릭 링크로 참조했다. 가중치를 Git에 추가하지 않는다.
- `pip_audit` 실행을 시도했으나 모듈이 설치되어 있지 않다. 의존성 감사 완료로 표현하지 않는다.
- 시스템 SciPy/NumPy와 requests 의존성 버전 경고가 있다.

```bash
cd /home/user/rl_ws/wt_console_rg3
pytest -q tests/s2r_console tests/test_object_registry.py tests/test_head_home.py
python3 scripts/ops/make_rh56f1_missions.py --check
node --check deploy/s2r_console/s2r_console/web/console.js
ruff check deploy/s2r_console/s2r_console scripts/ops/registered_locate.py
# 전체 저장소 검증은 pytest -q. 위 환경/기존 실패도 함께 기록한다.
```

## 최종 마무리 기록

최종 보완 테스트와 커밋 결과를 여기에 덧붙인다. 이 절에 완료 기록이 없으면
위 다섯 리뷰 항목의 최신 diff와 테스트를 먼저 확인하고, 실기 실행/병합을 진행하지 않는다.

### 2026-10-15 16:30 마무리 (Claude s2r_1st 가 이어받음)

- 위 다섯 리뷰 항목의 보완 diff 를 확인하고 커밋했다: **`aaa3227`**.
  - 실기 RH56F1 그림에 목 드라이버 상자(`목 상태 (head)`)가 생겨 `test_the_fake_rh56f1_picture_has_the_real_pictures_boxes` 가 실패했다.
    fake 에는 목이 없으므로 카메라처럼 실기 전용 상자로 허용 목록에 넣었다.
- 브라우저에서 새로 찾은 결함을 고쳤다: **`755b7ab`**.
  - 서버는 `workspace` · `actions` · `management` 를 `session` 아래로 보내는데, 화면은 최상위에서 읽었다.
    그래서 setting 칸에 `{}` 만 보이고, control 버튼과 policy 물체 선택이 비어 있었다(`82f956c` 통합 뒤 생긴 문제).
  - 관리 버튼은 이제 작업 id(home_right, rehome_left, head_control …)를 굵게 쓰고, 설명은 두 줄로 줄인다.
  - 관리 진행 상황은 원시 JSON 대신 글 한 줄로 보인다.
- 검증(fake 97, `--no-bridge`, 실기 실행 없음):
  - `pytest -q tests/s2r_console tests/test_object_registry.py tests/test_head_home.py` → **665 passed, 2 skipped**.
  - `make_rh56f1_missions.py --check` 일치. `node --check` 통과. ruff 통과.
  - 헤드리스 Chrome(CDP)으로 탭 세 개를 캡처했다.
    - setting: preflight · drivers · head_home · cups · shutdown · connect_right/left.
    - control: home · rehome · return · engage(좌우) · head_control · perception_refresh · policy_stop.
    - policy: FPP/locate · 물체×색 체크 · 정책 노드 · ＋task.
  - 승인 창은 단계 ID 입력 없이 **취소 / 승인 후 실행** 으로 뜨는 것을 확인하고 취소했다.
- 확인하지 못한 것:
  - 실기 브리지에서 RH56F1 손 상태가 실제로 표시되는지(fake no-bridge 라 미수신만 보임).
  - rg3 실행 연결의 실기 동작.
  - 전체 `pytest -q` 재실행.
- 병합과 push 는 하지 않았다. 브랜치 `feat/s2r-console-workspace` 에만 있다.

