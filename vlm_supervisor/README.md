# 독립 VLM 감독기

`vlm_supervisor`는 문장으로 계획을 만들고, runner의 노드 종료 뒤 사진과 상태로 다음 행동을 결정한다. 기본 모드는 **shadow**이며 승인 파일과 Trigger 서비스를 건드리지 않는다. `active`를 명시하면 승인 append 및 서비스 호출을 수행한다. `deploy` 코드를 import하거나 수정하지 않는다. `proto/`는 비교용 원형이며 실행 패키지는 이를 import하지 않는다.

## 실행

저장소 루트에서 실행한다. Python 3.10 이상과 `pyyaml`, `pillow`, `requests`가 필요하다. 실기 연결은 ROS 2 환경을 먼저 source한다. ROS가 없어도 패키지 import, 계획 검사, 판정 파싱, 단위 테스트, `--help`는 가능하다.

```bash
python -m vlm_supervisor --runner episode_runner --mode shadow \
  --sentence '블루컵과 핑크컵의 비드를 오렌지 쉐이커 컵에 옮겨줘' \
  --approvals /실제/실행기/logs/runs/episode_approvals.jsonl
```

`--mode active`로 실행 권한을 켠다. `--approvals`는 **runner가 실제 읽는 같은 파일**을 지정해야 한다. 기본값은 현재 디렉터리 기준 `logs/runs/episode_approvals.jsonl`이다. 다른 호스트에서 실행한다면 같은 승인 파일이 보이도록 마운트해야 한다. 서비스는 같은 ROS 도메인의 runner로 연결된다.

계획은 기록·검사용이다. 문장 계획은 **runner의 기존 순서를 자동 교체하지 않는다**. 개별 복구 노드 삽입은 별도 shell 명령으로 수행한다. 초기 사진 → 카드 기반 VLM 계획 → simulate 통과 뒤 READY 루프를 시작한다. 문장과 실제 runner 에피소드의 일치는 운영자가 확인해야 한다. 계획 검사 실패 시 active는 stop을 요청하고 종료한다.

## 인터페이스 계약

배포 구현이 바뀌면 아래 계약을 유지하거나 `interface.py`의 어댑터를 수정한다. ROS 접근은 interface와 shell 어댑터에 한정한다.

| 방향 | 통로 | 형식/의미 |
|---|---|---|
| 읽기 | `/policy_control/status/<runner>` | `std_msgs/String`의 JSON. 필수 키: `status,node,world,nodes,last,pending,history` |
| 읽기 | `/camera/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage`. sensor-data QoS로 수신. 노드 종료 상태 관찰 후 `--settle`(기본 0.5초) 이후 수신한 사진을 판정에 사용 |
| 쓰기 | `logs/.../episode_approvals.jsonl` | UTF-8 JSON 한 줄 append. `{what,at,operator,runner}` 및 기존 호환 필드 `t` 포함 |
| 호출 | `/policy_control/<runner>/next` | `std_srvs/Trigger`. READY의 현재 `node` id 승인 후 한 단계 실행 |
| 호출 | `/policy_control/<runner>/run` | `std_srvs/Trigger`. 인터페이스에서 노출. 감독 루프는 단계별 판정을 위해 `next`만 사용 |
| 호출 | `/policy_control/<runner>/stop` | `std_srvs/Trigger`. 위험/불명/미지원/실패 시 정지 요청 |
| 호출 | `/policy_control/<runner>/reset` | `std_srvs/Trigger`. 인터페이스에서 노출. 자동 복구로 사용하지 않음 |

상태 필드 의미:

| 키 | 계약 |
|---|---|
| `status` | 문자열. `READY`에서만 판단·승인. `SUCCESS/FAILURE/STOPPED`는 종료. 그 외는 대기 |
| `node` | 다음 실행할 현재 노드 id 문자열 또는 `{id: ...}`. 다음 동작이 없으면 null |
| `world` | 객체. 실행기가 선언한 상태이며 영상 검증 결과와 구별 |
| `nodes` | 노드 목록. 전체 문맥으로 VLM에 전달 |
| `last` | 직전 완료 노드의 `{node,code,reason,...}` 객체 또는 null. 초기에는 빈 객체/null |
| `pending` | null이면 일반 진행. 복구 대기가 있으면 자동 승인하지 않고 미지원 처리 |
| `history` | 완료 이력 배열. 반복 시도 식별에 사용 |

추가 필드 `busy`와 `attempts`가 있으면 각각 실행 중 승인 방지와 중복 판정 식별에 사용한다. `next_action`은 필요하지 않다. 원형 `fake_runner.py`는 `world/history`가 없으므로 새 계약으로 시험하려면 두 필드를 보충해야 한다.

승인 예시(기존 원형과 동일):

```json
{"what":"place_blue","t":1794300000.0,"at":"2026-11-10 17:40:00","operator":"vlm_supervisor","runner":"episode_runner"}
```

`t`는 Unix 초(float), `at`는 로컬 시간 `%F %T` 문자열이다. 승인 대상은 `node` id이고 `next/run` 직전에 기록한다. Trigger는 `success=true`가 필요하며, `message`가 JSON 객체라면 `ok=false`도 실패로 취급한다. `reasons`를 기록하고 실패하면 stop을 요청한다. 파일 append와 서비스 호출은 원자적 트랜잭션이 아니며, 다른 승인 주체와의 동시 실행은 runner 측에서 관리해야 한다.

## 판단과 미지원 훅

| 선택 | 의도 | 현재 active 동작 |
|---|---|---|
| A | 성공 → next | 현재 node 승인 후 next. pending이 있으면 미지원 처리 |
| B | 실패 → retry | shell에 직전 실패/판정 대상 노드 id 전달, 재기동 READY 후 루프 계속 |
| C | 실패 → rescan | shell restore snapshot 전용 세대 실행 |
| D | 실패 → home | 실패 노드의 팔을 찾아 원본 single-arm rehome 삽입 |
| E | 위험/불명 → stop | stop 요청 → 사람 확인 |

JSON 파싱 실패, 모순된 success/choice, 사진 부재, 백엔드 실패는 E다. VLM 판정 동안 상태가 바뀌면 결과를 기록하고 승인을 취소한다. 같은 완료 상태는 한 번만 처리한다. 상태 수신이 10초 이상 끊기면 active는 stop을 요청한다. stop 응답 실패 역시 기록하며 물리 정지 성공을 가정하지 않는다.

## Episode shell

별도 노드 `vlm_shell_<runner>`가 runner 프로세스를 소유한다. 기존 runner를 먼저 종료하고 shell을 실행한다. 동일 이름 runner를 따로 띄우지 않는다. shell은 deploy를 import하지 않으며 검증 도구와 runner를 subprocess로 실행한다.

```bash
python -m vlm_supervisor.shell --runner episode_runner_right \
  --episode config/episodes/cocktail_1c_e4000.yaml --run-dir logs/vlm_shell/right \
  --runner-cmd python3 deploy/policy_control/policy_control/episode_runner_node.py \
  --ros-args -p robot:=rh56f1 -p approvals:=logs/runs/episode_approvals.jsonl
```

`--runner-cmd`는 마지막 옵션이며 이후 인자를 모두 runner에 전달한다. 생략하면 현재 Python과 저장소의 runner 스크립트를 사용한다. `ros2 run` 또는 `python -m` 명령도 지정할 수 있다. shell은 끝에 episode 절대 경로와 동일 `__node` 이름을 ROS override로 추가한다. robot/approvals/policy_object/cup_object 등은 runner 명령에 지정하면 재기동마다 유지된다. shell과 supervisor는 같은 approvals 파일을 사용해야 한다. 새 승인은 새 runner READY 이후 작성하므로 시작 시각 이전 승인을 재사용하지 않는다.

| 방향 | 토픽/계약 |
|---|---|
| 명령 | `/vlm_shell/<runner>/cmd`, `std_msgs/String` JSON `{cmd_id, verb, args}` |
| 응답 | `/vlm_shell/<runner>/ack`, JSON `{cmd_id, ok, reasons, generation, episode_path}` |
| 상태 | `/vlm_shell/<runner>/status`, JSON `{generation, episode_path, lineage, runner_pid}` (0.5초 간격) |
| retry | `args: {node_id: "tilt_blue"}` — 해당 노드부터 끝까지 |
| rescan | `args: {}` — restore snapshot만 실행하고 terminal SUCCESS. 이후 작업 재개는 원본 node_id를 지정한 별도 retry 명령과 감독기 재실행이 필요 |
| home | `args: {side: "right"}` — 현재 노드 앞에 원본의 같은 팔 rehome 노드 복사. 생략하면 직전 실패 노드에서 팔 추론 |
| insert | `args: {spec: {id: "check", type: "confirm", text: "확인"}}` — 현재 노드 앞에 삽입 |
| human | `args: {text: "컵을 확인하세요"}` — 현재 노드 앞에 confirm 삽입 |

처리 순서는 busy/상태 검사 → stop 성공 확인 → 프로세스 그룹 SIGINT 및 대기(8초 뒤 SIGTERM, 추가 5초 뒤 SIGKILL) → `run_dir/shell/gen_<n>.yaml` 저장 → `episode_run.py --plan` subprocess 검증 → 같은 이름 재기동 → 새 READY 확인 → ack다. 실패하면 직전 성공 세대 YAML로 재기동하며 ack는 false이고 rollback 실패 이유도 포함한다. 세대 번호는 실패 시에도 파일 덮어쓰기를 피하도록 증가하므로 빈 번호가 생길 수 있다. 같은 cmd_id 응답은 shell 프로세스 수명 동안 캐시하고 재실행하지 않는다. 계보와 cmd_id 캐시는 메모리에 있으므로 shell 자체 재기동 간에는 이어지지 않는다.

active 인터페이스는 shell을 찾는 데 3초, ack에 최대 180초 기다린다. shell이 없거나 ack가 없으면 기존 `NotSupported + stop` 처리한다. 명령 실패 ack도 감독기가 stop하고 종료한다. shadow는 shell에 쓰지 않는다. retry/home 성공 후에는 새 READY 루프를 계속한다. **confirm 노드는 감독기가 자동 승인하지 않는다.** 운영자가 기존 승인/next 통로로 확인해야 진행한다.

파생 YAML은 나머지 최상위 설정을 복사하고, 기존 손 물체와 홀더 물체를 제외한 snapshot에 pose/두 손/참인 task_flags를 expect로 복원한다. 물체 좌표는 저장된 수치를 주입하지 않고 snapshot으로 다시 측정한다. deploy 문법상 명시적 snapshot objects 앞에 confirm이 필수이므로 `shell_confirm_<n>`을 추가한다. rescan에도 마지막 terminal이 필수이므로 `shell_done_<n>`을 덧붙인다. 따라서 restore 확인도 운영자 승인이 필요하다.

한계:

- 재기동마다 runner run_dir/episode_id가 바뀌며 checkpoints/attempts는 세대마다 초기화된다. rollback도 이전 실행 상태로 메모리 복원하는 것이 아니라 이전 YAML의 처음부터 시작한다.
- expect는 손 칸을 복원하지만 world.objects의 손/홀더 위치·과거 pose 기록을 직접 복원할 수 없다. 홀더 물체는 snapshot에서 제외하며 위치 메타데이터를 복구했다고 가정하면 안 된다.
- 다시 찍을 물체가 하나도 없으면 빈 objects가 전체 재촬영으로 해석되는 것을 피하기 위해 거부한다. lane 자식만의 재시도와 단일 팔 rehome 노드가 없는 팔의 home은 거부한다. 두 팔 trajectory의 실패 팔이 모호하면 명시적 side가 필요하다.
- 손에 물체가 있으면 기존 runner의 rehome 가드가 거부한다. shell이 그 가드를 우회하지 않는다. 부분 실행된 lanes를 통째로 retry하면 이미 끝난 팔도 다시 실행될 수 있으므로 운영자 판단이 필요하다.
- 정책 checkpoint가 로컬에 없으면 `--plan`이 실패하므로 shell은 파생 세대를 활성화하지 않는다. 외부 승인 주체와 shell 명령을 동시에 조작하지 않는다. cmd_id dedup은 물리 동작의 영속적 exactly-once 보장이 아니다.

## 모듈과 기록

- `interface.py`: 상태 파싱, 카메라, 승인, 네 서비스, 미지원 처리. ROS import는 생성 시점에만 수행. shell cmd 발행과 ack 대기도 담당.
- `backends.py`: 원형의 로컬 HTTP `/infer` 및 Gemini, 태평양 날짜 기준 일일 호출 한도·토큰 ledger, 순서별 fallback.
- `planner.py`: `config/skill_cards.yaml`로 문장 계획 생성 및 `simulate(steps)` 검사. 형식은 `[skill,hand,object,target]` 배열. 손·객체 배정, 잡기/운반/붓기/놓기 순서, 홀더 배정, 빈손 home, terminal 검사.
- `judge.py`: 종료 사진과 전체 상태·카드·계획을 입력해 A~E 파싱.
- `supervisor.py`: ROS 없는 결정 코어와 메인 루프.
- `cli.py`, `__main__.py`: 실행 진입점.

`simulate`는 물리 시뮬레이션이 아닌 기호 상태 검사다. 문장 의미 충족, 충돌, 수치 자세, 실제 비드 이동까지 증명하지 않는다.

`VLM_CHAIN` 기본값은 원형과 같은 `gemini-robotics-er-2-preview,local`이다. `--chain local`로 로컬만 사용할 수 있다. `VLM_LOCAL_URL`은 로컬 서버 주소(기본 원형의 `http://192.168.0.60:8765`), Gemini 키는 `GEMINI_API_KEY` 또는 `GEMINI_KEY_FILE`(기본 `~/.config/gemini_key`)로 지정한다. 키가 없어도 로컬 fallback은 동작한다. 로컬 HTTP는 마지막 사진 한 장을 `image_png_b64`로 전송하며 사진 없는 호출은 거부한다. Gemini를 체인에 넣으면 사진과 상태가 외부 API로 전송된다.

장부·기록 루트는 `VLM_SUP_HOME`(기본 `logs/vlm_supervisor`). `state/usage.json`에 날짜별 모델 성공 호출 수·토큰·차단 만료, `state/calls.jsonl`에 호출 기록을 저장한다. `VLM_CAP_<모델명의 비영숫자를 _로 치환>`으로 상한을 바꾼다. 기본 Gemini 상한 20은 원형의 관측값이며 서비스 보장값이 아니다. 429에서 받은 차단 정보와 다음 백엔드 fallback을 사용한다. 장부는 **단일 감독기 프로세스용**이며 여러 프로세스 간 전역 쿼터를 보장하지 않는다.

각 실행 폴더(`--log-dir`)에는 `start.jpg`, 검사한 `plan.json`, `decisions.jsonl`을 저장한다. 실물 runner/카메라와 외부 VLM 연결은 별도 통합 시험이 필요하다.

## ROS 없는 검증

```bash
uvx --with numpy --with pyyaml --with pillow --with requests pytest -q vlm_supervisor/tests
git diff --stat -- deploy/
```

테스트는 fake status JSON, 합성 사진과 가짜 백엔드를 사용한다. planner 상태 검사, 판정 파싱, 승인 줄 형식, active/shadow 흐름, 미지원 정지, 서비스 실패, stale 판정 취소, backend HTTP 형식과 ledger/fallback을 검증한다. 마지막 diff 출력은 비어 있어야 한다.
