# 독립 VLM 감독기

`vlm_supervisor`는 문장으로 계획을 만들고, runner의 노드 종료 뒤 사진과 상태로 다음 행동을 결정한다. 기본 모드는 **shadow**이며 승인 파일과 Trigger 서비스를 건드리지 않는다. `active`를 명시하면 승인 append 및 서비스 호출을 수행한다. `deploy`의 순수 실패 분류·복구 표를 선택적으로 import하며, import 실패 시 경고 후 보수적인 고정 표로 동작한다. deploy 파일은 수정하지 않는다. `proto/`는 비교용 원형이며 실행 패키지는 이를 import하지 않는다.

## 실행

저장소 루트에서 실행한다. Python 3.10 이상과 `pyyaml`, `pillow`, `requests`가 필요하다. 실기 연결은 ROS 2 환경을 먼저 source한다. ROS가 없어도 패키지 import, 계획 검사, 판정 파싱, 단위 테스트, `--help`는 가능하다.

```bash
python -m vlm_supervisor --runner episode_runner --mode shadow \
  --sentence '블루컵과 핑크컵의 비드를 오렌지 쉐이커 컵에 옮겨줘' \
  --approvals /실제/실행기/logs/runs/episode_approvals.jsonl
```

`--mode active`로 실행 권한을 켠다. `--approvals`는 **runner가 실제 읽는 같은 파일**을 지정해야 한다. 기본값은 현재 디렉터리 기준 `logs/runs/episode_approvals.jsonl`이다. 다른 호스트에서 실행한다면 같은 승인 파일이 보이도록 마운트해야 한다. 서비스는 같은 ROS 도메인의 runner로 연결된다.

sim 에서 돌릴 때는 ROS runner 대신 hdgp sim 브리지(HTTP)에 붙는다. rclpy 가 필요 없다.

```bash
python -m vlm_supervisor --interface sim --sim_url http://127.0.0.1:8790 --mode active \
  --sentence '블루컵과 핑크컵의 비드를 오렌지 쉐이커 컵에 옮겨줘'
```

브리지는 `GET /status`, `GET /frame`(JPEG, `X-Frame-Time`), `POST /cmd {"verb","args"}` 를 낸다. `next`·`stop` 과 셸 동작 `retry`·`rescan`·`home` 만 브리지로 보낸다. `run`·`reset` 은 실패로 돌려주고, `insert`·`human` 은 NotSupported 로 멈춘다. 브리지가 응답하지 않으면 상태는 `{}`(상태 끊김)이고 명령은 `ok: false` 다.

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

`VLM_CHAIN` 기본값은 `local8b`(Cosmos-Reason2-8B, 11.10 결정)다. 8B 서버는 같은 호스트의 `http://127.0.0.1:8766`(`VLM_LOCAL8B_URL`)에 띄운다. sim 은 server GPU0, 실기는 실기 호스트 쪽이다. `--chain gemini-robotics-er-2-preview,local8b,local`처럼 다른 순서를 줄 수 있다. `VLM_LOCAL_URL`은 로컬 서버 주소(기본 원형의 `http://192.168.0.60:8765`), Gemini 키는 `GEMINI_API_KEY` 또는 `GEMINI_KEY_FILE`(기본 `~/.config/gemini_key`)로 지정한다. 키가 없어도 로컬 fallback은 동작한다. 로컬 HTTP는 마지막 사진 한 장을 `image_png_b64`로 전송하며 사진 없는 호출은 거부한다. Gemini를 체인에 넣으면 사진과 상태가 외부 API로 전송된다.

장부·기록 루트는 `VLM_SUP_HOME`(기본 `logs/vlm_supervisor`). `state/usage.json`에 날짜별 모델 성공 호출 수·토큰·차단 만료, `state/calls.jsonl`에 호출 기록을 저장한다. `VLM_CAP_<모델명의 비영숫자를 _로 치환>`으로 상한을 바꾼다. 기본 Gemini 상한 20은 원형의 관측값이며 서비스 보장값이 아니다. 429에서 받은 차단 정보와 다음 백엔드 fallback을 사용한다. 장부는 **단일 감독기 프로세스용**이며 여러 프로세스 간 전역 쿼터를 보장하지 않는다.

각 실행 폴더(`--log-dir`)에는 `start.jpg`, 검사한 `plan.json`, `decisions.jsonl`을 저장한다. 실물 runner/카메라와 외부 VLM 연결은 별도 통합 시험이 필요하다.

## 접근 전 경로 판정 (`--approach-judge`, 기본 끔)

ap05 접근 정책은 장애물을 피하지 못한다. 이 옵션을 켜면 접근 노드를 실행하기 직전에 `approach.py` 의 `ApproachJudge` 를 판정 백엔드와 같은 체인으로 부른다. 판정 시점은 둘이다. READY 에서 `next` 를 보내기 전에는 다음 노드(`node`)를 본다. `node` 가 없을 때만 `next_action` 을 쓰고, 그 값이 `nodes` 목록의 id 여야 한다(sim 브리지의 `next_action` 은 서비스 이름 `next` 라 노드가 아니다). `retry` 를 보내기 전에는 다시 돌 노드(`last.node`)를 본다. 재실행도 접근이기 때문이다.

접근 노드는 `skill_cards.yaml` grasp 카드의 노드 예시(`grasp_blue`·`grasp_shaker`·`grasp_pink`)뿐이다. 손은 정책 이름 끝 `_r`/`_l`, 물체는 `target_object` 로 정한다. lanes 노드(`grasp_to_tilt_start` 등)는 손·물체를 몰라 접근 판정에서 뺀다. `recover:` 노드도 보지 않는다.

같은 에피소드·노드·시도 횟수에서는 한 번만 부른다. status 에 `episode_id` 나 `attempts` 가 없으면 같은 시도인지 알 수 없어 상태가 바뀔 때마다 다시 부른다. active 에서 clear 판정으로 `next`·`retry` 를 보내면 그 시도는 소비된 것으로 보고 지운다. 그래서 재시도·재기동 뒤에는 다시 판정한다. 판정이 쓰이지 않은 채 상태가 바뀌면(stale) 역시 다시 부른다.

`next`·`retry` 를 그대로 보내는 경우는 하나다. clear+proceed 이고 장애물 목록이 비어 있으며 기하·계획 단계와 어긋나지 않아야 한다. reorder(실행기가 순서 바꾸기를 지원하지 않음), other_hand(실행 경로 없음), blocked+human, clear 인데 장애물이 있는 응답, VLM 과 기하 불일치, 좌표가 숫자·유한값이 아닌 기하(판정 불가), 응답 오류, 카메라·백엔드 실패는 모두 active 에서 `human` 을 보내고 감독기를 멈춘다. 이때 `decision` 행에 `approach` 사유가 남는다. shadow 에서는 `decisions.jsonl` 에 `approach` 이벤트만 남기고 흐름은 바꾸지 않는다.

기하 판정은 status 의 `world` 에서 판정하는 손과 목표 물체 좌표를 읽는다. 둘 중 하나라도 없으면 기하 없음이고, 이때는 VLM 단독으로 판정한다(`approach` 이벤트의 `geom` 이 `none`). 장애물에서 빼는 것은 목표와 판정하는 손이 쥔 물체(`in_hand` 가 그 손)뿐이다. 반대 손이 든 물체는 장애물로 남긴다.

## ROS 없는 검증

```bash
uvx --with numpy --with pyyaml --with pillow --with requests pytest -q vlm_supervisor/tests
git diff --stat -- deploy/
```

테스트는 fake status JSON, 합성 사진과 가짜 백엔드를 사용한다. planner 상태 검사, 판정 파싱, 승인 줄 형식, active/shadow 흐름, 미지원 정지, 서비스 실패, stale 판정 취소, backend HTTP 형식과 ledger/fallback을 검증한다. 마지막 diff 출력은 비어 있어야 한다.

실패 코드·reason·outcome 전수 대조와 복구 허용 범위는 [FAILURE_AUDIT.md](FAILURE_AUDIT.md)에 정리했다. 금지/미지 코드는 사람 정지이며, pending 복구는 runner의 `recover:<node>:<action>` 승인 이름을 기록한 뒤 `next`로 실행한다.

## 에피소드 끝난 뒤 판정 (`bench/episode_review.py`)

실기 에피소드가 끝난 뒤 녹화된 카메라 bag 과 러너 기록만으로 경계 판정을 돌리고, 사람이 확인할 HTML 한 장을 만든다. ROS 노드가 아니라 로봇에는 영향이 없다.

```bash
# 1. laptop: 러너 기록 + 카메라 bag → 경계 사진과 bounds.jsonl
python3 -m vlm_supervisor.bench.real_frames <runs> --cam <cam> --spec <미션 yaml> --out <작업 폴더>
# 2. 작업 폴더를 server 로 복사(rsync). 코드는 server 의 이 저장소 체크아웃을 쓴다
# 3. server: GPU0 점유 확인 → 8B 서버 기동 → score → 서버 SIGTERM (GPU0 에 큰 프로세스가 있으면 멈춘다)
vlm_supervisor/tools/review_on_server.sh <작업 폴더>
# 4. review.jsonl 을 laptop 작업 폴더로 가져와 페이지를 만든다
PYTHONPATH=.:deploy/policy_control python3 -m vlm_supervisor.bench.episode_review page <작업 폴더>/review.jsonl
# 5. <작업 폴더>/index.html 을 열어 종합 결과 실패 · 사람 확인 경계를 사진으로 확인하고 "표시 결과 jsonl 복사" 로 넘긴다
# 6. 사람 표시와 비교: 역할별 러너만 · 종합 판정이 맞은 수
PYTHONPATH=.:deploy/policy_control python3 -m vlm_supervisor.bench.episode_review agree <작업 폴더>/review.jsonl <labels.jsonl>
```

경계 판정은 경계 묶음 모든 노드의 `node_checks`(judge_short.yaml)를 슬롯 기준으로 합친다. 확정 답 하나라도 기대와 다르면 막음, 그다음 오류·low_mass 가 있으면 판정불가, 모두 기대대로면 통과다. 검사가 없거나 전 사진이 없는 경계는 묻지 않고 판정불가다. 문턱은 score 를 돌린 시점의 judge_short.yaml 값이다. `score` 는 review.jsonl 에 이미 있는 경계를 건너뛰니 중간에 끊겨도 다시 돌리면 된다(다시 물을 경계는 그 줄을 지운다). 서버 없이 이미 있는 run_bench 결과를 다시 판정하려면 `score <bounds.jsonl> --from-bench <results.jsonl> [--variant short2]` 를 쓴다(프롬프트 문자열이 같은 응답만 쓰고, 없는 질문은 오류로 남는다). 페이지의 "러너 기록" 은 러너가 스스로 남긴 결과일 뿐 정답이 아니다.

종합 판정은 노드 id 로 역할을 정해 러너와 VLM 중 믿을 쪽을 고른다. 행에 노드별 `combined: [{node, role, runner, vlm, final, note}]` 와 경계 `final`(성공 · 실패 · 사람 확인)이 붙는다.

| 역할 | 노드 | 종합 결과 |
|---|---|---|
| outcome | `grasp_*`, `place_*` | VLM. 검사가 모두 기대대로면 성공(러너가 실패였으면 note "과정 경고"), 하나라도 반대면 실패. 오류 · low_mass · 검사 없음이면 러너 성공은 성공, 러너 실패는 사람 확인(러너 잡기 · 놓기 실패 신호는 결과로 믿지 않는다) |
| drop | `to_tilt_start_*`, `tilt_*` | 러너. VLM 이 "손에 없음" 으로 확정하면 실패(떨어뜨림 의심) |
| episode | `go_home_end`, `final` 의 `all_in_holders` | VLM, 판정불가면 사람 확인 |
| state | 그 밖 전부(`go_home_*`, `scene`, `home_*_after_*` …) | 러너만. 카메라로 볼 수 없는 이동 · 세팅이라 이 노드 검사는 `score` 때 묻지 않는다 |

같은 슬롯이 outcome 노드와 state 노드에 같이 걸리면 outcome 쪽에 쓴다. 놓기 끝 사진은 손이 컵을 가려 `place_*` 에 검사가 없을 수 있어서, 같은 묶음 state 노드(`home_*_after_*` 등)의 `cup_in_holder_b:<물체>` 는 그 물체를 놓는 place 노드로 옮긴다(`place_blue`↔`SRC_BLUE`, `place_pink`↔`SRC_PINK`, `place_shaker`↔`SHAKER`, 접미사로 못 찾으면 묶음 안 place 가 하나일 때만). 옮긴 슬롯은 `score` 때도 묻는다. 경계 `final` 은 노드 중 하나라도 실패면 실패, 실패 없이 사람 확인 노드가 있으면 사람 확인이다. 종합은 저장된 `checks` 를 슬롯 → 노드로 다시 나눠 계산하므로 옛 review.jsonl 도 `page` · `agree` 에 그대로 쓴다. `agree` 의 labels 줄은 `{"id","mark":"runner|vlm|both_wrong","runner_ok","vlm"}` 이고, 정답은 mark=runner 면 runner_ok, mark=vlm 이고 vlm 이 pass · block 이면 그 답이다. 그 밖(VLM 판정불가 · 둘 다 틀림)은 정답 없음으로 세지 않는다.
