# S2R 배포 운영 체계 — pour 실기 배포 + 읽기전용 콘솔

> 승인 후 이 파일을 `sim2real/docs/PLAN_S2R_CONSOLE_2026-09-21.md` 로 복사해 저장소에 남긴다(커밋은 요청 시).
> 작성 2026-09-21. 근거는 전부 실측 — 문서 기술과 어긋나는 곳은 §0 에 정정해 두었다.

---

## Context

**왜 지금 이것을 하는가.** 오늘부터 s2r 를 본격 진행한다. `hdgp/source/openarm/openarm/agnostic` 학습 구조는 어느 정도 완성됐고, 실기 쪽 `sim2real/deploy/policy_control` 도 계약 v2 + 노드 6개 + 테스트 593개로 성숙했다. 그런데 **정책 하나를 실기에 올리는 경로가 아직 한 번도 끝까지 통과된 적이 없다.** 양팔 물붓기(pour) 정책은 별도 스키마·전용 노드로 갈라져 미커밋 상태이고, 학습 산출물을 서버에서 로컬로 가져오는 도구가 없다.

**목표.** 정책마다 `*_inference_node.py`·전용 GUI·전용 launch 를 새로 만들지 않고, **계약과 프로파일을 등록하는 것만으로** 배포할 수 있게 한다. 그 위에 실기 세션에서 터미널 4개를 오가지 않아도 되는 운영 화면을 얹는다.

**첫 대상.** `log/server_mirror` 구조의 t2r 런 → pour 순서. 구체적으로 서버 `t2r_i18` 의 `ep_2500` 체크포인트.

**사용자 결정 (2026-09-21).**

| 항목 | 결정 |
|---|---|
| 1호 정책 | **i18 ep_2500 을 실기 후보로 준비**한다. 계약·검증까지 만들되 실기 실행은 §5 게이트를 넘긴 뒤 |
| pour 계약 분기 | **지금 통합하지 않는다.** i18 등록을 한 번 끝까지 통과시켜 실측한 뒤 재판단 |
| 콘솔 1차 형태 | **의존성 0 읽기전용 상태판.** 표준 라이브러리 `http.server`, `127.0.0.1` 바인딩, 버튼 없음, 새 상태기계 없음 |

---

## 0. 참고 MD 정정표 (실측)

업로드된 `sim2real_deployment_console_coding_agent_prompt.md` 는 큰 틀은 유효하나 아래가 현재 저장소와 다르다.

| MD 전제 | 실제 | 근거 |
|---|---|---|
| 계약 스키마 v1 | **v2 가 현행**. v1 은 로드 호환용 레거시 | `deploy/policy_control/policy_control/contract.py:29-30` |
| `deploy_contract.json` 이 단일 진실원천 | pour 는 **별도 스키마** `policy_control/pour_contract/v1`. 게다가 pd 용 control-only 계약이 **하나 더** 필요 — 정책 1개에 계약 파일 2개 | `pour_contract.py:22`, `contract_build.py:263-266` |
| 정책별 전용 노드 금지 | **이미 깨져 있다.** `pour_node.py` 301줄 + `pour_chain.launch.py` + `setup.py` 엔트리포인트 | `deploy/policy_control/setup.py` entry_points |
| `test_gui` 를 수동 조작 도구로 유지 | 유지해도 되나 **배포 체계에서 배제**. PickNik 예제 포크이고, `/openarm/*/eef_target` 은 구독자 0인 죽은 토픽이며, **게이트 없이 실손 JTC 를 발행**한다 | `test_gui/src/ros2node.cpp:113-131`, `README.md:1-30` |
| `config/mission_policy_control.yaml` 이 유효 | **낡아서 좌팔 미션이 죽어 있다.** `pd_selftest` 의 `blocked` 문구가 "도구가 아직 없다"인데 `deploy/policy_control/tools/pd_selftest.py` 는 09-07 부터 실재 → `goto_home` 이후 전 단계 영구 차단 | `config/mission_policy_control.yaml:43-45`, `scripts/mission_core.py:138-141` |
| `docs/measure/S2R_INTERFACE_EQUIVALENCE.md` | 존재하나 낡음. 대상이 구 자산 `openarm_tesollo_sensor_rl`, 대부분 행 미측정 | 파일 헤더 |
| 웹 스택 도입이 자연스럽다 | 저장소에 웹 스택 **0**. `~/.local` 에 `pydantic 2.13.3`·`flask 3.1.3` 누수 있음 + `.venv` 는 `include-system-site-packages=true` + 거기에 `torch 2.7.1+cu128`·`fabrics_sim.pth` 가 산다 → **venv 에 pydantic 계열 설치는 실질 위험**. node v22·PyPI 접속은 가능 | `.venv/pyvenv.cfg`, `pip index versions fastapi` rc=0 |
| 브라우저 UI 가 새 종류의 물건 | **전례가 이미 있다.** `scripts/vision/stream_head_view.py`(162줄)·`cup_view_stream.py`(215줄)가 stdlib `ThreadingHTTPServer` + `127.0.0.1` + ssh 터널로 운영 중 | 두 파일 |
| MVP 1~4 순서 | MVP1(읽기전용)만 지금 유효. MVP2~4 는 **배관이 한 번도 안 통한 상태에서 설계하면 근거가 없다** | §2 |

**MD 가 틀리지 않은 핵심 원칙 (그대로 채택)**: 계약을 UI 설정에 복제하지 않는다 · 정책명 분기 금지 · 임의 shell 실행 금지 · 실기는 명시 승인 + `--execute` · fake 가 기본 · 실기 자동 테스트 금지.

---

## 1. 지금 있는 것 (재사용 대상)

**계약·노드 층** — `policy_control`
- 계약 `deploy_contract/v2`: 최상위 12키, `sides` dict 로 양팔 표현, `primary_side` 미러. `contract.py`
- family 판정은 **빌드 시점에만**: `pour_bimanual` | `gripper_left` | `grasp_s2r`, 그 외 `SystemExit`. `contract_build.py:209-218`
- 노드 6개: `obs_node, policy_node, fabric_node, pd_node, episode_master, pour_node`. 순수 tick 로직은 전부 `chain.py` 의 4 스테이지에 있고 노드는 배선만 한다.
- 서비스 7개 `episode/{reset,start,stop,abort}` + `pd/{engage,goto_home,release}`, 전부 `Trigger`, 응답 `message` 는 `{"ok","reasons"}` JSON.
- status 4토픽 공통 필드 `node, phase, episode, seq, ok, reasons[], proc_ms` + `t_pub_ns`.
- **`execute:=false` 무발행이 5겹**: 파라미터 AND yaml(`pd_node.py:127`) · publisher 를 생성조차 안 함(`pd_backends.py:54-68`) · controller_manager 전 메서드 dry_run(`controller_switch.py`) · 손 PID SetParameters 차단 · engage 거부(`pd_state.py:170-171`). **이 다섯은 손대지 않는다.**
- 레지스트리 상태: obs 빌더는 dict + `@register` 19종으로 이미 확장 가능(`obs_segments.py:60`). **action decoder·pd backend·family 는 if/elif 체인** — 확장 저항점.

**운영 층** — 전부 순수(ROS 무의존), 감싸기만 하면 된다
- `scripts/mission_core.py` — `gate()` 가 차단·선행·산출물·체크포인트·승인 사유를 **한 번에 모아** 반환. 사유 문자열의 저작권자.
- `scripts/mission_stages.py` — 단계 → argv 조립, `{repo}`/`{artifact:}`/`{checkpoint:}`/`{params:}`, `manual`/`background` 구분.
- `scripts/ops/mission_run.py` — `--approve` · `--execute` · `start_new_session` + `pids` + `--abort` → `killpg` · append-only `state.jsonl`. **함정: `plan_evidence()` 는 "승인이 다 있다"고 가정한다 — 실행 가능 신호로 쓰면 안 된다.**
- `deploy/policy_control/tools/episode_ctl.py` — 7단계 시나리오 + `touches_real` 3개 승인 필수 + `parse_trigger()`.
- `deploy/policy_control/tools/status_to_csv.py` — 4노드 status 를 `seq` 로 join, `latency_ms`, p50/p95, seq 결손, `--jsonl`. **단 "N초 구독 후 종료" 형이라 상주 fan-out 이 없다.**
- fake 플랜트 완비: `scripts/fakes/*` 6개 + `launch/fake_plant.launch.py`(`plant_model: pd|rate`) + `tools/fake_plant_run.sh`/`fake_direct_run.sh`. **한계: MockArm 에 커플링·테이블 접촉이 없어 파지 성공은 fake 로 검증되지 않는다 — 배관만 증명한다.**

**테스트 현황 (실측)**: `pytest tests/policy_control -m "not gpu"` → **2 failed, 593 passed, 3 skipped**. 실패 2건은 구 자산 `openarm_dg5f-m_bi_rl.urdf` 에 `*_alias` 링크가 없어서 나는 기존 결함(`test_pc_fk_urdf.py:110`)이며 pour 와 무관. pour 만 `-k pour` → **70 passed**.

---

## 2. 정책 1개 등록에 실제로 드는 일

`docs/RUNBOOK_pour_bimanual.md` §1~§2 가 이미 7단계로 적어 놓았다.

1. `params/env.yaml`·`agent.yaml` 배치 → 2. `nn/*.pth` 배치 → 3. `trace_meta.json` 확보 → 4. `build_deploy_contract.py --run --sim-meta` → 5. pd 용 control-only 계약 **한 개 더** → 6. `contract_doc.py` 재생성 → 7. 픽스처 교체 후 parity 테스트

**콘솔이 줄일 수 있는 것: 4~7 의 타이핑뿐** (전부 이미 한 줄 명령). **줄이지 못하는 것: 1~3** — rsync/ssh 문제이고 로봇 PC 에 Isaac 이 없다.

**그래서 우선순위가 뒤집힌다.** 지금 pour 배포를 막고 있던 것을 실측으로 좁히면:

| 막힘 | 실측 결과 |
|---|---|
| 미러에 `params/`·`nn/` 이 없다 | 사실. `hdgp/scripts/reward_gen/t2r_round.py:57-66` 의 rsync 가 `--include=summaries/*** --include=test_history.md --exclude=*` 로 **설계상** 안 가져온다 |
| sim meta 가 없다 | **로컬만 없다. 서버에는 있다** — `t2r_i18/trace_i18_ep2500_adr30_64env_meta.json` 실재 확인 |
| `play.py` 의 meta 기록이 미커밋 | **틀렸다. 커밋됨** — hdgp `43ba1628`(09-20), HEAD 에 `play.py:842` `obs_next`, `:849` `_meta.json`. RUNBOOK §1 의 "uncommitted" 기술이 낡았다 |
| slew(`palm_cmd_max_step`) 드리프트 | **i19 에만 해당.** 도입 커밋 `329a0812` 가 iter_19 보상과 같은 커밋이고, **i18 `env.yaml` 에 그 키가 없다**(404행은 `palm_action_ema_alpha: 0.25` 뿐). → **i18 은 디코더 수정 불필요** |
| 체크포인트 미정 | 사용자가 i18 ep_2500 으로 결정 |

**결론: 남은 실질 차단은 하나다 — 서버 산출물 4종을 로컬 규약 디렉터리로 가져오는 도구가 없다.**
서버 `t2r_i18` 에 `nn/last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth`, `params/{env,agent}.yaml`, `trace_i18_ep2500_adr30_64env_meta.json`, trace npz 가 전부 있음을 확인했다.

---

## 3. 작업 순서

각 단계는 독립적으로 검증 가능하다. **GPU 테스트는 학습 프로세스 부재를 `nvidia-smi` 로 확인한 뒤에만** 돈다(현재 로컬 5090 에서 cup_pick 학습 중).

### P0 — 죽은 게이트 해제 · 미커밋 정리 (30분, 발행 없음)

1. `config/mission_policy_control.yaml:43-45` 의 `pd_selftest` `blocked` 문구 제거. 도구는 09-07 부터 실재한다.
2. 미커밋 pour 작업(미추적 26 + 수정 6, 약 8.8 MB)의 커밋 정책 결정. `tests/fixtures/policy_control/pour_i11/trace.npz` 8.8 MB 를 저장소에 넣을지가 쟁점 — **넣지 않고 `fetch_run.py` 로 재생성 가능하게 하는 쪽을 권고**한다.

검증: `python3 scripts/ops/mission_run.py --mission config/mission_policy_control.yaml --plan` 이 `goto_home` 이후 단계를 더 이상 차단하지 않는다. 발행 0건.

### P1 — `deploy/policy_control/tools/fetch_run.py` 신규 (hdgp 무수정)

hdgp 의 미러 스크립트는 건드리지 않고, sim2real 이 필요한 것만 따로 당긴다.

```
fetch_run.py --run t2r_i18 [--host server]
             [--root ~/rl_ws/hdgp/log/rl_games/open-short/both/pour-fab]
             [--out logs/policy/<run>]
             --checkpoint best|last|ep:2500|<filename>      # 자동선택 금지, 명시 요구
             [--sim-meta auto|<name>] [--trace none|auto]   # trace 기본 none (143 MB)
             [--list|--dry-run|--verify-only|--force]
```

받는 곳:
```
sim2real/logs/policy/<run>/
  params/env.yaml, params/agent.yaml
  nn/<선택된 1개>.pth        # 정확히 1개 → contract_build 의 "exactly one .pth" 규칙을 구조적으로 만족
  trace_meta.json            # 원격 <trace>_meta.json 을 규약명으로 rename
  fetch.json                 # host·remote_dir·hdgp_commit·파일별 sha256/md5/size
```

규약: `.staging/` 에 받고 원격 `sha256sum` 과 재해시 일치 후에만 `os.replace`(반쪽 디렉터리 불가) · 재실행 시 해시 동일이면 0바이트 "up to date" · ssh 실패 시 `fetch.json` 있으면 로컬 검증만 하고 성공, 없으면 붙여넣을 `scp` 명령 출력 · 서버에서는 `ls`/`sha256sum`/`git rev-parse`/rsync read 만 — **GPU 접촉 0**.

검증: `--list --run t2r_i19` 표 출력 → `--run t2r_i18 --checkpoint ep:2500` → 재실행이 0바이트 → `--host nosuch` 가 오프라인 경로 → 전후 `nvidia-smi` 동일.

### P2 — i18 계약 생성 · 문서 재생성 · parity

RUNBOOK §2 의 두 명령을 i18 로 실행한다.
1. `build_deploy_contract.py --run logs/policy/pour_i18 --sim-meta logs/policy/pour_i18/trace_meta.json` → `pour_contract.json`
2. pd 용 control-only 계약(`--asset --sides --home pour:<pour_contract.json>`) → `deploy_contract.json`
3. `contract_doc.py` 재생성 — 현재 `docs/CONTRACT_policy_control.md` 는 pour 절을 **테스트 픽스처에서 렌더**한 상태이고 `right_g1` 절이 중복돼 있다. 재생성 명령을 문서 머리에 박는다.

검증: `pytest tests/policy_control -q -m "not gpu" -k pour` 그린 · 계약의 `obs_dim/action_dim` 223/18 · `checkpoint_md5` 가 `fetch.json` 값과 일치 · `palm_cmd_max_step` 부재가 계약에 `max_step_source="absent(pre-slew run)"` 로 명시.

**이 단계가 끝나면 "정책 1개 등록"의 실제 소요가 처음으로 측정된다.** 그 숫자가 이후 자동화 판단의 근거다.

### P3 — 잔여 결함 (P1 과 병행 가능, 4-1 은 최우선)

| # | 결함 | 수정 |
|---|---|---|
| 1 | **`tests/policy_control/test_pour_fabric.py:131` 에 `@pytest.mark.gpu` 누락** → `-m "not gpu"` 로 돌려도 CUDA 를 쓴다. 이번 조사에서 실제로 학습 중인 5090 을 잠깐 건드렸다 | 마커 부착 + `conftest.py` 수집 훅: `fabrics_sim` import 나 `torch.cuda` 참조가 있는데 `gpu` 마커가 없으면 **수집 실패** |
| 2 | `pour_build.py:64-65` `float(lim.get("upper"))` 무가드 → 잠복 `TypeError`(현 자산에서는 미발현) | 가드에 `upper is None` 추가 → `PourContractError` |
| 3 | `pour_node.py:189` 가 `feed_inbox()` 의 스테일/미싱 소스 목록을 버린다 | `(measure, problems)` 반환으로 바꿔 status `reasons` 에 병합 |
| 4 | ruff 9건 (F401 3 · E702 3 · E741 3) | 정리. 게이트는 범위 한정 `ruff check policy_control tests/policy_control` = 0 |
| 5 | `docs/` 3종 낡음 — `CONTRACT_policy_control.md` 중복·빈 경로, `POLICY_CONTROL_STATUS_2026-09-06.md` 단계 수/테스트 수 불일치, `PLAN_S2R_CONTRACT_SYNC_2026-08-18.md` 완료 미표시 | 재생성·정정·아카이브 표시 |

검증: `ruff check policy_control tests/policy_control` = 0 · 전체 비GPU 스위트가 **여전히 2 failed(기존 alias 결함)/나머지 그린** · **테스트 실행 중 `nvidia-smi` 에 새 프로세스가 뜨지 않는다**(4-1 실증).

### P4 — 읽기전용 상태판 (의존성 0, 버튼 없음)

`deploy/policy_control/tools/status_board.py`. `scripts/vision/stream_head_view.py` 패턴 그대로 — stdlib `ThreadingHTTPServer`, 기본 바인딩 `127.0.0.1`, 원격 열람은 기존 ssh 터널.

내용은 세 덩어리뿐이고 **전부 기존 출처의 문자열을 그대로 옮긴다**:
- ⓐ `mission_core.plan()`/`gate()` 의 단계별 판정과 사유 — 콘솔이 사유를 새로 쓰지 않는다
- ⓑ 4노드 `status/*` 의 `phase`/`ok`/`reasons` 원문 + pd 의 `execute`/`estop`/`thermal`
- ⓒ `sources` stale · seq 결손 · 지연 p50/p95 — `status_to_csv.summarize()` 재사용

동시에 `status_to_csv.py` 의 구독·join·요약을 `deploy/policy_control/policy_control/status_join.py` 로 승격한다. **CLI 와 CSV 출력 포맷은 한 글자도 바꾸지 않는다**(`fake_plant_run.sh` 가 파싱한다).

**금지 사항 (스스로 거는 제약)**: 새 상태기계 금지 · 버튼 금지 · 새 의존성 금지 · `plan_evidence()` 호출 금지 · **250줄 넘으면 멈추고 재검토**.

검증: `tests/policy_control/test_pc_status_join.py` 골든 — 기존 `logs/policy_control/fake_*/status.jsonl` 실측 행을 먹여 `summarize()` 출력이 현행과 동일 · fake 체인 구동 중 브라우저에서 판정이 CLI 와 같은 문자열로 보인다.

### P5 — fake 체인으로 i18 배관 실증 (실기 아님)

`fake_plant.launch.py` + `pour_chain.launch.py` 를 **도메인 97** 에서 구동. `ROS_DOMAIN_ID` 를 명시적으로 주입한다 — 현 방어선은 `""`/`"0"` 거부뿐이라 실기 도메인 126 혼선을 못 막고, `fake_plant_run.sh:14` 의 `${ROS_DOMAIN_ID:-99}` 는 이미 export 돼 있으면 그대로 쓴다.

검증: 900스텝 완주 · `status.jsonl` 의 `not-ok` 행 0 · 지연 p95 < 0.5·dt · seq 결손 0 · 60 Hz 두 fabric 의 `proc_ms` 기록(실기 예산 판단용). **파지·붓기 성공은 판정하지 않는다** — MockArm 에 접촉이 없다.

### P6 — 실기 진입 전 안전망 (정책과 무관, 어떤 체크포인트든 필수)

1. **기울기 워치독** — 소스 컵 포즈에서 기울기를 계산해 계약의 `src_tilt_abort_deg`(예 125°) 초과 시 `episode/abort`. 정책이 기울기를 직접 지령하지 않으므로 계약만으로는 못 막는다.
2. **cross-arm guard 노드** — 두 `fabric_node` 가 이미 `/policy_control/palm_pose` 를 latched 로 낸다. 그 둘의 거리가 임계 이하면 `episode/abort`. **필요한 이유**: 배포 fabric 세계의 장애물은 테이블 박스 하나뿐이고(`pour_fabric.py:22-33`), 실기에서는 fabric 프로세스가 팔당 하나라 **서로의 존재를 모른다**. sim 은 두 팔이 한 articulation 이라 자가충돌이 처리되지만 실기는 아니다.
3. **체크포인트 게이트 스크립트** — §5 표를 trace npz 에 적용해 PASS/REJECT 를 출력.

검증: 게이트 스크립트를 i18 trace 에 적용 → 아래 표대로 2건 REJECT · 워치독·guard 단위 테스트.

### P7 — 보류 (P2 실측 후 재판단)

- **pour 계약 통합(v3 family)**: 조사 결과 통합 비용의 대부분은 이미 지불돼 있다 — pour obs 10 세그먼트 중 9개가 기존 빌더와 의미까지 동일하고, `SideCfg`/`PalmCfg`/`HandCfg` 가 pour 필드를 대부분 이미 갖고 있다. 실질 신규는 8개 정도(`family`, `SideCfg.role`, `rate.hold_steps`, `ActionCfg.filter`, `PalmCfg.delta_lo/hi` + 비대칭 convention, `synergy_grip3` 디코더, `FabricCfg` 3필드). **단 지금 하지 않는다** — 70개 테스트가 초록이고, P2 를 통과해야 "전용 경로라서 드는 추가 비용"이 숫자로 나온다.
- **레지스트리화**(decoder·pd backend·family 의 if/elif → `@register`): 통합을 하기로 하면 그 직전 단계. 동작 변경 0 이어야 하고, `contract_doc.py` 재생성 결과가 byte diff 0 인 것으로 검증한다.
- **콘솔 확장**(프로세스 감독·승인 원장·실행 버튼): P4 가 실기 세션에서 실제로 쓰인 뒤에 판단. 확장한다면 **API 프로세스가 rclpy 를 import 하지 않고** bridge 노드(구독 전용, 발행자 0개)와 unix socket 으로 분리하는 2프로세스 구조가 맞다 — 그래야 API 가 DDS 참가자가 될 수 없다.
- **i19 등록**: slew 가 들어간 첫 런이므로 `ActionCfg.filter{alpha,max_step}` 신설 + 필터를 `policy_core` 한 곳에서 적용(그래야 `last_action` obs 가 저절로 맞는다) + 서버 trace/meta 재수집이 필요하다. i18 배관이 통한 뒤 체크포인트만 갈아 끼운다.

---

## 4. 하지 않는 것

MD 의 아래 기능은 이 환경(로봇 PC 1대, 운영자 1명, 정책 2~3개)에서 만들지 않는다.

| 기능 | 이유 |
|---|---|
| ROS2 Node Graph 시각화 / 노드별 7상태 | 노드 4~6개다. expected graph 를 콘솔 설정에 적는 순간 원천이 launch 파일에서 UI 로 복제된다 — MD 자신의 원칙 위반. status 의 `reasons` 는 가변 길이라 7칸에 넣으면 정보가 준다 |
| 실시간 그래프 12종 | 운영자 1명은 12개를 못 본다. 라이브로 필요한 건 지연·seq 결손·HOLD 사유 3개이고 텍스트가 낫다 |
| 프로파일 복제 Wizard | 프로파일은 이미 갈라진 게 문제다(`config/robots/` vs `deploy/policy_control/config/robots/` 두 디렉터리). 필요한 건 복제가 아니라 참조 |
| 역할 기반 권한 / 웹 세션 승인 | **안전 회귀다.** 현 승인은 `--approve <id>` 로 argv 에 있고 프로세스와 함께 사라진다. 세션 쿠키로 옮기면 "단계당 1회"가 "로그인 1회"가 된다 |
| 런타임 `execute` 토글 | `pd_node.py:127` 이 `__init__` 에서 한 번 읽고 `add_on_set_parameters_callback` 이 없다. **발행 허가가 프로세스 수명에 묶여 있는 것이 안전장치다.** 토글을 만들면 그 보증이 메모리 안 bool 하나로 내려온다 |
| 콘솔 소프트 estop 버튼 | 웹서버→소켓→DDS 는 건물에서 가장 느린 estop 이고, 큰 빨간 버튼을 주면 운영자가 틀린 것에 손을 뻗도록 훈련된다. 콘솔은 estop **상태만** 배너로 표시하고 물리 estop 위치를 적는다 |
| `test_gui` 확장 | 게이트 없이 실손을 발행하는 유일한 코드다. 수동 조작이 필요하면 `palm_cmd.py`/`hand_cmd.py` 를 감싼다 |

---

## 5. i18 ep_2500 실기 진입 게이트

`LOOP_STATE.json` 의 `play_check_i18`(ADR30, 64env) 실측을 기준선으로 한다.

| 지표 | 조건 | i18 실측 | 판정 |
|---|---|---|---|
| `success_ever` | ≥ 0.70 | 0.828 | 통과 |
| `in_target_max` | ≥ 0.80 | 0.907 | 통과 |
| `spill` | ≤ 0.10 | 0.07 | 통과 |
| `min_cup_dist_med` | ≥ 0.12 m | 0.181 | 통과 |
| `pour_dir` (x≤0 비율) | ≥ 0.99 | 0.9999 | 통과 |
| `cup_collision_rate` | ≤ 0.01 | 0.002 | 통과 |
| **`src_tilt_med_peak`** | **≤ 120°** | **150.8°** | **탈락** |
| **`src_xy_travel`** + 중앙선 교차 | **≤ 0.25 m 이고 교차 없음** | **0.33 m, 교차** | **탈락** |
| `adr_level` | 기록 필수(맥락) | 30 | — |
| `rew_*` 파일명 값 | **게이트 아님** | 36408 | 보상 이터레이션마다 의미가 바뀐다 |

**두 탈락 항목의 실기 위험**
- **over-tilt 150°**: 수평을 지난 역전 자세다. 손바닥 법선 방향 중력 성분의 부호가 뒤집혀 컵이 떨어질 수 있고(sim 은 접촉 동결로 버티지만 실기 DG-5F 는 백래시가 있다), `r_aj_6/7` 이 한계 부근에 머물러 j7 발열(임계 70 °C)로 간다. 액체는 비드와 달리 100~110°에서 이미 흐르므로 150°는 전량 일시 배출이다.
- **소스 컵이 수신 쪽으로 넘어감**: 배포 fabric 세계에는 테이블 박스 하나뿐이고 실기에서는 팔마다 fabric 프로세스가 따로 돈다. 중앙선을 넘는 궤적은 **충돌 검출기 없이 60 Hz 로 반대 팔 영역을 지난다.**

**따라서 i18 은 P1~P5(계약·fake 검증)까지만 진행하고, 실기 실행은 아래 셋이 모두 충족된 뒤다.**
1. 사용자 영상 판정 완료
2. P6 의 기울기 워치독 + cross-arm guard 가동
3. i19(또는 후속)가 두 항목을 개선한 체크포인트를 내거나, i18 로 가되 reduced 파라미터(팜 rate ×0.5, `max_vel` 0.25)로 단계 승인

---

## 6. 위험과 중단 기준

| 위험 | 대응 |
|---|---|
| `status_join` 승격이 CSV 숫자를 바꾼다 | 현 구현은 무한 누적 후 사후 정렬이라 늦은 seq 도 합쳐진다. 스트리밍용 bounded 모드를 넣으면 `seq missing` 이 달라지고 그 값은 `fake_plant_run.sh` 의 합격 판정에 쓰인다. **두 모드를 분리하고 골든 테스트로 동등성을 잠근다** |
| 도메인 126 혼선 | 현 방어선은 `""`/`"0"` 거부뿐. 감독 대상 자식의 env 에 도메인을 **명시 주입**하고, `fake_plant_run.sh:14` 처럼 자기 기본값을 가진 스크립트는 래핑하거나 배제 |
| `pour_profiles.load_pair()` 가 hdgp 워킹트리를 실시간으로 읽는다 | 값은 계약에 복사되지만 **출처 기록이 없다**. `SideCfg.profile` 에 `hdgp_commit` + 모듈 3개 sha1 을 동결한다. 값 검증은 골든 트레이스가 이미 `box_lo/hi`·`hand_open`·`hand_grip` 을 전이적으로 커버하므로, **`arm_reset` 대조 1건만 `test_pour_golden` 에 추가**하면 구멍이 메워진다 |
| `palm_box_verified=False` (양팔 모두) | 계약·문서·status `warnings` 에 노출하고 콘솔 배지로 상시 표시. 등록은 막지 않되 `execute:=true` 전환 시 확인 |
| venv 오염 | 새 파이썬 패키지를 설치해야 하는 순간 멈추고 재검토. `.venv` 는 rclpy+torch+fabrics 가 동거하는 유일한 장소다 |

**중단 기준 3개**
1. P4 상태판이 **의존성 추가 없이 250줄 안에** 안 나오면 범위가 샌 것이다. 기준선: `stream_head_view.py` 162줄.
2. **venv 에 패키지를 하나라도 설치해야 하면** 중단하고 설계를 다시 본다.
3. `logs/policy/pour_i18/pour_contract.json` 이 **아직 없는데** 새 도구 코드가 1000줄을 넘으면 중단. "배포 전진 0 / 도구 1000줄"은 직전 세션에서 중단시킨 것과 같은 패턴이다.

---

## 7. 파일

**새로 만든다**
```
deploy/policy_control/tools/fetch_run.py                 P1  서버 산출물 수집
deploy/policy_control/tools/status_board.py              P4  읽기전용 상태판 (stdlib, ≤250줄)
deploy/policy_control/policy_control/status_join.py      P4  status 구독·join·summarize 승격
deploy/policy_control/tools/ckpt_gate.py                 P6  §5 게이트를 trace npz 에 적용
deploy/policy_control/policy_control/pour_guard.py       P6  기울기 워치독 + cross-arm guard
tests/policy_control/test_pc_status_join.py       P4  골든 — 승격 전후 동등성
tests/policy_control/test_pc_fetch_run.py         P1  멱등성·오프라인·해시 불일치
```

**수정한다**
```
config/mission_policy_control.yaml                P0  pd_selftest 의 낡은 blocked 제거
deploy/policy_control/tools/status_to_csv.py             P4  status_join 호출자로 (CLI·출력 포맷 불변)
deploy/policy_control/policy_control/pour_build.py       P3  upper 가드
deploy/policy_control/policy_control/pour_node.py        P3  feed_inbox 반환값 status 로
tests/policy_control/test_pour_fabric.py          P3  @pytest.mark.gpu
tests/policy_control/conftest.py                  P3  GPU 마커 누락 수집 훅
docs/CONTRACT_policy_control.md                   P2  재생성 (+ 생성 명령 머리에 명시)
docs/RUNBOOK_pour_bimanual.md                     P2  "play.py 미커밋" 기술 정정
docs/POLICY_CONTROL_STATUS_2026-09-06.md          P3  단계 수·테스트 수 정정
```

**손대지 않는다**: `mission_core.py` · `mission_stages.py` · `mission_run.py` CLI · `episode_ctl.py` · `chain.py` · `pd_law.py`/`pd_state.py` · `controller_switch.py` · `pd_backends.py` 의 5중 execute 게이트 · `fake_plant.launch.py` · policy_control 4노드.
**배제**: `test_gui/`.
**hdgp 는 읽기 전용** — 수정도 학습 기동도 하지 않는다.

---

---

## 진행 (2026-09-21 갱신 — 실행하면서 채운다)

| 단계 | 상태 | 실측 |
|---|---|---|
| P0 죽은 게이트 해제 | **완료** | `mission_policy_control.yaml` 의 낡은 `pd_selftest` blocked 제거 → `--plan` 9단계 **구조적 막힘 0** |
| P1 `fetch_run.py` | **완료** | 단위 26. 서버 `t2r_i18` 에서 params 2 + nn 1 + trace_meta + trace(142 MB) 수집. 재실행 0 바이트, 오프라인 로컬 검증 동작. GPU 프로세스 전후 동일 |
| P2 i18 계약 | **완료** | `pour_contract.json`(223/18, 60 Hz, asset short_bi) + pd control-only `asset_pour_i18/deploy_contract.json`. ckpt md5 `f50b09a6…` = `fetch.json` 일치. `palm_cmd_max_step` **없음**(slew 이전 런 확인). 계약 문서 재생성 — 제목을 계약 파일 경로로 바꿔 **중복 2건 해소**, 생성 명령을 문서 머리에 기록 |
| P2 검증 | **완료** | 신규 `test_pour_registered_run.py` 6개 통과: actor 재현 max err **2.1e-6**(3 env × 850 step), 디코더 palm/hand 목표 재현 < 1e-5 |
| P3 잔여 결함 | **완료** | GPU 마커 누락 수정 + `conftest` 가 마커 없는 테스트에서 CUDA 를 숨긴다 · `pour_build` upper 가드 · `pour_node` 가 소스 결손을 status `sources` 로 노출 · pour ruff 9건 0 · 문서 3종 정정 |
| P4 상태판 | **완료** | `status_join.py` 승격(기록 16개 실행에서 CSV·요약 **바이트 동일**, 62 테스트) + `status_board.py` 246줄, 표준 라이브러리만, 127.0.0.1, 버튼 없음 |
| P5 fake 체인 | 미착수 | |
| P6 안전망 | **완료(오프라인)** | `ckpt_gate.py` — i18 trace 를 직접 세어 **탈락 3건**(기울기 155.1° / 이탈 0.344 m / 중앙선 교차 64·64 env). `pour_guard.py` 코어 18 테스트 + `pour_guard_node.py`(발행 없음, `episode/abort` Trigger 하나만). 실기 구동은 P5 이후 |
| P7 보류 | — | 계약 통합·콘솔 확장은 P2 실측 후 재판단 |


**저장소 위생**: `logs/policy/**/trace.npz` 와 `tests/fixtures/**/trace.npz` 를 `.gitignore` 에 넣었다
(i18 trace 142 MB 가 추적 대상으로 잡혀 있었다). 골든 trace 를 쓰는 테스트는 파일이 없으면
받는 명령을 알려주고 skip 한다(`pour_trace_util.trace()`).


**한 가지 되돌린 것**: `pour_node` 가 소스 결손을 `episode/start` 거부 사유로 쓰게 했다가 되돌렸다.
좌우가 같은 토픽(`/joint_states`)을 쓰면 한쪽 `SourceSet` 스냅샷이 결손으로 보여도 인박스에는 값이
들어와 있어서 **거짓 거부**가 난다(`test_pour_node_ros` 가 잡았다). 결손 목록은 status `sources` 로만
노출하고, 기동 가부는 실제 측정을 보는 `start_refusals` 가 계속 정한다.
**남은 과제로 기록**: `start_refusals` 가 "한쪽 팔 소스가 통째로 없음"을 잡는지는 별도 확인이 필요하다.

**남은 ruff 28건은 이 작업과 무관한 파일의 기존 건이다**(`test_pc_selftest_judgement` 9, `test_pc_node_fabric` 6, `pd_selftest` 2 …). 관련 없는 파일은 건드리지 않았다.

**기존 실패 2건 유지**: `test_pc_fk_urdf.py` 의 `*_alias` 링크 부재(구 자산 `openarm_dg5f-m_bi_rl.urdf`). pour 와 무관.

---

## 8. grasp-fj-s2r 연결 (사용자 09.21 지시 — 다음 차례)

**대상**: `open-short/right/grasp-fj-t2r-rand/fj_rand_i01`, 학습 커밋 `9aa65abb`.
사용자가 `our_source/s2r_init_right/` 에 pth 3개 + 그 런의 `params/{env,agent}.yaml` 을 모으는 중(414 MB).
서버는 읽기만, s2r 동결 자산은 건드리지 않는다.

**인터페이스 (사용자 확인)**
- obs **133** / critic 157 / action **26 = 팔 7 증분 + 손 19 절대**
- 관절 순서·잠긴 관절은 사이드카 JSON 으로 함께 기록될 예정
- 보상 `reward_gen/grasp_fj_rand/iter_01/compute_reward.py` — 로컬에 있음. `env.yaml` 의
  `reward_code_path` 가 서버 절대경로라 로컬 경로로 고쳐야 재생된다
- 과제 코드 `tasks/grasp_fj_t2r/` — 로컬, 동결, 읽기만

**이것이 기존 세 family 와 다른 지점 (중요)**
`gripper_left` 는 `absolute_palm`, `grasp_s2r` 는 `delta_anchor` — **둘 다 팜 6D 포즈를 만들어
fabric 에 넘긴다.** fj 는 팔을 **관절 증분 7** 로 직접 지령한다. 즉 팔에 fabric 이 끼지 않는
새 제어 경로다. 손도 시너지 15 가 아니라 **절대 19**(한 관절 잠김)다. 그래서:
- 새 action decoder 2종이 필요하다 — 팔 `joint_delta`, 손 `direct_absolute`(잠긴 관절 제외)
- 계약에 "이 side 는 fabric 을 쓰지 않는다"를 표현할 자리가 필요하다. `SideCfg.fabric` 이
  이미 `None` 을 허용하므로(`contract.py:210` "None → this side has no Fabrics layer") 자리는 있다
- pd 는 관절 목표를 그대로 받으므로 기존 `arm_forward` 백엔드가 그대로 선다

**관련 사실 (직접 확인)**
- `grasp-s2r/f1_fresh` 는 obs **140** / act 21 이고, 155 와의 차이는 **촉각 15칸뿐**이다 —
  hdgp `ccfe4a5a:grasp_s2r_env.py:904` "★09.10 촉각은 관측에서 제거됐다(사용자 확정)".
  sim2real 의 `grasp_s2r_obs_builder.SEGMENTS` 가 155 로 굳어 있어 빌드가 막힌다
  (`_check_checkpoint_dims` 가 정확히 잡는다). 레이아웃을 변종 표로 바꾸면 풀린다.
  **fj 를 먼저 하기로 했으므로 이것은 보류**한다.

## 상시 제약
- 실기 동작은 사용자 승인 후에만. `--execute`/`execute:=false` 무발행 규약 유지.
- 커밋·푸시는 요청 시에만.
- 테스트는 실기 도메인(126) 금지 — 99/97/96 사용.
- 학습 중인 GPU 에서 무거운 CUDA 작업 금지. P3-1 이 이 규칙의 코드화다.
- 보고는 끝난 뒤 한 번, 요약으로.


## deploy/policies/ 등록소 (09.21)

쓸 정책을 `sim2real/deploy/policies/<id>/` 한 곳에 모은다. `logs/policy/` 는 실험 기록으로 그대로 둔다(옮기지 않았다 — 테스트·RUNBOOK 이 그 경로를 가리킨다).

| 무엇 | 어디 |
|---|---|
| 규약·점검 (순수) | `deploy/policy_control/policy_control/policy_registry.py` — 카드 `policy.yaml`, `fetch.json`, 계약 md5 == 받은 체크포인트 md5 |
| 등록 | `deploy/policy_control/tools/fetch_run.py` — 기본 `--out` 이 `deploy/policies/<run>`, `--host local` 로 이 PC 의 디렉터리도 출처가 된다, 출처 `README.md` 는 `SOURCE_README.md` 로 따라온다, 카드 초안은 **없을 때만** 쓴다 |
| 목록·점검 | `deploy/policy_control/tools/policies.py [--write-index] [--shallow]` — 문제가 있으면 rc 1 |
| git | 카드·fetch.json·params·계약은 추적, `nn/*.pth` · `trace.npz` 는 `.gitignore` |

등록된 것:

- `pour_i18` — **hold**. 계약(`pour_contract.json`)은 `logs/policy/pour_i18` 것과 경로 두 필드만 다르고 checkpoint md5 동일. hold 사유는 ckpt_gate REJECT 2건(기울기 155.1 deg, 중앙선 64/64).
- `grasp_fj_rand_i01` — **candidate, 계약 없음**. `our_source/s2r_init_right` 에서 `--host local` 로 받았다. obs 133 / act 26 (팔 7 증분 + 손 19 절대) — grasp_s2r(155/21) 과 다른 인터페이스이고 빌더가 없다.

같이 고친 것: `contract_build._refuse_other_interface` — grasp_fj_t2r 덤프는 grasp_s2r 의 키를 물려받아 그 family 로 판정되고, 전에는 `no value for joint r_hj_thumb_1` 에서 **우연히** 멈췄다. 이제 차원 불일치를 이유로 먼저 거절한다 (`test_pc_contract_interface_guard.py`). `right_g1` 계약은 재빌드해도 바이트 동일.

남은 일: grasp_fj family — obs 133 세그먼트 표, 관절공간 디코더(팔 증분 스케일·클립, 손 19 절대 → `-tl` 자산의 잠긴 관절 처리), pd 그룹. hdgp `grasp_fj_t2r/` 의 `_get_observations` · `_pre_physics_step` 을 읽고 시작한다.


## status 골든 테스트가 pour 기록을 빈 것으로 읽던 문제 (09.21)

`pour_fake_run.sh` 는 `status_to_csv.py --nodes pour_node` 로 찍는다. `test_pc_status_join.py` 의 골든 2종은 `logs/policy_control/*/` 을 전부 집어 **4노드 기본값**으로 되읽었고, pour 기록은 "no rows" 가 되어 10건이 떨어졌다(실행 5개 × 2).

- `status_join.nodes_from_fields()` — CSV 헤더 → 기록 당시의 `--nodes`. `row_fields` 의 역함수. 기록이 자기 노드 구성을 말하는 유일한 곳이 헤더다.
- 골든 테스트는 실행마다 헤더에서 노드를 되읽는다. 회귀 2건 추가(헤더 왕복, 다른 노드 기록을 기본값으로 읽으면 빈 결과).
- 도구·CSV 포맷·`summarize()` 문구는 그대로다.

전체 `-m "not gpu"`: **2 failed / 767 passed / 3 skipped**. 남은 2건은 기존 `test_pc_fk_urdf`(구 자산 `openarm_dg5f-m_bi_rl.urdf` 에 palm alias 링크 없음)다.

플레이크 1건 관찰: `test_pc_node_fabric.py::test_control_only_bad_palm_cmd_and_hand_cmd_are_reported_not_applied` 가 전체 실행 3회 중 1회 떨어졌다. 단독 3/3, 파일 단위 15/15, 다음 전체 실행에서는 통과. 떨어진 실행 중에 같은 PC 에서 Isaac Sim 학습(4096 env)이 기동하고 있었다(GPU 2.4 → 19 GB). 0.2 s 펌프로 기다리는 타이밍 테스트라 부하 탓으로 **추정**한다 — 실패 사유 문구는 그 실행에서 잘라내 버려 남지 않았다. 다시 보이면 `_pump` 고정 대기를 조건 대기로 바꾼다.


## s2r_console — 연결 그림과 스위치 (09.21, 사용자 지시로 §4 의 두 항목을 뒤집었다)

사용자 지시 세 번: (1) 입력·정책·실기 상태를 한 창에서, (2) 드라이버 온/오프도 같은 창에서, (3) 표가 아니라 **상자가 선으로 이어지고 정보가 흐르는 그림**으로, 입력 노드를 켜는 토글 포함. §4 의 "Node Graph 시각화 안 함" 과 P4 의 "버튼 없음" 은 이 지시로 철회한다. 철회하면서 §4 가 걱정한 것은 아래처럼 막았다.

| §4 의 걱정 | 지금 구조 |
|---|---|
| expected graph 를 UI 에 적으면 원천이 복제된다 | 그림은 프로파일 yaml 의 `diagram:` 한 곳(`diagram_spec.py`, 불변 레코드). 상자의 스위치는 **미션 yaml 의 배경 명령 키(`단계#번호`)만** 가리킨다. `test_every_shipped_diagram_unit_is_a_background_command_of_its_mission` 이 출고 프로파일 전부에서 이것을 잠근다. status 상자는 프로파일 `status_nodes` 에 있는 이름만 쓸 수 있다 |
| 7칸 상태에 넣으면 `reasons` 가 준다 | 상자는 노드가 status 로 말한 `phase`·`reasons` 를 그대로 옮긴다. 콘솔은 문턱을 따로 두지 않는다(`links.py` 와 같은 규칙) |
| 런타임 `execute` 토글 | **여전히 없다.** 스위치는 프로세스를 띄우고 내릴 뿐이고 pd 의 `execute` 는 argv 로만 정해진다 |
| 콘솔 소프트 estop | **여전히 없다.** estop 은 pd 상자가 fault 로 표시할 뿐이다 |

**프로세스 구조**는 P7 에 적어 둔 그대로다: `bridge.py` 는 구독만 하는 별도 프로세스(우리 코드의 publisher 0 · service client 0, rosout·parameter 서비스 끔 — 단 rclpy 가 만드는 `/parameter_events` publisher 는 못 막는다: 기동마다 use_sim_time 선언 이벤트 1건, `ROS_DOMAIN_ID` 불일치면 init 전에 거부)이고 NDJSON 을 stdout 으로 낸다. API 프로세스는 rclpy 를 import 하지 않는다. 기동은 `deploy/s2r_console/tools/console.sh`(PYTHONPATH 를 덮어쓰지 않고 앞에 붙인다 — 덮어쓰면 브리지가 `No module named rclpy` 로 죽는다).

| 모듈 | 하는 일 |
|---|---|
| `links.py` | 연결 사슬(브리지 → 입력 → 정책 → 실기 → 가드). 순수, 색 표(TONE)도 여기 |
| `diagram_spec.py` | 프로파일 `diagram:` 파싱·검증. boxes(id·col·kind·unit·ports) / wires(from·to·topic·meter·on_demand·stale_ms) |
| `diagram.py` | 관측 → 상자·선 상태. 순수. 선은 발행자·구독자·나이로 판정하되 **받는 노드가 제 입으로 더 나쁘게 말하면 그 말이 이긴다**. `on_demand` 선은 발행자 부재가 아니라 받는 쪽으로 판정. meter 없는 선은 "이어짐" 까지만 말하고 흐른다고 하지 않는다. 머리줄 요약은 끊긴 선을 `from → to (topic)` 으로 이름 붙여 말한다 |
| `units.py` | 스위치 규칙(순수). 아래 표 |
| `console.Console.toggle_unit` | 규칙 통과 시 감독자로 미션의 argv 를 띄우거나 내린다. `POST /api/unit {key, on}` — 리스 필요, **argv 는 HTTP 로 받지 않는다** |
| `web/console.js` `renderDiagram` | SVG 선 + 상자. 상태 지식은 JS 에 없다(서버가 tone 을 준다) |

**스위치가 거절하는 경우**(`units.on_reasons` / `off_reasons`, 거절은 사유 문장과 함께 409, 아무것도 바꾸지 않는다):

| 상황 | 켬 | 끔 |
|---|---|---|
| manual 명령(sudo·전원·다른 PC) | 거절 — 보이되 잠김 | — |
| `touches_real` 단계의 명령 | **거절** — 승인 원장은 단계 실행에만 있고 그림은 그것을 우회하지 않는다 | 아래 규칙 |
| 선행 단계 미완 | 거절(`needs`) | — |
| 제 단계가 실행 중 | 거절 | 거절 |
| pd 가 IDLE 이 아님 + pd 를 띄운 단위 | — | 거절(토크가 끊긴다) |
| pd 가 IDLE 이 아님 + **실기 프로파일** | — | **전부 거절** |
| pd 가 IDLE 이 아님 + fake 프로파일의 입력 단위 | — | 허용(고장 주입용) |

따라서 **실기 프로파일(`left_v2B25_real`, domain 126)에서 그림의 스위치로 켤 수 있는 것은 실기를 건드리지 않는 단계의 배경 명령뿐**이다. 드라이버·pd 는 지금처럼 미션 단계 승인 → 실행으로 뜬다. 이것은 의도한 한계다.

**실증(fake, domain 97, `pour_i18_fake`)**: CDP 로 화면을 몰아 확인 — 에피소드 도중 `plant#1`(컵 포즈 fake)을 스위치로 끄면 cup 상자 off, 선 missing, `pour_node` FAULT 로 넘어가고 머리줄이 끊긴 선을 이름으로 말한다. pd 가 TRACKING 인 동안 `pd_load#0` 끄기는 409 + 사유. 다시 켜면 선이 live 로 돌아온다. 끝난 뒤 남은 프로세스 0.

테스트: `tests/s2r_console/` 15 파일. 그림·스위치 몫은 `test_console_diagram.py` 25, `test_console_diagram_spec.py` 7, `test_console_units.py` 12.


## s2r_console — 그림 자동 생성 · FPP 까지 · 그림 밖 연결 (09.22, 사용자 지시)

지시: (1) FPP 연결부터 다른 ROS2 토픽까지 확인, (2) fabric IK · pd 등을 창이 작아도 그래프로 전부, (3) **pour 에 국한하지 말 것 — 정책을 바꾸면 노드·연결이 자동으로 생겨야 한다.**

전날의 그림은 프로파일 yaml 에 손으로 적은 것이었다(pour 하나). 정책이 바뀌면 그림이 안 따라온다. 이제 **적지 않는다.**

| 무엇 | 어디서 읽는가 (`s2r_console/wiring.py` — 순수, rclpy 없음) |
|---|---|
| 어떤 노드가 뜨는가 | 미션 명령(argv)이 부르는 launch·스크립트: `policy_chain` · `pour_chain` · `pd_controller` · `pour_guard_node` · `fake_plant` · `fake_cup_pose_pub` · `openarm.bimanual` · `dg5f_<side>_driver` |
| 노드가 몇 개인가 | 계약 json 의 `control_only` · `sides` · `primary_side` — launch 의 `chain_nodes` 와 같은 규칙 (정책 계약 → obs·policy·fabric, 제어 전용 → episode_master + fabric direct, `side:=both` → `fabric_node_<side>`) |
| 센서 전선 | robot yaml `sources.<역할>[_<팔>].topic`. 노드별로 실제 구독하는 역할만: obs = 전부(decoder_target 제외), fabric = arm·ee·object, pd = arm·ee, pour_node = arm·ee·tip_force |
| 구동 전선 | robot yaml `groups` + `pd_backends.forward_topic` (position·velocity·effort 셋 다 — position 만 센다) |
| 컵 토픽(pour) | 미션 명령의 `src_cup_topic:=` / `rcv_cup_topic:=` |
| 인지(FPP) | 물체 포즈를 미션이 fake 로 내지 않으면 **카메라 → FPP 추적(`fpp_<이름>`) → object_pose_node → 받는 노드** 를 그린다. 토픽 이름은 `scripts/object_registry.py`. 영상 토픽은 **구독하지 않는다**(그래프만 본다) |
| 받는 쪽 입력 이름 | 노드가 status 로 말하는 그대로: obs = 역할 이름, pd = `<팔>:arm|ee`, pour_node = `<role>:arm|hand|force|cup` · `fill` |

모르는 명령은 버리지 않고 전선 없는 상자로 남긴다. 계약·yaml 을 못 읽으면 예외 대신 그 사유를 상자에 적는다. 되먹임(joint_target → obs, action → obs)은 그리지 않는다(왼쪽 → 오른쪽 규칙).

전선의 새 속성 — 거짓 경보를 없애려고 넣었다(전부 리뷰가 실제 결함으로 확인한 것):
- `heard_by` — ros2_control 컨트롤러는 `/controller_manager` 가 아니라 **제 이름의 노드**로 구독한다. 전에는 실기 구동 전선이 항상 "구독 없음" 이었다.
- `muted` — pd 가 `execute:=false` 면 발행자를 만들지 않는다(`pd_backends._GuardedPublisher`). 끊김이 아니라 "무발행" 으로 표시.
- `episodic` — joint_target·obs·action 은 에피소드 동안만 흐른다. 내는 노드가 **살아서** running 이 아니라고 말할 때만 "쉬는 중"(held). 내는 노드가 조용하면 그대로 끊김.
- 상자의 `stages` — 한 프로세스 안의 단계를 칩으로: pour_node = 입력함 › obs 223 › policy → 18 › decoder › fabric IK ×2, fabric_node = decoder › fabric IK, pd = PD 법칙 › 컨트롤러 교대 › 발행|무발행.

**그림 밖 연결**: 브리지가 도메인 전체의 (토픽 → 내는 노드·받는 노드)를 그래프 조회만으로 읽어 보낸다(`rosgraph.py`, 구독 없음, 바뀔 때 + 5 s 마다). 그림이 선언하지 않은 토픽은 그림 아래 "그림 밖 연결 N개" 표에 나온다. fake 실측에서 나온 것: `/dynamic_joint_states` 를 pd 가 구독하는데 내는 쪽이 없다(fake 에는 온도가 없다), `/policy_control/estop` 은 내는 쪽이 없다(콘솔은 estop 을 내지 않는다 — 설계대로).

화면: 열 수를 CSS 에 넘겨 상자 폭을 나눈다(container query). 실기 일반 체인은 8열(카메라 → FPP → 센서 → obs → policy → fabric → pd → 구동)인데 1500 px 창에 가로 스크롤 없이 들어간다(브리지 없이 띄워 확인 — 도메인 126 에는 붙지 않았다).

검증: fake 도메인 97 에서 자동 생성 그림으로 미션 끝까지. 손으로 적었던(09.21 실측) 전선 20개·11개가 전부 생성 결과에 들어 있다. 에피소드 중 컵 입력 끄기 → FAULT · 선 missing, pd 끄기 409(사유 2개), 다시 켜기 → 복구, 종료 후 남은 프로세스 0. `tests/s2r_console` **317 통과**, ruff 0.

### 리뷰(Workflow, 5 관점 → 반증 검증, 31건 확정)에서 고친 것

| 심각도 | 결함 | 수정 |
|---|---|---|
| CRITICAL | 413 응답이 본문을 읽지 않고 연결을 유지 → 남은 본문이 다음 요청으로 해석된다(요청 밀반입). 다른 출처의 웹 페이지가 `pd_release`·`episode_abort`·lease 강탈을 보낼 수 있었다. 재현 확인 | 오류 응답은 전부 `Connection: close`. `Transfer-Encoding`·음수·비숫자 Content-Length 는 400. 회귀 테스트 6개(`test_console_server.py`) |
| HIGH | pd status 가 없거나 늙으면 phase 가 None = "자유" → 토크를 쥔 pd 를 끌 수 있었다(fail-open) | `PD_UNKNOWN` — 없다는 **증거**(콘솔이 띄운 단위가 죽음 · 그래프에 없음)가 있을 때만 자유. 종료(`end_reasons`)에도 같은 규칙 |
| HIGH | 다른 단계가 도는 동안 pd 단위를 끌 수 있었다(단계가 곧 engage 할 수 있다). 끄기 스레드가 재확인 없이 죽였다 | 실기·pd 단위는 어떤 단계든 도는 동안 잠금. 죽이기 직전에 규칙을 한 번 더 본다 |
| HIGH | 끈 표시가 키에만 붙어, 같은 키로 다시 뜬 프로세스의 **크래시**가 "운영자가 끔" 으로 보였다 | 끈 표시를 pid 와 함께 기록 |
| MEDIUM | `stop()` 이 그룹 리더만 기다렸다 — `ros2 launch` 가 먼저 끝나면 pd_node 가 고아로 남는다 | 그룹이 빌 때까지 기다리고 안 비면 그룹째 SIGKILL (`test_console_supervisor.py`) |

남긴 것(고치지 않음): lease `force` 에 인증이 없다(127.0.0.1 + ssh 터널 전제 — 설계), 그림 전체를 매 스냅샷 다시 그린다(성능), 모달 포커스 관리, 컨트롤러 active 요구가 구동 상자 판정에 반영되지 않는다(표에서는 판정한다).

### 남은 한계
- FPP 추적기·RealSense 의 **노드 이름**은 이 저장소에 없다(인지 PC 의 다른 저장소). 그래서 그 상자는 노드가 아니라 토픽의 발행자 유무로 판정한다. 실기 도메인에서 본 적 없다.
- `mission_pour.yaml`(실기 pour)은 아직 policy_control 체인을 쓰지 않는다(옛 Isaac-in-the-loop 경로) — 생성기가 아는 launch 가 없어 상자만 나온다. 실기 pour 미션을 `pour_chain` 으로 쓰면 그림은 따라온다.
- 새 family(예: grasp_fj — 관절공간 디코더, fabric 없음)가 **새 launch 파일**로 오면 `wiring._HANDLERS` 에 한 항목이 필요하다. 기존 launch 를 쓰는 한(계약·robot yaml 만 다름) 아무것도 안 고쳐도 된다.

### 초록이라고 말할 자격 (09.22 추가, TDD)

자동 생성 이후 화면을 다시 보다가 **거짓 초록 두 개**를 찾아 고쳤다. 셋 다 테스트를 먼저 빨갛게 만들고 고쳤다.

| 무엇 | 왜 거짓이었나 | 지금 |
|---|---|---|
| 머리말 "전부 이어짐" | 상자가 다 초록이면 **전선이 unknown 이어도** 초록이라고 했다. 브리지가 보고하지 않은 토픽은 "모르는 것"이지 "이어진 것"이 아니다 | 상자가 다 떠 있는데 확인 못한 전선이 있으면 `warn` 으로 그 전선을 이름까지 적는다. 의도된 무발행(pd `execute:=false`)과 가끔 오는 값은 그대로 초록 |
| 구동 전선이 inactive 컨트롤러로 흘러도 초록 | ros2_control 컨트롤러는 **configure 에서 구독을 만든다** — inactive 여도 구독자로 보인다. 토픽은 250 Hz 로 흐르는데 명령은 버려진다 | pd → 구동 전선이 live 이면 그 컨트롤러의 상태를 본다. active 가 아니면 `fault` + "명령이 버려진다". 조회 결과가 없으면 상태는 두고 "active 인지는 모른다" 만 적는다 |
| 좁은 상자에서 이름이 `robot_cont / rol` 로 쪼개짐 | 브라우저에서 재 보니 제목이 받은 폭이 **139 px 중 77 px** — 판정 낱말이 같은 줄을 먹었다 | 8열 모드에서 판정 낱말을 아랫줄로 내린다(제목 123 px). 머리말에 색도 붙였다(warn 을 낼 수 있게 됐는데 회색으로 나왔다) |

`tests/s2r_console` **323 통과**, ruff 0. 실기 프로파일 8열 그림을 브리지 없이 1500 px 창에서 재확인(도메인 126 미참가).

생성 결과를 실제로 찍어 보다가 둘 더 나왔다(둘 다 테스트 먼저).
- **fake 의 heard_by 는 컨트롤러가 아니다.** fake 플랜트에서는 구동 전선을 `/fake_arm_bridge` 가 듣는다. 컨트롤러 판정이 여기까지 따라와 "active 인지 모른다" 를 붙였다. 듣는 쪽이 상자 제 노드면 컨트롤러 이야기가 아니다 — 건너뛴다.
- **`execute:=false` 는 선언이지 사실이 아니다.** 실기 미션은 pd 를 `execute:=false` 로만 띄우고, execute:=true 재기동은 콘솔 밖 수동이다(`mission_policy_control.yaml` policy_reduced 의 note). 그림이 계속 "무발행" 이라고 하면 **실기가 움직이는데 그림은 조용하다고 말한다.** 이제 그 토픽에 발행자가 보이면 선언이 지고, 실제 상태 + "미션은 execute:=false 로 띄운다 — 그런데 발행자가 있다" 를 적는다.

`tests/s2r_console` **325 통과**.

### 자동 생성 그림 리뷰 (09.22, Workflow 4관점 → 3표 반증, 확정 8 · 기각 11)

첫 시도는 6 에이전트가 전부 모델 한도로 죽어 결과 0이었다. 다시 돌렸다(61 에이전트, 오류 0). **기각 11건 중 6건은 "검증하는 동안 내가 이미 고쳐서 재현되지 않은 것"** 이다.

| 심각도 | 결함 | 수정 |
|---|---|---|
| HIGH | **한 미션이 같은 노드를 두 번 띄우면 상자가 하나로 합쳐졌다.** `mission_dg5f_m_control.yaml` 은 pd 를 좌·우 따로 띄운다(`sides:=right` · `sides:=left`). 둘째 pd 는 상자도 스위치도 없고, 끄기 보호(`_robot_key`)는 첫째만 가리켰다 — **토크를 쥔 나머지 pd 를 끌 수 있었다** | 상자 id 에 단위를 붙여 둘로 나눈다(`pd@pd_load_left#0`). 보호도 키 하나가 아니라 **집합**(`_robot_keys`)으로 바꿔 pd 를 띄운 단위를 전부 잠근다. `_finish` 의 joint_target·episode 도 pd 마다 잇는다 |
| HIGH | **그리다 실패하면 반쪽이 남았다.** 핸들러가 중간에 예외를 내면 이미 그린 상자·전선은 그대로 남고 그 위에 "모름" 상자가 하나 더 붙어, 끊긴 체인이 그럴듯하게 보였다 | 명령 하나를 그리기 전에 장부를 찍고, 실패하면 되돌린 뒤 상자 하나로만 남긴다 |
| HIGH | **잡는 예외가 좁아 콘솔이 안 떴다.** `(OSError, ValueError, KeyError, yaml.YAMLError)` 밖(계약의 `sides` 가 목록이면 TypeError)은 `generate()` 를 뚫고 나가 `Session.__init__` 에서 죽었다 — 그림 하나 때문에 미션 화면 전체가 안 열린다 | 전부 잡는다. 최종 `parse_diagram` 실패도 `diagram_of` 가 받아 **그림 없이** 세션을 연다 |
| HIGH | **inactive 컨트롤러 판정이 손·그리퍼에는 없었다.** 팔만 매니저를 알고 있었다 | 그리퍼는 `/controller_manager`, DG-5F 손은 `/<namespace>/controller_manager` 를 상자에 준다. fake 에는 손 매니저가 없으므로 fake 보정에서 뗀다 |
| MEDIUM | **fake 손 좌·우가 같은 노드 이름으로 판정됐다.** `/fake_hand_state_pub` 은 좌·우 프로세스가 같은 이름을 쓴다 — 한쪽이 죽어도 이름이 남아 **죽은 손이 초록**이었다 | 같은 프로세스가 만드는 per-side 노드 `/dg5f_<side>/dg5f_<side>_controller` 로 판정한다 |
| MEDIUM | 재지 않는 전선이 `stale_ms` 를 달고 있었다 — 아무도 보지 않는 숫자다 | 재는 전선에만 한계를 단다(forward position 만) |
| MEDIUM | 실기 구동 상자가 영원히 "아직 켜지지 않았다" 였다 — 제 노드도 내보내는 토픽도 없어 판정 근거가 없고, 수동 단위의 "안 띄웠다" 로 떨어졌다 | 콘솔이 띄우지 않는 **수동** 단위는 "안 떴다" 고 말하지 않는다. 구동 상자는 **controller_manager 의 대답**을 근거로 쓴다 |
| LOW | 브리지가 죽어도 "그림 밖 연결" 목록이 그대로 보였다(마지막 스냅샷) | 브리지가 없으면 목록을 내리지 않는다 |
| LOW | 토픽 보고가 없을 때 무발행 선언을 그대로 믿었다 | 그럴 때는 **pd 가 스스로 한 말**(`execute`)을 본다. "명령 나감" 이면 선언이 낡은 것이므로 무발행이라고 하지 않는다 |
| LOW | `bridge.py` 가 "DDS 에 아무것도 쓰지 않는다" 고 적혀 있었다 | 사실과 맞췄다: rclpy 가 노드마다 `/parameter_events` publisher 를 만들고 기동 때 `use_sim_time` 선언 이벤트 1건이 나간다. 우리 코드의 publisher·service client 는 여전히 0 |

무발행(`muted`) 규칙도 두 갈래로 나눴다: **내는 쪽 사유**(pd `execute:=false`)는 정말 흐르면 지고, **받는 쪽 사유**(fake 손은 JTC 를 안 받는다)는 내는 쪽이 있어도 유지된다.

`tests/s2r_console` **339 통과**, ruff 0. fake 도메인 97 에서 전 과정 재검증(20/26 live · 컵 입력 차단 → FAULT · pd 끄기 409 · 복구 · 종료 후 남은 프로세스 0).

**아직 안 고친 것**: pd 의 `/dynamic_joint_states`(발열 게이트)·`/policy_control/estop` 은 전선으로 그리지 않는다 — 내는 쪽 상자가 그림에 없어서다. 둘 다 "그림 밖 연결" 표에는 나오고, estop 은 pd 상자가 래치 여부를 따로 말한다. 재지 않는 전선(`meter=False`)이 발행자 유무만으로 초록인 것도 그대로다 — 세려면 브리지가 구독해야 하고, 그건 카메라 토픽에서 일부러 피한 비용이다.

## s2r_console — 인지(FP++)를 vision-3090 쪽으로 잇는다 (09.22, 사용자 지시)

지시: "FPP 는 vision3090 쪽 연결해야함."

전까지 그림은 카메라·FPP 상자를 **토픽 발행자 유무**로만 판정했고(영상 토픽은 일부러 구독하지 않는다) 스위치가 없었다. 그런데 저장소에는 이미 그 PC 를 다루는 물건이 있었다 — 그것을 콘솔에 물렸다.

| 어디서 도는가 | 무엇 |
|---|---|
| **vision-3090** | RealSense 노드 · `fpp_<물체>` docker 컨테이너. `scripts/vision/*.sh` 가 저 PC 에서 돈다(`/home/usr/rl_ws`, 도메인 126) |
| 이 PC | `scripts/nodes/perception_launcher_node.py` — tailscale ssh 로 저 스크립트를 부르고, 저 PC 의 `docker ps`·프로세스를 읽어 `/perception/status` 를 1 Hz 로 낸다. 스스로는 아무것도 켜지 않는다 |
| 이 PC | `object_pose_node` — `/perception_plus_plus/<이름>/pose` → base_link → `/objects/<이름>/pose` |
| 이 PC | `scripts/ops/perception_ctl.py` — `/perception/cmd` 로 start/stop 을 보내는 사용자 면 |

콘솔에 넣은 것:
- **생성기**가 `perception_launcher_node.py` 를 알아본다 → 상자 `인지 런처 · vision-3090`, 카메라보다 **왼쪽 열**(그 PC 를 켜는 것이 먼저다), 미션 단위가 붙으므로 **스위치가 생긴다**. 인지 사슬에서 콘솔이 켤 수 있는 유일한 것이다.
- **상자에 호스트를 적는다**(`Box.host`). 카메라·FPP 는 `vision-3090` 칩이 붙는다 — 어디로 가서 고칠지가 달라진다.
- **판정의 진실원천이 바뀐다**: 브리지가 `/perception/status` 를 구독하고(구독 전용, String JSON), 카메라는 `camera_up`·`camera_hz` 로, 추적기는 **컨테이너 유무와 포즈 나이**로 판정한다. 런처의 `error`(ssh 실패 등)는 런처 상자를 FAULT 로 만든다. 런처가 없으면 예전처럼 토픽만으로 본다(약하지만 증거는 증거다).
- 실기 미션 `bringup` 에 런처를 **배경 단위**로 넣었다. 띄워도 저 PC 에 아무것도 켜지 않는다 — 상태만 읽는다. FP++ 를 실제로 켜는 것은 여전히 `perception_ctl.py start <물체>` 다.

실기 체인은 이제 **10열**이다(인지 런처 → 카메라 → FPP → object_pose → obs → policy → fabric → pd → 팔 구동·그리퍼 구동 → 모르는 수동 명령). 1500 px 창에서 가로 스크롤 **0**(브라우저 실측 `scrollWidth == clientWidth == 1462`). 열이 많아지면 간격과 상자 최소폭을 함께 줄인다.

곁들여 고친 것: 모르는 명령 상자가 "bash" 라고만 적혀 체인 사이에 끼어 있었다 — 맨 오른쪽으로 모으고 미션이 붙인 설명을 상자 안에 적는다.

**실측**: `ssh vision-3090 scripts/vision/status.sh` → `{"camera_up": false, "containers": {}, "viewer_up": false}`. 지금 인지 체인은 전부 꺼져 있다. `tests/s2r_console` **350 통과**, ruff 0.

**도메인 126 실측 (09.22, 사용자 승인 후)**: 런처를 띄우고 콘솔 브리지를 12 s 돌렸다.
- `/perception/status` 12건 수신(1 Hz). 스키마 일치: `{"camera_up":false,"camera_hz":0.0,"objects":{"shaker_closed":{...},"cup_big_s100":{...}},"viewer":false,"busy":false,"error":null}`
- 그 줄들을 그대로 `Feed` → `diagram.build` 에 먹인 결과: 런처 **live**(카메라 down · 컨테이너 둘 다 없음), 카메라 **off**("vision-3090 에서 카메라가 떠 있지 않다 — 인지 런처로 켤 것"), FPP 추적 **off**("컨테이너가 없다"), object_pose **missing**(노드 안 떠 있음). ssh 로 직접 잰 `{"camera_up": false, "containers": {}}` 와 일치한다.
- 브리지의 도메인 방어선도 같이 확인됐다: env `ROS_DOMAIN_ID` 가 비면 `--domain 126` 이라도 **거부**하고 fault 한 줄만 낸다.
- 런처·브리지는 확인 뒤 PID 로 종료. 도메인 126 에 남긴 프로세스 0.

화면(브리지를 붙인 실기 콘솔) 스크린샷은 못 찍었다 — 실기 도메인에 포트를 여는 상주 기동이 자동 승인에서 막힌다. 판정은 위 실측 데이터로 확인했다.
