# deploy 실패 상태 대조 (97fde86)

supervisor는 deploy의 `FailureCode`, `RecoveryManager`가 읽는 모듈 전역 `_TABLE`, `SAFETY`를 선택적으로 import한다. 설치되지 않은 환경은 경고 후 아래 고정 표로 fallback한다. 새 코드가 enum에만 추가되어도 복구 표에 없으면 정지한다. enum에도 없는 코드는 `unknown_code=true` 및 경고로 기록한다.

`last.code`와 `last.results[].signals.outcome`은 서로 다른 이름 공간이다. `reason`은 고정 enum이 아니라 센서 값·경로·서비스 응답·예외를 포함하는 자유 문자열이다. 아래는 코드 전수 목록과 reason 판정 규칙이며, 뒤의 부록은 실제 reason 생성식이다. `episode_runner_node`는 manager의 상태를 발행하며 서비스의 `reasons`를 `last.reason`으로 새로 정의하지 않는다.

| code (전수) | deploy 복구 | 변경 전 supervisor | 변경 후 supervisor |
|---|---|---|---|
| target_not_found, holder_not_found, lid_not_found | retry + refresh_objects/refresh_holders | VLM B/C/D 요청 | READY의 일치하는 pending만 recover; 없으면 사람 정지 |
| grasp_failed_left, grasp_failed_right, pose_mismatch | rollback_retry + checkpoint; 파지 실패는 open_hand | 일반 retry/home 가능 | runner pending으로만 rollback 승인; shell 우회 금지 |
| pour_failed, lock_failed | retry | VLM 복구 요청 | 일치하는 pending으로만 runner recover; pending 없으면 reason 문구와 무관하게 정지 |
| controller_error, observation_error, unknown, operator_declined, lift_guard_failed, policy_timeout | safe_stop | FAILURE에서 복구 차단 | READY/FAILURE 모두 복구 차단 |
| object_dropped_left, object_dropped_right, pour_partial, place_failed, shake_aborted | safe_stop (표에 없음) | retry/rescan/home 가능 | 사람 정지; VLM에 B/C/D 미제공, 잘못 응답해도 실행 차단 |
| none | 정상 | VLM 판정 | 정상 READY/SUCCESS만 진행; FAILURE에서는 정지 |
| waiting_confirm, confirmed | runner 메타 코드 | VLM/confirm 가드 | confirm 가드 유지; 운영자 확인 대기 자동 승인 없음 |
| place_failed_continued | runner가 continue_on_fail 적용함 | 일반 완료처럼 판정 | runner가 이미 적용한 결과만 시각 판정; supervisor가 예외를 새로 적용하지 않음 |
| done | proto 호환 코드 (실 runner enum 아님) | 정상 | 정상 호환 유지 |
| 미지 코드 / code 누락·None | 기본 safe_stop | 일부 복구 가능 | 미지 로그 + 사람 정지 |

`RECOVERY`는 shell 동사 목록이다. deploy 복구 action과 같은 표가 아니다. 이번 변경은 이 목록 자체를 rollback 구현으로 바꾸지 않고, 허용 action 검사를 앞에 둔다. 사람 정지는 shell에 새 에피소드를 만들도록 요청하지 않고 실제 stop 서비스를 호출한다. runner pending 복구는 `recover:<node>:<action>` 승인 이름을 확인하고 `next` 서비스를 통해 준비·checkpoint·retry 상한을 실행한다. 일치하는 승인 이름이 없으면 사람 정지한다. 준비 단계가 있는 retry를 pending 없이 shell로 실행하는 것도 막는다.

| outcome / 규칙 (code 아님) | 실제 의미와 노출 | 변경 전 → 변경 후 |
|---|---|---|
| clear | 래치 후 빼기 clearance ≥ 0.05 m | 일반 시각 판정 유지, 상세 outcome 기록 |
| withdraw_timeout | 빼기 120틱 상한; place_failed, continue 금지 | 일반 실패 취급 → safe_stop, 상세 기록 |
| episode_end | 학습 시간 끝 정상 stop; hand_empty=None | 카드에는 시간 초과 abort → 규칙 설명 교정, None을 unknown으로 보존 |
| opened_before_latch | 목표·실측 열림 0.15 rad, 3틱; 촉각/관절 힘 조건. holder 판정 z 허용 15 mm; 촉각 ≥ release_force_n이면 hand_empty=False. 촉각 없으면 검사 생략 기록 | 카드의 outcome=opened → 실제 outcome/힘 조건 반영, rule로 별도 분류 |
| place_hand_unknown | runner 로그 이벤트. opened_before_latch + cup_in_holder=False이면 손 걸림으로 보고 continue_on_fail 차단 | 구분 없음 → 같은 signals 조합에서 rule 태그, 사람 정지 |
| track abort | ready_carry.still=track, start_ok=False; 명령−실측 이탈 3틱, NaN도 위반 | 구분 없음 → track_abort 태그와 원문·signals 보존 |
| start_ok 거부 | 이송 후 시작 위치·축 문턱 미달. 축 상한 4.7°; 정책 미시작 | 구분 없음 → start_ok_rejected 태그, 미지/금지 코드 정지 |
| 홀더 refresh/jump 거부 | last_check result: written, measured, unstable, jump_rejected, no_markers, failed 및 읽기 실패 문자열. 사용 홀더 전부 안정·기존 자리 근처 재확인 예외는 deploy만 판단 | 구분 없음 → 상세가 status에 있으면 holder_refresh_rejected 태그; 시작 전 거부는 status가 생기지 않을 수도 있음 |
| rehome refused | 실패 컵 unknown 위치를 신선한 live(5초)에서도 못 얻으면 NodeResult(refused); 보통 code unknown | 일반 미지 실패 → rehome_refused 태그 + 사람 정지 |
| None / 누락 | outcome 미제공·손 상태 미확인 | 명시적 unknown 보존; None을 False/빈손으로 변환하지 않음 |

현재 rh_place의 outcome 값은 `None`, `clear`, `withdraw_timeout`, `opened_before_latch`, `episode_end`다. `opened`는 오래된 카드 표현으로 현재 outcome이 아니다. ready_carry의 `still` 값은 빈 문자열, ok, timeout, track이다. timeout만으로 반드시 거부되는 것은 아니고 최종 start_ok가 판단한다.

홀더 시작 검사: 마지막 기록 없음/비객체/읽기 실패, 미확인 사용 홀더, rc 실패/no_markers, epoch 없음·NaN·bool, 미래 >120초, 나이 >7일을 거부한다. 기본 점프 상한은 3 cm/10°. 과거 커밋의 cups 뒤 재측정 및 6시간 설명보다 현재 cup_holders 단계 확정·7일 규칙이 우선이다. 이는 시작 전 console/plan 검사이므로 `last.code=jump_rejected`라는 새 enum을 만들어 해석하지 않는다.

## reason → FailureCode 규칙 (위에서 먼저 일치하는 규칙 우선)

| reason 키워드 / 상태·신호 | code |
|---|---|
| lift_guard | lift_guard_failed |
| tool timeout | controller_error |
| place 시간 초과 | policy_timeout |
| holder pose missing | holder_not_found |
| 컵 자세가 없다 / cup pose missing / goal comes from the cup / cup outside region | target_not_found; lock이면 lid_not_found |
| not grasped | grasp_failed_left/right |
| held target / training start pose / rad from / start pose | pose_mismatch |
| estop / thermal / controller / pd phase / watchdog | controller_error |
| source / stale / missing / ticks without a valid step | observation_error |
| refused/aborted이고 위 키워드 없음 | unknown |
| pd_fault | controller_error (reason보다 우선) |
| status=error | trajectory는 controller_error, 그 외 observation_error |
| status=timeout | policy_timeout |
| completed의 dropped / attached=False / live_off_m>0.04 / setting_err_m>0.04 | object_dropped_left/right / grasp_failed_left/right / pose_mismatch; 들기 가드 성공 예외는 deploy 적용 |
| completed place: cup_in_holder=False / hand_empty=False / withdraw_timeout | place_failed |
| completed pour: pour_ok=False | partial이면 pour_partial, 아니면 pour_failed |
| completed lock: locked=False | lock_failed |
| 그 외 completed | none |

추가 reason: RecoveryManager의 안전 규칙·policy_timeout·재시도 소진·checkpoint 없음·복구 수단 없음 문구, runner의 줄 노드 정지·복구 준비 실패·rollback 실패·재snapshot 실패·승인 없음·확인 대기·continue 문구가 원인에 붙는다. 원문은 decision.runner_detail.reason 및 results에 보존하고 VLM 프롬프트에도 같은 구조를 전달한다. 규칙 태그와 물리 실패/미분류를 별도 필드로 기록하며 센서 증거 없는 물리 원인을 추측하지 않는다.

`continue_on_fail` 상태라도 `last.code`가 `policy_timeout` 또는 `place_failed`이면 supervisor는 정지한다. 이는 의도적으로 보수적인 정책이다. `place_failed_continued`는 next/stop/human만 허용하고, 규칙이 있는 다른 상태는 일치하는 pending 복구가 없으면 stop/human만 허용한다.

## 검증과 한계

실 deploy import 및 모든 enum/표 대조, import 실패 fallback 경고, 미지·None 코드, 금지 복구, pending rollback, 준비 단계 우회, rule 상세, VLM 금지 응답 차단을 회귀 검사한다. 실기·ROS 노드·sim·GPU·SSH 실행은 하지 않는다. 기본 작업 환경에서 policy_control 패키지가 import 경로에 없으면 fallback 경고가 정상이다. 비교용 proto 실행기는 패키지 진입점이 아니므로 runtime은 수정하지 않았고 카드 사본만 동기화했다. 기본 fallback은 deploy 업데이트를 자동으로 읽지 못하므로 배포 시 policy_control이 같은 환경에서 import되어야 한다. `_TABLE`은 비공개 API라 import 불가 시 fallback으로 전환되며, 그 외 형식/실행 오류는 전파한다. active 모드에서 fallback을 쓰면 error 이벤트를 기록한다.

## reason 생성식 부록

문자열은 런타임 값으로 확장되므로 고정 문자열의 유한 목록이 아니라 소스 생성식으로 열거한다. 서비스/정책이 주는 이유와 외부 도구 tail·예외 문자열도 그대로 포함될 수 있다.

### episode_ros.py

| 줄 | reason 생성식 |
|---|---|
| 666 | `f'cup pose missing — {exc}'` |
| 669 | `f'cup pose missing — FP++ 다시 찍기 rc {rc}: {tail}'` |
| 695 | `'cup pose missing — ' + ', '.join((f"{n}({why.get(n, '없음')})" for n in missing))` |
| 700 | `''` |
| 709 | `f'{type(exc).__name__}: {exc}'` |
| 725 | `f'{p.kind} 정책 노드가 아직 없다'` |
| 735 | `f'place tin_json: {side} 팔 tilt 직전 T_in 없음 — {why}'` |
| 739 | `'; '.join(why)` |
| 743 | `'episode runner stop'` |
| 745 | `'; '.join(why)` |
| 749 | `'episode runner stop'` |
| 751 | `f'lift_guard: {why}'` |
| 767 | `f"goal {p.setting} refused: {(res or {}).get('reasons', 'no answer')}"` |
| 769 | `'episode runner stop'` |
| 772 | `'; '.join(why)` |
| 777 | `'episode runner stop'` |
| 781 | `fault` |
| 789 | `f'no stop event in {limit:.0f} s'` |
| 793 | `reasons` |
| 795 | `f'operator stopped the policy node ({reasons})'` |
| 797 | `reasons` |
| 803 | `f'episode runner stop — 가드 고정 판정 중 정지 ({reasons})'` |
| 806 | `f'{GUARD_FAIL_TEXT}: {why} ({reasons})'` |
| 807 | `reasons` |
| 987 | `'noop'` |
| 995 | `f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}"` |
| 998 | `f'{side} hand holds {world.hand(side)} — rehome 계획기가 든 물체를 모른다'` |
| 1014 | `f'{side} rehome: {e}'` |
| 1030 | `'episode runner stop'` |
| 1033 | `f"{('no safe path — ' if k < 2 else '')}{Path(argv[1]).name} rc {rc}: {tail}"` |
| 1039 | `f"이미 홈: {','.join(skipped)}" if skipped else ''` |
| 1063 | `f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}"` |
| 1067 | `f'{traj.kind}: {side} 손이 비었다 — 컵을 쥔 상태에서만(손 안 컵 자리 기준)'` |
| 1069 | `f'{traj.kind}(빈손): {side} 손에 {held} — 빈손 이송은 손이 빈 때만'` |
| 1081 | `'episode runner stop'` |
| 1093 | `f'reach: {traj.object} 컵 자세가 없거나 움직인다({SNAPSHOT_WINDOW_S} s 창, {SNAPSHOT_STILL_M * 1000:.0f} mm) — FP++ 확인'` |
| 1097 | `f'reach: {traj.object} 토픽에서 실물 이름을 못 읽었다'` |
| 1122 | `'episode runner stop'` |
| 1124 | `why` |
| 1129 | `why` |
| 1132 | `'episode runner stop'` |
| 1137 | `f'{Path(argv[1]).name} rc {rc}: {tail}'` |
| 1141 | `f"들기 끝 · FP++ 컵 {('전' if before is None else '뒤')} 프레임 없음 — 들림 판정 보류"` |
| 1146 | `f'컵이 {rise * 100:.1f} cm 만 올라갔다(< {LIFT_RISE_MIN_M * 100:.0f} cm) — 못 쥐었거나 놓쳤다'` |
| 1148 | `f'컵 {rise * 100:.1f} cm 들림 · 유지 끝'` |
| 1149 | `''` |
| 1170 | `f"pd_{side} 가 TRACKING 이 아니다 · {pd['pd_phase']} {pd['pd_reasons']}"` |
| 1173 | `f'hand {node.action}: {side} 손에 {world.hand(side)} — 빈손에서만(펴면 떨어진다)'` |
| 1178 | `'episode runner stop'` |
| 1181 | `f"hand_release {side} 거부 · pd {pd['pd_phase']} {pd['pd_reasons']}: {tail}"` |
| 1188 | `f'손 펴짐 {errs}'` |
| 1191 | `'episode runner stop'` |
| 1192 | `f'손이 {HAND_OPEN_WAIT_S:.0f} s 안에 안 펴졌다 — max\|q\| {errs}(허용 {HAND_OPEN_TOL_RAD} rad, None = 손 실측 없음)'` |
| 1197 | `f'hand grasp: {contract} 가 없다(hand_contract {node.hand_contract})'` |
| 1202 | `'episode runner stop'` |
| 1204 | `tail` |
| 1206 | `f'hand_grasp rc 2: {tail}'` |
| 1208 | `f'hand_grasp rc 3: {tail}'` |
| 1209 | `f'hand_grasp rc {rc}: {tail}'` |
| 1441 | `f"pd_{side} 가 TRACKING 으로 안 돌아온다 · {pd['pd_phase']} {pd['pd_reasons']}"` |
| 1446 | `f"hand_release {side} 거부 · pd {pd['pd_phase']} {pd['pd_reasons']}: {tail}"` |
| 1449 | `''` |
| 1459 | `''` |
| 1461 | `f'cup pose missing — {names} 프레임이 5 s 동안 없다'` |
| 1468 | `f'holder pose missing: {missing} ({self.ep.holder_poses})'` |
| 1470 | `''` |
| 1471 | `f'모르는 준비 {step}'` |

### episode_runner.py

| 줄 | reason 생성식 |
|---|---|
| 320 | `'episode runner stop — 궤적 차례를 기다리는 중 정지'` |
| 425 | `f'{type(exc).__name__}: {exc}'` |
| 435 | `'cup outside region — ' + '; '.join(miss)` |
| 473 | `f"{self.last.get('reason') or code.value} — 줄 노드는 복구 없이 정지(다른 줄도 멈춤)"` |
| 484 | `reason` |
| 500 | `f'복구 준비 {pre} 실패: {res.reason}'` |
| 509 | `f'{rec.checkpoint} 로 못 돌아갔다: {res.reason}'` |
| 515 | `f'되돌아간 뒤 배치 재기록 실패: {res.reason}'` |

### episode_failure.py

| 줄 | reason 생성식 |
|---|---|
| 200 | `'policy_timeout — 현재 팔·손 목표 유지, 복구는 운영자 rehome'` |
| 202 | `f'{code.value} — 안전 규칙: 복구 없이 정지'` |
| 204 | `f'{code.value} — 재시도 {attempts}/{ep.max_retry(node)} 다 씀'` |
| 209 | `f'{code.value} — 손에 든 물체가 같은 되돌아갈 checkpoint(trajectory)가 없다'` |
| 210 | `f'{code.value} — {cp} 로 되돌아가 다시'` |
| 212 | `f'{code.value} — 이 저장소에 복구 수단이 없다(장면 재평가 · 재파지는 Step 3)'` |
| 213 | `f'{code.value} — 같은 노드 다시'` |

### rh_place_node.py

| 줄 | reason 생성식 |
|---|---|
| 382 | `f'준비 자세 이송 시간 초과 {lim:.1f} s · {rc.as_dict()}'` |
| 388 | `f'준비 자세 이송 끝 문턱 미달 → 놓기 정책 시작 안 함 · {rc.as_dict()}'` |
| 400 | `f'placed — 래치 {c.release_timed_steps} 틱 뒤 놓음 → 빼기 {self.chain.settle_t} 틱 ({self.chain.outcome}, 손–컵 {self.chain.clearance})'` |
| 403 | `f'placed — 놓음 {c.release_steps} 스텝 + 스크립트 {c.settle_steps} 스텝 끝'` |
| 419 | `f'opened before latch — 정책이 래치 전에 손을 폄(opened {self.chain.opened:.2f} · 실측 {self.chain.opened_meas:.2f} rad, 기준 {self.chain.open_min_rad:.2f}, {OPENED_END_TICKS} 틱 · 촉각 {self.chain.tact_max} N · 관절 힘 {self.chain.jf_max} g) · 놓임 판정은 cup_in_holder → 다음 노드(home)가 뺀다'` |
| 430 | `f'placed — 학습 시간 {self.max_s:.1f} s(episode_length_s) 끝 · 래치 {self.chain.latch} · {self.chain.latch_t} 틱 · 손 폄 {self.chain.opened:.2f} rad → 다음 노드(home)가 뺀다'` |

### pour_fj_node.py

| 줄 | reason 생성식 |
|---|---|
| 884 | `'user stop'` |
| 888 | `'abort requested'` |
| 1120 | `why` |
| 1201 | `str(exc)` |
| 1208 | `f'{self._gap} ticks without a valid step: {exc}'` |
| 1216 | `f'lift_guard tick: {type(exc).__name__}: {exc}'` |
| 1234 | `f'{path} 시간 초과 {self.max_s:.1f} s — 단계 {ag.phase}'` |
| 1236 | `f'episode time >= {self.max_s:.1f} s (학습 episode_length_s)'` |

