# logs/ 관리 규칙 (2026-11-09 제정)

## 왜 만들었나 (11.09 까지의 문제)

1. **필요 여부를 알 수 없다.** 러너가 `<시각>_<이름>` 폴더를 자동으로 만들 뿐, 그 런이 실험인지 디버그인지 실패한 시도인지 기록하지 않는다.
   `logs/episode/` 에만 64개가 있고, README 가 있는 폴더는 2개뿐이다.
2. **같은 종류가 여러 곳에 흩어져 있다.** `episode/` 와 `episodes/`가 따로 있다. bag 은 `bags/`, `calib/bags/`, `policy_control/real_runs/`, 영상은 `vision_runs/`, `report_media/`, `bags/*/camera` 로 나뉜다.
   로그 파일도 `perception/`, `console/`, `s2r_console_app.log` 로 흩어져 있다.
3. **날짜를 붙인 최상위 폴더가 많다.** `cup_seg_1107`, `holder_seg_1107`, `head_d435i_camera_info_1103`, `monitor_1104` 같은 폴더는 최상위만 봐서는 무엇인지, 아직 필요한지 알 수 없다.
4. **코드가 읽는 파일과 지워도 되는 파일이 섞여 있다.** 배포 계약(`policy/…/deploy_contract.json`), teach npz, 계획 npz 는 코드가 하드코딩 경로로 읽는다.
   그런데 같은 폴더에 일회성 로그가 함께 있다. 고정 파일명(`policy_control/rehome_left.npz`, `calib/*.png`)은 실행할 때마다 덮어써서 이전 값이 남지 않는다.
5. **재현 정보가 없다.** git HEAD 와 설정 사본을 남기는 러너는 `baseline_run.py` 와 `stationary_tracking.py` 둘뿐이다.
6. **쓰레기 로그가 크다.** 권한 오류를 반복 출력한 `realsense_try1_imu_denied.log` 하나가 1.1G 다. `calib/bags/` 원시 bag 은 4.3G 다.

## 원칙: 데이터마다 종류를 하나 정한다

| 종류 | 무엇 | 어디 | 수명 |
|---|---|---|---|
| **R 참조** | 코드나 설정이 경로로 읽는 것: 배포 계약, 캘리브 결과, teach 자세, 계획 npz, 학습된 마스크/모델 | 지금 경로 그대로 둔다(`policy/`, `place_teach/`, `policy_control/*.npz`, `calib/` 의 결과 파일) | 영구. 지우지 않는다. 덮어쓰지 않고 새 이름으로 만든 뒤 코드에서 가리키는 경로를 바꾼다 |
| **E 실험** | 승인받은 실험 하나에 속한 런 | `logs/exp/<YYMMDD>_<이름>/`. 원시 데이터는 러너 기본 경로에 두고 실험 README 에 등록한다 | 결과를 쓸 때까지 로컬에 둔다. 그 뒤 원시 데이터는 DATA 로 옮긴다 |
| **S 임시** | 디버그, 시운전, 가짜(fake) 실행, 콘솔·노드 로그, pid, 미션 상태 | `logs/runs/<YYMMDD>/<종류>/` (러너 기본 경로) | **실험 README 에 등록되지 않은 런은 모두 S 다.** 7일이 지나면 지워도 된다 |

핵심 규칙: **지키고 싶은 런은 실험 README 의 런 표에 적는다.** 표에 없는 런은 임시로 보고 정리할 때 지운다.
런이 끝난 뒤가 아니라 런을 시작하기 전이나 직후에 적는다.

## E 실험 폴더

```
logs/exp/<YYMMDD>_<이름>/          예: 261109_friction_right
  README.md      목적·판정 기준·런 표·결론 (틀은 logs/exp/_template/README.md)
  setup/         설정 사본, `git rev-parse HEAD` + `git status --short`, 계획
  result/        요약 표·그림 (작으니 오래 둔다)
```

- 폴더 이름: `<YYMMDD>_<무엇>_<side>`. 날짜는 실험을 시작한 날이다. 같은 실험을 다시 하면 `_b`, `_c` 를 붙인다. 폴더를 다시 쓰지 않는다.
- 원시 데이터(bag, episode, inhand npz 등)는 옮기지 않는다. 러너가 만든 경로를 그대로 둔다. 다른 README 와 기본 인자가 그 경로를 가리키고 있기 때문이다.
  대신 README 런 표에 경로를 적는다.

  | 런 | 원시 경로 | 판정 | 메모 |
  |---|---|---|---|
  | r01 | `logs/runs/261109/episode/101500_grasp_f_right` | ok | |
  | r02 | `logs/runs/261109/bags/102030_episode_grasp_f_right` | 버림 | 손가락 영점 틀어짐 |

  "버림"도 표에 적는다. 왜 버렸는지가 남아야 같은 실수를 반복하지 않는다. 다만 원시 데이터는 정리할 때 지워도 된다.
- 이미 실험 폴더를 갖고 있는 분석(`analysis/friction/` 처럼 repo 안에 있는 것)은 그 구조를 그대로 쓴다. 이 README 와 같은 형식이다.

## 새 코드를 쓸 때

- 런 기록 경로는 `scripts/run_paths.py` 하나로 정한다. 손으로 `logs/...` 를 만들지 않는다.
  - 런마다 폴더: `run_paths.run_dir("<종류>", label)` → `logs/runs/<YYMMDD>/<종류>/<HHMMSS>_<label>/`. 같은 초에 겹치면 `_2`, `_3` 을 붙인다.
  - 하루에 파일 하나: `run_paths.day_dir("<종류>") / "<이름>"` → `logs/runs/<YYMMDD>/<종류>/<이름>`.
  - 셸은 `$SIM2REAL/logs/runs/$(date +%y%m%d)/<종류>`, 미션 yaml 은 `{runs}/<종류>/…`(러너가 오늘 날짜 폴더로 푼다).
  - 날짜는 런을 시작한 날이다. 자정을 넘겨도 그 런의 파일은 시작한 날 폴더에 둔다.
- 앞선 런을 읽는 코드(가장 최근 기록, 남은 PID, 넘김 묶음)는 `run_paths.find_runs` · `latest` 로 모든 날을 본다. 오늘 폴더만 보지 않는다.
- `logs/` 바로 아래에 새 폴더를 만들지 않는다. R 파일만 예외다. 고정 파일명에 덮어쓰지 않는다. R 종류라면 새 이름으로 만든다.
- 경로는 repo 루트 기준 절대 경로로 정한다. cwd 에 따라 상대경로 `logs/...` 가 다른 곳을 가리키면 안 된다.
- 런 폴더에 `meta.json` 을 남긴다: `{"git": HEAD, "dirty": bool, "argv": [...], "host": ..., "start": ...}`.
- 자산(궤적) 버전이 결과를 가르면 이름에 자산을 넣는다. 지금 적용된 곳(11.09):
  - `tilt/<시각>_setting_<자산>/`, `tilt/<시각>_pour_<자산>/` (예: `_setting_base_arc6`, `_pour_traj_v12_240`) + `meta.json`
  - `episode/<시각>_<에피소드>/meta.json` 의 `assets` 에 궤적별 자산 버전(`tilt: tilt/traj_v12_240`)이 남는다. 같은 에피소드 이름이라도 v11·v12 가 여기서 갈린다.
- 노드 로그가 같은 오류를 반복하면 파일이 GB 단위로 커진다. 오래 띄우는 노드는 로그 크기나 줄 수에 상한을 둔다.

## 정리 (주 1회 또는 실험을 마칠 때)

1. 실험이 끝나면 README 결론과 `result/` 를 쓰고, 상태를 `완료` 로 바꾼다.
2. 끝난 날은 날짜 폴더째 옮긴다: `logs/runs/<YYMMDD>/` → `/media/user/DATA/<YYMMDD>/sim2real/logs/runs/<YYMMDD>/`.
   실험 README 런 표의 경로 옆에 `→ DATA/<YYMMDD>` 를 적는다.
3. 어느 실험에도 등록되지 않았고 7일이 지난 날짜 폴더는 지운다. 지우기 전에 목록을 사용자에게 보여 준다.
4. R 은 건드리지 않는다.

## 지금 있는 폴더 (11.09 날짜별로 바꾼 뒤)

| 폴더 | 종류 | 무엇 |
|---|---|---|
| `policy/` | R | 배포 계약 · 학생 마스크. 코드가 읽는다 |
| `place_teach/` | R | 놓기 teach npz |
| `policy_control/` | R | `rehome_left/right.npz`, 계획 npz(`home_path_*` · `approach_*` · `reset_*`) |
| `calib/` | R | 보정 결과(지금 없음). 겹친 그림 · 측정은 `runs/<날짜>/calib/` |
| `exp/` | E | 실험 폴더(`_template/` 틀) |
| `s2r_console/` | S | 콘솔 실기 평가 회차 `eval/<YYYY-MM-DD>.csv` · 정책 카드 배치 `workspaces/<로봇 해시>.json`. 콘솔이 직접 쓴다(옛 `runs/s2r_console`, 지금은 이리로 가는 링크) |
| `runs/<YYMMDD>/<종류>/` | S | 런 기록. 종류는 아래 |
| `runs/episode_approvals.jsonl` | S | 에피소드 실행기 승인 통로. 날짜와 무관하다(노드는 뜬 뒤 쓰인 줄만 읽는다) |

`runs/<YYMMDD>/` 아래 종류:

| 종류 | 무엇 |
|---|---|
| `episode/` | 에피소드 실행기 런 `<시각>_<에피소드>/` |
| `mission/` | 미션 러너 · 콘솔 세션 상태와 로그 `<시각>[__<프로필>]/` |
| `tilt/` | `tilt_now.py` · `tilt_pour.py` 런 `<시각>_setting_<자산>` · `_pour_<자산>` |
| `inhand/` | 손 안 넘김 묶음 npz. place 가 오늘 · 어제 것을 읽는다 |
| `bags/` | `rh56f1_record.sh` 등 bag `<시각>_<이름>/{arm,hand,camera}` |
| `rh56f1_record/` | bag 기록기 PID · 로그(이름별). stop 은 모든 날을 본다 |
| `rh56f1_tip_zero/` | 손끝 영점 시도 `tip_zero.jsonl` |
| `rh56f1_hand/` | 손 드라이버 PID · 로그(`rh56f1_hand_up.sh`) |
| `cpu/` | 프로세스별 CPU 기록 csv |
| `perception/` | 인지 노드 로그 · PID(`scripts/vision/*.sh`) |
| `console/` | 콘솔 앱 출력 `s2r_console_app.log` |
| `vision_runs/`, `report_media/` | 카메라 JPEG · FP++ 좌표, 보고서 영상 · 사진 |
| `stationary/` | stationary 기준선(`baseline_run.py`) · 추적(`stationary_tracking.py`) 런 |
| `eval_real/` | 실기 회차 기록 `tilt.jsonl` · `grasp.jsonl`(`eval_record.py`) |
| `sim_traj/` | sim 궤적 계획 `card_<카드>/<시각_팔_물체>/` |
| `grasp_c/` | 경로 C 쥔 관절 기록 `grips.jsonl` |
| `real_runs/` | DG-5F `policy_bag.py` · `joint_recorder.py` 기록 |
| `fake/` | fake 실행 스크립트 로그(`*_fake_run.sh` 등) |
| `calib/` | 보정 겹친 그림 · tiltsweep · 손 힘 측정 csv |
| `finger_meas/`, `joint_monitor/`, `head_stream_probe/` | 측정 도구 런 |
| `place_teach/` | teach 재생 로그(`.log`) |

### 옮긴 것 → `/media/user/DATA/261109/sim2real/logs/`

| 원래 경로 | 무엇 |
|---|---|
| `cup_seg_1107/` | 11.07 컵 분할 분석. `place_cup_gate.py` 근거, `test_pc_place_inputs.py` 일부 시험은 이 폴더가 없으면 건너뛴다 |
| `holder_seg_1107/`, `calib/holder_1107/` | 11.07 홀더 분할 데이터셋 · 원시 프레임 |
| `head_d435i_camera_info_1103/` | 11.03 머리 D435i camera_info · 캡처(arm_v2.yaml 주석 근거) |
| `monitor_1104/`, `perception/*1104.png`, `perception/realsense_board_jog.log` | 11.04~07 모니터 스크립트 · 외부 보정 확인 그림 |
| `console/console_20261108.log`, `rh56f1_hand/right.log` | 11.08 콘솔 · 손 로그 |
| `bags/` | (11.09 이전에 옮김) |

지운 것: `episodes/lanes_fake_1108/`(가짜 실행 잔재), `realsense_try1_imu_denied.log`(1.1G, 권한 오류 반복).
