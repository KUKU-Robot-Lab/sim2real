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
| **S 임시** | 디버그, 시운전, 가짜(fake) 실행, 콘솔·노드 로그, pid, 미션 상태 | 러너 기본 경로 | **실험 README 에 등록되지 않은 런은 모두 S 다.** 7일이 지나면 지워도 된다 |

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
  | r01 | `logs/episode/20261109_101500_grasp_f_right` | ok | |
  | r02 | `logs/bags/20261109_102030_grasp_f_right` | 버림 | 손가락 영점 틀어짐 |

  "버림"도 표에 적는다. 왜 버렸는지가 남아야 같은 실수를 반복하지 않는다. 다만 원시 데이터는 정리할 때 지워도 된다.
- 이미 실험 폴더를 갖고 있는 분석(`analysis/friction/` 처럼 repo 안에 있는 것)은 그 구조를 그대로 쓴다. 이 README 와 같은 형식이다.

## 새 코드를 쓸 때

- 출력 기본 경로는 `logs/<도구 종류>/<YYYYMMDD_HHMMSS>_<label>/` 로 한다. 같은 초에 충돌하면 `_k` 를 붙인다(`tilt_pour.py` 의 `fresh_dir`).
- 최상위 폴더 이름에 날짜 접미사(`_1107`)를 붙이지 않는다. 날짜는 하위 폴더 이름에 넣는다.
- 고정 파일명에 덮어쓰지 않는다. R 종류라면 새 이름으로 만든다.
- 경로는 repo 루트 기준 절대 경로로 정한다. cwd 에 따라 상대경로 `logs/...` 가 다른 곳을 가리키면 안 된다.
- 런 폴더에 `meta.json` 을 남긴다: `{"git": HEAD, "dirty": bool, "argv": [...], "host": ..., "start": ...}`.
- 자산(궤적) 버전이 결과를 가르면 이름에 자산을 넣는다. 지금 적용된 곳(11.09):
  - `logs/tilt/<시각>_setting_<자산>/`, `logs/tilt/<시각>_pour_<자산>/` (예: `_setting_base_arc6`, `_pour_traj_v12_240`) + `meta.json`
  - `logs/episode/<시각>_<에피소드>/meta.json` 의 `assets` 에 궤적별 자산 버전(`tilt: tilt/traj_v12_240`)이 남는다. 같은 에피소드 이름이라도 v11·v12 가 여기서 갈린다.
- 노드 로그가 같은 오류를 반복하면 파일이 GB 단위로 커진다. 오래 띄우는 노드는 로그 크기나 줄 수에 상한을 둔다.

## 정리 (주 1회 또는 실험을 마칠 때)

1. 실험이 끝나면 README 결론과 `result/` 를 쓰고, 상태를 `완료` 로 바꾼다.
2. 완료된 실험의 원시 데이터(런 표에 있는 경로)는 `/media/user/DATA/<YYMMDD>/sim2real/logs/<같은 상대경로>` 로 옮긴다. 이 형식은 이미 `DATA/261109/` 에서 쓰고 있다.
   README 런 표의 경로 옆에 `→ DATA/<YYMMDD>` 를 적는다.
3. 어느 실험에도 등록되지 않았고 7일이 지난 S 런은 지운다. 지우기 전에 목록을 사용자에게 보여 준다.
4. R 은 건드리지 않는다.

## 지금 있는 폴더 분류 (11.09 기준, 아직 옮기지 않았다)

| 폴더 | 종류 | 메모 |
|---|---|---|
| `policy/` | R | 계약과 마스크를 코드가 읽는다 |
| `place_teach/*.npz` | R | 재생 `.log` 는 S |
| `policy_control/*.npz`, `home_calib_*.log` | R | `fake_replay_*` 와 `press/` 는 S |
| `calib/` 결과(yaml·csv·`hand_real_1014/`) | R | `calib/bags/`(4.3G) 와 `holder_1107/`(1.3G) 원시 데이터는 S 후보 |
| `episode/`, `mission/`, `inhand/`, `tilt/`, `rh56f1_record/`, `rh56f1_tip_zero/`, `cpu/`, `console/`, `perception/*.log`, `monitor_1104/` | S | 실험 README 에 등록되지 않았다 |
| `episodes/lanes_fake_1108` | S | 쓰는 코드가 없다. 가짜 실행 잔재로 보인다 |
| `cup_seg_1107/`(671M), `holder_seg_1107/`(251M) | E (옛 형식) | 스크립트와 결과가 함께 있다. `exp/261107_cup_seg/` 로 옮길지 정해야 한다 |
| `head_d435i_camera_info_1103/` | R(`camera_info.json`) + S(1.1G 오류 로그) | |
