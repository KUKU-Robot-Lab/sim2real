# RH56F1 정지 마찰 실기 런북

실험 정의는 `docs/experiments/2026-11-07_rh56f1_stationary_tracking.md` 의 "2026-11-08 실기 대기 목록" 절에 있다. 이 문서에는 실행 순서만 적는다. 1108 런북(`logs/analysis/stationary_tracking_1108`, 11.09 폐기)을 새 폴더 구조로 옮겼다.

승인은 시험 하나(= 실험 폴더 하나) 단위로 받는다(11.09 사용자 "하나의 실험은 한번만 승인"). 그 시험의 PD 발행 기동, engage, trial 전부를 한 번에 승인받고 그 안에서는 다시 묻지 않는다. 다음 시험으로 넘어갈 때, 보호 정지 뒤 재개할 때, 계획(게인·자세)을 바꿀 때는 다시 승인받는다.

## 시험 목록 (팔마다 오른팔 먼저, 위에서 아래 순서)

| 시험 | 폴더 이름 예 | new_experiment.sh 인자 | trial |
|---|---|---|---|
| F1 끔 | `<날짜>_F1off_<side>` | `<side> off F1 2.0 3` | 24 |
| F3 | `<날짜>_F3_<side>` | `<side> off F3 3.0 4` | 32 |
| F4 | `<날짜>_F4_<side>` | `<side> off F4 0.4 3` | 24 |
| F1 켬 | `<날짜>_F1on_<side>` | `<side> fs F1 2.0 3` | 24 |
| F2 ×0.5 | `<날짜>_F2x05_<side>` | `<side> fs_i050 F2 2.0 3` | 24 |
| F2 ×1 | `<날짜>_F2x1_<side>` | `<side> fs_i100 F2 2.0 3` | 24 |
| F2 ×2 | `<날짜>_F2x2_<side>` | `<side> fs_i200 F2 2.0 3` | 24 |

F2 integral_rate j1~j4 = 0.1 / 0.2 / 0.4 N·m/s, 상한 1.0 N·m. F1 끔·F3·F4 는 같은 `_off` 설정이라 PD 재기동 없이 이어 할 수 있다. 계획의 `pd_config` 와 실행 중인 PD 의 보상 설정이 다르면 `run` 이 거부한다.

## 1. 준비 (오프라인, 명령 0개)

```bash
cd ~/rl_ws/sim2real
analysis/friction/common/scripts/new_experiment.sh 261109_F1off_right right off F1 2.0 3
```

폴더, README, 설정 사본, git 상태, 계획, trial 목록이 생긴다. 이름이 이미 있으면 거부한다. 다시 할 때는 `_b` 를 붙인다. README 의 목적·판정 기준을 채운다.

## 2. 실행 전 확인

1. 상대 팔이 계획 장면과 같은 자세인가. 오른팔은 `common/inputs/scene_right_fit01_head_from_attention.json`(왼팔 홈), 왼팔은 `common/inputs/right_pose_set_v1/scene_complete_attention_left_home.json`(오른팔 attention) 기준이다. 다르면 장면을 다시 잡아 `common/inputs` 에 새 이름으로 두고 새 폴더로 준비한다.
2. 대상 팔이 계획의 시작(홈) 자세인가. 다르면 `run` 이 거부한다.
3. 손 자세가 계획의 고정 손 자세(차렷 손)와 같은가.
4. 경로 주변이 비어 있는가. 받침·케이블·발열·CAN 상태는 운영자가 본다.
5. `ROS_DOMAIN_ID=50`.
6. 무발행 확인: 아래 trial 명령에서 `--execute` 를 빼고 돌려 "Dry run: 0 commands" 를 본다.
7. README 에 승인 시각을 적는다.

## 3. 실행 (시험 승인 한 번에 포함)

PD 발행 기동(기존 PD 는 운영자가 미션 절차로 정지한 뒤):

```bash
ROS_DOMAIN_ID=50 ros2 launch deploy/policy_control/launch/pd_controller.launch.py \
  contract:=logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json \
  robot:=deploy/policy_control/config/robots/rh56f1_<side>_real.yaml \
  pd_config:=deploy/policy_control/config/stationary_1108/pd_rh56f1_stationary_<side>_<cfg>.yaml \
  sides:=<side> execute:=true stage:=full
```

engage(IDLE → TRACKING, 제자리 10 s):

```bash
ROS_DOMAIN_ID=50 python3 deploy/policy_control/tools/episode_ctl.py --side <side> --only pd_engage --hold-s 10 --execute --approve pd_engage
```

trial 전부(`setup/trials.txt` 순서, 하나가 멈추면 거기서 끝):

```bash
E=analysis/friction/261109_F1off_right
for t in $(cat $E/setup/trials.txt); do
  ROS_DOMAIN_ID=50 .venv/bin/python deploy/policy_control/tools/stationary_tracking.py run \
    --plan $E/setup/plan/plan.json --trial $t --out $E/runs/$t --execute || break
done
```

## 4. 중단과 복구

1. 과열·추종·토크 보호, 접촉, stale, 경로·게인·보상 설정 불일치, 시작 자세 불일치가 나면 그 trial 에서 멈추고 다음 trial 로 넘어가지 않는다. 원시 기록, 마지막 명령·실측, 안전 사유는 `runs/<trial>/` 에 남는다. README 진행 기록에 적는다.
2. 복구 순서: 원인 확인 → 시작 자세·손 자세 확인 → 새 승인. 이어서 하면 남은 trial 만, 같은 `runs/<trial>` 은 다시 쓰지 않는다(`<trial>_b`).
3. 블록 종료: 운영자가 기존 경로로 홈에 둔다. 다음 승인된 bringup 에서 운영 설정(`pd_rh56f1.yaml` 계열, 보상 끔)으로 돌아가고 상태·게인을 다시 확인한다.

## 5. 정리

`result/` 에 trial별 요약 csv 와 그림을 두고 README 결과 한 줄과 `analysis/friction/README.md` 진행 표를 고친다.
