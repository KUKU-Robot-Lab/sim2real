# 배포 사용법 — 학습한 정책을 실기에 올린다

> **평소 운영은 운영 콘솔 한 화면에서 한다** → [README](../README.md) (드라이버 · 비전 · head · 관절 상태 · 미션 · 정지).
> 이 문서는 **콘솔 밖에서 하는 일**(정책 등록 · 계약 만들기)과, 콘솔이 안에서 부르는 명령의 참고표다.

이 문서가 배포 경로의 **시작점**이다. README.md 는 하드웨어 브링업과 Isaac Sim 연동까지만 다룬다.

여기 적힌 명령은 전부 그 도구의 `--help` / argparse 에서 확인한 것이다(2026-09-22).
플래그가 의심스러우면 문서를 믿지 말고 `--help` 를 보라 — 문서가 뒤처지면 그것이 문서의 잘못이다.

## 0. 전제

```bash
cd ~/rl_ws/sim2real
source /opt/ros/humble/setup.bash          # ROS. PYTHONPATH 를 덮어쓰지 말고 앞에 붙일 것
./scripts/setup/setup_check.sh policy      # 이 PC 가 정책 배포에 필요한 것을 갖췄는지
python3 -m pytest tests -q -m "not gpu"    # 실패 0 이어야 한다. GPU 를 쓰는 것은 뺀다
```

**전 구간 안전 규약** — 코드로 잠겨 있다. 문서가 아니라 규약이다.

| 규약 | 어디서 잠기나 |
|---|---|
| `--execute` / `execute:=true` 없이는 **아무것도 발행하지 않는다** | pd_node 는 publisher 를 만들지조차 않는다(`pd_backends._GuardedPublisher`) |
| 로봇을 움직이는 단계는 **단계마다 `--approve <id>`** | `mission_run.py` · `episode_ctl.py`. 승인은 argv 에 있고 프로세스와 함께 사라진다 |
| 실기 도메인은 `ROS_DOMAIN_ID=126` | fake 는 97/99 등. launch 가 0/미설정을 거부한다 |
| 비상정지는 **물리 버튼** | 콘솔에는 estop 버튼이 없다(웹서버→DDS 는 가장 느린 경로다) |

## 1. 정책 등록 — `deploy/policies/`

무엇을 쓸 수 있는지에 답하는 곳은 `deploy/policies/` 하나다(`logs/` 에는 옛 기록이 섞여 있다).

```bash
# 서버에서 계약 생성 입력만 받아온다(가중치·params·trace_meta). GPU 접촉 0, 읽기만 한다
python3 deploy/policy_control/tools/fetch_run.py --run t2r_i18 --checkpoint ep:2500 --list
python3 deploy/policy_control/tools/fetch_run.py --run t2r_i18 --checkpoint ep:2500

# 지금 등록된 것과 그 상태
python3 deploy/policy_control/tools/policies.py --shallow         # sha256 재해시 없이 빠르게
python3 deploy/policy_control/tools/policies.py --write-index      # deploy/policies/INDEX.md 갱신
```

정책 하나 = 디렉터리 하나. 한 팔의 묶음(체크포인트 여럿)이면 카드 `policy.yaml` 의 `checkpoint:` 가
후보 하나를 가리킨다. status 는 넷뿐이다: `candidate` → `verified` → `deployed`, 쓰지 않으면 `hold`.

## 2. 계약 만들기

계약은 정책과 실기 사이의 **유일한 인터페이스 선언**이다(관측 차원·액션 분해·게인·홈 자세).

```bash
# 정책 계약 (학습 런에서)
python3 deploy/policy_control/tools/build_deploy_contract.py \
    --run deploy/policies/<id> --checkpoint deploy/policies/<id>/nn/<선택>.pth \
    --sim-meta deploy/policies/<id>/trace_meta.json --out deploy/policies/<id>/deploy_contract.json

# pd 용 control-only 계약 (자산에서). 정책 1개에 계약 파일 2개인 이유다
python3 deploy/policy_control/tools/build_deploy_contract.py \
    --asset openarm_dg5f-m-short_bi_rl --sides right,left \
    --home pour:deploy/policies/<id>/pour_contract.json --out logs/policy/asset_<id>/deploy_contract.json

# 사람이 읽는 문서로 (생성물이지 원본이 아니다)
python3 deploy/policy_control/tools/contract_doc.py --out docs/CONTRACT_policy_control.md <계약들...>
```

pour 계열은 체크포인트가 실기에 나갈 자격이 있는지 먼저 본다:

```bash
python3 deploy/policy_control/tools/ckpt_gate.py --trace deploy/policies/<id>/trace.npz --json /tmp/gate.json
```

## 3. 하드웨어 없이 리허설 — fake 플랜트

배관을 증명한다(계약 로드·관측 조립·60 Hz 루프·에피소드 서비스·seq 결손). **파지/붓기 성공은 증명하지 않는다** — MockArm 에는 접촉이 없다.

```bash
ROS_DOMAIN_ID=97 deploy/policy_control/tools/pour_fake_run.sh 30 logs/policy_control/pour_fake1
ROS_DOMAIN_ID=99 deploy/policy_control/tools/fake_plant_run.sh 900 logs/policy_control/fake1
MODE=pd SIDE=left ROS_DOMAIN_ID=97 deploy/policy_control/tools/fake_plant_run.sh 0 logs/policy_control/fake_pd_left
SIDE=left ROS_DOMAIN_ID=96 deploy/policy_control/tools/fake_direct_run.sh logs/policy_control/direct_left
```

## 4. 미션 — 순서와 게이트

미션 yaml 이 단계·선행조건·승인·argv 의 **유일한 출처**다. argv 는 여기에만 있다.

```bash
python3 scripts/ops/mission_run.py --mission config/mission_policy_control.yaml --plan
python3 scripts/ops/mission_run.py --mission config/mission_policy_control.yaml --stage pd_load
python3 scripts/ops/mission_run.py --mission config/mission_policy_control.yaml \
    --stage pd_load --execute --approve pd_load
python3 scripts/ops/mission_run.py --resume 20260922_101500 --plan
python3 scripts/ops/mission_run.py --abort  20260922_101500
```

에피소드 한 판(로봇이 움직인다 — 승인 3개가 전부 있어야 시작한다):

```bash
# pd 는 팔마다 따로다(09.23) — --side 로 어느 팔인지 말한다(episode 서비스는 쪽이 없다)
python3 deploy/policy_control/tools/episode_ctl.py --side right --steps 250 --execute \
    --approve pd_engage --approve pd_goto_home --approve ep_start
```

## 5. 운영 콘솔 — `s2r_console`

터미널 네 개를 오가지 않기 위한 화면. 판정을 새로 만들지 않고 미션·계약·상태 토픽의 문자열을 옮긴다.

```bash
deploy/s2r_console/tools/console.sh --profile dg5f_m_real --port 8091 --operator <이름>   # 지금 로봇(DG-5F-M)
ssh -L 8091:127.0.0.1:8091 <이 PC>      # 원격은 터널로만. 127.0.0.1 바인딩이고 인증이 없다
```

- 연결 그림은 **미션 argv + 계약 + robot yaml 에서 자동 생성**된다. 정책을 바꾸면 노드·전선이 따라 바뀐다.
- 프로세스 스위치가 상자에 붙는다. pd 가 팔을 잡고 있거나 단계가 도는 동안에는 끄기가 거부된다(409).
- 화면만 볼 때는 `--no-bridge` (rclpy 없는 PC 에서도 뜬다).

## 6. 인지 — 카메라·FP++ 는 vision-3090 에서 돈다

```bash
# 이 PC: 저쪽을 ssh 로 켜고 끄며 /perception/status 를 1 Hz 로 낸다. 스스로는 아무것도 켜지 않는다
ROS_DOMAIN_ID=126 python3 scripts/nodes/perception_launcher_node.py --host vision-3090

# 물체를 골라 인지 체인을 켠다 (이름은 config/objects.yaml 로 검증·alias 해석)
python3 scripts/ops/perception_ctl.py list
python3 scripts/ops/perception_ctl.py start cup_big_s100 --viewer
python3 scripts/ops/perception_ctl.py status
python3 scripts/ops/perception_ctl.py stop --camera
```

저 PC 에서 직접 봐야 할 때만: `ssh vision-3090 bash ~/rl_ws/sim2real/scripts/vision/status.sh`
(`camera_up.sh` · `fpp_up.sh <이름> <yaml>` · `*_down.sh` 도 같은 곳에 있다. 전부 저 PC 경로다.)

## 7. 판정·기록

```bash
python3 deploy/policy_control/tools/status_to_csv.py --seconds 60 --out /tmp/run.csv --policy-dt 0.02
python3 deploy/policy_control/tools/status_board.py            # 버튼 없는 읽기전용 상태판(127.0.0.1)
python3 deploy/policy_control/tools/episode_judge.py --contract <계약> --seconds 30
```

## 정리

| 하고 싶은 것 | 명령 |
|---|---|
| 지금 쓸 수 있는 정책 | `deploy/policy_control/tools/policies.py --shallow` |
| 서버에서 정책 받기 | `deploy/policy_control/tools/fetch_run.py --run <런> --checkpoint <선택>` |
| 계약 만들기 | `deploy/policy_control/tools/build_deploy_contract.py --run deploy/policies/<id> …` |
| 하드웨어 없이 한 판 | `ROS_DOMAIN_ID=97 deploy/policy_control/tools/pour_fake_run.sh` |
| 미션 판정만 보기 | `scripts/ops/mission_run.py --mission <yaml> --plan` |
| 실기 세션 화면 | `deploy/s2r_console/tools/console.sh --profile <id> --operator <이름>` |
| 인지 켜기 | `scripts/ops/perception_ctl.py start <물체>` |
| 지연·seq 기록 | `deploy/policy_control/tools/status_to_csv.py --seconds 60 --out <csv>` |
