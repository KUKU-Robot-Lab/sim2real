# sim2real — 실기 운영 플랫폼

학습한 RL 정책을 **OpenArm 양팔 + Tesollo DG-5F 양손** 실기에 올리고 운영한다.
평소 운영은 **운영 콘솔(`s2r_console`) 한 화면**에서 한다 — 드라이버, 비전, 목(head), 관절 상태,
미션 단계, 정지까지. 명령어를 직접 치는 것은 sudo 가 필요한 준비 단계뿐이다.

## 시작

```bash
cd ~/rl_ws/sim2real
deploy/s2r_console/tools/console.sh --profile dg5f_m_real --operator <이름>
```

브라우저에서 `http://127.0.0.1:8091`. 다른 PC 에서 볼 때는 터널로만 연다(콘솔은 127.0.0.1 에만 붙고 인증이 없다):

```bash
ssh -L 8091:127.0.0.1:8091 <이 PC>
```

| 프로파일 | 무엇 | 도메인 |
|---|---|---|
| `dg5f_m_real` | **지금 로봇** — 양팔 DG-5F-M short, 제어 점검(pd → fabric direct), 한 팔씩 | 126 (실기) |
| `pour_i18_fake` | 양팔 물붓기 정책을 가짜 플랜트로 — 하드웨어 없이 전 과정 리허설 | 97 |
| `left_v2B25_real` | 옛 왼손 그리퍼 구성(2026-09-14 손 교체 전) | 126 (실기) |

`--profile` 없이 띄우면 화면에서 고른다. 화면만 볼 때(ROS 가 없는 PC)는 `--no-bridge`.

## 콘솔이 관리하는 것

| 대상 | 콘솔에서 | 어떻게 |
|---|---|---|
| 드라이버 — 팔 브링업(robot_control), DG-5F 우·좌 | 상자의 스위치로 **감독**하고 끈다 | 켜기는 `bringup` 단계를 승인·실행할 때. 끄기는 스위치(pd 가 팔을 잡은 동안·단계가 도는 동안은 거부) |
| 비전 — 카메라 · FP++ (vision-3090) | `sensors` 단계로 켜고 `sensors_off` 로 끈다 | 인지 런처가 저 PC 의 카메라·컨테이너 상태를 1 Hz 로 보고 → 상자에 `vision-3090` 표시 |
| 목(head) | 목 상태 퍼블리셔 스위치, 기준자세는 `bringup` 의 head_home | `/head/joint_states` 수신 주기가 상자에 나온다 |
| 관절 상태 | 팔(`/joint_states`) · 손(`/dg5f_*/joint_states`) · head 상자 | 수신 주기와 끊김(stale) 을 색·낱말로 |
| 정책 체인 · pd | pd · fabric · episode 상자의 스위치 + 미션 단계 | pd 가 팔을 잡은 동안은 끄기 거부 |
| 연결 | 노드와 토픽이 왼쪽 → 오른쪽 그림으로 | 정책·미션을 바꾸면 그림이 저절로 따라 바뀐다. 그림에 없는 토픽은 아래 "그림 밖 연결" 표 |
| 정지 | 화면 아래 정지 바 — 에피소드 정지 · 에피소드 중단 · PD 해제 | 조작 권한 없이도 누른다. **비상정지는 물리 버튼**이다 |

**아직 콘솔에 없는 것**

- **Isaac Sim 렌더링** — 브리지 패키지는 `robot/isaacsim_bridge/` 에 있지만 콘솔이 띄우지 않는다.
  렌더링에 쓸 장면(USD)과 기동 명령이 정해지면 미션 단위로 붙인다.
- **관절 값 자체** — 지금은 수신 주기·끊김만 보인다. 각 관절의 값·한계 여유는 표로 나오지 않는다.

## 실기 세션 순서 (`dg5f_m_real`)

미션은 `config/mission_dg5f_m_control.yaml`. 콘솔의 미션 패널이 이 순서를 그대로 보여 주고, 막힌 단계는
**왜 막혔는지**를 적는다.

1. **preflight** — 테스트 · 자산 계약 재생성과 게인 대조 · forward 컨트롤러 선언 확인. 읽기 전용이다
2. **sensors** — 인지 런처 → 목 상태 퍼블리셔 → 카메라 + FP++ 켜기. 실기를 움직이지 않는다
3. **bringup** (실기) — 콘솔이 순서대로 진행하고, `수동` 단계에서는 운영자의 확인을 기다린다
   1. 모터 전원(양팔) ON — 물리 스위치, 켠 뒤 확인
   2. CAN can0 · can1 — **sudo, 운영자 셸에서** (콘솔은 sudo 를 실행하지 않는다)
   3. 팔 브링업 — 콘솔이 띄운다
   4. 손 네트워크(Modbus TCP, NIC 둘) — **sudo, 운영자 셸에서**
   5. DG-5F 드라이버 우 · 좌 — 콘솔이 띄운다
   6. head_home — 목 기준자세 + I게인
4. **한 팔씩** — 오른팔: `pd_load_right` → `pd_selftest_right` → `goto_home_right` → `preset_right` →
   `preset_return_right` → `fabric_direct_right` → `release_right`.
   왼팔: `pd_load_left` → `pd_selftest_left` → `goto_home_left` → `fabric_direct_left` → `release_left`
   (왼팔에는 preset 단계가 없다). 두 팔은 bringup 뒤 서로 독립이라 어느 쪽을 먼저 해도 된다
5. 끝낼 때 — 정지 바의 **PD 해제**, 그다음 화면 오른쪽 아래 **run 끝내기**

## 안전 규약 — 코드로 잠겨 있다

| 규약 | 어디서 |
|---|---|
| `--execute` / `execute:=true` 없이는 아무것도 발행하지 않는다 | pd 는 발행자를 만들지조차 않는다(`pd_backends._GuardedPublisher`) |
| 실기 단계는 **단계마다 승인**. 실기 단계의 명령은 스위치로 켜지 않는다 | 미션 러너 · 콘솔 스위치 규칙 |
| `수동` 명령(sudo · 물리 조작)은 콘솔이 절대 실행하지 않는다 | 러너가 확인을 기다린다 |
| 실기 도메인은 `ROS_DOMAIN_ID=126`, 가짜는 97/99 | 브리지는 env 와 프로파일 도메인이 다르면 뜨기 전에 거부한다 |
| 콘솔은 로봇 명령을 발행하지 않는다 | 브리지는 구독 전용 별도 프로세스, API 프로세스는 rclpy 를 import 하지 않는다 |
| 명령(argv)은 미션 yaml 에서만 온다 | HTTP 로는 이름만 온다 |

## 화면 읽는 법

- **상자** — 노드 하나. 색과 낱말이 같이 나온다: 연결됨 · 보유(값을 들고 쉬는 중) · 꺼짐 · 모름 · 끊김 · 없음 ·
  고장 · 죽음. "모름" 은 정상이 아니다 — 브리지가 보지 못했다는 뜻이다.
- **스위치 아래 잠금 줄** — 지금 왜 못 누르는지, 무엇을 하면 되는지(예: "bringup 단계를 승인·실행하면 켜진다").
- **전선** — 토픽 하나. 받는 노드가 스스로 "못 받는다" 고 하면 토픽이 흘러도 끊긴 것으로 본다.
  pd 가 무발행(`execute:=false`)일 때의 구동 전선은 "꺼짐" 이지 끊김이 아니다.
- **머리말** — 상자가 전부 초록이어도 확인 못한 전선이 있으면 그 이름을 적는다.
- **`vision-3090` 표시** — 그 상자는 인지 PC 에서 돈다. 고치러 갈 곳이 다르다.

## 콘솔 밖에서 하는 일

| 하고 싶은 것 | 문서 |
|---|---|
| 새 정책을 등록하고 계약을 만든다 | [docs/USAGE_DEPLOY.md](docs/USAGE_DEPLOY.md) |
| 새 PC 를 세팅한다 | [INSTALL.md](INSTALL.md) · 진단 `./scripts/setup/setup_check.sh [control\|vision\|policy]` |
| 하드웨어 스택의 원리 · 배선을 손으로 확인 · Isaac Sim 브리지 · 드라이브 튜닝 | [robot/HARDWARE_AND_SIM.md](robot/HARDWARE_AND_SIM.md) |
| Isaac Sim ↔ ROS 2 실행 절차 | [robot/USAGE_ISAACSIM_ROS2.md](robot/USAGE_ISAACSIM_ROS2.md) |
| 양팔 물붓기 등록 · 검증 | [docs/RUNBOOK_pour_bimanual.md](docs/RUNBOOK_pour_bimanual.md) |

## 디렉토리 구성

```
sim2real/
├── README.md · INSTALL.md
├── deploy/                  정책 배포 — 운영의 중심
│   ├── s2r_console/            운영 콘솔(브라우저): 프로세스 스위치 · 연결 그림 · 미션 단계 · 정지
│   ├── policy_control/         배포 체인: 계약 · obs/policy/fabric/pd 노드 · launch · 도구
│   └── policies/               쓸 정책 등록소 (카드 policy.yaml + 가중치 + params)
├── config/                  미션 yaml(mission_*.yaml) · 로봇 프로필 · 물체 레지스트리   ※ 경로 고정
├── scripts/                 ops/(운영) · nodes/(ROS 노드) · vision/(인지 PC) · calib/ · probes/ · setup/
│                            최상위는 라이브러리                                        ※ 경로 고정
├── robot/                   하드웨어 · 시뮬레이터
│   ├── isaacsim_bridge/        Isaac Sim ROS 2 브리지 (렌더링 기준으로 쓴다)
│   ├── integrated_control/     OpenArm + Tesollo 통합 런치
│   ├── urdf/                   xacro · urdf · usd(LFS)
│   └── vendor/                 upstream 의존 패키지(openarm · inspire_ws)
├── tests/                   비GPU 회귀 (pytest tests -q -m "not gpu")
├── docs/                    배포 사용법 · 런북 · 계약 · reference/ · legacy/
├── logs/                    배포 입력 일부(logs/policy/* 계약)와 실험 기록            ※ 경로 고정
└── legacy/                  더는 쓰지 않는 것 — ros_pkgs/(COLCON_IGNORE) · scripts/(참조 0)
```

※ **경로 고정**: `config/` · `logs/` 는 학습 저장소 hdgp 가, `scripts/vision/` 은 인지 PC(vision-3090)의
체크아웃이 경로로 직접 읽는다. 옮기면 저쪽이 깨진다.

