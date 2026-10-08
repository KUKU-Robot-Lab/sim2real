# RH56F1 EtherCAT 실기 점검 순서 (arm4090) — 2026-10-03 준비

손 드라이버는 EtherCAT(`config/rh56f1_ports.yaml` transport ethercat, 손 하나 = NIC 하나 · 오른손 `enx00e04c6806e1` · 왼손 `enp6s0`).
구조 · 안전 규칙은 `docs/RH56F1_HAND.md` §7.

## ★10.03 결과 — EtherCAT 으로 두 손 손가락 제어 성공

- OP 가 안 되던 원인: **1 kHz 주기**. 손 MCU(LAN9252 · SSC)가 프로세스 데이터에 밀려 OP 요청을 읽지 못했다
  (ESC AL event 0x0220 bit0 · SM2 '쓰임 · 안 읽힘' 이 그대로). 500 · 250 · 100 Hz 는 바로 OP → 기본 `cycle_hz: 500`(dba78b2).
- 두 손 OP 30 s 안정(WKC 오류 0 · 왕복 < 0.07 ms). speed_set 500 → `rh56f1_axis_probe.py` 검지 하나:
  오른손 1743 → 1658 → 1742(굽힘 · 되돌림), 왼손 896 → 978 → 897(폄 · 되돌림), 다른 손가락 그대로. ENABLE_SET 1 = 동작 허용 확인.
- 이어서 두 손 동시에 나머지 5 축(중지 · 약지 · 새끼 ±0.15 rad · 엄지 굽힘 ±0.1 · 엄지 회전 +0.15)을 한 축씩: 모두 그 슬롯만 움직이고
  목표 ±3 안 · 되돌림 · 오류 0 · 온도 36~42 °C. 사용자 눈 확인 '움직임 모두 확인됨' — 두 손 6 축 전부 EtherCAT 으로 확인.
- 명령 뒤 상태 코드는 6 축 모두 1(쥐는 중). 목표가 매뉴얼 범위로 잘려 실제 각도와 3~4 차이(오른손 1743 vs 1740 · 왼손 896 vs 900)라
  계속 따라가는 중으로 보인다 — 온도 36~40 °C 로 문제는 없었지만 범위 끝에서는 지켜볼 것.

## ★10.03 양팔 홈 ↔ 차렷 · CPU 최적화 (2b644f0 · 88f449e · 319888a · robot_control 3c909cc)

- 홈 ↔ 차렷: 오른팔 · 왼팔 모두 성공(홈 오차 0.0099 · 0.0097 rad). 왼팔은 차렷 j4 −0.013 rad 가 하한 0 밖이라 engage 시작점과
  경로 진입 램프의 첫 목표가 '한계 밖'으로 HOLD → 목표는 늘 한계로 자르고 고장 판정은 0.05 rad 넘을 때만.
- CPU(같은 부하, fake 도메인): pd 프로세스 66.5 % → 17.3 %.
  ① robot_control 브링업에 pd 전용 사본 — /pd_state_broadcaster/joint_states 250 Hz · /pd_temp_broadcaster/dynamic_joint_states 10 Hz
     (/joint_states 750 Hz 는 기록용 그대로). rh56f1_*_real.yaml 이 사본을 읽는다 — **브링업은 robot_control 3c909cc 이후여야 한다**.
  ② pd 는 상태를 raw 로 받아 틱에서 마지막 것만 푼다. ③ lean_node — 기본 QoS 이벤트 · 파라미터 서비스 끔(executor wait set 비용이 66 %).
- 기록: 팔 · 손 기록기 분리(rh56f1_record.sh), 기록 중 bag 은 읽지 않는다(bag_rate_report.py 는 metadata.yaml 없으면 거부).
- 미확인: 실기 133 s 의 동시 멈춤(손 336 · pd 149 · /joint_states 88 ms). fake 2 분 부하에서는 재현 안 됨(pd 최대 간격 47 ms).
- 배경 실행한 ros2 launch 는 SIGINT 를 무시한다 → 노드 PID 에 SIGTERM.

## ★10.03 CPU 자동 배치 · 실시간 한도 (PC 가 바뀌어도 같은 절차)

- 실측(arm4090): 손 EtherCAT 마스터 · controller_manager 둘 다 `SCHED_FIFO … Operation not permitted` — 보통 우선순위로 돌았다.
  RTPRIO 한도 0. GNOME 터미널 아래 프로세스는 systemd 사용자 관리자(user@1000.service)의 한도를 받아서 limits.d 만으로는 안 바뀐다.
- 한 번(운영자): `sudo bash scripts/setup/rt_setup.sh` → 재부팅. limits.d · user@<uid>.service drop-in · ~/.config/systemd/user.conf
  세 곳에 rtprio 98 · memlock unlimited. 확인 `bash scripts/setup/rt_setup.sh --check`, 되돌리기 `--undo`.
- 매번(자동): 실기 미션 preflight 첫 명령 `check_host.py --robot rh56f1 --only cpu` — 한도가 80 미만이면 MISS 로 멈춘다(fake 미션에는 없다).
- 코어(자동): `policy_control/cpu_plan.py` 가 sysfs(물리 코어 · SMT 형제 · isolated)를 읽어 정한다 — 번호를 박지 않는다.
  EtherCAT 마스터 오른/왼은 각자 물리 코어 하나(isolcpus 가 있으면 그것, 없으면 큰 번호부터, cpu0 코어는 안 씀), 형제 스레드까지 비켜 두고
  pd · 정책 · 손 상태 · EtherCAT 노드는 그 밖(일반 코어)에서 돈다. 물리 코어가 4 개 미만이면 고정하지 않는다. arm4090 = cpu15 · cpu14.
  torch 스레드 = POLICY_CPU_THREADS 또는 1(10.04: 2 개는 CPU 2 배 · 0.5 ms 빠름). 끄기 `S2R_CPU_PIN=0`. 노드 로그 첫머리에 'CPU: …' 한 줄.

## ★10.03 밤 실기 측정 — 정책 노드 2개(대기) + FP++ (무발행, 로봇 정지)

- ★정정(10.04): 기록 · CPU 측정 동안 rh_aglt 두 노드는 내내 stopped 였다 — `max_episode_s -1` 은 '제한 없음'이 아니라
  계약의 학습 에피소드 길이(15 s)라 시작 15 s 뒤 끝났다. 아래 '정책 ×2 0.54' 는 대기 틱(60 Hz 측정 · 상태)의 값이고
  추론 부하는 아직 안 쟀다. joint_target 이 bag 에 0 개인 것도 이 때문이다. 다음 측정은 max_episode_s 를 크게(예 600) 준다.

- 순서: CAN 무해 프레임(0x5A5) ACK 확인 → 손 → 팔 브링업(/joint_states 0 이 아닌지 즉시 확인) → pd 무발행 ×2 → rh_aglt 양팔
  (CPU 추론, 컵 대신 FP++ 홀더 자세, reset_tol 4.0) → bag 기록 + 프로세스별 CPU → 역순 정리.
- 실시간: 손 마스터 FIFO 80(cpu15 · cpu14), controller_manager RT 스레드 FIFO 50. FP++ 컨테이너는 일반 코어(0-13,16-29).
- CPU(코어 수, 170 s 평균): FP++ 3.13 · pd ×2 0.62 · 정책 ×2 0.54 · 기록 ×2 0.40 · 손 상태 ×2 0.35 · controller_manager 0.30 ·
  카메라 0.25 · 손 EtherCAT 노드 ×2 0.21 · 마스터 ×2 0.02 → 합계 약 5.9 / 32. GPU 65 % · FP++ 2.1 GB.
- 멈춤(최대 간격): /joint_states 4.6 ms · pd 상태 12.5 ms · 정책 상태(60 Hz) 20.6 ms · 손 9.5 ms. 오후의 동시 멈춤(손 336 · pd 149 ·
  팔 88 ms)은 204 · 363 s 기록 모두에서 다시 안 나왔다. 기록 시작 0.6 s 에 손 토픽 333 ms 늦게 옴 1 회 — 샘플 시각은 4 ms 간격
  그대로라 기록기 시작 때 전달 지연(제어와 무관).
- 남은 것: pd 가 실기 30 %(fake 17 %) · FP++ 깊이 광선 방향 13 mm 짧음(배포는 z −8 mm 만 보정) · pd 발행 상태 측정.
- 보고서: ~/rl_ws/report/rh56f1_final_optimization_eli5.html. bag: logs/bags/20261003_231359_rt_cpu_check · 20261003_234451_policy_fpp_check(arm4090).

## ★10.08 실기 런 자동 기록 (CPU · 팔 지연)

- 실기 미션 단독 aglt(`policy_aglt_<side>`) · 에피소드 단계가 정책보다 먼저 bag(rh56f1_record.sh, pd applied · 손 명령 angle_set/angle_target 포함)과
  프로세스별 CPU(`tools/proc_cpu_record.py` → logs/cpu/<단계>.csv, 끝날 때 요약)를 띄운다. 10.03 은 정책이 15 s 에 멈춰 추론 부하를 못 쟀다 —
  이번에는 정책이 실제로 도는 동안 남는다. 요약만 다시: `python3 deploy/policy_control/tools/proc_cpu_record.py --summary logs/cpu/<이름>.csv`.
- 팔 명령 → 움직임 지연: `python3 deploy/policy_control/tools/arm_latency_report.py logs/bags/<런>/arm --side right [--json …]`
  (목표 → pd applied 집어감 · applied → 실측 관절별 p10/50/90). 10.03 home_return bag 으로 10.07 Grasping 회신 값을 재현한다(test_pc_run_tools).
  10.04 pd 폴링(cee5f88) 이후 실측은 첫 실기 런 bag 으로 다시 잰다.
- 단계가 도중에 실패해 기록이 남아도 다음 start 가 먼저 마무리하고, shutdown 이 남은 기록 · CPU 기록기를 내린다.

## ★10.03 고속 맞춤 (3ca354e · 4c1bfae)

| 구간 | 값 | 근거 |
|---|---|---|
| EtherCAT PDO | 500 Hz | 이 손의 상한. 1 kHz 는 OP 전이가 안 되고, OP 뒤 750 · 1000 Hz 로 바꾸면 OP 는 유지되지만 명령을 무시(검지 그대로). 500 초과 설정은 노드가 거부 |
| 손 상태 → ROS | 250 Hz | 500 을 나눠떨어지게(200 은 166.7 Hz 로 나왔다) · hand_check 기준 = 80 % |
| pd(팔 · 손) | 120 Hz | sim PD 120 Hz 와 같게 — 정책 60 Hz 한 스텝 = 2 틱 |
| pd → 손 명령 상한 | 120 Hz | RS485 의 30 Hz 상한을 풀었다 |
| 팔 컨트롤러 | 750 Hz | openarm_bimanual_controllers.yaml(바꾸지 않음) |
- setcap: `/etc/sudoers.d/rh56f1-setcap`(tools/ethercat/sudoers-rh56f1-setcap) 설치됨 — build.sh 가 다시 빌드한 뒤 비밀번호 없이 붙인다.

## 10.02 밤 상태

| 항목 | 상태 |
|---|---|
| SAFE_OP(상태만) | 두 손 1 kHz · 100 Hz 발행 · WKC 오류 0 · hand_check 통과 |
| OP(명령을 받는 상태) | ★안 됨 — OP 요청 뒤 AL 0x04 · code 0x0000 으로 SAFE_OP 에 머문다(거부가 아니라 대기). 원인 미확인 |
| 마스터 빌드 | 2768721(OP 진단 추가) — **setcap 이 아직 없다** |
| 손 전원 · 링크 | 꺼짐(NO-CARRIER) |
| USB-RS485(비상용) | 빠져 있음 |
| 벤더 자료 | 영문 매뉴얼 V1.2 §2.6 · 중문 매뉴얼 V1.0.0(FINGER_MODE · 일시정지 · 비상정지 SDO 추가) — EtherCAT 시작 순서 · ESI XML · 예제는 없다 |

## 0. 준비 (운영자)

```bash
# 손 전원 ON · 랜 케이블 확인 후
ip -br link show enx00e04c6806e1; ip -br link show enp6s0          # 둘 다 UP 이어야 한다
getcap ~/rl_ws/sim2real/tools/ethercat/rh56f1_ecat_master          # 비면 ↓
sudo setcap cap_net_raw,cap_net_admin=ep ~/rl_ws/sim2real/tools/ethercat/rh56f1_ecat_master
```

## 1. 통신 (손 무동작)

```bash
cd ~/rl_ws/sim2real
tools/ethercat/ecat_rtt enx00e04c6806e1 2000; tools/ethercat/ecat_rtt enp6s0 2000      # 잃음 0 · slave 1
tools/ethercat/ecat_rh56f1 safeop enx00e04c6806e1 3 1000                               # 잃음 0 · 각도 · 온도
```

## 2. 드라이버 무동작 (SAFE_OP)

```bash
bash deploy/policy_control/tools/rh56f1_hand_up.sh right --no-op
python3 deploy/policy_control/tools/rh56f1_hand_check.py --side right        # angle_actual 100 Hz · ✓ 통과
bash deploy/policy_control/tools/rh56f1_hand_down.sh right
```
- 띄우고 내리는 것은 반드시 `rh56f1_hand_up.sh` / `rh56f1_hand_down.sh` 로(PID 파일에 실제 노드 PID). 10.02 에 setsid 의
  fork PID 를 기록해 다른 프로세스에 신호를 보내고, 같은 NIC 에 마스터가 둘 붙었다.

## 3. OP (★실기 — 승인 뒤. 첫 명령 전에는 목표 = 지금 각도라 손은 제자리)

```bash
bash deploy/policy_control/tools/rh56f1_hand_up.sh right
tail -f /tmp/rh56f1_hand/right.log      # "[master] OP" 가 나오면 성공
```
- 3 s 안에 OP 가 안 되면 `[master] ESC 레지스터: …` 줄이 찍힌다(DL · AL · AL event · SM2/SM3 · 워치독 · DC). 그 줄을 보고 판단.
- 안 되면 실험을 하나씩(각각 내린 뒤 다시 띄운다):
  - A `--op-enable` — 명령 전에도 ENABLE_SET 1(목표 = 지금 각도)
  - B `--sync-type 1` — 0x1C32/33:01 에 SM 동기(1)를 쓴 뒤 시작(되돌리려면 `--sync-type 0`)
- 각 시도 뒤 `rh56f1_hand_down.sh right`.

## 4. 손가락 하나 (★실기 — 승인 뒤, OP 가 된 뒤에만)

```bash
source /opt/ros/humble/setup.bash; source ~/rl_ws/robot_control/ros_ws/install/setup.bash; export ROS_DOMAIN_ID=126
python3 deploy/policy_control/tools/rh56f1_axis_probe.py --side right --axis index_1 --by 0.15 --back            # 계획만
python3 deploy/policy_control/tools/rh56f1_axis_probe.py --side right --axis index_1 --by 0.15 --back --execute  # 실기
```
- 같은 `/hand_<side>/angle_set` 을 쓰므로 EtherCAT · RS485 어느 쪽이든 같은 명령이다. 한 축만, 다른 다섯 축은 -1.
- 왼손은 주먹 자세면 `--by -0.15`(펴는 쪽)로.

## 5. 비상용 — RS485 (OP 가 안 풀려 오늘 손을 꼭 움직여야 할 때)

1. 손 RS485 포트에 USB-RS485(FTDI BG0327KL = 오른손 · BG033STU = 왼손)를 다시 꽂는다 → `ls -l /dev/serial/by-id | grep FT232R`
2. `bash deploy/policy_control/tools/rh56f1_hand_up.sh right --transport rs485` (설정 파일 `rs485:` 블록 · 벤더 드라이버)
3. 4 와 같은 `rh56f1_axis_probe.py` 명령(09.30 에 이 경로로 양손 축 확인을 마쳤다)
- 미션 기본값은 EtherCAT 그대로다. RS485 는 `--transport rs485` 를 준 그 실행에만 쓴다.

## 6. 끝낼 때

```bash
bash deploy/policy_control/tools/rh56f1_hand_down.sh right; bash deploy/policy_control/tools/rh56f1_hand_down.sh left
```
