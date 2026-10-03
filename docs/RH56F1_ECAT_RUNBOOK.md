# RH56F1 EtherCAT 실기 점검 순서 (arm4090) — 2026-10-03 준비

손 드라이버는 EtherCAT(`config/rh56f1_ports.yaml` transport ethercat, 손 하나 = NIC 하나 · 오른손 `enx00e04c6806e1` · 왼손 `enp6s0`).
구조 · 안전 규칙은 `docs/RH56F1_HAND.md` §7.

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
