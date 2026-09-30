# arm4090 — RH56F1 로봇 PC 설정 기록 (2026-09-30 확인)

OpenArm 양팔 + Inspire RH56F1 양손 + 머리(XC330 ×2) + RealSense 가 모두 이 PC 에 붙는다. 점검은
`python3 scripts/setup/check_host.py --robot rh56f1`. 설치 절차는 `INSTALL.md`, 로봇 모듈은
`deploy/s2r_console/robots/openarm_rh56f1.yaml`.

## 장치

| 장치 | 연결 | 이 PC 에서의 이름 | 설정 파일 |
|---|---|---|---|
| 오른팔 | PCAN-USB Pro FD ch1 | `can0` (1M / 5M FD) | 미션 drivers 단계 |
| 왼팔 | PCAN-USB Pro FD ch2 | `can1` (1M / 5M FD) | 미션 drivers 단계 |
| 오른손 RH56F1 | FTDI FT232R BG0327KL · RS485 | `/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG0327KL-if00-port0` | `deploy/policy_control/config/rh56f1_ports.yaml` |
| 왼손 RH56F1 | FTDI FT232R BG033STU · RS485 | `/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG033STU-if00-port0` | 〃 |
| 머리 pan · tilt | U2D2(FT232H FT763P8T) · TTL · 1M | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT763P8T-if00-port0` | `config/head_home_rh56f1.yaml` |
| 카메라 | RealSense D435i serial 348122071637 · fw 5.16.0.1 · USB 3.2 | — | — |

- `ttyUSB0/1/2` 번호는 꽂는 순서로 바뀐다(09.30 같은 날 세 번 바뀜). 손 · 머리 설정은 by-id 경로만 쓴다.
- CAN 을 켜는 것은 sudo 라 운영자가 한다:
  `sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can0 up` (can1 도 같게).

## 확인한 값 (09.30, 읽기만)

- 팔: 두 버스 모두 모터 ID 1~7 응답(Damiao 파라미터 읽기 0x33). TMAX 54/54/28/28/10/10/10. ID 8(그리퍼 자리)은 없음 — RH 손이라 맞다.
- 손: 두 손 모두 Hand_ID 1 · 115200, 오류코드 0, 편 손 약 1750(네 손가락) · 엄지 1349 / 951.
- 머리: pan ID 1 · tilt ID 2 · 1M · 모드 3 · XC330(model 1240, fw 52), 5.2 V.

## 머리 — 5090 과 숫자가 다르다

- **모터끼리 직렬로 이으면 버스 전체가 멎는다.** pan · tilt 를 U2D2(허브)에 따로 꽂는다.
- 모터 EEPROM: Homing Offset **pan +1024 · tilt −1024**. 이 모터는 ±1024 틱(±90°) 밖의 offset 을 무시한다(써져도 안 먹음).
- tilt 혼이 5090 대비 약 184° 돌아 조립돼 있어 같은 화면에서 숫자가 다르다:

  | 같은 화면(5090 홈) | 5090 | arm4090 |
  |---|---|---|
  | pan | 2049 틱 (+0.1°) | 2015 틱 (−2.9°) |
  | tilt | 1820 틱 (−20.0°) | 2865 틱 (+71.8°) |

  맞춘 품질: 숙임 18.93°(기준 18.9) · 테이블 거리 0.61 m(0.614) · 볼트 무리 5~10 px · 회전 약 0.7°.
  도구 `scripts/calib/head_view_align.py`(읽기) · `head_view_autoalign.py`(--execute 로만 움직임) · 기준 `config/head_view_ref_5090.npz`.
- 카메라 외부 파라미터(`config/head_extrinsics.yaml`)는 5090 값이다. FP++ 를 이 PC 에 올리면 영점을 이 표로 바꿔 따로 둔다.
- 혼을 다시 조립하면 offset · `head_home_rh56f1.yaml` · 위 표를 다시 맞춘다.

## 소프트웨어

- 저장소 네 개(sim2real main · robot_control humble · hdgp main · urdf main) — hdgp 는 비공개라 ssh 주소로 받는다.
- robot_control 23 패키지 · policy_control symlink(`PYTHONNOUSERSITE=1` 로 빌드) · `.venv`(torch 2.7.1+cu128 등) · 정책 가중치 md5 = 5090.
- `~/rl_ws/dynamixel-tools`(머리 `scripts/calib/dxl.sh` 가 쓴다).
- 09.30: `check_host.py --robot rh56f1` MISS 0, `pytest -m "not gpu"` 1906 passed · 실패 0.
