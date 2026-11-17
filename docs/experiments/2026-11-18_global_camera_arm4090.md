# arm4090 글로벌 카메라(두 번째 D435i) 첫 확인 절차 (2026-11-18 준비, 실행은 다음 세션)

arm4090 에 D435i 가 두 대 붙었다. 이 문서는 다음 세션에서 사용자와 함께 위에서부터 차례로 돌린다.
목·팔·손은 움직이지 않는다. 목을 옮겨야 하는 단계가 나오면 멈추고 사용자 허락을 받는다.

## 장치

| 역할 | ns · 토픽 | rs_launch `serial_no`(Serial Number) | lsusb · /sys serial(Asic) | fw | USB |
|---|---|---|---|---|---|
| 머리 | `/camera` · `/camera/camera/*`(IMU 켬) | 348122071637 | 254343066627 | 5.16.0.1 | 4-1.1(허브 경유) |
| 글로벌 | `/global_camera` · `/global_camera/camera/*`(IMU 끔, TF 안 냄) | 261222075918 | 261143062487 | 5.15.1.55 | 4-2(직결) |

- `serial_no` 에는 rs-enumerate-devices 의 `Serial Number` 를 준다. lsusb 와 `/sys` 에 보이는 값은 `Asic Serial Number` 라서 이 값을 주면 장치를 못 찾는다(11.18 arm4090 `rs-enumerate-devices -s` 로 짝 확인).
- 시리얼 정본은 `deploy/policy_control/config/sites/arm_v2.yaml` 의 `rs_cameras` 다. `scripts/vision/camera_up.sh <head|global>` 가 `policy_control/site_ports.py camera <역할>` 로 읽는다. 다른 설치 장소에서는 `S2R_CAMERA_SITE` 로 고른다. 시리얼을 못 읽으면 카메라를 띄우지 않는다.
- 11.18 기준으로 머리 노드는 `serial_no` 없이 떠 있다(PID 2091047, `-r __ns:=/camera`). 이 노드는 그대로 둔다. 글로벌을 시리얼로 띄우면 남은 한 대를 잡으므로 머리와 다투지 않는다.

## 0. 준비 (arm4090)

```bash
ssh arm4090
cd ~/rl_ws/sim2real && git pull --ff-only && git log --oneline -1    # laptop HEAD 와 같은지 본다
source scripts/vision/common.sh                                      # 도메인 · LAN 프로필. 안 하면 토픽이 안 보인다
nvidia-smi --query-compute-apps=pid,used_memory --format=csv          # FP++ 컨테이너가 쓰는 GPU 확인(읽기만)
```

## 1. 시리얼 매칭

```bash
rs-enumerate-devices -s                       # Serial Number · Asic · fw · USB 3.2 두 줄
lsusb -t | grep -B1 -A1 uvcvideo              # 두 대 모두 5000M 인지 본다
python3 deploy/policy_control/policy_control/site_ports.py camera head      # → 348122071637
python3 deploy/policy_control/policy_control/site_ports.py camera global    # → 261222075918
grep -a "Device Serial No\|serial number" $(ls -t logs/runs/*/perception/realsense.log | head -1) | tail -2   # 지금 머리 노드가 잡은 장치
```

통과 조건: 머리 노드 로그의 시리얼이 348122071637 이어야 한다. 261222075918 이 찍혀 있으면 머리 노드가 글로벌 카메라를 잡고 있다는 뜻이다. 그때는 2번으로 가지 말고 사용자와 정한다(머리 노드를 내렸다가 `camera_up.sh head` 로 다시 띄우면 FP++ 가 끊긴다).

## 2. 글로벌 카메라 기동

```bash
bash scripts/vision/camera_up.sh global       # → "camera(global) up · serial 261222075918"
ros2 node list | grep camera                  # /camera/camera · /global_camera/camera
ros2 param get /global_camera/camera serial_no
ros2 param get /global_camera/camera publish_tf    # False — 두 노드가 같은 camera_link TF 를 내면 머리 TF 가 덮인다
```

로그는 `logs/runs/<YYMMDD>/perception/realsense_global.log` 에 쌓인다. 끌 때는 `bash scripts/vision/camera_down.sh global` 을 쓴다. 이 스크립트는 `__ns:=/global_camera` 노드만 내린다.

## 3. 두 카메라 동시 스트림 대역폭

```bash
for t in /camera/camera/color/image_raw /camera/camera/aligned_depth_to_color/image_raw /camera/camera/imu \
         /global_camera/camera/color/image_raw /global_camera/camera/aligned_depth_to_color/image_raw; do
  echo "== $t"; timeout 12 ros2 topic hz "$t" 2>/dev/null | grep average | tail -1; done
grep -ci "frames didn't arrive\|overflow\|timeout" logs/runs/$(date +%y%m%d)/perception/realsense*.log
sudo dmesg | tail -50 | grep -i "usb\|uvc"    # 사용자가 실행(읽기)
```

통과 조건은 다음과 같다. 컬러와 정렬 깊이는 두 카메라 모두 설정 fps(rs_launch 기본 30)의 95 % 이상이어야 한다. 머리 IMU 는 글로벌을 띄우기 전과 같아야 한다. FP++ 는 `ros2 topic hz /objects/<컵>/live_pose` 가 약 10 Hz 로 유지돼야 한다(11.08 정상값). 머리는 허브를 거치는 4-1.1 이고 글로벌은 직결 4-2 다. 둘 다 같은 4번 버스(xHCI 하나)를 쓰므로 프레임이 빠지면 글로벌을 다른 컨트롤러 포트로 옮기는 안을 사용자에게 올린다.

## 4. 마커 검출 (쓰기 없음)

```bash
ros2 topic echo --once /head/joint_states | grep -A2 position   # 머리가 camset(m1117 head_pose 1825/2833 틱)인지 본다
python3 scripts/calib/global_cam_extrinsics.py                  # --out 이 없으면 출력만 한다
```

출력에서 볼 것은 세 가지다. 첫째, 두 카메라에서 각각 마커 `[70, 71, 73]` 이 보여야 한다. 글로벌은 70 · 71 둘이 꼭 있어야 하고 73 은 있으면 좋다. 둘째, 코너 σ 는 0.5 px 이하여야 한다. 셋째, 머리 재풀이 결과가 m1117 대비 3 mm · 0.3° 안이어야 한다. 머리 재풀이가 이 범위를 벗어나면 머리가 camset 에 있지 않은 것이다. 이때 목을 옮기는 일은 사용자 허락을 받은 뒤 `head_view_feedback.py --execute` 로만 한다.

## 5. 외부 파라미터 풀기

```bash
python3 scripts/calib/global_cam_extrinsics.py --out config/global_camera_extrinsics_arm4090_global_$(date +%m%d).yaml
```

이 도구는 머리 m1117 과 같은 풀이를 쓴다(`head_view_feedback.solve_extrinsics`, 마커 정답은 `config/head_view_feedback_arm4090.yaml` 의 `anchor_truth`). 영상 토픽만 `/global_camera` 로 바뀐다. 초기값은 머리 영상에서 정한 코너 모서리 부호로 70 · 71 의 코너 8 개를 PnP 로 풀어 얻는다. 파일이 이미 있으면 쓰지 않는다. 결과 파일의 `provenance` 에 rms, 73 의 y, 비교 마커 차가 남는다.

통과 조건은 글로벌 재투영 rms 2.0 px 이하다(머리 m1117 은 0.73 px). 마커 73 의 자유 y 는 머리와 글로벌이 몇 mm 안에서 같아야 한다.

## 6. 같은 물체의 base 좌표 비교 (머리 FK 경로와 글로벌)

(a) 테이블 마커 72 를 쓴다. 72 는 들떠 있어 정답에서 뺐지만 두 카메라가 같은 물체를 보므로 비교에는 쓸 수 있다. 5번 출력의 `비교 마커 72: 글로벌−머리 [dx, dy, dz] mm` 를 본다.

(b) 컵으로 비교한다. 6x6_250 ArUco 하나(예: id 74, 한 변을 자로 잰다)를 컵 윗면에 붙여 테이블 위 두 카메라에 다 보이는 곳에 둔다.

```bash
python3 scripts/calib/global_cam_extrinsics.py --compare-id 74 --compare-size <m>
ros2 topic echo --once /objects/<컵>/pose      # 머리 FK 경로(FP++ → object_pose_node) 컵 base 좌표
```

머리 카메라의 74 base 좌표와 `/objects/<컵>/pose` 를 견준다. 차이는 마커 중심과 컵 원점 사이 오프셋만큼 나야 한다. 그다음 글로벌의 74 base 좌표와 견준다. 통과 기준안은 글로벌−머리 차 5 mm 이하다. 기준은 결과를 보고 사용자가 정한다. 컵 위치를 바꿔 가며 3곳에서 되풀이한다.

## 확인하지 못한 것

- 두 대 동시 스트림의 대역폭과 프레임 손실은 아직 재지 않았다(3번).
- realsense2_camera 4.58 에서 `serial_no:=_<시리얼>`, `camera_namespace`, `publish_tf:=false` 가 실제로 먹는지 실기에서 보지 않았다. 인자 목록에는 셋 다 있다.
- 글로벌 카메라 영상에 마커 70 · 71 이 들어오는지 모른다. 안 보이면 마운트를 옮기거나 마커를 하나 더 붙여야 한다(`anchor_truth` 추가).
- 글로벌 카메라의 깊이 tare 와 RGB 왜곡 계수는 아직 확인하지 않았다(머리는 11.17 에 EEPROM tare 를 했다). 이번 풀이는 RGB 만 쓴다.
- 글로벌 FP++ 는 `scripts/vision/fpp_global_up.sh`(컨테이너 fppg_*, ns /perception_plus_plus_global)로 머리와 따로 띄운다. 글로벌 object_pose_node 는 `S2R_FPP_NS`·`S2R_OBJECTS_NS` 를 둘 다 글로벌 값으로 export 한 셸에서만 돌린다(하나만 바꾸면 머리 토픽과 겹친다).
- 마커 75 · 76 은 `global_anchor_markers`(글로벌 전용)에 있다. 머리 풀이는 70 · 71 · 73 만 쓴다.
