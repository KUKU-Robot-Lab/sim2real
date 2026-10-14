# 목 카메라 보정 프로필 (2026-10-14)

FP++ 가 낸 카메라 프레임 물체 pose 를 base 좌표로 바꿀 때 쓰는 hand-eye 와 목 각도별 보정을 **프로필 파일 하나**로 고른다.
코드는 프로필 내용을 모른다. 프로필을 바꿔도 코드는 안 바뀐다.

| 조각 | 위치 | 역할 |
|---|---|---|
| 프로필(데이터) | `config/calib_profiles/<이름>.yaml` | hand-eye `T_neck_cam`, 각도별 xy 보정 표, 허용 각도 오차, strict 여부 |
| 읽기·검사(코드) | `scripts/calib_profile.py` | 로드, 각도 조회, 보정값 반환, 스키마 검사(순수, ROS 무관) |
| 적용(FP++) | `scripts/nodes/object_pose_node.py --calib-profile <이름>` | 프로필의 hand-eye 로 `T_base_cam` 을 목 각도마다 계산하고 각도별 보정을 더한다 |
| 적용(위치 추정) | perception_plus_plus `scripts/locate_object.py --calib-profile <이름>` | 같은 프로필을 쓴다 |
| 생성 | perception_plus_plus `scripts/make_calib_profile.py` | hand-eye 후보 + 영상(CAD 볼트)에서 각도 표를 만든다 |

## 쓰는 법
```
python3 scripts/nodes/object_pose_node.py --head-joint-topic /head/joint_states --calib-profile v6_1014
```
- `--head-joint-topic` 가 꼭 필요하다(각도별 보정이므로).
- **strict(기본)**: 목 각도가 프로필 표(허용 ±1°)에 없으면 그 pose 는 발행하지 않는다(경고만). 틀린 좌표는 없느니만 못하다.
- 프로필을 안 주면 지금까지와 똑같이 동작한다(기본 hand-eye `config/head_extrinsics.yaml`).

## 다른 환경에서 한 번에 만들기 (원테이크)
대화로 값을 주고받지 않고 **명령 하나**로 프로필을 만든다(perception_plus_plus 저장소, 로봇 PC):
```
python3 scripts/calibrate_profile.py --name <이름> --tilt -30 --pans="-60,-40,-20,0,20,40" --execute   # 목이 움직인다: 사용자 허락 뒤 한 번
python3 scripts/calibrate_profile.py --name <이름> --surveys <이미 찍은 서베이 폴더>...                  # 목 안 움직임
```
한 스크립트가 촬영(기준 자세 + tilt 고정 pan 목록, FP++ 포함) → CAD 구멍으로 시작 hand-eye → 구멍 짝짓기 → 맞춤 2회 → 각도별 보정 → 기존 정적 좌표계 앵커 →
아는 z → 프레임 하나씩 빼는 검증 → `config/calib_profiles/<이름>.yaml` 과 `report/survey/calib_<이름>/report.md` 까지 한다. 이미 있는 이름이면 거부한다.
필요한 것: 이 환경의 테이블이 CAD(env_v1)와 같을 것, 기준 자세(pan 0/tilt -20) 영상에서 구멍 12개 안팎이 보일 것, vision 호스트에서 FP++ 가 뜰 수 있을 것.
환경마다 hand-eye 가 달라도(구멍이 수십 px 어긋나도) 기준 자세에서 CAD 전체 탐색으로 시작값을 얻는다. 다른 hand-eye 를 기준으로 삼으려면 `--base-hand-eye <yaml>`.
10.14 실측(기존 서베이 6개, 41장)으로 같은 방법을 처음부터 돌려 보면 1.4 초에 프로필이 나오고, 앵커는 손으로 만든 것과 같다(회전 +1.1°, 이동 (+7.2, -14.3) mm).

## 현재 권장 (10.14 저녁)
- **프로필 `v9_1014_anchored_z`**: v6 hand-eye + 각도별 보정(22각도) + 기존 검증 좌표계 앵커 + **아는 z**(`known_z`).
  정지 pose 의 z 를 (상판 0.205 + 물체 원점 높이)로 맞춘다. FP++ z 는 깊이 편향으로 2~4 mm 낮았다. 오차가 ±15 mm 를 넘으면(들림·기움) FP++ 값 그대로,
  live 추적(`/live_pose`)은 쥐고 들면 높이가 변하므로 건드리지 않는다.
- **운용 자세(tilt -30 고정, pan 만)**: pan -60, -40, -20, 0, +20, +40 (6곳). 병(24 cm) 통째로 보이는 테이블 77%, 낮은 컵(7 cm) 93%.
  pan +60 은 오른손이 화면을 가린다. 이 자세만 프로필 각도 표에 있다(`locate_object.py --tilt -30 --calib-profile v9_1014_anchored_z`).
- **tilt -30 에서는 FP++ 스냅샷 노드에 tilt -30 용 camera extrinsics 를 줘야 한다**: `config/global_camera_extrinsics_tilt30.yaml`
  (`perception_launcher_node.py --camera-extrinsics ...`). 기준 자세 파일을 주면 후보 거르기가 후보를 버려 검출이 1/14 로 떨어졌다(5/11 로 회복).
- 알려진 문제(미해결, FP++ 쪽 버그로 보임): 목을 돌린 뒤 검출에 실패하면 FP++ 가 이전 자세의 pose 를 그대로 다시 낸다 → 값이 같으면 낡은 값이다
  (`survey.drop_stale` 이 걸러낸다). shaker 는 pan +20~+40 에서 사진에는 보이는데 새 값이 안 나온다.

## 프로필 두 개
- `v6_1014`: CAD 볼트로 맞춘 좌표계(CAD 기준). 기존 검증된 FP++ 출력과 기준 자세에서 x -5~-12 · y +7.5 · z +3 mm 다르다.
- `v6_1014_anchored`: 위와 같되 **기존 검증된 FP++ 좌표계에 붙였다**(`frame_anchor`: 회전 +1.02° · 이동 (+7.0, -14.0) mm · dz -3.1 mm).
  기준 자세에서 기존 출력과 x -1.2~-1.9 · y +0.3 · z ±0.4 mm 안으로 같고, 다른 각도도 같은 좌표계로 나온다. **기존 정책·궤적과 함께 쓸 때는 이쪽.**
  만들기: perception_plus_plus `scripts/anchor_profile.py`.

## 프로필 v6_1014 상세
- hand-eye: CAD 볼트만으로 맞춘 v6(perception_plus_plus `head_extrinsics_cad_v6_candidate.yaml`).
- 각도 표 17개: tilt -20 에서 pan -45~+45, tilt -25·-30 에서 pan -50~+30 일부. 보정은 0.3~3.2 mm 평행이동.
- 근거 데이터: survey 1013f/h/j, board_1014a. 같은 영상에서 hand-eye 와 보정을 맞췄으므로 표의 각도에서 가장 정확하고,
  표에 없는 각도는 검증되지 않았다.
- 새 각도를 더하거나 hand-eye 를 바꿀 때는 `make_calib_profile.py` 로 **새 이름의 파일**을 만든다(기존 파일을 덮어쓰지 않는다).

## 실측(같은 FP++ 영상을 세 방식으로 변환, 서베이 1013h·j, 자세 7곳)
| 변환 | 자세 간 표준편차 (shaker) | 최대 이탈 |
|---|---|---|
| 정적(기준 자세 값) | (60, 190) mm | 271 mm |
| 동적 hand-eye v2 | (11, 5) mm | 18 mm |
| 프로필 v6_1014 | (2.7, 1.0) mm | 5.1 mm |
같은 영상에서 hand-eye 를 맞춘 값이라 표의 각도에서만 이 정도다. FP++ 자체의 깊이 편향(z 2~3 mm 낮음)은 별개다.

## 목 시리얼 단독 소유 (2026-10-15, 사용자 결정 c)

`head_joint_publisher.py` 한 프로세스가 `/dev/ttyUSB0` 를 혼자 맡는다. 각도 발행(20 Hz)과 이동 요청을 한 스레드에서 번갈아 처리하므로
"Incorrect status packet" 이 나지 않는다(예전 충돌 원인: 읽기 프로세스와 쓰기 프로세스가 같은 버스를 같이 씀).

- 이동: `scripts/head_move.py --execute [--pan P --tilt T]`(인자 없으면 기준 자세). 포트를 열지 않고 `/head/move_request` 로 요청, `/head/move_status` 로 결과. 출력 줄은 `head_home.py` 검증과 같은 형식이라 `survey.parse_head_home` 이 읽는다. 종료 코드 0 도달, 1 실패, 2 거절·드라이버 없음.
- 이동 중에도 `/head/joint_states` 발행은 끊기지 않는다(object_pose_node 의 stale 보류가 그대로 동작).
- 정착 검증은 드라이버가 읽은 현재 각도로 한다: 최근 20샘플 폭 ≤ 0.2° 이고 1 s 이상 지나면 정지, 평균이 목표 ±0.3° 안이면 ok. 20 s 안에 못 서면 실패.
- 두 명령자 동시 금지: 이동 중 새 요청은 `busy` 로 거절. 다른 프로세스가 포트를 직접 열면(`head_home.py`·`head_compliant_hold.py`·`head_position_hold_node.py`) `port_lock` 이 거절한다.
- 범위: tilt ≤ -10°(인코더, 위쪽은 제한 없음), pan ±180°. 범위 밖 요청은 모터에 닿지 않는다.
- 하드웨어 없이 리허설: `ROS_DOMAIN_ID=99 python3 scripts/fakes/fake_head_driver.py`(같은 토픽·규약).
- 기동 순서: 드라이버(`head_joint_publisher.py`)를 먼저 띄운 뒤 locate·survey·calibrate 를 쓴다. `head_home.py --execute` 는 드라이버를 내린 뒤에만 쓴다.
- FP++ 스냅샷 노드의 camera 파일은 정책 운용(기준 자세 tilt -20)이면 기본 파일. tilt -30 스캔 때만 `global_camera_extrinsics_tilt30.yaml`.
