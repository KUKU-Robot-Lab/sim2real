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
