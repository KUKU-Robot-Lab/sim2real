# cup_pose 라이브 노드 + extrinsics 캘리브 — 설계

**날짜:** 2026-07-27
**범위:** cup 하나를 라이브 end-to-end로 완성하기 위한 **SP1(견고한 cup_pose ROS 노드)**
+ **SP2(T_base_cam extrinsics 캘리브)**. hdgp↔perception 자동 CAD 주입·다물체 일반화(SP3),
동적 목 보정, 핸드-아이 캘리브는 **별도 spec**.

## 1. 목표

카메라 → 견고한 6D cup pose → `/cup_pose`(robot base 프레임) → 정책 obs 까지 **cup 하나가
라이브로 흐르는 것**을 완성한다. 지금까지는 standalone 데모(`live_fp_demo.py`)만 모니터에서
검증됐고, ROS 파이프라인(노드→relay→`/cup_pose`)이 라이브로 연결·검증된 적 없다.

성공 기준:
1. 새 FP++ ROS 노드가 실기에서 손 드리프트(컵 들 때 손/몸 추종) 없이 cup을 추종하고,
   놓쳐도 in-place 재-앵커로 복구한다.
2. QR 기반 1회 정적 캘리브로 `T_base_cam`(PLACEHOLDER)을 실측값으로 채운다.
3. 노드 출력이 `cup_pose_relay`를 거쳐 `/cup_pose`(base)로 나오고, `sim2real_inference`가
   이를 구독해 그럴듯한 base 좌표(테이블 위 컵)를 받는다.

## 2. 확정된 설계 결정

- **FP++ upstream 서브모듈(`external/foundationpose_plus_plus`, pin 58aa715)은 수정하지 않는다.**
- **기존 `CupTrackingNode`/`TrackingManager`(perception_plus_plus_core)도 수정하지 않는다.**
  검증된 `live_fp_demo.py` 루프(detection-anchor + in-place re-anchor)를 담는 **새 ROS 노드**를
  만든다. live_fp_demo가 이미 어댑터를 직접 쓰는 방식(코어 무수정)이라 그걸 ROS 노드로 승격.
- **in-place 재-앵커는 FP++ 엔진 메서드(`estimator.register`/`cutie.initialize`/`kalman.initiate`)를
  호출만 한다** — FP++ 수정이 아니라 사용. 우리가 쓰는 건 전부 신규 파일 + sim2real relay 쪽뿐.
- **목 관절(카메라)은 캘리브한 home 자세에 고정**하고 운영한다. 동적 목 보정은 CAD 도착 후 별도.
- **T_base_cam 캘리브 = 정적 QR(테이블 부착)** 1회. 코드는 relay/yaml을 무수정,
  PLACEHOLDER 값만 실측값으로 채운다.
- 검출기는 기존 cup(COCO class 41) 그대로 — SP1은 cup 전용이라 검출 변경 없음.

## 3. 아키텍처

```
[vision-3090]  RealSense D435 (RGB/D/info)
      │
      ▼
  새 FP++ ROS 노드 (신규)                         ◄── live_fp_demo 루프 이식
      │ /perception_plus_plus/cup/pose  (PoseStamped, camera optical frame)
      ▼   (DDS, ROS_DOMAIN_ID=126, 동일 LAN)
  cup_pose_relay.py --in-type posestamped  (기존, 무수정)
      │      T_base_cam(SP2 실측) ∘ T_cam_cad ∘ T_cad_body(기존)
      ▼
  /cup_pose  (PoseStamped, robot base frame)
      ▼
  sim2real_inference.py (grasp, 106D obs, 기존)  →  grasp-v1 정책
```

기존 재사용: `cup_pose_relay.py`(posestamped 입력 경로 이미 존재), `global_camera_extrinsics.yaml`
(T_cad_body는 cup용 산출 완료, T_base_cam만 PLACEHOLDER), `sim2real_inference.py`.

## 4. SP1 — 견고한 cup_pose ROS 노드 (perception_plus_plus, 신규 노드)

### 4.1 문제 원인 (오늘 코드로 확인)

- **손 드리프트**: `TrackingManager`/`evaluate_quality`의 점프 게이팅은 손이 컵 바로 옆이라
  공간 점프가 작아 못 잡는다. Cutie 마스크가 손으로 전파되며 FP가 손을 추종.
- **복구 불가**: `TrackingManager` LOST 복구가 `adapter.reset()`+재`adapter.initialize()` 호출 →
  2번째 initialize가 Hydra(`GlobalHydra already initialized`)+CUDA tensor device 전역상태로 크래시.

### 4.2 해법 (live_fp_demo 검증된 두 수정을 새 노드에 이식)

- **(a) detection-anchored 유효성 게이트**: 매 프레임 YOLO cup bbox 검출, 추정 pose를 재투영
  (reproject)해 bbox+margin 밖이면 invalid. patience 프레임 연속 invalid + 검출 존재 시 재-앵커.
- **(b) in-place 재-앵커**: 기존 engine 재사용 — `estimator.register`(pose 재추정) +
  `cutie.initialize`(마스크 재시드) + `kalman.initiate`. 재구성 없음 → Hydra 안 건드림.
- **(c) fd-level stdout 억제**: FP가 프레임당 C/CUDA 프린트 → `os.dup2(devnull, 1)`로 fd1 억제,
  상태는 stderr. (라이브 FPS 1-3 → 11-12fps, 모션 추종의 실질 전제)

### 4.3 새 노드 구성

- **파일**: `perception_plus_plus/ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/anchored_node.py`
  (가칭). 신규 entrypoint로 등록(기존 `node.py`/`CupTrackingNode`는 그대로 둠).
- **구독**: RGB/aligned_depth/camera_info (message_filters ApproximateTimeSynchronizer) — 기존 노드와 동일.
- **발행**: `PoseStamped`(camera optical frame, object_to_camera) + `TrackingStatus` + TF —
  기존 노드와 **동일 토픽/계약** 유지(relay 무수정 소비).
- **로직**: `FoundationPosePlusPlusAdapter` + `YoloCupDetector` 직접 사용(TrackingManager 우회).
  1회만 `adapter.initialize`; 이후 매 프레임 `adapter.track` + detection-anchor 게이트 + 필요 시
  in-place 재-앵커. `bbox_depth_mask`(near-percentile35 + band + min-area fallback)로 마스크 생성.
- **순수 로직 분리**: `reproject`/`inside_box`/`bbox_depth_mask`/변환 헬퍼는 ROS 비의존 모듈로
  빼서 유닛테스트 가능하게 한다.

### 4.4 SP1 out-of-scope

- 컵 완전 가림(검출 없음) 시 즉시 재-앵커 불가 — 컵 재출현 시 자동 복구(문서화된 한계).
- 다물체·비-COCO 객체 검출(SP3).

## 5. SP2 — T_base_cam extrinsics 캘리브 (sim2real, 신규 스크립트)

### 5.1 방식

목 home 고정 전제, 테이블 부착 QR 1회 정적 캘리브.

```
카메라 1프레임 캡처 (RGB + camera_info K)
  → OpenCV QRCodeDetector: QR 4모서리 검출
  → cv2.solvePnP(QR 실측 3D 코너, 2D 코너, K, distCoeffs)  →  T_cam_qr
  → T_base_cam = T_base_qr ∘ inv(T_cam_qr)
  → global_camera_extrinsics.yaml 의 camera.position / orientation_wxyz 갱신
```

### 5.2 새 스크립트 구성

- **파일**: `sim2real/scripts/calibrate_camera_extrinsics.py`.
- **입력**:
  - `--qr-size` QR 물리 한 변 길이(m).
  - `T_base_qr` = QR 프레임의 robot base 기준 pose(테이블 실측). 인자 또는 별도 yaml.
  - 프레임 소스: 라이브 토픽 구독 또는 저장 NPZ(오프라인). K는 camera_info/NPZ에서.
- **출력**: `global_camera_extrinsics.yaml`의 `camera.position`/`orientation_wxyz`(wxyz) 갱신.
  relay·yaml 스키마 무수정.
- **정확도 보완**: 단일 QR PnP는 회전 정밀도가 낮을 수 있어 (a) 되도록 큰 QR
  (b) N프레임 평균 (c) 4모서리 재투영 잔차(px) 리포트로 sanity.

### 5.3 SP2 out-of-scope

- 동적 목(T_neck_cam via CAD + 목 joint 실시간 구독), 핸드-아이 캘리브.

## 6. 데이터 흐름 (통합)

```
QR 캘리브(1회) ─→ global_camera_extrinsics.yaml [T_base_cam 실측]
                          │
새 FP++ ROS 노드 ─ /perception_plus_plus/cup/pose (cam) ─→ cup_pose_relay
                          (T_base_cam ∘ T_cam_cad ∘ T_cad_body) ─→ /cup_pose (base) ─→ 정책
```

## 7. 테스트 / 검증

- **SP1 유닛(ROS 비의존)**: `reproject`(pose→uv), `inside_box`(margin 경계), `bbox_depth_mask`
  (near-band + fallback), 4x4↔pose 변환. 순수 로직 테스트.
- **SP1 라이브(vision-3090, 실기)**: 노드 기동 → `/perception_plus_plus/cup/pose` 발행 확인 →
  relay → `/cup_pose` 도달 확인 → 손 드리프트 유발 시 재-앵커 복구 관찰.
- **SP2 유닛**: 합성 QR corner→알려진 T_cam_qr 역산 검증, `T_base_cam = T_base_qr ∘ inv(T_cam_qr)`
  합성 변환 정확성, wxyz 직렬화 라운드트립. QR 재투영 잔차 계산.
- **SP2 sanity(실기)**: 캘리브 후 relay 기동 → `/cup_pose`가 테이블 위 그럴듯한 base 좌표에
  떨어지는지 확인.

## 8. 리스크

- **단일 QR PnP 회전 정밀도**: 회전 오차가 크면 향후 ArUco/ChArUco 보드로 승격(별도).
- **DDS 연결**: vision-3090 ↔ 로봇 PC 동일 LAN + ROS_DOMAIN_ID=126 필요. 미연결 시 라이브 검증 보류.
- **목 자세 재현성**: 운영 시 목이 캘리브 home 자세와 정확히 같아야 T_base_cam 유효.

## 9. 명시적 out-of-scope (각각 별도 spec)

- **SP3**: hdgp 태스크(체크포인트+객체 USD) → CAD mesh 자동 추출·주입 + T_cad_body 자동 산출 +
  검출 클래스 일반화 + pose를 정책 입력으로(cup 전용 → 임의 객체).
- 동적 목 보정(CAD + 목 joint 실시간), 핸드-아이 캘리브.
