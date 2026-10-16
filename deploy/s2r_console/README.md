# S2R 콘솔

기존 실행 도구를 사용한다. 모의 화면 확인은 다음과 같다.

```bash
PYTHONPATH=deploy/s2r_console python3 -m s2r_console \
  --profile rh56f1_fake --operator operator --no-bridge
```

실제 fake ROS 노드 검증에는 ROS 환경과 정책 실행용 Python 환경이 필요하다.
`--no-bridge`는 화면과 API 확인용이며 로봇 상태를 만들어 내지 않는다.

상단 control은 모든 탭에서 접근한다. setting에서 기반 연결을 준비하고,
control에서 engage·home·rehome·return과 목 자세를 실행한다. policy에서는
인지 입력과 물체를 선택하고 등록된 정책 노드를 배치한다.

정책 파일 선택과 배치 복원은 실행하지 않는다. 현재 실행을 연결한 정책은
rg3이며, 활성화한 물체의 **신선한 FPP live_pose**가 필요하다. locate 결과는
별도 입력으로 표시하며 FPP를 자동 대체하지 않는다. 측정되지 않은 색 조합은
기존 색상 프리셋을 사용한다.

모든 새 실행은 미리보기의 `취소 / 승인 후 실행`을 사용한다. 확인 ID는
현재 run·조작권·명령·계약·정책·물체·입력원에 묶이며 한 번만 사용할 수 있다.
정책 실행 도중 제어를 요청하면 정지와 PD 자세 유지 확인을 먼저 수행한다.
중단한 정책은 자동 재개하지 않는다.

정책 카드 배치는 `runs/s2r_console/workspaces/`에 로봇별로 저장한다.
작업 실행 중에는 실행 설정을 변경할 수 없으며, rg3 실행 중 다음 목 제어를
준비하기 위한 목 자세 설정은 변경할 수 있다.

영상은 `/api/video`가 기존 MJPEG 서비스를 중계한다. 기본 upstream은
`http://127.0.0.1:8080/stream`이며 다른 호스트나 SSH 터널을 사용한다면
콘솔 시작 전에 `S2R_VIDEO_URL`을 설정한다. 브라우저는 영상 주소를 선택할 수 없다.
영상이 끊기면 표시된 상태를 확인하고 `영상 다시 연결`을 사용한다.
RH56F1의 `cups` 및 FPP 인지 재추출은 vision-3090의 ROS 영상 구독형
MJPEG 뷰어를 함께 요청한다. 인지 런처가 로컬 8080 SSH 터널을 관리하며,
기존 터널이 있으면 재사용한다. 원격 뷰어는 모니터나 카메라 장치 직접 열기가
필요 없다. 원격 저장소에도 최신 `scripts/vision/viewer_up.sh`가 있어야 한다.

FPP와 locate를 함께 선택하면 FPP 재추출에 실패해도 locate를 실행한다.
성공한 locate 좌표는 유지하고 FPP 실패는 작업 결과에 따로 남긴다.
rg3는 계속 신선한 FPP `live_pose`가 필요하다. 계약·등록부 검사가 정상인
`candidate` 정책은 미리보기 경고와 운영자 승인으로 실기 시험할 수 있다.

shutdown은 팔 받침과 토크 해제를 확인한 뒤 관리하는 연결을 순서대로 종료하고
결과 화면을 유지한다. 콘솔 창을 닫을 때의 기존 실기 드라이버 보존 동작은
shutdown 작업과 별개다.

주요 자동 검증:

```bash
pytest -q tests/s2r_console tests/test_object_registry.py tests/test_head_home.py
python3 scripts/ops/make_rh56f1_missions.py --check
node --check deploy/s2r_console/s2r_console/web/console.js
```

실기 수락 검증은 실행 승인을 받은 뒤 연결 → 자세 제어 → 영상·인지 → rg3
순서로 수행한다.
