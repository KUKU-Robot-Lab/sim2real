# S2R 콘솔

기존 실행 도구를 사용한다. 모의 화면 확인은 다음과 같다.

```bash
PYTHONPATH=deploy/s2r_console python3 -m s2r_console \
  --profile rh56f1_fake --operator operator --no-bridge
```

실제 fake ROS 노드 검증에는 ROS 환경과 정책 실행용 Python 환경이 필요하다.
`--no-bridge`는 화면과 API 확인용이며 로봇 상태를 만들어 내지 않는다.

상단 control은 모든 탭에서 접근한다. setting에서 기반 연결을 준비하고,
control에서 engage·home·rehome·return과 목 자세를 실행한다. policy(10.21)는
위에서부터 FP++ 한 줄(판정 · 물체 자리, [영상]은 누를 때만) → 실험 목록 |
고른 실험(정책 사슬 · 컵 · 막힌 이유 · ▶ 실행) → 지금 에피소드 → 접은 칸
(정책 목록 · 회차 기록 · 인지 상세)이다. 실험 = 경로 하나: 파지 · 들기(1-B · C ·
D · E · F, 팔마다 `ctl_grasp_<경로>_<팔>`)와 tilt(D · C · B, 미션 단계 ▶).
사슬은 에피소드 파일과 정책 등록부에서 읽는다(`s2r_console/experiments.py`).
정책 카드 배치(←→↑↓)는 화면에서 뺐다(사용자 "일단 제거").

인지 상세에서 물체와 입력원을 고른다. locate 결과는 별도 입력으로 표시하며
FPP를 자동 대체하지 않는다. 측정되지 않은 색 조합은 기존 색상 프리셋을 사용한다.

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
콘솔의 locate 재추출은 현재 목 실측 자세에서만 여러 장을 촬영한다.
목 이동·탐색·홈 복귀 명령은 보내지 않으며, 촬영 전후 목 자세가 바뀌면 결과를 무효로 한다.
성공한 locate 좌표는 유지하고 FPP 실패는 작업 결과에 따로 남긴다.
rg3는 계속 신선한 FPP `live_pose`가 필요하다. 계약·등록부 검사가 정상인
`candidate` 정책은 미리보기 경고와 운영자 승인으로 실기 시험할 수 있다.
정책 파일이 삭제되면 해당 명령만 실행 불가 사유를 표시한다. 상태 조회와 다른
관리 작업은 계속 사용할 수 있다. 정책 시간 초과 시 자동 손 열기·복귀·재시도는
하지 않고 마지막 팔·손 목표를 유지한다. 복구는 운영자가 rehome으로 실행한다.
rg3 에피소드는 다른 실패도 자동 재시도하지 않는다.

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
