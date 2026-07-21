# grasp-v1 sim2real 연계 — 진행 ledger
plan: docs/superpowers/plans/2026-07-21-grasp-v1-sim2real-linkage.md
branch: feat/grasp-v1-sim2real-linkage
제약: 로봇 PC·카메라 미연결 → ROS 런타임 스텝은 준비만(게이팅). 순수 로직은 pytest 실검증. sim2real 단일 디렉토리. hdgp READ-ONLY.

- Task 1: complete (commits 2c0a0b9..9c25288, review clean; Minor[plan-mandated]: parity test 빈약 — 최종리뷰 트리아지)
- Task 2: complete (commits 9c25288..8bc45c6, review clean; 계획 '바닥중심' 가정→증거기반 정정 translation[0,0,0]. Minor: 미사용 aabb 파라미터[의도])
- Task 3: complete (commit 8bc45c6..aba8b04, review clean; 도메인 체크 exit0/1 실검증. Minor: 리포트 줄수 오기재[cosmetic])
- Task 4: complete (commit aba8b04..c20ccd5, review clean; 발행노드 실기동 검증. Step3(정책 dryrun) 하드웨어게이팅. Minor: 비가드 orbit split[비차단])
- Task 5: complete (commit c20ccd5..701c6e6, review clean; 체크포인트/vendor/토픽 전부 실검증. 발견: 기존 도크스트링 stale 경로, vendor 이미 in-repo)

## 최종 (2026-07-22)
- 전체 5 태스크 완료. 최종 whole-branch 리뷰(opus): READY TO MERGE. Critical/Important 0, Minor 4(전부 ship 가능):
  T1 파리티 테스트 명명 과장 / T2 미사용 aabb 파라미터[의도] / T4 비가드 orbit split[dev툴] / T5 인용라벨 → e9e96a7로 정정 완료.
- 브랜치 feat/grasp-v1-sim2real-linkage, 커밋 2c0a0b9..e9e96a7. 테스트 13 passed(ROS 무관).
- 하드웨어 게이팅(범위 밖, 문서화됨): T_base_cam 캘리브 / FP++ 라이브 ROS 노드 / 라이브 grasp 동작.
