# RH56F1 정지 마찰 실험

가설: 정지 직전 관절 속 마찰이 마지막 1~2° 를 붙잡는다. 중력 계산 오차가 작다는 것은 11.08 분석에서 확인했다. 정의는 `docs/experiments/2026-11-07_rh56f1_stationary_tracking.md`, 실행 절차는 `common/RUNBOOK.md`.

## 폴더

- `common/` 여러 실험이 같이 쓰는 입력(`inputs/`), 스크립트(`scripts/`), 런북
- `_template/` 실험 README 틀
- `<YYMMDD>_<시험>_<side>/` 실험 하나(= 승인 하나). `README.md`, `setup/`(설정 사본·git 상태·계획·trial 목록), `runs/<trial>/`(원시 기록, git 제외), `result/`(요약 표·그림)

같은 폴더를 다시 쓰지 않는다. 다시 하면 `_b`. 중단된 폴더도 지우지 않는다.

## 진행

| 폴더 | 시험 | 팔 | 상태 | 결과 한 줄 |
|---|---|---|---|---|
