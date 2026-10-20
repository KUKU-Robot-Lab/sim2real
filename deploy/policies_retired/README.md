# 옛 배포 정책(10.20 은퇴)

10.20 사용자: "newton 세션하고 aglt_new 세션에서 전달해준 grasp, approach 정책들로만 policy 노드의 grasping, approach 탭에
존재하게 해줘 그전 것들은 제거". 그래서 `deploy/policies/rh56f1/aglt/` 에는 새 정책 4개(right/left_approach_ap05 · right/left_rcg03 — 같은 날 rcg02 를 대체)만 남기고,
그 전 정책 9개와 대체된 rcg02 2개를 여기로 옮겼다.

- 상황판 정책 탭 · 정책 등록부(`policies.py` · `policy_registry.scan`)는 `deploy/policies/` 만 읽는다. 여기 있는 정책은 실행되지 않는다.
- 남긴 이유: 시험 고정본(계약 · env.yaml · 체크포인트)으로 쓴다. `.pth` 는 git 밖이다.
- 1-B 손 닫기 계약은 정책과 떼어 `config/hand/rh56f1_<팔>_hand.json`(옛 `<팔>_approach` 계약 그대로, md5 같음)을 쓴다.
- 되살리기: 폴더를 `deploy/policies/rh56f1/aglt/` 로 다시 옮기고 `python3 deploy/policy_control/tools/policies.py --write-index --shallow`.

| 폴더 | 원래 역할 |
|---|---|
| right_rg3wo · left_rg3wo | 경로 A 파지 정책(10.17), tilt 경로 A |
| right_neo03m · left_neo03 | 10.19 rg3 카드 기본 파지 정책 |
| right_rg3hp3 | 10.14 위치제어 파지 |
| right_approach · left_approach | 경로 C 접근(rg3wc ep800). 1-B 손 계약의 원본 |
| right_approach_rg3wg · left_approach_rg3wg | 10.18 접근 후보, tilt 경로 C |
| right_rcg02 · left_rcg02 | 10.20 파지 · 들기(rcg_02_nr ep1300 · rcg_02_nlm ep300). 같은 날 newton 최종 정책 rcg03(rcg_03_nr ep1400 · rcg_03_nl ep1500)으로 대체 |
