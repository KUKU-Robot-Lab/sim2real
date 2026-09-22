# archive — 더는 쓰지 않는 것

여기 있는 것은 **참조가 0** 인 상태로 확인된 뒤 옮겨졌다(2026-09-22 감사, 4관점 → 반증 검증).
지우지 않고 남긴 이유는 경위가 남아 있기 때문이다. 다시 쓸 일이 생기면 원래 자리로 되돌리면 된다.

## 왜 `scripts/` 밖인가

`tests/conftest.py` 와 `policy_control/policy_control/_paths.py` 는 **`scripts/` 의 모든 하위 디렉터리를
sys.path 에 얹는다**(`_SKIP` 은 `__pycache__` 계열뿐). 그래서 `scripts/` 안에 죽은 코드를 두면
모든 테스트와 policy_control import 가 그것을 import 경로에 달고 다닌다. 그래서 `scripts/archive/` 가
아니라 최상위 `archive/` 다.

## 들어 있는 것 (2026-09-22)

| 경로 | 무엇 | 대체된 것 |
|---|---|---|
| `scripts/deprecated/` | `sim2real_inference.py` · `sim2real_dryrun.py` | policy_control 체인 + `scripts/ops/mission_run.py` |
| `scripts/probes/` | `probe_head_push_response` · `probe_policy_on_sim_states` · `probe_s2r_right_record` | — (`probe_policy_on_sim_states` 는 parity 하네스였지만 테스트가 없어 계약 변경에 조용히 썩었다) |
| `scripts/analysis/` | `hand_status_monitor` · `monitor_left_side` · `monitor_right_side` | `policy_control/tools/status_board.py` · `s2r_console` |
| `scripts/calib/` | `cup_touch_survey` · `touch_probe_left` | — |
| `scripts/nodes/` | `npz_udp_feed` · `ros2_teleop_device` · `shaker_centroid_node` | shaker 과제 자체가 폐기 |
| `scripts/vision/` | `grab_frame` · `pose_stats_recorder` | `scripts/vision/grab_rgbd.py` · 인지 런처 |

문서에 남아 있는 언급(`sim2real_inference.py` 등)은 **경위 서술**이라 고치지 않았다.
