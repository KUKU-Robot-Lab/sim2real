"""★11.18 사용자("가"): 경로 F 잡기 노드(ap05 → rcg04)는 손 배포 보조 A · B · C 를 빼고 노드 기본값(학습 그대로 손)으로 돈다.
칵테일 카드는 같은 넘김 노드라도 보조를 그대로 둔다. lift_grasp_palm_m(정지 판정만)과 끊김 정지는 둘 다 남는다.
★11.18 사용자: 경로 F 는 FP++ 값 보조(튐 거르기 · 마지막 값 유지 · FK 부착 fallback/guard · 손 안 GRU · 컵 오프셋)도 끈다."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
ASSIST = ("hand_unfreeze_on_grasp:=true", "thumb_support_g:=80.0", "thumb_open_floor_rad:=0.06")
KEEP = ("lift_grasp_palm_m:=0.08",)
FPP_ASSIST = ("cup_hold_last", "attach_fallback_s", "attach_guard_m", "fpp_gate_jump_m", "fpp_gate_anchor_m",
              "cup_dropout_stop_s")
LEFT_OFFSET = "cup_obs_offset_base_m:=[0.0028,0.0283,0.0046]"   # ★11.18 경로 F 왼손만 넣는다


def _rcg04_nodes(name: str) -> dict[str, list[list[str]]]:
    """단계 → ap05 → rcg04 넘김 rh_aglt 노드 argv 들."""
    run = yaml.safe_load((REPO / "config" / name).read_text(encoding="utf-8"))["run"]
    out: dict[str, list[list[str]]] = {}

    def walk(x, sid):
        if isinstance(x, dict):
            av = x.get("argv")
            if isinstance(av, list) and any("rh_aglt_node.py" in str(a) for a in av) \
                    and any(str(a).startswith("handoff_contract:=") and "_rcg04/" in str(a) for a in av):
                out.setdefault(sid, []).append([str(a) for a in av])
            for v in x.values():
                walk(v, sid)
        elif isinstance(x, list):
            for v in x:
                walk(v, sid)

    for sid, stage in run.items():
        walk(stage, sid)
    return out


@pytest.mark.parametrize("mission", ["mission_rh56f1_control.yaml", "mission_rh56f1_control_cam5090.yaml",
                                     "mission_rh56f1_fake.yaml"])
def test_no_hand_assists_anywhere_and_route_f_keeps_only_the_left_offset(mission):
    nodes = _rcg04_nodes(mission)
    cocktail = {s for s in nodes if s.startswith("episode_cocktail")}
    route_f = set(nodes) - cocktail
    assert {"episode_grasp_f_right", "episode_grasp_f_left"} <= route_f and cocktail
    for sid, argvs in nodes.items():
        for av in argvs:
            assert all(k in av for k in KEEP), (sid, av)
            assert not any(a.split(":=")[0] in x for a in ASSIST for x in av), sid   # ★11.18 A · B · C 모두 제거
            if sid in cocktail:
                assert "cup_hold_last:=true" in av and "fpp_gate_anchor_m:=0.03" in av, sid
                assert "cup_dropout_stop_s:=1.0" in av, sid
            else:
                left = "ns:=left" in av
                assert (LEFT_OFFSET in av) == left, (sid, av)
                assert not any(x.startswith("cup_obs_offset_base_m:=") for x in av) or left, sid
                assert not any(x.startswith(k + ":=") for k in FPP_ASSIST for x in av), (sid, av)
                assert "inhand_est:='off'" in av, sid
