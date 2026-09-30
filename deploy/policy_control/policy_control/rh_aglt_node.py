"""rh_aglt_node: RH56F1 한 팔 aglt 정책(hdgp open-rh_{r,l}_aglt) — pour_fj_node 의 rh_aglt 계열로 띄운다(09.30).

  in   robot yaml(한 팔, rh56f1_<side>_*): 팔 /joint_states · 손 /hand_<side>/joint_states · 촉각 /hand_<side>/tip_forces
       컵 PoseStamped(base): 파라미터 cup_topic (기본 /objects/cup_big_s100/pose)
  out  /policy_control/joint_target (팔 7 · 손 6) · /policy_control/{obs,action,episode} · status/rh_aglt_node
  srv  /policy_control/episode/{reset,start,stop,abort} — reset 때 컵 자세로 목표(컵 + goal_offset)를 정한다

    python3 deploy/policy_control/policy_control/rh_aglt_node.py --ros-args -p contract:=… -p robot:=…
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control.pour_fj_node import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(family="rh_aglt"))
