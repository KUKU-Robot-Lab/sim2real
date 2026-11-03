#!/usr/bin/env python3
"""11.03 실기 bag 오프라인 재생 — 경로 F 넘김 규칙(창 평균 멈춤 · 3 cm)과 FP++ 기준점 거르기를 실측에 대 본다.

bag 의 /joint_states → 손바닥 FK(노드와 같은 UrdfChainFK · 계약 palm_body), /objects/<물체>/live_pose → FppGate,
노드 상태 토픽 메시지 하나 = 정책 틱 하나(60 Hz). 틱마다 새 ApproachGrasp(handoff=1) 의 after_policy 를 부른다.

★개루프다: 손바닥 경로는 실기에서 옛 규칙 · 옛 거르기로 돈 그대로다. 새 규칙이 넘긴 뒤(또는 실패한 뒤)의 동작과
  15:03 에서 거른 컵을 받았다면 정책이 어디로 갔을지는 이 재생으로 알 수 없다 — 넘긴 시각 · 그때 오차까지만 의미가 있다.

    .venv/bin/python scripts/analysis/replay_handoff_gate_1103.py
"""
from __future__ import annotations

import bisect
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy" / "policy_control"))

from policy_control import approach_grasp as AG  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402
from policy_control.contract_assets import ASSETS  # noqa: E402
from policy_control.fk_numpy import UrdfChainFK  # noqa: E402
from policy_control.fpp_gate import FppGate, GateCfg  # noqa: E402

BAGS = REPO / "logs" / "bags"
#: 미션 AGLT_LIVE_ARGS 와 같은 값 — 옛(실기에서 돈) 거르기 · 새 거르기
OLD_GATE = GateCfg(jump_m=0.02, dz_m=0.02, radius_m=0.2, max_hold_s=2.5)
NEW_GATE = GateCfg(jump_m=0.02, dz_m=0.02, radius_m=0.2, max_hold_s=2.5, anchor_m=0.03)
CASES = [  # (이름, bag, 팔, 계약 물체, FP++ 물체, t0, t1)
    ("12:23 오른손 blue", "20261103_122326_episode_cocktail_1c_e4000", "right", "source240_real", "source240b_blue",
     1793676331.2, 1793676348.0),
    ("12:23 왼손 shaker", "20261103_122326_episode_cocktail_1c_e4000", "left", "shaker_c_real", "shaker_c_orange",
     1793676372.5, 1793676389.5),
    ("15:03 오른손 blue", "20261103_150257_episode_cocktail_1c_e4000", "right", "source240_real", "source240b_blue",
     1793685989.9, 1793686005.0),
]


def _read(db: Path, topics: list[str], t0: float, t1: float) -> dict[str, list]:
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    kinds = {"js": JointState, "pose": PoseStamped, "str": String}
    c = sqlite3.connect(db)
    ids = {n: i for i, n in c.execute("select id, name from topics")}
    out: dict[str, list] = {}
    for name in topics:
        kind = "js" if name == "/joint_states" else "pose" if name.endswith("pose") else "str"
        rows = c.execute("select timestamp, data from messages where topic_id = ? and timestamp between ? and ? "
                         "order by timestamp", (ids[name], int((t0 - 1.0) * 1e9), int(t1 * 1e9)))
        out[name] = [(ts * 1e-9, deserialize_message(d, kinds[kind])) for ts, d in rows]
    return out


def _latest(ts: list[float], rows: list, t: float):
    i = bisect.bisect_right(ts, t) - 1
    return None if i < 0 else rows[i][1]


def replay(label, bag, side, obj, fpp, t0, t1) -> dict:
    c = A.select_object(A.load_contract(REPO / f"deploy/policies/rh56f1/aglt/{side}_approach_ap05/rh_aglt_contract.json"), obj)
    s = c.side()
    fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    hand0 = np.asarray(s.hand_open, float)
    st_topic, live_topic, snap_topic = (f"/policy_control/status/rh_aglt_node_{side}", f"/objects/{fpp}/live_pose",
                                        f"/objects/{fpp}/pose")
    d = _read(BAGS / bag / "arm" / "arm_0.db3", ["/joint_states", live_topic, snap_topic, st_topic], t0, t1)
    js = [r for r in d["/joint_states"] if f"openarm_{side}_joint1" in r[1].name]
    js_t = [t for t, _ in js]
    live_t = [t for t, _ in d[live_topic]]
    snap_t = [t for t, _ in d[snap_topic]]
    names = [f"openarm_{side}_joint{k}" for k in range(1, 8)]          # 계약 arm_joints(?_aj_1..7) 순서

    gates = {"old": FppGate(OLD_GATE), "new": FppGate(NEW_GATE)}
    ags = {k: AG.ApproachGrasp(AG.ApproachGraspCfg(handoff=1), c.policy_hz) for k in ("new", "old")}   # 새 넘김 규칙 × 거르기
    for a in ags.values():
        a.next = (None, None)
    ag = ags["new"]
    chain = NS(c=c, step_i=0, handover=lambda *_: None)
    out = {"label": label, "rows": [], "new": None, "old": None, "real": None, "far": None, "anchor": None}
    t_first = None
    for t, msg in d[st_topic]:
        st = json.loads(msg.data)
        g = st.get("approach_grasp") or {}
        if t < t0 or st.get("phase") != "running":
            continue
        if out["real"] is None and g.get("phase") not in (None, "approach"):
            out["real"] = (round(t - t0, 2), g.get("phase"), g.get("handoff_t_s"), g.get("why", "")[:60])
        if g.get("phase") not in (None, "approach") and out["new"] is not None and out["old"] is not None:
            break                                                          # 둘 다 넘김 뒤 — 개루프라 더 볼 것이 없다
        j = _latest(js_t, js, t)
        lv = _latest(live_t, d[live_topic], t)
        sn = _latest(snap_t, d[snap_topic], t)
        if j is None or lv is None:
            continue
        q = np.array([j.position[list(j.name).index(n)] for n in names], float)
        palm = np.asarray(fk.palm_pose(q, hand0).palm_pos, float)
        p = lv.pose.position
        live = ((p.x, p.y, p.z), (1.0, 0.0, 0.0, 0.0), lv.header.stamp.sec + lv.header.stamp.nanosec * 1e-9)
        cups = {k: np.asarray(gt.step(live, t, palm, active=True)[0], float) for k, gt in gates.items()}
        snap = None if sn is None else np.array([sn.pose.position.x, sn.pose.position.y, sn.pose.position.z])
        chain.step_i = int(st.get("step", 0))
        for k, a in ags.items():
            if a.phase != "approach":
                continue
            a.after_policy(chain, A.RaMeas(arm_q=q, arm_qd=np.zeros(7), hand_q={}, palm_pos=palm, palm_R=np.eye(3),
                                           tips=np.zeros((5, 3)), cup_pos=cups[k], cup_quat=np.array([1.0, 0, 0, 0]),
                                           tactile_n=np.zeros(5)))
            if a.phase != "approach":
                L = a.last
                off = float(np.hypot(L["off_r_m"], L["off_h_m"])) if "off_r_m" in L else float("nan")
                out[k] = (round(t - t0, 2), a.phase, round(off * 100, 2), L.get("palm_mps_win"))
        t_first = t if t_first is None else t_first
        rr, oh, orr = A.ready_pose_error(c, cups["new"], palm)
        e_snap = float("nan") if snap is None else float(np.hypot(*A.ready_pose_error(c, snap, palm)[1:]))
        out["rows"].append(dict(t=round(t - t0, 2), real_off=None if "off_r_m" not in g else round(float(np.hypot(g["off_r_m"], g["off_h_m"])) * 100, 2),
                                rep_off_old=round(float(np.hypot(*A.ready_pose_error(c, cups["old"], palm)[1:])) * 100, 2),
                                rep_off_new=round(float(np.hypot(oh, orr)) * 100, 2), off_snap=round(e_snap * 100, 2),
                                d_new=round(float(np.linalg.norm(cups["new"] - palm)), 3),
                                dz_old=round(float(palm[2] - cups["old"][2]) * 100, 2), dz_snap=None if snap is None else round(float(palm[2] - snap[2]) * 100, 2),
                                cup_new=np.round(cups["new"], 3).tolist(), cup_old=np.round(cups["old"], 3).tolist(),
                                live=np.round(live[0], 3).tolist(), palm=np.round(palm, 3).tolist(),
                                real_mps=g.get("palm_mps"), win_mps=ag.last.get("palm_mps_win") if ag.phase == "approach" else None))
    ng = gates["new"]
    anchor = np.asarray(ng.anchor[0], float)
    drift = max(float(np.linalg.norm(np.asarray(r["cup_new"]) - anchor)) for r in out["rows"])
    out["anchor"] = (np.round(anchor, 3).tolist(), ng.anchor_rejects, gates["old"].rejects, round(drift * 100, 2))
    out["start_d"] = out["rows"][0]["d_new"] if out["rows"] else None
    far = next((r for r in out["rows"] if r["d_new"] > AG.ApproachGraspCfg().far_abort_m), None)
    out["far"] = None if far is None else (far["t"], far["d_new"])
    out["max_d"] = max(r["d_new"] for r in out["rows"])
    return out


def main() -> None:
    print("# 11.03 넘김 · FP++ 기준점 거르기 오프라인 재생(개루프 — 넘김 뒤는 의미 없음)\n")
    tol = AG.ApproachGraspCfg().ready_tol_m * 100
    print(f"넘김 허용 {tol:g} cm · 창 평균 멈춤 · 손바닥-컵 상한 {AG.ApproachGraspCfg().far_abort_m} m\n")
    print("| 에피소드 | 실기 넘김 · 끝 | 새 규칙 + 옛 거르기: 넘김 · 오차 · 창 m/s | 새 규칙 + 기준점 거르기: 넘김 | 그때 오차 | 창 평균 m/s | 시작 손바닥-컵 | 최대 손바닥-컵 | 0.35 m 넘은 때 | 기준점 · 기준점 거름 · 옛 거름 · 거른 컵 최대 이탈 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    results = [replay(*cs) for cs in CASES]
    for r in results:
        new = r["new"]
        far = "-" if r["far"] is None else f"{r['far'][0]} s ({r['far'][1]} m)"
        old = "-" if r["old"] is None else f"{r['old'][0]} s ({r['old'][1]}) · {r['old'][2]} cm · {r['old'][3]}"
        print(f"| {r['label']} | {r['real']} | {old} | {'-' if new is None else f'{new[0]} s ({new[1]})'} | "
              f"{'-' if new is None else f'{new[2]} cm'} | {'-' if new is None else new[3]} | {r['start_d']} m | {r['max_d']} m | "
              f"{far} | "
              f"{r['anchor'][0]} · {r['anchor'][1]} · {r['anchor'][2]} · {r['anchor'][3]} cm |")
    for r in results:
        rows = r["rows"]
        fid = [abs(x["real_off"] - x["rep_off_old"]) for x in rows if x["real_off"] is not None]
        print(f"\n## {r['label']} — 재생 충실도: 실기 상태 오차 대비 옛 거르기 재생 오차 차이 중앙 {np.median(fid):.2f} cm · "
              f"최대 {max(fid):.2f} cm (n={len(fid)})\n")
        print("| t s | 실기 오차 cm | 재생 옛 cm | 재생 새 cm | 스냅숏 대비 cm | 높이 손바닥−컵 cm(live · 스냅숏) | 손바닥-컵(새) m | 실기 한 틱 m/s | 창 평균 m/s | live | 거른 컵(새) | 손바닥 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        last = -1.0
        for x in rows:
            if x["t"] - last < 0.5:
                continue
            last = x["t"]
            print(f"| {x['t']} | {x['real_off']} | {x['rep_off_old']} | {x['rep_off_new']} | {x['off_snap']} | {x['dz_old']} · {x['dz_snap']} | {x['d_new']} | "
                  f"{x['real_mps']} | {x['win_mps']} | {x['live']} | {x['cup_new']} | {x['palm']} |")


if __name__ == "__main__":
    main()
