#!/usr/bin/env python3
"""시험용 가짜 episode_runner — 실기와 같은 상태 JSON · next/stop 서비스 · 승인 파일 검사를 흉내 내고,
녹화된 실기 프레임(015557, place_blue · place_pink 실패 포함)을 노드가 끝날 때마다 카메라 토픽으로 낸다.
로봇은 없다. 반드시 ROS_LOCALHOST_ONLY=1 · 실기와 다른 도메인에서 띄운다(sup.sh test 가 77 로 띄움)."""
import json, os, sys, threading, time
from pathlib import Path
import rclpy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger

FR = Path(sys.argv[1]); APPR = Path(sys.argv[2])
NODES = [("go_home_start", "trajectory"), ("scene", "snapshot"), ("grasp_to_tilt_start", "lanes"), ("tilt_blue", "policy"),
         ("place_blue", "policy"), ("home_r_after_blue", "trajectory"), ("grasp_pink", "policy"), ("to_tilt_start_r2", "policy"),
         ("tilt_pink", "policy"), ("place_pink", "policy"), ("place_shaker", "policy"), ("go_home_end", "trajectory"), ("done", "terminal")]
files = sorted(p.name for p in FR.glob("*.jpg"))
def frame_for(nid):
    for pre in ("failure_", "exit_"):
        hit = [f for f in files if f"_{pre}{nid}." in f]
        if hit: return hit[0], pre == "failure_"
    return files[-1], False

rclpy.init(); n = rclpy.create_node("episode_runner")
pub = n.create_publisher(String, "/policy_control/status/episode_runner", 10)
ipub = n.create_publisher(CompressedImage, "/camera/camera/color/image_raw/compressed", 2)
S = {"i": 0, "status": "READY", "busy": False, "last": {}, "img": [f for f in files if "enter_go_home_start" in f][0], "attempts": {}}
def view():
    i = S["i"]
    return {"episode": "cocktail_1c_e4000", "status": S["status"], "busy": S["busy"], "phase": S["status"], "index": i,
            "node": NODES[i][0] if i < len(NODES) else None,
            "nodes": [{"id": a, "type": b, "state": "done" if k < i else ("current" if k == i else "pending")} for k, (a, b) in enumerate(NODES)],
            "last": S["last"], "pending": None, "attempts": S["attempts"],
            "next_action": NODES[i][0] if S["status"] == "READY" and i < len(NODES) - 1 else None}
def tick():
    pub.publish(String(data=json.dumps(view())))
    m = CompressedImage(); m.format = "jpeg"; m.data = (FR / S["img"]).read_bytes(); ipub.publish(m)
n.create_timer(0.2, tick)
def approved(what):
    try: rows = [json.loads(l) for l in APPR.read_text().splitlines()[-5:]]
    except FileNotFoundError: return False
    return any(r["what"] == what and time.time() - r["t"] < 30 for r in rows)
def work(nid, typ):
    time.sleep(1.5)
    f, failed = frame_for(nid); S["img"] = f
    S["attempts"][nid] = S["attempts"].get(nid, 0) + 1
    S["last"] = {"node": nid, "code": "PLACE_FAILED" if failed else "done", "reason": "holder latch not detected" if failed else ""}
    S["i"] += 1; S["busy"] = False
    if NODES[S["i"]][1] == "terminal": S["status"] = "SUCCESS"; S["last"] = {"node": "done", "code": "SUCCESS"}
    print(f"[fake] {nid} → {S['last']['code']} ({f})", flush=True)
def srv_next(req, res):
    v = view()
    if S["busy"] or v["next_action"] is None: res.success = False; res.message = json.dumps({"ok": False, "reasons": ["busy/none"]}); return res
    if not approved(v["next_action"]): res.success = False; res.message = json.dumps({"ok": False, "reasons": ["승인 없음"]}); return res
    S["busy"] = True; threading.Thread(target=work, args=NODES[S["i"]], daemon=True).start()
    res.success = True; res.message = json.dumps({"ok": True, "reasons": [f"started {v['next_action']}"]}); return res
def srv_stop(req, res):
    S["status"] = "STOPPED"; res.success = True; res.message = json.dumps({"ok": True, "reasons": ["stopped"]}); print("[fake] STOP", flush=True); return res
n.create_service(Trigger, "/policy_control/episode_runner/next", srv_next)
n.create_service(Trigger, "/policy_control/episode_runner/stop", srv_stop)
print("[fake] up", len(files), "frames", flush=True)
try: rclpy.spin(n)
except KeyboardInterrupt: pass
