#!/usr/bin/env python3
"""실기 전 점검(읽기만) — 실행기 · 머리 카메라 · 목 · FP++ 물체 · VLM 한도. 로봇을 움직이지 않는다."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import String
from vlm_backends import Router

rclpy.init(); n = rclpy.create_node("vlm_preflight")
got = {"status": None, "img": [], "head": None}
n.create_subscription(String, "/policy_control/status/episode_runner", lambda m: got.update(status=json.loads(m.data)), 10)
n.create_subscription(CompressedImage, "/camera/camera/color/image_raw/compressed", lambda m: got["img"].append(time.monotonic()), qos_profile_sensor_data)
n.create_subscription(JointState, "/head/joint_states", lambda m: got.update(head=dict(zip(m.name, m.position))), qos_profile_sensor_data)
t0 = time.monotonic()
while time.monotonic() - t0 < 4.0:
    rclpy.spin_once(n, timeout_sec=0.1)
topics = {t for t, _ in n.get_topic_names_and_types()}
bad = 0
def row(ok, what, detail=""):
    global bad
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {what:28s} {detail}")
print(f"[preflight] ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')}")
st = got["status"]
row(st is not None, "실행기 상태", "" if st is None else f"{st.get('episode')} · {st.get('status')} · 다음 {st.get('next_action')} · index {st.get('index')}")
if st is not None:
    row(st.get("episode") == "cocktail_1c_e4000", "에피소드 = cocktail_1c_e4000", str(st.get("episode")))
    row(st.get("status") == "READY" and not st.get("busy"), "실행기 READY · 대기", f"{st.get('status')} busy={st.get('busy')}")
    row(st.get("index") == 0, "처음부터(go_home_start)", f"index {st.get('index')} — 중간이면 reset 뒤 다시")
hz = (len(got["img"]) - 1) / (got["img"][-1] - got["img"][0]) if len(got["img"]) > 2 else 0
row(hz > 3, "머리 카메라 compressed", f"{hz:.1f} Hz")
row(got["head"] is not None, "목 /head/joint_states", "" if got["head"] is None else json.dumps({k: round(v, 3) for k, v in got["head"].items()}))
objs = sorted(t for t in topics if t.startswith("/objects/") and t.endswith("live_pose"))
row(len(objs) >= 3, "FP++ 물체 live_pose", ", ".join(o.split("/")[2] for o in objs) or "없음 — cups 단계")
r = Router()
for t in r.ledger.table(r.chain):
    ok = t["blocked_until"] is None and (t["remaining"] is None or t["remaining"] > 0)
    row(ok, f"VLM {t['model']}", f"오늘 {t['used_today']}회 · 남음 {t['remaining'] if t['remaining'] is not None else '∞'}"
        + (f" · {t['blocked_until']}까지 막힘" if t["blocked_until"] else "") + f" · 토큰 {t['tokens_in/out/think']}")
    if t["model"].startswith("local"):
        alive = r.backends[t["model"]].alive()
        row(alive, f"  {t['model']} 서버", r.backends[t["model"]].url)
print(f"[preflight] {'통과' if bad == 0 else f'✗ {bad}개 확인 필요'} (VLM 은 하나만 살아 있어도 진행 가능)")
