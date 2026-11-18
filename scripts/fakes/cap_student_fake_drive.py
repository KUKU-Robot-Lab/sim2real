#!/usr/bin/env python3
"""cap 깊이 학생 노드 fake 구동기(11.18) — run_cap_student_fake.sh 가 부른다. 도메인 97 전용.

체인 한 번(reset → start → 학생 구간은 stage/next · 스크립트 구간은 자동 대기) 뒤 거부 경로 셋(stop · 쥔 손 start · 깊이 끊김)을 돌리고
status · joint_target · /head/target_stream · episode · 서비스 응답을 <out>/events.jsonl 에 남긴 뒤 구간별 요약 <out>/summary.json 을 쓴다.
시작 자세 이동은 pd 대신 /policy_control/joint_target 을 fake plant 에 직접 낸다(= pd 가 붙든 마지막 목표 흉내, 노드 _held 가 읽는다).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "policy_control"))
from policy_control.rh_cap import CAPTOP_GRIP, CAPTOP_OPEN  # noqa: E402
from policy_control.ros_domain import REAL_DOMAIN  # noqa: E402
from policy_control.rh_cap_student_node import LIVE_ARM_START_Q, PHASES, SCRIPTED  # noqa: E402

ARM = [f"r_aj_{i}" for i in range(1, 8)]
HAND = ["r_hj_thumb_1", "r_hj_thumb_2", "r_hj_index_1", "r_hj_middle_1", "r_hj_ring_1", "r_hj_pinky_1"]
STUDENT_DWELL_S = 3.0           # 학생 구간에서 stage/next 전에 기다리는 시간
SCRIPT_TIMEOUT_S = {"descend": 8.0, "carry": 25.0}   # 하강 150 스텝 · 이송 900 + 60 스텝(60 Hz) 에 여유


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", default="rh_cap_node_right")
    ap.add_argument("--ns", default="right_cap")
    ap.add_argument("--episode-topic", default="/policy_control/right/episode")
    ap.add_argument("--depth-pid", type=int, default=0, help="깊이 끊김 시험에서 SIGTERM 할 fake 깊이 발행기 PID(이 스크립트가 띄운 것)")
    ap.add_argument("--quick", action="store_true", help="reset → start → search 한 구간만 보고 stop(깊이 config 대조 확인용)")
    args = ap.parse_args()
    dom = os.environ.get("ROS_DOMAIN_ID", "").strip()
    if not dom.isdecimal() or int(dom) in (0, REAL_DOMAIN):   # "050" 도 실기 50
        raise SystemExit(f"fake 도메인에서만(지금 ROS_DOMAIN_ID={dom!r})")

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    out = Path(args.out)
    log = (out / "events.jsonl").open("w", encoding="utf-8")
    lock = threading.Lock()
    st = {"status": None, "hand": None}

    def rec(kind: str, **kw) -> None:
        with lock:
            log.write(json.dumps({"t": time.monotonic(), "kind": kind, **kw}, ensure_ascii=False) + "\n")

    rclpy.init()
    node = Node("cap_student_fake_drive")
    base = f"/policy_control/{args.ns}"

    def on_status(m):
        b = json.loads(m.data)
        st["status"] = b
        cs = b.get("cap_student") or {}
        rec("status", phase=b.get("phase"), ok=b.get("ok"), reasons=b.get("reasons"), cap=cs.get("phase"), stage=cs.get("stage"),
            step=cs.get("step"), phase_steps=cs.get("phase_steps"), note=cs.get("note"), carry_wait=cs.get("carry_wait"),
            scripts=cs.get("scripts"), head_cmd=cs.get("head_cmd"), proc_ms=b.get("proc_ms"), tick=b.get("tick"), gap=b.get("gap"),
            poses_start=b.get("poses_start"))

    def on_target(m):
        d = dict(zip(m.name, m.position))
        rec("target", arm=[round(d.get(j, float("nan")), 4) for j in ARM], hand=[round(d.get(j, float("nan")), 4) for j in HAND])

    def on_hand(m):
        d = dict(zip(m.name, m.position))
        st["hand"] = [d.get(j) for j in HAND]

    node.create_subscription(String, f"/policy_control/status/{args.node}", on_status, 50)
    node.create_subscription(JointState, "/policy_control/joint_target", on_target, 200)
    node.create_subscription(String, "/head/target_stream", lambda m: rec("head_stream", body=m.data), 200)
    node.create_subscription(String, args.episode_topic, lambda m: rec("episode", body=m.data), 10)
    node.create_subscription(JointState, "/hand_right/joint_states", on_hand, 10)
    pub = node.create_publisher(JointState, "/policy_control/joint_target", 10)
    cli = {n: node.create_client(Trigger, f"{base}/{n}") for n in ("episode/reset", "episode/start", "episode/stop", "stage/next")}
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    def call(name: str, timeout: float = 60.0) -> dict:
        c = cli[name]
        if not c.wait_for_service(timeout_sec=timeout):
            r = {"ok": False, "msg": "service 없음"}
        else:
            f = c.call_async(Trigger.Request())
            t0 = time.monotonic()
            while not f.done() and time.monotonic() - t0 < timeout:
                time.sleep(0.02)
            r = {"ok": bool(f.result().success), "msg": f.result().message} if f.done() else {"ok": False, "msg": "응답 시간 초과"}
        rec("call", name=name, **r)
        print(f"[{name}] {r}", flush=True)
        return r

    def home(hand, secs: float = 2.0) -> None:
        """pd 시작 자세 이동 흉내 — 팔 LIVE_ARM_START_Q · 손 hand 를 fake plant 에 낸다(노드가 idle 일 때만)."""
        rec("home", hand=list(hand))
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            m = JointState()
            m.header.stamp = node.get_clock().now().to_msg()
            m.name, m.position = ARM + HAND, [float(v) for v in (*LIVE_ARM_START_Q, *hand)]
            pub.publish(m)
            time.sleep(0.02)
        time.sleep(0.5)

    def cap_phase():
        s = st["status"] or {}
        return s.get("phase"), (s.get("cap_student") or {}).get("phase"), (s.get("cap_student") or {}).get("note")

    def wait_until(pred, secs: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            if pred():
                return True
            time.sleep(0.05)
        return False

    # 1. 체인 끝까지 -------------------------------------------------------------
    t_wait = time.monotonic()
    wait_until(lambda: st["status"] is not None and st["hand"] is not None, 120.0)
    rec("mark", name="chain", node_up_s=round(time.monotonic() - t_wait, 2))
    home(CAPTOP_OPEN)
    if args.quick:
        ok = call("episode/reset", 90.0)["ok"]
        rec("mark", name="quick_reset", ok=ok)
        if ok:
            ok = call("episode/start")["ok"]
            rec("mark", name="quick_start", ok=ok)
        if ok:
            rec("mark", name="quick_search", ok=wait_until(lambda: cap_phase()[:2] == ("running", "search"), 5.0))
            time.sleep(2.0)
            call("episode/stop")
        time.sleep(0.5)
        ex.shutdown()
        rclpy.try_shutdown()      # 데몬 spin 스레드가 남은 채 끝나면 종료 때 core dump(11.18 183346)
        log.close()
        return 0 if ok else 1
    if args.depth_pid <= 0:
        raise SystemExit("체인 전체 실행은 --depth-pid 가 필요하다")
    if call("episode/reset", 90.0)["ok"] and call("episode/start")["ok"]:
        for ph in PHASES:
            if not wait_until(lambda: cap_phase()[0] == "running" and cap_phase()[1] == ph, 5.0):
                rec("mark", name="chain_stop", why=f"{ph} 에 들어가지 않음 · 지금 {cap_phase()}")
                break
            rec("mark", name="enter", phase=ph)
            if ph in SCRIPTED:
                left = wait_until(lambda: cap_phase()[1] != ph or cap_phase()[0] != "running" or cap_phase()[2], SCRIPT_TIMEOUT_S[ph])
                if cap_phase()[0] != "running":
                    rec("mark", name="chain_stop", why=f"{ph} 중 에피소드 끝 · {cap_phase()}")
                    break
                if cap_phase()[1] == ph:      # 자동 전환 없음 — 미도달 note 든 시간 초과든 stage/next 를 한 번 불러 본다
                    rec("mark", name="script_no_auto", phase=ph, note=cap_phase()[2], timed_out=not left)
                    if not call("stage/next", 10.0)["ok"]:
                        rec("mark", name="chain_stop", why=f"{ph} 자동 전환 없음 · stage/next 거부")
                        break
                continue
            time.sleep(STUDENT_DWELL_S)
            if cap_phase()[0] != "running":
                rec("mark", name="chain_stop", why=f"{ph} 중 에피소드 끝 · {cap_phase()}")
                break
            if not call("stage/next", 10.0)["ok"]:
                rec("mark", name="chain_stop", why=f"{ph} stage/next 거부")
                break
        time.sleep(1.0)
        if cap_phase()[0] == "running":
            call("episode/stop", 10.0)
    rec("mark", name="chain_end", final=list(cap_phase()))
    time.sleep(1.0)

    # 2. episode stop -------------------------------------------------------------
    rec("mark", name="test_stop")
    home(CAPTOP_OPEN)
    if call("episode/reset", 30.0)["ok"] and call("episode/start")["ok"]:
        time.sleep(2.0)
        call("episode/stop", 10.0)
        time.sleep(1.0)
    rec("mark", name="test_stop_end", final=list(cap_phase()))

    # 4. 깊이 끊김 ---------------------------------------------------------------------
    rec("mark", name="test_depth_cut")
    home(CAPTOP_OPEN, 3.0)
    if call("episode/reset", 30.0)["ok"] and call("episode/start")["ok"]:
        time.sleep(0.5)                                 # 학생 틱이 도는 중에 끊는다
        os.kill(args.depth_pid, signal.SIGTERM)        # 이 실행이 띄운 fake 깊이 발행기 PID 만
        rec("mark", name="depth_sigterm", pid=args.depth_pid)
        wait_until(lambda: cap_phase()[0] != "running", 10.0)
        time.sleep(1.0)
        if cap_phase()[0] == "running":
            call("episode/stop", 10.0)
    rec("mark", name="test_depth_cut_end", final=list(cap_phase()))
    call("episode/reset", 30.0)                          # 깊이 없이 reset → 깊이 대기 뒤 거부여야 한다
    rec("mark", name="test_depth_reset_end", final=list(cap_phase()))
    # 3. 쥔 손으로 start(마지막 — fake plant 는 한 번 쥐면 놓지 않는다). 깊이는 4 에서 끊었으니 이 시험 동안만 새로 띄운다
    rec("mark", name="test_hand_closed")
    dp = subprocess.Popen(["nice", "-n", "15", "python3", str(Path(__file__).with_name("fake_head_depth_pub.py")),
                           "--depth", os.environ.get("DEPTH_M", "0.6")], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rec("mark", name="depth2_pid", pid=dp.pid)
    try:
        time.sleep(2.0)
        home(CAPTOP_GRIP, 3.0)
        rec("mark", name="hand_meas", hand=st["hand"])
        if call("episode/reset", 30.0)["ok"]:
            if call("episode/start")["ok"]:
                time.sleep(1.0)
                call("episode/stop", 10.0)
        rec("mark", name="test_hand_closed_end", final=list(cap_phase()))
    finally:
        dp.terminate()                                 # 이 스크립트가 띄운 PID 에만 SIGTERM
        try:
            dp.wait(10.0)
        except subprocess.TimeoutExpired:
            rec("mark", name="depth2_left", pid=dp.pid)

    time.sleep(0.5)
    ex.shutdown()
    log.close()
    summarize(out)
    return 0


def summarize(out: Path) -> None:
    """events.jsonl → 구간별 표(joint_target Hz · 첫/마지막 · 머리 스트림 수 · proc_over · 끝난 이유)."""
    ev = [json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    marks = {e["name"]: e["t"] for e in ev if e["kind"] == "mark" and e["name"] in ("chain", "chain_end")}
    chain = [e for e in ev if marks.get("chain", 0) <= e["t"] <= marks.get("chain_end", float("inf"))]
    segs, cur = [], None
    for e in chain:
        if e["kind"] == "status" and e.get("phase") == "running" and e.get("cap") and (cur is None or cur["phase"] != e["cap"]):
            cur = {"phase": e["cap"], "stage": e["stage"], "t0": e["t"], "t1": e["t"], "targets": [], "head": 0, "status": []}
            segs.append(cur)
        if cur is None:
            continue
        if e["kind"] == "status":
            if e.get("phase") == "running" and e.get("cap") == cur["phase"]:
                cur["t1"] = e["t"]
            cur["status"].append(e)
        elif e["kind"] == "target":
            cur["targets"].append(e)
        elif e["kind"] == "head_stream":
            cur["head"] += 1
    rows = []
    for s in segs:
        tg, run = s["targets"], [x for x in s["status"] if x.get("phase") == "running"]
        dur = max(s["t1"] - s["t0"], 1e-6)
        last = s["status"][-1] if s["status"] else {}
        rows.append({"phase": s["phase"], "stage": s["stage"], "dur_s": round(dur, 2), "n_target": len(tg),
                     "target_hz": round(len(tg) / dur, 1), "first": tg[0] if tg else None, "last": tg[-1] if tg else None,
                     "head_stream": s["head"], "head_cmd_last": run[-1].get("head_cmd") if run else None,
                     "steps": (run[-1].get("phase_steps") if run else None), "tick_last": run[-1].get("tick") if run else None,
                     "scripts_last": run[-1].get("scripts") if run else None, "note_last": run[-1].get("note") if run else None,
                     "reasons_last": last.get("reasons"), "end_phase": last.get("phase")})
    calls = [e for e in ev if e["kind"] in ("call", "mark", "episode")]
    (out / "summary.json").write_text(json.dumps({"segments": rows, "calls_marks": calls}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
