#!/usr/bin/env python3
"""VLM 감독 — episode_runner 의 노드가 끝날 때마다 머리 카메라를 보고 VLM 이 성공 판정 · 다음 행동을 고른다.

  mode shadow : 판단만 기록. 실행기 승인은 사람이 콘솔에서 평소대로(로봇에 영향 없음)
  mode active : VLM 이 승인(next) · 복구 승인(recover:*) · 멈춤(stop)을 직접 부른다

순서는 실행기 YAML(cocktail_1c_e4000) 그대로(11.10 사용자: "현재 스킬 순서는 그대로"). VLM 은 처음에 한 문장으로 계획을
한 번 세우고(기록 · 대조용), 매 판정 노드 뒤에 A~E 를 고른다.
  A 성공 → 다음 노드 승인      B/C/D 복구 → 실행기에 복구 대기가 있으면 그 복구 승인, 없으면 멈춤(원한 복구는 기록)
  E 위험 · 불명 → 멈춤          VLM 전부 실패 · 호출 예산 초과 → 멈춤
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_backends import ROOT, Router  # noqa: E402

SENT = "블루컵과 핑크컵에 있는 비드를 오렌지 쉐이커 컵에 옮겨줘."
CAT = """Robot: two arms with black robotic hands. Objects on the table:
- SRC_BLUE = blue cup with beads (right hand only, home holder HOLDER_1)
- SRC_PINK = pink cup with beads (right hand only, home holder HOLDER_2)
- SHAKER = orange shaker cup (left hand only, home holder HOLDER_0)
Skills (each step = [skill, hand, object, target]):
- go_home [go_home, both|right, -, -]: move empty arm(s) to home. A hand holding the SHAKER cannot go home; use right only then.
- scene_snapshot [scene_snapshot, -, -, -]: locate all cups once at the start.
- grasp [grasp, right|left, OBJ, -]: empty hand approaches and lifts OBJ.
- to_tilt_start [to_tilt_start, right|left, OBJ, -]: carry the held OBJ to the pouring start pose. Both arms must do this before tilt.
- tilt [tilt, both, SRC_BLUE|SRC_PINK, SHAKER]: right hand pours its cup into the SHAKER held by the left hand, then returns.
- place [place, right|left, OBJ, HOLDER_n]: put the held OBJ into its holder and open the hand.
- terminal [terminal, -, -, SUCCESS]: end.
Rules: one cup per hand. A cup must be held to be carried, poured or placed. Keep the shaker in the left hand until all pouring is done."""
# 실행기 노드 id → VLM 에 보여 줄 단계(cocktail_1c_e4000.yaml). 여기 없는 노드는 판정 없이 넘긴다(rehome · snapshot 은 실행기 검사로 충분).
STEP = {
    "grasp_to_tilt_start": ('[["grasp","right","SRC_BLUE"],["to_tilt_start","right","SRC_BLUE"],["grasp","left","SHAKER"],["to_tilt_start","left","SHAKER"]]',
                            "right hand holds the blue cup and left hand holds the orange shaker, both lifted to the pouring start pose"),
    "tilt_blue": ('["tilt","both","SRC_BLUE","SHAKER"]', "beads of the blue cup were poured into the shaker; both cups still held upright"),
    "place_blue": ('["place","right","SRC_BLUE","HOLDER_1"]', "blue cup stands in its holder and the right hand is open/away"),
    "grasp_pink": ('["grasp","right","SRC_PINK"]', "right hand holds the pink cup lifted"),
    "to_tilt_start_r2": ('["to_tilt_start","right","SRC_PINK"]', "right hand holds the pink cup at the pouring start pose"),
    "tilt_pink": ('["tilt","both","SRC_PINK","SHAKER"]', "beads of the pink cup were poured into the shaker; both cups still held upright"),
    "place_pink": ('["place","right","SRC_PINK","HOLDER_2"]', "pink cup stands in its holder and the right hand is open/away"),
    "place_shaker": ('["place","left","SHAKER","HOLDER_0"]', "orange shaker stands in its holder and the left hand is open/away"),
}
DEFAULT_JUDGE = "grasp_to_tilt_start,tilt_blue,place_blue,to_tilt_start_r2,tilt_pink,place_pink,place_shaker"
OPTS = """A. The step succeeded: continue with the next step of the plan.
B. The step failed but the hand still holds the object: retry the same step.
C. The grasp failed (object not held): move that arm home, take a new scene snapshot, grasp again.
D. The held cup was lost or is not in the hand anymore: take a new scene snapshot and grasp it again before continuing.
E. Unsafe or unclear (human in the scene, object fallen over, cannot tell): stop and call the human."""
END = ("SUCCESS", "FAILURE", "STOPPED")


def parse_json(txt: str):
    txt = txt.split("</think>")[-1]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Sup:
    def __init__(self, a):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        self.a, self.rclpy, self.Trigger = a, rclpy, Trigger
        rclpy.init()
        self.node = rclpy.create_node("vlm_supervisor")
        self.lock = threading.Lock()
        self.st: dict = {}
        self.st_t = 0.0
        self.img: bytes | None = None
        self.img_t = 0.0
        self.node.create_subscription(String, f"/policy_control/status/{a.runner}", self._on_status, 10)
        self.node.create_subscription(CompressedImage, a.image_topic, self._on_img, 2)
        self.cli = {k: self.node.create_client(Trigger, f"/policy_control/{a.runner}/{k}") for k in ("next", "stop")}
        self.ex = MultiThreadedExecutor(num_threads=3)
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()
        self.router = Router()
        self.run_dir = Path(ROOT) / "runs" / time.strftime("%y%m%d_%H%M%S")
        (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.log_fh = open(self.run_dir / "decisions.jsonl", "a")
        self.calls = 0
        self.hist: list = []
        self.start_img: Image.Image | None = None
        self.plan = None

    # ---------------- ROS
    def _on_status(self, m):
        with self.lock:
            self.st = json.loads(m.data)
            self.st_t = time.monotonic()

    def _on_img(self, m):
        with self.lock:
            self.img = bytes(m.data)
            self.img_t = time.monotonic()

    def status(self) -> dict:
        with self.lock:
            return dict(self.st)

    def frame(self, after: float = 0.0, wait: float = 3.0) -> Image.Image | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait:
            with self.lock:
                if self.img is not None and self.img_t >= after:
                    return Image.open(io.BytesIO(self.img)).convert("RGB")
            time.sleep(0.05)
        return None

    def call(self, what: str, approve: str | None = None) -> tuple[bool, list]:
        if self.a.mode != "active":
            return True, ["shadow — 부르지 않음"]
        if approve:                                             # 실행기는 승인 파일 줄을 보고 움직인다(episode_cmd 와 같은 통로)
            write_approval(approve, "vlm_supervisor", self.a.approvals, runner=self.a.runner)
        cli = self.cli[what]
        if not cli.wait_for_service(timeout_sec=3.0):
            return False, [f"{what} 서비스 없음"]
        fut = cli.call_async(self.Trigger.Request())
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < 8:
            time.sleep(0.05)
        if not fut.done():
            return False, ["응답 없음"]
        res = fut.result()
        body = json.loads(res.message) if res.message.startswith("{") else {"ok": res.success, "reasons": [res.message]}
        return bool(body.get("ok")), list(body.get("reasons", []))

    # ---------------- 기록
    def log(self, **kw):
        kw["t"] = time.strftime("%F %T")
        self.log_fh.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.log_fh.flush()
        brief = {k: v for k, v in kw.items() if k not in ("raw", "plan")}
        print(f"[vlm] {json.dumps(brief, ensure_ascii=False)}", flush=True)

    def save(self, img: Image.Image, name: str) -> str:
        p = self.run_dir / "frames" / f"{len(list((self.run_dir / 'frames').iterdir())):03d}_{name}.jpg"
        img.save(p, quality=85)
        return p.name

    # ---------------- VLM
    def ask(self, images, text):
        if self.calls >= self.a.max_calls:
            raise RuntimeError(f"이번 실행 호출 예산 {self.a.max_calls} 소진")
        self.calls += 1
        return self.router.ask(images, text, self.a.level)

    def make_plan(self, img):
        q = (CAT + f'\n\nUser instruction (Korean): "{self.a.sentence}"\nLook at the image of the table. Write the full step list '
             'to complete the instruction, from start to finish. Each step is a list of 4 strings [skill, hand, object, target]. '
             'Answer with only the JSON list.')
        try:
            txt, meta = self.ask([img], q)
        except Exception as e:                                  # noqa: BLE001 — 계획은 기록용, 실패해도 진행
            self.log(ev="plan_error", err=str(e))
            return
        m = re.search(r"\[.*\]", txt.split("</think>")[-1], re.S)
        try:
            self.plan = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            self.plan = None
        self.log(ev="plan", model=meta["model"], sec=meta["sec"], n_steps=len(self.plan or []), plan=self.plan, raw=txt[-600:])

    def judge(self, node_id: str, last: dict, img: Image.Image) -> dict:
        step, goal = STEP[node_id]
        code = str(last.get("code") or "")
        ok_sig = code.upper() in ("", "DONE", "COMPLETED", "SUCCESS", "OK")
        sig = "the controller reported the step DONE" if ok_sig else f"the controller reported the step FAILED ({code}: {str(last.get('reason',''))[:120]})"
        q = (CAT + f'\n\nUser instruction (Korean): "{self.a.sentence}"\n'
             + (f"Your plan: {json.dumps(self.plan)}\n" if self.plan else "")
             + f'Steps finished so far and your judgements: {json.dumps(self.hist) if self.hist else "none"}\n'
             f"The robot just finished step {step}; {sig}. The controller signal can be wrong, trust what you see.\n"
             f"Goal of this step: {goal}.\n"
             "Image 1 = table at the start of the episode. Image 2 = now.\n"
             f"Did step {step} achieve its goal? Then choose what to do next:\n{OPTS}\n"
             'Answer with only JSON: {"success": true|false, "choice": "A".."E", "reason": "<one short sentence>"}')
        imgs = [self.start_img, img] if self.start_img is not None else [img]
        txt, meta = self.ask(imgs, q)
        a = parse_json(txt) or {}
        ch = str(a.get("choice", "?"))[:1].upper()
        if ch not in "ABCDE":
            ch = "E"                                            # 못 읽으면 위험 쪽으로
        out = {"success": a.get("success"), "choice": ch, "reason": a.get("reason", txt[-160:]), "model": meta["model"],
               "sec": meta["sec"], "tok": meta.get("in_tok", 0) + meta.get("out_tok", 0) + meta.get("think_tok", 0),
               "fallbacks": meta.get("fallbacks", []), "ctrl_ok": ok_sig}
        self.hist.append([node_id, "success" if a.get("success") else "failed", ch])
        return out

    # ---------------- 본 루프
    def wait_status(self, timeout=10.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.status():
                return True
            time.sleep(0.1)
        return False

    def stop(self, why: str):
        ok, rs = self.call("stop")
        self.log(ev="stop", why=why, ok=ok, reasons=rs)

    def run(self) -> int:
        a = self.a
        print(f"[vlm] mode={a.mode} runner={a.runner} judge={sorted(a.judge)} 남은 호출: {self.router.calls_left()}")
        print(f"[vlm] 기록: {self.run_dir}")
        if not self.wait_status():
            print("[vlm] ✗ 실행기 상태가 안 온다 — 콘솔에서 episode_cocktail_1c_e4000 노드를 먼저 띄운다")
            return 2
        img = self.frame(wait=5.0)
        if img is None:
            print(f"[vlm] ✗ 카메라 {a.image_topic} 가 안 온다 — 4090 camera_up 확인")
            return 2
        self.start_img = img
        self.save(img, "start")
        if a.plan:
            self.make_plan(img)
        if a.mode == "active" and not a.yes:                    # 첫 동작은 사람이 한 번 시작(그 뒤는 VLM)
            st = self.status()
            input(f"[vlm] 실행기 {st.get('episode')} · 다음 {st.get('next_action')} — Enter 를 누르면 VLM 이 진행한다 (Ctrl-C 취소) ")
        seen = None
        stop_at = None
        while True:
            st = self.status()
            phase = st.get("status")
            if phase in END:
                self.log(ev="end", status=phase, last=st.get("last"), calls=self.calls, left=self.router.calls_left())
                return 0 if phase == "SUCCESS" else 1
            if st.get("busy") or phase != "READY":
                time.sleep(0.2)
                continue
            last = st.get("last") or {}
            key = json.dumps([last, st.get("attempts")], sort_keys=True, default=str)
            if key == seen:                                     # 이미 처리한 상태 — shadow 는 사람이 다음을 누를 때까지 대기
                time.sleep(0.2)
                continue
            seen = key
            nid, nxt, pend = last.get("node"), st.get("next_action"), st.get("pending")
            decision = {"choice": "A", "reason": "판정 안 하는 노드"}
            if nid in self.a.judge:
                t_end = time.monotonic()
                time.sleep(a.settle)
                img = self.frame(after=t_end + a.settle - 0.05)
                if img is None:
                    decision = {"choice": "E", "reason": "카메라 프레임 없음"}
                else:
                    fname = self.save(img, f"after_{nid}")
                    try:
                        decision = self.judge(nid, last, img)
                    except Exception as e:                      # noqa: BLE001
                        decision = {"choice": "E", "reason": f"VLM 실패: {e}"}
                    decision["frame"] = fname
            ch = decision["choice"]
            if ch == "A" and nxt and not str(nxt).startswith("recover:"):
                act, approve = "next", nxt
            elif ch in "BCD" and pend and str(nxt).startswith("recover:"):
                act, approve = "next", nxt                       # 실행기 자체 복구를 VLM 이 승인
            elif ch == "A" and nxt is None:
                act, approve = "wait", None                      # terminal 직전 등 — 실행기가 스스로 닫는다
            else:
                act, approve = "stop", None
            self.log(ev="decide", node=nid, last_code=last.get("code"), next_action=nxt, pending=pend, act=act, **decision)
            if act == "next":
                if a.mode == "active":
                    ok, rs = self.call("next", approve)
                    self.log(ev="next", approve=approve, ok=ok, reasons=rs)
                    if not ok:
                        self.stop(f"next 거부: {rs}")
                        return 1
                else:
                    print(f"[vlm] (shadow) 사람이 '{approve}' 를 누르기를 기다림")
            elif act == "stop":
                if a.mode == "active":
                    self.stop(f"VLM {ch}: {decision.get('reason')}")
                    return 1
                print(f"[vlm] (shadow) VLM 이라면 여기서 멈춤: {decision.get('reason')}")
                stop_at = stop_at or nid
            time.sleep(0.2)

    def close(self):
        try:
            self.ex.shutdown()
            self.node.destroy_node()
            self.rclpy.try_shutdown()
        except Exception:                                       # noqa: BLE001
            pass


def write_approval(what, operator, path, runner):
    row = {"what": what, "t": time.time(), "operator": operator, "at": time.strftime("%F %T"), "runner": runner}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("shadow", "active"), default="shadow")
    ap.add_argument("--runner", default="episode_runner")
    ap.add_argument("--sentence", default=SENT)
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw/compressed")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="판정할 노드 id (쉼표)")
    ap.add_argument("--no-plan", dest="plan", action="store_false")
    ap.add_argument("--level", default=os.environ.get("GLEVEL", "low"), choices=("minimal", "low", "medium", "high"))
    ap.add_argument("--max-calls", type=int, default=12, help="이번 실행 VLM 호출 상한(계획 1 + 판정 7 = 8 이 기본 소비)")
    ap.add_argument("--settle", type=float, default=0.5, help="노드 끝난 뒤 사진 찍기 전 대기(s)")
    ap.add_argument("--yes", action="store_true", help="active 시작 확인(Enter) 생략 — 시험용")
    ap.add_argument("--approvals", default=os.path.expanduser("~/rl_ws/sim2real/logs/runs/episode_approvals.jsonl"))
    a = ap.parse_args()
    a.judge = {x for x in a.judge.split(",") if x}
    if a.mode == "active" and os.environ.get("ROS_DOMAIN_ID") in (None, "", "0"):
        print("[vlm] ✗ ROS_DOMAIN_ID 0/미설정 — sup.sh 로 띄운다")
        return 2
    s = Sup(a)
    try:
        return s.run()
    except KeyboardInterrupt:
        if a.mode == "active":
            s.stop("operator Ctrl-C")
        return 130
    finally:
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
