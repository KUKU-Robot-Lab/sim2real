#!/usr/bin/env python3
"""fake 도메인에서 상황판(Console)으로 에피소드 하나를 끝까지 굴린다 — 단계 준비 → 에피소드 단계 → [다음]/[연속 실행].

    ROS_DOMAIN_ID=97 python3 deploy/s2r_console/tools/run_fake_episode.py --episode pick_place_right --mode step
    ROS_DOMAIN_ID=97 python3 deploy/s2r_console/tools/run_fake_episode.py --episode pick_place_right --mode run

운영자가 누르는 것(단계 실행 · 수동 확인 · 에피소드 확인 입력)을 같은 Console API 로 부른다. 실기 프로파일이면 거부.
fake 손은 촉각이 0 이라 집기 정책은 쥠을 못 만든다 — 에피소드는 정책 성공이 아니라 배관(승인 · 노드 · 궤적 · 실패 → 복구 →
정지)을 본다. 결과: 단계 결과 · 에피소드 상태 · 실행기 기록(마지막 20 줄).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
STAGE_TIMEOUT_S = 900.0
EPISODE_TIMEOUT_S = 900.0
PREP = {"right": ("preflight", "head_home", "cups", "cup_holders", "drivers", "hand_right", "hand_check_right",
                  "pd_load_right", "pd_arm_right", "home_right"),
        "left": ("preflight", "head_home", "cups", "cup_holders", "drivers", "hand_left", "hand_check_left",
                 "pd_load_left", "pd_arm_left", "home_left")}


def _wait_stage(con, s, stage: str, *, stop_at_manual: str = "") -> object:
    """단계를 돌리며 수동 확인을 자동으로 누른다. stop_at_manual 이 든 수동 단계에서는 누르지 않고 돌아온다."""
    runner = con._runner_of(s, stage)
    acked: set[int] = set()
    t0 = time.monotonic()
    while runner is not None and runner.active:
        for st in runner.view()["steps"]:
            if st["status"] == "waiting" and st["index"] not in acked:
                if stop_at_manual and stop_at_manual in st["note"]:
                    return runner
                con.ack(st["index"], True, stage_id=stage)
                acked.add(st["index"])
        if time.monotonic() - t0 > STAGE_TIMEOUT_S:
            con.abort_stage(stage)
            break
        time.sleep(0.2)
    return runner


def _episode_status(con) -> dict | None:
    return (con.snapshot().get("session") or {}).get("episode_runner")


def shot(port: int, out: Path) -> None:
    """상황판 화면 한 장(헤드리스 크롬) — 에피소드 패널을 눈으로 확인한다(rules/ui-pages.md)."""
    import subprocess
    out.parent.mkdir(parents=True, exist_ok=True)
    # chrome --screenshot 는 SSE 가 열린 상황판에서 JS 가 그린 화면을 못 담는다 — CDP 로 그린 뒤 찍는다(screenshot_cdp.mjs)
    try:
        r = subprocess.run(["node", str(HERE / "screenshot_cdp.mjs"), f"http://127.0.0.1:{port}/", str(out), "5000"],
                           capture_output=True, text=True, timeout=60)
        print(f"  📷 {out} {r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr.strip()[-200:]}", flush=True)
    except (OSError, subprocess.SubprocessError) as exc:      # 화면은 확인용 — 리허설을 멈추지 않는다
        print(f"  📷 실패 {out.name}: {exc}", flush=True)


def drive(con, mode: str, name: str, on_step=None) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30 and not (_episode_status(con) or {}).get("next_action"):
        time.sleep(0.5)
    if mode == "run":
        con.episode("run", operator="fake-episode", typed=f"episode:{name}")
    while time.monotonic() - t0 < EPISODE_TIMEOUT_S:
        st = _episode_status(con) or {}
        if st.get("phase") in ("SUCCESS", "FAILURE", "STOPPED") and not st.get("busy"):
            return st
        if mode == "step" and not st.get("busy") and st.get("next_action"):
            before = (st.get("index"), st.get("next_action"), len(st.get("history") or []))
            print(f"  ▶ 다음: {st['next_action']}", flush=True)
            con.episode("next", operator="fake-episode", typed=st["next_action"])
            if on_step:
                on_step(st["next_action"])
            t1 = time.monotonic()
            while time.monotonic() - t1 < 120:                 # 실행기가 받아 busy 가 되거나 상태가 바뀔 때까지
                cur = _episode_status(con) or {}
                if cur.get("busy") or (cur.get("index"), cur.get("next_action"), len(cur.get("history") or [])) != before:
                    break
                time.sleep(0.3)
        time.sleep(1.0)
    return _episode_status(con) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--episode", default="pick_place_right")
    ap.add_argument("--mode", choices=("step", "run"), default="step")
    ap.add_argument("--profile", default="rh56f1_fake")
    ap.add_argument("--serve", type=int, default=0, help="이 포트로 상황판 HTTP 를 같이 띄운다(127.0.0.1)")
    ap.add_argument("--shots", type=Path, default=None, help="--serve 와 함께: 에피소드 패널 화면을 여기에 찍는다")
    args = ap.parse_args()
    from s2r_console import console as C

    con = C.Console(bridge=True)
    s = con.open(args.profile, operator="fake-episode")
    if s.profile.is_real:
        con.shutdown()
        raise SystemExit("실기 프로파일에는 쓰지 않는다")
    side = "left" if args.episode.endswith("_left") else "right"
    rows = []
    server = None
    if args.serve:
        from s2r_console.server import serve
        import threading
        server = serve(con, bind="127.0.0.1", port=args.serve)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    snap = (lambda tag: shot(args.serve, args.shots / f"{args.episode}_{tag}.png")) if args.serve and args.shots else None
    try:
        for stage in PREP[side]:
            print(f"▶ {stage}", flush=True)
            con.run_stage(stage, operator="fake-episode")
            r = _wait_stage(con, s, stage)
            rows.append((stage, r.outcome if r else None))
            if not r or r.outcome != "DONE":
                print(f"✗ {stage} {r.outcome if r else None}: {s.state.note}", flush=True)
                return 1
        stage = f"episode_{args.episode}"
        print(f"▶ {stage}", flush=True)
        con.run_stage(stage, operator="fake-episode")
        _wait_stage(con, s, stage, stop_at_manual="에피소드 진행")
        time.sleep(2.0)
        if snap:
            snap("0_ready")
        st = drive(con, args.mode, args.episode, on_step=(lambda what: snap(f"step_{what.replace(':', '_')}")) if snap else None)
        if snap:
            snap("9_end")
        print(f"\n[fake-episode] {args.episode} · {args.mode} → {st.get('phase')} · last {json.dumps(st.get('last'), ensure_ascii=False)[:400]}")
        for h in (st.get("history") or [])[-20:]:
            print(f"   {h.get('timestamp', '')[11:19]} {h.get('event'):<17} {h.get('state_id') or '':<16} {h.get('result', '') or ''} "
                  f"{(h.get('termination_reason') or '')[:120]}")
        for nd in (con.snapshot().get("session") or {}).get("nodes") or []:
            stn = nd.get("status") or {}
            print(f"   [{nd['name']}] phase {stn.get('phase')} · reasons {stn.get('reasons')}")
        r = _wait_stage(con, s, stage)                     # 남은 수동 확인 → 노드 정지
        rows.append((stage, r.outcome if r else None))
        return 0 if st.get("phase") in ("SUCCESS", "FAILURE", "STOPPED") else 1
    finally:
        if server is not None:
            server.shutdown()
        kept = con.shutdown()
        print(f"[fake-episode] 단계 {rows} · 남은 프로세스 {kept or '없음'}")


if __name__ == "__main__":
    raise SystemExit(main())
