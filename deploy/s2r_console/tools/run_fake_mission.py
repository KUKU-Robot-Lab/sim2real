#!/usr/bin/env python3
"""fake 미션을 콘솔로 처음부터 끝까지 밟고, 오른팔 · 오른손 · 왼팔 · 왼손이 단계마다 목표에 닿았는지 판정한다.

    deploy/s2r_console/tools/run_fake_mission.sh                 # 도메인 97, 프로파일 dg5f_m_fake
    deploy/s2r_console/tools/run_fake_mission.sh --with-viewer   # Isaac 뷰어 단계도 실행(기본은 건너뜀)

화면 없이 `Console` 을 그대로 쓴다 — 운영자가 누르는 승인 · 실행 · 수동 확인을 같은 API 로 부른다.
수동 확인은 자동으로 "정상" 으로 답하되, 그 전에 관절 상태를 재서 판정한다(팔이 초기 자세인가 · 손이 그대로인가).
실기 도메인에는 붙지 않는다(프로파일이 fake 가 아니면 거부).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "sample_joints.py"
STAGE_TIMEOUT_S = 900.0          # preflight(pytest 전부) 가 5 분 남짓
ARM_TOL = 0.02                   # rad — pd 정착 허용(settle.tol 과 같은 자리수)
HAND_TOL = 0.05                  # rad — 손 속도 제한 램프가 끝났는가


@dataclass
class Report:
    rows: list = field(default_factory=list)
    failed: bool = False

    def add(self, what: str, ok: bool, detail: str = "") -> None:
        self.rows.append((what, ok, detail))
        self.failed |= not ok
        print(f"  {'✓' if ok else '✗'} {what}" + (f" — {detail}" if detail else ""), flush=True)


def sample(domain: int) -> dict[str, float]:
    env = {**os.environ, "ROS_DOMAIN_ID": str(domain)}
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.6"], env=env, capture_output=True,
                         text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"sample_joints 실패: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])["q"]


def worst(q: dict[str, float], want: dict[str, float]) -> tuple[float, str]:
    missing = [j for j in want if j not in q]
    if missing:
        return float("inf"), f"관절 상태 없음 {missing[:3]}"
    j = max(want, key=lambda k: abs(q[k] - want[k]))
    return abs(q[j] - want[j]), f"{j} {q[j]:+.3f} (목표 {want[j]:+.3f})"


def targets(contract_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    c = json.loads(contract_path.read_text())
    out = {}
    for side, s in c["sides"].items():
        arm = dict(zip(s["arm_joints"], s["home_arm"]))
        hand = {j: float(s["home_hand"][j]) for j in s["hand_joints"] if j in s["home_hand"]}
        out[side] = {"arm": arm, "hand": hand}
    return out


def check_side(report: Report, domain: int, side: str, tgt: dict, *, arm: bool, hand: bool, when: str) -> None:
    q = sample(domain)
    if arm:
        err, what = worst(q, tgt[side]["arm"])
        report.add(f"{when}: {side} 팔이 초기 자세", err < ARM_TOL, f"최대 오차 {err:.4f} rad · {what}")
    if hand:
        err, what = worst(q, tgt[side]["hand"])
        report.add(f"{when}: {side} 손이 초기 손 자세", err < HAND_TOL, f"최대 오차 {err:.4f} rad · {what}")


def check_hand_not_home(report: Report, domain: int, side: str, tgt: dict, when: str) -> None:
    """팔만 움직인 직후 — 손은 아직 초기 손 자세가 아니어야 한다(0 에서 시작한 fake 손 · pd home_hand: keep)."""
    q = sample(domain)
    err, what = worst(q, tgt[side]["hand"])
    report.add(f"{when}: {side} 손은 아직 움직이지 않았다(팔 먼저)", err >= HAND_TOL, f"초기 손 자세와의 차 {err:.3f} · {what}")


def run(args) -> int:
    from s2r_console import console as C

    con = C.Console(bridge=True)
    s = con.open(args.profile, operator="fake-e2e")
    if s.profile.is_real:
        con.shutdown()
        raise SystemExit("실기 프로파일에는 쓰지 않는다")
    domain = s.profile.domain
    tgt = targets(Path(con.repo) / s.mission.artifacts["contract"])
    report = Report()
    skip = set(args.skip) | ({"viewer"} if not args.with_viewer else set())
    print(f"[fake-e2e] {args.profile} · 도메인 {domain} · run {s.run_id} · 건너뜀 {sorted(skip)}", flush=True)
    try:
        while s.state.status != "DONE":
            stage = s.state.stage
            if stage in skip:
                con.skip_stage(stage, operator="fake-e2e")
                print(f"» {stage} 건너뜀", flush=True)
                continue
            print(f"▶ {stage}", flush=True)
            con.run_stage(stage, operator="fake-e2e")
            acked: set[int] = set()
            t0 = time.monotonic()
            while s.runner is not None and s.runner.active:
                for st in s.runner.view()["steps"]:
                    if st["status"] == "waiting" and st["index"] not in acked:
                        before_ack(report, domain, stage, st, tgt)
                        con.ack(st["index"], True)
                        acked.add(st["index"])
                if time.monotonic() - t0 > STAGE_TIMEOUT_S:
                    con.abort_stage()
                    report.add(f"{stage} 시간 초과", False)
                    break
                time.sleep(0.2)
            outcome = s.runner.outcome if s.runner else None
            report.add(f"{stage} 완료", outcome == "DONE", "" if outcome == "DONE" else f"{outcome}: {s.state.note}")
            if outcome != "DONE":
                for st in s.runner.view()["steps"]:
                    if st["status"] in ("failed", "aborted") and st["kind"] not in ("manual", "stop"):
                        print(con.log_tail(st["key"])[-1500:], flush=True)
                break
            after_stage(report, domain, stage, tgt)
    finally:
        kept = con.shutdown()
        if kept:
            report.add("종료 후 남은 프로세스 없음", False, ", ".join(kept))
    print("\n[fake-e2e] " + ("실패" if report.failed else "전부 통과") + f" · {sum(ok for _, ok, _ in report.rows)}/{len(report.rows)}")
    return 1 if report.failed else 0


def before_ack(report: Report, domain: int, stage: str, step: dict, tgt: dict) -> None:
    """수동 확인 직전 — 운영자가 눈으로 볼 것을 재서 판정한다."""
    side = "right" if stage.endswith("_right") else "left" if stage.endswith("_left") else None
    if stage.startswith("home_") and side and "도착" in step["note"]:
        check_side(report, domain, side, tgt, arm=True, hand=False, when="팔 이동 뒤")
        check_hand_not_home(report, domain, side, tgt, when="팔 이동 뒤")


def after_stage(report: Report, domain: int, stage: str, tgt: dict) -> None:
    side = "right" if stage.endswith("_right") else "left" if stage.endswith("_left") else None
    if stage.startswith("home_") and side:
        time.sleep(3.0)                                  # 손 속도 제한 램프
        check_side(report, domain, side, tgt, arm=True, hand=True, when="손 홈 뒤")
    elif stage.startswith("selftest_") and side:
        check_side(report, domain, side, tgt, arm=True, hand=False, when="셀프테스트 뒤")
    elif stage.startswith("return_") and side:
        import numpy as np
        path = HERE.parents[2] / "deploy" / "policy_control" / "paths" / f"home_{side}.npz"
        start = [float(v) for v in np.load(path)["meta_start"]]
        want = {f"{side[0]}_aj_{i}": v for i, v in enumerate(start, 1)}
        err, what = worst(sample(domain), want)
        report.add(f"되짚기 뒤: {side} 팔이 시작 자세로 돌아왔다", err < ARM_TOL, f"최대 오차 {err:.4f} rad · {what}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="dg5f_m_fake")
    ap.add_argument("--with-viewer", action="store_true", help="Isaac 뷰어 단계도 실행(GPU · 1~2 분)")
    ap.add_argument("--skip", nargs="*", default=[], help="건너뛸 단계 id (skippable 인 것만)")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
