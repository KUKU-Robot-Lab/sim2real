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
#: 다른 단계와 **양자택일**이라 한 번에 걷지 않는 단계 — 비상 복귀(reset_*)는 정상 복귀(return_*)의 대체 경로다.
#: 둘 다 밟으면 reset 이 pd 를 풀고 차렷까지 내린 뒤 return 이 "홈에서 정착"을 하려 해서 거부된다(09.23 fake).
#: 비상 복귀 자체는 test_pc_reset_to_rest 가 따로 잠근다.
EXCLUSIVE = ("reset_right", "reset_left")
SKIP_WAIT_S = 10.0             # s — release 직후 pd 상태가 새로 들어올 시간(STALE_S 보다 넉넉히)
ARM_TOL = 0.02                   # rad — pd 정착 허용(settle.tol 과 같은 자리수)
#: rad — 손 속도 제한 램프가 끝났는가. pd 가 **일부러** 한계 안쪽 HAND_LIMIT_MARGIN 으로 물려 지령하므로
#: (09.23 실기: 계약 홈이 굽힘 관절 하한 0.0 그 자체라 손가락이 꺾이고 드라이버가 error 423 을 냈다)
#: 계약 홈과는 그 여유만큼 **어긋나 서는 것이 정상**이다. 그래서 허용은 그 여유보다 커야 한다.
HAND_TOL = 0.08
#: 팔만 움직인 직후 "손은 아직 그대로"를 판정하는 문턱 — 이쪽은 여유와 무관하게 크게 본다.
HAND_MOVED_TOL = 0.3


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


def skip_when_pd_settles(con, stage: str, wait_s: float = SKIP_WAIT_S) -> None:
    """건너뛰기는 pd 상태를 모르거나 팔을 잡은 동안 거부된다(정당한 게이트). release 직후에는 상태 토픽이
    한 박자 늦어 그 게이트에 걸린다(09.25 fake: release_left 직후 sensors_off). 잠깐 기다렸다 다시 묻고,
    그래도 거부면 사유를 붙여 실패한다."""
    from s2r_console import console as C

    deadline = time.monotonic() + wait_s
    while True:
        try:
            con.skip_stage(stage, operator="fake-e2e")
            return
        except C.ConsoleError as exc:
            if time.monotonic() >= deadline:
                raise SystemExit(f"✗ {stage} 를 건너뛸 수 없다: {list(getattr(exc, 'reasons', ()))}") from exc
            time.sleep(0.5)


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
    report.add(f"{when}: {side} 손은 아직 움직이지 않았다(팔 먼저)", err >= HAND_MOVED_TOL, f"초기 손 자세와의 차 {err:.3f} · {what}")


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
    if "sensors" in skip:                            # 켠 적이 없으면 끌 것도 없다(런처 구독자 0 → rc 1)
        skip.add("sensors_off")
    print(f"[fake-e2e] {args.profile} · 도메인 {domain} · run {s.run_id} · 건너뜀 {sorted(skip)} · "
          f"지나침 {sorted(EXCLUSIVE)}", flush=True)
    try:
        # 미션 차례대로 하나씩 — 창(lane)이 있으면 동시에도 되지만, e2e 는 **모든 단계를 한 번씩** 밟는 것이 일이다.
        # yaml 순서는 선행(needs)을 이미 만족한다(mission_core 가 뒤 단계를 선행으로 두는 것을 막는다).
        for stage in [x.id for x in s.mission.stages]:
            if stage in EXCLUSIVE:
                # 지나친다 — **건너뛰기가 아니다**. 건너뛰기는 pd 가 팔을 잡고 있으면 거부되고(정당하다),
                # 완료로도 적힌다. 대체 경로는 걷지 않았을 뿐 끝낸 것이 아니다.
                print(f"» {stage} 지나침 (대체 경로 — {stage.replace('reset_', 'return_')} 를 걷는다)", flush=True)
                continue
            if stage in skip:
                skip_when_pd_settles(con, stage)
                print(f"» {stage} 건너뜀", flush=True)
                continue
            print(f"▶ {stage}", flush=True)
            con.run_stage(stage, operator="fake-e2e")
            runner = con._runner_of(s, stage)
            acked: set[int] = set()
            t0 = time.monotonic()
            while runner is not None and runner.active:
                for st in runner.view()["steps"]:
                    if st["status"] == "waiting" and st["index"] not in acked:
                        before_ack(report, domain, stage, st, tgt)
                        con.ack(st["index"], True, stage_id=stage)
                        acked.add(st["index"])
                if time.monotonic() - t0 > STAGE_TIMEOUT_S:
                    con.abort_stage(stage)
                    report.add(f"{stage} 시간 초과", False)
                    break
                time.sleep(0.2)
            outcome = runner.outcome if runner else None
            report.add(f"{stage} 완료", outcome == "DONE", "" if outcome == "DONE" else f"{outcome}: {s.state.note}")
            if outcome != "DONE":
                for st in runner.view()["steps"]:
                    if st["status"] in ("failed", "aborted") and st["kind"] not in ("manual", "stop"):
                        try:                          # 기동 전에 막힌 단계(콘솔 밖 런치 감지 등)는 로그가 없다
                            print(con.log_tail(st["key"])[-1500:], flush=True)
                        except C.ConsoleError as exc:
                            print(f"  (로그 없음: {exc})", flush=True)
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
