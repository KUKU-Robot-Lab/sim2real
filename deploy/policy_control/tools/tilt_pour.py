#!/usr/bin/env python3
"""tilt 붓기 — 지금 두 팔 자세(세팅 자세, 컵을 쥔 채)에서 바로. 상황판 control 탭 [tilt_pour] 가 부른다.
★10.19 사용자 "tilt 시작은 시작 자세부터 만들어 주는 게 아니라, 내가 미리 세팅 자세를 만들어 둘 테니 그 상태에서 바로 틸팅".
세팅 자세로 가는 것은 [tilt_now](tilt_now.py). 에피소드 tilt 노드(episode_ros._run_planned)와 같은 두 걸음:

  1. plan_tilt.py — 실측 두 팔 자세 → hdgp tilt_real_start(학습과 같은 계산) → 검사(IK 오차 · 관절 여유 · 명령 속도 · sim 출발
     범위 · 탁자 여유). 검사가 하나라도 걸리면 rc 2 — 움직이지 않는다.
  2. replay_to_pd.py — 두 팔 관절(r_aj_1..7, l_aj_1..7)을 한 시각표로 재생(D4 기본 궤적 + 시작 자세 복귀). 손 목표는 그대로
     (pd 가 붙든 쥔 손). 재생 직전 실측이 계획 자세에서 start_gap_rad 넘게 벗어나면 replay 가 거부한다.

기록: logs/tilt/<날짜_시각>_pour/(q.json · check.json · tilt_cmd.npz · plan_tilt.log) + 실기면 bag(rh56f1_record.sh tilt_pour).

    python3 deploy/policy_control/tools/tilt_pour.py                  # 계획 · 검사만(움직이지 않음)
    python3 deploy/policy_control/tools/tilt_pour.py --execute        # 검사 통과면 재생
종료 코드: 0 = 끝(재생했거나 계획만) · 2 = 검사 실패(움직이지 않음) · 3 = 핀 · 기준 파일 불일치 · 그 밖 = 계획 · 재생 실패.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parent
SIM2REAL = TOOLS.parents[2]
ASSET = TOOLS.parent / "config" / "tilt" / "base_arc6"
JOINTS = [f"{p}_aj_{i}" for p in ("r", "l") for i in range(1, 8)]
RECORD = TOOLS / "rh56f1_record.sh"
RECORD_NAME = "tilt_pour"


def record_topics() -> list[str]:
    """bag 에 같이 싣는 것 — 팔 지령 · pd 상태 · pd 가 실제로 적용한 목표(지연 도구)."""
    out = ["/policy_control/joint_target"]
    for s in ("right", "left"):
        out += [f"/policy_control/status/pd_{s}", f"/policy_control/pd_{s}/applied"]
    return out


def run(argv: list[str]) -> int:
    print("$", " ".join(argv), flush=True)
    return subprocess.run(argv).returncode


def fresh_dir(root: Path, tag: str) -> Path:
    """같은 초에 두 번(계획만 → 실행) 눌러도 앞 기록을 덮지 않는다."""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = root / f"{stamp}_{tag}"
    k = 1
    while out.exists() and k < 100:
        out, k = root / f"{stamp}_{tag}_{k}", k + 1
    return out


def fail_summary(out: Path) -> str:
    """check.json 의 걸린 항목 — 콘솔 로그에서 rc 만 보이지 않게(10.19 사용자 rc=2)."""
    try:
        chk = json.loads((out / "check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    bad = [f"{k} {v.get('value')} (한계 {v.get('limit')})" for k, v in (chk.get("checks") or {}).items() if not v.get("ok", True)]
    return " · ".join(bad)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--out-root", type=Path, default=SIM2REAL / "logs" / "tilt")
    ap.add_argument("--rate-scale", type=float, default=None, help="없으면 자산 policy.yaml rate_scale")
    ap.add_argument("--no-record", action="store_true", help="bag 기록 안 함(fake)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((a.asset / "policy.yaml").read_text(encoding="utf-8")) or {}
    rate = float(a.rate_scale if a.rate_scale is not None else cfg.get("rate_scale", 1.0))
    gap = float(cfg.get("start_gap_rad", 0.05))
    out = fresh_dir(a.out_root, "pour")
    rc = run([sys.executable, str(TOOLS / "plan_tilt.py"), "--asset", str(a.asset), "--out", str(out)])
    if rc != 0:
        print(f"[tilt_pour] 계획 · 검사 rc {rc} — 움직이지 않았다 · 걸린 항목: {fail_summary(out) or '(check.json 없음)'} · "
              f"{out}/check.json", flush=True)
        return rc
    chk = json.loads((out / "check.json").read_text(encoding="utf-8"))
    print(f"[tilt_pour] 검사 통과 · {float(chk.get('T_end_s', float('nan'))):.1f} s · rate {rate:g} · 시작 간극 한계 {gap} rad · {out}",
          flush=True)
    if not a.execute:
        print("[tilt_pour] 계획만 — --execute 로 재생")
        return 0
    recording = not a.no_record
    if recording:
        env = {**os.environ, "EXTRA": " ".join(record_topics())}
        print(f"$ bash {RECORD} start {RECORD_NAME}", flush=True)
        if subprocess.run(["bash", str(RECORD), "start", RECORD_NAME], env=env).returncode != 0:
            print("[tilt_pour] bag 기록 시작 실패 — 기록 없이 진행하지 않는다", flush=True)
            return 1
    try:
        rc = run([sys.executable, str(TOOLS / "replay_to_pd.py"), "--npz", str(out / "tilt_cmd.npz"), "--joints", ",".join(JOINTS),
                  "--rate-scale", repr(rate), "--max-start-gap", repr(gap), "--execute"])
    finally:
        if recording:
            subprocess.run(["bash", str(RECORD), "stop", RECORD_NAME])
    print(f"[tilt_pour] 재생 {'끝' if rc == 0 else f'실패 rc {rc}'} · {out}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
