#!/usr/bin/env python3
"""joint 정책 실기 기록(joint_recorder.py 의 npz)을 읽어 제어기가 정책을 따라갔는지 표로 낸다. 파일만 읽는다.

    python3 deploy/policy_control/tools/joint_trace_report.py --latest --side left
    python3 deploy/policy_control/tools/joint_trace_report.py --npz logs/policy_control/real_runs/<파일>.npz

무엇을 재는지는 policy_control/joint_trace.py 머리말에 있다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control.joint_trace import render, summarize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_recorder import OUT_DIR  # noqa: E402


def newest(out_dir: Path, side: str) -> Path | None:
    """가장 최근 기록 — bag run 폴더(policy_bag.py) 또는 옛 npz(joint_recorder.py). 이름 앞이 시각이라 이름 순이 시간 순."""
    cands = [p for p in out_dir.glob(f"*__{side}") if p.is_dir()] + list(out_dir.glob(f"*__{side}.npz"))
    return max(cands, key=lambda p: p.name) if cands else None


def trace_of(path: Path) -> Path:
    """run 폴더면 trace.npz(없으면 bag 에서 만든다), npz 면 그대로."""
    if path.is_dir():
        t = path / "trace.npz"
        if not t.is_file():
            from bag_to_trace import convert
            t = convert(path)
        return t
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", type=Path, default=None, help="trace npz 또는 bag run 폴더")
    ap.add_argument("--latest", action="store_true", help="--out-dir 에서 --side 의 가장 최근 기록")
    ap.add_argument("--side", choices=("left", "right"), default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    path = args.npz
    if path is None:
        if not (args.latest and args.side):
            raise SystemExit("--npz 또는 --latest --side <left|right> 가 필요하다")
        path = newest(args.out_dir, args.side)
        if path is None:
            raise SystemExit(f"✗ {args.out_dir} 에 {args.side} 기록이 없다")
    path = trace_of(path)
    d = np.load(path, allow_pickle=False)
    print(f"[report] {path}")
    errs = [str(e) for e in d["meta_errors"]]
    if errs:
        print(f"[report] 기록 중 디코드 실패: {errs}")
    print(render(summarize(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
