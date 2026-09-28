#!/usr/bin/env python3
"""policy_bag.py 가 남긴 run 폴더(policy/ · sensors/ bag)를 분석용 trace.npz 로 바꾼다. 파일만 읽는다.

    python3 deploy/policy_control/tools/bag_to_trace.py logs/policy_control/real_runs/<run 폴더>

시각은 bag 이 받은 시각(recv)이다 — 정책 노드 · pd 와 같은 PC 시계라 묶음끼리 바로 맞는다.
trace.npz 의 모양은 joint_trace.summarize 가 읽는 그대로다(policy_control/trace_acc.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402,F401
from policy_control.joint_contract import load_contract  # noqa: E402
from policy_control.sources import load_robot_cfg, select_side  # noqa: E402
from policy_control.trace_acc import BAG_GROUPS, TraceAccumulator  # noqa: E402


def read_bag(path: Path):
    """(topic, 받은 시각 s, 메시지) 를 시간 순으로."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        yield topic, t_ns * 1e-9, deserialize_message(raw, types[topic])


def convert(run: Path) -> Path:
    meta = json.loads((run / "meta.json").read_text())
    c = load_contract(Path(meta["contract"]))
    cfg = select_side(load_robot_cfg(Path(meta["robot"])), c.side)
    kinds = {t: v[0] for t, v in meta["topics"].items()}
    msgs = []
    for g in BAG_GROUPS:
        if (run / g / "metadata.yaml").is_file():
            msgs += [(t, topic, m) for topic, t, m in read_bag(run / g) if topic in kinds]
    msgs.sort(key=lambda x: x[0])                   # 묶음 둘을 한 시간축으로 — SourceSet 신선도가 시각을 본다
    acc = TraceAccumulator(c, cfg)
    for t, topic, m in msgs:
        acc.add(kinds[topic], t, m)
    out = run / "trace.npz"
    np.savez(out, **acc.to_arrays(meta))
    print(f"[bag→trace] {run.name} · {acc.counts()} → {out.name}"
          + (f" · 디코드 실패 {acc.errors}" if acc.errors else ""), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    convert(ap.parse_args().run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
