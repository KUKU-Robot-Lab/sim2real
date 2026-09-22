#!/usr/bin/env python3
"""4 노드의 /policy_control/status/* (JSON) 를 seq 로 join → CSV.

지연(obs 발행 → pd 적용), 홉별 proc_ms, seq 결손, HOLD 사유가 한 표에 남는다.
라이브 구독 또는 rosbag2 재생 어느 쪽이든 같은 토픽이므로 같은 코드다.

join·요약 자체는 `policy_control.status_join` 에 있다 — 상태판도 같은 코드를 쓴다.
이 파일은 그 코어의 ROS 껍질일 뿐이고, CLI 와 출력 포맷은 예전과 같다.

    python3 deploy/policy_control/tools/status_to_csv.py --seconds 60 --out logs/policy_control/run.csv
    # 기록해 둔 덤프를 다시 표로 (ROS 불필요)
    python3 deploy/policy_control/tools/status_to_csv.py --from-jsonl logs/.../status.jsonl --out /tmp/run.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_control.status_join import NODES, StatusJoiner, join_jsonl, row_fields, summarize  # noqa: E402


def _collect_live(seconds: float, jsonl: Path | None, nodes=NODES) -> StatusJoiner:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    joiner = StatusJoiner()
    rclpy.init()
    node = Node("status_to_csv")
    raw = open(jsonl, "w") if jsonl else None

    def on_status(name):
        def cb(msg):
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            if raw is not None:
                raw.write(json.dumps({"topic": name, **d}, ensure_ascii=False) + "\n")
            joiner.offer(name, d)
        return cb

    for n in nodes:
        node.create_subscription(String, f"/policy_control/status/{n}", on_status(n), 50)
    t0 = time.time()
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if raw is not None:
        raw.close()
    return joiner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy-dt", type=float, default=None, help="예산 표시용 (s)")
    ap.add_argument("--jsonl", type=Path, default=None, help="모든 status 메시지를 그대로(한 줄 JSON) 남긴다 — 디버그용")
    ap.add_argument("--from-jsonl", type=Path, default=None, help="구독 대신 기록된 덤프에서 표를 만든다")
    ap.add_argument("--nodes", default=",".join(NODES),
                    help="join 할 status 노드 (쉼표). pour 체인은 한 노드다: --nodes pour_node")
    args = ap.parse_args(argv)

    nodes = tuple(n.strip() for n in args.nodes.split(",") if n.strip())
    if args.from_jsonl is not None:
        rows = join_jsonl(args.from_jsonl.read_text(errors="replace").splitlines(), nodes)
    else:
        rows = _collect_live(args.seconds, args.jsonl, nodes).rows(nodes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row_fields(rows))
        w.writeheader()
        w.writerows(rows)
    print(summarize(rows, args.policy_dt), "→", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
