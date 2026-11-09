#!/usr/bin/env python3
"""11.07 실기 bag color(JPEG) → head_mask_node.MaskDetector → 160x90 비트 npz(키 = color stamp ns, 'config' = 검출 설정 JSON).

student_bag_replay.py --mask-bits 의 입력. YOLO 는 .venv-yolo 에만 있어 두 단계로 나눈다.
  source /opt/ros/humble/setup.bash
  .venv-yolo/bin/python scripts/analysis/mask_bag_bits.py --bag logs/runs/<YYMMDD>/bags/<bag> --policy-dir deploy/policies/rh56f1/aglt/right_student1105mA \
      --out logs/analysis/<이름>/mask_bits.npz
처리 시간(디코드 제외, 검출 + 비트) p50 · p90 을 함께 적는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "nodes"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from head_mask_node import MaskDetector, resolve_yolo  # noqa: E402
from student_bag_replay import T_COLOR, decode_jpeg_bgr, read_bag, stamp_ns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", required=True)
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", action="store_true", help="fp16 추론(fp32 와 비교용)")
    args = ap.parse_args()
    import torch
    from sensor_msgs.msg import CompressedImage

    weights, y = resolve_yolo(args.policy_dir, None)     # conf · imgsz · 인스턴스 규칙 = 묶음 값(노드와 같게)
    det = MaskDetector(weights, y["imgsz"], y["conf"], top1=y["top1_per_class"], cross_iou=y["cross_class_iou"], half=args.half)
    if det.md5 != str(y["md5"]):
        raise SystemExit(f"{weights} md5 {det.md5} ≠ student.yaml {y['md5']}")
    out, ms = {}, []
    for _, _, m in read_bag(REPO / args.bag / "camera", {T_COLOR: CompressedImage}):
        bgr = decode_jpeg_bgr(m.format, bytes(m.data))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out[str(stamp_ns(m))] = det(bgr)
        ms.append((time.perf_counter() - t0) * 1e3)
    if not out:
        raise SystemExit(f"{args.bag} 에 {T_COLOR} 가 없다")
    a = np.array(ms[1:] or ms)
    cfg = {**det.config(), "bag": args.bag, "frames": len(out), "proc_ms_p50": float(np.median(a)),
           "proc_ms_p90": float(np.percentile(a, 90))}
    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, config=json.dumps(cfg, ensure_ascii=False), **out)
    print(json.dumps({k: cfg[k] for k in ("frames", "proc_ms_p50", "proc_ms_p90", "weights_md5")}))


if __name__ == "__main__":
    main()
