#!/usr/bin/env python3
"""저장된 홈 경로를 지금 재생해도 되는가 — 팔이 경로의 시작점에 있고, 경로가 지금 계약으로 만든 것인가. 구독만 한다.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/check_path_start.py --npz deploy/policy_control/paths/home_right.npz

09.22 사용자: "실기 동작의 경우 미리 최적 경로를 저장해두고 진행해야 함" — 경로는 오프라인에서 한 번 계획 · 충돌 검사 ·
Isaac 확인을 거쳐 저장소에 둔다(deploy/policy_control/paths/). 실기에서는 다시 계산하지 않고 이 검사만 한다.

  · 이 팔의 7관절 실측이 경로 시작점에서 --tol(기본 0.05 rad) 안인가 — 아니면 재생 진입 램프가 검사하지 않은 길로 간다
  · 경로를 만든 계약(meta_contract_sha1)이 지금 계약과 같은가 — 홈이 바뀌었으면 경로 끝이 홈이 아니다
  · 관절 상태가 살아 있는가 — 전부 정확히 0.0 이면 엔코더를 못 읽는 것이다(09.22: CAN RX 0 인데 0.0 을 냈다)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SIM2REAL = Path(__file__).resolve().parents[3]
SAMPLE = SIM2REAL / "deploy" / "s2r_console" / "tools" / "sample_joints.py"
CONTRACT = SIM2REAL / "logs" / "policy" / "asset_openarm_dg5f-m-short_bi_rl" / "deploy_contract.json"


def measure() -> dict[str, float]:
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.6"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"✗ 관절 상태를 못 읽었다: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])["q"]


def verdict(q: dict[str, float], joints: list[str], start: np.ndarray, tol: float, allow_exact_zero: bool = False) -> list[str]:
    """재생하면 안 되는 이유. 비면 재생해도 된다."""
    missing = [j for j in joints if j not in q]
    if missing:
        return [f"관절 상태 없음 {missing} — 드라이버가 떠 있는가"]
    now = np.array([q[j] for j in joints])
    if np.all(now == 0.0) and not allow_exact_zero:
        return ["관절 상태가 전부 정확히 0.0 — 엔코더를 못 읽는 것이다(모터 전원 · CAN 응답을 볼 것)"]
    err = np.abs(now - start)
    k = int(np.argmax(err))
    if err[k] > tol:
        return [f"팔이 경로 시작점에 있지 않다 — {joints[k]} {now[k]:+.3f} (시작점 {start[k]:+.3f}, 허용 {tol})"]
    return []


def hand_verdict(q: dict[str, float], hand_q_text: str, tol: float) -> list[str]:
    """경로를 그 손 자세로 검사했다면(meta_hand_q) 지금 손도 그 자세여야 한다 — pd 는 팔이 움직이는 동안 손을 그대로 둔다."""
    want = {}
    for item in (hand_q_text or "").split(","):
        if item.strip():
            k, _, v = item.partition("=")
            want[k.strip()] = float(v)
    if not want:
        return []
    missing = [k for k in want if k not in q]
    if missing:
        return [f"손 관절 상태 없음 {missing[:3]}"]
    k = max(want, key=lambda j: abs(q[j] - want[j]))
    if abs(q[k] - want[k]) > tol:
        return [f"손이 경로를 검사한 자세가 아니다 — {k} {q[k]:+.3f} (검사 자세 {want[k]:+.3f}, 허용 {tol})"]
    return []


def sphere_verdict(q: dict[str, float], side: str, radius: float) -> list[str]:
    """경로를 손 봉투 구로 계획했다면(meta_hand_sphere) 지금 손이 그 구 안에 있어야 한다.

    손가락 자세를 맞추는 대신 **봉투에 들어가는가**만 본다 — 손 전원을 껐다 켤 때마다 손가락이
    다른 자세로 자리 잡아도, 충분히 오므려져 있으면 저장 경로가 그대로 유효하다(09.23 사용자).
    """
    W = _world()
    links, joints, _ = W.parse_urdf(W.URDF_DEFAULT)
    hand = [j for j in q if f"{side[0]}_hj_" in j]
    if not hand:
        return [f"손 관절 상태가 없다({side}) — 손 드라이버가 떠 있는가"]
    r, worst = W.hand_radius(links, joints, {j: float(q[j]) for j in hand}, side)
    if r > radius:
        return [f"손이 경로 봉투(반지름 {radius * 100:.1f} cm)를 {(r - radius) * 100:.1f} cm 넘는다 — "
                f"{worst} 가 손바닥에서 {r * 100:.1f} cm. 손가락을 더 오므릴 것"]
    print(f"[path] 손 봉투 {r * 100:.1f} / {radius * 100:.1f} cm ({worst}) — 구 안")
    return []


def _world():
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("home_path_world", here / "home_path_world.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["home_path_world"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--contract", type=Path, default=CONTRACT)
    ap.add_argument("--tol", type=float, default=0.05, help="시작점 허용 [rad] (관절별 최대)")
    ap.add_argument("--allow-exact-zero", action="store_true", help="fake 플랜트 전용 — 팔이 정확히 0 에서 시작한다")
    ap.add_argument("--hand-tol", type=float, default=0.2, help="손 자세 허용 [rad] (경로를 실측 손으로 검사했을 때)")
    args = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID", "") in ("", "0"):
        raise SystemExit("✗ ROS_DOMAIN_ID 가 비었거나 0 — 거부")
    d = np.load(args.npz)
    joints = [str(j) for j in d["meta_joints"]]
    reasons = []
    want = str(d["meta_contract_sha1"])
    have = hashlib.sha1(args.contract.read_bytes()).hexdigest()
    if want != have:
        reasons.append(f"경로를 만든 계약({want[:10]})이 지금 계약({have[:10]})과 다르다 — 경로를 다시 만들 것")
    q = measure()
    reasons += verdict(q, joints, np.asarray(d["meta_start"], dtype=float), args.tol, args.allow_exact_zero)
    sphere = float(d["meta_hand_sphere"]) if "meta_hand_sphere" in d else float("nan")
    if sphere == sphere:                      # 봉투로 계획한 경로 — 손 자세는 구 안에 있기만 하면 된다
        side = "right" if str(d["meta_joints"][0]).startswith("r_") else "left"
        reasons += sphere_verdict(q, side, sphere)
    elif "meta_hand_q" in d and str(d["meta_hand_q"]) and not args.allow_exact_zero:
        reasons += hand_verdict(q, str(d["meta_hand_q"]), args.hand_tol)
    print(f"[path] {args.npz.name} · {d['meta_method']} · {len(d['arm_target'])} 프레임 · 최소 여유 "
          f"{float(d['meta_min_clearance_non_escape']):.3f} m")
    for r in reasons:
        print(f"  ✗ {r}")
    if reasons:
        return 1
    print("  ✓ 팔이 경로 시작점에 있고, 경로는 지금 계약으로 만든 것이다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
