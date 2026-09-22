#!/usr/bin/env python3
"""pour 체크포인트가 실기에 나갈 자격이 있는가 — play trace 에서 직접 계산해 판정한다.

LOOP_STATE 에 사람이 적어 둔 숫자를 읽는 대신 `trace_*.npz` 에서 다시 센다. 기준은
`docs/PLAN_S2R_CONSOLE_2026-09-21.md` §5 이고, 근거는 각 항목의 `why` 에 적혀 있다.

    python3 deploy/policy_control/tools/ckpt_gate.py --trace logs/policy/pour_i18/trace.npz
    # 통과 0 · 탈락 4 · trace 를 못 읽으면 2

**통과는 "실기로 나가도 된다"가 아니라 "수치 게이트에서 걸리는 것이 없다"는 뜻이다.**
사용자 영상 판정과 안전망(기울기 워치독 · cross-arm guard)은 별도다.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 기준값. 바꿀 때는 왜 바꾸는지 같이 적는다 — 이 숫자가 실기 진입의 유일한 수치 관문이다.
SUCCESS_MIN = 0.70
IN_TARGET_MIN = 0.80
SPILL_MAX = 0.10
CUP_DIST_MIN_M = 0.12          # 실기 컵 포즈 오차와 림 접촉 사이의 마진 (컵 원점 정합 미검증)
POUR_DIR_MIN = 0.99            # 붓는 방향이 -x 여야 한다. +x 는 로봇 몸통 쪽이다
TILT_POUR_DEG = 45.0           # 이보다 기울어야 "붓는 중"으로 본다
TILT_PEAK_MAX_DEG = 120.0      # 넘어가면 손바닥 법선 중력 성분 부호가 뒤집히고 j7 이 한계 부근에 머문다
XY_EXCURSION_MAX_M = 0.25      # 시작 위치 대비. 팔마다 fabric 이 따로 도니 크게 움직일수록 위험하다
CROSS_FRAC_MAX = 0.0           # 소스 컵이 수신 쪽 반면으로 넘어가면 충돌 검출기가 없다


@dataclass(frozen=True)
class Row:
    name: str
    value: float
    limit: float
    ok: bool
    kind: str                   # gate | context
    why: str


def _tilt_deg(up: np.ndarray) -> np.ndarray:
    """컵 위쪽 축과 월드 z 의 각도. 90도를 넘으면 컵이 수평을 지나 뒤집힌 것이다."""
    return np.degrees(np.arccos(np.clip(up[..., 2], -1.0, 1.0)))


def evaluate(z, *, src="src", rcv="rcv") -> list[Row]:
    def g(k):
        return np.asarray(z[f"{k}"])

    success = float(g("success").max(axis=0).mean())
    in_target = float(g("in_target").max(axis=0).mean())
    spill = float(g("spill")[-1].mean())

    dist = np.linalg.norm(g(f"{src}_cup_pos") - g(f"{rcv}_cup_pos"), axis=-1)
    cup_dist = float(np.median(dist.min(axis=0)))

    pos = g(f"{src}_cup_pos")
    excursion = float(np.median(np.linalg.norm(pos[..., :2] - pos[0][None, :, :2], axis=-1).max(axis=0)))

    up = g(f"{src}_cup_up")
    tilt = _tilt_deg(up)
    tilt_peak = float(np.median(tilt.max(axis=0)))
    pouring = tilt > TILT_POUR_DEG
    pour_dir = float((up[..., 0][pouring] <= 0).mean()) if pouring.any() else 1.0

    # 수신 컵이 반대 부호 y 에 있을 때만 '중앙선'이 의미가 있다.
    rcv_y = float(np.median(g(f"{rcv}_cup_pos")[0, :, 1]))
    sign = 1.0 if rcv_y >= 0 else -1.0
    cross = float(((pos[..., 1] * sign) > 0).any(axis=0).mean())

    rows = [
        Row("success_ever", success, SUCCESS_MIN, success >= SUCCESS_MIN, "gate",
            "에피소드 중 한 번이라도 성공한 비율"),
        Row("in_target_max", in_target, IN_TARGET_MIN, in_target >= IN_TARGET_MIN, "gate",
            "성공 플래그가 아니라 실제로 옮겨진 양"),
        Row("spill_last", spill, SPILL_MAX, spill <= SPILL_MAX, "gate",
            "흘린 비율 — 실기에서는 액체가 바닥으로 간다"),
        Row("min_cup_dist_m", cup_dist, CUP_DIST_MIN_M, cup_dist >= CUP_DIST_MIN_M, "gate",
            "두 컵이 가장 가까웠을 때. 실기 컵 포즈 오차의 마진"),
        Row("pour_dir_neg_x", pour_dir, POUR_DIR_MIN, pour_dir >= POUR_DIR_MIN, "gate",
            f"기울기 {TILT_POUR_DEG:.0f}도 초과 구간에서 붓는 방향이 -x 인 비율"),
        Row("src_tilt_peak_deg", tilt_peak, TILT_PEAK_MAX_DEG, tilt_peak <= TILT_PEAK_MAX_DEG, "gate",
            "소스 컵 최대 기울기. 90도를 넘으면 수평을 지난 것이다"),
        Row("src_xy_excursion_m", excursion, XY_EXCURSION_MAX_M, excursion <= XY_EXCURSION_MAX_M, "gate",
            "시작 위치 대비 최대 이동. 배포 fabric 세계에는 테이블 박스뿐이다"),
        Row("src_crosses_centre", cross, CROSS_FRAC_MAX, cross <= CROSS_FRAC_MAX, "gate",
            "소스 컵이 수신 쪽 반면으로 넘어간 env 비율. 팔마다 fabric 이 따로 돌아 서로를 모른다"),
        Row("envs", float(g("success").shape[1]), 0.0, True, "context", "play 한 env 수"),
        Row("steps", float(g("success").shape[0]), 0.0, True, "context", "에피소드 길이"),
    ]
    if "fill_level" in getattr(z, "files", []):
        fl = float(np.median(g("fill_level")[-1]))
        rows.append(Row("fill_level_med", fl, 0.0, True, "context", "소스 컵 수위 — 맥락일 뿐 게이트 아님"))
    return rows


def report(rows: list[Row]) -> str:
    out = [f"{'':5} {'지표':<22}{'값':>10}{'기준':>10}   설명"]
    for r in rows:
        if r.kind == "context":
            out.append(f"{'':5} {r.name:<22}{r.value:>10.3f}{'—':>10}   {r.why}")
            continue
        mark = "PASS " if r.ok else "REJECT"
        out.append(f"{mark:<5} {r.name:<22}{r.value:>10.3f}{r.limit:>10.3f}   {r.why}")
    bad = [r.name for r in rows if r.kind == "gate" and not r.ok]
    out.append("")
    out.append("판정: 통과" if not bad else f"판정: 탈락 — {', '.join(bad)}")
    out.append("통과해도 실기 진입은 사용자 영상 판정 + 기울기 워치독 + cross-arm guard 이후다.")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--trace", type=Path, required=True, help="hdgp play.py 가 남긴 trace npz")
    ap.add_argument("--json", type=Path, default=None, help="판정을 JSON 으로도 남긴다")
    args = ap.parse_args(argv)
    try:
        z = np.load(args.trace)
    except (OSError, ValueError) as exc:
        print(f"[ckpt_gate] trace 를 못 읽었다: {exc}")
        return 2
    rows = evaluate(z)
    print(f"[ckpt_gate] {args.trace}")
    print(report(rows))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"trace": str(args.trace),
             "rows": [{"name": r.name, "value": r.value, "limit": r.limit, "ok": r.ok, "kind": r.kind}
                      for r in rows],
             "ok": all(r.ok for r in rows if r.kind == "gate")}, indent=1) + "\n")
    return 0 if all(r.ok for r in rows if r.kind == "gate") else 4


if __name__ == "__main__":
    raise SystemExit(main())
