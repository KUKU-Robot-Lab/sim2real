#!/usr/bin/env python3
"""board_jog 접촉 기록 → 팔 관절 영점 오프셋 후보 맞춤(읽기만, 로봇을 움직이지 않는다).

11.15 왼팔 j6 −6.13° 를 찾은 분석을 도구로 옮겼다(정리 sim2real logs/runs/261115/fk_zero_summary.md).
모델: 실제 손끝 = FK(q_읽기 + dq) — dq 는 관절 1~2 개만 0 이 아니다. 검지 끝(T_tip_fk 를 만든 같은 Kin)과
코너 위치(board_pose)의 차를 최소제곱(Gauss-Newton)으로 줄인다. 조합 전부를 jog RMS 로 줄 세우고, 위 몇 개는
한 점 빼고 맞춤(LOO)으로 흔들림을 본다. 자세가 비슷한 점만 있으면 여러 후보가 똑같이 맞는다 — 손목(j5)을 크게 바꾼 점이나
자세가 아주 다른 접촉(--z-contact, 예: 주먹 새끼 상판)을 같이 넣어야 갈린다.

    python3 deploy/policy_control/tools/fit_board.py --side left logs/runs/261115/board_jog/left_test2.json \\
        --z-contact logs/runs/261115/fk_fit/fist_left_1115.json
    # 영점을 바꾸기 전에 찍은 기록을 지금 영점 기준으로 볼 때: --q-shift 6:-6.13 (새 읽기 = 옛 + 각도)

--z-contact JSON: {"q_arm": [7 rad], "q_hand": [6 rad], "links": ["l_hl_pinky_2", ...], "z_err_mm": 1.8}
  = 그 자세에서 links 원점 중 가장 낮은 것의 FK 높이 − 실제 높이(재 둔 값). 후보 dq 로 이 값이 얼마로 바뀌는지 같이 맞춘다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_jog import Kin  # noqa: E402

Z_WEIGHT = 1.0   # 접촉 높이 잔차 한 개의 무게(jog 점 하나의 xyz 한 성분과 같게)
AMBIGUOUS_MM = 1.0   # 1·2위 jog RMS 차가 이보다 작으면 후보가 갈리지 않는다고 알린다(손 맞춤 흔들림 2~4 mm)


def load_points(paths: list[Path], shift: np.ndarray) -> tuple[list, dict]:
    """→ ([(이름, q_arm, q_hand, 코너 xyz)], 첫 파일 메타). 지운 점은 뺀다."""
    pts, meta = [], None
    for p in paths:
        d = json.loads(Path(p).read_text())
        meta = meta or d
        for k in ("side", "urdf", "hand_q"):     # 다른 팔 · 손 자세 기록을 섞으면 첫 파일 FK 로 틀리게 맞춘다
            if d.get(k) != meta.get(k):
                raise SystemExit(f"{p}: {k} 가 첫 파일과 다르다 — 같은 팔 · 같은 손 자세 기록만")
        for r in d["points"]:
            if r.get("deleted"):
                continue
            tgt = r.get("corner_xyz_base_display") or r["target_xyz_base_display"]
            pts.append((f"{Path(p).stem}:{r['target_id']}", np.asarray(r["q_arm"], float) + shift,
                        np.asarray(r["q_hand"], float), np.asarray(tgt, float)))
    if not pts or meta is None:
        raise SystemExit("기록 점이 없다")
    return pts, meta


class Problem:
    def __init__(self, kin: Kin, pts: list, z: dict | None):
        self.kin, self.pts, self.z = kin, pts, z
        if z:
            self.z0 = self._z_low(np.zeros(7))

    def _z_low(self, dq: np.ndarray) -> float:
        q = np.asarray(self.z["q_arm"], float) + dq
        T = self.kin.tree.transforms(self.z["links"], self.kin._q(q, np.asarray(self.z["q_hand"], float)))
        return min(float(T[n][2, 3]) for n in self.z["links"])

    def jog_res(self, dq: np.ndarray, pts: list | None = None) -> np.ndarray:
        return np.concatenate([self.kin.tip(q + dq, h)[:3, 3] - t for _, q, h, t in (pts or self.pts)])

    def z_res(self, dq: np.ndarray) -> float | None:
        return None if not self.z else self.z["z_err_mm"] / 1000 + self._z_low(dq) - self.z0

    def fit(self, idx: list[int], pts: list | None = None, iters: int = 20) -> np.ndarray:
        def res(dq):
            r = self.jog_res(dq, pts)
            return np.r_[r, Z_WEIGHT * self.z_res(dq)] if self.z else r
        dq = np.zeros(7)
        for _ in range(iters):
            r = res(dq)
            J = np.zeros((len(r), len(idx)))
            for k, j in enumerate(idx):
                e = np.zeros(7)
                e[j] = 1e-5
                J[:, k] = (res(dq + e) - r) / 1e-5
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            dq[idx] += step
            if np.abs(step).max() < 1e-7:
                break
        return dq


def rms_mm(r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(r ** 2)) * 1000)


def report(prob: Problem, max_joints: int, top: int) -> list[dict]:
    r0 = prob.jog_res(np.zeros(7)).reshape(-1, 3) * 1000
    print(f"점 {len(prob.pts)}개 · 지금 영점 그대로: 평균 {np.round(r0.mean(0), 1).tolist()} mm, RMS {rms_mm(r0 / 1000):.1f} mm"
          + (f", 접촉 z {prob.z['z_err_mm']:+.1f} mm" if prob.z else ""))
    for (name, *_), e in zip(prob.pts, r0):
        print(f"  {name:28s} {np.round(e, 1).tolist()}")
    rows = []
    for n in range(1, max_joints + 1):
        for idx in itertools.combinations(range(7), n):
            dq = prob.fit(list(idx))
            z = prob.z_res(dq)
            rows.append({"joints": [i + 1 for i in idx], "deg": np.round(np.degrees(dq[list(idx)]), 2).tolist(),
                         "jog_rms_mm": round(rms_mm(prob.jog_res(dq)), 2), "z_mm": None if z is None else round(z * 1000, 1),
                         "dq": dq})
    # 접촉 높이가 있으면 jog RMS² + z² 로, 없으면 jog RMS 로 줄 세운다
    rows.sort(key=lambda r: r["jog_rms_mm"] ** 2 + (r["z_mm"] or 0.0) ** 2)
    print("후보(관절 · 오프셋 ° · jog RMS mm · 접촉 z mm · 한 점 빼기 최대 mm):")
    for r in rows[:top]:
        loo = []
        for i in range(len(prob.pts)):
            rest = prob.pts[:i] + prob.pts[i + 1:]
            dq = prob.fit([j - 1 for j in r["joints"]], rest)
            loo.append(np.linalg.norm(prob.jog_res(dq, [prob.pts[i]])) * 1000)
        r["loo_max_mm"] = round(max(loo), 1)
        print(f"  j{r['joints']} {r['deg']} · {r['jog_rms_mm']:.1f} · {r['z_mm']} · {r['loo_max_mm']}")
    if len(rows) > 1 and rows[1]["jog_rms_mm"] - rows[0]["jog_rms_mm"] < AMBIGUOUS_MM:
        print(f"⚠ 1·2위 jog RMS 차 {rows[1]['jog_rms_mm'] - rows[0]['jog_rms_mm']:.1f} mm < {AMBIGUOUS_MM} — 후보가 갈리지 않는다. "
              "자세가 다른 점(j5 크게 바꾸기)이나 --z-contact 를 더 넣고 영점을 옮긴다")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("jsons", nargs="+", type=Path, help="board_jog --out 기록(같은 팔)")
    ap.add_argument("--side", choices=["right", "left"], required=True)
    ap.add_argument("--q-shift", action="append", default=[], help="j:deg — 모든 점 읽기에 더한다(영점 바꾸기 전 기록용)")
    ap.add_argument("--z-contact", type=Path, help="자세가 다른 접촉 높이 제약 JSON(위 설명)")
    ap.add_argument("--max-joints", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(argv)
    shift = np.zeros(7)
    for s in a.q_shift:
        try:
            j, deg = s.split(":")
            j = int(j)
            assert 1 <= j <= 7
            shift[j - 1] += np.radians(float(deg))
        except (ValueError, AssertionError):
            ap.error(f"--q-shift {s!r} — j:deg (j 1~7)")
    pts, meta = load_points(a.jsons, shift)
    if meta.get("side") not in (None, a.side):
        raise SystemExit(f"기록 팔 {meta.get('side')} ≠ --side {a.side}")
    kin = Kin(Path(meta["urdf"]), a.side, np.asarray(meta["hand_q"], float))
    z = json.loads(a.z_contact.read_text()) if a.z_contact else None
    report(Problem(kin, pts, z), a.max_joints, a.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
