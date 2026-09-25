#!/usr/bin/env python3
"""**지금 손 자세에서** 손가락을 접는다 — 손가락끼리 닿지 않는 만큼만, 봉투 구 안에 들 때까지.

    ROS_DOMAIN_ID=126 python3 deploy/policy_control/tools/plan_hand_fold.py --side left          # 실측에서
    python3 deploy/policy_control/tools/plan_hand_fold.py --side left --q 'l_hj_index_2=0.3,...'  # 오프라인

09.23 사용자: "손가락들이 서로 충돌이 일어나지 않게 모을 순 없는건가? 현재 JOINT STATE 기반해서."

고정된 주먹(pd yaml `hand_path_pose`) 하나로 보내면 못 닿는 손가락이 있을 때(오른손 새끼 error 409,
왼손 엄지가 검지에 걸림) 그 자세에 영영 도달하지 못한다. 여기서는 **실측에서 출발해 손가락마다 따로**
접을 수 있는 만큼 접고, 결과가 봉투 구 안인지 숫자로 말한다.

★충돌 판정은 **원본 삼각 메쉬**로 한다. 볼록 껍질은 손에서 못 쓴다 — 나란히 붙은 손가락의 껍질이 서로를
삼켜서 **편 손에서도** 1.8~4.5 mm 겹쳤다고 나온다(09.23 실측). 껍질로 게이트를 걸면 전부 막히거나 전부 통과한다.
꼭짓점 구름 최단거리라 면-면 실제 거리보다 조금 낙관적이다(주먹에서 4.6 mm vs 면 기준 4.19 mm).

접는 방향은 pd yaml 의 주먹 자세다 — 그 자세를 **목표가 아니라 방향**으로 쓴다(손가락마다 0..1 배).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SIM2REAL = HERE.parents[2]
SAMPLE = SIM2REAL / "deploy" / "s2r_console" / "tools" / "sample_joints.py"
PD_CONFIG = SIM2REAL / "deploy" / "policy_control" / "config" / "pd_dg5f_m_short.yaml"
FINGERS = ("index", "middle", "ring", "pinky", "thumb")     # 엄지를 마지막에 — 남들이 자리를 잡은 뒤 넣는다
#: 손가락끼리 이만큼은 떨어져 있어야 한다 [m]. 주먹에서 가장 가까운 쌍이 약지↔새끼 0.95 mm 라 그보다 낮게 잡는다
#: (그 쌍은 나란히 붙어 도는 구조다 — 더 벌리려 하면 접지를 못한다).
MARGIN_M = 0.0008
#: 두 링크의 경계구가 이보다 멀면 메쉬를 보지 않는다 [m] — 손 하나에 링크 쌍이 130 개다.
PRUNE_M = 0.03
STEPS = 6                                                   # 손가락마다 이등분 탐색 횟수


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


W = _load("home_path_world")


def measure() -> dict[str, float]:
    out = subprocess.run([sys.executable, str(SAMPLE), "--seconds", "0.6"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"✗ 관절 상태를 못 읽었다: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])["q"]


def parse_q(text: str) -> dict[str, float]:
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            k, _, v = item.partition("=")
            out[k.strip()] = float(v)
    return out


def finger_of(link: str) -> str:
    """`r_hl_index_2` → `index`. 손가락이 아니면(손바닥 · 어댑터) 빈 문자열."""
    if "_hl_" not in link:
        return ""
    name = link.split("_hl_")[1].split("_")[0]
    return name if name in FINGERS else ""


class HandGeometry:
    """손 링크의 **원본 메쉬** 꼭짓점 — 링크 프레임에 한 번만 쌓아 두고 자세마다 옮긴다."""

    def __init__(self, links, joints, side: str) -> None:
        import trimesh

        self.links, self.joints, self.side = links, joints, side
        pre = side[0] + "_hl_"
        self.verts: dict[str, np.ndarray] = {}
        for name, link in links.items():
            if pre not in name:
                continue
            pts = []
            for Tc, kind, data in link.collisions:
                if kind != "mesh":
                    continue
                path, scale = data
                m = trimesh.load(str(path), force="mesh")
                v = np.asarray(m.vertices, float) * np.asarray(scale, float)
                T = np.asarray(Tc, float)
                pts.append(v @ T[:3, :3].T + T[:3, 3])
            if pts:
                self.verts[name] = np.vstack(pts)
        #: 볼 쌍 — 다른 손가락끼리만(같은 손가락의 이웃 마디는 접으면 당연히 붙는다).
        names = sorted(self.verts)
        self.pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                      if finger_of(a) and finger_of(b) and finger_of(a) != finger_of(b)]
        self.radius = {n: float(np.linalg.norm(v - v.mean(0), axis=1).max()) for n, v in self.verts.items()}

    def placed(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        T = W.link_transforms(self.links, self.joints, q)
        return {n: v @ T[n][:3, :3].T + T[n][:3, 3] for n, v in self.verts.items()}

    def worst_pair(self, q: dict[str, float]) -> tuple[float, str, str]:
        """가장 가까운 **다른 손가락** 쌍의 거리 [m] 와 그 쌍."""
        pts = self.placed(q)
        mids = {n: v.mean(0) for n, v in pts.items()}
        best, wa, wb = np.inf, "", ""
        for a, b in self.pairs:
            gap = float(np.linalg.norm(mids[a] - mids[b])) - self.radius[a] - self.radius[b]
            if gap > PRUNE_M or gap > best:
                continue
            d = float(cKDTree(pts[b]).query(pts[a])[0].min())
            if d < best:
                best, wa, wb = d, a, b
        return best, wa, wb


def fold(geo: HandGeometry, now: dict[str, float], target: dict[str, float], margin: float) -> dict[str, float]:
    """손가락마다 `now → target` 을 얼마나 갈 수 있는지 이등분으로 찾는다. 앞서 정한 손가락을 그대로 두고 본다."""
    q = dict(now)
    for finger in FINGERS:
        keys = [k for k in target if f"_hj_{finger}_" in k]
        if not keys:
            continue
        def at(t: float) -> dict[str, float]:
            return {**q, **{k: now.get(k, target[k]) + (target[k] - now.get(k, target[k])) * t for k in keys}}
        lo, hi = 0.0, 1.0
        if geo.worst_pair(at(hi))[0] >= margin:
            q = at(hi)
            print(f"  {finger:<7} 100 % (끝까지)")
            continue
        for _ in range(STEPS):
            mid = (lo + hi) / 2
            if geo.worst_pair(at(mid))[0] >= margin:
                lo = mid
            else:
                hi = mid
        d, a, b = geo.worst_pair(at(lo))
        print(f"  {finger:<7} {lo:4.0%}      (더 접으면 {a.split('_hl_')[1]}↔{b.split('_hl_')[1]} 가 {margin*1000:.1f} mm 아래)"
              if lo < 1.0 else f"  {finger:<7} 100 %")
        q = at(lo)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--q", default=None, help="실측 대신 쓸 손 자세 CSV — 없으면 로봇에서 읽는다")
    ap.add_argument("--pd-config", type=Path, default=PD_CONFIG, help="접는 **방향**을 주는 주먹 자세")
    ap.add_argument("--sphere", type=float, default=W.HAND_SPHERE_DEFAULT, help="봉투 구 반지름 [m]")
    ap.add_argument("--margin-mm", type=float, default=MARGIN_M * 1000, help="손가락끼리 최소 거리 [mm]")
    ap.add_argument("--frozen", default="", help="움직이지 않는 관절 CSV(고장 등) — 실측값에 묶는다")
    ap.add_argument("--out", type=Path, default=None, help="찾은 자세를 CSV 로 쓴다(--hand-q 에 그대로 넣는다)")
    ap.add_argument("--write-pd", action="store_true",
                    help="찾은 자세를 pd 설정 3개의 hand_path_pose 로 쓴다 — **봉투 구 안일 때만**. "
                         "구 안이면 저장 경로는 그대로 쓸 수 있다(경로는 자세가 아니라 구로 검사했다)")
    args = ap.parse_args(argv)

    pose = (yaml.safe_load(args.pd_config.read_text()).get("hand_path_pose") or {}).get(args.side)
    if not pose:
        raise SystemExit(f"✗ {args.pd_config} 에 hand_path_pose.{args.side} 가 없다")
    target = {str(k): float(v) for k, v in pose.items()}
    q_all = parse_q(args.q) if args.q else measure()
    pre = f"{args.side[0]}_hj_"
    now = {k: float(q_all.get(k, 0.0)) for k in target}
    missing = [k for k in target if k not in q_all]
    if missing and not args.q:
        raise SystemExit(f"✗ 손 관절 상태 없음 {missing[:4]} — 손 드라이버가 떠 있는가")
    frozen = {k.strip() for k in args.frozen.split(",") if k.strip()}
    if frozen:
        target = {k: (now[k] if k in frozen else v) for k, v in target.items()}
        print(f"[fold] 묶인 관절 {sorted(frozen)} — 실측값에 고정한다")

    links, joints, _ = W.parse_urdf(W.URDF_DEFAULT)
    geo = HandGeometry(links, joints, args.side)
    margin = args.margin_mm / 1000.0
    r0 = W.hand_radius(links, joints, {k: v for k, v in q_all.items() if pre in k} or now, args.side)[0]
    d0, a0, b0 = geo.worst_pair(now)
    print(f"[fold] {args.side} 지금 — 봉투 {r0 * 100:.1f} cm · 손가락 최소 거리 {d0 * 1000:.1f} mm "
          f"({a0.split('_hl_')[1]}↔{b0.split('_hl_')[1]})")
    print(f"[fold] 접는다(손가락끼리 {args.margin_mm:.1f} mm 이상 유지):")
    q = fold(geo, now, target, margin)

    r, worst = W.hand_radius(links, joints, q, args.side)
    d, a, b = geo.worst_pair(q)
    print(f"[fold] 결과 — 봉투 {r * 100:.1f} / {args.sphere * 100:.1f} cm ({worst.split('_hl_')[1]}) · "
          f"손가락 최소 거리 {d * 1000:.1f} mm ({a.split('_hl_')[1]}↔{b.split('_hl_')[1]})")
    csv = ",".join(f"{k}={v:.4f}" for k, v in sorted(q.items()))
    if args.out:
        args.out.write_text(csv + "\n")
        print(f"[fold] 저장 {args.out}")
    if r > args.sphere:
        print(f"✗ 접어도 봉투 구를 {(r - args.sphere) * 100:.1f} cm 넘는다 — 그 손으로는 저장 경로를 쓸 수 없다. "
              f"걸린 손가락을 고치거나 --hand-sphere {r + 0.005:.3f} 로 경로를 다시 계획할 것", file=sys.stderr)
        return 1
    print("✓ 봉투 구 안 — 이 자세로 팔을 옮겨도 된다")
    print(f"  --hand-q '{csv}'")
    if args.write_pd:
        cap = _load("capture_hand_pose")
        lim = cap.limits()
        safe, moved = cap.clamped(q, lim, cap.MARGIN)
        if moved:
            print(f"[fold] 한계 안쪽 {cap.MARGIN} rad 로 물린 관절 {moved}")
        r2 = W.hand_radius(links, joints, safe, args.side)[0]
        if r2 > args.sphere:
            raise SystemExit(f"✗ 한계로 물리니 봉투를 넘는다({r2*100:.1f} cm) — 쓰지 않는다")
        other = "left" if args.side == "right" else "right"
        keep = {str(k): float(v) for k, v in
                ((yaml.safe_load(args.pd_config.read_text()).get("hand_path_pose") or {}).get(other) or {}).items()}
        wrote = cap.write_configs({args.side: safe, other: keep})
        print(f"[fold] pd 설정에 기록 {wrote} — 저장 경로는 그대로다(구 안이라 다시 계획할 필요가 없다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
