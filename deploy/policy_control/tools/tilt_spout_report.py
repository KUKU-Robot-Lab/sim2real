#!/usr/bin/env python3
"""실기 tilt 노드(tilt_blue · tilt_pink) 주둥이 끝 ↔ shaker 입구 오프라인 표 — 에피소드 폴더 + bag 만 읽는다(로봇 · GPU 없음).

    .venv/bin/python deploy/policy_control/tools/tilt_spout_report.py --episode-dir logs/runs/261116/episode/100033_cocktail_1c_e4000 [더 …]

점 정의 = plan_tilt_traj 와 같다(그 노드 tilt_<id>/check.json 의 T_in_r · T_in_l · 자산 setup_<컵>.npz geo):
  붓는 컵 입구 중심 = FK_r(실측 q) · T_in_r 원점 + src_mouth_z · 컵 z (plan_tilt_traj.mouth_xy_mm),
  주둥이 끝(lip) = 입구 중심 + src_lip_r · 내리막 방향(hdgp tilt_traj/plan.py src_origin 의 d),
  shaker 입구 = FP++ live 원점 + (rcv_z_off + rcv_mouth_z) · z  ·  왼손 FK_l · T_in_l(원점 + rcv_z_off 틀) + rcv_mouth_z · z.
가장 많이 기울어진 구간(기울기 ≥ 최대 − MAX_BAND_DEG)에서 수평 거리 · 높이(주둥이 끝 − 입구)를 낸다.
출력: stdout markdown + logs/analysis/tilt_spout/<날짜>_<에피소드>.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import run_bag as RB
from policy_control import tin_live as TL

MAX_BAND_DEG = 3.0       # 최대 기울기 유지 구간(t_hold 2 s)을 덮는 폭
PATH_EVERY = 6           # JSON 경로 표본 60 Hz → 10 Hz
EZ = np.array([0.0, 0.0, 1.0])


def code_lines() -> list[str]:
    """tilt 계획이 이 런 실측 T_in 으로 다시 도는지 — episode_ros 의 해당 줄(작업 트리 그대로)."""
    src = RB.SIM2REAL / "deploy/policy_control/policy_control/episode_ros.py"
    pats = [r"_measure_tin\(node, tin_objs", r"why = self\._planner_tin\(node", r'argv \+= \["--tin-json"']
    out = []
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if any(re.search(p, line) for p in pats):
            out.append(f"{src.relative_to(RB.SIM2REAL)}:{i}: {line.strip()}")
    return out


def spout(C: np.ndarray, mouth_z: float, lip_r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C (N,4,4) 컵 자세 → (입구 중심, 주둥이 끝, 기울기 °). 내리막 d = −(ez − (ez·z)z)/|…| (서 있으면 0)."""
    z = C[:, :3, 2]
    m = C[:, :3, 3] + mouth_z * z
    h = EZ[None] - z[:, 2:3] * z
    n = np.linalg.norm(h, axis=1, keepdims=True)
    d = -np.divide(h, n, out=np.zeros_like(h), where=n > 1e-9)
    return m, m + lip_r * d, np.degrees(np.arccos(np.clip(z[:, 2], -1, 1)))


def fixed_T(path: Path, key: str) -> np.ndarray:
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))[key]["T"], float)


def node_report(ep_dir: Path, w: dict, bag: Path, live_topics: dict) -> dict:
    nid = w["id"]
    r = {"node": nid, "notes": []}
    chk_p = ep_dir / f"tilt_{nid}" / "check.json"
    if not chk_p.is_file():
        r["notes"].append(f"{chk_p.name} 없음 — 계획 기록이 없다")
        return r
    chk = json.loads(chk_p.read_text(encoding="utf-8"))
    r["planner_tin"] = chk.get("tin")
    r["replanned_with_run_tin"] = str(chk.get("tin", "")).endswith(f"tilt_tin_{nid}.json")
    r["plan_mouth_xy_mm"] = (chk.get("checks", {}).get("mouth_xy_mm") or {}).get("value")
    with np.load(Path(chk["asset"]) / "ref" / f"setup_{chk['cup']}.npz", allow_pickle=False) as S:
        geo = json.loads(str(S["geo_json"]))
    r["geo"] = {k: geo[k] for k in ("src_mouth_z", "src_lip_r", "rcv_mouth_z", "rcv_z_off")}
    T_r, T_l = np.asarray(chk["T_in_r"], float), np.asarray(chk["T_in_l"], float)
    rec = {s: json.loads(p.read_text(encoding="utf-8")) for s in ("right", "left")
           if (p := ep_dir / f"tin_{nid}_{s}.json").is_file()}
    r["tin_source"] = {s: d.get("source") for s, d in rec.items()}
    r["tin_r_z"] = float(T_r[2, 3])
    if w["t1"] is None:
        r["notes"].append("노드 끝 사건 없음")
        return r
    t, Q = RB.arm_series(bag, w["t0"], w["t1"])
    if len(t) < 2:
        r["notes"].append("/joint_states 없음")
        return r
    fk_r, fk_l = TL.palm_fk("right"), TL.palm_fk("left")
    Pr = np.array([fk_r(q) for q in Q["right"]])
    Pl = np.array([fk_l(q) for q in Q["left"]])
    mouth, lip, tilt = spout(Pr @ T_r, geo["src_mouth_z"], geo["src_lip_r"])
    Cl = Pl @ T_l
    inlet_fk = Cl[:, :3, 3] + geo["rcv_mouth_z"] * Cl[:, :3, 2]
    sel = tilt >= tilt.max() - MAX_BAND_DEG
    r.update(max_tilt_deg=float(tilt.max()), window_s=float(sel.sum() / 60.0), n=int(sel.sum()))
    hxy = lambda a, b: np.linalg.norm((a - b)[:, :2], axis=1) * 1e3  # noqa: E731
    d = hxy(lip[sel], inlet_fk[sel])
    dz = (lip[sel, 2] - inlet_fk[sel, 2]) * 1e3
    r.update(lip_fkl_xy_mean=float(d.mean()), lip_fkl_xy_max=float(d.max()), lip_fkl_dz_min=float(dz.min()), lip_fkl_dz_mean=float(dz.mean()),
             mouth_fkl_xy_max=float(hxy(mouth[sel], inlet_fk[sel]).max()))
    obj = (rec.get("left") or {}).get("object", "SHAKER")
    topic = live_topics.get(obj)
    rows = RB.read(RB.db_of(bag, "arm"), [topic], w["t0"] - 30.0, w["t1"])[topic] if topic else []
    inlet_fpp = None
    if rows:
        tr = RB.PoseTrack(rows)
        got = [tr.at(x) for x in t]
        ok = np.array([g is not None for g in got])
        inlet_fpp = np.full((len(t), 3), np.nan)
        age = np.full(len(t), np.nan)
        for k, g in enumerate(got):
            if g is not None:
                inlet_fpp[k] = g[0] + (geo["rcv_z_off"] + geo["rcv_mouth_z"]) * TL.quat_to_R(g[1])[:, 2]
                age[k] = g[2]
        s2 = sel & ok
        if s2.any():
            d = hxy(lip[s2], inlet_fpp[s2])
            dz = (lip[s2, 2] - inlet_fpp[s2, 2]) * 1e3
            r.update(lip_fpp_xy_mean=float(d.mean()), lip_fpp_xy_max=float(d.max()), lip_fpp_dz_min=float(dz.min()),
                     lip_fpp_dz_mean=float(dz.mean()), fpp_age_max_s=float(np.nanmax(age[s2])),
                     fpp_vs_fkl_inlet_mm=float(np.nanmean(np.linalg.norm(inlet_fpp[s2] - inlet_fk[s2], axis=1)) * 1e3))
    else:
        r["notes"].append(f"shaker FP++ live 가 이 노드 창(−30 s 포함)에 없음({topic})")
    # 고정 T_in(tin_fixed) 이었다면 — 같은 실측 관절로 주둥이 끝이 얼마나 달라지나
    fb = (rec.get("right") or {}).get("fallback_file")
    if fb and (RB.SIM2REAL / fb).is_file():
        F_r = fixed_T(RB.SIM2REAL / fb, "T_in_r")
        _, lip_f, _ = spout(Pr @ F_r, geo["src_mouth_z"], geo["src_lip_r"])
        dd = (lip_f[sel] - lip[sel]) * 1e3
        r.update(fixed_tin_r_z=float(F_r[2, 3]), fixed_lip_shift_mm=float(np.linalg.norm(dd, axis=1).mean()),
                 fixed_lip_shift_dz_mm=float(dd[:, 2].mean()), fixed_lip_shift_xy_mm=float(np.linalg.norm(dd[:, :2], axis=1).mean()))
    else:
        r["notes"].append(f"고정 T_in 파일 없음({fb})")
    k = np.arange(0, len(t), PATH_EVERY)
    r["path"] = {"t": (t[k] - t[0]).round(3).tolist(), "tilt_deg": tilt[k].round(2).tolist(), "lip": lip[k].round(4).tolist(),
                 "inlet_fk_l": inlet_fk[k].round(4).tolist(),
                 "inlet_fpp": None if inlet_fpp is None else np.where(np.isfinite(inlet_fpp[k]), inlet_fpp[k].round(4), None).tolist()}
    return r


def episode(ep_dir: Path, bag: Path | None, roots: list[Path]) -> dict:
    evs = RB.events(ep_dir)
    bag = bag or RB.find_bag(ep_dir, roots)
    meta, _, live = RB.episode_ctx(ep_dir)
    wins = RB.node_windows(evs, lambda e: e.get("policy_name") == "tilt" or str(e.get("state_id", "")).startswith("tilt_"))
    plan_logs = [p.name for p in sorted((ep_dir / "tools").glob("*plan_tilt_traj.log"))
                 if "--tin-json" in p.read_text(encoding="utf-8", errors="replace")]
    return {"episode_dir": str(ep_dir), "bag": str(bag), "episode": meta.get("episode"), "plan_logs_with_tin_json": plan_logs,
            "nodes": [node_report(ep_dir, w, bag, live) for w in wins]}


ROWS = [("계획 T_in = 이 런 실측 json", "replanned_with_run_tin"), ("T_in 출처 오른손", ("tin_source", "right")),
        ("T_in 출처 왼손", ("tin_source", "left")), ("T_in_r z m", "tin_r_z"), ("고정 T_in_r z m", "fixed_tin_r_z"),
        ("계획 mouth_xy mm(check)", "plan_mouth_xy_mm"), ("최대 기울기 °", "max_tilt_deg"), ("구간 s", "window_s"),
        ("끝↔입구(FP++) 수평 평균 mm", "lip_fpp_xy_mean"), ("끝↔입구(FP++) 수평 최대 mm", "lip_fpp_xy_max"),
        ("끝−입구(FP++) 높이 최소 mm", "lip_fpp_dz_min"), ("끝−입구(FP++) 높이 평균 mm", "lip_fpp_dz_mean"),
        ("FP++ 나이 최대 s", "fpp_age_max_s"), ("끝↔입구(왼손 FK) 수평 평균 mm", "lip_fkl_xy_mean"),
        ("끝↔입구(왼손 FK) 수평 최대 mm", "lip_fkl_xy_max"), ("끝−입구(왼손 FK) 높이 최소 mm", "lip_fkl_dz_min"),
        ("입구 중심↔입구(왼손 FK) 수평 최대 mm", "mouth_fkl_xy_max"), ("입구 FP++ − 왼손 FK mm", "fpp_vs_fkl_inlet_mm"),
        ("고정 T_in 이면 끝 이동 mm", "fixed_lip_shift_mm"), ("  그중 높이 mm", "fixed_lip_shift_dz_mm"),
        ("  그중 수평 mm", "fixed_lip_shift_xy_mm")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--episode-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--bag-dir", type=Path, default=None, help="에피소드 하나일 때 bag 폴더를 직접")
    ap.add_argument("--bag-root", type=Path, action="append", default=None, help="bag 을 찾을 곳(여럿) — 기본 logs/runs/<날짜>/bags · DATA 백업")
    a = ap.parse_args(argv)
    if a.bag_dir and len(a.episode_dir) > 1:
        ap.error("--bag-dir 는 에피소드 하나에만")
    print("tilt 계획 T_in(코드, 작업 트리):")
    for line in code_lines():
        print("  " + line)
    cols, recs = [], []
    for ep_dir in a.episode_dir:
        ep_dir = ep_dir.resolve()
        rep = episode(ep_dir, a.bag_dir, a.bag_root or RB.default_roots(ep_dir))
        out = RB.out_path("tilt_spout", ep_dir)
        out.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"[tilt] {ep_dir.name} · bag {Path(rep['bag']).name} · --tin-json 계획 로그 {rep['plan_logs_with_tin_json']} → "
              f"{out.relative_to(RB.SIM2REAL)}")
        for r in rep["nodes"]:
            cols.append(f"{ep_dir.parent.parent.name}/{ep_dir.name.split('_')[0]} {r['node']}")
            recs.append(r)
    print()
    cell = lambda r, k: (r.get(k[0]) or {}).get(k[1]) if isinstance(k, tuple) else r.get(k)  # noqa: E731
    print(RB.md_table(cols, [(name, [cell(r, k) for r in recs]) for name, k in ROWS]))
    for col, r in zip(cols, recs):
        for n in r["notes"]:
            print(f"- {col}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
