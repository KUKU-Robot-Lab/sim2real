#!/usr/bin/env python3
"""실기 에피소드마다 잡기(aglt) · 놓기(place) 정책 입출력을 sim 기준과 비교하고 콘솔 토글 조합별로 묶는다(읽기 전용).

  nice -n 15 .venv/bin/python deploy/policy_control/tools/sim_real_compare.py --run-dir logs/runs/261116 [--episode 150115_cocktail_1c_n07b]

회차(logs/runs/<날짜>) 분석의 진입점 하나. 잡기 기준 = sim 결정론 trace(정책 → 폴더 표 config/sim_real_traces.json),
놓기 = place_run_report 의 학습 시작 뱅크 비교, 기울이기 = tilt_spout_report. 잡기 · 놓기는 정책 재현 시험(policy_parity)도 한다.
분포 비교: 대각 Mahalanobis + 칸별 KS · Wasserstein-1(sim 표준편차 단위), KS 순 어긋난 칸. 기준 N < 500 이면 경고.
결과: logs/analysis/sim_real/<날짜>/<에피소드>.json · summary.md(토글 문자열별 표).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import place_run_report as PR
import policy_parity as PP
import run_bag as RB
import tilt_spout_report as TS
from policy_control import rh_aglt as A
from policy_control import rh_place as RP
from policy_control.episode_spec import format_toggles

NO_TOGGLES = "기본(미기록)"
Z_BIG = PR.Z_BIG
SD_FLOOR = 1e-6       # sim 에서 상수인 칸 → z 대신 절대 차이
N_MIN = 500           # 기준(sim trace · 뱅크) 표본이 이보다 적으면 분포 비교 경고
SIM_TABLE = RB.SIM2REAL / "deploy/policy_control/config/sim_real_traces.json"


def sim_folder(policy: str) -> Path | None:
    """정책 → sim trace 폴더(표에 없으면 None = sim 기준 없음)."""
    rel = json.loads(SIM_TABLE.read_text(encoding="utf-8")).get(policy)
    return Path.home() / rel if rel else None


_SIM: dict = {}


def toggles_label(meta: dict) -> str:
    t = meta.get("toggles")
    return format_toggles(t) if t else NO_TOGGLES


def sim_pool(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """s*/trace.npz 의 alive 스텝 obs(N,94) · act(N,13)."""
    if folder not in _SIM:
        O, U = [], []
        for f in sorted(folder.glob("s*/trace.npz")):
            b = np.load(f)
            m = b["alive"].astype(bool)
            O.append(b["obs"][m])
            U.append(b["act"][m])
        if not O:
            raise FileNotFoundError(f"sim trace 없음 {folder}")
        _SIM[folder] = (np.concatenate(O), np.concatenate(U))
    return _SIM[folder]


def tick_compare(S: np.ndarray, R: np.ndarray, names: list[str]) -> dict:
    """sim 분포 S(N,D) 대비 실기 틱 R(T,D): 칸별 |z|>3 틱 비율, 대각 Mahalanobis(√mean z²) 중앙값, 큰 칸."""
    mu, sd = S.mean(0), S.std(0)
    const = sd < SD_FLOOR
    z = np.where(const, 0.0, (R - mu) / np.where(const, 1.0, sd))
    frac = (np.abs(z) > Z_BIG).mean(0)
    med = np.median(np.abs(z), 0)
    order = [k for k in np.argsort(-med) if not const[k]][:6]
    var = np.flatnonzero(~const)
    ks = {k: stats.ks_2samp(S[:, k], R[:, k]).statistic for k in var}
    w1 = {k: stats.wasserstein_distance(S[:, k], R[:, k]) / sd[k] for k in var}
    ks_top = sorted(var, key=lambda k: -ks[k])[:6]
    return dict(n_tick=int(len(R)), n_sim=int(len(S)), small_ref=bool(len(S) < N_MIN),
                ks_top=[(names[k], round(float(ks[k]), 2), round(float(w1[k]), 2)) for k in ks_top],
                ks_max=round(float(max(ks.values(), default=0.0)), 2), w1_max=round(float(max(w1.values(), default=0.0)), 2),
                maha_med=round(float(np.median(np.sqrt((z ** 2).mean(1)))), 2),
                n_cell_z3=int((frac > 0.5).sum()),                      # 틱 절반 넘게 |z|>3 인 칸 수
                top=[(names[k], round(float(med[k]), 1), round(float(frac[k]), 2)) for k in order],
                sim_const_absdiff=sorted((names[k], round(float(np.abs(R[:, k] - mu[k]).max()), 4))
                                         for k in np.flatnonzero(const)))


def cell_names(layout) -> list[str]:
    return [f"{n}[{i}]" for n, w in layout for i in range(w)]


def grasp_node(w: dict, bag: Path, ep_yaml: dict) -> dict:
    role = w["enter"].get("policy_name")
    pol = (ep_yaml.get("policies") or {}).get(role) or {}
    policy = pol.get("handoff") or pol.get("policy", "")
    side = pol.get("side") or ("right" if role.endswith("_r") else "left")
    r = {"node": w["id"], "stage": "grasp", "policy": policy, "notes": []}
    end = w["end"] or {}
    r["outcome"] = end.get("result")
    r["reason"] = str(end.get("termination_reason") or "")[:80]
    t0, t1 = w["t0"], (w["t1"] or w["t0"] + 600) + 1.0
    st, ot, at = f"/policy_control/status/rh_aglt_node_{side}", f"/policy_control/{side}/obs", f"/policy_control/{side}/policy_action"
    M = RB.read(RB.db_of(bag, "arm"), [st, ot], t0 - 5.0, t1)
    P = RB.read(RB.db_of(bag, "pd"), [at], t0 - 5.0, t1)
    if not P[at]:   # aglt 는 설계상 policy_action 을 안 낸다(pour_fj 계열 전용) → /action = 정책 출력 13 칸(clip_actions 클립 뒤 · 디코더 ±1 클립 전)
        at = f"/policy_control/{side}/action"
        P = RB.read(RB.db_of(bag, "arm"), [at], t0 - 5.0, t1)
        r["notes"].append("act = /action(aglt 정책 출력, clip_actions 뒤 · 디코더 클립 전)")
    stat = [(t, json.loads(m.data)) for t, m in M[st]]
    # handoff 뒤(approach_grasp.phase == policy · 접근 없으면 running 전부)만 — sim trace 가 그 정책 구간
    run = [t for t, d in stat if d.get("phase") == "running" and (d.get("approach_grasp") or {}).get("phase", "policy") == "policy"]
    if stat:
        last = stat[-1][1]
        r.update(freeze_rule=last.get("freeze_rule"), lift_dz=last.get("lift_dz"), lift_steps=last.get("lift_steps"))
    if not run:
        r["notes"].append("handoff 뒤 running 구간 없음")
        return r
    a, b = min(run), max(run) + 0.1
    c = A.load_contract(PR.POLICIES / policy / "rh_aglt_contract.json")
    r["parity"] = PP.parity(c, [x for x in M[ot] if a <= x[0] <= b], [x for x in P[at] if a <= x[0] <= b])
    folder = sim_folder(policy)
    if folder is None:
        r["notes"].append(f"sim 기준 없음({policy})")
        return r
    O = np.array([list(m.data) for t, m in M[ot] if a <= t <= b])
    U = np.array([list(m.data) for t, m in P[at] if a <= t <= b])
    S_o, S_u = sim_pool(folder)
    if len(O) and O.shape[1] == S_o.shape[1]:
        r["obs"] = tick_compare(S_o, O, cell_names(A.actor_layout(c)))
        names = cell_names(A.actor_layout(c))
        hq = [k for k, n in enumerate(names) if n.startswith("hand_q[")]
        r["hand_q_real_end"] = np.round(O[-1, hq], 3).tolist()
    else:
        r["notes"].append(f"obs 틱 {len(O)} · 폭 {O.shape[1] if len(O) else '-'} ≠ sim {S_o.shape[1]}")
    if len(U) and U.shape[1] == S_u.shape[1]:
        r["action"] = tick_compare(S_u, U, [f"a[{i}]" for i in range(U.shape[1])])
    else:
        r["notes"].append(f"action 틱 {len(U)}")
    return r


def place_parity(w: dict, bag: Path, policy: str) -> dict:
    c = RP.load_contract(PR.POLICIES / policy / "rh_place_contract.json")
    side = {"r": "right", "l": "left"}.get(c.side().side, c.side().side)
    base = f"/policy_control/{side}_place"
    t1 = (w["t1"] if w["t1"] is not None else w["t0"] + 600) + 1.0
    P = RB.read(RB.db_of(bag, "pd"), [f"{base}/obs", f"{base}/policy_action"], w["t0"], t1)
    return PP.parity(c, P[f"{base}/obs"], P[f"{base}/policy_action"])


def place_rows(ep_dir: Path, bag: Path, evs: list[dict]) -> list[dict]:
    rep = PR.episode(ep_dir, bag, [])
    wins = RB.node_windows(evs, lambda e: str(e.get("policy_name") or "").startswith("rh_place"))   # PR.episode 와 같은 창 · 같은 순서
    rows = []
    for w, n in zip(wins, rep["nodes"]):
        n = {"stage": "place", **n}
        if n.get("bank_n") is not None and n["bank_n"] < N_MIN:
            n["notes"] = [*n.get("notes", []), f"뱅크 N {n['bank_n']} < {N_MIN}"]
        try:
            n["parity"] = place_parity(w, bag, n["policy"])
        except (FileNotFoundError, KeyError, ValueError) as e:
            n["parity"] = {"note": f"재현 못 함 {type(e).__name__}: {e}"}
        rows.append(n)
    return rows


def tilt_rows(ep_dir: Path, bag: Path) -> list[dict]:
    rep = TS.episode(ep_dir, bag, [])
    return [{"stage": "tilt", **n} for n in rep["nodes"]]


def episode(ep_dir: Path, roots: list[Path]) -> dict:
    evs = RB.events(ep_dir)
    meta, ep_yaml, _ = RB.episode_ctx(ep_dir)
    out = {"episode_dir": str(ep_dir), "episode": meta.get("episode"), "toggles": toggles_label(meta), "nodes": [], "notes": []}
    try:
        bag = RB.find_bag(ep_dir, roots)
    except FileNotFoundError as e:
        out["notes"].append(f"bag 없음: {e}")
        return out
    out["bag"] = str(bag)
    for w in RB.node_windows(evs, lambda e: str(e.get("policy_name") or "").startswith("rh_aglt")):
        try:
            out["nodes"].append(grasp_node(w, bag, ep_yaml))
        except (FileNotFoundError, KeyError, ValueError) as e:
            out["nodes"].append({"node": w["id"], "stage": "grasp", "notes": [f"{type(e).__name__}: {e}"]})
    for tag, fn in (("place", lambda: place_rows(ep_dir, bag, evs)), ("tilt", lambda: tilt_rows(ep_dir, bag))):
        try:
            out["nodes"] += fn()
        except (FileNotFoundError, KeyError, ValueError) as e:
            out["notes"].append(f"{tag}: {type(e).__name__}: {e}")
    return out


def _f(x, nd: int = 1) -> str | None:
    return None if x is None else f"{x:.{nd}f}"


def row(ep: dict, n: dict) -> list:
    notes = "; ".join(n.get("notes") or [])
    if n["stage"] == "grasp":
        o, u = n.get("obs") or {}, n.get("action") or {}
        top = " ".join(f"{a}:{z}" for a, z, _ in (o.get("top") or [])[:3])
        ks = " ".join(f"{a}:{k}/{w}" for a, k, w in (o.get("ks_top") or [])[:3])
        if o.get("small_ref"):
            notes = f"⚠ sim N {o['n_sim']} < {N_MIN}; " + notes
        return [ep["name"], n["node"], "grasp", n.get("policy"), o.get("n_tick"), o.get("maha_med"), o.get("n_cell_z3"), top, ks,
                u.get("maha_med"), u.get("ks_max"), PP.fmt(n.get("parity")), f"{n.get('outcome')} lift_dz={n.get('lift_dz')}", notes]
    if n["stage"] == "tilt":
        res = (f"최대 {_f(n.get('max_tilt_deg'))}° · 끝↔입구 FP++ 수평 최대 {_f(n.get('lip_fpp_xy_max'))} mm · "
               f"높이 최소 {_f(n.get('lip_fpp_dz_min'))} mm · T_in {n.get('tin_source')}")
        return [ep["name"], n["node"], "tilt", "tilt", None, None, None, None, None, None, None, None, res, notes]
    top = " ".join(f"{a}:{z}" for a, z in (n.get("top_z") or [])[:3])
    return [ep["name"], n["node"], "place", n.get("policy"), n.get("policy_ticks"), None, n.get("n_z3"), top, None, None, None,
            PP.fmt(n.get("parity")), f"{n.get('outcome')} holder={n.get('cup_in_holder')}", notes]


HEAD = ["에피소드", "노드", "단계", "정책", "틱", "obs Maha 중앙", "obs |z|>3 칸", "obs 큰 칸(|z| 중앙)",
        "obs KS 큰 칸(KS/W1σ)", "act Maha 중앙", "act KS 최대", "재현 max|Δ|", "결과", "메모"]


def summary(eps: list[dict]) -> str:
    lines = []
    for tog in sorted({e["toggles"] for e in eps}):
        lines += [f"## 토글 {tog}", "", "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
        for e in (e for e in eps if e["toggles"] == tog):
            lines += ["| " + " | ".join("" if v is None else str(v) for v in row(e, n)) + " |" for n in e["nodes"]]
            lines += [f"| {e['name']} | - | - |" + " |" * (len(HEAD) - 4) + f" {x} |" for x in e["notes"]]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="logs/runs/<날짜>")
    ap.add_argument("--episode", nargs="*", default=None, help="episode/ 아래 폴더 이름(없으면 전부)")
    ap.add_argument("--bag-root", type=Path, action="append", default=None)
    a = ap.parse_args(argv)
    run = a.run_dir.resolve()
    dirs = [run / "episode" / n for n in a.episode] if a.episode else sorted(p for p in (run / "episode").iterdir() if p.is_dir())
    out = RB.SIM2REAL / "logs" / "analysis" / "sim_real" / run.name
    out.mkdir(parents=True, exist_ok=True)
    eps = []
    for d in dirs:
        if not (d / "meta.json").is_file() or not any(d.glob("episode_*.jsonl")):
            print(f"[skip] {d.name}: meta.json · episode_*.jsonl 없음", file=sys.stderr)
            continue
        e = episode(d, a.bag_root or RB.default_roots(d))
        e["name"] = d.name
        (out / f"{d.name}.json").write_text(json.dumps(e, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        eps.append(e)
    md = summary(eps)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {out.relative_to(RB.SIM2REAL)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
