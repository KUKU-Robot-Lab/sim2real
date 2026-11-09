#!/usr/bin/env python3
"""실험 폴더의 runs/<trial>/events.jsonl 에서 trial별 정지 구간 지표를 뽑아 result/trials.csv 로 쓴다.

창: planned_path_end 기준 0~15 s(유지 전체), 3~13 s(판정 창).
e = (measured - pd_target) * direction [deg]. 음수 = 목표 못 미침, 양수 = 지나침.
사용: analyze_runs.py <실험 폴더> [...]
"""
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "deploy/policy_control"))
from policy_control import stationary_tracking as S  # noqa: E402

DIAG = '"/policy_control/pd_'
TOL = 0.2  # deg, F2 도달 기준
DEAD = 0.03  # deg, 부호 반전 판정 데드밴드(엔코더 한 칸 0.022°보다 크게)


def load(run):
    meta = json.loads((run / "manifest.json").read_text())["metadata"]
    side = meta["side"]
    topic = f"/policy_control/pd_{side}/diagnostics"
    t, q, tgt, ti, tsf, tpd, prot, cont = [], [], [], [], [], [], [], []
    # 11.10~ 실험은 bag(runs/<trial>/bag/…), 그 전은 events.jsonl — S.load_run_events 가 고른다
    for e in S.load_run_events(run, topics=(topic,)):
        d = e["message"]
        if isinstance(d.get("data"), str):
            d = json.loads(d["data"])
        t.append(d["stamp"])
        q.append(d["measured"][:7])
        tgt.append(d["pd_target"][:7])
        ti.append(d["torque_integral"][:7])
        tsf.append(d["torque_static_friction"][:7])
        tpd.append(d["torque_pd"][:7])
        prot.append(d["protected"])
        cont.append(d["contact"])
    return meta, *(np.asarray(x, float) for x in (t, q, tgt, ti, tsf, tpd)), np.asarray(prot), np.asarray(cont)


def plan_trial(exp, tid):
    for p in sorted((exp / "setup" / "rebase").glob(f"{tid}_*/plan/plan.json")) + [exp / "setup/plan/plan.json"]:
        for tr in json.loads(p.read_text())["trials"]:
            if tr["id"] == tid:
                return tr
    raise KeyError(tid)


def reversals(e):
    s = np.sign(e[np.abs(e) > DEAD])
    return int(np.count_nonzero(np.diff(s))) if s.size > 1 else 0


def one(args):
    exp, run = args
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    meta, t, q, tgt, ti, tsf, tpd, prot, cont = load(run)
    tr = plan_trial(exp, run.name.removesuffix("_b"))
    j, dirn = tr["joint"], tr["direction"]
    pe = meta.get("planned_path_end")
    row = dict(trial=run.name, stage=tr["stage"], pose=tr["pose"], joint=j + 1, direction=dirn,
               repeat=tr["repeat"], offset_deg=tr["offset_deg"], failure=meta.get("failure"),
               eligible=summ.get("eligible"), n=len(t))
    if pe is None or len(t) == 0:
        return row
    rel = t - pe
    hold = (rel >= 0) & (rel <= 15)
    win = (rel >= 3) & (rel <= 13)
    row.update(hold_n=int(hold.sum()), win_n=int(win.sum()), rel_last_s=float(rel[-1]))
    if win.sum() < 10:
        return row
    e_all = np.degrees(q - tgt)  # 모든 관절, 부호 없음
    e = e_all[:, j] * dirn
    row.update(
        protected=bool(prot[hold].any()), contact=bool(cont[hold].any()),
        e_end_deg=float(e[hold][0]) if hold.any() else math.nan,
        e_med_deg=float(np.median(e[win])),
        e_p2p_deg=float(np.ptp(e[win])),
        e_min_hold_deg=float(e[hold].min()), e_max_hold_deg=float(e[hold].max()),
        overshoot_deg=float(max(0.0, e[hold].max())),
        reversals_hold=reversals(e[hold]),
        ti_med_nm=float(np.median(ti[win, j])) * dirn,
        tsf_med_nm=float(np.median(tsf[win, j])) * dirn,
        tpd_med_nm=float(np.median(tpd[win, j])) * dirn,
        ti_max_abs_nm=float(np.abs(ti[hold, j]).max()),
        other_max_abs_med_deg=float(max(abs(np.median(e_all[win, k])) for k in range(4) if k != j)),
    )
    inside = np.abs(e[hold]) <= TOL
    # 0.2° 안에 들어가서 창 끝까지 머문 첫 시각
    idx = None
    for i in range(len(inside) - 1, -1, -1):
        if not inside[i]:
            idx = i + 1
            break
    else:
        idx = 0
    th = rel[hold]
    row["t_in_0p2_s"] = float(th[idx]) if idx < len(th) else math.nan
    row["t_first_0p2_s"] = float(th[np.argmax(inside)]) if inside.any() else math.nan
    return row


def safe(args):
    try:
        return one(args)
    except Exception as exc:  # run 하나의 오류가 전체를 멈추지 않게
        return dict(trial=args[1].name, error=repr(exc))


def main():
    jobs = []
    for a in sys.argv[1:]:
        exp = Path(a).resolve()
        jobs += [(exp, r) for r in sorted((exp / "runs").iterdir()) if (r / "manifest.json").exists()]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(safe, jobs))
    by = {}
    for (exp, _), r in zip(jobs, rows):
        by.setdefault(exp, []).append(r)
    for exp, rs in by.items():
        out = exp / "result"
        out.mkdir(exist_ok=True)
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(out / "trials.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rs)
        print(out / "trials.csv", len(rs))


if __name__ == "__main__":
    main()
