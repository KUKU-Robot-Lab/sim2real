#!/usr/bin/env python3
"""sweep 실험 폴더 → result/sweep.csv (관절 × 속도 단계 × 방향).

trial 하나에 관절 7개가 이어져 있다(plan trial["segments"]). 관절 구간 안에서 목표 속도가
단계 속도의 80% 이상인 정속 틱만 쓰고, 그 틱의 모터 토크(pd + 보상 합)를 방향별로 평균한다.
정속이면 관성 항은 0 이고 중력은 ± 왕복 평균에서 상쇄되므로
  tau(+v) - tau(-v) = 2 (Fc + Fv·v)
속도 단계 4개에 직선을 맞추면 Fc(절편) · Fv(기울기)가 나온다.
사용: analyze_sweep.py <실험 폴더> [...]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "deploy/policy_control"))
from policy_control import stationary_tracking as S  # noqa: E402

TORQUE_KEYS = ("torque_pd", "torque_gravity", "torque_static_friction", "torque_integral", "torque_feedforward")


def load(run, side):
    topic = f"/policy_control/pd_{side}/diagnostics"
    t, q, tgt, tau = [], [], [], []
    for e in S.load_run_events(run, topics=(topic,)):
        d = e["message"]
        if isinstance(d.get("data"), str):
            d = json.loads(d["data"])
        if d.get("phase") != "TRACKING" or "user_target" not in d:
            continue
        t.append(d["stamp"])
        q.append(d["measured"][:7])
        tgt.append(d["user_target"][:7])
        tau.append(np.sum([np.asarray(d.get(k) or [0.0] * 7, float)[:7] for k in TORQUE_KEYS], axis=0))
    return np.asarray(t), np.asarray(q, float), np.asarray(tgt, float), np.asarray(tau, float)


def segment_rows(exp, run, trial, speeds):
    side = json.loads((run / "manifest.json").read_text())["metadata"]["side"]
    t, q, tgt, tau = load(run, side)
    if len(t) < 10:
        return []
    frames = np.asarray(trial["frames"], float)
    rows = []
    for seg in trial["segments"]:
        j = seg["joint"]
        ref = frames[seg["start"]:seg["end"], j]
        # 계획 구간의 목표 궤적 범위로 실측 틱을 고른다: 그 관절만 움직이는 동안
        lo, hi = ref.min() - 1e-4, ref.max() + 1e-4
        others = [k for k in range(7) if k != j]
        moving = np.abs(np.ptp(ref)) > 0
        if not moving:
            continue
        vt = np.gradient(tgt[:, j], t)
        vm = np.gradient(q[:, j], t)
        still_other = np.all(np.abs(tgt[:, others] - frames[seg["start"], others]) < 1e-4, axis=1)
        inside = still_other & (tgt[:, j] >= lo) & (tgt[:, j] <= hi)
        for v in speeds:
            pair = {}
            for sign in (1, -1):
                m = inside & (np.abs(vt - sign * v) < 0.2 * v)
                pair[sign] = m
                if m.sum() < 3:
                    continue
                rows.append({
                    "exp": exp.name, "trial": trial["id"], "joint": j + 1, "speed": v, "dir": sign,
                    "n": int(m.sum()),
                    "tau_mean": float(tau[m, j].mean()), "tau_std": float(tau[m, j].std()),
                    "vel_meas_mean": float(vm[m].mean()),
                    "track_err_deg": float(np.degrees(np.median(np.abs(q[m, j] - tgt[m, j])))),
                })
    return rows


def fit(rows):
    out = []
    for j in sorted({r["joint"] for r in rows}):
        vs, half = [], []
        for v in sorted({r["speed"] for r in rows if r["joint"] == j}):
            p = [r for r in rows if r["joint"] == j and r["speed"] == v and r["dir"] == 1]
            n = [r for r in rows if r["joint"] == j and r["speed"] == v and r["dir"] == -1]
            if p and n:
                vs.append(v)
                half.append((np.mean([r["tau_mean"] for r in p]) - np.mean([r["tau_mean"] for r in n])) / 2)
        if len(vs) >= 2:
            fv, fc = np.polyfit(vs, half, 1)
        elif vs:
            fv, fc = 0.0, half[0]
        else:
            continue
        out.append({"joint": j, "Fc_Nm": float(fc), "Fv_Nms_per_rad": float(fv),
                    "speeds": vs, "half_diff_Nm": [round(float(h), 4) for h in half]})
    return out


def main():
    for arg in sys.argv[1:]:
        exp = Path(arg).resolve()
        plan = json.loads((exp / "setup/plan/plan.json").read_text())
        speeds = [float(v) for v in plan.get("speeds") or ()]
        by_id = {t["id"]: t for t in plan["trials"]}
        rows = []
        for run in sorted((exp / "runs").glob("*/")):
            trial = by_id.get(run.name)
            if trial is None or not trial.get("segments") or not (run / "manifest.json").exists():
                continue
            rows += segment_rows(exp, run, trial, speeds)
        out = exp / "result"
        out.mkdir(exist_ok=True)
        if rows:
            with open(out / "sweep.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        fits = fit(rows)
        (out / "sweep_fit.json").write_text(json.dumps(fits, indent=2, ensure_ascii=False))
        print(f"{exp.name}: {len(rows)} rows")
        for r in fits:
            print(f"  j{r['joint']} Fc={r['Fc_Nm']:+.3f} N·m  Fv={r['Fv_Nms_per_rad']:+.3f} N·m·s/rad  {r['half_diff_Nm']}")


if __name__ == "__main__":
    main()
