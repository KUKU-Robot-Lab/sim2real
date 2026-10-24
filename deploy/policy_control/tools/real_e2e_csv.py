#!/usr/bin/env python3
"""★10.25 실기 에피소드 기록 → e2e CSV(hdgp scripts/tools/e2e_chain/e2e_contract.yaml per_episode 35 열, sim episodes.csv 와 같은 순서).

입력 = 에피소드 실행기 폴더 logs/episode/<episode_id>_<episode>/ (episode_<id>.jsonl · scene.json). 한 폴더 = 한 행.
실기에서 얻지 못하는 값은 빈 칸(열은 빼지 않는다). 채우는 규칙(learn:e2e-aglt 10.25):
  cond = gru · latch = est · layout = real(인자로 바꿈) · seed = 회차 번호(입력 순, --seed0 부터) · env = episode_id
  cup = 에피소드 YAML 의 place(없으면 aglt) 노드 policy_object · x, y = scene.json 컵 첫 자세(FP++, base_link)
  grasp_ok = aglt 노드가 COMPLETED 로 나감 · grasp_end = 그 종료 사유 앞머리 · t_grasp_s = 그 노드 시간(접근 + 넘김 + 잡기)
  ap_ok = aglt 실패 사유가 접근 단계(approach · 접근)면 0, aglt 를 지났거나 다른 실패면 1
  carry_* = place 노드 신호 ready_carry(이송 단계가 있을 때만): carry_ok = 최대 스텝 안에 끝남(sim 과 같은 뜻) · carry_steps = 틱 ·
            carry_err_mm = 계획 앵커 오차(완벽 추종 가정의 계획 값 — sim 의 지각 컵 끝 오차와 같은 양은 아니다)
  place_ok = 홀더 안(놓은 뒤 FP++ 컵: xy < 0.035 · |dz| < 0.03) 이고 손이 빔 · grade · perfect 는 비움(실기 기울기 측정 없음)
  place_reasons = place 노드 종료 사유 원문 · t_place_s = place 노드 시간(이송 포함) · phys_fault = pd 고장 신호
  fail_stage = phys | approach | grasp | carry | place | none — 판정할 근거가 없으면 빈 칸(e2e_ok 도 빈 칸).
  place 노드가 failure 로 끝나면 신호가 없어도 place 실패. 노드가 안 떠 있거나 거부(정책이 안 돎)면 빈 칸
signals 는 10.25 실행기부터 jsonl 에 남는다 — 그 전 기록은 place_ok · carry_* 가 빈 칸이다.

  python3 deploy/policy_control/tools/real_e2e_csv.py logs/episode/20261026_*_pick_place_f_right --out <csv>
  (--out 없으면 ~/rl_ws/our_source/rh_aglt_runs/real/<첫 날짜>_e2e/episodes.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

SIM2REAL = Path(__file__).resolve().parents[3]
EPISODES = SIM2REAL / "config" / "episodes"
COLS = ("side", "seed", "cond", "latch", "layout", "env", "cup", "x", "y", "ap_ok", "ap_steps", "grasp_ok", "grasp_end",
        "t_handoff_s", "t_grasp_s", "ho_phi_deg", "ho_dphi_deg", "ho_cup_move_mm", "est_err_mm", "est_err_deg",
        "carry_ok", "carry_steps", "carry_err_mm", "carry_tilt_deg", "carry_palm_cup_mm", "seam_drift_mm", "seam_rot_deg",
        "place_ok", "grade", "perfect", "place_reasons", "phys_fault", "t_place_s", "e2e_ok", "fail_stage")
FAIL_STAGES = ("approach", "grasp", "carry", "place", "phys", "none")
APPROACH_RE = re.compile(r"approach|접근", re.I)
INFRA_RE = re.compile(r"서비스가 없다|set_parameters|refused|거부")   # 정책이 돌지 않은 준비 실패 — 판정 보류
END_EVENTS = ("exit", "failure", "stopped")


class RealE2eError(RuntimeError):
    pass


def _rows(run: Path) -> list[dict]:
    files = sorted(run.glob("episode_*.jsonl"))
    if not files:
        raise RealE2eError(f"{run}: episode_*.jsonl 없음")
    return [json.loads(line) for f in files for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def _last_end(rows: list[dict], prefix: str) -> dict | None:
    """그 정책(rh_aglt_* | rh_place_*) 노드의 마지막 끝 행 — 복구 재시도가 있으면 마지막 시도."""
    got = [r for r in rows if r.get("event") in END_EVENTS and str(r.get("policy_name") or "").startswith(prefix)]
    return got[-1] if got else None


def _signals(row: dict | None) -> dict:
    """역할별 신호({role: {...}}) → 한 정책 노드의 신호(노드마다 역할 하나)."""
    sig = (row or {}).get("signals") or {}
    return next(iter(sig.values()), {}) if isinstance(sig, dict) else {}


def _side(rows: list[dict], episode: str) -> str:
    for r in rows:
        m = re.match(r"rh_(aglt|place)_([rl])$", str(r.get("policy_name") or ""))
        if m:
            return {"r": "r", "l": "l"}[m.group(2)]
    return "r" if "right" in episode else "l" if "left" in episode else ""


def _cup(episode: str) -> str:
    p = EPISODES / f"{episode}.yaml"
    if not p.is_file():
        return ""
    spec = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    nodes = [n for n in spec.get("sequence", []) or [] if isinstance(n, dict) and n.get("policy_object")]
    place = [n for n in nodes if str(n.get("name", "")).startswith("rh_place")]
    return str((place or nodes or [{}])[0].get("policy_object", ""))


def _xy(run: Path) -> tuple:
    p = run / "scene.json"
    if not p.is_file():
        return "", ""
    objs = json.loads(p.read_text(encoding="utf-8")).get("objects") or {}
    pose = objs.get("CUP") or next(iter(objs.values()), None)
    return (round(float(pose[0][0]), 4), round(float(pose[0][1]), 4)) if pose else ("", "")


def _b(v) -> str | int:
    return "" if v is None else int(bool(v))


def _f(v, nd: int = 3):
    return "" if v in (None, "") else round(float(v), nd)


def row_of(run: Path, seed: int, cond: str, latch: str, layout: str) -> dict:
    rows = _rows(run)
    head = rows[0]
    episode, env = str(head.get("episode", "")), str(head.get("episode_id", ""))
    out: dict[str, object] = dict.fromkeys(COLS, "")
    out.update(side=_side(rows, episode), seed=seed, cond=cond, latch=latch, layout=layout, env=env, cup=_cup(episode))
    out["x"], out["y"] = _xy(run)
    g, p = _last_end(rows, "rh_aglt"), _last_end(rows, "rh_place")
    ps = _signals(p)
    phys = any(_signals(r).get("pd_fault") for r in rows if r.get("event") in END_EVENTS)
    out["phys_fault"] = int(phys)
    stage = ""
    if g is not None:
        g_ok = g.get("event") == "exit"
        reason = str(g.get("termination_reason") or "")
        ap_fail = (not g_ok) and bool(APPROACH_RE.search(reason))
        out.update(ap_ok=int(not ap_fail), grasp_ok=int(g_ok), grasp_end=reason.split(":")[0][:40],
                   t_grasp_s=_f(g.get("duration"), 2))
        if not g_ok:
            stage = "approach" if ap_fail else "grasp"
    if p is not None:
        rc = ps.get("ready_carry")
        if isinstance(rc, dict):
            plan = rc.get("plan") or {}
            out.update(carry_ok=_b(rc.get("carry_ok")), carry_steps=rc.get("t", ""), carry_err_mm=_f(plan.get("anchor_err_mm"), 2))
            if not stage and rc.get("carry_ok") is False:
                stage = "carry"
        hold, empty = ps.get("cup_in_holder"), ps.get("hand_empty")
        place_ok = None if hold is None or empty is None else bool(hold and empty)
        if place_ok is None and p.get("event") == "failure" and not INFRA_RE.search(str(p.get("termination_reason") or "")):
            place_ok = False                         # 신호가 없는 기록(10.25 전)이라도 place 노드가 실패로 끝났으면 실패
        out.update(place_ok=_b(place_ok), place_reasons=str(p.get("termination_reason") or ""), t_place_s=_f(p.get("duration"), 2))
        if not stage:
            stage = "" if place_ok is None else ("none" if place_ok else "place")
    elif not stage and g is not None:
        stage = "place"                              # 잡기는 됐는데 place 노드가 돌지 않았다(중간 정지)
    if phys:
        stage = "phys"
    out["fail_stage"] = stage
    out["e2e_ok"] = "" if stage == "" else int(stage == "none")
    assert stage in FAIL_STAGES + ("",), stage
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="에피소드 실행기 폴더(logs/episode/<id>_<episode>)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed0", type=int, default=1, help="첫 회차 번호")
    ap.add_argument("--cond", default="gru")
    ap.add_argument("--latch", default="est")
    ap.add_argument("--layout", default="real")
    a = ap.parse_args(argv)
    runs = sorted(r for r in a.runs if r.is_dir())
    if not runs:
        raise SystemExit("에피소드 폴더가 없다")
    rows = [row_of(r, a.seed0 + i, a.cond, a.latch, a.layout) for i, r in enumerate(runs)]
    out = a.out or (Path.home() / "rl_ws/our_source/rh_aglt_runs/real" / f"{str(rows[0]['env'])[:8]}_e2e" / "episodes.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    n = {s: sum(r["fail_stage"] == s for r in rows) for s in FAIL_STAGES + ("",)}
    print(f"{out} · {len(rows)} 행 · fail_stage {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
