"""policy 탭 'tilt 평가' 묶음(10.17 사용자 "policy 부분에 내가 말한 부분들이 시각화 잘되게") — 읽기만 한다.

docs/PLAN_GRASP_TILT_EVAL_2026-10-17.md 의 것을 한 화면에 모은다:
  paths    경로 A · C · B 에피소드(config/episodes/tilt_base_*.yaml) — 노드 순서, 지금 쓸 수 있는지(정책 등록부 · 손 계약),
           언제 쓰는지(결정 5: A → C → B), 실행할 미션 단계 이름.
  run      지금 실행기 에피소드가 tilt_base_* 이면 그 실행 폴더의 계획 검사(이송 carry_* · tilt_* check.json)와
           손 쥐기 결과(hand_* · 경로 B) — 실기 전에 볼 수(탁자 여유 · 두 팔 사이 · 속도 · IK 오차 · sim 출발 범위).
  records  회차 기록(tools/eval_record.py → logs/eval_real/<날짜>/) 요약과 최근 줄 — 성공 정의(결정 4)와 파지 확인(결정 1).
파일은 mtime 이 같으면 다시 읽지 않는다(상황판 스냅샷이 초마다 부른다).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time
from typing import Mapping

from . import _paths

#: (경로, 에피소드, 이름, 언제) — 결정 5 의 순서
PATHS = (("A", "tilt_base_a", "파지 정책(rg3wo)", "먼저 — 파지 확인(결정 1) 쪽당 4/5 이상이면"),
         ("C", "tilt_base_c", "접근 정책 + 스크립트 파지", "A 가 쪽당 4/5 미만이면(결정 5)"),
         ("B", "tilt_base_b", "수동 파지(확인 두 번)", "A · C 가 안 되면 — 사람이 컵을 손에 댄다"))
CRITERION = ("성공 = tilt(복귀까지) 끝난 뒤 받는 컵(shaker)에 넣은 비드 10 개 전부 — D4 판정 success_fill_ratio 1.0(결정 4 · 11). "
             "sim 은 붓는 중 1/3 s 성립이면 성공, 실기는 끝에 센다(복귀 중 흘림도 실패)")
GRASP_RULE = "파지 확인 = 쪽당 5 회, 7 cm 들고 정책이 멈춘 뒤 손이 3 s 유지되면 1 회 성공 · 4/5 이상이면 경로 A(결정 1 · 5)"
RECORD_CMD = "python3 deploy/policy_control/tools/eval_record.py tilt --path A --in-rcv 10 --in-src 0 --spilled 0 --video <파일>"
#: tilt check.json 검사 이름 → (보이는 이름, 단위 배율, 단위)
TILT_LABEL = {"src_cup_table_clear_m": ("병 · 탁자 여유", 100.0, "cm"), "rcv_cup_table_clear_m": ("받는 컵 · 탁자 여유", 100.0, "cm"),
              "src_vmax_rad_s": ("병 팔 최대 속도", 1.0, "rad/s"), "rcv_vmax_rad_s": ("받는 팔 최대 속도", 1.0, "rad/s"),
              "src_ik_err_max_cm_deg": ("병 IK 오차", 1.0, "cm · °"), "rcv_ik_err_max_cm_deg": ("받는 컵 IK 오차", 1.0, "cm · °"),
              "src_min_margin_deg": ("병 팔 관절 여유", 1.0, "°"), "rcv_min_margin_deg": ("받는 팔 관절 여유", 1.0, "°"),
              "src_start_cup_tilt_deg": ("병 출발 기울기", 1.0, "°"), "rcv_start_cup_tilt_deg": ("받는 컵 출발 기울기", 1.0, "°"),
              "src_start_cup_in_sim_box": ("병 출발이 sim 출발 범위 안", 0.0, ""),
              "rcv_start_cup_in_sim_box": ("받는 컵 출발이 sim 출발 범위 안", 0.0, "")}
SIDE_KO = {"right": "오른팔", "left": "왼팔"}

_cache: dict = {}


def _cached(key: str, paths: list[Path], make):
    """paths 의 mtime 이 그대로면 지난 값. 파일이 없으면 mtime 0."""
    stamp = tuple(p.stat().st_mtime if p.exists() else 0.0 for p in paths)
    hit = _cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    val = make()
    _cache[key] = (stamp, val)
    return val


def _round(v, k: float = 1.0, nd: int = 2):
    if isinstance(v, (int, float)):
        return round(float(v) * k, nd)
    if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
        return [round(float(x) * (k or 1.0), nd) for x in v]
    return v


# ---------------------------------------------------------------- paths
def _path_card(repo: Path, key: str, name: str, title: str, when: str) -> dict:
    from policy_control import episode_spec as S
    from policy_control import policy_registry as R
    from policy_control.episode_flow import flow_of

    ep_path = repo / "config" / "episodes" / f"{name}.yaml"
    try:
        ep = S.load(ep_path)
    except Exception as exc:                          # noqa: BLE001 — 화면은 이유를 보여 준다
        return {"key": key, "episode": name, "title": title, "when": when, "error": f"{type(exc).__name__}: {exc}"}
    entries = {e.id: e for e in R.scan(repo / "deploy" / "policies", deep=False)}
    av = S.availability(ep, entries)
    hands = S.hand_problems(ep, repo / "deploy" / "policies")
    problems = [f"{r}: {w}" for r, w in av.items() if w] + [f"손 {n}: {w}" for n, w in hands.items() if w]
    flow = flow_of(ep)["nodes"]
    return {"key": key, "episode": name, "title": title, "when": when, "stage": f"episode_{name}",
            "policies": {r: b.policy for r, b in ep.policies.items()},
            "nodes": [{"id": n["id"], "type": n["type"], "title": n["title"], "line": (n["lines"] or [""])[0]} for n in flow],
            "confirms": sum(n["type"] == "confirm" for n in flow), "problems": problems, "ok": not problems}


def paths_view(repo: Path) -> list[dict]:
    files = [repo / "config" / "episodes" / f"{n}.yaml" for _, n, _, _ in PATHS]
    pol = repo / "deploy" / "policies" / "rh56f1" / "aglt"
    return _cached("paths", files + [pol], lambda: [_path_card(repo, *p) for p in PATHS])


# ---------------------------------------------------------------- run
def _carry_card(node: str, d: Path) -> dict:
    chk = json.loads((d / "check.json").read_text(encoding="utf-8"))
    rows = []
    for side, c in (chk.get("checks") or {}).items():
        rows += [{"name": f"{SIDE_KO.get(side, side)} 탁자 여유", "value": _round(c.get("table_clear_m"), 100.0), "unit": "cm",
                  "limit": 1.0, "ok": (c.get("table_clear_m") or 0) >= 0.01},
                 {"name": f"{SIDE_KO.get(side, side)} 최대 속도", "value": _round(c.get("vmax_rad_s")), "unit": "rad/s",
                  "limit": 2.0, "ok": (c.get("vmax_rad_s") or 9) <= 2.0},
                 {"name": f"{SIDE_KO.get(side, side)} 도착(관절 · 컵)", "value": [c.get("end_q_err_rad"), c.get("end_cup_err_mm")],
                  "unit": "rad · mm", "limit": None, "ok": bool(c.get("arrived")) and bool(c.get("plan_ok"))}]
    arms = chk.get("arms_clear") or {}
    if arms:
        rows += [{"name": "두 팔 사이(팔 점)", "value": _round(arms.get("min_arm_point_m"), 100.0), "unit": "cm", "limit": 10.0,
                  "ok": (arms.get("min_arm_point_m") or 0) >= 0.10},
                 {"name": "두 컵 겉면 사이", "value": _round(arms.get("min_cup_gap_m"), 100.0), "unit": "cm", "limit": 3.0,
                  "ok": (arms.get("min_cup_gap_m") or 0) >= 0.03}]
    return {"node": node, "kind": "carry", "ok": bool(chk.get("ok")), "T_end_s": chk.get("T_end_s"),
            "sides": chk.get("sides") or list((chk.get("checks") or {})), "rows": rows}


def _tilt_card(node: str, d: Path) -> dict:
    chk = json.loads((d / "check.json").read_text(encoding="utf-8"))
    rows = []
    for k, c in (chk.get("checks") or {}).items():
        label, scale, unit = TILT_LABEL.get(k, (k, 1.0, ""))
        rows.append({"name": label, "value": _round(c.get("value"), scale) if scale else None, "unit": unit,
                     "limit": _round(c.get("limit"), scale) if scale else None, "ok": bool(c.get("ok"))})
    meta = chk.get("teach_meta") or {}
    tilt = (meta.get("src_tilt_max_deg") or {}).get("new")
    return {"node": node, "kind": "tilt", "ok": bool(chk.get("ok")), "T_end_s": chk.get("T_end_s"),
            "tilt_max_deg": _round(tilt, 1.0, 1), "rows": rows}


def _hand_card(node: str, d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("hand_grasp_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        out.append({"node": node, "kind": "hand", "side": r.get("side"), "ok": r.get("rc") == 0, "phase": r.get("phase"),
                    "why": r.get("why"), "latched": r.get("latched_fingers"), "ticks": r.get("ticks")})
    return out


def run_view(runner: Mapping | None) -> dict | None:
    if not runner or not str(runner.get("episode", "")).startswith("tilt_base_"):
        return None
    run_dir = Path(str(runner.get("run_dir") or ""))
    if not run_dir.is_dir():
        return {"episode": runner.get("episode"), "run_dir": str(run_dir), "cards": []}
    order = [n.get("id") for n in (runner.get("nodes") or [])]
    subs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.split("_", 1)[0] in ("carry", "tilt", "hand"))
    files = [p / "check.json" for p in subs] + [f for p in subs for f in p.glob("hand_grasp_*.json")]

    def make() -> list:
        cards = []
        for p in subs:
            kind, node = p.name.split("_", 1)
            try:
                if kind == "carry" and (p / "check.json").is_file():
                    cards.append(_carry_card(node, p))
                elif kind == "tilt" and (p / "check.json").is_file():
                    cards.append(_tilt_card(node, p))
                elif kind == "hand":
                    cards += _hand_card(node, p)
            except (OSError, ValueError) as exc:
                cards.append({"node": node, "kind": kind, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return sorted(cards, key=lambda c: order.index(c["node"]) if c["node"] in order else 99)
    return {"episode": runner.get("episode"), "run_dir": str(run_dir), "cards": _cached(f"run:{run_dir}", files, make)}


# ---------------------------------------------------------------- records
_ev = None


def _eval_record():
    """tools/eval_record.py(성공 판정 · 요약의 정본) — 한 번만 읽는다."""
    global _ev
    if _ev is None:
        p = _paths.POLICY_CONTROL / "tools" / "eval_record.py"
        spec = importlib.util.spec_from_file_location("eval_record_view", p)
        if spec is None or spec.loader is None:
            raise ImportError(f"{p} 를 읽지 못했다")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ev = mod
    return _ev


def records_view(root: Path | None = None) -> dict:
    ev = _eval_record()
    root = Path(root or ev.ROOT)
    day = root / time.strftime("%Y%m%d")
    files = [day / "tilt.jsonl", day / "grasp.jsonl"]

    def rows(p: Path) -> list:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []

    def make() -> dict:
        return {"day": day.name, "summary": ev.summary(root), "tilt": rows(files[0])[-12:], "grasp": rows(files[1])[-12:]}
    return {**_cached(f"rec:{day}", files, make), "criterion": CRITERION, "grasp_rule": GRASP_RULE, "command": RECORD_CMD}


def view(repo: Path, runner: Mapping | None) -> dict:
    """상황판 스냅샷 session.tilt — 한 칸이 실패해도 다른 칸은 낸다(이유는 errors)."""
    out: dict = {"errors": []}
    for key, fn in (("paths", lambda: paths_view(repo)), ("run", lambda: run_view(runner)), ("records", records_view)):
        try:
            out[key] = fn()
        except Exception as exc:                      # noqa: BLE001 — 표시용 집계가 상황판을 멈추지 않게
            out[key] = None
            out["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
    return out
