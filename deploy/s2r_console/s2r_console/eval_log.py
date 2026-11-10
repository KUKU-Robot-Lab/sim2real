"""실기 평가 탭(11.02) — 평가 에피소드 목록과 회차 기록(CSV). 로봇에는 아무것도 보내지 않는다.

에피소드: EVAL_EPISODES 에 적은 `config/episodes/<이름>.yaml` 만 그 순서로 읽는다(11.05 사용자, 파일 없는 이름은 건너뛴다).
실행은 이 모듈이 하지 않는다 — 화면이 기존 길(미션 단계 episode_<이름> → 에피소드 실행기 [연속 실행] 승인)로 보낸다.

회차 기록: `logs/s2r_console/eval/<YYYY-MM-DD>.csv`(utf-8, 머리 한 줄, 날짜마다 파일). 서버가 round · time · episode ·
policy_r · policy_l 을 채운다(policy id 는 클라이언트가 아니라 에피소드 yaml 에서 읽는다). 기록일 뿐이라 lease 가 필요 없다.
"""
from __future__ import annotations

import csv
import io
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Mapping

import yaml

from . import tilt_view as TV

#: ★11.05 사용자: 탭에는 이 셋만 — 다른 cocktail_* yaml 은 지우지 않고 목록에서만 뺀다
#: ★11.09 사용자: 놓기 학생 에피소드(F 잡기 → 놓기 학생) 두 장을 더한다
#: ★11.11 deploy 리뷰 MEDIUM: 놓기 학생 색 · 홀더 조합(make_rh56f1_missions.PLACE_M_VARIANTS)도 평가 회차로 적는다
EVAL_EPISODES = ("cocktail_1c_e4000", "grasp_s_right", "grasp_s_left", "place_m_right", "place_m_left",
                 "place_m_blue_h2_right", "place_m_pink_h1_right", "place_m_pink_h2_right")
COLUMNS = ("round", "time", "episode", "policy_r", "policy_l", "result", "fail_stage", "fail_type", "note")
RESULTS = ("성공", "실패", "중단")
#: 화면 · 실행기 말을 저장 값으로 — 실행기 phase(SUCCESS · FAILURE · STOPPED)와 영어 별칭도 받는다
_RESULT_ALIAS = {"success": "성공", "fail": "실패", "failure": "실패", "abort": "중단", "stopped": "중단", "stop": "중단"}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_TEXT_MAX = {"fail_stage": 80, "fail_type": 80, "note": 500}
#: CSV 열 → (놓기 역할, 손). 놓기 역할이 없는 에피소드(잡기)는 그 손 정책의 student(없으면 policy)를 쓴다
PLACE_ROLES = {"policy_r": ("rh_place_r", "right"), "policy_l": ("rh_place_l", "left")}
_LOCK = threading.Lock()


class EvalError(ValueError):
    """입력이 잘못됐다 — api 가 400 으로 돌려준다."""


def eval_dir(repo: Path) -> Path:
    return repo / "logs" / "s2r_console" / "eval"


def episode_files(repo: Path) -> list[Path]:
    d = repo / "config" / "episodes"
    return [d / f"{n}.yaml" for n in EVAL_EPISODES if (d / f"{n}.yaml").is_file()]


def _holders(raw: Mapping) -> list[dict]:
    """놓기 노드마다 {node, object, holder 번호} — sequence 의 target_holder 를 holders 표로 번호로 바꾼다."""
    names = raw.get("holders") if isinstance(raw.get("holders"), Mapping) else {}
    out = []
    for node in raw.get("sequence") or []:
        if not isinstance(node, Mapping) or "target_holder" not in node:
            continue
        key = node.get("target_holder")
        out.append({"node": str(node.get("id", "")), "object": str(node.get("source_object", "")),
                    "holder": names.get(key, key)})
    return out


def _empty(name: str, error: str) -> dict:
    return {"name": name, "error": error, "version": None, "policy_r": "", "policy_l": "", "holders": [], "stages": []}


def read_episode(path: Path) -> dict:
    """에피소드 파일 하나 → 화면 칸. 읽지 못하면 error 만 채운다(목록은 계속 그린다)."""
    name = path.stem
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return _empty(name, f"{type(exc).__name__}: {exc}")
    if not isinstance(raw, Mapping):
        return _empty(name, "최상위가 매핑이 아니다")
    pol = raw.get("policies") if isinstance(raw.get("policies"), Mapping) else {}
    head = raw.get("episode") if isinstance(raw.get("episode"), Mapping) else {}

    def pid(role: str, side: str) -> str:
        b = pol.get(role)
        if isinstance(b, Mapping):
            # 놓기 학생 에피소드(11.09)는 student 가 실제로 도는 정책 — policy 는 교사(계약 출처)다
            return str(b.get("student") or b.get("policy", ""))
        # 실행되는 것은 student 쪽(깊이 학생) — 같은 손 항목이 여럿이면 yaml 순서 첫 항목
        b = next((v for v in pol.values() if isinstance(v, Mapping) and v.get("side") == side), {})
        return str(b.get("student") or b.get("policy") or "")

    stages = [str(n.get("id")) for n in raw.get("sequence") or []
              if isinstance(n, Mapping) and n.get("id") and n.get("type") != "terminal"]
    return {"name": name, "error": "", "version": head.get("version"),
            **{col: pid(*rs) for col, rs in PLACE_ROLES.items()}, "holders": _holders(raw), "stages": stages}


def episodes(repo: Path) -> list[dict]:
    """EVAL_EPISODES 중 파일이 있는 것. 파일 mtime · 개수가 같으면 다시 읽지 않는다."""
    d = repo / "config" / "episodes"
    # 목록의 세 경로를 늘 넣는다(없으면 mtime 0) — 파일이 생기거나 지워지면 캐시가 풀린다
    return TV._cached(f"eval:{repo}", [d, *(d / f"{n}.yaml" for n in EVAL_EPISODES)],
                      lambda: [read_episode(p) for p in episode_files(repo)])


def view(repo: Path, stage_ids) -> dict:
    """상황판 스냅샷 session.eval — 에피소드 칸 + 미션 단계(episode_<이름>)가 있는지. 실패해도 상황판은 돈다."""
    ids = set(stage_ids)
    try:
        items = [{**e, "stage": f"episode_{e['name']}", "has_stage": f"episode_{e['name']}" in ids} for e in episodes(repo)]
        return {"episodes": items, "dir": str(eval_dir(repo).relative_to(repo))}
    except Exception as exc:                          # noqa: BLE001 — 표시용 집계가 상황판을 멈추지 않게
        return {"episodes": [], "error": f"{type(exc).__name__}: {exc}"}


# ── 회차 기록 ──────────────────────────────────────────────────────────
def _path(repo: Path, date: str) -> Path:
    if not isinstance(date, str) or not _DATE.match(date):
        raise EvalError(f"날짜는 YYYY-MM-DD 여야 한다: {date!r}")
    return eval_dir(repo) / f"{date}.csv"


def read_rows(repo: Path, date: str) -> list[dict]:
    path = _path(repo, date)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [{k: (row.get(k) or "") for k in COLUMNS} for row in csv.DictReader(fh)]


def normalize_result(value) -> str:
    text = str(value or "").strip()
    out = text if text in RESULTS else _RESULT_ALIAS.get(text.lower(), "")
    if not out:
        raise EvalError(f"result 는 {' · '.join(RESULTS)} 중 하나여야 한다: {text!r}")
    return out


def _text(data: Mapping, key: str) -> str:
    value = data.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise EvalError(f"{key} 는 문자열이어야 한다")
    value = " ".join(value.split())                   # 줄바꿈 · 탭 없이 한 줄
    limit = _TEXT_MAX[key]
    if len(value) > limit:
        raise EvalError(f"{key} 가 너무 길다({len(value)} > {limit})")
    return value


def append_round(repo: Path, data: Mapping, *, now: datetime | None = None) -> dict:
    """한 회차를 오늘 파일 끝에 붙인다. round · time · policy id 는 서버가 채운다. 돌려주는 값 = 쓴 행 + date."""
    now = now or datetime.now()
    name = data.get("episode")
    if not isinstance(name, str) or name not in EVAL_EPISODES or not _NAME.match(name):
        raise EvalError(f"episode 는 {' · '.join(EVAL_EPISODES)} 중 하나여야 한다: {name!r}")
    src = repo / "config" / "episodes" / f"{name}.yaml"
    if not src.is_file():
        raise EvalError(f"에피소드 파일이 없다: config/episodes/{name}.yaml")
    ep = read_episode(src)
    if ep["error"]:
        raise EvalError(f"{name}: {ep['error']}")
    result = normalize_result(data.get("result"))
    stage = "" if result == "성공" else _text(data, "fail_stage")
    if stage and stage not in ep["stages"]:
        raise EvalError(f"fail_stage {stage!r} 는 {name} 의 단계가 아니다")
    fail_type, note = _text(data, "fail_type"), _text(data, "note")
    date = now.strftime("%Y-%m-%d")
    path = _path(repo, date)
    with _LOCK:
        rows = read_rows(repo, date)
        last = max((int(r["round"]) for r in rows if str(r["round"]).isdigit()), default=0)
        row = {"round": str(last + 1), "time": now.strftime("%H:%M:%S"), "episode": name,
               "policy_r": ep["policy_r"], "policy_l": ep["policy_l"], "result": result, "fail_stage": stage,
               "fail_type": fail_type, "note": note}
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
        if not path.is_file() or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as fh:
            fh.write(buf.getvalue())
    return {**row, "date": date}


def summary(rows: list[dict]) -> dict:
    """n · 성공 · 성공률 · 실패(중단 포함) 단계별 개수 · 에피소드별 n/성공."""
    n = len(rows)
    ok = sum(r["result"] == "성공" for r in rows)
    by_stage: dict[str, int] = {}
    by_episode: dict[str, dict] = {}
    for r in rows:
        if r["result"] != "성공":
            key = r["fail_stage"] or "단계 없음"
            by_stage[key] = by_stage.get(key, 0) + 1
        e = by_episode.setdefault(r["episode"], {"n": 0, "success": 0})
        e["n"] += 1
        e["success"] += r["result"] == "성공"
    return {"n": n, "success": ok, "fail": sum(r["result"] == "실패" for r in rows),
            "abort": sum(r["result"] == "중단" for r in rows), "rate": (ok / n) if n else None,
            "by_stage": by_stage, "by_episode": by_episode}


def recent_fail_types(repo: Path, *, upto: str, limit: int = 30, days: int = 14) -> list[str]:
    """최근 fail_type(새것부터, 중복 없이) — upto 날짜 파일과 그 앞 날짜 파일들에서."""
    files = sorted((p.stem for p in eval_dir(repo).glob("*.csv") if _DATE.match(p.stem) and p.stem <= upto), reverse=True)
    seen: dict[str, None] = {}
    for date in files[:days]:
        for r in reversed(read_rows(repo, date)):
            if r["fail_type"] and r["fail_type"] not in seen:
                seen[r["fail_type"]] = None
                if len(seen) >= limit:
                    return list(seen)
    return list(seen)


def rounds_view(repo: Path, date: str | None = None, *, now: datetime | None = None) -> dict:
    """GET /api/eval/rounds 본문 — 그 날짜(없으면 오늘)의 행 · 요약 · 최근 fail_type."""
    today = (now or datetime.now()).strftime("%Y-%m-%d")
    date = date or today
    rows = read_rows(repo, date)
    return {"ok": True, "date": date, "today": today, "file": str(_path(repo, date).relative_to(repo)),
            "columns": list(COLUMNS), "rows": rows, "summary": summary(rows),
            "fail_types": recent_fail_types(repo, upto=max(today, date))}
