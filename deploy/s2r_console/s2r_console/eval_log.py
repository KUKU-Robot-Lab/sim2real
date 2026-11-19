"""실기 평가 탭(11.02) — 평가 에피소드 목록과 회차 기록(CSV). 로봇에는 아무것도 보내지 않는다.

에피소드: EVAL_TAB 에 적은 `config/episodes/<이름>.yaml` 만 그 순서로 읽는다(11.05 사용자, 파일 없는 이름은 건너뛴다).
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
#: ★11.11 사용자 "EVAL 부분에서 STUDENT 정책들도 POLICY 탭으로 옮기면서" — 탭 목록은 EVAL_TAB 만. 기록 이름 검사는 그대로
#:   EVAL_EPISODES(학생 잡기 색 판 grasp_s_pink_right 도 받는다 — 회차 기록은 policy 탭 실행 뒤에도 적는다)
#: ★11.16 사용자 "안 B 로 세팅" — 파랑만 · 오른손 놓기 n07(cocktail_1c_n07b)을 e4000 과 따로 카드로
#: ★11.16 _t 변형(양손 놓기 T_in 에서 CUP_OBS_OFFSET_BASE_M 를 뺀 판) 두 장을 원판 카드 뒤에 따로
#: ★11.16 _t 카드 두 개(cocktail_1c_n07b_t · cocktail_1c_e4000_t)는 탭에서 뺀다 — n07b 카드의 tin_subtract_offset 토글이 같은 일을
#:   한다(TOGGLE_EPISODES). yaml · 미션 단계는 남겨 둔다(옛 회차 기록 이름 검사 = EVAL_EPISODES)
#: ★11.20 사용자: cocktail_1c_e4000 → cocktail_e2e 이름 변경 — 탭은 새 이름, 옛 이름은 11.20 이전 회차 기록 검사용으로 EVAL_EPISODES 에 남긴다
#: ★11.20 사용자 "cocktail_rrl 버전으로 tilt 가 잔차 강화학습 버전으로 할 카드를 미리" — 잔차 정책(tilt_lab)이 아직 없어 yaml · 미션 단계
#:   없이 '준비 중' 칸만 둔다(EVAL_PENDING). 정책이 오면 config/episodes/cocktail_rrl.yaml · 생성기 단계를 만들고 여기서 뺀다
EVAL_TAB = ("cocktail_e2e", "cocktail_1c_n07b", "cocktail_rrl")
EVAL_PENDING = {"cocktail_rrl": "준비 중 — tilt = 잔차 강화학습 궤적(tilt_lab: tilt 직전 파지 · 비드 개수 → 궤적 수정). "
                                "잡기 · 이송 · 놓기는 cocktail_e2e 와 같게 둘 예정. 정책이 오면 yaml · 미션 단계를 만든다"}
EVAL_EPISODES = ("cocktail_e2e", "cocktail_1c_e4000", "cocktail_1c_n07b", "grasp_s_right", "grasp_s_left", "place_m_right", "place_m_left",
                 "place_m_blue_h2_right", "place_m_pink_h1_right", "place_m_pink_h2_right", "grasp_s_pink_right")
COLUMNS = ("round", "time", "episode", "policy_r", "policy_l", "result", "fail_stage", "fail_type", "note", "toggles")
#: ★11.16 실험 토글이 붙는 카드(episode_spec.TOGGLES) — 단계 실행 때 고른 값이 실행기 · aglt 노드 인자로 간다
#: ★11.20 사용자 "cocktail_e2e 에서는 tilt 에서 v12 또는 v14 모델 고를 수 있고" — e2e 는 tilt 자산 고르기만. 나머지 토글은 받지 않고
#:   yaml · 자산 그대로 값으로 채운다(TOGGLE_KEYS). tilt_comp 는 고른 tilt 자산 policy.yaml 값(실행기는 토글이 있으면 토글 값을 쓴다 —
#:   n07b 기본값 끔을 e2e 에 씌우면 e2e 의 tilt_comp 가 꺼진다)
TOGGLE_EPISODES = ("cocktail_1c_n07b", "cocktail_e2e")
#: 카드 → 받는 토글 이름(없으면 전부)
TOGGLE_KEYS = {"cocktail_e2e": ("tilt",)}
#: 손 보조 토글 끔 = rh_aglt 노드 파라미터 기본값(GripAssist 끔). 켬 = 미션 그대로(AGLT_LIVE_ARGS)
#:   토글 이름 → (노드 파라미터, 끈 값)
HAND_ASSIST_OFF = {"hand_unfreeze_on_grasp": ("hand_unfreeze_on_grasp", "false"), "thumb_support_g": ("thumb_support_g", "0.0"),
                   "thumb_open_floor": ("thumb_open_floor_rad", "0.0")}
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


def episode_files(repo: Path, names=None) -> list[Path]:
    d, names = repo / "config" / "episodes", EVAL_TAB if names is None else names
    return [d / f"{n}.yaml" for n in names if (d / f"{n}.yaml").is_file()]


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


def episodes(repo: Path, names=None) -> list[dict]:
    """names(기본 EVAL_TAB) 중 파일이 있는 것. 파일 mtime · 개수가 같으면 다시 읽지 않는다."""
    d, names = repo / "config" / "episodes", EVAL_TAB if names is None else names
    # 목록의 경로를 늘 넣는다(없으면 mtime 0) — 파일이 생기거나 지워지면 캐시가 풀린다
    return TV._cached(f"eval:{repo}:{','.join(names)}", [d, *(d / f"{n}.yaml" for n in names)],
                      lambda: [read_episode(p) for p in episode_files(repo, names)])


def _tab_cards(repo: Path) -> list[dict]:
    """탭 카드 = EVAL_TAB 순서의 파일 칸 + ★11.20 준비 중 칸(EVAL_PENDING — 파일 · 미션 단계 없이 설명만, 파일이 생기면 파일 칸)."""
    got = {e["name"]: e for e in episodes(repo)}
    return [got[n] if n in got else {**_empty(n, ""), "nodes": [], "pending": EVAL_PENDING[n]}
            for n in EVAL_TAB if n in got or n in EVAL_PENDING]


def view(repo: Path, stage_ids) -> dict:
    """상황판 스냅샷 session.eval — 에피소드 칸 + 미션 단계(episode_<이름>)가 있는지. 실패해도 상황판은 돈다."""
    ids = set(stage_ids)
    try:
        items = [{**e, "stage": f"episode_{e['name']}", "has_stage": f"episode_{e['name']}" in ids} for e in _tab_cards(repo)]
        # ★11.11 기록 폼은 EVAL_EPISODES 전부 — policy 탭에서 돈 grasp_s · place_m 회차도 적는다(탭 카드는 EVAL_TAB 만)
        return {"episodes": items, "record": episodes(repo, EVAL_EPISODES), "dir": str(eval_dir(repo).relative_to(repo))}
    except Exception as exc:                          # noqa: BLE001 — 표시용 집계가 상황판을 멈추지 않게
        return {"episodes": [], "record": [], "error": f"{type(exc).__name__}: {exc}"}


# ── 실험 토글 ──────────────────────────────────────────────────────────
def stage_toggles(stage_id: str, raw) -> str:
    """화면이 보낸 {이름: 값} → 실행기 문자열("k=v,..."). 토글 카드 단계만 받는다. 잘못된 값은 EvalError."""
    from policy_control import episode_spec as S
    if stage_id not in {f"episode_{n}" for n in TOGGLE_EPISODES}:
        raise EvalError(f"{stage_id}: 토글을 받는 단계가 아니다")
    if not isinstance(raw, dict) or not all(isinstance(v, (bool, int, float, str)) for v in raw.values()):
        raise EvalError("toggles 는 {이름: bool|수|정책 이름} 여야 한다")
    keys = TOGGLE_KEYS.get(stage_id.removeprefix("episode_"))
    if keys is not None and set(raw) - set(keys):
        raise EvalError(f"{stage_id}: 받는 토글은 {list(keys)} 뿐이다 — {sorted(set(raw) - set(keys))}")
    text = ",".join(f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in raw.items())
    try:
        t = S.parse_toggles(text)
        if keys is not None and "tilt_comp" not in keys:    # 카드가 안 받는 tilt_comp = 고른 tilt 자산 policy.yaml 값(yaml 그대로)
            asset = S.TILT_CHOICES[t["tilt"]]
            try:
                cfg = yaml.safe_load((S.TRAJ_ASSETS / asset / "policy.yaml").read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                raise EvalError(f"{asset}/policy.yaml 을 못 읽었다 — {exc}") from None
            t["tilt_comp"] = bool(cfg.get("tilt_comp"))
        return S.format_toggles(t)
    except S.EpisodeSpecError as exc:
        raise EvalError(str(exc)) from None


def _regen(repo: Path, mission, runbook, stage_id: str, real: bool, ep) -> list:
    """미션 생성기(make_rh56f1_missions._episode_run)로 이 에피소드 단계 명령을 다시 만든다 — 정책 고르기용.
    다른 단계(정지 대상 head_home 등)는 지금 runbook 그대로 두고 이 단계만 바꿔 검증한다."""
    import sys
    from dataclasses import replace

    from mission_stages import _spec_from_raw, _validate_stops, commands_for
    ops = str(repo / "scripts" / "ops")
    if ops not in sys.path:
        sys.path.insert(0, ops)
    import make_rh56f1_missions as G
    specs = tuple(_spec_from_raw(r, stage_id) for r in G._episode_run(stage_id.removeprefix("episode_"), real, ep=ep))
    # 다른 단계의 정지 목록(shutdown 등)은 번호로 이 단계 배경 명령을 가리킨다 — 번호가 바뀌므로 이 단계 몫을 새 배경 명령 전부로 바꾼다
    pre = f"{stage_id}#"
    bg = tuple(f"{pre}{i}" for i, c in enumerate(specs) if c.background)
    cmds = {}
    for sid, old in runbook.commands.items():
        out = []
        for c in old:
            if sid != stage_id and any(k.startswith(pre) for k in c.stop):
                c = replace(c, stop=tuple(dict.fromkeys([*(k for k in c.stop if not k.startswith(pre)), *bg])))
            out.append(c)
        cmds[sid] = tuple(out)
    book = replace(runbook, commands={**cmds, stage_id: specs})
    try:
        _validate_stops(book.commands, mission)
    except ValueError as exc:
        raise EvalError(f"고른 정책으로 만든 단계가 미션과 맞지 않는다 — {exc}") from None
    return commands_for(book, mission, stage_id, repo=repo, execute=True)


def toggle_commands(commands: list, text: str, *, regen=None) -> list:
    """미션 단계 명령에 토글을 입힌 새 목록. 정책을 바꾸면 regen(ep_or_None) 으로 단계를 다시 만든다 — regen(None) = yaml 그대로
    다시 만든 것이 지금 미션 명령과 같아야 한다(다르면 미션이 생성기와 어긋났다 → 거부). 그 밖은 실행기 -p toggles:= ·
    시작 검사 --toggles · 끈 손 보조 인자를 aglt 노드 파라미터 기본값(끔)으로."""
    from dataclasses import replace

    from policy_control import episode_spec as S
    t = S.parse_toggles(text)
    if any(t[k] != S.TOGGLES[k] for k in S.POLICY_CHOICES):
        if regen is None:
            raise EvalError("정책 고르기는 미션 단계를 다시 만들어야 한다")
        if [c.argv for c in regen(None)] != [c.argv for c in commands]:
            raise EvalError("미션 yaml 이 생성기와 다르다 — make_rh56f1_missions.py 로 미션을 다시 만든 뒤 정책을 고를 것")
        commands = regen(t)
    off = {name: val for key, (name, val) in HAND_ASSIST_OFF.items() if not t[key]}
    out = []
    for cmd in commands:
        argv = tuple(cmd.argv)
        if any("episode_runner_node.py" in a for a in argv):
            argv += ("-p", f"toggles:={text}")
        if any(a.endswith("episode_run.py") for a in argv) and "--plan" in argv:
            argv += ("--toggles", text)
        if off and any("rh_aglt" in a and a.endswith("_node.py") for a in argv):
            argv = tuple(next((f"{k}:={v}" for k, v in off.items() if a.startswith(f"{k}:=")), a) for a in argv)
        out.append(cmd if argv == tuple(cmd.argv) else replace(cmd, argv=argv))
    return out


def apply_stage_toggles(repo: Path, mission, runbook, real: bool, stage_id: str, commands: list, raw) -> tuple[list, str]:
    """확인 창 · 단계 실행이 같이 쓰는 입구 — (토글 입힌 명령, 실행기 문자열)."""
    from policy_control import episode_spec as S
    text = stage_toggles(stage_id, raw)
    path = repo / "config" / "episodes" / f"{stage_id.removeprefix('episode_')}.yaml"

    def regen(t):
        ep = S.load(path)
        return _regen(repo, mission, runbook, stage_id, real, ep if t is None else S.apply_toggles(ep, t))
    try:
        return toggle_commands(commands, text, regen=regen), text
    except S.EpisodeSpecError as exc:
        raise EvalError(str(exc)) from None


# ── 회차 기록 ──────────────────────────────────────────────────────────
def _path(repo: Path, date: str) -> Path:
    if not isinstance(date, str) or not _DATE.match(date):
        raise EvalError(f"날짜는 YYYY-MM-DD 여야 한다: {date!r}")
    return eval_dir(repo) / f"{date}.csv"


def _header(path: Path) -> list | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), None)


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


def append_round(repo: Path, data: Mapping, *, now: datetime | None = None, toggles: str = "") -> dict:
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
               "fail_type": fail_type, "note": note, "toggles": toggles}
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
        if _header(path) not in (None, list(COLUMNS)):   # ★11.16 toggles 열 이전 파일 — 새 열로 다시 쓴다(옛 행은 빈 값)
            whole = io.StringIO()
            w = csv.DictWriter(whole, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            path.write_text(whole.getvalue(), encoding="utf-8")
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
