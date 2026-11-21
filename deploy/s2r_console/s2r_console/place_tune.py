"""★11.18 고정 놓기 튜닝 탭(web/place_tune.js) — 서버 쪽.

움직이는 일은 전부 기존 미션 단계 place_fixed_<컵>_descend_<팔> 를 그대로 탄다: 확인창 승인 · lease · needs ·
pd TRACKING 검사 · 작업 칸 · 실기/fake 는 그 run 의 미션 그대로(execute_args 도 단계에 등록된 그대로).
설정 tune 이 붙으면 그 단계의 place_fixed.py 줄만 --until descend → --jog / --save-offset / --resume 로 바꾼다(transform).
새 발행 코드는 없다 — place_fixed.py → plan_place_fixed → replay_traj_rh --hand-start applied.

jog 승인(설계): 'jog 켜기'(arm) 는 lease 를 가진 사람이 확인창을 보고 켠다 — 켤 때의 lease token 을 같이 적는다.
켜진 동안 그 컵의 jog 클릭은 화면이 확인창 없이 prepare → execute 한다. 서버는 jog 를 받을 때마다(describe) 켜져 있는지,
요청 lease 가 켤 때 lease 와 같은지 본다(다르면 거부하고 끈다). 꺼지는 때: 화면이 탭을 떠날 때 · 컵을 바꿀 때,
pd 가 TRACKING 이 아닐 때(view), 마지막 view 폴링이 POLL_GONE_S 넘게 없을 때(화면이 닫힘), 켜거나 jog 가 성공한 뒤
ARM_TTL_S 가 지났을 때, 오프셋 저장 · 마저 놓기 · 일반 하강 · tilt_now 를 실행할 때(executed).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from .console import ConsoleError
from .console_state import STALE_S
from . import pd_names as PD

CUPS = {"blue": ("right", "blue_h1"), "pink": ("right", "pink_h2"), "shaker": ("left", "shaker_h0")}   # = plan_place_fixed.CUPS
OPS = ("jog", "save", "resume")
KEYS = frozenset({"tune", "jog", "stamp"})          # 화면이 보내는 설정 — 그 밖(누적 등)은 서버가 정한다
JOG_MM, JOG_DEG = (-5.0, -1.0, 0.0, 1.0, 5.0), (-1.0, 0.0, 1.0)   # 사용자 결정: x·y·z ±1/±5 mm, 손목 yaw ±1°
ARM_TTL_S = 120.0                                   # 켜기 · jog 성공마다 다시 센다
POLL_GONE_S = 10.0                                  # 화면 view 폴링(1 s)이 이만큼 없으면 끈다
LOG = logging.getLogger(__name__)
_STAGE = re.compile(r"^place_fixed_(blue|pink|shaker)_descend_(right|left)$")
_STAMP = re.compile(r"^\d{6}_\d{6}_\d{3}$")         # YYMMDD_HHMMSS_ms — 화면이 정한다(prepare ↔ execute 같은 명령)

_lock = threading.Lock()
UNSAVED_MSG = "jog 한 위치를 먼저 저장하세요(저장 안 하고 놓으면 교시값이 남지 않습니다)"
REASONS = {   # 플래너 check.json 키 → 화면 문장(없는 키는 그대로)
    "jog_anchor_mm": "하강 끝에서 40 mm·15° 넘게 벗어남",
    "jog_anchor_deg": "하강 끝에서 40 mm·15° 넘게 벗어남",
    "release_drop_m": "놓는 높이가 허용 창(-5~30 mm) 밖",
    "release_xy_m": "컵이 홀더 구멍 중심에서 너무 벗어남",
    "resume_pos_mm": "저장한 계획 위치와 지금 손 위치가 다름",
    "resume_ang_deg": "저장한 계획 위치와 지금 손 위치가 다름",
    "cup_bore_clear_m": "컵이 홀더 링에 닿음",
    "cup_table_m": "컵이 탁자에 닿음",
}
_armed: dict = {}                                   # {"cup", "side", "since", "seen", "who", "token"} 또는 빈 dict


def stage_id(cup: str) -> str:
    return f"place_fixed_{cup}_descend_{CUPS[cup][0]}"


def parse_jog(text) -> list[float]:
    """'dx,dy,dz,dyaw'(mm · deg) — 한 축만, 허용 값만."""
    try:
        v = [float(x) for x in str(text).split(",")]
    except ValueError:
        raise ConsoleError("jog 는 숫자 4 개(dx,dy,dz,dyaw)", code=400) from None
    if len(v) != 4 or any(x not in JOG_MM for x in v[:3]) or v[3] not in JOG_DEG or sum(x != 0 for x in v) != 1:
        raise ConsoleError("jog 는 한 축만 ±1/±5 mm 또는 yaw ±1°", code=400)
    return v


def armed(cup: str | None = None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    with _lock:
        if _armed and (now - _armed["since"] > ARM_TTL_S or now - _armed.get("seen", _armed["since"]) > POLL_GONE_S):
            _armed.clear()
        return dict(_armed) if _armed and (cup is None or _armed["cup"] == cup) else {}


def _touch(key: str, now: float | None = None) -> None:
    with _lock:
        if _armed:
            _armed[key] = time.time() if now is None else now


def _same_lease(console, token: str | None) -> bool:
    """켤 때 token 이 지금 살아 있는 lease 인가. 쓰기 요청은 api 가 지금 lease 로 check 한 뒤에만 오므로 = 요청 lease 와 같은가."""
    from .lease import LeaseError
    try:
        console.lease.check(token)
    except (LeaseError, AttributeError):
        return False
    return True


def disarm() -> None:
    with _lock:
        _armed.clear()


def take(action_id: str, parameters: dict, console=None) -> dict:
    """describe 맨 앞: tune 설정을 떼어 검사하고 돌려준다(기존 설정 검사를 지난 뒤 다시 붙인다).
    jog 는 켜져 있고, 지금 lease 가 켤 때 lease 와 같아야 한다(다르면 끈다)."""
    tune = {k: parameters.pop(k) for k in list(parameters) if k in KEYS}
    if not tune:
        return {}
    m = _STAGE.match(action_id)
    if not m or CUPS[m.group(1)][0] != m.group(2):
        raise ConsoleError("tune 설정은 place_fixed_<컵>_descend_<팔> 에만", code=400)
    if tune.get("tune") not in OPS or not _STAMP.match(str(tune.get("stamp", ""))):
        raise ConsoleError(f"tune 은 {'/'.join(OPS)} 중 하나 · stamp 는 YYMMDD_HHMMSS_ms", code=400)
    if tune["tune"] == "jog":
        parse_jog(tune.get("jog"))
        a = armed(m.group(1))
        if not a:
            raise ConsoleError("이 탭에서 jog 켜기를 먼저 승인할 것(꺼졌다: 탭 이동 · pd TRACKING 아님 · 시간 초과)")
        if not _same_lease(console, a.get("token")):
            disarm()
            raise ConsoleError("jog 를 켠 lease 와 지금 lease 가 다르다 — jog 를 껐다. 다시 켤 것")
    elif "jog" in tune:
        raise ConsoleError("jog 값은 tune=jog 에만", code=400)
    if tune["tune"] == "resume":
        repo = getattr(console, "repo", None)
        if repo is None or unsaved_jog(_results(Path(repo), m.group(1), name="*")):
            raise ConsoleError(UNSAVED_MSG, code=400)
    return tune


def executed(action_id: str, parameters: dict) -> None:
    """actions.execute 가 부른다: tune jog 가 아닌 하강 · 놓기 · tilt_now · 저장 · 마저 놓기를 실행하면 jog 를 끈다(기준이 바뀐다)."""
    if (action_id == "tilt_now" or action_id.startswith("place_fixed_")) and (parameters or {}).get("tune") != "jog":
        disarm()


def out_dir(repo: Path, cup: str, op: str, stamp: str) -> Path:
    return Path(repo) / "logs" / "runs" / stamp[:6] / "place_fixed" / f"{cup}_tune_{op}_{stamp[7:]}"


def transform(action_id: str, parameters: dict, cmd, repo: Path):
    """place_fixed.py 줄만 바꾼다(손 확인 · 다른 줄은 그대로). 순수 — prepare 와 execute 에서 같은 명령이어야 한다."""
    if "tune" not in parameters or not any(str(a).endswith("place_fixed.py") for a in cmd.argv):
        return cmd
    cup, op, stamp = _STAGE.match(action_id).group(1), parameters["tune"], parameters["stamp"]
    argv = list(cmd.argv)
    if "--until" not in argv:
        raise ConsoleError("단계 명령에 --until 이 없다 — 미션 place_fixed 단계가 바뀌었다")
    i = argv.index("--until")
    del argv[i:i + 2]
    if op == "jog":
        extra = ["--jog", ",".join(f"{v:g}" for v in parse_jog(parameters.get("jog")))]
    else:
        extra = ["--save-offset"] if op == "save" else ["--resume"]
    argv += [*extra, "--out", str(out_dir(repo, cup, op, stamp))]
    return replace(cmd, argv=tuple(argv), note=f"{cmd.note} → 튜닝 {op}")


def _results(repo: Path, cup: str, since: float = 0.0, days: int = 2, name: str = "tune_*") -> list[dict]:
    """최근 튜닝 런 — 시간순. name="*" 이면 튜닝 아닌 하강 · 놓기 런도.
    result.json 이 없거나 깨진 런(재생 중 · estop · 크래시)도 rc None 으로 넣는다 — mode · execute 는 meta.json(없으면 이름 · None)."""
    root, rows = Path(repo) / "logs" / "runs", []
    for day in sorted(root.glob("[0-9]" * 6))[-days:]:
        for d in (day / "place_fixed").glob(f"{cup}_{name}"):
            if not d.is_dir():
                continue
            p = d / "result.json"
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                float(r.get("time", 0))
            except (OSError, ValueError, TypeError, AttributeError) as exc:   # 없거나 깨진 기록은 성공으로 치지 않고 보인다
                if p.exists():
                    LOG.warning("place_tune: %s 읽기 실패 %s: %s", p, type(exc).__name__, exc)
                try:
                    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                    meta = meta if isinstance(meta, dict) else {}
                except (OSError, ValueError):
                    meta = {}
                try:
                    mtime = d.stat().st_mtime
                except OSError:
                    mtime = 0.0
                r = {"rc": None, "mode": meta.get("mode") or ("jog" if d.name.startswith(f"{cup}_tune_jog_") else None),
                     "execute": meta.get("execute"), "time": mtime,
                     "error": f"result.json {'깨짐: ' + type(exc).__name__ if p.exists() else '없음(실행 중 · 중단)'}"}
            if float(r.get("time", 0)) >= since:
                rows.append({**r, "dir": str(d)})
    return sorted(rows, key=lambda r: r["time"])


def unsaved_jog(results: list[dict]) -> bool:
    """마지막 실행 하강 · 전체 놓기 뒤 실행 jog 가 있고 그 뒤 성공한 저장이 없다 → 마저 놓으면 교시값이 사라진다.
    jog 는 rc 와 상관없이 막는다(estop · 크래시 · 실행 중이어도 손이 움직였을 수 있다). 드라이런이 확인된 jog 만 뺀다."""
    pending = False
    for r in results:
        if r.get("mode") == "jog":
            pending = pending or r.get("execute") is not False   # execute 를 모르면 실행으로 본다
        elif r.get("rc") == 0 and (r.get("mode") == "save" or (r.get("mode") in ("descend", "all") and r.get("execute"))):
            pending = False
    return pending


def anchor_from(results: list[dict]) -> list[float] | None:
    """마지막으로 성공한 실행 jog 끝의 하강 끝 계획 차이(홀더 mm 3 · yaw deg) — 플래너 check.json jog_anchor_*.
    한계 판정은 플래너가 한다(이 값은 화면 표시용). 저장 · 마저 놓기가 그 뒤에 성공했으면 None(기준이 바뀜)."""
    last = None
    for r in results:
        if r.get("rc") == 0 and r.get("execute") and r.get("mode") in ("jog", "save", "resume"):
            last = r
    if last is None or last.get("mode") != "jog":
        return None
    try:
        chk = json.loads((Path(last["dir"]) / "plan" / "check.json").read_text(encoding="utf-8"))
        return [*chk["jog_anchor_mm"]["holder_mm"], chk["jog_anchor_deg"]["yaw_deg"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.warning("place_tune: %s check.json 읽기 실패 %s: %s", last["dir"], type(exc).__name__, exc)
        return None


def _phase(console, side: str) -> str | None:
    s = console.session
    if s is None:
        return None
    obs = s.feed.observed()
    st = obs.status.get(PD.name(side))
    if not st or obs.age_s.get(PD.name(side), 1e9) > STALE_S:
        return None
    return st.get("phase")


def _summary(r: dict) -> dict:
    d = Path(r["dir"])
    out = {k: r.get(k) for k in ("rc", "mode", "execute", "jog", "time")} | {"dir": r["dir"]}
    try:
        out["failed"] = [k for k, c in json.loads((d / "plan" / "check.json").read_text(encoding="utf-8")).items()
                         if isinstance(c, dict) and not c.get("ok", True)]
        out["reasons"] = list(dict.fromkeys(REASONS.get(k, k) for k in out["failed"]))
    except (OSError, ValueError):
        pass
    try:
        rec = json.loads((d / "plan" / "offset.json").read_text(encoding="utf-8"))
        out["offset"] = {k: rec.get(k) for k in ("ok", "before", "after", "delta", "backup")}
    except (OSError, ValueError):
        pass
    return out


def view(console, cup: str) -> dict:
    if cup not in CUPS:
        raise ConsoleError(f"컵은 {'/'.join(CUPS)} 중 하나", code=400)
    side, name = CUPS[cup]
    repo = Path(console.repo)
    try:
        offset = yaml.safe_load((repo / "deploy/policy_control/config/place_fixed" / f"{name}.yaml")
                                .read_text(encoding="utf-8"))["offset"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        offset = {"error": f"{type(exc).__name__}: {exc}"}
    phase = _phase(console, side)
    arm = armed(cup)
    if arm and phase not in ("RAMPING", "TRACKING"):   # jog 재생 중 RAMPING 은 정상(replay_traj_rh MOVING)
        disarm()
        arm = {}
    recent = _results(repo, cup)
    if arm:
        _touch("seen")
        ok_jogs = [r for r in recent if r.get("mode") == "jog" and r.get("rc") == 0 and r.get("execute")]
        if ok_jogs and float(ok_jogs[-1]["time"]) > arm["since"]:   # jog 가 성공할 때마다 TTL 을 다시 센다
            _touch("since", float(ok_jogs[-1]["time"]))
            arm = armed(cup)
    anchor = anchor_from(recent)
    return {"ok": True, "cup": cup, "side": side, "stage": stage_id(cup), "config": f"config/place_fixed/{name}.yaml",
            "offset": offset, "pd_phase": phase, "armed": bool(arm), "armed_left_s": round(ARM_TTL_S - (time.time() - arm["since"])) if arm else 0,
            "cum": anchor if anchor is not None else [0.0] * 4,      # 하강 끝 계획에서 얼마나(마지막 jog 기준)
            "limits": {"step_mm": 10, "step_deg": 3, "cum_mm": 40, "cum_deg": 15},
            "unsaved_jog": unsaved_jog(_results(repo, cup, name="*")), "unsaved_msg": UNSAVED_MSG,
            "recent": [_summary(r) for r in recent[-5:]][::-1]}


def set_armed(console, data: dict, who: str, token: str | None = None) -> dict:
    cup, on = data.get("cup"), data.get("on")
    if not isinstance(on, bool) or (on and cup not in CUPS):
        raise ConsoleError("cup(blue/pink/shaker) 과 on(true/false) 이 필요하다", code=400)
    if not on:
        disarm()
        return {"ok": True, "armed": False}
    if not token:
        raise ConsoleError("jog 켜기는 lease 가 있어야 한다", code=403)
    console._need()
    side = CUPS[cup][0]
    if _phase(console, side) != "TRACKING":
        raise ConsoleError(f"{side} pd 가 TRACKING 이 아니다 — jog 를 켜지 않는다")
    now = time.time()
    with _lock:
        _armed.clear()
        _armed.update(cup=cup, side=side, since=now, seen=now, who=who, token=token)
    if console.session is not None:
        console.session.event("place_tune", f"jog 켜기 · {cup} · {side} ({who})")
    return {"ok": True, "armed": True, "since": datetime.now().isoformat(timespec="seconds")}
