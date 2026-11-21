"""tilt_comp_ab --bringup — 콘솔(s2r_console)의 미션 단계로 로봇을 준비한다(드라이버 · 손 · pd 양팔 발행 모드).

명령을 새로 쓰지 않는다. 콘솔 HTTP API(127.0.0.1:8091)로 미션 단계를 실행한다:
  · 콘솔이 꺼져 있으면 이 PC 에 띄운다(setsid — 실험이 끝나도 콘솔과 그 프로세스는 남는다. 화면은 브라우저로 그대로 본다).
  · 배경 프로세스(드라이버 · pd)의 주인은 콘솔이다 → 실험기가 끝나도 살아 있고, 콘솔 화면 · 승인 원장 · run 기록과 어긋나지 않는다.
    (콘솔 runner 를 파이썬에서 직접 부르면 프로세스가 이 실험기와 같이 죽고, 떠 있는 콘솔은 그 프로세스를 모른다 — 그래서 API.)
  · ★수동 단계(모터 전원 · 손 전원 · 랜)는 터미널에서 안내하고 Enter 를 받은 뒤 콘솔에 확인을 보낸다. 다음 자동 확인(CAN 수신 등)은 콘솔이 돌린다.
  · 단계 하나라도 실패 · 타임아웃이면 거기서 멈춘다(실험 진입 금지). sudo 가 필요한 MISS 는 preflight 실패로 멈추고 안내를 보인다.
  · teardown(11.17 사용자): 실험이 끝나면 손 · pd · drivers 유닛을 끄고 그 단계들을 rewind 해 다음 실행이 처음부터 다시 오른다.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

SIM2REAL = Path(__file__).resolve().parents[3]
for _p in ("scripts", "deploy/s2r_console"):
    if str(SIM2REAL / _p) not in sys.path:
        sys.path.insert(0, str(SIM2REAL / _p))

URL = "http://127.0.0.1:8091"
OPERATOR = "tilt_comp_ab"
TARGETS = ("pd_arm_right", "pd_arm_left")     # 이 둘의 needs 를 따라간 단계 전부(probe_* 는 needs 가 아니라 빠진다)
STAGE_TIMEOUT = 300.0     # 수동 확인을 기다린 시간은 뺀다 — preflight(계약 재생성 + pytest)가 가장 길다
CONSOLE_UP_TIMEOUT = 40.0
PROFILE_REAL, PROFILE_FAKE = "rh56f1_real_cam5090", "rh56f1_fake"   # ★11.08 로봇 · 카메라 · 목 모두 5090
# teardown 순서(11.17 팀장이 실기에서 한 순서): 끄기는 손 → pd → drivers, rewind 는 팔마다 pd → 손, 마지막 drivers
OFF_ORDER = ("hand_left", "hand_right", "pd_arm_left", "pd_load_left", "pd_arm_right", "pd_load_right", "drivers")
REWIND_ORDER = ("pd_arm_left", "pd_load_left", "hand_left", "pd_arm_right", "pd_load_right", "hand_right", "drivers")
UNIT_DOWN_S = 10.0        # 콘솔은 끄기를 받은 뒤 그룹째 SIGTERM → 최대 5 s 뒤 SIGKILL
UNIT_RETRY = ("PD 해제", "pd 상태를 모른다")   # release 직후 콘솔의 pd 상태가 늦게 따라올 때의 거부 — 이것만 다시 시도한다


class BringupError(RuntimeError):
    pass


# ---------------------------------------------------------------- 계획(읽기만)
def profile_mission(profile_id: str) -> Path:
    raw = yaml.safe_load((SIM2REAL / "deploy/s2r_console/profiles" / f"{profile_id}.yaml").read_text(encoding="utf-8"))
    return SIM2REAL / raw["mission"]


def plan(mission_yaml: Path, targets=TARGETS) -> list[dict]:
    """targets 의 needs 를 깊이 우선으로 따라 실행 순서대로 — 단계마다 콘솔이 낼 명령(종류 · 설명 · argv)."""
    import mission_core as MC
    from mission_stages import commands_for, load_runbook
    raw = yaml.safe_load(Path(mission_yaml).read_text(encoding="utf-8"))
    mission = MC.load_mission(raw)
    runbook = load_runbook(raw.get("run", {}), mission)
    order: list[str] = []

    def visit(sid: str) -> None:
        if sid in order:
            return
        for n in MC.stage_by_id(mission, sid).needs:
            visit(n)
        order.append(sid)
    for t in targets:
        visit(t)
    out = []
    for sid in order:
        st = MC.stage_by_id(mission, sid)
        cmds = [{"kind": "stop" if getattr(c, "stop", ()) else "manual" if c.manual else "background" if c.background else "foreground",
                 "note": c.note, "argv": [str(a) for a in c.argv]}
                for c in commands_for(runbook, mission, sid, repo=SIM2REAL, execute=True)]
        out.append({"id": sid, "title": st.title, "touches_real": st.touches_real, "skippable": st.skippable,
                    "needs": list(st.needs), "commands": cmds})
    return out


def already_up(sid: str, graph: dict) -> bool | None:
    """ROS 그래프로 본 '이미 떠 있다'. None = 판단 거리가 없다(읽기 전용 점검 · preflight).
    graph = {"topics": 발행자가 있는 토픽 집합, "pd": {팔: status dict | None}}"""
    side = sid.rsplit("_", 1)[-1]
    pd = graph.get("pd", {}).get(side)
    if sid == "drivers":
        return "/joint_states" in graph["topics"]
    if sid.startswith("hand_check_"):
        return None
    if sid.startswith("hand_"):
        return f"/hand_{side}/joint_states" in graph["topics"]
    if sid.startswith("pd_load_"):
        return pd is not None                       # 무발행이든 발행이든 pd 가 떠 있다
    if sid.startswith("pd_arm_"):
        return bool(pd and pd.get("execute"))
    return None


def decide(stage: dict, done: bool | None, up: bool | None) -> tuple[str, str]:
    """(동작, 이유). 동작 = run | skip(콘솔이 이미 함) | console_skip(콘솔 밖에서 떠 있음 → 콘솔에 건너뜀) | fail."""
    if done:
        if up is False:
            return "fail", "콘솔에는 끝났다고 되어 있는데 ROS 에 없다 — 콘솔에서 이 단계를 되감고(rewind) 다시"
        return "skip", "콘솔에서 이미 끝남"
    if up:
        if stage["skippable"]:
            return "console_skip", "콘솔 밖에서 이미 떠 있다 — 콘솔에는 건너뜀으로 적는다"
        return "fail", "콘솔 밖에서 이미 떠 있는데 콘솔이 건너뛸 수 없는 단계다 — 두 번 띄우지 않게 멈춘다(그 프로세스를 내리거나 콘솔에서 띄울 것)"
    return "run", ""


def print_plan(stages: list[dict], done: dict, graph: dict | None, console_note: str, out=print) -> None:
    out(f"[bringup] 콘솔 {URL} — {console_note}")
    for st in stages:
        up = None if graph is None else already_up(st["id"], graph)
        act, why = decide(st, done.get(st["id"]), up)
        tag = {"run": "실행", "skip": "건너뜀", "console_skip": "건너뜀(콘솔 밖)", "fail": "★멈춤"}[act]
        out(f"  {st['id']:18s} [{tag}]{' 실기' if st['touches_real'] else ''} {st['title']}" + (f" — {why}" if why else ""))
        for c in st["commands"]:
            mark = "  ★손으로 확인 → Enter" if c["kind"] == "manual" else ""
            out(f"      ({c['kind']}) {c['note'][:110]}{mark}")
            if c["kind"] in ("foreground", "background"):
                out(f"         $ {' '.join(c['argv']).replace(str(SIM2REAL) + '/', '')[:240]}")
    if graph is None:
        out("  (ROS 그래프를 보지 않았다 — 건너뛰기 판단은 실행 때 다시 한다)")


# ---------------------------------------------------------------- 콘솔 HTTP
class Api:
    def __init__(self, url: str = URL, http=None) -> None:
        self.url, self.token = url.rstrip("/"), None
        self._http = http or self._urllib

    def _urllib(self, method: str, path: str, body: dict | None, headers: dict) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={**headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {"error": str(e)}

    def get(self, path: str) -> dict:
        code, body = self._http("GET", path, None, {})
        if code != 200:
            raise BringupError(f"GET {path}: {code} {body}")
        return body

    def post(self, path: str, data: dict, method: str = "POST") -> dict:
        h = {"X-S2R-Console": "1", **({"X-S2R-Lease": self.token} if self.token else {})}
        code, body = self._http(method, path, data, h)
        if code not in (200, 202):
            why = "; ".join(map(str, body.get("reasons") or ()))
            raise BringupError(f"{path}: {body.get('error', body)}" + (f" ({why})" if why else ""))
        return body

    def up(self) -> bool:
        try:
            self.get("/api/state")
            return True
        except (OSError, BringupError):
            return False

    def lease(self) -> None:
        self.token = self.post("/api/lease", {"operator": OPERATOR})["token"]

    def release(self) -> None:
        if self.token:
            try:
                self.post("/api/lease", {}, method="DELETE")
            except (OSError, BringupError):
                pass
            self.token = None


def start_console(profile: str, log: Path, api: Api) -> None:
    """콘솔을 이 PC 에 띄운다(setsid — 실험기가 끝나도 남는다). 프로파일은 띄우면서 연다."""
    # 콘솔이 띄우는 손 드라이버 · pd launch 는 이 셸의 환경을 물려받는다 — 오버레이가 없으면 그 단계에서 바로 죽는다
    missing = [p for p in ("rh56f1_interfaces", "policy_control") if importlib.util.find_spec(p) is None]
    if missing:
        raise BringupError(f"{', '.join(missing)} 를 못 찾는다 — 먼저 `source ~/rl_ws/robot_control/ros_ws/install/setup.bash"
                           f" && source ~/rl_ws/sim2real/install/setup.bash` 뒤 다시 실행")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SIM2REAL / "deploy/s2r_console"), os.environ.get("PYTHONPATH")]))}
    with open(log, "ab") as f:
        subprocess.Popen([sys.executable, "-m", "s2r_console", "--profile", profile, "--operator", OPERATOR],
                         cwd=SIM2REAL, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < CONSOLE_UP_TIMEOUT:
        if api.up():
            return
        time.sleep(0.5)
    raise BringupError(f"콘솔이 {CONSOLE_UP_TIMEOUT:.0f} s 안에 안 떴다 — {log}")


def mission_rows(state: dict) -> dict:
    return {r["id"]: r for r in ((state.get("session") or {}).get("mission") or {}).get("rows", [])}


def _runner(state: dict, sid: str) -> dict | None:
    for v in ((state.get("session") or {}).get("runners") or {}).values():
        if v.get("stage") == sid:
            return v
    return None


# ---------------------------------------------------------------- 실행
def run_stage(api: Api, sid: str, ask=input, out=print, timeout: float = STAGE_TIMEOUT, poll: float = 0.5) -> None:
    """단계 하나 — 실행을 걸고 끝날 때까지. 수동 단계는 터미널 확인 → ack. 실패 · 타임아웃이면 BringupError."""
    api.post("/api/stage/run", {"stage": sid})
    acked: set[int] = set()
    t0 = time.time()
    while True:
        api.post("/api/lease/renew", {})
        r = _runner(api.get("/api/state"), sid)
        if r is not None:
            for s in r["steps"]:
                if s["kind"] == "manual" and s["status"] == "waiting" and s["index"] not in acked:
                    out(f"\n  ★[{sid}] 손으로 할 일: {s['note']}")
                    ok = ask("    했고 정상이면 Enter · 아니면 n + Enter: ").strip().lower() not in ("n", "no")
                    api.lease()                      # 사람을 기다린 사이 lease(30 s)가 만료됐을 수 있다
                    api.post("/api/stage/ack", {"stage": sid, "index": s["index"], "ok": ok})
                    acked.add(s["index"])
                    t0 = time.time()                 # 사람을 기다린 시간은 타임아웃에서 뺀다
            if not r.get("active") and r.get("outcome"):
                if r["outcome"] == "DONE":
                    return
                bad = next((s for s in r["steps"] if s["status"] == "failed"), None)
                tail = ""
                if bad and bad.get("key"):
                    try:
                        tail = api.get(f"/api/log?key={bad['key']}").get("text", "")[-1500:]
                    except BringupError:
                        pass
                raise BringupError(f"{sid} {r['outcome']}: {r.get('note', '')}"
                                   + (f"\n    실패 단계: {bad['note']} — {bad.get('detail', '')}" if bad else "")
                                   + (f"\n    로그 끝:\n{tail}" if tail else ""))
        if time.time() - t0 > timeout:
            api.post("/api/stage/abort", {"stage": sid})
            raise BringupError(f"{sid}: {timeout:.0f} s 안에 안 끝났다 — 콘솔에 중단을 보냈다")
        time.sleep(poll)


def bringup(api: Api, profile: str, graph_fn, domain: int, *, log: Path, ask=input, out=print) -> dict:
    """콘솔 미션으로 TARGETS 까지. graph_fn() → already_up 의 graph. 돌려주는 dict 는 메타 json 에 그대로 싣는다."""
    rec: dict = {"profile": profile, "console_started": False, "stages": [], "ok": False}
    try:
        if not api.up():
            out(f"[bringup] 콘솔이 꺼져 있다 — 띄운다(프로파일 {profile}, 로그 {log})")
            start_console(profile, log, api)
            rec["console_started"] = True
        api.lease()
        sess = api.get("/api/state").get("session")
        if sess is None:
            api.post("/api/run/open", {"profile": profile})
            sess = api.get("/api/state")["session"]
        prof = sess["profile"]
        if prof["id"] != profile:
            raise BringupError(f"콘솔에 다른 프로파일({prof['id']})이 열려 있다 — 이 실험은 {profile}. 콘솔에서 run 을 끝내거나 --profile {prof['id']}")
        if int(prof["domain"]) != domain:
            raise BringupError(f"콘솔 프로파일 도메인 {prof['domain']} ≠ 이 셸 ROS_DOMAIN_ID {domain}")
        rec["run_id"] = sess.get("run_id")
        stages = plan(Path(prof["mission"]))
        rows, graph = mission_rows(api.get("/api/state")), graph_fn()
        # 11.17 사용자: 실험 뒤 drivers 까지 내리고 다음에 다시 켜는 흐름 — 콘솔엔 끝났는데 ROS 에 없는 단계는 자동으로 되감는다
        # (되감기는 기록만 바꾼다. 다시 띄우는 실기 단계는 아래 'bringup' 승인을 그대로 거친다). 의존 단계도 콘솔이 같이 되감는다.
        for st in stages:
            sid = st["id"]
            if rows.get(sid, {}).get("done") and already_up(sid, graph) is False:
                api.post("/api/stage/rewind", {"stage": sid})
                out(f"[bringup] ↶ {sid} 되감음 — 콘솔엔 끝났는데 ROS 에 없다(다시 띄운다)")
                rec.setdefault("rewound", []).append(sid)
                rows = mission_rows(api.get("/api/state"))
        acts = []
        for st in stages:
            act, why = decide(st, rows.get(st["id"], {}).get("done"), already_up(st["id"], graph))
            if act == "fail":
                raise BringupError(f"{st['id']}: {why}")
            acts.append((st, act, why))
        need = [st["id"] for st, act, _ in acts if act == "run" and st["touches_real"] and not rows.get(st["id"], {}).get("approved")]
        if need:
            out(f"[bringup] 실기 단계 승인 필요: {' · '.join(need)}")
            if ask("  이 단계들을 승인하고 실행하려면 'bringup' 입력: ").strip() != "bringup":
                raise BringupError("승인 안 함 — 아무것도 실행하지 않았다")
            api.lease()                              # 입력을 기다린 사이 lease(30 s)가 만료됐을 수 있다
        for st, act, why in acts:
            sid = st["id"]
            if act == "skip":
                out(f"[bringup] {sid} 건너뜀 — {why}")
            elif act == "console_skip":
                api.post("/api/stage/skip", {"stage": sid})
                out(f"[bringup] {sid} 건너뜀 — {why}")
            else:
                if sid in need:
                    api.post("/api/approve", {"stage": sid, "typed": sid, "note": "tilt_comp_ab --bringup 터미널 일괄 승인"})
                out(f"[bringup] ▶ {sid} — {st['title']}", )
                run_stage(api, sid, ask=ask, out=out)
                out(f"[bringup] ✓ {sid}")
            rec["stages"].append({"id": sid, "action": act, "why": why, "t": time.time()})
        rec["ok"] = True
        return rec
    except BringupError as exc:
        rec["error"] = str(exc)
        return rec
    finally:
        api.release()


def unit_off(api: Api, key: str, retry_s: float = 0.0, poll: float = 0.5) -> None:
    """콘솔 유닛 하나 끄기. 콘솔의 pd 상태는 상태 토픽 묶음이라 release 직후 잠깐 잡은 채로 남는다 — 그 거부만 retry_s 동안 다시."""
    t0 = time.time()
    while True:
        try:
            api.post("/api/unit", {"key": key, "on": False})
            return
        except BringupError as exc:
            if not any(m in str(exc) for m in UNIT_RETRY) or time.time() - t0 > retry_s:
                raise
        time.sleep(poll)


def _alive(api: Api, sids) -> list[str]:
    units = (api.get("/api/state").get("session") or {}).get("units") or {}
    return sorted((k for k, v in units.items() if k.split("#")[0] in sids and v.get("alive")),
                  key=lambda k: (OFF_ORDER.index(k.split("#")[0]), k))


def teardown(api: Api, stages, out=print, *, keep_drivers: str | None, retry_s: float = 0.0, poll: float = 0.5) -> dict:
    """11.17 사용자 "실험 종료하면 자동으로 bringup까지 종료". `stages`(손 · pd 단계) 중 콘솔에 살아 있는 유닛을 끄고, 그것들이 실제로
    내려간 것을 본 뒤 drivers 를 끈다. 문제없이 내린 단계는 rewind 해 다음 브링업이 처음부터 오르게 한다. 이번 실행이 브링업하지
    않았어도 살아 있는 유닛은 끈다. ★keep_drivers = drivers 를 남길 이유(None 이면 내린다) — 호출자는 두 팔 pd 가 팔을 놓은(IDLE)
    뒤에만 None 을 준다(토크가 빠진다). 손 · pd 를 하나라도 못 내리면 drivers 는 남긴다. 돌려주는 dict 는 메타에 싣는다."""
    rec: dict = {"off": [], "rewound": [], "failed": [], "drivers_kept": keep_drivers}
    bad: set[str] = set()
    stages = [s for s in stages if s != "drivers"] + ([] if keep_drivers else ["drivers"])
    try:
        api.lease()
        for group in ([s for s in OFF_ORDER if s in stages and s != "drivers"], [s for s in ("drivers",) if s in stages]):
            if group == ["drivers"] and bad:
                rec["drivers_kept"] = f"앞 유닛을 다 내리지 못했다({' · '.join(sorted(bad))})"
                bad.add("drivers")
                break
            sent = []
            for key in _alive(api, group):
                try:
                    unit_off(api, key, retry_s, poll)
                    sent.append(key)
                except BringupError as exc:
                    rec["failed"].append(f"{key}: {exc}")
                    bad.add(key.split("#")[0])
            t0 = time.time()
            while (up := [k for k in _alive(api, group) if k in sent]) and time.time() - t0 < UNIT_DOWN_S:
                time.sleep(poll)
            for k in up:                                   # 끄기 직전 규칙을 다시 본 콘솔이 거뒀을 수 있다
                rec["failed"].append(f"{k}: 끄기를 보냈는데 {UNIT_DOWN_S:g} s 뒤에도 떠 있다")
                bad.add(k.split("#")[0])
            rec["off"] += [k for k in sent if k not in up]
        for sid in REWIND_ORDER:
            if sid in stages and sid not in bad and mission_rows(api.get("/api/state")).get(sid, {}).get("done"):
                try:
                    api.post("/api/stage/rewind", {"stage": sid})
                    rec["rewound"].append(sid)
                except BringupError as exc:
                    rec["failed"].append(f"rewind {sid}: {exc}")
    except (OSError, BringupError) as exc:
        rec["failed"].append(f"teardown: {exc}")
        if not any(k.split("#")[0] == "drivers" for k in rec["off"]):
            rec["drivers_kept"] = rec["drivers_kept"] or f"콘솔 오류: {exc}"
    finally:
        api.release()
    for k in rec["off"]:
        out(f"[teardown] ▼ {k}")
    for f in rec["failed"]:
        out(f"[teardown] ✗ {f}")
    if rec["rewound"]:
        out(f"[teardown] ↶ rewind {' · '.join(rec['rewound'])} — 다음 실행이 처음부터 다시 오른다")
    if rec["drivers_kept"]:
        out(f"[teardown] ★팔 브링업(drivers)은 남김 — {rec['drivers_kept']}")
    return rec
