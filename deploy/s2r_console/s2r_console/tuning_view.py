"""tuning 탭 — RH56F1 정지 마찰 실험(analysis/friction) 을 팔마다 보여 준다. 읽기만 한다.

정본은 analysis/friction/common/RUNBOOK.md · common/scripts/new_experiment.sh 다. 이 파일은 그 표와 명령 줄을
화면에 옮겨 적을 뿐 실행하지 않는다(로봇 명령 0). 실험 폴더 = <YYMMDD>_<시험>_<side>/ (setup/ · runs/ · result/).
trial 결과는 stationary_tracking.py run 이 쓰는 runs/<trial>/summary.json(pass · eligible · excluded_reasons).
파일은 mtime 이 같으면 다시 읽지 않는다(상황판 스냅샷이 초마다 부른다).
"""
from __future__ import annotations

import json
import re
import shlex
from datetime import datetime
from pathlib import Path

from .tilt_view import _cached

ROOT = Path("analysis/friction")
ARMS = ("right", "left")
#: RUNBOOK "시험 목록" 표 그대로 — (키, 이름, 폴더 이름 조각, cfg, stage, offset_deg, repeats, trial 수). 팔마다 위에서 아래 순서.
TESTS = (("F1off", "F1 끔", "F1off", "off", "F1", "2.0", "3", 24),
         ("F3", "F3", "F3", "off", "F3", "3.0", "4", 32),
         ("F4", "F4", "F4", "off", "F4", "0.4", "3", 24),
         ("F1on", "F1 켬", "F1on", "fs", "F1", "2.0", "3", 24),
         ("F2x05", "F2 ×0.5", "F2x05", "fs_i050", "F2", "2.0", "3", 24),
         ("F2x1", "F2 ×1", "F2x1", "fs_i100", "F2", "2.0", "3", 24),
         ("F2x2", "F2 ×2", "F2x2", "fs_i200", "F2", "2.0", "3", 24))
CONTRACT = "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json"
#: 셋 다 source 해야 pd_node 가 rh56f1_interfaces 를 찾는다(11.09 실측: 빠지면 ModuleNotFoundError)
ROS_ENV = ("source /opt/ros/humble/setup.bash && source ~/rl_ws/robot_control/ros_ws/install/setup.bash && "
           "source ~/rl_ws/sim2real/install/setup.bash && ")
DDS = ("ROS_DOMAIN_ID=50 ROS_LOCALHOST_ONLY=0 "
       "FASTRTPS_DEFAULT_PROFILES_FILE=/run/user/1000/s2r_fastdds_lan_192.168.0.50.xml")
#: 다시 하면 이름 끝에 _b · _c(new_experiment.sh "use a new name, e.g. <name>_b")
_NAME = re.compile(r"^(\d{6})_(.+)_(right|left)(?:_[b-z])?$")
_CFG = re.compile(r"^pd_rh56f1_stationary_(right|left)_(.+)\.yaml$")


def _pd_launch(side: str, cfg: str) -> str:
    pd_cfg = f"deploy/policy_control/config/stationary_1108/pd_rh56f1_stationary_{side}_{cfg}.yaml"
    return (f"ros2 launch deploy/policy_control/launch/pd_controller.launch.py "
            f"contract:={CONTRACT} robot:=deploy/policy_control/config/robots/rh56f1_{side}_real.yaml "
            f"pd_config:={pd_cfg} sides:={side} execute:=true stage:=full")


def _engage(side: str) -> str:
    return (f"python3 deploy/policy_control/tools/episode_ctl.py --side {side} --only pd_engage "
            "--hold-s 10 --execute --approve pd_engage")


def commands(side: str, test: tuple, folder: str, remaining: list | None = None) -> dict:
    """한 실험의 명령 줄(복사해 쓰는 글) — RUNBOOK 1 · 3 절과 같은 인자.
    remaining 을 주면 trial 줄은 남은 것만 돈다(RUNBOOK 4 절: 이어서 하면 남은 trial 만)."""
    _key, _label, _frag, cfg, stage, off, rep, _n = test
    return {
        "prepare": f"analysis/friction/common/scripts/new_experiment.sh {folder} {side} {cfg} {stage} {off} {rep}",
        "pd": f"{ROS_ENV}{DDS} {_pd_launch(side, cfg)}",
        "engage": f"{DDS} {_engage(side)}",
        "hold": f"{ROS_ENV}{DDS} ros2 service call /policy_control/pd_{side}/hold std_srvs/srv/Trigger",
        "trials": (f"E=analysis/friction/{folder}; for t in {' '.join(remaining) if remaining else '$(cat $E/setup/trials.txt)'}; do {DDS} "
                   ".venv/bin/python deploy/policy_control/tools/stationary_tracking.py run "
                   "--plan $E/setup/plan/plan.json --trial $t --out $E/runs/$t --execute || break; done"),
    }


def _sh(body: str) -> tuple:
    """콘솔 감독자가 띄울 argv — ROS 셋 · DDS 를 다 싣는다(빠지면 rh56f1_interfaces ModuleNotFoundError, 11.09)."""
    return ("bash", "-c", f"{ROS_ENV}{DDS} exec {body}")


def new_argv(repo: Path, side: str, key: str) -> tuple[str, tuple]:
    """새 실험 준비(new_experiment.sh, 로봇 명령 0) — (새 폴더 이름, argv). 이름은 오늘 날짜 · 있으면 _b …"""
    test = next(t for t in TESTS if t[0] == key)
    names = {p.name for p in (repo / ROOT).glob("*") if p.is_dir()}
    name = _free_name(names, f"{datetime.now().strftime('%y%m%d')}_{test[2]}_{side}")
    _k, _l, _f, cfg, stage, off, rep, _n = test
    return name, ("bash", str(repo / ROOT / "common/scripts/new_experiment.sh"), name, side, cfg, stage, off, rep)


def plan(repo: Path, side: str, name) -> dict:
    """[실험 시작] 한 번에 도는 명령 — pd 기동 · engage · 남은 trial. 막는 이유가 있으면 reasons."""
    m = _NAME.match(name) if isinstance(name, str) else None
    if m is None or m.group(3) != side or not (repo / ROOT / name).is_dir():
        return {"reasons": [f"{side} 팔의 실험 폴더를 고를 것"], "trials": []}
    e = experiment(repo / ROOT / name)
    reasons = []
    if not e["prepared"] or not (repo / ROOT / name / "setup/plan/plan.json").is_file():
        reasons.append("준비 안 된 폴더다 — 새 실험 준비 먼저")
    if not e["cfg"]:
        reasons.append("setup/ 에 PD 설정(pd_rh56f1_stationary_*.yaml)이 없다")
    partial = [t["id"] for t in e["trials"] if t["state"] == "partial"]
    if partial:      # 끊긴 runs/<trial>/ 을 덮어쓰지 않는다 — 사람이 보고 정리한다
        reasons.append(f"끊긴 trial {', '.join(partial)} — runs/ 폴더를 다시 쓰지 않는다(정리하거나 새 폴더로)")
    todo = [t["id"] for t in e["trials"] if t["state"] == "todo"]
    if not todo:
        reasons.append("남은 trial 이 없다")
    if any(not re.fullmatch(r"\w[\w.-]*", t) for t in todo):     # trials.txt 는 파일 입력 — runs/ 밖으로 못 나가게
        reasons.append("trials.txt 에 쓸 수 없는 trial 이름이 있다")
    E = f"{ROOT}/{name}"
    q = shlex.quote
    return {"reasons": reasons, "n": e["n"], "done": e["done"],
            "pd": _sh(_pd_launch(side, e["cfg"])), "engage": _sh(_engage(side)),
            "trials": [(t, _sh(f".venv/bin/python deploy/policy_control/tools/stationary_tracking.py run "
                               f"--plan {E}/setup/plan/plan.json --trial {q(t)} --out {q(f'{E}/runs/{t}')} --execute"))
                       for t in todo]}


def _trial(run: Path) -> dict:
    """runs/<trial>/ 하나 — summary.json 이 없으면 '기록 중 또는 중간에 끊김'."""
    p = run / "summary.json"
    if not p.is_file():
        return {"id": run.name, "state": "partial", "why": "summary.json 없음(도는 중이거나 끊김)"}
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"id": run.name, "state": "partial", "why": f"summary.json 읽기 실패: {exc}"}
    reasons = [str(r) for r in s.get("excluded_reasons") or []]
    ok = bool(s.get("pass")) and s.get("eligible", True) is not False
    return {"id": run.name, "state": "pass" if ok else "excluded", "why": " · ".join(reasons)}


def experiment(folder: Path) -> dict:
    """폴더 하나의 진행 — 준비됨 · trial n/N · 다음 trial · 멈춘 trial."""
    m = _NAME.match(folder.name)
    setup = folder / "setup"
    cfgs = [c.group(2) for c in (_CFG.match(p.name) for p in setup.glob("pd_rh56f1_stationary_*.yaml")) if c]
    tf = setup / "trials.txt"
    planned = [t.strip() for t in tf.read_text(encoding="utf-8").splitlines() if t.strip()] if tf.is_file() else []
    runs = {r.name: _trial(r) for r in sorted((folder / "runs").glob("*")) if r.is_dir()}
    done = [t for t in planned if t in runs]
    stopped = next((runs[t] for t in done if runs[t]["state"] != "pass"), None)
    nxt = next((t for t in planned if t not in runs), None)
    result = sorted(p.name for p in (folder / "result").glob("*")) if (folder / "result").is_dir() else []
    return {"name": folder.name, "date": m.group(1) if m else "", "test": m.group(2) if m else "",
            "side": m.group(3) if m else "", "cfg": cfgs[0] if cfgs else "",
            "path": f"{ROOT}/{folder.name}", "prepared": bool(planned), "n": len(planned), "done": len(done),
            "passed": sum(runs[t]["state"] == "pass" for t in done), "next": nxt,
            "stopped": stopped, "complete": bool(planned) and nxt is None,
            "trials": [runs.get(t) or {"id": t, "state": "todo", "why": ""} for t in planned],
            "extra_runs": sorted(set(runs) - set(planned)), "result": result}


def _free_name(names: set, base: str) -> str:
    """같은 폴더를 다시 쓰지 않는다 — 있으면 _b, _c …(README 규칙)."""
    for name in (base, *(f"{base}_{c}" for c in "bcdefghijklmnopqrstuvwxy")):
        if name not in names:
            return name
    return f"{base}_z"


def _build(repo: Path, today: str) -> dict:
    base = repo / ROOT
    folders = sorted(p for p in base.glob("[0-9]*_*") if p.is_dir() and _NAME.match(p.name))
    exps = [experiment(p) for p in folders]
    names = {e["name"] for e in exps}
    sides = {}
    for side in ARMS:
        rows = []
        for t in TESTS:
            mine = [e for e in exps if e["side"] == side and e["test"] == t[2]]
            fresh = _free_name(names, f"{today}_{t[2]}_{side}")
            rows.append({"key": t[0], "label": t[1], "cfg": t[3], "stage": t[4], "offset_deg": t[5], "repeats": t[6],
                         "trials": t[7], "folders": [e["name"] for e in mine], "new_name": fresh,
                         "commands": {"new": commands(side, t, fresh),
                                      **{e["name"]: commands(side, t, e["name"], [x["id"] for x in e["trials"] if x["state"] == "todo"]
                                                           if e["done"] else None) for e in mine}}})
        sides[side] = rows
    return {"root": str(ROOT), "experiments": {e["name"]: e for e in exps}, "sides": sides,
            "runbook": f"{ROOT}/common/RUNBOOK.md"}


def view(repo: Path) -> dict:
    """상황판 스냅샷 session.tuning — 실패해도 상황판은 돈다(이유는 error)."""
    base = repo / ROOT
    today = datetime.now().strftime("%y%m%d")
    try:
        folders = sorted(p for p in base.glob("[0-9]*_*") if p.is_dir())
        # runs/<trial>/ 이 생기거나 summary.json 이 써지면 다시 읽는다
        watch = [base, *folders, *(f / "runs" for f in folders), *(f / "result" for f in folders),
                 *(r / "summary.json" for f in folders for r in (f / "runs").glob("*"))]
        return _cached(f"tuning:{today}", watch, lambda: _build(repo, today))
    except Exception as exc:                          # noqa: BLE001 — 표시용 집계가 상황판을 멈추지 않게
        return {"experiments": {}, "sides": {}, "error": f"{type(exc).__name__}: {exc}"}
