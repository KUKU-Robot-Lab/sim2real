#!/usr/bin/env python3
"""실기 에피소드 기록과 같이 도는 stationary 기준선 기록(11.08 사용자 승인). 구독만, 명령 발행 없음.

  start <이름> --mission <미션 yaml> [--episode <에피소드 yaml>] --sides right[,left]
      logs/analysis/stationary_tracking_1108/runs/baseline_<YYYYMMDD>_<NN>/ 를 실행 시각에 고르고(NN = 다음 빈 번호)
      snapshot.json + copies/(미션 · 에피소드 · PD 설정 · 게인 · 정책 계약) + README.md(bag 경로)를 쓴 뒤
      팔마다 stationary_recorder.py 를 <폴더>/<팔>/ 로 띄운다(setsid, 이 명령은 바로 끝난다).
  stop <이름>   기록기에 SIGINT, README 에 끝 시각 · 실제 bag 폴더를 적는다.
상태 파일은 logs/analysis/stationary_tracking_1108/active/<이름>.json. ROS 환경은 rh56f1_record.sh 와 같다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "logs/analysis/stationary_tracking_1108"
RECORDER = REPO / "deploy/policy_control/tools/stationary_recorder.py"
SECONDS = 3600                                   # stationary_recorder 상한. 보통 stop 이 먼저 끊는다
POLICY_ID = re.compile(r"^rh56f1/[A-Za-z0-9_]+/[A-Za-z0-9_]+$")
ROS_ENV = os.environ.get("RH56F1_RECORD_ENV") or (
    f"source /opt/ros/humble/setup.bash; source $HOME/rl_ws/robot_control/ros_ws/install/setup.bash; "
    f"source '{REPO}/config/ros_domain.env'; export ROS_DOMAIN_ID=${{ROS_DOMAIN_ID:-$S2R_REAL_DOMAIN}}")


def next_run_dir(base: Path, day: str) -> Path:
    runs = base / "runs"
    taken = [int(m.group(1)) for p in runs.glob(f"baseline_{day}_*") if (m := re.fullmatch(rf"baseline_{day}_(\d+)", p.name))]
    return runs / f"baseline_{day}_{max(taken, default=0) + 1:02d}"


def _md5(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _policy_ids(node) -> set[str]:
    if isinstance(node, dict):
        return set().union(*(_policy_ids(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(_policy_ids(v) for v in node)) if node else set()
    return {node} if isinstance(node, str) and POLICY_ID.match(node) else set()


def _policy_entry(contract: Path) -> dict:
    data = json.loads(contract.read_text(encoding="utf-8"))
    ckpt = data.get("checkpoint")
    ep = re.search(r"ep_?(\d+)", Path(ckpt).name) if ckpt else None
    return {"contract": str(contract.relative_to(REPO)), "checkpoint": ckpt, "checkpoint_md5": data.get("checkpoint_md5"),
            "ep": int(ep.group(1)) if ep else None, "run_dir": data.get("run_dir")}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def _resolve_gains(rel: str) -> Path | None:
    for root in (REPO.parent, Path.home() / "rl_ws"):
        if (root / rel).is_file():
            return root / rel
    return None


def write_snapshot(out: Path, name: str, mission: Path, episode: Path | None) -> dict:
    copies = out / "copies"
    copies.mkdir()
    mission_doc = yaml.safe_load(mission.read_text(encoding="utf-8"))
    artifacts = mission_doc.get("artifacts", {})
    files = {"mission": mission}
    if episode is not None:
        files["episode"] = episode
    pd = {}
    for key, rel in sorted(artifacts.items()):
        if key == "pd" or key.startswith("pd_"):
            path = REPO / rel
            files[key] = path
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            gains_rel = (raw.get("gains") or {}).get("yaml")
            gains_path = _resolve_gains(gains_rel) if gains_rel else None
            pd[key] = {"file": rel, "gains": raw.get("gains"), "stationary_compensation": raw.get("stationary_compensation"),
                       "gains_file": str(gains_path) if gains_path else None,
                       "gains_values": yaml.safe_load(gains_path.read_text(encoding="utf-8")) if gains_path else None}
            if gains_path:
                files[f"{key}_gains"] = gains_path
    ids = _policy_ids(yaml.safe_load(episode.read_text(encoding="utf-8"))) if episode else set()
    contracts = {pid: sorted((REPO / "deploy/policies" / pid).glob("*_contract.json")) for pid in sorted(ids)}
    contracts.update({key: [REPO / rel] for key, rel in artifacts.items() if str(rel).endswith("_contract.json")})
    policies = {}
    for key, paths in contracts.items():
        existing = [p for p in paths if p.is_file()]
        policies[key] = [_policy_entry(p) for p in existing] or {"error": "contract 없음"}
        for p in existing:
            files[f"policy:{key}:{p.name}"] = p
    for key, path in files.items():
        dest = copies / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + "__" + path.name)
        shutil.copy2(path, dest)
    status = _git("status", "--porcelain", "--untracked-files=no")
    snap = {"name": name, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "sim2real_commit": _git("rev-parse", "HEAD"), "sim2real_dirty": bool(status), "dirty_files": status.splitlines(),
            "mission": str(mission.relative_to(REPO)), "episode": str(episode.relative_to(REPO)) if episode else None,
            "pd_configs": pd, "policies": policies,
            "files": {k: {"path": str(p), "md5": _md5(p)} for k, p in files.items()}}
    (out / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snap


def _readme(out: Path, name: str, bag_note: str, sides: list[str]) -> None:
    (out / "README.md").write_text(
        f"# stationary 기준선 기록 {out.name}\n\n- 에피소드 기록 이름: `{name}`\n- 팔: {', '.join(sides)}"
        f" (팔마다 `<팔>/` 아래 stationary_recorder 출력)\n- bag: {bag_note}\n"
        "- 설정 · 정책 · 커밋: `snapshot.json`, 원본 사본 `copies/`\n", encoding="utf-8")


def start(args) -> int:
    state = BASE / "active" / f"{args.name}.json"
    if state.exists():
        print(f"{args.name} 기준선 기록이 남아 있다 — 먼저 stop", file=sys.stderr)
        stop(argparse.Namespace(name=args.name))
    out = next_run_dir(BASE, dt.date.today().strftime("%Y%m%d"))
    out.mkdir(parents=True)
    sides = [s for s in args.sides.split(",") if s]
    write_snapshot(out, args.name, Path(args.mission).resolve(), Path(args.episode).resolve() if args.episode else None)
    _readme(out, args.name, f"`logs/bags/<YYYYMMDD_HHMMSS>_{args.name}/`(rh56f1_record.sh, stop 때 실제 경로를 적는다)", sides)
    pids = {}
    for side in sides:
        cmd = (f"{ROS_ENV}; exec nice -n 10 python3 '{RECORDER}' --side {side} --seconds {SECONDS} --out '{out / side}'")
        proc = subprocess.Popen(["bash", "-c", cmd], stdin=subprocess.DEVNULL, stdout=(out / f"{side}.log").open("w"),
                                stderr=subprocess.STDOUT, start_new_session=True)
        pids[side] = proc.pid
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"out": str(out), "pids": pids, "started": time.time(), "sides": sides}) + "\n")
    time.sleep(1.0)
    dead = [s for s, p in pids.items() if not Path(f"/proc/{p}").exists()]
    print(f"stationary 기준선 기록 -> {out.relative_to(REPO)} pid {pids}")
    if dead:                                         # bag 은 그대로 돈다. 기준선만 빠졌다고 알린다
        print(f"경고: 기준선 기록기 {dead} 가 바로 끝났다 — {out.relative_to(REPO)}/<팔>.log", file=sys.stderr)
    return 0                                         # 기준선 실패로 에피소드를 막지 않는다


def stop(args) -> int:
    state = BASE / "active" / f"{args.name}.json"
    if not state.exists():
        print(f"{args.name} 기준선 기록 없음(이미 끝남)")
        return 0
    info = json.loads(state.read_text())
    for pid in info["pids"].values():
        try:
            os.kill(pid, signal.SIGINT)              # stationary_recorder 가 SIGINT 에 저장하고 끝낸다
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for pid in info["pids"].values():
        while time.monotonic() < deadline and Path(f"/proc/{pid}").exists():
            time.sleep(0.2)
    alive = [p for p in info["pids"].values() if Path(f"/proc/{p}").exists()]
    bags = sorted((p for p in (REPO / "logs/bags").glob(f"*_{args.name}") if p.stat().st_mtime >= info["started"] - 60),
                  key=lambda p: p.name)
    out = Path(info["out"])
    bag = f"`{bags[-1].relative_to(REPO)}/`" if bags else f"찾지 못함(`logs/bags/*_{args.name}`)"
    _readme(out, args.name, bag, info["sides"])
    with (out / "README.md").open("a", encoding="utf-8") as f:
        f.write(f"- 끝: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}"
                + (f" · 15 s 안에 안 끝난 기록기 pid {alive}" if alive else "") + "\n")
    state.unlink()
    print(f"stationary 기준선 기록 끝 -> {out.relative_to(REPO)} bag {bag}" + (f" · 남은 pid {alive}" if alive else ""))
    return 0                                         # 남은 pid 는 위에 적었다. bag 마무리를 막지 않는다


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("start")
    a.add_argument("name")
    a.add_argument("--mission", required=True)
    a.add_argument("--episode")
    a.add_argument("--sides", default="right")
    b = sub.add_parser("stop")
    b.add_argument("name")
    args = ap.parse_args(argv)
    return start(args) if args.cmd == "start" else stop(args)


if __name__ == "__main__":
    sys.exit(main())
