#!/usr/bin/env python3
"""bag_gate — 에피소드 bag(arm · hand · pd)에 회 구간만 남긴다(11.22 사용자 '녹화량 줄이기', general_lab 과 맞춘 조건).

    python3 scripts/nodes/bag_gate.py watch --gate /policy_control/status/<runner> --name episode_<x> --start-cmd '<기록 시작 명령>'
    python3 scripts/nodes/bag_gate.py wait --name episode_<x>        # 미션 '기록 끝' 단계 — 마무리기가 끝날 때까지 기다림
    (close 는 watch 가 띄우는 마무리기다)

전에는 미션 처음부터 끝까지(승인 대기 · 에피소드 사이 포함) 기록해 11.20 e2e 한 세션이 pd 4.7G · hand 2.7G 였다.
Humble 의 ros2 bag record(rosbag2 0.15)는 일시정지 서비스가 없다(11.22 시험: /rec_t 에 파라미터 서비스만) → 끄고 켜는 대신
  - 기록기는 미션 단계에서 그대로 시작한다(토픽 찾기를 미리 끝내 둔다).
  - 회가 끝나면(SUCCESS · FAILURE · STOPPED, 또는 상태가 --post 초 끊김) --post 초 뒤 rh56f1_record.sh stop 으로 끈다
    (손을 편 뒤 홈 복귀 중 컵이 앉는 +8~20 s, 실패 뒤 상태까지 — general_lab 11.22).
  - 끈 뒤 arm · hand · pd 에서 '첫 노드 시작 - --pre 초' 앞 메시지를 지우고(VACUUM) ros2 bag reindex 로 metadata.yaml 을 다시 쓴다.
    transient_local 토픽(손끝 영점 등)은 잘린 앞쪽의 마지막 한 개를 남기고 시각만 자른 지점으로 옮긴다. camera 는 자르지 않는다
    (stop 이 바로 mp4 를 만들기 시작한다). 잘라낸 내용은 bag 폴더의 bag_trim.json.
  - --post 안에 다음 회가 시작되면 마무리를 취소하고 이어 쓴다. 이미 껐으면 --start-cmd 로 새 bag 을 연다(이때는 앞 여유 없음).
마무리는 따로 떨어진 프로세스(setsid)라 미션의 '에피소드 노드들 정지'가 watch 를 먼저 꺼도 끝까지 간다.
시각은 모두 이 PC 의 벽시계(ns) — rosbag2 가 메시지에 찍는 수신 시각과 같은 시계다.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[2]
RECORD_SH = SIM2REAL / "deploy/policy_control/tools/rh56f1_record.sh"
_ENDED = ("SUCCESS", "FAILURE", "STOPPED")
TRIM_KINDS = ("arm", "hand", "pd")
CANCELLED = 3            # close 가 대기 중 SIGTERM 을 받았다 — bag 은 아직 돈다
PID_FILE = "bag_gate_close.pid"
STOPPED_FILE = "bag_gate_stopped"     # close 가 stop 을 마쳤다(watch 가 다시 열기 전에 본다)


class Gate:
    """상태 JSON → 'open'(회 시작) · 'close'(회 끝) · None. 같은 회 안의 반복 상태는 None."""

    def __init__(self, post_s: float) -> None:
        self.post_s = post_s
        self.active = False
        self.seen = False          # 회를 한 번이라도 봤다(그 전에는 끌 것이 없다)
        self.last_msg: float | None = None
        self.episode_id: str | None = None

    def update(self, data: str, now: float) -> str | None:
        try:
            s = json.loads(data)
            status, eid, runs = s.get("status"), s.get("episode_id"), s.get("runs") or {}
        except (ValueError, AttributeError):
            return None
        self.last_msg = now
        active = bool(runs) and status not in _ENDED
        if active:
            self.episode_id = eid
        if active == self.active:
            return None
        self.active = active
        if active:
            self.seen = True
            return "open"
        return "close" if self.seen else None

    def tick(self, now: float) -> str | None:
        """상태가 --post 초 넘게 안 오면(러너가 죽음) 끝으로 본다."""
        if self.active and self.last_msg is not None and now - self.last_msg > self.post_s:
            self.active = False
            return "close"
        return None


# ---------------------------------------------------------------- 기록 폴더 · 자르기
def record_dir(name: str) -> Path | None:
    """rh56f1_record.sh 의 name_dirs 와 같은 자리(<runs>/<날>/rh56f1_record/<이름>) — 기록 중(arm.pid)인 것, 없으면 최근 것."""
    env = os.environ.get("RH56F1_RECORD_DIR")
    cands = [Path(env) / name] if env else sorted((SIM2REAL / "logs/runs").glob(f"*/rh56f1_record/{name}"))
    cands = [d for d in cands if d.is_dir()]
    live = [d for d in cands if (d / "arm.pid").exists()]
    return (live or cands or [None])[-1]


def _latched_ids(c: sqlite3.Connection) -> set[int]:
    ids = set()
    for tid, qos in c.execute("SELECT id, offered_qos_profiles FROM topics"):
        if qos and "durability: 1" in qos:       # rmw 열거값 1 = transient_local
            ids.add(tid)
    return ids


def trim_bag(bag: Path, since_ns: int, reindex: bool = True) -> dict:
    """since_ns 앞 메시지를 지운다(transient_local 은 토픽마다 마지막 하나를 since_ns 시각으로 남김)."""
    dbs = sorted(bag.glob("*.db3"))
    out = {"bag": bag.name, "bytes_before": sum(p.stat().st_size for p in dbs), "dropped": 0, "latched_kept": 0}
    for db in dbs:
        c = sqlite3.connect(db)
        try:
            keep = []
            for tid in _latched_ids(c):
                row = c.execute("SELECT id FROM messages WHERE topic_id=? AND timestamp<? ORDER BY timestamp DESC LIMIT 1",
                                (tid, since_ns)).fetchone()
                if row:
                    keep.append(row[0])
            if keep:
                q = ",".join("?" * len(keep))
                c.execute(f"UPDATE messages SET timestamp=? WHERE id IN ({q})", (since_ns, *keep))
                out["latched_kept"] += len(keep)
            n = c.execute("DELETE FROM messages WHERE timestamp<?", (since_ns,)).rowcount
            c.commit()
            out["dropped"] += n
            if n:
                c.execute("VACUUM")
        finally:
            c.close()
    out["bytes_after"] = sum(p.stat().st_size for p in dbs)
    if reindex and out["dropped"]:
        r = subprocess.run(["bash", "-c", f"source /opt/ros/humble/setup.bash && ros2 bag reindex '{bag}' -s sqlite3"],
                           capture_output=True, text=True)
        out["reindex_rc"] = r.returncode
    return out


def _recorder_pids(d: Path) -> dict[str, int]:
    out = {}
    for k in TRIM_KINDS:
        try:
            out[k] = int((d / f"{k}.pid").read_text().strip())
        except (OSError, ValueError):
            pass
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------- 하위 명령
def cmd_close(args: argparse.Namespace) -> int:
    d = record_dir(args.name)
    if d is None:
        print(f"[bag_gate] {args.name}: 기록 폴더 없음", flush=True)
        return 1
    (d / STOPPED_FILE).unlink(missing_ok=True)
    (d / PID_FILE).write_text(str(os.getpid()))
    try:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(CANCELLED))
        time.sleep(args.after)
        for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):    # 여기부터는 끝까지 간다(자르는 도중 끊기지 않게)
            signal.signal(sg, signal.SIG_IGN)
        last = d / "last_out"
        out = Path(last.read_text().strip()) if last.exists() else None     # stop 이 last_out 을 지운다 — 먼저 읽는다
        pids = _recorder_pids(d)                                             # stop 이 *.pid 를 지운다 — 먼저 읽는다
        subprocess.run(["bash", str(RECORD_SH), "stop", args.name], check=False)
        (d / STOPPED_FILE).touch()
        if out is None or args.since <= 0:
            return 0
        alive = [k for k, pid in pids.items() if _alive(pid)]
        if alive:       # 기록기가 아직 쓰는 bag 의 sqlite 를 지우면 bag 이 깨진다 — 자르지 않고 둔다
            print(f"[bag_gate] {out.name}: 기록기 {alive} 가 아직 살아 있어 앞 자르기를 건너뜀(bag 은 전체 기록)", flush=True)
            return 0
        os.nice(19)
        res = {"since_ns": args.since, "since": time.strftime("%H:%M:%S", time.localtime(args.since / 1e9)),
               "pre_s": args.pre, "post_s": args.after, "bags": []}
        for k in TRIM_KINDS:
            if (out / k).is_dir():
                r = trim_bag(out / k, args.since)
                res["bags"].append(r)
                print(f"[bag_gate] {out.name}/{k}: {r['dropped']} 개 잘라냄 "
                      f"{r['bytes_before'] / 1e6:.0f} → {r['bytes_after'] / 1e6:.0f} MB", flush=True)
        (out / "bag_trim.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    finally:
        (d / PID_FILE).unlink(missing_ok=True)


def cmd_wait(args: argparse.Namespace) -> int:
    d = record_dir(args.name)
    pf = d / PID_FILE if d else None
    t0 = time.monotonic()
    while pf is not None and pf.exists() and time.monotonic() - t0 < args.timeout:
        try:
            os.kill(int(pf.read_text().strip() or 0), 0)
        except (ValueError, ProcessLookupError):
            break
        time.sleep(1.0)
    waited = time.monotonic() - t0
    if waited > 1:
        print(f"[bag_gate] {args.name} 마무리 기다림 {waited:.0f} s", flush=True)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile
    from std_msgs.msg import String

    gate = Gate(args.post)
    st: dict = {"t_open": None, "closer": None, "stopped": False}

    def log(msg: str) -> None:
        print(f"[bag_gate] {time.strftime('%H:%M:%S')} {msg}", flush=True)

    def on_open() -> None:
        p = st["closer"]
        if p is not None:
            st["closer"] = None
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
            try:
                rc = p.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                rc = None                     # 이미 stop · 자르기 중 — stop 이 끝난 것만 기다린다(자르기는 두고)
            if rc in (CANCELLED, -signal.SIGTERM):   # -15 = 신호 처리기를 걸기 전(대기 시작 전)에 꺼짐 — 아무것도 안 껐다
                log(f"회 {gate.episode_id} 시작 — 마무리 취소, 이어 씀")
                return
            d = record_dir(args.name)
            t0 = time.monotonic()
            while rc is None and d is not None and not (d / STOPPED_FILE).exists() and p.poll() is None \
                    and time.monotonic() - t0 < 30:
                time.sleep(0.2)
            st["stopped"] = True
        if st["stopped"]:
            subprocess.run(["bash", "-lc", args.start_cmd], check=False)
            st["stopped"], st["t_open"] = False, time.time_ns()
            log(f"회 {gate.episode_id} 시작 — bag 을 새로 염(앞 여유 없음)")
            return
        if st["t_open"] is None:
            st["t_open"] = time.time_ns()
            log(f"회 {gate.episode_id} 시작 — 이 시각 - {args.pre:g} s 앞은 나중에 잘라냄")

    def on_close(reason: str) -> None:
        since = (st["t_open"] or 0) - int(args.pre * 1e9) if st["t_open"] else 0
        d = record_dir(args.name)
        logf = open(d / "bag_gate_close.log", "a") if d is not None else subprocess.DEVNULL   # watch 가 먼저 꺼져도 출력이 막히지 않게
        st["closer"] = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "close", "--name", args.name, "--after", str(args.post),
             "--since", str(since), "--pre", str(args.pre)], start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=logf, stderr=subprocess.STDOUT)
        if d is not None:
            (d / PID_FILE).write_text(str(st["closer"].pid))   # wait 가 마무리기가 PID 를 쓰기 전 틈에 지나치지 않게
            logf.close()
        log(f"{reason} — {args.post:g} s 뒤 bag 마무리")

    def on_status(m) -> None:
        ev = gate.update(m.data, time.monotonic())
        if ev == "open":
            on_open()
        elif ev == "close":
            on_close(f"회 {gate.episode_id} 끝")

    def on_tick() -> None:
        if gate.tick(time.monotonic()) == "close":
            on_close(f"상태 {args.post:g} s 끊김")

    rclpy.init()
    node = rclpy.create_node("bag_gate_" + args.name.replace("-", "_"))
    node.create_subscription(String, args.gate, on_status, QoSProfile(depth=10))
    node.create_timer(0.5, on_tick)
    log(f"{args.gate} → {args.name} bag 을 회 구간(-{args.pre:g} s ~ +{args.post:g} s)만 남김")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="runner 상태를 보고 회가 끝나면 마무리기를 띄운다")
    w.add_argument("--gate", required=True, help="에피소드 runner 상태 토픽(/policy_control/status/<runner>)")
    w.add_argument("--name", required=True, help="rh56f1_record.sh 기록 이름(episode_<x>)")
    w.add_argument("--start-cmd", required=True, help="이미 끈 뒤 다음 회가 시작될 때 다시 여는 명령(bash -lc)")
    w.add_argument("--pre", type=float, default=5.0, help="첫 노드 시작 앞에 남기는 초")
    w.add_argument("--post", type=float, default=30.0, help="회가 끝난 뒤 더 쓰는 초")
    c = sub.add_parser("close", help="(watch 가 띄움) --after 초 뒤 stop + 앞 자르기")
    c.add_argument("--name", required=True)
    c.add_argument("--after", type=float, required=True)
    c.add_argument("--since", type=int, required=True, help="이 시각(ns) 앞을 자른다. 0 이면 안 자른다")
    c.add_argument("--pre", type=float, default=0.0, help="기록용(bag_trim.json)")
    t = sub.add_parser("wait", help="마무리기가 끝날 때까지 기다림")
    t.add_argument("--name", required=True)
    t.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv)
    return {"watch": cmd_watch, "close": cmd_close, "wait": cmd_wait}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
