"""실기 에피소드 폴더 + bag 읽기 공용(place_run_report · tilt_spout_report). 파일만 읽는다(ROS 노드 · 로봇 없음).

  episode 폴더 = logs/runs/<YYMMDD>/episode/<HHMMSS>_<이름> (episode_*.jsonl · meta.json · tilt_tin_*.json …)
  bag 폴더     = <root>/<HHMMSS>_episode_<이름>/{arm,pd,hand}/*_0.db3 — 에피소드 시각보다 조금 앞에 시작한다
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SIM2REAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM2REAL / "deploy" / "policy_control"))
BACKUP_RUNS = Path("/media/user/DATA/5090_backup_1116/sim2real/logs/runs")   # 11.16 이전 bag 백업
BAG_LEAD_S = 60.0          # bag 은 에피소드 폴더 시각보다 이 안에서 먼저 시작한다(실측 3~4 s)


def default_roots(ep_dir: Path) -> list[Path]:
    day = ep_dir.parent.parent.name
    return [SIM2REAL / "logs" / "runs" / day / "bags", BACKUP_RUNS / day / "bags"]


def find_bag(ep_dir: Path, roots: list[Path]) -> Path:
    """에피소드 폴더 <HHMMSS>_<이름> → 같은 이름 · 시각이 BAG_LEAD_S 안에서 가장 가까운 앞선 bag 폴더."""
    hhmmss, name = ep_dir.name.split("_", 1)
    t_ep = int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])
    best = None    # shortcut: 자정을 넘긴 bag(전날 폴더)은 못 찾는다 — 그때는 --bag-dir
    for root in roots:
        for d in root.glob(f"*_episode_{name}") if root.is_dir() else []:
            h = d.name.split("_", 1)[0]
            dt = t_ep - (int(h[:2]) * 3600 + int(h[2:4]) * 60 + int(h[4:]))
            if 0 <= dt <= BAG_LEAD_S and (best is None or dt < best[0]):
                best = (dt, d)
    if best is None:
        raise FileNotFoundError(f"{ep_dir.name}: bag 없음({', '.join(map(str, roots))}) — --bag-dir 로 준다")
    return best[1]


def db_of(bag: Path, group: str) -> Path:
    got = sorted((bag / group).glob("*.db3"))
    if not got:
        raise FileNotFoundError(f"{bag}/{group}: db3 없음")
    return got[0]


def read(db: Path, topics: list[str], t0: float | None = None, t1: float | None = None) -> dict:
    """{topic: [(받은 시각 s, 메시지)]} — 없는 토픽은 빈 목록. t0 · t1(s)로 자른다."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    have = {n: (i, t) for i, n, t in con.execute("select id, name, type from topics")}
    lo = -1 if t0 is None else int(t0 * 1e9)
    hi = 1 << 62 if t1 is None else int(t1 * 1e9)
    out = {}
    for name in topics:
        if name not in have:
            out[name] = []
            continue
        i, typ = have[name]
        T = get_message(typ)
        rows = con.execute("select timestamp, data from messages where topic_id=? and timestamp between ? and ? order by timestamp",
                           [i, lo, hi])
        out[name] = [(ts * 1e-9, deserialize_message(d, T)) for ts, d in rows]
    con.close()
    return out


def events(ep_dir: Path) -> list[dict]:
    """episode_*.jsonl → 사건 목록(각각 't' = 이 PC 시계 epoch s, bag 시각과 같은 시계)."""
    out = []
    for f in sorted(ep_dir.glob("episode_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                d["t"] = datetime.strptime(d["timestamp"], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
                out.append(d)
    if not out:
        raise FileNotFoundError(f"{ep_dir}: episode_*.jsonl 없음")
    return out


def episode_ctx(ep_dir: Path) -> tuple[dict, dict, dict]:
    """meta.json · 그 에피소드 yaml(지금 작업 트리 판, 없으면 {}) · 물체 → FP++ live 토픽."""
    import yaml
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    p = Path(meta.get("episode_path", ""))
    ep_yaml = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.is_file() else {}
    live = {k: str(v.get("topic", "")).rsplit("/", 1)[0] + "/live_pose" for k, v in (ep_yaml.get("objects") or {}).items()}
    return meta, ep_yaml, live


def node_windows(evs: list[dict], pred) -> list[dict]:
    """pred(enter 사건) 가 참인 노드마다 {id, enter, t0, t1, end} — 끝 = 그 뒤 첫 exit · failure · stop 사건."""
    out = []
    for k, e in enumerate(evs):
        if e["event"] != "enter" or not pred(e):
            continue
        end = next((x for x in evs[k + 1:] if x["state_id"] == e["state_id"] and x["event"] in ("exit", "failure", "stop")), None)
        out.append({"id": e["state_id"], "enter": e, "t0": e["t"], "t1": end["t"] if end else None, "end": end})
    return out


def runner_flow(bag: Path) -> dict:
    """실행기 status 첫 메시지의 flow → {노드 id: lines} (정책 노드 lines[0] = '<역할> · <정책 id>', lines[1] = '<물체> → …')."""
    con = sqlite3.connect(f"file:{db_of(bag, 'arm')}?mode=ro", uri=True)
    names = [n for (n,) in con.execute("select name from topics where name like '/policy_control/status/episode_runner%'")]
    con.close()
    for name in names:
        rows = read(db_of(bag, "arm"), [name])[name][:1]
        if rows:
            d = json.loads(rows[0][1].data)
            return {n["id"]: n.get("lines", []) for n in (d.get("flow") or {}).get("nodes", [])}
    return {}


def pose7(msg) -> tuple[float, np.ndarray, np.ndarray]:
    """PoseStamped → (header 시각 s, 위치, 쿼터니언 wxyz)."""
    p, o, st = msg.pose.position, msg.pose.orientation, msg.header.stamp
    return st.sec + st.nanosec * 1e-9, np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z])


class PoseTrack:
    """FP++ PoseStamped 열 — at(t) = t 까지 받은 마지막 값(노드가 그 순간 본 것)과 그 나이(t − header 시각)."""

    def __init__(self, rows: list) -> None:
        self.t = np.array([t for t, _ in rows])
        self.v = [pose7(m) for _, m in rows]

    def at(self, t: float):
        k = int(np.searchsorted(self.t, t, side="right")) - 1
        if k < 0:
            return None
        stamp, p, q = self.v[k]
        return p, q, float(t - stamp)


def arm_series(bag: Path, t0: float, t1: float, rate_hz: float = 60.0) -> tuple[np.ndarray, dict]:
    """/joint_states → (시각, {"right": (N,7), "left": (N,7)}) — canonical r_aj · l_aj 순서, rate_hz 로 솎는다."""
    from policy_control.episode_ros import ARM_JOINT_NAME
    rows = read(db_of(bag, "arm"), ["/joint_states"], t0, t1)["/joint_states"]
    ts, R, L, last = [], [], [], -1.0
    for t, m in rows:
        if t - last < 1.0 / rate_hz:
            continue
        q = {}
        for n, v in zip(m.name, m.position):
            g = ARM_JOINT_NAME.fullmatch(n)
            if g:
                q[(g.group(1), int(g.group(2)))] = float(v)
        if len(q) < 14:
            continue
        last = t
        ts.append(t)
        R.append([q[("right", i)] for i in range(1, 8)])
        L.append([q[("left", i)] for i in range(1, 8)])
    return np.array(ts), {"right": np.array(R), "left": np.array(L)}


def out_path(kind: str, ep_dir: Path) -> Path:
    d = SIM2REAL / "logs" / "analysis" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ep_dir.parent.parent.name}_{ep_dir.name}.json"


def md_table(cols: list[str], rows: list[tuple[str, list]]) -> str:
    """지표 행 × 노드 열 markdown 표."""
    fmt = lambda v: "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| 지표 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| " + name + " | " + " | ".join(fmt(v) for v in vals) + " |" for name, vals in rows]
    return "\n".join(lines)
