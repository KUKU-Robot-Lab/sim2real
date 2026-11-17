"""실기 bag 재생 — 러너 이벤트(episode_*.jsonl)의 단계마다 signals.SensorHub 판정을 다시 내고 정답 라벨과 맞춘다.

    python3 -m vlm_supervisor.bench.replay_bags --runs ~/rl_ws/real_bags_vlmsup \
        --labels vlm_supervisor/bench/labels_real_1117.csv [--fk] [--hand-only] [--out replay.jsonl]

runs 아래 <YYMMDD>/episode/<HHMMSS>_<이름>/episode_*.jsonl 과 <YYMMDD>/bags/<HHMMSS>_episode_<이름>/{hand,arm} 를 짝짓는다
(같은 날 · 같은 이름 · 시작 시각 15 s 안). 에피소드 설정은 config/episodes/<이름>.yaml(지금 판) — 그때 판과 노드가 다르면 건너뛴다.
--fk: 손바닥 위치를 자산 URDF FK 로 풀어 물체 손 거리를 쓴다(policy_control 필요). 없으면 물체 높이만 본다.
읽기만 한다(bag sqlite 는 읽기 전용 연결).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

from policy_control import episode_spec as S

from .. import signals as G
from . import bagread

_REPO = Path(__file__).resolve().parents[2]
EPISODES = _REPO / "config" / "episodes"
KST = dt.timezone(dt.timedelta(hours=9))
RUN_EXEC = ("policy", "trajectory", "hand")


def epoch(text: str) -> float:
    """러너 jsonl 시각(KST, 시간대 표기 없음) → epoch s."""
    t = dt.datetime.fromisoformat(text)
    return (t if t.tzinfo else t.replace(tzinfo=KST)).timestamp()


def find_bag(runs: Path, day: str, ep_dir: str, name: str) -> Path | None:
    hh = int(ep_dir[:6])
    for b in sorted((runs / day / "bags").glob(f"*_episode_{name}")):
        if abs(int(b.name[:6]) - hh) <= 15 and (b / "hand").is_dir():
            return b
    return None


def stages(events: list[dict]):
    """jsonl → (state_id, t0, t1, 이벤트) — 실행 노드(정책 · 궤적 · 손)의 exit · failure · stopped 마다."""
    enters = {}
    for e in events:
        if e.get("event") == "enter":
            enters[e["state_id"]] = epoch(e["timestamp"])
            continue
        if e.get("event") not in ("exit", "failure", "stopped") or e.get("executor_type") not in RUN_EXEC:
            continue
        t1 = epoch(e.get("end_time") or e["timestamp"])
        yield e["state_id"], enters.get(e["state_id"], t1 - float(e.get("duration") or 0.0)), t1, e


def feed(hub: G.SensorHub, bag: Path) -> dict:
    """bag 손 · 팔 토픽을 hub 에 넣는다. 팔 · 손 관절(FK 용)은 돌려준다."""
    joints: dict = {}
    want = lambda n: (n.startswith("/hand_") and n.split("/")[2] in (  # noqa: E731
        "angle_actual", "angle_set", "force_actual", "tip_forces_zeroed", "joint_states")) \
        or (n.startswith("/objects/") and n.endswith("/live_pose")) or n == "/joint_states"
    for sub in ("hand", "arm"):
        if not (bag / sub).is_dir():
            continue
        for topic, t, v in bagread.read(bag / sub, want):
            if topic.startswith("/objects/"):
                hub.objs.feed(topic.split("/")[2], t, v[:3])
                continue
            if topic == "/joint_states" or topic.endswith("/joint_states"):
                joints.setdefault(topic, []).append((t, v[0], v[1]))
                continue
            side, field = topic.split("/")[1][5:], topic.split("/")[2]
            h = hub.hand.get(side)
            if h is None:
                continue
            {"angle_actual": h.feed_angle, "angle_set": h.feed_target, "force_actual": h.feed_force,
             "tip_forces_zeroed": h.feed_tip}[field](t, v)
    return joints


def palm_fn(joints: dict):
    """관절 기록 → palm(side, t). policy_control FK(계약 rh56f1) 를 쓴다."""
    import bisect

    import numpy as np
    from policy_control.contract_assets import ASSETS
    from policy_control.fk_numpy import UrdfChainFK
    from policy_control.sources import load_profile, load_robot_cfg
    c = json.loads((_REPO / "logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json").read_text())
    prof = load_profile(load_robot_cfg(_REPO / "deploy/policy_control/config/robots/rh56f1_bi_real.yaml").joint_profiles)
    fks = {}
    for s in c["sides"].values():
        fk = UrdfChainFK(ASSETS[c["asset"]["name"]].urdf, s["arm_joints"], s["hand_joints"], s["palm_body"], s["tip_bodies"])
        fks[s["side"]] = (fk, [(prof[j]["source"], prof[j]["sign"]) for j in s["arm_joints"]],
                          [(prof[j]["source"], prof[j]["sign"]) for j in s["hand_joints"]])

    def pick(topic, srcs, t):
        rows = joints.get(topic) or []
        i = bisect.bisect_right([r[0] for r in rows], t) - 1
        if i < 0:
            return None
        _, names, vals = rows[i]
        try:
            return np.array([vals[names.index(n)] * g for n, g in srcs])
        except ValueError:
            return None

    def palm(side: str, t: float):
        fk, a, h = fks[side]
        qa, qh = pick("/joint_states", a, t), pick(f"/hand_{side}/joint_states", h, t)
        if qa is None or qh is None:
            return None
        return tuple(float(x) for x in fk.palm_pose(qa, qh).palm_pos)
    return palm


def load_labels(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(r["day"], r["ep"], r["state"], r["side"]): r for r in csv.DictReader(f)}


def score(verdict: str, label: str) -> str:
    if label not in ("ok", "fail"):
        return "라벨없음"
    if verdict == G.UNSURE:
        return "애매"
    return "맞음" if verdict == label else "틀림"


def replay(runs: Path, labels: dict, use_fk: bool, out=None, hand_only: bool = False) -> Counter:
    tally: Counter = Counter()
    cfg = G.load_cfg()
    for jf in sorted(runs.glob("2611*/episode/*/episode_*.jsonl")):
        day, ep_dir = jf.parts[-4], jf.parent.name
        events = [json.loads(x) for x in jf.read_text().splitlines() if x.strip()]
        if not events:
            continue
        name = events[0].get("episode", "")
        bag = find_bag(runs, day, ep_dir, name)
        spec_path = EPISODES / f"{name}.yaml"
        if bag is None or not spec_path.exists():
            continue
        ep = S.load(spec_path)
        nodes = {n.id: n for n in ep.nodes}
        hub = G.SensorHub(ep, cfg, maxlen=10 ** 7)
        joints = feed(hub, bag)
        if hand_only:                                # 손 센서만의 성능 — 라벨이 FP++ 물체 자리를 근거로 쓰므로 순환을 뺀다
            hub.objs = G.ObjectBuffer()
        hub.palm = palm_fn(joints) if use_fk else None
        for sid, t0, t1, e in stages(events):
            node = nodes.get(sid)
            if node is None:
                continue
            for v in hub(node, t0, t1):
                side = v.feats["side"]
                lab = labels.get((day, ep_dir, sid, side), {})
                sc = score(v.label, lab.get("label", ""))
                tally[(v.feats["kind"], sc)] += 1
                row = {"day": day, "ep": ep_dir, "state": sid, "side": side, "kind": v.feats["kind"],
                       "runner": e.get("result"), "sensor": v.label, "label": lab.get("label"), "score": sc,
                       "reasons": list(v.reasons), "feats": {k: v.feats[k] for k in v.feats if k not in ("side", "kind", "node")}}
                if out is not None:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return tally


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path(__file__).with_name("labels_real_1117.csv"))
    ap.add_argument("--fk", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--hand-only", action="store_true", help="물체(FP++) 표를 빼고 손 센서만으로 판정")
    a = ap.parse_args(argv)
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    try:
        tally = replay(a.runs.expanduser(), load_labels(a.labels), a.fk, out, a.hand_only)
    finally:
        if out:
            out.close()
    kinds = sorted({k for k, _ in tally})
    cols = ("맞음", "틀림", "애매", "라벨없음")
    print("| 단계 | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for k in kinds:
        print(f"| {k} | " + " | ".join(str(tally[(k, c)]) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
