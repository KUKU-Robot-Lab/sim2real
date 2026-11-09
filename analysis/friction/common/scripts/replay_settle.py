"""Replay recorded hold data through the current StationaryRuntime settle logic.

Feeds measured q/qd and the recorded user target of every diagnostics row into a
fresh runtime with the run's compensation config (forced to the given fs/ki), so we
see where the new standstill test would arm fs / update the integral. Open-loop:
the recorded motion is what the old (inactive) compensator produced.
usage: replay_settle.py RUN_DIR [RUN_DIR ...]
"""
import collections, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "rl_ws/sim2real/deploy/policy_control"))
from policy_control.pd_law import PdCommand  # noqa: E402
from policy_control.pd_stationary import StationaryRuntime  # noqa: E402
from policy_control import stationary_tracking as S  # noqa: E402

TOPIC = "/policy_control/pd_right/diagnostics"


def rows(run):
    side = json.load(open(run / "manifest.json"))["metadata"]["side"]
    topic = TOPIC.replace("right", side)
    for d in S.load_run_events(run, topics=(topic,)):   # bag(11.10~) 또는 events.jsonl
        m = d["message"]
        yield json.loads(m["data"]) if isinstance(m.get("data"), str) else m


def replay(run):
    data = list(rows(run))
    cfg = dict(data[0]["compensation_config"])
    cfg.update(fs=[0.3] * 4 + [0.0] * 3, integral_enabled=True, ki=[14.0] * 4 + [0.0] * 3)
    g = data[0]["gains"]
    rt = StationaryRuntime(cfg, kp=g["real_kp"], kd=g["real_kd"],
                           torque_limit=np.ones(7) * 7.0, blend_sec=1.0)
    blocks, settled, armed, upd, first = collections.Counter(), 0, 0, 0, None
    prev = None
    for i, m in enumerate(data):
        t = m.get("t_mono", i / 120.0)
        dt = 1.0 / 120.0 if prev is None else min(max(t - prev, 1e-3), 0.1)
        prev = t
        q = np.asarray(m["motor_position"])
        cmd = PdCommand(q, np.asarray(m["motor_velocity"]), np.zeros(7),
                        tuple(m["limited"]), False)
        _, d = rt.apply(cmd, q_meas=np.asarray(m["measured"]), qd_meas=np.asarray(m["velocity"]),
                        target=np.asarray(m["user_target"]), target_velocity=np.zeros(7),
                        phase=m["phase"], contact=m["contact"], stale=m["stale"],
                        episode=m["episode"], dt=dt, gravity=np.zeros(7), request=np.zeros(7))
        settled += d["settled"]; armed += d["fs_armed"]; upd += any(d["integral_updating"])
        if d["settled"] and first is None:
            first = i
        for b in d["settle_block"]:
            blocks[b] += 1
    n = len(data)
    return n, settled / n, armed / n, upd, first, dict(blocks)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        n, s, a_, u, f, b = replay(Path(a))
        print(f"{Path(a).name:32s} n={n} settled={s:.2f} armed={a_:.2f} upd={u} first={f} {b}")
