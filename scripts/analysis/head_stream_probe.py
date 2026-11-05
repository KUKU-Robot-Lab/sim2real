#!/usr/bin/env python3
"""목 연속 모드(/head/target_stream) 실측 — 계단 · 정지 · 사인 · 삼각파 목표를 20 Hz 로 보내고 응답을 기록 · 분석한다.

포트는 열지 않는다. head_joint_publisher 가 이미 떠 있어야 한다(목표 → /head/target_stream, 응답 ← /head/joint_states ·
/head/stream_log · /head/move_status). 각도는 모두 인코더 deg(tick·360/4095 − 180). tilt + = 숙임(틱 증가 = 숙임).

  python3 scripts/analysis/head_stream_probe.py --dry-run            # 계획 표만
  python3 scripts/analysis/head_stream_probe.py                      # 실측(camset 자세에서 2° 안일 때만 시작)
  python3 scripts/analysis/head_stream_probe.py --analyze logs/analysis/head_stream_probe_<MMDD_HHMMSS>

시각: 보낸 시각은 이 프로세스의 time.time(), joint_states stamp 는 드라이버의 벽시계(읽기 가운데)다 — 같은 호스트에서 돌린다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import head_motion as hm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CAMSET = REPO / "config" / "head_camset_rh56f1.yaml"
RATE_HZ = 20.0
JOINT_NAMES = ("head_j_pan", "head_j_tilt")  # head_joint_publisher.JOINT_NAMES
START_TOL_DEG = 2.0
MOVE_DEG = 0.2  # 데드타임: 목표 송신 뒤 각도가 이만큼 처음 넘은 stamp(정밀 지연은 fit_fopdt)
SIM = {"delay_ms": (0.0, 67.0), "tau_s": (0.03, 0.15), "vel_dps": (60.0, 180.0)}  # sim 가정
TRI_AMP, TRI_VEL = 15.0, 34.0
SINE_AMP, SINE_HZ = 5.0, 0.5

#: (구간, 길이 s, 종류, 축, 크기 deg). 크기는 --amp 배율.
SEGMENTS = (
    ("hold0", 3.0, "hold", None, 0.0),
    ("pan+5", 2.0, "hold", "pan", 5.0), ("back1", 2.0, "hold", None, 0.0),
    ("pan-5", 2.0, "hold", "pan", -5.0), ("back2", 2.0, "hold", None, 0.0),
    ("tilt+5", 2.0, "hold", "tilt", 5.0), ("back3", 2.0, "hold", None, 0.0),  # tilt + = 숙임
    ("tilt-5", 2.0, "hold", "tilt", -5.0), ("back4", 2.0, "hold", None, 0.0),
    ("pan_sine", 6.0, "sine", "pan", SINE_AMP),
    ("pan_tri", 6.0, "tri", "pan", TRI_AMP),
    ("hold_end", 3.0, "hold", None, 0.0),
)


# ---------------------------------------------------------------- 계획
def build_plan(pan0: float, tilt0: float, amp: float = 1.0) -> list[tuple[float, str, float, float]]:
    """[(보낼 시각 s, 구간, pan, tilt)] 20 Hz."""
    plan, t0 = [], 0.0
    tri_period = 4 * TRI_AMP / TRI_VEL  # 진폭 15° 를 34 °/s 로 오르내린다(배율은 속도에 걸린다)
    for seg, dur, kind, axis, size in SEGMENTS:
        for k in range(round(dur * RATE_HZ)):
            s = k / RATE_HZ
            if kind == "sine":
                off = size * amp * math.sin(2 * math.pi * SINE_HZ * s)
            elif kind == "tri":
                off = size * amp * (2 / math.pi) * math.asin(math.sin(2 * math.pi * s / tri_period))
            else:
                off = size * amp
            plan.append((round(t0 + s, 6), seg, pan0 + (off if axis == "pan" else 0.0),
                         tilt0 + (off if axis == "tilt" else 0.0)))
        t0 += dur
    return plan


def plan_problem(plan) -> str:
    """연속 모드 범위 밖 목표가 있으면 사유."""
    for axis, i, (lo, hi) in (("pan", 2, hm.STREAM_PAN_RANGE_DEG), ("tilt", 3, hm.STREAM_TILT_RANGE_DEG)):
        vals = [p[i] for p in plan]
        if min(vals) < lo or max(vals) > hi:
            return f"{axis} 목표 [{min(vals):+.2f}, {max(vals):+.2f}]° 가 연속 모드 범위 [{lo:g}, {hi:g}]° 밖이다"
    return ""


def start_problem(pan: float, tilt: float, pan0: float, tilt0: float, tol: float = START_TOL_DEG) -> str:
    err = max(abs(pan - pan0), abs(tilt - tilt0))
    if err > tol:
        return f"현재 자세 pan {pan:+.2f} tilt {tilt:+.2f} 가 camset {pan0:+.2f}/{tilt0:+.2f} 에서 {err:.2f}° (> {tol}°) 떨어져 있다"
    return ""


def plan_table(plan) -> list[str]:
    lines = [f"{'구간':<9}{'시작 s':>7}{'길이 s':>7}{'pan 범위':>20}{'tilt 범위':>20}{'최대 °/s':>9}"]
    for seg in dict.fromkeys(p[1] for p in plan):
        rows = [p for p in plan if p[1] == seg]
        pan, tilt = [r[2] for r in rows], [r[3] for r in rows]
        vel = max([abs(b - a) * RATE_HZ for a, b in zip(pan, pan[1:])] + [abs(b - a) * RATE_HZ for a, b in zip(tilt, tilt[1:])] + [0])
        lines.append(f"{seg:<9}{rows[0][0]:>7.2f}{len(rows) / RATE_HZ:>7.2f}"
                     f"{min(pan):>+10.2f}{max(pan):>+10.2f}{min(tilt):>+10.2f}{max(tilt):>+10.2f}{vel:>9.1f}")
    return lines


# ---------------------------------------------------------------- 분석(순수)
def crossing(t, f, level: float, interp: bool = True) -> float | None:
    """f 가 level 에 처음 닿는 시각(앞 표본과 선형 보간, interp=False 면 그 표본 stamp)."""
    hit = np.nonzero(f >= level)[0]
    if not len(hit):
        return None
    i = hit[0]
    if i == 0 or not interp:
        return float(t[i])
    return float(t[i - 1] + (level - f[i - 1]) / (f[i] - f[i - 1]) * (t[i] - t[i - 1]))


def fit_fopdt(t_u, u, t_y, y, dt: float = 0.001) -> dict:
    """y = 1차 지연(τ) ∘ 순수 지연(d) ∘ u(ZOH). 격자 탐색(5 ms → 1 ms). {delay_ms, tau_s, rmse_deg}."""
    t_u, u, t_y, y = map(np.asarray, (t_u, u, t_y, y))
    keep = t_y >= t_u[0]
    t_y, y = t_y[keep], y[keep]
    grid = np.arange(t_u[0], t_y[-1] + dt, dt)
    meas = np.full(len(grid), np.nan)
    meas[np.clip(np.round((t_y - grid[0]) / dt).astype(int), 0, len(grid) - 1)] = y
    hit = ~np.isnan(meas)

    def sse(ds, taus):
        a = (1.0 - np.exp(-dt / taus))[None, :]
        k = np.searchsorted(t_u, grid[None, :] - ds[:, None], side="right") - 1
        U = u[np.clip(k, 0, None)]
        state = np.full((len(ds), len(taus)), y[0])
        err = np.zeros_like(state)
        for i in range(len(grid)):
            if hit[i]:
                err += (state - meas[i]) ** 2
            state += a * (U[:, i:i + 1] - state)
        return err

    best = None
    ds, taus = np.arange(0.0, 0.2501, 0.005), np.arange(0.005, 0.4001, 0.005)
    for _ in range(2):
        err = sse(ds, taus)
        i, j = np.unravel_index(np.argmin(err), err.shape)
        best = (ds[i], taus[j], err[i, j])
        ds = np.clip(ds[i] + np.arange(-6, 7) * 0.001, 0.0, None)
        taus = np.clip(taus[j] + np.arange(-6, 7) * 0.001, 0.001, None)
    d, tau, e = best
    return {"delay_ms": round(float(d) * 1e3, 1), "tau_s": round(float(tau), 4),
            "rmse_deg": round(math.sqrt(float(e) / hit.sum()), 3),
            "in_sim_range": bool(SIM["delay_ms"][0] <= d * 1e3 <= SIM["delay_ms"][1] and SIM["tau_s"][0] <= tau <= SIM["tau_s"][1])}


def segments(tc, seg, pan, tilt) -> list[dict]:
    out = []
    names = list(dict.fromkeys(seg))
    for n, name in enumerate(names):
        m = seg == name
        t = tc[m]
        end = tc[seg == names[n + 1]][0] if n + 1 < len(names) else t[-1] + 1 / RATE_HZ
        out.append({"seg": name, "t0": float(t[0]), "t1": float(end), "pan": pan[m], "tilt": tilt[m], "tc": t,
                    "const": bool(np.ptp(pan[m]) < 1e-9 and np.ptp(tilt[m]) < 1e-9)})
    return out


def step_metrics(t, y, t_step: float, t_end: float, y1: float) -> dict:
    before = y[(t >= t_step - 0.3) & (t < t_step)]
    y0 = float(before.mean()) if len(before) else float(y[t < t_step][-1])
    w = (t >= t_step) & (t <= t_end)
    tw, sgn = t[w], math.copysign(1.0, y1 - y0)
    f, size = (y[w] - y0) * sgn, abs(y1 - y0)

    def rel(level, interp=True):
        c = crossing(tw, f, level, interp)
        return None if c is None else c - t_step

    t10, t90, dead = rel(0.1 * size), rel(0.9 * size), rel(MOVE_DEG, interp=False)  # 첫 stamp — 표본 간격(50 ms)만큼 늦게 잡힐 수 있다
    last = y[(t > t_end - 0.5) & (t <= t_end)]
    return {"from": round(y0, 3), "to": round(y1, 3),
            "deadtime_ms": None if dead is None else round(dead * 1e3, 1),
            "rise_ms": None if t10 is None or t90 is None else round((t90 - t10) * 1e3, 1),
            "overshoot_deg": round(max(float(f.max()) - size, 0.0), 3) if len(f) else None,
            "settle_err_deg": round(float(last.mean()) - y1, 3) if len(last) else None}


def track_metrics(tc, c, t, y, w0: float, w1: float) -> dict:
    """교차상관 지연(목표 ZOH vs 실측, 0~500 ms) · 진폭비(표준편차 비) · 실측 속도 p95."""
    g = np.arange(w0, w1, 0.001)
    cmd = c[np.clip(np.searchsorted(tc, g, side="right") - 1, 0, None)]
    cmd = cmd - cmd.mean()
    lags = np.arange(0, 501)
    scores = []
    for lag in lags:
        m = np.interp(g + lag * 1e-3, t, y)
        m = m - m.mean()
        scores.append(float(cmd @ m) / (np.linalg.norm(cmd) * np.linalg.norm(m) + 1e-12))
    lag = int(lags[int(np.argmax(scores))])
    meas = np.interp(g, t, y)
    sel = (t >= w0) & (t <= w1)
    vel = np.abs(np.diff(y[sel]) / np.diff(t[sel])) if sel.sum() > 2 else np.array([0.0])
    return {"lag_ms": lag, "corr": round(max(scores), 4), "amp_ratio": round(float(meas.std() / (cmd.std() + 1e-12)), 3),
            "vel_p95_dps": round(float(np.percentile(vel, 95)), 1)}


def stream_log_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    lat = np.array([r["write_t"] - r["recv_t"] for r in rows]) * 1e3
    tick = np.diff([r["write_t"] for r in rows]) * 1e3
    clamps = [r.get("clamp") or {} for r in rows]
    pct = lambda a, q: round(float(np.percentile(a, q)), 2) if len(a) else None  # noqa: E731
    return {"n": len(rows), "recv_to_write_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95)},
            "tick_ms": {"p50": pct(tick, 50), "p95": pct(tick, 95), "max": round(float(tick.max()), 2) if len(tick) else None},
            "clamp_any_frac": round(sum(any(c.values()) for c in clamps) / len(rows), 4),
            "clamp_range_frac": round(sum("range" in c.values() for c in clamps) / len(rows), 4)}


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([r[k] for r in rows], dtype=object if k == "seg" else float) for k in (rows[0] if rows else {})}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def analyze(folder: Path) -> dict:
    folder = Path(folder)
    tg, js = _read_csv(folder / "targets.csv"), _read_csv(folder / "joint_states.csv")
    if not tg or not js:
        raise SystemExit(f"{folder}: targets.csv · joint_states.csv 가 비었다")
    log = _jsonl(folder / "stream_log.jsonl")
    tc, seg = tg["t_send"], tg["seg"].astype(str)
    t, meas = js["stamp"], {"pan": js["pan_deg"], "tilt": js["tilt_deg"]}
    segs = segments(tc, seg, tg["pan_deg"], tg["tilt_deg"])
    summary = {"folder": str(folder), "steps": [], "still": [], "tracking": [], "sim_assumption": SIM}
    for prev, s in zip([None] + segs, segs):
        if s["const"]:
            half = (t >= (s["t0"] + s["t1"]) / 2) & (t <= s["t1"])
            if half.sum() >= 3:
                summary["still"].append({"seg": s["seg"], "n": int(half.sum()),
                                         **{f"{a}_std_deg": round(float(meas[a][half].std()), 4) for a in meas}})
            if prev is not None:
                jump = {a: abs(s[a][0] - prev[a][-1]) for a in meas}
                axis = max(jump, key=jump.get)
                if jump[axis] > 0.5:
                    summary["steps"].append({"seg": s["seg"], "axis": axis,
                                             **step_metrics(t, meas[axis], s["t0"], s["t1"], float(s[axis][0]))})
        else:
            axis = max(meas, key=lambda a: np.ptp(s[a]))
            summary["tracking"].append({"seg": s["seg"], "axis": axis,
                                        **track_metrics(s["tc"], s[axis], t, meas[axis], s["t0"] + 1.0, s["t1"])})
    summary["stream_log"] = stream_log_stats(log)
    summary["fopdt"] = {}
    for a in meas:
        if np.ptp(tg[f"{a}_deg"]) < 1.0:  # 이 축 목표가 안 움직였으면 τ 를 못 잰다(맞춤값이 거짓 범위 밖으로 나온다)
            continue
        fits = {"target": fit_fopdt(tc, tg[f"{a}_deg"], t, meas[a])}
        if log:
            fits["goal"] = fit_fopdt([r["write_t"] for r in log], [r[f"goal_{a}_deg"] for r in log], t, meas[a])
        summary["fopdt"][a] = fits
    summary["move_status"] = _jsonl(folder / "move_status.jsonl")
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    _plot(folder, tc, tg, t, meas, log)
    return summary


def _plot(folder, tc, tg, t, meas, log) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, a in zip(axes, ("pan", "tilt")):
        ax.step(tc - tc[0], tg[f"{a}_deg"], where="post", label="target(sent)", lw=1)
        if log:
            ax.step(np.array([r["write_t"] for r in log]) - tc[0], [r[f"goal_{a}_deg"] for r in log], where="post", label="goal(written)", lw=1)
        ax.plot(t - tc[0], meas[a], ".-", ms=3, lw=0.8, label="joint_states")
        ax.set_ylabel(f"{a} [enc deg]")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t [s] (first target = 0)")
    fig.tight_layout()
    fig.savefig(folder / "probe.png", dpi=110)
    plt.close(fig)


def print_summary(s: dict) -> None:
    print("계단: 구간 축 데드타임ms 상승ms 오버슈트° 정착오차°")
    for r in s["steps"]:
        print(f"  {r['seg']:<9}{r['axis']:<5}{r['deadtime_ms']!s:>8}{r['rise_ms']!s:>8}{r['overshoot_deg']!s:>9}{r['settle_err_deg']!s:>9}")
    print("정지 떨림(뒤 절반 std °): " + ", ".join(f"{r['seg']} {r['pan_std_deg']}/{r['tilt_std_deg']}" for r in s["still"]))
    for r in s["tracking"]:
        print(f"추종 {r['seg']}: 지연 {r['lag_ms']} ms · 진폭비 {r['amp_ratio']} · 속도 p95 {r['vel_p95_dps']} °/s · corr {r['corr']}")
    print(f"stream_log: {json.dumps(s['stream_log'], ensure_ascii=False)}")
    for a, fits in s["fopdt"].items():
        for k, f in fits.items():
            print(f"1차 지연 {a} {k}→각도: 지연 {f['delay_ms']} ms · τ {f['tau_s']} s · rmse {f['rmse_deg']}° · sim 범위 안 {f['in_sim_range']}")
    print(f"sim 가정: {SIM}")


# ---------------------------------------------------------------- 실측(rclpy)
FINISH_MAX_S = 3.0  # 60 °/s 로 15° 복귀 + 여유


def _finish(send, js, pan0: float, tilt0: float) -> None:
    """기준 자세 목표를 도달할 때까지(최대 FINISH_MAX_S) 보낸다. 0.5 s 끊기면 드라이버는 마지막 goal 을 두고 이탈한다."""
    t0 = time.time()
    while time.time() - t0 < FINISH_MAX_S:
        send("final", pan0, tilt0)
        time.sleep(1 / RATE_HZ)
        reached = js and not start_problem(js[-1][2], js[-1][3], pan0, tilt0)
        if reached and time.time() - t0 > 0.5:  # 도달해도 0.5 s 는 보낸다
            break
    time.sleep(1.0)  # 이탈 상태 · 마지막 각도 받기
    if not js:
        print("경고: 마무리 뒤 각도를 못 받았다", file=sys.stderr)
        return
    pan, tilt = js[-1][2], js[-1][3]
    problem = start_problem(pan, tilt, pan0, tilt0)
    print(f"마무리 각도 pan {pan:.2f} tilt {tilt:.2f}" + (f" — 기준 자세 아님: {problem}" if problem else ""),
          file=sys.stderr if problem else sys.stdout)


def run(args, pan0: float, tilt0: float, plan) -> int:
    import threading

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    from head_fk_chain import encoder_from_urdf

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl-C 뒤에도 컨텍스트가 살아 있어야 기준 자세를 보낸다

    def _interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt)  # kill · ssh 끊김에도 기준 자세로 마무리
    signal.signal(signal.SIGHUP, _interrupt)
    node = Node("head_stream_probe")
    js, log, status, sent = [], [], [], []

    def on_js(msg):
        try:
            i, j = msg.name.index(JOINT_NAMES[0]), msg.name.index(JOINT_NAMES[1])
        except ValueError:
            return
        pan, tilt = encoder_from_urdf(math.degrees(msg.position[i]), math.degrees(msg.position[j]))
        js.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.time(), pan, tilt))

    node.create_subscription(JointState, "/head/joint_states", on_js, 10)
    node.create_subscription(String, hm.STREAM_LOG_TOPIC, lambda m: log.append(m.data), 50)
    node.create_subscription(String, hm.STATUS_TOPIC, lambda m: status.append({"recv_t": time.time(), "data": m.data}), 10)
    pub = node.create_publisher(String, hm.STREAM_TOPIC, 10)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    def send(seg, pan, tilt):
        seq = len(sent)
        pub.publish(String(data=json.dumps({"seq": seq, "pan_deg": pan, "tilt_deg": tilt})))
        sent.append((time.time(), seq, seg, pan, tilt))

    out = REPO / "logs" / "analysis" / f"head_stream_probe_{time.strftime('%m%d_%H%M%S')}"
    started = False
    try:
        for _ in range(30):  # DDS 발견 대기 최대 3 s, 첫 메시지부터 1 s 모은다
            if js:
                break
            time.sleep(0.1)
        time.sleep(1.0)
        now = [r for r in js if r[1] >= time.time() - 1.0]
        if not now:
            print("/head/joint_states 를 1 s 동안 못 받았다 — head_joint_publisher · ROS_DOMAIN_ID 확인", file=sys.stderr)
            return 2
        problem = start_problem(float(np.mean([r[2] for r in now])), float(np.mean([r[3] for r in now])), pan0, tilt0)
        if problem:
            print(f"시작 안 함: {problem}", file=sys.stderr)
            return 2
        started = True
        t0 = time.time()
        for t_rel, seg, pan, tilt in plan:
            time.sleep(max(0.0, t0 + t_rel - time.time()))
            send(seg, pan, tilt)
    except KeyboardInterrupt:
        print("중단 — 기준 자세로 마무리한다", file=sys.stderr)
    finally:
        old = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}  # 마무리를 끊지 않게
        if started:
            _finish(send, js, pan0, tilt0)
        try:
            if started:
                out.mkdir(parents=True, exist_ok=True)
                _write(out, sent, js, log, status)  # 종료 처리보다 먼저: 실기 실측을 잃지 않게
                print(f"기록: {out}")
                if not log:
                    print(f"경고: {hm.STREAM_LOG_TOPIC} 를 못 받았다 — goal 맞춤 생략", file=sys.stderr)
                if status:
                    print(f"마지막 move_status: {status[-1]['data']}")
        finally:
            ex.shutdown()
            spin.join(timeout=2.0)  # spin 스레드가 살아 있는 채로 끝나면 terminate(core dump)
            node.destroy_node()
            rclpy.try_shutdown()
            for s, h in old.items():
                signal.signal(s, h)
    if not started:
        return 2
    print_summary(analyze(out))
    return 0


def _write(out: Path, sent, js, log, status) -> None:
    with open(out / "targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_send", "seq", "seg", "pan_deg", "tilt_deg"])
        w.writerows(sent)
    with open(out / "joint_states.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp", "recv_t", "pan_deg", "tilt_deg"])
        w.writerows(js)
    (out / "stream_log.jsonl").write_text("".join(line + "\n" for line in log))
    (out / "move_status.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in status))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=["steps"], default="steps")
    ap.add_argument("--amp", type=float, default=1.0, help="진폭 배율")
    ap.add_argument("--config", type=Path, default=DEFAULT_CAMSET, help="기준(camset) 자세 yaml")
    ap.add_argument("--dry-run", action="store_true", help="계획 표만 출력")
    ap.add_argument("--analyze", type=Path, help="기록 폴더를 다시 분석")
    args = ap.parse_args()
    if args.analyze:
        print_summary(analyze(args.analyze))
        return 0

    from head_home import load_head_home

    cfg = load_head_home(args.config)
    by_name = {name: cfg.targets_deg[i] for i, name in cfg.names.items()}
    pan0, tilt0 = by_name["pan"], by_name["tilt"]
    plan = build_plan(pan0, tilt0, args.amp)
    print(f"기준 {args.config.name}: pan {pan0:+.3f} tilt {tilt0:+.3f} (인코더 deg) · {len(plan)} 목표 · {len(plan) / RATE_HZ:.1f} s · amp ×{args.amp:g}")
    print("\n".join(plan_table(plan)))
    problem = plan_problem(plan)
    if problem:
        print(f"거부: {problem}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    return run(args, pan0, tilt0, plan)


if __name__ == "__main__":
    raise SystemExit(main())
