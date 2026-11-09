#!/usr/bin/env python3
"""잡은 뒤 FP++ 컵(live · snapshot)과 FK 컵(손바닥 FK ⊗ 넘김 묶음 rel)을 손별 · 단계별로 비교한다(11.04 사용자).

    python3 scripts/analysis/grasp_fk_vs_fpp.py --episode <logs/runs/<YYMMDD>/episode/<시각>_<이름> | latest> \
        [--analysis logs/analysis/grasp_fk_vs_fpp_<MMDD>_<HHMMSS>]   # bag 은 에피소드 시각으로 logs/runs/*/bags 에서 찾는다
출력(항상): <에피소드>/grasp_record/<성공한 grasp_* 단계>/record.json · <시점>.jpg(머리 카메라 원본) · <시점>_overlay.png
      — 시점 = 잡기 끝 · 끝 + 1 s · 다음 단계 0.2 s 전. 각 시점의 FP++ live · /pose, FK 컵, 차이(mm · deg), 손 상태, 묶음.
      다시 돌려도 같은 파일(시각 · 값만 쓴다). 미션 후처리: nice -n 10 로 돌려도 된다(rh56f1_interfaces 를 source 해야 손 각도 · 힘).
출력(--analysis): report.md(손별 · 단계별 표, 놓친 시각, 잡은 직후 묶음 대조), <잡기 id>.png(단계 구간 표시),
      frames_<잡기 id>/ (머리 카메라에 live 컵 · FK 컵을 겹친 놓친 순간 앞뒤 프레임, camera bag 이 있을 때).

- 단계: 에피소드 폴더 *.jsonl 의 enter → exit/failure/safe_stop. 잡기 = 계획서(config/episodes/<이름>.yaml)의 kind aglt 정책 단계.
- FK 컵: place 노드와 같은 UrdfChainFK(계약 asset URDF · palm_body, pour_fj_node) 로 /joint_states(프로필 이름 · 부호) 를 풀고
  넘김 묶음(logs/runs/<YYMMDD>/inhand/*_rh_aglt_node_<팔>_*.npz) 의 rel_p · rel_q 를 붙인다(rh_place.cup_from_palm 과 같은 식).
- live 끊김: place 노드 CUP_STALE_S(0.5 s, 받은 시각) 를 넘게 새 live_pose 가 없던 30 Hz 칸의 비율.
- 거리: live 한 장마다 그 영상 stamp 의 FK 와 잰다(지연 빼고 기하만). 자세는 전체 회전과 컵 축(z) 기울기 차 둘 다 —
  원통 컵은 축 둘레 yaw 가 FP++ 에서 정해지지 않고 object_pose_node 가 twist 를 0 으로 바꿔 내므로(remove_twist) 축 차만 의미가 있다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / "scripts" / "analysis"), str(_ROOT / "scripts"), str(_ROOT / "deploy" / "policy_control")]

import episode_video_overlay as V  # noqa: E402
import run_paths  # noqa: E402

STALE_S = 0.5            # pour_fj_node.CUP_STALE_S
TOL_M = 0.05             # rh_place.LIVE_FK_TOL_M
GRID_HZ = 30.0
SUSTAIN_S = 1.0          # 이만큼 이어진 이탈을 '놓침'으로 본다(한두 장 튐과 구분)
JF_NAMES = ("thumb_bend", "thumb_rot", "index", "middle", "ring", "pinky")    # /hand_*/joint_forces 순서(robot yaml)
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
ROBOT_YAML = _ROOT / "deploy" / "policy_control" / "config" / "robots" / "rh56f1_bi_real.yaml"


# ============================================================================ 순수 로직
def parse_steps(lines: list[dict]) -> list[tuple[str, float, float, str]]:
    """에피소드 jsonl 줄들 → [(단계 id, 시작 s, 끝 s, 결과)]. 끝 = exit · failure · safe_stop 중 처음. 운영자 stop 줄은 뺀다."""
    out, open_ = [], {}
    for d in lines:
        ev, sid = d.get("event"), d.get("state_id")
        t = ts_epoch(d["timestamp"])
        if ev == "enter":
            open_[sid] = t
        elif ev in ("exit", "failure", "safe_stop") and sid in open_:
            out.append((sid, open_.pop(sid), t, str(d.get("result") or ev)))
    return sorted(out, key=lambda r: r[1])


def ts_epoch(text: str) -> float:
    # shortcut: 에피소드 jsonl 시각은 시간대 없는 로컬 시각 — 이 PC 시간대(KST)로 읽는다. 다른 시간대에서 돌리면 TZ=Asia/Seoul
    return dt.datetime.fromisoformat(text).timestamp()


def phase_at(steps, t: float) -> str:
    """t 의 단계 이름. 단계 사이 빈 시간은 'after_<앞 단계>'."""
    prev = "start"
    for sid, a, b, _ in steps:
        if a <= t <= b:
            return sid
        if b < t:
            prev = sid
    return f"after_{prev}"


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def axis_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(Ra[:, 2] @ Rb[:, 2]), -1.0, 1.0))))


def last_index(times: np.ndarray, t) -> np.ndarray:
    """정렬된 times 에서 t 이하 마지막 칸(없으면 -1). t 는 스칼라나 배열."""
    return np.searchsorted(times, t, side="right") - 1


def stale_mask(grid: np.ndarray, recv: np.ndarray, stale_s: float = STALE_S) -> np.ndarray:
    """30 Hz 칸마다 마지막 live 를 받은 지 stale_s 넘었으면 True(받은 적 없음도 True)."""
    i = last_index(recv, grid)
    age = np.where(i >= 0, grid - recv[np.clip(i, 0, None)], np.inf)
    return age > stale_s


def first_sustained(t: np.ndarray, bad: np.ndarray, hold_s: float = SUSTAIN_S) -> float | None:
    """bad 가 hold_s 이상 이어진 첫 구간의 시작 시각(없으면 None)."""
    start = None
    for ti, b in zip(t, bad):
        if b and start is None:
            start = ti
        elif not b:
            start = None
        if start is not None and ti - start >= hold_s:
            return float(start)
    return None


def stats(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    if v.size == 0:
        return {"med": np.nan, "p90": np.nan, "max": np.nan}
    return {"med": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())}


# ============================================================================ 입력
@dataclass
class Grasp:
    step: str
    side: str            # right | left
    obj_topic: str       # /objects/<물체>
    contract: Path       # 같은 팔 place 계약(FK 정의)
    bundle: Path


def plan_grasps(ep_dir: Path, steps) -> tuple[str, list[Grasp]]:
    lines = [json.loads(ln) for f in sorted(ep_dir.glob("*.jsonl")) for ln in f.read_text().splitlines() if ln.strip()]
    name = lines[0]["episode"]
    spec = yaml.safe_load((_ROOT / "config" / "episodes" / f"{name}.yaml").read_text(encoding="utf-8"))
    pol = spec["policies"]
    place = {p["side"]: p["policy"] for p in pol.values() if p.get("kind") == "place"}
    done = {s[0]: s for s in steps}
    out = []
    for it in spec["sequence"]:
        p = pol.get(it.get("name") or "")
        if it.get("type") != "policy" or not p or p.get("kind") != "aglt" or it["id"] not in done:
            continue
        if done[it["id"]][3] != "COMPLETED":              # 성공한 잡기만(묶음이 있는 것)
            continue
        side = p["side"]
        topic = spec["objects"][it["target_object"]]["topic"].rsplit("/", 1)[0]
        _, a, b, _ = done[it["id"]]
        out.append(Grasp(it["id"], side, topic, _ROOT / "deploy" / "policies" / place[side] / "rh_place_contract.json",
                         find_bundle(side, a, b)))
    return name, out


def find_bundle(side: str, t_enter: float, t_exit: float) -> Path:
    """그 잡기 단계 안(끝 + 10 s)에 저장된 넘김 묶음 — 파일 이름 HHMMSS 와 날짜 폴더로 찾는다."""
    day = dt.datetime.fromtimestamp(t_enter)
    hits = []
    for f in sorted(run_paths.day_dir("inhand", day).glob(f"*_rh_aglt_node_{side}_*.npz")):
        t = dt.datetime.combine(day.date(), dt.datetime.strptime(f.name[:6], "%H%M%S").time()).timestamp()
        if t_enter - 1 <= t <= t_exit + 10:
            hits.append(f)
    if len(hits) != 1:
        raise SystemExit(f"{side} 넘김 묶음이 {len(hits)}개 {[h.name for h in hits]} — 잡기 구간 {t_enter:.0f}~{t_exit:.0f}")
    return hits[0]


def joint_series(bag: V.Bag, topic: str, names: list[str], signs, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
    """JointState → (받은 시각 s (N,), 값 (N, len(names))). 이름이 빠진 메시지는 건너뛴다."""
    ts, vals = [], []
    for t, m in bag.read(topic, t0, t1):
        try:
            idx = [m.name.index(n) for n in names]
        except ValueError:
            continue
        ts.append(t * 1e-9)
        vals.append([m.position[i] for i in idx])
    return np.asarray(ts), np.asarray(vals, float) * np.asarray(signs, float)


def array_series(bag: V.Bag, topic: str, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [(t * 1e-9, list(m.data)) for t, m in bag.read(topic, t0, t1)]
    if not rows:
        return np.zeros(0), np.zeros((0, 0))
    return np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows], float)


def pose_series(bag: V.Bag, topic: str, t0: int, t1: int):
    """PoseStamped → (받은 s, stamp s, pos (N,3), R (N,3,3))."""
    rows = bag.read(topic, t0, t1)
    recv = np.asarray([t * 1e-9 for t, _ in rows])
    stamp = np.asarray([V.stamp_ns(m.header) * 1e-9 for _, m in rows])
    pos = np.asarray([[m.pose.position.x, m.pose.position.y, m.pose.position.z] for _, m in rows]).reshape(-1, 3)
    R = np.asarray([V.quat_wxyz_to_R([m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z])
                    for _, m in rows]).reshape(-1, 3, 3)
    return recv, stamp, pos, R


# ============================================================================ 한 잡기
@dataclass
class CupFK:
    """place 노드와 같은 FK 컵 = 손바닥 FK(UrdfChainFK, 계약 palm_body) ⊗ 넘김 묶음 rel(rh_place.cup_from_palm)."""
    fk: object
    arm_src: tuple       # (source 이름들, 부호들) — /joint_states
    hand_src: tuple      # /hand_<팔>/joint_states
    rel_p: np.ndarray
    rel_q: np.ndarray    # wxyz
    R_rel: np.ndarray

    @classmethod
    def of(cls, g: Grasp) -> "CupFK":
        from policy_control.contract_assets import ASSETS
        from policy_control.fk_numpy import UrdfChainFK
        from policy_control.place_handover import load as load_bundle
        from policy_control.sources import load_profile, load_robot_cfg
        c = json.loads(g.contract.read_text())
        s = next(v for v in c["sides"].values() if v["side"] == g.side)
        fk = UrdfChainFK(ASSETS[c["asset"]].urdf, s["arm_joints"], s["hand_joints"], s["palm_body"], s["tip_bodies"])
        prof = load_profile(load_robot_cfg(ROBOT_YAML).joint_profiles)
        src = lambda js: ([prof[j]["source"] for j in js], [prof[j]["sign"] for j in js])   # noqa: E731
        b = load_bundle(g.bundle)
        q = np.asarray(b.rel_q, float)
        return cls(fk, src(s["arm_joints"]), src(s["hand_joints"]), np.asarray(b.rel_p, float), q, V.quat_wxyz_to_R(q))

    def at(self, arm: V.Bag, hand: V.Bag, side: str, t0: int, t1: int):
        """[t0, t1] ns 의 관절을 읽어 → fk_cup(t 배열) 함수. 관절 기록이 없으면 SystemExit."""
        ta, qa = joint_series(arm, "/joint_states", *self.arm_src, t0, t1)
        th, qh = joint_series(hand, f"/hand_{side}/joint_states", *self.hand_src, t0, t1)
        if ta.size == 0 or th.size == 0:
            raise SystemExit(f"{side}: {t0 * 1e-9:.1f}~{t1 * 1e-9:.1f} 팔 또는 손 관절 기록이 없다")

        def fk_cup(t):
            """시각 배열 → (palm pos, palm R, FK 컵 pos, FK 컵 R). 관절은 그 시각 이전 마지막 표본."""
            ia, ih = np.clip(last_index(ta, t), 0, None), np.clip(last_index(th, t), 0, None)
            P, R = [], []
            for i, j in zip(np.atleast_1d(ia), np.atleast_1d(ih)):
                p = self.fk.palm_pose(qa[i], qh[j])
                P.append(p.palm_pos)
                R.append(p.extra["palm_rot"])
            P, R = np.asarray(P).reshape(-1, 3), np.asarray(R).reshape(-1, 3, 3)
            return P, R, P + R @ self.rel_p, R @ self.R_rel
        return fk_cup


def analyze(g: Grasp, steps, arm: V.Bag, hand: V.Bag, t_end: float) -> dict:
    model = CupFK.of(g)
    rel_p, R_rel = model.rel_p, model.R_rel
    t_grasp = next(st for st in steps if st[0] == g.step)
    t0, t1 = int(t_grasp[1] * 1e9), int(t_end * 1e9)
    hs = f"/hand_{g.side}"
    fk_cup = model.at(arm, hand, g.side, t0 - int(1e9), t1)

    lv = pose_series(arm, f"{g.obj_topic}/live_pose", t0, t1)
    sn = pose_series(arm, f"{g.obj_topic}/pose", t0, t1)

    def compare(series):
        recv, stamp, pos, R = series
        palm, palmR, cp, cR = fk_cup(stamp)
        d = np.linalg.norm(pos - cp, axis=1) if len(pos) else np.zeros(0)
        rot = np.array([rot_deg(a, b) for a, b in zip(R, cR)])
        ax = np.array([axis_deg(a, b) for a, b in zip(R, cR)])
        in_palm = np.einsum("nji,nj->ni", palmR, pos - palm) if len(pos) else np.zeros((0, 3))   # R^T(live − palm)
        tilt = lambda RR: np.degrees(np.arccos(np.clip(RR[:, 2, 2], -1, 1))) if len(RR) else np.zeros(0)   # noqa: E731 컵 축 ↔ base z
        return {"recv": recv, "stamp": stamp, "d": d, "rot": rot, "ax": ax, "in_palm": in_palm,
                "tilt": tilt(R), "fk_tilt": tilt(cR),
                "pos": pos, "R": R, "fk": cp, "phase": [phase_at(steps, t) for t in recv]}

    live, snap = compare(lv), compare(sn)
    grid = np.arange(t_grasp[1], t_end, 1.0 / GRID_HZ)
    stale = stale_mask(grid, live["recv"])
    il = np.clip(last_index(live["recv"], grid), 0, None)
    far = np.where(stale, False, live["d"][il] > TOL_M) if live["d"].size else np.zeros_like(stale)
    fk_grid_tilt = np.degrees(np.arccos(np.clip(fk_cup(grid)[3][:, 2, 2], -1, 1)))
    after = grid >= t_grasp[2]                                     # 잡기(들기) 끝 뒤
    lost = first_sustained(grid[after], (stale | far)[after])
    first_stale = first_sustained(grid[after], stale[after], 0.0)
    first_far = first_sustained(grid[after], far[after], 0.0)

    tj, jf = array_series(hand, f"{hs}/joint_forces", t0 - int(1e9), t1)        # 앞 1 s = 잡기 전 기준(start)
    tt, tip = array_series(hand, f"{hs}/tip_forces_zeroed", t0 - int(1e9), t1)
    return {"g": g, "rel_p": rel_p, "R_rel": R_rel, "live": live, "snap": snap, "grid": grid, "stale": stale, "far": far,
            "lost": lost, "first_stale": first_stale, "first_far": first_far, "jf": (tj, jf), "tip": (tt, tip),
            "grasp_end": t_grasp[2], "fk_cup": fk_cup, "fk_grid_tilt": fk_grid_tilt}


# ============================================================================ 표 · 그림
def clock(t: float | None) -> str:
    return "-" if t is None else dt.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-5]


def phase_rows(r: dict, steps) -> list[dict]:
    live, snap = r["live"], r["snap"]
    gph = [phase_at(steps, t) for t in r["grid"]]
    tj, jf = r["jf"]
    tt, tip = r["tip"]
    t_in = r["grid"][0]                                            # 그 앞 1 s = 잡기 전 기준 'start'
    jph = ["start" if t < t_in else phase_at(steps, t) for t in tj]
    tph = ["start" if t < t_in else phase_at(steps, t) for t in tt]
    out = []
    for ph in ["start"] + list(dict.fromkeys(gph)):
        m = np.array([p == ph for p in live["phase"]], bool)
        ms = np.array([p == ph for p in snap["phase"]], bool)
        gm = np.array([p == ph for p in gph])
        jm, tm = np.array([p == ph for p in jph], bool), np.array([p == ph for p in tph], bool)
        out.append({"phase": ph, "dur": gm.sum() / GRID_HZ, "n": int(m.sum()), "d": stats(1e3 * live["d"][m]),
                    "hz": m.sum() / max(gm.sum() / GRID_HZ, 1e-9), "tilt": np.median(live["tilt"][m]) if m.any() else np.nan,
                    "fk_tilt": np.median(r["fk_grid_tilt"][gm]) if gm.any() else np.nan,
                    "rot": stats(live["rot"][m]), "ax": stats(live["ax"][m]),
                    "over": float((live["d"][m] > TOL_M).mean()) if m.any() else np.nan,
                    "cut": float(r["stale"][gm].mean()) if gm.any() else np.nan, "snap_d": stats(1e3 * snap["d"][ms])["med"],
                    "in_palm": 1e3 * np.median(live["in_palm"][m], axis=0) if m.any() else np.full(3, np.nan),
                    "jf": np.mean(jf[jm], axis=0) if jm.any() else None, "tip": np.mean(tip[tm], axis=0) if tm.any() else None})
    return out


def f(v, n=0) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


def report(results: list[dict], steps, name: str, ep_dir: Path, bag_dir: Path) -> str:
    L = [f"# 잡은 뒤 FP++ 컵 vs FK 컵 — {name}", "",
         f"에피소드 `{ep_dir}` · bag `{bag_dir}`. 거리 = live 한 장마다 그 stamp 의 FK 컵(손바닥 FK ⊗ 묶음 rel)과. "
         f"끊김 = 받은 지 {STALE_S} s 넘은 30 Hz 칸 비율. 회전 = 전체(참고만 — FP++ 출력은 대칭축 twist 를 0 으로 바꾼다) · 축 = 컵 z 축 기울기 차(의미 있는 값). 손 힘 = 단계 평균(joint_forces g · tip_forces_zeroed).",
         "", "| 단계 | 시작 | 끝 | 결과 |", "|---|---|---|---|"]
    L += [f"| {sid} | {clock(a)} | {clock(b)} | {res} |" for sid, a, b, res in steps]
    for r in results:
        g = r["g"]
        L += ["", f"## {g.step} ({g.side}, {g.obj_topic})", "",
              f"묶음 `{g.bundle.relative_to(_ROOT)}` rel_p(mm) {np.round(1e3 * r['rel_p'], 1).tolist()} · 컵 축(palm) "
              f"{np.round(r['R_rel'][:, 2], 3).tolist()}", "",
              f"- 들기 끝 {clock(r['grasp_end'])} 뒤 처음 끊김(> {STALE_S} s) {clock(r['first_stale'])}"
              f"({phase_at(steps, r['first_stale']) if r['first_stale'] else '-'}), 처음 5 cm 넘음 {clock(r['first_far'])}"
              f"({phase_at(steps, r['first_far']) if r['first_far'] else '-'}), "
              f"{SUSTAIN_S:g} s 넘게 이어진 이탈 시작 {clock(r['lost'])}({phase_at(steps, r['lost']) if r['lost'] else '-'})"]
        k = r["live"]
        near = (k["recv"] >= r["grasp_end"] - 1.0) & (k["recv"] <= r["grasp_end"])
        if near.any():
            L.append(f"- 들기 끝 1 s live 컵(palm 좌표 mm) 중앙 {np.round(1e3 * np.median(k['in_palm'][near], 0), 1).tolist()} "
                     f"vs 묶음 {np.round(1e3 * r['rel_p'], 1).tolist()} → 위치 차 {f(1e3 * np.median(k['d'][near]), 1)} mm · "
                     f"회전 {f(float(np.median(k['rot'][near])), 1)}° · 축 {f(float(np.median(k['ax'][near])), 1)}° ({int(near.sum())}장) · "
                     f"컵 축 기울기(base z 대비) live {f(float(np.median(k['tilt'][near])), 1)}° · FK {f(float(np.median(k['fk_tilt'][near])), 1)}°")
        L += ["", "| 단계 | 초 | live 장(Hz) | 축 기울기 live/FK ° | 위치 중앙/p90/최대 mm | 회전 중앙/p90 ° | 축 중앙/p90/최대 ° | >5 cm | 끊김 | snap 중앙 mm "
              "| live palm 좌표 중앙 mm | joint_forces g " + "/".join(JF_NAMES) + " | tip " + "/".join(TIP_NAMES) + " |",
              "|" + "---|" * 13]
        for row in phase_rows(r, steps):
            d, ro, ax = row["d"], row["rot"], row["ax"]
            jf = "-" if row["jf"] is None else "/".join(f"{v:.0f}" for v in row["jf"])
            tip = "-" if row["tip"] is None else "/".join(f"{v:.1f}" for v in row["tip"])
            L.append(f"| {row['phase']} | {row['dur']:.1f} | {row['n']}({f(row['hz'], 1)}) | {f(row['tilt'])}/{f(row['fk_tilt'])} | {f(d['med'])}/{f(d['p90'])}/{f(d['max'])} | "
                     f"{f(ro['med'])}/{f(ro['p90'])} | {f(ax['med'])}/{f(ax['p90'])}/{f(ax['max'])} | "
                     f"{f(100 * row['over'])} % | {f(100 * row['cut'])} % | {f(row['snap_d'])} | "
                     f"{np.round(row['in_palm'], 0).astype(int).tolist() if not np.isnan(row['in_palm']).any() else '-'} | {jf} | {tip} |")
    return "\n".join(L) + "\n"


def plot(r: dict, steps, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k, sn, g = r["live"], r["snap"], r["g"]
    T0 = r["grid"][0]
    x = lambda t: np.asarray(t) - T0                                # noqa: E731
    fig, ax = plt.subplots(6, 1, figsize=(14, 17), sharex=True)
    ax[0].plot(x(k["recv"]), 1e3 * k["d"], ".", ms=3, label="live − FK")
    ax[0].plot(x(sn["recv"]), 1e3 * sn["d"], ".", ms=2, alpha=0.5, label="snapshot − FK")
    ax[0].axhline(1e3 * TOL_M, color="r", ls="--", lw=1, label="place LIVE_FK_TOL 50 mm")
    ax[0].set_ylabel("pos diff mm"), ax[0].set_yscale("symlog", linthresh=10)
    ax[1].plot(x(k["recv"]), k["ax"], ".", ms=3, label="cup axis diff")
    ax[1].plot(x(k["recv"]), k["rot"], ".", ms=2, alpha=0.4, label="full rot diff")
    ax[1].plot(x(k["recv"]), k["tilt"], "x", ms=3, label="live cup tilt vs base z")
    ax[1].plot(x(r["grid"]), r["fk_grid_tilt"], "-", lw=0.8, label="FK cup tilt vs base z")
    ax[1].set_ylabel("orientation diff deg")
    for i, c in enumerate("rgb"):
        ax[2].plot(x(k["recv"]), 1e3 * k["in_palm"][:, i], ".", ms=2, color=c, label=f"live palm {'xyz'[i]}")
        ax[2].axhline(1e3 * r["rel_p"][i], color=c, ls="--", lw=1)
    ax[2].set_ylabel("live cup in palm frame mm\n(dashed = bundle rel_p)"), ax[2].set_ylim(-200, 200)
    ax[3].plot(x(r["grid"]), r["stale"].astype(int), drawstyle="steps-post", label=f"live stale (> {STALE_S} s)")
    ax[3].plot(x(r["grid"]), r["far"].astype(int) * 0.5, drawstyle="steps-post", label="live > 5 cm")
    ax[3].set_ylabel("stale / off")
    tj, jf = r["jf"]
    for i, n in enumerate(JF_NAMES[: jf.shape[1] if jf.size else 0]):
        ax[4].plot(x(tj), jf[:, i], lw=0.8, label=n)
    ax[4].set_ylabel("joint_forces g")
    tt, tip = r["tip"]
    for i, n in enumerate(TIP_NAMES[: tip.shape[1] if tip.size else 0]):
        ax[5].plot(x(tt), tip[:, i], lw=0.8, label=n)
    ax[5].set_ylabel("tip_forces_zeroed")
    ax[5].set_xlabel(f"s (0 = {clock(T0)} {g.step} enter)")
    cols = plt.cm.tab10.colors
    for a in ax:
        for j, (sid, s0, s1, _) in enumerate(steps):
            if s1 >= T0:
                a.axvspan(x(s0), x(s1), color=cols[j % 10], alpha=0.10)
        if r["lost"] is not None:
            a.axvline(x(r["lost"]), color="k", ls=":", lw=1.2)
        a.legend(loc="upper left", fontsize=7, ncol=6)
        a.grid(alpha=0.3)
    for j, (sid, s0, s1, _) in enumerate(steps):
        if s1 >= T0:
            ax[0].text(x(max(s0, T0)), ax[0].get_ylim()[1], sid, fontsize=7, va="top", rotation=90)
    fig.suptitle(f"{g.step} ({g.side}) FP++ live vs FK cup (palm FK x bundle rel) - dotted = lost {clock(r['lost'])}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def frames(r: dict, bag_dir: Path, out: Path, around: float, offsets=(-1.0, -0.5, 0.0, 0.5, 1.0, 2.0), tag: str = "lost") -> list[Path]:
    """놓친 순간 앞뒤 카메라 프레임에 live 컵(굵은 축) · snapshot(가는 축) · FK 컵(자홍 + · 축)을 겹친다."""
    import cv2
    cam = V.Bag(bag_dir / "camera")
    info = cam.read(V.COLOR_INFO)
    if not info:
        return []
    K, D = np.array(info[0][1].k, float).reshape(3, 3), np.array(info[0][1].d, float)
    T_cal = np.asarray(V._base_from_camera(V.DEFAULT_EXTRINSICS), float)
    q_cal = V.calib_head_urdf_deg(V.DEFAULT_EXTRINSICS)
    lo, hi = int((around + min(offsets) - 1) * 1e9), int((around + max(offsets) + 1) * 1e9)
    head = V.track(cam.read(V.HEAD, lo, hi), value=lambda m: tuple(float(np.degrees(m.position[m.name.index(j)]))
                                                                    for j in ("head_j_pan", "head_j_tilt")))
    imgs = cam.read(V.COLOR, lo, hi)
    k, sn = r["live"], r["snap"]
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for off in offsets:
        tw = around + off
        t, msg = min(imgs, key=lambda im: abs(im[0] * 1e-9 - tw))
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        ts = V.stamp_ns(msg.header) * 1e-9
        q, _ = V.at(head, t, 500_000_000)
        T = V.head_corrected(T_cal, q_cal, q)
        _, _, cp, cR = r["fk_cup"](np.array([ts]))
        lines = [(f"{clock(ts)}  {off:+.1f}s from {tag}  {r['g'].step}", (255, 255, 255))]
        for series, thick, ln, lab in ((k, 3, 0.06, "live"), (sn, 1, 0.03, "snap")):
            i = int(last_index(series["stamp"], ts))
            if i >= 0:
                age = ts - series["stamp"][i]
                draw_R(img, K, D, T, series["pos"][i], series["R"][i], ln, thick, f"{lab} {age * 1e3:.0f}ms")
                lines.append((f"{lab}: age {age * 1e3:.0f} ms  |{lab}-FK| {1e3 * series['d'][i]:.0f} mm  ax {series['ax'][i]:.0f} deg",
                              (255, 255, 255)))
        draw_R(img, K, D, T, cp[0], cR[0], 0.05, 2, "FK cup", cross=True)
        V.draw_text(img, lines)
        p = out / f"{tag}_{off:+.1f}s.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def draw_R(img, K, D, T, pos, R, length: float, thick: int, label: str, cross: bool = False) -> None:
    """V.draw_axes 와 같은 그림(x 빨강 · y 초록 · z 파랑)을 회전 행렬로. cross = 원점에 자홍 +."""
    import cv2
    uv = V.project(K, D, T, np.vstack([pos, pos + length * np.asarray(R).T]))
    if np.isnan(uv[0]).any():
        return
    o = tuple(int(v) for v in uv[0])
    for k, bgr in enumerate(((0, 0, 255), (0, 200, 0), (255, 0, 0))):
        if not np.isnan(uv[k + 1]).any():
            cv2.line(img, o, tuple(int(v) for v in uv[k + 1]), bgr, thick, cv2.LINE_AA)
    if cross:
        cv2.drawMarker(img, o, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(img, label, (o[0] + 6, o[1] + (18 if cross else -6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,   # FK 글자는 아래로(live 와 안 겹침)
                (255, 0, 255) if cross else (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================================ 잡기 기록(grasp_record)
RECORD_POINTS = (("end", "잡기 단계가 끝난 시각"), ("end_plus_1s", "끝난 뒤 1 s"), ("before_next", "다음 단계 시작 0.2 s 전"))
HAND_TOPICS = ("joint_states", "angle_actual", "force_actual", "joint_forces", "joint_forces_zeroed", "tip_forces", "tip_forces_zeroed")


def record_times(steps, step_id: str, t_bag_end: float) -> dict[str, float]:
    """잡기 끝 · 끝 + 1 s · 다음 단계 시작 0.2 s 전(다음 단계가 없으면 bag 끝 0.2 s 전)."""
    i = next(k for k, st in enumerate(steps) if st[0] == step_id)
    end = steps[i][2]
    nxt = steps[i + 1][1] if i + 1 < len(steps) else t_bag_end
    return {"end": end, "end_plus_1s": end + 1.0, "before_next": max(end, nxt - 0.2)}


def _r(v, n: int = 6):
    """json 용 반올림(다시 돌려도 같은 파일)."""
    if v is None:
        return None
    a = np.asarray(v, float)
    return round(float(a), n) if a.ndim == 0 else [round(float(x), n) for x in a.reshape(-1)]


def _pose(pos, R) -> dict:
    from grasp_s2r_core import _quat_from_matrix
    return {"pos_m": _r(pos), "quat_wxyz": _r(_quat_from_matrix(np.asarray(R, float)))}


def _last(bag: V.Bag, topic: str, t: float, lookback_s: float):
    """t 이하에 받은 마지막 메시지 (받은 시각 s, msg) — 없거나 타입을 못 풀면 (None, 이유)."""
    try:
        rows = bag.read(topic, int((t - lookback_s) * 1e9), int(t * 1e9))
    except (ImportError, ValueError) as e:      # rh56f1_interfaces 를 source 하지 않았다
        return None, f"unavailable: {type(e).__name__} {e} (source robot_control/ros_ws/install/setup.bash)"
    return (rows[-1][0] * 1e-9, rows[-1][1]) if rows else (None, f"no message in {lookback_s:g} s")


def _msg_pose(m) -> tuple[np.ndarray, np.ndarray]:
    o = m.pose.orientation
    return np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z]), V.quat_wxyz_to_R([o.w, o.x, o.y, o.z])


def _hand_state(hand: V.Bag, side: str, t: float) -> dict:
    out = {}
    for name in HAND_TOPICS:
        tr, m = _last(hand, f"/hand_{side}/{name}", t, 0.5)
        if tr is None:
            out[name] = {"error": m}
        elif name == "joint_states":
            out[name] = {"age_s": _r(t - tr, 4), "names": list(m.name), "position_rad": _r(m.position)}
        elif hasattr(m, "joint_values"):
            out[name] = {"age_s": _r(t - tr, 4), "names": list(m.joint_names), "values": [int(v) for v in m.joint_values]}
        else:
            out[name] = {"age_s": _r(t - tr, 4), "values": _r(m.data),
                         **({"names": list(JF_NAMES)} if name.startswith("joint_forces") else
                            {"names": list(TIP_NAMES)} if name.startswith("tip_forces") else {})}
    return out


def _camera(cam: V.Bag | None, t: float):
    """t 이전 마지막 컬러 프레임(stamp 기준, 없으면 가장 가까운 것) → (stamp s, msg) | (None, None)."""
    if cam is None:
        return None, None
    rows = cam.read(V.COLOR, int((t - 0.5) * 1e9), int((t + 0.5) * 1e9))
    if not rows:
        return None, None
    before = [r for r in rows if V.stamp_ns(r[1].header) * 1e-9 <= t]
    t_rec, m = before[-1] if before else min(rows, key=lambda r: abs(V.stamp_ns(r[1].header) * 1e-9 - t))
    return V.stamp_ns(m.header) * 1e-9, m


def _overlay(cam: V.Bag, msg, ts: float, cup: dict, fk_pos, fk_R, title: str, path: Path) -> None:
    import cv2
    info = cam.read(V.COLOR_INFO, int((ts - 5) * 1e9), int((ts + 5) * 1e9)) or cam.read(V.COLOR_INFO)
    K, D = np.array(info[0][1].k, float).reshape(3, 3), np.array(info[0][1].d, float)
    T_cal = np.asarray(V._base_from_camera(V.DEFAULT_EXTRINSICS), float)
    head = cam.read(V.HEAD, int((ts - 0.5) * 1e9), int(ts * 1e9))
    q = None if not head else tuple(float(np.degrees(head[-1][1].position[head[-1][1].name.index(j)]))
                                    for j in ("head_j_pan", "head_j_tilt"))
    T = V.head_corrected(T_cal, V.calib_head_urdf_deg(V.DEFAULT_EXTRINSICS), q)
    img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    lines = [(title, (255, 255, 255))]
    for key, ln, thick in (("live", 0.06, 3), ("pose", 0.03, 1)):
        c = cup.get(key)
        if c is not None:
            draw_R(img, K, D, T, c["pos"], c["R"], ln, thick, f"{key} {c['age'] * 1e3:.0f}ms")
            d = cup[f"{key}_minus_fk"]
            lines.append((f"{key}: age {c['age'] * 1e3:.0f} ms |{key}-FK| {d['pos_mm']:.0f} mm  rot {d['rot_deg']:.0f}  "
                          f"axis {d['axis_deg']:.0f} deg", (255, 255, 255)))
        else:
            lines.append((f"{key}: none", (0, 0, 255)))
    draw_R(img, K, D, T, fk_pos, fk_R, 0.05, 2, "FK cup", cross=True)
    V.draw_text(img, lines)
    cv2.imwrite(str(path), img)


def record_grasp(g: Grasp, steps, arm: V.Bag, hand: V.Bag, cam: V.Bag | None, t_bag_end: float, out: Path,
                 meta: dict) -> dict:
    """잡기 하나 → out/record.json + <시점>.jpg(원본) · <시점>_overlay.png. 같은 입력이면 같은 파일."""
    model = CupFK.of(g)
    out.mkdir(parents=True, exist_ok=True)
    st = next(x for x in steps if x[0] == g.step)
    times = record_times(steps, g.step, t_bag_end)
    rec = {**meta, "state_id": g.step, "side": g.side, "object_topic": g.obj_topic,
           "step": {"enter": _r(st[1], 3), "exit": _r(st[2], 3), "enter_clock": clock(st[1]), "exit_clock": clock(st[2]),
                    "result": st[3]},
           "bundle": {"path": str(g.bundle.relative_to(_ROOT)), "rel_p_m": _r(model.rel_p), "rel_q_wxyz": _r(model.rel_q)},
           "fk": {"contract": str(g.contract.relative_to(_ROOT)), "palm_body": model.fk.palm_body,
                  "urdf": str(model.fk.tree.path)},
           "conventions": {"frame": "base_link", "quat": "wxyz",
                           "fk_cup": "palm FK(UrdfChainFK, 계약 palm_body) ⊗ bundle rel (rh_place.cup_from_palm)",
                           "X_minus_fk": "X 의 영상 stamp 시각 관절로 푼 FK 컵과 비교(지연 제외). axis = 컵 z 축 각도 차. "
                                         "rot_deg 은 참고만 — object_pose_node 가 대칭축 twist 를 0 으로 바꿔 낸다(remove_twist)",
                           "points": dict(RECORD_POINTS)},
           "points": {}}
    for key, t in times.items():
        lv = _last(arm, f"{g.obj_topic}/live_pose", t, 30.0)
        ps = _last(arm, f"{g.obj_topic}/pose", t, 30.0)
        stamps = [V.stamp_ns(m.header) * 1e-9 for tr, m in (lv, ps) if tr is not None]
        fk_cup = model.at(arm, hand, g.side, int((min([t] + stamps) - 0.2) * 1e9), int(t * 1e9))
        palm, palmR, cp, cR = (a[0] for a in fk_cup(np.array([t])))
        pt = {"t": _r(t, 3), "clock": clock(t), "palm_fk": _pose(palm, palmR), "fk_cup": _pose(cp, cR)}
        draw = {}
        for name, (tr, m) in (("live", lv), ("pose", ps)):
            if tr is None:
                pt[name], pt[f"{name}_minus_fk"] = {"error": m}, None
                continue
            pos, R = _msg_pose(m)
            ts = V.stamp_ns(m.header) * 1e-9
            p2, R2, c2, cR2 = (a[0] for a in fk_cup(np.array([ts])))
            diff = {"pos_mm": _r(1e3 * np.linalg.norm(pos - c2), 2), "vec_mm": _r(1e3 * (pos - c2), 2),
                    "rot_deg": _r(rot_deg(R, cR2), 2), "axis_deg": _r(axis_deg(R, cR2), 2),
                    "in_palm_mm": _r(1e3 * (R2.T @ (pos - p2)), 2), "bundle_rel_mm": _r(1e3 * model.rel_p, 2)}
            pt[name] = {**_pose(pos, R), "stamp": _r(ts, 3), "age_s": _r(t - ts, 3), "recv_age_s": _r(t - tr, 3)}
            pt[f"{name}_minus_fk"] = diff
            draw[name], draw[f"{name}_minus_fk"] = {"pos": pos, "R": R, "age": t - ts}, diff
        pt["hand"] = _hand_state(hand, g.side, t)
        ts_img, img = _camera(cam, t)
        if img is None:
            pt["camera"] = None
        else:
            jpg, png = out / f"{key}.jpg", out / f"{key}_overlay.png"
            jpg.write_bytes(bytes(img.data))           # 원본 그대로(CompressedImage jpeg)
            _, _, ci, cRi = (a[0] for a in fk_cup(np.array([ts_img])))
            _overlay(cam, img, ts_img, draw, ci, cRi, f"{g.step} {key} {clock(ts_img)}", png)
            pt["camera"] = {"jpg": jpg.name, "overlay": png.name, "format": img.format, "stamp": _r(ts_img, 3),
                            "dt_s": _r(ts_img - t, 3)}
        rec["points"][key] = pt
    (out / "record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rec


# ============================================================================ 에피소드 · bag 찾기


def resolve_episode(arg: str) -> Path:
    if arg != "latest":
        p = Path(arg)
        p = p if p.is_absolute() or p.exists() else _ROOT / p
        if not p.is_dir():
            raise SystemExit(f"에피소드 폴더가 없다: {arg}")
        return p.resolve()
    cands = [d for d in run_paths.find_runs("episode") if d.is_dir() and run_paths.run_time(d) and any(d.glob("*.jsonl"))]
    if not cands:
        raise SystemExit(f"{run_paths.RUNS}/*/episode 에 에피소드가 없다")
    return cands[-1]


def find_bag(ep_dir: Path, name: str) -> Path:
    """에피소드 시작 시각 앞 10 분 안에 시작한 logs/runs/*/bags/<시각>_episode_<이름> 중 가장 늦은 것."""
    t_ep = run_paths.run_time(ep_dir)
    if t_ep is None:
        raise SystemExit(f"{ep_dir}: logs/runs/<YYMMDD>/episode/<HHMMSS>_… 형식이 아니다 — --bag 으로 준다")
    t_ep = t_ep.timestamp()
    best = None
    for d in run_paths.find_runs("bags", f"*_episode_{name}"):
        t = run_paths.run_time(d)
        if t is None:
            continue
        t = t.timestamp()
        if t_ep - 600 <= t <= t_ep + 5 and (best is None or t > best[0]):
            best = (t, d)
    if best is None:
        raise SystemExit(f"{ep_dir.name}: {run_paths.RUNS}/*/bags 에 맞는 bag(*_episode_{name}, 에피소드 10 분 전 ~ 5 s 뒤 시작)이 없다")
    return best[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--episode", required=True, help="logs/runs/<YYMMDD>/episode/<시각>_<이름> 또는 latest")
    ap.add_argument("--bag", type=Path, default=None, help="기본 = 에피소드 시각으로 logs/runs/*/bags/ 에서 찾는다")
    ap.add_argument("--analysis", type=Path, default=None,
                    help="전체 시계열 분석(report.md · 단계 그림 · 놓침 프레임)도 이 폴더에 쓴다(기본 = 잡기 기록만)")
    ap.add_argument("--no-frames", action="store_true", help="--analysis 의 놓침 카메라 프레임을 뽑지 않는다")
    ap.add_argument("--at", nargs="*", default=[], help="--analysis: 놓침 말고도 프레임을 뽑을 시각 HH:MM:SS[.f]")
    args = ap.parse_args(argv)
    ep = resolve_episode(args.episode)
    lines = [json.loads(ln) for f in sorted(ep.glob("*.jsonl")) for ln in f.read_text().splitlines() if ln.strip()]
    steps = parse_steps(lines)
    name, grasps = plan_grasps(ep, steps)
    bag = (args.bag if args.bag is None or args.bag.is_absolute() else _ROOT / args.bag) or find_bag(ep, name)
    print(f"[grasp_fk_vs_fpp] 에피소드 {ep} · bag {bag} · 성공한 잡기 {[g.step for g in grasps]}")
    if not grasps:
        return 0
    arm, hand = V.Bag(bag / "arm"), V.Bag(bag / "hand")
    cam = V.Bag(bag / "camera") if any((bag / "camera").glob("*.db3")) else None
    t_end = max(c.execute("select max(timestamp) from messages").fetchone()[0] for c in arm.cons) * 1e-9
    meta = {"episode": name, "episode_dir": str(ep.relative_to(_ROOT)) if ep.is_relative_to(_ROOT) else str(ep),
            "bag": str(bag.relative_to(_ROOT)) if bag.is_relative_to(_ROOT) else str(bag)}
    for g in grasps:
        rec = record_grasp(g, steps, arm, hand, cam, t_end, ep / "grasp_record" / g.step, meta)
        d = {k: (p["live_minus_fk"] or {}).get("pos_mm") for k, p in rec["points"].items()}
        print(f"[grasp_record] {g.step}: live−FK mm {d} → {ep / 'grasp_record' / g.step}")
    if args.analysis is not None:
        out = args.analysis if args.analysis.is_absolute() else _ROOT / args.analysis
        out.mkdir(parents=True, exist_ok=True)
        results = []
        for g in grasps:
            r = analyze(g, steps, arm, hand, t_end)
            plot(r, steps, out / f"{g.step}.png")
            if not args.no_frames and cam is not None:
                t = r["lost"] or r["first_far"] or r["first_stale"]
                day = dt.datetime.fromtimestamp(r["grid"][0]).date()
                extra = [dt.datetime.combine(day, dt.datetime.strptime(a, "%H:%M:%S.%f" if "." in a else "%H:%M:%S").time()).timestamp()
                         for a in args.at]
                for k, tt in enumerate(([t] if t is not None else []) + extra):
                    tag = "lost" if k == 0 and t is not None else clock(tt).replace(":", "")
                    ps = frames(r, bag, out / f"frames_{g.step}", tt, tag=tag)
                    print(f"[frames] {g.step} {tag}: {[p.name for p in ps]}")
            results.append(r)
        (out / "report.md").write_text(report(results, steps, name, ep, bag), encoding="utf-8")
        print(f"[grasp_fk_vs_fpp] → {out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
