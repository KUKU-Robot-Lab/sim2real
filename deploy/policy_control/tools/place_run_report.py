#!/usr/bin/env python3
"""실기 place 노드(rh_place_node) 오프라인 비교표 — 에피소드 폴더 + bag 만 읽는다(로봇 · ROS 노드 · GPU 없음).

    .venv/bin/python deploy/policy_control/tools/place_run_report.py --episode-dir logs/runs/261116/episode/100033_cocktail_1c_e4000 [더 …]

place 노드(place_blue · place_pink · place_shaker · place_cup …)마다 한 열: 정책 id · md5(계약) · 래치 방식, ready_carry,
시작 틱 관측(손바닥 틀 컵 z vs 학습 뱅크 · 컵−자리 dz · 뱅크 범위 밖 · |z|>3 차원 수, 앞 78 칸 = 촉각 · prev_action · 래치 빼고),
손 열기(t15 · 20 틱 열림), 래치, 손 편 순간 컵 높이(FK⊗T_in · FP++), 끝(outcome · cup_in_holder …), FP++−FK 컵 차.
틱 = 정책 틱(policy_action 이 나온 obs 틱, 준비 이송 틱 제외) 0 부터. 출력: stdout markdown + logs/analysis/place_compare/<날짜>_<에피소드>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_bag as RB
from policy_control import rh_aglt as A
from policy_control import rh_place as RP
from policy_control import tin_live as TL
from policy_control.contract_assets import ASSETS
from policy_control.fk_numpy import UrdfChainFK
from policy_control.place_head_pose import HOLDER_TOP_Z

POLICIES = RB.SIM2REAL / "deploy" / "policies"
#: 계약 폴더에 bank/ 가 없는 정책의 학습 시작 뱅크(README start_bank_path)
BANKS = {"rh56f1/place/right_n07": Path.home() / "rl_ws/our_source/rh_aglt_runs/eval/1023_grasp_bank/carry_end_rcg04_r.npz"}
PHYS = ("arm_q", "arm_qd", "arm_q_target", "hand_q", "hand_err", "palm_pos", "palm_rot", "cup_rel_palm", "cup_up",
        "tips_rel_cup", "kp_rel_goal", "goal_rel_palm")      # 78 칸 — 뱅크로 다시 만들 수 있는 관측
Z_BIG = 3.0
_BANK: dict = {}          # (정책, 물체) → bank_obs — 여러 에피소드를 한 번에 볼 때 다시 만들지 않는다
OPEN20 = 20


def open_latch(hand0, ht, hq, close_dir, fingers, latch, thr: float = RP.OPEN_MIN_RAD) -> dict:
    """정책 틱별 손 목표 ht · 실측 hq (K,6) → t15(손 목표 열림 ≥ thr 첫 틱) · 20 틱 열림 · 실측 손 편 틱 · 래치 첫 틱.
    열림 = 네 손가락 평균 close_dir·(hand0 − q) — rh_place.PlaceChain.opened 와 같은 식."""
    f = list(fingers)
    op = np.mean(close_dir[f] * (hand0[f] - ht[:, f]), axis=1)
    opm = np.mean(close_dir[f] * (hand0[f] - hq[:, f]), axis=1)
    first = lambda m: int(np.argmax(m)) if m.any() else None  # noqa: E731
    lat = None if latch is None else np.asarray(latch) > 0.5
    return {"t15": first(op >= thr), "open20": float(op[OPEN20 - 1]) if len(op) >= OPEN20 else None,
            "t_rel": first(opm >= thr), "latch_tick": None if lat is None else first(lat),
            "latch_all0": None if lat is None else not lat.any(), "opened": op, "opened_meas": opm}


def slices(c) -> dict:
    out, k = {}, 0
    for n, w in A.actor_layout(c):
        out[n] = slice(k, k + w)
        k += w
    return out


def palm_R(o, I) -> np.ndarray:
    R = np.zeros((3, 3))
    R[:, 0], R[:, 1] = o[I["palm_rot"]][:3], o[I["palm_rot"]][3:]
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    return R


def n_phys(I) -> int:
    return I[PHYS[-1]].stop


BANK_CONST_STD = 1e-6  # 뱅크에서 이 칸이 사실상 한 값(예: carry_end 뱅크 joint_vel 전부 0) → z 대신 절대 차이


def bank_compare(B, x, cells, phys):
    """뱅크 B(N, D) 대비 x(D). std≈0 칸은 z·범위 밖 집계에서 빼고 '뱅크 상수' 로 따로 절대 차이를 낸다."""
    mu, sd, lo, hi = B.mean(0), B.std(0), B.min(0), B.max(0)
    const = sd < BANK_CONST_STD
    z = np.where(const, 0.0, (x - mu) / np.where(const, 1.0, sd))
    bc = {}
    for nm, k in cells:
        if const[k]:
            bc[nm] = round(max(bc.get(nm, 0.0), float(abs(x[k] - mu[k]))), 4)
    return dict(n_out=int((((x < lo) | (x > hi)) & ~const).sum()), n_z3=int((np.abs(z) > Z_BIG).sum()),
                n_bank_const=int(const.sum()), bank_const=sorted(bc.items()),
                top_z=[(n, round(float(z[k]), 1)) for n, k in sorted(
                    ((nm, k) for nm, k in cells if nm in phys and not const[k]), key=lambda e: -abs(z[e[1]]))[:5]])



def bank_obs(c, path: Path, obj: str | None):
    """학습 시작 뱅크 → 앞 78 칸 관측(core.bank_obs · bank07 식) · 손바닥 틀 컵 z · node(없으면 None)."""
    s = c.side()
    b = np.load(path, allow_pickle=True)
    jn = list(b["joint_names"])
    hi, ai = [jn.index(n) for n in s.hand_joints], [jn.index(n) for n in s.arm_joints]
    fk = UrdfChainFK(ASSETS[c.asset].urdf, s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)
    cc = A.select_object(c, obj) if c.object_names else c
    H = np.asarray(b["holder_pos"], float)
    B, pz = [], []
    for i in range(len(b["joint_pos"])):
        aq, hq = b["joint_pos"][i][ai].astype(float), b["joint_pos"][i][hi].astype(float)
        p = fk.palm_pose(aq, hq)
        Rm, pp = np.asarray(p.extra["palm_rot"]), np.asarray(p.palm_pos)
        cr = b["cup_root"][i].astype(float)
        cp, cq = cr[:3], cr[3:7]
        g = RP.seat_goal(cc, H[i][0 if H.shape[1] == 1 else int(b["target"][i])])
        tips = np.asarray(p.tips).reshape(5, 3)
        parts = [aq, b["joint_vel"][i][ai], b["arm_q_target"][i], hq,
                 np.clip((b["hand_target"][i] - hq) / c.joint_err_norm, -1, 1), pp, np.concatenate([Rm[:, 0], Rm[:, 1]]),
                 cp - pp, A.quat_apply(cq, np.array([0, 0, 1.0])), (tips - cp).reshape(-1),
                 (A.keypoints(cc, cp, cq) - A.keypoints(cc, g.pos, g.quat)).reshape(-1), g.pos - pp]
        B.append(np.clip(np.concatenate([np.asarray(x, float).reshape(-1) for x in parts]), -c.obs_clip, c.obs_clip))
        pz.append(float((Rm.T @ (cp - pp))[2]))
    return np.array(B), np.array(pz), (np.asarray(b["node"]) if "node" in b.files else None)


def bank_path(policy: str, side: str) -> Path:
    p = BANKS.get(policy, POLICIES / policy / "bank" / f"bank_{side}_train.npz")
    if not p.is_file():
        raise FileNotFoundError(f"학습 뱅크 없음 {p}")
    return p


def cup_world(o, I, rel_p, rel_q, bz):
    """FK⊗T_in 컵 원점 · 바닥점(관측 손바닥 자세 ⊗ 붙인 자세)."""
    R = palm_R(o, I)
    pos = o[I["palm_pos"]] + R @ rel_p
    up = R @ TL.quat_to_R(rel_q)[:, 2]
    return pos, pos + bz * up


def fpp_at(track, t, bz):
    got = None if track is None else track.at(t)
    if got is None:
        return None
    p, q, age = got
    return p, p + bz * TL.quat_to_R(q)[:, 2], age, TL.tilt_deg(q)


def node_report(w: dict, bag: Path, flow: dict, live_topics: dict, ep_yaml: dict) -> dict:
    nid = w["id"]
    lines = flow.get(nid) or []
    pnode = next((n for n in ep_yaml.get("sequence", []) if n.get("id") == nid), {})
    role = w["enter"].get("policy_name")
    policy = lines[0].split(" · ")[1] if lines and " · " in lines[0] else \
        str(((ep_yaml.get("policies") or {}).get(role) or {}).get("policy", ""))
    obj = lines[1].split(" → ")[0] if len(lines) > 1 else pnode.get("source_object")
    cpath = POLICIES / policy / "rh_place_contract.json"
    c = RP.load_contract(cpath)
    raw = json.loads(cpath.read_text(encoding="utf-8"))
    side = {"r": "right", "l": "left"}.get(c.side().side, c.side().side)
    cs = c.side()
    I = slices(c)
    t0, t1 = w["t0"], (w["t1"] if w["t1"] is not None else w["t0"] + 600) + 1.0
    base = f"/policy_control/{side}_place"
    P = RB.read(RB.db_of(bag, "pd"), [f"{base}/obs", f"{base}/hand_target", f"{base}/policy_action"], t0, t1)
    st_topic = f"/policy_control/status/rh_place_node_{side}"
    live_topic = live_topics.get(obj)
    S = RB.read(RB.db_of(bag, "arm"), [st_topic] + ([live_topic] if live_topic else []), t0 - 30.0, t1)
    r = {"node": nid, "policy": policy, "ckpt_md5": raw.get("checkpoint_md5"), "object": obj,
         "latch_mode": "GRU est" if c.latch_est else f"없음(래치 컵 = {'actor 컵' if c.cup_obs == 'live' else '붙인 자세'})",
         "notes": []}
    stat = [json.loads(m.data) for t, m in S[st_topic] if t >= t0]
    last = next((d for d in reversed(stat) if d.get("place")), {})
    pl = last.get("place") or {}
    sig = ((w["end"] or {}).get("signals") or {}).get(role) or {}
    rc = sig.get("ready_carry") or pl.get("ready_carry") or {}
    r["ready_carry"] = {k: rc.get(k) for k in ("carry_ok", "start_ok", "err_mm", "axis_deg", "t")} if rc else None
    attach = pl.get("attach") or {}
    obs = P[f"{base}/obs"]
    if not obs or not P[f"{base}/policy_action"]:
        r["notes"].append("obs · policy_action 없음(정책이 돌지 않았다)")
        return r
    to = np.array([t for t, _ in obs])
    O = np.array([list(m.data) for _, m in obs])
    j = np.array([int(np.argmin(np.abs(to - t))) for t, _ in P[f"{base}/policy_action"]])
    th = np.array([t for t, _ in P[f"{base}/hand_target"]])
    HT = np.array([list(m.data) for _, m in P[f"{base}/hand_target"]])
    ht = HT[[int(np.argmin(np.abs(th - to[k]))) for k in j]]
    j0 = int(j[0])
    o0 = O[j0]
    hand0 = O[j0 - 1][I["hand_q"]] if c.ready_carry and j0 > 0 else o0[I["hand_q"]]   # 이송 끝 틱 실측 = _place_begin
    close_dir = np.sign(np.asarray(cs.hand_grip, float) - np.asarray(cs.hand_open, float))
    fingers = [i for i, n in enumerate(cs.hand_joints) if "thumb" not in n]
    ol = open_latch(hand0, ht, O[j][:, I["hand_q"]], close_dir, fingers, O[j][:, I["latch"]] if "latch" in I else None)
    dz = lambda o: float((o[I["cup_rel_palm"]] - o[I["goal_rel_palm"]])[2])  # noqa: E731
    lat = next((d["place"] for d in stat if (d.get("place") or {}).get("latch") and d["place"].get("latch_cup") and d["place"].get("seat")), None)
    r["latch_dz_status"] = None if lat is None else float(lat["latch_cup"][2] - lat["seat"][2])   # 래치 입력 컵(추정기 보정 뒤) − 자리
    r.update(policy_ticks=len(j), t15=ol["t15"], open20=ol["open20"], latch_tick=ol["latch_tick"], latch_all0=ol["latch_all0"],
             latch_dz=None if ol["latch_tick"] is None else dz(O[j[ol["latch_tick"]]]), t_rel=ol["t_rel"], start_dz=dz(o0))
    # 손바닥 틀 컵 z — 붙인 자세(T_in) 와 관측
    rel_p = np.asarray(attach.get("rel_p") or [np.nan] * 3, float)
    rel_q = np.asarray(attach.get("rel_q") or [1, 0, 0, 0], float)
    r["tin_z"], r["tin_source"] = float(rel_p[2]), attach.get("source") or attach.get("attach_source") or "bundle"
    r["obs_palm_cup_z"] = float((palm_R(o0, I).T @ o0[I["cup_rel_palm"]])[2])
    np_ = n_phys(I)
    try:
        bp = bank_path(policy, side)
        geo_obj = pnode.get("policy_object") or (c.object_names[0] if c.object_names else None)
        key = (policy, geo_obj)
        if key not in _BANK:
            _BANK[key] = bank_obs(c, bp, geo_obj)
        B, pz, bnode = _BANK[key]
        if bnode is not None and nid in set(bnode.tolist()):
            B, pz = B[bnode == nid], pz[bnode == nid]
        cells = [(nm, k) for nm, w_ in A.actor_layout(c) for k in range(I[nm].start, I[nm].stop) if k < np_]
        r.update(bank=str(bp), bank_n=len(B), bank_palm_cup_z=float(np.median(pz)), tin_z_minus_bank=float(rel_p[2] - np.median(pz)),
                 **bank_compare(B, o0[:np_], cells, PHYS))
    except (FileNotFoundError, KeyError, ValueError, RP.RhPlaceError, A.RhAgltError) as exc:
        r["notes"].append(f"뱅크 열 비움: {exc}")
    # 컵 높이(바닥점 − 홀더 링 윗면) · FP++−FK
    geo = c.object_geometry.get(pnode.get("policy_object") or "", None) or next(iter(c.object_geometry.values()), {})
    bz = float(geo.get("bottom_z", -c.cup_half_height))
    hz = (last.get("holder_pose") or [np.nan] * 3)[2]
    track = RB.PoseTrack([(t, m) for t, m in S[live_topic]]) if live_topic and S[live_topic] else None
    if track is None:
        r["notes"].append(f"FP++ live 가 이 노드 창(−30 s 포함)에 없음({live_topic})")

    def at(k_obs, t, tag):
        fk_p, fk_b = cup_world(O[k_obs], I, rel_p, rel_q, bz)
        f = fpp_at(track, t, bz)
        r[f"{tag}_h_fk_mm"] = (fk_b[2] - hz - HOLDER_TOP_Z) * 1e3
        if f is not None:
            r[f"{tag}_h_fpp_mm"], r[f"{tag}_fpp_age_s"] = (f[1][2] - hz - HOLDER_TOP_Z) * 1e3, f[2]
            r[f"{tag}_fpp_fk_mm"], r[f"{tag}_fpp_fk_dz_mm"] = float(np.linalg.norm(f[0] - fk_p)) * 1e3, float(f[0][2] - fk_p[2]) * 1e3
        return f

    at(j0, to[j0], "start")
    if ol["t_rel"] is not None:
        k = int(j[ol["t_rel"]])
        at(k, to[k], "rel")
    t_end = w["t1"] if w["t1"] is not None else to[-1]
    k_end = int(np.searchsorted(to, t_end, side="right")) - 1
    f_end = at(k_end, t_end, "end")
    r.update(outcome=sig.get("outcome") or pl.get("outcome"), cup_in_holder=sig.get("cup_in_holder"), hand_empty=sig.get("hand_empty"),
             released_by_force=last.get("released_by_force"), end_tilt_fpp_deg=None if f_end is None else f_end[3],
             cup_src=pl.get("cup_src"), end_event=(w["end"] or {}).get("event"))
    return r


def episode(ep_dir: Path, bag: Path | None, roots: list[Path]) -> dict:
    evs = RB.events(ep_dir)
    bag = bag or RB.find_bag(ep_dir, roots)
    meta, ep_yaml, live = RB.episode_ctx(ep_dir)
    flow = RB.runner_flow(bag)
    wins = RB.node_windows(evs, lambda e: str(e.get("policy_name") or "").startswith("rh_place"))
    return {"episode_dir": str(ep_dir), "bag": str(bag), "episode": meta.get("episode"),
            "nodes": [node_report(w, bag, flow, live, ep_yaml) for w in wins]}


ROWS = [("정책", "policy"), ("ckpt md5(계약)", "ckpt_md5"), ("래치 방식", "latch_mode"),
        ("rc carry_ok", ("ready_carry", "carry_ok")), ("rc start_ok", ("ready_carry", "start_ok")),
        ("rc err_mm", ("ready_carry", "err_mm")), ("rc axis_deg", ("ready_carry", "axis_deg")), ("rc 틱", ("ready_carry", "t")),
        ("T_in 출처", "tin_source"), ("T_in palm z m", "tin_z"), ("뱅크 palm 컵 z 중앙 m", "bank_palm_cup_z"),
        ("T_in z − 뱅크 m", "tin_z_minus_bank"), ("시작 컵−자리 dz m", "start_dz"), ("뱅크 범위 밖 칸(78, 뱅크 상수 칸 제외)", "n_out"),
        ("|z|>3 칸(78, 뱅크 상수 칸 제외)", "n_z3"), ("뱅크 N", "bank_n"), ("t15 틱", "t15"), ("20 틱 열림 rad", "open20"),
        ("래치 틱", "latch_tick"), ("래치 입력 계속 0", "latch_all0"), ("래치 때 dz(actor 컵) m", "latch_dz"), ("래치 때 dz(래치 컵, status) m", "latch_dz_status"), ("정책 틱 수", "policy_ticks"),
        ("손 편 틱(실측)", "t_rel"), ("손 편 때 컵 높이 FK mm", "rel_h_fk_mm"), ("손 편 때 컵 높이 FP++ mm", "rel_h_fpp_mm"),
        ("손 편 때 FP++ 나이 s", "rel_fpp_age_s"), ("outcome", "outcome"), ("cup_in_holder", "cup_in_holder"),
        ("hand_empty", "hand_empty"), ("released_by_force", "released_by_force"), ("끝 컵 기울기 FP++ °", "end_tilt_fpp_deg"),
        ("cup_src", "cup_src"), ("시작 FP++ 나이 s", "start_fpp_age_s"), ("FP++−FK 시작 mm", "start_fpp_fk_mm"), ("FP++−FK 시작 dz mm", "start_fpp_fk_dz_mm"),
        ("FP++−FK 손 편 mm", "rel_fpp_fk_mm"), ("끝 FP++ 나이 s", "end_fpp_age_s"), ("FP++−FK 끝 mm", "end_fpp_fk_mm"), ("FP++−FK 끝 dz mm", "end_fpp_fk_dz_mm"),
        ("|z| 큰 칸(뱅크 상수 칸 제외)", "top_z"), ("뱅크 상수 칸 수(std≈0, z 안 씀)", "n_bank_const"),
        ("뱅크 상수 칸: 이름 · |실기−뱅크값| 최대", "bank_const")]


def cell(r: dict, key):
    if isinstance(key, tuple):
        return (r.get(key[0]) or {}).get(key[1])
    return r.get(key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--episode-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--bag-dir", type=Path, default=None, help="에피소드 하나일 때 bag 폴더를 직접")
    ap.add_argument("--bag-root", type=Path, action="append", default=None, help="bag 을 찾을 곳(여럿) — 기본 logs/runs/<날짜>/bags · DATA 백업")
    a = ap.parse_args(argv)
    if a.bag_dir and len(a.episode_dir) > 1:
        ap.error("--bag-dir 는 에피소드 하나에만")
    cols, recs = [], []
    for ep_dir in a.episode_dir:
        ep_dir = ep_dir.resolve()
        rep = episode(ep_dir, a.bag_dir, a.bag_root or RB.default_roots(ep_dir))
        out = RB.out_path("place_compare", ep_dir)
        out.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"[place] {ep_dir.name} · bag {Path(rep['bag']).name} → {out.relative_to(RB.SIM2REAL)}")
        for r in rep["nodes"]:
            cols.append(f"{ep_dir.parent.parent.name}/{ep_dir.name.split('_')[0]} {r['node']}")
            recs.append(r)
    print()
    print(RB.md_table(cols, [(name, [cell(r, k) for r in recs]) for name, k in ROWS]))
    for col, r in zip(cols, recs):
        for n in r["notes"]:
            print(f"- {col}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
