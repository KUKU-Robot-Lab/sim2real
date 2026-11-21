#!/usr/bin/env python3
"""자세 유지 측정(중력 · 마찰 동정) — tilt_comp_ab --pose 의 자세 yaml · 창 궤적 · 단계 목록.

    python3 deploy/policy_control/tools/tilt_comp_ab.py --pose deploy/policy_control/config/pd_comp_ab/pose_hold.yaml --grip open
    (계획만, --execute 로 실행. 브링업 · lease · 조건 yaml · teardown 은 tilt_comp_ab 그대로)

자세마다 대상 관절을 목표 ±offset 에서 출발해 목표로 접근(move_s) → 유지(hold_s), reps 회(d = +1, −1 교대). 창 하나 = 재생 하나
(replay_to_pd, 두 팔 14 관절, --max-start-gap). 분석기는 창 끝 hold_s 의 실측 − applied 를 창별 · d 별로 낸다.
손: open = 자세에 닿은 뒤 계약 home_hand 로 펴고 떠나기 전 되돌린다(이동은 늘 pd hand_path_pose 주먹 — 홈 경로 · tilt 를 검사한 손).
    closed · object = 홈에서 한 번 grasp 정책 hand_grip 으로 닫고(object 는 편 뒤 Enter 대기) 끝에 홈에서 푼다.
    ★11.18 사용자 "손 home 값은 편 손": 홈에 서 있는 동안은 계약 home_hand(편 손)로 끝낸다 — 주먹은 이동 중에만.
    푸는 자리가 홈이라 이동 검사(--hand-start pd 주먹)는 그대로다. 홈 편 손 여유 = 테이블 윗면 198 mm.
    held = ★11.18 사용자 "현재 상태에서 그냥 pd 들만 바꿔서 바로 진행": 이미 쥔 채 홈에 서 있다고 보고 손 단계를 하나도 내지
    않는다(차렷 · 주먹 · 쥐기 · 놓기 없음, 놓기는 사람이). 사이클 = engage → 홈 끝점 확인 → goto_home(pd yaml home_hand keep —
    손 지령 없음) → 자세 측정 → 홈. 자세 · 검사는 object 와 같다(grip_poses · 쥔 손 hand_grip 으로 충돌 검사).

이 파일 단독 실행 = 준비: 자세 yaml → <out>/*.npz · windows.json 과 plan_home_path --check-only 충돌 검사(실기에 아무것도 안 보냄).
tilt_comp_ab --execute 가 브링업 전에 <실험 폴더>/pose 로 부른다 — 검사 하나라도 실패면 시작 전 거부.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

TOOLS = Path(__file__).resolve().parent
SIM2REAL = TOOLS.parents[2]
GRIPS = ("open", "closed", "object", "held")
FROMS = ("home", "tilt_start", "max_tilt")
DT = 1 / 60                     # 기록 주기 — tilt_cmd.npz 와 같게
OUT_S, PAUSE_S = 2.0, 0.5        # 목표 → 오프셋 지점 이동 · 그 자리 멈춤(접근 전)
HAND_S, HAND_HOLD_S = 2.0, 1.0   # 손 바꾸기 · 그 뒤 붙들기
REF_TOL_DEG = 0.05               # yaml 의 home · tilt 숫자 ↔ 기준(계약 · --traj) 허용
#: 쥐는 손 모양 = 배포 잡기 정책 rcg04 계약 hand_grip(정책 손 행동의 쥔 끝, 두 손 같은 값 [1.2, 0.24, 1.08, 1.08, 0.85, 0.85])
GRIP_CONTRACT = SIM2REAL / "deploy/policies/rh56f1/aglt/{side}_rcg04/rh_aglt_contract.json"


def load_pose(path: Path) -> dict:
    """자세 yaml 읽기 · 검사. 잘못되면 ValueError(움직이기 전)."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    spec = {k: d.get(k, v) for k, v in (("offset_deg", 3.0), ("move_s", 4.0), ("hold_s", 3.0), ("reps", 2),
                                          ("joints", [5, 6, 7]), ("transit_rad_s", 0.3), ("grip_poses", list(FROMS)))}
    if not (0 < spec["offset_deg"] <= 10 and 1 <= spec["move_s"] <= 20 and 1 <= spec["hold_s"] <= 20
            and isinstance(spec["reps"], int) and 1 <= spec["reps"] <= 10 and 0 < spec["transit_rad_s"] <= 0.5):
        raise ValueError(f"{path}: offset_deg (0,10] · move_s · hold_s [1,20] · reps 1~10 · transit_rad_s (0,0.5] 밖")
    poses, names = [], set()
    for p in d.get("poses") or []:
        n, fr, jn = str(p.get("name", "")), p.get("from", "home"), p.get("joints", spec["joints"])
        arms = [p.get("right"), p.get("left")] + [x.get(s) for x in p.get("via") or [] for s in ("right", "left")]
        why = ("이름은 영문 · 숫자 · _" if not re.fullmatch(r"\w+", n) else "이름 중복" if n in names else
               f"from 은 {FROMS}" if fr not in FROMS else
               "joints 는 1~7 목록" if not jn or not all(isinstance(j, int) and 1 <= j <= 7 for j in jn) else
               "right · left(· via) 는 숫자 7 개" if not all(isinstance(a, list) and len(a) == 7 for a in arms) else
               "max_tilt 는 frame(정수)" if fr == "max_tilt" and not isinstance(p.get("frame"), int) else
               "via 는 from: home 에만" if p.get("via") and fr != "home" else
               "open_sides 는 right · left 목록" if not set(p.get("open_sides", ["right"])) <= {"right", "left"} else "")
        if why:
            raise ValueError(f"{path}: 자세 {n or '?'}: {why}")
        names.add(n)
        poses.append({"name": n, "from": fr, "joints": list(jn), "frame": p.get("frame"),
                      "open_sides": list(p.get("open_sides", ["right", "left"])),
                      "q": np.radians(p["right"] + p["left"]),
                      "via": [np.radians(v["right"] + v["left"]) for v in p.get("via") or []]})
    if not poses or len([p for p in poses if p["from"] != "home"]) != len({p["from"] for p in poses if p["from"] != "home"}):
        raise ValueError(f"{path}: poses 가 비었거나 tilt_start · max_tilt 가 둘 이상")
    if set(spec["grip_poses"]) - names:
        raise ValueError(f"{path}: grip_poses 에 없는 자세 {sorted(set(spec['grip_poses']) - names)}")
    return {**spec, "path": str(path), "poses": poses}


def selected(spec: dict, grip: str) -> list[dict]:
    """이번 단계에서 재는 자세 — open 은 전부, closed · object 는 grip_poses 만(yaml 순서)."""
    return [p for p in spec["poses"] if grip == "open" or p["name"] in spec["grip_poses"]]


def refs(spec: dict, contract: dict, traj: Path) -> dict:
    """기준 자세(rad, 14) — home = 계약 home_arm, tilt_start = traj 첫 프레임, max_tilt = traj[frame]. yaml 숫자와 어긋나면 ValueError."""
    home = np.asarray(contract["sides"]["right"]["home_arm"] + contract["sides"]["left"]["home_arm"], float)
    q = np.load(traj)["arm_target"]
    out = {}
    for p in spec["poses"]:
        ref = (home if p["name"] == "home" else q[0] if p["from"] == "tilt_start" else
               q[p["frame"]] if p["from"] == "max_tilt" else None)
        if ref is None:
            continue
        gap = float(np.degrees(np.abs(ref - p["q"]).max()))
        if gap > REF_TOL_DEG:
            raise ValueError(f"자세 {p['name']}: yaml 숫자가 기준과 {gap:.3f}° 다르다(허용 {REF_TOL_DEG}°) — yaml 을 고칠 것")
        out[p["name"]] = ref
    out["_home"] = home
    return out


def cos_seg(a: np.ndarray, b: np.ndarray, sec: float) -> np.ndarray:
    """a → b 코사인 속도(끝 속도 0), 끝 포함 · 처음 뺌."""
    n = max(1, round(sec / DT))
    s = (1 - np.cos(np.pi * np.arange(1, n + 1) / n)) / 2
    return a + s[:, None] * (b - a)


def transit_s(a: np.ndarray, b: np.ndarray, rad_s: float) -> float:
    return max(1.0, float(np.abs(b - a).max()) * np.pi / 2 / rad_s)   # 코사인 최대 속도 = 평균 × π/2


def window(q: np.ndarray, idx: list[int], d: int, spec: dict) -> np.ndarray:
    """목표 q → q + d·off(대상 관절, 두 팔) → 멈춤 → q 로 접근(move_s) → 유지(hold_s)."""
    off = np.zeros(14)
    off[[j - 1 for j in idx] + [j + 6 for j in idx]] = d * np.radians(spec["offset_deg"])
    pts = [q[None], cos_seg(q, q + off, OUT_S), np.repeat([q + off], round(PAUSE_S / DT), 0),
           cos_seg(q + off, q, spec["move_s"]), np.repeat([q], round(spec["hold_s"] / DT), 0)]
    return np.vstack(pts)


def save(path: Path, frames: np.ndarray) -> float:
    np.savez(path, arm_target=frames.astype(np.float64), meta_step_dt=np.float64(DT))
    return len(frames) * DT


def hand_vals(contract: dict, pd_yaml: dict, side: str) -> dict:
    sd = contract["sides"][side]
    g = json.loads(Path(str(GRIP_CONTRACT).format(side=side)).read_text(encoding="utf-8"))["sides"]["arm"]
    names = list(sd["home_hand"])
    if g["hand_joints"] != names:
        raise ValueError(f"{side}: 잡기 정책 손 관절 {g['hand_joints']} ≠ 계약 {names}")
    fist = pd_yaml["hand_path_pose"][side]
    return {"names": names, "open": [sd["home_hand"][n] for n in names], "grip": list(g["hand_grip"]),
            "fist": [fist[n] for n in names]}


SIDE_KO = {"right": "오른", "left": "왼"}


def hand_plan(grip: str, sides=("right", "left")) -> tuple[list, list]:
    """홈에서 손 바꾸기 차례(오른 → 왼, 한 손씩). 항목 = (npz 이름, 손 | None(둘 다), 바꿀 모양, Enter 문구 | None). 시작 · 끝."""
    start, end = [], []
    for s in sides:
        if grip == "object":
            start += [(f"hand_open_{s}", s, "open", None), (f"hand_grip_{s}", s, "grip", f"{SIDE_KO[s]}손에 쉐이커를 넣고")]
            end.append((f"hand_release_{s}", s, "open", f"{SIDE_KO[s]}손 쉐이커를 받을 준비가 되면"))
        elif grip == "closed":
            start.append((f"hand_grip_{s}", s, "grip", None))
            end.append((f"hand_release_{s}", s, "open", None))
    if grip == "object":
        #: ★11.18 사용자 "손 home 값은 편 손": 홈에 서 있는 동안의 손 = 계약 home_hand(편 손). 주먹(pd hand_path_pose)은
        #: 이동 중에만 쓴다. 홈 편 손 여유는 테이블 윗면(table_8) 198 mm — 차렷 편 손이 밑판을 6.5 mm 파고드는 것(10.01
        #: 기록 7.5 mm)과 달리 홈에서는 닿지 않는다(URDF v1 FK, 같은 식으로 차렷 −6.5 mm 재현).
        end.append(("hand_home_open", None, "open", None))
    return start, end


def hand_moves(H: dict, grip: str, sides=("right", "left")) -> list:
    """hand_plan 차례대로 (npz 이름, 손 12 전, 후). 바꾸지 않는 손은 직전 모양을 유지한다."""
    cur, out = np.asarray(H["fist"], float), []
    start, end = hand_plan(grip, sides)
    for name, side, shape, _ in start + end:
        nxt = cur.copy()
        for s in [side] if side else sides:
            i = 6 * ("right", "left").index(s)
            nxt[i: i + 6] = np.asarray(H[shape], float)[i: i + 6]
        out.append((name, cur, nxt))
        cur = nxt
    return out


def make(spec: dict, grip: str, out: Path, traj: Path, contract_path: Path, pd_path: Path) -> dict:
    """npz 와 windows.json 을 out 에 쓴다. 반환 = windows.json 내용."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    pd_yaml = yaml.safe_load(pd_path.read_text(encoding="utf-8"))
    ref = refs(spec, contract, traj)
    hands = {s: hand_vals(contract, pd_yaml, s) for s in ("right", "left")}
    H = {k: np.asarray(hands["right"][k] + hands["left"][k], float) for k in ("open", "grip", "fist")}
    out.mkdir(parents=True, exist_ok=True)
    meta = {"grip": grip, "spec": spec["path"], "hand_joints": hands["right"]["names"] + hands["left"]["names"],
            "hand": {k: v.tolist() for k, v in H.items()}, "poses": [], "windows": [], "checks": []}
    home = ref["_home"]

    def hand_npz(name: str, q: np.ndarray, h0: np.ndarray, h1: np.ndarray) -> float:
        arm = np.repeat([q], round((HAND_S + HAND_HOLD_S) / DT) + 1, 0)
        hand = np.vstack([h0[None], cos_seg(h0, h1, HAND_S), np.repeat([h1], round(HAND_HOLD_S / DT), 0)])
        return save(out / f"{name}.npz", np.hstack([arm, hand]))

    for name, h0, h1 in hand_moves(H, grip):
        meta[f"{name}_s"] = hand_npz(name, home, h0, h1)
    tq = np.load(traj)["arm_target"]
    for p in selected(spec, grip):
        q = ref.get(p["name"], p["q"])
        rec = {"name": p["name"], "from": p["from"], "joints": p["joints"], "q_deg": np.degrees(q).round(3).tolist(),
               "open_sides": p["open_sides"] if grip == "open" else []}
        path = [home]
        if p["from"] == "home" and np.abs(q - home).max() > 1e-6:
            segs = [cos_seg(a, b, transit_s(a, b, spec["transit_rad_s"])) for a, b in zip([home, *p["via"]], [*p["via"], q])]
            path = [home[None], *segs]
            rec["transit_s"] = save(out / f"to_{p['name']}.npz", np.vstack(path))
        if p["from"] == "max_tilt":
            rec["to_max_s"] = save(out / "tilt_to_max.npz", tq[: p["frame"] + 1])
            rec["from_max_s"] = save(out / "tilt_from_max.npz", tq[p["frame"]:])
        if grip == "open":
            ho = np.concatenate([H["open" if s in p["open_sides"] else "fist"][6 * i: 6 * i + 6] for i, s in enumerate(("right", "left"))])
            rec["hand_s"] = hand_npz(f"hand_{p['name']}", q, H["fist"], ho)
        wins = []
        for r in range(1, spec["reps"] + 1):
            for d in (1, -1):
                k = len(wins) + 1
                f = window(q, p["joints"], d, spec)
                wins.append(f)
                meta["windows"].append({"id": f"win_{p['name']}_{k}", "pose": p["name"], "d": d, "rep": r, "grip": grip,
                                        "joints": p["joints"], "offset_deg": spec["offset_deg"], "hold_s": spec["hold_s"],
                                        "q_deg": rec["q_deg"], "dur_s": save(out / f"win_{p['name']}_{k}.npz", f)})
        rec["check"] = np.vstack([*(path[1:] if len(path) > 1 else []), *wins])
        meta["poses"].append(rec)
    return meta


def check_cmds(meta: dict, out: Path, grip: str) -> list[tuple[str, list[str]]]:
    """plan_home_path --check-only 명령(팔마다 · 손마다). 반대팔은 홈과 그 자세 둘 다(두 팔이 같이 움직인다 — 끝점만 본다)."""
    from plan_rehome import HOLDERS, ROBOTS
    cmds = []
    names = meta["hand_joints"]
    for rec in meta["poses"]:
        frames = rec.pop("check")
        for i, side in enumerate(("right", "left")):
            npz = out / f"check_{rec['name']}_{side}.npz"
            save(npz, frames[:, 7 * i: 7 * i + 7])
            other = "left" if side == "right" else "right"
            oq = ",".join(f"{other[0]}_aj_{j + 1}={np.radians(v):.5f}" for j, v in enumerate(rec["q_deg"][7 * (1 - i): 7 * (1 - i) + 7]))
            hq = {"pd": ["--hand-start", "pd"], "open": ["--hand-start", "contract"],
                  "grip": ["--hand-start", "measured", "--hand-q",
                           ",".join(f"{n}={v}" for n, v in zip(names[6 * i: 6 * i + 6], meta["hand"]["grip"][6 * i: 6 * i + 6]))]}
            for h in (("pd", "open") if side in rec["open_sides"] else ("pd",)) if grip == "open" else ("grip",):
                cmds.append((f"{rec['name']}/{side}/{h}",
                             [sys.executable, str(TOOLS / "plan_home_path.py"), *ROBOTS["rh56f1"], "--holders", str(HOLDERS),
                              "--side", side, "--other-arm", "home", "--other-q", oq, *hq[h], "--check-only", "--npz", str(npz)]))
    return cmds


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(TOOLS))
    from tilt_comp_ab import CONTRACT, PD_DEFAULT, TRAJ
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--grip", choices=GRIPS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--traj", type=Path, default=TRAJ)
    ap.add_argument("--fake", action="store_true", help="손 이동 손 모양을 fake pd yaml 에서")
    ap.add_argument("--no-check", action="store_true", help="충돌 검사 생략(시험용)")
    a = ap.parse_args(argv)
    try:
        spec = load_pose(a.pose)
        meta = make(spec, a.grip, a.out, a.traj, CONTRACT, PD_DEFAULT["fake" if a.fake else "real"])
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"✗ pose_hold: {exc}")
        return 1
    cmds = check_cmds(meta, a.out, a.grip)
    bad = []
    for name, cmd in ([] if a.no_check else cmds):
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = [ln for ln in (r.stdout + r.stderr).splitlines() if "[check]" in ln or "✗" in ln][-1:] or ["(출력 없음)"]
        meta["checks"].append({"name": name, "rc": r.returncode, "line": tail[0]})
        print(f"[pose_hold] 검사 {name}: {'통과' if r.returncode == 0 else '★실패'} {tail[0]}", flush=True)
        if r.returncode:
            bad.append(name)
    (a.out / "windows.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[pose_hold] {a.out}: 자세 {len(meta['poses'])} · 창 {len(meta['windows'])} · 검사 {len(meta['checks'])}"
          + (f" ★실패 {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


def pose_steps(ab, spec: dict, grip: str, cond: dict, cdir: Path, traj: Path, rate: float, fake: bool) -> list:
    """한 사이클 단계(tilt_comp_ab.Step). 홈 · tilt 이송 · 되돌림 · 쉬기 단계는 cycle_steps 의 것을 그대로 잘라 쓴다."""
    S, R, L = ab.Step, ab.ARM_JOINTS["right"], ab.ARM_JOINTS["left"]
    base = ab.cycle_steps(cond, cdir, traj, rate, fake)
    at = {s.id: i for i, s in enumerate(base)}
    pdir, arm = cdir.parent / "pose", R + L    # 준비(main)가 실험 폴더에 한 번 만든다 — 조건이 바뀌어도 같다
    x = tuple(cond.get("replay_args", ())) if "pose" in cond.get("replay_on", ab.SEGMENTS) else ()
    hands = [f"{s}_hj_{n}" for s in "rl" for n in ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")]

    def rp(sid: str, name: str, est: float, *extra: str, joints=arm, gap: bool = True):
        g = ("--max-start-gap", repr(ab.START_GAP)) if gap else ()
        return S(sid, ab.replay(pdir / f"{name}.npz", joints, 1.0, *g, *extra, *x), est_s=est)

    hand_s = HAND_S + HAND_HOLD_S
    win_s = OUT_S + PAUSE_S + spec["move_s"] + spec["hold_s"]
    if grip == "held":   # 손 단계 없이 홈에서 — 끝점 0.3 rad 가드(★11.18 pd 교체 때 쉐이커 든 오른 j7 release 처짐 9°로 0.15 초과)(pd 교체 때 release 처짐) 뒤 직선 정착
        home_chk = {s: S(f"check_at_home_{s}", (ab.PY, str(ab.TOOLS / "check_path_start.py"), "--npz", str(ab.HOME_NPZ[s]),
                                                "--contract", str(ab.CONTRACT), "--at", "end", "--tol", "0.3")) for s in ab.SIDES}
        st = [base[at[f"engage_{s}"]] for s in ab.SIDES] + [x for s in ab.SIDES for x in (home_chk[s], base[at[f"settle_home_{s}"]])]
        st.append(base[at["still_home"]])
    else:
        st = base[: at["still_home"] + 1]
    def hand_steps(plan: list) -> list:   # Enter 대기 중 Ctrl-C = 기존 중단 경로(손 상태 유지, drivers 남김)
        out = []
        for name, _, _, enter in plan:
            out += [S(f"enter_{name}", (enter,), kind="enter")] if enter else []
            out += [rp(name, name, hand_s, joints=arm + hands, gap=False), S(f"still_{name}", kind="still")]
        return out

    hand_start, hand_end = hand_plan(grip)
    st += hand_steps(hand_start)
    sel = selected(spec, grip)
    home = next((p["q"] for p in spec["poses"] if p["name"] == "home"), None)
    est = {p["name"]: sum(transit_s(a, b, spec["transit_rad_s"]) for a, b in zip([home, *p["via"]], [*p["via"], p["q"]]))
           if home is not None and p["from"] == "home" else 0.0 for p in sel}
    for i, p in enumerate(sel):
        n, k = p["name"], 2 * spec["reps"]
        prev = sel[i - 1]["from"] if i else None
        nxt = sel[i + 1]["from"] if i + 1 < len(sel) else None
        if p["from"] != "home" and not (p["from"] == "max_tilt" and prev == "tilt_start"):   # tilt 시작으로 이송
            st += base[at["plan_tilt_start_right"]: at["pre_tilt_hold"]] + [S("still_tilt_start", kind="still")]
        if p["from"] == "max_tilt":
            st += [S("tilt_to_max", ab.replay(pdir / "tilt_to_max.npz", arm, rate, "--max-start-gap", repr(ab.START_GAP), *x),
                     est_s=p["frame"] * DT / rate), S("still_max", kind="still")]
        transit = p["from"] == "home" and n != "home"
        if transit:
            st += [rp(f"to_{n}", f"to_{n}", est[n]), S(f"still_{n}", kind="still")]
        if grip == "open":
            st.append(rp(f"hand_{n}", f"hand_{n}", hand_s, joints=arm + hands, gap=False))
        st += [rp(f"win_{n}_{j}", f"win_{n}_{j}", win_s) for j in range(1, k + 1)]
        if grip == "open":
            st.append(rp(f"hand_{n}_back", f"hand_{n}", hand_s, "--reverse", joints=arm + hands, gap=False))
        if transit:
            st += [rp(f"back_{n}", f"to_{n}", est[n], "--reverse"), S(f"still_{n}_back", kind="still")]
        if p["from"] == "max_tilt":
            st += [S("tilt_from_max", ab.replay(pdir / "tilt_from_max.npz", arm, rate, "--max-start-gap", repr(ab.START_GAP), *x),
                     est_s=(ab.npz_seconds(traj, 1.0) - p["frame"] * DT) / rate), S("still_after_tilt", kind="still")]
        if p["from"] != "home" and not (p["from"] == "tilt_start" and nxt == "max_tilt"):   # 홈으로 되돌림
            st += base[at["back_left"]: at["check_rest_right"]]
    if grip == "held":   # 홈에서 끝 — 차렷으로 내려가지 않는다(쥔 물체를 몸 쪽으로 가져가지 않게), 다음 사이클도 홈에서
        return st
    return st + hand_steps(hand_end) + base[at["check_rest_right"]:]


if __name__ == "__main__":
    raise SystemExit(main())
