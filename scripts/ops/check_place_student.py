#!/usr/bin/env python3
"""놓기 학생 실기 시작 전 검사 — 읽기 전용, ROS · GPU 를 띄우지 않는다(11.08). 형식은 check_student_mask_A 와 같다.

    .venv-student/bin/python scripts/ops/check_place_student.py --student-dir <student.yaml 폴더> [--side right]
        [--contract <rh_place_contract.json>] [--mission config/mission_rh56f1_control.yaml] [--target-class blue]

ckpt md5 · hdgp 스냅숏 커밋 · 외부 파라미터(r20u5) md5 · place_student.check_meta · 목 동역학(meta head ↔ 미션 목 드라이버)
· 미션 --max-tilt ≥ 82 · 홀더 CAD · 홀더 자세 · YOLO md5 와 /head_mask/config · target_class ↔ ckpt one_obj · head_sag md5 · 64 MB SHM.
불일치가 하나라도 있으면 종료 코드 1. 확인불가 · 경고는 실패로 치지 않는다.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "policy_control"))
sys.path.insert(0, str(ROOT / "scripts"))
import head_motion as hm  # noqa: E402
from policy_control import hdgp_pin  # noqa: E402
from policy_control import place_head_pose as HP  # noqa: E402
from policy_control import place_holder_proj as HPJ  # noqa: E402
from policy_control import place_student as PS  # noqa: E402
from policy_control import rh_place as P  # noqa: E402
from policy_control import student_mask as SM  # noqa: E402
from policy_control.hdgp_pin import md5  # noqa: E402
from policy_control.student_policy import SNAPSHOT_CFG  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_student_mask_A", ROOT / "scripts/ops/check_student_mask_A.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)               # 잡기 검사: check_shm · mask_node_config · HD(head_depth_node) 를 다시 쓴다
OK, BAD, NA, WARN = G.OK, G.BAD, G.NA, "경고"

CONTRACTS = {s: ROOT / f"deploy/policies/rh56f1/place/{d}/rh_place_contract.json"
             for s, d in (("right", "right_n10_e4000"), ("left", "left_n09_e3800"))}
MISSION = ROOT / "config/mission_rh56f1_control.yaml"
SAG_MD5 = "91ef9d699334d02340277f896264cbb4"
TILT_NEED = 82.0                           # hm.STREAM_TILT_RANGE_DEG 상한 = 학생 tilt 한계
HOLDER_POSES = ROOT / "config/cup_holder_poses_5090_v1.yaml"
META_GROUPS = (("arch", "'arch'"), ("prop_cols", "'prop_cols'"), ("mask", "'mask."), ("head.rule", "'head.rule'"),
               ("floor_cut_z_m", "'depth_pre.floor_cut_z_m'"))


def head_drivers(mission: Path) -> list[tuple[str, list[str]]]:
    """미션 yaml 안 head_joint_publisher.py argv 전부 → [(위치, argv)]."""
    import yaml

    out = []

    def walk(x, where):
        if isinstance(x, dict):
            argv = x.get("argv")
            if isinstance(argv, list) and any("head_joint_publisher.py" in str(a) for a in argv):
                out.append((where, [str(a) for a in argv]))
            for k, v in x.items():
                walk(v, f"{where}.{k}" if where else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{where}[{i}]")

    walk(yaml.safe_load(mission.read_text(encoding="utf-8")), "")
    return out


def _arg(argv: list[str], flag: str, default: float) -> float:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return float(argv[i + 1])
        if a.startswith(flag + "="):
            return float(a.split("=", 1)[1])
    return default


def check_head(meta_head: dict, mission: Path, row) -> None:
    drivers = head_drivers(mission)
    if not drivers:
        row("미션 목 드라이버", "head_joint_publisher argv", f"{mission.name} 에 없음", NA)
    for where, argv in drivers:
        vel = _arg(argv, "--stream-max-vel", hm.STREAM_MAX_VEL_DPS)
        acc = _arg(argv, "--stream-max-acc", hm.STREAM_MAX_ACC_DPS2)
        tilt = _arg(argv, "--max-tilt", hm.MAX_TILT_DEG)
        for name, real, dr_key, ev_key in (("속도 °/s", vel, "rate_dps", "rate_eval_dps"),
                                           ("가속 °/s²", acc, "accel_dps2", "accel_eval_dps2")):
            dr, ev = meta_head.get(dr_key), meta_head.get(ev_key)
            if dr is None:
                row(f"목 {name} DR {where}", f"meta head.{dr_key}", f"키 없음 · 실기 {real:g}", NA)
            else:
                row(f"목 {name} DR {where}", f"{dr_key} {list(dr)} 안", f"실기 {real:g}", OK if dr[0] <= real <= dr[1] else BAD)
            if ev is None:
                row(f"목 {name} eval {where}", f"meta head.{ev_key}", f"키 없음 · 실기 {real:g}", NA)
            else:
                row(f"목 {name} eval {where}", f"{ev_key} {ev:g}", f"실기 {real:g}", OK if float(ev) == real else WARN)
        row(f"미션 --max-tilt {where}", f"≥ {TILT_NEED:g}", f"{tilt:g}" + ("" if "--max-tilt" in " ".join(argv) else "(기본값)"),
            OK if tilt >= TILT_NEED else BAD)


def check_bundle(d: Path, c, mission: Path, target_class: str, rows: list) -> None:
    import torch
    import yaml

    side = c.side().side

    def row(item, want, got, verdict):
        rows.append((side, item, str(want), str(got), verdict))

    card = yaml.safe_load((d / "student.yaml").read_text(encoding="utf-8")) or {}
    ck, ex, yo = card["ckpt"], card["extrinsics"], (card.get("mask") or {}).get("yolo") or {}
    # 1. ckpt · hdgp 핀 · 외부 파라미터
    got = md5(d / ck["path"])
    row("ckpt md5", ck["md5"], got, OK if got == ck["md5"] else BAD)
    snap = hdgp_pin.load_snapshot_cfg(SNAPSHOT_CFG)["commit"]
    commit = str((card.get("hdgp") or {}).get("commit"))
    row("hdgp 스냅숏 커밋", snap, commit, OK if commit == snap else BAD)
    got = md5(d / ex["path"])
    row("외부 파라미터 md5 = student.yaml", ex["md5"], got, OK if got == ex["md5"] else BAD)
    row("외부 파라미터 md5 = r20u5 학습 값", G.HD.EXTRINSICS_MD5, got, OK if got == G.HD.EXTRINSICS_MD5 else BAD)
    # 2. check_meta(노드가 시작할 때 부르는 그대로)
    meta = torch.load(d / ck["path"], map_location="cpu", weights_only=True)
    z_cut = PS.TABLE_Z - float(G.HD.real_depth.FLOOR_MARGIN_M)
    try:
        PS.check_meta(meta, c, ex["md5"], z_cut)
        err = ""
    except ValueError as exc:
        err = str(exc)
    row("check_meta 전체", "통과", err or "통과", BAD if err else OK)
    for name, key in META_GROUPS:
        row(f"check_meta {name}", "통과", "어긋남" if key in err else "통과", BAD if key in err else OK)
    # 3 · 4. 목 동역학 · tilt 상한
    check_head(meta.get("head") if isinstance(meta.get("head"), dict) else {}, mission, row)
    # 5. 홀더 CAD · 자세
    for p in (HPJ.HOLDER_MESH, HOLDER_POSES):
        row(f"{p.relative_to(ROOT)}", "있음", md5(p) if p.is_file() else "없음", OK if p.is_file() else BAD)
    # 6. YOLO · 마스크 노드 설정 · 목표 클래스
    ref = G.HM.DEFAULT_WEIGHTS
    if not yo:
        row("student.yaml mask.yolo", "있음", "없음", BAD)
    else:
        got = md5(d / yo["path"])
        row("YOLO md5 = student.yaml", yo["md5"], got, OK if got == yo["md5"] else BAD)
        row("YOLO md5 = seg_s640_A", md5(ref) if ref.is_file() else f"{ref} 없음", got,
            NA if not ref.is_file() else (OK if got == md5(ref) else BAD))
        try:
            SM.check_mask_config(G.mask_node_config(d), yo, meta.get("depth_pre") or {})
            row("/head_mask/config 대조", "통과", "통과", OK)
        except Exception as exc:  # noqa: BLE001 — 어떤 검사가 막았든 표에 이유를 남긴다
            row("/head_mask/config 대조", "통과", f"{type(exc).__name__}: {exc}", BAD)
    tgt = target_class or str((card.get("mask") or {}).get("target_class") or "")
    one = meta.get("one_obj")
    if one is None:
        row("target_class ↔ ckpt one_obj", SM.OBJ_TARGETS, f"{tgt} · one_obj 키 없음", NA)
    else:
        try:
            warn = SM.check_target(one, tgt)
            row("target_class ↔ ckpt one_obj", f"{one} → {SM.OBJ_TARGETS.get(one)}", tgt, WARN if warn else OK)
        except ValueError as exc:
            row("target_class ↔ ckpt one_obj", f"{one} → {SM.OBJ_TARGETS.get(one)}", f"{tgt}: {exc}", BAD)
    # 7. head_sag 표
    got = md5(HP.SAG_TABLE) if HP.SAG_TABLE.is_file() else "없음"
    row(f"{HP.SAG_TABLE.relative_to(ROOT)}", SAG_MD5, got, OK if got == SAG_MD5 == HP.SAG_TABLE_MD5 else BAD)


def run(argv: list[str] | None = None) -> list[tuple]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student-dir", required=True, help="놓기 학생 묶음(student.yaml 폴더)")
    ap.add_argument("--side", choices=sorted(CONTRACTS), default="right")
    ap.add_argument("--contract", default=None, help="rh_place_contract.json(기본 = 손별 배포 놓기 계약)")
    ap.add_argument("--mission", default=str(MISSION))
    ap.add_argument("--target-class", default="", help="빈 값 = student.yaml mask.target_class")
    args = ap.parse_args(argv)
    c = P.load_contract(Path(args.contract or CONTRACTS[args.side]))
    if c.side().side != args.side:
        raise SystemExit(f"계약 손 {c.side().side} ≠ --side {args.side}")
    rows: list[tuple] = []
    check_bundle(Path(args.student_dir).expanduser().resolve(), c, Path(args.mission), args.target_class, rows)
    G.check_shm(rows)                        # 8. 64 MB SHM
    return rows


def main(argv: list[str] | None = None) -> int:
    rows = run(argv)
    w = max(len(r[1]) for r in rows)
    for r in rows:
        print(f"{r[4]:<5} {r[0]:<5} {r[1]:<{w}}  기대 {r[2]}  ·  실제 {r[3]}")
    bad = sum(r[4] == BAD for r in rows)
    print(f"\n불일치 {bad} · 확인불가 {sum(r[4] == NA for r in rows)} · 경고 {sum(r[4] == WARN for r in rows)}"
          f" · OK {sum(r[4] == OK for r in rows)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
