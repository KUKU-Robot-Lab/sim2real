#!/usr/bin/env python3
"""목표 마스크 학생 A(1105mA) 실기 평가 시작 전 검사 — 읽기 전용, ROS · GPU 를 띄우지 않는다(11.07).

    .venv-student/bin/python scripts/ops/check_student_mask_A.py [--sides right left]

손별: ckpt md5 · YOLO best.pt md5 = student.yaml mask.yolo.md5 · target_class ↔ ckpt one_obj · 얼린 외부 파라미터 md5(r20u5)
· 묶음 head_pose(틱) = 지금 config 의 r20u5 head_pose · StudentPolicy.from_dir 전체 검사 · 노드가 낼 /head_mask/config ·
/head_depth/config 를 오프라인으로 만들어 check_mask_config · check_depth_config 에 넣는다(노드 기본 인자 그대로).
공통: 64 MB SHM 프로파일(템플릿 · 생성본) · FP++ 컨테이너의 /opt/params/fastdds_lan.xml · 카메라 노드 프로파일의 segment_size.
불일치가 하나라도 있으면 종료 코드 1. 확인불가(컨테이너 · 카메라가 아직 안 떠 있음)는 실패로 치지 않는다.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "policy_control"))
from policy_control import student_mask as SM  # noqa: E402
from policy_control.hdgp_pin import md5  # noqa: E402
from policy_control.student_policy import StudentPolicy, check_depth_config  # noqa: E402

BUNDLE = ROOT / "deploy/policies/rh56f1/aglt/{side}_student1105mA"
LIVE_EXTR = ROOT / "config/global_camera_extrinsics_arm4090_r20u5.yaml"
FPP_PROFILE = "/opt/params/fastdds_lan.xml"
OK, BAD, NA = "OK", "불일치", "확인불가"


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts/nodes"))   # 노드는 스크립트로 돌 때처럼 옆 모듈(object_pose_node)을 import 한다
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/nodes" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HD, HM = _load("head_depth_node"), _load("head_mask_node")


def mask_node_config(policy_dir: Path) -> dict:
    """head_mask_node --policy-dir 가 기본 인자로 낼 /head_mask/config(YOLO 를 올리지 않고 main 과 같은 값으로)."""
    weights, card = HM.resolve_yolo(str(policy_dir), None)
    det = object.__new__(HM.MaskDetector)
    det.weights, det.md5, det.device = Path(weights), md5(weights), "cuda:0"
    det.imgsz, det.conf = int(card.get("imgsz", 640)), float(card.get("conf", 0.25))
    det.top1 = bool(card.get("top1_per_class", SM.DET_RULES["top1_per_class"]))
    det.cross_iou = float(card.get("cross_class_iou", SM.DET_RULES["cross_class_iou"]))
    det.half = False
    return det.config()


def depth_node_config(policy_dir: Path) -> dict:
    """head_depth_node --policy-dir(--rgb 없음)가 기본 인자로 낼 /head_depth/config."""
    extr, want, rgb = HD.resolve_extrinsics(str(policy_dir), None, False)
    _, got = HD.load_extrinsics(extr)
    if got != want:
        raise HD.PinError(f"외부 파라미터 {extr} md5 {got} ≠ {want}")
    rd = HD.real_depth
    return HD.node_config(rd.BLOCK, rd.MIN_VALID, HD.TABLE_Z, rd.FLOOR_MARGIN_M, extr, got, rgb)


def check_side(side: str, rows: list) -> None:
    import yaml

    d = BUNDLE.with_name(BUNDLE.name.format(side=side))
    card = yaml.safe_load((d / "student.yaml").read_text(encoding="utf-8"))

    def row(item, want, got, ok):
        rows.append((side, item, str(want), str(got), OK if ok else BAD))

    ck, yo, ex = card["ckpt"], card["mask"]["yolo"], card["extrinsics"]
    got = md5(d / ck["path"])
    row(f"ckpt md5 (it{ck['it']})", ck["md5"], got, got == ck["md5"])
    got = md5(d / yo["path"])
    row("YOLO best.pt md5", yo["md5"], got, got == yo["md5"])
    got = md5(d / ex["path"])
    row("외부 파라미터 md5", HD.EXTRINSICS_MD5, got, got == ex["md5"] == HD.EXTRINSICS_MD5)
    hp, live = HD.load_head_pose(d / ex["path"]), HD.load_head_pose(LIVE_EXTR)
    row("head_pose 틱(묶음 = config r20u5)", live and live.ticks, hp and hp.ticks,
        hp is not None and live is not None and hp == live)
    try:
        pol = StudentPolicy.from_dir(d, device="cpu")
    except Exception as exc:   # noqa: BLE001 — 어떤 검사가 막았든 표에 이유를 싣는다
        row("StudentPolicy.from_dir", "통과", f"{type(exc).__name__}: {exc}", False)
        return
    row("StudentPolicy.from_dir", "통과", "통과", True)
    one = pol.meta.get("one_obj")
    tgt = SM.CLASSES[pol.target]
    try:
        warn = SM.check_target(one, tgt)
        row("target_class ↔ ckpt one_obj", f"{SM.OBJ_TARGETS.get(str(one), ('?',))[0]} ({one})",
            f"{tgt}{' · ' + warn if warn else ''}", warn is None and one == card["one_obj"])
    except ValueError as exc:
        row("target_class ↔ ckpt one_obj", one, exc, False)
    for item, fn in (("/head_mask/config 대조", lambda: SM.check_mask_config(mask_node_config(d), pol.yolo, pol.depth_pre)),
                     ("/head_depth/config 대조(rgb 끔)",
                      lambda: check_depth_config(pol.depth_pre, depth_node_config(d), mask=pol.mask_on))):
        try:
            fn()
            row(item, "같음", "같음", True)
        except Exception as exc:   # noqa: BLE001
            row(item, "같음", f"{type(exc).__name__}: {exc}", False)


def shm_sizes(path: str) -> list[int] | None:
    """Fast-DDS XML 의 SHM segment_size 목록(못 읽으면 None)."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    return [int(t.findtext("{*}segment_size") or 0) for t in root.findall(".//{*}transport_descriptor")
            if (t.findtext("{*}type") or "").strip() == "SHM"]


def _docker(*args: str) -> str | None:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def check_shm(rows: list) -> None:
    def row(item, got, verdict):
        rows.append(("공통", item, f"SHM ≥ {HM.SHM_MIN}", str(got), verdict))

    def judge(sizes):
        return NA if sizes is None else (OK if sizes and max(sizes) >= HM.SHM_MIN else BAD)

    seen = {}
    tmpl = str(ROOT / "config/dds/fastdds_lan.xml.in")
    seen["템플릿"] = shm_sizes(tmpl)
    row("config/dds/fastdds_lan.xml.in", seen["템플릿"], judge(seen["템플릿"]) if seen["템플릿"] is not None else BAD)
    gen = sorted({*glob.glob(os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "s2r_fastdds_lan_*.xml")),
                  *glob.glob("/tmp/s2r_fastdds_lan_*.xml")})
    if not gen:
        row("lan_profile.sh 생성본", "없음(source 전)", NA)
    for p in gen:
        seen[p] = shm_sizes(p)
        row(f"생성본 {p}", seen[p], judge(seen[p]))
    names = _docker("ps", "--format", "{{.Names}}")
    fpp = [n for n in (names or "").split() if n.startswith("fpp_")]
    if names is None or not fpp:
        row("FP++ 컨테이너", "docker 못 읽음" if names is None else "도는 컨테이너 없음", NA)
    for n in fpp:
        mounts = json.loads(_docker("inspect", "-f", "{{json .Mounts}}", n) or "[]")
        src = next((m.get("Source") for m in mounts if m.get("Destination") == FPP_PROFILE), None)
        seen[n] = shm_sizes(src) if src else []
        row(f"FP++ {n} {FPP_PROFILE}", f"{src} {seen[n]}" if src else "마운트 없음(기본 SHM 512 KiB)", judge(seen[n]))
    cams = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            if b"realsense2_camera" in Path(p).read_bytes():
                cams.append(Path(p).parent)
        except OSError:
            continue
    if not cams:
        row("카메라 노드(realsense2_camera)", "도는 프로세스 없음", NA)
    for c in cams:
        try:
            env = dict(kv.split("=", 1) for kv in (c / "environ").read_text(errors="replace").split("\0") if "=" in kv)
        except OSError as exc:
            row(f"카메라 PID {c.name}", f"environ 못 읽음({exc.strerror})", NA)
            continue
        prof = env.get("FASTRTPS_DEFAULT_PROFILES_FILE")
        seen[f"cam{c.name}"] = shm_sizes(prof) if prof else []
        row(f"카메라 PID {c.name} 프로파일", f"{prof} {seen[f'cam{c.name}']}" if prof else "없음(기본 SHM 512 KiB)",
            judge(seen[f"cam{c.name}"]))
    known = {k: max(v) for k, v in seen.items() if v}
    if len(known) > 1:
        row("segment_size 서로 같은가", known, OK if len(set(known.values())) == 1 else BAD)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sides", nargs="+", choices=("right", "left"), default=["right", "left"])
    args = ap.parse_args(argv)
    rows: list[tuple] = []
    for side in args.sides:
        check_side(side, rows)
    check_shm(rows)
    w = max(len(r[1]) for r in rows)
    for r in rows:
        print(f"{r[4]:<5} {r[0]:<5} {r[1]:<{w}}  기대 {r[2]}  ·  실제 {r[3]}")
    bad = sum(r[4] == BAD for r in rows)
    print(f"\n불일치 {bad} · 확인불가 {sum(r[4] == NA for r in rows)} · OK {sum(r[4] == OK for r in rows)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
