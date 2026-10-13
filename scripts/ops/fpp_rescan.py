#!/usr/bin/env python3
"""컵 좌표를 지금 카메라로 다시 찍는다 — FP++ 묶음 컨테이너(fpp_cups)를 끄지 않고(10.08 사용자).

    python3 scripts/ops/fpp_rescan.py cyl60 cyl60_blue cyl60_pink [--wait 120]

미션 cups 단계가 FP++ 를 켠 뒤 부른다. 처음 실행이면 컨테이너가 켜지며 찍는 한 번을 기다리고(다시 명령하지 않는다),
단계를 다시 실행하면(상황판 '↶ 여기서 다시') /perception_plus_plus/snapshot/cmd 로 다시 찍게 하고 그 회차가 끝나길
기다린다. 다시 찍는 동안 컨테이너는 그 컵의 옛 좌표를 내지 않는다 — 옛 값이 정책 입력으로 새지 않는다.
끝나면 컵마다 base 좌표(/objects/<컵>/pose)와 판정을 찍는다. 못 찾은 컵이 있으면 rc 1.
로봇은 움직이지 않는다 — 카메라 영상만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import math
import sys
import time
from pathlib import Path

STATUS_TOPIC = "/perception_plus_plus/snapshot/status"
CMD_TOPIC = "/perception_plus_plus/snapshot/cmd"
TABLE_TOP_Z = 0.205
CUP_ORIGIN_Z = 0.085            # cyl60 원점 = 원통 중심(바닥 위)
Z_TOL = 0.008
TILT_MAX_DEG = 3.0
X_RANGE = (0.1, 0.4)


def first_action(status: dict | None) -> str:
    """처음 할 일 — 상태가 아직 없으면 기다리고, 첫 회차(generation 0)가 안 끝났으면 그것을 기다리고, 아니면 다시 찍게 한다."""
    if status is None:
        return "wait_status"
    if int(status.get("generation", 0)) == 0:
        return "wait_round"
    return "command"


def round_done(status: dict, after: int, names: list[str]) -> bool:
    """after 보다 새 회차(찍기 한 바퀴)가 끝났고 물어본 컵을 다 찾았는가. 못 찾은 컵은 노드가 retry_s 마다 다시 찾는다."""
    objs = status.get("objects", {})
    return int(status.get("generation", 0)) > after and all(objs.get(n, {}).get("found") for n in names)


def merge_status(st: dict, status: dict) -> None:
    """묶음별 최신 상태 — 묶음 컨테이너가 여럿이면 같은 토픽에 각자 낸다(10.09 병 + 쉐이커)."""
    key = status.get("group") or "+".join(sorted(status.get("objects", {})))
    st[key] = status


def groups_for(st: dict, names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for g, status in st.items():
        mine = [n for n in names if n in status.get("objects", {})]
        if mine:
            out[g] = mine
    return out


def missing_names(st: dict, names: list[str]) -> list[str]:
    """어느 컨테이너 상태에도 없는 물체."""
    known = {n for status in st.values() for n in status.get("objects", {})}
    return [n for n in names if n not in known]


def all_done(st: dict, after: dict[str, int], names: list[str]) -> bool:
    if missing_names(st, names):
        return False
    return all(round_done(st[g], after.get(g, 0), mine) for g, mine in groups_for(st, names).items())


def missing(status: dict, names: list[str]) -> list[str]:
    objs = status.get("objects", {})
    return [f"{n}: {objs.get(n, {}).get('why', '상태 없음')}" for n in names if not objs.get(n, {}).get("found")]


#: ★10.13 사용자 "둘중 하나라도 인지되게": 한 물체의 상태가 먼저 오면 나머지 묶음 상태는 이만큼만 더 기다린다
STATUS_GRACE_S = 5.0


def found_names(st: dict, after: dict[str, int], names: list[str]) -> list[str]:
    """after 보다 새 회차가 끝난 묶음에서 찾은 물체(names 순서)."""
    hit = set()
    for g, mine in groups_for(st, names).items():
        if int(st[g].get("generation", 0)) <= after.get(g, 0):
            continue
        objs = st[g].get("objects", {})
        hit |= {n for n in mine if objs.get(n, {}).get("found")}
    return [n for n in names if n in hit]


def enough_done(st: dict, after: dict[str, int], names: list[str], require_all: bool = False) -> bool:
    """★10.13 다 찾았거나, (require_all 이 아니면) 물어본 물체가 든 묶음마다 새 회차가 끝났고 하나라도 찾았다."""
    if all_done(st, after, names):
        return True
    if require_all or missing_names(st, names):
        return False
    groups = groups_for(st, names)
    settled = all(int(st[g].get("generation", 0)) > after.get(g, 0) for g in groups)
    return settled and bool(found_names(st, after, names))


def lost_reasons(st: dict, names: list[str]) -> list[str]:
    """★10.13 못 찾은 물체와 이유('<물체>: <이유>') — 상태가 아예 없는 물체(묶음 컨테이너가 안 떴다)도 넣는다."""
    absent = missing_names(st, names)
    out = [f"{n}: 상태 없음 — 이 물체의 fpp_<묶음> 컨테이너가 안 떠 있다" for n in absent]
    out += [x for g, mine in groups_for(st, names).items() for x in missing(st[g], mine)]
    return out


def check_cup(name: str, xyz: tuple[float, float, float], tilt_deg: float, origin_above_bottom: float = CUP_ORIGIN_Z) -> list[str]:
    """테이블 위에 서 있는 물체 기준의 경고들(빈 목록 = 통과). 원점 높이는 물체마다(레지스트리, 쉐이커 0.065)."""
    x, _, z = xyz
    warn = []
    z_want = TABLE_TOP_Z + origin_above_bottom
    if abs(z - z_want) > Z_TOL:
        warn.append(f"z {z:.3f} — 기대 {z_want:.3f} ± {Z_TOL * 1e3:.0f} mm")
    if tilt_deg > TILT_MAX_DEG:
        warn.append(f"기울기 {tilt_deg:.1f}° > {TILT_MAX_DEG:.0f}°")
    if not X_RANGE[0] <= x <= X_RANGE[1]:
        warn.append(f"x {x:.3f} — {X_RANGE[0]}~{X_RANGE[1]} 밖")
    return warn


def _origins(names: list[str]) -> dict[str, float]:
    """물체마다 원점 높이(바닥 위) — 레지스트리를 못 읽으면 비워 둔다(cyl60 값으로 판정)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from object_registry import load_registry
        reg = load_registry()
        return {n: float(reg.get(n).origin_above_bottom_m) for n in names}
    except (OSError, ValueError, KeyError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("cups", nargs="+")
    ap.add_argument("--wait", type=float, default=120.0, help="전체 대기 상한(s)")
    # ★10.13 기본은 하나라도 찾으면 통과(못 찾은 물체는 경고) — 전부 필요할 때만 --require-all
    ap.add_argument("--require-all", action="store_true", help="물어본 물체를 전부 찾아야 통과")
    ap.add_argument("--phase", choices=("auto", "status", "full"), default="auto",
                    help="auto: 상황판(ROS_LOCALHOST_ONLY≠1)이면 상태 · 명령은 localhost DDS 자식이, base 좌표는 이 프로세스가")
    # ★10.13 5090 + vision-3090: FP++ 컨테이너 · 상태 토픽은 인지 PC 의 localhost DDS 에만 있다 — 상태 단계를 그 PC 에서 ssh 로
    ap.add_argument("--status-host", default="", help="FP++ 가 도는 인지 PC(ssh 이름). 비우면 이 PC 의 localhost DDS")
    args = ap.parse_args(argv)
    if not args.status_host and args.phase == "auto":
        # ★10.13 --status-host 가 없으면 이 PC 의 인지 런처가 맡은 PC — 콘솔은 run 시작 때 명령을 고정해 옛 명령으로 다시 실행해도 맞는다
        args.status_host = launcher_host()
    if args.status_host and args.phase == "auto":
        return _split_run(args)
    if args.phase == "auto" and os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        return _split_run(args)
    return _run(args, only_status=args.phase == "status")


def _split_run(args) -> int:
    """10.09 실기: FP++ 컨테이너는 localhost 전용 DDS(ROS_LOCALHOST_ONLY=1)라 상황판 노드(=0)가 상태 토픽을 못 본다.
    상태 · 명령은 localhost 자식이 하고(결과를 FPP_STATUS 줄로), base 좌표(/objects, 로봇 쪽 DDS)는 이 프로세스가 본다."""
    import subprocess
    env = {**os.environ, "ROS_LOCALHOST_ONLY": "1"}
    try:
        child = subprocess.run(status_argv(args.cups, args.wait, args.require_all, args.status_host),
                               env=env, capture_output=True, text=True, timeout=args.wait + 60)
    except subprocess.TimeoutExpired:
        print(f"[rescan] 상태 단계가 {args.wait + 60:.0f} s 안에 안 끝났다({args.status_host or 'localhost'})", file=sys.stderr)
        return 1
    st = None
    for line in child.stdout.splitlines():
        if line.startswith(STATUS_MARK):
            st = json.loads(line[len(STATUS_MARK):])
        else:
            print(line)
    if child.stderr.strip():
        print(child.stderr.strip()[-1500:], file=sys.stderr)
    if child.returncode != 0 or st is None:
        return child.returncode or 1
    return _run(args, only_status=False, preset=st)


STATUS_MARK = "FPP_STATUS "
#: 인지 PC 의 저장소(홈 기준 — PC 마다 사용자 이름이 다르다, perception_launcher 와 같은 규약)
REMOTE_SIM2REAL = "rl_ws/sim2real"


#: perception_launcher_node.py 의 --host 기본값(인지 PC) — 테스트가 그 파일과 맞는지 본다
LAUNCHER_DEFAULT_HOST = "vision-3090"
LOCAL_HOSTS = ("local", "localhost", "127.0.0.1")


def launcher_host(proc: Path = Path("/proc")) -> str:
    """이 PC 에서 도는 perception_launcher_node 가 맡은 인지 PC(ssh 이름). 런처가 없거나 local 이면 '' — 이 PC 의 DDS."""
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            args = [a.decode(errors="replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if not any(Path(a).name == "perception_launcher_node.py" for a in args):
            continue
        host = LAUNCHER_DEFAULT_HOST
        for i, a in enumerate(args):
            if a == "--host" and i + 1 < len(args):
                host = args[i + 1]
            elif a.startswith("--host="):
                host = a.split("=", 1)[1]
        return "" if host in LOCAL_HOSTS else host
    return ""


def status_argv(cups: list[str], wait: float, require_all: bool, host: str = "") -> list[str]:
    """상태 · 명령 단계(localhost DDS) 명령. host 가 있으면 그 PC 에서 ssh 로(common.sh 가 도메인 · localhost 를 맞춘다)."""
    tail = [*cups, "--wait", f"{wait:g}", "--phase", "status", *(["--require-all"] if require_all else [])]
    if not host:
        return [sys.executable, str(Path(__file__).resolve()), *tail]
    remote = (f"cd {REMOTE_SIM2REAL} && source scripts/vision/common.sh && "
              f"python3 scripts/ops/fpp_rescan.py {' '.join(tail)}")
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote]


def _run(args, only_status: bool, preset: dict | None = None) -> int:

    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("fpp_rescan")
    box: dict = {"st": {}, "poses": {}}
    node.create_subscription(String, STATUS_TOPIC, lambda m: merge_status(box["st"], json.loads(m.data)), 10)
    pub = node.create_publisher(String, CMD_TOPIC, 10)
    t_end = time.monotonic() + args.wait

    def spin_until(pred) -> bool:
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if pred():
                return True
        return False

    try:
        if preset is not None:                       # 상태는 localhost 자식이 끝냈다 — base 좌표만 본다
            box["st"] = preset
            return _report(node, box, args, spin_until)
        # ★10.13 하나라도 상태가 오면 나머지 묶음은 STATUS_GRACE_S 만 더 기다린다(--require-all 이면 끝까지)
        spin_until(lambda: len(missing_names(box["st"], args.cups)) < len(args.cups))
        if args.require_all:
            spin_until(lambda: not missing_names(box["st"], args.cups))
        else:
            t_grace = time.monotonic() + STATUS_GRACE_S
            spin_until(lambda: not missing_names(box["st"], args.cups) or time.monotonic() > t_grace)
        absent = missing_names(box["st"], args.cups)
        if absent and (args.require_all or len(absent) == len(args.cups)):
            print(f"[rescan] {', '.join(absent)} 를 내는 묶음 FP++ 컨테이너 상태가 {args.wait:.0f} s 안에 "
                  "안 온다 — fpp_<묶음> 이 떠 있는가 · docker logs 로 확인", file=sys.stderr)
            return 1
        if absent:
            print(f"[rescan] ⚠ {', '.join(absent)} 는 FP++ 상태가 안 온다(fpp_<묶음> 컨테이너) — 나머지로 계속",
                  file=sys.stderr)
        names = [n for n in args.cups if n not in absent]
        st = box["st"]
        groups = groups_for(st, names)
        after = {g: int(st[g].get("generation", 0)) for g in groups}
        again = [n for g, mine in groups.items() if first_action(st[g]) == "command" for n in mine]
        if again:
            spin_until(lambda: pub.get_subscription_count() > 0)
            pub.publish(String(data=",".join(again)))
            print(f"[rescan] 다시 찍기 요청 — {', '.join(again)}")
        if len(again) < len(names):
            print("[rescan] FP++ 가 켜지며 찍는 첫 회차를 기다린다 — " + ", ".join(n for n in names if n not in again))
        if not spin_until(lambda: enough_done(box["st"], after, names, args.require_all)):
            lost = [x for g, mine in groups_for(box["st"], names).items() for x in missing(box["st"][g], mine)]
            errs = " · ".join(f"{g}: {s_.get('error')}" for g, s_ in box["st"].items() if s_.get("error"))
            # ★10.13 시간 안에 하나라도 찾았으면 나머지는 경고로 남기고 통과(--require-all 이면 실패)
            if args.require_all or not found_names(box["st"], after, names):
                print(f"[rescan] {args.wait:.0f} s 안에 다 못 찾았다 — {' · '.join(lost) or '회차가 안 끝났다'} · "
                      f"오류 {errs or '-'}", file=sys.stderr)
                return 1
        if only_status:
            print(STATUS_MARK + json.dumps(box["st"], ensure_ascii=False))
            return 0
        return _report(node, box, args, spin_until)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _report(node, box: dict, args, spin_until) -> int:
    """찾은 물체의 base 좌표(/objects/<물체>/pose)와 판정을 찍는다."""
    from geometry_msgs.msg import PoseStamped
    if True:
        st = box["st"]
        groups = groups_for(st, args.cups)
        info_of = {n: st[g]["objects"][n] for g, mine in groups_for(st, args.cups).items() for n in mine}
        # ★10.13 못 찾은 물체는 경고로 찍고 찾은 물체만 base 좌표를 기다린다
        lost = lost_reasons(st, args.cups)
        why_lost = {x.split(":")[0]: x.split(": ", 1)[-1] for x in lost}
        t_round = time.time()

        def on_pose(name):
            def f(m):
                box["poses"][name] = m
            return f
        for n in args.cups:
            node.create_subscription(PoseStamped, f"/objects/{n}/pose", on_pose(n), 10)
        found = [n for n in args.cups if n not in {x.split(":")[0] for x in lost}]
        origin = _origins(args.cups)
        spin_until(lambda: all(n in box["poses"] for n in found))
        rounds = ", ".join(f"{g} {st[g].get('generation')}" for g in groups)
        print(f"[rescan] 회차 {rounds} · {time.strftime('%H:%M:%S', time.localtime(t_round))}")
        for n in args.cups:
            info = info_of.get(n, {})
            m = box["poses"].get(n)
            if m is None:
                print(f"  {n:11s} 없음 — {why_lost.get(n) or info.get('why', 'base 자세가 안 온다(object_pose_node)')}")
                continue
            p, q = m.pose.position, m.pose.orientation
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
            warn = check_cup(n, (p.x, p.y, p.z), tilt, origin.get(n, CUP_ORIGIN_Z))
            print(f"  {n:11s} x {p.x:.3f} y {p.y:+.3f} z {p.z:.3f} 기울기 {tilt:.1f}° · 색 {info.get('color_score')} · "
                  f"흔들림 {info.get('spread_mm')} mm  {'✓' if not warn else '⚠ ' + ' · '.join(warn)}")
        ok = [n for n in found if n in box["poses"]]
        if lost or len(ok) < len(args.cups):
            print(f"[rescan] ⚠ 못 찾은 물체 — {', '.join(n for n in args.cups if n not in ok)}"
                  f" · 이 물체를 쓰는 에피소드는 정책 노드가 reset 을 거부한다(컵 자세 없음 → TARGET_NOT_FOUND)", file=sys.stderr)
        if not ok:
            return 1
        return 1 if args.require_all and len(ok) < len(args.cups) else 0


if __name__ == "__main__":
    sys.exit(main())
