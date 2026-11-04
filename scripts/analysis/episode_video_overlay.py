#!/usr/bin/env python3
"""실기 에피소드 bag 의 머리 카메라 영상에 FP++ 물체 자세 · 손바닥 · 손끝 · 넘김 시각을 겹친다(11.04 사용자).

    python3 scripts/analysis/episode_video_overlay.py logs/bags/<시각>_episode_<이름> --node grasp_shaker
    python3 scripts/analysis/episode_video_overlay.py logs/bags/<시각>_episode_<이름> --t0 40 --t1 55   # camera 첫 프레임부터 초
출력: <bag>/overlay_<구간>/frames/00000.png … · overlay.mp4 (--out 으로 바꾼다)

입력 bag = rh56f1_record.sh 의 arm(정책 · 실행기 상태 · /objects/*) 과 camera(camera_bag_recorder.py) 묶음.
- 물체: /objects/<물체>/live_pose(굵은 축) · /pose(가는 축, 스냅샷). base 프레임 body 자세다. 영상 stamp 이전에 찍힌 것 중
  가장 늦은 것(FP++ 가 이 프레임이나 그 전 프레임에서 본 자세)을 그리고, 영상보다 --max-age 넘게 늙었으면 뺀다.
- 손바닥 · 손끝 · 정책 컵: /policy_control/<팔>/obs 의 rh_aglt 관측(rh_aglt.ACTOR_LAYOUT 의 palm_pos · cup_rel_palm ·
  tips_rel_cup, base 프레임). 정책 컵(+)은 정책이 본 컵(쥔 뒤는 붙인 자세)이라 FP++ 축과 다르면 그 차이가 보인다.
- 넘김: rh_aglt 상태의 approach_grasp.phase 가 approach → policy 로 바뀐 시각(접근 정책 → 쥐기 정책). 노드 시작도 표시한다.
- 카메라: config/global_camera_extrinsics_arm4090_r20u5.yaml(그 head_pose 에서 잰 값) × 목 FK. camera bag 의
  /head/joint_states 로 T_base_cam(q) = T_base_neck(q) · T_base_neck(q_cal)⁻¹ · T_base_cam_cal. 목 기록이 없으면 정적 값.
시각: 영상 · 물체는 header stamp, 정책 · 실행기 상태 · 목은 bag 기록 시각(같은 PC system clock)으로 맞춘다.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / "scripts"), str(_ROOT / "deploy" / "policy_control")]

from head_fk_chain import t_base_neck, urdf_from_encoder  # noqa: E402
from head_position_hold_node import tick_to_deg  # noqa: E402
from object_registry import _base_from_camera  # noqa: E402

DEFAULT_EXTRINSICS = _ROOT / "config" / "global_camera_extrinsics_arm4090_r20u5.yaml"
COLOR = "/camera/camera/color/image_raw/compressed"
COLOR_INFO = "/camera/camera/color/camera_info"
HEAD = "/head/joint_states"
OBJ_RE = re.compile(r"^/objects/([^/]+)/(live_pose|pose)$")
RUNNER_RE = re.compile(r"^/policy_control/status/episode_runner(_\w+)?$")
AGLT_RE = re.compile(r"^/policy_control/status/rh_aglt_node_(right|left)$")
OBS_RE = re.compile(r"^/policy_control/(right|left)/obs$")
SIDE_BGR = {"right": (0, 140, 255), "left": (255, 160, 0)}


class OverlayError(RuntimeError):
    pass


# ============================================================================ bag 읽기
class Bag:
    """rosbag2 sqlite3 폴더 하나(읽기 전용). read() = [(기록 시각 ns, 메시지)]."""

    def __init__(self, path: Path) -> None:
        dbs = sorted(Path(path).glob("*.db3"))
        if not dbs:
            raise OverlayError(f"{path}: *.db3 가 없다")
        self.cons = [sqlite3.connect(f"file:{p}?mode=ro", uri=True) for p in dbs]
        self.types = {n: t for c in self.cons for n, t in c.execute("select name, type from topics")}

    def read(self, topic: str, t0: int | None = None, t1: int | None = None) -> list:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        if topic not in self.types:
            return []
        cls, out = get_message(self.types[topic]), []
        for c in self.cons:
            q = ("select m.timestamp, m.data from messages m join topics t on m.topic_id = t.id where t.name = ?"
                 + (" and m.timestamp >= ?" if t0 is not None else "") + (" and m.timestamp <= ?" if t1 is not None else "")
                 + " order by m.timestamp")
            args = [topic] + [v for v in (t0, t1) if v is not None]
            out += [(int(ts), deserialize_message(d, cls)) for ts, d in c.execute(q, args)]
        return sorted(out, key=lambda r: r[0])

    def first_time(self) -> int:
        return min(c.execute("select min(timestamp) from messages").fetchone()[0] for c in self.cons)


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def json_rows(bag: Bag, topic: str, t0: int | None = None, t1: int | None = None) -> list[tuple[int, dict]]:
    out = []
    for t, m in bag.read(topic, t0, t1):
        try:
            out.append((t, json.loads(m.data)))
        except json.JSONDecodeError:
            continue                    # 상태 토픽에 섞인 JSON 아닌 줄은 건너뛴다(구간 · 넘김 판정에 안 쓴다)
    return out


# ============================================================================ 구간 · 사건 (순수 로직)
def node_segment(rows: list[tuple[int, dict]], node: str) -> tuple[int, int]:
    """실행기 상태 [(t, {'node': …})] → 그 노드가 현재였던 처음 · 끝 시각. 여러 번이면 마지막 회."""
    runs, cur = [], None
    for t, s in rows:
        if s.get("node") == node:
            cur = [t, t] if cur is None else [cur[0], t]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    if not runs:
        seen = sorted({s.get("node") for _, s in rows if s.get("node")})
        raise OverlayError(f"노드 {node!r} 가 실행기 상태에 없다 — 있는 노드: {seen}")
    return runs[-1][0], runs[-1][1]


def node_events(rows: list[tuple[int, dict]]) -> list[tuple[int, str]]:
    out, prev = [], None
    for t, s in rows:
        n = s.get("node")
        if n and n != prev:
            out.append((t, f"node {n}"))
        prev = n or prev
    return out


def handoff_events(rows: list[tuple[int, dict]], side: str) -> list[tuple[int, str]]:
    """rh_aglt 상태 → approach_grasp.phase 가 approach → policy 로 바뀐 시각(에피소드마다)."""
    out, prev = [], {}
    for t, s in rows:
        ag = (s.get("approach_grasp") or {}).get("phase")
        ep = s.get("episode")
        if ag == "policy" and prev.get(ep) == "approach":
            out.append((t, f"HANDOFF {side[0].upper()}"))
        if ag:
            prev[ep] = ag
    return out


def latest_before(times: list[int], t: int) -> int | None:
    """정렬된 times 에서 t 이하인 마지막 칸 번호."""
    i = bisect.bisect_right(times, t) - 1
    return i if i >= 0 else None


# ============================================================================ 기하
def head_corrected(T_cal: np.ndarray, q_cal_urdf_deg: tuple[float, float],
                   q_urdf_deg: tuple[float, float] | None) -> np.ndarray:
    """T_base_cam(q) = T_base_neck(q) · T_base_neck(q_cal)⁻¹ · T_base_cam_cal. q 가 없으면 정적 값."""
    if q_urdf_deg is None:
        return T_cal
    return t_base_neck(*q_urdf_deg) @ np.linalg.inv(t_base_neck(*q_cal_urdf_deg)) @ T_cal


def calib_head_urdf_deg(extrinsics: Path) -> tuple[float, float]:
    hp = (yaml.safe_load(Path(extrinsics).read_text(encoding="utf-8")) or {}).get("head_pose") or {}
    if "pan_tick" not in hp or "tilt_tick" not in hp:
        raise OverlayError(f"{extrinsics}: head_pose.pan_tick · tilt_tick 가 없다(목 FK 보정 기준)")
    return urdf_from_encoder(tick_to_deg(hp["pan_tick"]), tick_to_deg(hp["tilt_tick"]))


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(K: np.ndarray, D: np.ndarray, T_base_cam: np.ndarray, pts_base: np.ndarray) -> np.ndarray:
    """base 점(N,3) → 픽셀(N,2). 카메라 뒤(z ≤ 1 cm)는 nan."""
    import cv2
    T_cam_base = np.linalg.inv(T_base_cam)
    p = (T_cam_base[:3, :3] @ np.asarray(pts_base, float).reshape(-1, 3).T).T + T_cam_base[:3, 3]
    uv, _ = cv2.projectPoints(p.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    uv[p[:, 2] <= 0.01] = np.nan
    return uv


def obs_slices() -> dict[str, slice]:
    from policy_control.rh_aglt import ACTOR_LAYOUT
    out, i = {}, 0
    for name, n in ACTOR_LAYOUT:
        out[name] = slice(i, i + n)
        i += n
    return out


def palm_from_obs(obs, sl: dict[str, slice]) -> dict | None:
    """rh_aglt 관측 → base 프레임 손바닥 · 정책 컵 · 손끝 5. 앞쪽 칸이 모자라면 None(다른 정책의 obs)."""
    o = np.asarray(obs, float)
    if o.size < sl["tips_rel_cup"].stop:
        return None
    palm = o[sl["palm_pos"]]
    cup = palm + o[sl["cup_rel_palm"]]
    return {"palm": palm, "cup": cup, "tips": o[sl["tips_rel_cup"]].reshape(5, 3) + cup}


# ============================================================================ 그리기
@dataclass(frozen=True)
class Track:
    times: list            # 정렬된 시각(ns)
    values: list


def track(rows, key=lambda t, m: t, value=lambda m: m) -> Track:
    pairs = sorted(((key(t, m), value(m)) for t, m in rows), key=lambda p: p[0])
    return Track([p[0] for p in pairs], [p[1] for p in pairs])


def at(tr: Track, t: int, max_age_ns: int):
    i = latest_before(tr.times, t)
    if i is None or t - tr.times[i] > max_age_ns:
        return None, None
    return tr.values[i], t - tr.times[i]


def draw_axes(img, K, D, T, pos, quat, length: float, thick: int, label: str) -> None:
    import cv2
    R = quat_wxyz_to_R(quat)
    pts = np.vstack([pos, pos + length * R.T])
    uv = project(K, D, T, pts)
    if np.isnan(uv[0]).any():
        return
    o = tuple(int(v) for v in uv[0])
    for k, bgr in enumerate(((0, 0, 255), (0, 200, 0), (255, 0, 0))):     # x 빨강 · y 초록 · z 파랑
        if not np.isnan(uv[k + 1]).any():
            cv2.line(img, o, tuple(int(v) for v in uv[k + 1]), bgr, thick, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (o[0] + 6, o[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hand(img, K, D, T, side: str, h: dict) -> None:
    import cv2
    uv = project(K, D, T, np.vstack([h["palm"], h["cup"], h["tips"]]))
    bgr = SIDE_BGR[side]
    if not np.isnan(uv[0]).any():
        p = tuple(int(v) for v in uv[0])
        cv2.circle(img, p, 9, bgr, 2, cv2.LINE_AA)
        cv2.putText(img, f"palm {side[0].upper()}", (p[0] + 10, p[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
    if not np.isnan(uv[1]).any():
        cv2.drawMarker(img, tuple(int(v) for v in uv[1]), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
    for k in range(5):
        if not np.isnan(uv[2 + k]).any():
            cv2.circle(img, tuple(int(v) for v in uv[2 + k]), 4, bgr, -1, cv2.LINE_AA)


def draw_text(img, lines: list[tuple[str, tuple]], org=(10, 22)) -> None:
    import cv2
    for k, (text, bgr) in enumerate(lines):
        y = org[1] + 20 * k
        cv2.putText(img, text, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 1, cv2.LINE_AA)


def draw_timeline(img, t: int, t0: int, t1: int, events: list[tuple[int, str]]) -> None:
    import cv2
    h, w = img.shape[:2]
    x = lambda tt: int(20 + (w - 40) * (tt - t0) / max(1, t1 - t0))     # noqa: E731
    cv2.line(img, (20, h - 20), (w - 20, h - 20), (200, 200, 200), 2)
    for te, name in events:
        if t0 <= te <= t1:
            col = (0, 0, 255) if name.startswith("HANDOFF") else (0, 255, 255)
            cv2.line(img, (x(te), h - 30), (x(te), h - 10), col, 2)
    cv2.circle(img, (x(t), h - 20), 6, (255, 255, 255), -1)


# ============================================================================ 실행
def window(arm: Bag, cam: Bag, args) -> tuple[int, int, str]:
    pad = int(args.pad * 1e9)
    if args.node:
        rows = [r for tp in arm.types if RUNNER_RE.match(tp) for r in json_rows(arm, tp)]
        a, b = node_segment(sorted(rows, key=lambda r: r[0]), args.node)
        return a - pad, b + pad, args.node
    if args.t0 is None or args.t1 is None:
        raise OverlayError("--node 또는 --t0 · --t1 중 하나")
    base = cam.first_time()
    return base + int(args.t0 * 1e9), base + int(args.t1 * 1e9), f"t{args.t0:g}-{args.t1:g}"


def render(bag_dir: Path, args) -> Path:
    import cv2
    arm, cam = Bag(bag_dir / "arm"), Bag(bag_dir / "camera")
    t0, t1, tag = window(arm, cam, args)
    out = args.out or bag_dir / f"overlay_{tag}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    max_age = int(args.max_age * 1e9)

    info = cam.read(COLOR_INFO)
    if not info:
        raise OverlayError(f"{bag_dir}/camera 에 {COLOR_INFO} 가 없다")
    K, D = np.array(info[0][1].k, float).reshape(3, 3), np.array(info[0][1].d, float)
    T_cal = np.asarray(_base_from_camera(args.extrinsics), float)
    q_cal = calib_head_urdf_deg(args.extrinsics)
    head = track(cam.read(HEAD, t0 - max_age, t1),
                 value=lambda m: tuple(float(np.degrees(m.position[m.name.index(j)])) for j in ("head_j_pan", "head_j_tilt")))
    if not head.times:
        print(f"[overlay] {HEAD} 기록이 없다 — 정적 외부 파라미터({args.extrinsics.name})로 그린다")

    ev_lo, ev_hi = t0 - 60_000_000_000, t1          # 넘김은 앞선 상태를 봐야 바뀐 순간을 안다
    events = node_events(sorted((r for tp in arm.types if RUNNER_RE.match(tp) for r in json_rows(arm, tp, ev_lo, ev_hi)),
                                key=lambda r: r[0]))
    aglt = {}
    for tp in arm.types:
        m = AGLT_RE.match(tp)
        if m:
            rows = json_rows(arm, tp, ev_lo, ev_hi)
            events += handoff_events(rows, m.group(1))
            aglt[m.group(1)] = track(rows, value=lambda s: s)
    events.sort()
    runner = track([r for tp in arm.types if RUNNER_RE.match(tp) for r in json_rows(arm, tp, ev_lo, ev_hi)],
                   value=lambda s: s.get("node", ""))
    sl = obs_slices()
    hands = {OBS_RE.match(tp).group(1): track(arm.read(tp, t0 - max_age, t1), value=lambda m: palm_from_obs(m.data, sl))
             for tp in arm.types if OBS_RE.match(tp)}
    objs = {}
    for tp in arm.types:
        m = OBJ_RE.match(tp)
        if m and not m.group(1).startswith("cup_holder"):
            objs[(m.group(1), m.group(2))] = track(arm.read(tp, t0 - 5 * max_age, t1), key=lambda t, msg: stamp_ns(msg.header),
                                                   value=lambda msg: (np.array([msg.pose.position.x, msg.pose.position.y,
                                                                                msg.pose.position.z]),
                                                                      np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                                                                                msg.pose.orientation.y, msg.pose.orientation.z])))

    frames = cam.read(COLOR, t0, t1)
    if not frames:
        raise OverlayError(f"{bag_dir}/camera 에 그 구간({(t1 - t0) / 1e9:.1f} s) {COLOR} 프레임이 없다")
    fps = args.fps or max(1.0, round((len(frames) - 1) / max(1e-9, (frames[-1][0] - frames[0][0]) / 1e9), 1))
    writer = None
    for k, (t, msg) in enumerate(frames):
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise OverlayError(f"프레임 {k}: JPEG 를 못 풀었다(format {msg.format!r})")
        ts = stamp_ns(msg.header)
        q, _ = at(head, t, max_age)
        T = head_corrected(T_cal, q_cal, q)
        for (name, kind), tr in objs.items():
            pose, age = at(tr, ts, max_age)
            if pose is not None:
                live = kind == "live_pose"
                draw_axes(img, K, D, T, pose[0], pose[1], 0.06 if live else 0.03, 3 if live else 1,
                          f"{name} {age / 1e6:.0f}ms" if live else "")
        for side, tr in hands.items():
            h, _ = at(tr, t, max_age)
            if h is not None:
                draw_hand(img, K, D, T, side, h)
        node, _ = at(runner, t, 10 * max_age)
        lines = [(f"t {(t - t0) / 1e9 - args.pad:+6.2f}s  node {node or '-'}", (255, 255, 255))]
        for side, tr in aglt.items():
            s, _ = at(tr, t, max_age)
            if s:
                ag = s.get("approach_grasp") or {}
                lines.append((f"{side[0].upper()} aglt {s.get('phase')}/{ag.get('phase', '-')} palm-cup "
                              f"{ag.get('palm_cup_m', '-')}", SIDE_BGR[side]))
        for te, name in events:
            if t0 <= te <= t1:
                near = abs(t - te) < 300_000_000
                lines.append((f"{'>> ' if near else ''}{name} @ {(te - t0) / 1e9 - args.pad:+.2f}s",
                               (0, 0, 255) if near else (200, 200, 200)))
        draw_text(img, lines)
        draw_timeline(img, t, t0, t1, events)
        cv2.imwrite(str(out / "frames" / f"{k:05d}.png"), img)
        if writer is None:
            writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        writer.write(img)
    writer.release()
    print(f"[overlay] {len(frames)} 프레임 · {fps:g} fps · 사건 {[(round((te - t0) / 1e9 - args.pad, 2), n) for te, n in events if t0 <= te <= t1]}")
    print(f"[overlay] → {out}/overlay.mp4 · {out}/frames/")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("bag", type=Path, help="logs/bags/<시각>_episode_<이름>(arm · camera 하위 폴더)")
    ap.add_argument("--node", help="실행기 노드 id(예: grasp_shaker) — 그 노드가 현재였던 구간")
    ap.add_argument("--t0", type=float, help="camera bag 첫 메시지부터 초")
    ap.add_argument("--t1", type=float)
    ap.add_argument("--pad", type=float, default=1.0, help="--node 구간 앞뒤 여유 초(실행기 상태가 2 Hz 라 경계가 0.5 s 흔들린다)")
    ap.add_argument("--max-age", type=float, default=0.5, help="영상보다 이만큼 넘게 늙은 자세 · 관측은 그리지 않는다(초)")
    ap.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    ap.add_argument("--fps", type=float, default=None, help="mp4 빈도(기본 = 기록 빈도)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.node:
        args.pad = 0.0
    try:
        render(args.bag, args)
    except OverlayError as e:
        print(f"[overlay] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
