#!/usr/bin/env python3
"""실기 에피소드 머리 카메라 bag → mp4(11.18 사용자 "이제는 카메라 영상 남겨줘").

    python3 scripts/analysis/camera_bag_to_mp4.py logs/runs/<YYMMDD>/bags/<시각>_episode_<이름>/camera [--out …] [--fps …]
출력 기본 = <bag 폴더>/../camera.mp4. 컬러 JPEG(/camera/camera/color/image_raw/compressed)만 쓰고 깊이는 뺀다.
프레임 왼쪽 위 = 기록 시각(현지 HH:MM:SS.ss)과 첫 프레임부터 초. fps 기본 = 기록 빈도((메시지 수 - 1) / 길이).
에피소드 끝에 미션(_chain_camera_bag) · rh56f1_record.sh stop 이 nice 19 로 부른다. 기록 중(metadata.yaml 없음)이거나
컬러 프레임이 없으면 종료 코드 1.
★11.18 사용자(보고서): bag 에 FP++ live 추적(/perception_plus_plus/<물체>/live_pose, 카메라 프레임)이 있으면 옆에
camera_bbox.mp4 도 만든다(camera.mp4 는 그대로). 물체 메시 AABB(config/objects.yaml aabb)를 그 자세로 놓아 K 로만 투영한
3D 상자 · 이름 · 나이(영상 stamp - 자세 stamp). 0.5 s 넘게 늙으면 회색 가는 선, 2 s 넘으면 안 그린다. 물체 색은 이름 순서로 고정.
"""
from __future__ import annotations

import argparse
import datetime
import heapq
import itertools
import re
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_video_overlay import (COLOR, COLOR_INFO, Bag, OverlayError, at, project,  # noqa: E402
                                   quat_wxyz_to_R, stamp_ns, track)
from object_registry import load_registry  # noqa: E402  (episode_video_overlay 가 scripts/ 를 경로에 넣는다)

cv2.setNumThreads(1)   # 카메라 PC 에서 FP++ 옆에 돈다 — 코어를 넓게 잡지 않는다

POSE_RE = re.compile(r"^/perception_plus_plus/([^/]+)/live_pose$")
STALE_NS, HIDE_NS = 500_000_000, 2_000_000_000
PALETTE = ((0, 200, 255), (255, 120, 0), (200, 0, 255), (0, 220, 0), (255, 0, 160), (0, 90, 255), (255, 255, 0), (160, 255, 200))
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]   # 꼭짓점 번호 = x·y·z 비트


def _label(img: np.ndarray, text: str) -> None:
    for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 2)):   # 검은 테두리 위 흰 글자
        cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)


def box_px(K: np.ndarray, D: np.ndarray, pos, quat_wxyz, aabb) -> np.ndarray:
    """물체(메시) 프레임 AABB 8 꼭짓점 → 픽셀(8,2). FP++ 자세는 카메라 프레임이라 외부 파라미터 · 목 FK 없이 K 로만 투영한다."""
    corners = np.array(list(itertools.product(*zip(*aabb))), float)
    return project(K, D, np.eye(4), np.asarray(pos, float) + corners @ quat_wxyz_to_R(quat_wxyz).T)


def draw_box(img: np.ndarray, uv: np.ndarray, bgr: tuple, label: str, stale: bool) -> None:
    color, thick = ((150, 150, 150), 1) if stale else (bgr, 2)
    for i, j in EDGES:
        if not (np.isnan(uv[i]).any() or np.isnan(uv[j]).any()):
            cv2.line(img, tuple(int(v) for v in uv[i]), tuple(int(v) for v in uv[j]), color, thick, cv2.LINE_AA)
    ok = uv[~np.isnan(uv).any(axis=1)]
    if len(ok):
        x, y = (int(v) for v in ok[np.argmin(ok[:, 1])])      # 화면에서 가장 높은 꼭짓점 위에, 화면 안으로 당겨서
        x, y = min(max(x - 20, 4), img.shape[1] - 12 * len(label)), min(max(y - 8, 64), img.shape[0] - 8)
        for c, t in (((0, 0, 0), 4), (color, 2)):
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, t, cv2.LINE_AA)


def load_boxes(bag: Bag) -> tuple[list, np.ndarray | None, np.ndarray | None]:
    """bag 의 FP++ live 추적 → [(이름, aabb, 색, Track(header stamp))], K, D. 자세나 camera_info 가 없으면 빈 목록.
    색 = bag 의 물체 이름 순서(한 영상 안에서 8 개까지 겹치지 않고, 같은 물체 묶음이면 회차마다 같다)."""
    topics = sorted(t for t in bag.types if POSE_RE.match(t))
    info = bag.read(COLOR_INFO) if topics else []
    if not info:
        if topics:
            print(f"camera_bag_to_mp4: {COLOR_INFO} 가 없어 상자를 못 그린다", file=sys.stderr)
        return [], None, None
    reg, out = load_registry(), []
    for tp in topics:
        name = POSE_RE.match(tp).group(1)
        try:
            spec = reg.get(name)
        except ValueError:
            print(f"camera_bag_to_mp4: {name} 가 레지스트리에 없어 상자를 뺀다", file=sys.stderr)
            continue
        tr = track(bag.read(tp), key=lambda t, m: stamp_ns(m.header),
                   value=lambda m: ((m.pose.position.x, m.pose.position.y, m.pose.position.z),
                                    (m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z)))
        out.append((name, spec.aabb, PALETTE[len(out) % len(PALETTE)], tr))
    m = info[0][1]
    return out, np.array(m.k, float).reshape(3, 3), np.array(m.d, float)


def draw_boxes(img: np.ndarray, boxes: list, K: np.ndarray, D: np.ndarray, t_img: int) -> None:
    for name, aabb, bgr, tr in boxes:
        pose, age = at(tr, t_img, HIDE_NS)
        if pose is not None:
            draw_box(img, box_px(K, D, *pose, aabb), bgr, f"{name} {age / 1e9:.2f}s", age > STALE_NS)


def convert(bag_dir: Path, out: Path | None = None, fps: float | None = None) -> tuple[list[Path], int]:
    if not (bag_dir / "metadata.yaml").exists():
        raise OverlayError(f"{bag_dir}: metadata.yaml 가 없다 — 기록 중이거나 기록기가 마무리하지 못했다")
    bag = Bag(bag_dir)
    q = "select m.timestamp, m.data from messages m join topics t on m.topic_id = t.id where t.name = ? order by m.timestamp"
    stats = [c.execute("select count(*), min(m.timestamp), max(m.timestamp) from messages m join topics t "
                       "on m.topic_id = t.id where t.name = ?", (COLOR,)).fetchone() for c in bag.cons]
    n = sum(s[0] for s in stats)
    if not n:
        raise OverlayError(f"{bag_dir}: {COLOR} 프레임이 없다")
    t_first = min(s[1] for s in stats if s[0])
    span = (max(s[2] for s in stats if s[0]) - t_first) / 1e9
    fps = fps or ((n - 1) / span if n > 1 and span > 0 else 1.0)

    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    out = out or bag_dir.parent / "camera.mp4"
    writers, parts, written = [], [], 0
    try:
        try:                                               # 리뷰 M: 상자 판이 깨져도 원본 camera.mp4 는 남긴다
            boxes, K, D = load_boxes(bag)
        except Exception as e:
            print(f"[camera_bag_to_mp4] FP++ 상자 생략: {type(e).__name__}: {e}", file=sys.stderr)
            boxes, K, D = [], None, None
        outs = [out] + ([out.with_name(out.stem + "_bbox" + out.suffix)] if boxes else [])
        parts = [o.with_name(o.stem + ".part" + o.suffix) for o in outs]   # 중간에 끊긴 파일을 완성본으로 오인하지 않게
        # 쪼개진 db3 도 기록 시각 순으로, 한 장씩(긴 에피소드 JPEG 전부를 메모리에 올리지 않는다)
        for ts, data in heapq.merge(*(c.execute(q, (COLOR,)) for c in bag.cons), key=lambda r: r[0]):
            msg = deserialize_message(data, CompressedImage)
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if not writers:
                for p in parts:
                    writers.append(cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1]))
                    if not writers[-1].isOpened():
                        raise OverlayError(f"{p}: VideoWriter 를 열지 못했다")
            clock = datetime.datetime.fromtimestamp(ts / 1e9).strftime("%H:%M:%S.%f")[:-4]
            _label(img, f"{clock}  +{(ts - t_first) / 1e9:6.1f}s")
            writers[0].write(img)
            if boxes:                                      # 원본을 쓴 뒤 같은 장에 그린다(자세 · 영상 둘 다 header stamp)
                try:
                    draw_boxes(img, boxes, K, D, stamp_ns(msg.header))
                except Exception as e:                     # 그리기 오류는 그 장만 상자 없이
                    print(f"[camera_bag_to_mp4] 상자 그리기 실패 {ts}: {type(e).__name__}: {e}", file=sys.stderr)
                writers[1].write(img)
            written += 1
    except Exception:                                      # 리뷰 M: 깨진 db3 · 해독 오류에 .part 를 남기지 않는다
        for p in parts:
            p.unlink(missing_ok=True)
        raise
    finally:
        for w in writers:
            w.release()
        for c in bag.cons:                                 # 리뷰 M: sqlite 연결을 닫는다
            c.close()
    if not written:
        for p in parts:
            p.unlink(missing_ok=True)
        raise OverlayError(f"{bag_dir}: 컬러 프레임 {n} 개를 하나도 풀지 못했다")
    for p, o in zip(parts, outs):
        p.replace(o)
    return outs, written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path, help="camera bag 폴더(<시각>_episode_<이름>/camera)")
    ap.add_argument("--out", type=Path, help="기본 <bag 폴더>/../camera.mp4")
    ap.add_argument("--fps", type=float, help="기본 = 기록 빈도")
    a = ap.parse_args(argv)
    try:
        outs, n = convert(a.bag, a.out, a.fps)
    except (OverlayError, sqlite3.DatabaseError) as e:
        print(f"camera_bag_to_mp4: {e}", file=sys.stderr)
        return 1
    print(f"{' · '.join(map(str, outs))}: {n} 프레임")
    return 0


if __name__ == "__main__":
    sys.exit(main())
