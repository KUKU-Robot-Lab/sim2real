#!/usr/bin/env python3
"""실기 에피소드 머리 카메라 bag → mp4(11.18 사용자 "이제는 카메라 영상 남겨줘").

    python3 scripts/analysis/camera_bag_to_mp4.py logs/runs/<YYMMDD>/bags/<시각>_episode_<이름>/camera [--out …] [--fps …]
출력 기본 = <bag 폴더>/../camera.mp4. 컬러 JPEG(/camera/camera/color/image_raw/compressed)만 쓰고 깊이는 뺀다.
프레임 왼쪽 위 = 기록 시각(현지 HH:MM:SS.ss)과 첫 프레임부터 초. fps 기본 = 기록 빈도((메시지 수 - 1) / 길이).
에피소드 끝에 미션(_chain_camera_bag) · rh56f1_record.sh stop 이 nice 19 로 부른다. 기록 중(metadata.yaml 없음)이거나
컬러 프레임이 없으면 종료 코드 1.
"""
from __future__ import annotations

import argparse
import datetime
import heapq
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_video_overlay import COLOR, Bag, OverlayError  # noqa: E402

cv2.setNumThreads(1)   # 카메라 PC 에서 FP++ 옆에 돈다 — 코어를 넓게 잡지 않는다


def _label(img: np.ndarray, text: str) -> None:
    for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 2)):   # 검은 테두리 위 흰 글자
        cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)


def convert(bag_dir: Path, out: Path | None = None, fps: float | None = None) -> tuple[Path, int]:
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
    part = out.with_name(out.stem + ".part" + out.suffix)   # 중간에 끊긴 파일을 완성본으로 오인하지 않게
    writer, written = None, 0
    try:
        # 쪼개진 db3 도 기록 시각 순으로, 한 장씩(긴 에피소드 JPEG 전부를 메모리에 올리지 않는다)
        for ts, data in heapq.merge(*(c.execute(q, (COLOR,)) for c in bag.cons), key=lambda r: r[0]):
            msg = deserialize_message(data, CompressedImage)
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if writer is None:
                writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
                if not writer.isOpened():
                    raise OverlayError(f"{part}: VideoWriter 를 열지 못했다")
            clock = datetime.datetime.fromtimestamp(ts / 1e9).strftime("%H:%M:%S.%f")[:-4]
            _label(img, f"{clock}  +{(ts - t_first) / 1e9:6.1f}s")
            writer.write(img)
            written += 1
    except Exception:                                      # 리뷰 M: 깨진 db3 · 해독 오류에 .part 를 남기지 않는다
        part.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.release()
        for c in bag.cons:                                 # 리뷰 M: sqlite 연결을 닫는다
            c.close()
    if not written:
        part.unlink(missing_ok=True)
        raise OverlayError(f"{bag_dir}: 컬러 프레임 {n} 개를 하나도 풀지 못했다")
    part.replace(out)
    return out, written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path, help="camera bag 폴더(<시각>_episode_<이름>/camera)")
    ap.add_argument("--out", type=Path, help="기본 <bag 폴더>/../camera.mp4")
    ap.add_argument("--fps", type=float, help="기본 = 기록 빈도")
    a = ap.parse_args(argv)
    try:
        out, n = convert(a.bag, a.out, a.fps)
    except (OverlayError, sqlite3.DatabaseError) as e:
        print(f"camera_bag_to_mp4: {e}", file=sys.stderr)
        return 1
    print(f"{out}: {n} 프레임")
    return 0


if __name__ == "__main__":
    sys.exit(main())
