#!/usr/bin/env python3
"""컵홀더 자세 자동 추정 노드 — 머리 카메라 컬러 영상 → /objects/<holder>/pose (base_link).

  python3 scripts/nodes/cup_holder_pose_node.py                          # config/cup_holders.yaml
  python3 scripts/nodes/cup_holder_pose_node.py --write                  # 안정되면 config/cup_holder_poses_arm4090.yaml 갱신
  python3 scripts/nodes/cup_holder_pose_node.py --write --timeout-s 30   # ★11.10 한 번 재기: 안정되면 1회 쓰고 끝(cups 단계)

★11.10 한 번 재기(--timeout-s N): 세 홀더가 안정되면 --out 의 직전 값과 홀더별 xy 차를 보고 --max-jump-m(기본 0.03) 안이면
  쓰고(written), 넘으면 쓰지 않는다(jump_rejected). N 초 안에 안정 못 하면 쓰지 않고 끝(unstable · 한 번도 못 보면 no_markers).
  결과는 한 줄 로그 + <out>.last_check.json(시각 · 결과 · 홀더별 dxy · dyaw). 종료 코드는 늘 0 — cups 를 막지 않는다.
  ★11.11: 결과가 written 이 아니면(또는 기록이 낡으면) 홀더를 쓰는 에피소드 시작 검사(episode_run --require-holders)가 막는다.
  계속 재기(--timeout-s 0, cup_holders 단계)도 쓸 때마다 written 을 남긴다.
  점프 검사는 이 모드에서만 — 계속 쓰는 --write(cup_holders 단계)는 홀더를 옮긴 뒤 다시 재는 용도라 점프를 막지 않는다.

하는 일(순수부는 scripts/calib/cup_holder_tracker.py):
  · period 마다 최신 컬러 한 장으로 추정한다 — ArUco → 직전 자세 추적(0.2 s) → 놓친 id 만 전체 무늬 탐색(~6 s).
    추정은 작업 스레드에서 돌아 영상 콜백을 막지 않는다. 한 번에 하나만 돈다.
  · 홀더별 최근 window 장 중앙값을 PoseStamped 로 낸다(yaw 만, roll·pitch 0). latched(transient_local)라
    늦게 붙은 구독자도 마지막 값을 받는다. 연속 lost_after 장 못 보면 그 홀더는 내지 않는다.
  · /cup_holders/status (std_msgs/String JSON): ok(세 홀더 모두 안정) · 홀더별 src(aruco|track|template) · 폭 · rms · NCC.

⚠ 외부 파라미터는 머리 head_home_rh56f1 자세 한 장에서 맞춘 정적 값이다(arm4090 에는 머리 관절 토픽이 없어 노드가
  확인하지 못한다). head_home 뒤에 띄운다. 로봇을 움직이지 않는다 — 카메라 영상만 읽는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "calib"))
sys.path.insert(0, str(_HERE.parent))

from cup_holder_pose import DEFAULT_CFG, DEFAULT_OUT, load_cfg, write_poses, yaw_quat_wxyz  # noqa: E402
from cup_holder_tracker import HolderTracker  # noqa: E402
from cup_pose_relay import load_extrinsics  # noqa: E402
from object_registry import output_topic  # noqa: E402
from table_cad_extrinsics import TABLE_TOP_Z, T_from  # noqa: E402

STATUS_TOPIC = "/cup_holders/status"
COLOR_TOPIC = "/camera/camera/color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"
#: --write 는 직전 기록에서 이만큼 바뀌었을 때만 다시 쓴다
REWRITE_XY_M = 0.003
REWRITE_YAW_RAD = np.radians(2.0)


def image_to_gray(encoding: str, height: int, width: int, data: bytes) -> np.ndarray:
    import cv2
    buf = np.frombuffer(data, np.uint8)
    if encoding in ("rgb8", "bgr8"):
        img = buf.reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if encoding == "rgb8" else cv2.COLOR_BGR2GRAY)
    if encoding in ("mono8", "8UC1"):
        return buf.reshape(height, width).copy()
    raise ValueError(f"영상 encoding {encoding} 미지원(rgb8 · bgr8 · mono8)")


def should_rewrite(last: dict[str, np.ndarray] | None, now: dict[str, np.ndarray]) -> bool:
    if last is None or set(last) != set(now):
        return True
    for n, p in now.items():
        q = last[n]
        dyaw = abs((p[3] - q[3] + np.pi) % (2 * np.pi) - np.pi)
        if np.linalg.norm(p[:2] - q[:2]) > REWRITE_XY_M or dyaw > REWRITE_YAW_RAD:
            return True
    return False


def jump_check(prev: dict[str, np.ndarray] | None, now: dict[str, np.ndarray], max_jump_m: float,
               max_yaw_deg: float = 10.0) -> tuple[bool, dict]:
    """★11.10 직전 파일 값(prev, 없으면 비교 없이 통과)과 홀더별 xy · yaw 차 — (모두 max_jump_m · max_yaw_deg 안인가,
    {홀더: {dxy_m, dyaw_deg}}). 직전 파일에 없는 홀더는 비교하지 않는다(dxy None). ★11.11 deploy 리뷰 H2: yaw 상한도 본다."""
    diffs = {}
    for n, p in now.items():
        q = (prev or {}).get(n)
        if q is None:
            diffs[n] = {"dxy_m": None, "dyaw_deg": None}
            continue
        dyaw = (p[3] - q[3] + np.pi) % (2 * np.pi) - np.pi
        diffs[n] = {"dxy_m": round(float(np.linalg.norm(p[:2] - q[:2])), 4), "dyaw_deg": round(float(np.degrees(dyaw)), 2)}
    ok = all(d["dxy_m"] is None or (d["dxy_m"] <= max_jump_m and abs(d["dyaw_deg"]) <= max_yaw_deg) for d in diffs.values())
    return ok, diffs


def load_written(path: Path) -> dict[str, np.ndarray] | None:
    """write_poses 가 쓴 파일 → {홀더: [x, y, z, yaw]}(없으면 None)."""
    import yaml
    if not path.exists():
        return None
    hs = (yaml.safe_load(path.read_text()) or {}).get("holders") or {}
    return {n: np.array([*h["position"], h["yaw_rad"]], float) for n, h in hs.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--extrinsics", type=Path, default=None, help="기본: cfg 의 camera_extrinsics")
    ap.add_argument("--period", type=float, default=1.0, help="추정 주기 s (추정이 더 오래 걸리면 그만큼 늦어진다)")
    ap.add_argument("--window", type=int, default=5, help="중앙값 · 안정 판정 장 수")
    ap.add_argument("--lost-after", type=int, default=5, help="연속 이 장 수만큼 못 보면 그 홀더를 내지 않는다")
    ap.add_argument("--table-z", type=float, default=TABLE_TOP_Z)
    ap.add_argument("--no-template", action="store_true", help="ArUco 만 쓴다(마커를 흰 여백으로 다시 붙인 뒤)")
    ap.add_argument("--write", action="store_true", help=f"세 홀더가 안정되면 {DEFAULT_OUT.name} 를 갱신")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--timeout-s", type=float, default=0.0, help="★11.10 한 번 재기(0 = 계속): 안정되면 1회 쓰고 끝, 시간 안에 못 하면 안 쓰고 끝")
    ap.add_argument("--max-jump-m", type=float, default=0.03, help="한 번 재기에서 직전 파일과 홀더별 xy 차 상한(넘으면 쓰지 않음)")
    ap.add_argument("--max-yaw-deg", type=float, default=10.0, help="★11.11 한 번 재기에서 직전 파일과 홀더별 yaw 차 상한°(넘으면 쓰지 않음)")
    args = ap.parse_args(argv)
    once = args.timeout_s > 0
    check_path = Path(str(args.out) + ".last_check.json")
    done, fin = threading.Event(), threading.Lock()     # fin — 작업 스레드 기록과 시간 초과 기록이 겹치지 않게
    seen = [False]      # 한 번 재기 — 한 홀더라도 본 적 있나(시간 초과 때 unstable / no_markers 가르기)

    def record(result: str, diffs: dict) -> None:
        # ★11.11 deploy 리뷰 M1 · M2: 시작 검사는 epoch(시간대 없음)로 나이를 잰다 — epoch 가 없는 기록은 옛 노드가 쓴 것으로 막는다
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "epoch": round(time.time(), 3), "result": result, "out": str(args.out),
               "max_jump_m": args.max_jump_m, "max_yaw_deg": args.max_yaw_deg, "holders": diffs}
        check_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
        print(f"[cup_holder_check] {result} · " + (" · ".join(
            f"{n} dxy {d['dxy_m'] if d['dxy_m'] is not None else '—'} m dyaw {d['dyaw_deg'] if d['dyaw_deg'] is not None else '—'}°"
            for n, d in diffs.items()) or "홀더 없음") + f" → {check_path.name}", flush=True)
        if result != "written":     # 콘솔 runner 가 '⚠ ' 줄을 성공 단계에도 경고로 띄운다
            last = (time.strftime("%m-%d %H:%M", time.localtime(args.out.stat().st_mtime)) if args.out.exists() else "없음")
            big = " · ".join(f"{n} dxy {d['dxy_m']} m" for n, d in diffs.items() if d["dxy_m"] is not None)
            print(f"⚠ 홀더 다시 재기 {result} — {args.out.name} 안 바꿈(마지막으로 쓴 시각 {last})" + (f" · {big}" if big else "")
                  + (" · 홀더가 옮겨졌으면 cup_holders 단계로 다시 잴 것" if result == "jump_rejected" else "")
                  + " · 홀더를 쓰는 에피소드 시작 검사가 막는다", flush=True)

    cfg = load_cfg(args.cfg)
    ext_path = args.extrinsics or cfg.extrinsics
    ext = load_extrinsics(ext_path)
    tracker = HolderTracker(cfg, T_from(ext.cam_pos, ext.cam_quat), table_z=args.table_z, window=args.window,
                            lost_after=args.lost_after, use_template=not args.no_template)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

    class CupHolderPoseNode(Node):
        def __init__(self) -> None:
            super().__init__("cup_holder_pose_node")
            self._pubs = {n: self.create_publisher(PoseStamped, output_topic(n), latched) for n in cfg.names}
            self._status = self.create_publisher(String, STATUS_TOPIC, latched)
            self.create_subscription(Image, COLOR_TOPIC, self._on_image, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, qos_profile_sensor_data)
            self._lock = threading.Lock()
            self._frame = None
            self._K = None
            self._busy = False
            self._written: dict[str, np.ndarray] | None = None
            self.create_timer(args.period, self._tick)
            self.get_logger().info(
                f"홀더 {list(cfg.names)} → {[output_topic(n) for n in cfg.names]} · 외부 파라미터 {ext_path.name} · "
                f"공유 축 {list(cfg.shared)} · {'ArUco 만' if args.no_template else 'ArUco + 무늬 맞춤'}")

        def _on_info(self, msg) -> None:
            self._K = np.asarray(msg.k, float).reshape(3, 3)

        def _on_image(self, msg) -> None:
            with self._lock:
                self._frame = msg

        def _tick(self) -> None:
            with self._lock:
                if self._busy or self._frame is None or self._K is None:
                    return
                msg, self._frame, self._busy = self._frame, None, True
            threading.Thread(target=self._work, args=(msg, self._K.copy()), daemon=True).start()

        def _work(self, msg, K) -> None:
            try:
                gray = image_to_gray(msg.encoding, msg.height, msg.width, bytes(msg.data))
                tracker.update(gray, K)
                self._publish(msg.header.stamp)
            except Exception as e:      # 한 장 실패로 노드가 죽지 않게 — 상태로 알린다
                tracker.last_error = f"{type(e).__name__}: {e}"
                self.get_logger().error(tracker.last_error, throttle_duration_sec=5.0)
                self._status.publish(String(data=json.dumps(tracker.status(), ensure_ascii=False)))
            finally:
                with self._lock:
                    self._busy = False

        def _write(self, poses) -> None:
            by_id = {i: poses[n] for n, i in zip(cfg.names, cfg.ids)}
            write_poses(args.out, cfg, by_id, f"cup_holder_pose_node · extrinsics={ext_path.name} · "
                                               f"shared={','.join(cfg.shared)} · window={args.window} · "
                                               f"쓴 시각 {time.strftime('%Y-%m-%dT%H:%M:%S')}")
            self._written = {n: p.copy() for n, p in poses.items()}
            self.get_logger().info(f"기록 {args.out}")

        def _publish(self, stamp) -> None:
            poses = tracker.poses()
            for name, p in poses.items():
                out = PoseStamped()
                out.header.stamp = stamp
                out.header.frame_id = ext.base_frame
                out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, p[:3])
                (out.pose.orientation.w, out.pose.orientation.x,
                 out.pose.orientation.y, out.pose.orientation.z) = yaw_quat_wxyz(float(p[3]))
                self._pubs[name].publish(out)
            st = tracker.status()
            self._status.publish(String(data=json.dumps(st, ensure_ascii=False)))
            seen[0] = seen[0] or any(h["src"] for h in st["holders"].values())
            if once and st["ok"]:
                with fin:
                    if done.is_set():
                        return
                    ok, diffs = jump_check(load_written(args.out), poses, args.max_jump_m, args.max_yaw_deg)
                    if ok and args.write:
                        self._write(poses)
                    record("written" if ok and args.write else "measured" if ok else "jump_rejected", diffs)
                    done.set()
            elif not once and args.write and st["ok"] and should_rewrite(self._written, poses):
                with fin:     # ★11.11 cup_holders 단계(계속 재기)도 결과를 남긴다 — 사람이 보고 다시 잰 값이라 시작 검사를 푼다
                    diffs = jump_check(load_written(args.out), poses, float("inf"), float("inf"))[1]
                    self._write(poses)
                    record("written", diffs)
            if tracker.frames % 10 == 1:
                self.get_logger().info(" · ".join(
                    f"{n} {h['src'] or '—'}{' ✓' if h['stable'] else ''}" for n, h in st["holders"].items())
                    + f" · {st['ms']:.0f} ms" + (f" · {st['error']}" if st["error"] else ""))

    rclpy.init()
    node = CupHolderPoseNode()
    try:
        if not once:
            rclpy.spin(node)
        else:
            end = time.monotonic() + args.timeout_s
            while not done.is_set() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
            with fin:
                if not done.is_set():
                    done.set()       # 늦게 끝난 작업 스레드가 시간 초과 뒤에 쓰지 않게
                    record("unstable" if seen[0] else "no_markers", {})
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
