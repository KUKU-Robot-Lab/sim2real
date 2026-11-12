#!/usr/bin/env python3
"""D435i JSON-lines helper: snap returns PNG/NPY base64; no remote run files needed."""
import argparse
import base64
import io
import json
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from sym_head import SessionEnded, cleanup_signals, watch_stdin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    a = ap.parse_args()
    pipe, started = None, False
    try:
        requests = watch_stdin()
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(a.serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        prof = pipe.start(cfg)
        started = True
        dev = prof.get_device()
        scale = dev.first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipe.wait_for_frames(1000)
        print(json.dumps({"serial": dev.get_info(rs.camera_info.serial_number), "name": dev.get_info(rs.camera_info.name),
                          "fw": dev.get_info(rs.camera_info.firmware_version), "depth_scale": scale}), flush=True)
        while True:
            request = requests.get()
            if request["cmd"] == "close":
                break
            if request["cmd"] != "snap":
                raise ValueError(f"unknown command: {request}")
            for _ in range(5):
                pipe.wait_for_frames(1000)
            f = align.process(pipe.wait_for_frames(1000))
            c, d = f.get_color_frame(), f.get_depth_frame()
            i = c.profile.as_video_stream_profile().get_intrinsics()
            color = np.asanyarray(c.get_data()).copy()
            depth = np.asanyarray(d.get_data()).astype(np.float32) * scale
            ok, png = cv2.imencode(".png", color)
            if not ok:
                raise OSError("png encoding failed")
            npy = io.BytesIO()
            np.save(npy, depth, allow_pickle=False)
            print(json.dumps({"color_b64": base64.b64encode(png).decode("ascii"),
                              "depth_b64": base64.b64encode(npy.getvalue()).decode("ascii"), "depth_scale": scale,
                              "intrinsics": {"w": i.width, "h": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx,
                                             "ppy": i.ppy, "model": str(i.model), "coeffs": list(i.coeffs)},
                              "t": time.time(), "t_str": time.strftime("%F %T"), "frame_ts_ms": f.get_timestamp(),
                              "depth_valid_frac": float((depth > 0).mean())}), flush=True)
    except SessionEnded:
        pass
    except Exception as e:
        print(json.dumps({"error": repr(e)}), flush=True)
        return 1
    finally:
        cleanup_signals()
        if started and pipe is not None:
            pipe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
