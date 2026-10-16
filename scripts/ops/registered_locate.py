#!/usr/bin/env python3
"""Use the existing locate scan with the object registry and operating camera pose.

Only --execute authorizes real head movement. Output ends with a machine-readable
JSON result so the console can invalidate failed attempts instead of reusing poses.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


def install_core(repo):
    core = Path(repo).resolve().parent / 'perception_plus_plus'
    if not (core / 'perception_plus_plus_core/head_scan/locate.py').is_file():
        raise ValueError(f'locate core unavailable: {core}')
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))


def load_spec(repo, object_id):
    """Map registered mesh (container mount → host), hue and sim body origin."""
    import trimesh
    from object_registry import load_registry
    from fpp_color_pick import hue_ranges
    from perception_plus_plus_core.head_scan.locate import ObjectSpec
    repo = Path(repo).resolve()
    registered = load_registry(repo / 'config/objects.yaml').get(object_id)
    if registered.fpp.get('group') not in ('source200', 'source240', 'source240b', 'shaker_c'):
        raise ValueError('color locate unsupported for this object')
    mesh_path = Path(registered.fpp['mesh_path'])
    if mesh_path.parts[:2] == ('assets', 's2r_meshes'):
        mesh_path = Path('assets/meshes') / mesh_path.name
    mesh_path = (repo / mesh_path).resolve()
    if not mesh_path.is_relative_to(repo / 'assets'):
        raise ValueError('registered mesh must be under assets')
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('registered asset is not a triangle mesh')
    ranges = hue_ranges(registered.fpp['color'])
    if 'hue_measured' in registered.fpp:
        hue = float(registered.fpp['hue_measured'][0])
    elif len(ranges) == 1:
        hue = sum(ranges[0]) / 2
    else:
        hue = ((ranges[0][0] + ranges[-1][1] + 180) / 2) % 180
    # Registered variants are z-up with the mesh already at the sim body origin.
    # Applying CAD→body here keeps added registry models aligned with FP++ output.
    import numpy as np
    from scipy.spatial.transform import Rotation
    q = registered.cad_to_body_quat
    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    vertices = (np.asarray(mesh.vertices) * registered.fpp['mesh_scale_to_meters'] - registered.cad_to_body_pos) @ rotation
    return ObjectSpec(name=registered.name, vertices=vertices, faces=mesh.faces,
                      origin_above_bottom=registered.origin_above_bottom_m, hue=hue)


def run_scan(request, spec, head, camera, *, return_pose=(0.0, -20.0), **kwargs):
    from perception_plus_plus_core.head_scan import locate, survey
    survey.validate_poses([return_pose])

    class OperatingHead:
        def move(self, pan_deg: float, tilt_deg: float):
            return head.move(pan_deg, tilt_deg)

        def go_home(self):
            return head.move(*return_pose)

    return locate.locate_object(request, spec, OperatingHead(), camera, **kwargs)


def read_current_head(timeout_s=3.0):
    """Read a fresh, stationary encoder pose from ROS. Never publish a head command."""
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from head_fk_chain import encoder_from_urdf
    from perception_plus_plus_core.head_scan.survey import HeadReading
    rclpy.init()
    node = Node('locate_current_head')
    samples = []

    def receive(msg):
        names = list(msg.name)
        try:
            angles = [msg.position[names.index(n)] for n in ('head_j_pan', 'head_j_tilt')]
        except (ValueError, IndexError):
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not np.all(np.isfinite(angles)) or abs(time.time() - stamp) > .5:
            return
        samples.append((time.monotonic(), encoder_from_urdf(*np.degrees(angles))))

    node.create_subscription(JointState, '/head/joint_states', receive, 10)
    began = time.monotonic()
    try:
        while time.monotonic() - began < timeout_s:
            rclpy.spin_once(node, timeout_sec=.05)
            if len(samples) >= 3 and samples[-1][0] - samples[0][0] >= .2:
                values = np.asarray([angles for _, angles in samples])
                if np.any(np.ptp(values, axis=0) > .3):
                    raise RuntimeError('목이 움직이고 있다 — 정착 후 현재 자세 재추출 필요')
                pan, tilt = values.mean(axis=0)
                return HeadReading(float(pan), float(tilt), True)
        raise RuntimeError('현재 목 실측이 없다 — /head/joint_states 확인 필요')
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_current_pose(spec, camera, read_pose, **kwargs):
    """Use several frames at the measured pose, with no move or return command."""
    from perception_plus_plus_core.head_scan import locate
    initial = read_pose()

    class StationaryHead:
        def read(self):
            current = read_pose()
            if not current.ok or max(abs(current.pan_deg - initial.pan_deg),
                                     abs(current.tilt_deg - initial.tilt_deg)) > .3:
                raise RuntimeError('촬영 중 목 자세가 바뀌었다 — 현재 자세에서 다시 추출할 것')
            return current

        def move(self, pan_deg, tilt_deg):
            return self.read()

        def go_home(self):
            return self.read()

    request = locate.LocateRequest(spec.name, poses=[(initial.pan_deg, initial.tilt_deg)], n_frames=3)
    return locate.locate_object(request, spec, StationaryHead(), camera, **kwargs)


@contextmanager
def cooperative_cancel():
    """Finish the current move and let the scan restore the operating pose."""
    stopped = False

    def cancel(signum, frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield lambda: stopped
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object_id')
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--return-pan', type=float, default=0)
    parser.add_argument('--return-tilt', type=float, default=-20)
    parser.add_argument('--calib-profile', default='')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--current-pose', action='store_true', help='현재 목 실측 자세에서만 촬영·추출(목 이동 없음)')
    parser.add_argument('--fake', metavar='X,Y')
    args = parser.parse_args(argv)
    if args.calib_profile and not re.fullmatch(r'[A-Za-z0-9_-]+', args.calib_profile):
        parser.error('invalid calibration profile name')
    install_core(args.repo)
    from perception_plus_plus_core.head_scan import adapters, calib_profile, fakes, locate, survey
    spec = load_spec(args.repo, args.object_id)
    return_pose = (args.return_pan, args.return_tilt)
    survey.validate_poses([return_pose])
    profile = calib_profile.load_profile(args.repo / 'config/calib_profiles' / f'{args.calib_profile}.yaml') if args.calib_profile else None
    request = locate.LocateRequest(spec.name)
    if not args.execute and not args.fake:
        print(json.dumps({'dry_run': True, 'object_id': spec.name, 'return_pose': return_pose}))
        return 0
    if args.fake:
        import cv2
        import numpy as np
        truth = tuple(float(v) for v in args.fake.split(','))
        if len(truth) != 2 or not all(np.isfinite(truth)):
            parser.error('--fake requires finite X,Y')
        head = fakes.FakeHead()
        if args.current_pose:
            head.moves.append(return_pose)  # Synthetic camera's already-measured pose; no command is sent.
        pixel = cv2.cvtColor(np.array([[[int(spec.hue), 255, 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        color = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        camera = fakes.FakeCamera(head, truth, spec, color)
    else:
        busy = adapters.busy_head_commanders()
        if busy:
            raise RuntimeError('another head commander is active: ' + '; '.join(busy))
        head = None if args.current_pose else adapters.Sim2RealHead(args.repo)
        camera = adapters.SshCamera()
    with cooperative_cancel() as cancelled:
        if args.current_pose:
            read_pose = (lambda: survey.HeadReading(*return_pose, True)) if args.fake else read_current_head
            result = run_current_pose(spec, camera, read_pose, profile=profile, should_stop=cancelled)
        else:
            result = run_scan(request, spec, head, camera, return_pose=return_pose, profile=profile, should_stop=cancelled)
        stopped = cancelled()
    valid = result.found and result.home_ok and not stopped
    print(json.dumps({'object_id': spec.name, 'source': 'locate', 'frame': 'base', 'timestamp': time.time(),
                      'position': [round(float(v), 4) for v in (result.x, result.y, result.z)] if valid else None,  # m, 0.1 mm
                      'valid': valid,
                      'reason': 'cancelled' if stopped else result.reason if not valid else '', 'home_ok': result.home_ok}, allow_nan=False))
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
