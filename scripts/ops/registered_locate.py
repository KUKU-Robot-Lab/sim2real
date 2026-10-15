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
        pixel = cv2.cvtColor(np.array([[[int(spec.hue), 255, 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        color = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        camera = fakes.FakeCamera(head, truth, spec, color)
    else:
        busy = adapters.busy_head_commanders()
        if busy:
            raise RuntimeError('another head commander is active: ' + '; '.join(busy))
        head, camera = adapters.Sim2RealHead(args.repo), adapters.SshCamera()
    with cooperative_cancel() as cancelled:
        result = run_scan(request, spec, head, camera, return_pose=return_pose, profile=profile, should_stop=cancelled)
        stopped = cancelled()
    valid = result.found and result.home_ok and not stopped
    print(json.dumps({'object_id': spec.name, 'source': 'locate', 'frame': 'base', 'timestamp': time.time(),
                      'position': [round(float(v), 4) for v in (result.x, result.y, result.z)] if valid else None,  # m, 0.1 mm 'valid': valid,
                      'reason': 'cancelled' if stopped else result.reason if not valid else '', 'home_ok': result.home_ok}, allow_nan=False))
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
