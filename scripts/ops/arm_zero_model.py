"""Read-only URDF view, FK/gravity and conservative swept mesh AABB checks."""
from __future__ import annotations

import copy
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from arm_zero_calibrate import Abort, ROOT, REFERENCES, Settings, digest, vector, reference


def required(root: ET.Element, path: str) -> ET.Element:
    result = root.find(path)
    if result is None:
        raise Abort(f'URDF missing {path}')
    return result


def numbers(text, default='0 0 0'):
    return vector([float(v) for v in (text or default).split()], 'xyz', 3)


def rotation(axis, angle):
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def origin(element):
    out = np.eye(4)
    if element is not None:
        r, p, y = numbers(element.get('rpy'))
        out[:3, :3] = rotation(np.array([0, 0, 1]), y) @ rotation(np.array([0, 1, 0]), p) @ rotation(np.array([1, 0, 0]), r)
        out[:3, 3] = numbers(element.get('xyz'))
    return out


class Model:
    def __init__(self, cfg, side, hand):
        self.cfg, self.side, self.hand = cfg, side, hand
        path = (ROOT / cfg['model']['urdf']).resolve()
        self.source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tree = copy.deepcopy(ET.parse(path).getroot())
        remove_at = cfg['model']['remove_at']
        if set(remove_at) != {'r_hj_mount', 'l_hj_mount'} or cfg['model']['physical_removal'] != 'rh56f1_and_adaptor':
            raise Abort('removal boundary must match RH56F1 + ADAPTOR at bare L7')
        self.removed = set()
        if hand == 'detached':
            for boundary in remove_at:
                joint = self.tree.find(f"joint[@name='{boundary}']")
                if joint is None or required(joint, 'parent').attrib['link'] != boundary[0] + '_al_7':
                    raise Abort('URDF removal boundary mismatch')
                self.removed.add(required(joint, 'child').attrib['link'])
            while True:
                children = {required(j, 'child').attrib['link'] for j in self.tree.findall('joint')
                            if required(j, 'parent').attrib['link'] in self.removed}
                if children <= self.removed:
                    break
                self.removed |= children
            for element in list(self.tree):
                if ((element.tag == 'link' and element.get('name') in self.removed)
                        or (element.tag == 'joint' and required(element, 'child').attrib['link'] in self.removed)):
                    self.tree.remove(element)
        # This calibration view is explicitly arms/body only. Head geometry and
        # joints remain in the source asset, outside this tool's scope.
        head_mount = self.tree.find("joint[@name='head_j_mount']")
        excluded = set()
        if head_mount is not None:
            excluded.add(required(head_mount, 'child').attrib['link'])
            while True:
                children = {required(j, 'child').attrib['link'] for j in self.tree.findall('joint')
                            if required(j, 'parent').attrib['link'] in excluded}
                if children <= excluded:
                    break
                excluded |= children
            for element in list(self.tree):
                if ((element.tag == 'link' and element.get('name') in excluded)
                        or (element.tag == 'joint' and required(element, 'child').attrib['link'] in excluded)):
                    self.tree.remove(element)
        self.links = {e.get('name'): e for e in self.tree.findall('link')}
        joints = self.tree.findall('joint')
        roots = set(self.links) - {required(j, 'child').attrib['link'] for j in joints}
        if len(roots) != 1:
            raise Abort('URDF is not one connected tree')
        self.root = roots.pop()
        self.joints = []
        reached = {self.root}
        while joints:
            ready = [j for j in joints if required(j, 'parent').attrib['link'] in reached]
            if not ready:
                raise Abort('URDF contains disconnected/cyclic joints')
            for j in ready:
                self.joints.append(j)
                reached.add(required(j, 'child').attrib['link'])
                joints.remove(j)
        # Use the concrete filters exported for this exact asset. Do not expand
        # wildcard audit rules (keep_collision entries represent real contact).
        self.filter_source = path.with_name(path.stem + '_manifest.yaml')
        manifest = yaml.safe_load(self.filter_source.read_text())
        if manifest.get('asset') != path.stem:
            raise Abort('collision manifest does not match URDF asset')
        pairs = manifest.get('self_collision_filtered_pairs')
        if not isinstance(pairs, list):
            raise Abort('asset manifest missing collision filter list')
        self.asset_exclusions = set()
        for pair in pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise Abort('invalid manifest collision filter pair')
            if all(link in self.links for link in pair):
                self.asset_exclusions.add(frozenset(pair))
        # The existing OpenArm v2 MoveIt SRDF disables collisions across arm
        # joints (Adjacent). Resolve the same topology with this URDF's names.
        self.adjacent_exclusions = {
            frozenset((required(j, 'parent').attrib['link'], required(j, 'child').attrib['link']))
            for j in self.joints
            if '_al_' in required(j, 'parent').attrib['link'] and '_al_' in required(j, 'child').attrib['link']
        }
        self.colliders = None
        self.bounds = None
        self.names = [f'{side[0]}_aj_{i}' for i in range(1, 8)]
        self.limits = np.array([[float(required(self.tree, f"joint[@name='{n}']/limit").attrib[k])
                                for k in ('lower', 'upper')] for n in self.names])
        self.mass = {}
        for name, link in self.links.items():
            inertial = link.find('inertial')
            if inertial is not None:
                self.mass[name] = (float(required(inertial, 'mass').attrib['value']), origin(inertial.find('origin'))[:3, 3])
        self.payloads = cfg['model']['adaptor_payloads'] if hand == 'attached' else {}

    def poses(self, q):
        values = dict(self.cfg['model']['finger_positions'] or {})
        other = 'left' if self.side == 'right' else 'right'
        oq = self.cfg['sides'][self.side]['other_arm_q']
        values.update({f'{other[0]}_aj_{i + 1}': v for i, v in enumerate(oq or [0] * 7)})
        values.update(zip(self.names, vector(q, 'q')))
        poses, axes = {self.root: np.eye(4)}, {}
        for j in self.joints:
            name = j.get('name')
            transform = poses[required(j, 'parent').attrib['link']] @ origin(j.find('origin'))
            if j.get('type') not in ('fixed', 'revolute', 'continuous'):
                raise Abort('unsupported joint type')
            if j.get('type') != 'fixed':
                angle = values.get(name, 0.)
                mimic = j.find('mimic')
                if mimic is not None:
                    angle = values.get(mimic.get('joint'), 0.) * float(mimic.get('multiplier', 1)) + float(mimic.get('offset', 0))
                values[name] = angle
                axis = numbers(required(j, 'axis').attrib['xyz'], '1 0 0')
                axes[name] = (transform[:3, 3].copy(), transform[:3, :3] @ axis)
                turn = np.eye(4)
                turn[:3, :3] = rotation(axis, angle)
                transform = transform @ turn
            poses[required(j, 'child').attrib['link']] = transform
        return poses, axes

    def descendants(self, link):
        result = {link}
        for j in self.joints:
            if required(j, 'parent').attrib['link'] in result:
                result.add(required(j, 'child').attrib['link'])
        return result

    def gravity(self, q):
        poses, axes = self.poses(q)
        result = []
        for name in self.names:
            point, axis = axes[name]
            child = required(self.tree, f"joint[@name='{name}']/child").attrib['link']
            torque = 0.
            for link in self.descendants(child):
                items = [self.mass[link]] if link in self.mass else []
                if link in (self.payloads or {}):
                    p = self.payloads[link]
                    items.append((float(p['mass']), vector(p['com'], 'adaptor COM', 3)))
                for mass, com in items:
                    world = poses[link][:3, :3] @ com + poses[link][:3, 3]
                    torque -= axis @ np.cross(world - point, np.array([0., 0., -9.80665 * mass]))
            result.append(torque)
        return np.array(result)

    def flange(self, q):
        return self.poses(q)[0][f'{self.side[0]}_al_7']

    def load_bounds(self):
        import trimesh
        self.bounds = {}
        try:
            import importlib
            fcl = importlib.import_module('fcl')
        except ImportError:
            self.colliders = None  # conservative box fallback; never waive overlap
        else:
            self.fcl = fcl
            self.colliders = {}
        for name, link in self.links.items():
            vertices = []
            for c in link.findall('collision'):
                mesh = c.find('geometry/mesh')
                if mesh is None:
                    raise Abort(f'{name}: unsupported non-mesh collision')
                path = mesh.attrib['filename'].removeprefix('file://')
                geometry = trimesh.load(path, force='mesh', process=False)
                if not isinstance(geometry, trimesh.Trimesh):
                    raise Abort('collision mesh could not be loaded')
                v = np.asarray(geometry.vertices) * numbers(mesh.get('scale'), '1 1 1')
                t = origin(c.find('origin'))
                vertices.append(v @ t[:3, :3].T + t[:3, 3])
            payload = (self.payloads or {}).get(name)
            if payload is not None:
                lo, hi = vector(payload['box_lower'], 'adaptor lower', 3), vector(payload['box_upper'], 'adaptor upper', 3)
                vertices.append(np.array([[x, y, z] for x in (lo[0], hi[0])
                                          for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
            if vertices:
                v = np.vstack(vertices)
                if self.colliders is not None:
                    # Convex hull contains the source mesh, so refining a broad
                    # AABB pair cannot miss intersections of the modeled geometry.
                    hull = trimesh.convex.convex_hull(v)
                    faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).flatten()
                    shape = self.fcl.Convex(np.asarray(hull.vertices), len(hull.faces), faces)
                    self.colliders[name] = self.fcl.CollisionObject(shape)
                lo, hi = v.min(0), v.max(0)
                self.bounds[name] = np.array([[x, y, z] for x in (lo[0], hi[0])
                                             for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    def guard(self, q, limit_slack=0.):
        q = vector(q, 'model q')
        if np.any(q < self.limits[:, 0] - limit_slack - .000001) or np.any(q > self.limits[:, 1] + limit_slack + .000001):
            raise Abort('model joint limit exceeded')
        if self.bounds is None:
            self.load_bounds()
        assert self.bounds is not None
        poses = self.poses(q)[0]
        boxes = {}
        for name, points in self.bounds.items():
            t = poses[name]
            p = points @ t[:3, :3].T + t[:3, 3]
            boxes[name] = (p.min(0), p.max(0))
        if self.colliders is not None:
            for name, collider in self.colliders.items():
                collider.setTransform(self.fcl.Transform(poses[name][:3, :3], poses[name][:3, 3]))
        exclusions = self.asset_exclusions | self.adjacent_exclusions | {frozenset(p) for p in self.cfg['geometry']['exclusions']}
        margin = float(self.cfg['geometry']['margin_m'])
        active = self.descendants(f'{self.side[0]}_al_2')
        for a in active & boxes.keys():
            alo, ahi = boxes[a]
            for b, (blo, bhi) in boxes.items():
                if a == b or frozenset((a, b)) in exclusions:
                    continue
                if np.all(ahi + margin >= blo) and np.all(bhi + margin >= alo):
                    if (self.colliders is not None
                            and a in self.colliders and b in self.colliders):
                        distance = self.fcl.distance(self.colliders[a], self.colliders[b],
                                                     self.fcl.DistanceRequest(), self.fcl.DistanceResult())
                        # Own-arm links have intentional small assembly gaps
                        # (e.g. al_2/al_0: 10.9mm at nominal zero). Do not apply
                        # the 20mm external/inter-arm margin to these gaps.
                        own = a.startswith(self.side[0] + '_al_') and (b.startswith(self.side[0] + '_al_') or b == 'body_link')
                        threshold = float(self.cfg['geometry'].get('self_margin_m', margin)) if own else margin
                        if distance > max(1e-6, threshold):
                            continue
                    raise Abort(f'conservative mesh boxes overlap: {a}/{b}')
            for obstacle in self.cfg['geometry']['obstacles'] or []:
                lo, hi = vector(obstacle['lower'], 'obstacle lower', 3), vector(obstacle['upper'], 'obstacle upper', 3)
                if np.any(lo >= hi):
                    raise Abort('invalid obstacle box')
                if np.all(ahi + margin >= lo) and np.all(hi + margin >= alo):
                    raise Abort(f'obstacle clearance: {a}/{obstacle["name"]}')


def prepare(cfg, side, hand, command=None, joint=None):
    """command: measure/apply/verify 이면 그 명령이 실제로 가는 경로만 검사한다(None = 전부, 보수적)."""
    if cfg['version'] != 1 or cfg['sides'][side]['reference_deg'] != list(REFERENCES[side]):
        raise Abort('configuration reference/version mismatch')
    if cfg['sides'][side]['can'] != {'right': 'can0', 'left': 'can1'}[side]:
        raise Abort('CAN side mismatch')
    model = Model(cfg, side, hand)
    auto = bool(cfg['operation'].get('auto_profile'))
    pending = auto and cfg['operation']['start_raw'] is None
    blockers = []
    requirements = [('model', 'bare_l7_audit'), ('search', 'torque_review'),
                ('operation', 'session'), ('operation', 'supported_on_disable'),
                ('operation', 'table_and_cables_clear'), ('operation', 'can_ownership_review'),
                ('operation', 'no_external_zero_writes'), ('operation', 'route_review'),
                ('operation', 'measure_routes'), ('operation', 'verify_route'), ('geometry', 'review')]
    if hand == 'attached':
        requirements += [('model', 'attached_payload_audit'), ('model', 'finger_positions'), ('model', 'adaptor_payloads')]
    for section, key in requirements:
        if pending and section == 'operation' and key == 'measure_routes':
            continue
        if not cfg[section].get(key):
            blockers.append(f'{section}.{key}: unverified')
    measurement_routes = cfg['operation']['measure_routes'] or {}
    if not pending and set(map(str, measurement_routes)) != set(map(str, range(1, 8))):
        blockers.append('measure_routes must contain all seven preparation routes')
    for key, route in measurement_routes.items():
        if not route:
            blockers.append(f'empty preparation route: {key}')
        for waypoint in route:
            vector(waypoint, 'measurement waypoint')
    verification_route = cfg['operation']['verify_route'] or []
    if len(verification_route) < 2:
        blockers.append('verify_route requires multiple poses')
    for waypoint in verification_route:
        vector(waypoint, 'verification waypoint')
    if cfg['operation']['hand_state'] != hand:
        blockers.append('operation.hand_state does not match requested hand state')
    for key, val in [('other_arm_q', cfg['sides'][side]['other_arm_q']),
                     ('start_raw', cfg['operation']['start_raw'])]:
        if val is None and not pending:
            blockers.append(f'{key}: missing')
        elif val is None:
            pass
        else:
            vector(val, key)
    if cfg['geometry']['obstacles'] is None:
        blockers.append('geometry.obstacles: missing surveyed environment (empty list requires review)')
    try:
        Settings(**{k: v for k, v in cfg['search'].items() if k != 'torque_review'})
    except (Abort, TypeError) as exc:
        blockers.append(f'search settings: {exc}')
    if hand == 'attached':
        fingers = cfg['model']['finger_positions'] or {}
        for j in model.joints:
            name = j.attrib['name']
            if '_hj_' in name and j.get('type') != 'fixed' and j.find('mimic') is None:
                if name not in fingers or not np.isfinite(fingers[name]):
                    blockers.append(f'missing/nonfinite actual finger angle: {name}')
                else:
                    lim = required(j, 'limit')
                    if not float(lim.attrib['lower']) <= fingers[name] <= float(lim.attrib['upper']):
                        blockers.append(f'finger outside URDF limit: {name}')
        for link in ('r_al_7', 'l_al_7'):
            payload = (cfg['model']['adaptor_payloads'] or {}).get(link)
            try:
                if payload is None or not np.isfinite(payload['mass']) or payload['mass'] < 0:
                    raise Abort('missing/invalid mass')
                vector(payload['com'], 'adaptor COM', 3)
                lo, hi = vector(payload['box_lower'], 'adaptor lower', 3), vector(payload['box_upper'], 'adaptor upper', 3)
                if np.any(lo >= hi):
                    raise Abort('invalid adaptor geometry')
            except (Abort, KeyError) as exc:
                blockers.append(f'{link} adaptor: {exc}')
    self_margin = cfg['geometry'].get('self_margin_m', cfg['geometry']['margin_m'])
    if not np.isfinite(self_margin) or self_margin < .002:
        raise Abort('own-arm modeled assembly clearance must be at least 0.002m')
    if cfg['geometry']['margin_m'] < .02 or not np.isfinite(cfg['geometry']['margin_m']):
        raise Abort('collision margin must be finite and at least 0.02m')
    # Even a blocked plan inspects every provided route with a complete environment.
    # (waypoint, limit_slack): 보정 전 좌표(measure/apply)는 런타임 guard 와 같은 max_offset 여유를 준다. verify 는 보정 뒤 좌표라 여유 없음.
    slack = float(cfg['search']['max_offset'])
    paths = []
    use_measure = command in (None, 'measure', 'apply')
    for jt, routes in (cfg['operation']['measure_routes'] or {}).items():
        if not use_measure or (command == 'apply' and joint is not None and int(jt) != joint):
            continue
        paths.extend((w, slack) for w in routes)
        if auto and routes:
            hit = list(routes[-1])
            hit[int(jt) - 1] = reference(side, int(jt))
            paths.append((hit, slack))
            backed = list(hit)
            direction = -1 if int(jt) == 4 or (side == 'right' and int(jt) in (1, 2)) else 1
            backed[int(jt) - 1] -= direction * cfg['search']['backoff']
            paths.append((backed, slack))
    if command in (None, 'measure', 'verify'):
        paths.extend((w, 0. if command == 'verify' else slack) for w in (cfg['operation']['verify_route'] or []))
    path_checked = 0
    if paths and cfg['sides'][side]['other_arm_q'] is not None and cfg['geometry']['obstacles'] is not None:
        try:
            for (a, sa), (b, sb) in zip(paths, paths[1:]):
                a, b = vector(a, 'waypoint'), vector(b, 'waypoint')
                for t in np.linspace(0, 1, max(2, int(np.max(np.abs(b - a)) / .005) + 2)):
                    model.guard(a + (b - a) * t, limit_slack=max(sa, sb))
                    path_checked += 1
        except (Abort, OSError, ValueError) as exc:
            blockers.append(f'path: {exc}')
    return model, {'ready': not blockers and not pending, 'requires_live_snapshot': pending, 'side': side, 'hand': hand,
                   'model_hash': model.source_hash, 'config_hash': digest(cfg),
                   'scope': 'arms_and_body',
                   'collision_filter_source': str(model.filter_source),
                   'asset_collision_filter_count': len(model.asset_exclusions),
                   'removed_links': sorted(model.removed),
                   'l7_mass_kg': {s: model.mass[f'{s}_al_7'][0] for s in ('r', 'l')},
                   'path_samples_checked': path_checked, 'blockers': blockers}
