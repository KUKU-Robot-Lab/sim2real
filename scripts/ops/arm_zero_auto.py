"""Manufacturer-derived, deterministic single-arm calibration preparation.

No device access here. All generated motion is checked before enable and again
at each control tick. A failure never changes gains, torque limits or routes.
"""
from __future__ import annotations

import copy
from datetime import datetime

import numpy as np

from arm_zero_calibrate import ORDER, Runner, reference, vector

PROFILE = 'openarm-can-1.2.9-arm-sequence-v2'


def interpolate_waypoints(start, goals, step=.25):
    current = vector(start, 'route start').copy()
    out = []
    for goal in goals:
        goal = vector(goal, 'route goal')
        count = max(1, int(np.ceil(np.max(np.abs(goal - current)) / step)))
        out.extend((current + (goal - current) * i / count).tolist() for i in range(1, count + 1))
        current = goal.copy()
    return out


def measurement_routes(start, side):
    q = vector(start, 'model starting pose').copy()
    original_j2 = float(q[1])
    direction = -1 if side == 'right' else 1
    goals = {}
    def stage(joint, changes):
        points = []
        for index, value in changes:
            q[index - 1] = value
            points.append(q.copy())
        d = -1 if joint == 4 or (side == 'right' and joint in (1, 2)) else 1
        # Manufacturer searches from the current pose: no near-stop preposition.
        if not points:
            points.append(q.copy())
        goals[joint] = points
        # Three measured approaches finish backed off .07 from the true stop.
        q[joint - 1] = reference(side, joint) - d * .07
    stage(4, [])
    stage(3, [(4, 0.), (2, original_j2 - direction * np.deg2rad(5))])
    stage(5, [(3, 0.), (2, original_j2), (4, np.pi / 2)])
    stage(6, [(5, 0.)])
    stage(7, [(6, 0.)])
    stage(2, [(7, 0.)])
    stage(1, [(2, 0.)])
    return {joint: [point.tolist() for point in goals[joint]] for joint in ORDER}


def configure(base, side, hand):
    cfg = copy.deepcopy(base)
    op = cfg['operation']
    if op.get('auto_profile') not in (None, PROFILE) and not op.get('latest_analysis'):
        op['verify_route'] = None  # retire unrelated task-home/table poses
    op['session'] = op['session'] or datetime.now().strftime('%Y%m%d') + '_' + side
    op.update(auto_profile=PROFILE, hand_state=hand, start_raw=None,
              measure_routes=None, route_review=PROFILE)
    cfg['sides'][side]['other_arm_q'] = None
    cfg['model'].pop('head_positions', None)
    cfg['geometry']['self_margin_m'] = .002  # own-arm assembly clearance; inter-arm/obstacle stays 20mm
    cfg['geometry']['obstacles'] = cfg['geometry']['obstacles'] or []
    cfg['geometry']['review'] = 'operator-cleared-workspace; robot filters from asset manifest'
    if not op['verify_route']:
        # Same arm-only poses for both hand states: manufacturer zero and elbow +90.
        elbow = [0.] * 7
        elbow[3] = float(np.pi / 2)
        op['verify_route'] = [[0.] * 7, elbow, [0.] * 7]
    # A from-attention search includes the full 90deg travel, unlike the old
    # .55rad near-stop approach. Fixed bounds, never expanded after a failure.
    cfg['search'].update(max_travel=2.0, timeout=100., total_travel=30., total_timeout=1800.,
                         torque_review='manufacturer 1.2.9 arm gains/zero feedforward; '
                         'directional gravity-subtracted stop torque; URDF effort caps; '
                         'real-hardware validation pending')
    return cfg


def resolve(cfg, side, target_raw, other_raw, delta):
    cfg = copy.deepcopy(cfg)
    cfg['operation']['start_raw'] = vector(target_raw, 'target raw').tolist()
    cfg['operation']['initial_delta'] = vector(delta, 'coordinate delta').tolist()
    # Opposite arm must remain supported/disabled, in its current encoder epoch.
    cfg['sides'][side]['other_arm_q'] = vector(other_raw, 'other arm').tolist()
    start = vector(target_raw, 'target raw') - vector(delta, 'coordinate delta')
    cfg['operation']['measure_routes'] = measurement_routes(start, side)
    return cfg


class AutoRunner(Runner):
    """Manufacturer arm gains/ordering, with bounded slower motion and repeat checks.

    Source: openarm_can/setup/openarm-can-zero-position-calibration (Apache-2.0,
    Copyright 2025 Enactic, Inc.). No gripper motor or head is initialized.
    """
    moving = ()
    stage_joint = None

    def control_parameters(self, state, target, search):
        kp = np.array([300., 300., 150., 150., 40., 40., 30.])
        kd = np.array([2.5, 2.5, 2.5, 2.5, .8, .8, .8])
        for idx in self.moving:
            kp[idx], kd[idx] = (280., 2.) if idx == 1 and self.stage_joint in (1, 2) else (52., 1.5)
        if search is not None:
            kp[search], kd[search] = (280., 2.1) if search == 0 else (45., 1.2)
        # The manufacturer commands zero feedforward, including holding joints.
        # Model gravity remains used for residual contact detection only.
        return kp, kd, np.zeros(7)

    precise = False

    def position_settled(self, state, goal):
        """영점 FF 0 제어는 중력 때문에 q = 목표 - g/Kp 에 선다. 이동(준비) 종료는 모델이 예측한 처짐과의 차이로 본다.
        예측이 30 % 빗나가도 통과하는 폭 + position_tolerance. 영점 저장 직전 정밀 도달(precise)은 원래 판정."""
        if self.precise or not self.moving:
            return super().position_settled(state, goal)
        kp, _, _ = self.control_parameters(state, goal, None)
        sag = np.asarray(self.gravity(self.coordinates.model(state.q)), float) / kp
        err = np.abs((state.q - goal) + sag)
        return bool(np.all(err <= self.s.position_tolerance + .3 * np.abs(sag)))

    def apply_joint(self, joint):
        self.precise = True  # 0° 로 가서 영점을 쓰는 순간은 처짐을 허용하지 않는다
        try:
            return super().apply_joint(joint)
        finally:
            self.precise = False

    def measure_joint(self, side, joint):
        self.stage_joint = joint
        return super().measure_joint(side, joint)

    def locate(self, side, joint, direction):
        # At attention j4 can already touch its stop. Only the discarded locating
        # pass allows this; each of the three measured passes must actually move.
        ref = reference(side, joint) + self.coordinates.delta[joint - 1]
        return self.approach(joint, direction, locating_reference=ref)

    def move(self, goal):
        start = self.read().q
        self.moving = tuple(np.flatnonzero(np.abs(np.asarray(goal) - start) > self.s.position_tolerance))
        try:
            state = None
            for point in interpolate_waypoints(start, [goal]):
                state = super().move(point)
            assert state is not None
            return state
        finally:
            self.moving = ()

    def finish_measurement(self):
        # Manufacturer ends with j1 at zero, then lowers j4 from +90 to zero.
        # These are measured model zeros, not old raw zero targets.
        for joint in (1, 4):
            goal = self.read().q.copy()
            goal[joint - 1] = self.coordinates.delta[joint - 1]
            self.move(goal)
