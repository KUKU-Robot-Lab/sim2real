"""Viewer state must not turn into commanded pose when publishing is disabled."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy/policy_control/tools'))
import board_jog as B  # noqa: E402


class Kin:
    def tip(self, arm, hand):
        T = np.eye(4)
        T[:3, 3] = [arm[0], hand[0], arm[1]]
        return T


def snap():
    return {'pub': False, 'meas': np.ones(7) * .1, 'hmeas': np.ones(6) * .2,
            'cmd': np.ones(7) * .3, 'hand': np.ones(6) * .4, 'contact': False}


def test_dry_still_shows_measured_robot():
    p = B.sim_packet(0, 'right', snap(), Kin(), {})
    assert p['state_source'] == 'measured'
    assert np.allclose(p['q_show'], [.1] * 7 + [.2] * 6)
    assert p['q_cmd'] is None
    assert np.allclose(p['tip_xyz_base'], [.1, .2, .1])


def test_no_state_is_explicitly_virtual():
    s = snap()
    s['meas'] = None
    p = B.sim_packet(0, 'right', s, Kin(), {})
    assert p['state_source'] == 'virtual'
    assert np.allclose(p['q_show'], [.3] * 7 + [.4] * 6)


def test_publish_does_not_change_measured_display():
    s = snap()
    before = B.sim_packet(0, 'right', s, Kin(), {})
    s['pub'] = True
    after = B.sim_packet(0, 'right', s, Kin(), {})
    assert before['q_show'] == after['q_show']
    assert np.allclose(after['q_cmd'], [.3] * 7 + [.4] * 6)


def test_viewer_does_not_clamp_measured_angles():
    import ast
    from types import SimpleNamespace

    class Tensor(np.ndarray):
        def clone(self):
            return self.copy()

        def unsqueeze(self, axis):
            return np.expand_dims(self, axis).view(Tensor)

    torch = SimpleNamespace(tensor=lambda x, **kw: np.array(x, dtype=kw.get('dtype')).view(Tensor),
                            zeros_like=np.zeros_like, long=int)
    source = Path(B.S2R) / 'robot/isaacsim_bridge/viewer/board_viewer.py'
    tree = ast.parse(source.read_text())
    mirror = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Mirror')
    ns = {'torch': torch, '_log': lambda x: None}
    exec(compile(ast.Module(body=[mirror], type_ignores=[]), str(source), 'exec'), ns)
    written = []
    art = SimpleNamespace(joint_names=['r_aj_1'], device='cpu',
                          data=SimpleNamespace(soft_joint_pos_limits=np.array([[[-1., 1.]]]).view(Tensor)),
                          write_joint_state_to_sim=lambda q, qd, joint_ids: written.append(q.copy()))
    ns['Mirror'](art, {}).apply({'r_aj_1': -1.2})
    assert written[0][0, 0] == -1.2


def test_state_audit_reads_named_applied_joints_and_tip_origin():
    sys.path.insert(0, str(Path(B.S2R) / 'robot/isaacsim_bridge/viewer'))
    import board_packet as P
    pkt = P.decode(__import__('json').dumps(B.sim_packet(10., 'right', snap(), Kin(), {})).encode())
    names = list(reversed(pkt.names))
    row = P.state_audit(pkt, names, list(reversed(pkt.q_show)), [1.1, 2.2, 3.1], [1.1, 2.2, 3.1],
                        [1., 2., 3.], 11., 11.01, 'recorded-replay')
    assert row['joint_max_abs_error_rad'] == 0.
    assert np.allclose(row['physical_tip_base_m'], [.1, .2, .1])
    assert row['physical_tip_error_mm'] < 1e-10
    assert row['state_source'] == 'measured'
    assert row['audit_origin'] == 'recorded-replay'


def test_state_audit_reports_missing_state_instead_of_hiding_it():
    sys.path.insert(0, str(Path(B.S2R) / 'robot/isaacsim_bridge/viewer'))
    import board_packet as P
    pkt = P.decode(__import__('json').dumps(B.sim_packet(10., 'right', snap(), Kin(), {})).encode())
    row = P.state_audit(pkt, [], [], None, None, [0., 0., 0.], 11., 11.01, 'recorded-replay')
    assert row['missing_joints'] == list(pkt.names)
    assert row['joint_max_abs_error_rad'] is None
    assert row['physical_tip_error_mm'] is None
    assert row['usd_tip_base_m'] is None
