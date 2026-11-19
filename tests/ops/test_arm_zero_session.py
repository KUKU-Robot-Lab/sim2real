"""Operator launcher checks; no CAN or ROS process is started."""
from argparse import Namespace
import importlib.util
from unittest.mock import patch

import yaml

import arm_zero_session as session
from arm_zero_auto import ORDER


def test_offline_run_does_not_start_recorder_or_check_hardware(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'operation': {}}))
    args = Namespace(config=path, execute=False, side='right', hand='detached')
    with patch.object(session.subprocess, 'call', return_value=2) as call, patch.object(importlib.util, 'find_spec', side_effect=AssertionError('hardware dependency inspection')):
        assert session.run(args) == 2
    assert call.call_count == 1
    assert 'plan' in call.call_args.args[0]
    assert '--execute' not in call.call_args.args[0]


def test_setup_preserves_baseline_routes_on_enter(tmp_path):
    cfg = yaml.safe_load((session.ROOT / 'config/arm_zero_calibration.yaml').read_text())
    routes = {j: [[.12345678912345678] * 7] for j in ORDER}
    baseline = [[.12345678912345678] * 7, [.2] * 7]
    cfg['operation']['measure_routes'] = routes
    cfg['operation']['verify_route'] = baseline
    cfg['search']['torque_cap'] = [6] * 7
    cfg['search']['contact_torque'] = [2] * 7
    cfg['search']['torque_review'] = 'test'
    cfg['model']['head_positions'] = {'head_j_pan': 0, 'head_j_tilt': 0}
    cfg['geometry'].update(obstacles=[], review='test')
    cfg['operation']['route_review'] = 'test'
    path = tmp_path / 'test.yaml'
    path.write_text(yaml.safe_dump(cfg))
    args = Namespace(config=path, side='right', hand='detached')
    with patch('builtins.input', side_effect=AssertionError('setup must not ask questions')):
        assert session.setup(args) == 0
    actual = yaml.safe_load(path.read_text())
    assert actual['operation']['verify_route'] == baseline
    assert actual['operation']['measure_routes'] is None
    assert actual['operation']['auto_profile']
    assert actual['operation']['start_raw'] is None


def test_default_actions_follow_two_stage_workflow(monkeypatch):
    actions = []
    monkeypatch.setattr(session, 'run', lambda args: actions.append(args.action) or 0)
    assert session.main(['run', '--side', 'right', '--hand', 'detached']) == 0
    assert session.main(['run', '--side', 'right', '--hand', 'attached']) == 0
    assert actions == ['measure', 'verify']


def test_unresolved_apply_run_blocks_new_motion_before_dependencies(tmp_path):
    import pytest
    cfg = yaml.safe_load((session.ROOT / 'config/arm_zero_calibration.yaml').read_text())
    cfg['operation']['pending_zero_run'] = 'interrupted_apply'
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    args = Namespace(config=path, execute=True, side='right', hand='detached', action='measure')
    with patch.object(importlib.util, 'find_spec', side_effect=AssertionError('must block first')):
        with pytest.raises(RuntimeError, match='영점 저장 런'):
            session.run(args)
