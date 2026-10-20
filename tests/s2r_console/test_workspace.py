from pathlib import Path
import shutil

import pytest
from s2r_console.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]


def workspace(tmp_path):
    shutil.copytree(ROOT / 'config', tmp_path / 'config')
    return Workspace(tmp_path, 'openarm_rh56f1')


def test_registry_matrix_preserves_measurement():
    ws = Workspace(ROOT, 'openarm_rh56f1')
    objects = ws.snapshot()['objects']
    for model in ('src200', 'src240', 'shaker'):
        assert {o['color'] for o in objects if o['model'] == model} == {'skyblue', 'orange', 'pink'}
    measured = next(o for o in objects if o['id'] == 'source200_orange')
    assert measured['measured'] and measured['hue_range'] == [1.8, 23.0]
    assert len([o for o in objects if o['model'] == 'holder']) == 3


def test_camera_change_rejects_cached_fpp_until_new_acquisition_stamp(tmp_path):
    import time
    ws = workspace(tmp_path)
    old_stamp = time.time() - 1
    topic = '/objects/source200_orange/live_pose'
    ws.ingest_fpp({'objects': {topic: {'pos': [.2, .1, .3], 'age_s': 0, 'stamp': old_stamp}}})
    ws.invalidate_results('camera changed')
    ws.ingest_fpp({'objects': {topic: {'pos': [.2, .1, .3], 'age_s': 0, 'stamp': old_stamp}}})
    assert not next(o for o in ws.objects if o['id'] == 'source200_orange')['results']['fpp']['valid']
    ws.ingest_fpp({'objects': {topic: {'pos': [.2, .1, .3], 'age_s': 0, 'stamp': time.time()}}})
    assert next(o for o in ws.objects if o['id'] == 'source200_orange')['results']['fpp']['valid']


def test_head_workspace_respects_robot_upward_limit(tmp_path):
    ws = workspace(tmp_path)
    with pytest.raises(ValueError, match='tilt'):
        ws.update({'head': {'tilt': -9}})


def test_selection_atomic_invalidation_and_sources(tmp_path):
    ws = workspace(tmp_path)
    old = ws.fingerprint()
    ws.update({'perception': {'fpp': True, 'locate': True}, 'objects': [
        {'id': 'source200_orange', 'enabled': True, 'input_source': 'locate'},
        {'id': 'source200_pink', 'enabled': True}]})
    assert old != ws.fingerprint()
    ws.set_result('source200_orange', 'locate', {'position': [0.2, 0.1, 0.3], 'frame': 'base', 'timestamp': 10, 'valid': True})
    ws.set_result('source200_orange', 'locate', {'valid': False, 'reason': 'not found'})
    obj = next(o for o in ws.snapshot()['objects'] if o['id'] == 'source200_orange')
    assert obj['results']['locate']['position'] is None
    assert not obj['results']['fpp']['valid']
    before = ws.snapshot()
    with pytest.raises(ValueError):
        ws.update({'perception': {'fpp': False}, 'objects': [{'id': 'missing', 'enabled': True}]})
    assert ws.snapshot() == before


def test_card_layout_restore_and_validation(tmp_path):
    shutil.copytree(ROOT / 'deploy/policies', tmp_path / 'deploy/policies', ignore=shutil.ignore_patterns('*.pth', '*.npz'))
    ws = workspace(tmp_path)
    policy = next(p for p in ws.snapshot()['catalog'] if p['task'] == 'grasping')   # ★10.20 옛 rg3 기본 정책은 치웠다 — 배치는 잡기 정책이면 된다
    state = ws.update({'cards': [{'id': 'a', 'task': 'grasping', 'policy_id': policy['id'], 'side': policy['side'], 'object_id': 'source200_orange', 'x': 20, 'y': 40}]})
    assert state['cards'][0]['x'] == 20
    assert Workspace(tmp_path, 'openarm_rh56f1').snapshot()['cards'][0]['x'] == 20
    with pytest.raises(ValueError):
        ws.update({'cards': [{'id': 'a', 'task': 'grasping', 'policy_id': '../bad', 'object_id': 'source200_orange'}]})
    ws.update({'reset_layout': True})
    assert ws.snapshot()['cards'][0]['x'] == 0
    ws.update({'cards': []})
    assert ws.snapshot()['cards'] == []


def test_safe_registered_commands_and_holder_rejection(tmp_path):
    ws = workspace(tmp_path)
    ws.update({'head': {'pan': 10, 'tilt': -30, 'calib_profile': 'v9_1014_anchored_z'}})
    argv = ws.locate_argv('source200_orange')
    assert '--current-pose' in argv
    assert argv[argv.index('--calib-profile') + 1] == 'v9_1014_anchored_z'
    with pytest.raises(ValueError):
        ws.locate_argv('cup_holder_0')
    with pytest.raises(ValueError):
        ws.update({'head': {'calib_profile': '../../bad'}})


def test_locate_uses_registry_and_returns_operating_pose_on_failure():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts/ops'))
    import registered_locate
    core_root = ROOT.parent / 'perception_plus_plus'
    if not core_root.exists():
        pytest.skip('sibling locate core unavailable')
    registered_locate.install_core(ROOT)
    from perception_plus_plus_core.head_scan import fakes, locate
    spec = registered_locate.load_spec(ROOT, 'source200_orange')
    assert spec.origin_above_bottom == .085
    assert spec.hue == pytest.approx(12.4)
    assert len(spec.vertices) > 0
    head = fakes.FakeHead()
    class BrokenCamera:
        def capture(self, n_frames=1):
            raise RuntimeError('camera disconnected')
    result = registered_locate.run_scan(locate.LocateRequest('source200_orange', poses=[(0, -20)]), spec,
                                        head, BrokenCamera(), return_pose=(10, -30))
    assert not result.found
    assert result.home_ok
    assert head.moves[-1] == (10, -30)


def test_fpp_observation_age_invalidates_old_coordinates(tmp_path):
    from s2r_console.console_state import Observed
    ws = workspace(tmp_path)
    obs = Observed(objects={'/objects/source200_orange/pose': {'pos': [.2, .1, .3], 'age_s': .1}}, objects_age_s=.1)
    ws.ingest_fpp(obs)
    obj = next(o for o in ws.snapshot()['objects'] if o['id'] == 'source200_orange')
    assert obj['results']['fpp']['valid']
    ws.ingest_fpp(Observed(objects=obs.objects, objects_age_s=5))
    obj = next(o for o in ws.snapshot()['objects'] if o['id'] == 'source200_orange')
    assert not obj['results']['fpp']['valid']
    assert obj['results']['fpp']['position'] is None


def test_fpp_live_tracking_is_distinguished_from_snapshot(tmp_path):
    ws = workspace(tmp_path)
    topic = '/objects/source200_orange'
    ws.ingest_fpp({topic + '/pose': {'pos': [.2, .1, .3], 'age_s': .1}})
    obj = next(o for o in ws.snapshot()['objects'] if o['id'] == 'source200_orange')
    assert obj['results']['fpp']['valid'] and not obj['results']['fpp']['live']
    ws.ingest_fpp({topic + '/pose': {'pos': [.2, .1, .3], 'age_s': .1},
                   topic + '/live_pose': {'pos': [.21, .1, .3], 'age_s': .05}})
    obj = next(o for o in ws.snapshot()['objects'] if o['id'] == 'source200_orange')
    assert obj['results']['fpp']['live']
    assert obj['results']['fpp']['position'][0] == .21


def test_cancelled_locate_returns_operating_pose_without_scanning():
    import registered_locate
    registered_locate.install_core(ROOT)
    from perception_plus_plus_core.head_scan import fakes, locate
    spec = registered_locate.load_spec(ROOT, 'source200_orange')
    head = fakes.FakeHead()
    result = registered_locate.run_scan(locate.LocateRequest(spec.name), spec, head, object(),
                                        return_pose=(15, -30), should_stop=lambda: True)
    assert not result.found and result.home_ok
    assert head.moves == [(15, -30)]


def test_locate_sigterm_requests_cooperative_cancellation():
    import signal
    import registered_locate
    old = signal.getsignal(signal.SIGTERM)
    with registered_locate.cooperative_cancel() as cancelled:
        assert not cancelled()
        signal.raise_signal(signal.SIGTERM)
        assert cancelled()
    assert signal.getsignal(signal.SIGTERM) == old


@pytest.mark.parametrize('found,home_ok', [(True, True), (False, True), (True, False)])
def test_locate_cli_reports_explicit_validity(monkeypatch, capsys, found, home_ok):
    import json
    from types import SimpleNamespace
    import registered_locate
    monkeypatch.setattr(registered_locate, 'run_scan', lambda *a, **kw: SimpleNamespace(
        found=found, home_ok=home_ok, x=.2622, y=-.1526, z=.29, reason='not found'))
    rc = registered_locate.main(['source240b_blue', '--repo', str(ROOT), '--fake', '.26,-.15'])
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result['valid'] is (found and home_ok)
    assert rc == (0 if found and home_ok else 1)
    assert (result['position'] is not None) is result['valid']


def test_unselected_fpp_does_not_repopulate_results(tmp_path):
    ws = workspace(tmp_path)
    ws.update({'perception': {'fpp': False, 'locate': True}})
    ws.ingest_fpp({'/objects/source240b_blue/live_pose': {'pos': [.2, -.1, .29], 'age_s': 0}})
    obj = next(o for o in ws.objects if o['id'] == 'source240b_blue')
    assert not obj['results']['fpp']['valid']
    assert obj['results']['fpp']['reason'] == 'FPP not selected'


def test_console_locate_uses_current_pose_without_return_motion(tmp_path):
    ws = workspace(tmp_path)
    argv = ws.locate_argv('source240b_blue')
    assert '--current-pose' in argv
    assert '--return-pan' not in argv and '--return-tilt' not in argv


def test_current_pose_locate_never_moves_and_rejects_pose_changes(monkeypatch):
    from types import SimpleNamespace
    import registered_locate
    registered_locate.install_core(ROOT)
    from perception_plus_plus_core.head_scan import locate, survey
    calls = []
    def core(request, spec, head, camera, **kwargs):
        assert request.poses == [(12.0, -20.0)]
        assert request.n_frames >= 2
        assert head.move(12, -20).pan_deg == 12
        calls.append(head.go_home().pan_deg)
        return SimpleNamespace(found=True, home_ok=True)
    monkeypatch.setattr(locate, 'locate_object', core)
    reading = survey.HeadReading(12, -20, True)
    result = registered_locate.run_current_pose(SimpleNamespace(name='cup'), object(), lambda: reading)
    assert result.found and calls == [12]
    readings = iter([reading, survey.HeadReading(13, -20, True)])
    with pytest.raises(RuntimeError, match='목'):
        registered_locate.run_current_pose(SimpleNamespace(name='cup'), object(), lambda: next(readings))


def test_fixed_pose_fake_cli_extracts_coordinates_without_scanning(capsys):
    import json
    import registered_locate
    rc = registered_locate.main(['source240b_blue', '--repo', str(ROOT), '--current-pose', '--fake', '.26,-.15'])
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert rc == 0 and result['valid']
    assert result['position'][:2] == pytest.approx([.26, -.15], abs=.003)


def test_current_pose_missing_measurement_never_falls_back_to_a_head_move(monkeypatch):
    import registered_locate
    registered_locate.install_core(ROOT)
    from perception_plus_plus_core.head_scan import adapters
    monkeypatch.setattr(adapters, 'busy_head_commanders', lambda: [])
    monkeypatch.setattr(adapters, 'SshCamera', lambda: object())
    def forbidden(*args, **kwargs):
        pytest.fail('current-pose extraction must not create a head motion adapter')
    monkeypatch.setattr(adapters, 'Sim2RealHead', forbidden)
    def missing():
        raise RuntimeError('현재 목 실측이 없다')
    monkeypatch.setattr(registered_locate, 'read_current_head', missing)
    with pytest.raises(RuntimeError, match='현재 목 실측'):
        registered_locate.main(['source240b_blue', '--repo', str(ROOT), '--current-pose', '--execute'])
