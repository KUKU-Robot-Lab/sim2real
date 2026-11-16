"""plan → episode node order mapping and shell reorder, without ROS."""
import copy
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
EP = ROOT / 'config/episodes/cocktail_1c_e4000.yaml'
FALLBACK = ROOT / 'vlm_supervisor/config/plan_cocktail_1c_e4000.json'


def episode():
    return yaml.safe_load(EP.read_text())


def fallback():
    return json.loads(FALLBACK.read_text())


def test_fallback_plan_maps_to_original_order_unchanged():
    from vlm_supervisor.plan_map import map_plan
    ep = episode()
    result = map_plan(fallback(), ep)
    assert result['ok'], result['reasons']
    assert not result['changed']
    assert result['order'] == [n['id'] for n in ep['sequence']]
    assert {row['node'] for row in result['table'][2:6]} == {'grasp_to_tilt_start'}


def test_lanes_children_inner_order_free():
    from vlm_supervisor.plan_map import map_plan
    steps = fallback()
    steps[2:6] = [steps[3], steps[2], steps[5], steps[4]]  # left first inside the lane block
    result = map_plan(steps, episode())
    assert result['ok'] and not result['changed'], result['reasons']


PINK_FIRST = [
    ['go_home', 'both', '-', '-'], ['scene_snapshot', '-', '-', '-'],
    ['grasp', 'right', 'SRC_PINK', '-'], ['grasp', 'left', 'SHAKER', '-'],
    ['to_tilt_start', 'right', 'SRC_PINK', '-'], ['to_tilt_start', 'left', 'SHAKER', '-'],
    ['tilt', 'both', 'SRC_PINK', 'SHAKER'], ['place', 'right', 'SRC_PINK', 'HOLDER_2'],
    ['go_home', 'right', '-', '-'],
    ['grasp', 'right', 'SRC_BLUE', '-'], ['to_tilt_start', 'right', 'SRC_BLUE', '-'],
    ['tilt', 'both', 'SRC_BLUE', 'SHAKER'], ['place', 'right', 'SRC_BLUE', 'HOLDER_1'],
    ['place', 'left', 'SHAKER', 'HOLDER_0'], ['go_home', 'both', '-', '-'],
    ['terminal', '-', '-', 'SUCCESS']]


def without_clear_first():
    """Cards minus 먼저_치울_물체, so the YAML-structure checks below stay reachable as a second guard."""
    from vlm_supervisor.planner import load_cards
    cards = load_cards()
    for obj in cards['객체_사전']:
        obj.pop('먼저_치울_물체', None)
    return cards


def test_pink_first_rejected_by_simulate_blue_first_rule():
    """User 11.16: pink is outboard of blue, grasping pink first knocks blue over."""
    from vlm_supervisor.planner import simulate
    from vlm_supervisor.plan_map import map_plan
    result = simulate(PINK_FIRST)
    assert not result['ok'] and "grasp SRC_PINK requires ['SRC_BLUE'] off the table" in result['errors'][0]
    assert result['errors'][0].startswith('step 2:')
    assert simulate(fallback())['ok']
    mapped = map_plan(PINK_FIRST, episode())
    assert not mapped['ok'] and 'plan fails simulate' in mapped['reasons'][0]


def test_pink_first_rejected_by_lanes_grouping():
    from vlm_supervisor.planner import simulate
    from vlm_supervisor.plan_map import map_plan
    cards = without_clear_first()
    assert simulate(PINK_FIRST, cards)['ok']  # without the card rule, the YAML structure still refuses it
    result = map_plan(PINK_FIRST, episode(), cards)
    assert not result['ok'] and not result['changed']
    assert 'lanes node grasp_to_tilt_start' in result['reasons'][0]


def test_flat_yaml_pink_first_rejected_by_expect_and_lift_guard():
    """Without lanes (and without the card rule) the same order still fails on expect and lift_guard."""
    from vlm_supervisor.plan_map import map_plan
    ep = episode()
    lanes = ep['sequence'][2]
    ep['sequence'][2:3] = [c for lane in lanes['lanes'] for c in lane['nodes']]
    assert map_plan(fallback(), ep)['ok']
    result = map_plan(PINK_FIRST, ep, without_clear_first())
    assert not result['ok']
    text = ' '.join(result['reasons'])
    assert 'grasp_pink: expect hands' in text
    assert 'grasp_blue: lift_guard object SRC_PINK not on table' in text


def test_step_without_yaml_node_and_missing_node_rejected():
    from vlm_supervisor.plan_map import map_plan
    ep = episode()
    ep['sequence'] = [n for n in ep['sequence'] if n['id'] != 'home_r_after_blue']
    result = map_plan(fallback(), ep)
    assert not result['ok'] and 'no unused yaml node' in result['reasons'][0]
    steps = [s for s in fallback() if s != ['go_home', 'right', '-', '-']]
    result = map_plan(steps, episode())
    assert not result['ok'] and 'home_r_after_blue' in result['reasons'][0]


def test_lanes_child_split_off_rejected():
    from vlm_supervisor.plan_map import map_plan
    steps = fallback()
    steps.insert(5, steps.pop(3))  # grasp_shaker after to_tilt_start right: still simulate-valid
    steps.insert(4, ['go_home', 'right', '-', '-'])  # breaks the lane block apart
    result = map_plan(steps, episode())
    assert not result['ok']


def test_passive_nodes_keep_place_before_successor():
    from vlm_supervisor.plan_map import map_plan
    ep = episode()
    ep['sequence'].insert(1, {'id': 'check', 'type': 'confirm', 'text': 'x'})
    result = map_plan(fallback(), ep)
    assert result['ok'] and not result['changed']
    assert result['order'][:3] == ['go_home_start', 'check', 'scene']


def test_simulate_allows_partial_plan_but_not_held_cup_at_terminal():
    from vlm_supervisor.planner import simulate
    head = [['go_home', 'both', '-', '-'], ['scene_snapshot', '-', '-', '-'], ['grasp', 'right', 'SRC_BLUE', '-']]
    tail = [['go_home', 'left', '-', '-'], ['terminal', '-', '-', 'SUCCESS']]
    assert simulate(head + [['place', 'right', 'SRC_BLUE', 'HOLDER_1']] + tail)['ok']
    result = simulate(head + tail)
    assert not result['ok'] and 'empty hands' in result['errors'][0]


def started(node='go_home_start', last=None):
    return {'status': 'READY', 'busy': False, 'node': node, 'index': 0, 'last': last or {},
            'world': {}}


def flat_episode():
    ep = episode()
    lanes = ep['sequence'][2]
    ep['sequence'][2:3] = [c for lane in lanes['lanes'] for c in lane['nodes']]
    return ep


def flat_swap(ep):
    """Left arm block before right arm block: simulate-valid, keeps checkpoint/auto nodes in place."""
    ids = [n['id'] for n in ep['sequence']]
    assert ids[2:6] == ['grasp_blue', 'to_tilt_start_r1', 'grasp_shaker', 'to_tilt_start_l']
    return ids[:2] + ids[4:6] + ids[2:4] + ids[6:]


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_map_plan_accepts_flat_swap_and_inverse_round_trips():
    from vlm_supervisor.plan_map import map_plan, steps_for
    ep = flat_episode()
    order = flat_swap(ep)
    result = map_plan(steps_for(order, ep), ep)
    assert result['ok'] and result['changed'] and result['order'] == order, result['reasons']
    ep = episode()
    ids = [n['id'] for n in ep['sequence']]
    back = map_plan(steps_for(ids, ep), ep)
    assert back['ok'] and back['order'] == ids and not back['changed'], back['reasons']


def test_map_plan_rejects_moved_auto_confirm_and_unsigned_lane_child():
    from vlm_supervisor.plan_map import map_plan, steps_for
    ep = flat_episode()
    ep['sequence'][4]['auto'] = True  # grasp_shaker: otherwise a valid swap
    result = map_plan(steps_for(flat_swap(ep), ep), ep)
    assert not result['ok'] and result['reasons'] == ['grasp_shaker: checkpoint/auto node must keep its position']
    ep = flat_episode()
    ep.setdefault('sequence').insert(1, {'id': 'check', 'type': 'confirm', 'text': 'x'})
    result = map_plan(steps_for(flat_swap_with(ep), ep), ep)
    assert not result['ok'] and 'confirm nodes not in the plan' in result['reasons'][-1]
    lanes = episode()
    lanes['sequence'][2]['lanes'][0]['nodes'].append({'id': 'mystery', 'type': 'hand', 'action': 'open'})
    result = map_plan(fallback(), lanes)
    assert not result['ok'] and 'children without a signature' in result['reasons'][0]


def test_map_plan_rejects_reorder_around_unsigned_hand_node():
    from vlm_supervisor.plan_map import map_plan, steps_for
    ep = flat_episode()
    ep['sequence'].insert(1, {'id': 'open_r', 'type': 'hand', 'action': 'open'})
    result = map_plan(steps_for(flat_swap_with(ep), ep), ep)
    assert not result['ok'] and 'unmapped nodes without a signature' in result['reasons'][-1], result['reasons']
    assert 'open_r' in result['reasons'][-1]
    ids = [n['id'] for n in ep['sequence']]
    same = map_plan(steps_for(ids, ep), ep)
    assert same['ok'] and same['order'] == ids and not same['changed'], same['reasons']


def test_shell_retry_after_reorder_follows_new_order(tmp_path):
    from vlm_supervisor.shell.shell_node import ShellController
    src = tmp_path / 'flat.yaml'
    src.write_text(yaml.safe_dump(flat_episode(), allow_unicode=True, sort_keys=False), encoding='utf-8')
    adapter = Adapter()
    shell = ShellController(yaml.safe_load(src.read_text()), src, tmp_path, adapter)
    order = flat_swap(flat_episode())  # grasp_shaker, to_tilt_start_l before grasp_blue, to_tilt_start_r1
    assert shell.handle({'cmd_id': 'r', 'verb': 'reorder', 'args': {'order': order, 'sha256': sha(src)}})['ok']
    assert shell.original_sha == sha(src)
    adapter.status = lambda: {**started('grasp_blue'), 'status': 'FAILURE'}
    ack = shell.handle({'cmd_id': 't', 'verb': 'retry', 'args': {'node_id': 'grasp_blue'}})
    assert ack['ok'], ack
    seq = [n['id'] for n in yaml.safe_load(Path(ack['episode_path']).read_text())['sequence']]
    tail = seq[seq.index('grasp_blue'):]
    assert tail == order[order.index('grasp_blue'):]
    assert 'grasp_shaker' not in seq and 'to_tilt_start_l' not in seq  # already done before grasp_blue


def flat_swap_with(ep):
    ids = [n['id'] for n in ep['sequence']]
    k = ids.index('grasp_blue')
    return ids[:k] + ids[k + 2:k + 4] + ids[k:k + 2] + ids[k + 4:]


def test_derive_reorder_before_start_only():
    from vlm_supervisor.shell.derive import derive
    original = flat_episode()
    saved = copy.deepcopy(original)
    ids = [n['id'] for n in original['sequence']]
    swapped = flat_swap(original)
    result = derive(original, started(), {'verb': 'reorder', 'args': {'order': swapped}})
    assert original == saved
    assert [n['id'] for n in result['sequence']] == swapped  # no shell_confirm/shell_restore prefix
    assert result['sequence'][2] == original['sequence'][4]
    assert result['sequence'][2] is not original['sequence'][4]
    with pytest.raises(ValueError, match='not in original'):
        derive(original, started(), {'verb': 'reorder', 'args': {'order': ids[:-1] + ['nope']}})
    with pytest.raises(ValueError, match='permutation'):
        derive(original, started(), {'verb': 'reorder', 'args': {'order': ids[:-1]}})
    with pytest.raises(ValueError, match='permutation'):
        derive(original, started(), {'verb': 'reorder', 'args': {'order': ids + ['grasp_blue']}})
    with pytest.raises(ValueError, match='before the episode starts'):
        derive(original, started('scene', {'node': 'go_home_start'}), {'verb': 'reorder', 'args': {'order': ids}})
    with pytest.raises(ValueError, match='before the episode starts'):
        derive(original, started(last={'node': 'go_home_start'}), {'verb': 'reorder', 'args': {'order': ids}})


def test_derive_reorder_plan_checks():
    from vlm_supervisor.shell.derive import derive
    ep = flat_episode()
    ids = [n['id'] for n in ep['sequence']]
    first = [ids[1], ids[0]] + ids[2:]
    with pytest.raises(ValueError, match='first node type'):
        derive(ep, started(), {'verb': 'reorder', 'args': {'order': first}})
    auto = copy.deepcopy(ep)
    auto['sequence'][4]['checkpoint'] = 'X'
    with pytest.raises(ValueError, match='checkpoint/auto'):
        derive(auto, started(), {'verb': 'reorder', 'args': {'order': flat_swap(auto)}})
    pink = ids[:2] + [ids[ids.index('grasp_pink')]] + [x for x in ids[2:] if x != 'grasp_pink']
    with pytest.raises(ValueError, match='plan check'):
        derive(ep, started(), {'verb': 'reorder', 'args': {'order': pink}})
    lanes = episode()
    lanes['sequence'][2]['lanes'][0]['nodes'].append({'id': 'mystery', 'type': 'hand', 'action': 'open'})
    lids = [n['id'] for n in lanes['sequence']]
    with pytest.raises(ValueError, match='children without a signature'):
        derive(lanes, started(), {'verb': 'reorder', 'args': {'order': lids}})


class Adapter:
    def __init__(self):
        self.started = []

    def status(self): return started()
    def stop(self): return {'ok': True}
    def terminate(self): pass
    def validate(self, path): return True
    def start(self, path): self.started.append(path)
    def wait_ready(self): return True
    def pid(self): return 1


def plan_order_of(path, ids):
    import subprocess
    import sys
    run = subprocess.run([sys.executable, str(ROOT / 'deploy/policy_control/tools/episode_run.py'),
                          '--episode', str(path), '--plan'], cwd=ROOT, capture_output=True, text=True)
    # rc 1 = a role is unavailable on this host (scratch copies have no nn/ checkpoints); a load error tracebacks
    assert run.returncode in (0, 1) and 'Traceback' not in run.stderr, (run.stdout + run.stderr)[-2000:]
    assert run.returncode == 0 or '✗' in run.stdout
    return [line.split()[0] for line in run.stdout.splitlines()
            if line.startswith('  ') and line.split() and line.split()[0] in ids]


def test_shell_reorder_uses_original_and_plan_validates(tmp_path):
    from vlm_supervisor.shell.shell_node import ShellController
    src = tmp_path / 'flat.yaml'
    src.write_text(yaml.safe_dump(flat_episode(), allow_unicode=True, sort_keys=False), encoding='utf-8')
    adapter = Adapter()
    shell = ShellController(yaml.safe_load(src.read_text()), src, tmp_path, adapter)
    shell.episode = {'sequence': []}  # derived generations must not be the reorder source
    order = flat_swap(flat_episode())
    ack = shell.handle({'cmd_id': 'r', 'verb': 'reorder', 'args': {'order': order, 'sha256': sha(src)}})
    assert ack['ok'], ack
    path = Path(ack['episode_path'])
    assert [n['id'] for n in yaml.safe_load(path.read_text())['sequence']] == order
    assert plan_order_of(path, set(order)) == order
    # sha is checked against the start file, not the moving generation path
    ack = shell.handle({'cmd_id': 'r2', 'verb': 'reorder', 'args': {'order': order, 'sha256': sha(path)}})
    assert not ack['ok'] and 'sha256 mismatch' in ack['reasons'][0]


def test_shell_reorder_rejects_sha_mismatch_and_started_runner():
    from vlm_supervisor.shell.shell_node import ShellController
    adapter = Adapter()
    shell = ShellController(episode(), EP, Path('/nonexistent-unused'), adapter)
    ids = [n['id'] for n in episode()['sequence']]
    ack = shell.handle({'cmd_id': 'a', 'verb': 'reorder', 'args': {'order': ids, 'sha256': '0' * 64}})
    assert not ack['ok'] and 'sha256 mismatch' in ack['reasons'][0] and adapter.started == []
    ack = shell.handle({'cmd_id': 'b', 'verb': 'reorder', 'args': {'order': ids}})
    assert not ack['ok'] and 'sha256 mismatch' in ack['reasons'][0]
    adapter.status = lambda: started('scene', {'node': 'go_home_start'})
    ack = shell.handle({'cmd_id': 'c', 'verb': 'reorder', 'args': {'order': ids, 'sha256': sha(EP)}})
    assert not ack['ok'] and 'before the episode starts' in ack['reasons'][0] and adapter.started == []


class Iface:
    """Runner status as the real EpisodeManager.view() keys: episode, episode_id, status, index, node, nodes, last."""

    def __init__(self, ack=None, ready=True, status=None, order=None):
        self.ack, self.ready, self.calls, self.status_calls = ack, ready, [], 0
        self.order = order or [n['id'] for n in episode()['sequence']]
        self.eid, self._status = '20261116_100000', status

    def status(self):
        self.status_calls += 1
        if self._status is not None:
            return self._status
        return {'status': 'READY', 'episode': 'cocktail_1c_e4000', 'episode_id': self.eid, 'index': 0,
                'node': self.order[0], 'nodes': [{'id': x} for x in self.order], 'last': {}}

    def perform(self, action, **kwargs):
        self.calls.append((action, kwargs))
        if self.ready:
            self.order, self.eid = kwargs['args']['order'], '20261116_100005'
        return None if self.ack is None else dict(self.ack)

    def call(self, service):
        self.calls.append(service)
        return {'ok': True}


CHANGED = ['go_home_start', 'scene'] + [n['id'] for n in yaml.safe_load(EP.read_text())['sequence']][2:]
CHANGED[3], CHANGED[4] = CHANGED[4], CHANGED[3]


def fake_changed(monkeypatch):
    from vlm_supervisor import plan_map
    monkeypatch.setattr(plan_map, 'map_plan', lambda *a: {'ok': True, 'changed': True, 'order': CHANGED,
                                                          'table': [], 'reasons': []})


def test_apply_plan_order_same_order_and_fallback(capsys):
    from vlm_supervisor.plan_map import apply_plan_order, steps_for
    logs, iface = [], Iface({'ok': True})
    ok, steps = apply_plan_order('apply', {'source': 'vlm', 'steps': fallback()}, iface, logs.append, 'active')
    assert ok and steps == steps_for([n['id'] for n in episode()['sequence']], episode()) and iface.calls == []
    assert logs[-1]['ok'] and not logs[-1]['changed'] and logs[-1]['sha256'] == sha(EP)
    ok, steps = apply_plan_order('shadow', {'source': 'fallback', 'steps': fallback()}, iface, logs.append, 'active')
    assert ok and logs[-1]['applied'] is False and 'fallback plan is never applied' in logs[-1]['reasons']


def test_apply_plan_order_off_touches_nothing():
    from unittest.mock import Mock
    from vlm_supervisor.plan_map import apply_plan_order
    iface, log = Mock(), Mock()
    assert apply_plan_order('off', {'source': 'vlm', 'steps': [1]}, iface, log, 'active') == (True, [1])
    assert iface.status.call_count == 0 and iface.perform.call_count == 0 and iface.call.call_count == 0
    assert log.call_count == 0


@pytest.mark.parametrize('plan_order,mode', [('shadow', 'active'), ('apply', 'shadow')])
def test_apply_plan_order_shadow_never_performs(monkeypatch, plan_order, mode):
    from vlm_supervisor.plan_map import apply_plan_order, steps_for
    fake_changed(monkeypatch)
    logs, iface = [], Iface({'ok': True})
    ok, steps = apply_plan_order(plan_order, {'source': 'vlm', 'steps': ['x']}, iface, logs.append, mode)
    assert ok and iface.calls == [] and logs[-1]['applied'] is False
    ep = episode()
    assert steps == steps_for([n['id'] for n in ep['sequence']], ep)  # original order runs: inverse of the table
    if mode != 'active':
        assert 'mode!=active' in logs[-1]['reasons']


@pytest.mark.parametrize('status', [{}, {'episode': None}, {'episode': 'no_such_episode'}])
def test_apply_plan_order_mapping_failure_runs_original(status):
    from vlm_supervisor.plan_map import apply_plan_order
    logs, iface = [], Iface({'ok': True}, status=status)
    ok, steps = apply_plan_order('apply', {'source': 'vlm', 'steps': fallback()}, iface, logs.append, 'active')
    assert ok and steps is None and iface.calls == [] and logs[-1]['applied'] is False
    assert 'mapping error' in logs[-1]['reasons'][-1]


def test_apply_plan_order_refuses_runner_not_at_start(monkeypatch):
    from vlm_supervisor.plan_map import apply_plan_order
    fake_changed(monkeypatch)
    ids = [n['id'] for n in episode()['sequence']]
    for status in ({'index': 1}, {'last': {'node': 'go_home_start'}}, {'node': 'scene'},
                   {'nodes': [{'id': x} for x in ids[:-1]]}):
        logs, iface = [], Iface({'ok': True})
        base = iface.status()
        iface._status = {**base, **status}
        ok, _ = apply_plan_order('apply', {'source': 'vlm', 'steps': ['x']}, iface, logs.append, 'active')
        assert ok and iface.calls == [] and logs[-1]['applied'] is False, status
        assert 'not sent: original order runs' in logs[-1]['reasons']


def test_apply_plan_order_sends_sha_and_waits_new_episode(monkeypatch):
    from vlm_supervisor.plan_map import apply_plan_order, steps_for
    fake_changed(monkeypatch)
    logs, iface = [], Iface({'ok': True})
    ok, steps = apply_plan_order('apply', {'source': 'vlm', 'steps': ['x']}, iface, logs.append, 'active')
    assert ok and steps == steps_for(CHANGED, episode()) and logs[-1]['applied']  # what the runner now executes
    assert iface.calls == [('reorder', {'active': True, 'args': {'order': CHANGED, 'sha256': sha(EP)}})]


@pytest.mark.parametrize('ack,ready', [(None, True), ({'ok': True}, False), ({'ok': False}, True)])
def test_apply_plan_order_unsure_or_rejected_stops(monkeypatch, ack, ready):
    from vlm_supervisor.plan_map import apply_plan_order, UNSURE
    fake_changed(monkeypatch)
    logs, iface = [], Iface(ack, ready=ready)
    ok, steps = apply_plan_order('apply', {'source': 'vlm', 'steps': ['x']}, iface, logs.append, 'active',
                                 ready_timeout=0.3)
    assert not ok and steps is None and iface.calls[-1] == 'stop' and logs[-1]['applied'] is False
    assert 'final_status' in logs[-1]
    assert (UNSURE in logs[-1]['reasons']) == (ack is None or not ready)
