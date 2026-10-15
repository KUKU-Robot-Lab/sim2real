"""One approval authorizes only the displayed action in this run and lease."""
import json
from concurrent.futures import ThreadPoolExecutor


from s2r_console.api import handle


def request(console, path, data, token):
    return handle(console, 'POST', path, query={}, body=json.dumps(data).encode(),
                  headers={'Host': 'localhost', 'X-S2R-Console': '1', 'X-S2R-Lease': token}, client='test')


def test_prepare_does_not_run_and_execute_consumes_confirmation(console):
    console.open('t_fake', operator='op')
    token = console.lease.acquire('op').token
    code, prepared = request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, token)
    assert code == 200
    assert console.session.supervisor.table() == []
    payload = {'confirmation_id': prepared['confirmation_id']}
    assert request(console, '/api/actions/execute', payload, token)[0] == 202
    assert request(console, '/api/actions/execute', payload, token)[0] == 409
    console.session.runner.join(3)
    assert console.session.runner.outcome == 'DONE'


def test_changed_contract_invalidates_confirmation(console, tiny_repo):
    console.open('t_fake', operator='op')
    token = console.lease.acquire('op').token
    code, prepared = request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, token)
    assert code == 200
    (tiny_repo / 'data/contract.json').write_text('{"v": 2}')
    assert request(console, '/api/actions/execute', {'confirmation_id': prepared['confirmation_id']}, token)[0] == 409
    assert console.session.supervisor.table() == []


def test_confirmation_cannot_transfer_to_new_lease(console):
    console.open('t_fake', operator='op')
    token = console.lease.acquire('op').token
    code, prepared = request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, token)
    assert code == 200
    other = console.lease.acquire('other', force=True).token
    assert request(console, '/api/actions/execute', {'confirmation_id': prepared['confirmation_id']}, other)[0] == 409


def test_settings_change_invalidates_confirmation(console):
    console.open('t_fake', operator='op')
    token = console.lease.acquire('op').token
    code, prepared = request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, token)
    assert code == 200
    assert request(console, '/api/workspace', {'perception': {'fpp': False, 'locate': True}}, token)[0] == 200
    assert request(console, '/api/actions/execute', {'confirmation_id': prepared['confirmation_id']}, token)[0] == 409


def test_action_routes_require_lease_and_never_accept_argv(console):
    console.open('t_fake', operator='op')
    assert request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, '')[0] == 409
    token = console.lease.acquire('op').token
    assert request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check', 'argv': ['false']}, token)[0] == 400


def test_concurrent_execution_is_accepted_only_once(console):
    console.open('t_fake', operator='op')
    token = console.lease.acquire('op').token
    _, prepared = request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, token)
    data = {'confirmation_id': prepared['confirmation_id']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(lambda _: request(console, '/api/actions/execute', data, token)[0], range(2)))
    assert sorted(codes) == [202, 409]


def test_expired_confirmation_cannot_execute(console):
    console.open('t_fake', operator='op')
    token = console.lease.acquire('op').token
    _, prepared = request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, token)
    console.confirmations.pending[prepared['confirmation_id']]['expires'] = 0
    assert request(console, '/api/actions/execute', {'confirmation_id': prepared['confirmation_id']}, token)[0] == 409
    assert console.session.supervisor.table() == []


def test_approved_stage_does_not_leave_reusable_legacy_approval(console):
    from dataclasses import replace
    from s2r_console import ledger, survey
    console.open('t_fake', operator='op')
    s = console.session
    s.mission = replace(s.mission, stages=tuple(replace(stage, touches_real=True) if stage.id == 'check' else stage
                                               for stage in s.mission.stages))
    token = console.lease.acquire('op').token
    code, prepared = request(console, '/api/actions/prepare', {'action': 'stage', 'stage': 'check'}, token)
    assert code == 200
    assert request(console, '/api/actions/execute', {'confirmation_id': prepared['confirmation_id']}, token)[0] == 202
    s.runner.join(3)
    basis = survey.all_basis(s.mission, repo=console.repo, cache=s.cache)
    assert ledger.valid_approvals(ledger.read(s.ledger_path), basis) == frozenset()


def test_legacy_episode_cannot_bypass_rg3_fpp_requirement(console):
    from s2r_console.feed import line
    from s2r_console.console import ConsoleError
    import pytest
    console.open('t_fake', operator='op')
    console.session.workspace.update({'perception': {'fpp': False, 'locate': True}})
    with console.session.feed_lock:
        console.session.feed.ingest(line('status', node='episode_runner', data={
            'episode': 'grasp_rg3_right', 'next_action': 'grasp_lift', 'phase': 'READY', 'busy': False}))
    with pytest.raises(ConsoleError, match='FPP'):
        console.episode('run', operator='op', typed='episode:grasp_rg3_right')
    assert console.session.supervisor.table() == []


def test_changed_policy_stop_targets_require_new_confirmation(console, monkeypatch):
    console.open('t_fake', operator='op')
    token = console.lease.acquire('op').token
    targets = []
    monkeypatch.setattr(console.actions, 'describe', lambda *args: {
        'id': 'control', 'group': 'control', 'stages': [], 'parameters': {},
        'enabled': True, 'title': 'control', 'commands': [], 'reasons': [],
        'stop_targets': list(targets), 'warnings': ['정책을 정지한다'] if targets else []})
    _, prepared = request(console, '/api/actions/prepare', {'action': 'control'}, token)
    targets.append('episode_policy#0')
    assert request(console, '/api/actions/execute', {'confirmation_id': prepared['confirmation_id']}, token)[0] == 409
    assert console.session.supervisor.table() == []
