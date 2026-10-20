"""★10.19 사용자 "정책 live 하나랑 sim 에서 나온 궤적 play 하나, 둘 다 선택할 수 있게" — 잡기 카드의 sim 궤적 계획 · 재생 버튼.

가짜 프로세스만 쓴다(로봇 · server 없음). sim_traj 명령 자체는 tests/policy_control/test_pc_sim_traj.py.
"""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np

from s2r_console import simtraj as ST
from test_console_actions import rg3_workspace, setup_actions, wait

POLICY = 'rh56f1/aglt/right_rg3_test'


def grasp_card(console):
    obj = rg3_workspace(console)
    w = console.session.workspace
    w.cards[0]['task'] = 'grasping'
    obj['results']['fpp']['position'] = [0.30, -0.10, 0.05]
    return w.cards[0], obj


def make_plan(repo, name='0101_000000_right_source200_real', **meta):
    d = repo / ST.ROOT / 'card_chosen' / name
    d.mkdir(parents=True)
    m = {'policy': POLICY, 'side': 'right', 'object': 'source200_real', 'cup_xy': [0.30, -0.10], 'n_ok': 8, 'n': 8,
         'env': 3, 'made': '2026-10-19 21:19:00', 'problems': []}
    m.update(meta)
    np.savez(d / 'plan.npz', q=np.zeros((2, 13)), meta=json.dumps(m))
    return d / 'plan.npz'


def pd_feed(console, monkeypatch, phase='TRACKING'):
    obs = SimpleNamespace(status={'pd_right': {'execute': True, 'phase': phase}}, age_s={'pd_right': 0},
                          joints_age_s=0, joint_ages={}, rosgraph=None, rosgraph_age_s=None, objects={}, objects_age_s=None)
    monkeypatch.setattr(console.session.feed, 'observed', lambda: obs)
    return obs


def test_a_grasping_card_offers_live_policy_and_sim_trajectory(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'episode_grasp_d_right': [{'argv': ['true']}]})
    grasp_card(console)
    specs = m._specs(console.session)
    assert {'rg3_chosen', 'simplan_chosen', 'simrun_chosen'} <= set(specs)
    assert specs['simplan_chosen'][1] == 'sim' and specs['simrun_chosen'][1] == 'policy'
    assert all(m.describe(a)['card_id'] == 'chosen' for a in ('rg3_chosen', 'simplan_chosen', 'simrun_chosen'))
    # 접근 카드(경로 C)는 sim 에서 들기까지 안 가므로 sim 버튼이 없다
    console.session.workspace.cards[0]['task'] = 'approaching'
    assert not {'simplan_chosen', 'simrun_chosen'} & set(m._specs(console.session))


def test_the_plan_reads_the_live_cup_and_never_touches_an_arm(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'episode_grasp_d_right': [{'argv': ['true']}]})
    card, obj = grasp_card(console)
    item = m.describe('simplan_chosen')
    assert item['enabled'], item['reasons']                      # pd 상태 없이도 된다 — 로봇을 안 움직인다
    argv = item['commands'][0]['argv']
    assert argv[2:] == ['plan', '--policy', POLICY, '--object', 'source200_real', '--cup-topic',
                        '/objects/source200_blue/live_pose', '--out-root', str(ST.card_root(tiny_repo.resolve(), card))]
    assert m.describe('simplan_chosen')['commands'] == item['commands']    # 승인 근거가 시각에 따라 안 바뀐다
    assert m._resources(console.session, 'simplan_chosen', 'sim', {}) == {'sim_server'}
    obj['results']['fpp']['live'] = False
    assert any('live_pose' in r for r in m.describe('simplan_chosen')['reasons'])
    obj['results']['fpp']['live'] = True
    calls = []
    monkeypatch.setattr(m, '_run_commands', lambda s, sid, cmds, **k: calls.append((sid, [c.argv for c in cmds])))
    m.execute('simplan_chosen', operator='test')
    assert wait(m)['outcome'] == 'DONE'
    assert calls == [('simplan_right', [tuple(argv)])]


def test_playback_needs_this_cards_plan_and_the_cup_within_1_cm(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'episode_grasp_d_right': [{'argv': ['true']}]})
    card, obj = grasp_card(console)
    pd_feed(console, monkeypatch)
    assert any('계획이 없다' in r for r in m.describe('simrun_chosen')['reasons'])
    make_plan(tiny_repo, 'a_other_object', object='shaker_c_real')         # 다른 물체 · 문제 있는 계획은 안 쓴다
    make_plan(tiny_repo, 'b_problem', problems=['팔 속도 2.0 rad/s'])
    assert ST.latest_plan(tiny_repo, card, obj) is None
    good = make_plan(tiny_repo, 'c_good')
    item = m.describe('simrun_chosen')
    assert item['enabled'], item['reasons']
    notes = [c['argv'] for c in item['commands']]
    assert 'start' in notes[0][-1] and notes[2][-2:] == ['stop', 'simtraj_right']
    assert notes[1][2:] == ['run', str(good), '--cup-topic', '/objects/source200_blue/live_pose', '--execute']
    assert item['commands'][1]['background'] and any('★실기 동작' in w and '8/8' in w for w in item['warnings'])
    assert m.describe('simplan_chosen')['live']['sim']['path'] == str(good)
    obj['results']['fpp']['position'] = [0.32, -0.10, 0.05]
    assert any('2.0 cm' in r and '다시 계획' in r for r in m.describe('simrun_chosen')['reasons'])
    # 팔 상태는 정책 카드와 같이 본다
    obj['results']['fpp']['position'] = [0.30, -0.10, 0.05]
    pd_feed(console, monkeypatch, phase='HOLD')
    assert any('PD 상태 HOLD' in r for r in m.describe('simrun_chosen')['reasons'])


def _fake_sim(m, monkeypatch, run_argv):
    from mission_stages import Command
    cmds = [Command(('true',), note='rec start'), Command(tuple(run_argv), note='play', background=True),
            Command(('true',), note='rec stop')]
    plan = {'path': ST.Path('fake/plan.npz'), 'meta': {'made': 'now', 'cup_xy': [0.30, -0.10], 'n_ok': 8, 'n': 8}}
    monkeypatch.setattr(m, '_sim_commands', lambda aid, card, obj: (plan, cmds))


def test_stop_ends_the_playback_then_confirms_pd_holds(console, tiny_repo, monkeypatch):
    """정책 정지 · 같은 팔 제어가 재생을 끊는다 — SIGTERM(sim_traj 가 episode stop 을 보낸다) → pd/hold → 내부 목표 확인 → 기록 끝."""
    m = setup_actions(console, tiny_repo, {'episode_grasp_d_right': [{'argv': ['true']}]})
    grasp_card(console)
    pd_feed(console, monkeypatch)
    ready = tiny_repo / 'sim_ready'           # 처리기를 단 뒤에 만든다 — 그 전에 TERM 이 가면 rc −15(경쟁)
    _fake_sim(m, monkeypatch, [sys.executable, '-c',
                               'import pathlib, signal, sys, time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(130)); '
                               f'pathlib.Path({str(ready)!r}).touch(); time.sleep(30)'])
    monkeypatch.setattr(console, '_pd_sides', lambda s, obs=None: ['right'])
    calls = []
    monkeypatch.setattr(m, '_trigger', lambda s, service, side, expect=(): calls.append((service, side)))
    monkeypatch.setattr(m, '_confirm_hold', lambda s, side, began: calls.append(('confirm', side)))
    m.execute('simrun_chosen', operator='test')
    sup = console.session.supervisor
    end = time.monotonic() + 5
    while not (sup.is_alive('simtraj_right#1') and ready.exists()) and time.monotonic() < end:
        time.sleep(.01)
    assert sup.is_alive('simtraj_right#1') and ready.exists()
    job = m._jobs['simrun_chosen']
    assert m._interrupts(job, 'policy_stop', 'control') and m._interrupts(job, 'rehome_right', 'control')
    assert m.workspace_busy({'cards': []}) == 'simrun_chosen'
    job.cancel_policy.set()
    assert wait(m)['outcome'] == 'DONE'
    assert sup.rc('simtraj_right#1') == 130
    assert calls == [('pd/hold', 'right'), ('confirm', 'right')]
    assert sup.rc('simtraj_right_cleanup#0') == 0


def test_a_refused_playback_fails_with_the_tool_reason(console, tiny_repo, monkeypatch):
    m = setup_actions(console, tiny_repo, {'episode_grasp_d_right': [{'argv': ['true']}]})
    grasp_card(console)
    pd_feed(console, monkeypatch)
    _fake_sim(m, monkeypatch, [sys.executable, '-c',
                               'import sys, time; time.sleep(.3); print("  거부: 컵이 계획 자리에서 3.0 cm(한계 1 cm) — 다시 plan"); sys.exit(2)'])
    monkeypatch.setattr(console, '_pd_sides', lambda s, obs=None: ['right'])
    calls = []
    monkeypatch.setattr(m, '_trigger', lambda s, service, side, expect=(): calls.append(service))
    monkeypatch.setattr(m, '_confirm_hold', lambda s, side, began: calls.append('confirm'))
    m.execute('simrun_chosen', operator='test')
    result = wait(m)
    assert result['outcome'] == 'FAILED'
    assert 'rc=2' in result['note'] and '거부: 컵이 계획 자리에서 3.0 cm' in result['note']
    assert calls == ['pd/hold', 'confirm']                          # 그래도 pd 가 지금 자세를 붙드는지 확인한다


def test_the_console_and_the_tool_use_the_same_cup_tolerance():
    sys.path.insert(0, str(ST.Path(__file__).resolve().parents[2] / 'deploy/policy_control/tools'))
    import sim_traj
    assert ST.CUP_TOL_M == sim_traj.CUP_TOL_M
    assert ST.refusal("a\n[sim_traj] x\n  거부: pd phase HOLD\n") == '거부: pd phase HOLD'
