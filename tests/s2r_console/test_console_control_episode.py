"""★10.20 사용자 "위치 세팅이 이상함 policy 창서도 보이게 해야지" — 1-B(정책 없는 제어 모듈 에피소드)를 policy 탭에서 바로 실행.

정책 카드 없이 팔마다 고정 작업 ctl_grasp_b_<팔>. 미션 단계 episode_grasp_b_<팔> 의 에피소드 파일을 그대로 돌리고(정책 · 물체 바꿈 없음),
rg3 카드와 같은 순서(띄움 → 승인한 연속 실행 → 끝까지 → 정지 · pd 붙듦 → 기록 마무리)로 정리한다.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from s2r_console.actions import CONTROL_PREFIX, control_side
from test_console_actions import setup_actions, wait

REPO = Path(__file__).resolve().parents[2]
# 미션 생성기가 만드는 모양 — 수동 확인(물체) · 띄움 · 수동(진행 경계) · 정리
COMMANDS = [{'argv': ['true'], 'manual': True},
            {'argv': ['true'], 'note': 'runner'},
            {'argv': ['true'], 'manual': True},
            {'argv': ['true'], 'note': 'record stop'},
            {'argv': ['true'], 'note': 'report media'}]


def _episodes(repo, sides=('right', 'left')):
    (repo / 'config/episodes').mkdir(parents=True, exist_ok=True)
    cups = {'right': 'source240b_blue', 'left': 'shaker_c_orange'}
    for side in sides:
        (repo / f'config/episodes/grasp_b_{side}.yaml').write_text(yaml.safe_dump({
            'objects': {'CUP': {'topic': f'/objects/{cups[side]}/pose'}}, 'policies': {}, 'sequence': []}))


def _lifecycle(console, tiny_repo, monkeypatch, phase_after_run):
    m = setup_actions(console, tiny_repo, {'episode_grasp_b_right': COMMANDS, 'home_right': [{'argv': ['true']}]})
    _episodes(tiny_repo, ('right',))
    status = {'episode': 'grasp_b_right', 'phase': 'READY'}
    obs = SimpleNamespace(status={'episode_runner_right': status, 'pd_right': {'execute': True, 'phase': 'TRACKING'}},
                          age_s={'episode_runner_right': 0, 'pd_right': 0}, joints_age_s=0, joint_ages={},
                          rosgraph=None, rosgraph_age_s=None)
    monkeypatch.setattr(console.session.feed, 'observed', lambda: obs)
    calls = []
    monkeypatch.setattr(m, '_preempt', lambda s, *a: calls.append(('stop -> hold', a)))

    def run(s, sid, cmds):
        calls.append((sid, cmds))
        if sid == 'action_episode_run_right':
            status['phase'] = phase_after_run
    monkeypatch.setattr(m, '_run_commands', run)
    return m, calls


def test_control_side_reads_the_arm_from_the_action_id():
    assert control_side('ctl_grasp_b_right') == 'right'
    assert control_side('ctl_grasp_b_left') == 'left'


def test_grasp_b_is_a_policy_tab_action_for_each_arm_without_a_card(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'episode_grasp_b_right': COMMANDS, 'episode_grasp_b_left': COMMANDS})
    _episodes(tiny_repo)
    items = {a['id']: a for a in m.catalog()}
    for side, cup in (('right', 'source240b_blue'), ('left', 'shaker_c_orange')):
        a = items[f'{CONTROL_PREFIX}grasp_b_{side}']
        assert a['group'] == 'policy' and a['control'] is True and a['card_id'] is None
        assert a['stages'] == [f'episode_grasp_b_{side}']
        assert any(cup in w for w in a['warnings']), a['warnings']        # 무엇을 FP++ 로 보는지 미리 보인다
        assert '1-B' in a['title'] and a['title'].startswith(side)
    # 카드 작업은 control 이 아니다(정책 카드 아래에 그린다)
    assert all(not a['control'] for a in m.catalog() if not a['id'].startswith(CONTROL_PREFIX))


def test_path_c_is_a_fixed_action_that_shows_the_script_grip(console, tiny_repo):
    """★10.21 사용자 "카드 배치 일단 제거" — 카드로만 돌던 경로 C 는 ctl_grasp_c_<팔>. 에피소드 파일의 컵으로
    카드 때처럼 스크립트 쥐기 단계 · 마지막 쥔 관절을 보인다."""
    m = setup_actions(console, tiny_repo, {'episode_grasp_c_right': COMMANDS})
    (tiny_repo / 'config/episodes').mkdir(parents=True, exist_ok=True)
    (tiny_repo / 'config/episodes/grasp_c_right.yaml').write_text(yaml.safe_dump({
        'objects': {'CUP': {'topic': '/objects/source240b_blue/pose'}}, 'policies': {}, 'sequence': []}))
    a = {x['id']: x for x in m.catalog()}['ctl_grasp_c_right']
    assert a['control'] is True and a['stages'] == ['episode_grasp_c_right'] and '경로 C' in a['title']
    assert a['live']['script']['steps'].startswith('접근 정책') and a['live']['script']['phase'] is None


def test_grasp_b_actions_appear_only_when_the_mission_has_the_stage(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'home_right': [{'argv': ['true']}]})
    assert not [a for a in m.catalog() if a['id'].startswith(CONTROL_PREFIX)]


def test_two_arms_share_head_and_perception_but_not_the_arm(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'episode_grasp_b_right': COMMANDS, 'episode_grasp_b_left': COMMANDS})
    s = console.session
    assert m._resources(s, 'ctl_grasp_b_right', 'policy', {}) == {'right', 'head', 'perception'}
    assert m._resources(s, 'ctl_grasp_b_left', 'policy', {}) == {'left', 'head', 'perception'}


def test_success_runs_the_unchanged_episode_then_holds_and_finishes_recording(console, tiny_repo, monkeypatch):
    m, calls = _lifecycle(console, tiny_repo, monkeypatch, 'SUCCESS')
    m.execute('ctl_grasp_b_right', operator='test')
    assert wait(m)['outcome'] == 'DONE'
    launch, run, hold, cleanup = calls
    assert launch[0] == 'episode_grasp_b_right' and len(launch[1]) == 2          # 진행 경계 앞까지
    argv = run[1][0].argv
    assert run[0] == 'action_episode_run_right' and 'episode:grasp_b_right' in argv and 'episode_runner_right' in argv
    assert hold[0] == 'stop -> hold' and hold[1] == ({'right'},)
    assert cleanup[0] == 'episode_grasp_b_right_cleanup'
    assert [c.note for c in cleanup[1]] == ['record stop', 'report media']     # 기록 마무리 · 보고 사진 · 영상
    assert not list(console.session.run_dir.glob('rg3_*.yaml'))                 # 정책 · 물체를 바꾼 사본을 만들지 않는다
    assert 'episode_grasp_b_right' in console.session.state.completed


def test_failure_still_holds_and_finishes_recording(console, tiny_repo, monkeypatch):
    m, calls = _lifecycle(console, tiny_repo, monkeypatch, 'FAILURE')
    m.execute('ctl_grasp_b_right', operator='test')
    result = wait(m)
    assert result['outcome'] == 'FAILED' and '1-B 종료: FAILURE' in result['note']
    assert calls[-2][0] == 'stop -> hold'
    assert calls[-1][0] == 'episode_grasp_b_right_cleanup'


def test_a_control_command_on_that_arm_cancels_the_episode(console, tiny_repo, monkeypatch):
    m, calls = _lifecycle(console, tiny_repo, monkeypatch, 'RUNNING')
    m.execute('ctl_grasp_b_right', operator='test')
    end = __import__('time').monotonic() + 3
    while not any(c[0] == 'action_episode_run_right' for c in calls) and __import__('time').monotonic() < end:
        __import__('time').sleep(.01)
    m.execute('home_right', operator='test')
    assert wait(m)['outcome'] == 'DONE'
    cleanup = next(i for i, c in enumerate(calls) if c[0] == 'episode_grasp_b_right_cleanup')
    home = next(i for i, c in enumerate(calls) if c[0] == 'home_right')
    assert cleanup < home
    assert 'episode_grasp_b_right' not in console.session.state.completed


def test_policy_tab_runs_control_episodes_from_the_experiment_list():
    """★10.21 사용자 "정책을 찾기 어렵고 실험 진행이 어려움": 제어 에피소드 버튼 8 개를 늘어놓던 칸 대신 실험 목록 —
    고른 실험 · 팔의 ctl_<에피소드>_<팔> 을 같은 승인 창(action-prepare)으로 연다. 경로 C(카드로만 돌던 것)도 고정 작업이다."""
    from s2r_console.actions import CONTROL_EPISODES, CONTROL_LABELS
    from s2r_console.experiments import GRASP

    html = (REPO / 'deploy/s2r_console/s2r_console/web/index.html').read_text()
    js = (REPO / 'deploy/s2r_console/s2r_console/web/console.js').read_text()
    tab = html[html.index('id="tab-policy"'):html.index('id="common-bar"')]
    assert 'id="xp-list"' in tab and 'id="xp-main"' in tab
    assert 'control-ep-panel' not in html and 'control-ep-actions' not in js
    assert 'data-act="action-prepare" data-arg="${esc(side.action)}"' in js
    assert [head for _, head, _ in GRASP] == list(CONTROL_EPISODES) == list(CONTROL_LABELS)   # 목록 = 실행 작업
    assert 'grasp_c' in CONTROL_EPISODES and CONTROL_LABELS['grasp_c'] == '경로 C'


def test_real_mission_exposes_grasp_b_on_both_arms():
    m = yaml.safe_load((REPO / 'config/mission_rh56f1_control.yaml').read_text())
    ids = {st['id']: st for st in m['stages']}
    for side, lane in (('right', 'arm_right'), ('left', 'arm_left')):
        st = ids[f'episode_grasp_b_{side}']
        assert st['group'] == 'policy' and st['lane'] == lane
        cmds = m['run'][f'episode_grasp_b_{side}']
        assert sum(bool(c.get('manual')) for c in cmds) >= 2                      # 진행 경계(정리 앞 수동 단계)가 있다
