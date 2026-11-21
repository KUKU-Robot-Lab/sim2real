"""★11.11 콘솔 수정 — 선점이 목 드라이버를 남기는지, refresh 사전 검사, 학생 표시, 기록 폼 목록."""
from pathlib import Path
from types import SimpleNamespace

from s2r_console import eval_log as EL
from s2r_console import experiments as XP
from s2r_console.actions import HEAD_TOPIC, head_missing_reason

from test_console_actions import setup_actions

REPO = Path(__file__).resolve().parents[2]


def test_preemption_leaves_the_head_driver_running(console, tiny_repo):
    m = setup_actions(console, tiny_repo, {'home_right': [{'argv': ['true']}]})
    s = console.session
    s.supervisor.spawn('episode_x#0', stage='episode_x', note='p', argv=['sleep', '30'], background=True, manual=False)
    s.supervisor.spawn('episode_x#head', stage='episode_x', note='h',
                       argv=['sleep', '30', 'scripts/nodes/head_joint_publisher.py'], background=True, manual=False)
    assert m._policy_targets(s) == ['episode_x#0']


def _obs(age_ms, topics_age_s=0.5):
    return SimpleNamespace(topics={HEAD_TOPIC: {'n': 3, 'age_ms': age_ms}}, topics_age_s=topics_age_s)


def test_refresh_precheck_needs_a_live_head_driver_and_fresh_head_angles():
    alive = [{'argv': ['python3', 'head_joint_publisher.py'], 'alive': True}]
    dead = [{'argv': ['python3', 'head_joint_publisher.py'], 'alive': False}]
    assert head_missing_reason(alive, _obs(20)) is None
    assert 'head_home 다시' in head_missing_reason(dead, _obs(20))
    assert 'head_home 다시' in head_missing_reason(alive, _obs(None))
    assert head_missing_reason(alive, _obs(None, topics_age_s=None)) is None     # 브리지 집계가 없으면 막지 않는다


def test_student_episode_shows_the_student_not_the_contract_policy():
    from policy_control import episode_spec as S
    ep = S.load(REPO / 'config/episodes/grasp_s_right.yaml')
    ids = [x['id'] for x in XP._steps(ep) if x['kind'] == 'policy']
    assert ep.policies['rh_aglt_r'].student in ids and ep.policies['rh_aglt_r'].policy not in ids
    meta = XP._student_meta(REPO, ep.policies['rh_aglt_r'].student)
    assert meta['card'] == 'student.yaml' and '.pt' in meta['checkpoint']


def test_eval_record_list_is_all_eval_episodes_while_tab_stays():
    v = EL.view(REPO, [])
    assert [e['name'] for e in v['episodes']] == [n for n in EL.EVAL_TAB if (REPO / f'config/episodes/{n}.yaml').is_file()
                                                  or n in EL.EVAL_PENDING]
    assert {e['name'] for e in v['record']} == {n for n in EL.EVAL_EPISODES if (REPO / f'config/episodes/{n}.yaml').is_file()}
