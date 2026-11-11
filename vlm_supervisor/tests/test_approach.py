import json

import pytest
from PIL import Image

from vlm_supervisor.approach import CLEAR_M, ApproachJudge, geom_check

IMG = Image.new('RGB', (8, 8))
STEP = ['grasp', 'right', 'SRC_BLUE', '-']


class Backend:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def ask(self, images, text, **kwargs):
        self.prompts.append(text)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer, {'model': 'fake'}


def ans(**kw):
    base = {'hand': 'right', 'object': 'SRC_BLUE', 'verdict': 'clear', 'obstacles': [],
            'action': 'proceed', 'first': None, 'reason': 'path free'}
    return '<think>x</think>' + json.dumps(base | kw)


def test_clear_parses_and_normalizes_names():
    r = ApproachJudge(Backend(ans(object='src blue'))).check(IMG, STEP)
    assert r['valid'] and r['verdict'] == 'clear' and r['action'] == 'proceed'
    assert r['object'] == 'SRC_BLUE' and not r['step_mismatch'] and r['geom'] is None


def test_hand_verdict_action_case_normalized():
    r = ApproachJudge(Backend(ans(hand=' Right', verdict='CLEAR', action='Proceed'))).check(IMG, STEP)
    assert r['valid'] and (r['hand'], r['verdict'], r['action']) == ('right', 'clear', 'proceed')


def test_blocked_reorder():
    r = ApproachJudge(Backend(ans(verdict='blocked', obstacles=['shaker'], action='reorder',
                                  first='SHAKER'))).check(IMG, STEP)
    assert r['valid'] and r['obstacles'] == ['SHAKER'] and r['first'] == 'SHAKER'


@pytest.mark.parametrize('answer', [
    'not json', ans(object='GREEN_CUP'), ans(obstacles=['TABLE']), ans(verdict='clear', action='reorder'),
    ans(verdict='blocked', action='proceed'), ans(verdict='blocked', obstacles=[], action='human'),
    ans(verdict='blocked', obstacles=['SHAKER'], action='reorder', first=None), ans(hand='both'),
    RuntimeError('server down'), ans(object=['SRC_BLUE']), ans(verdict={'a': 1}), ans(obstacles='SHAKER'),
    ans(obstacles=[['SHAKER']]), ans(first=3), ans(obstacles=['SHAKER']),  # clear 인데 장애물
])
def test_fallback_is_unknown_human(answer):
    r = ApproachJudge(Backend(answer)).check(IMG, STEP)
    assert not r['valid'] and r['verdict'] == 'unknown' and r['action'] == 'human' and r['error']
    assert (r['hand'], r['object']) == ('right', 'SRC_BLUE')


def test_no_image_does_not_call_backend():
    b = Backend(ans())
    r = ApproachJudge(b).check(None, STEP)
    assert r['action'] == 'human' and not b.prompts


def test_geom_and_disagree():
    j = ApproachJudge(Backend(ans()))
    world = {'right_hand': {'pos': [0.3, -0.4, 0.3]}, 'SRC_BLUE': {'pos': [0.3, 0.0, 0.05]},
             'SHAKER': {'pos': [0.3 + CLEAR_M / 2, -0.2, 0.05]}, 'SRC_PINK': {'pos': [0.6, -0.2, 0.05]}}
    g = geom_check(world, 'right', 'SRC_BLUE', j.names)
    assert g['verdict'] == 'blocked' and g['obstacles'] == ['SHAKER'] and g['dist_m']['SRC_PINK'] == 0.3
    r = j.check(IMG, STEP, world)
    assert r['geom']['verdict'] == 'blocked' and r['disagree']
    world['SHAKER']['pos'] = [0.3 - 2 * CLEAR_M, 0.0, 0.05]  # 목표 옆이지만 멀다
    assert not j.check(IMG, STEP, world)['disagree']
    assert geom_check({'SRC_BLUE': {'pos': [0, 0, 0]}}, 'right', 'SRC_BLUE', j.names) is None


def test_geom_bridge_form_skips_in_hand_and_goal_neighbor():
    j = ApproachJudge(Backend(ans()))
    objs = {'right_hand': {'pos': [0.3, -0.4, 0.3]}, 'SRC_BLUE': {'pos': [0.3, 0.0, 0.05]},
            'SHAKER': {'pos': [0.3, -0.2, 0.05], 'in_hand': 'right'},  # 길 위지만 판정하는 손이 쥠
            'SRC_PINK': {'pos': [0.3, 0.07, 0.05]}}  # 목표 뒤 0.07 m: 자르지 않으면 막힘, 자른 끝(-0.04)에서 0.11 > CLEAR_M
    g = geom_check({'objects': objs}, 'right', 'SRC_BLUE', j.names)
    assert g['verdict'] == 'clear' and 'SHAKER' not in g['dist_m'] and 'SRC_BLUE' not in g['dist_m']
    objs['SHAKER'].pop('in_hand')
    assert geom_check({'objects': objs}, 'right', 'SRC_BLUE', j.names)['obstacles'] == ['SHAKER']
    objs['SHAKER']['in_hand'] = 'left'  # 반대 손이 든 물체는 장애물로 남는다
    assert geom_check({'objects': objs}, 'right', 'SRC_BLUE', j.names)['obstacles'] == ['SHAKER']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None, 'x', True])
def test_geom_bad_coordinate_is_unknown_and_disagrees(bad):
    j = ApproachJudge(Backend(ans()))
    world = {'right_hand': {'pos': [0.3, -0.4, 0.3]}, 'SRC_BLUE': {'pos': [0.3, 0.0, 0.05]},
             'SRC_PINK': {'pos': [bad, 0.5, 0.05]}}
    assert geom_check(world, 'right', 'SRC_BLUE', j.names)['verdict'] == 'unknown'
    r = j.check(IMG, STEP, world)
    assert r['geom']['verdict'] == 'unknown' and r['disagree']


class SeqBackend(Backend):
    def __init__(self, answers):
        super().__init__(None)
        self.answers = list(answers)

    def ask(self, images, text, **kwargs):
        self.prompts.append(text)
        return self.answers.pop(0), {'model': 'fake'}


def test_invalid_answer_is_reasked_once():
    bad = ans(verdict='blocked', obstacles=[], action='reorder', first='HOLDER_2')
    b = SeqBackend([bad, ans()])
    r = ApproachJudge(b).check(IMG, STEP)
    assert len(b.prompts) == 2 and r['valid'] and r['verdict'] == 'clear'
    assert r['retried_from']['error'] == 'blocked without obstacles'


def test_invalid_twice_stays_human():
    bad = ans(verdict='blocked', obstacles=[], action='reorder')
    b = SeqBackend([bad, bad])
    r = ApproachJudge(b).check(IMG, STEP)
    assert len(b.prompts) == 2 and not r['valid'] and r['action'] == 'human'


def test_backend_error_is_not_reasked():
    b = Backend(RuntimeError('timeout'))
    r = ApproachJudge(b).check(IMG, STEP)
    assert len(b.prompts) == 1 and r['action'] == 'human'
