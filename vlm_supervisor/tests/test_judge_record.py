import json

from PIL import Image

from vlm_supervisor.failure_policy import allowed_actions, runner_detail
from vlm_supervisor.judge import Judge
from vlm_supervisor.supervisor import Supervisor
from vlm_supervisor.tests.test_package import Backend, FakeInterface, status

OK = '{"success":true,"choice":"A","reason":"ok"}'


def failed_status():
    return status() | {'status': 'FAILURE', 'pending': {'node': 'tilt_blue', 'action': 'retry', 'reason': 'x'},
                       'last': {'node': 'tilt_blue', 'code': 'pour_failed', 'reason': 'cup tipped',
                                'results': [{'role': 'r', 'status': 'failed', 'reason': 'spill', 'signals': {}}]},
                       'history': [{'event': 'done', 'result': 'FAILURE', 'termination_reason': 'spill'}]}


def test_blind_hides_runner_verdict_and_off_is_unchanged():
    st = failed_status()
    off, blind = Backend(OK), Backend(OK)
    judge = Judge(off, 'sentence', ['plan'])
    judge.judge(st, Image.new('RGB', (2, 2)))
    out = Judge(blind, 'sentence', ['plan'], blind=True).judge(st, Image.new('RGB', (2, 2)))
    # off: e5cdb75 프롬프트 식을 그대로 다시 만들어 바이트 비교한다.
    detail = runner_detail(st)
    allowed = allowed_actions(st, detail)
    head = off.prompts[0].split('Choose exactly one allowed choice: ')[0]
    choices = off.prompts[0].split('Choose exactly one allowed choice: ')[1].split('. Return only')[0]
    expected = (head + 'Choose exactly one allowed choice: ' + choices + '. '
                'Return only {"success":bool,"choice":"A".."E","reason":"short explanation"}. '
                'success must be true exactly for A.\n'
                + json.dumps({'sentence': 'sentence', 'plan': ['plan'], 'status': st, 'runner_detail': detail,
                              'allowed_actions': sorted(allowed), 'skill_cards': judge.cards}, ensure_ascii=False))
    assert off.prompts[0] == expected
    assert head.startswith('Judge the completed node in last, using the CURRENT photograph and status JSON. ')
    p = blind.prompts[0]
    assert 'judge success from the photograph alone' in p
    for leak in ('pour_failed', 'cup tipped', 'spill', 'runner_detail', 'FAILURE', 'pending'):
        assert leak not in p.split('\n', 1)[1].split('"skill_cards"')[0], leak
    assert '"status": {"last": {"node": "tilt_blue"}}' in p
    assert out['blind'] is True


def test_each_judge_call_saves_image_prompt_and_raw(tmp_path):
    log = tmp_path / 'sup' / 'ts' / 'decisions.jsonl'
    sup = Supervisor(FakeInterface(tmp_path / 'approvals.jsonl'), Judge(Backend(OK), blind=True),
                     mode='shadow', log_path=log)
    sup.step(status())
    row = json.loads(log.read_text().splitlines()[-1])
    assert row['decision']['files'] == ['decisions/0001_tilt_blue_judge.jpg', 'decisions/0001_tilt_blue_judge.json']
    assert 'prompt' not in row['decision'] and row['decision']['blind'] is True
    saved = json.loads((log.parent / row['decision']['files'][1]).read_text())
    assert saved['raw'] == OK and saved['prompt'].startswith('Judge the completed node')
    assert Image.open(log.parent / row['decision']['files'][0]).size == (2, 2)


def test_record_failure_does_not_block_decision(tmp_path):
    class BadImage:
        def convert(self, mode):
            raise OSError('disk full')
    sup = Supervisor(FakeInterface(tmp_path / 'a.jsonl'), Judge(Backend(OK)), log_path=tmp_path / 'd.jsonl')
    row = sup.record('n', 'judge', BadImage(), {'action': 'next', 'raw': OK, 'prompt': 'p'})
    assert row == {'action': 'next', 'raw': OK, 'files': ['decisions/0001_n_judge.json']}  # 쓴 파일만


def test_blind_prompt_is_identical_across_runner_outcomes():
    ok = status()
    fail = failed_status() | {'last': failed_status()['last'] | {'code': 'zz_unknown_code'}}
    pend = status() | {'pending': {'action': 'rollback_retry', 'checkpoint': 'cp'},
                       'next_action': 'recover:tilt_blue:rollback_retry',
                       'last': {'node': 'tilt_blue', 'code': 'grasp_failed_right', 'reason': 'slip'}}
    assert runner_detail(fail)['unknown_code'] and 'recover' in allowed_actions(pend, runner_detail(pend))
    prompts = []
    for st in (ok, fail, pend):
        b = Backend(OK)
        Judge(b, 'sentence', ['plan'], blind=True).judge(st, Image.new('RGB', (2, 2)))
        prompts.append(b.prompts[0])
    assert prompts[0] == prompts[1] == prompts[2]
    assert 'A=success/next, B=failure/retry, C=failure/rescan, D=' in prompts[0] and 'runner approval' not in prompts[0]


def test_record_keeps_first_prompt_only_in_json(tmp_path):
    sup = Supervisor(FakeInterface(tmp_path / 'a.jsonl'), Judge(Backend(OK)), log_path=tmp_path / 'd.jsonl')
    row = sup.record('x' * 200, 'approach', Image.new('RGB', (2, 2)),
                     {'valid': True, 'prompt': 'p2', 'retried_from': {'error': 'e', 'prompt': 'p1'}})
    assert row['retried_from'] == {'error': 'e'}
    stem = '0001_' + 'x' * 80 + '_approach'
    assert row['files'] == [f'decisions/{stem}.jpg', f'decisions/{stem}.json']
    saved = json.loads((tmp_path / row['files'][1]).read_text())
    assert saved['retried_from']['prompt'] == 'p1' and saved['prompt'] == 'p2'
