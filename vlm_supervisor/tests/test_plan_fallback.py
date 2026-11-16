import json

import pytest
from PIL import Image

from vlm_supervisor import cli
from vlm_supervisor.planner import Planner, simulate


class Backend:
    def __init__(self, answer):
        self.answer = answer

    def ask(self, images, text, **kwargs):
        assert kwargs['max_new_tokens'] == 1024
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer, {'model': 'fake'}


def test_planner_normalizes_dict_steps_and_missing_fields():
    raw = json.dumps([{'skill': 'go_home', 'hand': 'both'},
                      ['scene_snapshot', '-', '-', '-'],
                      {'skill': 'terminal', 'target': 'SUCCESS'}])
    result = Planner(Backend(raw)).plan('test')
    assert result['steps'] == [['go_home', 'both', '-', '-'],
                               ['scene_snapshot', '-', '-', '-'],
                               ['terminal', '-', '-', 'SUCCESS']]
    assert result['simulation']['ok']


@pytest.mark.parametrize('answer,source,code', [
    ('[{"skill":"go_home","hand":"right"}]', 'fallback', 0),
    ('[{"skill":"go_home"', 'fallback', 0),
    ('[["terminal","-","-","SUCCESS"]]', 'vlm', 0),
    (RuntimeError('network down'), None, 2),
])
def test_cli_plan_handling(tmp_path, monkeypatch, capsys, answer, source, code):
    calls = []

    class Interface:
        def __init__(self, *args):
            pass

        def frame(self, **kwargs):
            return Image.new('RGB', (2, 2))

        def perform(self, action, **kwargs):
            calls.append(action)
            return {'ok': True}

        def close(self):
            calls.append('close')

    def run(supervisor):
        assert simulate(supervisor.judge.plan)['ok']
        calls.append('run')
        return 0

    monkeypatch.setattr(cli, 'RosInterface', Interface)
    monkeypatch.setattr(cli, 'Router', lambda *args: Backend(answer))
    monkeypatch.setattr(cli.Supervisor, 'run', run)
    assert cli.main(['--sentence', 'test', '--mode', 'active', '--log-dir', str(tmp_path)]) == code
    run_dir, = tmp_path.iterdir()
    output = capsys.readouterr().out
    if source:
        plan = json.loads((run_dir / 'plan.json').read_text())
        assert plan['source'] == source
        assert 'run' in calls and 'stop' not in calls
        if source == 'fallback':
            assert plan['simulation']['world']['poured'] == ['SRC_BLUE', 'SRC_PINK']
            event, = [row for line in (run_dir / 'decisions.jsonl').read_text().splitlines()
                      if (row := json.loads(line))['event'] == 'plan_fallback']
            assert event['event'] == 'plan_fallback'
            assert event['vlm_answer_head'] == answer[:400]
            assert event['reason']
            assert 'plan_fallback' in output
    else:
        assert 'run' not in calls and 'stop' in calls
        assert 'startup_error' in output


def test_fixed_fallback_passes_simulation():
    path = cli.build_parser().parse_args(['--sentence', 'test']).plan_fallback
    from pathlib import Path
    result = simulate(json.loads(Path(path).read_text()))
    assert result['ok'], result['errors']
    assert result['world']['poured'] == ['SRC_BLUE', 'SRC_PINK']


def test_invalid_fallback_keeps_startup_error(tmp_path, monkeypatch, capsys):
    class Interface:
        def __init__(self, *args):
            pass

        def frame(self, **kwargs):
            return Image.new('RGB', (2, 2))

        def perform(self, action, **kwargs):
            assert action == 'stop'
            return {'ok': True}

        def close(self):
            pass

    fallback = tmp_path / 'invalid.json'
    fallback.write_text('[]')
    monkeypatch.setattr(cli, 'RosInterface', Interface)
    monkeypatch.setattr(cli, 'Router', lambda *args: Backend('[]'))
    assert cli.main(['--sentence', 'test', '--log-dir', str(tmp_path / 'runs'),
                     '--plan-fallback', str(fallback)]) == 2
    assert 'invalid fallback plan' in capsys.readouterr().out
    assert not list(tmp_path.glob('runs/*/plan.json'))


@pytest.mark.parametrize('plan_order', ['shadow', 'apply'])
def test_cli_judge_plan_with_mapping_error_and_unreadable_fallback(tmp_path, monkeypatch, plan_order):
    seen = []

    class Interface:
        def __init__(self, *args):
            pass

        def frame(self, **kwargs):
            return Image.new('RGB', (2, 2))

        def status(self):
            raise RuntimeError('no runner')  # mapping error -> executed None

        def perform(self, action, **kwargs):
            return {'ok': True}

        def close(self):
            pass

    monkeypatch.setattr(cli, 'RosInterface', Interface)
    monkeypatch.setattr(cli, 'Router', lambda *args: Backend('[["terminal","-","-","SUCCESS"]]'))
    monkeypatch.setattr(cli.Supervisor, 'run', lambda sup: seen.append(sup.judge.plan) or 0)
    assert cli.main(['--sentence', 'test', '--log-dir', str(tmp_path), '--plan_order', plan_order,
                     '--plan-fallback', str(tmp_path / 'missing.json')]) == 0
    assert seen == [[['terminal', '-', '-', 'SUCCESS']]]


def test_planner_prompt_shows_blue_first_reason_and_rejects_pink_first():
    pink_first = [['go_home', 'both', '-', '-'], ['scene_snapshot', '-', '-', '-'],
                  ['grasp', 'right', 'SRC_PINK', '-']]
    seen = {}

    class Spy(Backend):
        def ask(self, images, text, **kwargs):
            seen['prompt'] = text
            return super().ask(images, text, **kwargs)

    with pytest.raises(ValueError, match='SRC_BLUE'):
        Planner(Spy(json.dumps(pink_first))).plan('test')
    assert '먼저_치울_물체' in seen['prompt'] and '파랑을 먼저 잡아' in seen['prompt']
