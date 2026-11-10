import json

import pytest
from PIL import Image

from vlm_supervisor.backends import Ledger, Local, Router
from vlm_supervisor.cli import build_parser


def test_cli_shadow_default():
    args = build_parser().parse_args(['--runner', 'runner', '--sentence', '옮겨줘'])
    assert args.mode == 'shadow'


def test_ledger_cap_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv('VLM_CAP_test', '1')
    ledger = Ledger(tmp_path)
    assert ledger.usable('test')
    ledger.count('test', {'in_tok': 3})
    assert not Ledger(tmp_path).usable('test')
    assert json.loads((tmp_path / 'usage.json').read_text())


def test_router_missing_key_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('GEMINI_KEY_FILE', str(tmp_path / 'absent'))
    monkeypatch.setattr(Local, 'ask', lambda self, images, text, level='low': ('ok', {'model': 'local'}))
    router = Router(['gemini-test', 'local'], state_dir=tmp_path)
    text, meta = router.ask([Image.new('RGB', (2, 2))], 'prompt')
    assert text == 'ok'
    assert meta['fallbacks']
    assert (tmp_path / 'calls.jsonl').exists()


def test_local_text_only_plan_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match='image'):
        Local(Ledger(tmp_path)).ask([], 'plan')


def test_ledger_resets_on_new_day(tmp_path, monkeypatch):
    import vlm_supervisor.backends as backends

    monkeypatch.setenv('VLM_CAP_test', '1')
    monkeypatch.setattr(backends, 'pt_day', lambda: '2026-11-10')
    ledger = Ledger(tmp_path)
    ledger.count('test', {})
    assert not ledger.usable('test')
    monkeypatch.setattr(backends, 'pt_day', lambda: '2026-11-11')
    assert ledger.usable('test')


@pytest.mark.parametrize("tokens", [None, 1024])
def test_local_http_contract_and_ledger(tmp_path, monkeypatch, tokens):
    import base64
    import io
    import vlm_supervisor.backends as backends

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'text': '{"choice":"A"}'}

    sent = []

    def post(url, *, timeout, json):
        sent.append((url, timeout, json))
        return Response()

    monkeypatch.setattr(backends.requests, 'post', post)
    monkeypatch.setenv('VLM_LOCAL_URL', 'http://example.invalid:8765/')
    local = Local(Ledger(tmp_path))
    answer, _ = local.ask([Image.new('RGB', (2, 2))], 'judge',
                          **({} if tokens is None else {'max_new_tokens': tokens}))
    url, timeout, body = sent[0]
    assert url == 'http://example.invalid:8765/infer'
    assert timeout > 0
    assert 'judge' in body['prompt']
    assert body['max_new_tokens'] == (400 if tokens is None else tokens)
    image = Image.open(io.BytesIO(base64.b64decode(body['image_png_b64'])))
    assert image.size == (2, 2)
    assert answer == '{"choice":"A"}'
    assert local.ledger._m('local')['n'] == 1


def test_gemini_daily_quota_blocks_and_routes_to_local(tmp_path, monkeypatch):
    import vlm_supervisor.backends as backends

    class Response:
        status_code = 429
        text = 'PerDay quota exceeded'

    monkeypatch.setenv('GEMINI_API_KEY', 'test-value')
    monkeypatch.setattr(backends.requests, 'post', lambda *a, **kw: Response())
    monkeypatch.setattr(Local, 'ask', lambda self, images, text, level: ('local result', {'model': 'local'}))
    router = Router(['gemini-test', 'local'], state_dir=tmp_path)
    answer, meta = router.ask([Image.new('RGB', (2, 2))], 'judge')
    assert answer == 'local result'
    assert router.ledger.blocked('gemini-test')
    assert meta['fallbacks']


def test_router_plan_tokens_only_forwarded_to_local(tmp_path, monkeypatch):
    router = Router(['gemini-test', 'local'], state_dir=tmp_path)

    class Cloud:
        def ask(self, images, text, level):
            raise RuntimeError('cloud unavailable')

    router.backends['gemini-test'] = Cloud()
    def ask(self, images, text, level, max_new_tokens=400):
        return str(max_new_tokens), {'model': 'local'}

    monkeypatch.setattr(Local, 'ask', ask)
    answer, meta = router.ask([], 'plan', max_new_tokens=1024)
    assert answer == '1024'
    assert 'cloud unavailable' in meta['fallbacks'][0]

    class CloudSuccess:
        def ask(self, images, text, level):
            return 'cloud plan', {'model': 'gemini-test'}

    router.backends['gemini-test'] = CloudSuccess()
    assert router.ask([], 'plan', max_new_tokens=1024)[0] == 'cloud plan'
