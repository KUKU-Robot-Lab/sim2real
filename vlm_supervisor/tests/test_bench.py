"""bench: 합성 기록으로 항목 참값(build_set)과 변형별 호출·집계(run_bench)를 본다. VLM 서버는 로컬 스텁이다."""
import base64
import collections
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml
from PIL import Image

from vlm_supervisor.bench import build_set, run_bench
from vlm_supervisor.planner import load_cards

RAW = '```json\n{"success": true, "choice": "A", "reason": "all good"}\n```'  # 저장된 VLM 답은 실패 경계에서도 늘 성공이다
DARK = '{"success": false, "choice": "E", "reason": "nothing visible"}'


def ex(node, policy=None, result='COMPLETED', **signals):
    """env0000_episode.jsonl 의 exit 행."""
    return {'event': 'exit', 'state_id': node, 'policy_name': policy, 'result': result,
            'signals': {policy: signals} if signals else {}}


def place(node, policy, grade):
    return ex(node, policy, hand_empty=True, scripted_judge={'grade': grade})


def write_seed(root, name, exits, decisions, objects=None):
    """시드 폴더 하나. decisions = (node, kind). kind 가 None 이면 사진·기록을 못 남긴 판정, objects = 끝 상태(홀더 번호, 등급)."""
    sup = root / name / 'sup' / '20261116_000000_000000'
    (sup / 'decisions').mkdir(parents=True)
    rows = [{'event': 'plan_fallback'}]
    for n, (node, kind) in enumerate(decisions, 1):
        stem, files = f'{n:04d}_{node}_{kind}', []
        if kind:
            color = (20 * n, 200, 60 * int(name[-1]))  # 경계마다 다른 색: 어느 사진을 보냈는지 가린다
            Image.new('RGB', (8, 6), color).save(sup / 'decisions' / f'{stem}.jpg')
            raw = DARK if (name, node) == ('seed1', 'place_pink') else RAW  # 다시 물으면 답이 달라질 기록 하나
            (sup / 'decisions' / f'{stem}.json').write_text(json.dumps(
                {'node': node, 'kind': kind, 'prompt': f'PROMPT {name} {stem}', 'raw': raw}), encoding='utf-8')
            files = [f'decisions/{stem}.jpg', f'decisions/{stem}.json']
        event = {None: 'decision', 'judge': 'decision'}.get(kind, kind)  # place_path·approach 는 판정 경계가 아니다
        rows.append({'event': event, 'node': node, 'decision': {'success': True, 'choice': 'A', 'raw': RAW, 'files': files}}
                    if event in ('decision', 'final_judge') else {'event': event, 'node': node})
    (sup / 'decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (root / name / 'env0000_episode.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in [{'event': 'enter', 'state_id': 'go_home_start'}, *exits]), encoding='utf-8')
    if objects is not None:
        (root / name / 'result.jsonl').write_text(json.dumps(
            {'objects': {k: {'holder': h, 'grade': g} for k, (h, g) in objects.items()}}) + '\n', encoding='utf-8')


def make_logs(root):
    # seed0: lanes 잡기 → 붓기 → place_blue 가 실제로 실패(컵이 넘어짐)하고 승인 없이 home → place_shaker 뒤 승인 없이 끝
    write_seed(root, 'seed0', [
        ex('go_home_start', 'go_home'), ex('scene'),
        ex('grasp_shaker', 'rh_aglt_l', dropped=False), ex('grasp_blue', 'rh_aglt_r', dropped=False),
        ex('to_tilt_start_l', 'to_tilt_start_l'), ex('to_tilt_start_r1', 'to_tilt_start_r'), ex('grasp_to_tilt_start'),
        ex('tilt_blue', 'tilt'), place('place_blue', 'rh_place_r', 'fail'), ex('home_r_after_blue', 'go_home_r'),
        place('place_shaker', 'rh_place_l', 'great'), ex('go_home_end', 'go_home'),
    ], [(None, 'judge'), ('go_home_start', 'judge'), ('scene', 'judge'), ('grasp_to_tilt_start', 'judge'), ('tilt_blue', 'judge'),
        ('place_blue', 'place_path'), ('home_r_after_blue', 'judge'), ('go_home_end', 'final_judge'),
    ], {'blue': (1, 'fail'), 'pink': (2, 'fail'), 'shaker': (0, 'great')})
    # seed1: 놓은 직후 great 인데 끝 상태가 fail(뒤에 건드림), dropped 신호 없는 잡기, partial(기운 채 구멍 안). 셋 다 못 정함
    write_seed(root, 'seed1', [
        ex('go_home_start', 'go_home'), ex('scene'), ex('grasp_blue', 'rh_aglt_r', dropped=False),
        ex('to_tilt_start_r1', 'to_tilt_start_r'), place('place_blue', 'rh_place_r', 'great'),
        ex('grasp_pink', 'rh_aglt_r'), place('place_pink', 'rh_place_r', 'partial'),
    ], [('go_home_start', 'judge'), ('scene', 'judge'), ('grasp_blue', 'judge'), ('to_tilt_start_r1', 'judge'),
        ('place_blue', 'judge'), ('grasp_pink', 'judge'), ('place_pink', 'judge'),
    ], {'blue': (1, 'fail'), 'pink': (2, 'partial')})
    # seed2: 아직 도는 시드(result.jsonl 없음). 사진 없는 판정, 떨어뜨린 잡기, 끊긴 이송, 손을 못 뗀 놓기, exit 행이 없는 판정
    write_seed(root, 'seed2', [
        ex('go_home_start', 'go_home'), ex('scene'), ex('grasp_pink', 'rh_aglt_r', result='FAILED', dropped=True),
        ex('grasp_shaker', 'rh_aglt_l', dropped=False), ex('to_tilt_start_l', 'to_tilt_start_l', result='ABORTED'),
        ex('place_shaker', 'rh_place_l', hand_empty=False, scripted_judge={'grade': 'fail'}),
    ], [(None, 'judge'), ('go_home_start', 'judge'), ('scene', 'judge'), ('grasp_pink', None), ('grasp_shaker', 'judge'),
        ('to_tilt_start_l', 'judge'), ('place_shaker', 'judge'), ('tilt_pink', 'judge')])
    return root


@pytest.fixture
def server():
    """가짜 VLM 서버: 검은 사진이면 no·실패, 아니면 긴 프롬프트에는 늘 성공, 짧은 질문에는 empty 를 물을 때만 yes."""
    seen = []

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            photos = [Image.open(io.BytesIO(base64.b64decode(b))) for b in req['images_b64']]
            dark = photos[-1].convert('L').getextrema() == (0, 0)
            seen.append({**req, 'sizes': [p.size for p in photos], 'dark': dark})
            if 'choices' in req:
                yes = 0.9 if not dark and 'empty' in req['prompt'].split('Question:')[1] else 0.2
                body = {'text': '', 'scores': {'yes': yes, 'no': 1 - yes}, 'choice_mass': 1.0, 'gen_s': 0.01}
            else:
                body = {'text': DARK if dark else RAW, 'gen_s': 0.01}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(('127.0.0.1', 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_port}', seen
    httpd.shutdown()


def test_build_truth_comes_from_runner_signals_not_vlm(tmp_path, capsys):
    logs, out = make_logs(tmp_path / 'logs'), tmp_path / 'set' / 'items.jsonl'
    assert build_set.main([str(logs), '--out', str(out)]) == 0
    items = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    by = {i['id']: i for i in items}
    assert len(by) == len(items)
    assert {(i['truth'], i['kind']) for i in items} == {('yes', 'pos'), ('no', 'neg_fail'), ('no', 'neg_cross')}
    # 승인 없이 이어진 place_blue 는 그 뒤 경계(home_r_after_blue) 사진에 붙는다. 저장된 VLM 답은 성공이지만 참값은 sim 등급 fail
    fail = by['seed0/0007_home_r_after_blue_judge/cup_in_holder:SRC_BLUE']
    assert (fail['truth'], fail['kind']) == ('no', 'neg_fail') and 'scripted_judge.grade=fail' in fail['truth_src']
    assert fail['question'] == 'Is the blue cup standing upright inside a holder?'
    assert fail['image'].endswith('0007_home_r_after_blue_judge.jpg') and fail['image_before'].endswith('0005_tilt_blue_judge.jpg')
    assert fail['decision_json'].endswith('0007_home_r_after_blue_judge.json')
    # lanes 경계 하나에 두 손의 잡기가 같이 붙고, 손에 든 컵은 홀더 질문의 쉬운 반례가 된다
    lane = 'seed0/0004_grasp_to_tilt_start_judge/'
    assert by[lane + 'cup_in_hand:left:SHAKER']['kind'] == by[lane + 'cup_in_hand:right:SRC_BLUE']['kind'] == 'pos'
    assert by[lane + 'cup_in_hand:left:SHAKER']['question'] == "Is the robot's left hand holding the orange shaker cup?"
    assert by[lane + 'cup_in_holder:SRC_BLUE']['kind'] == 'neg_cross'
    assert by[lane + 'cup_lifted:SRC_BLUE']['image_swap'].endswith('0001_None_judge.jpg')  # 잡기 조건이 모두 깨진 사진
    # place_shaker 는 마지막 final 판정에 붙고 terminal 질문도 거기 붙는다
    final = 'seed0/0008_go_home_end_final_judge/'
    assert by[final + 'cup_in_holder:SHAKER']['kind'] == 'pos' and by[final + 'all_in_holders']['kind'] == 'neg_fail'
    # 실패 뒤 상태는 다른 경계의 쉬운 반례로 쓰지 않는다. tilt 경계에는 제 질문이 없다. 첫 경계에는 직전 사진이 없다
    assert final + 'cup_in_holder:SRC_BLUE' not in by
    assert {i['kind'] for i in items if '/0005_tilt_blue_judge/' in i['id']} == {'neg_cross'}
    assert by['seed0/0001_None_judge/cup_in_holder:SRC_BLUE']['image_before'] is None
    # 못 정한 것은 항목을 만들지 않고 이유만 센다
    assert not any(i['question_id'] == 'cup_in_holder' and i['kind'] != 'neg_cross' for i in items if i['seed'] == 'seed1')
    assert 'seed1/0006_grasp_pink_judge/cup_in_hand:right:SRC_PINK' not in by  # dropped 신호가 없으면 쥐었다고 보지 않는다
    assert by['seed1/0004_to_tilt_start_r1_judge/cup_in_hand:right:SRC_BLUE']['kind'] == 'pos'  # 이송 노드의 컵 = 그 손이 쥔 컵
    assert by['seed1/0007_place_pink_judge/holders_empty']['kind'] == 'neg_cross'  # 기운 채라도 홀더 안이면 빈 홀더가 아니다
    # 떨어뜨린 잡기는 no. 사진 없는 경계(grasp_pink)의 노드는 다음 경계(grasp_shaker)가 이어받는다
    assert by['seed2/0005_grasp_shaker_judge/cup_in_hand:right:SRC_PINK']['kind'] == 'neg_fail'
    # 손을 못 뗀 놓기도 구멍 밖(fail)이면 홀더 질문은 no 가 확실하다
    assert by['seed2/0007_place_shaker_judge/cup_in_holder:SHAKER']['kind'] == 'neg_fail'
    assert {i['question_id'] for i in items} == {'holders_empty', 'cup_in_hand', 'cup_lifted', 'cup_in_holder', 'all_in_holders'}
    skipped = build_set.build(str(logs))[1]
    assert skipped['경계: 판정 사진·기록 없음'] == skipped['경계: 노드의 exit 행 없음'] == 1
    for reason in ('grade=partial', '끝 grade=fail', '쥔 컵을 모름', 'SRC_PINK=out'):
        assert any(reason in key for key in skipped), reason
    printed = capsys.readouterr().out
    assert f'항목 {len(items)}개' in printed and '참값을 못 정해 뺀 것' in printed


def test_run_bench_calls_server_per_variant_and_scores(tmp_path, server):
    url, seen = server
    items, _ = build_set.build(str(make_logs(tmp_path / 'logs')))
    rows = run_bench.run(items, url, tmp_path / 'out')
    assert [json.loads(line) for line in (tmp_path / 'out' / 'results.jsonl').read_text(encoding='utf-8').splitlines()] == rows
    assert not any('error' in r for r in rows)
    s = run_bench.stats(rows)
    own = collections.defaultdict(list)  # 경계 → 그 노드의 제 질문 참값
    for i in items:
        own[i['decision_json']] += [i['truth']] if i['kind'] != 'neg_cross' else []
    n_fail, n_ok = sum('no' in v for v in own.values()), sum(bool(v) and 'no' not in v for v in own.values())
    n_swap = len({i['decision_json'] for i in items if i['image_swap']})
    assert n_fail >= 2 and 0 < n_swap < len(own)
    # long: 경계당 한 번, 저장된 프롬프트 전문을 기록 때와 같은 머리말·토큰 상한으로 보낸다
    long_sent = [q for q in seen if 'choices' not in q]
    assert len(long_sent) == 2 * len(own) + n_swap
    assert all(q['prompt'].startswith('(Only the CURRENT image is attached.)\nPROMPT seed') and q['max_new_tokens'] == 400
               and len(q['images_b64']) == 1 for q in long_sent)
    same = [(r['same_raw'], r['same_choice']) for r in rows if r['variant'] == 'long_same']
    assert same.count((True, True)) == len(own) - 1 and same.count((False, False)) == 1  # seed1 place_pink 기록만 다르다
    lane = next(i for i in items if i['id'].startswith('seed0/0004_'))
    sent = [q['images_b64'][0] for q in long_sent if q['prompt'].endswith('PROMPT seed0 0004_grasp_to_tilt_start_judge')]
    assert sent[:2] == [run_bench.b64(lane['image']), run_bench.b64(lane['image_swap'])] and sent[2] == run_bench.black(lane['image'])
    # 스텁은 사진이 검지 않으면 긴 프롬프트에 늘 성공이라고 답한다: 실패 경계를 통과시키고 바꾼 사진에도 답이 그대로다
    assert s['long_same']['n'] == s['long_black']['n'] == len(own) and s['long_swap']['n'] == n_swap
    assert (s['long_same']['acc'], s['long_same']['fp_fail'], s['long_same']['fh']) == ((n_ok, n_ok + n_fail), (n_fail, n_fail), (0, n_ok))
    assert (s['long_swap']['fp_cross'], s['long_swap']['flip']) == ((n_swap, n_swap), (0, n_swap))
    assert s['long_black']['flip'] == (len(own), len(own)) and s['long_black']['acc'] == (0, 0)
    # short: 항목당 한 번, yes/no 점수만. 스텁의 답(empty 를 물으면 yes)으로 기대값을 따로 센다
    short_sent = [q for q in seen if 'choices' in q]
    assert all(q['choices'] == ['yes', 'no'] and q['max_new_tokens'] == 0 for q in short_sent)
    assert collections.Counter(len(q['images_b64']) for q in short_sent) == {
        1: 2 * len(items), 2: sum(i['image_before'] is not None for i in items)}
    assert all(q['sizes'][-1] == (8, 6) for q in seen) and sum(q['dark'] for q in seen) == len(own) + len(items)
    at_lane = [i for i in items if i['image'] == lane['image']]  # 한 경계의 항목들로 문구 틀과 사진 순서(직전, 지금)를 본다
    pair = [run_bench.b64(lane['image_before']), run_bench.b64(lane['image'])]
    assert sorted(q['prompt'] for q in short_sent if q['images_b64'] == pair) == sorted(
        run_bench.SHORT2.format(question=i['question']) for i in at_lane)
    assert sorted(q['prompt'] for q in short_sent if q['images_b64'] == pair[1:]) == sorted(
        run_bench.SHORT1.format(question=i['question']) for i in at_lane)

    def guess(i):
        return 'yes' if 'empty' in i['question'] else 'no'

    assert s['short1']['acc'] == (sum(guess(i) == i['truth'] for i in items), len(items))
    for kind, key, said in (('neg_cross', 'fp_cross', 'yes'), ('neg_fail', 'fp_fail', 'yes'), ('pos', 'fh', 'no')):
        of_kind = [i for i in items if i['kind'] == kind]
        assert s['short1'][key] == (sum(guess(i) == said for i in of_kind), len(of_kind)), key
    assert s['short2']['n'] == sum(i['image_before'] is not None for i in items) < s['short1']['n'] == len(items)
    assert s['short_black']['flip'] == (sum(guess(i) == 'yes' for i in items), len(items)) and s['short_black']['acc'] == (0, 0)
    text = run_bench.summary(items, rows, 'items.jsonl', url)
    assert f'시드 3개 · 경계 {len(own)}개 · 항목 {len(items)}개' in text and repr(run_bench.SHORT2) in text
    check = next(line for line in text.splitlines() if 'long_same 자체 점검' in line)
    assert check.count(f'{len(own) - 1}/{len(own)} (') == 2  # 글자 일치와 choice 일치
    assert all(f'| {v} |' in text for v in run_bench.VARIANTS)
    assert 'short1 1.00/1.00 · short2 1.00/1.00 · short_black 1.00/1.00' in text  # 선택지 질량은 변형별로


def test_cli_runs_chosen_variants_and_records_server_errors(tmp_path, server):
    logs, out = make_logs(tmp_path / 'logs'), tmp_path / 'set' / 'items.jsonl'
    build_set.main([str(logs), '--out', str(out)])
    assert run_bench.main([str(out), '--url', server[0], '--out', str(tmp_path / 'ok'), '--variants', 'short1']) == 0
    table = (tmp_path / 'ok' / 'summary.md').read_text(encoding='utf-8')
    assert '| short1 |' in table and '| long_same |' not in table
    with pytest.raises(FileExistsError):  # 잰 결과는 덮어쓰지 않는다
        run_bench.main([str(out), '--url', server[0], '--out', str(tmp_path / 'ok'), '--variants', 'short1'])
    with socket.socket() as sock:  # 닫힌 포트: 서버가 죽어 있어도 예외로 끝나지 않고 오류로 적힌다
        sock.bind(('127.0.0.1', 0))
        dead = f'http://127.0.0.1:{sock.getsockname()[1]}'
    items = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()][:3]
    rows = run_bench.run(items, dead, tmp_path / 'dead', ['short1'])
    assert len(rows) == 3 and all('error' in r and 'answer' not in r for r in rows)
    assert run_bench.stats(rows)['short1'] == {
        'n': 0, 'errors': 3, 'acc': (0, 0), 'fp_cross': (0, 0), 'fp_fail': (0, 0), 'fh': (0, 0), 'flip': (0, 0),
        'sec_med': None, 'sec_max': None}


def test_questions_match_truth_table_and_cards():
    questions = yaml.safe_load(build_set.QUESTIONS.read_text(encoding='utf-8'))
    skills = {card['id'] for card in load_cards()['스킬']}
    start = {'w': {**dict.fromkeys(questions['objects'], 'table'), 'left': None, 'right': None},
             'why': collections.defaultdict(str)}
    for q in questions['questions']:
        assert set(q['at']) <= skills and 'bead' not in q['text'] and q['card'] and q['truth'], q['id']
        q['text'].format(hand='right', obj='blue cup')  # 틀은 {hand}·{obj} 만 쓴다
        assert build_set.truth_of((q['id'], 'right', 'SRC_BLUE'), start, True)[0] in ('yes', 'no'), q['id']
    assert all(e['card'] and e['why'] for e in questions['excluded'])
