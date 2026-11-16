"""bench: 합성 기록으로 항목 참값(build_set)과 변형별 호출·집계(run_bench)를 본다. VLM 서버는 로컬 스텁이다."""
import base64
import collections
import io
import json
import socket
import threading
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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
    # 이송 exit 의 dropped 신호는 seed0 왼손에만 있다(실제 기록에는 없다). 없는 쪽은 손 안이라는 yes 를 못 정한다
    # seed0: lanes 잡기 → 붓기 → place_blue 가 실제로 실패(컵이 넘어짐)하고 승인 없이 home → place_shaker 뒤 승인 없이 끝
    write_seed(root, 'seed0', [
        ex('go_home_start', 'go_home'), ex('scene'),
        ex('grasp_shaker', 'rh_aglt_l', dropped=False), ex('grasp_blue', 'rh_aglt_r', dropped=False),
        ex('to_tilt_start_l', 'to_tilt_start_l', dropped=False), ex('to_tilt_start_r1', 'to_tilt_start_r'), ex('grasp_to_tilt_start'),
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
    """가짜 VLM 서버: 검은 사진이면 no·실패, 아니면 긴 프롬프트에는 늘 성공, 짧은 질문에는 empty 를 물을 때만 yes.

    mode['long'] = 'garbage'(JSON 아닌 글), mode['short'] = noscores · tie · lowmass · flag(서버가 low_mass 로 표시) · list 로 답을 망가뜨린다.
    """
    seen, mode = [], {}

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            photos = [Image.open(io.BytesIO(base64.b64decode(b))) for b in req['images_b64']]
            dark = photos[-1].convert('L').getextrema() == (0, 0)
            seen.append({**req, 'sizes': [p.size for p in photos], 'dark': dark})
            if 'choices' in req:
                yes = 0.9 if not dark and 'empty' in req['prompt'].split('Question:')[1] else 0.2
                probs = {'yes': yes, 'no': 1 - yes}
                body = {'text': '', 'scores': probs, 'probs': probs, 'choice_mass': 1.0, 'gen_s': 0.01}
                body = {'noscores': {'text': ''}, 'tie': {**body, 'scores': {'yes': 0.5, 'no': 0.5}},
                        'lowmass': {**body, 'choice_mass': 0.2}, 'list': [1, 2],
                        'flag': {'text': '', 'scores': {}, 'probs': probs, 'choice_mass': 0.01, 'low_mass': True},
                        }.get(mode.get('short'), body)
            else:
                body = {'text': 'It looks fine to me.' if mode.get('long') == 'garbage' else DARK if dark else RAW, 'gen_s': 0.01}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(('127.0.0.1', 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_port}', seen, mode
    httpd.shutdown()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_build_truth_comes_from_runner_signals_not_vlm(tmp_path, capsys):
    logs, out = make_logs(tmp_path / 'logs'), tmp_path / 'set' / 'items.jsonl'
    assert build_set.main([str(logs), '--out', str(out)]) == 0
    items = read(out)
    by = {i['id']: i for i in items}
    assert len(by) == len(items)
    assert {(i['truth'], i['kind']) for i in items} == {('yes', 'pos'), ('no', 'neg_fail'), ('no', 'neg_cross')}
    # 승인 없이 이어진 place_blue 는 그 뒤 경계(home_r_after_blue) 사진에 붙는다. 저장된 VLM 답은 성공이지만 참값은 sim 등급 fail
    fail = by['seed0/0007_home_r_after_blue_judge/cup_in_holder:SRC_BLUE']
    assert (fail['truth'], fail['kind']) == ('no', 'neg_fail') and 'scripted_judge.grade=fail' in fail['truth_src']
    assert fail['question'] == 'Is the blue cup standing upright inside a holder?'
    assert fail['image'].endswith('0007_home_r_after_blue_judge.jpg') and fail['image_before'].endswith('0005_tilt_blue_judge.jpg')
    assert fail['decision_json'].endswith('0007_home_r_after_blue_judge.json')
    # lanes 경계 하나에 두 손의 잡기·이송이 같이 붙는다. 이송 exit 에 dropped=false 가 있는 왼손만 yes, 신호가 없는 오른손은 못 정함
    lane = 'seed0/0004_grasp_to_tilt_start_judge/'
    assert by[lane + 'cup_in_hand:left:SHAKER']['kind'] == by[lane + 'cup_lifted:SHAKER']['kind'] == 'pos'
    assert by[lane + 'cup_in_hand:left:SHAKER']['question'] == "Is the robot's left hand holding the orange shaker cup?"
    assert lane + 'cup_in_hand:right:SRC_BLUE' not in by and lane + 'cup_lifted:SRC_BLUE' not in by
    assert by[lane + 'cup_in_holder:SRC_BLUE']['kind'] == 'neg_cross'  # 손에 든 컵은 홀더 질문의 쉬운 반례다(떨어뜨렸어도 no)
    assert by[lane + 'cup_lifted:SHAKER']['image_swap'].endswith('0001_None_judge.jpg')  # 잡기 조건이 모두 깨진 사진
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
    assert by['seed1/0003_grasp_blue_judge/cup_in_hand:right:SRC_BLUE']['kind'] == 'pos'
    assert 'seed1/0004_to_tilt_start_r1_judge/cup_in_hand:right:SRC_BLUE' not in by  # 이송 exit 에 신호가 없다
    assert by['seed1/0007_place_pink_judge/holders_empty']['kind'] == 'neg_cross'  # 기운 채라도 홀더 안이면 빈 홀더가 아니다
    # 떨어뜨린 잡기는 no. 사진 없는 경계(grasp_pink)의 노드는 다음 경계(grasp_shaker)가 이어받고, 그 경계는 직전 사진이 없다
    assert by['seed2/0005_grasp_shaker_judge/cup_in_hand:right:SRC_PINK']['kind'] == 'neg_fail'
    assert by['seed2/0005_grasp_shaker_judge/cup_in_hand:right:SRC_PINK']['image_before'] is None
    assert by['seed2/0003_scene_judge/holders_empty']['image_before'].endswith('0002_go_home_start_judge.jpg')
    # 손을 못 뗀 놓기도 구멍 밖(fail)이면 홀더 질문은 no 가 확실하다
    assert by['seed2/0007_place_shaker_judge/cup_in_holder:SHAKER']['kind'] == 'neg_fail'
    assert {i['question_id'] for i in items} == {'holders_empty', 'cup_in_hand', 'cup_lifted', 'cup_in_holder', 'all_in_holders'}
    skipped = build_set.build(str(logs))[1]
    assert skipped['경계: 판정 사진·기록 없음'] == skipped['경계: 노드의 exit 행 없음'] == 1
    for reason in ('grade=partial', '끝 grade=fail', '쥔 컵을 모름', 'SRC_PINK=out', '이송 exit 에 dropped 신호가 없어'):
        assert any(reason in key for key in skipped), reason
    assert json.loads((out.parent / 'skipped.json').read_text(encoding='utf-8')) == dict(skipped)
    # 경계별 제 질문 수 / 정한 수 / 못 정한 수. long 참값 yes 는 못 정한 것이 0 일 때만이다
    bnd = {b['id']: b for b in read(out.parent / 'boundaries.jsonl')}
    assert {k: bnd[lane[:-1]][k] for k in ('home', 'decided', 'undecided', 'long_truth')} == {
        'home': 4, 'decided': 2, 'undecided': 2, 'long_truth': None}  # 정한 둘은 yes 지만 한쪽 손을 못 정했다
    assert bnd['seed1/0003_grasp_blue_judge']['long_truth'] == 'yes' and bnd['seed1/0003_grasp_blue_judge']['undecided'] == 0
    assert bnd[final[:-1]]['long_truth'] == 'no' and bnd['seed0/0005_tilt_blue_judge']['home'] == 0
    assert bnd['seed2/0006_to_tilt_start_l_judge']['undecided'] == 1  # 끊긴 이송: 질문으로 풀지 못한 제 질문도 못 정한 수다
    assert set(bnd) >= {'/'.join(i['id'].split('/')[:2]) for i in items}
    printed = capsys.readouterr().out
    assert f'항목 {len(items)}개' in printed and '참값을 못 정해 뺀 것' in printed
    with pytest.raises(FileExistsError):  # 옆 파일까지 섞이지 않게, 있으면 거부한다
        build_set.main([str(logs), '--out', str(out)])
    out.unlink()
    with pytest.raises(FileExistsError):
        build_set.main([str(logs), '--out', str(out)])


def test_build_guards(tmp_path):
    # 손이 하나로 정해지지 않는 잡기, 목표 홀더(HOLDER_1)가 result.jsonl 에 없는 fail 놓기는 추측하지 않는다
    write_seed(tmp_path, 'seed3', [
        ex('scene'), ex('grasp_blue', 'rh_aglt', dropped=False), ex('grasp_pink', 'rh_aglt_r', dropped=False),
        place('place_pink', 'rh_place_r', 'fail'), ex('grasp_shaker', 'rh_aglt_l', dropped=False), place('place_shaker', 'rh_place_l', 'fail'),
    ], [('scene', 'judge'), ('grasp_blue', 'judge'), ('grasp_pink', 'judge'), ('place_pink', 'judge'), ('grasp_shaker', 'judge'),
        ('place_shaker', 'judge')], {'pink': (1, 'fail'), 'shaker': (0, 'fail')})
    items, skipped, bounds = build_set.build(str(tmp_path))
    by, bnd = {i['id']: i for i in items}, {b['id']: b for b in bounds}
    assert any('grasp_blue 의 손이 하나로 정해지지 않음' in k for k in skipped)
    assert not any(i['id'].startswith('seed3/0002_grasp_blue_judge/') and i['kind'] != 'neg_cross' for i in items)
    assert bnd['seed3/0002_grasp_blue_judge']['long_truth'] is None and bnd['seed3/0002_grasp_blue_judge']['undecided'] >= 1
    assert 'seed3/0004_place_pink_judge/cup_in_holder:SRC_PINK' not in by and any('HOLDER_2 행 없음' in k for k in skipped)
    assert by['seed3/0006_place_shaker_judge/cup_in_holder:SHAKER']['kind'] == 'neg_fail'  # 목표 홀더 행이 있으면 그대로 no
    # 마지막 줄만 잘린 jsonl 은 경고하고 넘어간다. 중간 줄이 깨지면 멈춘다
    log = tmp_path / 'seed3' / 'env0000_episode.jsonl'
    whole = log.read_text(encoding='utf-8')
    log.write_bytes((whole + '{"event": "exit", "state_id": "끝').encode()[:-1])  # 글자 중간에서 끊김
    with pytest.warns(UserWarning, match='잘린 마지막 줄'):
        assert [i['id'] for i in build_set.build(str(tmp_path))[0]] == list(by)
    log.write_text(whole.replace('\n', '\n{"event": "ex\n', 1), encoding='utf-8')
    with pytest.raises(ValueError, match='2번째 줄이 깨졌다'):
        build_set.build(str(tmp_path))
    log.write_text(whole, encoding='utf-8')
    # 한 시드에 sup 폴더가 둘이면 멈춘다
    (tmp_path / 'seed3' / 'sup' / '20261117_000000_000000').mkdir()
    (tmp_path / 'seed3' / 'sup' / '20261117_000000_000000' / 'decisions.jsonl').write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='sup 폴더가 2개'):
        build_set.build(str(tmp_path))


def test_run_bench_calls_server_per_variant_and_scores(tmp_path, server):
    url, seen, _ = server
    items, _, brows = build_set.build(str(make_logs(tmp_path / 'logs')))
    bnd = {b['id']: b for b in brows}
    rows = run_bench.run(items, bnd, url, tmp_path / 'out')
    meta, *written = read(tmp_path / 'out' / 'results.jsonl')
    assert written == rows and not any('error' in r for r in rows)
    assert meta['meta']['url'] == url and meta['meta']['short1'] == run_bench.SHORT1 and meta['meta']['long_tokens'] == 400
    s = run_bench.stats(rows, items, bnd)
    own = {run_bench.bid(i) for i in items}  # 항목이 있는 경계
    n_fail, n_ok = (sum(bnd[k]['long_truth'] == t for k in own) for t in ('no', 'yes'))
    n_swap = len({i['decision_json'] for i in items if i['image_swap']})
    assert n_fail >= 2 and n_ok >= 2 and n_ok + n_fail < len(own) and 0 < n_swap < len(own)
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
    # 행에는 보낸 문구·사진 경로·서버 응답 원문이 남는다
    row = next(r for r in rows if r['variant'] == 'long_black' and r['id'] == 'seed0/0004_grasp_to_tilt_start_judge')
    assert row['prompt'].endswith('PROMPT seed0 0004_grasp_to_tilt_start_judge') and row['images'] == ['black:' + lane['image']]
    assert row['resp'] == {'text': DARK, 'gen_s': 0.01} and row['sec'] > 0
    row = next(r for r in rows if r['variant'] == 'short2' and r['id'] == lane['id'])
    assert row['images'] == [lane['image_before'], lane['image']] and row['resp']['probs'] == row['resp']['scores']
    # 스텁은 사진이 검지 않으면 긴 프롬프트에 늘 성공이라고 답한다: 실패 경계를 통과시키고 바꾼 사진에도 답이 그대로다.
    # lanes 경계처럼 제 질문을 한쪽만 정한 경계는 답이 맞든 틀리든 채점에서 빠진다(unknown)
    same, swap, dark = s['long_same', None], s['long_swap', None], s['long_black', None]
    assert same['n'] == dark['n'] == len(own) and swap['n'] == n_swap and same['got'] == (len(own), len(own))
    assert (same['acc'], same['acc_all'], same['unknown']) == ((n_ok, n_ok + n_fail),) * 2 + (len(own) - n_ok - n_fail,)
    assert (same['yes_recall'], same['no_fail'], same['no_cross'], same['balanced']) == ((n_ok, n_ok), (0, n_fail), (0, 0), 0.5)
    assert (same['always_yes'], same['always_no']) == ((n_ok, n_ok + n_fail), (n_fail, n_ok + n_fail))
    assert (swap['no_cross'], swap['flip'], swap['yes_rate']) == ((0, n_swap), (0, n_swap), (n_swap, n_swap))
    assert dark['flip'] == dark['base_yes_flip'] == (len(own), len(own)) and dark['yes_rate'] == (0, len(own)) and dark['acc'] == (0, 0)
    # short: 항목당 한 번, yes/no 점수만. 스텁의 답(empty 를 물으면 yes)으로 기대값을 따로 센다
    short_sent = [q for q in seen if 'choices' in q]
    assert all(q['choices'] == ['yes', 'no'] and q['max_new_tokens'] == 0 for q in short_sent)
    n_two = sum(i['image_before'] is not None for i in items)
    assert collections.Counter(len(q['images_b64']) for q in short_sent) == {1: 2 * len(items), 2: n_two}
    assert all(q['sizes'][-1] == (8, 6) for q in seen) and sum(q['dark'] for q in seen) == len(own) + len(items)
    at_lane = [i for i in items if i['image'] == lane['image']]  # 한 경계의 항목들로 문구 틀과 사진 순서(직전, 지금)를 본다
    pair = [run_bench.b64(lane['image_before']), run_bench.b64(lane['image'])]
    assert sorted(q['prompt'] for q in short_sent if q['images_b64'] == pair) == sorted(
        run_bench.SHORT2.format(question=i['question']) for i in at_lane)
    assert sorted(q['prompt'] for q in short_sent if q['images_b64'] == pair[1:]) == sorted(
        run_bench.SHORT1.format(question=i['question']) for i in at_lane)

    def guess(i):
        return 'yes' if 'empty' in i['question'] else 'no'

    def hits(kind, of=items):
        of_kind = [i for i in of if i['kind'] in kind]
        return sum(guess(i) == i['truth'] for i in of_kind), len(of_kind)

    one = s['short1', 0.5]
    assert one == s['short1', 0.0]  # 스텁의 choice_mass 는 1.0 이라 문턱이 아무것도 빼지 않는다
    assert one['acc'] == one['acc_all'] == hits(('pos', 'neg_cross', 'neg_fail')) and one['got'] == (len(items), len(items))
    assert (one['yes_recall'], one['no_cross'], one['no_fail']) == (hits(('pos',)), hits(('neg_cross',)), hits(('neg_fail',)))
    assert one['always_yes'] == (sum(i['truth'] == 'yes' for i in items), len(items))
    two = [i for i in items if i['image_before'] is not None]
    assert s['short2', 0.5]['n'] == n_two < one['n'] == len(items)
    assert s['short2', 0.5]['pair']['short1']['acc'] == hits(('pos', 'neg_cross', 'neg_fail'), two) != one['acc']
    n_yes = sum(guess(i) == 'yes' for i in items)
    dark = s['short_black', 0.5]
    assert n_yes and dark['base_yes_flip'] == (n_yes, n_yes)  # 검은 사진에서는 yes 가 모두 no 로 바뀐다
    assert dark['flip'] == (n_yes, len(items)) and dark['yes_rate'] == (0, len(items)) and dark['acc'] == (0, 0)
    text = run_bench.summary(items, bnd, {'경계: 판정 사진·기록 없음': 1}, [meta, *rows], 'items.jsonl')
    assert f'시드 3개 · 경계 {len(own)}개 · 항목 {len(items)}개' in text and repr(run_bench.SHORT2) in text
    assert f'yes {n_ok} · no {n_fail} · 못 정함 {len(own) - n_ok - n_fail}' in text and '- 1건 경계: 판정 사진·기록 없음' in text
    check = next(line for line in text.splitlines() if 'long_same 자체 점검' in line)
    assert check.count(f'{len(own) - 1}/{len(own)} (') == 2  # 글자 일치와 choice 일치
    assert all(f'| {v} |' in text for v in run_bench.VARIANTS) and '| short1 | 0 |' in text and '| short1 | 0.5 |' in text
    assert f'짝 비교(문턱 0.5, short2 가 불린 항목 {n_two}개만)' in text
    assert 'short1 1.00/1.00 · short2 1.00/1.00 · short_black 1.00/1.00' in text  # 선택지 질량은 변형별로


def test_broken_answers_are_counted_apart(tmp_path, server):
    url, _, mode = server
    logs = make_logs(tmp_path / 'logs')
    items, _, brows = build_set.build(str(logs))
    bnd, few = {b['id']: b for b in brows}, [i for i in items if i['id'].startswith('seed0/0007_')]
    n = len(few)
    # long 의 답이 판정 JSON 이 아니면 no 가 아니라 무효다. 분모에서 빠지고 보수 정답률에서만 오답이다
    mode['long'] = 'garbage'
    rows = run_bench.run(few, bnd, url, tmp_path / 'garbage', ['long_same'])
    s = run_bench.stats(rows, few, bnd)['long_same', None]
    assert [(r['answer'], r['choice']) for r in rows] == [('invalid', 'E')]
    assert (s['invalid'], s['got'], s['acc'], s['acc_all'], s['no_fail']) == (1, (0, 1), (0, 0), (0, 1), (0, 0))
    # short: 동점은 어느 문턱에서나 보류, 질량 0.2 는 문턱 0.5 에서만 보류(문턱 0 줄에서는 채점), 서버가 low_mass 로 표시한 답도 보류
    for kind, at0, at5 in (('tie', 0, 0), ('lowmass', n, 0), ('flag', 0, 0)):
        mode['short'] = kind
        rows = run_bench.run(few, bnd, url, tmp_path / kind, ['short1'])
        s = run_bench.stats(rows, few, bnd)
        assert {r['answer'] for r in rows} == {'abstain'} and not any('error' in r for r in rows), kind
        assert (s['short1', 0.0]['got'], s['short1', 0.5]['got']) == ((at0, n), (at5, n)), kind
        assert s['short1', 0.5]['abstain'] == n and s['short1', 0.5]['acc_all'][1] == n and s['short1', 0.5]['acc'] == (0, 0)
    assert rows[0]['resp']['low_mass'] is True and rows[0]['p_yes'] is None
    # 점수 없는 응답·dict 아닌 응답은 오류로 남는다
    for kind in ('noscores', 'list'):
        mode['short'] = kind
        rows = run_bench.run(few, bnd, url, tmp_path / kind, ['short1'])
        assert len(rows) == n and all('error' in r and 'answer' not in r for r in rows), kind
    mode.clear()
    # 사진이 없거나 판정 기록이 깨져도 run 은 끝까지 간다. 그 행만 오류다
    lost = few[0]
    Path(lost['image']).unlink()
    Path(lost['decision_json']).write_text('{"prompt": ', encoding='utf-8')
    other = next(i for i in items if i['id'].startswith('seed0/0003_'))
    rows = run_bench.run([*few, other], bnd, url, tmp_path / 'lost', ['long_same', 'short1', 'short_black'])
    bad = [r for r in rows if 'error' in r]
    assert len(bad) == 1 + 2 * n and {r['id'] for r in rows if 'error' not in r} == {other['id'], run_bench.bid(other)}
    assert 'JSONDecodeError' in bad[0]['error'] and 'FileNotFoundError' in bad[1]['error']
    s = run_bench.stats(rows, [*few, other], bnd)
    assert (s['short1', 0.5]['error'], s['short1', 0.5]['got']) == (n, (1, n + 1))


def test_cli_runs_chosen_variants_and_records_server_errors(tmp_path, server):
    logs, out = make_logs(tmp_path / 'logs'), tmp_path / 'set' / 'items.jsonl'
    build_set.main([str(logs), '--out', str(out)])
    assert run_bench.main([str(out), '--url', server[0], '--out', str(tmp_path / 'ok'), '--variants', 'short1']) == 0
    table = (tmp_path / 'ok' / 'summary.md').read_text(encoding='utf-8')
    assert '| short1 |' in table and '| long_same |' not in table and '## 빠진 항목' in table
    with pytest.raises(FileExistsError):  # 잰 결과는 덮어쓰지 않는다
        run_bench.main([str(out), '--url', server[0], '--out', str(tmp_path / 'ok'), '--variants', 'short1'])
    with pytest.raises(SystemExit):  # 주소도 --rescore 도 없으면 부르지 않는다
        run_bench.main([str(out), '--out', str(tmp_path / 'none')])
    with socket.socket() as sock:  # 닫힌 포트: 서버가 죽어 있어도 예외로 끝나지 않고 오류로 적힌다
        sock.bind(('127.0.0.1', 0))
        dead = f'http://127.0.0.1:{sock.getsockname()[1]}'
    items, bnd, _ = run_bench.load_set(out)
    rows = run_bench.run(items[:3], bnd, dead, tmp_path / 'dead', ['short1'])
    assert len(rows) == 3 and all('error' in r and 'answer' not in r for r in rows)
    s = run_bench.stats(rows, items[:3], bnd)['short1', 0.5]
    assert (s['error'], s['got'], s['acc'], s['acc_all'], s['sec_med']) == (3, (0, 3), (0, 0), (0, 3), None)


def test_rescore_matches_run_and_reads_old_results(tmp_path, server):
    logs, out = make_logs(tmp_path / 'logs'), tmp_path / 'set' / 'items.jsonl'
    build_set.main([str(logs), '--out', str(out)])
    assert run_bench.main([str(out), '--url', server[0], '--out', str(tmp_path / 'run')]) == 0
    first, n_sent = (tmp_path / 'run' / 'summary.md').read_text(encoding='utf-8'), len(server[1])
    # 서버를 부르지 않고 같은 글을 다시 쓴다
    assert run_bench.main([str(out), '--rescore', str(tmp_path / 'run' / 'results.jsonl'), '--out', str(tmp_path / 'again')]) == 0
    assert (tmp_path / 'again' / 'summary.md').read_text(encoding='utf-8') == first and len(server[1]) == n_sent
    assert not (tmp_path / 'again' / 'results.jsonl').exists()
    # 옛 형식(49742ef): meta 행이 없고 필드가 적다. long 의 파싱 실패도 answer=no, 참값은 옛 규칙으로 적혀 있다
    keep = ('variant', 'id', 'seed', 'node', 'question_id', 'kind', 'text', 'choice', 'same_raw', 'same_choice',
            'p_yes', 'p_no', 'choice_mass', 'sec', 'gen_s')
    rows = read(tmp_path / 'run' / 'results.jsonl')[1:]
    old = [{**{k: r[k] for k in keep if k in r}, 'truth': 'yes', 'answer': 'yes' if r['answer'] == 'yes' else 'no'} for r in rows]
    (tmp_path / 'old').mkdir()
    (tmp_path / 'old' / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in old), encoding='utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert run_bench.main([str(out), '--rescore', str(tmp_path / 'old' / 'results.jsonl')]) == 0
    again = (tmp_path / 'old' / 'summary.md').read_text(encoding='utf-8')
    assert again.split('## 응답 상태')[1] == first.split('## 응답 상태')[1] and 'meta 행이 없다(옛 형식)' in again
    # results 에만 있는 id, items 에만 있는 id 는 수로 적는다. 쓰다 끊긴 마지막 줄은 경고만 한다
    cut = [r for r in old if r['variant'] == 'short1']
    cut[0] = {**cut[0], 'id': 'seed9/0001_x/holders_empty'}
    (tmp_path / 'old' / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in cut) + '{"variant": "sho', encoding='utf-8')
    with pytest.warns(UserWarning, match='잘린 마지막 줄'):
        items, bnd, _ = run_bench.load_set(out)
        s = run_bench.stats(build_set.jl(tmp_path / 'old' / 'results.jsonl'), items, bnd)['short1', 0.5]
    assert (s['missing'], s['extra'], s['got']) == (1, 1, (len(items) - 1, len(items)))
    # boundaries.jsonl 이 없는 옛 items 는 정한 제 질문만으로 long 참값을 만든다(lanes 경계가 yes 로 돌아간다)
    (out.parent / 'boundaries.jsonl').unlink()
    assert run_bench.load_set(out)[1]['seed0/0004_grasp_to_tilt_start_judge'] == {
        'id': 'seed0/0004_grasp_to_tilt_start_judge', 'undecided': None, 'long_truth': 'yes'}


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
