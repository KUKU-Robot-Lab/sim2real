"""real_set: 가짜 bounds.jsonl + 사람 라벨로 실기 항목(종류·건너뛰기·라벨 검사·마지막 장면)을 본다."""
import json

import pytest

from test_bench import make_logs
from vlm_supervisor.bench import build_set, real_set

S = 'E1'


def nd(node, policy=None):
    return {'id': node, 'type': 'policy', 'policy': policy, 'event': 'exit', 'result': 'COMPLETED'}


def bound(k, nodes, end=False):
    jpg = f"{S}/{k:02d}_{nodes[-1]['id']}.jpg"
    return {'id': jpg[:-4], 'seed': S, 'k': k, 'head': nodes[0]['id'], 'node': nodes[-1]['id'], 'nodes': nodes,
            'end': end, 'jpg': jpg, 'before': None if k == 0 else f'{S}/{k - 1:02d}_x.jpg'}


BOUNDS = [
    bound(0, [nd('scene')]),
    bound(1, [nd('grasp_blue', 'rh_aglt_r')]),
    bound(2, [nd('place_blue', 'rh_place_r'), nd('home_r_after_blue', 'go_home_r')]),  # 놓기 실패(컵이 구멍 밖)
    bound(3, [nd('grasp_pink', 'rh_aglt_r')]),  # 라벨 없음
    bound(4, [nd('go_home_end', 'go_home')], end=True),
]
TABLE = {'SRC_BLUE': 'table', 'SRC_PINK': 'table', 'SHAKER': 'table'}
LABELS = {f'{S}/00_scene': TABLE, f'{S}/01_grasp_blue': {**TABLE, 'SRC_BLUE': 'right'},
          f'{S}/02_home_r_after_blue': {**TABLE, 'SRC_BLUE': 'out'}, f'{S}/04_go_home_end': TABLE}


def write(tmp_path, labels=LABELS, bounds=BOUNDS):
    frames = tmp_path / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    (frames / 'bounds.jsonl').write_text(''.join(json.dumps(b) + '\n' for b in bounds), encoding='utf-8')
    lab = tmp_path / 'labels.jsonl'
    lab.write_text(''.join(json.dumps({'id': i, 'w': w, 'sure': True}) + '\n' for i, w in labels.items()), encoding='utf-8')
    return frames, lab


def test_kinds_skips_and_final_question(tmp_path):
    frames, lab = write(tmp_path)
    items, skipped, brows = real_set.build(frames, lab)
    kind = {i['id']: i['kind'] for i in items}
    assert kind[f'{S}/00_scene/holders_empty'] == 'pos'
    assert kind[f'{S}/01_grasp_blue/cup_in_hand:right:SRC_BLUE'] == 'pos'
    assert kind[f'{S}/02_home_r_after_blue/cup_in_holder:SRC_BLUE'] == 'neg_fail'  # 라벨 out = 놓기 실패
    assert kind[f'{S}/04_go_home_end/cup_in_hand:right:SRC_BLUE'] == 'neg_cross'
    assert kind[f'{S}/04_go_home_end/all_in_holders'] == 'neg_fail'  # end 경계의 마지막 장면 질문
    assert {k for i, k in kind.items() if 'all_in_holders' in i and '/04_' not in i} == {'neg_cross'}
    no_end = real_set.build(*write(tmp_path / 'no_end', bounds=[*BOUNDS[:4], {**BOUNDS[4], 'end': False}]))[0]
    assert not any(i['question_id'] == 'all_in_holders' for i in no_end)  # end 가 없으면 마지막 장면 질문도 없다
    assert skipped['경계: 라벨 없음'] == 1 and not any('/03_' in i for i in kind)
    it = {i['id']: i for i in items}
    one, four = it[f'{S}/01_grasp_blue/cup_in_hand:right:SRC_BLUE'], it[f'{S}/04_go_home_end/all_in_holders']
    assert one['image'] == str((frames / f'{S}/01_grasp_blue.jpg').resolve())
    assert one['image_before'] == str((frames / f'{S}/00_x.jpg').resolve())
    assert four['image_before'] is None  # 앞 경계에 라벨이 없다
    assert one['truth_src'].endswith('사람 라벨(labels.jsonl)')
    assert [b['id'] for b in brows] == [f'{S}/00_scene', f'{S}/01_grasp_blue', f'{S}/02_home_r_after_blue', f'{S}/04_go_home_end']
    assert brows[2]['nodes'] == ['place_blue', 'home_r_after_blue']


def test_same_fields_as_sim(tmp_path):
    sim_items, _, sim_bounds = build_set.build(str(make_logs(tmp_path / 'logs')))
    items, _, brows = real_set.build(*write(tmp_path))
    base = set.intersection(*(set(i) for i in sim_items))
    assert {k for i in items for k in i} <= {k for i in sim_items for k in i}
    assert all(base <= set(i) for i in items)
    assert {k for b in brows for k in b} == {k for b in sim_bounds for k in b} | {'nodes'}


@pytest.mark.parametrize('labels, msg', [
    ({**LABELS, f'{S}/99_typo': TABLE}, 'bounds.jsonl 에 없다'),
    ({**LABELS, f'{S}/00_scene': {'SRC_BLUE': 'table', 'SRC_PINK': 'table'}}, '컵 이름'),
    ({**LABELS, f'{S}/00_scene': {**TABLE, 'SHAKER': 'floor'}}, '밖이다'),
    ({**LABELS, f'{S}/00_scene': {**TABLE, 'SHAKER': 'left', 'SRC_PINK': 'left'}}, '두 컵이 left'),
])
def test_label_errors(tmp_path, labels, msg):
    with pytest.raises(ValueError, match=msg):
        real_set.build(*write(tmp_path, labels))


def test_cli_writes_three_files_and_refuses_overwrite(tmp_path, capsys):
    frames, lab = write(tmp_path)
    out = tmp_path / 'set' / 'items.jsonl'
    assert real_set.main([str(frames), '--labels', str(lab), '--out', str(out)]) == 0
    assert all((out.parent / f).exists() for f in ('items.jsonl', 'boundaries.jsonl', 'skipped.json'))
    assert '경계 4개' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        real_set.main([str(frames), '--labels', str(lab), '--out', str(out)])
