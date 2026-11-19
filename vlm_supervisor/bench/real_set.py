"""실기 시험대 항목 만들기: 경계 사진(real_frames 의 bounds.jsonl) + 사람 라벨 → sim 과 같은 형식의 items.jsonl.

실행(저장소 루트): python3 -m vlm_supervisor.bench.real_set <frames 폴더> --labels <라벨 jsonl> --out <새 폴더>/items.jsonl
옆에 boundaries.jsonl(행마다 그 경계의 노드 id 목록 nodes 를 더한다)·skipped.json 을 쓴다. 셋 중 하나라도 있으면 멈춘다.
라벨 한 줄: {"id": "<seed>/<kk>_<node>", "w": {컵: table|left|right|holder|leaning|out|null}}. 다른 키는 읽지 않는다.
world 는 라벨(사람이 사진으로 본 값)이고, 제 질문은 그 경계 묶음에서 끝난 노드들로 정한다. 항목 만들기는 build_set.bound_items 와 같다.
실기에는 판정 기록(decision json)이 없어 decision_json 은 사진 옆의 없는 .json 경로다. long 변형은 그 행이 오류로 남는다.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import yaml

from .build_set import QUESTIONS, SIDES, bound_items, check_questions, dump, home_questions, hands_of, jl, node_table

VALUES = {'table', 'left', 'right', 'holder', 'leaning', 'out', None}


def load_labels(path, ids: set, cups) -> dict:
    """라벨 id → 컵 값 dict. id 가 bounds 에 없거나, 컵 이름·값이 틀렸거나, 두 컵이 같은 손이면 ValueError."""
    out = {}
    for row in jl(path):
        lid, w = row.get('id'), row.get('w') or {}
        if lid not in ids:
            raise ValueError(f'{path}: 라벨 id {lid!r} 가 bounds.jsonl 에 없다')
        if lid in out:
            raise ValueError(f'{path}: 라벨 id {lid!r} 가 두 번 있다')
        if set(w) != set(cups):
            raise ValueError(f'{path}: {lid} 의 컵 이름 {sorted(w)} 가 questions.yaml objects {sorted(cups)} 와 다르다')
        bad = {k: v for k, v in w.items() if v not in VALUES}
        if bad:
            raise ValueError(f'{path}: {lid} 의 컵 값 {bad} 은 {sorted(VALUES, key=str)} 밖이다')
        for h in SIDES:
            if list(w.values()).count(h) > 1:
                raise ValueError(f'{path}: {lid} 에서 두 컵이 {h} 손 안이다')
        out[lid] = w
    return out


def seed_bounds(rows: list, frames: Path, labels: dict, name: str, Q: dict, nodes: dict, skipped) -> list:
    """한 에피소드(k 순)의 bounds 행 → bound_items 가 읽는 경계 목록. 라벨이 없는 경계는 빼고 다음 경계의 직전 사진도 없앤다."""
    bounds, gap = [], False
    for r in rows:
        if r['id'] not in labels:
            skipped['경계: 라벨 없음'] += 1
            gap = True
            continue
        cups = labels[r['id']]
        w = {**{c: cups[c] for c in Q['objects']}, **{h: next((c for c, v in cups.items() if v == h), None) for h in SIDES}}
        home, n0 = [], sum(skipped.values())
        for nd in r['nodes']:
            kind, card = nodes.get(nd['id'], (None, {}))
            hands = hands_of(nd.get('policy') or '')
            if kind in ('grasp', 'place') and len(hands) != 1:  # apply_exit 과 같다: 손을 모르면 그 노드의 질문을 만들지 않는다
                skipped[f"노드: {nd['id']} 의 손이 하나로 정해지지 않음(정책 이름 {nd.get('policy') or ''!r})"] += 1
                continue
            home += home_questions(kind, nd['id'], card, hands, w, Q, skipped)
        if r.get('end'):
            home += home_questions('terminal', 'final', {}, list(SIDES), w, Q, skipped)
        jpg = (frames / r['jpg']).resolve()
        bounds.append({'stem': jpg.stem, 'node': r['node'], 'nodes': [nd['id'] for nd in r['nodes']],
                       'files': {'jpg': str(jpg), 'json': str(jpg.with_suffix('.json'))},  # 없는 파일: run_bench 의 경계 id(bid)용
                       'w': w, 'why': dict.fromkeys(Q['objects'], f'사람 라벨({name})'),
                       'home': list(dict.fromkeys(home)), 'unsure': set(), 'lost': sum(skipped.values()) - n0,
                       'before': None if gap or not r.get('before') else str((frames / r['before']).resolve())})
        gap = False
    return bounds


def build(frames, labels_path):
    """(항목 목록, 뺀 건수 Counter(이유별), 경계 행 목록)."""
    Q = yaml.safe_load(QUESTIONS.read_text(encoding='utf-8'))
    check_questions(Q)
    frames = Path(frames)
    rows = jl(frames / 'bounds.jsonl')
    labels = load_labels(labels_path, {r['id'] for r in rows}, Q['objects'])
    by_seed = collections.defaultdict(list)
    for r in rows:
        by_seed[r['seed']].append(r)
    nodes, skipped, items, brows = node_table(), collections.Counter(), [], []
    for seed, rs in by_seed.items():
        bounds = seed_bounds(sorted(rs, key=lambda r: r['k']), frames, labels, Path(labels_path).name, Q, nodes, skipped)
        got, rows_b = bound_items(seed, bounds, Q, skipped, cross_yes=True)  # 라벨은 사진을 본 값이라 tilt·home 의 "아직 쥐고 있나"도 묻는다
        items += got
        brows += [{**row, 'nodes': b['nodes']} for row, b in zip(rows_b, bounds)]
    return items, skipped, brows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('frames', help='real_frames 출력 폴더(bounds.jsonl 과 사진)')
    ap.add_argument('--labels', required=True, help='사람 라벨 jsonl')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    side = {'items': out, 'bounds': out.with_name('boundaries.jsonl'), 'skipped': out.with_name('skipped.json')}
    if any(p.exists() for p in side.values()):
        raise FileExistsError(f"이미 있다: {[str(p) for p in side.values() if p.exists()]}. --out 에 새 폴더를 준다")
    items, skipped, bounds = build(args.frames, args.labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    side['items'].write_text(dump(items), encoding='utf-8')
    side['bounds'].write_text(dump(bounds), encoding='utf-8')
    side['skipped'].write_text(json.dumps(dict(skipped.most_common()), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    kinds = collections.Counter((i['question_id'], i['kind']) for i in items)
    total = collections.Counter(i['kind'] for i in items)
    print(f"에피소드 {len({i['seed'] for i in items})}개 · 경계 {len(bounds)}개 · 항목 {len(items)}개 "
          f"(pos {total['pos']} · neg_cross {total['neg_cross']} · neg_fail {total['neg_fail']}) → {args.out}")
    for qid in dict.fromkeys(i['question_id'] for i in items):
        print(f"  {qid}: pos {kinds[qid, 'pos']} · neg_cross {kinds[qid, 'neg_cross']} · neg_fail {kinds[qid, 'neg_fail']}")
    long = collections.Counter(b['long_truth'] for b in bounds)
    print(f"long 참값(경계 {len(bounds)}개): yes {long['yes']} · no {long['no']} · 못 정함 {long[None]}")
    print(f'참값을 못 정해 뺀 것 {sum(skipped.values())}건')
    for why, n in skipped.most_common():
        print(f'  {n}건 {why}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
