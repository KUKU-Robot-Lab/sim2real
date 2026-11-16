"""VLM 판정 시험대 항목 만들기: 저장된 경계 사진 + 러너·sim 참값 → items.jsonl.

실행(저장소 루트): python3 -m vlm_supervisor.bench.build_set <logs_root> --out items.jsonl
<logs_root>/seed*/ 에서 읽는 것: sup/*/decisions.jsonl(경계와 그 사진·판정 기록 경로), env0000_episode.jsonl(노드 exit 행의
sim 신호), result.jsonl(놓은 컵의 끝 상태). 참값은 exit 행을 순서대로 반영한 world 에서만 정한다. VLM 답은 읽지 않는다.
항목 종류: pos(제 경계, 참값 yes) · neg_fail(제 경계, 노드가 실제로 실패해 no) · neg_cross(다른 경계, world 상 분명히 no).
질문 문구와 참값 근거는 questions.yaml.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

import yaml

from ..planner import load_cards

QUESTIONS = Path(__file__).with_name('questions.yaml')
SIDES = ('left', 'right')
#: 놓기 3단계 등급(oneshot place_judge: 구멍 안·기울기 10° 미만 / 구멍 안·기울기 10° 이상 / 구멍 밖) → world 의 컵 값
GRADE = {'great': 'holder', 'partial': 'leaning', 'fail': 'out'}


def jl(path) -> list:
    """jsonl 을 읽는다. 파일이 없으면 빈 목록(끝나지 않은 시드에는 result.jsonl 이 없다)."""
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def node_table() -> dict:
    """skill_cards 의 노드 예시: 노드 id → (스킬 id, 노드 인자)."""
    return {node['id']: (card['id'], node) for card in load_cards()['스킬']
            for node in (card.get('실행_방법') or {}).get('노드_예시') or []}


def apply_exit(w: dict, why: dict, ex: dict, kind, card: dict, end: dict) -> list:
    """exit 행 하나를 world 에 반영하고 그 노드가 쓴 손을 돌려준다.

    컵 값: table · left/right(그 손 안) · holder · leaning · out(놓기 fail, 떨어뜨림) · None(모름). 손 값: 쥔 컵 이름 · None.
    """
    node, role, result = ex.get('state_id'), ex.get('policy_name') or '', ex.get('result')
    hands = [h for h in SIDES if role.endswith('_' + h[0])] or list(SIDES)  # 손은 정책 이름 끝 _r/_l(supervisor 와 같다)
    sig = (ex.get('signals') or {}).get(role) or {}
    if kind == 'grasp':
        hand, obj, dropped = hands[0], card['target_object'], sig.get('dropped')
        held = result == 'COMPLETED' and dropped is False
        w[hand], w[obj] = (obj, hand) if held else (None, 'out' if dropped is True else None)
        why[obj] = f'exit[{node}].result={result}, signals.{role}.dropped={dropped}'
    elif kind == 'place':
        hand, obj, empty = hands[0], card['source_object'], sig.get('hand_empty')
        grade = (sig.get('scripted_judge') or {}).get('grade')
        why[obj] = f'exit[{node}].signals.{role}.scripted_judge.grade={grade}, hand_empty={empty}'
        last = end.get(card.get('target_holder'))
        if last not in (None, grade):  # 놓은 뒤 컵이 움직였다. 언제 바뀌었는지 몰라 이 컵은 못 정함으로 둔다
            grade, why[obj] = None, why[obj] + f', result.jsonl objects 끝 grade={last}'
        # 손을 못 뗀 놓기(hand_empty≠true)는 컵이 손 안인지 홀더 안인지 모른다. 구멍 밖(fail)만 확실하다
        w[hand], w[obj] = None, GRADE.get(grade) if empty is True or grade == 'fail' else None
    elif result != 'COMPLETED':  # 끝까지 못 간 궤적 뒤에는 쥔 컵이 어디 있는지 모른다
        for hand in hands:
            held = w[hand]
            if held:
                w[hand] = w[held] = None
                why[held] = f'exit[{node}].result={result}'
    return hands


def home_questions(kind, node, card: dict, hands: list, w: dict, Q: dict, skipped) -> list:
    """이 노드 종류의 제 질문을 (질문 id, 손, 컵) 으로 푼다. 컵은 노드 인자, 없으면(to_tilt_start) 그 손이 쥔 컵."""
    out = []
    for q in Q['questions']:
        if kind not in q['at']:
            continue
        for hand in hands if '{hand}' in q['text'] else [None]:
            obj = None
            if '{obj}' in q['text']:
                obj = card.get('target_object') or card.get('source_object') or (w.get(hand) if w.get(hand) in Q['objects'] else None)
                if obj is None:
                    skipped[f"{q['id']}: {node} 에서 그 손이 쥔 컵을 모름"] += 1
                    continue
            out.append((q['id'], hand, obj))
    return out


def truth_of(inst, b: dict, home: bool):
    """world 에서 질문의 참값. (yes | no | None(못 정함), 근거). 본 값이 하나라도 no 쪽이면 no, 모두 yes 쪽이면 yes."""
    qid, hand, obj = inst
    w = b['w']
    cups = [k for k in w if k not in SIDES]
    away = {'table', 'out', *SIDES}  # 홀더 안이 아닌 컵 값
    keys, yes, no = {
        'cup_in_hand': ([obj], {hand}, (away | {'holder', 'leaning'}) - {hand}),
        'cup_lifted': ([obj], set(SIDES), {'table'}),
        'cup_in_holder': ([obj], {'holder'}, away),
        'all_in_holders': (cups, {'holder'}, away),
        'holders_empty': (cups, {'table', *SIDES}, {'holder', 'leaning'}),
    }[qid]
    if not home:
        no = no - {'out'}  # 실패 뒤 상태(out)는 그 노드의 제 경계에서만 반례(neg_fail)로 쓴다. 쉬운 반례가 아니다
    vals = [w[k] for k in keys]
    truth = 'no' if any(v in no for v in vals) else 'yes' if all(v in yes for v in vals) else None
    return truth, '; '.join(f"{k}={w[k]} ← {b['why'][k]}" for k in keys)


def swap_photo(n: int, bounds: list):
    """long_swap 용 사진: 이 경계에서 yes 인 제 질문이 가장 많이 no 로 뒤집히는 같은 시드 경계(동률이면 먼 경계). 없으면 None."""
    met = [i for i in bounds[n]['home'] if truth_of(i, bounds[n], True)[0] == 'yes']
    broken, _, m = max(((sum(truth_of(i, b, False)[0] == 'no' for i in met), abs(m - n), m)
                        for m, b in enumerate(bounds) if m != n), default=(0, 0, n))
    return bounds[m]['files']['jpg'] if broken else None


def seed_items(seed_dir: str, Q: dict, nodes: dict, skipped) -> list:
    seed = os.path.basename(seed_dir)
    exits = [o for o in jl(os.path.join(seed_dir, 'env0000_episode.jsonl')) if o.get('event') == 'exit']
    res = (jl(os.path.join(seed_dir, 'result.jsonl')) or [{}])[0]
    end = {f"HOLDER_{o.get('holder')}": o.get('grade') for o in (res.get('objects') or {}).values()}  # 홀더 → 끝 등급
    w = {**dict.fromkeys(Q['objects'], 'table'), **dict.fromkeys(SIDES)}
    why = dict.fromkeys(Q['objects'], '시작 상태(grasp·place exit 전)')
    bounds, p = [], -1  # p = 앞 경계까지 반영한 마지막 exit 행
    for dec in sorted(glob.glob(os.path.join(seed_dir, 'sup', '*', 'decisions.jsonl'))):
        for row in jl(dec):
            if row.get('event') not in ('decision', 'final_judge'):
                continue
            files = {f.rsplit('.', 1)[-1]: os.path.join(os.path.dirname(dec), f)
                     for f in (row.get('decision') or {}).get('files') or []}
            node = row.get('node')  # 첫 READY 는 끝난 노드가 없다(None)
            j = p if node is None else next((i for i in range(p + 1, len(exits)) if exits[i].get('state_id') == node), None)
            if j is None or set(files) != {'jpg', 'json'}:
                skipped['경계: 노드의 exit 행 없음' if j is None else '경계: 판정 사진·기록 없음'] += 1
                # p 를 두면 이 구간의 노드는 다음 경계가 이어받는다.
                # shortcut: 이어받은 질문도 그 경계 시점 world 로 답한다. 같은 컵의 잡기와 놓기가 한 구간에 묶이면 잡기 질문이
                # neg_fail 로 적힌다(vlm_full_1116 에는 사진 없는 경계가 없다). 생기면 exit 시점 참값과 다른 질문을 뺀다.
                continue
            home = []
            for ex in exits[p + 1:j + 1]:  # 승인 없이 이어진 노드(place_blue → home_r_after_blue)와 lanes 는 한 경계다
                kind, card = nodes.get(ex.get('state_id'), (None, {}))
                hands = apply_exit(w, why, ex, kind, card, end)
                home += home_questions(kind, ex.get('state_id'), card, hands, w, Q, skipped)
            if row['event'] == 'final_judge':
                home += home_questions('terminal', 'final', {}, list(SIDES), w, Q, skipped)
            p = j
            bounds.append({'stem': Path(files['json']).stem, 'node': node, 'files': files,
                           'w': dict(w), 'why': dict(why), 'home': home})
    asked = list(dict.fromkeys(i for b in bounds for i in b['home']))  # 이 시드에서 제 경계가 있는 질문
    text = {q['id']: q['text'] for q in Q['questions']}
    items = []
    for n, b in enumerate(bounds):
        swap = swap_photo(n, bounds)
        for inst in asked:
            qid, hand, obj = inst
            own = inst in b['home']  # 이 경계에서 끝난 노드의 제 질문인가
            truth, src = truth_of(inst, b, own)
            if truth is None and own:
                skipped[f'{qid}: {src}'] += 1
            if truth is None or (truth == 'yes' and not own):  # 다른 경계의 yes 는 만들지 않는다
                continue
            items.append({
                'id': '/'.join([seed, b['stem'], ':'.join(a for a in inst if a)]), 'seed': seed, 'node': b['node'],
                'image': b['files']['jpg'], 'image_before': bounds[n - 1]['files']['jpg'] if n else None, 'image_swap': swap,
                'question_id': qid, 'question': text[qid].format(hand=hand, obj=Q['objects'][obj] if obj else None),
                'truth': truth, 'kind': 'pos' if truth == 'yes' else 'neg_fail' if own else 'neg_cross',
                'truth_src': src, 'decision_json': b['files']['json']})
    return items


def build(logs_root: str):
    """(항목 목록, 못 정해 뺀 건수 Counter)."""
    Q = yaml.safe_load(QUESTIONS.read_text(encoding='utf-8'))
    nodes, skipped, items = node_table(), collections.Counter(), []
    for seed_dir in sorted(glob.glob(os.path.join(os.path.abspath(logs_root), 'seed*'))):
        items += seed_items(seed_dir, Q, nodes, skipped)
    return items, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('logs_root')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    items, skipped = build(args.logs_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(''.join(json.dumps(i, ensure_ascii=False) + '\n' for i in items), encoding='utf-8')
    kinds = collections.Counter((i['question_id'], i['kind']) for i in items)
    total = collections.Counter(i['kind'] for i in items)
    print(f"시드 {len({i['seed'] for i in items})}개 · 경계 {len({i['decision_json'] for i in items})}개 · 항목 {len(items)}개 "
          f"(pos {total['pos']} · neg_cross {total['neg_cross']} · neg_fail {total['neg_fail']}) → {args.out}")
    for qid in dict.fromkeys(i['question_id'] for i in items):
        print(f"  {qid}: pos {kinds[qid, 'pos']} · neg_cross {kinds[qid, 'neg_cross']} · neg_fail {kinds[qid, 'neg_fail']}")
    print(f'참값을 못 정해 뺀 것 {sum(skipped.values())}건')
    for why, n in skipped.most_common():
        print(f'  {n}건 {why}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
