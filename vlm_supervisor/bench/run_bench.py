"""VLM 판정 시험대 실행: items.jsonl 을 변형별로 VLM 서버에 물어 results.jsonl 과 summary.md 를 쓴다.

실행(저장소 루트): python3 -m vlm_supervisor.bench.run_bench items.jsonl --url http://127.0.0.1:8767 --out <dir> [--variants ...]
변형:
  long_same    저장된 판정 프롬프트 전문 + 같은 사진. 저장된 raw 와 같은 답이 나와야 한다(시험대 자체 점검).
  long_swap    같은 프롬프트 + 그 노드의 성공 조건이 깨진 같은 시드의 다른 경계 사진(items 의 image_swap).
  long_black   같은 프롬프트 + 같은 크기의 검은 사진.
  short1       사진 1장 + 질문 하나, yes/no 선택지 점수.
  short2       전후 사진 2장 + 같은 질문.
  short_black  검은 사진 + 같은 질문.
long 은 경계당 한 번(성공 = judge.parse_judgement 의 choice A), short 는 항목당 한 번 부른다.
서버 계약: POST /infer {"prompt", "images_b64": [...], "max_new_tokens", "choices"(선택)} → {"text", "scores", "choice_mass", "gen_s"}.
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import statistics
import time
from pathlib import Path

import requests
from PIL import Image

from ..judge import parse_judgement

# short 변형의 프롬프트 문구는 이 세 상수뿐이다. 바꿔 비교할 때 여기만 고친다(쓴 문구는 summary.md 에 찍힌다).
# 장면 설명은 경계 사진에 보이는 대로다: 파랑·분홍 컵은 병 모양이고 흰 홀더 셋이 탁자 안쪽에 있다.
SCENE = ('a two-arm robot at a table with a blue cup and a pink cup (both bottle-shaped), '
         'an orange shaker cup and three white cup holders')
SHORT1 = (f'This photograph shows {SCENE}. '
          'Answer the question about the photograph with yes or no.\nQuestion: {question}')
SHORT2 = (f'These two photographs show {SCENE}. '
          'The first was taken before the last robot action and the second after it. '
          'Answer the question about the second photograph with yes or no.\nQuestion: {question}')
CHOICES = ['yes', 'no']
# 기록 당시 backends.Local.ask 가 저장된 prompt 앞에 붙여 보낸 문구와 토큰 상한. 달라지면 long_same 이 raw 를 재현하지 못한다.
LONG_PREFIX = '(Only the CURRENT image is attached.)\n'
LONG_TOKENS = 400
TIMEOUT_S = 180
VARIANTS = ('long_same', 'long_swap', 'long_black', 'short1', 'short2', 'short_black')
BASE = {'long_swap': 'long_same', 'long_black': 'long_same', 'short_black': 'short1'}  # 사진 민감도의 기준 변형


def b64(path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def black(path) -> str:
    """path 의 사진과 같은 해상도의 검은 사진."""
    buf = io.BytesIO()
    Image.new('RGB', Image.open(path).size).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def infer(row: dict, url: str, prompt: str, images: list, tokens: int, choices=None):
    """서버를 한 번 부른다. 실패는 row['error'] 에 적고 None 을 돌려준다(한 건이 실패해도 나머지 측정은 남긴다)."""
    body = {'prompt': prompt, 'images_b64': images, 'max_new_tokens': tokens, **({'choices': choices} if choices else {})}
    t0 = time.perf_counter()
    try:
        r = requests.post(url.rstrip('/') + '/infer', json=body, timeout=TIMEOUT_S)
        if r.status_code != 200:
            raise ValueError(f'http {r.status_code} {r.text[:200]}')
        resp = r.json()
        if choices and not all(isinstance((resp.get('scores') or {}).get(c), (int, float)) for c in choices):
            raise ValueError(f'응답에 선택지 점수가 없다: {str(resp)[:200]}')
    except (requests.RequestException, ValueError) as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return None
    row.update(sec=round(time.perf_counter() - t0, 3), gen_s=resp.get('gen_s'))
    return resp


def ask_long(variant: str, its: list, url: str):
    """경계 하나에 저장된 판정 프롬프트 전문을 다시 보낸다. its = 그 경계의 항목들. 부를 사진이 없으면 None."""
    it = its[0]
    photo = it.get('image_swap') if variant == 'long_swap' else it['image']
    if photo is None:
        return None
    rec = json.loads(Path(it['decision_json']).read_text(encoding='utf-8'))
    own = [i['truth'] for i in its if i['kind'] != 'neg_cross']  # 이 경계에서 끝난 노드의 성공 조건(제 질문)
    # 참값: long_swap 은 성공 조건이 깨진 사진이라 no, long_black 은 없음, long_same 은 제 질문이 모두 yes 인가
    truth = {'long_swap': 'no', 'long_black': None}.get(variant, 'no' if 'no' in own else 'yes' if own else None)
    row = {'variant': variant, 'id': f"{it['seed']}/{Path(it['decision_json']).stem}", 'seed': it['seed'], 'node': it['node'],
           'truth': truth, 'kind': {'yes': 'pos', 'no': 'neg_cross' if variant == 'long_swap' else 'neg_fail'}.get(truth)}
    resp = infer(row, url, LONG_PREFIX + rec['prompt'], [black(photo) if variant == 'long_black' else b64(photo)], LONG_TOKENS)
    if resp is None:
        return row
    text = resp.get('text') or ''
    choice = parse_judgement(text)['choice']
    row.update(answer='yes' if choice == 'A' else 'no', choice=choice, text=text)
    if variant == 'long_same':
        raw = rec.get('raw') or ''
        row.update(same_raw=text == raw, same_choice=choice == parse_judgement(raw)['choice'])
    return row


def ask_short(variant: str, it: dict, url: str):
    """항목 하나를 짧은 질문으로 묻는다. short2 인데 직전 경계 사진이 없으면 None."""
    if variant == 'short2' and not it.get('image_before'):
        return None
    photo = black(it['image']) if variant == 'short_black' else b64(it['image'])
    images = [b64(it['image_before']), photo] if variant == 'short2' else [photo]
    row = {'variant': variant, 'id': it['id'], 'seed': it['seed'], 'node': it['node'], 'question_id': it['question_id'],
           'truth': it['truth'], 'kind': it['kind']}
    resp = infer(row, url, (SHORT2 if variant == 'short2' else SHORT1).format(question=it['question']), images, 0, CHOICES)
    if resp is not None:
        yes, no = (resp['scores'][c] for c in CHOICES)
        row.update(answer='yes' if yes > no else 'no', p_yes=yes, p_no=no, choice_mass=resp.get('choice_mass'))  # 동률은 no(보류)
    return row


def run(items: list, url: str, out: Path, variants=VARIANTS) -> list:
    """변형별로 서버를 부르고 results.jsonl 을 줄 단위로 쓴다(중간에 끊겨도 받은 답은 남는다).

    out 에 results.jsonl 이 이미 있으면 FileExistsError 로 멈춘다(잰 결과를 덮어쓰지 않는다. 실행마다 새 폴더를 준다).
    """
    out.mkdir(parents=True, exist_ok=True)
    bounds = collections.defaultdict(list)  # 판정 기록(경계) → 그 경계의 항목
    for it in items:
        bounds[it['decision_json']].append(it)
    rows = []
    with (out / 'results.jsonl').open('x', encoding='utf-8') as fh:
        for variant in variants:
            calls = ((ask_long, its) for its in bounds.values()) if variant.startswith('long') else ((ask_short, it) for it in items)
            for fn, arg in calls:
                row = fn(variant, arg, url)
                if row is not None:
                    rows.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                    fh.flush()
    return rows


def stats(rows: list) -> dict:
    """변형별 집계. 비율은 (분자, 분모). 검은 사진 변형은 참값이 없어 정답률을 내지 않는다."""
    answer = {(r['variant'], r['id']): r['answer'] for r in rows if 'answer' in r}
    out = {}
    for variant in dict.fromkeys(r['variant'] for r in rows):
        ok = [r for r in rows if r['variant'] == variant and 'answer' in r]
        scored = [r for r in ok if r['truth'] and 'black' not in variant]

        def said(kind, ans):  # 그 종류의 항목 가운데 ans 라고 답한 수
            of_kind = [r for r in scored if r['kind'] == kind]
            return sum(r['answer'] == ans for r in of_kind), len(of_kind)

        base = [(r['answer'], answer[BASE[variant], r['id']]) for r in ok if (BASE.get(variant), r['id']) in answer]
        secs = [r['sec'] for r in ok]
        out[variant] = {
            'n': len(ok), 'errors': sum(r['variant'] == variant and 'error' in r for r in rows),
            'acc': (sum(r['answer'] == r['truth'] for r in scored), len(scored)),
            'fp_cross': said('neg_cross', 'yes'), 'fp_fail': said('neg_fail', 'yes'), 'fh': said('pos', 'no'),
            'flip': (sum(a != b for a, b in base), len(base)),
            'sec_med': round(statistics.median(secs), 2) if secs else None, 'sec_max': max(secs, default=None)}
    return out


def frac(pair) -> str:
    k, n = pair
    return f'{k}/{n} ({100 * k / n:.0f} %)' if n else '-'


def summary(items: list, rows: list, items_path, url: str) -> str:
    kinds = collections.Counter(i['kind'] for i in items)
    same = [r for r in rows if r['variant'] == 'long_same' and 'answer' in r]
    lines = [
        '# VLM 판정 시험대 요약', '',
        f'- 항목 `{items_path}` · 서버 `{url}` · {time.strftime("%F %T")}',
        f"- 시드 {len({i['seed'] for i in items})}개 · 경계 {len({i['decision_json'] for i in items})}개 · 항목 {len(items)}개"
        f"(pos {kinds['pos']} · neg_cross {kinds['neg_cross']} · neg_fail {kinds['neg_fail']}). "
        '같은 시드·같은 사진에서 나온 항목은 서로 독립이 아니다. 표본 크기는 항목 수가 아니라 시드·경계 수로 읽는다.']
    if same:
        lines.append(f"- long_same 자체 점검(저장된 raw 와 비교): 글자까지 같음 {frac((sum(r['same_raw'] for r in same), len(same)))} · "
                     f"choice 같음 {frac((sum(r['same_choice'] for r in same), len(same)))}. "
                     '저장된 사진은 q95 JPEG 이라 기록 때 보낸 원본 프레임과 픽셀이 조금 다르다.')
    mass = collections.defaultdict(list)
    for r in rows:
        if isinstance(r.get('choice_mass'), (int, float)):
            mass[r['variant']].append(r['choice_mass'])
    if mass:  # 점수는 yes·no 끼리 정규화한 값이라, 이 몫이 작으면 답이 아니라 잡음이다
        lines.append('- short 선택지 질량(choice_mass = 첫 토큰 확률 가운데 yes·no 몫) 중앙값/최소: '
                     + ' · '.join(f'{v} {statistics.median(m):.2f}/{min(m):.2f}' for v, m in mass.items())
                     + '. 낮으면 모델의 첫 토큰이 yes/no 가 아니어서 그 답을 믿기 어렵다.')
    lines += ['', '| 변형 | n | 오류 | 정답률 | 거짓 통과 neg_cross | 거짓 통과 neg_fail | 거짓 보류 | 사진 민감도 | 지연 중앙값 s | 지연 최대 s |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for variant, s in stats(rows).items():
        lines.append(f"| {variant} | {s['n']} | {s['errors']} | {frac(s['acc'])} | {frac(s['fp_cross'])} | {frac(s['fp_fail'])} | "
                     f"{frac(s['fh'])} | {frac(s['flip'])} | {s['sec_med']} | {s['sec_max']} |")
    lines += [
        '', '읽는 법',
        '- 거짓 통과는 참값이 no 인데 yes 라고 답한 비율이다(long 은 choice A). 거짓 보류는 참값이 yes 인데 no 라고 답한 비율이다.',
        '- 정답률은 항목 종류 비율에 끌려간다(short 는 대부분 neg_cross). 종류별 칸을 같이 본다.',
        '- long_same 의 참값은 그 경계에서 끝난 노드의 제 질문(pos·neg_fail)이 모두 yes 인지다. 제 질문은 노드 성공 조건 가운데 '
        '참값을 정할 수 있는 것뿐이다(questions.yaml). 제 질문이 없는 경계(첫 READY, go_home_start, tilt)는 정답률에서 빠진다. '
        'blind 프롬프트는 마지막 노드 이름만 받으므로 승인 없이 이어진 place(place_blue·place_shaker)는 이름이 프롬프트에 없다.',
        '- long_swap 은 그 노드의 성공 조건이 깨진 다른 경계 사진을 주므로 참값이 no 다. 그런 사진이 같은 시드에 없는 경계는 부르지 않는다.',
        '- 사진 민감도는 같은 경계·항목에서 기준 변형과 답이 달라진 비율이다. long_swap·long_black 의 기준은 long_same, '
        'short_black 의 기준은 short1 이다. 검은 사진 변형은 참값이 없어 정답률을 내지 않는다.',
        '- short2 는 직전 경계 사진이 없는 첫 경계 항목을 부르지 않는다. 지연은 요청 한 번의 왕복 시간이다.',
        '', 'short 문구', f'- SHORT1: `{SHORT1!r}`', f'- SHORT2: `{SHORT2!r}`']
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('items')
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = ap.parse_args(argv)
    items = [json.loads(line) for line in Path(args.items).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = run(items, args.url, Path(args.out), args.variants)
    text = summary(items, rows, args.items, args.url)
    (Path(args.out) / 'summary.md').write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
