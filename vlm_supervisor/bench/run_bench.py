"""VLM 판정 시험대 실행: items.jsonl 을 변형별로 VLM 서버에 물어 results.jsonl 과 summary.md 를 쓴다.

실행(저장소 루트): python3 -m vlm_supervisor.bench.run_bench items.jsonl --url http://127.0.0.1:8767 --out <dir> [--variants ...]
  --questions id1 id2 ... 를 주면 그 질문 id 의 항목만 부르고 채점한다(meta 에 적혀 --rescore 가 그대로 쓴다).
  long 변형은 경계 단위라 --questions 로 호출 수가 줄지 않는다(그 질문의 항목이 있는 경계를 모두 부른다). short 만 잴 때는 --variants 를 같이 준다.
변형:
  long_same    저장된 판정 프롬프트 전문 + 같은 사진. 저장된 raw 와 같은 답이 나와야 한다(시험대 자체 점검).
  long_swap    같은 프롬프트 + 그 노드의 성공 조건이 깨진 같은 시드의 다른 경계 사진(items 의 image_swap).
  long_black   같은 프롬프트 + 같은 크기의 검은 사진.
  short1       사진 1장 + 질문 하나, yes/no 선택지 점수.
  short2       전후 사진 2장 + 같은 질문.
  short_black  검은 사진 + 같은 질문.
long 은 경계당 한 번(성공 = judge.parse_judgement 의 choice A), short 는 항목당 한 번 부른다.
서버 계약: POST /infer {"prompt", "images_b64": [...], "max_new_tokens", "choices"(선택)}
  → {"text", "scores", "probs", "choice_mass", "gen_s", "low_mass"(선택)}.
다시 채점(서버를 부르지 않는다): python3 -m vlm_supervisor.bench.run_bench items.jsonl --rescore <results.jsonl> [--out <dir>]
  items.jsonl 과 그 옆 boundaries.jsonl·skipped.json(build_set 이 같이 쓴다)으로 참값을 다시 붙여 summary.md 를 쓴다.
  results.jsonl 은 원자료(text, p_yes, p_no, choice_mass)만 읽으므로 meta 행과 prompt 가 없는 옛 형식도 된다.
results.jsonl: 첫 줄은 {"meta": ...}(items sha256, 문구 상수, url, 시각, 토큰 상한), 나머지는 호출 하나에 한 줄.
  여러 실행의 results.jsonl 을 이어 붙여(cat) --rescore 해도 된다. meta 행을 모두 읽어 질문은 합집합(--questions 없이 잰 실행이
  하나라도 있으면 전체)으로 채점하고, min_mass 가 서로 다르면 --min-mass 를 줘야 한다. url·items sha256 이 다르면 summary 맨 위에 적는다.
  answer = yes | no | invalid(long 의 답이 판정 JSON 이 아님) | abstain(short 의 선택지 질량이 문턱 미만이거나 동점).
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import statistics
import time
from pathlib import Path

import requests
import yaml
from PIL import Image

from ..judge import parse_judgement
from .build_set import QUESTIONS, dump, jl

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
MIN_MASS = 0.5  # short 답을 채점에 넣는 choice_mass 문턱(--min-mass). 표에는 문턱 0 줄도 같이 낸다
INVALID = 'invalid judgement JSON'  # parse_judgement 가 파싱 실패 때 주는 reason(choice 는 E)
BAD_INPUT = (OSError, ValueError, KeyError, TypeError)  # 사진 없음·깨진 decision json: 그 행의 error 로 남긴다
VARIANTS = ('long_same', 'long_swap', 'long_black', 'short1', 'short2', 'short_black')
FIXED = {'long_swap': 'no', 'long_black': None}  # 경계 참값과 무관하게 정해진 long 참값
BASE = {'long_swap': 'long_same', 'long_black': 'long_same', 'short_black': 'short1'}  # 사진 민감도의 기준 변형
# 채점 표·사진 민감도에 쓰는 집계 칸. 문구 변형(항목의 base)을 뺀 원 질문 항목으로만 낸다
HEAD = ('acc', 'acc_all', 'yes_recall', 'no_cross', 'no_fail', 'balanced', 'always_yes', 'always_no')


def b64(path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def black(path) -> str:
    """path 의 사진과 같은 해상도의 검은 사진."""
    buf = io.BytesIO()
    Image.new('RGB', Image.open(path).size).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def bid(it: dict) -> str:
    """항목 → 그 경계의 id(long 행의 id)."""
    return f"{it['seed']}/{Path(it['decision_json']).stem}"


def load_set(items_path) -> tuple:
    """(항목, 경계 id → 경계 행, 뺀 건수 dict | None). 경계·뺀 건수는 items.jsonl 옆 파일에서 읽는다.

    boundaries.jsonl 이 없는 옛 items 는 항목의 제 질문만으로 long 참값을 만든다(못 정한 제 질문은 모른다 = undecided None).
    """
    items, side = jl(items_path), Path(items_path).parent
    bnd = {b['id']: b for b in jl(side / 'boundaries.jsonl')}
    if not bnd:
        own = collections.defaultdict(list)
        for it in items:
            own[bid(it)] += [it['truth']] if it['kind'] != 'neg_cross' and not it.get('base') else []  # 문구 변형은 제 질문이 아니다
        bnd = {k: {'id': k, 'undecided': None, 'long_truth': 'no' if 'no' in v else 'yes' if v else None} for k, v in own.items()}
    skipped = side / 'skipped.json'
    return items, bnd, json.loads(skipped.read_text(encoding='utf-8')) if skipped.exists() else None


def metas_of(rows: list) -> list:
    """results 행 가운데 meta 행의 내용(이어 붙인 results 는 여럿이다)."""
    return [r['meta'] for r in rows if 'meta' in r]


def meta_questions(metas: list):
    """잴 때 고른 질문의 합집합. meta 가 없거나 --questions 없이 잰 실행이 하나라도 있으면 None(전체)."""
    if not metas or not all(m.get('questions') for m in metas):
        return None
    return list(dict.fromkeys(q for m in metas for q in m['questions']))


def expected(variant: str, items: list, bnd: dict) -> dict:
    """그 변형이 불러야 할 단위: id → (참값 yes | no | None, 종류). long 은 경계, short 는 항목.

    long_same 의 참값은 경계 행의 long_truth, long_swap 은 성공 조건이 깨진 사진이라 no, 검은 사진 변형은 없음.
    """
    if not variant.startswith('long'):
        return {it['id']: (None if variant == 'short_black' else it['truth'], it['kind']) for it in items
                if variant != 'short2' or it.get('image_before')}
    out = {}
    for it in items:
        if bid(it) in out or (variant == 'long_swap' and not it.get('image_swap')):
            continue
        truth = FIXED[variant] if variant in FIXED else bnd[bid(it)]['long_truth']
        out[bid(it)] = (truth, {'yes': 'pos', 'no': 'neg_cross' if variant == 'long_swap' else 'neg_fail'}.get(truth))
    return out


def answer_of(row: dict, min_mass: float) -> str:
    """결과 행 → yes | no | invalid | abstain | error. 원자료(text·p_yes·p_no·choice_mass)로 정하므로 옛 행도 같은 규칙이다."""
    if 'error' in row:
        return 'error'
    if row['variant'].startswith('long'):
        got = parse_judgement(row.get('text') or '')
        return 'invalid' if (got['choice'], got['reason']) == ('E', INVALID) else 'yes' if got['choice'] == 'A' else 'no'
    yes, no, mass = row.get('p_yes'), row.get('p_no'), row.get('choice_mass')
    if yes is None or no is None or not (yes > no or no > yes):  # 점수 없음(low_mass)·동점·NaN
        return 'abstain'
    if min_mass > 0 and not (isinstance(mass, (int, float)) and mass >= min_mass):  # 질량이 없거나 NaN 이어도 문턱 줄에서는 보류
        return 'abstain'
    return 'yes' if yes > no else 'no'


def infer(row: dict, url: str, prompt: str, images: list, tokens: int, choices=None):
    """서버를 한 번 부른다. 실패는 row['error'] 에 적고 None 을 돌려준다(한 건이 실패해도 나머지 측정은 남긴다)."""
    body = {'prompt': prompt, 'images_b64': images, 'max_new_tokens': tokens, **({'choices': choices} if choices else {})}
    row['prompt'] = prompt
    t0 = time.perf_counter()
    try:
        r = requests.post(url.rstrip('/') + '/infer', json=body, timeout=TIMEOUT_S)
        if r.status_code != 200:
            raise ValueError(f'http {r.status_code} {r.text[:200]}')
        resp = r.json()
        if not isinstance(resp, dict):
            raise ValueError(f'응답이 dict 가 아니다: {str(resp)[:200]}')
        scores = resp.get('scores')
        if choices and not resp.get('low_mass') and not (
                isinstance(scores, dict) and all(isinstance(scores.get(c), (int, float)) for c in choices)):
            raise ValueError(f'응답에 선택지 점수가 없다: {str(resp)[:200]}')
    except (requests.RequestException, ValueError) as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return None
    row.update(sec=round(time.perf_counter() - t0, 3), gen_s=resp.get('gen_s'), resp=resp)
    return resp


def ask_long(variant: str, its: list, url: str, exp: dict, min_mass: float):
    """경계 하나에 저장된 판정 프롬프트 전문을 다시 보낸다. its = 그 경계의 항목들. 부르지 않는 경계(exp 에 없음)면 None."""
    it = its[0]
    if bid(it) not in exp:
        return None
    photo = it['image_swap'] if variant == 'long_swap' else it['image']
    truth, kind = exp[bid(it)]
    row = {'variant': variant, 'id': bid(it), 'seed': it['seed'], 'node': it['node'], 'truth': truth, 'kind': kind,
           'images': [f'black:{photo}' if variant == 'long_black' else photo]}
    try:
        rec = json.loads(Path(it['decision_json']).read_text(encoding='utf-8'))
        prompt, image = LONG_PREFIX + rec['prompt'], black(photo) if variant == 'long_black' else b64(photo)
    except BAD_INPUT as exc:
        return {**row, 'error': f'{type(exc).__name__}: {exc}'}
    resp = infer(row, url, prompt, [image], LONG_TOKENS)
    if resp is None:
        return row
    text = resp.get('text')
    text = text if isinstance(text, str) else ''  # 판정 글이 없으면 무효로 센다
    row.update(text=text, choice=parse_judgement(text)['choice'])
    row['answer'] = answer_of(row, min_mass)
    if variant == 'long_same':
        raw = rec.get('raw') or ''
        row.update(same_raw=text == raw, same_choice=row['choice'] == parse_judgement(raw)['choice'])
    return row


def ask_short(variant: str, it: dict, url: str, exp: dict, min_mass: float):
    """항목 하나를 짧은 질문으로 묻는다. short2 인데 직전 경계 사진이 없으면(exp 에 없음) None."""
    if it['id'] not in exp:
        return None
    paths = [it['image_before'], it['image']] if variant == 'short2' else [it['image']]
    row = {'variant': variant, 'id': it['id'], 'seed': it['seed'], 'node': it['node'], 'question_id': it['question_id'],
           'truth': it['truth'], 'kind': it['kind'],
           'images': [f'black:{p}' if variant == 'short_black' else p for p in paths]}
    try:
        images = [black(p) if variant == 'short_black' else b64(p) for p in paths]
    except BAD_INPUT as exc:
        return {**row, 'error': f'{type(exc).__name__}: {exc}'}
    resp = infer(row, url, (SHORT2 if variant == 'short2' else SHORT1).format(question=it['question']), images, 0, CHOICES)
    if resp is not None:
        scores = resp.get('scores') or {}  # low_mass 응답은 scores 가 비어 있다 → p 없음 → abstain
        row.update(p_yes=scores.get('yes'), p_no=scores.get('no'), choice_mass=resp.get('choice_mass'))
        row['answer'] = answer_of(row, min_mass)
    return row


def run(items: list, bnd: dict, url: str, out: Path, variants=VARIANTS, min_mass: float = MIN_MASS, meta=None) -> list:
    """변형별로 서버를 부르고 results.jsonl 을 줄 단위로 쓴다(중간에 끊겨도 받은 답은 남는다). 돌려주는 것은 호출 행(meta 제외).

    out 에 results.jsonl 이 이미 있으면 FileExistsError 로 멈춘다(잰 결과를 덮어쓰지 않는다. 실행마다 새 폴더를 준다).
    """
    out.mkdir(parents=True, exist_ok=True)
    bounds = collections.defaultdict(list)  # 판정 기록(경계) → 그 경계의 항목
    for it in items:
        bounds[it['decision_json']].append(it)
    rows = []
    with (out / 'results.jsonl').open('x', encoding='utf-8') as fh:
        fh.write(dump([{'meta': {**(meta or {}), 'url': url, 'time': time.strftime('%F %T'), 'variants': list(variants),
                                 'min_mass': min_mass, 'long_tokens': LONG_TOKENS, 'long_prefix': LONG_PREFIX, 'scene': SCENE,
                                 'short1': SHORT1, 'short2': SHORT2, 'choices': CHOICES, 'timeout_s': TIMEOUT_S}}]))
        for variant in variants:
            exp = expected(variant, items, bnd)
            todo = ((ask_long(variant, its, url, exp, min_mass) for its in bounds.values()) if variant.startswith('long')
                    else (ask_short(variant, it, url, exp, min_mass) for it in items))
            for row in todo:
                if row is not None:
                    rows.append(row)
                    fh.write(dump([row]))
                    fh.flush()
    return rows


def score(units: list) -> dict:
    """(참값, 종류, 답) 목록 → 집계. 비율은 (분자, 분모). 답 = yes | no | invalid | abstain | error | missing(행 없음).

    acc·재현율은 yes/no 로 답한 것만, acc_all(보수 정답률)은 오류·무효·보류·행 없음을 오답으로 세어 참값 있는 것 전체가 분모다.
    """
    said = collections.Counter(a for _, _, a in units)
    known = [u for u in units if u[0]]
    hit = [u for u in known if u[2] in CHOICES]

    def recall(*kinds):
        of_kind = [u for u in hit if u[1] in kinds]
        return sum(t == a for t, _, a in of_kind), len(of_kind)

    pos, neg, n_pos = recall('pos'), recall('neg_cross', 'neg_fail'), sum(t == 'yes' for t, _, _ in known)
    return {'n': len(units), 'got': (said['yes'] + said['no'], len(units)), 'unknown': len(units) - len(known),
            **{k: said[k] for k in ('error', 'invalid', 'abstain', 'missing')},
            'acc': (sum(t == a for t, _, a in hit), len(hit)), 'acc_all': (sum(t == a for t, _, a in known), len(known)),
            'yes_recall': pos, 'no_cross': recall('neg_cross'), 'no_fail': recall('neg_fail'),
            'balanced': (pos[0] / pos[1] + neg[0] / neg[1]) / 2 if pos[1] and neg[1] else None,
            'always_yes': (n_pos, len(known)), 'always_no': (len(known) - n_pos, len(known))}


def stats(rows: list, items: list, bnd: dict, min_mass: float = MIN_MASS, variants=None) -> dict:
    """(변형, 문턱) → 집계. short 는 문턱 0(전부 채점)과 min_mass 두 줄, long 은 문턱이 없어 한 줄(문턱 None).

    참값은 행에 적힌 것을 쓰지 않고 items·bnd 로 다시 붙인다. extra = results 에만 있는 id 수, missing = items 에만 있는 id 수.
    """
    by: dict[str, dict] = {}
    dup: collections.Counter = collections.Counter()  # 같은 (변형, id) 가 다시 나온 행 수. 채점은 마지막 행으로 한다
    for r in rows:
        dup[r['variant']] += r['id'] in by.setdefault(r['variant'], {})
        by[r['variant']][r['id']] = r
    qof = {it['id']: it['question_id'] for it in items}
    varied = {it['id'] for it in items if it.get('base')}  # 문구 변형 항목. 머리 채점·사진 민감도에서 뺀다(long 의 경계 id 는 안 걸린다)
    out = {}
    names: list[str] = [*(variants or []), *by]
    for variant in dict.fromkeys(names):
        exp, got = expected(variant, items, bnd), by.get(variant, {})
        short = not variant.startswith('long')
        of_q = collections.defaultdict(list)  # 질문 id → 항목 id (short 만)
        twins = collections.defaultdict(list)  # (원 질문 id, 뒤집힌 질문 id) → [(원 항목 id, 뒤집힌 항목 id)]
        for it in items if short else []:
            if it['id'] in exp:
                of_q[it['question_id']].append(it['id'])
                if it.get('pair') in exp and 'black' not in variant:
                    twins[qof[it['pair']], it['question_id']].append((it['pair'], it['id']))
        for mass in sorted({0.0, min_mass}) if short else [None]:
            def ans(v, i):
                return answer_of(by[v][i], mass or 0.0) if i in by.get(v, ()) else 'missing'

            def scored(ids, v=variant):
                return score([(*exp[i], ans(v, i)) for i in ids])

            def rates(ids):
                a = [x for x in (ans(variant, i) for i in ids) if x not in ('error', 'missing')]
                return {'yes_rate': (a.count('yes'), len(a)), 'yes_of_choice': (a.count('yes'), a.count('yes') + a.count('no'))}

            plain = [i for i in exp if i not in varied]
            s, head = scored(exp), scored(plain)  # 응답 상태는 부른 것 전부, 채점 칸은 원 질문 항목만
            s.update({k: head[k] for k in HEAD}, n_head=len(plain))
            base = BASE.get(variant) if BASE.get(variant) in by else None  # 기준 변형이 results 에 없으면 견주지 않는다
            # 기준 변형도 yes/no 로 답했고 이 변형이 응답(오류 아님)한 단위만 견준다
            pairs = [(ans(variant, i), ans(base, i)) for i in plain if i in got and i in by[base]] if base else []
            pairs = [(a, b) for a, b in pairs if b in CHOICES and a != 'error']
            secs = [r['sec'] for r in got.values() if 'sec' in r]
            no_mass = sum('error' not in r and not (isinstance(r.get('choice_mass'), (int, float)) and r['choice_mass'] >= 0)
                          for i, r in got.items() if i in exp) if short else None  # NaN >= 0 은 거짓
            s.update(extra=len(set(got) - set(exp)), dup=dup[variant], no_mass=no_mass, base=base, **rates(plain),
                     flip=(sum(a != b for a, b in pairs), len(pairs)),
                     base_yes_flip=(sum(a != 'yes' for a, b in pairs if b == 'yes'), sum(b == 'yes' for _, b in pairs)),
                     sec_med=round(statistics.median(secs), 2) if secs else None, sec_max=max(secs, default=None))
            if variant == 'short2' and 'short1' in by:  # 같은 항목(short2 가 불린 것)끼리 short1 과 견준다
                called = [i for i in plain if i in got]
                s['pair'] = {'short1': scored(called, 'short1'), 'short2': scored(called)}
            if short:
                s['by_q'] = {q: {**scored(ids), **rates(ids)} for q, ids in of_q.items()}
                # 짝 일치: 원 질문과 뒤집힌 질문의 답이 서로 반대일 때만 원 질문의 답을 채택하고 나머지는 보류로 센다
                s['twin'] = {}
                for k, v in twins.items():
                    said = [(ans(variant, o), ans(variant, i)) for o, i in v]
                    s['twin'][k] = {**score([(*exp[o], a if {a, b} == set(CHOICES) else 'abstain') for (o, _), (a, b) in zip(v, said)]),
                                    'both_yes': said.count(('yes', 'yes')), 'both_no': said.count(('no', 'no'))}
            out[variant, mass] = s
    return out


def frac(pair) -> str:
    k, n = pair
    return f'{k}/{n} ({100 * k / n:.0f} %)' if n else '-'


def summary(items: list, bnd: dict, skipped, rows: list, items_path, min_mass=None, questions=None) -> str:
    """rows = results.jsonl 의 모든 행(meta 행 포함, 이어 붙인 results 면 여럿). 서버 실행 직후와 --rescore 가 같은 글을 낸다.

    min_mass 가 None 이면 meta 의 min_mass(잴 때 쓴 문턱), meta 에도 없으면 MIN_MASS. meta 끼리 다르면 ValueError(문턱을 줘야 한다).
    questions = items 를 거른 질문 id 목록(거르지 않았으면 None). 요약에 적기만 한다.
    """
    metas = metas_of(rows)

    def vals(key) -> list:
        return list(dict.fromkeys(m.get(key) for m in metas))

    def show(xs) -> str:
        return ' / '.join('기록 없음' if x is None else str(x) for x in xs) or '기록 없음'

    rows = [r for r in rows if 'variant' in r]
    asked, ran = min_mass, vals('min_mass')
    if asked is None and len(ran) > 1:
        raise ValueError(f'meta 행 {len(metas)}개의 min_mass 가 서로 다르다({show(ran)}). min_mass 를 준다')
    min_mass = asked if asked is not None else ran[0] if ran and ran[0] is not None else MIN_MASS
    kinds = collections.Counter(i['kind'] for i in items)
    called = expected('long_same', items, bnd)  # 지금 items 의 경계가 아닌 행은 자체 점검에 넣지 않는다
    same = [r for r in rows if 'same_raw' in r and r['id'] in called]
    here = [bnd[k] for k in dict.fromkeys(bid(i) for i in items)]  # 항목이 있는 경계
    long = collections.Counter(b['long_truth'] for b in here)
    sha = hashlib.sha256(Path(items_path).read_bytes()).hexdigest() if Path(items_path).is_file() else None
    old = any(b['undecided'] is None for b in here)  # boundaries.jsonl 이 없어 load_set 이 항목으로 만든 경계
    lines = ['# VLM 판정 시험대 요약', '']
    for key, what in (('url', '서버 주소'), ('items_sha256', 'items sha256')):  # 이어 붙인 실행끼리 조건이 다르면 맨 위에 적는다
        if len(vals(key)) > 1:
            lines.append(f'- 경고: results 의 meta 행 {len(metas)}개가 서로 다른 {what}로 잰 것이다({show(str(x)[:12] if key != "url" and x else x for x in vals(key))}). '
                         '한 표로 합쳐 읽을 때 조건 차이를 감안한다.')
    lines += [
        f"- 항목 `{items_path}`(sha256 {str(sha)[:12]}) · 서버 `{show(vals('url'))}` · 잰 시각 {show(vals('time'))}"
        + ('' if metas else ' · results.jsonl 에 meta 행이 없다(옛 형식). 보낸 문구와 토큰 상한은 그때 코드의 상수다.'),
        f"- 시드 {len({i['seed'] for i in items})}개 · 경계 {len(here)}개 · 항목 {len(items)}개"
        f"(pos {kinds['pos']} · neg_cross {kinds['neg_cross']} · neg_fail {kinds['neg_fail']}). "
        '같은 시드·같은 사진에서 나온 항목은 서로 독립이 아니다. 표본 크기는 항목 수가 아니라 시드·경계 수로 읽는다.',
        f"- long 참값(항목이 있는 경계 {len(here)}개): yes {long['yes']} · no {long['no']} · 못 정함 {long[None]}"
        + ('. boundaries.jsonl 이 없어 못 정한 제 질문 수를 모른다(옛 items). yes 는 정한 제 질문만 본 값이다.'
           if old else
           f"(제 질문 없음 {sum(not b['home'] for b in here)} · 제 질문 가운데 못 정한 것이 있음 "
           f"{sum(b['long_truth'] is None and b['home'] > 0 for b in here)})")]
    if not old:
        lines.append(f"- boundaries.jsonl 전체: 경계 {len(bnd)}개 · long 참값 못 정함 {sum(b['long_truth'] is None for b in bnd.values())}개"
                     '(항목이 없는 경계 포함).')
    if len(metas) > 1:
        lines.append(f'- meta 행 {len(metas)}개(results 를 이어 붙였다). 실행별 값: ' + ' / '.join(
            f"{n}) url {m.get('url')} · items {str(m.get('items_sha256'))[:12]} · min_mass {m.get('min_mass')} · "
            f"질문 {' '.join(m.get('questions') or ['전체'])} · 변형 {' '.join(m.get('variants') or ['기록 없음'])}"
            for n, m in enumerate(metas, 1)))
    ran_q = meta_questions(metas)
    if questions or ran_q:
        lines.append(f"- 채점에 쓴 질문: {' '.join(questions or ['전체'])}. 불러야 할 수는 그 항목만 센다."
                     + ('' if set(questions or ()) == set(ran_q or ()) else f" 잴 때(meta) 고른 질문은 {' '.join(ran_q or ['전체'])} 다."))
    odd = [x for x in ran if x not in (None, asked)]
    if asked is not None and odd:
        lines.append(f"- 주의: 잴 때 문턱(meta min_mass)은 {show(f'{x:g}' for x in odd)} 인데 --min-mass {asked:g} 로 다시 채점했다.")
    odd = [x for x in vals('items_sha256') if x not in (None, sha)]
    if odd:
        lines.append(f"- 주의: results 는 다른 items(sha256 {show(x[:12] for x in odd)})로 잰 것이다. 참값은 지금 items 로 다시 붙였다.")
    have = {i['question_id'] for i in items}
    for q in yaml.safe_load(QUESTIONS.read_text(encoding='utf-8'))['questions']:
        if q.get('pair_of') and (q['id'] in have) != (q['pair_of'] in have):  # 지금 questions.yaml 의 짝 기준
            one, other = (q['id'], q['pair_of']) if q['id'] in have else (q['pair_of'], q['id'])
            lines.append(f'- 짝 일치 표에 {q["pair_of"]} + {q["id"]} 줄이 없다: 채점 항목에 {one} 만 있고 짝 {other} 가 없다'
                         '(--questions 로 짝 한쪽만 고르면 빠진다).')
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
    st = stats(rows, items, bnd, min_mass, [v for m in metas for v in m.get('variants') or ()])
    n_var = sum(1 for i in items if i.get('base'))
    plain_note = (f'문구 변형 항목(questions.yaml 의 base 가 있는 질문, {n_var}개)은 이 표에 넣지 않는다. short 줄은 원 질문 항목 '
                  f'{len(items) - n_var}개만 센 것이고, 변형은 질문별 표와 짝 일치 표에서 본다.')

    def name(key):
        return f"| {key[0]} | {'-' if key[1] is None else f'{key[1]:g}'} |"

    def pct(x):
        return '-' if x is None else f'{100 * x:.0f} %'

    lines += ['', '## 응답 상태', '',
              '| 변형 | 문턱 | 불러야 할 수 | 답 받음(coverage) | 오류 | 무효 | 보류 | 행 없음(items 에만) | results 에만 | 중복 행 | 질량 없음·NaN | 참값 못 정함 | 지연 중앙값 s | 지연 최대 s |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines += [f"{name(k)} {s['n']} | {frac(s['got'])} | {s['error']} | {s['invalid']} | {s['abstain']} | {s['missing']} | {s['extra']} | {s['dup']} | {'-' if s['no_mass'] is None else s['no_mass']} | "
              f"{'-' if 'black' in k[0] else s['unknown']} | {s['sec_med']} | {s['sec_max']} |" for k, s in st.items()]
    lines += ['', '## 채점', '', plain_note, '',
              '| 변형 | 문턱 | 정답률(답한 것) | 보수 정답률 | yes 재현율 | no 재현율 neg_cross | no 재현율 neg_fail | 균형 정확도 | 항상 yes | 항상 no |',
              '|---|---|---|---|---|---|---|---|---|---|']
    lines += [f"{name(k)} {frac(s['acc'])} | {frac(s['acc_all'])} | {frac(s['yes_recall'])} | {frac(s['no_cross'])} | {frac(s['no_fail'])} | "
              f"{pct(s['balanced'])} | {frac(s['always_yes'])} | {frac(s['always_no'])} |" for k, s in st.items() if 'black' not in k[0]]
    for k, s in st.items():
        if 'pair' in s:
            lines += ['', f"짝 비교(문턱 {k[1]:g}, short2 가 불린 항목 {s['pair']['short2']['n']}개만): " + ' / '.join(
                f"{v} 정답률 {frac(p['acc'])} · 보수 {frac(p['acc_all'])} · 균형 {pct(p['balanced'])}" for v, p in s['pair'].items())
                + ' (원 질문 항목만)']
    if any(s.get('by_q') for s in st.values()):
        lines += ['', '## 질문별(short)', '',
                  '| 질문 id | 변형 | 문턱 | 항목 수 | yes 재현율 | no 재현율 neg_cross | no 재현율 neg_fail | 균형 정확도 | yes 답 비율 | yes/(yes+no) |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for q in dict.fromkeys(q for s in st.values() for q in s.get('by_q', ())):
            lines += [f"| {q} {name(k)} {s['by_q'][q]['n']} | {frac(s['by_q'][q]['yes_recall'])} | {frac(s['by_q'][q]['no_cross'])} | "
                      f"{frac(s['by_q'][q]['no_fail'])} | {pct(s['by_q'][q]['balanced'])} | {frac(s['by_q'][q]['yes_rate'])} | "
                      f"{frac(s['by_q'][q]['yes_of_choice'])} |" for k, s in st.items() if q in s.get('by_q', ())]
    if any(s.get('twin') for s in st.values()):
        lines += ['', '## 짝 일치(원 질문 + 뒤집힌 질문)', '',
                  '| 원 질문 | 뒤집힌 질문 | 변형 | 문턱 | 짝 수 | 보류율 | 둘 다 yes | 둘 다 no | yes 재현율 | no 재현율 neg_cross | no 재현율 neg_fail | 균형 정확도 | 보수 정답률 |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        lines += [f"| {a} | {b} {name(k)} {t['n']} | {frac((t['n'] - t['got'][0], t['n']))} | {t['both_yes']} | {t['both_no']} | {frac(t['yes_recall'])} | {frac(t['no_cross'])} | "
                  f"{frac(t['no_fail'])} | {pct(t['balanced'])} | {frac(t['acc_all'])} |"
                  for k, s in st.items() for (a, b), t in s.get('twin', {}).items()]
    lines += ['', '## 사진 민감도', '', plain_note, '',
              '| 변형 | 문턱 | 기준 변형 | yes 답 비율 | yes/(yes+no) | 기준과 답이 다름 | 기준에서 yes 였던 것 가운데 바뀜 |', '|---|---|---|---|---|---|---|']
    lines += [f"{name(k)} {s['base'] or BASE[k[0]] + '(results 에 없음)'} | {frac(s['yes_rate'])} | {frac(s['yes_of_choice'])} | "
              f"{frac(s['flip'])} | {frac(s['base_yes_flip'])} |" for k, s in st.items() if k[0] in BASE]
    lines += ['', '## 빠진 항목', '']
    if skipped is None:
        lines.append('- skipped.json 이 없다(옛 items). build_set 이 뺀 건수를 모른다.')
    else:
        lines.append(f'- build_set 이 참값을 못 정했거나 기록이 없어 뺀 것 {sum(skipped.values())}건(skipped.json). '
                     '"경계:" 는 경계 전체, "노드:" 는 노드 하나, 나머지는 질문 하나다.')
        lines += [f'  - {n}건 {why}' for why, n in skipped.items()]
    lines += [
        '', '## 읽는 법', '',
        '- 답은 yes · no · 무효(long 의 답이 판정 JSON 이 아님) · 보류(short 의 choice_mass 가 문턱 미만이거나 yes·no 동점) · 오류(호출 실패, '
        '사진·판정 기록을 못 읽음) · 행 없음으로 나눈다. long 의 yes 는 choice A 다.',
        '- 정답률과 재현율은 yes/no 로 답한 것만 센다. 보수 정답률은 참값이 있는 단위 전체가 분모이고 오류·무효·보류·행 없음을 오답으로 센다. '
        '변형끼리는 coverage 와 보수 정답률을 같이 보고 견준다.',
        '- yes 재현율은 참값 yes 가운데 yes 라고 답한 비율, no 재현율은 참값 no 가운데 no 라고 답한 비율이다(거짓 통과 = 1 − no 재현율). '
        '균형 정확도는 yes 재현율과 no 재현율(neg_cross·neg_fail 합침)의 평균이다. 항상 yes·항상 no 는 그 답만 낼 때의 보수 정답률이다.',
        f'- 문턱 0 줄은 choice_mass 와 무관하게 전부 채점하고(동점만 보류), 문턱 {min_mass:g} 줄은 그 미만과 choice_mass 가 없거나 NaN 인 답'
        '(응답 상태 표의 질량 없음·NaN)을 보류로 뺀다. long 은 문턱이 없다. 중복 행은 같은 (변형, id) 가 다시 나온 수이고 마지막 행으로 채점한다.',
        '- 질문별 표는 short 변형을 질문 id 로 나눈 것이다. 검은 사진 변형은 참값이 없어 yes 답 비율만 본다. yes 답 비율의 분모는 '
        '응답(보류 포함), yes/(yes+no) 의 분모는 yes·no 로 답한 것이다. 뒤집힌 질문(questions.yaml invert)은 참값이 반대라 '
        '원 질문의 pos 가 neg_fail 로, 원 질문의 neg_cross·neg_fail 이 pos 로 들어간다.',
        '- 짝 일치는 같은 (경계, 물체)에서 원 질문과 뒤집힌 질문의 답이 yes·no 로 서로 반대일 때만 원 질문의 답을 채택한다. '
        '둘이 같거나 한쪽이 보류·오류·행 없음이면 보류다. 재현율은 채택한 것만, 보수 정답률은 보류를 오답으로 센다. 참값은 원 질문 것이다. '
        '보류 가운데 두 답이 같은 것은 둘 다 yes(무엇을 물어도 yes)와 둘 다 no 로 나눠 적는다. 나머지 보류는 한쪽이 보류·오류·행 없음이다.',
        ('- long_same 의 참값은 boundaries.jsonl 이 없어 항목으로 남은 제 질문(pos·neg_fail)만으로 정했다. no 가 하나라도 있으면 no, '
         '남은 것이 모두 yes 면 yes(못 정해 빠진 제 질문이 있어도 yes 다), 제 질문 항목이 없는 경계는 못 정함이다. '
         if old else
         '- long_same 의 참값은 그 경계에서 끝난 노드의 제 질문으로 정한다. no 가 하나라도 있으면 no, 모두 정했고 모두 yes 면 yes, '
         '못 정한 제 질문이 하나라도 있거나 제 질문이 없는 경계(첫 READY, go_home_start, tilt)는 못 정함이라 채점에서 빠진다. ') +
        'blind 프롬프트는 마지막 노드 이름만 받으므로 승인 없이 이어진 place(place_blue·place_shaker)는 이름이 프롬프트에 없다.',
        '- long_swap 은 그 노드의 성공 조건이 깨진 다른 경계 사진을 주므로 참값이 no 다. 그런 사진이 같은 시드에 없는 경계는 부르지 않는다.',
        '- 사진 민감도는 기준 변형이 yes/no 로 답했고 이 변형도 응답한 단위에서 센다(무효·보류로 바뀐 것도 바뀐 것이다). '
        '검은 사진 변형은 참값이 없어 채점 표에 없다. 검은 사진에 yes 라고 답하는 비율이 높으면 사진을 보지 않고 답한다는 뜻이다.',
        '- short2 는 직전 경계 사진이 없는 항목(첫 경계, 앞 경계의 사진이 없는 경계)을 부르지 않는다. 지연은 요청 한 번의 왕복 시간이다.',
        '', '## short 문구', '',
        *(f'- SHORT1: `{x!r}`' for x in [x for x in vals('short1') if x] or [SHORT1]),
        *(f'- SHORT2: `{x!r}`' for x in [x for x in vals('short2') if x] or [SHORT2])]
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('items')
    ap.add_argument('--url', help='VLM 서버 주소. --rescore 가 아니면 필요하다')
    ap.add_argument('--out', help='results.jsonl·summary.md 를 쓸 폴더. --rescore 에서는 summary.md 만 쓴다(기본: results.jsonl 옆)')
    ap.add_argument('--rescore', metavar='RESULTS_JSONL', help='서버를 부르지 않고 이 results.jsonl 을 다시 채점한다')
    ap.add_argument('--min-mass', type=float, help=f'short 채점 문턱. 기본: 잴 때는 {MIN_MASS}, --rescore 는 results meta 의 값(없으면 {MIN_MASS})')
    ap.add_argument('--variants', nargs='+', choices=VARIANTS, help='기본: 전부')
    ap.add_argument('--questions', nargs='+', metavar='ID', help='이 질문 id 의 항목만 부르고 채점한다. --rescore 의 기본은 results meta 의 값(meta 행이 여럿이면 합집합). '
                         'long 변형은 경계 단위라 호출 수가 줄지 않는다')
    args = ap.parse_args(argv)
    items, bnd, skipped = load_set(args.items)
    if args.rescore:
        if args.url or args.variants:
            ap.error('--rescore 는 서버를 부르지 않는다. --url·--variants 와 같이 주지 않는다')
        if not Path(args.rescore).is_file():
            ap.error(f'{args.rescore} 이 없다')
        results, out = Path(args.rescore), Path(args.out or Path(args.rescore).parent)
        out.mkdir(parents=True, exist_ok=True)
    elif args.url and args.out:
        results, out = Path(args.out) / 'results.jsonl', Path(args.out)
    else:
        ap.error('--url 과 --out 을 주거나 --rescore 를 준다')
    rows = jl(results) if args.rescore else []
    metas = metas_of(rows)
    if args.min_mass is None and len({m.get('min_mass') for m in metas}) > 1:
        ap.error(f"results 의 meta 행 {len(metas)}개가 서로 다른 min_mass 로 잰 것이다({[m.get('min_mass') for m in metas]}). --min-mass 를 준다")
    asked = args.questions or meta_questions(metas)
    if asked:
        if set(asked) - {i['question_id'] for i in items}:
            ap.error(f"items 에 없는 질문 id: {sorted(set(asked) - {i['question_id'] for i in items})}")
        items = [i for i in items if i['question_id'] in asked]
    if not args.rescore:
        run(items, bnd, args.url, out, args.variants or VARIANTS, MIN_MASS if args.min_mass is None else args.min_mass,
            {'items': str(args.items), 'items_sha256': hashlib.sha256(Path(args.items).read_bytes()).hexdigest(),
             **({'questions': asked} if asked else {})})
        rows = jl(results)  # 실행 직후에도 파일을 다시 읽어 같은 길로 채점한다
    text = summary(items, bnd, skipped, rows, args.items, args.min_mass, asked)
    (out / 'summary.md').write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
