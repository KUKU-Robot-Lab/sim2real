"""문장 → 카드 기반 계획. simulate는 선언 expect를 센서 사실로 취급하지 않는다."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

CARDS_PATH = Path(__file__).parent / 'config' / 'skill_cards.yaml'


def load_cards(path=CARDS_PATH) -> dict:
    cards = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cards, dict):
        raise ValueError('skill cards must be a mapping')
    return cards


def parse_json(text: str, expected_type):
    text = text.split('</think>')[-1].strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != ('[' if expected_type is list else '{'):
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, expected_type):
            return value
    raise ValueError('VLM response has no valid JSON ' + expected_type.__name__)


def simulate(steps, cards=None) -> dict:
    """Validate [skill, hand, object, target] using symbolic hand/pose/holder state."""
    cards = cards or load_cards()
    allowed = {card['id'] for card in cards['스킬']}
    objects = {obj['시스템_이름']: obj for obj in cards['객체_사전']}
    hands = {'right': None, 'left': None}
    ready = {'right': False, 'left': False}
    locations = {name: 'table' for name in objects}
    poured = set()
    scene = False
    terminal = False
    errors = []
    if not isinstance(steps, list) or not steps:
        return {'ok': False, 'errors': ['plan must be a nonempty list'], 'world': {}}
    for index, step in enumerate(steps):
        try:
            if terminal:
                raise ValueError('steps after terminal')
            if not isinstance(step, list) or len(step) != 4 or not all(isinstance(x, str) for x in step):
                raise ValueError('step must have four strings')
            skill, hand, obj, target = step
            if skill not in allowed:
                raise ValueError('unknown card: ' + skill)
            if skill == 'scene_snapshot':
                scene = True
            elif skill == 'go_home':
                if hand not in {'both', 'right', 'left'}:
                    raise ValueError('invalid home hand')
                sides = list(hands) if hand == 'both' else [hand]
                if any(hands[side] is not None for side in sides):
                    raise ValueError('home requires empty hands')
                for side in sides:
                    ready[side] = False
            elif skill == 'terminal':
                if target != 'SUCCESS' or any(hands.values()):
                    raise ValueError('terminal requires SUCCESS and empty hands')
                terminal = True
            elif skill == 'human':
                raise ValueError('human confirmation cannot be simulated as success')
            elif skill == 'tilt':
                if (hand != 'both' or obj not in {'SRC_BLUE', 'SRC_PINK'} or target != 'SHAKER'
                        or hands != {'right': obj, 'left': 'SHAKER'} or not all(ready.values())):
                    raise ValueError('tilt requires both held cups at tilt start')
                poured.add(obj)
            else:
                if obj not in objects or hand != objects[obj]['손']:
                    raise ValueError('object/hand mismatch')
                if skill == 'grasp':
                    if not scene or hands[hand] or locations[obj] != 'table':
                        raise ValueError('grasp requires snapshot, empty hand and table object')
                    hands[hand], locations[obj], ready[hand] = obj, hand, False
                elif skill == 'to_tilt_start':
                    if hands[hand] != obj:
                        raise ValueError('carry requires held object')
                    ready[hand] = True
                elif skill == 'place':
                    if hands[hand] != obj or target != objects[obj]['홀더']:
                        raise ValueError('place requires held object and matching holder')
                    if obj == 'SHAKER' and any(
                            s[0] == 'tilt' for s in steps[index + 1:] if isinstance(s, list) and s):
                        raise ValueError('keep shaker until pouring is done')
                    hands[hand], locations[obj], ready[hand] = None, target, False
                else:
                    raise ValueError('unsupported simulation card: ' + skill)
        except ValueError as exc:
            errors.append(f'step {index}: {exc}')
            break
    if not terminal:
        errors.append('plan must end with terminal')
    return {'ok': not errors, 'errors': errors,
            'world': {'hands': hands, 'locations': locations, 'poured': sorted(poured)}}


class Planner:
    def __init__(self, backend, cards_path=CARDS_PATH):
        self.backend = backend
        self.cards = load_cards(cards_path)
        self.last_answer = ""

    def plan(self, sentence, image=None):
        prompt = ('Plan the Korean user instruction using only these skill cards. '
                  'Return only a JSON list of [skill, hand, object, target], four strings per step. '
                  'Use "-" for unused values; grasp/carry/place need the object and hand. '
                  'Begin with go_home and scene_snapshot; end with ["terminal","-","-","SUCCESS"]. '
                  'One object per hand; carry both cups to_tilt_start before tilt. '
                  'Use only named objects and assigned holders. No executable code.\n'
                  + json.dumps(self.cards, ensure_ascii=False)
                  + '\nUser instruction: ' + sentence)
        raw, meta = self.backend.ask([] if image is None else [image], prompt, max_new_tokens=1024)
        self.last_answer = raw
        steps = parse_json(raw, list)
        steps = [[step.get(key, "-") for key in ("skill", "hand", "object", "target")]
                 if isinstance(step, dict) else step for step in steps]
        result = simulate(steps, self.cards)
        if not result['ok']:
            raise ValueError('invalid VLM plan: ' + '; '.join(result['errors']))
        return {'sentence': sentence, 'steps': steps, 'simulation': result, 'meta': meta}
