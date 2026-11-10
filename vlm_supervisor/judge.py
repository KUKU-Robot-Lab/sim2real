"""사진과 상태를 분리해 관찰하고 A~E 결정을 파싱한다."""
from __future__ import annotations

import json

from .planner import load_cards, parse_json

ACTIONS = {'A': 'next', 'B': 'retry', 'C': 'rescan', 'D': 'home', 'E': 'stop'}


def parse_judgement(raw: str) -> dict:
    fallback = {'success': False, 'choice': 'E', 'action': 'stop', 'reason': 'invalid judgement JSON'}
    try:
        value = parse_json(raw, dict)
    except (ValueError, TypeError, AttributeError):
        return fallback
    choice = value.get('choice')
    success = value.get('success')
    if (not isinstance(choice, str) or choice not in ACTIONS or type(success) is not bool
            or (choice == 'A') != success or not isinstance(value.get('reason'), str)):
        return fallback
    return {'success': success, 'choice': choice, 'action': ACTIONS[choice], 'reason': value['reason']}


class Judge:
    def __init__(self, backend, sentence='', plan=None):
        self.backend, self.sentence, self.plan = backend, sentence, plan
        self.cards = load_cards()

    def judge(self, status: dict, image) -> dict:
        if image is None:
            return {'success': False, 'choice': 'E', 'action': 'stop', 'reason': '카메라 프레임 없음'}
        prompt = ('Judge the completed node in last, using the CURRENT photograph and status JSON. '
                  'world/expect are controller declarations, not visual proof. Report contradictions. '
                  'Do not infer hidden bead counts from a photograph. '
                  'Choose exactly A=success/next, B=failure/retry, C=failure/rescan, '
                  'D=failure/home (only empty hands), E=unsafe or unclear/stop and human. '
                  'B/C/D are requests only: the runner has no recovery hooks. '
                  'Return only {"success":bool,"choice":"A".."E","reason":"short explanation"}. '
                  'success must be true exactly for A.\n'
                  + json.dumps({'sentence': self.sentence, 'plan': self.plan,
                                'status': status, 'skill_cards': self.cards}, ensure_ascii=False))
        raw, meta = self.backend.ask([image], prompt)
        return {**parse_judgement(raw), 'raw': raw, 'meta': meta}
