"""사진과 상태를 분리해 관찰하고 A~E 결정을 파싱한다."""
from __future__ import annotations

import json

from .planner import load_cards, parse_json
from .failure_policy import allowed_actions, runner_detail

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
    def __init__(self, backend, sentence='', plan=None, blind=False):
        self.backend, self.sentence, self.plan, self.blind = backend, sentence, plan, blind
        self.cards = load_cards()

    def judge(self, status: dict, image) -> dict:
        if image is None:
            return {'success': False, 'choice': 'E', 'action': 'stop', 'reason': '카메라 프레임 없음'}
        detail = runner_detail(status)
        allowed = allowed_actions(status, detail)
        if self.blind:  # blind 은 선택지도 상태와 무관하게 A~E 고정(허용 밖은 아래 clamp 가 stop)
            choices = ['A=success/next', 'B=failure/retry', 'C=failure/rescan',
                       'D=failure/home (only confirmed empty hands)', 'E=unsafe or unclear/stop and human']
        else:
            choices = ['E=unsafe or unclear/stop and human']
            if 'next' in allowed or 'recover' in allowed:
                choices.insert(0, 'A=success/next (pending recovery requires runner approval)')
            if 'retry' in allowed:
                choices.append('B=failure/retry')
            if 'rescan' in allowed:
                choices.append('C=failure/rescan')
            if 'home' in allowed:
                choices.append('D=failure/home (only confirmed empty hands)')
        payload = {'sentence': self.sentence, 'plan': self.plan,
                   'status': status, 'runner_detail': detail, 'allowed_actions': sorted(allowed),
                   'skill_cards': self.cards}
        if self.blind:  # status 는 runner 결과(code·reason·results·pending·history·world 갱신)를 담아 노드 이름만 남긴다
            last = status.get('last')
            payload['status'] = {'last': {'node': last.get('node') if isinstance(last, dict) else None}}
            del payload['runner_detail'], payload['allowed_actions']
        prompt = ('Judge the completed node in last, using the CURRENT photograph and status JSON. '
                  'world/expect are controller declarations, not visual proof. Report contradictions. '
                  'Do not infer hidden bead counts from a photograph. '
                  'Rule terminations are distinct from physical failures; None/missing means unknown. '
                  'opened_before_latch does not prove empty hands: use holder seated z 15 mm and tactile release threshold. '
                  'Never override deploy no-retry rules or blocked continue_on_fail. '
                  + 'Choose exactly one allowed choice: ' + ', '.join(choices) + '. '
                  'Return only {"success":bool,"choice":"A".."E","reason":"short explanation"}. '
                  'success must be true exactly for A.\n'
                  + ('Blind mode: runner outcome fields are withheld; judge success from the photograph alone.\n'
                     if self.blind else '')
                  + json.dumps(payload, ensure_ascii=False))
        raw, meta = self.backend.ask([image], prompt)
        decision = parse_judgement(raw)
        if decision['action'] not in allowed and not (decision['action'] == 'next' and 'recover' in allowed):
            decision = {'success': False, 'choice': 'E', 'action': 'stop',
                        'reason': 'deploy recovery rule forbids model action', 'model_decision': decision}
        return {**decision, 'raw': raw, 'meta': meta, 'prompt': prompt, **({'blind': True} if self.blind else {})}
