import pytest
from PIL import Image

from vlm_supervisor.place_path import PlacePathJudge, blocked, candidates, draw, parse_choice, project

HOLDER = [0.3, 0.0, 0.05]


def world(**extra):
    objs = {'SRC_BLUE': {'pos': [0.3, -0.3, 0.20], 'in_hand': 'right'}, 'HOLDER_1': {'pos': HOLDER}}
    return {'objects': objs | extra, 'right_hand': {'pos': [0.3, -0.3, 0.25]}}


# 카메라: base 원점 위 1 m 에서 아래를 봄. base x → 이미지 u, base y → 이미지 v.
K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
T = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]]


def test_candidates_geometry():
    c = {x['name']: x for x in candidates(world(), 'right', HOLDER)}
    assert [x['id'] for x in c.values()] == [1, 2, 3, 4]
    for x in c.values():
        assert x['cup'][0] == [0.3, -0.3, 0.2] and x['cup'][-1] == pytest.approx(HOLDER)
        # 손 웨이포인트 = 컵 + (손 - 컵) 고정 오프셋
        assert [h[2] - p[2] for h, p in zip(x['hand'], x['cup'])] == pytest.approx([0.05] * len(x['cup']))
    assert max(p[2] for p in c['lift_high']['cup']) > max(p[2] for p in c['direct']['cup']) + 0.05
    assert c['detour_left']['cup'][1][0] < 0.3 < c['detour_right']['cup'][1][0]


def test_candidates_need_geometry():
    assert candidates(world(), 'right', None) == []
    assert candidates({'objects': {}}, 'right', HOLDER) == []


def test_blocked_only_near_paths_and_ignores_holder_and_held_cup():
    w = world(X={'pos': [0.3, -0.15, 0.2]}, FAR={'pos': [0.8, 0.5, 0.0]})
    bad = blocked(candidates(w, 'right', HOLDER), w, 'right', ignore=('HOLDER_1',))
    assert bad == {1: ['X']}


def test_parse_choice():
    assert parse_choice('think {"choice": 2, "reason": "ok"}', 4, {}) == {'choice': 2, 'reason': 'ok', 'valid': True}
    assert not parse_choice('no json', 4, {})['valid']
    assert 'out of' in parse_choice('{"choice": 7, "reason": ""}', 4, {})['error']
    assert 'BLOCKED' in parse_choice('{"choice": 1, "reason": ""}', 4, {1: ['X']})['error']


def test_project_and_draw_synthetic_camera():
    assert project([0, 0, 0], K, T) == pytest.approx((320, 240))
    assert project([0.1, 0.2, 0], K, T) == pytest.approx((370, 140))
    assert project([0, 0, 2.0], K, T) is None  # 카메라 뒤
    img = Image.new('RGB', (640, 480))
    out = draw(img, candidates(world(), 'right', HOLDER), K, T)
    u, v = project(HOLDER, K, T)  # 경로 끝(홀더) 투영점 근처에 선이 그려진다
    near = [out.getpixel((round(u) + du, round(v) + dv)) for du in range(-3, 4) for dv in range(-3, 4)]
    assert out is not img and any(p != (0, 0, 0) for p in near)
    assert img.getpixel((round(u), round(v))) == (0, 0, 0)  # 원본은 그대로
    assert draw(img, [], None, T) is None


class Backend:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def ask(self, images, text, **kwargs):
        self.prompts.append(text)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a, {}


IMG = Image.new('RGB', (640, 480))


def test_choose_valid_with_camera_in_world():
    b = Backend('{"choice": 3, "reason": "clear"}')
    w = world() | {'camera': {'K': K, 'T_cam_base': T}}
    r = PlacePathJudge(b).choose(IMG, w, 'right', 'HOLDER_1')
    assert (r['choice'], r['valid'], r['drawn']) == (3, True, True)
    assert len(b.prompts) == 1 and 'arrow' in b.prompts[0]


def test_choose_rejects_blocked_then_retries_then_falls_back():
    w = world(X={'pos': [0.3, -0.15, 0.2]})
    b = Backend('{"choice": 1, "reason": "x"}', 'garbage')
    r = PlacePathJudge(b).choose(IMG, w, 'right', 'HOLDER_1')
    assert (r['choice'], r['valid'], r['drawn']) == (1, False, False)
    assert len(b.prompts) == 2 and 'BLOCKED' in b.prompts[0] and b.prompts[1].startswith('Your previous answer')
    assert r['blocked'] == {1: ['X']}


def test_choose_retry_succeeds():
    b = Backend('{"choice": 9, "reason": ""}', '{"choice": 2, "reason": "high"}')
    r = PlacePathJudge(b).choose(IMG, world(), 'right', HOLDER)
    assert (r['choice'], r['valid']) == (2, True) and 'retried_from' in r


@pytest.mark.parametrize('image,w,err', [(IMG, {'objects': {}}, 'no geometry'), (None, world(), 'no camera frame')])
def test_choose_fallback_without_inputs(image, w, err):
    r = PlacePathJudge(Backend()).choose(image, w, 'right', HOLDER)
    assert (r['choice'], r['valid']) == (1, False) and err in r['error']


def test_backend_error_falls_back_without_retry():
    b = Backend(RuntimeError('down'))
    r = PlacePathJudge(b).choose(IMG, world(), 'right', HOLDER)
    assert (r['choice'], r['valid'], len(b.prompts)) == (1, False, 1)
