"""real_frames·bagread: 러너 기록 → 경계 묶기와 사진 시각, 작은 db3 로 CompressedImage 꺼내기."""
import sqlite3
import struct

import pytest

from vlm_supervisor.bench import bagread, real_frames

SPEC = {'sequence': [
    {'id': 'a'}, {'id': 'b', 'auto': True},
    {'id': 'L', 'lanes': [[{'id': 'l1'}], {'nodes': [{'id': 'l2'}]}]},
    {'id': 'p'}, {'id': 'h', 'auto': True},
    {'id': 'done', 'type': 'terminal'},
]}


def test_groups_auto_and_lanes():
    assert real_frames.groups(SPEC) == {'a': (0, 'a'), 'b': (0, 'a'), 'L': (1, 'L'), 'l1': (1, 'L'), 'l2': (1, 'L'),
                                        'p': (2, 'p'), 'h': (2, 'p')}


def r(t, node, event, **kw):
    return {'timestamp': t, 'state_id': node, 'event': event, **kw}


ROWS = [
    r(0, 'a', 'enter'), r(1, 'a', 'exit', result='COMPLETED'), r(1, 'b', 'enter'), r(2, 'b', 'exit'),
    r(5, 'L', 'enter'), r(5, 'l1', 'enter'), r(5, 'l2', 'enter'), r(6, 'l1', 'exit'), r(7, 'l2', 'exit'),
    r(7, 'L', 'exit', executor_type='lanes'),
    r(10, 'p', 'enter'), r(11, 'p', 'failure', termination_reason='place_failed'),
    r(11.5, 'p', 'judge', signals={'rh_place_r': {'hand_empty': True}}),  # 실패한 놓기의 신호는 failure 뒤에 온다
    r(12, 'p', 'enter'), r(13, 'p', 'exit'), r(13, 'h', 'enter'), r(14, 'h', 'exit'),
    r(15, 'done', 'enter'),
]


def test_episode_bounds_groups_retry_failure_and_end():
    out = real_frames.episode_bounds(ROWS, real_frames.groups(SPEC), float)
    got = [(b['head'], b['node'], [n['id'] for n in b['nodes']], b['t_fin'], round(b['t_photo'], 6), b['gap_s'], b['end'])
           for b in out]
    assert got == [
        ('a', 'b', ['a', 'b'], 2, 3.0, 3, False),       # 다음 진입까지 3 s → 종료 + 1 s
        ('L', 'l2', ['l1', 'l2'], 7, 8.0, 3, False),    # lanes 행 자체는 노드가 아니다
        ('p', 'p', ['p'], 11, 11.9, 1, True),           # 재시도: 다음 진입 0.1 s 앞
        ('p', 'h', ['p', 'h'], 14, 15.0, None, True),   # 끝: 종료 + 1 s
    ]
    failed = out[2]['nodes'][0]
    assert (failed['event'], failed['reason'], failed['t_enter']) == ('failure', 'place_failed', 10)
    assert failed['signals'] == {'rh_place_r': {'hand_empty': True}}


def test_frame_at_tolerance_and_missing_bag():
    st = [(int(10e9), 'db'), (int(10.2e9), 'db')]
    cams = [(10.0, 10.2, st)]
    assert real_frames.frame_at(cams, 10.15) == ((int(10.2e9), 'db'), None)
    assert real_frames.frame_at(cams, 10.45) == ((int(10.2e9), 'db'), None)  # 끝 + TOL 안
    assert real_frames.frame_at(cams, 10.6) == (None, '그 시각의 카메라 bag 없음')
    gap = [(10.0, 20.0, [(int(10e9), 'db'), (int(20e9), 'db')])]
    assert real_frames.frame_at(gap, 15.0) == (None, '프레임이 목표 시각에서 멀다')


def cdr_image(fmt: str, data: bytes) -> bytes:
    def s(x: str) -> bytes:
        b = x.encode() + b'\0'
        return struct.pack('<I', len(b)) + b

    b = b'\x00\x01\x00\x00' + struct.pack('<iI', 1, 2) + s('cam')  # 캡슐 헤더 · stamp · frame_id
    b += b'\0' * (-(len(b) - 4) % 4) + s(fmt)
    b += b'\0' * (-(len(b) - 4) % 4)
    return b + struct.pack('<I', len(data)) + data


def make_db(path, rows):
    c = sqlite3.connect(path)
    c.execute('create table topics (id integer primary key, name text, type text)')
    c.execute('create table messages (id integer primary key, topic_id integer, timestamp integer, data blob)')
    c.executemany('insert into topics values (?, ?, ?)', [(1, real_frames.TOPIC, 'img'), (2, '/other', 'x')])
    c.executemany('insert into messages (topic_id, timestamp, data) values (?, ?, ?)', rows)
    c.commit()
    c.close()


def test_bagread_stamps_at_cimage(tmp_path):
    a, b = tmp_path / 'a_0.db3', tmp_path / 'a_1.db3'
    make_db(a, [(1, 30, cdr_image('jpeg', b'JPG30')), (2, 20, b'other')])
    make_db(b, [(1, 10, cdr_image('rgb8; jpeg compressed bgr8', b'JPEG10'))])
    assert bagread.stamps(tmp_path, real_frames.TOPIC) == [(10, b), (30, a)]
    assert bagread.cimage(bagread.at(b, real_frames.TOPIC, 10)) == ('rgb8; jpeg compressed bgr8', b'JPEG10')
    assert bagread.cimage(bagread.at(a, real_frames.TOPIC, 30)) == ('jpeg', b'JPG30')
    with pytest.raises(KeyError):
        bagread.at(a, real_frames.TOPIC, 20)  # 그 시각은 다른 토픽이다
