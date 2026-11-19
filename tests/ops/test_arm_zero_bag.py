import importlib.util
from pathlib import Path
import struct
import sys

import pytest

OPS = Path(__file__).resolve().parents[2] / 'scripts/ops'
sys.path.insert(0, str(OPS))


def test_decode_string_cdr():
    spec = importlib.util.find_spec('arm_zero_bag')
    assert spec is not None, 'bag analysis not implemented'
    from arm_zero_bag import decode_string
    data = b'{"kind":"state"}\x00'
    assert decode_string(b'\x00\x01\x00\x00' + struct.pack('<I', len(data)) + data) == {'kind': 'state'}
    with pytest.raises(ValueError):
        decode_string(b'\x00\x01\x00\x00\xff\xff\xff\xff')


def test_sqlite_analysis_outputs_unknown_history(tmp_path):
    import sqlite3
    import json
    from arm_zero_bag import main
    from arm_zero_runtime import TOPIC
    import yaml
    bag = tmp_path / 'test.db3'
    c = {'kind': 'candidate', 'candidate_id': 'c', 'side': 'right', 'session': 's',
         'model_hash': 'm', 'delta': [0.] * 7, 'hits': {}, 'version': 1}
    events = [c, {'kind': 'zero_attempt', 'candidate_id': 'c', 'joint': 7}]
    with sqlite3.connect(bag) as db:
        db.execute('CREATE TABLE topics(id INTEGER, name TEXT, type TEXT)')
        db.execute('CREATE TABLE messages(topic_id INTEGER, timestamp INTEGER, data BLOB)')
        db.execute('INSERT INTO topics VALUES(1, ?, ?)', (TOPIC, 'std_msgs/msg/String'))
        for i, e in enumerate(events):
            raw = json.dumps(e).encode() + b'\x00'
            data = b'\x00\x01\x00\x00' + struct.pack('<I', len(raw)) + raw
            db.execute('INSERT INTO messages VALUES(1, ?, ?)', (i, data))
    out = tmp_path / 'analysis'
    main([str(bag), '--out', str(out)])
    history = yaml.safe_load((out / 'history.yaml').read_text())
    assert history['unknown_joints'] == [7]
    assert history['saved_joints'] == []
    assert '미확정' in (out / 'comparison.md').read_text()
    assert not yaml.safe_load((out / 'candidate.yaml').read_text())['reviewed_no_contact']


def test_preflight_failure_keeps_first_reason_without_candidate_error():
    from arm_zero_bag import incomplete_summary
    result = incomplete_summary([
        {'kind': 'startup_snapshot'},
        {'kind': 'aborted', 'phase': 'preflight', 'reason': 'head envelope overlaps'},
    ])
    assert result['reason'] == 'head envelope overlaps'
    assert not result['candidate_available']
    assert result['command_count'] == 0


def test_incomplete_bag_produces_diagnostic_not_a_candidate(tmp_path):
    import json
    import sqlite3
    from arm_zero_bag import main
    from arm_zero_runtime import TOPIC
    import yaml
    bag = tmp_path / 'failed.db3'
    with sqlite3.connect(bag) as db:
        db.execute('CREATE TABLE topics(id INTEGER, name TEXT, type TEXT)')
        db.execute('CREATE TABLE messages(topic_id INTEGER, timestamp INTEGER, data BLOB)')
        db.execute('INSERT INTO topics VALUES(1, ?, ?)', (TOPIC, 'std_msgs/msg/String'))
        raw = json.dumps({'kind': 'aborted', 'phase': 'preflight', 'reason': 'model clearance blocked'}).encode() + b'\x00'
        db.execute('INSERT INTO messages VALUES(1, 1, ?)', (b'\x00\x01\x00\x00' + struct.pack('<I', len(raw)) + raw,))
    out = tmp_path / 'diagnostic'
    assert main([str(bag), '--out', str(out)]) == 2
    result = yaml.safe_load((out / 'run_status.yaml').read_text())
    assert result['reason'] == 'model clearance blocked'
    assert result['command_count'] == 0
    assert not (out / 'candidate.yaml').exists()
    assert not (out / 'history.yaml').exists()


def test_existing_zero_before_after_inspection_requires_matching_reference():
    from arm_zero_bag import inspection_report
    from arm_zero_calibrate import Abort
    import copy
    baseline = dict(kind='zero_inspection', side='right', session='s', model_hash='m', hand='detached', mean=[.01] * 7)
    attached = dict(baseline, hand='attached', mean=[.02] * 7)
    records, report = inspection_report([baseline, attached])
    assert len(records) == 2 and 'j7' in report
    assert '자동 합격 판정 없음' in report
    with pytest.raises(Abort, match='preceding detached'):
        inspection_report([attached])
    wrong = copy.deepcopy(attached)
    wrong['model_hash'] = 'different'
    with pytest.raises(Abort, match='does not match'):
        inspection_report([baseline, wrong])


def test_inspection_reference_requires_durable_matching_finished():
    from arm_zero_bag import inspection_report
    from arm_zero_calibrate import Abort
    record = dict(kind='zero_inspection', run_id='check-1', side='right', session='s', model_hash='m', hand='detached', mean=[0.] * 7)
    with pytest.raises(Abort, match='matching finished'):
        inspection_report([record])
    with pytest.raises(Abort, match='matching finished'):
        inspection_report([record, dict(kind='finished', run_id='other-run')])
    records, report = inspection_report([record, dict(kind='finished', run_id='check-1')])
    assert records == [record] and 'j7' in report
