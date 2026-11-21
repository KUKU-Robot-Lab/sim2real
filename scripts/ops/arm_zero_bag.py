#!/usr/bin/env python3
"""Derive candidate/history and a verification table from existing sqlite rosbag(s).

No ROS initialization or hardware access. Outputs are analysis, never raw recorders.
Run again with the measurement bag AND all subsequent apply/verify bags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import struct

import numpy as np
import yaml

from arm_zero_calibrate import Abort
from arm_zero_runtime import TOPIC, reduce_events


def decode_string(data):
    if len(data) < 9 or data[:4] not in (b'\x00\x01\x00\x00', b'\x00\x00\x00\x00'):
        raise ValueError('expected CDR std_msgs/String')
    size = struct.unpack('<I' if data[1] == 1 else '>I', data[4:8])[0]
    if size < 1 or len(data) < 8 + size or data[7 + size] != 0:
        raise ValueError('truncated CDR String')
    return json.loads(data[8:7 + size].decode('utf-8'))


def read_bags(paths):
    files = set()
    for path in paths:
        files.update(path.rglob('*.db3') if path.is_dir() else [path])
    if not files:
        raise Abort('no sqlite rosbag files found')
    events = []
    sources = []
    for path in sorted(files):
        # URI read-only mode cannot accidentally create/modify a bag.
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            topics = db.execute('SELECT id, type FROM topics WHERE name=?', (TOPIC,)).fetchall()
            for topic_id, msg_type in topics:
                if msg_type != 'std_msgs/msg/String':
                    raise Abort('unexpected calibration topic type')
                for stamp, data in db.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (topic_id,)):
                    events.append((stamp, decode_string(data)))
                stat = path.stat()
                sources.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return [event for _, event in sorted(events, key=lambda item: item[0])], sources


def comparison(events):
    """FK is labelled inferred; it is never presented as physical measurement."""
    rows = ['# RH56F1 부착 전후 검증 (bag 분석)', '',
            'FK는 엔코더·URDF로 추정한 값이다. 실측 플랜지/손 장착 오차와 전원 재투입 확인은 별도이다.', '',
            '| 상태 | 자세 | 최대 관절 추종 오차 (deg) | FK 플랜지 xyz (m) |',
            '|---|---|---:|---|']
    baseline = {}
    for e in events:
        if e['kind'] != 'verification':
            continue
        key = e['pose_index']
        goal = np.asarray(e['model_target'])
        if e['hand'] == 'detached':
            baseline[key] = goal
        elif key not in baseline or not np.allclose(goal, baseline[key], atol=1e-10, rtol=0):
            raise Abort('attached verification lacks matching detached target')
        error = float(np.rad2deg(np.max(np.abs(np.asarray(e['model_q']) - goal))))
        xyz = np.asarray(e['flange_fk'])[:3, 3].round(5).tolist()
        rows.append(f'| {e["hand"]} | {key} | {error:.4f} | {xyz} |')
    rows += ['', '합격 판정: 미확정. 물리적 대칭·전원 재투입·실측 플랜지·손 상대 방향 확인 전 완료로 선언하지 않는다.',
             '부착 전후 차이를 평균하거나 영점에 더하지 않는다. j3 과거 5.34° 재적용 금지; 외력 j7 사건 제외.', '']
    return '\n'.join(rows)


def inspection_report(events):
    records = [e for e in events if e['kind'] == 'zero_inspection']
    current = records[-1]
    if current.get('run_id') and not any(e.get('kind') == 'finished' and e.get('run_id') == current['run_id'] for e in events):
        raise Abort('inspection bag missing matching finished event; reference not updated')
    baseline = next((e for e in records if e['hand'] == 'detached' and e['side'] == current['side']), None)
    if current['hand'] == 'attached' and baseline is None:
        raise Abort('attached inspection requires preceding detached reference bag')
    if baseline is not None and any(baseline[k] != current[k] for k in ('side', 'session', 'model_hash')):
        raise Abort('inspection baseline does not match side/session/model')
    lines = ['# 기존 영점 확인', '', '모터 구동과 영점 저장 없이 읽은 값이다. 실제 링크·플랜지 기준 자세의 일치는 운영자가 확인해야 한다.', '',
             '| 관절 | 손 제거 (deg) | 손 부착 (deg) | 부착−제거 (deg) |', '|---|---:|---:|---:|']
    for i in range(7):
        a = float(np.rad2deg(baseline['mean'][i])) if baseline else None
        b = float(np.rad2deg(current['mean'][i])) if current['hand'] == 'attached' else None
        lines.append(f'| j{i + 1} | {a if a is not None else "미측정"} | {b if b is not None else "미측정"} | {b - a if b is not None and a is not None else "미측정"} |')
    lines += ['', '자동 합격 판정 없음. 기존 엔코더 영점과 실제 기계 영점의 일치 여부를 확인한 뒤 보정 필요성을 판단한다.',
              '손 하중에 따른 차이를 영점에 더하거나 두 결과를 평균하지 않는다.', '']
    return records, '\n'.join(lines)


def incomplete_summary(events):
    failures = [e for e in events if e.get('kind') == 'aborted']
    reason = failures[0].get('reason', 'unknown') if failures else '후보 생성 전 종료; 앞선 실행 오류를 확인하세요.'
    return dict(status='incomplete', candidate_available=False,
                reason=reason, control_phase_entered=any(e.get('kind') == 'started' for e in events),
                command_count=sum(e.get('kind') == 'command' for e in events))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bags', type=Path, nargs='+')
    parser.add_argument('--failure-reason', default='')
    parser.add_argument('--out', type=Path, required=True, help='new analysis directory (will not overwrite)')
    args = parser.parse_args(argv)
    events, sources = read_bags(args.bags)
    if any(e['kind'] == 'zero_inspection' for e in events):
        records, report = inspection_report(events)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / 'zero_inspection.yaml').write_text(yaml.safe_dump(dict(records=records, source_bags=sources), allow_unicode=True))
        (args.out / 'comparison.md').write_text(report)
        print(args.out / 'comparison.md')
        return 0
    if not any(e['kind'] == 'candidate' for e in events):
        summary = incomplete_summary(events)
        if args.failure_reason and not any(e['kind'] == 'aborted' for e in events):
            summary['reason'] = args.failure_reason
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / 'run_status.yaml').write_text(yaml.safe_dump(summary, allow_unicode=True))
        print('실행 미완료:', summary['reason'])
        print('중단 진단:', args.out / 'run_status.yaml')
        return 2
    candidate, history = reduce_events(events)
    candidate['source_bags'] = sources
    history['source_bags'] = sources
    report = comparison(events)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'candidate.yaml').write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True))
    (args.out / 'history.yaml').write_text(yaml.safe_dump(history, sort_keys=False, allow_unicode=True))
    (args.out / 'comparison.md').write_text(report)
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
