#!/usr/bin/env python3
"""Operator entry point: offline setup, then one recorded calibration command."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import os
import signal
import threading
from collections import deque
from pathlib import Path
import subprocess
import sys

import yaml

from arm_zero_calibrate import ROOT


def confirm(label):
    if input(label + ' [확인했으면 yes]: ').strip().lower() != 'yes':
        raise RuntimeError('확인되지 않아 중단합니다.')


def checks(cfg, side, hand):
    print('지금 실행할 팔:', side, '/ 손 상태:', hand)
    confirm('양팔에 지지대가 있고 disable 시에도 안전합니까?')
    confirm('손과 ADAPTOR 분리/부착 상태, 탁자·케이블 여유를 확인했습니까?')
    confirm('팔 CAN 제어 프로세스를 운영자가 정상 종료했고 별도 영점 변경이 없습니까?')
    op = cfg['operation']
    op.update(hand_state=hand, supported_on_disable=True, table_and_cables_clear=True,
              can_ownership_review='operator confirmed ' + datetime.now().isoformat(), no_external_zero_writes=True)

    confirm('탁자·물체·케이블·지지대를 팔 이동 경로 밖으로 정리했습니까?')


def setup(args):
    from arm_zero_auto import configure
    cfg = yaml.safe_load((args.config if args.config.exists() else ROOT / 'config/arm_zero_calibration.yaml').read_text())
    defaults = yaml.safe_load((ROOT / 'config/arm_zero_calibration.yaml').read_text())
    for key in ('torque_cap', 'contact_torque', 'torque_review'):
        if cfg['search'].get(key) is None:
            cfg['search'][key] = defaults['search'][key]
    cfg = configure(cfg, args.side, args.hand)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print('자동 설정 저장:', args.config)
    print('시작 관절값·반대 팔 상태는 실행 시 자동 수신합니다. 경유점 입력은 없습니다.')
    print('제조사 팔 순서 j4→j3→j5→j6→j7→j2→j1, gripper/목 지령 없음.')
    print('detached: 측정 후보 생성 → 관절별 apply → verify; attached: 같은 좌표로 verify.')
    print('제조사 게인 사용. 탐색 지령 속도는 0.05 rad/s로 제한하며 실제 속도 0.1 rad/s 초과 시 중단합니다.')
    print('실제 시작 자세에서 전체 준비/탐색 경로 검사를 통과한 뒤에만 enable합니다.')
    return 0


def run(args):
    if not args.config.exists():
        setup(args)
    cfg = yaml.safe_load(args.config.read_text())
    if getattr(args, 'action', None):
        from arm_zero_auto import configure
        cfg = configure(cfg, args.side, args.hand)
    if args.hand == 'attached' and getattr(args, 'action', None) in ('measure', 'apply'):
        raise RuntimeError('RH56F1 부착 단계는 verify입니다. 손 하중 차이를 영점에 저장하지 않습니다.')
    if not args.execute:
        if getattr(args, 'action', None) == 'inspect':
            print('기존 영점 읽기 전용 확인. --execute일 때만 상태를 수신합니다. 모터 지령·영점 저장 없음.')
            return 0
        args.config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return subprocess.call([sys.executable, str(ROOT / 'scripts/ops/arm_zero_calibrate.py'), 'plan',
                                '--side', args.side, '--hand', args.hand, '--config', str(args.config)])
    if args.action != 'inspect' and cfg['operation'].get('pending_zero_run'):
        raise RuntimeError('미확정 영점 저장 런: ' + cfg['operation']['pending_zero_run']
                           + '; 원본 bag의 저장 이력 확인 전 재구동할 수 없습니다.')
    # Dependency checks before starting any recorder/process with device access.
    for name in ('rclpy', 'openarm_can'):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f'{name} 모듈이 없습니다. ROS 환경과 arm-zero venv를 확인하세요.')
    if os.environ.get('ROS_DOMAIN_ID') != '50':
        raise RuntimeError('ROS_DOMAIN_ID=50 환경에서 실행해야 합니다.')
    latest = cfg['operation'].get('latest_analysis')
    if latest and args.action in ('apply', 'verify'):
        args.candidate = args.candidate or Path(latest) / 'candidate.yaml'
        args.history = args.history or Path(latest) / 'history.yaml'
    if args.candidate is not None and args.action in ('apply', 'verify'):
        candidate = yaml.safe_load(args.candidate.read_text())
        if not candidate.get('reviewed_no_contact'):
            confirm('측정 후보를 검토했고 몸통·케이블 접촉과 외력이 없었습니까?')
            candidate['reviewed_no_contact'] = True
            args.candidate.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True))
    if args.action == 'apply' and (args.joint is None or args.candidate is None or args.history is None):
        raise RuntimeError('apply에는 --joint, --candidate, --history가 필요합니다.')
    if args.action == 'inspect':
        confirm('손/ADAPTOR 상태가 지정한 상태이고, 팔을 지지한 채 동일한 물리적 영점 기준 자세에 두었습니까?')
        confirm('팔 CAN 제어를 종료했고 모터가 disable 상태입니까?')
        if args.hand == 'attached' and not cfg['operation'].get('latest_inspection'):
            raise RuntimeError('먼저 --hand detached로 기준 영점을 확인하세요.')
    else:
        checks(cfg, args.side, args.hand)
    cfg_path = args.config.resolve()
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    core = [sys.executable, str(ROOT / 'scripts/ops/arm_zero_calibrate.py'), args.action,
            '--side', args.side, '--hand', args.hand, '--config', str(cfg_path)]
    for key in ('joint', 'candidate', 'history'):
        if getattr(args, key) is not None:
            core += ['--' + key, str(getattr(args, key))]
    # Execute offline planning first. Failure never starts a recorder or opens CAN.
    plan = 0 if args.action == 'inspect' else subprocess.call([sys.executable, str(ROOT / 'scripts/ops/arm_zero_calibrate.py'), 'plan',
                            '--side', args.side, '--hand', args.hand, '--config', str(cfg_path)]
                           + (['--for', args.action] if args.action in ('measure', 'apply', 'verify') else [])
                           + (['--joint', str(args.joint)] if args.joint is not None else []))
    if plan:
        # Only live-state absence is deferred. Other configuration errors stop here.
        from arm_zero_model import prepare
        _, report = prepare(cfg, args.side, args.hand,
                            command=args.action if args.action in ('measure', 'apply', 'verify') else None, joint=args.joint)
        if report['blockers'] or not report.get('requires_live_snapshot'):
            return plan
    print('실행할 명령:', ' '.join(core + ['--execute']))
    confirm('이 명령을 지금 실행할까요?')
    recorder = ROOT / 'deploy/policy_control/tools/rh56f1_record.sh'
    name = 'arm_zero_' + args.side + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    env = dict(os.environ, HAND_SIDES=args.side)
    bag_path = None
    status = 2
    errors = deque(maxlen=32)
    # Own only this unique recording name. Never stop any preexisting recorder.
    try:
        subprocess.run(['bash', str(recorder), 'start', name], env=env, check=True)
        if args.action == 'apply':
            cfg['operation']['pending_zero_run'] = name
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        child = subprocess.Popen(core + ['--execute'], stderr=subprocess.PIPE, text=True, bufsize=1)
        def forward_errors():
            assert child.stderr is not None
            for line in child.stderr:
                print(line, end='', file=sys.stderr, flush=True)
                errors.append(line)
        reader = threading.Thread(target=forward_errors, daemon=True)
        reader.start()
        while True:
            try:
                status = child.wait()
                reader.join()
                break
            except KeyboardInterrupt:
                # Let the controller execute its own disable/finally before ending
                # the recording. subprocess.call would kill it on interruption.
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                print('제어 도구의 정지·disable 정리를 기다립니다.', flush=True)
    finally:
        # Read the recorder's path before stop clears its last_out marker.
        marker = ROOT / 'logs/runs' / datetime.now().strftime('%y%m%d') / 'rh56f1_record' / name / 'last_out'
        if marker.exists():
            bag_path = Path(marker.read_text().strip())
        subprocess.run(['bash', str(recorder), 'stop', name], env=env, check=False)
        print('기록 이름:', name)
    if bag_path is not None:
        from arm_zero_bag import main as analyze
        paths = [bag_path]
        if args.history is not None:
            history = yaml.safe_load(args.history.read_text())
            paths.extend(Path(source['path']) for source in history.get('source_bags', []))
        if args.action == 'inspect' and args.hand == 'attached':
            baseline = yaml.safe_load((Path(cfg['operation']['latest_inspection']) / 'zero_inspection.yaml').read_text())
            paths.extend(Path(source['path']) for source in baseline['source_bags'])
        output = ROOT / 'logs/analysis/arm_zero' / name
        try:
            analysis_status = analyze([*[str(path) for path in paths], '--out', str(output),
                                       '--failure-reason', ''.join(errors).strip()])
        except (RuntimeError, ValueError, OSError) as exc:
            print('분석 산출물을 만들지 못했습니다. 원본 bag은 보존됩니다:', exc)
        else:
            if analysis_status:
                print('영점 확인 미완료: 기준 결과를 갱신하지 않았습니다.' if args.action == 'inspect' else '측정 미완료: 후보를 갱신하지 않았습니다. 위의 최초 중단 원인을 확인하세요.')
                return status or analysis_status
            if args.action == 'apply':
                # A failed write can leave an UNKNOWN epoch. Preserve its new
                # history too; never silently select the pre-write history.
                cfg['operation']['latest_analysis'] = str(output)
                cfg['operation'].pop('pending_zero_run', None)
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            if status:
                print('실행 중단: apply의 미확정 저장 이력을 포함해 확인하세요.'
                      if args.action == 'apply' else '실행 중단: 이전 기준 결과는 유지합니다.')
                return status
            if args.action == 'inspect':
                cfg['operation']['latest_inspection'] = str(output)
            else:
                cfg['operation']['latest_analysis'] = str(output)
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            print('영점 확인·전후 비교표 자동 생성:' if args.action == 'inspect' else '후보·부분 저장 이력·비교표 자동 생성:', output)
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('setup', 'run'))
    p.add_argument('--side', required=True, choices=('right', 'left'))
    p.add_argument('--hand', default='detached', choices=('detached', 'attached'))
    p.add_argument('--config', type=Path)
    p.add_argument('--action', choices=('inspect', 'measure', 'apply', 'verify'), default=None, help='기본: detached=measure, attached=verify')
    p.add_argument('--joint', type=int, choices=range(1, 8))
    p.add_argument('--candidate', type=Path)
    p.add_argument('--history', type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    args.action = args.action or ('measure' if args.hand == 'detached' else 'verify')
    if args.config is None:
        args.config = ROOT / f'config/local/arm_zero_{args.side}.yaml'
    try:
        return setup(args) if args.command == 'setup' else run(args)
    except (RuntimeError, ValueError, OSError, KeyboardInterrupt) as exc:
        print(f'중단: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
