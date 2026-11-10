"""python -m vlm_supervisor entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .backends import ROOT, Router
from .interface import RosInterface
from .judge import Judge
from .planner import Planner
from .supervisor import Supervisor


def build_parser():
    parser = argparse.ArgumentParser(description='독립 VLM 감독기 (기본: shadow)')
    parser.add_argument('--runner', default='episode_runner')
    parser.add_argument('--mode', choices=['shadow', 'active'], default='shadow')
    parser.add_argument('--sentence', required=True)
    parser.add_argument('--approvals', default='logs/runs/episode_approvals.jsonl')
    parser.add_argument('--log-dir', default=str(Path(ROOT) / 'runs'))
    parser.add_argument('--chain', help='쉼표로 구분한 백엔드 순서; 기본 VLM_CHAIN')
    parser.add_argument('--settle', type=float, default=0.5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.settle < 0:
        raise SystemExit('--settle must be nonnegative')
    interface = None
    supervisor = None
    run_dir = Path(args.log_dir) / (time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns() % 1000000:06d}')
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        interface = RosInterface(args.runner, args.approvals)
        router = Router(args.chain.split(',') if args.chain else None)
        judge = Judge(router, args.sentence)
        supervisor = Supervisor(interface, judge, args.mode, run_dir / 'decisions.jsonl', args.settle)
        print(f'mode={args.mode} runner={args.runner} logs={run_dir}', flush=True)
        image = interface.frame(after=time.monotonic(), wait=5)
        if image is None:
            raise RuntimeError('카메라 프레임 없음')
        image.save(run_dir / 'start.jpg')
        plan = Planner(router).plan(args.sentence, image)
        (run_dir / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        judge.plan = plan['steps']
        return supervisor.run()
    except KeyboardInterrupt:
        if interface:
            interface.perform('stop', active=args.mode == 'active')
        return 130
    except Exception as exc:
        event: dict = {'event': 'startup_error', 'reason': str(exc)}
        if interface:
            event['result'] = interface.perform('stop', active=args.mode == 'active')
        if supervisor:
            supervisor.log(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        return 2
    finally:
        if interface:
            interface.close()
