"""python -m vlm_supervisor entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .approach import ApproachJudge
from .backends import ROOT, Router
from .interface import RosInterface
from .judge import Judge
from .planner import Planner, simulate
from .sim_interface import SimInterface
from .supervisor import Supervisor


def build_parser():
    parser = argparse.ArgumentParser(description='독립 VLM 감독기 (기본: shadow)')
    parser.add_argument('--runner', default='episode_runner')
    parser.add_argument('--mode', choices=['shadow', 'active'], default='shadow')
    parser.add_argument('--sentence', required=True)
    parser.add_argument('--approvals', default='logs/runs/episode_approvals.jsonl')
    parser.add_argument('--log-dir', default=str(Path(ROOT) / 'runs'))
    parser.add_argument('--chain', help='쉼표로 구분한 백엔드 순서; 기본 VLM_CHAIN')
    parser.add_argument('--plan-fallback', default=str(
        Path(__file__).parent / 'config' / 'plan_cocktail_1c_e4000.json'))
    parser.add_argument('--settle', type=float, default=0.5)
    parser.add_argument('--interface', choices=['ros', 'sim'], default='ros')
    parser.add_argument('--approach-judge', action='store_true',
                        help='잡기(접근) 노드 직전에 VLM 으로 경로 방해물 판정(같은 백엔드)')
    parser.add_argument('--sim_url', default='http://127.0.0.1:8790', help='hdgp sim 브리지 주소(--interface sim)')
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
        if args.interface == 'sim':
            interface = SimInterface(args.sim_url, args.approvals)
        else:
            interface = RosInterface(args.runner, args.approvals)
        router = Router(args.chain.split(',') if args.chain else None)
        judge = Judge(router, args.sentence)
        supervisor = Supervisor(interface, judge, args.mode, run_dir / 'decisions.jsonl', args.settle,
                                approach_judge=ApproachJudge(router) if args.approach_judge else None)
        print(f'mode={args.mode} runner={args.runner} logs={run_dir}', flush=True)
        image = interface.frame(after=time.monotonic(), wait=5)
        if image is None:
            raise RuntimeError('카메라 프레임 없음')
        image.save(run_dir / 'start.jpg')
        planner = Planner(router)
        try:
            plan = planner.plan(args.sentence, image)
            plan['source'] = 'vlm'
        except ValueError as exc:
            event = {'event': 'plan_fallback', 'reason': str(exc),
                     'vlm_answer_head': planner.last_answer[:400]}
            supervisor.log(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            steps = json.loads(Path(args.plan_fallback).read_text(encoding='utf-8'))
            result = simulate(steps, planner.cards)
            if not result['ok']:
                raise ValueError('invalid fallback plan: ' + '; '.join(result['errors']))
            plan = {'sentence': args.sentence, 'steps': steps, 'simulation': result,
                    'source': 'fallback'}
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
