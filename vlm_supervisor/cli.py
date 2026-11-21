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
from .place_path import PlacePathJudge
from .plan_map import apply_plan_order
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
        Path(__file__).parent / 'config' / 'plan_cocktail_e2e.json'))
    parser.add_argument('--settle', type=float, default=0.5)
    parser.add_argument('--interface', choices=['ros', 'sim'], default='ros')
    parser.add_argument('--approach-judge', action='store_true',
                        help='잡기(접근) 노드 직전에 VLM 으로 경로 방해물 판정(같은 백엔드)')
    parser.add_argument('--place-path-judge', action='store_true',
                        help='놓기 노드 직전에 VLM 이 후보 경로 번호를 고른다(기록만, 실행 동작은 그대로; 같은 백엔드)')
    parser.add_argument('--judge_blind', action='store_true',
                        help='judge 프롬프트에서 runner 판정(code·reason·runner_detail 등)을 빼고 사진만으로 판정')
    parser.add_argument('--plan_order', choices=['off', 'shadow', 'apply'], default='off',
                        help='VLM 계획 순서를 실행 순서로: shadow=매핑만 로그, apply=시작 전 shell reorder(fallback 계획은 적용 안 함)')
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
        judge = Judge(router, args.sentence, blind=args.judge_blind)
        supervisor = Supervisor(interface, judge, args.mode, run_dir / 'decisions.jsonl', args.settle,
                                approach_judge=ApproachJudge(router) if args.approach_judge else None,
                                place_path_judge=PlacePathJudge(router) if args.place_path_judge else None)
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
        keep_going, executed = apply_plan_order(args.plan_order, plan, interface, supervisor.log, args.mode)
        if not keep_going:
            return 2
        # shadow/off judge exactly as e5cdb75; apply follows what the runner executes.
        judge.plan = plan['steps']
        if args.plan_order == 'apply':
            if executed is not None:
                judge.plan = executed
            else:
                try:
                    judge.plan = json.loads(Path(args.plan_fallback).read_text(encoding='utf-8'))
                except (OSError, ValueError) as exc:
                    supervisor.log({'event': 'judge_plan_fallback_unreadable', 'reason': repr(exc)})
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
