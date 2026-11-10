"""Run an owned runner and its episode shell."""
import argparse
import re
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--run-dir', default='logs/vlm_shell')
    parser.add_argument('--ready-timeout', type=float, default=30)
    parser.add_argument('--runner-cmd', nargs=argparse.REMAINDER,
                        default=[sys.executable, str(Path(__file__).resolve().parents[2] /
                                                       'deploy/policy_control/policy_control/episode_runner_node.py')])
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.runner):
        parser.error('runner must be a ROS name segment')
    if not args.runner_cmd or args.ready_timeout <= 0:
        parser.error('runner-cmd must be nonempty and ready-timeout positive')
    from .shell_node import run_node
    run_node(args)
    return 0
