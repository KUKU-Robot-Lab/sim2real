#!/usr/bin/env python3
"""에피소드 정의 점검 · 모의 실행 — 로봇 없이(ROS 없이). 실기 · fake 실행은 episode_runner_node + episode_cmd.py(상황판).

    python3 deploy/policy_control/tools/episode_run.py --episode config/episodes/pick_place_right.yaml --plan
    python3 deploy/policy_control/tools/episode_run.py --episode … --dry-run                        # 정상 경로
    python3 deploy/policy_control/tools/episode_run.py --episode … --dry-run --inject pick_cup=grasp_failed_right
    python3 deploy/policy_control/tools/episode_run.py --episode … --dry-run --step                 # 노드마다(구분 실행 모양)

--plan: 순서 · 각 노드가 부를 정책(등록부) · 실기에서 못 쓰는 이유(availability). rc 1 = 실기 불가 역할이 있다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402
from policy_control import episode_spec as S  # noqa: E402
from policy_control import policy_registry as R  # noqa: E402
from policy_control.episode_fake import FakeExecutor  # noqa: E402
from policy_control.episode_runner import FAILURE, SUCCESS, EpisodeManager  # noqa: E402


def _inject(items: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for it in items:
        node, _, codes = it.partition("=")
        out.setdefault(node, []).extend(c for c in codes.split(",") if c)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--episode", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inject", action="append", default=[], help="노드=실패코드[,실패코드] (모의 실패)")
    ap.add_argument("--step", action="store_true", help="모의 실행을 노드 하나씩(구분 실행 모양)")
    ap.add_argument("--require-holders", action="store_true",
                    help="--plan: 고정 홀더 자세 파일에 쓰는 홀더가 다 있어야 rc 0(실기 에피소드 단계 시작 검사)")
    args = ap.parse_args(argv)
    path = args.episode if args.episode.is_absolute() else _paths.SIM2REAL / args.episode
    ep = S.load(path)
    if args.plan:
        entries = {e.id: e for e in R.scan(_paths.SIM2REAL / "deploy" / "policies", deep=False)}
        av = S.availability(ep, entries)
        print(f"[episode] {ep.name} v{ep.version} · 노드 {len(ep.nodes)} · 물체 {list(ep.objects)} · 홀더 {dict(ep.holders)}")
        for n in ep.nodes:
            what = n.name if n.type in ("trajectory",) else " + ".join(
                f"{j.role}→{ep.policies[j.role].policy or '없음'}" for j in n.jobs) or n.result or ""
            print(f"  {n.id:<24} {n.type:<16} {what}  {json.dumps(dict(n.expect), ensure_ascii=False) if n.expect else ''}"
                  + (f"  [checkpoint {n.checkpoint}]" if n.checkpoint else ""))
        bad = {r: why for r, why in av.items() if why}
        for r, why in av.items():
            print(f"  {'✓' if not why else '✗'} {r}: {why or ep.policies[r].policy}")
        holders = []
        if args.require_holders:
            from policy_control.episode_ros import load_holder_poses
            hp = Path(ep.holder_poses)
            holders = S.holder_problems(ep, load_holder_poses(hp if hp.is_absolute() else _paths.SIM2REAL / hp))
            for why in holders:
                print(f"  ✗ 홀더: {why}")
            if not holders and ep.holders:
                print(f"  ✓ 홀더 {dict(ep.holders)} — {ep.holder_poses}")
        return 1 if bad or holders else 0
    ex = FakeExecutor(ep, inject=_inject(args.inject))
    rows: list[dict] = []
    m = EpisodeManager(ep, ex, approve=lambda what, why: True, log=rows.append, episode_id="dry-run")
    if args.step:
        while m.status not in (SUCCESS, FAILURE):
            m.step()
    else:
        m.run()
    for r in rows:
        print(f"[{r['timestamp'][11:23]}] {r['event']:<17} {str(r['state_id']):<24} "
              f"{r.get('result', '') or ''} {r.get('termination_reason', '') or ''}".rstrip())
    print(f"[episode] {m.status} · {json.dumps(m.world.as_dict(), ensure_ascii=False)}")
    return 0 if m.status == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
