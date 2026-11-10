#!/usr/bin/env python3
"""에피소드 정의 점검 · 모의 실행 — 로봇 없이(ROS 없이). 실기 · fake 실행은 episode_runner_node + episode_cmd.py(상황판).

    python3 deploy/policy_control/tools/episode_run.py --episode config/episodes/pick_place_right.yaml --plan
    python3 deploy/policy_control/tools/episode_run.py --episode … --dry-run                        # 정상 경로
    python3 deploy/policy_control/tools/episode_run.py --episode … --dry-run --inject pick_cup=grasp_failed_right
    python3 deploy/policy_control/tools/episode_run.py --episode … --dry-run --step                 # 노드마다(구분 실행 모양)
    python3 deploy/policy_control/tools/episode_run.py --episode … --dry-run --delay grasp_shaker=2 --jsonl <경로>   # ★11.08 lanes 시각

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
from policy_control.episode_runner import FAILURE, SUCCESS, EpisodeManager, jsonl_logger  # noqa: E402


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
    ap.add_argument("--delay", action="append", default=[], help="노드=초 — 모의 정책 · 궤적이 걸리는 시간(lanes 순서 보기)")
    ap.add_argument("--jsonl", type=Path, help="모의 실행 기록을 실기 실행기와 같은 jsonl 로도 쓴다")
    ap.add_argument("--require-holders", action="store_true",
                    help="--plan: 고정 홀더 자세 파일에 쓰는 홀더가 다 있고 마지막 홀더 재기(<파일>.last_check.json)가 "
                         "written · 6 h 안이어야 rc 0(실기 에피소드 단계 시작 검사)")
    args = ap.parse_args(argv)
    path = args.episode if args.episode.is_absolute() else _paths.SIM2REAL / args.episode
    ep = S.load(path)
    if args.plan:
        entries = {e.id: e for e in R.scan(_paths.SIM2REAL / "deploy" / "policies", deep=False)}
        av = S.availability(ep, entries)
        print(f"[episode] {ep.name} v{ep.version} · 노드 {len(ep.nodes)} · 물체 {list(ep.objects)} · 홀더 {dict(ep.holders)}")
        for n in ep.nodes:
            what = n.name if n.type in ("trajectory",) else " + ".join(
                f"{j.role}→{ep.policies[j.role].policy or '없음'}" for j in n.jobs) or n.result or (
                f"{n.action} {','.join(n.sides)}" if n.type == "hand" else n.text[:40])
            print(f"  {n.id:<24} {n.type:<16} {what}  {json.dumps(dict(n.expect), ensure_ascii=False) if n.expect else ''}"
                  + (f"  [checkpoint {n.checkpoint}]" if n.checkpoint else ""))
        bad = {r: why for r, why in av.items() if why}
        for r, why in av.items():
            b = ep.policies[r]
            print(f"  {'✓' if not why else '✗'} {r}: {why or b.policy}"
                  + (f" → 넘김 {b.handoff}(경로 D)" if b.handoff and not why else ""))
        holders, notes = [], []
        if args.require_holders:
            from policy_control.episode_ros import load_holder_poses
            hp = Path(ep.holder_poses)
            poses = load_holder_poses(hp if hp.is_absolute() else _paths.SIM2REAL / hp)
            holders = S.holder_problems(ep, poses)
            if not holders and ep.holders:         # ★11.11 deploy 리뷰 H1 — 마지막 홀더 재기가 written 이고 낡지 않았나
                import time
                cp = Path(str(hp if hp.is_absolute() else _paths.SIM2REAL / hp) + ".last_check.json")
                try:
                    check = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else None
                except (OSError, ValueError) as e:
                    check = {"result": f"읽기 실패({type(e).__name__})"}
                holders = S.holder_check_problems(ep, check, time.time())
            if not holders:                        # ★10.28 놓기 홀더가 계약 학습 범위 안인가 — 놓기 노드 reset 과 같은 검사를 미리
                from policy_control import rh_place as P
                root = _paths.SIM2REAL / "deploy" / "policies"
                contracts = {r: P.load_contract(root / b.policy / (getattr(entries.get(b.policy), "contract", "")
                                                                   or "rh_place_contract.json"))
                             for r, b in ep.policies.items() if b.kind == "place" and b.policy and not av.get(r)}
                holders, notes = S.place_holder_checks(ep, poses, contracts, P.HOLDER_TOL_M)
            for why in holders:
                print(f"  ✗ 홀더: {why}")
            for why in notes:                      # ★10.29 카드 holder_tol_m 로 받은 범위 밖 홀더 — 막지 않고 보인다
                print(f"  ! 홀더: {why}")
            if not holders and ep.holders:
                print(f"  ✓ 홀더 {dict(ep.holders)} — {ep.holder_poses}")
        hands = S.hand_problems(ep, _paths.SIM2REAL / "deploy" / "policies")      # ★10.17 경로 B 손 쥐기 계약
        for nid, why in hands.items():
            print(f"  {'✓' if not why else '✗'} 손 {nid}: {why or 'hand_contract 있음'}")
        return 1 if bad or holders or any(hands.values()) else 0
    ex = FakeExecutor(ep, inject=_inject(args.inject),
                      delay={k: float(v) for k, _, v in (d.partition("=") for d in args.delay)})
    rows: list[dict] = []
    write = jsonl_logger(args.jsonl) if args.jsonl else (lambda row: None)
    m = EpisodeManager(ep, ex, approve=lambda what, why: True, log=lambda row: (rows.append(row), write(row)),
                       episode_id="dry-run")
    if args.step:
        while m.status not in (SUCCESS, FAILURE):
            m.step()
    else:
        m.run()
    for r in rows:
        print(f"[{r['timestamp'][11:23]}] {r['event']:<17} {str(r['state_id']):<24} {r.get('lane', '') or '':<6}"
              f"{r.get('result', '') or ''} {r.get('termination_reason', '') or ''}".rstrip())
    print(f"[episode] {m.status} · {json.dumps(m.world.as_dict(), ensure_ascii=False)}")
    return 0 if m.status == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
