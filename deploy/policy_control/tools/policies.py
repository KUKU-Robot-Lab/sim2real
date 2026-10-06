#!/usr/bin/env python3
"""`sim2real/deploy/policies/` 에 **무엇이 등록돼 있고 쓸 수 있는 상태인가**를 본다.

    python3 deploy/policy_control/tools/policies.py                 # 목록 + 점검 (문제가 있으면 rc 1)
    python3 deploy/policy_control/tools/policies.py --write-index   # deploy/policies/INDEX.md 갱신 — 계열별 설명 · 학습 조건 차이 ·
                                                                    # sim/실기 개별 실행법(policy_index.py). 설명은 카드 summary · eval
    python3 deploy/policy_control/tools/policies.py --shallow       # 138 MB 체크포인트 재해시를 건너뛴다

정책 폴더는 `deploy/policies/<손>/<과제>/<팔>_<태그>/`(10.06) — 규칙을 벗어난 폴더도 문제로 센다.
등록은 `fetch_run.py`, 계약은 `build_deploy_contract.py`, 상태는 각 `policy.yaml` 을 사람이 고친다.
이 도구는 읽고 점검만 한다 — 규약은 `policy_control/policy_registry.py` 에 있다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM2REAL / "deploy" / "policy_control"))

from policy_control import policy_index as X  # noqa: E402
from policy_control import policy_registry as R  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--root", type=Path, default=SIM2REAL / "deploy" / "policies")
    ap.add_argument("--shallow", action="store_true", help="sha256 재해시 없이 존재·크기만 본다")
    ap.add_argument("--write-index", action="store_true", help=f"<root>/{R.INDEX} 를 다시 쓴다")
    args = ap.parse_args(argv)

    entries = R.scan(args.root, deep=not args.shallow)
    if not entries:
        print(f"[policies] {args.root} 에 등록된 정책이 없다")
        return 0
    bad = 0
    for e in entries:
        problems = (*e.issues, *R.layout_issues(e))
        bad += bool(problems)
        print(f"{'ok ' if not problems else '✗  '}{e.id:<28} {e.status:<10} {e.card.get('side', ''):<6} "
              f"{e.contract or '(계약 없음)':<22} {e.checkpoint or '-'}")
        for i in problems:
            print(f"     - {i}")
    if args.write_index:
        (args.root / R.INDEX).write_text(X.render(entries, repo=SIM2REAL))
        print(f"[policies] {args.root / R.INDEX} 갱신")
    print(f"[policies] {len(entries)} 개 · 문제 {bad} 개")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
