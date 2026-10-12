#!/usr/bin/env python3
"""RH56F1 오른팔 캡 단계 런(hdgp open-rh_r_cap, cap_phase grasp · seat) → rh_cap 계약(policy_control/rh_cap.py). 10.12 사용자 간단 배포.

    python3 tools/build_rh_cap_contract.py --run deploy/policies/rh56f1/cap/right_capg_j04 [--checkpoint nn/<pth>] [--out …]

캡 기하는 학습과 같은 hdgp assets/cyl_cap/<cap_asset_dir>/cap_c<N>.json 에서 읽고, 관측 폭은 체크포인트에서 확인한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402
from policy_control import rh_cap as C  # noqa: E402
from policy_control.checkpoint_dims import checkpoint_contract  # noqa: E402
from policy_control.pour_profiles import load_profile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rh_aglt_contract import ASSET, URDF, _checkpoint, real_admittance  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="params/{env,agent}.yaml 이 있는 런 폴더")
    ap.add_argument("--checkpoint", help="nn/ 에 하나가 아니면 명시")
    ap.add_argument("--hdgp", default=str(_paths.RL_WS / "hdgp"))
    ap.add_argument("--out", help="기본 <run>/rh_cap_contract.json")
    ap.add_argument("--skip-admittance-check", action="store_true", help="실기 드라이버 설정 파일이 없는 호스트(fake)")
    args = ap.parse_args(argv)
    run = Path(args.run).resolve()
    ck = _checkpoint(run, args.checkpoint)
    obs, act = checkpoint_contract(ck)
    if act != 13:
        raise SystemExit(f"{ck}: 행동 {act} ≠ 13")
    c = C.build(run, ck, load_profile(Path(args.hdgp), "rh56f1_right"), URDF, asset=ASSET, hdgp_root=Path(args.hdgp),
                obs_dim=obs)
    if c.hand_command == "admittance" and not args.skip_admittance_check:
        real, reg_per_rad = real_admittance()
        bad = A.admittance_mismatch(c.hand_admittance, real, reg_per_rad)
        if bad:
            raise SystemExit("학습 어드민턴스가 실기 드라이버와 다르다 — " + " · ".join(bad))
        c = A.with_run(c, notes=[*c.notes, f"실기 드라이버 어드민턴스와 명목값 같음(손가락 {reg_per_rad:.1f} 칸/rad, 2 % 안)"])
    out = Path(args.out) if args.out else run / "rh_cap_contract.json"
    out.write_text(c.to_json())
    print(f"[rh_cap] {c.task} · obs {c.obs_dim} / act {c.action_dim} · {c.policy_hz:.0f} Hz · {c.episode_s} s → {out}")
    for n in c.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
