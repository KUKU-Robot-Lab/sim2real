#!/usr/bin/env python3
"""RH56F1 한 팔 aglt(hdgp open-rh_{r,l}_aglt) 런 → rh_aglt 계약(policy_control/rh_aglt.py). 09.30 사용자: RH56F1 정책 deploy 연결.

    python3 tools/build_rh_aglt_contract.py --run deploy/policies/rh56f1/aglt/right_i03 [--checkpoint nn/<pth>] [--out …]
    python3 tools/build_rh_aglt_contract.py --run deploy/policies/rh56f1/place/right_i09    # rh_place 런 → rh_place_contract.json(10.04)

손 관측 순서는 문제가 없다 — rh_aglt 는 손 관절을 이름(프로필 순)으로 찾는다. 왼팔은 hdgp tasks/rh_aglt_l/profile.py 와
같은 규칙(이름 r_ → l_, 손 값 그대로)으로 RH56F1_RIGHT 에서 만든다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402
from policy_control import rh_aglt as A  # noqa: E402
from policy_control import rh_place as P  # noqa: E402
from policy_control import rh56f1_ecat as E  # noqa: E402
from policy_control import rh56f1_map  # noqa: E402
from policy_control.pour_profiles import load_profile  # noqa: E402

ASSET = "openarm_rh56f1_bi_rl"
URDF = _paths.RL_WS / "hdgp" / "assets" / "robot" / ASSET / f"{ASSET}.urdf"


def _checkpoint(run: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else run / p
    found = sorted((run / "nn").glob("*.pth"))
    if len(found) != 1:
        raise SystemExit(f"{run}/nn 에 .pth 가 {len(found)} 개 — --checkpoint 로 고를 것")
    return found[0]


PORTS = Path(__file__).resolve().parents[1] / "config" / "rh56f1_ports.yaml"


def real_admittance() -> tuple[dict, float]:
    """실기 드라이버가 쓸 어드민턴스(rh56f1_ports.yaml ethercat + 정본 robot_control 계약) · 네 손가락 레지스터/rad(변환표 index_1)."""
    import yaml
    eth = yaml.safe_load(PORTS.read_text())["ethercat"]
    adm = E.with_contract(eth).get("admittance") or {}
    if not adm or not bool(adm.get("enabled", True)):
        raise SystemExit(f"{PORTS}: 실기 어드민턴스가 없다(정본 계약 control.admittance)")
    hmap = rh56f1_map.load(rh56f1_map.DEFAULT_PATH)
    ax = hmap.axes[hmap.joint_order.index("index_1")]
    return adm, abs(ax.reg[1] - ax.reg[0]) / abs(ax.rad[1] - ax.rad[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="params/{env,agent}.yaml 이 있는 런 폴더")
    ap.add_argument("--checkpoint", help="nn/ 에 하나가 아니면 명시")
    ap.add_argument("--hdgp", default=str(_paths.RL_WS / "hdgp"))
    ap.add_argument("--out", help="기본 <run>/rh_aglt_contract.json")
    args = ap.parse_args(argv)
    run = Path(args.run).resolve()
    place = P.is_rh_place_run(A.read_env(run / "params" / "env.yaml"))
    c = (P.build if place else A.build)(run, _checkpoint(run, args.checkpoint), load_profile(Path(args.hdgp), "rh56f1_right"),
                                        URDF, asset=ASSET)
    if c.hand_command == "admittance":          # ★10.08 학습 어드민턴스 명목값 = 실기 드라이버 값이어야 한다
        real, reg_per_rad = real_admittance()
        bad = A.admittance_mismatch(c.hand_admittance, real, reg_per_rad)
        if bad:
            raise SystemExit("학습 어드민턴스가 실기 드라이버와 다르다 — " + " · ".join(bad))
        c = A.with_run(c, notes=[*c.notes, f"실기 드라이버 어드민턴스(정본 robot_control components/rh56f1.yaml · 손가락 "
                                             f"{reg_per_rad:.1f} 칸/rad)와 명목값 같음(2 % 안)"])
    out = Path(args.out) if args.out else run / ("rh_place_contract.json" if place else "rh_aglt_contract.json")
    out.write_text(c.to_json())
    s = c.side()
    print(f"[rh_aglt] {c.task} · {s.side} · obs {c.obs_dim} / act {c.action_dim} · {c.policy_hz:.0f} Hz · "
          f"hold {c.hold_steps} · 목표 +{c.goal_offset} → {out}")
    for n in c.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
