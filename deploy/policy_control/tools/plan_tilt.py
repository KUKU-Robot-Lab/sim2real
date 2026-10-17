#!/usr/bin/env python3
"""tilt 기본 궤적을 지금 두 팔 자세에서 계획 · 검사한다 — 움직이지 않는다(관절 상태 구독만).

  실측(또는 --q-json) → 자산 policy.yaml 의 hdgp md5 · 기준 파일 md5 확인 → hdgp tilt_real_start.py(학습과 같은 계산)
  → 검사(IK 오차 · 관절 여유 · 명령 속도 · sim 출발 범위 · 탁자 여유) → <out>/tilt_cmd.npz · check.json
재생(실기, 사용자 승인 뒤):
  replay_to_pd.py --npz <out>/tilt_cmd.npz --joints r_aj_1,..,r_aj_7,l_aj_1,..,l_aj_7 --rate-scale 1.0 --max-start-gap 0.05 --execute
종료 코드: 0 = 재생해도 됨 · 2 = 검사 실패(재생 금지) · 3 = 핀 · 기준 파일 불일치 · 그 밖 = 계획 못 함.

    python3 deploy/policy_control/tools/plan_tilt.py --out logs/tilt/<run>
    python3 deploy/policy_control/tools/plan_tilt.py --q-json q.json --out /tmp/t      # 로봇 없이(오프라인 검사)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

SIM2REAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM2REAL / "deploy" / "policy_control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_control import hdgp_pin as H  # noqa: E402

ASSET = SIM2REAL / "deploy" / "policy_control" / "config" / "tilt" / "base_arc6"
TOOL = "scripts/tools/pour_bank/tilt_real_start.py"
ARMS = {"right": [f"r_aj_{i}" for i in range(1, 8)], "left": [f"l_aj_{i}" for i in range(1, 8)]}
JOINTS = ARMS["right"] + ARMS["left"]


def table_z(cfg: dict) -> float:
    src = SIM2REAL / str(cfg["table"]["from"])
    z = (yaml.safe_load(src.read_text(encoding="utf-8")) or {}).get("table_top_z")
    if not isinstance(z, (int, float)):
        raise H.PinError(f"{src}: table_top_z 가 없다")
    return float(z)


def current_q(q_json: str) -> dict[str, list[float]]:
    """{"right": [7], "left": [7]} — 파일이 있으면 그것, 없으면 /joint_states 실측(canonical 이름)."""
    if q_json:
        return json.loads(Path(q_json).read_text(encoding="utf-8"))
    from check_path_start import measure          # 실측 구독(sample_joints) — 관절이 다 들어올 때까지 기다린다
    q = measure(JOINTS)
    return {side: [float(q[j]) for j in names] for side, names in ARMS.items()}


def plan(asset: Path, out: Path, q: dict, hdgp: Path, python: str) -> tuple[int, dict]:
    cfg = H.load_policy(asset)
    H.verify(hdgp, cfg["hdgp"]["md5"])
    H.verify_refs(asset, cfg.get("ref") or {})
    out.mkdir(parents=True, exist_ok=True)
    (out / "q.json").write_text(json.dumps(q))
    argv = [python, str(hdgp / TOOL), "--q_json", str(out / "q.json"), "--ref_dir", str(asset / "ref"),
            "--table_z", repr(table_z(cfg)), "--out", str(out)]
    r = subprocess.run(argv, capture_output=True, text=True)
    (out / "plan_tilt.log").write_text(f"$ {' '.join(argv)}\n{r.stdout}\n{r.stderr}")
    if r.returncode not in (0, 2):
        raise RuntimeError(f"tilt_real_start rc {r.returncode}: {r.stderr.strip()[-400:]}")
    return r.returncode, json.loads((out / "check.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--asset", type=Path, default=ASSET)
    ap.add_argument("--q-json", default="", help='{"right": [7], "left": [7]} — 없으면 실측')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable, help="hdgp 도구를 돌릴 인터프리터(numpy · scipy · yaml)")
    a = ap.parse_args(argv)
    try:
        rc, chk = plan(a.asset, a.out, current_q(a.q_json), H.hdgp_dir(), a.python)
    except H.PinError as exc:
        print(f"✗ {exc}")
        return 3
    bad = {k: v["value"] for k, v in chk["checks"].items() if not v["ok"]}
    print(json.dumps({"ok": chk["ok"], "fail": bad, "T_end_s": chk["T_end_s"], "secs": chk["secs"],
                      "rcv_table_clear_m": chk["checks"].get("rcv_cup_table_clear_m", {}).get("value"),
                      "npz": str(a.out / "tilt_cmd.npz")}, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
