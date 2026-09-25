#!/usr/bin/env python3
"""실기 미션 → fake 미션. 단계 · 묶음 · 순서 · 승인은 그대로 두고, 실기에만 있는 것만 fake 로 바꾼다.

    python3 scripts/ops/make_fake_mission.py            # config/mission_dg5f_m_fake.yaml 을 다시 쓴다
    python3 scripts/ops/make_fake_mission.py --check    # 커밋된 파일이 지금 실기 미션에서 나온 것인지(테스트가 부른다)

왜 생성하나: 오른팔 · 오른손 · 왼팔 · 왼손이 이 프레임워크에서 끝까지 도는지 fake 플랜트로 먼저 확인한다(09.22 사용자).
fake 미션을 손으로 따로 두면 실기 미션을 고칠 때마다 어긋난다 — 확인한 것이 실기가 아니게 된다.

바꾸는 것(그 밖은 한 글자도 바꾸지 않는다):
  · robot yaml → *_fake, pd 설정 → pd_dg5f_m_short_fake.yaml (execute 는 launch 인자가 정한다: 무발행 단계 false)
  · drivers → fake 플랜트(양팔 MockArm, hands:=none) · hand_<팔> 의 손 드라이버 → 그 손만 띄우는 fake 플랜트
    (arm:=false, 손은 pd 의 드라이버 JTC 를 따르고 경로 기준 자세에서 조금 어긋나 시작)
  · 모든 ros2 launch 에 fake:=true (도메인 0 거부)
  · head_home → 목 하드웨어가 없다고 적는 echo (단계는 남긴다). vision-3090 인지는 진짜로 켠다(로봇이 아니다)
  · 정지 대상 drivers#<n> → drivers#0 (fake 플랜트)
  · touches_real 을 모두 뗀다 — fake 는 실기를 건드리지 않으므로 승인이 없다(단계 · 순서 · 수동 확인은 그대로)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "config" / "mission_dg5f_m_control.yaml"
FAKE = REPO / "config" / "mission_dg5f_m_fake.yaml"

ARTIFACTS = {
    "robot_right": "deploy/policy_control/config/robots/dg5f_m_right_fake.yaml",
    "robot_left": "deploy/policy_control/config/robots/dg5f_m_left_fake.yaml",
    "robot_bi": "deploy/policy_control/config/robots/dg5f_m_bi_fake.yaml",
    "pd": "deploy/policy_control/config/pd_dg5f_m_short_fake.yaml",
    "pd_exec": "deploy/policy_control/config/pd_dg5f_m_short_fake.yaml",
}
PLANT = {
    # plant_model rate: pd 모델은 팔 실측 캘리브(hdgp/log/logs/r2s_autotune/…/right_arm_best_calibration.json)의 마찰을
    # 요구하는데 그 파일이 이제 없다(로컬·서버 모두, 09.22). 값을 지어 넣지 않는다 — rate 는 배선 · 순서 · 도달만 본다.
    "note": "fake 플랜트 — 양팔 MockArm(rate: 배선 · 순서 · 도달만, 처짐 · 마찰 없음) + 컵 포즈. 손은 hand_<팔> 단계가 따로 띄운다",
    "argv": ["ros2", "launch", "{repo}/deploy/policy_control/launch/fake_plant.launch.py", "side:=both",
             "robot:={artifact:robot_bi}", "contract:={artifact:contract}", "pd_config:={artifact:pd}",
             "hands:=none", "plant_model:=rate"],
    "background": True,
}


def _hand(side: str) -> dict:
    """실기 손 드라이버 자리 — 그 손만 띄우는 fake(팔 브리지 없음)."""
    return {"note": f"fake 손({side}) — pd 의 드라이버 JTC 를 따른다, 경로 기준 자세에서 조금 어긋나 시작",
            "argv": ["ros2", "launch", "{repo}/deploy/policy_control/launch/fake_plant.launch.py", f"side:={side}",
                     "robot:={artifact:robot_bi}", "contract:={artifact:contract}", "pd_config:={artifact:pd}",
                     "arm:=false", "hand_follow:=jtc", "hand_start:=path"],
            "background": True}
# vision-3090(카메라 · FP++)은 로봇이 아니라 fake 에서도 진짜로 켠다(09.22 사용자: 언제든 쓸 수 있다). 목만 없다.
NO_HARDWARE = {"head_home": "fake — 목 하드웨어가 없다(실기에서는 기준자세 + 목 퍼블리셔)"}
HEADER = """# ★생성 파일 — 고치지 말 것. scripts/ops/make_fake_mission.py 가 config/mission_dg5f_m_control.yaml 에서 만든다.
#   실기 미션을 고친 뒤: python3 scripts/ops/make_fake_mission.py  (테스트가 --check 로 어긋남을 잡는다)
# 실기와 같은 단계 · 순서 · 승인을 fake 플랜트(도메인 97)에서 밟는다 — 오른팔 · 오른손 · 왼팔 · 왼손 확인용.
"""


def _fake_argv(argv: list) -> list:
    out = list(argv)
    if out[:2] == ["ros2", "launch"] and not any(str(a).startswith("fake:=") for a in out):
        out.append("fake:=true")
    return out


def _arm_start(artifacts: dict) -> list[str]:
    """fake 팔을 저장 홈 경로의 시작점(실측 차렷)에서 시작시킨다 — 실기와 같은 시작점 검사를 fake 에서도 밟는다."""
    import numpy as np

    parts = []
    for side in ("right", "left"):
        path = REPO / str(artifacts.get(f"path_{side}", ""))
        if path.is_file():
            q = np.load(path)["meta_start"]
            parts.append(f"{side}=" + ",".join(f"{float(v):.6f}" for v in q))
    return [f"arm_start:={';'.join(parts)}"] if parts else []


def convert(real: dict) -> dict:
    fake = yaml.safe_load(yaml.safe_dump(real, allow_unicode=True))          # 깊은 복사
    fake["name"] = f"{real['name']} (fake)"
    fake["artifacts"] = {**real["artifacts"], **ARTIFACTS}
    for st in fake["stages"]:                     # fake 프로파일은 실기 단계를 가리키지 않는다(test_console_profiles) — 승인도 없다
        st.pop("touches_real", None)
    run = fake["run"]
    if not any(c.get("background") for c in real["run"]["drivers"]):
        raise SystemExit("실기 drivers 에 배경 명령이 없다 — 변환 규칙을 다시 볼 것")
    run["drivers"] = [{**PLANT, "argv": PLANT["argv"] + _arm_start(fake["artifacts"])}]
    for side in ("right", "left"):                # 손 드라이버 명령만 fake 로 — 앞의 수동 확인은 그대로(키 번호가 실기와 같다)
        cmds = run[f"hand_{side}"]
        hits = [i for i, c in enumerate(cmds) if c.get("background") and "dg5f_driver" in c.get("argv", ())]
        if len(hits) != 1:
            raise SystemExit(f"실기 hand_{side} 에 손 드라이버 배경 명령이 하나가 아니다 — 변환 규칙을 다시 볼 것")
        cmds[hits[0]] = _hand(side)
    for stage, note in NO_HARDWARE.items():
        run[stage] = [{"note": note, "argv": ["echo", note]}]
    alive = {f"{st}#{i}" for st, cmds in run.items() for i, c in enumerate(cmds) if c.get("background")}
    for stage, cmds in run.items():
        for c in cmds:
            if c.get("stop"):
                keys = ("drivers#0" if k.startswith("drivers#") else k for k in c["stop"])
                c["stop"] = [k for k in dict.fromkeys(keys) if k in alive]     # fake 에 없는 프로세스(목 · 인지)는 뺀다
            elif c.get("argv") and stage != "drivers" and "fake_plant.launch.py" not in " ".join(c["argv"]):
                c["argv"] = _fake_argv(c["argv"])
                if any(str(a).endswith("check_path_start.py") for a in c["argv"]):
                    c["argv"].append("--allow-exact-zero")        # fake 팔은 정확히 0 에서 시작한다(엔코더 미수신 검사를 끈다)
        run[stage] = [c for c in cmds if not ("stop" in c and not c["stop"])]
    return fake


def render(real_path: Path = REAL) -> str:
    real = yaml.safe_load(real_path.read_text(encoding="utf-8"))
    return HEADER + yaml.safe_dump(convert(real), allow_unicode=True, sort_keys=False, width=120)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="쓰지 않고 커밋된 파일과 비교만")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        same = FAKE.exists() and FAKE.read_text(encoding="utf-8") == text
        print("일치" if same else f"{FAKE.name} 가 실기 미션과 어긋난다 — python3 scripts/ops/make_fake_mission.py")
        return 0 if same else 1
    FAKE.write_text(text, encoding="utf-8")
    print(f"썼다: {FAKE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
