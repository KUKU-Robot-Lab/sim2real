#!/usr/bin/env python3
"""rg3 잡기 에피소드(config/episodes/grasp_rg3_<side>.yaml)의 정책을 학습 런 체크포인트로 교체한다 — 한 명령, 순서 고정.
★10.14 s2r_1st: 위치제어(hand_adm_enable false) 재학습 aglt_{r,l}_rg3w 를 다음 날 실기에 올리려고 리허설한 순서 그대로다.
리허설에서 잡힌 사고: (1) 계약을 만들기 전에 미션을 만들면 contract:= 에 폴더 경로가 들어갔다(지금은 생성기가 거부)
(2) tests/policy_control/test_pc_episode_grasp_rg3.py 가 정책 id 를 고정해 둔다 — 같이 바꾼다.

    python3 scripts/ops/swap_rg3_policy.py --side right --host server \\
        --run ~/rl_ws/hdgp_grasp_w/log/rl_games/open-rh/right/aglt/aglt_r_rg3w --checkpoint last_..._ep_N_rew_R.pth \\
        --tag right_rg3w --expect position            # 계획만 출력
    ... --execute                                     # 받기 · 카드 · 계약 · 에피소드 · 미션 · INDEX · 로드 점검 · 시험

커밋은 하지 않는다(사용자 · 세션이 자기 파일만 git add). .pth 는 .gitignore — 실기 PC 에 이 명령으로 직접 받는다.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
POL = REPO / "deploy" / "policies"
PY = sys.executable                     # 이 스크립트를 돌린 파이썬(venv) 그대로 — 워크트리에는 .venv 가 없다
EPISODE = REPO / "config" / "episodes" / "grasp_rg3_{side}.yaml"
PIN_TEST = REPO / "tests" / "policy_control" / "test_pc_episode_grasp_rg3.py"
ROLE = {"right": "rh_aglt_r", "left": "rh_aglt_l"}
TESTS = ["test_pc_rh_aglt.py", "test_pc_policy_index.py", "test_pc_episode_grasp_rg3.py", "test_pc_episode.py",
         "test_pc_grip_assist.py", "test_pc_hand_block.py", "test_pc_rh56f1_ecat.py", "test_pc_rh56f1_pos_guard.py",
         "test_pc_pd_backends.py"]


def run(cmd: list[str], **kw) -> str:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def swap_episode(text: str, side: str, new: str) -> str:
    """에피소드 yaml 의 rh_aglt_<s> 줄 policy 만 바꾼다(주석 · 다른 줄은 그대로)."""
    pat = re.compile(rf"^(\s*{ROLE[side]}:\s*\{{policy:\s*)rh56f1/aglt/[\w.-]+", re.M)
    out, n = pat.subn(rf"\g<1>rh56f1/aglt/{new}", text)
    if n != 1:
        raise SystemExit(f"{ROLE[side]} 정책 줄이 {n} 개 — 손으로 본다")
    return out


def swap_pin(text: str, side: str, new: str) -> str:
    pat = re.compile(rf'("{side}":\s*")rh56f1/aglt/[\w.-]+(")')
    out, n = pat.subn(rf"\g<1>rh56f1/aglt/{new}\g<2>", text, count=1)
    if n != 1:
        raise SystemExit(f"{PIN_TEST.name} 의 POLICY[{side}] 를 못 찾았다")
    return out


def write_index(new_id: str) -> None:
    """INDEX.md 는 레지스트리 전체가 깨끗할 때만 다시 쓴다. .pth 는 git 밖이라 새 워크트리 · 다른 PC 에서는 남의 정책이
    '체크포인트 없음'으로 보이고, 그대로 쓰면 INDEX 가 오염된다(10.14 리허설). 새 정책 자체가 깨졌으면 멈춘다."""
    p = subprocess.run([PY, "deploy/policy_control/tools/policies.py", "--shallow"], cwd=REPO, text=True, capture_output=True)
    bad = [ln.split()[1] for ln in p.stdout.splitlines() if ln.startswith("✗") and len(ln.split()) > 1]
    if new_id in bad:
        raise SystemExit(f"새 정책 {new_id} 가 레지스트리 점검에 걸렸다:\n{p.stdout[-2000:]}")
    if bad:
        print(f"[swap] INDEX.md 는 그대로 둔다 — 다른 정책 {len(bad)} 개가 이 트리에서 점검 실패(.pth 없음 등). "
              f"체크포인트가 다 있는 트리에서 policies.py --write-index --shallow")
        return
    run([PY, "deploy/policy_control/tools/policies.py", "--write-index", "--shallow"], cwd=REPO)


def card(template: dict, a: argparse.Namespace, mode: str) -> dict:
    c = dict(template)
    c.update(id=f"rh56f1/aglt/{a.tag}", status="candidate", checkpoint=a.checkpoint,
             source=f"{a.host}:{a.run}/nn/{a.checkpoint}",
             summary=f"{template.get('summary', '')} · ★손 = {mode}(10.14 위치제어 재학습)".replace("어드민턴스 · ", ""))
    c["note"] = f"swap_rg3_policy.py 로 등록(템플릿 {template.get('id')}). 학습 env.yaml 덤프 그대로."
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--host", required=True, help="ssh 호스트(학습 런이 있는 곳)")
    ap.add_argument("--run", required=True, help="원격 런 폴더(params/ nn/)")
    ap.add_argument("--checkpoint", required=True, help="nn/ 의 .pth 이름")
    ap.add_argument("--tag", required=True, help="deploy/policies/rh56f1/aglt/<tag>")
    ap.add_argument("--expect", choices=("position", "admittance"), required=True, help="계약 hand_command 가 이것이 아니면 멈춘다")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    ep_path = Path(str(EPISODE).format(side=a.side))
    hit = re.search(rf"{ROLE[a.side]}:\s*\{{policy:\s*rh56f1/aglt/([\w.-]+)", ep_path.read_text())
    if hit is None:
        raise SystemExit(f"{ep_path.name} 에 {ROLE[a.side]} 정책 줄이 없다")
    old = hit.group(1)
    dst = POL / "rh56f1" / "aglt" / a.tag
    print(f"[swap] {a.side}: {old} → {a.tag} ({a.host}:{a.run}/nn/{a.checkpoint}) · 기대 손 = {a.expect}")
    if not a.execute:
        print("계획만 — --execute 로 실행")
        return 0
    if dst.exists():
        raise SystemExit(f"{dst} 가 이미 있다 — 다른 --tag 를 쓰거나 손으로 정리")
    (dst / "nn").mkdir(parents=True)
    (dst / "reward").mkdir()
    run(["scp", "-q", "-r", f"{a.host}:{a.run}/params", str(dst)])
    run(["scp", "-q", f"{a.host}:{a.run}/nn/{a.checkpoint}", str(dst / "nn")])
    m = re.search(r"^reward_code_path:\s*(\S+)", (dst / "params" / "env.yaml").read_text(), re.M)
    if not m:
        raise SystemExit("env.yaml 에 reward_code_path 가 없다")
    run(["scp", "-q", f"{a.host}:{m.group(1)}", str(dst / "reward" / "compute_reward.py")])
    template = yaml.safe_load((POL / "rh56f1" / "aglt" / old / "policy.yaml").read_text())
    (dst / "policy.yaml").write_text(yaml.safe_dump(card(template, a, a.expect), allow_unicode=True, sort_keys=False))
    run([PY, "deploy/policy_control/tools/build_rh_aglt_contract.py", "--run", str(dst)], cwd=REPO)   # 계약 → 그다음 미션
    sys.path.insert(0, str(REPO / "deploy" / "policy_control"))
    from policy_control import rh_aglt as A
    c = A.load_contract(dst / "rh_aglt_contract.json")
    if c.hand_command != a.expect:
        raise SystemExit(f"계약 hand_command={c.hand_command} ≠ --expect {a.expect} — 학습 env hand_adm_enable 확인")
    ep_path.write_text(swap_episode(ep_path.read_text(), a.side, a.tag))
    PIN_TEST.write_text(swap_pin(PIN_TEST.read_text(), a.side, a.tag))
    run([PY, "scripts/ops/make_rh56f1_missions.py"], cwd=REPO)
    write_index(f"rh56f1/aglt/{a.tag}")
    check = ("import sys, numpy as np; sys.path[:0]=['deploy/policy_control','tests/policy_control']\n"
             "import test_pc_rh_aglt as T; from policy_control import rh_aglt as A, pour_fj_node as N\n"
             "from policy_control.joint_policy import JointPolicy\n"
             f"c=A.load_contract('{dst}/rh_aglt_contract.json')\n"
             "for o in (c.object_names or [None]):\n"
             "    ch=N.AgltChain(c, JointPolicy(c,'cpu')); ch.object_name=o; m={'arm':T._meas(c)}; ch.reset(m)\n"
             "    for _ in range(12): obs,act,_t=ch.step(m)\n"
             "    assert act.size==13 and np.all(np.isfinite(act)) and np.all(np.isfinite(obs)), o\n"
             "print('로드 · 12 스텝 ok', list(c.object_names), c.hand_command)\n")
    print(run([PY, "-c", check], cwd=REPO).strip().splitlines()[-1])
    out = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=line",
                          *[f"tests/policy_control/{t}" for t in TESTS]], cwd=REPO, text=True, capture_output=True,
                         env={**__import__("os").environ, "PYTHONPATH": "deploy/policy_control:tests/policy_control"})
    print(out.stdout.strip().splitlines()[-1])
    if out.returncode:
        print(out.stdout[-3000:])
        raise SystemExit("시험 실패 — 위를 본다")
    print(f"[swap] 끝. 바뀐 파일: {ep_path.relative_to(REPO)} · {PIN_TEST.relative_to(REPO)} · config/mission_rh56f1_*.yaml · "
          f"(INDEX.md 는 위 메시지대로) · {dst.relative_to(REPO)}/ (커밋은 직접)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
