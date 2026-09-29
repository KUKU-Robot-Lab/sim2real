#!/usr/bin/env python3
"""새 PC 점검 — git pull · 빌드 뒤 이 PC 가 local5090 과 같은 배포 환경인가. **읽기만** 한다(설치 · 수정 없음).

09.29 사용자: "git pull 하고 build 하면 5090 과 동일하게 세팅할 수 있는 거지?" — git 에 없는 것(가중치 · venv · 빌드 ·
PC 별 설정)과 저장소 배치를 한 번에 본다. 고칠 방법은 INSTALL.md 의 Step 번호로 알려 준다.

    python3 scripts/setup/check_host.py                     # 전부
    python3 scripts/setup/check_host.py --robot rh56f1      # RH56F1 로봇 PC(arm4090) 기준
    python3 scripts/setup/check_host.py --robot dg5f --fetch  # 원격과 비교(git fetch — 네트워크)

rc 0 = MISS 없음(WARN 은 있을 수 있다), 1 = MISS 있음.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
REPOS = {"sim2real": "main", "robot_control": "humble", "hdgp": "main", "urdf": None}
#: 5090 기준 버전(2026-09-29). 다르면 WARN — 학습 · 재생 결과가 달라질 수 있다
VENV_PACKAGES = {"torch": "2.7.1+cu128", "rl-games": "1.6.1", "mujoco": "2.3.0", "numpy": "1.26.4",
                 "scipy": "1.13.1", "trimesh": None, "PyYAML": None}   # 배포(dist) 이름
ROBOT_CONTROL_PKGS = {"common": ["openarm_bringup", "openarm_hardware", "openarm_description"],
                      "dg5f": ["dg5f_driver", "delto_hardware"],
                      "rh56f1": ["rh56f1_driver", "rh56f1_interfaces", "inspire_control_ros2"]}
MISSIONS = {"dg5f": ["config/mission_dg5f_m_control.yaml"], "rh56f1": ["config/mission_rh56f1_control.yaml"]}
HDGP_FILES = ["assets/robot/openarm_dg5f-m-short_bi_rl/openarm_dg5f-m-short_bi_rl.urdf",
              "assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf",
              "assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl_manifest.yaml",
              "assets/simulation_setting/env_v1/usd/env_v1.usda",
              "source/openarm/openarm/agnostic/modules/robot_profiles.py",
              "source/openarm/openarm/agnostic/tasks/pour_fabric_mimic/bimanual.py",
              "source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/config/__init__.py"]


@dataclass
class Report:
    rows: list = field(default_factory=list)

    def ok(self, what: str) -> None:
        self.rows.append(("OK", what, ""))

    def warn(self, what: str, fix: str = "") -> None:
        self.rows.append(("WARN", what, fix))

    def miss(self, what: str, fix: str) -> None:
        self.rows.append(("MISS", what, fix))

    def section(self, title: str) -> None:
        self.rows.append(("==", title, ""))

    def print(self) -> int:
        for kind, what, fix in self.rows:
            if kind == "==":
                print(f"\n== {what} ==")
                continue
            print(f"  [{kind:4}] {what}" + (f"\n         → {fix}" if fix else ""))
        n = {k: sum(1 for r in self.rows if r[0] == k) for k in ("OK", "WARN", "MISS")}
        print(f"\n요약: OK {n['OK']} · WARN {n['WARN']} · MISS {n['MISS']}")
        return 1 if n["MISS"] else 0


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else ""


def check_repos(rep: Report, fetch: bool) -> None:
    rep.section("저장소 배치 · 동기화 (~/rl_ws 에 나란히)")
    for name, branch in REPOS.items():
        repo = RL_WS / name
        if not (repo / ".git").exists():
            rep.miss(f"{repo} 없음(git 저장소)", "INSTALL.md Step 3 — 네 저장소를 ~/rl_ws 에 나란히 clone")
            continue
        if fetch:
            _git(repo, "fetch", "-q")
        cur = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if branch and cur != branch:
            rep.warn(f"{name}: 브랜치 {cur} (5090 은 {branch})", f"git -C {repo} checkout {branch}")
        head = _git(repo, "status", "-sb").splitlines()[:1]
        state = head[0] if head else "?"
        if "behind" in state:
            rep.miss(f"{name}: 원격보다 뒤 — {state}", f"git -C {repo} pull")
        elif "ahead" in state:
            rep.warn(f"{name}: 원격보다 앞(push 안 한 커밋) — {state}")
        else:
            rep.ok(f"{name} {_git(repo, 'rev-parse', '--short', 'HEAD')} ({state}{', fetch 안 함' if not fetch else ''})")
    for rel in HDGP_FILES:
        if not (RL_WS / "hdgp" / rel).exists():
            rep.miss(f"hdgp/{rel} 없음", "hdgp 를 pull — sim2real 은 이 자산 · 모듈을 읽는다")
    gains = RL_WS / "urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml"
    (rep.ok if gains.exists() else lambda w: rep.miss(w, "urdf 저장소 pull"))(f"팔 게인 {gains.relative_to(RL_WS)}")


def check_venv(rep: Report) -> None:
    rep.section("sim2real/.venv (정책 추론 · 계획 · 테스트)")
    py = SIM2REAL / ".venv/bin/python"
    if not py.exists():
        rep.miss(".venv 없음", "INSTALL.md Step 4 — python3 -m venv --system-site-packages .venv 뒤 패키지 설치")
        return
    cfg = (SIM2REAL / ".venv/pyvenv.cfg").read_text()
    if "include-system-site-packages = true" not in cfg:
        rep.miss(".venv 가 시스템 패키지를 못 본다(rclpy 가 안 보인다)", "venv 를 --system-site-packages 로 다시 만든다")
    code = ("import json\nfrom importlib import metadata\nout={}\n"
            f"for m in {list(VENV_PACKAGES)!r}:\n"
            "  try:\n    out[m]=metadata.version(m)\n"
            "  except Exception: out[m]=None\n"
            "try:\n  import torch; out['_cuda']=bool(torch.cuda.is_available())\nexcept Exception: out['_cuda']=None\n"
            "print(json.dumps(out))")
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120)
    try:
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        rep.miss(".venv 파이썬을 실행하지 못했다", r.stderr.strip()[-200:])
        return
    for mod, want in VENV_PACKAGES.items():
        have = got.get(mod)
        if have is None:
            rep.miss(f".venv 에 {mod} 없음", f"{py} -m pip install {mod}"
                     + (f"=={want.split('+')[0]}" if want else ""))
        elif want and have != want:
            rep.warn(f".venv {mod} {have} (5090 은 {want})")
        else:
            rep.ok(f".venv {mod} {have}")
    if got.get("_cuda") is False:
        rep.warn("torch 가 CUDA 를 못 본다 — 정책 노드는 device:=cpu 로 돈다(미션 기본)")


def check_builds(rep: Report, robot: str) -> None:
    rep.section("빌드 (colcon)")
    link = SIM2REAL / "build/policy_control/policy_control"
    target = SIM2REAL / "deploy/policy_control/policy_control"
    if not (SIM2REAL / "install/policy_control").exists():
        rep.miss("sim2real policy_control 빌드 없음",
                 "INSTALL.md Step 3 — colcon build --packages-select policy_control --base-paths deploy --symlink-install")
    elif not link.exists() or link.resolve() != target.resolve():
        rep.miss("policy_control 이 복사 설치다(--symlink-install 아님) — pd_node 가 import 에서 죽는다",
                 "rm -rf build/policy_control install/policy_control 뒤 --symlink-install 로 다시")
    else:
        rep.ok("policy_control symlink 빌드")
    ws = RL_WS / "robot_control/ros_ws/install"
    if not (ws / "setup.bash").exists():
        rep.miss("robot_control ros_ws 빌드 없음", "INSTALL.md Step 3 — robot_control/ros_ws/build.sh")
        return
    want = ROBOT_CONTROL_PKGS["common"] + (ROBOT_CONTROL_PKGS[robot] if robot in ROBOT_CONTROL_PKGS else
                                           ROBOT_CONTROL_PKGS["dg5f"] + ROBOT_CONTROL_PKGS["rh56f1"])
    for pkg in want:
        (rep.ok if (ws / pkg).exists() else lambda w: rep.miss(w, "robot_control/ros_ws/build.sh (RH56F1 은 "
                                                               "INSTALL.md Step 3-A 의 apt 의존성 먼저)"))(
            f"robot_control 패키지 {pkg}")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_policies(rep: Report, source_host: str) -> None:
    rep.section("정책 가중치 deploy/policies/*/nn (git 에 없다 — .gitignore)")
    sys.path.insert(0, str(SIM2REAL / "deploy/policy_control"))
    from policy_control import policy_registry as R   # noqa: PLC0415
    for e in R.scan(SIM2REAL / "deploy/policies", deep=False):
        if e.status == "hold":
            continue
        name = e.card.get("checkpoint") or ""
        pth = e.path / "nn" / str(name)
        if not name or not pth.exists():
            rep.miss(f"{e.id}: nn/{name or '?'} 없음",
                     f"rsync -av {source_host}:~/rl_ws/sim2real/deploy/policies/{e.id}/nn/ {e.path}/nn/")
            continue
        md5 = ""
        if e.contract:
            try:
                md5 = R.contract_checkpoint_md5(json.loads((e.path / e.contract).read_text()))
            except (OSError, ValueError):
                md5 = ""
        if md5 and _md5(pth) != md5:
            rep.miss(f"{e.id}: nn/{name} md5 가 계약과 다르다", "5090 의 파일을 다시 복사")
        else:
            rep.ok(f"{e.id}: nn/{name}" + (" · md5 = 계약" if md5 else " · 계약 없음(md5 대조 없음)"))


def check_missions(rep: Report, robot: str) -> None:
    rep.section("미션 산출물(실기 미션이 가리키는 파일)")
    import yaml   # noqa: PLC0415
    names = MISSIONS.get(robot) or [m for v in MISSIONS.values() for m in v]
    for rel in names:
        arts = (yaml.safe_load((SIM2REAL / rel).read_text()) or {}).get("artifacts") or {}
        for key, path in arts.items():
            p = SIM2REAL / path
            (rep.ok if p.exists() else lambda w: rep.miss(w, "git pull (계약 · 경로는 git 에 있다)"))(
                f"{Path(rel).stem}: {key} → {path}")
        for key in [k for k in arts if k.startswith("path_")]:
            import numpy as np   # noqa: PLC0415
            d = np.load(SIM2REAL / arts[key])
            want = str(d["meta_contract_sha1"])
            have = hashlib.sha1((SIM2REAL / arts["contract"]).read_bytes()).hexdigest()
            if want != have:
                rep.miss(f"{Path(rel).stem}: {key} 를 만든 계약 ≠ 지금 계약", "홈 경로를 다시 계획(plan_home_path.py)")


def check_host_specific(rep: Report, robot: str) -> None:
    rep.section("이 PC 에만 있는 것")
    dom = os.environ.get("ROS_DOMAIN_ID", "")
    (rep.ok if dom == "126" else lambda w: rep.warn(w, "실기는 export ROS_DOMAIN_ID=126 (fake 는 콘솔이 97 로 띄운다)"))(
        f"ROS_DOMAIN_ID={dom or '(없음)'}")
    if robot in ("dg5f", "rh56f1", "all"):
        for can in ("can0", "can1"):
            (rep.ok if Path(f"/sys/class/net/{can}").exists() else
             lambda w: rep.warn(w, "USB-CAN 연결 · 이름 확인(미션 drivers 단계가 can0 · can1 을 쓴다)"))(f"CAN {can}")
    if robot in ("rh56f1", "all"):
        import yaml   # noqa: PLC0415
        ports = yaml.safe_load((SIM2REAL / "deploy/policy_control/config/rh56f1_ports.yaml").read_text())
        for side in ("right", "left"):
            port = ports[side]["port"]
            (rep.ok if Path(port).exists() else
             lambda w: rep.warn(w, "손 연결 확인 후 config/rh56f1_ports.yaml 의 port 를 이 PC 값으로(ls -l /dev/serial/by-id)"))(
                f"RH56F1 {side} {ports[side]['transport']} {port}")
    if robot in ("dg5f", "all"):
        rep.warn("DG-5F 손 네트워크(/32 경로)는 미션 hand_<side> 단계의 hand_net_dual.sh --apply 로(최초 1회)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", choices=("all", "dg5f", "rh56f1"), default="all", help="이 PC 에 붙은 로봇")
    ap.add_argument("--fetch", action="store_true", help="git fetch 로 원격과 비교(네트워크)")
    ap.add_argument("--source-host", default="<5090 PC>", help="가중치를 받아 올 PC(rsync 안내에 쓴다)")
    args = ap.parse_args(argv)
    rep = Report()
    check_repos(rep, args.fetch)
    check_venv(rep)
    check_builds(rep, args.robot)
    check_policies(rep, args.source_host)
    check_missions(rep, args.robot)
    check_host_specific(rep, args.robot)
    rc = rep.print()
    print("다음: python3 -m pytest tests -q -m 'not gpu' (저장소 루트, .venv) — 실패 0 이어야 실기")
    return rc


if __name__ == "__main__":
    sys.exit(main())
