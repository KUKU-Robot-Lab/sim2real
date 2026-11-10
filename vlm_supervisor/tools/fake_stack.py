"""fake 미션 단계를 이어서 띄워 두는 시험용 런처 — 감독기 · shell 통합 시험 전용.

mission_run CLI 는 단계가 끝나면 배경 자식을 정지한다(여러 단계를 들고 있는 것은 콘솔뿐). 감독기 시험에는
드라이버 · pd · 홈 · 에피소드 노드가 함께 살아 있어야 하고, 에피소드 runner 는 vlm_supervisor.shell 이
소유해야 한다. 그래서 미션 정의(commands_for)를 그대로 읽어 단계 순서대로 실행하고, --omit 에 걸리는
명령(기본: episode_runner_node.py)만 빼고 배경 자식을 SIGTERM 까지 들고 있는다. deploy 코드는 고치지 않는다.

    ROS_DOMAIN_ID=97 python -m vlm_supervisor.tools.fake_stack --repo ~/rl_ws/sim2real \
        --mission config/mission_rh56f1_fake.yaml --stages cups drivers hand_right ... episode_cocktail_1c_e4000
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def plan(commands_by_stage: list[tuple[str, tuple]], omit: list[str]) -> list[tuple[str, Any]]:
    """(stage, commands) → 실행할 (key, command) 목록. 수동 · omit 은 뺀다.

    stop 은 노드 교체용(수동 명령 앞)만 남긴다. 단계 안에서 수동 명령 뒤에 오는 stop 은 그 단계의 정리라
    (예: 에피소드 단계의 [사람이 실행 → 끝나면 노드 정지]) 여기서 돌리면 shell 의 runner 가 쓸 노드가 READY 전에
    내려간다. 그런 stop 은 빼고 종료 때 한꺼번에 내린다."""
    out = []
    for stage, cmds in commands_by_stage:
        after_manual = False
        for i, cmd in enumerate(cmds):
            if cmd.manual:
                after_manual = True
                continue
            if cmd.stop and after_manual:
                continue
            if not cmd.stop and any(o in " ".join(cmd.argv) for o in omit):
                continue
            out.append((f"{stage}#{i}", cmd))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path("~/rl_ws/sim2real").expanduser())
    ap.add_argument("--mission", required=True)
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--omit", nargs="*", default=["episode_runner_node.py"])
    ap.add_argument("--pids", type=Path, help="배경 자식 pid 를 적을 파일")
    a = ap.parse_args(argv)
    if os.environ.get("ROS_DOMAIN_ID") in (None, "", "0", "50"):
        print("ROS_DOMAIN_ID 가 fake 도메인이 아니다(0 · 50 · 없음 거부)")
        return 2
    repo = a.repo.resolve()
    sys.path.insert(0, str(repo / "scripts" / "ops"))
    sys.path.insert(0, str(repo / "scripts"))
    import mission_run as MR  # type: ignore[import-not-found]  # noqa: E402 — 미션 정의를 deploy 와 같은 코드로 읽는다

    mission, runbook = MR._load(repo / a.mission)
    steps = plan([(s, MR.commands_for(runbook, mission, s, repo=repo, execute=True)) for s in a.stages], a.omit)
    alive: dict[str, subprocess.Popen] = {}

    stuck: dict[str, subprocess.Popen] = {}

    def stop(keys) -> None:
        """SIGTERM 뒤 10 s 안에 안 내려간 자식은 SIGKILL 하지 않고 stuck 에 남겨 pid 파일에 적는다."""
        for k in keys:
            p = alive.pop(k, None)
            if p is None or p.poll() is not None:
                continue
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                print(f"! {k} pid {p.pid} 가 SIGTERM 10 s 뒤에도 살아 있다", flush=True)
                stuck[k] = p

    def write_pids() -> None:
        if a.pids:
            a.pids.write_text("".join(f"{p.pid} {k}\n" for k, p in {**alive, **stuck}.items() if p.poll() is None))

    def on_term(*_) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)  # 기동 도중 SIGTERM 도 finally 정리를 거친다
    try:
        for key, cmd in steps:
            if cmd.stop:
                print(f"■ {key} 정지 {cmd.stop}", flush=True)
                stop(cmd.stop)
                continue
            print(f"▶ {key} {'[배경] ' if cmd.background else ''}{cmd.note[:100]}", flush=True)
            p = subprocess.Popen(cmd.argv, cwd=str(repo), start_new_session=True)
            if cmd.background:
                alive[key] = p
                write_pids()
                time.sleep(1.0)
                continue
            if p.wait() != 0:
                print(f"✗ {key} 실패 rc={p.returncode}", flush=True)
                return 1
        print("READY — 배경 자식", len(alive), "개를 들고 있다(SIGTERM 으로 정지)", flush=True)
        while True:
            dead = [k for k, p in alive.items() if p.poll() is not None]
            for k in dead:
                print(f"! {k} 종료 rc={alive.pop(k).returncode}", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop(list(reversed(list(alive))))
        write_pids()
    return 1 if stuck else 0


if __name__ == "__main__":
    sys.exit(main())
