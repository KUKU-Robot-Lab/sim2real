"""자식 프로세스 감독 — 새 세션으로 띄우고, 그룹째 정지하고, 출력을 파일로 받는다.

HTTP 계층은 **argv 를 받지 않는다**. argv 는 미션 yaml 의 runbook 에서만 나오고
(`mission_stages.commands_for`), 여기로는 이미 해석된 `Command` 가 온다. 셸을 거치지 않는다.

자식의 env 에는 프로파일의 `ROS_DOMAIN_ID` 를 **명시적으로** 넣는다 — 운영자가 엉뚱한 셸에서
콘솔을 띄웠어도 자식이 부모의 도메인을 물려받지 않는다.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

#: SIGTERM 뒤 SIGKILL 까지 기다리는 시간 [s]. mission_run.KILL_GRACE_S 와 같다.
KILL_GRACE_S = 5.0
KILL_REAP_S = 2.0            # SIGKILL 뒤에 그룹이 비기를 기다리는 시간


class SupervisorError(RuntimeError):
    pass


#: ros2 launch 가 자식이 죽었을 때 찍는 문구(launch/actions/execute_local.py).
CHILD_DIED = "process has died"
#: spawner 가 **이미 떠 있는** 컨트롤러에 다시 붙다가 죽을 때의 근거 문구(09.23 실기 DG-5F).
#: 이 죽음은 하드웨어가 정상이라는 뜻이다 — 실패로 보면 멀쩡한 드라이버를 다시 띄우게 된다.
SPAWNER_OK_REASONS = ("already loaded", "can not be configured from 'active' state")
_SPAWNER_CMD = re.compile(r"controller_manager/spawner ([^']*?) -c ")


def benign_spawner_death(line: str, log_lines: Sequence[str]) -> bool:
    """죽은 spawner 가 맡은 컨트롤러가 이미 로드·active 여서 난 죽음인가. 순수.

    죽음 줄의 cmd 에서 컨트롤러 이름을 뽑아, 같은 로그의 `[spawner_<이름>]` 줄에서 근거를 찾는다.
    이름을 하나라도 확인하지 못하면 진짜 죽음으로 본다(보수적)."""
    m = _SPAWNER_CMD.search(line)
    if m is None:
        return False
    names = [n for n in m.group(1).split() if not n.startswith("-")]
    if not names:
        return False
    return all(any(f"spawner_{n}" in ln and any(r in ln for r in SPAWNER_OK_REASONS) for ln in log_lines)
               for n in names)


def launch_target(argv) -> str | None:
    """`ros2 launch <pkg> <file>` / `ros2 launch <path>` 의 launch 파일 이름. launch 가 아니면 None."""
    argv = [str(a) for a in argv]
    if argv[:2] != ["ros2", "launch"]:
        return None
    files = [a for a in argv[2:] if a.endswith((".launch.py", ".launch.xml", ".launch.yaml"))]
    return os.path.basename(files[0]) if files else None


def foreign_launches(target: str, procs, own_pgids) -> list[tuple[int, str]]:
    """같은 launch 파일로 떠 있는 **콘솔 밖** 프로세스 (pid, cmdline). `procs` 는 (pid, pgid, cmdline) 들.

    09.22 실기: 3 시간 전 run 이 남긴 팔 브링업이 살아 있는데 새 콘솔이 브링업을 한 번 더 띄웠다 — 같은 CAN 에 둘이 붙었다.
    """
    out = []
    for pid, pgid, cmd in procs:
        if pgid in own_pgids:
            continue
        words = cmd.split()
        if "launch" in words and any(os.path.basename(w) == target for w in words):
            out.append((pid, cmd[:160]))
    return out


def scan_processes():
    """/proc 에서 (pid, pgid, cmdline). 읽을 수 없는 것은 건너뛴다."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            pid = int(d.name)
            yield pid, os.getpgid(pid), cmd
        except (OSError, ProcessLookupError):
            continue


def child_env(base: Mapping[str, str], *, domain: int, run_id: str) -> dict[str, str]:
    """부모 env 에 도메인과 run id 를 덮어쓴다. 순수."""
    env = dict(base)
    env["ROS_DOMAIN_ID"] = str(domain)
    env["S2R_RUN_ID"] = run_id
    env["PYTHONUNBUFFERED"] = "1"
    return env


@dataclass
class Proc:
    key: str               # "<stage>#<n>"
    stage: str
    note: str
    argv: tuple[str, ...]
    background: bool
    popen: subprocess.Popen
    log: Path
    started: float
    log_pos: int = 0       # 이 기동이 쓰기 시작한 로그 위치 — 앞의 줄은 지난 기동의 것이다(09.23)

    @property
    def rc(self) -> int | None:
        return self.popen.poll()

    def as_dict(self) -> dict:
        rc = self.rc
        return {"key": self.key, "stage": self.stage, "note": self.note, "argv": list(self.argv),
                "background": self.background, "pid": self.popen.pid, "rc": rc,
                "alive": rc is None, "log": self.log.name, "age_s": round(time.time() - self.started, 1)}


class Supervisor:
    def __init__(self, *, cwd: Path, run_dir: Path, env: Mapping[str, str]) -> None:
        self._cwd, self._run_dir, self._env = cwd, run_dir, dict(env)
        self._procs: dict[str, Proc] = {}
        self._lock = threading.Lock()
        (run_dir / "proc").mkdir(parents=True, exist_ok=True)

    # ── 띄우기 ──────────────────────────────────────────────────────────
    def spawn(self, key: str, *, stage: str, note: str, argv: Sequence[str], background: bool, manual: bool) -> Proc:
        if manual:
            raise SupervisorError("manual 명령은 콘솔이 실행하지 않는다 — 운영자가 다른 셸에서 한다")
        if not argv:
            raise SupervisorError("빈 argv")
        with self._lock:
            old = self._procs.get(key)
            if old is not None and old.rc is None:
                raise SupervisorError(f"{key} 가 이미 떠 있다 (pid {old.popen.pid})")
            log = self._run_dir / "proc" / f"{key.replace('#', '_')}.log"
            fh = log.open("ab")
            fh.write(f"\n$ {' '.join(argv)}\n".encode())
            fh.flush()
            log_pos = log.stat().st_size            # 같은 단계를 다시 띄우면 로그가 이어 쓰인다 — 여기서부터 본다
            try:
                popen = subprocess.Popen(list(argv), cwd=str(self._cwd), env=self._env, stdin=subprocess.DEVNULL,
                                         stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                fh.write(f"[console] 띄우지 못했다: {exc}\n".encode())
                fh.close()
                raise SupervisorError(f"{argv[0]}: {exc}") from exc
            fh.close()
            proc = Proc(key=key, stage=stage, note=note, argv=tuple(argv), background=background,
                        popen=popen, log=log, started=time.time(), log_pos=log_pos)
            self._procs[key] = proc
            self._write_pids()
            return proc

    # ── 보기 ────────────────────────────────────────────────────────────
    def table(self) -> list[dict]:
        with self._lock:
            return [p.as_dict() for p in self._procs.values()]

    def is_alive(self, key: str) -> bool:
        with self._lock:
            proc = self._procs.get(key)
            return proc is not None and proc.rc is None

    def rc(self, key: str) -> int | None:
        with self._lock:
            proc = self._procs.get(key)
        return None if proc is None else proc.rc

    def alive(self) -> list[Proc]:
        with self._lock:
            return [p for p in self._procs.values() if p.rc is None]

    def tail(self, key: str, n_bytes: int = 8192) -> str:
        with self._lock:
            proc = self._procs.get(key)
        if proc is None:
            raise SupervisorError(f"모르는 프로세스: {key}")
        with proc.log.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n_bytes))
            return fh.read().decode("utf-8", errors="replace")

    def own_pgids(self) -> set[int]:
        with self._lock:
            out = set()
            for p in self._procs.values():
                try:
                    out.add(os.getpgid(p.popen.pid))
                except ProcessLookupError:
                    continue
            return out

    def foreign_launches(self, argv) -> list[tuple[int, str]]:
        target = launch_target(argv)
        return [] if target is None else foreign_launches(target, scan_processes(), self.own_pgids())

    def child_deaths(self, key: str) -> list[str]:
        """이 프로세스의 로그에서 `ros2 launch` 가 알린 자식 죽음 줄. launch 는 자식이 죽어도 살아 있어서
        `is_alive` 만 보면 "떠 있음" 이다 — 09.22 fake 플랜트의 팔 브리지가 뜨자마자 죽었는데 drivers 가 완료로 넘어갔다."""
        with self._lock:
            proc = self._procs.get(key)
        if proc is None:
            return []
        with proc.log.open("rb") as fh:                    # 이번 기동이 쓴 부분만 — 지난 기동의 죽음을 다시 읽지 않는다
            fh.seek(min(proc.log_pos, proc.log.stat().st_size))
            lines = fh.read().decode("utf-8", "replace").splitlines()
        return [ln.strip() for ln in lines
                if CHILD_DIED in ln and not benign_spawner_death(ln, lines)]

    # ── 정지 ────────────────────────────────────────────────────────────
    def stop(self, keys: Sequence[str] | None = None) -> int:
        """그룹째 SIGTERM, 유예 뒤 SIGKILL. 정지시킨 수를 낸다."""
        with self._lock:
            targets = [p for k, p in self._procs.items() if (keys is None or k in keys) and p.rc is None]
        for p in targets:
            _signal_group(p.popen, signal.SIGTERM)
        # 리더(`ros2 launch`)가 먼저 끝나도 그룹에 노드가 남을 수 있다 — 그룹이 **빌 때까지** 기다리고, 안 비면 그룹째 죽인다.
        deadline = time.time() + KILL_GRACE_S
        for p in targets:
            if not _wait_group_empty(p.popen, deadline):
                _signal_group(p.popen, signal.SIGKILL)
                _wait_group_empty(p.popen, time.time() + KILL_REAP_S)
        with self._lock:
            self._write_pids()
        return len(targets)

    def _write_pids(self) -> None:
        """`mission_run.py --abort <run_id>` 가 읽는 것과 같은 파일·같은 형식."""
        live = [str(p.popen.pid) for p in self._procs.values() if p.rc is None]
        path = self._run_dir / "pids"
        if live:
            path.write_text("\n".join(live) + "\n")
        elif path.exists():
            path.unlink()


def _signal_group(popen: subprocess.Popen, sig: int) -> None:
    """`start_new_session=True` 로 띄웠으므로 그룹 id = 리더의 pid 다. 리더가 이미 거둬졌어도 그룹은 남아 있을 수 있다."""
    try:
        os.killpg(popen.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _group_alive(pgid: int) -> bool:
    """그룹에 좀비가 아닌 프로세스가 남아 있는가. 좀비(거두지 않은 리더)는 신호 0 에 답하므로 /proc 으로 가른다."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            fields = open(f"/proc/{entry.name}/stat", encoding="utf-8", errors="replace").read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
            return True
    return False


def _wait_group_empty(popen: subprocess.Popen, deadline: float) -> bool:
    while True:
        popen.poll()                                  # 리더를 거둔다 — 거두지 않으면 좀비로 남는다
        if not _group_alive(popen.pid):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.05)
