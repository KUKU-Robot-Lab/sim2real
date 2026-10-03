"""CPU 배치 — 이 PC 의 코어를 그 자리에서 읽어 실시간 프로세스 자리와 torch 스레드 수를 정한다.

10.03 사용자: "sim2real 은 CPU 최적화가 자동이어야 한다 — 어떤 PC 에서 세팅할지 모른다." 그래서 코어 번호를
박지 않는다. sysfs(물리 코어 · SMT 형제 · online · isolated)만 읽는 순수 함수 + 얇은 적용부.

  · 실시간 역할(EtherCAT 마스터 오른/왼, 1 kHz)마다 물리 코어 하나. isolcpus 로 떼어 둔 코어가 있으면 그것부터,
    없으면 번호가 큰 코어부터. cpu0 이 든 코어는 쓰지 않는다(부팅 · IRQ 기본 자리).
    마스터는 그 코어의 첫 스레드에 꽂고, 형제 스레드까지 '예약'으로 비켜 둔다 — 우리 노드는 예약 밖(general)에서 돈다.
  · 물리 코어가 역할 수 + 4 보다 적으면 고정하지 않는다(작은 VM · 노트북 — 고정하면 오히려 굶는다).
    손이 없는 PC(DG-5F)도 역할 자리 두 코어를 비켜 두지만, 비켜 두는 것은 우리 노드뿐이라 다른 프로그램은 그대로 쓴다.
  · torch 스레드: POLICY_CPU_THREADS 가 양의 정수면 그것. 아니면 1(09.28: 코어 전부 47 ms · 2 개 4 ms,
    10.04: 1 개 4.2 ms 에 CPU 는 2 개의 절반).
  · 끄기: S2R_CPU_PIN=0 (고정만 끈다. torch 스레드 수는 그대로 정한다).

실시간 우선순위(SCHED_FIFO) 자체는 여기서 주지 않는다 — 마스터 · controller_manager 가 스스로 요청하고, 그것이
되려면 PC 의 RT 한도가 열려 있어야 한다(scripts/setup/rt_setup.sh, 점검은 check_host.py --only cpu).

    python3 -m policy_control.cpu_plan          # 이 PC 의 배치를 한 줄로
    python3 policy_control/cpu_plan.py --general   # 일반 코어 cpulist("0-13,16-29") — fpp_up.sh 가 컨테이너를 여기에 묶는다
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

SYSFS = Path("/sys/devices/system/cpu")
RT_ROLES = ("ecat_right", "ecat_left")
TORCH_THREADS = 1                # ★10.04 실측(rh_aglt LSTM 1024, CPU 추론, 같은 부하): 2 개 CPU 65 % · 3.7 ms/스텝 →
                                 #   1 개 31.6 % · 4.2 ms — 2 개는 OpenMP 스레드가 연산 사이에 바쁘게 기다려 코어를 더 쓴다
PIN_ENV = "S2R_CPU_PIN"
THREADS_ENV = "POLICY_CPU_THREADS"
MIN_GENERAL_CORES = 4                    # 실시간 코어를 떼고도 우리 노드(pd 3 스레드 · 정책 · 상태 · 노드)에 남길 물리 코어
RT_PRIO_NEEDED = 80                      # rh56f1_ecat_master 가 요청하는 SCHED_FIFO 순위 (controller_manager 50)
_SCHED = {os.SCHED_OTHER: "OTHER", os.SCHED_FIFO: "FIFO", os.SCHED_RR: "RR",
          getattr(os, "SCHED_BATCH", -1): "BATCH", getattr(os, "SCHED_IDLE", -2): "IDLE"}


def parse_cpulist(text: str) -> tuple[int, ...]:
    """커널 cpulist("0-3,8,10-11") → (0, 1, 2, 3, 8, 10, 11)."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(sorted(set(out)))


def format_cpulist(cpus) -> str:
    """(0, 1, 2, 3, 8, 10, 11) → "0-3,8,10-11" (커널 · docker --cpuset-cpus 형식). 순수."""
    out, run = [], []
    for c in sorted(set(int(c) for c in cpus)):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


@dataclass(frozen=True)
class Topology:
    cores: tuple[tuple[int, ...], ...]          # 물리 코어마다 켜져 있는 논리 CPU (첫 CPU 순)
    isolated: frozenset[int] = frozenset()

    @property
    def cpus(self) -> tuple[int, ...]:
        return tuple(sorted(c for core in self.cores for c in core))


def read_topology(sysfs: Path = SYSFS) -> Topology:
    """sysfs 만 읽는다 — 프로세스 affinity 는 보지 않으므로 이미 고정된 노드가 물어도 같은 답이 나온다."""
    online = parse_cpulist(_read(sysfs / "online")) or tuple(range(os.cpu_count() or 1))
    groups: dict[tuple[int, ...], list[int]] = {}
    for cpu in online:
        sib = parse_cpulist(_read(sysfs / f"cpu{cpu}" / "topology" / "thread_siblings_list")) or (cpu,)
        groups.setdefault(sib, []).append(cpu)
    cores = tuple(sorted(tuple(sorted(v)) for v in groups.values()))
    return Topology(cores, frozenset(parse_cpulist(_read(sysfs / "isolated"))))


@dataclass(frozen=True)
class CpuPlan:
    rt: Mapping[str, int] = field(default_factory=dict)     # 역할 → 논리 CPU 하나 (비면 고정 안 함)
    reserved: tuple[int, ...] = ()                          # 실시간 코어(형제 포함) — 우리 노드는 비켜 간다
    general: tuple[int, ...] = ()                           # 우리 노드 자리
    torch_threads: int = TORCH_THREADS
    note: str = ""

    def describe(self) -> str:
        where = (" · ".join(f"{r}→cpu{c}" for r, c in self.rt.items()) + " (형제 스레드까지 비켜 둠)"
                 if self.rt else "고정 안 함")
        return f"{self.note} — {where} · 노드 {len(self.general)} 스레드 자리 · torch {self.torch_threads}"


def _torch_threads(general_cores: int, env: Mapping[str, str]) -> int:
    raw = env.get(THREADS_ENV, "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return min(TORCH_THREADS, max(1, general_cores - 1))


def make_plan(topo: Topology, roles: Iterable[str] = RT_ROLES, env: Mapping[str, str] | None = None) -> CpuPlan:
    env = os.environ if env is None else env
    roles = tuple(roles)
    cores = topo.cores
    usable = tuple(c for c in cores if not set(c) <= topo.isolated)     # isolated 코어는 우리 노드 자리가 아니다
    head = f"물리 {len(cores)} · 논리 {len(topo.cpus)}"

    def unpinned(why: str) -> CpuPlan:
        general = tuple(sorted(c for core in usable for c in core))
        return CpuPlan(general=general, torch_threads=_torch_threads(len(usable), env), note=f"{head} · {why}")

    if env.get(PIN_ENV, "1") == "0":
        return unpinned(f"{PIN_ENV}=0 — 고정 안 함")
    iso = [c for c in cores if set(c) <= topo.isolated and 0 not in c]
    rest = [c for c in cores if c not in iso and 0 not in c]
    cand = iso[::-1] + rest[::-1]                                      # 떼어 둔 코어 → 큰 번호 순
    if len(cores) < len(roles) + MIN_GENERAL_CORES or len(cand) < len(roles):
        return unpinned(f"물리 코어가 적어(< {len(roles) + MIN_GENERAL_CORES}) 고정 안 함")
    picked = cand[:len(roles)]
    general_cores = [c for c in usable if c not in picked]
    return CpuPlan(rt={r: core[0] for r, core in zip(roles, picked)},
                   reserved=tuple(sorted(c for core in picked for c in core)),
                   general=tuple(sorted(c for core in general_cores for c in core)),
                   torch_threads=_torch_threads(len(general_cores), env), note=head)


def current_plan(sysfs: Path = SYSFS) -> CpuPlan:
    """노드가 부르는 입구 — sysfs 가 이상해도 예외 대신 '고정 안 함' 계획을 돌려준다(최적화일 뿐, 못 해도 노드는 돈다)."""
    try:
        return make_plan(read_topology(sysfs))
    except Exception as e:                                     # noqa: BLE001 — 파싱 · 권한 무엇이든
        return CpuPlan(general=tuple(sorted(os.sched_getaffinity(0))), note=f"코어 읽기 실패({type(e).__name__}: {e}) — 고정 안 함")


# ── 적용부(부작용) — 실패는 예외 대신 이유 문자열로 돌려준다: 고정은 최적화일 뿐, 못 해도 노드는 돈다 ─────────

def _threads(pid: int) -> list[int]:
    try:
        return sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return [pid]


def pin(pid: int, cpus: Iterable[int]) -> str | None:
    """pid 의 스레드 전부를 cpus 에 고정한다(뒤에 생기는 스레드는 물려받는다). None = 됐다."""
    mask = set(cpus)
    if not mask:
        return "빈 CPU 목록"
    try:
        for tid in _threads(pid):
            os.sched_setaffinity(tid, mask)
    except (OSError, OverflowError, ValueError) as e:
        return f"{e.strerror or e} (cpu {sorted(mask)[:4]}…)" if len(mask) > 4 else f"{e.strerror or e} (cpu {sorted(mask)})"
    return None


def pinned_argv(argv: list[str], role: str, plan: CpuPlan | None = None) -> tuple[list[str], str]:
    """실시간 역할 프로세스(EtherCAT 마스터)를 계획된 코어에서 **시작**시키는 argv 와 로그 한 줄.

    ★뜬 뒤에 고정하면 안 된다 — 마스터는 setcap(cap_net_raw) 실행 파일이라 권한 없는 부모가 sched_setaffinity 를 걸면
    EPERM 이다(commoncap: 대상의 권한이 더 크면 거부, 10.03 /usr/bin/ping 으로 확인). taskset 이 제 affinity 를 정한 뒤
    exec 하면 affinity 가 exec 를 넘어 이어지고 pid 도 그대로다(PR_SET_PDEATHSIG 도 마스터가 exec 뒤에 건다).
    """
    plan = plan or current_plan()
    cpu = plan.rt.get(role)
    if cpu is None:
        return list(argv), f"CPU: {role} 고정 안 함 — {plan.note}"
    taskset = shutil.which("taskset")
    if taskset is None:
        return list(argv), f"CPU: {role} 고정 안 함 — taskset 없음(util-linux)"
    # 감싼 taskset 이 실패하면 마스터가 exec 되지 않는다(cpuset 이 걸린 컨테이너 · 오프라인 코어) — 먼저 한 번 시험
    try:
        trial = subprocess.run([taskset, "-c", str(cpu), "true"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as e:
        return list(argv), f"CPU: {role} 고정 안 함 — taskset 시험 실패({e})"
    if trial.returncode != 0:
        why = (trial.stderr or trial.stdout).strip().splitlines()[-1:] or [f"rc {trial.returncode}"]
        return list(argv), f"CPU: {role} 고정 안 함 — cpu{cpu} 를 못 잡는다({why[0]})"
    return [taskset, "-c", str(cpu), *argv], f"CPU: {role} → cpu{cpu}"


def keep_off_rt(plan: CpuPlan | None = None) -> str:
    """이 프로세스를 일반 코어로 — 노드 main 첫머리에서 부른다. 로그에 쓸 한 줄을 돌려준다(예외는 안 낸다)."""
    plan = plan or current_plan()
    if not plan.rt:
        return f"CPU: {plan.note}"
    why = pin(os.getpid(), plan.general)
    return f"CPU: 일반 코어 {len(plan.general)} 스레드에서 돈다" + (f" — 고정 실패({why})" if why else "")


def sched_name(pid: int) -> str:
    try:
        return _SCHED.get(os.sched_getscheduler(pid), "?")
    except OSError:
        return "?"


def rt_limit() -> int:
    """이 프로세스가 올릴 수 있는 SCHED_FIFO 최대 순위(RLIMIT_RTPRIO soft). 0 = 실시간 못 씀."""
    return int(resource.getrlimit(resource.RLIMIT_RTPRIO)[0])


def memlock_limit() -> int:
    """mlockall 한도(바이트). -1 = 무제한."""
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return -1 if soft == resource.RLIM_INFINITY else int(soft)


def governors(sysfs: Path = SYSFS) -> tuple[str, ...]:
    """CPU 마다의 주파수 governor (cpufreq 가 없는 VM 은 빈 튜플)."""
    return tuple(g for g in (_read(p).strip() for p in sorted(sysfs.glob("cpu[0-9]*/cpufreq/scaling_governor"))) if g)


if __name__ == "__main__":
    import sys
    plan = current_plan()
    if "--general" in sys.argv[1:]:
        # 셸 · 컨테이너용: 실시간 코어를 비켜 둔 자리. 고정하지 않는 PC 면 빈 줄(호출하는 쪽이 제한을 안 건다)
        print(format_cpulist(plan.general) if plan.rt else "")
    else:
        print(plan.describe())
