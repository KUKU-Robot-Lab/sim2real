"""cpu_plan — 이 PC 의 코어를 읽어 실시간 자리 · torch 스레드를 정한다(가짜 sysfs 로 PC 여러 종류를 흉내 낸다)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from policy_control import cpu_plan as C


def _sysfs(tmp_path: Path, siblings: dict[int, str], online: str, isolated: str = "") -> Path:
    root = tmp_path / "cpu"
    for cpu, sib in siblings.items():
        d = root / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(sib + "\n")
    (root / "online").write_text(online + "\n")
    (root / "isolated").write_text(isolated + "\n")
    return root


def _smt(tmp_path: Path, cores: int, **kw) -> Path:
    """AMD/Intel 데스크톱처럼 형제가 cpu i 와 i+cores 인 PC (arm4090 = 16 코어 · 32 스레드)."""
    sib = {i: f"{i % cores},{i % cores + cores}" for i in range(2 * cores)}
    return _sysfs(tmp_path, sib, f"0-{2 * cores - 1}", **kw)


def _flat(tmp_path: Path, cores: int, **kw) -> Path:
    """SMT 없는 PC · VM."""
    return _sysfs(tmp_path, {i: str(i) for i in range(cores)}, f"0-{cores - 1}", **kw)


def test_cpulist_ranges_and_singles():
    assert C.parse_cpulist("0-3,8,10-11\n") == (0, 1, 2, 3, 8, 10, 11)
    assert C.parse_cpulist("\n") == ()


def test_reads_physical_cores_with_their_smt_siblings(tmp_path):
    topo = C.read_topology(_smt(tmp_path, 16))
    assert len(topo.cores) == 16
    assert topo.cores[0] == (0, 16) and topo.cores[15] == (15, 31)


def test_offline_cpus_are_left_out(tmp_path):
    root = _smt(tmp_path, 4)
    (root / "online").write_text("0-2,4-6\n")            # cpu3 · cpu7(코어 3) 꺼짐
    assert C.read_topology(root).cores == ((0, 4), (1, 5), (2, 6))


def test_desktop_gives_each_master_its_own_high_core_and_keeps_the_sibling_free(tmp_path):
    p = C.make_plan(C.read_topology(_smt(tmp_path, 16)), env={})
    assert dict(p.rt) == {"ecat_right": 15, "ecat_left": 14}
    assert p.reserved == (14, 15, 30, 31)
    assert not set(p.reserved) & set(p.general) and len(p.general) == 28
    assert p.torch_threads == 1


def test_the_smallest_pinned_machine_keeps_four_cores_for_the_nodes_and_skips_cpu0(tmp_path):
    p = C.make_plan(C.read_topology(_flat(tmp_path, 6)), env={})
    assert dict(p.rt) == {"ecat_right": 5, "ecat_left": 4} and p.general == (0, 1, 2, 3)
    assert p.torch_threads == 1


def test_isolated_cpu0_core_is_still_never_used(tmp_path):
    p = C.make_plan(C.read_topology(_flat(tmp_path, 8, isolated="0")), env={})
    assert 0 not in p.rt.values() and 0 not in p.general        # 떼어 둔 cpu0 은 누구의 자리도 아니다


@pytest.mark.parametrize("cores", [3, 4, 5])
def test_a_small_machine_is_not_pinned(tmp_path, cores):
    p = C.make_plan(C.read_topology(_flat(tmp_path, cores)), env={})
    assert dict(p.rt) == {} and p.reserved == () and p.general == tuple(range(cores))
    assert p.torch_threads == 1 and "고정 안 함" in p.note


def test_isolated_cores_are_used_first_and_kept_out_of_general(tmp_path):
    p = C.make_plan(C.read_topology(_flat(tmp_path, 8, isolated="4-5")), env={})
    assert dict(p.rt) == {"ecat_right": 5, "ecat_left": 4}
    assert p.general == (0, 1, 2, 3, 6, 7)


def test_pinning_can_be_turned_off(tmp_path):
    p = C.make_plan(C.read_topology(_smt(tmp_path, 16)), env={C.PIN_ENV: "0"})
    assert dict(p.rt) == {} and len(p.general) == 32 and C.PIN_ENV in p.note


@pytest.mark.parametrize("raw, want", [("4", 4), ("x", 1), ("0", 1)])
def test_torch_threads_env_wins_when_valid(tmp_path, raw, want):
    p = C.make_plan(C.read_topology(_smt(tmp_path, 16)), env={C.THREADS_ENV: raw})
    assert p.torch_threads == want


def test_the_plan_does_not_depend_on_the_askers_own_cores(tmp_path):
    """이미 일반 코어에 고정된 노드가 물어도 같은 답이어야 두 손 노드 · 다른 노드가 같은 배치를 쓴다(sysfs 만 읽는다)."""
    root = _smt(tmp_path, 16)
    wide = C.make_plan(C.read_topology(root), env={})
    before = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, {min(before)})
        narrow = C.make_plan(C.read_topology(root), env={})
    finally:
        os.sched_setaffinity(0, before)
    assert narrow == wide


def test_a_broken_sysfs_gives_an_unpinned_plan_not_an_exception(tmp_path):
    root = _smt(tmp_path, 4)
    (root / "cpu1" / "topology" / "thread_siblings_list").write_text("garbage\n")
    p = C.current_plan(root)
    assert dict(p.rt) == {} and "고정 안 함" in p.note and p.general


def test_pin_reports_a_reason_instead_of_raising():
    assert C.pin(os.getpid(), sorted(os.sched_getaffinity(0))) is None      # 지금 자리 그대로 = 바뀌는 것 없음
    why = C.pin(os.getpid(), (100_000,))
    assert isinstance(why, str) and why
    assert C.pin(os.getpid(), ()) is not None


def test_describe_names_the_roles_and_threads(tmp_path):
    text = C.make_plan(C.read_topology(_smt(tmp_path, 16)), env={}).describe()
    assert "cpu15" in text and "cpu14" in text and "torch 1" in text


def test_the_master_starts_on_its_planned_core(tmp_path):
    """setcap 마스터는 뜬 뒤 고정할 수 없어(EPERM) 시작할 때 taskset 으로 — 그 자리가 exec 뒤에도 남는지."""
    import shutil
    import subprocess
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2 or shutil.which("taskset") is None:
        pytest.skip("CPU 하나뿐 · taskset 없음")
    plan = C.CpuPlan(rt={"ecat_right": cpus[-1]}, reserved=(cpus[-1],), general=tuple(cpus[:-1]), note="t")
    argv, note = C.pinned_argv(["sleep", "5"], "ecat_right", plan)
    assert note == f"CPU: ecat_right → cpu{cpus[-1]}" and argv[-2:] == ["sleep", "5"]
    child = subprocess.Popen(argv)
    try:
        for _ in range(50):                                    # taskset 이 exec 할 때까지
            if Path(f"/proc/{child.pid}/comm").read_text().strip() == "sleep":
                break
            __import__("time").sleep(0.02)
        assert Path(f"/proc/{child.pid}/comm").read_text().strip() == "sleep"     # pid 그대로 = 노드가 기다리는 그 프로세스
        assert os.sched_getaffinity(child.pid) == {cpus[-1]}
    finally:
        child.kill()
        child.wait()
    same, why = C.pinned_argv(["sleep", "5"], "ecat_left", plan)          # 계획에 없는 역할은 그대로 둔다
    assert same == ["sleep", "5"] and "고정 안 함" in why


def test_a_core_taskset_cannot_take_leaves_the_master_unwrapped():
    """감싼 taskset 이 실패하면 마스터가 아예 안 뜬다(cpuset 컨테이너 · 오프라인 코어) — 시험해 보고 못 잡으면 감싸지 않는다."""
    import shutil
    if shutil.which("taskset") is None:
        pytest.skip("taskset 없음")
    plan = C.CpuPlan(rt={"ecat_right": 99_999}, general=(0,), note="t")
    argv, why = C.pinned_argv(["true"], "ecat_right", plan)
    assert argv == ["true"] and "cpu99999" in why and "고정 안 함" in why


def test_keep_off_rt_without_a_plan_only_reports(tmp_path):
    before = os.sched_getaffinity(0)
    assert C.keep_off_rt(C.CpuPlan(general=tuple(before), note="작은 PC")) == "CPU: 작은 PC"
    assert os.sched_getaffinity(0) == before


@pytest.mark.parametrize("cpus, want", [((0, 1, 2, 3, 8, 10, 11), "0-3,8,10-11"), ((5,), "5"), ((), ""),
                                        (tuple(range(14)) + tuple(range(16, 30)), "0-13,16-29")])
def test_cpulist_round_trips_in_the_kernel_format(cpus, want):
    assert C.format_cpulist(cpus) == want
    assert C.parse_cpulist(want) == tuple(sorted(cpus))


def test_the_general_cpulist_cli_is_empty_when_the_pc_is_not_pinned(tmp_path):
    """fpp_up.sh 가 이 출력으로 --cpuset-cpus 를 건다 — 고정하지 않는 PC(작은 · S2R_CPU_PIN=0)는 빈 줄이어야 제한이 안 걸린다."""
    import subprocess
    import sys
    script = Path(C.__file__)
    off = subprocess.run([sys.executable, str(script), "--general"], capture_output=True, text=True,
                         env={**os.environ, C.PIN_ENV: "0"})
    assert off.returncode == 0 and off.stdout.strip() == ""
    on = subprocess.run([sys.executable, str(script), "--general"], capture_output=True, text=True,
                        env={k: v for k, v in os.environ.items() if k != C.PIN_ENV})
    plan = C.current_plan()
    assert on.stdout.strip() == (C.format_cpulist(plan.general) if plan.rt else "")


def test_fpp_containers_are_kept_off_the_realtime_cores():
    text = (Path(__file__).resolve().parents[2] / "scripts/vision/fpp_up.sh").read_text()
    assert "cpu_plan.py\" --general" in text and '"${CPUSET[@]}"' in text
