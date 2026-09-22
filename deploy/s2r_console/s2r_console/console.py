"""콘솔의 한가운데 — 프로파일 하나를 열어 run 하나를 굴린다.

여기가 **유일하게 상태를 바꾸는 곳**이다. HTTP 계층(`api.py`)은 이 클래스의 메서드를 부를 뿐이고,
판정은 전부 남의 것이다: 단계 가드는 `mission_core.gate`, 승인의 유효성은 `ledger`, 배너는 `console_state`.

콘솔이 스스로 소유하는 것은 넷뿐이다 — run 디렉터리, 승인 원장, 띄운 자식들, 조작 lease.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from . import _paths, ledger, survey
from . import units as U
from .console_state import STALE_S, derive
from .diagram import _LAUNCHER as PERCEPT_BOX
from .diagram import build as build_diagram
from .diagram_spec import Diagram
from .feed import Feed
from .lease import Lease
from .links import chain
from .profiles import Profile, scan
from .rosgraph import REPEAT_S as GRAPH_REPEAT_S
from .runner import SETTLE_S, StageRunner, step_kind
from .supervisor import Supervisor, SupervisorError, child_env
from .wiring import generate as generate_diagram
from .wiring import PERCEPTION_STATUS as PERCEPT_STATUS

import mission_core as MC  # noqa: E402
import mission_run  # noqa: E402
import policy_control.policy_registry as registry  # noqa: E402
from mission_stages import commands_for, load_runbook  # noqa: E402

#: 언제나 누를 수 있는 정지 동작. **argv 는 여기 코드에만 있다** — HTTP 로는 이름만 온다.
#: estop 은 없다: 웹서버→DDS 를 거치는 estop 은 가장 느린 estop 이고, 빨간 버튼은 물리 버튼이어야 한다.
QUICK = {
    "episode_stop": ("에피소드 정지", "정책 루프를 멈춘다. pd 는 팔을 잡은 채로 남는다.", ["episode/stop"]),
    "episode_abort": ("에피소드 중단", "abort 이벤트를 낸다. pd 는 HOLD 로 간다.", ["episode/abort"]),
    "pd_release": ("PD 해제", "역블렌드로 토크를 내리고 JTC 로 돌려준다 → IDLE.", ["pd/release", "--expect-pd", "IDLE"]),
}
_TRIGGER = _paths.POLICY_CONTROL / "tools" / "trigger.py"
_BRIDGE_RESTART_S = 3.0
_BRIDGE_RESTART_MAX_S = 30.0
GRAPH_STALE_S = 3 * GRAPH_REPEAT_S   # 그래프는 REPEAT_S 마다 다시 온다 — 세 번 거르면 못 본 것으로 친다
_BRIDGE_LIVED_S = 10.0          # 이보다 오래 살았으면 "거듭 죽는 중" 이 아니다
_RCLPY_HINT = "콘솔을 띄운 셸에서 `source /opt/ros/humble/setup.bash` 를 먼저 할 것 (PYTHONPATH 를 덮어쓰지 말 것)"


def death_reason(rc: int, log_tail: str) -> str:
    """브리지가 죽은 이유 한 줄. `rc=1` 만으로는 운영자가 할 수 있는 일이 없다."""
    lines = [ln.strip() for ln in log_tail.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    if "No module named 'rclpy'" in last:
        return f"rc={rc} · rclpy 를 import 할 수 없다 — {_RCLPY_HINT}"
    return f"rc={rc} · {last[:200]}" if last else f"rc={rc}"


def restart_delay(fast_deaths: int) -> float:
    """연속으로 빨리 죽은 횟수 → 다음 시도까지 기다릴 초 (3, 6, 12, 24, 30, 30 …)."""
    return min(_BRIDGE_RESTART_MAX_S, _BRIDGE_RESTART_S * 2 ** max(0, fast_deaths - 1))


class ConsoleError(RuntimeError):
    def __init__(self, message: str, *, code: int = 409, reasons: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code, self.reasons = code, reasons or (message,)


def mission_units(profile: Profile, *, repo: Path) -> dict[str, U.UnitCmd]:
    """프로파일의 미션이 가진 배경·수동 명령 — 그림의 스위치와 자동 생성의 재료. argv 는 미션 yaml 에서만 온다."""
    raw = yaml.safe_load(profile.mission.read_text(encoding="utf-8"))
    mission = MC.load_mission(raw)
    runbook = load_runbook(raw.get("run", {}), mission)
    return U.index_units(mission, {st.id: commands_for(runbook, mission, st.id, repo=repo, execute=True) for st in mission.stages})


def diagram_of(profile: Profile, units: Mapping[str, U.UnitCmd], *, repo: Path) -> Diagram | None:
    """프로파일이 그림을 손으로 적었으면 그것, 아니면 미션·계약·robot yaml 에서 만든다(정책이 바뀌면 따라 바뀐다).

    만들지 못하면 None — 그림은 편의다. 그것 때문에 세션이 안 열리면 운영자는 미션 자체를 못 본다.
    """
    if profile.diagram is not None:
        return profile.diagram
    try:
        return generate_diagram(units, repo=repo, status_nodes=profile.status_nodes)
    except Exception:                                   # noqa: BLE001 — 사유는 화면의 단위 목록에 그대로 남는다
        return None


def bridge_argv(profile: Profile, diagram: Diagram | None = None) -> list[str]:
    """구독 전용 브리지. 스택 토픽은 프로파일이 선언했을 때만 넘긴다. 순수."""
    diagram = diagram if diagram is not None else profile.diagram
    argv = [sys.executable, "-m", "s2r_console.bridge", "--domain", str(profile.domain),
            "--nodes", *profile.status_nodes]
    topics = [] if profile.stack is None else [t.name for t in profile.stack.topics]
    watch: list[str] = []
    if diagram is not None:                             # 그림의 전선 — 세는 것과 그래프만 보는 것
        topics = list(dict.fromkeys([*topics, *diagram.metered()]))
        watch = [t for t in diagram.watched() if t not in topics]
    argv += ["--topics", *topics] if topics else []
    argv += ["--watch", *watch] if watch else []
    if diagram is not None and any(b.id == PERCEPT_BOX for b in diagram.boxes):
        argv += ["--perception", PERCEPT_STATUS]         # vision-3090 의 카메라·컨테이너는 런처만 안다
    return argv


def probe_argv(profile: Profile, diagram: Diagram | None = None) -> list[str] | None:
    """컨트롤러 프로브. 스택·그림 어디에도 controller_manager 가 없으면 띄우지 않는다(None). 순수."""
    managers = [] if profile.stack is None else [m.name for m in profile.stack.managers]
    managers = list(dict.fromkeys([*managers, *(b.manager for b in (diagram.boxes if diagram else ()) if b.manager)]))
    if not managers:
        return None
    return [sys.executable, "-m", "s2r_console.ctl_probe", "--domain", str(profile.domain), "--managers", *managers]


class _Pipe:
    """자식 프로세스를 띄우고 stdout(NDJSON)을 `Feed` 에 흘려 넣는다. 죽으면 다시 띄운다.

    브리지와 컨트롤러 프로브가 같이 쓴다. 다른 것은 인자와, 죽었을 때 `Feed` 의 어느 쪽에 알리느냐(`died`)뿐이다.
    """

    def __init__(self, name: str, argv: Sequence[str], died: Callable[[str], None], feed: Feed,
                 feed_lock: threading.Lock, env: Mapping[str, str], log: Path) -> None:
        self._name, self._argv, self._died = name, list(argv), died
        self._feed, self._lock, self._log = feed, feed_lock, log
        self._env = dict(env)
        root = str(Path(__file__).resolve().parents[1])
        self._env["PYTHONPATH"] = root + os.pathsep + self._env.get("PYTHONPATH", "")
        self._popen: subprocess.Popen | None = None
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        p = self._popen
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

    def _loop(self) -> None:
        argv = self._argv
        fast = 0
        while not self._stopping.is_set():
            began, at = time.monotonic(), self._log.stat().st_size if self._log.exists() else 0
            with self._log.open("ab") as err:
                try:
                    self._popen = subprocess.Popen(argv, env=self._env, stdin=subprocess.DEVNULL,
                                                   stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
                except OSError as exc:
                    with self._lock:
                        self._died(str(exc))
                    return
                assert self._popen.stdout is not None
                for text in self._popen.stdout:
                    with self._lock:
                        self._feed.ingest(text)
                rc = self._popen.wait()
            if self._stopping.is_set():
                return
            with self._log.open("rb") as fh:       # 이번 시도가 쓴 부분만 — 로그는 append 다
                fh.seek(at)
                tail = fh.read()[-4096:].decode("utf-8", errors="replace")
            with self._lock:
                self._died(death_reason(rc, tail))
            if rc == 2:          # 자식이 도메인을 스스로 거부했다 — 다시 띄워 봐야 같다
                return
            fast = fast + 1 if time.monotonic() - began < _BRIDGE_LIVED_S else 1
            self._stopping.wait(restart_delay(fast))


class Session:
    def __init__(self, profile: Profile, *, repo: Path, operator: str, bridge: bool) -> None:
        self.profile, self.repo, self.operator = profile, repo, operator
        raw = yaml.safe_load(profile.mission.read_text(encoding="utf-8"))
        self.mission = MC.load_mission(raw)
        self.runbook = load_runbook(raw.get("run", {}), self.mission)
        self.run_id = f"{datetime.now():%Y%m%d_%H%M%S}__{profile.id}"
        self.run_dir = mission_run.MISSION_LOG_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state = MC.initial_state(self.mission)
        self.skipped: frozenset[str] = frozenset()          # 실행하지 않고 넘긴 단계 — 목록에 "건너뜀" 으로 보인다
        self.cache = survey.DigestCache()
        self.ledger_path = self.run_dir / "approvals.jsonl"
        self.intents_path = self.run_dir / "intents.jsonl"
        env = child_env(os.environ, domain=profile.domain, run_id=self.run_id)
        self.supervisor = Supervisor(cwd=repo, run_dir=self.run_dir, env=env)
        self.feed = Feed(profile.status_nodes, expect_domain=profile.domain, latency=profile.latency)
        self.feed_lock = threading.Lock()
        self.runner: StageRunner | None = None
        self.policy = None if profile.policy_dir is None else registry.check(profile.policy_dir, deep=False)
        self.started = time.time()
        self._dead_seen: set[str] = set()
        #: 그림의 스위치가 다루는 것 — 미션의 배경·수동 명령. argv 는 여기(미션 yaml)에서만 온다.
        self.units = U.index_units(self.mission, {st.id: commands_for(self.runbook, self.mission, st.id, repo=repo, execute=True)
                                                  for st in self.mission.stages})
        #: 연결 그림 — 프로파일이 적지 않았으면 미션·계약·robot yaml 에서 만든다. 정책을 바꾸면 그림이 따라 바뀐다.
        self.diagram = diagram_of(profile, self.units, repo=repo)
        #: 운영자가 끈 것 {키: 그때의 pid} — 죽은 것과 구별한다. pid 까지 적어야 그 뒤 다시 뜬 프로세스의 크래시를 가리지 않는다.
        self.units_stopped: dict[str, int | None] = {}
        proc = self.run_dir / "proc"
        probe = probe_argv(profile, self.diagram) if bridge else None
        self.bridge = (_Pipe("bridge", bridge_argv(profile, self.diagram), self.feed.bridge_died, self.feed, self.feed_lock, env,
                             proc / "bridge.log") if bridge else None)
        self.probe = (None if probe is None else
                      _Pipe("ctl_probe", probe, self.feed.probe_died, self.feed, self.feed_lock, env, proc / "ctl_probe.log"))
        mission_run.save_state(self.run_id, self.state)
        (self.run_dir / "run.json").write_text(json.dumps({
            "run_id": self.run_id, "profile": profile.as_dict(), "operator": operator,
            "started": datetime.now().isoformat(timespec="seconds"),
            "basis": dict(survey.all_basis(self.mission, repo=repo, cache=self.cache)),
        }, ensure_ascii=False, indent=2))
        for pipe in (self.bridge, self.probe):
            if pipe is not None:
                pipe.start()

    def event(self, kind: str, text: str) -> None:
        with self.feed_lock:
            self.feed.events.append({"t": time.time(), "kind": kind, "text": text})


class Console:
    def __init__(self, *, repo: Path = _paths.SIM2REAL, profiles_dir: Path | None = None, bridge: bool = True) -> None:
        self.repo = repo
        self.profiles_dir = profiles_dir or Path(__file__).resolve().parents[1] / "profiles"
        self.lease = Lease()
        self.session: Session | None = None
        self._bridge = bridge
        self._lock = threading.RLock()

    # ── 열고 닫기 ───────────────────────────────────────────────────────
    def open(self, profile_id: str, *, operator: str) -> Session:
        with self._lock:
            if self.session is not None:
                raise ConsoleError(f"run {self.session.run_id} 이 열려 있다 — 먼저 끝낼 것")
            good, bad = scan(self.profiles_dir, repo=self.repo)
            profile = next((p for p in good if p.id == profile_id), None)
            if profile is None:
                why = next((v for k, v in bad.items() if Path(k).stem == profile_id), "그런 프로파일이 없다")
                raise ConsoleError(f"{profile_id}: {why}", code=404)
            self.session = Session(profile, repo=self.repo, operator=operator, bridge=self._bridge)
            self.session.event("run", f"run 시작 — {profile.title} (도메인 {profile.domain}, {profile.domain_class})")
            return self.session

    def end_reasons(self) -> list[str]:
        """지금 run 을 끝내면 왜 안 되는가. 비어 있으면 된다."""
        s = self._need()
        out = []
        if s.runner is not None and s.runner.active:
            out.append(f"단계 {s.runner.stage_id} 가 실행 중이다")
        with s.feed_lock:
            obs = s.feed.observed()
        phase = self._pd_phase(s, obs, {p["key"]: p for p in s.supervisor.table()})
        if phase == U.PD_UNKNOWN:
            out.append("pd 상태를 모른다 — 브리지 복구 후 PD 해제를 확인할 것 (그래도 끝내려면 강제 종료)")
        elif phase not in U.PD_FREE:
            out.append(f"pd 가 {phase} 다 — 프로세스를 죽이기 전에 먼저 PD 해제(release)를 할 것")
        stack = U.stack_end_reason(real=s.profile.is_real, stack_keys=self._stack_keys(s),
                                   procs={p["key"]: p for p in s.supervisor.table()})
        if stack:
            out.append(stack)
        return out

    @staticmethod
    def _stack_keys(s: "Session") -> set[str]:
        """팔·손 드라이버를 띄운 단위 — 그림에서 controller_manager 를 가진 구동 상자의 단위."""
        return set() if s.diagram is None else {b.unit for b in s.diagram.boxes if b.manager and b.unit}

    def end(self, *, force: bool = False) -> int:
        with self._lock:
            s = self._need()
            reasons = self.end_reasons()
            if reasons and not force:
                raise ConsoleError("run 을 끝낼 수 없다", reasons=tuple(reasons))
            if s.runner is not None and s.runner.active:
                s.runner.abort()
                s.runner.join(timeout=8)
            n = s.supervisor.stop()
            for pipe in (s.bridge, s.probe):
                if pipe is not None:
                    pipe.stop()
            self._intent(s, "run/end", {"force": force, "stopped": n, "overrode": reasons})
            self.session = None
            return n

    def shutdown(self) -> list[str]:
        """콘솔 프로세스가 내려갈 때. fake 는 자식을 모두 정지한다.

        실기에서는 팔·손 드라이버만 **남긴다** — 팔 브링업을 내리면 모터가 전부 풀린다(Ctrl+C 한 번에 팔이 떨어졌다).
        그 밖의 자식(인지 런처·목 퍼블리셔·pd …)과 브리지·프로브는 정지한다. 남긴 키를 돌려준다(없으면 빈 목록).
        """
        with self._lock:
            s = self.session
            if s is None:
                return []
            procs = {p["key"]: p for p in s.supervisor.table()}
            kept = U.keep_on_exit(real=s.profile.is_real, stack_keys=self._stack_keys(s), procs=procs)
            if not kept:
                self.end(force=True)
                return []
            if s.runner is not None and s.runner.active:
                s.runner.abort()
                s.runner.join(timeout=8)
            others = [k for k, p in procs.items() if p.get("alive") and k not in kept]
            if others:
                s.supervisor.stop(others)
            for pipe in (s.bridge, s.probe):
                if pipe is not None:
                    pipe.stop()
            self._intent(s, "run/detach", {"kept": kept, "stopped": others})
            self.session = None
            return kept

    # ── 승인 ────────────────────────────────────────────────────────────
    def approve(self, stage_id: str, *, operator: str, typed: str, note: str = "") -> None:
        with self._lock:
            s = self._need()
            stage = self._current_stage(s, stage_id)
            if not stage.touches_real:
                raise ConsoleError(f"{stage_id} 는 실기를 건드리지 않는다 — 승인이 필요 없다", code=400)
            if typed != stage_id:
                raise ConsoleError("확인 입력이 단계 id 와 다르다", code=400)
            structural = self._structural(s, stage_id)
            if structural:
                raise ConsoleError("막힌 단계는 승인할 수 없다", reasons=tuple(structural))
            basis = survey.stage_basis(s.mission, stage_id, repo=self.repo, cache=s.cache)
            ledger.append(s.ledger_path, ledger.Entry("approve", stage_id, operator, _now(), basis, note))
            s.event("approve", f"{operator} 가 {stage_id} 를 승인했다")
            self._intent(s, "approve", {"stage": stage_id, "operator": operator, "basis": basis})

    def revoke(self, stage_id: str, *, operator: str, note: str = "운영자 취소") -> None:
        with self._lock:
            s = self._need()
            MC.stage_by_id(s.mission, stage_id)
            ledger.append(s.ledger_path, ledger.Entry("revoke", stage_id, operator, _now(), {}, note))
            s.event("approve", f"{stage_id} 승인 취소 — {note}")

    # ── 단계 실행 ───────────────────────────────────────────────────────
    def run_stage(self, stage_id: str, *, operator: str) -> None:
        with self._lock:
            s = self._need()
            stage = self._current_stage(s, stage_id)
            if s.state.status == MC.STATUS_DONE:
                raise ConsoleError("미션이 끝났다")
            result = MC.gate(s.mission, stage_id, s.state, self._evidence(s))
            if not result.ok:
                raise ConsoleError(f"{stage_id} 를 지금 실행할 수 없다", reasons=tuple(result.reasons))
            commands = commands_for(s.runbook, s.mission, stage_id, repo=self.repo, execute=True)
            s.state = MC.begin(s.state)
            mission_run.save_state(s.run_id, s.state)
            s.event("stage", f"▶ {stage_id} — {stage.title}")
            self._intent(s, "stage/run", {"stage": stage_id, "operator": operator,
                                          "argv": [list(c.argv) for c in commands]})
            s.runner = StageRunner(stage_id, commands, s.supervisor, on_done=self._stage_done)
            s.runner.start()

    def skip_stage(self, stage_id: str, *, operator: str) -> None:
        """지금 단계를 실행하지 않고 넘긴다 — 오른팔만 할 때 왼팔 단계처럼 미션이 `skippable` 로 선언한 것만.

        넘긴 단계는 완료로 친다(뒤 단계의 선행 조건이 풀린다). 원장·기록에 "건너뜀" 으로 남는다.
        """
        with self._lock:
            s = self._need()
            stage = self._current_stage(s, stage_id)
            reasons = self._skip_reasons(s, stage)
            if reasons:
                raise ConsoleError(f"{stage_id} 를 건너뛸 수 없다", reasons=tuple(reasons))
            s.state = MC.advance(s.mission, MC.begin(s.state), MC.STATUS_DONE, note=f"건너뜀 ({operator})")
            s.skipped = s.skipped | {stage_id}
            mission_run.save_state(s.run_id, s.state)
            s.event("stage", f"» {stage_id} 건너뜀 — {operator}")
            self._intent(s, "stage/skip", {"stage": stage_id, "operator": operator})

    def rewind(self, stage_id: str, *, operator: str) -> None:
        """끝낸(또는 건너뛴) 단계로 돌아가 그 단계부터 다시 진행한다 — 09.22 사용자: "잘못되면 되돌아가서 진행할 수가 없다".

        떠 있는 프로세스는 그대로 둔다(다시 실행하면 러너가 "이미 떠 있음" 으로 넘긴다). 실기 단계는 승인을 다시 받는다
        (승인은 실행 한 번에 쓰이고 사라진다). 단계가 도는 중에는 되돌리지 않는다.
        """
        with self._lock:
            s = self._need()
            if s.runner is not None and s.runner.active:
                raise ConsoleError(f"단계 {s.runner.stage_id} 가 실행 중이다 — 끝나거나 중단한 뒤에 되돌릴 것")
            try:
                MC.stage_by_id(s.mission, stage_id)
            except KeyError as exc:
                raise ConsoleError(f"모르는 단계: {stage_id}", code=404) from exc
            ids = [st.id for st in s.mission.stages]
            if stage_id not in s.state.completed:
                raise ConsoleError(f"{stage_id} 는 아직 끝낸 단계가 아니다 — 끝낸 단계로만 돌아간다")
            cut = ids.index(stage_id)
            s.state = MC.MissionState(stage=stage_id, status=MC.STATUS_PENDING,
                                      completed=tuple(x for x in s.state.completed if ids.index(x) < cut),
                                      cycle=s.state.cycle, note=f"되돌림 ({operator})")
            s.skipped = frozenset(x for x in s.skipped if ids.index(x) < cut)
            mission_run.save_state(s.run_id, s.state)
            s.event("stage", f"↶ {stage_id} 로 되돌림 — {operator}")
            self._intent(s, "stage/rewind", {"stage": stage_id, "operator": operator})

    def _skip_reasons(self, s: Session, stage) -> list[str]:
        with s.feed_lock:
            obs = s.feed.observed()
        phase = self._pd_phase(s, obs, {p["key"]: p for p in s.supervisor.table()})
        return U.skip_reasons(skippable=stage.skippable, busy=s.runner is not None and s.runner.active, pd_phase=phase)

    def ack(self, index: int, ok: bool) -> None:
        with self._lock:
            s = self._need()
            if s.runner is None or not s.runner.active:
                raise ConsoleError("실행 중인 단계가 없다")
            try:
                s.runner.ack(index, ok)
            except ValueError as exc:
                raise ConsoleError(str(exc)) from exc
            self._intent(s, "stage/ack", {"stage": s.runner.stage_id, "index": index, "ok": ok})

    def abort_stage(self) -> None:
        with self._lock:
            s = self._need()
            if s.runner is None or not s.runner.active:
                raise ConsoleError("실행 중인 단계가 없다")
            s.runner.abort()
            self._intent(s, "stage/abort", {"stage": s.runner.stage_id})

    def _stage_done(self, stage_id: str, outcome: str, note: str) -> None:
        with self._lock:
            s = self.session
            if s is None or s.state.stage != stage_id:
                return
            stage = MC.stage_by_id(s.mission, stage_id)
            s.state = MC.advance(s.mission, s.state, outcome, note=note)
            mission_run.save_state(s.run_id, s.state)
            if stage.touches_real:
                # 승인은 **한 번의 실행**에 대한 것이다 — 반복(loop_to)으로 돌아와도 다시 받는다.
                ledger.append(s.ledger_path, ledger.Entry("revoke", stage_id, "console", _now(), {}, f"실행에 쓰였다 ({outcome})"))
            mark = {"DONE": "✓", "FAILED": "✗", "ABORTED": "■"}.get(outcome, "?")
            s.event("stage", f"{mark} {stage_id} {outcome}" + (f" — {note}" if note else ""))
            release = outcome != MC.STATUS_DONE and stage.touches_real and self._pd_holds(s)
        if release:
            # 실기 단계가 실패 · 중단했는데 pd 가 팔을 잡고 있으면 풀어 둔다(JTC 가 그 자리를 잡는다).
            # 09.22: 저장 경로 시작점 검사에서 멈췄는데 pd 는 engage 된 채 남았다 — 그 검사는 episode_ctl 이 아니라 해제하지 않는다.
            try:
                self.quick("pd_release", client=f"자동: {stage_id} {outcome}")
            except ConsoleError as exc:
                with self._lock:
                    if self.session is s:
                        s.event("quick", f"자동 PD 해제를 못 했다 — 정지 바의 PD 해제를 누를 것 ({exc})")

    # ── 그림의 스위치 ───────────────────────────────────────────────────
    def toggle_unit(self, key: str, on: bool, *, operator: str) -> None:
        """미션의 배경 명령 하나를 켜거나 끈다. 규칙은 `units.on_reasons` / `off_reasons` 가 정한다."""
        with self._lock:
            s = self._need()
            unit = s.units.get(key)
            if unit is None:
                raise ConsoleError(f"모르는 단위: {key}", code=404)
            v = self._units_view(s)[key]
            reasons = v["why_on"] if on else v["why_off"]
            if reasons:
                raise ConsoleError(f"{key} 를 지금 {'켤' if on else '끌'} 수 없다", reasons=tuple(reasons))
            if on:
                try:
                    s.supervisor.spawn(key, stage=unit.stage, note=unit.note, argv=unit.argv, background=True, manual=False)
                except SupervisorError as exc:
                    raise ConsoleError(str(exc)) from exc
                s.units_stopped.pop(key, None)
            else:
                s.units_stopped[key] = v["pid"]
            s.event("unit", f"{'▲ 켬' if on else '▼ 끔'} {key} — {unit.note} ({operator})")
            self._intent(s, f"unit/{'on' if on else 'off'}", {"key": key, "operator": operator, "argv": list(unit.argv)})
        target = self._unit_settle if on else self._unit_stop
        threading.Thread(target=target, args=(s, key, unit.note), daemon=True).start()

    def _robot_keys(self, s: Session) -> set[str] | None:
        """pd 노드를 띄운 단위의 키 **전부**. 그림에 pd 상자가 없으면 None(= 어느 단위가 pd 인지 모른다 — 전부 조심해서 다룬다).

        빈 집합은 pd 상자는 있는데 콘솔이 띄우지 않는다는 뜻이다(스위치가 없다).
        """
        pd_boxes = [] if s.diagram is None else [b for b in s.diagram.boxes if b.id.startswith("pd") or b.status == "pd"]
        return None if not pd_boxes else {b.unit for b in pd_boxes if b.unit}

    def _pd_holds(self, s: Session) -> bool:
        """pd 가 팔을 잡고 있을 수 있는가(모르는 것도 잡은 쪽으로). 콘솔이 띄운 pd 가 없으면 아니다."""
        procs = {p["key"]: p for p in s.supervisor.table()}
        keys = self._robot_keys(s) or set()
        if not any(procs.get(k, {}).get("alive") for k in keys):
            return False
        with s.feed_lock:
            obs = s.feed.observed()
        phase = self._pd_phase(s, obs, {p["key"]: p for p in s.supervisor.table()})
        return phase not in U.PD_FREE

    def _pd_phase(self, s: Session, obs, procs: Mapping[str, Mapping]) -> str | None:
        """끄기·종료 규칙이 볼 pd phase — 조용한 pd 는 자유가 아니라 `U.PD_UNKNOWN` 이다."""
        keys = self._robot_keys(s)
        # None = 어느 단위가 pd 인지 모른다(그림 없음) · 빈 집합 = 그림은 있는데 pd 를 콘솔이 띄우지 않는다(역시 모른다)
        pd_alive = None if not keys else any(procs.get(k, {}).get("alive") for k in keys)
        d = s.diagram
        pd_ros = () if d is None else tuple(n for b in d.boxes if b.status == "pd" for n in b.ros)
        seen = obs.rosgraph is not None and obs.rosgraph_age_s is not None and obs.rosgraph_age_s <= GRAPH_STALE_S
        pd_in_graph = bool(set(pd_ros) & set(obs.rosgraph.get("nodes") or ())) if seen and pd_ros else None
        return U.pd_phase_of(obs.status.get("pd"), obs.age_s.get("pd"), stale_s=STALE_S, pd_alive=pd_alive,
                             real=s.profile.is_real, pd_in_graph=pd_in_graph)

    @staticmethod
    def _unit_settle(s: Session, key: str, note: str) -> None:
        """단계 러너와 같은 확인 — 띄운 직후에 죽으면 그렇다고 말한다."""
        time.sleep(SETTLE_S)
        if not s.supervisor.is_alive(key):
            s.event("proc", f"{key} ({note}) 가 뜨자마자 끝났다 rc={s.supervisor.rc(key)} — 로그를 볼 것")

    def _unit_stop(self, s: Session, key: str, note: str) -> None:
        """그룹째 SIGTERM → 유예 → SIGKILL. 최대 5 s 걸리므로 콘솔 락 밖에서 한다(화면이 멎지 않게).

        죽이기 직전에 규칙을 **한 번 더** 본다 — 끄기를 받은 뒤 이 스레드가 돌기까지 사이에 단계가 pd 를 걸 수 있다.
        """
        with self._lock:
            why = [] if self.session is not s else self._units_view(s)[key]["why_off"]
            if why:
                s.units_stopped.pop(key, None)
                s.event("unit", f"{key} ({note}) 끄기를 거뒀다 — {'; '.join(why)}")
                return
        s.supervisor.stop([key])
        s.event("unit", f"{key} ({note}) 정지됨 rc={s.supervisor.rc(key)}")

    def _units_view(self, s: Session, obs=None) -> dict[str, dict]:
        if obs is None:
            with s.feed_lock:
                obs = s.feed.observed()
        robot_keys = self._robot_keys(s)
        busy = s.runner.stage_id if s.runner is not None and s.runner.active else None
        procs = {p["key"]: p for p in s.supervisor.table()}
        pd_phase = self._pd_phase(s, obs, procs)
        stopped = {k for k, pid in s.units_stopped.items() if procs.get(k, {}).get("pid") == pid}
        return U.views(s.units, procs, stopped=stopped, busy_stage=busy, completed=s.state.completed,
                       pd_phase=pd_phase, robot_keys=robot_keys, real=s.profile.is_real)

    # ── 언제나 되는 정지 동작 ───────────────────────────────────────────
    def quick(self, name: str, *, client: str) -> None:
        if name not in QUICK:
            raise ConsoleError(f"모르는 동작: {name}", code=404)
        with self._lock:
            s = self._need()
            label, _, args = QUICK[name]
            argv = ["python3", str(_TRIGGER), *args, "--execute", "--service-timeout", "5"]
            key = f"quick#{name}"
            try:
                s.supervisor.spawn(key, stage="quick", note=label, argv=argv, background=False, manual=False)
            except SupervisorError as exc:
                raise ConsoleError(str(exc)) from exc
            s.event("quick", f"■ {label} 요청 ({client})")
            self._intent(s, f"quick/{name}", {"client": client})
        threading.Thread(target=self._quick_wait, args=(s, key, label), daemon=True).start()

    @staticmethod
    def _quick_wait(s: Session, key: str, label: str) -> None:
        while s.supervisor.is_alive(key):
            time.sleep(0.1)
        rc = s.supervisor.rc(key)
        s.event("quick", f"{label} {'완료' if rc == 0 else f'실패 (rc={rc}) — 로그를 볼 것'}")

    def log_tail(self, key: str) -> str:
        s = self._need()
        if key in ("bridge", "ctl_probe"):
            path = s.run_dir / "proc" / f"{key}.log"
            return path.read_text(errors="replace")[-16384:] if path.is_file() else ""
        try:
            return s.supervisor.tail(key, 16384)
        except SupervisorError as exc:
            raise ConsoleError(str(exc), code=404) from exc

    # ── 화면 ────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            good, bad = scan(self.profiles_dir, repo=self.repo)
            out = {"t": time.time(), "lease": self.lease.view(),
                   "profiles": [p.as_dict() for p in good], "bad_profiles": bad, "session": None,
                   "quick": [{"name": k, "label": v[0], "help": v[1]} for k, v in QUICK.items()]}
            if self.session is not None:
                out["session"] = self._session_view(self.session)
            return out

    def _session_view(self, s: Session) -> dict:
        self._note_dead(s)
        with s.feed_lock:
            obs = s.feed.observed()
            metrics = s.feed.window.metrics(policy_dt=s.profile.policy_dt)
            events = list(s.feed.events)[-80:]
            bridge = {"up": s.feed.bridge_up(), "domain": s.feed.domain, "graph": list(s.feed.graph),
                      "bad_lines": s.feed.bad_lines, "enabled": s.bridge is not None}
            rosgraph = obs.rosgraph
        banner = derive(obs, expected_nodes=s.profile.status_nodes)
        links = chain(obs, expected_nodes=s.profile.status_nodes, domain=bridge["domain"],
                      domain_class=s.profile.domain_class, stack=s.profile.stack)
        units = self._units_view(s, obs)
        diagram = None if s.diagram is None else build_diagram(obs, s.diagram, units=units)
        nodes = [{"name": n, "age_s": None if n not in obs.age_s else round(obs.age_s[n], 2),
                  "stale": obs.age_s.get(n, 1e9) > 2.0, "status": obs.status.get(n)} for n in s.profile.status_nodes]
        return {"run_id": s.run_id, "run_dir": str(s.run_dir), "operator": s.operator,
                "uptime_s": round(time.time() - s.started), "profile": s.profile.as_dict(),
                "banner": banner.as_dict(), "links": links, "diagram": diagram, "rosgraph": rosgraph, "units": units, "bridge": bridge, "nodes": nodes, "episode": obs.episode,
                "mission": self._mission_view(s), "runner": None if s.runner is None else s.runner.view(),
                "procs": s.supervisor.table(), "metrics": metrics, "events": events,
                "policy": self._policy_view(s), "end_reasons": self.end_reasons()}

    def _mission_view(self, s: Session) -> dict:
        entries = ledger.read(s.ledger_path)
        basis = survey.all_basis(s.mission, repo=self.repo, cache=s.cache)
        valid = ledger.valid_approvals(entries, basis)
        stale = ledger.stale_reasons(entries, basis)
        ev = survey.gather(s.mission, repo=self.repo, approvals=valid, cache=s.cache)
        rows_struct = MC.plan(s.mission, s.state, mission_run.plan_evidence(s.mission, ev))
        busy = s.runner is not None and s.runner.active
        finished = s.state.status == MC.STATUS_DONE
        skip_why = None if finished else self._skip_reasons(s, MC.stage_by_id(s.mission, s.state.stage))
        rows = []
        for row in rows_struct:
            st = row.stage
            current = st.id == s.state.stage and not finished
            structural = list(row.result.reasons)
            approved = st.id in valid
            cmds = commands_for(s.runbook, s.mission, st.id, repo=self.repo, execute=True)
            rows.append({
                "id": st.id, "title": st.title, "touches_real": st.touches_real,
                "done": st.id in s.state.completed, "current": current,
                "status": s.state.status if current else ("DONE" if st.id in s.state.completed else "PENDING"),
                "reasons": structural, "approved": approved, "approval_stale": stale.get(st.id, []),
                "can_approve": current and st.touches_real and not approved and not structural and not busy,
                "can_run": current and not structural and (approved or not st.touches_real) and not busy,
                "group": st.group, "skippable": st.skippable, "skipped": st.id in s.skipped,
                "can_rewind": st.id in s.state.completed and not busy,
                "can_skip": current and st.skippable and not skip_why, "skip_why": skip_why if current and st.skippable else [],
                "commands": [{"note": c.note, "argv": list(c.argv), "kind": step_kind(c), "stop": list(c.stop)} for c in cmds],
            })
        groups = [{"id": g.id, "title": g.title, "motion": g.motion} for g in s.mission.groups]
        return {"name": s.mission.name, "stage": s.state.stage, "status": s.state.status, "cycle": s.state.cycle,
                "note": s.state.note, "loop_to": s.mission.loop_to, "rows": rows, "groups": groups}

    @staticmethod
    def _policy_view(s: Session) -> dict | None:
        e = s.policy
        if e is None:
            return None
        return {"id": e.id, "status": e.status, "card": dict(e.card), "checkpoint": e.checkpoint,
                "contract": e.contract, "issues": list(e.issues), "path": str(e.path)}

    # ── 안 ──────────────────────────────────────────────────────────────
    def _need(self) -> Session:
        if self.session is None:
            raise ConsoleError("열린 run 이 없다 — 먼저 프로파일을 열 것")
        return self.session

    @staticmethod
    def _current_stage(s: Session, stage_id: str):
        try:
            stage = MC.stage_by_id(s.mission, stage_id)
        except KeyError as exc:
            raise ConsoleError(f"모르는 단계: {stage_id}", code=404) from exc
        if stage_id != s.state.stage:
            raise ConsoleError(f"지금 단계는 {s.state.stage} 다 — {stage_id} 는 차례가 아니다")
        if s.runner is not None and s.runner.active:
            raise ConsoleError(f"단계 {s.runner.stage_id} 가 실행 중이다")
        return stage

    def _evidence(self, s: Session) -> MC.Evidence:
        basis = survey.all_basis(s.mission, repo=self.repo, cache=s.cache)
        valid = ledger.valid_approvals(ledger.read(s.ledger_path), basis)
        return survey.gather(s.mission, repo=self.repo, approvals=valid, cache=s.cache)

    def _structural(self, s: Session, stage_id: str) -> list[str]:
        ev = mission_run.plan_evidence(s.mission, self._evidence(s))
        return list(MC.gate(s.mission, stage_id, s.state, ev).reasons)

    @staticmethod
    def _note_dead(s: Session) -> None:
        """배경 프로세스가 죽은 것을 사건으로 남긴다 — 한 번씩만."""
        for p in s.supervisor.table():
            tag = f"{p['key']}@{p['pid']}"
            if p["background"] and not p["alive"] and tag not in s._dead_seen:
                s._dead_seen.add(tag)
                if p["key"] in s.units_stopped and s.units_stopped[p["key"]] == p["pid"]:
                    # 운영자가 끈 바로 그 프로세스 — 그 사건은 toggle_unit 이 남겼다
                    continue
                s.event("proc", f"배경 프로세스 종료: {p['key']} ({p['note']}) rc={p['rc']}")

    @staticmethod
    def _intent(s: Session, action: str, detail: Mapping) -> None:
        with s.intents_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": _now(), "action": action, **detail}, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


__all__ = ["Console", "ConsoleError", "QUICK", "replace"]
