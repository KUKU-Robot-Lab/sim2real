"""단위 — 미션 runbook 의 배경 명령 하나. 그림의 상자에 붙은 스위치가 이것을 켜고 끈다.

스위치는 미션의 규칙을 **돌아가지 않는다**:
  · argv 는 미션 yaml 에서만 온다. HTTP 로는 키(`단계#번호`)와 켬/끔만 온다.
  · 실기를 건드리는 단계의 명령은 여기서 켜지 않는다 — 승인 원장은 단계 실행에만 있고, 그것을 우회하는 길을 만들지 않는다.
  · manual 명령은 콘솔이 실행하지 않는다 (sudo·전원 순서·다른 PC). 스위치는 보이되 잠긴다.
  · 선행 단계가 끝나지 않았으면 켜지 않는다 — 단계 순서에는 이유(`needs_why`)가 있다.
  · pd 가 팔을 잡고 있는 동안에는 pd 를 끄지 않는다(토크가 그대로 끊긴다). 실기에서는 아무것도 끄지 않는다.

규칙은 순수 함수다. 프로세스를 다루는 쪽은 `console.Console.toggle_unit` 이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping

#: pd 가 팔을 잡고 있지 않은 phase. 이 밖의 값이면 "잡고 있다" 로 친다(모르는 phase 도 안전한 쪽으로).
PD_FREE = (None, "", "IDLE")
#: pd 상태를 **모른다** — 없다(None)와 다르다. 떠 있는데 조용한 pd 를 "자유" 로 치면 토크를 건 채로 죽이게 된다.
PD_UNKNOWN = "UNKNOWN"


def pd_phase_of(status: Mapping | None, age_s: float | None, *, stale_s: float, pd_alive: bool | None, real: bool,
                pd_in_graph: bool | None) -> str | None:
    """끄기 규칙이 볼 pd phase. None 은 "pd 가 없다고 **증명된다**" 일 때만 돌려준다.

    `pd_alive` — 콘솔이 띄운 pd 단위가 살아 있는가. 어느 단위가 pd 인지 모르면(그림 없는 프로파일) None.
    `pd_in_graph` — 살아 있는 ROS 그래프에 pd 노드가 보이는가(True/False), 그래프를 못 봤으면 None.
    """
    if status is not None and age_s is not None and age_s <= stale_s:
        return str(status.get("phase") or "")
    if pd_alive or pd_in_graph:
        return PD_UNKNOWN
    gone = pd_alive is False or pd_in_graph is False          # 없다는 **증거** — 콘솔이 띄운 단위가 죽었거나 그래프에 없다
    if status is not None and not gone:
        return PD_UNKNOWN          # 말하던 pd 가 조용해졌다 — 사라졌다는 증거 없이는 자유로 치지 않는다
    if real and (pd_in_graph is None or pd_alive is None):
        return PD_UNKNOWN          # 실기인데 없다는 증거가 없다 — 콘솔 밖에서 띄운 pd 가 있을 수 있다
    return None


@dataclass(frozen=True)
class UnitCmd:
    key: str                  # "<stage>#<n>" — 러너·감독자가 쓰는 프로세스 키와 같다
    stage: str
    index: int
    note: str
    argv: tuple[str, ...]
    kind: str                 # background | manual | foreground
    touches_real: bool
    needs: tuple[str, ...]


def index_units(mission, commands_by_stage: Mapping[str, tuple]) -> dict[str, UnitCmd]:
    """미션의 배경·수동 명령 전부. 전경 명령은 켜 둘 것이 아니라서 단위가 아니다."""
    out: dict[str, UnitCmd] = {}
    for stage in mission.stages:
        for i, c in enumerate(commands_by_stage.get(stage.id, ())):
            kind = "manual" if c.manual else ("background" if c.background else "foreground")
            if kind == "foreground":
                continue
            key = f"{stage.id}#{i}"
            out[key] = UnitCmd(key=key, stage=stage.id, index=i, note=c.note or " ".join(c.argv[:3]), argv=tuple(c.argv),
                               kind=kind, touches_real=stage.touches_real, needs=tuple(stage.needs))
    return out


def _busy(unit: "UnitCmd", busy_stage: str | None) -> list[str]:
    """제 단계가 도는 동안만 잠근다 — 러너가 같은 키를 띄우거나 지켜보는 중이다. 남의 단계(긴 episode)는 막지 않는다."""
    return [f"단계 {busy_stage} 가 실행 중이다 — 이 명령을 그 단계가 다루고 있다. 끝난 뒤에 할 것"] if busy_stage == unit.stage else []


def on_reasons(unit: UnitCmd, *, alive: bool, busy_stage: str | None, completed: Collection[str]) -> list[str]:
    """왜 지금 켤 수 없는가. 비어 있으면 켜도 된다."""
    if unit.kind == "manual":
        return ["수동 명령이다 — 콘솔은 실행하지 않는다. 운영자 셸에서 직접 띄울 것"]
    if unit.kind != "background":
        return ["배경 프로세스가 아니다"]
    if alive:
        return ["이미 떠 있다"]
    if unit.touches_real:
        return [f"실기를 건드리는 단계({unit.stage})의 명령이다 — 미션에서 승인하고 그 단계를 실행할 것"]
    waiting = [n for n in unit.needs if n not in completed]
    if waiting:
        return [f"선행 단계 {', '.join(waiting)} 가 아직 끝나지 않았다", *_busy(unit, busy_stage)]
    return _busy(unit, busy_stage)


def off_reasons(unit: UnitCmd, *, alive: bool, busy_stage: str | None, pd_phase: str | None,
                robot_unit: bool, real: bool) -> list[str]:
    """왜 지금 끌 수 없는가. `robot_unit` = 이 단위가 pd 노드를 띄운 것인가."""
    if not alive:
        return ["떠 있지 않다"]
    out = _busy(unit, busy_stage)
    guarded = robot_unit or real
    if guarded and busy_stage is not None and busy_stage != unit.stage:
        # 도는 단계가 곧 pd 를 걸 수 있다 — 상태 스냅샷(최대 STALE_S 묵음)만 믿고 끄지 않는다.
        out.append(f"단계 {busy_stage} 가 실행 중이다 — 실기/pd 단위는 단계가 끝난 뒤에 끌 것")
    if guarded and pd_phase == PD_UNKNOWN:
        out.append("pd 상태를 모른다 — 브리지/상태 토픽을 먼저 살릴 것 (떠 있는 pd 가 팔을 잡고 있을 수 있다)")
    elif guarded and pd_phase not in PD_FREE:
        what = "pd 를 죽이면 토크가 그대로 끊긴다" if robot_unit else "실기에서는 pd 가 팔을 잡은 동안 아무것도 끄지 않는다"
        out.append(f"pd 가 {pd_phase} 다 — 먼저 PD 해제를 할 것 ({what})")
    return out


def view(unit: UnitCmd, proc: Mapping | None, *, stopped: bool, why_on: list[str], why_off: list[str]) -> dict:
    """화면·그림이 읽는 모양. `proc` 은 감독자의 프로세스 행(없으면 한 번도 띄운 적 없다)."""
    alive = bool(proc and proc.get("alive"))
    return {"key": unit.key, "stage": unit.stage, "note": unit.note, "kind": unit.kind, "argv": list(unit.argv),
            "touches_real": unit.touches_real, "alive": alive, "started": proc is not None,
            "stopped": bool(stopped and not alive), "pid": None if proc is None else proc.get("pid"),
            "rc": None if proc is None else proc.get("rc"), "age_s": None if proc is None else proc.get("age_s"),
            "can_on": not why_on, "can_off": not why_off, "why_on": why_on, "why_off": why_off}


def views(units: Mapping[str, UnitCmd], procs: Mapping[str, Mapping], *, stopped: Collection[str], busy_stage: str | None,
          completed: Collection[str], pd_phase: str | None, robot_keys: Collection[str] | None,
          real: bool) -> dict[str, dict]:
    """단위 전부의 화면 모양. `robot_keys` 를 모르면(그림이 없다) 모든 단위를 pd 처럼 조심해서 다룬다.

    pd 를 두 번 띄우는 미션이 있다(좌·우 따로) — 키 하나가 아니라 **전부**를 지킨다.
    """
    out = {}
    for key, u in units.items():
        alive = bool(procs.get(key, {}).get("alive"))
        why_on = on_reasons(u, alive=alive, busy_stage=busy_stage, completed=completed)
        why_off = off_reasons(u, alive=alive, busy_stage=busy_stage, pd_phase=pd_phase,
                              robot_unit=robot_keys is None or key in robot_keys, real=real)
        out[key] = view(u, procs.get(key), stopped=key in stopped, why_on=why_on, why_off=why_off)
    return out
