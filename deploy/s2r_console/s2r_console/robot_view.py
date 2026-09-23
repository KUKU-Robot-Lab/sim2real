"""로봇 상태 표 — 팔 7축 · 손 5지 4관절을 계약의 목표와 나란히 놓는다. 순수(ROS·파일 접근 없음).

09.23 사용자: "로봇 joint state 들에 대한 정보가 그래프나 표가 있으면 좋겠는데". 그날 손가락이 계약 홈
(굽힘 관절 하한 0.0)으로 밀려 꺾였는데, 화면에는 그 값이 어디에도 보이지 않아 손으로 재야 했다.

입력은 브리지가 준 `{관절: [위치, 속도, 토크]}` 와 계약의 팔·손 홈이다. 여기서는 **읽은 것만** 늘어놓는다 —
무엇이 이상한지는 `state` 한 글자로 표시하고(ok · limit · off), 판정 기준은 아래 상수뿐이다.

09.23 사용자: "디폴트는 joint state 고, vel 이나 effort 들도. 나중에 finger tip sensor 도 써야 할 수 있어서
미리 만들기" — 줄마다 채널을 **전부** 담고 화면이 하나를 고른다. 새 채널(촉각 등)은 `CHANNELS` 에 이름과
단위를 더하고 브리지가 그 값을 보내면 그대로 뜬다. 값이 없는 채널은 빈 칸이다(0 이 아니다).
"""
from __future__ import annotations

from typing import Mapping, Sequence

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
#: 화면이 고를 수 있는 값 채널 — (키, 이름, 단위). `pos` 만 목표·한계와 견준다.
#: 손끝 촉각은 그 값을 내는 원이 생기면 `("tactile", "촉각", "N")` 을 더하면 된다(09.23 대비).
CHANNELS = (("pos", "위치", "rad"), ("vel", "속도", "rad/s"), ("eff", "토크", "N·m"))
#: 목표와 이만큼 벌어지면 화면에서 굵게 — 팔은 pd 정착 공차(0.01)의 5배, 손은 속도 제한 한 틱 몫
ARM_OFF_RAD = 0.05
HAND_OFF_RAD = 0.10
#: 관절 한계에서 이 안쪽이면 "끝점" 으로 표시한다 — 09.23 손가락이 꺾인 그 자리
LIMIT_NEAR_RAD = 0.02


def arm_joints(side: str) -> list[str]:
    return [f"{side[0]}_aj_{i}" for i in range(1, 8)]


def hand_joints(side: str) -> list[str]:
    return [f"{side[0]}_hj_{f}_{j}" for f in FINGERS for j in range(1, 5)]


def _chan(q: Sequence | None, i: int) -> float | None:
    """브리지가 준 [위치, 속도, 토크, …] 의 i 번째. 없으면 None — 0 으로 채우지 않는다."""
    if not q or i >= len(q) or q[i] is None:
        return None
    return float(q[i])


def _row(name: str, q: Sequence | None, target: float | None, limits: Sequence | None,
         off_rad: float) -> dict:
    vals = {key: _chan(q, i) for i, (key, _, _) in enumerate(CHANNELS)}
    pos, eff = vals["pos"], vals["eff"]
    err = None if (pos is None or target is None) else pos - target
    lo, hi = (None, None) if not limits else (float(limits[0]), float(limits[1]))
    state = "missing" if pos is None else "ok"
    if pos is not None and lo is not None and hi is not None \
            and (pos - lo <= LIMIT_NEAR_RAD or hi - pos <= LIMIT_NEAR_RAD):
        state = "limit"
    if err is not None and abs(err) > off_rad:
        state = "off"
    return {"joint": name, "pos": pos, "effort": eff, "vals": vals, "target": target, "err": err,
            "lower": lo, "upper": hi, "state": state}


def group(title: str, names: Sequence[str], joints: Mapping[str, Sequence],
          targets: Mapping[str, float], limits: Mapping[str, Sequence], off_rad: float) -> dict:
    rows = [_row(n, joints.get(n), targets.get(n), limits.get(n), off_rad) for n in names]
    seen = [r for r in rows if r["pos"] is not None]
    worst = max((r for r in rows if r["err"] is not None), key=lambda r: abs(r["err"]), default=None)
    return {"title": title, "rows": rows, "seen": len(seen), "total": len(rows),
            "worst": None if worst is None else worst["joint"],
            "worst_err": None if worst is None else worst["err"],
            "limit": [r["joint"] for r in rows if r["state"] == "limit"]}


def view(joints: Mapping[str, Sequence], contract_sides: Mapping[str, Mapping],
         limits: Mapping[str, Sequence], *, age_s: float | None) -> dict:
    """화면이 그대로 그리는 모양. `contract_sides` 는 계약의 `sides`(팔·손 홈이 목표다)."""
    groups = []
    for side in ("right", "left"):
        if side not in contract_sides:            # 계약에 없는 팔은 그리지 않는다(한 팔짜리 계약)
            continue
        cfg = contract_sides.get(side) or {}
        arm_t = dict(zip(cfg.get("arm_joints") or arm_joints(side), cfg.get("home_arm") or []))
        hand_t = {str(k): float(v) for k, v in (cfg.get("home_hand") or {}).items()}
        ko = "오른" if side == "right" else "왼"
        groups.append(group(f"{ko}팔", cfg.get("arm_joints") or arm_joints(side), joints, arm_t, limits, ARM_OFF_RAD))
        groups.append(group(f"{ko}손", cfg.get("hand_joints") or hand_joints(side), joints, hand_t, limits,
                            HAND_OFF_RAD))
    return {"groups": [g for g in groups if g["total"]],
            "channels": [{"key": k, "name": n, "unit": u} for k, n, u in CHANNELS],
            "age_s": None if age_s is None else round(age_s, 2),
            "stale": age_s is None or age_s > 2.0}
