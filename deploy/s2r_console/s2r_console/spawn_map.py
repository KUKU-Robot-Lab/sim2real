"""학습 가능 영역 지도 — 정책이 학습한 컵 소환 박스(xy)와 지금 FP++ 물체 자리를 위에서 본 그림 하나로. 순수.

10.13 사용자: "Deploy 화면에 가능영역이 오버레이로 보였으면 좋겠어". 그날 실기 121432 는 컵을 오른팔 소환 박스
y 밖 9 cm(0.342, −0.394)에 놓았고, sim 재현에서도 거의 실패했다 — 화면에서 미리 보였어야 했다.

  boxes_from   미션 artifacts + 로봇 slots → 쪽마다 rh_aglt 계약의 goal_box xy(= 소환 박스, goal_box_xy_margin 0).
               goal_box 가 없는 계약(붓기 · 옛 계약)은 건너뛴다. 읽기 실패도 건너뛴다 — 표시용이다.
  view         박스 + 물체 요약 → 물체마다 어느 박스 안인가 · 판정 tone. 판정 여유는 배포 시작 거부와 같다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

#: policy_control.pour_fj_node.START_SPAWN_TOL_M 과 같은 값(테스트가 잠근다) — 이만큼 밖이면 rh_aglt 시작이 거부된다
START_SPAWN_TOL_M = 0.02
SIDES = ("right", "left")
_KO = {"right": "오른팔", "left": "왼팔"}


def object_name(topic: str) -> str:
    parts = [p for p in topic.split("/") if p]
    return parts[1] if len(parts) >= 3 and parts[0] == "objects" else topic


def boxes_from(artifacts: Mapping[str, str], slots: Mapping[str, str], repo: Path) -> list[dict]:
    out = []
    for side in SIDES:
        key = slots.get(side)
        rel = artifacts.get(key) if key else None
        if not rel:
            continue
        try:
            c = json.loads((Path(repo) / str(rel)).read_text())
            lo, hi = [float(v) for v in c["goal_box_min"][:2]], [float(v) for v in c["goal_box_max"][:2]]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(lo) != 2 or len(hi) != 2 or lo[0] >= hi[0] or lo[1] >= hi[1]:
            continue
        out.append({"side": side, "label": _KO[side], "policy": Path(str(rel)).parent.name,
                    "lo": [round(v, 4) for v in lo], "hi": [round(v, 4) for v in hi]})
    return out


def _inside(xy, box, tol: float) -> bool:
    return all(box["lo"][i] - tol <= xy[i] <= box["hi"][i] + tol for i in range(2))


def _margin_m(xy, box) -> float:
    """박스 밖으로 나간 거리(안이면 0) — 축마다 넘친 양의 최댓값."""
    return max(max(box["lo"][i] - xy[i], xy[i] - box["hi"][i], 0.0) for i in range(2))


def view(boxes: list[dict], objects: Mapping[str, Mapping | None], stale: bool,
         tol: float = START_SPAWN_TOL_M) -> dict | None:
    """objects = {토픽: PoseWindow.summary | None}. 박스가 하나도 없으면 None(그릴 것이 없다)."""
    if not boxes:
        return None
    objs = []
    for topic, s in objects.items():
        if not s or not s.get("pos") or stale:
            continue
        xy = [float(s["pos"][0]), float(s["pos"][1])]
        inside = [b["side"] for b in boxes if _inside(xy, b, 0.0)]
        near = [b["side"] for b in boxes if _inside(xy, b, tol)]
        if inside:
            tone, say = "ok", " · ".join(_KO[x] for x in inside) + " 범위 안"
        elif near:
            tone, say = "warn", " · ".join(_KO[x] for x in near) + f" 경계 밖 {tol * 100:.0f} cm 이내 — 시작은 받지만 안쪽이 낫다"
        else:
            d = min(_margin_m(xy, b) for b in boxes)
            tone, say = "bad", f"모든 팔 범위 밖 {d * 100:.1f} cm — 시작이 거부된다, 컵을 안으로"
        objs.append({"name": object_name(topic), "x": round(xy[0], 4), "y": round(xy[1], 4),
                     "age_s": s.get("age_s"), "tone": tone, "say": say})
    return {"boxes": boxes, "tol_m": tol, "objects": objs, "stale": stale}
