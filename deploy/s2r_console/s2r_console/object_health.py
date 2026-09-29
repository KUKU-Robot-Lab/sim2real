"""FP++ 건강 — 컵 자세 흐름(브리지가 잰 것)과 vision-3090 상태(인지 런처가 읽은 것)를 상태창 한 칸으로. 순수.

09.29 사용자: "fpp 가 제대로 되는지 확인하는 걸 … 상태창에서 확인할 수 있으면 좋겠는데".
그날 FP++ 는 컨테이너가 Up 인 채 추적 노드가 CUDA OOM(학습과 GPU 공유)으로 죽어 자세를 하나도 내지 않았고,
전날 2회차는 선 컵을 174° 로 뒤집어 본 채 정책이 시작됐다. 이 칸은 둘 다 정책 전에 보이게 한다.

  PoseWindow  브리지가 자세 한 개마다 add(받은 시각, 메시지 시각, 위치, 쿼터니언 w x y z) — 최근 WINDOW_S 만 든다.
              지연 = 받은 시각 − 메시지 시각(두 PC 시계가 맞다는 가정, 09.28 실측 0.3 s 로 일정했다).
  view        브리지 요약 + 런처 상태 → 줄마다 {label, value, tone}, 칸 전체 tone = 가장 나쁜 줄.
              모르는 것(런처 상태 없음)은 'mute' — 나쁨으로 치지 않고 모른다고 쓴다.
"""
from __future__ import annotations

import math
from collections import deque
from statistics import median, pstdev
from typing import Mapping, Sequence

WINDOW_S = 2.0
HZ_OK, HZ_MIN = 10.0, 5.0            # 09.28 실기 13 Hz. 정책은 60 Hz 로 최신값을 쓴다
LAT_OK_S, LAT_BAD_S = 0.5, 1.0       # 09.28 실기 0.3 s
STALE_S = 1.0                        # 이만큼 안 오면 끊긴 것
TILT_MAX_DEG = 15.0                  # joint_node 리셋 · 시작 검사(START_TILT_MAX_DEG)와 같은 값
STILL_MM = 5.0                       # 선 컵의 최근 흔들림. 손에 든 컵은 움직이니 경고까지만
GPU_FREE_MIN_MIB = 1024              # 09.29 FP++ 가 296 MiB 를 더 잡다 죽었다(남은 157 MiB)
REPORT_STALE_S = 3.0                 # 브리지 보고가 이보다 오래되면 칸 전체를 믿지 않는다
_RANK = {"mute": -1, "ok": 0, "warn": 1, "bad": 2}   # 모름만 있으면 mute, 하나라도 확인되면 그 판정


def object_name(topic: str) -> str:
    """/objects/<이름>/pose → <이름>."""
    parts = [p for p in topic.split("/") if p]
    return parts[1] if len(parts) >= 3 and parts[0] == "objects" else topic


def tilt_deg(quat: Sequence[float]) -> float:
    """물체 z 축과 세계 z 의 각 — 학습의 넘어짐 판정과 같다(fj_core_env._get_dones)."""
    _, x, y, _ = quat
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


class PoseWindow:
    def __init__(self, keep_s: float = WINDOW_S) -> None:
        self.keep_s = keep_s
        self._rows: deque = deque()

    def add(self, recv: float, stamp: float, pos: Sequence[float], quat: Sequence[float]) -> None:
        self._rows.append((float(recv), float(stamp), tuple(map(float, pos)), tuple(map(float, quat))))
        while self._rows and self._rows[0][0] < recv - self.keep_s:
            self._rows.popleft()

    def summary(self, now: float) -> dict | None:
        """None = 한 번도 못 받았다."""
        if not self._rows:
            return None
        recent = [r for r in self._rows if r[0] >= now - self.keep_s]
        last = self._rows[-1]
        out = {"hz": round(len(recent) / self.keep_s, 1), "age_s": round(now - last[0], 2),
               "pos": [round(v, 4) for v in last[2]], "tilt_deg": round(tilt_deg(last[3]), 1),
               "lat_med_s": None, "lat_max_s": None, "gap_max_s": None, "still_mm": None}
        if recent:
            lat = [r[0] - r[1] for r in recent]
            out.update(lat_med_s=round(median(lat), 3), lat_max_s=round(max(lat), 3))
            recv = [r[0] for r in recent]
            out["gap_max_s"] = round(max((b - a for a, b in zip(recv, recv[1:])), default=0.0), 3)
        if len(recent) >= 3:
            out["still_mm"] = round(max(pstdev(r[2][i] for r in recent) for i in range(3)) * 1e3, 1)
        return out


def _row(label: str, value: str, tone: str) -> dict:
    return {"label": label, "value": value, "tone": tone}


def _launcher_rows(name: str, perception: Mapping | None) -> list[dict]:
    if perception is None:
        return [_row("추적 노드", "인지 런처 상태 없음 — sensors 단계가 떠 있는가", "mute"),
                _row("GPU (vision-3090)", "모른다", "mute")]
    info = (perception.get("objects") or {}).get(name) or {}
    cont, crash = info.get("container"), info.get("crash")
    if crash:
        node = _row("추적 노드", f"죽었다 — {crash}", "bad")
    elif not cont:
        node = _row("추적 노드", "컨테이너 없음", "bad")
    elif not str(cont).startswith("Up"):
        node = _row("추적 노드", f"컨테이너 {cont}", "bad")
    else:
        node = _row("추적 노드", f"컨테이너 {cont}", "ok")
    gpu = perception.get("gpu") or {}
    used, total = gpu.get("used_mib"), gpu.get("total_mib")
    if used is None or total is None:
        mem = _row("GPU (vision-3090)", "모른다(옛 status.sh)", "mute")
    else:
        free = total - used
        mem = _row("GPU (vision-3090)", f"{used / 1024:.1f} / {total / 1024:.1f} GB · 여유 {free / 1024:.1f} GB",
                   "bad" if free < GPU_FREE_MIN_MIB else "ok")
    return [node, mem]


def _stream_rows(s: Mapping | None) -> list[dict]:
    if s is None:
        return [_row("자세 수신", "한 번도 안 왔다", "bad")]
    if s["age_s"] > STALE_S:
        recv = _row("자세 수신", f"끊김 — 마지막 {s['age_s']:.1f} s 전", "bad")
    else:
        recv = _row("자세 수신", f"마지막 {s['age_s'] * 1e3:.0f} ms 전", "ok")
    hz = s["hz"]
    out = [recv, _row("주기", f"{hz:.1f} Hz (기준 ≥ {HZ_OK:.0f})",
                      "ok" if hz >= HZ_OK else "warn" if hz >= HZ_MIN else "bad")]
    lat = s.get("lat_med_s")
    if lat is not None:
        out.append(_row("지연", f"{lat:.2f} s (최대 {s['lat_max_s']:.2f})",
                        "ok" if lat <= LAT_OK_S else "warn" if lat <= LAT_BAD_S else "bad"))
    t = s["tilt_deg"]
    out.append(_row("기울기", f"{t:.1f}° (선 컵 ≤ {TILT_MAX_DEG:.0f}°, 넘으면 리셋이 거부)",
                    "ok" if t <= TILT_MAX_DEG else "bad"))
    if s.get("still_mm") is not None:
        out.append(_row("흔들림 (2 s)", f"{s['still_mm']:.1f} mm", "ok" if s["still_mm"] <= STILL_MM else "warn"))
    x, y, z = s["pos"]
    out.append(_row("위치 (base)", f"x {x:.3f} · y {y:.3f} · z {z:.3f} m", "ok"))
    return out


def worst(tones) -> str:
    return max(tones, key=lambda t: _RANK.get(t, 0), default="mute")


def view(objects: Mapping[str, Mapping | None], report_age_s: float | None, perception: Mapping | None,
         topics: Sequence[str] = ()) -> dict:
    """objects = {토픽: PoseWindow.summary | None}. topics 는 요약이 아직 없어도 칸을 낼 토픽."""
    stale = report_age_s is None or report_age_s > REPORT_STALE_S
    names = list(dict.fromkeys([*objects, *topics])) or [
        f"/objects/{n}/pose" for n in ((perception or {}).get("objects") or {})]
    out = []
    for topic in names:
        name = object_name(topic)
        rows = _launcher_rows(name, perception) + _stream_rows(None if stale else objects.get(topic))
        tone = worst(r["tone"] for r in rows)
        out.append({"name": name, "topic": topic, "rows": rows, "tone": tone})
    tone = worst(o["tone"] for o in out) if out else "mute"
    say = {"ok": "정책 입력으로 써도 된다", "warn": "주의 — 값은 오지만 기준 밖이 있다",
           "bad": "정책을 돌리지 말 것", "mute": "FP++ 자세 토픽이 없다"}[tone]
    return {"tone": tone, "say": say, "stale": stale, "objects": out}
