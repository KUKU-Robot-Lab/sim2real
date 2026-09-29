"""FP++ 건강 칸 — 09.29 사용자: "fpp 가 제대로 되는지 … 상태창에서 확인할 수 있으면 좋겠는데".

그날 FP++ 는 컨테이너가 Up 인 채 추적 노드가 CUDA OOM 으로 죽어 자세를 하나도 내지 않았다(학습과 GPU 공유).
전날 2회차는 선 컵을 174° 로 뒤집어 본 채 정책이 시작됐다. 이 칸이 둘 다 정책 전에 보여야 한다.
"""
from __future__ import annotations

import math

import pytest

from s2r_console.object_health import PoseWindow, object_name, view

pytestmark = pytest.mark.unit

UP = (1.0, 0.0, 0.0, 0.0)
CUP = "/objects/cup_big_s100/pose"


def tilted(deg: float):
    h = math.radians(deg) / 2
    return (math.cos(h), math.sin(h), 0.0, 0.0)


def stream(hz=13.0, lat=0.3, quat=UP, secs=2.0, t0=100.0, jitter_mm=0.0):
    w = PoseWindow()
    n = int(secs * hz)
    for i in range(n):
        t = t0 + i / hz
        dx = jitter_mm * 1e-3 * (1 if i % 2 else -1)
        w.add(t, t - lat, (0.26 + dx, 0.11, 0.287), quat)
    return w, t0 + (n - 1) / hz


def percept(container="Up 3 minutes", crash=None, used=19000, total=24576):
    return {"objects": {"cup_big_s100": {"container": container, "pose_age_s": 0.1, "crash": crash}},
            "gpu": {"used_mib": used, "total_mib": total}, "error": None}


def rows(v):
    return {r["label"]: r for r in v["objects"][0]["rows"]}


def test_a_healthy_standing_cup_is_ok():
    w, now = stream()
    s = w.summary(now)
    assert s["hz"] == pytest.approx(13.0, abs=1.0) and s["lat_med_s"] == pytest.approx(0.3, abs=1e-6)
    assert s["tilt_deg"] < 0.1 and s["still_mm"] < 0.1
    v = view({CUP: s}, 0.5, percept())
    assert v["tone"] == "ok" and v["objects"][0]["name"] == "cup_big_s100"


def test_a_crashed_tracker_in_an_up_container_is_bad_even_without_poses():
    """09.29: 컨테이너 Up · 추적 노드 OOM · 자세 0."""
    v = view({}, 0.5, percept(crash="torch.OutOfMemoryError: CUDA out of memory", used=23934))
    r = rows(v)
    assert v["tone"] == "bad"
    assert r["추적 노드"]["tone"] == "bad" and "OutOfMemoryError" in r["추적 노드"]["value"]
    assert r["GPU (vision-3090)"]["tone"] == "bad"                 # 남은 0.6 GB < 1 GB
    assert r["자세 수신"]["tone"] == "bad"


def test_a_flipped_cup_is_bad_and_says_why():
    """09.28 2회차: 선 컵을 174° 로 봤다."""
    w, now = stream(quat=tilted(174.0))
    r = rows(view({CUP: w.summary(now)}, 0.5, percept()))
    assert r["기울기"]["tone"] == "bad" and "174" in r["기울기"]["value"]


def test_slow_late_or_stale_poses_are_flagged():
    w, now = stream(hz=4.0)
    assert rows(view({CUP: w.summary(now)}, 0.5, percept()))["주기"]["tone"] == "bad"
    w, now = stream(hz=8.0)
    assert rows(view({CUP: w.summary(now)}, 0.5, percept()))["주기"]["tone"] == "warn"
    w, now = stream(lat=1.2)
    assert rows(view({CUP: w.summary(now)}, 0.5, percept()))["지연"]["tone"] == "bad"
    w, now = stream()
    s = w.summary(now + 3.0)                                        # 3 s 동안 안 왔다
    assert s["hz"] == 0.0 and rows(view({CUP: s}, 0.5, percept()))["자세 수신"]["tone"] == "bad"


def test_jitter_is_only_a_warning_because_a_held_cup_moves():
    w, now = stream(jitter_mm=8.0)
    r = rows(view({CUP: w.summary(now)}, 0.5, percept()))
    assert r["흔들림 (2 s)"]["tone"] == "warn"


def test_no_launcher_status_is_said_not_guessed():
    w, now = stream()
    v = view({CUP: w.summary(now)}, 0.5, None)
    r = rows(v)
    assert r["추적 노드"]["tone"] == "mute" and r["GPU (vision-3090)"]["tone"] == "mute"
    assert v["tone"] == "ok"                                         # 모르는 것은 나쁨으로 치지 않는다


def test_a_stale_bridge_report_is_bad():
    w, now = stream()
    assert view({CUP: w.summary(now)}, 5.0, percept())["tone"] == "bad"


def test_object_name_comes_from_the_topic():
    assert object_name(CUP) == "cup_big_s100"
