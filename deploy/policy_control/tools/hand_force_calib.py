#!/usr/bin/env python3
"""hand_force_calib — RH56F1 손가락 하나씩 접촉 뒤 단계 쥐기 기록(실기 손가락 보정, 팔 없이).

10.13 s2r_1st(사용자 10.14 계획 "손가락제어 실기 확인 및 튜닝"): 실기 force_actual(액추에이터 힘의 손끝 환산값,
매뉴얼 §2.5.16 · §2.5.12 "힘팔이 짧을수록 커진다")이 sim 접촉력과 다른 양이라, 손가락별로
  접근   : 지금 각도(기준)에서 --step-reg 씩 --dwell-s 마다 닫는다 → force 가 쉬는 값 + --contact-g 를 넘으면 접촉
  누르기 : 접촉 레지스터에서 --past 목록만큼 더 깊은 목표를 차례로 --dwell-s 씩
  유지   : 마지막 목표를 --hold-s 동안(리드스크루가 힘을 쥐고 있는지 · 전류가 빠지는지)
  풀기   : 그 손가락만 기준 레지스터로
를 하고, 50 Hz 로 시각 · 단계 · 목표 · 각도 · 힘 · 전류 · 손끝 촉각 법선(0.01 N)을 CSV 에 남긴다.
다른 손가락에는 -1(그 축 유지)을 보낸다. 운영자는 고정한 컵(또는 물체)을 손가락 앞에 대 준다.

  기본 명령 = /hand_<s>/angle_set(위치, 모드 0 — 위치제어 학습 정책과 같은 경로).
  --topic angle_target 이면 마스터 어드민턴스 경로(비교용).
  중단  그 손가락 힘 > --abort-g · 전류 > --abort-ma 가 --abort-s 동안 · 상태 끊김 · Ctrl-C → 그 손가락을 기준으로.

    python3 deploy/policy_control/tools/hand_force_calib.py --side right --finger index      # 계획만(손 무동작)
    … --execute --out logs/calib/right_index.csv                                           # 실기 — 실행 직전 사용자 허락

손은 rh56f1_ecat_node 가 떠 있어야 한다(같은 도메인). 측정 전 빈손 · 편 손에서 힘 영점(--force-calibrate)을 권한다.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palm_cmd import check_domain  # noqa: E402

#: 레지스터 슬롯(드라이버 순) — rh56f1_hand_map.yaml slot_order. 손끝 촉각 5 개도 같은 순(새끼부터 · 엄지 = 4).
SLOTS = ("pinky", "ring", "middle", "index", "thumb_2")
#: 닫힘 쪽 끝(레지스터 ↓ = 굽힘) — 매뉴얼 표 36 · hand_map: 네 손가락 900, 엄지 굽힘 1100
REG_CLOSED = {"pinky": 900, "ring": 900, "middle": 900, "index": 900, "thumb_2": 1100}
LOG_HZ = 50.0
PUB_HZ = 50.0
STATE_TIMEOUT_S = 3.0
REST_S = 1.0             # 시작 뒤 이만큼 힘 평균 = 쉬는 값
ABORT_G = 1300.0         # 10.06 모드 0 포화 1.1~1.85 kg — 그 아래에서 멈춘다
ABORT_MA = 750.0         # 계약 current_limit 800 mA 바로 아래
ABORT_S = 0.3


class CalibError(ValueError):
    pass


@dataclass(frozen=True)
class Cfg:
    slot: int
    step_reg: int
    dwell_s: float
    contact_g: float
    past: tuple[int, ...]
    hold_s: float
    abort_g: float
    abort_ma: float
    abort_s: float


def make_cfg(finger: str, step_reg: int, dwell_s: float, contact_g: float, past: str, hold_s: float,
             abort_g: float, abort_ma: float, abort_s: float) -> Cfg:
    if finger not in SLOTS:
        raise CalibError(f"손가락 {finger!r} — {SLOTS} 중 하나(엄지 회전은 재지 않는다)")
    p = tuple(int(x) for x in past.split(",") if x.strip())
    if not p or any(v < 0 for v in p) or list(p) != sorted(p):
        raise CalibError(f"--past 는 0 이상 오름차순 레지스터 목록: {past!r}")
    if not 1 <= step_reg <= 50 or dwell_s <= 0 or hold_s < 0 or contact_g <= 0:
        raise CalibError("step_reg 1..50 · dwell_s > 0 · hold_s ≥ 0 · contact_g > 0")
    if not 0 < abort_g <= 2000 or not 0 < abort_ma <= 1500 or abort_s <= 0:
        raise CalibError("abort_g 0..2000 · abort_ma 0..1500 · abort_s > 0")
    return Cfg(SLOTS.index(finger), int(step_reg), float(dwell_s), float(contact_g), p, float(hold_s),
               float(abort_g), float(abort_ma), float(abort_s))


def approach_next(target: int, step_reg: int, finger: str) -> int | None:
    """다음 접근 목표(한 칸 더 닫음). 닫힘 끝을 넘으면 None — 물체에 닿지 않은 것."""
    nxt = int(target) - int(step_reg)
    return None if nxt < REG_CLOSED[finger] else nxt


def press_targets(contact_reg: int, past: tuple[int, ...], finger: str) -> list[int]:
    """접촉 레지스터에서 past 만큼 더 깊은 목표(닫힘 끝에서 자른다, 같은 값은 한 번)."""
    out: list[int] = []
    for p in past:
        v = max(int(contact_reg) - int(p), REG_CLOSED[finger])
        if not out or v != out[-1]:
            out.append(v)
    return out


def touched(force_g: float, rest_g: float, contact_g: float) -> bool:
    return float(force_g) - float(rest_g) > float(contact_g)


def over_for(history: list[tuple[float, float]], now: float, limit: float, span_s: float) -> bool:
    """(시각, 값) 기록에서 최근 span_s 동안 줄곧 limit 넘었나."""
    recent = [v for t, v in history if now - t <= span_s]
    return bool(history) and now - history[0][0] >= span_s and bool(recent) and min(recent) > limit


def describe(c: Cfg, base_reg: int | None) -> str:
    f = SLOTS[c.slot]
    b = "지금 각도" if base_reg is None else str(base_reg)
    return (f"{f}(슬롯 {c.slot}): 기준 {b} → {c.step_reg} 레지스터씩 {c.dwell_s:g} s 마다 닫음(끝 {REG_CLOSED[f]}) · "
            f"접촉 = 쉬는 힘 + {c.contact_g:g} g → 접촉 + {list(c.past)} 레지스터 각 {c.dwell_s:g} s → 유지 {c.hold_s:g} s → 기준. "
            f"중단: 힘 > {c.abort_g:g} g · 전류 > {c.abort_ma:g} mA {c.abort_s:g} s")


def run(c: Cfg, side: str, topic: str, out: Path) -> int:  # pragma: no cover - 실기
    import rclpy
    from rclpy.node import Node
    from rh56f1_interfaces.msg import GetAngleAct1, GetCurrentAct1, GetForceAct1, SetAngle1, TouchData1

    rclpy.init()
    node = Node(f"hand_force_calib_{side}")
    pub = node.create_publisher(SetAngle1, f"/hand_{side}/{topic}", 10)
    st = {"angle": None, "force": None, "cur": None, "touch": None, "t_angle": 0.0, "hand_id": 1}

    def on_angle(m):
        st["angle"], st["t_angle"], st["hand_id"] = list(m.joint_values), time.monotonic(), int(m.hand_id)

    node.create_subscription(GetAngleAct1, f"/hand_{side}/angle_actual", on_angle, 10)
    node.create_subscription(GetForceAct1, f"/hand_{side}/force_actual", lambda m: st.__setitem__("force", list(m.joint_values)), 10)
    node.create_subscription(GetCurrentAct1, f"/hand_{side}/current_actual", lambda m: st.__setitem__("cur", list(m.joint_values)), 10)
    node.create_subscription(TouchData1, f"/hand_{side}/touch_data", lambda m: st.__setitem__("touch", list(m.finger_forces)), 10)
    stop: dict[str, str | None] = {"why": None}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("why", "신호"))
    j = c.slot
    f_hist: list[tuple[float, float]] = []
    c_hist: list[tuple[float, float]] = []

    def send(reg: int) -> None:
        m = SetAngle1()
        m.header.stamp = node.get_clock().now().to_msg()
        m.hand_id = st["hand_id"]
        v = [-1] * 6
        v[j] = int(reg)
        m.joint_values = v
        pub.publish(m)

    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["t", "phase", "target", "angle", "force_g", "current_ma", "touch_normal_0.01N"])
    t0 = time.monotonic()
    last_log = [0.0]

    def tick(phase: str, target: int, t_end: float) -> bool:
        while time.monotonic() < t_end:
            send(target)
            t_next = time.monotonic() + 1.0 / PUB_HZ
            while time.monotonic() < t_next:
                rclpy.spin_once(node, timeout_sec=0.002)
            now = time.monotonic()
            if st["force"] is not None:
                f_hist.append((now, float(st["force"][j])))
            if st["cur"] is not None:
                c_hist.append((now, float(st["cur"][j])))
            del f_hist[:max(0, len(f_hist) - 500)], c_hist[:max(0, len(c_hist) - 500)]
            if now - last_log[0] >= 1.0 / LOG_HZ and st["angle"] is not None:
                last_log[0] = now
                tch = st["touch"][j] if st["touch"] is not None and j < len(st["touch"]) else ""
                w.writerow([f"{now - t0:.3f}", phase, target, st["angle"][j], (st["force"] or [""] * 6)[j],
                            (st["cur"] or [""] * 6)[j], tch])
            if stop["why"] is None and f_hist and f_hist[-1][1] > c.abort_g:
                stop["why"] = f"힘 {f_hist[-1][1]:.0f} g > {c.abort_g:g}"
            if stop["why"] is None and over_for(c_hist, now, c.abort_ma, c.abort_s):
                stop["why"] = f"전류 > {c.abort_ma:g} mA {c.abort_s:g} s"
            if stop["why"] is None and now - st["t_angle"] > STATE_TIMEOUT_S:
                stop["why"] = "손 상태 끊김"
            if stop["why"]:
                return False
        return True

    rc = 0
    try:
        t_wait = time.monotonic() + STATE_TIMEOUT_S
        while time.monotonic() < t_wait and (st["angle"] is None or st["force"] is None):
            rclpy.spin_once(node, timeout_sec=0.05)
        if st["angle"] is None or st["force"] is None:
            print(f"✗ /hand_{side}/angle_actual · force_actual 이 안 온다 — 드라이버 · 도메인 확인", flush=True)
            return 2
        base = int(st["angle"][j])
        print("[계획] " + describe(c, base), flush=True)
        ok = tick("rest", base, time.monotonic() + REST_S)
        rest = float(np.mean([v for _, v in f_hist])) if f_hist else float(st["force"][j])
        print(f"[rest] 쉬는 힘 {rest:.0f} g", flush=True)
        target, contact = base, None
        while ok and contact is None:
            nxt = approach_next(target, c.step_reg, SLOTS[j])
            if nxt is None:
                print(f"✗ 닫힘 끝 {REG_CLOSED[SLOTS[j]]} 까지 접촉 없음 — 물체 위치 확인", flush=True)
                rc = 1
                break
            target = nxt
            ok = tick("approach", target, time.monotonic() + c.dwell_s)
            if ok and touched(f_hist[-1][1] if f_hist else rest, rest, c.contact_g):
                contact = target
                print(f"[contact] 레지스터 {contact} · 각도 {st['angle'][j]} · 힘 {f_hist[-1][1]:.0f} g", flush=True)
        if ok and contact is not None:
            tgt = contact
            for tgt in press_targets(contact, c.past, SLOTS[j]):
                if not (ok := tick(f"press{contact - tgt}", tgt, time.monotonic() + c.dwell_s)):
                    break
                print(f"[press +{contact - tgt}] 목표 {tgt} · 각도 {st['angle'][j]} · 힘 {f_hist[-1][1]:.0f} g · "
                      f"전류 {(st['cur'] or [0] * 6)[j]} mA", flush=True)
            if ok:
                ok = tick("hold", tgt, time.monotonic() + c.hold_s)
                print(f"[hold] 끝 힘 {f_hist[-1][1]:.0f} g · 전류 {(st['cur'] or [0] * 6)[j]} mA", flush=True)
        if not ok:
            print(f"✗ 중단({stop['why']}) — 기준으로", flush=True)
            rc = 1
        stop["why"] = None
        for _ in range(int(PUB_HZ)):
            send(base)
            rclpy.spin_once(node, timeout_sec=1.0 / PUB_HZ)
        print(f"[release] 기준 {base} 보냄 · 기록 {out}", flush=True)
    finally:
        fh.close()
        node.destroy_node()
        rclpy.shutdown()
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--finger", choices=SLOTS, required=True)
    ap.add_argument("--topic", choices=("angle_set", "angle_target"), default="angle_set")
    ap.add_argument("--step-reg", type=int, default=5)
    ap.add_argument("--dwell-s", type=float, default=1.0)
    ap.add_argument("--contact-g", type=float, default=120.0, help="쉬는 힘 위 이만큼이면 접촉(10.06 측정과 같은 문턱)")
    ap.add_argument("--past", default="0,10,20,40,80", help="접촉 뒤 더 닫을 레지스터(오름차순)")
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--abort-g", type=float, default=ABORT_G)
    ap.add_argument("--abort-ma", type=float, default=ABORT_MA)
    ap.add_argument("--abort-s", type=float, default=ABORT_S)
    ap.add_argument("--out", default=None, help="CSV(기본 logs/calib/<날짜>_<side>_<finger>.csv)")
    ap.add_argument("--execute", action="store_true", help="실기 손에 보낸다(없으면 계획만)")
    ap.add_argument("--allow-domain-0", action="store_true")
    a = ap.parse_args(argv)
    try:
        c = make_cfg(a.finger, a.step_reg, a.dwell_s, a.contact_g, a.past, a.hold_s, a.abort_g, a.abort_ma, a.abort_s)
    except CalibError as e:
        print(f"✗ {e}")
        return 2
    print(f"계획 {a.side} /{a.topic}: " + describe(c, None))
    if not a.execute:
        print("(계획만 — 실기는 --execute)")
        return 0
    why = check_domain(dict(os.environ), a.allow_domain_0)
    if why:
        print(f"✗ {why}")
        return 2
    out = Path(a.out or f"logs/calib/{time.strftime('%Y%m%d_%H%M%S')}_{a.side}_{a.finger}.csv")
    return run(c, a.side, a.topic, out)


if __name__ == "__main__":
    sys.exit(main())
