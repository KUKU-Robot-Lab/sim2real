#!/usr/bin/env python3
"""sim = real 머리 시야 피드백 — IMU 중력과 /head/joint_states 를 보며 tilt 를 맞춘다(--execute 일 때만 움직인다).

10.30 사용자: 최종 목표는 정책이 원하는 시야(sim 카메라)와 실기 시야를 같게 하는 것이고, 그 피드백 제어를 IMU 와 joint state 로 한다.
11.03 사용자: "FP++ 를 받는 것도 중요하지만 sim 과 real 의 카메라 시야가 같아야 의미가 있다" — 기준을 camset 으로 옮겼다.

  기준  외부 파라미터 config/global_camera_extrinsics_arm4090_r20u5.yaml(camset, 미션 extrinsics). 10.28 실기 시야를 sim 시야에
        맞춘 자세(멈춘 자리 1825/2815)에서 잰 값이고 FP++ base 좌표가 이 값으로만 맞다.
        sim 카메라 출처 = hdgp scripts/tools/e2e_chain/cocktail/headcam.py --static global_camera_extrinsics_arm4090.yaml(10.27 home)
        --static_rel -20,-5(URDF 목 FK 근사) → videos/cocktail_1028_b/headcam_static/headcam_1028.json 의 static_dpan-20_dtilt-5,
        숙임 67.67°. r20u5 숙임과의 차(10.28 볼트 맞춤 잔차)는 설정 sim_pitch_deg 로 같이 찍는다.
  측정  카메라 숙임 = D435i 가속도(중력)로 잰 광축의 아래 각. 처짐 · 조립 유격 · 손으로 건드린 것까지 보인다(인코더는 못 본다).
        yaw(pan 방향 어긋남)는 중력으로 못 잰다. 인코더 각 = /head/joint_states(URDF 규약 → 인코더 deg 로 되돌린다).
  검사  --check: 움직이지 않고 |IMU 숙임 − 기준 숙임| ≤ 설정 view_check_tol_deg 인지만 본다(미션 head_view_check 단계).
  목표  --target extrinsics(기본): 외부 파라미터 카메라의 숙임. --target <deg>: 숙임 각을 직접(0 = 정면 수평, 상한 찾기용).
  제어  남은 숙임 오차를 인코더 이동으로 바꿔 head_move 요청을 보낸다. 비율(Δ숙임/Δ인코더)은 매 단계 다시 잰다.
        한 번에 --max-step° 이하, tilt 는 [--min-tilt, --max-tilt] 안, 비율이 RATIO_OK 밖이면 멈춘다.
  IMU 0점 imu_pitch_bias = (시야를 맞춘 자세의 IMU 숙임) − (외부 파라미터 숙임). D435i IMU 는 공장 보정이 없어 0점이 치우친다.
        카메라에 붙은 값이라 기준 파일을 바꿔도 그대로 쓴다. --calib-bias 는 지금 자세가 기준 시야와 맞았다고 보고 다시 잰다.

  python3 scripts/head_view_feedback.py --check                 # 시야 검사만(미션 단계): 지금 숙임 · 기준 · 차 · 판정
  python3 scripts/head_view_feedback.py                         # 재기만: 지금 숙임 · 목표 · 남은 오차 · 다음 tilt
  python3 scripts/head_view_feedback.py --execute               # 기준 시야로 맞춘다(목이 움직인다 — 사용자 허락 뒤)
  python3 scripts/head_view_feedback.py --execute --target 0 --min-tilt -8   # 정면 수평 찾기(상한)
  python3 scripts/head_view_feedback.py --calib-bias            # IMU 0점 다시 재기

ROS: ROS_DOMAIN_ID 는 드라이버 · 카메라와 같아야 한다. 목 드라이버(head_joint_publisher)와 realsense(IMU 켬,
scripts/vision/camera_up.sh)가 떠 있어야 한다.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

REPO = _HERE.parent
DEFAULT_EXTRINSICS = REPO / "config" / "global_camera_extrinsics_arm4090_r20u5.yaml"   # camset = 미션 extrinsics
DEFAULT_CONFIG = REPO / "config" / "head_view_feedback_arm4090.yaml"
IMU_TOPIC = "/camera/camera/imu"
JOINT_TOPIC = "/head/joint_states"
#: 10.30 처짐 지도: 36~61° 구간은 처짐이 1.5 → 3.8° 로 바뀌어 구간 비율이 0.84~1.22 로 정상이다. 그 밖은 1.00 ± 0.03.
RATIO_OK = (0.7, 1.3)


# ---------- 순수부(ROS · 하드웨어 무관) ----------

def gravity_angles(ax: float, ay: float, az: float) -> tuple[float, float]:
    """광학 프레임(x 오른쪽 · y 아래 · z 앞) 가속도 → (숙임 deg, 옆 기울기 deg). 정지 때 가속도 = 위쪽 단위벡터 × g."""
    g = math.sqrt(ax * ax + ay * ay + az * az)
    if g < 5.0:
        raise ValueError(f"가속도 크기 {g:.2f} m/s² — 중력이 아니다(IMU 이상)")
    return math.degrees(math.asin(max(-1.0, min(1.0, -az / g)))), math.degrees(math.atan2(ax, -ay))


def quat_to_matrix(wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def extrinsics_angles(wxyz) -> tuple[float, float]:
    """base ← 광학 프레임 회전 → IMU 와 같은 정의의 (숙임, 옆 기울기). base z 가 중력 위쪽이라고 본다."""
    up = quat_to_matrix(wxyz)[2, :]  # base 위쪽을 카메라 프레임으로 = R^T e_z
    return gravity_angles(*(up * 9.81))


def load_extrinsics_pitch(path: Path = DEFAULT_EXTRINSICS) -> tuple[float, float]:
    cam = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["camera"]
    return extrinsics_angles(cam["orientation_wxyz"])


def view_error(raw_pitch: float, bias: float, ref_pitch: float, tol: float) -> tuple[float, bool]:
    """(IMU 숙임 − 0점) − 기준 숙임, 그리고 |차| ≤ tol 인가. +면 기준보다 숙였다."""
    err = raw_pitch - bias - ref_pitch
    return err, abs(err) <= tol


RECALIB_EPILOG = """머리를 --execute 로 옮겼으면 그 자세에서 외부 파라미터를 다시 재야 FP++ 좌표가 맞는다(옮긴 자리는 head_pose ±4 틱 밖일 수
있고, 그러면 object_pose_node 가 HeadOffCalibPose 로 발행을 멈춘다). 로봇을 움직이지 않고, 목 · 카메라가 붙은 PC 에서:
  1. 목 드라이버(head_joint_publisher)를 내린다 — --head-config 가 포트를 직접 읽어 틱을 head_pose 에 적는다(포트 단독 소유).
  2. python3 scripts/calib/table_cad_extrinsics.py --init config/global_camera_extrinsics_arm4090_r20u5.yaml \\
             --write --head-config config/head_home_rh56f1.yaml
  3. 새 head_pose 틱에 맞춰 config/head_camset_rh56f1.yaml 의 deg 를 고친다(tests/s2r_console/test_rh56f1_mission.py 가 비교).
  4. 드라이버를 다시 띄우고 그 자세에서 --calib-bias 로 IMU 0점을 다시 잰 뒤 --check 가 ✓ 인지 본다.
자세한 절차: docs/HOST_arm4090.md "시야 검사"."""


@dataclass(frozen=True)
class Limits:
    min_tilt: float = 60.0     # 인코더 deg — 작을수록 머리가 든다
    max_tilt: float = 74.8
    max_step: float = 10.0
    tol: float = 0.2           # 숙임 deg
    max_iters: int = 12


def next_tilt(enc: float, pitch_err: float, ratio: float, lim: Limits) -> float:
    """남은 숙임 오차(측정 − 목표, +면 너무 숙임) → 다음 인코더 목표. 큰 오차는 max_step 으로 끊고 가까우면 90 % 만 간다."""
    step = pitch_err / ratio
    step = max(-lim.max_step, min(lim.max_step, step if abs(step) > 2.0 else 0.9 * step))
    return max(lim.min_tilt, min(lim.max_tilt, enc - step))


def ratio_from(prev: tuple[float, float], cur: tuple[float, float], fallback: float) -> float:
    """(인코더, 숙임) 두 점 → Δ숙임/Δ인코더. 이동이 0.5° 미만이면 이전 값을 쓴다."""
    d_enc = cur[0] - prev[0]
    return (cur[1] - prev[1]) / d_enc if abs(d_enc) >= 0.5 else fallback


@dataclass
class Step:
    tilt_cmd: float
    enc_pan: float
    enc_tilt: float
    pitch: float       # IMU 0점 보정 뒤 숙임(외부 파라미터와 같은 기준)
    roll: float
    err: float
    ratio: float


# ---------- ROS 부 ----------

class Sensors:
    def __init__(self, node, rclpy) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu, JointState
        from head_fk_chain import encoder_from_urdf
        self._node, self._rclpy, self._e2u = node, rclpy, encoder_from_urdf
        self._acc: list[tuple[float, float, float]] = []
        self.joint: tuple[float, float] | None = None
        node.create_subscription(Imu, IMU_TOPIC, self._on_imu, qos_profile_sensor_data)
        node.create_subscription(JointState, JOINT_TOPIC, self._on_joint, 10)

    def _on_imu(self, m) -> None:
        a = m.linear_acceleration
        self._acc.append((a.x, a.y, a.z))

    def _on_joint(self, m) -> None:
        pos = dict(zip(m.name, m.position))
        self.joint = self._e2u(math.degrees(pos["head_j_pan"]), math.degrees(pos["head_j_tilt"]))

    def read(self, n: int = 400, settle_s: float = 0.5, timeout_s: float = 8.0) -> tuple[float, float, float, float, float]:
        """(숙임, 옆, IMU 흔들림 σ m/s², 인코더 pan, 인코더 tilt) — 멈춘 뒤 n 개 평균."""
        end = time.monotonic() + settle_s
        while time.monotonic() < end:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        self._acc.clear()
        end = time.monotonic() + timeout_s
        while (len(self._acc) < n or self.joint is None) and time.monotonic() < end:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        if len(self._acc) < n // 4:
            raise RuntimeError(f"IMU {IMU_TOPIC} 가 안 들어온다({len(self._acc)} 개) — realsense 를 IMU 켜고 띄울 것")
        if self.joint is None:
            raise RuntimeError(f"{JOINT_TOPIC} 가 안 들어온다 — 목 드라이버를 띄울 것")
        cols = list(zip(*self._acc))
        mean = [statistics.fmean(c) for c in cols]
        sd = max(statistics.pstdev(c) for c in cols)
        pitch, roll = gravity_angles(*mean)
        return pitch, roll, sd, self.joint[0], self.joint[1]


def sweep_targets(start: float, step: float, lim: Limits) -> list[float]:
    """start 에서 step 씩 한계까지(한계 자체 포함)."""
    if step == 0:
        raise ValueError("step 이 0 이다")
    out, t = [], start
    while True:
        t = max(lim.min_tilt, min(lim.max_tilt, t + step))
        if out and abs(t - out[-1]) < 1e-9 or abs(t - start) < 1e-9 and not out:
            return out
        out.append(t)


def grid_plan(pans: list[float], tilts: list[float]) -> list[tuple[float, float, str]]:
    """pan 마다 tilts 순서대로(up) 갔다가 거꾸로(down). 같은 자리 두 번 재기는 뺀다. (pan, tilt, 방향) 목록."""
    plan = []
    for pan in pans:
        plan += [(pan, t, "up") for t in tilts]
        plan += [(pan, t, "down") for t in tilts[-2::-1]]
    return plan


def _grid(args, lim: Limits, sens, head_move, bias: float) -> int:
    pans = [float(v) for v in args.grid_pans.split(",")]
    tilts = [float(v) for v in args.grid_tilts.split(",")]
    for t in tilts:
        if not lim.min_tilt <= t <= lim.max_tilt:
            raise SystemExit(f"tilt {t} 가 [{lim.min_tilt}, {lim.max_tilt}] 밖")
    plan = grid_plan(pans, tilts)
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    prev = None
    print(f"{len(plan)} 자세 · pan {pans} · tilt {tilts[0]}→{tilts[-1]} 양방향", flush=True)
    for i, (pan, tilt, d) in enumerate(plan):
        rc, _ = head_move.request_move(pan, tilt, max_tilt=lim.max_tilt)
        if rc == 2:
            print("❌ 드라이버가 요청을 거절했다"); return 1
        raw, roll, sd, ep, et = sens.read()
        row = {"i": i, "pan_cmd": pan, "tilt_cmd": tilt, "dir": d, "enc_pan": round(ep, 3), "enc_tilt": round(et, 3),
               "pitch": round(raw - bias, 3), "roll": round(roll, 3), "sd": round(sd, 4), "sag": round(raw - bias - et, 3)}
        if out:
            out.write(json.dumps(row) + "\n"); out.flush()
        print(f"[{i + 1}/{len(plan)}] pan {ep:+6.2f} tilt {et:+6.2f} {d:4} 숙임 {row['pitch']:+6.2f} 처짐 {row['sag']:+5.2f} 옆 {roll:+5.2f}",
              flush=True)
        if prev and prev["pan_cmd"] == pan and abs(et - prev["enc_tilt"]) >= 2:
            r = (row["pitch"] - prev["pitch"]) / (et - prev["enc_tilt"])
            if not 0.5 <= r <= 1.5:
                print(f"❌ 비율 {r:.3f} — 기구 이상 의심, 멈춘다"); return 1
        prev = row
    return 0


def _sweep(args, lim: Limits, sens, head_move, pan: float, tilt: float, pitch: float, bias: float) -> int:
    rows = [{"tilt_cmd": None, "enc_tilt": tilt, "pitch": pitch}]
    print(f"{'목표':>8} {'인코더':>8} {'숙임':>8} {'구간 비율':>8} {'누적 비율':>8}")
    print(f"{'-':>8} {tilt:+8.2f} {pitch:+8.2f}")
    code = 0
    for cmd in sweep_targets(tilt, args.sweep, lim):
        rc, _ = head_move.request_move(pan, cmd, max_tilt=lim.max_tilt)
        if rc == 2:
            print("❌ 드라이버가 요청을 거절했다"); code = 1; break
        raw, roll, sd, pan2, tilt2 = sens.read()
        p2 = raw - bias
        r = ratio_from((rows[-1]["enc_tilt"], rows[-1]["pitch"]), (tilt2, p2), float("nan"))
        r0 = ratio_from((rows[0]["enc_tilt"], rows[0]["pitch"]), (tilt2, p2), float("nan"))
        rows.append({"tilt_cmd": cmd, "enc_tilt": tilt2, "enc_pan": pan2, "pitch": p2, "roll": roll, "sd": sd, "ratio": r})
        print(f"{cmd:+8.2f} {tilt2:+8.2f} {p2:+8.2f} {r:8.3f} {r0:8.3f}  옆 {roll:+.2f}", flush=True)
        if not 0.5 <= r <= 1.5:
            print(f"❌ 구간 비율 {r:.3f} — 기구 이상 의심, 멈춘다"); code = 1; break
        if p2 <= args.sweep_stop_pitch:
            print(f"✓ 숙임 {p2:+.2f}° ≤ {args.sweep_stop_pitch}° — 정면을 지났다"); break
    if args.out:
        Path(args.out).write_text(json.dumps({"bias": bias, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0], epilog=RECALIB_EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="extrinsics", help="'extrinsics'(sim 시야) 또는 숙임 deg")
    p.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="imu_pitch_bias 를 읽고(--calib-bias 면 쓰는) 파일")
    p.add_argument("--check", action="store_true",
                   help="움직이지 않고 시야만 검사 — |IMU 숙임 − 기준 숙임| 이 설정 view_check_tol_deg 를 넘으면 종료 코드 1")
    p.add_argument("--execute", action="store_true", help="목을 움직인다(사용자 허락 뒤에만)")
    p.add_argument("--calib-bias", action="store_true", help="지금 자세 = sim 시야라 보고 IMU 0점을 다시 재서 설정에 쓴다")
    p.add_argument("--min-tilt", type=float, default=Limits.min_tilt)
    p.add_argument("--max-tilt", type=float, default=Limits.max_tilt)
    p.add_argument("--max-step", type=float, default=Limits.max_step)
    p.add_argument("--tol", type=float, default=Limits.tol)
    p.add_argument("--out", help="기록 JSON 경로")
    p.add_argument("--sweep", type=float, metavar="STEP",
                   help="처짐 지도: 지금 tilt 에서 STEP°씩(음수 = 머리를 든다) --min-tilt/--max-tilt 까지 가며 인코더 · 숙임을 기록. "
                        "숙임이 --sweep-stop-pitch 를 넘거나 비율이 0.5~1.5 밖이면 멈춘다")
    p.add_argument("--grid", action="store_true",
                   help="보정표 측정: --grid-pans 각 pan 에서 --grid-tilts 를 순서대로(든다) 갔다가 거꾸로(숙인다) 돌아오며 기록. "
                        "--out 에 한 줄씩(jsonl) 바로 쓴다 — 중간에 끊겨도 남는다")
    p.add_argument("--grid-pans", default="-30,-20,-10,0,10,20,30")
    p.add_argument("--grid-tilts", default="74.8,70,65,60,55,50,45,40,35,30,25,20,15,10,5,0,-2")
    p.add_argument("--sweep-stop-pitch", type=float, default=-2.0, help="sweep 이 멈추는 숙임(보정 뒤 deg, 기본 정면 −2°)")
    args = p.parse_args(argv)

    ext_pitch, ext_roll = load_extrinsics_pitch(Path(args.extrinsics))
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    bias = float(cfg.get("imu_pitch_bias", 0.0))
    if args.check and "view_check_tol_deg" not in cfg:
        raise SystemExit(f"❌ {cfg_path} 에 view_check_tol_deg 가 없다 — 검사 허용치를 모른다")
    target = ext_pitch if args.target == "extrinsics" else float(args.target)
    lim = Limits(args.min_tilt, args.max_tilt, args.max_step, args.tol)

    import rclpy
    from rclpy.node import Node
    import head_move
    rclpy.init()
    node = Node("head_view_feedback")
    try:
        sens = Sensors(node, rclpy)
        raw_pitch, roll, sd, pan, tilt = sens.read()
        print(f"지금: 인코더 pan {pan:+.2f}° tilt {tilt:+.2f}° | IMU 숙임 {raw_pitch:+.2f}°(보정 뒤 {raw_pitch - bias:+.2f}°) "
              f"옆 {roll:+.2f}° σ {sd:.3f} | 외부 파라미터 숙임 {ext_pitch:+.2f}° 옆 {ext_roll:+.2f}° | 목표 {target:+.2f}°")
        if args.check:
            tol = float(cfg["view_check_tol_deg"])
            err, ok = view_error(raw_pitch, bias, target, tol)
            tick = round(tilt * 4096 / 360) + 2048
            print(f"시야 검사: IMU 숙임(0점 {bias:+.2f}° 보정) {raw_pitch - bias:+.2f}° − 기준({Path(args.extrinsics).name}) "
                  f"{target:+.2f}° = {err:+.2f}°(허용 ±{tol}°) | 인코더 tilt 틱 {tick}")
            if "sim_pitch_deg" in cfg:
                print(f"  참고: sim 카메라 숙임 {cfg['sim_pitch_deg']:+.2f}° 와 차 {raw_pitch - bias - cfg['sim_pitch_deg']:+.2f}°")
            if ok:
                print("✓ 시야 숙임이 기준과 같다(yaw 는 IMU 로 못 잰다)")
                return 0
            print(f"❌ 시야가 sim 과 다르다 — 캘리브 자세 다시 잡기 필요: 숙임 차 {err:+.2f}°(+ = 더 숙임) > ±{tol}°. "
                  "카메라 · 머리를 건드렸는지 보고, 맞춘 뒤에는 외부 파라미터를 다시 잰다(--help 끝)")
            return 1
        if args.calib_bias:
            new = raw_pitch - ext_pitch
            cfg.update(imu_pitch_bias=round(new, 3), imu_roll_bias=round(roll - ext_roll, 3),
                       bias_measured_at={"pan_deg": round(pan, 3), "tilt_deg": round(tilt, 3),
                                         "time": time.strftime("%Y-%m-%d %H:%M"), "extrinsics": Path(args.extrinsics).name})
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"IMU 0점 {bias:+.3f} → {new:+.3f}° 를 {cfg_path} 에 썼다")
            return 0
        pitch = raw_pitch - bias
        steps: list[Step] = []
        ratio = float(cfg.get("ratio_guess", 0.98))
        if args.grid and args.execute:
            return _grid(args, lim, sens, head_move, bias)
        if args.sweep and args.execute:
            return _sweep(args, lim, sens, head_move, pan, tilt, pitch, bias)
        if not args.execute:
            print(f"남은 오차 {pitch - target:+.2f}° → 다음 tilt {next_tilt(tilt, pitch - target, ratio, lim):+.2f}°(--execute 없음, 움직이지 않는다)")
            return 0
        code = 1
        for _ in range(lim.max_iters):
            err = pitch - target
            if abs(err) <= lim.tol:
                code = 0
                break
            cmd = next_tilt(tilt, err, ratio, lim)
            if abs(cmd - tilt) < 0.05:
                print(f"❌ tilt 한계 [{lim.min_tilt}, {lim.max_tilt}] 에 막혔다(남은 오차 {err:+.2f}°)")
                break
            rc, _ = head_move.request_move(pan, cmd, max_tilt=lim.max_tilt)
            if rc == 2:
                print("❌ 드라이버가 요청을 거절했다")
                break
            raw2, roll, sd, pan2, tilt2 = sens.read()
            pitch2 = raw2 - bias
            ratio = ratio_from((tilt, pitch), (tilt2, pitch2), ratio)
            steps.append(Step(cmd, pan2, tilt2, pitch2, roll, pitch2 - target, ratio))
            print(f"  tilt 목표 {cmd:+.2f}° 실제 {tilt2:+.2f}° | 숙임 {pitch2:+.2f}° 오차 {pitch2 - target:+.2f}° 옆 {roll:+.2f}° | 비율 {ratio:.3f}",
                  flush=True)
            pan, tilt, pitch = pan2, tilt2, pitch2  # 멈춰도 최종 보고는 실제로 선 자리
            if not RATIO_OK[0] <= ratio <= RATIO_OK[1]:
                print(f"❌ 비율 {ratio:.3f} — 카메라가 인코더만큼 안 움직인다(풀림 · 미끄러짐?), 그 자리에서 멈춘다")
                break
        tick = round(tilt * 4096 / 360) + 2048
        print(("✓" if code == 0 else "❌") + f" 끝: 인코더 tilt {tilt:+.2f}°(틱 {tick}) 숙임 {pitch:+.2f}° 목표 {target:+.2f}°")
        if args.out:
            Path(args.out).write_text(json.dumps({"target": target, "bias": bias, "ok": code == 0, "final_tilt_deg": tilt,
                                                  "final_tick": tick, "steps": [asdict(s) for s in steps]},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
        return code
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
