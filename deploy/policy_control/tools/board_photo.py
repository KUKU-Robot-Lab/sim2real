"""board_jog 점마다 사진 확인(11.05 사용자) — 머리 카메라 컬러 한 장 · 오버레이 · photo_check, 그리고 사진용 카메라 노드 켜기 · 끄기.

사진: 기록 순간(접촉 정지 기록 창) 받은 원본 컬러(image_raw — 압축 토픽은 드라이버 지연이 생겨 쓰지 않는다)를 별 스레드(Worker)가
jpg 로 쓰고, 카메라 외부 파라미터(--extrinsics) · color camera_info K 로 (a) 실측 FK 손끝 +, (b) 목표 교차점 o(맞춤 자세),
(c) 그 사진에서 검출한 ChArUco 교차점 점을 그린다. photo_check = 목표 id 검출 픽셀 ↔ FK 손끝 투영 픽셀 거리(px, 손끝 깊이로 mm 근사).
손가락이 목표를 가리면 검출이 안 된다 → px_err null + why. 같은 사진의 보드 PnP 로 목표 픽셀을 옮긴 값(px_err_pnp)도 함께 적는다.

카메라(Camera): camera_up.sh 와 같은 launch · 인자 · 환경(이 프로세스 환경 = board_jog.sh 의 도메인 · lan_profile)으로 띄우되 그
스크립트의 pgrep/pkill -f(이름 패턴 kill)는 쓰지 않는다. 우리가 띄운 프로세스 그룹에만 SIGTERM, SIGKILL 은 쓰지 않는다.
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

S2R = Path(__file__).resolve().parents[3]
BOARD_DIR = S2R / "logs/analysis/fk_touch_probe_1105"   # board_cam_fit(detect · load_T_base_cam · render) — cv2 4.7+(.venv)
COLOR_TOPIC = "/camera/camera/color/image_raw"            # realsense2_camera 4.58 rs_launch 기본 이름(camera_up.sh · vision_recorder.py)
INFO_TOPIC = "/camera/camera/color/camera_info"
CAM_NODE = "realsense2_camera_node"
# camera_up.sh 의 launch 와 같은 인자(align_depth 는 FP++ · IMU 는 head_view_check 가 쓴다 — 다른 사용처와 같은 카메라 설정을 낸다)
LAUNCH = ("source /opt/ros/humble/setup.bash; exec ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true "
          "enable_gyro:=true enable_accel:=true unite_imu_method:=2")
AUTO_WAIT_S, STOP_WAIT_S = 1.5, 5.0
JPG_Q = 92


# ------------------------------------------------------------------ 영상 → 확인(순수 · cv2)
def _bf():
    sys.path.insert(0, str(BOARD_DIR))
    import board_cam_fit as BF  # noqa: PLC0415

    return BF


def to_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image(rgb8 · bgr8 · mono8) → BGR uint8."""
    import cv2

    ch = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(msg.encoding)
    if ch is None:
        raise ValueError(f"인코딩 {msg.encoding} 은 받지 않는다(rgb8 · bgr8 · mono8)")
    a = np.frombuffer(msg.data, np.uint8)   # rclpy uint8[] = array.array — 복사 없이
    a = a.reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    return {"rgb8": lambda: cv2.cvtColor(a, cv2.COLOR_RGB2BGR), "bgr8": lambda: a.copy(),
            "mono8": lambda: cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)}[msg.encoding]()


def proj(K: np.ndarray, T_cam_base: np.ndarray, P) -> tuple[np.ndarray, np.ndarray]:
    """base 점들 → (픽셀 Nx2, 카메라 깊이 N). 왜곡은 무시(D435 컬러 plumb_bob 0, board_cam_fit 과 같다)."""
    Pc = np.atleast_2d(P) @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    uv = Pc @ np.asarray(K, float).T
    return uv[:, :2] / uv[:, 2:3], Pc[:, 2]


def check(bgr: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, tip, tgt, ti: int, charuco: bool) -> tuple[dict, np.ndarray]:
    """→ (photo_check, 오버레이). tip · tgt = base m(실측 FK 손끝 · 맞춤 목표), ti = 목표 교차점 id."""
    import cv2

    BF = _bf()
    T_cam_base = np.linalg.inv(T_base_cam)
    (tip_px, tgt_px), (z_tip, _) = proj(K, T_cam_base, [tip, tgt])
    mm = float(z_tip) / float(K[0][0])   # 손끝 깊이에서 1 px 의 mm 근사(m/px · 1000 은 아래)
    pc = {"tip_px": np.round(tip_px, 1).tolist(), "target_px_fit": np.round(tgt_px, 1).tolist(), "n_corners": 0,
          "target_px_det": None, "px_err": None, "mm_err_approx": None, "target_px_pnp": None, "px_err_pnp": None,
          "mm_err_pnp_approx": None, "board_reproj_px": None, "why": None}
    det = BF.detect(bgr, np.asarray(K, float)) if charuco else None
    ov = bgr.copy()
    if not charuco:
        pc["why"] = "ChArUco 보드만 검출한다"
    elif det is None:
        pc["why"] = "보드 검출 실패(교차점 6 개 미만)"
    else:
        ids = [int(np.argmin(np.linalg.norm(BF.CORNERS - o, axis=1))) for o in det["objp"]]
        pc.update(n_corners=len(ids), board_reproj_px=round(float(det["reproj"]), 2))
        for uv in det["imgp"]:
            cv2.circle(ov, tuple(int(round(v)) for v in uv), 3, (0, 220, 255), -1)
        pnp = BF.project(np.asarray(K, float), det["T"], BF.CORNERS[ti:ti + 1])[0]
        e = float(np.linalg.norm(pnp - tip_px))
        pc.update(target_px_pnp=np.round(pnp, 1).tolist(), px_err_pnp=round(e, 1), mm_err_pnp_approx=round(e * mm * 1000, 1))
        if ti in ids:
            uv = det["imgp"][ids.index(ti)]
            e = float(np.linalg.norm(uv - tip_px))
            pc.update(target_px_det=np.round(uv, 1).tolist(), px_err=round(e, 1), mm_err_approx=round(e * mm * 1000, 1))
        else:
            pc["why"] = f"목표 교차점 id {ti} 가 검출되지 않음(손가락이 가림?) — 보드 PnP 값만"
    cv2.circle(ov, tuple(int(round(v)) for v in tgt_px), 9, (0, 200, 0), 2)
    cv2.drawMarker(ov, tuple(int(round(v)) for v in tip_px), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    txt = (f"tip+ target(fit)o  det id{ti}: " + (f"{pc['px_err']:.1f}px ~{pc['mm_err_approx']:.1f}mm" if pc["px_err"] is not None else "none")
           + (f"  pnp: {pc['px_err_pnp']:.1f}px ~{pc['mm_err_pnp_approx']:.1f}mm" if pc["px_err_pnp"] is not None else ""))
    cv2.putText(ov, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(ov, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return pc, ov


class Worker:
    """사진 쓰기 · 검출을 제어 스레드(60 Hz) 밖에서. put(job) → done 큐에 (점 index, photo_check)."""

    def __init__(self, extrinsics: Path):
        self.extr, self.T = Path(extrinsics), None
        self.jobs: queue.Queue = queue.Queue()
        self.done: queue.Queue = queue.Queue()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def put(self, job: dict) -> None:
        self.jobs.put(job)

    def close(self, timeout: float = 10.0) -> None:
        self.jobs.put(None)
        self.th.join(timeout)

    def _run(self) -> None:
        while (job := self.jobs.get()) is not None:
            try:
                self.done.put((job["i"], self.process(job)))
            except Exception as error:   # noqa: BLE001 — 사진 실패는 기록을 막지 않는다(경고만)
                self.done.put((job["i"], {"status": "error", "why": f"{type(error).__name__}: {error}"}))

    def process(self, job: dict) -> dict:
        import cv2

        if self.T is None:
            self.T = _bf().load_T_base_cam(self.extr)
        bgr = to_bgr(job["msg"])
        path = Path(job["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        pc, ov = check(bgr, job["K"], self.T, job["tip"], job["tgt"], job["ti"], job["charuco"])
        ovp = path.with_name(path.stem + "_overlay.jpg")
        for p, im in ((path, bgr), (ovp, ov)):
            if not cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, JPG_Q]):
                raise OSError(f"jpg 저장 실패 {p}")
        return {"status": "done", "photo": path.name, "overlay": ovp.name, "frame_age_s": job["age"], "extrinsics": str(self.extr), **pc}


def summary(points: list, bid: str, res_max_mm: float) -> dict:
    """이번 실행(seeded 아님 · 산) 점의 맞춤 잔차 · photo_check 요약. 순수."""
    P = [p for p in points if not p["deleted"] and p["board_id"] == bid and "seeded_from" not in p]
    R = [(p["target_id"], p["fit_res_mm"]) for p in P if p.get("fit_res_mm") is not None]
    ph = [p.get("photo_check") or {} for p in P]
    det = [c["mm_err_approx"] for c in ph if c.get("mm_err_approx") is not None]
    pnp = [c["mm_err_pnp_approx"] for c in ph if c.get("mm_err_pnp_approx") is not None]
    why: dict = {}
    for c in ph:
        if c.get("px_err") is None:
            k = (c.get("why") or c.get("status") or "사진 없음").split(" — ")[0].split("(")[0].strip()   # 이유 종류별(수치 뺌)
            why[k] = why.get(k, 0) + 1
    r = np.array([v for _, v in R]) if R else np.zeros(0)
    stat = lambda v: None if not len(v) else {"mean": round(float(np.mean(v)), 2), "max": round(float(np.max(v)), 2)}   # noqa: E731
    return {"board_id": bid, "n_points": len(P), "n_with_res": len(R), "fit_res_mm": stat(r),
            "over_mm": res_max_mm, "over": [t for t, v in R if v > res_max_mm],
            "photo": {"n_photo": sum(c.get("status") == "done" for c in ph), "n_det": len(det), "det_mm": stat(det),
                      "n_pnp": len(pnp), "pnp_mm": stat(pnp), "no_det_why": why}}


def summary_text(s: dict) -> str:
    r, ph = s["fit_res_mm"], s["photo"]
    return (f"{s['board_id']} 이번 실행 {s['n_points']} 점 · 맞춤 잔차 " + ("—" if r is None else f"평균 {r['mean']:.1f} · 최대 {r['max']:.1f} mm")
            + f" · > {s['over_mm']:g} mm {len(s['over'])} 점" + (f"({', '.join(s['over'])})" if s["over"] else "")
            + f" · 사진 {ph['n_photo']} · 목표 검출 {ph['n_det']}" + ("" if ph["det_mm"] is None else f"(평균 {ph['det_mm']['mean']:.1f} mm)")
            + ("" if ph["pnp_mm"] is None else f" · PnP 평균 {ph['pnp_mm']['mean']:.1f} · 최대 {ph['pnp_mm']['max']:.1f} mm"))


# ------------------------------------------------------------------ 카메라 노드(우리가 띄운 것만 끈다)
class Camera:
    """frames() = 지금까지 받은 컬러 영상 수(RosIO). owner: ours(우리가 띄움) · foreign(영상 또는 남의 노드 프로세스) · none."""

    def __init__(self, frames, hz, log: Path, popen=subprocess.Popen, proc_root: Path = Path("/proc"), killpg=os.killpg):
        self.frames, self.hz, self.log, self.popen, self.proc_root, self.killpg = frames, hz, Path(log), popen, Path(proc_root), killpg
        self.proc, self.note, self.lock = None, "확인 전", threading.Lock()

    def mine(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def node_pids(self) -> list[int]:
        """argv[0] 이 realsense2_camera_node 인 프로세스(그 이름을 인자로 품은 셸은 빼려고 argv[0] 만 본다), 우리 그룹 제외."""
        out = []
        for d in self.proc_root.glob("[0-9]*"):
            try:
                argv0 = (d / "cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
                if Path(argv0).name != CAM_NODE:
                    continue
                pgrp = int((d / "stat").read_text().rsplit(")", 1)[1].split()[2])
            except (OSError, IndexError, ValueError):
                continue
            if not (self.mine() and pgrp == self.proc.pid):
                out.append(int(d.name))
        return out

    def owner(self) -> str:
        return "ours" if self.mine() else "foreign" if self.hz() > 0 or self.node_pids() else "none"

    def status(self) -> str:
        o = self.owner()
        who = {"ours": f"우리가 띄움(PID {self.proc.pid if self.proc else '-'})", "foreign": "남의 것(끄지 않는다)", "none": "없음"}[o]
        return f"카메라 {self.hz():.0f} Hz · {who}" + (f" · {self.note}" if self.note else "")

    def auto(self, wait: float = AUTO_WAIT_S, sleep=time.sleep) -> str:
        """--camera auto: wait 안에 영상 → 그대로(남의 것) · 남의 노드 프로세스만 → 경고 · 둘 다 없으면 우리가 띄운다."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait:
            if self.frames() > 0:
                self.note = "이미 떠 있는 카메라를 쓴다"
                return self.note
            sleep(0.05)
        if pids := self.node_pids():
            self.note = f"★ 카메라 프로세스는 있는데 영상 없음(PID {pids}) — 띄우지 않는다, 사진 없이 기록"
            return self.note
        return self.start()

    def start(self) -> str:
        with self.lock:
            if self.mine():
                return "카메라 켜기 무시 — 이미 우리가 띄웠다"
            if self.hz() > 0 or self.node_pids():
                self.note = "카메라 켜기 거부 — 이미 다른 카메라(영상 · 노드 프로세스)가 있다"
                return self.note
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log, "ab") as lf:
                self.proc = self.popen(["bash", "-c", LAUNCH], stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                       start_new_session=True)   # setsid — 그룹 PID = 이 PID, 끌 때 이 그룹만
            self.note = f"카메라 켬 PID {self.proc.pid}(로그 {self.log.name})"
            return self.note

    def stop(self, block: bool = False) -> str:
        """우리가 띄운 그룹에만 SIGTERM. STOP_WAIT_S 뒤 남으면 경고만(SIGKILL 금지)."""
        with self.lock:
            if not self.mine():
                self.note = "카메라 끄기 거부 — 우리가 띄운 카메라가 아니다" if self.owner() == "foreign" else "끌 카메라 없음"
                return self.note
            p = self.proc
            try:
                self.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        def wait() -> None:
            try:
                p.wait(STOP_WAIT_S)
                self.note = f"카메라 끔 PID {p.pid}"
            except subprocess.TimeoutExpired:
                self.note = f"★ 카메라 PID {p.pid} 가 SIGTERM {STOP_WAIT_S:g} s 뒤에도 남음 — 직접 확인(SIGKILL 하지 않음)"
                print(self.note, file=sys.stderr, flush=True)

        self.note = f"카메라 끄는 중 PID {p.pid}"
        wait() if block else threading.Thread(target=wait, daemon=True).start()
        return self.note


# ------------------------------------------------------------------ 자체 검사(ROS 없음 · 가짜 Popen · /proc · 합성 영상)
def selftest() -> None:
    import tempfile
    from types import SimpleNamespace

    import cv2

    BF = _bf()
    # (p1) 합성 영상: 카메라 0.5 m 앞 보드(살짝 기울임), base = 카메라 프레임. 손끝 = 목표 bc09 → 0 mm, +5 mm 옮김 → ≈ 5 mm
    K = np.array([[600.0, 0, 640], [0, 600.0, 360], [0, 0, 1]])
    T_cb = np.eye(4)
    T_cb[:3, :3] = cv2.Rodrigues(np.array([0.25, -0.15, 0.05]))[0]
    T_cb[:3, 3] = [-0.10, -0.07, 0.50]
    img = BF.render(T_cb, K)
    P = BF.CORNERS @ T_cb[:3, :3].T + T_cb[:3, 3]
    ti, x_b = 9, T_cb[:3, 0]
    pc, ov = check(img, K, np.eye(4), P[ti], P[ti], ti, True)
    assert pc["n_corners"] >= 20 and pc["px_err"] is not None and pc["mm_err_approx"] < 0.5 and pc["why"] is None, pc
    pc5, _ = check(img, K, np.eye(4), P[ti] + 0.005 * x_b, P[ti], ti, True)
    assert abs(pc5["mm_err_approx"] - 5.0) < 0.6 and abs(pc5["mm_err_pnp_approx"] - 5.0) < 0.6, pc5
    occ = img.copy()   # 손가락이 목표를 가림 → 목표 id 검출 없음 → px_err null + 이유, PnP 값은 남는다
    cv2.circle(occ, tuple(int(v) for v in pc["target_px_det"]), 14, (40, 40, 40), -1)
    pco, _ = check(occ, K, np.eye(4), P[ti] + 0.005 * x_b, P[ti], ti, True)
    assert pco["px_err"] is None and "검출되지 않음" in pco["why"] and abs(pco["mm_err_pnp_approx"] - 5.0) < 1.0, pco
    blank, _ = check(np.full((720, 1280, 3), 170, np.uint8), K, np.eye(4), P[ti], P[ti], ti, True)
    grid, _ = check(img, K, np.eye(4), P[ti], P[ti], ti, False)
    assert blank["why"].startswith("보드 검출 실패") and grid["why"].startswith("ChArUco") and ov.shape == img.shape
    rgb = SimpleNamespace(encoding="rgb8", height=2, width=3, step=10, data=bytes(range(20)))   # step 패딩 1 바이트
    assert to_bgr(rgb).shape == (2, 3, 3) and to_bgr(rgb)[1, 0].tolist() == [12, 11, 10]
    with tempfile.TemporaryDirectory() as td:   # Worker: jpg 둘 + done 큐
        ex = Path(td) / "extr.yaml"
        ex.write_text("camera:\n  position: [0, 0, 0]\n  orientation_wxyz: [1, 0, 0, 0]\n")
        w = Worker(ex)
        msg = SimpleNamespace(encoding="bgr8", height=720, width=1280, step=3840, data=img.tobytes())
        w.put({"i": 7, "msg": msg, "K": K, "tip": P[ti], "tgt": P[ti], "ti": ti, "path": Path(td) / "ph" / "007_charuco_bc09.jpg",
               "age": 0.1, "charuco": True})
        i, r = w.done.get(timeout=20)
        w.close()
        assert i == 7 and r["status"] == "done" and r["px_err"] < 1.0 and (Path(td) / "ph" / r["overlay"]).stat().st_size > 0, r
    print(f"(p1) 사진 확인: 합성 영상 교차점 {pc['n_corners']} 개 · 손끝 = 목표 {pc['px_err']:.2f} px · +5 mm → 검출 {pc5['mm_err_approx']:.2f} mm "
          f"(PnP {pc5['mm_err_pnp_approx']:.2f}) · 가림 → px_err null(PnP {pco['mm_err_pnp_approx']:.2f} mm) · 빈 영상 · grid 이유 · Worker jpg 2 장")
    # (p2) 요약
    pts = [{"board_id": "charuco", "target_id": f"bc{k:02d}", "deleted": False, "fit_res_mm": v, "photo_check": c}
           for k, v, c in ((1, 1.0, {"status": "done", "px_err": 3.0, "mm_err_approx": 1.5, "mm_err_pnp_approx": 2.0}),
                           (2, 4.0, {"status": "done", "px_err": None, "why": "목표 교차점 id 2 가 검출되지 않음 — x", "mm_err_pnp_approx": 4.0}),
                           (3, 2.0, {"status": "none", "why": "카메라 영상 없음"}))]
    pts += [{"board_id": "charuco", "target_id": "bc00", "deleted": False, "fit_res_mm": 9.0, "seeded_from": "x#0"}]
    s = summary(pts, "charuco", 3.0)
    assert s["n_points"] == 3 and s["fit_res_mm"] == {"mean": 2.33, "max": 4.0} and s["over"] == ["bc02"], s
    assert s["photo"]["n_photo"] == 2 and s["photo"]["n_det"] == 1 and s["photo"]["no_det_why"] == {"목표 교차점 id 2 가 검출되지 않음": 1, "카메라 영상 없음": 1}
    assert "bc02" in summary_text(s)
    # (p3) 카메라 분기 · 끄기(가짜 Popen · /proc · killpg)
    calls, kills = [], []

    class FakeProc:
        def __init__(self, pid: int, hang: bool = False):
            self.pid, self.hang, self.rc = pid, hang, None

        def poll(self):
            return self.rc

        def wait(self, timeout=None):
            if self.hang:
                raise subprocess.TimeoutExpired("x", timeout)
            self.rc = 0
            return 0

    def popen(cmd, **kw):
        calls.append((cmd, kw))
        return FakeProc(5000 + len(calls), hang=len(calls) > 1)

    with tempfile.TemporaryDirectory() as td:
        pr = Path(td)

        def proc(pid: int, argv: list, pgrp: int) -> None:
            (pr / str(pid)).mkdir()
            (pr / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
            (pr / str(pid) / "stat").write_text(f"{pid} (x y) S 1 {pgrp} {pgrp} 0")

        proc(100, ["bash", "-c", "pkill -f realsense2_camera_node"], 100)   # 이름을 인자로 품은 셸 — 카메라가 아니다
        n = {"f": 0}
        cam = Camera(lambda: n["f"], lambda: 30.0 if n["f"] else 0.0, pr / "cam.log", popen, pr, lambda pid, sg: kills.append((pid, sg)))
        n["f"] = 3   # (a) 영상이 온다 → 띄우지 않고 남의 것
        assert "이미" in cam.auto(0.2) and not calls and cam.owner() == "foreign"
        assert "거부" in cam.stop() and not kills   # 남의 카메라는 끄지 않는다
        n["f"] = 0   # (b) 영상 없음 + 남의 노드 프로세스(argv[0]) → 경고만
        proc(200, ["/opt/ros/humble/lib/realsense2_camera/realsense2_camera_node", "--ros-args"], 200)
        assert "프로세스는 있는데" in cam.auto(0.1) and not calls and cam.node_pids() == [200]
        assert "거부" in cam.start() and not calls
        (pr / "200" / "cmdline").unlink()   # (c) 둘 다 없음 → 우리가 띄운다(setsid)
        assert "켬" in cam.auto(0.1) and len(calls) == 1 and cam.owner() == "ours"
        assert calls[0][0] == ["bash", "-c", LAUNCH] and calls[0][1]["start_new_session"] and "rs_launch.py" in LAUNCH
        proc(201, ["/opt/ros/humble/lib/realsense2_camera/realsense2_camera_node"], 5001)   # 우리 그룹의 노드는 남의 것이 아니다
        assert cam.node_pids() == [] and "이미" in cam.start() and len(calls) == 1
        cam.stop(block=True)   # 종료 정리 = 우리 그룹에만 SIGTERM
        assert kills == [(5001, signal.SIGTERM)] and not cam.mine() and "끔" in cam.note, (kills, cam.note)
        (pr / "201").joinpath("cmdline").unlink()
        cam.start()   # 두 번째는 SIGTERM 에 안 끝남 → 경고만(SIGKILL 없음)
        cam.stop(block=True)
        assert kills[-1] == (5002, signal.SIGTERM) and len(kills) == 2 and "남음" in cam.note, cam.note
    print("(p2) 요약(잔차 평균 · 최대 · 3 mm 넘는 점 · 사진 검출 · 이유별) (p3) 카메라 auto: 영상 있음 → 그대로 · 남의 노드 프로세스만 → "
          "경고 · 둘 다 없음 → setsid 로 띄움 / 남의 것 끄기 거부 · 끄기 = 우리 그룹 SIGTERM · 안 끝나면 경고만")
