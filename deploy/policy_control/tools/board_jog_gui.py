"""board_jog 조작 창(PyQt5). 제어 · 안전 판정은 board_jog.Ctl(제어 스레드)에 있고 이 창은 요청을 넣고 스냅숏을 그리기만 한다.

11.05 사용자: "조작 ui가 구려. 키패드 같은 상/하, 전진(+x)/후진(-x), 왼쪽(+y)/오른쪽(-y) 화표로 되어있어야지."
배치: 맨 위 상태 띠 · 왼쪽 이동 패드 · 회전 패드 · 단계 · 주요 동작 줄 · 오른쪽 목표 · 기록(위에서 본 2D 로 목표 고르기).
11.05 사용자 "최외각 직사각형 4점만 내가 맞춰주고, 나머진 점들은 자동화": 상태 띠 둘째 줄 = 보드 자세(추정 · 4점 · 전체 맞춤) · 자동 상태.
[자동 다음 점](N, 고른 보드 안에서만 — 보드를 바꾸면 해제) · 확인 없이 자동 하강 · 건너뛰기(X) · Enter = 제어 스레드의 auto_descend
(자동 확인 대기에서만 하강, 그 밖에는 무시). 창 포커스를 잃으면 자동을 끈다. 받아들인 맞춤은 보드 표시 자세(그림 · 목표 · sim)를 덮는다.
3D 는 sim 창(Isaac, board_jog.sim_sender UDP)이 그린다.
"""
from __future__ import annotations

import json
import signal
import sys
import time
import traceback

import numpy as np

import board_jog as B

KO_FONT = ["Noto Sans CJK JP", "DejaVu Sans"]   # 같은 ttc, matplotlib 은 JP 이름으로 등록(한글 포함)
REPEAT_DELAY_S = 0.35   # 누르고 있으면 이만큼 뒤부터 반복(한 번 누름이 두 단계가 되지 않게)
# 색(ui-ux-pro-max 팔레트: 위험 #DC2626 · 강조 #EA580C · 기본 #2563EB, 글자 대비 4.5:1 이상)
RED, ORANGE, BLUE, AMBER, GREEN, GRAY, LIGHT = "#DC2626", "#EA580C", "#2563EB", "#F59E0B", "#16A34A", "#4B5563", "#E5E7EB"
QSS = f"""
QMainWindow, QWidget#root {{ background: #F8FAFC; }}
QFrame#strip {{ background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; }}
QGroupBox {{ font-weight: bold; border: 1px solid #CBD5E1; border-radius: 6px; margin-top: 10px; background: #FFFFFF; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QPushButton {{ min-height: 32px; padding: 4px 10px; border: 1px solid #94A3B8; border-radius: 6px; background: #F1F5F9; color: #0F172A; }}
QPushButton:pressed {{ background: #CBD5E1; }}
QPushButton:disabled {{ color: #94A3B8; background: #F1F5F9; border-color: #E2E8F0; }}
QPushButton[pad="true"] {{ min-width: 104px; min-height: 54px; font-size: 15px; font-weight: bold; background: #E0E7FF; border: 2px solid #6366F1; }}
QPushButton[pad="true"]:pressed {{ background: #A5B4FC; }}
QPushButton[rot="true"] {{ min-width: 150px; min-height: 42px; font-size: 14px; font-weight: bold; background: #ECFEFF; border: 2px solid #0891B2; }}
QPushButton[rot="true"]:pressed {{ background: #A5F3FC; }}
QPushButton[stepbtn="true"] {{ min-width: 64px; min-height: 32px; font-size: 14px; }}
QPushButton[stepbtn="true"]:checked {{ background: {BLUE}; color: #FFFFFF; font-weight: bold; border-color: #1E40AF; }}
QPushButton#descend {{ min-height: 60px; font-size: 15px; font-weight: bold; background: {AMBER}; color: #000000; border-color: #B45309; }}
QPushButton#lift {{ min-height: 60px; font-size: 15px; font-weight: bold; background: {BLUE}; color: #FFFFFF; border-color: #1E40AF; }}
QPushButton#stop {{ min-height: 72px; font-size: 22px; font-weight: bold; background: {RED}; color: #FFFFFF; border: 3px solid #7F1D1D; }}
QPushButton#stop:pressed {{ background: #991B1B; }}
QPushButton#home {{ min-height: 60px; font-size: 15px; font-weight: bold; }}
QPushButton#record {{ min-height: 48px; font-size: 18px; font-weight: bold; background: {GREEN}; color: #000000; border-color: #166534; }}
QPushButton#record:disabled {{ background: {LIGHT}; color: #6B7280; }}
QPushButton#auto {{ min-height: 40px; font-size: 14px; font-weight: bold; background: #EDE9FE; border: 2px solid #7C3AED; }}
QLabel[cap="true"] {{ color: #475569; font-size: 11px; }}
QLabel[val="true"] {{ font-size: 15px; font-weight: bold; color: #0F172A; }}
QLabel[note="true"] {{ color: #64748B; font-size: 11px; }}
"""


def run_gui(ctl, kin, lo, hi, boards: dict, srcs: dict, smoke: str | None) -> int:
    """boards = {board_id: 4x4 표시 자세}, srcs = {board_id: pose_display 사전(editable 포함)} — 창에서 옮기면 둘 다 바뀐다."""
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PyQt5 import QtCore, QtGui, QtWidgets as W

    Qt = QtCore.Qt
    app = W.QApplication.instance() or W.QApplication(sys.argv[:1])
    win = W.QMainWindow()
    win.setWindowTitle(f"board_jog — {kin.side} {'★EXECUTE' if ctl.execute else ('demo' if ctl.demo else 'dry')}")
    win.setStyleSheet(QSS)
    bids = list(boards)
    st = {"boards": {b: T.copy() for b, T in boards.items()}, "bid": bids[0], "target": 0, "rec_ver": -1, "closing": False,
          "error": None, "dead": False, "n_jog": 0,
          "fit_ver": -1, "fits": {}, "est": {}, "auto_key": None}   # est = 맞춤이 덮기 전 추정 자세(맞춤이 사라지면 되돌린다)
    hold = {"key": None, "t0": 0.0, "wait": 0}   # 누르고 있는 (축, 부호) · 누른 시각 · 기다리는 요청 번호

    def fail(where: str, text: str) -> None:
        """창 쪽 예외 → 알리고 창 닫기(closeEvent → ctl.stop → 해제 · stop · 원본 저장). 같은 길로 한 번만."""
        st["error"] = st["error"] or where
        B.say(f"★ 창 오류({where}) — 해제 · 종료로 간다\n{text}", err=True)
        if not st["closing"]:
            try:
                win.close()
            finally:
                app.quit()

    def guarded(where: str, fn):
        def w(*a):
            try:
                return fn(*a)
            except Exception:   # noqa: BLE001
                fail(where, traceback.format_exc())
        return w

    sys.excepthook = lambda tp, v, tb: fail("excepthook", "".join(traceback.format_exception(tp, v, tb)))

    clickable: list = []   # fn 이 붙은 버튼(스모크가 전부 한 번씩 누른다)

    def btn(text: str, fn, tip: str = "", name: str = "") -> W.QPushButton:
        b = W.QPushButton(text)
        b.setFocusPolicy(Qt.NoFocus)   # space 가 포커스 버튼을 누르지 않게(space = 정지)
        b.setCursor(Qt.PointingHandCursor)
        if fn is not None:   # clicked(bool) 인자를 버린다 — 0 인자 lambda 에 들어가 TypeError → 창이 닫혔다(11.05 실기)
            g = guarded(text.split("\n")[0], fn)
            b.clicked.connect(lambda _checked=False: g())
            clickable.append(b)
        if tip:
            b.setToolTip(tip)
        if name:
            b.setObjectName(name)
        return b

    def label(text: str = "", kind: str = "") -> W.QLabel:
        lb = W.QLabel(text)
        if kind:
            lb.setProperty(kind, True)
        return lb

    root = W.QWidget()
    root.setObjectName("root")
    vroot = W.QVBoxLayout(root)
    vroot.setSpacing(8)

    # ---------------- 상태 띠
    strip = W.QFrame()
    strip.setObjectName("strip")
    sv = W.QVBoxLayout(strip)
    sl = W.QHBoxLayout()
    sl.setSpacing(14)
    sv.addLayout(sl)
    lbl_mode = W.QLabel()
    lbl_mode.setAlignment(Qt.AlignCenter)
    lbl_mode.setMinimumSize(190, 52)
    chk = W.QCheckBox("실기 발행")
    chk.setFocusPolicy(Qt.NoFocus)
    chk.setStyleSheet("QCheckBox { font-size: 15px; font-weight: bold; } QCheckBox::indicator { width: 24px; height: 24px; }")
    chk.setToolTip("--execute 로 시작했고 시작 검사를 통과했을 때만 켜진다. 켜는 순간 pd · 측정 · 처짐 검사를 다시 본다")
    chk.clicked.connect(lambda on: ctl.call("publish", bool(on)))
    lamp = W.QLabel("무접촉")
    lamp.setAlignment(Qt.AlignCenter)
    lamp.setMinimumSize(130, 52)

    def metric(caption: str, widget: W.QWidget) -> W.QWidget:
        box = W.QWidget()
        bl_ = W.QVBoxLayout(box)
        bl_.setContentsMargins(0, 0, 0, 0)
        bl_.setSpacing(2)
        bl_.addWidget(label(caption, "cap"))
        bl_.addWidget(widget)
        return box

    def bar(rng: int) -> tuple[W.QProgressBar, W.QLabel, W.QWidget]:
        """막대 + 오른쪽 값 글자(글자를 막대 안에 그리면 채운 경계에서 잘려 보인다)."""
        b = W.QProgressBar()
        b.setRange(0, rng)
        b.setMinimumWidth(110)
        b.setTextVisible(False)
        v = label("—", "val")
        v.setMinimumWidth(96)
        box = W.QWidget()
        hl = W.QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(b)
        hl.addWidget(v)
        return b, v, box

    (bar_g, val_g, box_g), (bar_n, val_n, box_n) = bar(300), bar(200)
    bar_g.setToolTip(f"검지 관절 힘 |Δ| g(기준선 대비). 접촉 중 이동은 {B.FORCE_G:g} g 넘게 오르면 멈춘다")
    bar_n.setToolTip(f"검지 촉각 Δ N(기준선 대비). 접촉 중 이동은 {B.TIP_N:g} N 넘게 오르면 멈춘다")
    lbl_z, lbl_lim, lbl_tr = label("", "val"), label("", "val"), label("", "val")
    lbl_pd = label("", "note")
    sl.addWidget(lbl_mode)
    sl.addWidget(chk)
    sl.addWidget(lamp)
    sl.addWidget(metric("검지 힘", box_g))
    sl.addWidget(metric("촉각", box_n))
    sl.addWidget(metric("손끝 z 하한까지", lbl_z))
    sl.addWidget(metric("관절 한계 여유(가장 가까운)", lbl_lim))
    tr_box = W.QWidget()
    tl = W.QVBoxLayout(tr_box)
    tl.setContentsMargins(0, 0, 0, 0)
    tl.setSpacing(2)
    tl.addWidget(label("추종 오차", "cap"))
    tl.addWidget(lbl_tr)
    lbl_hold = label("", "note")
    tl.addWidget(lbl_hold)
    tl.addWidget(lbl_pd)
    sl.addWidget(tr_box)
    sl.addStretch(1)
    sl2 = W.QHBoxLayout()
    sl2.setSpacing(14)
    sv.addLayout(sl2)
    lbl_fit, lbl_auto = W.QLabel(), W.QLabel()
    lbl_fit.setToolTip(f"바깥 4 모서리 기록 → 강체 맞춤(대각선 차 > {B.A.DIAG_WARN_MM:g} mm 경고 · > {B.A.DIAG_REJECT_MM:g} mm 거부), "
                       f"산 기록 {B.A.REFIT_MIN} 점부터 전체 맞춤. FK 파라미터 보정은 fit_board.py")
    sl2.addWidget(lbl_fit, 2)
    sl2.addWidget(lbl_auto, 3)
    vroot.addWidget(strip)
    lbl_msg = W.QLabel()
    lbl_msg.setWordWrap(True)
    lbl_msg.setStyleSheet("font-size: 14px; padding: 4px 6px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px;")
    vroot.addWidget(lbl_msg)

    body = W.QHBoxLayout()
    body.setSpacing(10)
    vroot.addLayout(body, 1)
    left = W.QVBoxLayout()
    right = W.QVBoxLayout()
    body.addLayout(left, 0)
    pads_w = W.QWidget()   # 이동 · 회전 · 단계: 화면이 낮으면 이 칸만 스크롤, 아래 동작 줄(정지 · 손 POINT)은 늘 보인다
    pads = W.QVBoxLayout(pads_w)
    pads.setContentsMargins(0, 0, 0, 0)
    body.addLayout(right, 1)

    # ---------------- 조그(누르고 있으면 반복: 이동이 끝나 idle 이 된 뒤에만 다음 단계, 쌓지 않는다)
    g_t, g_r = W.QButtonGroup(win), W.QButtonGroup(win)
    tip_frame = W.QCheckBox("검지 끝 프레임 기준으로 조그(버튼의 방향 이름과 달라진다)")
    tip_frame.setFocusPolicy(Qt.NoFocus)

    def send_jog(axis: str, sign: int) -> None:
        k = B.AXES.index(axis)
        step = B.TRANS_STEPS_MM[g_t.checkedId()] / 1000 if k < 3 else np.radians(B.ROT_STEPS_DEG[g_r.checkedId()])
        ctl.call("jog", axis, sign, step, "tip" if tip_frame.isChecked() else "base")
        hold["wait"], st["n_jog"] = ctl.n_put, st["n_jog"] + 1

    def press(key: tuple) -> None:
        if st["dead"]:
            return
        hold.update(key=key, t0=time.monotonic())
        send_jog(*key)

    def release(key: tuple | None = None) -> None:
        if key is None or hold["key"] == key:
            hold["key"] = None

    def hold_tick() -> None:
        if hold["key"] is None or time.monotonic() - hold["t0"] < REPEAT_DELAY_S:
            return
        s = ctl.get_snap()
        if not s or s["n_done"] < hold["wait"] or s["mode"] != "idle":
            return
        if not s["msg"].startswith("조그 "):   # 거부 · 접촉 정지 · 힘 상승 정지 뒤에는 반복을 끊는다
            hold["key"] = None
            return
        send_jog(*hold["key"])

    keyname = {("x", 1): "↑", ("x", -1): "↓", ("y", 1): "←", ("y", -1): "→", ("z", 1): "PgUp", ("z", -1): "PgDn"}
    keyname |= {v: k.upper() for k, v in B.KEYS.items() if B.AXES.index(v[0]) >= 3}

    def jbtn(text: str, key: tuple, tip: str, prop: str = "pad") -> W.QPushButton:
        b = btn(f"{text}\n[{keyname[key]}]", None, tip)
        b.setProperty(prop, True)
        b.pressed.connect(guarded("jog", lambda: press(key)))
        b.released.connect(lambda: release(key))
        return b

    g_move = W.QGroupBox("이동 — base 축(로봇 뒤에서 앞을 보는 기준: 왼쪽 = 로봇의 왼쪽 +y)")
    ml = W.QHBoxLayout(g_move)
    cross = W.QGridLayout()
    cross.setSpacing(8)
    cross.addWidget(jbtn("▲ 전진 +x", ("x", 1), "로봇 앞쪽(base +x)"), 0, 1)
    cross.addWidget(jbtn("◀ 왼쪽 +y", ("y", 1), "로봇의 왼쪽(base +y)"), 1, 0)
    centre = label("검지 끝", "cap")
    centre.setAlignment(Qt.AlignCenter)
    cross.addWidget(centre, 1, 1)
    cross.addWidget(jbtn("▶ 오른쪽 −y", ("y", -1), "로봇의 오른쪽(base −y)"), 1, 2)
    cross.addWidget(jbtn("▼ 후진 −x", ("x", -1), "로봇 쪽으로(base −x)"), 2, 1)
    ml.addLayout(cross)
    ml.addSpacing(24)
    vert = W.QVBoxLayout()
    vert.setSpacing(8)
    vert.addWidget(jbtn("▲ 위 +z", ("z", 1), "base +z"))
    vert.addWidget(label("높이", "cap"), 0, Qt.AlignCenter)
    vert.addWidget(jbtn("▼ 아래 −z", ("z", -1), "base −z — 접촉 중 · 힘 끊김 · z 하한 아래로는 거부"))
    ml.addLayout(vert)
    ml.addStretch(1)
    pads.addWidget(g_move)

    g_rot = W.QGroupBox("회전 — 검지 끝 점은 제자리, base 축 둘레")
    rl_ = W.QGridLayout(g_rot)
    rl_.setSpacing(8)
    rot = (("⟲ 손목 돌리기", ("yaw", 1), "위에서 봐 반시계(base z 축 +)"), ("⟳ 손목 돌리기", ("yaw", -1), "위에서 봐 시계(base z 축 −)"),
           ("앞으로 기울이기", ("pitch", 1), "손목이 전진(+x) 쪽으로(base y 축 +)"), ("뒤로 기울이기", ("pitch", -1), "손목이 후진(−x) 쪽으로(base y 축 −)"),
           ("왼쪽으로 기울이기", ("roll", -1), "손목이 왼쪽(+y) 쪽으로(base x 축 −)"), ("오른쪽으로 기울이기", ("roll", 1), "손목이 오른쪽(−y) 쪽으로(base x 축 +)"))
    for i, (text, key, tip) in enumerate(rot):
        rl_.addWidget(jbtn(text, key, tip, "rot"), i // 2, i % 2)
    pads.addWidget(g_rot)

    g_step = W.QGroupBox("한 단계")
    sg = W.QGridLayout(g_step)
    for row, (grp, vals, unit) in enumerate(((g_t, B.TRANS_STEPS_MM, "mm"), (g_r, B.ROT_STEPS_DEG, "°"))):
        sg.addWidget(label("이동" if row == 0 else "회전", "cap"), row, 0)
        grp.setExclusive(True)
        for i, v in enumerate(vals):
            b = btn(f"{v:g} {unit}", None)
            b.setCheckable(True)
            b.setProperty("stepbtn", True)
            b.setChecked(i == 0)   # 가장 작은 단계로 시작
            grp.addButton(b, i)
            sg.addWidget(b, row, 1 + i)
    adv = W.QToolButton()
    adv.setText("고급 ▸")
    adv.setCheckable(True)
    adv.setFocusPolicy(Qt.NoFocus)
    adv.toggled.connect(lambda on: (tip_frame.setVisible(on), adv.setText("고급 ▾" if on else "고급 ▸")))
    tip_frame.setVisible(False)
    sg.addWidget(adv, 2, 0)
    sg.addWidget(tip_frame, 2, 1, 1, 4)
    pads.addWidget(g_step)
    pads.addStretch(1)
    scroll = W.QScrollArea()
    scroll.setWidget(pads_w)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(W.QFrame.NoFrame)
    scroll.setFocusPolicy(Qt.NoFocus)   # 방향키 · PgUp/Dn 이 스크롤로 가지 않게(조그 키)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    left.addWidget(scroll, 1)

    # ---------------- 주요 동작(위험 순서: 내림 → 올림 → 정지 → 복귀)
    acts_w = W.QWidget()
    acts = W.QHBoxLayout(acts_w)
    acts.setContentsMargins(0, 0, 0, 0)
    acts.setSpacing(8)

    def stop() -> None:
        hold["key"] = None
        ctl.call("stop")

    acts.addWidget(btn(f"▼ 천천히 하강\n{B.DESC_V * 1000:.0f} mm/s", lambda: ctl.call("descend"),
                       f"base −z, 접촉 즉시 멈춤 · 최대 {B.DESC_MAX * 1000:.0f} mm · 손끝 z 하한에서 멈춤", "descend"), 2)
    acts.addWidget(btn(f"▲ 위로 {B.LIFT_M * 1000:.0f} mm", lambda: ctl.call("lift"), "접촉 중에도 된다", "lift"), 2)
    b_stop = btn("■ 정지\n[space]", stop, "명령 = 실측 고정", "stop")
    acts.addWidget(b_stop, 3)
    acts.addWidget(btn("시작 자세 복귀", lambda: ctl.call("home"), "접촉 중이면 거부 — 먼저 위로 빼기", "home"), 2)
    left.addWidget(acts_w)
    acts2_w = W.QWidget()
    acts2 = W.QHBoxLayout(acts2_w)
    acts2.setContentsMargins(0, 0, 0, 0)
    acts2.addWidget(btn("손 POINT [P]", lambda: ctl.call("point"), "손을 --grip 자세로(검지만 편 손)"))
    if not ctl.hold_on:   # 유지 켬: 기준선은 하강 · 자동 정지 때 스스로 잡는다 — 사람이 다시 누르는 방식이 아니다(11.05 사용자)
        acts2.addWidget(btn("영점 기준선 다시", lambda: ctl.call("rebase"), "검지가 떠 있을 때 0.3 s 평균"))
    lbl_cam = label("", "note")   # 사진용 머리 카메라: 상태 · 켜기 · 끄기(끄기 = 우리가 띄운 것만) — 이 줄 빈자리에(창 높이를 늘리지 않게)
    lbl_cam.setWordWrap(True)
    acts2.addWidget(lbl_cam, 1)
    b_cam_on = b_cam_off = None
    if ctl.cam is not None:
        b_cam_on = btn("카메라 켜기", lambda: ctl.cam.start(), "영상도 카메라 노드 프로세스도 없을 때만 rs_launch(camera_up.sh 와 같은 인자)를 띄운다")
        b_cam_off = btn("카메라 끄기", lambda: ctl.cam.stop(), "우리가 띄운 카메라 그룹에만 SIGTERM(남의 카메라는 끄지 않는다)")
        acts2.addWidget(b_cam_on)
        acts2.addWidget(b_cam_off)
    acts3_w = W.QWidget()
    acts3 = W.QHBoxLayout(acts3_w)
    acts3.setContentsMargins(0, 0, 0, 0)
    acts3.addWidget(btn("▶ 남은 점 전체 자동 [N]" if ctl.auto_all else "▶ 자동 다음 점 [N]", lambda: ctl.call("auto", True),
                        f"고른 보드의 4 모서리 맞춤 뒤: 보드 면 최고점 + {B.AUTO_CLEAR_M * 1000:.0f} mm 로 올려(+z) 수평 이동 → "
                        f"위 {B.AUTO_HOVER_M * 1000:.0f} mm 까지 {B.AUTO_APPROACH_V * 1000:.0f} mm/s 이하 → 확인 대기 → [하강]/Enter → "
                        f"접촉 기록 → 다음(N 한 번에 {B.AUTO_MAX_PTS} 점). space = 정지 · 자동 해제", "auto"), 2)
    chk_free = W.QCheckBox("확인 없이 자동 하강")
    chk_free.setFocusPolicy(Qt.NoFocus)
    chk_free.setToolTip(f"켜면 위 {B.AUTO_HOVER_M * 1000:.0f} mm 확인 대기를 건너뛴다(N 뒤 첫 점은 확인, 기록 aligned_by = auto). "
                        "space 말고 다른 이유로 자동이 꺼지면 같이 꺼진다")
    chk_free.toggled.connect(lambda on: ctl.call("auto_free", bool(on)))
    acts3.addWidget(chk_free, 1)
    acts3.addWidget(btn("건너뛰기 [X]", lambda: ctl.call("skip"), "확인 대기 중인 목표를 이번 자동에서 빼고 다음으로"), 1)
    left.addWidget(acts2_w)
    left.addWidget(acts3_w)

    # ---------------- 목표 · 기록
    g_tg = W.QGroupBox("목표")
    tg = W.QGridLayout(g_tg)
    cb_b, cb_c = W.QComboBox(), W.QComboBox()
    cb_b.addItems(bids)
    for cb in (cb_b, cb_c):
        cb.setFocusPolicy(Qt.NoFocus)
        cb.setMinimumHeight(32)
    cb_c.currentIndexChanged.connect(lambda i: st.update(target=max(i, 0)))
    lbl_d = W.QLabel()
    lbl_d.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
    lbl_d.setStyleSheet("font-size: 17px; font-weight: bold;")
    tg.addWidget(label("보드", "cap"), 0, 0)
    tg.addWidget(cb_b, 0, 1)
    tg.addWidget(label("목표점", "cap"), 0, 2)
    tg.addWidget(cb_c, 0, 3)
    tg.addWidget(lbl_d, 1, 0, 1, 4)
    tg.addWidget(label(("목표까지 = 명령 6D 손끝(유지 보정으로 실측이 갈 곳) 기준" if ctl.hold_on else "목표까지 = 현재 FK 기준")
                       + ". 실물 손끝 · 보드와 다를 수 있다(풀이 전)", "note"), 2, 0, 1, 4)

    fig = Figure(figsize=(6.4, 4.6), tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(260)   # 1080 px 화면(창 ~990 px)에서 기록 표가 2~3 줄 보이게 300 → 260
    canvas.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Expanding)
    ax2 = fig.add_subplot(111)
    c2, = ax2.plot([], [], "k+", ms=7)
    tgt2, = ax2.plot([], [], "o", mfc="none", mec="m", ms=14, mew=2)
    tip2m, = ax2.plot([], [], "o", color="tab:blue", ms=7, label="tip FK (meas)")
    tip2c, = ax2.plot([], [], "x", color="0.5", ms=7, label="cmd")
    rec2, = ax2.plot([], [], "o", color=GREEN, ms=6, zorder=3, label="recorded")
    cor2, = ax2.plot([], [], "s", mfc="none", mec=ORANGE, ms=12, mew=2, zorder=2, label="outer 4")
    ax2.set_aspect("equal", adjustable="datalim")   # 칸 모양에 맞춰 범위를 넓힌다(축척은 같게)
    # 패드와 같은 방향: 가로 = base y(왼쪽이 +y, xlim 을 뒤집어 그림), 세로 = base x(위가 전진 +x). 데이터 좌표 = (y mm, x mm)
    ax2.set_xlabel("← 왼쪽 +y        base y [mm]        오른쪽 −y →", fontsize=8, fontfamily=KO_FONT)
    ax2.set_ylabel("↑ 전진 +x     base x [mm]", fontsize=8, fontfamily=KO_FONT)
    ax2.tick_params(labelsize=7)
    ax2.legend(loc="lower right", fontsize=7)
    labels: list = []
    tg.addWidget(canvas, 3, 0, 1, 4)
    tg.addWidget(label("위에서 본 그림 · 점을 눌러 목표를 고른다. 보드 위치는 표시용 추정, 배치(마커 순서)는 인쇄 그대로", "note"), 4, 0, 1, 4)
    pose_w = W.QWidget()
    pl = W.QHBoxLayout(pose_w)
    pl.setContentsMargins(0, 0, 0, 0)
    pl.addWidget(label("보드 표시 자세", "cap"))
    spins = []
    for name, lo_, hi_, stp in (("x mm", -800, 800, 1), ("y mm", -800, 800, 1), ("yaw°", -180, 180, 0.5)):
        s = W.QDoubleSpinBox()
        s.setRange(lo_, hi_)
        s.setSingleStep(stp)
        s.setDecimals(1)
        s.setFocusPolicy(Qt.ClickFocus)   # 시작 때 포커스를 가져가 방향키를 먹지 않게(누른 칸만)
        pl.addWidget(label(name, "cap"))
        pl.addWidget(s)
        spins.append(s)
    lbl_src = label("", "note")
    lbl_src.setToolTip("gui · file(x, y, yaw) 는 옮길 수 있다. npz · T_base_board 는 고정")
    pl.addWidget(lbl_src)
    pl.addStretch(1)
    tg.addWidget(pose_w, 5, 0, 1, 4)
    right.addWidget(g_tg, 5)

    g_rec = W.QGroupBox("기록")
    rg = W.QGridLayout(g_rec)
    b_rec = btn("● 기록", lambda: ctl.call("record", st["bid"], st["target"], st["boards"][st["bid"]].copy(),
                                           json.loads(json.dumps(srcs[st["bid"]]))),
                "접촉 표시등 + 최근 0.5 s 접촉 ≥ 90 % + 팔 정지일 때만", "record")
    lbl_why = W.QLabel()
    lbl_dir = label("", "note")
    table = W.QTableWidget(0, 7)
    table.setHorizontalHeaderLabels(["#", "보드", "목표", "방위°", "고각°", "Δg", "지움"])
    table.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(W.QAbstractItemView.SelectRows)
    table.setFocusPolicy(Qt.NoFocus)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.setMinimumHeight(110)
    b_del = btn("선택 행 지움/되살림", lambda: ctl.call("delete", table.currentRow()) if table.currentRow() >= 0 else None)
    lbl_cnt = W.QLabel()
    # 줄바꿈 글자는 열 높이를 희망 높이로 고정해(heightForWidth) 1080 px 화면에서 그림 칸과 겹쳤다 — 한 줄로 두고 넘치면 잘림 · 전문은 툴팁
    for lb_ in (lbl_why, lbl_cnt):
        lb_.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Preferred)
    rg.addWidget(b_rec, 0, 0, 1, 2)
    rg.addWidget(lbl_why, 1, 0, 1, 2)
    rg.addWidget(lbl_dir, 2, 0, 1, 2)
    rg.addWidget(table, 3, 0, 1, 2)
    rg.addWidget(b_del, 4, 0)
    rg.addWidget(lbl_cnt, 4, 1)
    lbl_j = W.QLabel()
    lbl_j.setTextFormat(Qt.RichText)
    lbl_j.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
    rg.addWidget(lbl_j, 5, 0, 1, 2)
    right.addWidget(g_rec, 2)

    win.setCentralWidget(root)
    ag = (win.screen() or app.primaryScreen()).availableGeometry()   # 1080 px 화면에서 아래 동작 줄이 잘렸다(11.05)
    win.resize(min(1480, ag.width()), min(980, ag.height() - 40))   # 40 = 제목줄 몫

    # ---------------- 보드 · 목표
    def all_pts() -> list:
        """[(board_id, 목표 index, base xyz)] — 클릭 선택 · 그리기."""
        return [(b, i, p) for b, T in st["boards"].items() for i, p in enumerate(B.to_base(B.BOARDS[b]["targets"], T))]

    def draw_board() -> None:
        for t in labels:
            t.remove()
        labels.clear()
        A = np.array([p for _, _, p in all_pts()])
        c2.set_data(A[:, 1] * 1000, A[:, 0] * 1000)
        recs = ctl.records()
        done = {(p["board_id"], p["target_id"]) for p in recs if not p["deleted"]}
        Pc, Pr = np.zeros((0, 3)), np.zeros((0, 3))
        for b, T in st["boards"].items():
            C, sp = B.to_base(B.BOARDS[b]["targets"], T), B.BOARDS[b]
            Pc = np.r_[Pc, C[list(sp["outer"])]]
            Pr = np.r_[Pr, C[[i for i, n in enumerate(sp["names"]) if (b, n) in done]].reshape(-1, 3)]
        cor2.set_data(Pc[:, 1] * 1000, Pc[:, 0] * 1000)
        rec2.set_data(Pr[:, 1] * 1000, Pr[:, 0] * 1000)
        s = ctl.get_snap()
        if s:   # 자동 차례 번호(지금 손끝에서 탐욕 최근접 — 기록 · 맞춤 · 자동 목표가 바뀔 때 다시)
            start = kin.tip(s["cmd"] + s["e0"], s["hand"])[:3, 3]
            for k, (_b, _i, x) in enumerate(B.auto_queue(recs, st["fits"], {tuple(v) for v in s["auto_skip"]}, start, st["bid"])):
                labels.append(ax2.annotate(str(k + 1), (x[1] * 1000, x[0] * 1000), xytext=(-9, -10), textcoords="offset points",
                                           fontsize=6, color=BLUE, clip_on=True))
        for b, T in st["boards"].items():
            sp = B.BOARDS[b]
            C = B.to_base(sp["targets"], T)
            grid = sp["type"] != "charuco"
            names = [n.split("_")[1] if grid else str(i) for i, n in enumerate(sp["names"])]   # grid: c0..c3, charuco: corner id
            labels.extend(ax2.annotate(names[i], (C[i, 1] * 1000, C[i, 0] * 1000), xytext=(3, 3), textcoords="offset points",
                                       fontsize=8, fontweight="bold", clip_on=True) for i in range(len(C)))
            for xy, text in sp["marks"]:   # 마커 이름은 연하게 작게(목표 글자와 겹쳐도 목표가 읽히게)
                m = B.to_base(np.r_[xy, 0.0][None], T)[0]
                labels.append(ax2.text(m[1] * 1000, m[0] * 1000, text, fontsize=5, color="0.72", ha="center", va="center",
                                       clip_on=True, zorder=0))
            labels.append(ax2.text(C[:, 1].max() * 1000, C[:, 0].max() * 1000 + 6, b, fontsize=8, color="tab:purple", clip_on=True))

    def show_pose() -> None:
        b = st["bid"]
        x, y, _z, yaw = B.board_xyyaw(st["boards"][b])
        for s, v in zip(spins, (x * 1000, y * 1000, yaw)):
            s.blockSignals(True)
            s.setValue(v)
            s.setEnabled(bool(srcs[b].get("editable")))
            s.blockSignals(False)
        lbl_src.setText(srcs[b]["source"].split(" ")[0] + ("" if srcs[b].get("editable") else " 고정"))

    def board_edit() -> None:
        b = st["bid"]
        if not srcs[b].get("editable"):
            return
        z = float(st["boards"][b][2, 3])
        st["boards"][b] = B.board_T(spins[0].value() / 1000, spins[1].value() / 1000, z, spins[2].value())
        srcs[b].update(T_base_board=st["boards"][b].tolist(), x=spins[0].value() / 1000, y=spins[1].value() / 1000, z=z,
                       yaw_deg=spins[2].value())
        ctl.call("est", b, spins[2].value())   # 맞춤 yaw 검사(180° 순서) 기준
        draw_board()

    for s in spins:
        s.editingFinished.connect(guarded("board_edit", board_edit))
        s.editingFinished.connect(s.clearFocus)   # Enter 뒤 방향키를 다시 조그로

    def on_board(i: int) -> None:
        st["bid"] = bids[i]
        ctl.call("board", st["bid"])   # 자동은 이 보드 안에서만 — 다른 보드에서 돌던 자동은 제어 스레드가 해제
        cb_c.blockSignals(True)
        cb_c.clear()
        cb_c.addItems(B.BOARDS[st["bid"]]["names"])
        cb_c.blockSignals(False)
        cb_c.setCurrentIndex(0)
        st["target"] = 0
        show_pose()

    cb_b.currentIndexChanged.connect(guarded("board_select", on_board))
    i0 = bids.index(ctl.sel_bid) if ctl.sel_bid in bids else 0   # --seed: 불러온 보드를 고른 채 시작
    cb_b.blockSignals(True)
    cb_b.setCurrentIndex(i0)
    cb_b.blockSignals(False)
    on_board(i0)

    def on_click(ev) -> None:
        if ev.inaxes is not ax2 or ev.xdata is None:
            return
        pts = all_pts()
        d = np.linalg.norm(np.array([(p[1], p[0]) for _, _, p in pts]) * 1000 - [ev.xdata, ev.ydata], axis=1)   # 그림 좌표 = (y, x)
        if d.min() < 15:
            b, i, _ = pts[int(d.argmin())]
            cb_b.setCurrentIndex(bids.index(b))   # on_board 가 목표 목록을 바꾼 뒤
            cb_c.setCurrentIndex(i)

    canvas.mpl_connect("button_press_event", guarded("click", on_click))
    draw_board()

    # ---------------- 키보드: 방향키 · PageUp/Down · 글자 키(B.KEYS). 누름 · 뗌을 보고 반복은 hold_tick 이 한다
    keymap = {Qt.Key_Up: ("x", 1), Qt.Key_Down: ("x", -1), Qt.Key_Left: ("y", 1), Qt.Key_Right: ("y", -1),
              Qt.Key_PageUp: ("z", 1), Qt.Key_PageDown: ("z", -1)}
    keymap |= {getattr(Qt, f"Key_{k.upper()}"): v for k, v in B.KEYS.items()}

    class KeyFilter(QtCore.QObject):
        def eventFilter(self, obj, ev) -> bool:   # noqa: N802 — Qt 이름
            t = ev.type()
            if t == QtCore.QEvent.KeyPress and ev.key() in actkeys and not st["dead"]:
                if isinstance(app.focusWidget(), (W.QAbstractSpinBox, W.QLineEdit)):
                    return False
                if not ev.isAutoRepeat():
                    guarded("key", actkeys[ev.key()])()
                return True
            if t in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease) and ev.key() in keymap:
                if isinstance(app.focusWidget(), (W.QAbstractSpinBox, W.QLineEdit)):
                    return False   # 보드 자세 입력 중에는 키를 그 칸에
                if not ev.isAutoRepeat():
                    guarded("key", press)(keymap[ev.key()]) if t == QtCore.QEvent.KeyPress else release(keymap[ev.key()])
                return True
            return False

    kf = KeyFilter(win)
    app.installEventFilter(kf)
    def on_state(state) -> None:
        release()   # 창이 포커스를 잃으면 뗌을 못 받으니 반복을 끊는다
        if state != Qt.ApplicationActive:   # 조건 없이(스냅숏이 늦어도) — 자동이 없으면 제어 스레드가 무시
            ctl.call("auto", False, "창 포커스 잃음")

    app.applicationStateChanged.connect(guarded("focus", on_state))

    def enter() -> None:   # 판정은 제어 스레드(auto_descend): 자동 확인 대기에서만 하강, 그 밖에는 무시하고 상태 띠에
        ctl.call("auto_descend")

    actkeys = {Qt.Key_N: lambda: ctl.call("auto", True), Qt.Key_X: lambda: ctl.call("skip"), Qt.Key_P: lambda: ctl.call("point"),
               Qt.Key_Return: enter, Qt.Key_Enter: enter}
    assert not set(actkeys) & set(keymap), "단축키가 조그 키와 겹친다"
    W.QShortcut(QtGui.QKeySequence(Qt.Key_Space), win, activated=stop)
    hold_timer = QtCore.QTimer()
    hold_timer.timeout.connect(guarded("hold", hold_tick))
    hold_timer.start(30)

    # ---------------- 그리기
    def paint(w: W.QWidget, bg: str, fg: str, size: int = 16) -> None:
        w.setStyleSheet(f"background:{bg}; color:{fg}; font-size:{size}px; font-weight:bold; border-radius:6px; padding:4px 10px;")

    def tone(m: float, bad: float, warn: float) -> str:
        return RED if m < bad else ("#B45309" if m < warn else "#0F172A")

    def color(q, j) -> str:
        return tone(min(q - lo[j], hi[j] - q), 0.05, 0.1)

    def mark_dead() -> None:
        """제어 스레드가 죽으면 조그 · 발행 · 기록을 막고 창을 경고색으로(M4)."""
        st["dead"] = True
        hold["key"] = None
        for w in (chk, g_move, g_rot, g_step, acts_w, acts2_w, acts3_w, g_rec):
            w.setEnabled(False)
        root.setStyleSheet("QWidget#root { background: #F1948A; }")

    def refresh() -> None:
        if not ctl.alive and not st["closing"] and not st["dead"]:
            mark_dead()
        s = ctl.get_snap()
        if not s:
            lbl_msg.setText(ctl.msg)
            paint(lbl_mode, GRAY, "#FFFFFF")
            lbl_mode.setText("시작 검사 중")
            return
        h = s["hmeas"] if s["hmeas"] is not None else s["hand"]
        q = s["meas"] if s["meas"] is not None else s["cmd"]
        T, Tc = kin.tip(q, h), kin.tip(s["cmd"], s["hand"])
        # 상태 띠
        if st["dead"]:
            paint(lbl_mode, RED, "#FFFFFF")
            lbl_mode.setText("제어 멈춤")
            lbl_msg.setText(f"★ 제어 스레드 멈춤 — 조그 · 발행 막힘: {ctl.fatal or ctl.msg}")
        else:
            paint(lbl_mode, ORANGE, "#000000", 18) if s["pub"] else paint(lbl_mode, GRAY, "#FFFFFF")
            lbl_mode.setText("★ 실기 발행 중" if s["pub"] else ("DEMO · 발행 0" if ctl.demo else "DRY · 발행 0"))
            lbl_msg.setText(f"[{s['mode']}] {s['msg']}")
        chk_free.blockSignals(True)   # 제어 스레드가 끈 '확인 없이' 를 체크에도
        chk_free.setChecked(bool(s["auto_free"]))
        chk_free.blockSignals(False)
        chk.blockSignals(True)
        chk.setChecked(s["pub"])
        chk.setEnabled(ctl.execute and not ctl.demo and not s["start_fail"] and not st["dead"])
        chk.blockSignals(False)
        if s["contact"]:
            paint(lamp, GREEN, "#000000", 20)
        else:
            paint(lamp, AMBER, "#000000", 18) if s["stale"] else paint(lamp, LIGHT, "#374151", 18)
        lamp.setText("● 접촉" if s["contact"] else ("힘 끊김" if s["stale"] else "○ 무접촉"))
        dg = None if s["jf"] is None else float(s["jf"][B.INDEX_JF] - s["jf0"])
        dn = None if s["tip"] is None else float(s["tip"][B.INDEX_TIP] - s["tip0"])
        bar_g.setValue(int(np.clip(abs(dg or 0), 0, 300)))
        val_g.setText("—" if dg is None else f"Δ {dg:+.0f} g")
        bar_n.setValue(int(np.clip((dn or 0) * 100, 0, 200)))
        val_n.setText("—" if dn is None else f"Δ {dn:+.2f} N")
        zm = (Tc[2, 3] - ctl.z_floor) * 1000
        lbl_z.setText(f"{zm:+.1f} mm" if np.isfinite(zm) else "없음")
        lbl_z.setStyleSheet(f"color:{tone(zm, 3, 8)}")
        mj = np.minimum(q - lo, hi - q)
        j = int(np.argmin(mj))
        lbl_lim.setText(f"j{j + 1}  {mj[j]:.3f} rad")
        lbl_lim.setStyleSheet(f"color:{tone(mj[j], 0.05, 0.1)}")
        lbl_tr.setText("— (dry)" if not s["pub"] or s["track"] is None else f"{s['track']:.3f} rad")
        e6 = B.pose_err(Tc, T)   # 손끝 실측 FK − 명령 6D(유지 켬이면 0 으로 간다)
        lbl_hold.setText(f"유지 보정 c 최대 {np.abs(s['hold_c']).max():.3f} rad · {s['hold_state']} · 손끝 실측−명령 "
                         f"{np.linalg.norm(e6[:3]) * 1000:.1f} mm {np.degrees(np.linalg.norm(e6[3:])):.1f}°")
        if ctl.cam is not None:
            o = ctl.cam.owner()
            b_cam_on.setEnabled(o == "none")   # 제어 스레드가 죽어도 카메라는 다룬다
            b_cam_off.setEnabled(o == "ours")
        cam = ctl.cam.status() if ctl.cam is not None else ("사진 끔" if ctl.photo is None else "카메라 관리 없음(demo · --photo_topic '')")
        lbl_cam.setText(cam + (f" · {s['photo_note']}" if s["photo_note"] else ""))
        lbl_cam.setToolTip(lbl_cam.text())
        pd = s["pd"]
        lbl_pd.setText(f"pd {pd['phase']} · {pd['target']} · estop {pd['estop']}")
        # 보드 자세: 받아들인 맞춤이 표시 자세를 덮고(그림 · 목표 · sim), 맞춤이 사라지면 추정으로 되돌린다
        redraw = False
        if s["fit_ver"] != st["fit_ver"]:
            st["fit_ver"], st["fits"], redraw = s["fit_ver"], ctl.get_fits(), True
            for b_ in bids:
                f = st["fits"].get(b_)
                if f is not None and f["ok"]:
                    st["est"].setdefault(b_, (st["boards"][b_].copy(), srcs[b_]))
                    st["boards"][b_], srcs[b_] = f["T"].copy(), B.A.fit_pose(f)
                elif b_ in st["est"]:
                    st["boards"][b_], srcs[b_] = st["est"].pop(b_)
            show_pose()
        b = st["bid"]
        f, who = st["fits"].get(b), (f" [{b}]" if len(bids) > 1 else "")
        if f is not None and f["ok"]:
            lbl_fit.setText(f"보드 자세{who}: {f['kind']} 맞춤 (잔차 최대 {f['res_mm'].max():.1f} mm)" + (f" · ★{f['warn']}" if f["warn"] else ""))
            lbl_fit.setStyleSheet(f"font-size:14px; font-weight:bold; color:{'#B45309' if f['warn'] else '#166534'};")
        else:
            src = (st["est"].get(b) or (None, srcs[b]))[1]["source"]
            est = f"보드 자세{who}: {'카메라' if src.startswith('npz') else '입력'} 추정(미확정)"
            lbl_fit.setText(est + ("" if f is None else f" · ★4점 맞춤 거부: {f['why']}"))
            lbl_fit.setStyleSheet(f"font-size:14px; font-weight:bold; color:{RED if f is not None else GRAY};")
        a = s["auto"]
        key = None if a is None else (a["bid"], a["ti"])
        if key != st["auto_key"]:   # 자동 목표를 목표 칸 · 그림 · sim 목표로
            st["auto_key"], redraw = key, True
            if key is not None:
                cb_b.setCurrentIndex(bids.index(key[0]))
                cb_c.setCurrentIndex(key[1])
        if a is None and s["summary"]:
            lbl_auto.setText(f"자동 끝 — {s['summary']}")
            lbl_auto.setToolTip(lbl_auto.text())
            lbl_auto.setStyleSheet("font-size:13px; font-weight:bold; color:#1E3A8A;")
        elif a is None:
            lbl_auto.setText(("남은 점 전체 자동 대기 — 맞춤 받아들인 뒤 N(처음 "
                              f"{ctl.confirm_first} 점은 Enter 확인)" if ctl.auto_all else "자동 꺼짐 — 4 모서리 맞춤 뒤 [자동 다음 점]/N")
                             + (" · 확인 없이 하강 켬" if s["auto_free"] else ""))
            lbl_auto.setStyleSheet(f"font-size:14px; color:{GRAY};")
        else:
            lbl_auto.setText(B.auto_wait_text(a["name"]) if a["phase"] == "wait" else
                             f"자동 · {a['name']} · {B.AUTO_PH[a['phase']]}" + (" · 확인 없이 하강" if s["auto_free"] else "") + " · 정지 = space")
            lbl_auto.setStyleSheet("font-size:14px; font-weight:bold; padding:2px 6px; border-radius:4px; "
                                   + ("background:#FEF3C7; color:#000000;" if a["phase"] == "wait" else "background:#EDE9FE; color:#4C1D95;"))
        # 목표
        c = B.to_base(B.BOARDS[b]["targets"], st["boards"][b])[st["target"]]
        ctl.view = {"boards": dict(st["boards"]), "target": c.copy()}   # sim 송신(별 스레드)이 통째로 읽는다
        d = (c - (Tc if ctl.hold_on else T)[:3, 3]) * 1000   # 유지 켬: 명령 6D 자세 = 실측이 갈 곳 기준
        word = (("전진", "후진"), ("왼쪽", "오른쪽"), ("위", "아래"))
        lbl_d.setText("   ".join(f"d{a} {v:+6.1f} ({word[k][v < 0]})" for k, (a, v) in enumerate(zip("xyz", d))) + "  mm")
        tgt2.set_data([c[1] * 1000], [c[0] * 1000])
        tip2m.set_data([T[1, 3] * 1000], [T[0, 3] * 1000])
        tip2c.set_data([Tc[1, 3] * 1000], [Tc[0, 3] * 1000])
        span = max(60.0, float(np.abs(d[:2]).max()) + 20)   # 목표가 멀면 넓혀 같이 보이게
        if a is not None:   # 자동 중에는 보드 전체(모서리 · 기록 · 차례 번호)가 보이게
            span = max(span, float(np.abs(B.to_base(B.BOARDS[b]["targets"], st["boards"][b])[:, :2] - T[:2, 3]).max()) * 1000 + 15)
        r = ax2.bbox.width / ax2.bbox.height if ax2.bbox.height > 0 else 1.0   # 칸 가로세로 비 — 같은 축척을 범위로 맞춰
        hy = max(span, span / r)                                              # datalim 이 한쪽 범위를 줄이지 않게
        ax2.set_xlim(T[1, 3] * 1000 + hy * r, T[1, 3] * 1000 - hy * r)   # 뒤집음: 왼쪽 = +y
        ax2.set_ylim(T[0, 3] * 1000 - hy, T[0, 3] * 1000 + hy)
        canvas.draw_idle()
        # 기록
        ok = bool(s["can_record"]) and ctl.doc is not None and not st["dead"]
        b_rec.setEnabled(ok)
        why = ("제어 스레드 멈춤" if st["dead"] else "--out 없음(demo) — 기록하지 않는다" if ctl.doc is None else s["rec_why"])
        lbl_why.setText("기록할 수 있다" if ok else f"기록 불가: {why}")
        lbl_why.setToolTip(lbl_why.text())
        lbl_why.setStyleSheet(f"color:{'#166534' if ok else '#B45309'}; font-weight:bold;")
        az, el = B.dir_deg(T[:3, 2])
        lbl_dir.setText(f"손끝 z 축 방위 {az:+.0f}° · 고각 {el:+.0f}° (같은 점을 다른 손목 방향으로도 기록)")
        cell = [f"<td style='color:{color(q[i], i)}'>j{i + 1} {q[i]:+.3f} 여유 {mj[i]:.3f}&nbsp;&nbsp;</td>" for i in range(7)]
        cell.append("<td>" + ("" if s["meas"] is not None else "(측정 없음 — 명령)") + "</td>")
        lbl_j.setText("<table>" + "".join(f"<tr>{''.join(cell[r:r + 4])}</tr>" for r in (0, 4)) + "</table>")   # 2 줄(높이 절약)
        if redraw or s["records_ver"] != st["rec_ver"]:
            draw_board()
        if s["records_ver"] != st["rec_ver"]:
            st["rec_ver"] = s["records_ver"]
            recs = ctl.records()
            table.setRowCount(len(recs))
            for i, p in enumerate(recs):
                vals = [i, p["board_id"], p["target_id"], f"{p['tip_dir_deg'][0]:+.0f}", f"{p['tip_dir_deg'][1]:+.0f}",
                        p["force"]["index_jf_delta_g"], "예" if p["deleted"] else ""]
                for k, v in enumerate(vals):
                    table.setItem(i, k, W.QTableWidgetItem("—" if v is None else str(v)))
            live: dict = {}
            for p in recs:
                if not p["deleted"]:
                    live.setdefault((p["board_id"], p["target_id"]), []).append(p["tip_dir_base"])
            lbl_cnt.setText("목표별 기록 수(손목 방향 수): "
                            + (", ".join(f"{b}:{t} {len(v)}({B.dir_count(v)})" for (b, t), v in sorted(live.items())) or "없음"))
            lbl_cnt.setToolTip(lbl_cnt.text())

    timer = QtCore.QTimer()
    timer.timeout.connect(guarded("refresh", refresh))
    timer.start(100)

    def close_all(*_a) -> None:
        if not st["closing"]:
            win.close()

    for sg_ in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sg_, close_all)
    pump = QtCore.QTimer()   # Qt 루프 안에서도 파이썬 시그널 처리기가 돌게
    pump.timeout.connect(lambda: None)
    pump.start(200)

    def close_event(ev) -> None:
        st["closing"] = True
        hold["key"] = None
        try:
            lbl_msg.setText("끝내는 중 — 해제 · stop · 원본 저장")
            app.processEvents()
            ctl.stop()
        finally:
            ev.accept()

    win.closeEvent = close_event
    if smoke:
        from PyQt5.QtTest import QTest

        def click_test() -> None:
            """그림 좌표 (y, x) → 화면 픽셀 → QTest 클릭 → on_click 이 같은 (보드, 목표) 를 고르는지."""
            refresh()
            canvas.draw()
            (x0, x1), (y0, y1) = ax2.get_xlim(), ax2.get_ylim()
            cur = (st["bid"], st["target"])
            seen = [(b, i, p) for b, i, p in all_pts() if (b, i) != cur
                    and min(x0, x1) < p[1] * 1000 < max(x0, x1) and y0 < p[0] * 1000 < y1]
            b, i, p = max(seen, key=lambda t: (t[2][1], t[2][0]))   # 가장 왼쪽(+y) · 위(+x) 쪽 점 — 방향이 뒤집히면 다른 점이 잡힌다
            px, py = ax2.transData.transform((p[1] * 1000, p[0] * 1000))
            r = getattr(canvas, "device_pixel_ratio", 1) or 1
            QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QtCore.QPoint(int(round(px / r)), int(round((fig.bbox.height - py) / r))))
            ok = (st["bid"], st["target"]) == (b, i)
            B.say(f"스모크 클릭: {b}:{B.BOARDS[b]['names'][i]} (base x {p[0] * 1000:.1f}, y {p[1] * 1000:.1f} mm) → "
                  f"{st['bid']}:{B.BOARDS[st['bid']]['names'][st['target']]} {'OK' if ok else '틀림'}")
            if not ok:
                st["error"] = st["error"] or "smoke-click"

        def click_all() -> None:
            """btn 으로 만든 버튼을 전부 마우스로 한 번씩(demo) → fail() 이 불리면 st["error"] 가 남는다. 정지로 끝낸다."""
            for b in clickable + [b_stop]:
                QTest.mouseClick(b, Qt.LeftButton)
            B.say(f"스모크 버튼 클릭 {len(clickable)} 개 → {'OK' if st['error'] is None else '틀림 ' + st['error']}")

        def shot() -> None:
            click_all()
            if st["closing"]:
                return
            refresh()
            canvas.draw()
            win.grab().save(smoke)
            B.say(f"스모크 화면 → {smoke} · 창 {win.width()}x{win.height()} (최소 {win.minimumSizeHint().height()}) · "
                  f"PageDown 누르고 있기로 조그 {st['n_jog']} 회")
            win.close()

        QtCore.QTimer.singleShot(1200, lambda: QTest.keyPress(win, Qt.Key_PageDown))   # 누르고 있기 → 이동이 끝날 때마다 1 mm 씩
        QtCore.QTimer.singleShot(2800, lambda: QTest.keyRelease(win, Qt.Key_PageDown))
        def auto_test() -> None:
            """(--out 이 있을 때) demo 가짜 4 모서리 → 맞춤이 표시 자세를 덮나 → 자동 다음 점 → 위 10 mm 확인 대기에서 화면."""
            if ctl.doc is None:
                QtCore.QTimer.singleShot(200, guarded("smoke", shot))
                return
            if ctl.get_snap().get("mode") != "idle":   # 앞 시험의 하강(면이 멀면 40 mm)이 끝난 뒤
                QtCore.QTimer.singleShot(300, guarded("smoke-auto", auto_test))
                return
            ctl.call("demo_corners", ctl.sel_bid)
            t0 = time.monotonic()

            def poll() -> None:
                s = ctl.get_snap()
                a = s.get("auto")
                if s.get("fit_ver", 0) > 0 and a is None and not st.get("auto_sent"):
                    st["auto_sent"] = True
                    ctl.call("auto", True)
                if (a and a["phase"] == "wait") or time.monotonic() - t0 > 20:
                    ok = bool(a and a["phase"] == "wait") and "맞춤" in lbl_fit.text() and srcs[ctl.sel_bid]["source"].startswith("fit")
                    B.say(f"스모크 자동: {lbl_fit.text()} · {lbl_auto.text()} → {'OK' if ok else '틀림 — ' + ctl.msg}")
                    if not ok:
                        st["error"] = st["error"] or "smoke-auto"
                    QtCore.QTimer.singleShot(300, guarded("smoke", shot))
                    return
                QtCore.QTimer.singleShot(200, guarded("smoke-auto", poll))

            QtCore.QTimer.singleShot(300, guarded("smoke-auto", poll))

        QtCore.QTimer.singleShot(4200, lambda: ctl.call("descend"))
        QtCore.QTimer.singleShot(10000, guarded("smoke-click", click_test))
        QtCore.QTimer.singleShot(10300, guarded("smoke-auto", auto_test))
    win.show()
    # QSS 가 들어간 뒤(보인 뒤)에 재야 스크롤 칸이 패드 오른쪽을 자르지 않는다
    QtCore.QTimer.singleShot(0, lambda: scroll.setMinimumWidth(pads_w.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()))
    try:
        app.exec_()
    finally:
        sys.excepthook = sys.__excepthook__
        app.removeEventFilter(kf)
    return 0 if ctl.fatal is None and st["error"] is None else 1
