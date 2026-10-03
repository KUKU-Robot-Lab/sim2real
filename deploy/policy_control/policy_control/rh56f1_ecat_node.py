"""RH56F1 손 EtherCAT 드라이버 노드 — 벤더 RS485 드라이버(robot_control rh56f1_driver)와 같은 토픽 · 메시지.

10.02 사용자: RS485 대신 EtherCAT(정책 제어 포함). 1 kHz 루프는 자식 프로세스 tools/ethercat/rh56f1_ecat_master
(SOEM · cap_net_raw)가 돌고, 이 노드는 ROS 쪽만 맡는다(setcap 실행 파일은 ROS 라이브러리를 못 읽는다).

  발행  /hand_<side>/angle_actual  (GetAngleAct1, 슬롯 순 새끼부터 · 벤더와 같은 단위)   state_hz
        /hand_<side>/force_actual  (GetForceAct1)  · /hand_<side>/current_actual (GetCurrentAct1)
        /hand_<side>/touch_data    (TouchData1 — palm_data 는 3 영역 × (법선 · 접선 · 방향))
        /hand_<side>/ecat_status   (std_msgs/String JSON, 1 Hz — OP · WKC · 왕복 · 오류 · 상태 · 온도)
  구독  /hand_<side>/angle_set (SetAngle1, -1 = 그 축 유지) · force_set (SetForce1) · speed_set (SetSpeed1)

    python3 deploy/policy_control/policy_control/rh56f1_ecat_node.py --side right            # config/rh56f1_ports.yaml
    python3 deploy/policy_control/policy_control/rh56f1_ecat_node.py --side right --no-op    # SAFE_OP · 상태만(손 무동작)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

if __package__ in (None, ""):              # 파일 경로로 띄울 때(미션 명령)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import rh56f1_ecat as E  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PORTS = Path(__file__).resolve().parents[1] / "config" / "rh56f1_ports.yaml"
HEARTBEAT_S = 0.1
FIRST_STATE_TIMEOUT_S = 15.0


def ecat_config(ports: dict, side: str) -> tuple[str, dict]:
    """(ifname, ethercat 블록). 좌우 NIC 가 같으면 거부 — 손 하나 = NIC 하나(일반 스위치는 폭주한다)."""
    hand = ports.get(side) or {}
    other = ports.get("left" if side == "right" else "right") or {}
    if hand.get("transport") != "ethercat":
        raise E.EcatError(f"{side}: transport 가 ethercat 이 아니다({hand.get('transport')!r})")
    ifname = str(hand.get("ifname") or "")
    if not ifname:
        raise E.EcatError(f"{side}: ifname 이 없다")
    if other.get("transport") == "ethercat" and other.get("ifname") == ifname:
        raise E.EcatError(f"좌우 손이 같은 NIC {ifname} — 손 하나에 NIC 하나")
    return ifname, dict(ports.get("ethercat") or {})


class MasterLink:
    """마스터 프로세스 + 유닉스 데이터그램 소켓. ROS 없음(테스트에서 가짜 마스터와 붙인다)."""

    def __init__(self, side: str, sock_dir: str | None = None):
        self.dir = sock_dir or tempfile.mkdtemp(prefix=f"rh56f1_ecat_{side}_")
        self.master_sock = os.path.join(self.dir, "master.sock")
        self.node_sock = os.path.join(self.dir, "node.sock")
        if os.path.exists(self.node_sock):
            os.unlink(self.node_sock)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(self.node_sock)
        self.sock.settimeout(0.5)
        self.proc: subprocess.Popen | None = None
        self.send_errors = 0

    def start(self, argv: list[str]) -> None:
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def send(self, payload: bytes) -> bool:
        try:
            self.sock.sendto(payload, self.master_sock)
            return True
        except (FileNotFoundError, ConnectionRefusedError, BlockingIOError, OSError):
            self.send_errors += 1
            return False

    def recv(self) -> E.EcatState | None:
        try:
            buf = self.sock.recv(E.STATE_SIZE + 16)
        except socket.timeout:
            return None
        return E.unpack_state(buf)

    def stop(self, timeout: float = 3.0) -> int | None:
        rc = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()                 # 마스터: hold 50 주기 → INIT
            try:
                rc = self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                rc = self.proc.wait()
        elif self.proc is not None:
            rc = self.proc.returncode
        self.sock.close()
        for p in (self.node_sock, self.master_sock):
            if os.path.exists(p):
                os.unlink(p)
        return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("right", "left"), required=True)
    ap.add_argument("--ports", default=str(DEFAULT_PORTS))
    ap.add_argument("--no-op", action="store_true", help="SAFE_OP 에 머문다 — 상태만 읽고 손은 명령을 쓰지 않는다")
    ap.add_argument("--op-enable", action="store_true", help="OP 실험: 명령 전에도 ENABLE_SET 을 켠다(목표 = 지금 각도)")
    ap.add_argument("--sync-type", type=int, default=None, help="OP 실험: 0x1C32/33:01 에 쓸 값(0 free run · 1 SM 동기)")
    args, _ = ap.parse_known_args(argv)

    ifname, cfg = ecat_config(yaml.safe_load(Path(args.ports).read_text()) or {}, args.side)
    if args.op_enable:
        cfg["op_enable"] = True
    if args.sync_type is not None:
        cfg["sync_type"] = args.sync_type
    binary = str((REPO / cfg.get("master", "tools/ethercat/rh56f1_ecat_master")).resolve())
    if not os.access(binary, os.X_OK):
        print(f"✗ 마스터 {binary} 가 없다 — bash tools/ethercat/build.sh 뒤 sudo setcap cap_net_raw,cap_net_admin=ep {binary}")
        return 1
    link = MasterLink(args.side)
    margv = E.master_argv(binary, ifname, link.master_sock, link.node_sock, cfg, args.no_op)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from rh56f1_interfaces.msg import (GetAngleAct1, GetCurrentAct1, GetForceAct1, SetAngle1, SetForce1, SetSpeed1,
                                       TouchData1)

    rclpy.init()
    node = Node(f"rh56f1_ecat_{args.side}")
    log = node.get_logger()
    ns = f"/hand_{args.side}"
    names = E.joint_names(args.side)
    hand_id = int((yaml.safe_load(Path(args.ports).read_text()) or {}).get(args.side, {}).get("hand_id", 1))
    pubs = {"angle": node.create_publisher(GetAngleAct1, f"{ns}/angle_actual", 10),
            "force": node.create_publisher(GetForceAct1, f"{ns}/force_actual", 10),
            "current": node.create_publisher(GetCurrentAct1, f"{ns}/current_actual", 10),
            "touch": node.create_publisher(TouchData1, f"{ns}/touch_data", 10),
            "status": node.create_publisher(String, f"{ns}/ecat_status", 10)}
    book = E.CommandBook()
    lock = threading.Lock()
    last = {"state": None, "status_t": 0.0, "count": 0}

    from builtin_interfaces.msg import Time

    def stamp_of(s: E.EcatState) -> Time:
        ns_ = E.sample_time_ns(node.get_clock().now().nanoseconds, time.monotonic_ns(), s.t_ns)
        return Time(sec=ns_ // 1_000_000_000, nanosec=ns_ % 1_000_000_000)

    def six(msg_type, values, stamp):
        m = msg_type()
        m.header.stamp = stamp
        m.header.frame_id = ns
        m.hand_id = hand_id
        m.joint_values = [int(v) for v in values]
        m.joint_names = names
        return m

    def publish(s: E.EcatState) -> None:
        st = stamp_of(s)                        # ★10.03 하드웨어 샘플 시각 — 네 토픽이 같은 도장(bag 에서 한 샘플로 묶인다)
        pubs["angle"].publish(six(GetAngleAct1, s.angle, st))
        pubs["force"].publish(six(GetForceAct1, s.force, st))
        pubs["current"].publish(six(GetCurrentAct1, s.current, st))
        t = TouchData1()
        t.header.stamp = st
        for k, v in s.touch().items():
            setattr(t, k, [int(x) for x in v])
        pubs["touch"].publish(t)
        now = time.monotonic()
        if now - last["status_t"] >= 1.0:
            last["status_t"] = now
            d = s.summary()
            d.update(ifname=ifname, no_op=args.no_op, states=last["count"], send_errors=link.send_errors)
            pubs["status"].publish(String(data=json.dumps(d, ensure_ascii=False)))

    def on_cmd(kind: str, values) -> None:
        vals = list(values)
        try:
            payload = {"angle": book.angle, "force": book.force, "speed": book.speed}[kind](vals)
        except E.EcatError as e:
            log.error(f"{kind}_set 거부: {e}")
            return
        if args.no_op and kind == "angle":
            log.warning("--no-op(SAFE_OP) — 각도 명령은 손이 쓰지 않는다", throttle_duration_sec=5.0)
        with lock:
            link.send(payload)

    node.create_subscription(SetAngle1, f"{ns}/angle_set", lambda m: on_cmd("angle", m.joint_values), 10)
    node.create_subscription(SetForce1, f"{ns}/force_set", lambda m: on_cmd("force", m.joint_values), 10)
    node.create_subscription(SetSpeed1, f"{ns}/speed_set", lambda m: on_cmd("speed", m.joint_values), 10)

    def heartbeat() -> None:
        with lock:
            link.send(E.CommandBook.heartbeat())
    node.create_timer(HEARTBEAT_S, heartbeat)

    stop = threading.Event()

    def pump_log() -> None:
        for line in link.proc.stdout:          # type: ignore[union-attr]
            line = line.rstrip()
            # rclpy 는 같은 호출 자리에서 심각도를 바꾸면 ValueError — 자리를 나눈다
            if "✗" in line or "⚠" in line:
                log.warning(line)
            else:
                log.info(line)

    def pump_state() -> None:
        while not stop.is_set():
            try:
                s = link.recv()
            except E.EcatError as e:
                log.error(f"상태 해석 실패: {e}", throttle_duration_sec=5.0)
                continue
            if s is None:
                if link.proc is not None and link.proc.poll() is not None:
                    log.error(f"마스터가 끝났다(rc {link.proc.returncode}) — 노드를 끝낸다")
                    stop.set()
                    break
                continue
            last["state"], last["count"] = s, last["count"] + 1
            publish(s)

    log.info(f"{ns} EtherCAT {ifname} — 마스터 {' '.join(margv)}")
    link.start(margv)
    threading.Thread(target=pump_log, daemon=True).start()
    threading.Thread(target=pump_state, daemon=True).start()
    t0 = time.monotonic()
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
            if last["state"] is None and time.monotonic() - t0 > FIRST_STATE_TIMEOUT_S:
                log.error(f"마스터 상태가 {FIRST_STATE_TIMEOUT_S:.0f} s 안에 안 온다 — 손 전원 · 케이블 · setcap")
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        rc = link.stop()
        log.info(f"마스터 종료 rc {rc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if last["state"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
