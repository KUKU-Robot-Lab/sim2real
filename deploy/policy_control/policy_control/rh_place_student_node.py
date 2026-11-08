"""rh_place_student_node: 놓기 학생(깊이 + 홀더 · 컵 마스크 + 규칙 머리) 실기 노드 — rh_place_node(PlaceNode) 위에 정책만 바꾼다.

  in   PlaceNode 입력 전부(팔 · 손 · 홀더 자세 · 넘김 묶음 · joint_target)
       /camera/camera/aligned_depth_to_color/image_raw(16UC1 1280x720, 원시 — head_depth_node 는 거치지 않는다: 목이 움직이면 그 노드는
         깊이를 보류한다. 잡기 학생 경로를 그대로 두려고 이 노드가 프레임별 자세 · 바닥 컷 · 홀더 투영을 직접 한다)
       /head_mask/image · /head_mask/config(head_mask_node --policy-dir <이 묶음>, 바꾸지 않음) · /head/joint_states
  out  PlaceNode 출력 + /head/target_stream(std_msgs/String JSON {"seq","pan_deg","tilt_deg"}, 정책 틱마다 = 2 Hz 연속 모드 조건 충족)
  학생 묶음 = -p student_dir:=<student.yaml 폴더>(ckpt · 외부 파라미터 md5 핀) · -p target_class:=blue|pink|orange(빈 값 = 묶음 기본).
  /head/joint_states 가 HEAD_MAX_AGE_S 넘게 없으면 머리 명령을 멈추고 abort. 깊이 · 마스크가 낡으면 틱 오류(gap 예산 뒤 abort).

    python3 deploy/policy_control/policy_control/rh_place_student_node.py --ros-args -r __node:=rh_place_student_right \\
        -p ns:=right -p contract:=<rh_place_contract.json> -p robot:=… -p student_dir:=<묶음> [-p holder:=2]
"""
from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import codec  # noqa: E402
from policy_control import place_student as PS  # noqa: E402
from policy_control import student_mask as SM  # noqa: E402
from policy_control.pour_fj_node import Family, PourFjNodeError, main  # noqa: E402
from policy_control.rh_place_node import PlaceNode  # noqa: E402

SRC_DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"   # = head_depth_node.SRC_DEPTH
HEAD_TOPIC, STREAM_TOPIC = "/head/joint_states", "/head/target_stream"   # = head_motion.STREAM_TOPIC
HEAD_CMD_FAIL_MAX = 6    # look_at 풀이 연속 실패 틱 상한(60 Hz 에서 0.1 s) — 넘으면 abort(드라이버 0.5 s 멈춤보다 먼저)
FRAME_FAIL_WARN = 3      # 깊이 프레임 처리 연속 실패가 이만큼이면 status reasons 에 올린다
CARRY_STALE_WARN = 6     # 이송 중 학생 입력 낡음 연속 틱이 이보다 많으면 status reasons 에 올린다(60 Hz 에서 0.1 s)


def _ns(stamp) -> int:
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class PlaceStudentNode(PlaceNode):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)                             # 정책은 계약을 읽은 뒤에 만든다(chain 은 첫 reset 에 생긴다)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import String

        for name, default in (("student_dir", ""), ("target_class", ""), ("num_threads", 4), ("head_stream", True)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not str(p("student_dir")):
            raise PourFjNodeError("student_dir:= 학생 묶음 폴더가 필요하다")
        self._policy = PS.PlaceStudentPolicy(str(p("student_dir")), self.contract, str(p("target_class")),
                                             device=self.device, num_threads=int(p("num_threads")))
        self.fam = Family(**{**self.fam.__dict__, "chain": PS.StudentPlaceChain})
        self._holder_yaw: dict[int, float] = {}
        self._head_on = bool(p("head_stream"))
        self._raw = None                                   # 가장 새 원시 깊이(처리 전) — 일꾼이 하나만 처리한다(밀린 것은 버림)
        self._raw_ev, self._stop = threading.Event(), threading.Event()
        self._frame_err, self._frame_fail = "", 0
        self._head_fail, self._head_err = 0, ""
        self._mask_cfg: dict | None = None
        sp = self._policy
        self.create_subscription(Image, SRC_DEPTH, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, SM.MASK_TOPIC, sp.masks.on_msg, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, SM.MASK_CONFIG_TOPIC, self._on_mask_cfg, latched)
        self.create_subscription(JointState, HEAD_TOPIC,
                                 lambda m: sp.head.on_state(_ns(m.header.stamp), m.name, m.position, time.time_ns()), 50)
        self._pub_head = self.create_publisher(String, STREAM_TOPIC, 10)
        self._String = String
        self._worker = threading.Thread(target=self._work, name="place_depth", daemon=True)
        self._worker.start()
        self.get_logger().info(f"놓기 학생 ckpt md5 {sp.md5} · 목표 {sp.target} · 머리 스트림 {'켬' if self._head_on else '끔'}")

    # ---------------------------------------------------------------- inputs
    def _on_holder(self, msg, k: int | None = None) -> None:
        super()._on_holder(msg, k)
        try:                                                # 위치와 같이 디코딩에 성공한 자세만 yaw 로 남긴다
            quat = codec.decode_pose(msg).quat
        except (codec.CodecError, ValueError):
            return
        self._holder_yaw[self.holder_id if k is None else k] = _yaw(quat)

    def _on_mask_cfg(self, msg) -> None:
        import json
        try:
            self._mask_cfg = json.loads(msg.data)
        except ValueError as exc:
            self._mask_cfg, self._frame_err = None, f"{SM.MASK_CONFIG_TOPIC} JSON 아님: {exc}"

    def _on_depth(self, msg) -> None:
        self._raw = msg
        self._raw_ev.set()

    def _work(self) -> None:
        """원시 깊이 → Frame(바닥 컷 · 블록 중앙값 · 홀더 투영, 약 30 ms). 틱 스레드를 막지 않게 따로 돈다."""
        sp = self._policy
        while not self._stop.is_set():
            if not self._raw_ev.wait(0.5):
                continue
            self._raw_ev.clear()
            try:
                sp.frame = self._make_frame(self._raw)
                self._frame_err, self._frame_fail = "", 0
            except Exception as exc:   # noqa: BLE001 — 일꾼이 죽으면 깊이가 조용히 낡는다. 무엇이든 남기고 계속 돈다
                self._frame_fail += 1
                known = isinstance(exc, (ValueError, PS.PlaceInputStale))
                err = str(exc) if known else f"{type(exc).__name__}: {exc}"
                if re.sub(r"[-\d.]+", "#", err) != re.sub(r"[-\d.]+", "#", self._frame_err):   # 수치만 바뀐 같은 사유는 한 번만
                    msg = f"깊이 프레임 처리 실패 — {err}"
                    if known:   # rclpy 는 호출 줄마다 심각도를 고정한다. 한 줄에서 바꾸면 ValueError 로 일꾼이 죽는다
                        self.get_logger().warning(msg)
                    else:
                        self.get_logger().error(msg)
                self._frame_err = err

    def _make_frame(self, msg) -> PS.Frame:
        hold = self._holder
        if hold is None:
            raise ValueError("목표 홀더 자세가 아직 없다")
        if msg.encoding != "16UC1" or msg.step != msg.width * 2:
            raise ValueError(f"깊이 {msg.encoding} step {msg.step} — 16UC1 만")
        d = np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width)
        st = _ns(msg.header.stamp)
        if (time.time_ns() - st) * 1e-9 < PS.MIN_AGE_S:          # 미래 프레임은 버린다(마지막 정상 프레임 유지)
            raise ValueError(f"깊이 stamp 가 {(st - time.time_ns()) * 1e-9:.3f} s 미래 — 시계 어긋남?")
        T = self._policy.hc.cam_pose_from_encoder(*self._policy.head.at(st))
        return self._policy.image.frame(d, st, T, hold, self._holder_yaw.get(self.holder_id, 0.0))

    # ---------------------------------------------------------------- episode
    def _reset_chain(self, meas: dict) -> None:
        sp = self._policy
        sp.now_ns = time.time_ns()
        if self._mask_cfg is None:
            raise PS.PlaceInputStale(f"{SM.MASK_CONFIG_TOPIC} 를 못 받았다 — head_mask_node --policy-dir 가 도는가")
        SM.check_mask_config(self._mask_cfg, sp.cfg["mask"]["yolo"], sp.meta["depth_pre"])
        why = sp.masks.alive(sp.now_ns)
        if why:
            raise PS.PlaceInputStale(why)
        why = PS.start_refusal(sp.head.latest(sp.now_ns), sp.q0)
        if why:
            raise PourFjNodeError(why)
        if self._holder_yaw.get(self.holder_id) is None:
            raise PS.PlaceInputStale(f"홀더 {self.holder_id} 자세(yaw)를 아직 못 받았다")
        why = ("없다" if sp.frame is None else
               PS.age_refusal((sp.now_ns - sp.frame.stamp_ns) * 1e-9, sp.max_depth_age_s, "놓기 깊이"))
        if why:
            raise PS.PlaceInputStale(f"놓기 깊이 프레임 {why} {self._frame_err}")
        self._head_fail, self._head_err = 0, ""
        sp.chain = self.chain
        super()._reset_chain(meas)

    def _on_tick(self) -> None:
        running = self.book.phase == "running" and self.chain is not None
        self._policy.now_ns = time.time_ns()
        if running:
            try:
                q = self._policy.head.latest(self._policy.now_ns)
            except PS.PlaceInputStale as exc:
                self._end("abort", f"머리 명령 멈춤 · {exc}")
                return
        super()._on_tick()
        if running and self._head_on and self.book.phase == "running":
            self._head_tick(q)

    def _head_tick(self, q) -> None:
        """머리 규칙 한 틱. 학생 입력이 이 틱 신선하지 않으면(정책 틱 실패 · 이송 중 낡음) 새로 풀지 않고 마지막 명령을 다시 보낸다."""
        sp = self._policy
        stale = sp.ok_ns != sp.now_ns
        if stale and sp.look.cmd is None:
            return                                          # 첫 신선한 틱 전 — 보낼 명령이 아직 없다
        try:
            if stale:
                body = sp.look.command(q, None, None, hold=True)
            else:
                palm_p, palm_R = sp.last_palm
                body = sp.look.command(q, palm_p + palm_R @ sp.image.c_cup, self._holder, hold=self.chain.settling)
        except ValueError as exc:                          # look_at 풀이 실패 — 이 틱은 건너뛰고 연속이면 멈춘다
            self._head_fail, self._head_err = self._head_fail + 1, str(exc)
            if self._head_fail > HEAD_CMD_FAIL_MAX:
                self._end("abort", f"머리 명령 {self._head_fail} 틱 연속 실패 · {exc}")
            return
        except Exception as exc:                           # 버그 — status 에 남기고 멈춘 뒤 그대로 던진다
            self._head_err = f"{type(exc).__name__}: {exc}"
            self._status(False, [f"머리 명령 예외 · {self._head_err}"])
            self._end("abort", f"머리 명령 예외 · {self._head_err}")
            raise
        if not stale:                                      # 실패 셈은 새로 푼 명령이 성공했을 때만 비운다(유지 재전송은 아님)
            self._head_fail, self._head_err = 0, ""
        self._pub_head.publish(self._String(data=body))

    def _warnings(self) -> list[str]:
        """실행 중 성공 틱에도 status reasons 에 보일 입력 · 머리 문제."""
        sp, out = self._policy, []
        if self._head_fail:
            out.append(f"머리 명령 {self._head_fail} 틱 연속 실패(> {HEAD_CMD_FAIL_MAX} 면 abort) · {self._head_err}")
        if sp.head.bad:
            out.append(f"{HEAD_TOPIC} 불량 {sp.head.bad} 연속({sp.head.bad_why})")
        if not self._worker.is_alive():
            out.append("깊이 일꾼 스레드가 죽었다 — 깊이가 더 오지 않는다")
        if self._frame_fail >= FRAME_FAIL_WARN:
            out.append(f"깊이 프레임 {self._frame_fail} 연속 실패 · {self._frame_err}")
        run = getattr(self.chain, "carry_stale_run", 0)
        if run > CARRY_STALE_WARN:
            out.append(f"이송 중 학생 입력 낡음 {run} 틱 연속(LSTM 이 진행하지 않는다) · {self.chain.carry_input_err}")
        return out

    def _status(self, ok: bool, reasons, extra=None) -> None:
        super()._status(ok, [*reasons, *self._warnings()], extra)

    def _status_extra(self) -> dict:
        sp, ch = self._policy, self.chain
        dt = sp.head.dt_ms
        return {**super()._status_extra(), "student": {
            **sp.last_in, "mask": sp.masks.status(), "frame_err": self._frame_err, "frame_fail": self._frame_fail,
            "worker_alive": self._worker.is_alive(), "head_bad": sp.head.bad, "head_depth_dt_ms": None if dt is None else round(dt, 1),
            "carry_stale": {"n": getattr(ch, "carry_stale_n", 0), "run": getattr(ch, "carry_stale_run", 0),
                            "last": getattr(ch, "carry_input_err", "")},
            "head_cmd": None if sp.look.cmd is None else [round(float(v), 2) for v in sp.look.cmd]}}

    def destroy_node(self) -> None:
        self._stop.set()
        super().destroy_node()


if __name__ == "__main__":
    raise SystemExit(main(family="rh_place", node_cls=PlaceStudentNode))
