"""정책 실기 기록의 토픽 목록과 메시지 → 배열 변환. 실시간 기록기(joint_recorder)와 bag 변환기(bag_to_trace)가 같이 쓴다.

09.28: Python 기록기 하나가 팔 joint_states(~700/s) 와 촉각 5 × 100 Hz 를 한 줄로 받다 촉각을 버렸다(손가락당 9 Hz).
그래서 기록은 `ros2 bag record`(C++)에 맡기고 묶음을 나눈다(사용자: "따로 따로 관리") — 이 모듈은 어느 토픽을
어느 묶음에 넣는지와, bag 에서 읽은 메시지를 분석용 npz(joint_trace.summarize 가 읽는 모양)로 쌓는 법을 정한다.

  kind          토픽                                         묶음
  obs · act     /policy_control/{obs,action}                policy
  tgt           /policy_control/joint_target                policy
  app           /policy_control/pd_<side>/applied           policy
  arm · hand    robot yaml arm · ee 소스                      policy  (정책 노드와 같은 SourceSet 변환 → canonical)
  obj           robot yaml object 소스                        policy
  jn · pd       /policy_control/status/{joint_node,pd_<side>} policy
  episode       /policy_control/episode                     policy
  tip<i>        /dg5f_<side>/fingertip_<i>_broadcaster/wrench sensors (F/T 센서 손)
  tac<i>        /dg5f_<side>/tactile/finger_<i>              sensors (촉각 손, 09.28 왼손 TACTILE_M 3×5 mono8)
"""
from __future__ import annotations

import json

import numpy as np

from . import codec
from .joint_contract import JointContract
from .sources import SourceSet

NS = "/policy_control"
BAG_GROUPS = ("policy", "sensors")


def topics(c: JointContract, cfg) -> dict[str, tuple[str, str, str]]:
    """{토픽: (kind, 메시지 타입 'pkg/msg/Type', 묶음)}."""
    side = c.side
    out = {
        f"{NS}/obs": ("obs", "std_msgs/msg/Float64MultiArray", "policy"),
        f"{NS}/action": ("act", "std_msgs/msg/Float64MultiArray", "policy"),
        f"{NS}/joint_target": ("tgt", "sensor_msgs/msg/JointState", "policy"),
        f"{NS}/pd_{side}/applied": ("app", "sensor_msgs/msg/JointState", "policy"),
        cfg.sources["arm"].topic: ("arm", "sensor_msgs/msg/JointState", "policy"),
        cfg.sources["ee"].topic: ("hand", "sensor_msgs/msg/JointState", "policy"),
        f"{NS}/status/joint_node": ("jn", "std_msgs/msg/String", "policy"),
        f"{NS}/status/pd_{side}": ("pd", "std_msgs/msg/String", "policy"),
        f"{NS}/episode": ("episode", "std_msgs/msg/String", "policy"),
    }
    obj = cfg.sources.get("object")
    if obj is not None:
        out[obj.topic] = ("obj", "geometry_msgs/msg/PoseStamped", "policy")
    for i in range(1, 6):
        out[f"/dg5f_{side}/fingertip_{i}_broadcaster/wrench"] = (f"tip{i}", "geometry_msgs/msg/WrenchStamped", "sensors")
        out[f"/dg5f_{side}/tactile/finger_{i}"] = (f"tac{i}", "sensor_msgs/msg/Image", "sensors")
    return out


def _stack(rows, width: int) -> np.ndarray:
    return np.stack(rows) if rows else np.zeros((0, width))


class TraceAccumulator:
    """메시지를 받은 시각 t [s] 와 함께 쌓는다. 순수(ROS 메시지 객체만 읽는다)."""

    def __init__(self, c: JointContract, cfg) -> None:
        self.c, self.cfg = c, cfg
        self.src = SourceSet(cfg)
        self.R: dict[str, list] = {k: [] for k in ("obs", "act", "tgt", "app", "arm", "hand", "cur", "obj", "jn", "pd",
                                                    "episode", "tip", "tac")}
        self.names: dict[str, tuple | None] = {"tgt": None, "app": None, "cur": None}
        self.errors: dict[str, int] = {}

    def add(self, kind: str, t: float, m) -> None:
        try:
            self._add(kind, float(t), m)
        except (codec.CodecError, ValueError, KeyError) as exc:
            key = f"{kind}: {type(exc).__name__}"
            self.errors[key] = self.errors.get(key, 0) + 1

    def _named(self, key: str, t: float, m) -> None:
        n = tuple(m.name)
        if self.names[key] is None:
            self.names[key] = n
        if n != self.names[key]:
            return
        q = np.asarray(m.position, float)
        qd = np.asarray(m.velocity, float) if len(m.velocity) == len(n) else np.full(len(n), np.nan)
        tau = np.asarray(m.effort, float) if len(m.effort) == len(n) else np.full(len(n), np.nan)
        seq = -1
        if key == "tgt":
            _, _, s = str(m.header.frame_id).rpartition(":")
            seq = int(s) if s.isdigit() else -1
        self.R[key].append((t, seq, q, qd, tau))

    def _add(self, kind: str, t: float, m) -> None:
        R = self.R
        if kind in ("obs", "act"):
            s = codec.decode_float_array(m)
            R[kind].append((t, s.seq, s.data.copy()))
        elif kind in ("tgt", "app"):
            self._named(kind, t, m)
        elif kind in ("arm", "hand"):
            source = self.cfg.sources["arm" if kind == "arm" else "ee"]
            self.src.update_from_joint_state(source.name, codec.decode_joint_state(m), t)
            st = self.src.snapshot(t)
            if kind == "arm" and st.arm_q is not None:
                R["arm"].append((t, np.asarray(st.arm_q, float), np.asarray(st.arm_qd, float)))
            if kind == "hand" and st.ee_q is not None:
                R["hand"].append((t, np.asarray(st.ee_q, float)))
                self._named("cur", t, m)                  # 드라이버 원래 이름으로 전류(effort, mA)
        elif kind == "obj":
            p = codec.decode_pose(m)
            R["obj"].append((t, np.asarray(p.pos, float), np.asarray(p.quat, float)))
        elif kind in ("jn", "pd", "episode"):
            R[kind].append((t, str(m.data)))
        elif kind.startswith("tip"):
            f, w = m.wrench.force, m.wrench.torque
            R["tip"].append((t, int(kind[3:]), np.array([f.x, f.y, f.z, w.x, w.y, w.z], float)))
        elif kind.startswith("tac"):
            raw = np.frombuffer(bytes(m.data), dtype=np.uint16 if m.encoding == "mono16" else np.uint8).astype(float)
            cells = np.full(18, np.nan)
            cells[: min(18, raw.size)] = raw[:18]
            R["tac"].append((t, int(kind[3:]), cells))

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.R.items()}

    def to_arrays(self, meta: dict) -> dict:
        c, R = self.c, self.R
        ee = self.cfg.sources["ee"]
        ee_names = list(ee.joints) + list(ee.mirror)
        tn, an, cn = self.names["tgt"] or (), self.names["app"] or (), self.names["cur"] or ()
        col = lambda k, i: np.array([r[i] for r in R[k]])  # noqa: E731
        return {
            "meta_task": np.array(c.task), "meta_side": np.array(c.side),
            "meta_policy_hz": np.float64(c.policy_hz), "meta_json": np.array(json.dumps(meta, ensure_ascii=False)),
            "meta_errors": np.array([f"{k} ×{v}" for k, v in self.errors.items()]),
            "obs_t": col("obs", 0), "obs_seq": np.array([r[1] for r in R["obs"]], int),
            "obs": _stack([r[2] for r in R["obs"]], c.obs_dim),
            "act_t": col("act", 0), "act_seq": np.array([r[1] for r in R["act"]], int),
            "act": _stack([r[2] for r in R["act"]], c.action_dim),
            "tgt_t": col("tgt", 0), "tgt_seq": np.array([r[1] for r in R["tgt"]], int), "tgt_names": np.array(tn),
            "tgt_q": _stack([r[2] for r in R["tgt"]], len(tn)), "tgt_qd": _stack([r[3] for r in R["tgt"]], len(tn)),
            "app_t": col("app", 0), "app_names": np.array(an),
            "app_q": _stack([r[2] for r in R["app"]], len(an)), "app_qd": _stack([r[3] for r in R["app"]], len(an)),
            "app_tau": _stack([r[4] for r in R["app"]], len(an)),
            "arm_t": col("arm", 0), "arm_names": np.array(c.arm_joints),
            "arm_q": _stack([r[1] for r in R["arm"]], 7), "arm_qd": _stack([r[2] for r in R["arm"]], 7),
            "hand_t": col("hand", 0), "hand_names": np.array(ee_names),
            "hand_q": _stack([r[1] for r in R["hand"]], len(ee_names)),
            "cur_t": col("cur", 0), "cur_names": np.array(cn), "cur_mA": _stack([r[4] for r in R["cur"]], len(cn)),
            "obj_t": col("obj", 0), "obj_pos": _stack([r[1] for r in R["obj"]], 3),
            "obj_quat": _stack([r[2] for r in R["obj"]], 4),
            "jn_t": col("jn", 0), "jn_json": np.array([r[1] for r in R["jn"]]),
            "pd_t": col("pd", 0), "pd_json": np.array([r[1] for r in R["pd"]]),
            "ep_t": col("episode", 0), "ep_json": np.array([r[1] for r in R["episode"]]),
            "tip_t": col("tip", 0), "tip_idx": np.array([r[1] for r in R["tip"]], int),
            "tip_wrench": _stack([r[2] for r in R["tip"]], 6),
            "tac_t": col("tac", 0), "tac_idx": np.array([r[1] for r in R["tac"]], int),
            "tac": _stack([r[2] for r in R["tac"]], 18),
        }
