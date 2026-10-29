"""rh_place_node: RH56F1 한 팔 컵 홀더 놓기(hdgp open-rh_{r,l}_place) — pour_fj_node 의 rh_place 계열 + 이 노드 고유 부분(10.04).

  in   robot yaml(한 팔, rh56f1_<side>_*): 팔 /joint_states · 손 /hand_<side>/joint_states · 촉각 · 관절 힘
       컵 PoseStamped(base): cup_topic(기본 /objects/cyl60/pose) — reset 순간에만 쓴다(그 뒤는 손바닥에 붙은 컵)
       홀더 PoseStamped(base, latched): holder_topic(기본 /objects/cup_holder_<holder>/pose) — 목표 = 원점 + (0, 0, seat_dz)
       /policy_control/joint_target — pd 가 붙잡고 있는 마지막 목표(aglt 가 낸 것) = 인계 팔 · 손 목표(sim 뱅크 arm_q_target)
  out  /policy_control/joint_target (팔 7 · 손 6) · /policy_control/<ns>/{obs,action,episode} · status/<노드>
  srv  /policy_control/<ns>/episode/{reset,start,stop,abort} — 놓은 뒤 스크립트가 끝나면 스스로 stop

    python3 deploy/policy_control/policy_control/rh_place_node.py --ros-args -r __node:=rh_place_node_right -p ns:=right \\
        -p contract:=deploy/policies/rh56f1/place/right_i09/rh_place_contract.json -p robot:=… [-p holder:=2]

★10.22 newton_05 계약(latch_s2r · attach_source bundle · holder_range):
  · 붙인 자세 = 잡기 노드가 남긴 grasp_state 넘김 묶음(place_handover) — bundle:=auto(그 팔 rh_aglt_node 의 가장 새 묶음,
    bundle_max_age_s 안 · 아직 안 쓴 것 · 손 실측이 묶음과 bundle_hand_tol_rad 안) 또는 파일 경로. 없으면 reset 거부(FP++ 로 안 붙인다).
  · 래치 컵 = 손바닥 FK ⊗ 래치 추정기(정책 폴더 latch/, md5 핀) — place_latch_est.
  · holder = 실물 홀더 번호(로봇 왼쪽 +y 부터 0 · 1 · 2). 학습 목표 홀더는 sim 슬롯 번호라 번호로 거르지 않고, 그 홀더 원점이 학습
    범위(holder_range)에서 holder_tol_m 넘게 밖이면 reset 거부. ★10.29 holder_tol_m 은 reset 마다 다시 읽는다(실행기가 job 마다
    넣는다 · 카드가 그 job 에 넓힌 허용, 없으면 0.02).
  · 시간 초과(학습 episode_length_s): 놓기 전이면 abort, 놓은 뒤(빼기 중)면 stop.
★10.25 준비 자세 이송(계약 ready_carry — newton_07_r · 06_l): reset 때 끝 자세를 풀고(Mover, 움직이지 않음) start 뒤 첫 틱부터
  Mover 행동으로 슬롯 위까지 옮긴 다음 정책. 에피소드 시간은 이송이 끝난 틱부터 센다. 이송 자체 상한 =
  (max_steps + settle_steps) 틱 + READY_CARRY_SLACK_S — 넘으면 abort(Mover 가 그 전에 끝나야 한다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

import time  # noqa: E402

READY_CARRY_SLACK_S = 2.0

from policy_control import codec  # noqa: E402
from policy_control import place_handover as PH  # noqa: E402
from policy_control.pour_fj_node import NS, Family, PourFjNode, PourFjNodeError, main  # noqa: E402
from policy_control.rh_place import (HOLDER_TOL_M, JOINT_FREE_G, OPEN_MIN_RAD, RhPlaceError,  # noqa: E402
                                     holder_refusals, holder_tol_of)


def holder_candidates(text: str, first: int) -> list[int]:
    """★10.28 holders:="1,2" → 구독할 실물 홀더 번호(시작 holder 를 늘 넣는다 · 정렬). 빈 문자열이면 시작 holder 하나."""
    try:
        got = {int(x) for x in str(text).split(",") if x.strip()}
    except ValueError as exc:
        raise PourFjNodeError(f"holders:={text!r} 는 쉼표로 가른 홀더 번호여야 한다") from exc
    return sorted(got | {int(first)})


class PlaceNode(PourFjNode):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from policy_control.raw_poll import poll_subscription

        for name, default in (("holder", -1), ("holders", ""), ("holder_topic", ""), ("allow_measured_start", False),
                              ("joint_free_g", JOINT_FREE_G), ("release_open_min_rad", OPEN_MIN_RAD), ("require_grasp", True),
                              # ★10.22 newton_05 — 넘김 묶음 · 홀더 범위
                              ("bundle", "auto"), ("bundle_max_age_s", 900.0), ("bundle_hand_tol_rad", 0.05),
                              ("holder_tol_m", HOLDER_TOL_M)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not bool(p("require_grasp")):                   # fake 손은 촉각이 0 — 리허설에서만 끈다
            self.fam = Family(**{**self.fam.__dict__, "refusals": lambda _c, _m, _t: []})
        self.holder_id = self._check_holder_id(int(p("holder")))
        self.holder_tol = holder_tol_of(p("holder_tol_m"))    # ★10.29 reset 과 같은 검사(0 < 값 ≤ 0.10 m)
        self.bundle_param = str(p("bundle") or "auto").strip()
        self.bundle_max_age = float(p("bundle_max_age_s"))
        self.bundle_hand_tol = float(p("bundle_hand_tol_rad"))
        self._bundle: PH.Bundle | None = None                # 이번 에피소드의 넘김 묶음(attach_source bundle)
        self._bundles_used: list = []
        self._holder_excess: dict = {}
        self._latch_est = None
        if self.contract.latch_est:                          # ★10.22 래치 추정기 — 정책 폴더 latch/ (md5 핀)
            from policy_control.place_latch_est import LatchEst
            root = Path(str(p("contract"))).resolve().parent
            self._latch_est = LatchEst(root / self.contract.latch_est["file"], self.contract.latch_est["md5"], self.contract.obs_dim)
        self._rc_chain = None
        if self.contract.ready_carry:                        # ★10.25 Mover 사슬(고정 hdgp 사본 URDF) — 한 번만 읽는다
            from policy_control import place_ready_carry as RCY
            s = self.contract.side()
            self._rc_chain = RCY.load_chain(s.palm_body, list(s.arm_joints))
            self.get_logger().info(f"준비 자세 이송 켬 · {self.contract.ready_carry}")
        self._carry_t0: float | None = None
        self.holder_topic = str(p("holder_topic")) or f"/objects/cup_holder_{self.holder_id}/pose"
        # ★10.28 칵테일(같은 팔이 홀더 1 · 2 에 놓는다) — 그 팔 놓기 job 의 홀더를 모두 구독해 두고 reset 에서 고른다
        #   (holders:=1,2, 미션 생성기가 넣는다 · 없으면 holder 하나). holder_topic 은 시작 holder 에만 쓴다.
        self._holder_topics = {k: (self.holder_topic if k == self.holder_id else f"/objects/cup_holder_{k}/pose")
                               for k in holder_candidates(str(p("holders")), self.holder_id)}
        self._holder_poses: dict[int, np.ndarray] = {}
        self.attach = {}                                   # 붙은 컵은 PlaceChain 이 든다(인계 순간 한 번, sim attached)
        self._holder: np.ndarray | None = None
        self._held_q: dict[str, float] = {}
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for k, topic in self._holder_topics.items():
            self.create_subscription(PoseStamped, topic, lambda msg, k=k: self._on_holder(msg, k), latched)
        self._JointState = JointState
        self._poll_held = poll_subscription(self._poll, JointState, f"{NS}/joint_target",
                                            QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        c = self.contract
        if c.latch_s2r:                                    # ★10.22 newton_05 — 놓음은 래치 뒤 시간(촉각 · 관절 힘 안 봄)
            self.get_logger().info(
                f"place · 홀더 {self.holder_id} ({self.holder_topic}) · 래치 {c.latch_debounce} 틱(추정기 "
                f"{'켬' if self._latch_est is not None else '없음'}) → 손 펴기 {c.release_timed_steps} 틱 → 놓음 → 빼기 "
                f"{c.withdraw} · 붙인 자세 {c.attach_source}(bundle:={self.bundle_param}, {self.bundle_max_age:.0f} s · "
                f"손 {self.bundle_hand_tol} rad 안) · 홀더 범위 {c.holder_range} ± {self.holder_tol} m")
        else:
            self.get_logger().info(f"place · 홀더 {self.holder_id} ({self.holder_topic}) · 놓음 손끝 < "
                                   f"{c.release_force_n} N · 관절 힘 < {float(p('joint_free_g'))} g · 손 목표가 "
                                   f"{float(p('release_open_min_rad'))} rad 넘게 열린 뒤")

    def _check_holder_id(self, k: int) -> int:
        """홀더 번호 → 쓸 번호. holder_range 계약은 실물 번호(≥ 0)를 받아 자세 범위로 거르고(reset), 옛 계약은 학습 목표 홀더만."""
        c = self.contract
        if c.holder_range:                                 # ★10.22 학습 목표 = sim 슬롯 번호 — 실물 번호는 자세 범위로 거른다
            if k < 0:
                raise PourFjNodeError("holder:=<실물 홀더 번호> 가 필요하다(학습 목표 홀더 번호는 sim 슬롯이라 실물 번호가 아니다) — "
                                      f"학습 범위 x {c.holder_range['x']} · y {c.holder_range['y']}")
            return k
        h = c.target_holders[0] if k < 0 else k
        if h not in c.target_holders:
            raise PourFjNodeError(f"holder {h} 는 학습 목표 홀더 {c.target_holders} 가 아니다")
        return h

    def _select_holder(self) -> None:
        """★10.28 같은 팔이 두 홀더에 놓는 칵테일(파랑 → 1 · 분홍 → 2): reset 마다 holder 파라미터를 다시 읽는다 — 실행기가
        reset 전에 넣는다(episode_ros.job_params). 자세는 시작 때 holders:= 의 홀더를 모두 구독해 두고 고른다(reset 은 콜백을
        기다릴 수 없다). 바꿀 수 없으면 PourFjNodeError — reset 이 움직이기 전에 거부한다.
        ★10.29 holder_tol_m 도 job 마다 다시 읽는다(카드가 분홍 → 홀더 2 에만 넓힌 허용 · 다른 job 은 0.02)."""
        try:
            tol = holder_tol_of(self.get_parameter("holder_tol_m").value)
        except RhPlaceError as exc:
            raise PourFjNodeError(str(exc)) from exc
        k = self._check_holder_id(int(self.get_parameter("holder").value))
        if k not in self._holder_topics:
            raise PourFjNodeError(f"홀더 {k} 는 이 노드가 구독하지 않는다(holders {sorted(self._holder_topics)}) — "
                                  "미션 생성기가 그 팔 놓기 job 의 홀더를 holders:= 로 다 넣는다")
        if k != self.holder_id:
            pose = self._holder_poses.get(k)
            if pose is None:
                raise PourFjNodeError(f"홀더 {k} 자세({self._holder_topics[k]})를 아직 못 받았다 — cup_holders 단계가 내는지 본다")
            self.get_logger().info(f"홀더 바꿈 {self.holder_id} → {k} {np.round(pose, 4).tolist()}")
            self.holder_id, self._holder_excess = k, {}
        self._holder = self._holder_poses.get(k, self._holder)
        self.holder_tol = tol

    # ---------------------------------------------------------------- inputs
    def _on_holder(self, msg, k: int | None = None) -> None:
        k = self.holder_id if k is None else k
        try:
            pos = np.asarray(codec.decode_pose(msg).pos, float)
        except (codec.CodecError, ValueError) as exc:
            self._errors["holder"] = str(exc)
            return
        self._holder_poses[k] = pos
        if k == self.holder_id:
            self._holder = pos

    def _drain(self) -> None:
        super()._drain()
        from rclpy.serialization import deserialize_message
        from policy_control.raw_poll import take_all
        for raw, _t in take_all(self._poll_held):             # 여러 팔 노드가 같은 토픽에 낸다 — 관절 이름으로 합친다
            msg = deserialize_message(raw, self._JointState)
            self._held_q.update(zip(msg.name, (float(v) for v in msg.position)))

    def _held(self, meas: dict):
        s = self.contract.side()
        if not all(j in self._held_q for j in list(s.arm_joints) + list(s.hand_joints)):
            return None
        arm = np.array([self._held_q[j] for j in s.arm_joints])
        err = float(np.abs(arm - meas["arm"].arm_q).max())
        if err > self.reset_tol:
            raise PourFjNodeError(f"pd 가 붙잡은 마지막 목표가 팔 실측에서 {err:.3f} rad(> {self.reset_tol}) — "
                                  "aglt 바로 뒤가 아니다(그 사이 다른 경로가 돌았다)")
        return arm, np.array([self._held_q[j] for j in s.hand_joints])

    # ---------------------------------------------------------------- 고리
    def _cup_for(self, role: str, advance: bool):
        if getattr(self.chain, "rel", None) is not None:
            return lambda palm_pos, palm_R, _tact, _st: self.chain.cup_from_palm(palm_pos, palm_R)
        if self._bundle is not None:                       # ★10.22 reset 측정 — 컵 = 손바닥 FK ⊗ 넘김 묶음(FP++ 안 봄)
            from policy_control.place_latch_est import _quat_mul
            from policy_control.rh_place import _matrix_to_quat
            b = self._bundle
            return lambda palm_pos, palm_R, _tact, _st: (np.asarray(palm_pos, float) + np.asarray(palm_R, float) @ b.rel_p,
                                                         _quat_mul(_matrix_to_quat(np.asarray(palm_R, float)), b.rel_q))
        if self.contract.attach_source == "bundle":       # 묶음 전 — 상태 표시만(이 계약은 FP++ 컵을 쓰지 않는다)
            return lambda palm_pos, _R, _tact, _st: (np.asarray(palm_pos, float), np.array([1.0, 0.0, 0.0, 0.0]))
        return super()._cup_for(role, advance)

    def _hand_now(self) -> np.ndarray:
        s = self.contract.side()
        st = self.srcs[s.role].snapshot(time.monotonic())
        if st.ee_q is None or "ee" in set(st.stale) | set(st.missing):
            raise PourFjNodeError(f"{s.side}: 손 관절 측정이 없다(stale)")
        hand = dict(zip(st.ee_names, st.ee_q))
        return np.array([float(hand[j]) for j in s.hand_joints])

    def _pick_bundle(self) -> PH.Bundle:
        """★10.22 이번 place 의 넘김 묶음 — auto 면 그 팔 잡기 노드의 가장 새 묶음, 아니면 파일 경로."""
        side = self.contract.side().side
        obj = str(self.get_parameter("object").value).strip() or None
        hand = self._hand_now()
        if self.bundle_param == "auto":
            return PH.pick(side, object_name=obj, hand_q=hand, hand_tol_rad=self.bundle_hand_tol,
                           max_age_s=self.bundle_max_age, used=self._bundles_used)
        b = PH.load(self.bundle_param)
        bad = PH.check(b, side=side, object_name=obj, hand_q=hand, hand_tol_rad=self.bundle_hand_tol, max_age_s=self.bundle_max_age)
        if bad:
            raise PH.HandoverError(f"넘김 묶음 {b.path.name}: " + " · ".join(bad))
        return b

    def _srv_reset(self, req, res):
        try:
            self._select_holder()                          # ★10.28 이번 job 의 홀더(실행기가 reset 전에 holder 를 넣는다)
        except PourFjNodeError as exc:
            if self.book.phase == "armed":
                self.book.phase = "idle"
            return self._reply(res, False, [f"reset: {exc}"])
        if self.chain is not None:
            self.chain.rel = None                          # 새 인계 — 컵은 다시 붙인다(FP++ 한 번 또는 넘김 묶음)
        self._bundle = None
        if self.contract.attach_source == "bundle":
            self._drain()
            try:
                self._bundle = self._pick_bundle()
            except (PH.HandoverError, PourFjNodeError, KeyError) as exc:
                if self.book.phase == "armed":
                    self.book.phase = "idle"
                return self._reply(res, False, [f"reset: {exc}"])
        out = super()._srv_reset(req, res)
        ok = bool(getattr(out, "success", False))
        if ok and self._bundle is not None:
            self._bundles_used.append(str(self._bundle.path))
            self.get_logger().info(f"place 붙인 자세 = 넘김 묶음 {self._bundle.as_dict()}")
        return out

    def _reset_chain(self, meas: dict) -> None:
        self.chain.object_name = str(self.get_parameter("object").value).strip() or None   # ★10.12 다물체 place(잡기와 같은 물체)
        self.chain.joint_free_g = float(self.get_parameter("joint_free_g").value)
        self.chain.open_min_rad = float(self.get_parameter("release_open_min_rad").value)
        held = self._held(meas)
        self.chain.fk = self.fk[self.contract.side().role]     # ★10.13 이송: 손바닥 자코비안(실측 FK)
        if self._holder is not None and self.contract.holder_range:   # ★10.22 학습 범위 밖 홀더는 움직이기 전에 거부
            bad, self._holder_excess = holder_refusals(self.contract, self._holder, self.holder_tol)
            if bad:
                raise RhPlaceError(f"홀더 {self.holder_id}: " + " · ".join(bad))
            if any(v > 0 for v in self._holder_excess.values()):
                self.get_logger().warning(f"홀더 {self.holder_id} {np.round(self._holder, 4).tolist()} 가 학습 범위 밖 "
                                          f"{self._holder_excess} m(허용 {self.holder_tol} m 안이라 진행)")
        self.chain.est = self._latch_est
        self.chain.rc_chain = self._rc_chain
        b = self._bundle
        self.chain.attach_info = {} if b is None else b.as_dict()
        self.chain.reset(meas, holder=self._holder, held=held,
                         allow_measured=bool(self.get_parameter("allow_measured_start").value),
                         attach=None if b is None else (b.rel_p, b.rel_q))
        if self.chain.rcarry is not None:
            self.get_logger().info(f"준비 자세 이송 계획 · {self.chain.rcarry.plan}")
        rel = self.chain.rel[0]
        self.get_logger().info(f"place reset · 자리 {self.chain.as_dict()['seat']} · 시작 목표 "
                               f"{'pd 마지막 joint_target' if held is not None else '실측(allow_measured_start)'} · "
                               f"손바닥 기준 컵 {np.round(rel, 3).tolist()}")

    def _after_step(self) -> bool:
        c = self.contract
        if self._carry_t0 is not None:                     # ★10.25 이송 중 — 에피소드 시계는 이송이 끝난 틱부터
            now = time.monotonic()
            rc = self.chain.rcarry
            if rc is not None and rc.on:
                self._t_start = now
                lim = (rc.cfg.max_steps + rc.mv.cfg.settle_steps) / float(c.policy_hz) + READY_CARRY_SLACK_S
                if now - self._carry_t0 > lim:
                    self._end("abort", f"준비 자세 이송 시간 초과 {lim:.1f} s · {rc.as_dict()}")
                    return True
                return False
            self._carry_t0 = None
            self._t_start = now
            self.get_logger().info(f"준비 자세 이송 끝 → 정책 · {None if rc is None else rc.as_dict()}")
        if self.chain.settle_t == 0:                       # 놓음이 난 스텝 — 첫 실기 문턱 정하기용 값
            self.get_logger().info(f"place 놓음 판정 · {self.chain.as_dict()}")
        if c.latch_s2r and self.chain.latch and self.chain.latch_t == 0 and not self.chain.settling and not getattr(self, "_latch_logged", False):
            self._latch_logged = True
            self.get_logger().info(f"place 래치 · {self.chain.as_dict()}")
        if self.chain.done:
            if c.withdraw:
                self._end("stop", f"placed — 래치 {c.release_timed_steps} 틱 뒤 놓음 → 빼기 {self.chain.settle_t} 틱 "
                                  f"({self.chain.outcome}, 손–컵 {self.chain.clearance})")
            else:
                self._end("stop", f"placed — 놓음 {c.release_steps} 스텝 + 스크립트 {c.settle_steps} 스텝 끝")
            return True
        if c.latch_s2r and self.max_s > 0 and time.monotonic() - self._t_start >= self.max_s and not self.chain.settling:
            self._end("abort", f"place 시간 초과 {self.max_s:.1f} s(학습 episode_length_s) — 놓기 전 "
                               f"(래치 {self.chain.latch} · {self.chain.latch_t} 틱)")   # 빼기 중 시간 초과는 부모가 stop
            return True
        return False

    def _srv_start(self, req, res):
        self._latch_logged = False
        out = super()._srv_start(req, res)
        rc = getattr(self.chain, "rcarry", None)
        self._carry_t0 = time.monotonic() if rc is not None and rc.on else None
        return out

    def _status_extra(self) -> dict:
        return {"holder": self.holder_id, "holder_pose": None if self._holder is None else [round(float(v), 4) for v in self._holder],
                **({"holder_excess_m": self._holder_excess} if self.contract.holder_range else {}),
                **({"place": self.chain.as_dict()} if getattr(self.chain, "as_dict", None) else {})}


if __name__ == "__main__":
    raise SystemExit(main(family="rh_place", node_cls=PlaceNode))
