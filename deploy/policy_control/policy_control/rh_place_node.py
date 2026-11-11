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
  · 학습 episode_length_s 끝: 놓기 전이어도 stop(★11.10 — 판정은 실행기 cup_in_holder), 놓은 뒤(빼기 중)면 stop.
  · ★11.11 정책 손 목표 · 실측 손가락이 둘 다 open_min_rad 넘게 열리면(OPENED_END_TICKS 연속) 그 틱에 stop(outcome "opened_before_latch", 이 규칙은 래치 전에만).
★10.25 준비 자세 이송(계약 ready_carry — newton_07_r · 06_l): reset 때 끝 자세를 풀고(Mover, 움직이지 않음) start 뒤 첫 틱부터
  Mover 행동으로 슬롯 위까지 옮긴 다음 정책. 에피소드 시간은 이송이 끝난 틱부터 센다. 이송 자체 상한 =
  (max_steps + settle_steps + 멈춤 확인 STILL_MAX_TICKS) 틱 + READY_CARRY_SLACK_S — 넘으면 abort(Mover 가 그 전에 끝나야 한다).
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
OPENED_END_TICKS = 3      # ★11.11 정책 손 목표 · 실측 손가락이 둘 다 열린 틱이 이만큼 연속(60 Hz 50 ms)이면 끝
OPENED_MEAS_FRAC = 1.0    # ★11.11 "다 핀" 쪽 — 실측 평균 ≥ open_min_rad. 0.5(0.075 rad)는 sim 1111b place_blue 에서 컵을 쥔 채
                          #   0.12 rad(안착 높이 +40.5 mm)에 끝냈다. 실기 bag 184024 는 707.1 s 0.067 → 707.2 s 0.548 로 뛰어 종료 시각 거의 같음

from policy_control import codec  # noqa: E402
from policy_control import place_handover as PH  # noqa: E402
from policy_control.pour_fj_node import NS, Family, PourFjNode, PourFjNodeError, main  # noqa: E402
from policy_control.rh_place import (ATTACH_SOURCES, HOLDER_FLOOR_Z, HOLDER_TOL_M, JOINT_FREE_G,  # noqa: E402
                                     OPEN_MIN_RAD, RhPlaceError, holder_refusals, holder_tol_of)
from policy_control import tin_live as TL  # noqa: E402
from policy_control.place_ready_carry import STILL_MAX_TICKS as RCY_STILL_MAX_TICKS  # noqa: E402

TIN_FPP_TILT_DEG = 15.0     # ★11.07 tin_json — FP++ 를 FK 와 비교해 남기는 조건(컵이 거의 서 있고 탁자 위)


def holder_candidates(text: str, first: int) -> list[int]:
    """★10.28 holders:="1,2" → 구독할 실물 홀더 번호(시작 holder 를 늘 넣는다 · 정렬). 빈 문자열이면 시작 holder 하나."""
    try:
        got = {int(x) for x in str(text).split(",") if x.strip()}
    except ValueError as exc:
        raise PourFjNodeError(f"holders:={text!r} 는 쉼표로 가른 홀더 번호여야 한다") from exc
    return sorted(got | {int(first)})


def opened_both(chain) -> bool:
    """★11.11 이 틱에 정책 손 목표와 실측 손가락이 둘 다 열렸다(네 손가락 평균, 인계 · 정책 시작 손 기준)."""
    return chain.opened >= chain.open_min_rad and chain.opened_meas >= OPENED_MEAS_FRAC * chain.open_min_rad


def hand_released(chain, release_force_n: float) -> bool | None:
    """★11.11 손이 컵을 놓았다 — 손끝 촉각 < release_force_n 이고 관절 힘 < joint_free_g(rh_place.is_free · episode_ros 8d9b873 문턱).
    sim 1111 place_blue: 0.16~0.18 rad 로 열린 채 손 힘 22~44 N 으로 컵을 쥐고 있었다. 촉각 0·NaN(토픽 없음 → pour_fj_node 가 0 을 넣음)과
    관절 힘 None·NaN 은 신호 없음. 둘 다 없으면 None — 부르는 쪽이 각도만으로 판단한다."""
    tact, jf = chain.tact_max, chain.jf_max
    if jf is not None and not np.isfinite(jf):           # 채널 하나라도 NaN(np.max 전파) = 모름 = 쥔 것(is_free 와 같은 fail-closed)
        return False
    has_t = tact is not None and bool(np.isfinite(tact)) and tact > 0.0
    has_j = jf is not None and bool(np.isfinite(jf))
    if not (has_t or has_j):
        return None
    return not ((has_t and tact >= release_force_n) or (has_j and jf >= chain.joint_free_g))


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
                              ("holder_tol_m", HOLDER_TOL_M),
                              # ★11.04 사용자 "place 는 묶은 값만": 계약이 cup_obs live 여도 FP++ 를 넣지 않고 손바닥 FK ⊗ 묶음만.
                              #   실기 131817 — tilt 뒤 FP++ 가 쥔 컵을 놓치고 탁자 아래 자세를 내 놓기가 멈췄다. true 면 live(5 cm 문)
                              ("place_live_cup", False),
                              # ★11.07 실행기가 job 마다 넣는다: attach_source:=tin_json(빈 값 = 계약) · tin_json:=런 폴더 T_in json
                              ("attach_source", ""), ("tin_json", "")):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not bool(p("require_grasp")):                   # fake 손은 촉각이 0 — 리허설에서만 끈다
            self.fam = Family(**{**self.fam.__dict__, "refusals": lambda _c, _m, _t: []})
        self.holder_id = self._check_holder_id(int(p("holder")))
        self.holder_tol = holder_tol_of(p("holder_tol_m"))    # ★10.29 reset 과 같은 검사(0 < 값 ≤ 0.10 m)
        self.bundle_param = str(p("bundle") or "auto").strip()
        self.bundle_max_age = float(p("bundle_max_age_s"))
        self.bundle_hand_tol = float(p("bundle_hand_tol_rad"))
        self._bundle: PH.Bundle | None = None                # 이번 에피소드의 넘김 묶음(attach_source bundle) 또는 T_in(tin_json)
        self._tin = False                                    # ★11.07 이번 reset 의 붙인 자세가 tin_json 인가
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
        self._opened_n = 0
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
            if self._tin:                                  # ★11.07 tin_json — 제어는 FK ⊗ T_in 만, FP++ 는 차이 기록만
                return lambda palm_pos, palm_R, _tact, _st: self._tin_cup(role, palm_pos, palm_R)
            if self.contract.cup_obs == "live":            # ★11.02 'true' 런 — FP++ 신선분을 사슬에 넣고, 없으면 FK(끊김 · 가림)
                use_live = bool(self.get_parameter("place_live_cup").value)

                def live(palm_pos, palm_R, _tact, _st):
                    got = self._fresh_cup(role) if use_live else None
                    self.chain.live_cup = None if got is None else (np.asarray(got[0], float), np.asarray(got[1], float))
                    return self.chain.cup_obs(np.asarray(palm_pos, float), np.asarray(palm_R, float))
                return live
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

    def _tin_cup(self, role, palm_pos, palm_R):
        """★11.07 손바닥 FK ⊗ T_in(정책 관측 · 래치 같은 값). FP++ 가 그럴듯하면(탁자 위 · 기울기 15° 안) 차이만 로그 — 미끄러짐 관찰용."""
        self.chain.live_cup = None
        fk = self.chain.cup_obs(np.asarray(palm_pos, float), np.asarray(palm_R, float))
        got = self._fresh_cup(role)
        if got is not None:
            p, q = np.asarray(got[0], float), np.asarray(got[1], float)
            tilt = TL.tilt_deg(q)
            if p[2] > HOLDER_FLOOR_Z and tilt <= TIN_FPP_TILT_DEG:
                d = (p - fk[0]) * 1000
                self.get_logger().info(f"tin_json FP++ − FK 컵 {np.round(d, 1).tolist()} mm(|{np.linalg.norm(d):.1f}|) · "
                                       f"기울기 FP++ {tilt:.1f}° / FK {TL.tilt_deg(fk[1]):.1f}°", throttle_duration_sec=1.0)
        return fk

    def _attach_source(self) -> str:
        src = str(self.get_parameter("attach_source").value).strip() or self.contract.attach_source
        if src not in ATTACH_SOURCES:
            raise PourFjNodeError(f"attach_source {src!r} — {ATTACH_SOURCES} 중 하나")
        return src

    def _tin_bundle(self) -> PH.Bundle:
        """★11.07 실행기가 넘긴 tilt 직전 T_in json → 붙인 자세(넘김 묶음과 같은 자리로). 이 값에는 cup_obs_offset 를 더하지 않는다 —
        T_in 이 이미 FK 손바닥 → 카메라 컵이라, 묶음의 카메라−FK 보정을 또 더하면 두 번 옮긴다."""
        path = str(self.get_parameter("tin_json").value).strip()
        if not path:
            raise TL.TinError("attach_source tin_json 인데 tin_json 경로가 비었다(실행기가 tilt 직전 T_in 을 못 냈다)")
        side = self.contract.side().side
        rel_p, rel_q, d = TL.load_tin(path, {"r": "right", "l": "left"}.get(side, side))
        return PH.Bundle(path=Path(path), side=PH.SIDES.get(side, side), object_contract=str(d.get("object", "")), rel_p=rel_p, rel_q=rel_q,
                         hand_q=np.zeros(0), valid=True, n_steps=int((d.get("measured") or {}).get("n_fresh") or 0),
                         created=Path(path).stat().st_mtime,
                         meta={"node": d.get("node"), "episode": d.get("episode"), "source": d.get("source"),
                               "fallback_reason": d.get("fallback_reason", ""), "z_off_applied": d.get("z_off_applied")})

    def _attach_info(self) -> dict:
        """사슬 attach_info — 묶음 as_dict, tin_json 이면 출처(measured | fallback) · fallback 사유까지."""
        b = self._bundle
        if b is None:
            return {}
        if not self._tin:
            return b.as_dict()
        return {**b.as_dict(), "attach_source": "tin_json", "source": b.meta.get("source"),
                "fallback_reason": b.meta.get("fallback_reason", ""), "z_off_applied": b.meta.get("z_off_applied")}

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
        self._tin = False
        try:
            src = self._attach_source()
        except PourFjNodeError as exc:
            if self.book.phase == "armed":
                self.book.phase = "idle"
            return self._reply(res, False, [f"reset: {exc}"])
        if src in ("bundle", "tin_json"):
            self._drain()
            try:
                self._bundle = self._tin_bundle() if src == "tin_json" else self._pick_bundle()
                self._tin = src == "tin_json"
            except (PH.HandoverError, PourFjNodeError, KeyError, TL.TinError, OSError, ValueError) as exc:
                if self.book.phase == "armed":
                    self.book.phase = "idle"
                return self._reply(res, False, [f"reset: {exc}"])
        out = super()._srv_reset(req, res)
        ok = bool(getattr(out, "success", False))
        if ok and self._bundle is not None:
            self._bundles_used.append(str(self._bundle.path))
            self.get_logger().info(f"place 붙인 자세 = {'tilt 직전 T_in' if self._tin else '넘김 묶음'} {self._bundle.as_dict()}"
                                   + (f" · 출처 {self._bundle.meta.get('source')} · 원점 틀로 되돌린 z_off {self._bundle.meta.get('z_off_applied')}" if self._tin else ""))
            if self._tin and self._bundle.meta.get("source") != "measured":
                self.get_logger().warning(f"place 붙인 자세 = 고정 T_in(실측 아님): {self._bundle.meta.get('fallback_reason', '')}")
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
        self.chain.attach_info = self._attach_info()
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
                lim = (rc.cfg.max_steps + rc.mv.cfg.settle_steps + RCY_STILL_MAX_TICKS) / float(c.policy_hz) + READY_CARRY_SLACK_S
                if now - self._carry_t0 > lim:
                    self._end("abort", f"준비 자세 이송 시간 초과 {lim:.1f} s · {rc.as_dict()}")
                    return True
                return False
            self._carry_t0 = None
            self._t_start = now
            if rc is not None and not rc.start_ok:         # ★11.10 컵이 sim 시작 분포 밖(place_ready_carry START_*) — 정책 전에 멈춘다
                self._end("abort", f"준비 자세 이송 끝 문턱 미달 → 놓기 정책 시작 안 함 · {rc.as_dict()}")
                return True
            self.get_logger().info(f"준비 자세 이송 끝 → 정책 · {None if rc is None else rc.as_dict()}")
            if rc is not None and rc.still == "timeout":    # ★11.11 멈춤 확인 상한을 넘겼다 — 그대로 진행
                self.get_logger().warning(f"준비 자세 이송 끝 팔이 {RCY_STILL_MAX_TICKS} 틱 안에 멈추지 않음 → 그대로 정책 · {rc.as_dict()}")
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
        # ★11.11 사용자 "정책 기준으론 손가락을 다 피게 되어있으니까 손가락 다 피면 에피소드 끝" — 6701ae9 가 지운 규칙을 되살리되
        #   정책 손 목표(opened ≥ open_min_rad)와 실측 손가락(opened_meas ≥ OPENED_MEAS_FRAC · open_min_rad)이 같이 OPENED_END_TICKS 연속.
        #   래치가 켜졌으면 손은 래치가 편다 → 기존 놓기 · 빼기 경로. 학습 시간 끝 stop(episode_end)은 그대로.
        if c.latch_s2r and not self.chain.latch and not self.chain.settling and not self.chain.done:
            rel = hand_released(self.chain, self.contract.release_force_n) if opened_both(self.chain) else False
            if rel is None and not getattr(self, "_rel_unknown_logged", False):
                self._rel_unknown_logged = True
                self.get_logger().warning(f"래치 전 손 폄 — 촉각 · 관절 힘 신호 없음(tact {self.chain.tact_max} · jf {self.chain.jf_max}) "
                                          f"→ 각도만으로 판단")
            self._opened_n = self._opened_n + 1 if rel is not False else 0   # 손 힘이 남았으면(쥔 채 벌어짐) 세지 않는다
            if self._opened_n >= OPENED_END_TICKS:
                self.chain.outcome = "opened_before_latch"   # ★11.11 래치 전 — 놓였는지는 실행기가 cup_in_holder 로 판정
                self.chain.released_cup = self.chain.latch_cup
                self.get_logger().info(f"place 정책이 래치 전에 손 폄 → 끝 · {self.chain.as_dict()}")
                self._end("stop", f"opened before latch — 정책이 래치 전에 손을 폄(opened {self.chain.opened:.2f} · "
                                  f"실측 {self.chain.opened_meas:.2f} rad, 기준 {self.chain.open_min_rad:.2f}, {OPENED_END_TICKS} 틱 · "
                                  f"촉각 {self.chain.tact_max} N · 관절 힘 {self.chain.jf_max} g) · "
                                  f"놓임 판정은 cup_in_holder → 다음 노드(home)가 뺀다")
                return True
        if c.latch_s2r and self.max_s > 0 and time.monotonic() - self._t_start >= self.max_s and not self.chain.settling:
            # ★11.10 사용자 "place 노드의 '열리면 끝' 규칙 제거" — 정책을 학습 episode_length_s 끝까지 돌린다. 끝은 abort 가 아니라
            #   정상 종료(stop), 들어갔는지는 에피소드 실행기가 cup_in_holder(FP++ live · 묶은 컵 FK)로 판정한다. 빼기는 다음 rehome 노드가.
            self.chain.outcome = "episode_end"
            self.chain.released_cup = self.chain.latch_cup
            self.get_logger().info(f"place 학습 시간 끝 · {self.chain.as_dict()}")
            self._end("stop", f"placed — 학습 시간 {self.max_s:.1f} s(episode_length_s) 끝 · 래치 {self.chain.latch} · "
                              f"{self.chain.latch_t} 틱 · 손 폄 {self.chain.opened:.2f} rad → 다음 노드(home)가 뺀다")   # 빼기 중 시간 초과는 부모가 stop
            return True
        return False

    def _srv_start(self, req, res):
        self._latch_logged = False
        self._opened_n = 0
        self._rel_unknown_logged = False
        out = super()._srv_start(req, res)
        rc = getattr(self.chain, "rcarry", None)
        self._carry_t0 = time.monotonic() if rc is not None and rc.on else None
        return out

    def _end(self, kind: str, reason: str) -> None:
        """★11.11 끝낸 틱의 chain(outcome · phase)을 끝 이벤트 바로 뒤 status 로 한 번 더 낸다.

        전에는 같은 틱에서 outcome 을 정하고 _end 만 불러, 실행기가 stop 을 본 순간 가진 status 는 앞 running 틱 것(outcome None)
        이었다 → opened_before_latch 가 hand_empty False 로 실패 판정(e2e 709 · 725 oneshot). 실행기는 phase != running 인
        status 를 기다려 판정한다(episode_ros._status_after_end)."""
        super()._end(kind, reason)
        try:
            self._status(True, list(self._errors.values()))
        except Exception as exc:                  # 끝 status 는 보조 — 실패해도 다음 idle 틱 status 가 같은 내용을 낸다
            self.get_logger().warning(f"끝 status 발행 실패: {exc}")

    def _status_extra(self) -> dict:
        return {"holder": self.holder_id, "holder_pose": None if self._holder is None else [round(float(v), 4) for v in self._holder],
                "release_force_n": self.contract.release_force_n,     # 실행기 opened_before_latch 촉각 판정 문턱
                **({"holder_excess_m": self._holder_excess} if self.contract.holder_range else {}),
                **({"place": self.chain.as_dict()} if getattr(self.chain, "as_dict", None) else {})}


if __name__ == "__main__":
    raise SystemExit(main(family="rh_place", node_cls=PlaceNode))
