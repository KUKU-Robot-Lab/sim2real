"""rh_place 계열 — RH56F1 한 팔 컵 홀더 놓기(hdgp open-rh_{r,l}_place, 51013e96). 10.04 사용자: 성공한 정책부터 실기에.

관측 96 · 행동 13 · 디코더는 rh_aglt 와 같다(rh_aglt.build_obs · RaDecoder 를 그대로 쓴다). 다른 것:
  · 목표 = 홀더 자리: 홀더 원점(컵 축 위 받침 윗면, /objects/cup_holder_<k>/pose) + (0, 0, HOLDER_FLOOR_Z − cup_bottom_z), 세운 컵.
  · 시작 = aglt 가 컵을 쥐고 멈춘 상태의 인계. 팔 · 손 목표는 그 순간의 목표(sim 뱅크 arm_q_target · hand_target = aglt 마지막 q*,
    배포 = pd 가 붙잡고 있는 마지막 joint_target), hold 0, LSTM 0.
  · 컵 관측 = 인계 순간 손바닥 기준 상대 자세로 붙인 것(sim cup_obs_mode "attached") — 에피소드 내내 손바닥 FK 로만.
  · 놓음 = 손가락 · 손바닥 접촉 < 1 N 이 release_steps 연속 → settle_steps 동안 스크립트(손 = 편 손 행동, 팔 = 시작 관절 쪽 최대 증분)
    → 끝. 실기 판정은 손끝 촉각 전부 < 1 N 이고 관절 힘(있으면) 전부 < 문턱(쥠 판정과 같은 300 g, 임시)이며, 네 손가락 손
    목표가 인계 때보다 open 쪽으로 OPEN_MIN_RAD 넘게 열린 뒤(실기는 손바닥 · 첫마디 접촉을 못 본다 — 10.04 PLACE 검토).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from policy_control import hand_block as HB
from policy_control import place_carry as PC
from policy_control import place_ready_carry as RCY
from policy_control import rh_aglt as A
from policy_control.cup_attach import AttachCfg, grasp_signal

SCHEMA = "policy_control/rh_place_contract/v1"
TASK_PREFIX = "open-rh_{}_place"
#: hdgp tasks/rh_place_r/place_geom.py:12 — 홀더 원점 기준 컵이 앉는 바닥 높이
HOLDER_FLOOR_Z = -0.025
JOINT_FREE_G = AttachCfg.joint_force_g
#: 놓음 안전 게이트(10.04 PLACE 검토) — 네 손가락 손 목표가 인계 때보다 open 쪽으로 평균 이만큼 열린 뒤에만 '빈 손'을 센다.
#  sim 놓음은 정책이 손을 연 뒤에만 났다. 실기는 손바닥 · 첫마디 접촉을 못 봐서 손끝이 비어도 마디로 쥐고 있을 수 있다.
OPEN_MIN_RAD = 0.15


class RhPlaceError(A.RhAgltError):
    pass


@dataclass(frozen=True)
class PlaceContract(A.RaContract):
    seat_dz: float = 0.0                 # 홀더 원점 → 자리(컵 원점) 높이
    target_holders: list = field(default_factory=list)
    release_steps: int = 0
    settle_steps: int = 0
    release_force_n: float = 0.0         # 놓음 = 접촉이 모두 이 힘 아래(env contact_force_threshold)
    # ★10.12 관측 래치(hdgp release_latch_obs — 관측 끝 1 칸 "latch"): 지각 컵(손바닥 FK 부착)이 자리 위 dz 안 · 바닥 중심 xy
    #   dxy 안 · 기울기 tilt_deg 안이면 켜지고 리셋까지 유지(놓기 전만). 빈 값 = 이 필드 전 계약(관측 96, 래치 없음).
    latch_obs: dict = field(default_factory=dict)        # {dz, dxy, tilt_deg, bottom_xy}
    # ★10.12 놓음 판정 — "contact"(손끝 촉각 · 관절 힘, 이 필드 전) | "block_follow_open"(막힘 신호 · 따라감 · 열림량, sim 과 같은 식)
    release_rule: str = "contact"
    release_open_rad: list = field(default_factory=list)  # block_follow_open: (엄지굽힘, 검지, 중지, 약지, 새끼) 쥔 자세에서 연 양 문턱
    # ★10.13 놓기 기울기 게이트(hdgp release_max_tilt_deg): 놓음 조건이 release_steps 틱 찬 틱에 컵 기울기(세계 z) ≤ 이 값일
    #   때만 놓음 — 배포는 지각 컵(손바닥 FK 부착) 자세로 같은 식. 0 = 게이트 없음(이 필드 전 계약).
    release_max_tilt_deg: float = 0.0
    # ★10.14 A안(hdgp release_open_zero_skip): True 면 release_open_rad 가 0 인 축은 연 양 조건에서 뺀다(기본 False = 이전 동작)
    release_open_zero_skip: bool = False
    # ★10.13 IK 이송(hdgp carry_enable): 리셋 뒤 손바닥 DLS 로 슬롯 위까지 옮긴 다음 정책(place_carry). 빈 값 = 이송 없음.
    carry: dict = field(default_factory=dict)
    # ★10.22 newton_05(hdgp rh_place_r 16f25c219 학습) — 기본값 = 이 필드들 전 계약(i09 · i01 동작 그대로)
    latch_debounce: int = 1                  # 래치 기하 조건 연속 틱(latch_debounce_steps)
    latch_dz_objs: dict = field(default_factory=dict)   # 물체 → 래치 높이(latch_dz_objs/vals, 나머지는 latch_obs.dz)
    latch_s2r: bool = False                  # 관측 래치 = 손 펴기 래치(쥠 조건 없음, 놓기 전만 켜짐 · 리셋까지 유지)
    latch_est: dict = field(default_factory=dict)       # 래치 컵 자세 추정기 {file(정책 폴더 기준), md5, source}
    latch_arm_free: bool = False             # 래치 뒤 놓기 전 = 손만 편 손 행동(강제), 팔은 정책
    release_timed_steps: int = 0             # > 0 = 놓음 = 래치 뒤 이 틱(손가락 · 접촉 판정 없음)
    withdraw: dict = field(default_factory=dict)        # 놓은 뒤 정책이 팔을 뺀다 {clear_m, max_steps, hand_open}
    neighbor_obs: str = ""                   # "" | "zero" — 관측 끝 이웃 홀더 6 칸(학습 내내 0 이던 런)
    attach_source: str = "fpp"               # 붙인 자세: fpp(리셋 순간 FP++ 한 장) | bundle(잡기의 grasp_state 넘김 묶음)
    holder_range: dict = field(default_factory=dict)    # 학습 목표 홀더 원점 범위 {x: [lo, hi], y: [lo, hi]} (base m)
    # ★10.25 준비 자세 이송(place_ready_carry): 이송 끝 뱅크로 학습한 런(newton_07_r · 06_l) — 리셋 뒤 Mover 로 슬롯 위까지 옮긴 다음
    #   정책(RNN · 직전 행동 · 래치 추정기 0 에서). 빈 값 = 이송 없음(들기 끝에서 바로 정책).
    ready_carry: dict = field(default_factory=dict)


#: ★10.22 hdgp _nb_obs 순서 — +y 이웃(슬롯 0) (dx, dy, 있음), −y 이웃(슬롯 2) (dx, dy, 있음)
NB_OBS = ("nb_py_dx", "nb_py_dy", "nb_py_on", "nb_ny_dx", "nb_ny_dy", "nb_ny_on")
LATCH_DIR = "latch"
ATTACH_SOURCES = ("fpp", "bundle")


def is_rh_place_run(env: Mapping) -> bool:
    return ("target_holders" in env and str(env.get("cup_obs_mode", "")) == "attached"
            and int(env.get("hold_steps", -1)) == 0 and str(env.get("profile_name", "")).startswith("rh56f1_"))


RELEASE_RULES = ("contact", "block_follow_open")
RELEASE_AXES = ("thumb_flex", "index", "middle", "ring", "pinky")


def latch_obs_of(env: Mapping) -> dict:
    """hdgp release_latch_obs → {dz, dxy, tilt_deg, bottom_xy}. dxy 는 커리큘럼(curr_latch_dxy (시작, 끝))이 끝 칸에서
    시작한 런이면 끝값(10.12 place: 덤프 latch_dxy 는 커리큘럼이 덮는 기본값). 커리큘럼 중간이면 거부."""
    if not bool(env.get("release_latch_obs", False)):
        return {}
    if bool(env.get("latch_obs_requires_grasp", False)):
        raise RhPlaceError("latch_obs_requires_grasp — 관측 래치에 쥠 조건이 붙은 런은 배포가 같은 값을 못 만든다")
    dxy = float(env["latch_dxy"])
    if bool(env.get("curr_enabled", False)) and env.get("curr_latch_dxy") is not None:
        n, k = int(env["curr_num_increments"]), int(env.get("curr_start_increment", 0))
        if k != n:
            raise RhPlaceError(f"curr_latch_dxy 커리큘럼이 끝 칸에서 시작하지 않았다({k}/{n}) — 학습 끝의 latch_dxy 를 모른다")
        dxy = float(list(env["curr_latch_dxy"])[1])
    return {"dz": float(env["latch_dz"]), "dxy": dxy, "tilt_deg": float(env["latch_tilt_deg"]),
            "bottom_xy": bool(env.get("latch_bottom_xy", False))}


def holder_range_of(env: Mapping) -> dict:
    """학습 목표 홀더 원점 범위(hdgp place_geom.sample_holder_poses): x = holder_x − U(0, dr_x_back), y = holder_ys[목표] ± dr_y
    ± 공통 y(holder_y_shift_fixed ≥ 0 이면 그 값, 아니면 커리큘럼 끝 holder_y_shift_max). 실물 배치 가지(holder_real_frac)는
    공통 y 가 없는 jitter 라 더 좁다. 키가 없으면 빈 값(검사 없음)."""
    if "holder_ys" not in env or "holder_x" not in env:
        return {}
    ys = [float(v) for v in env["holder_ys"]]
    t = [int(v) for v in env["target_holders"]]
    fixed = float(env.get("holder_y_shift_fixed", -1.0))
    common = fixed if fixed >= 0.0 else float(env.get("holder_y_shift_max", 0.0))
    dy = float(env.get("holder_dr_y", 0.0)) + common
    if fixed < 0.0 and float(env.get("holder_real_frac", 0.0)) > 0.0:
        dy = max(dy, float(env.get("holder_real_jitter", 0.0)))
    x = float(env["holder_x"])
    return {"x": [round(x - float(env.get("holder_dr_x_back", 0.0)), 4), round(x, 4)],
            "y": [round(min(ys[i] for i in t) - dy, 4), round(max(ys[i] for i in t) + dy, 4)]}


def s2r_fields(env: Mapping, run_dir: Path, latch: dict, obs_dim: int) -> tuple[dict, list]:
    """★10.22 newton_05 계열 키 → 계약 필드 · 노트. 배포가 같은 값을 못 만드는 설정은 거부한다(조용히 다른 정책을 돌리지 않는다)."""
    out: dict = {}
    notes: list = []
    if bool(env.get("latch_true_pose", False)):
        raise RhPlaceError("latch_true_pose — 래치가 sim 참 컵 자세를 본 진단 런은 배포가 같은 입력을 못 만든다")
    objs, vals = list(env.get("latch_dz_objs") or ()), list(env.get("latch_dz_vals") or ())
    if len(objs) != len(vals):
        raise RhPlaceError(f"latch_dz_objs {objs} · vals {vals} 길이가 다르다")
    if objs:
        out["latch_dz_objs"] = {str(k): float(v) for k, v in zip(objs, vals)}
    db = int(env.get("latch_debounce_steps", 1) or 1)
    if db > 1:
        out["latch_debounce"] = db
    s2r = bool(env.get("latch_s2r", False))
    if s2r and not latch:
        raise RhPlaceError("latch_s2r 인데 release_latch_obs 가 꺼져 있다")
    out["latch_s2r"] = s2r
    est = str(env.get("latch_est_path") or "")
    if est:
        if not s2r:
            raise RhPlaceError("latch_est_path 는 latch_s2r 런에서만 배포한다(래치 입력 = 추정기 보정 자세)")
        f = Path(run_dir) / LATCH_DIR / Path(est).name
        if not f.is_file():
            raise RhPlaceError(f"래치 추정기 {f} 가 없다 — 학습 {est} 를 정책 폴더 {LATCH_DIR}/ 에 받는다")
        from policy_control.place_latch_est import LatchEst, md5
        mu, sd = LatchEst.stats(f)
        if mu.size != obs_dim:
            raise RhPlaceError(f"래치 추정기 입력 {mu.size} ≠ 정책 관측 {obs_dim}")
        out["latch_est"] = {"file": f"{LATCH_DIR}/{f.name}", "md5": md5(f), "source": est}
        notes.append(f"★래치 컵 = 손바닥 FK ⊗ 추정기(GRU) 보정 자세({out['latch_est']['file']} md5 {out['latch_est']['md5'][:8]}) — "
                     "입력 = 정책 관측(래치 칸은 갱신 전), 리셋 뒤 첫 틱은 보정 0. 정책 관측의 컵은 붙인 자세 그대로")
    elif s2r:
        notes.append("래치 컵 = 붙인 자세(손바닥 FK)")
    if bool(env.get("neighbor_obs", False)):
        if int(env.get("extra_actor_obs", 0)) != 1 + len(NB_OBS):
            raise RhPlaceError(f"neighbor_obs 인데 extra_actor_obs {env.get('extra_actor_obs')} ≠ 7(래치 1 + 이웃 6)")
        # 이웃 칸은 시작 뱅크에 holder_pos 가 있을 때만 켜진다(_place_neighbors). 래치 추정기 학습 데이터(같은 env 롤아웃)가
        #   그 칸을 늘 0 으로 봤으면(mu 0 · std = 바닥값) 정책도 0 만 봤다 — 그 증거가 없으면 배포 값을 정할 수 없다.
        if not est:
            raise RhPlaceError("neighbor_obs: 이웃 칸이 학습에서 늘 0 이었다는 증거(래치 추정기 통계)가 없다")
        tail_mu, tail_sd = mu[-len(NB_OBS):], sd[-len(NB_OBS):]
        if float(np.abs(tail_mu).max()) > 1e-6 or float(tail_sd.max()) > 1.1e-3:
            raise RhPlaceError(f"neighbor_obs: 추정기 통계에서 이웃 칸이 0 이 아니다(mu {tail_mu} · std {tail_sd}) — "
                               "실물 이웃 홀더 관측을 배포가 아직 만들지 않는다")
        out["neighbor_obs"] = "zero"
        notes.append("★이웃 홀더 관측 6 칸 = 0 — 학습 시작 뱅크에 홀더 자세가 없어 이웃 칸이 늘 꺼져 있었다(래치 추정기 데이터 "
                      "mu 0 · std 0.001 로 확인). 이웃 홀더 자체는 sim 장면에 ±0.15 m 간격으로 있었다")
    if bool(env.get("latch_arm_free", False)):
        out["latch_arm_free"] = True
    k = int(env.get("release_timed_steps", 0) or 0)
    if k > 0:
        if not latch:
            raise RhPlaceError("release_timed_steps 는 관측 래치가 있어야 한다")
        out["release_timed_steps"] = k
    if bool(env.get("policy_withdraw", False)):
        out["withdraw"] = {"clear_m": float(env["withdraw_clear_m"]), "max_steps": int(env["withdraw_max_steps"]),
                           "hand_open": bool(env.get("withdraw_hand_open", False))}
        if float(env.get("assist_withdraw_p0", 0.0)) > 0.0:
            notes.append(f"학습 빼기 보조 assist_withdraw_p0 {env['assist_withdraw_p0']}(일부 에피소드 스크립트) — 배포는 늘 정책")
    if s2r:
        if not (out.get("release_timed_steps") and out.get("withdraw")):
            raise RhPlaceError("latch_s2r 배포는 시간 기준 놓음(release_timed_steps) · 정책 빼기(policy_withdraw) 런만 — "
                               "다른 조합은 아직 배포 단계 순서를 옮기지 않았다")
        if bool(env.get("carry_enable", False)):
            raise RhPlaceError("latch_s2r 배포에 IK 이송(carry) 은 아직 없다")
        w = out["withdraw"]
        notes.append(f"★s2r 래치(10.14): 래치 컵 원점 z − 자리 z ≤ {latch['dz']}"
                     + (f"(물체별 {out['latch_dz_objs']})" if objs else "")
                     + f" · 바닥 중심 xy ≤ {latch['dxy']} m · 기울기 ≤ {latch['tilt_deg']}° 가 {db} 틱 연속이면 켜지고 리셋까지 유지 — "
                     f"관측 래치 칸 = 손 펴기 래치(쥠 조건 없음)")
        notes.append(f"★놓기(10.14): 래치 뒤 손 = 편 손 행동"
                     + (" · 팔 = 정책" if out.get("latch_arm_free") else "") + f" → 래치 {k} 틱 뒤 놓음(손가락 · 접촉 판정 없음) → "
                     f"빼기: 손 {'편 손 고정' if w['hand_open'] else '정책'} · 팔 정책, 손바닥 · 손끝이 놓은 컵에서 {w['clear_m']} m "
                     f"떨어지거나 {w['max_steps']} 틱이면 끝(stop). 놓기 전 시간 초과는 abort")
        notes.append("학습 전용(배포는 따라 하지 않는다): lost_terminate(참값 손–컵 거리) · goal_obs_err · s2r_adr · "
                     "fpp_bias(붙일 때 FP++ 편향 — 배포는 attach_source 가 정한다)")
    return out, notes


def build(run_dir: Path, checkpoint: Path, right_profile, urdf: Path, *, asset: str, attach: str = "fpp",
          ready_carry_max: int = 0) -> PlaceContract:
    env_p = Path(run_dir) / "params" / "env.yaml"
    env = A.read_env(env_p)
    if not is_rh_place_run(env):
        raise RhPlaceError(f"{env_p}: rh_place 런이 아니다(target_holders · cup_obs_mode attached · hold 0)")
    if attach not in ATTACH_SOURCES:
        raise RhPlaceError(f"attach {attach!r} — {ATTACH_SOURCES}")
    latch = latch_obs_of(env)
    base = A.build_from_env(Path(run_dir), checkpoint, right_profile, urdf, asset=asset, env=env, extra_obs=1 if latch else 0)
    nb = list(NB_OBS) if bool(env.get("neighbor_obs", False)) else []
    obs_dim = base.obs_dim + (1 if latch else 0) + len(nb)
    s2r, s2r_notes = s2r_fields(env, Path(run_dir), latch, obs_dim)
    from policy_control.checkpoint_dims import checkpoint_contract
    ck_obs, ck_act = checkpoint_contract(Path(checkpoint))
    if (ck_obs, ck_act) != (obs_dim, base.action_dim):
        raise RhPlaceError(f"체크포인트 (obs, act) ({ck_obs}, {ck_act}) ≠ 계약 ({obs_dim}, {base.action_dim}) — 관측 칸 해석이 틀렸다")
    rule = str(env.get("release_rule", "contact"))
    open_rad = [float(v) for v in env.get("release_open_rad", ())] if rule == "block_follow_open" else []
    max_tilt = float(env.get("release_max_tilt_deg", 0.0))
    zero_skip = bool(env.get("release_open_zero_skip", False))
    try:
        carry = PC.CarryCfg.from_env(env)
    except PC.CarryError as exc:
        raise RhPlaceError(f"{env_p}: {exc}") from exc
    rcy = None
    if ready_carry_max > 0:
        try:
            rcy = RCY.ReadyCarryCfg.from_env(env, ready_carry_max)
        except RCY.ReadyCarryError as exc:
            raise RhPlaceError(f"{env_p}: {exc}") from exc
    elif "carry_end" in str(env.get("start_bank_path", "")):
        raise RhPlaceError(f"{env_p}: 이송 끝 뱅크({env['start_bank_path']})로 학습한 런 — --ready-carry 가 필요하다"
                           "(이송 없이 들기 끝에서 바로 시작하면 시작 분포가 다르다)")
    _, origin_z = A.cup_geometry(env)                       # 컵 원점 → 바닥 = −origin_z
    # 아래에서 다시 정하는 키는 뺀다(10.13 — 10.12 RaContract.extra_actor_obs 뒤로 이 빌더가 중복 키로 깨져 있었다)
    kw = {k: v for k, v in asdict(base).items() if not k.startswith("goal_") and k not in ("extra_actor_obs", "obs_dim")}
    kw["sides"] = base.sides
    c = PlaceContract(**{**kw, "schema": SCHEMA, "task": TASK_PREFIX.format(base.side().side[0]),
                         "goal_offset": [0.0, 0.0, 0.0],
                         "notes": [n for n in base.notes if not n.startswith(("첫 목표", "목표 입력", "목표 달성"))] + [
                             f"목표 = 홀더 원점 + (0, 0, {HOLDER_FLOOR_Z + origin_z:.3f}) · 세운 컵 — 학습 목표 홀더 "
                             f"{list(env['target_holders'])} (/objects/cup_holder_<k>/pose)",
                             "시작 = aglt 가 쥐고 멈춘 상태: 팔 · 손 목표는 pd 가 붙잡은 마지막 joint_target, hold 0, LSTM 0",
                             f"놓음 = 접촉 < {float(env['contact_force_threshold'])} N {int(env['release_steps'])} 스텝 → "
                             f"{int(env['settle_steps'])} 스텝 스크립트(손 펴기 · 팔 시작 관절로) → 끝 — 배포는 손끝 촉각 · 관절 힘으로 "
                             f"보고 네 손가락 손 목표가 인계보다 {OPEN_MIN_RAD} rad 넘게 열린 뒤에만 센다(손바닥 · 첫마디 접촉 없음)"]},
                       seat_dz=HOLDER_FLOOR_Z + origin_z, target_holders=[int(h) for h in env["target_holders"]],
                       release_steps=int(env["release_steps"]), settle_steps=int(env["settle_steps"]),
                       release_force_n=float(env["contact_force_threshold"]),
                       latch_obs=latch, release_rule=rule, release_open_rad=open_rad, release_max_tilt_deg=max_tilt, release_open_zero_skip=zero_skip,
                       carry={} if carry is None else carry.as_dict(),
                       extra_actor_obs=(["latch"] if latch else []) + nb,
                       obs_dim=obs_dim, attach_source=attach, holder_range=holder_range_of(env),
                       ready_carry={} if rcy is None else rcy.as_dict(), **s2r)
    if (latch or rule != "contact") and not c.latch_s2r:
        c = replace(c, notes=list(c.notes) + [
            f"★관측 래치(10.12): 관측 끝 1 칸 — 지각 컵 원점 z − 자리 z ≤ {latch.get('dz')} · 바닥 중심 xy ≤ {latch.get('dxy')} m · "
            f"기울기 ≤ {latch.get('tilt_deg')}° 면 켜지고 리셋까지 유지(놓기 전만)" if latch else "관측 래치 없음",
            f"★놓음(10.12) {rule}: 막힘 latch 없음 AND 모든 축이 목표를 따라감(앞섬 < G) AND 쥔 자세에서 연 양 ≥ "
            f"{open_rad} rad 이 {int(env['release_steps'])} 틱 — 관측 래치가 켜진 뒤에만" if rule != "contact" else
            "놓음 = 접촉(손끝 촉각 · 관절 힘)"])
    if max_tilt > 0.0 and not c.release_timed_steps:     # 시간 기준 놓음에는 안 쓴다(sim k_rel > 0)
        c = replace(c, notes=list(c.notes) + [
            f"★놓기 기울기 게이트(10.13): 놓음 조건이 {int(env['release_steps'])} 틱 찬 틱에 지각 컵 기울기 ≤ {max_tilt}° 일 때만 "
            "놓음(sim 은 실제 컵 기울기) — 기울어진 채로는 놓지 않고 정책이 세울 때까지 기다린다"])
    if carry is not None:
        c = replace(c, notes=list(c.notes) + [
            f"★IK 이송(10.13): 리셋 뒤 손바닥 DLS 로 컵을 (홀더 xy, 상판 + 원점 높이 + {carry.dz}) 까지 — 축당 {carry.speed} m/틱, "
            f"{carry.min_steps} 틱 뒤 {carry.tol} m 안이거나 {carry.max_steps} 틱이면 정책. 이송 중 정책 forward 는 돌리고 출력은 버림"
            f"(LSTM · prev_action 이어짐). 에피소드 {c.episode_s} s 는 이송 포함(sim 과 같음)"])
    if rcy is not None:
        c = replace(c, notes=list(c.notes) + [
            f"★준비 자세 이송(10.25, place_ready_carry): 리셋 뒤 ready_carry Mover 로 컵을 (홀더 xy, 상판 + 원점 높이 + {rcy.dz}) 앵커까지"
            f"(작업 공간 하강 {rcy.plan_down} · xy ±{rcy.plan_xy} · 기울기 ±{rcy.plan_tilt_deg}° · {rcy.v_max} m/틱 · {rcy.qd_max} rad/틱, "
            f"상한 {rcy.max_steps} 틱) — 학습 시작 뱅크 = 이송 끝({env.get('start_bank_path')}). 이송 중 정책 · 래치 추정기는 돌리지 않고, "
            f"끝난 다음 틱부터 정책(RNN · 직전 행동 · 추정기 0). 에피소드 {c.episode_s} s 는 이송 뒤부터 센다. 배포 잡음 0"])
    if s2r_notes or attach == "bundle" or c.holder_range:
        c = replace(c, notes=[n for n in c.notes if not (c.latch_s2r and n.startswith(("놓음 =", "★관측 래치(10.12)", "★놓음(10.12)")))]
                    + s2r_notes
                    + ([f"★붙인 자세 = 잡기의 grasp_state 넘김 묶음 rel_p · rel_q(learn:place 10.21 · 10.22 'FP++ 는 처음부터 없음') — "
                        "리셋 순간 FP++ 스냅숏을 쓰지 않는다. 묶음이 없거나 이번 잡기 것이 아니면 reset 거부"] if attach == "bundle" else [])
                    + ([f"학습 목표 홀더 원점 범위 x {c.holder_range['x']} · y {c.holder_range['y']} m — 실물 홀더가 밖이면 노드가 "
                        "holder_tol_m 까지만 받는다"] if c.holder_range else []))
    validate(c)
    return c


def validate(c: PlaceContract) -> None:
    A.validate(c)
    if c.hold_steps != 0 or not c.target_holders or c.release_steps < 1 or c.settle_steps < 1 or c.release_force_n <= 0:
        raise RhPlaceError("hold 0 · 목표 홀더 · release/settle 스텝 · 놓음 힘")
    want_extra = ["latch"] + (list(NB_OBS) if c.neighbor_obs else [])
    if c.latch_obs:
        if set(c.latch_obs) != {"dz", "dxy", "tilt_deg", "bottom_xy"} or list(c.extra_actor_obs) != want_extra:
            raise RhPlaceError(f"latch_obs {c.latch_obs} · extra_actor_obs {c.extra_actor_obs} — dz · dxy · tilt_deg · bottom_xy, "
                               f"관측 끝 {want_extra}")
        if c.latch_obs["bottom_xy"] and not c.object_geometry:
            raise RhPlaceError("latch bottom_xy 는 물체별 bottom_z(object_geometry)가 있어야 한다")
    elif c.extra_actor_obs:
        raise RhPlaceError(f"extra_actor_obs {c.extra_actor_obs} 인데 latch_obs 가 없다")
    if not 0.0 <= float(c.release_max_tilt_deg) < 90.0:
        raise RhPlaceError(f"release_max_tilt_deg {c.release_max_tilt_deg} — 0(없음) 또는 (0, 90)°")
    if c.carry:
        try:
            PC.CarryCfg.from_dict(c.carry)
        except PC.CarryError as exc:
            raise RhPlaceError(str(exc)) from exc
    if c.ready_carry:
        if c.carry or not c.latch_s2r:
            raise RhPlaceError("ready_carry 는 latch_s2r 계약에서만, IK 이송(carry)과 같이 쓰지 않는다")
        try:
            RCY.ReadyCarryCfg.from_dict(c.ready_carry)
        except RCY.ReadyCarryError as exc:
            raise RhPlaceError(str(exc)) from exc
    if c.release_rule not in RELEASE_RULES:
        raise RhPlaceError(f"release_rule {c.release_rule!r} — {RELEASE_RULES}")
    if c.release_rule == "block_follow_open":
        if not c.hand_block or len(c.release_open_rad) != len(RELEASE_AXES) or min(c.release_open_rad) < 0 \
                or max(c.release_open_rad) <= 0:
            raise RhPlaceError(f"block_follow_open: hand_block 과 release_open_rad {len(RELEASE_AXES)} 개(≥ 0, 하나는 > 0) "
                               f"{RELEASE_AXES} — 10.12 place 결정 (0.0, 0.10, 0.10, 0.10, 0.10)")
    # ★10.22 newton_05 계열
    if c.attach_source not in ATTACH_SOURCES:
        raise RhPlaceError(f"attach_source {c.attach_source!r} — {ATTACH_SOURCES}")
    if c.neighbor_obs not in ("", "zero"):
        raise RhPlaceError(f"neighbor_obs {c.neighbor_obs!r} — '' | 'zero'")
    if c.neighbor_obs and not c.latch_obs:
        raise RhPlaceError("neighbor_obs 는 관측 래치 뒤 칸 — latch_obs 가 있어야 한다")
    if int(c.latch_debounce) < 1:
        raise RhPlaceError(f"latch_debounce ≥ 1: {c.latch_debounce}")
    bad = [k for k in c.latch_dz_objs if k not in (c.object_names or [])]
    if bad:
        raise RhPlaceError(f"latch_dz_objs {bad} 가 object_names {c.object_names} 에 없다")
    if c.latch_est and set(c.latch_est) != {"file", "md5", "source"}:
        raise RhPlaceError(f"latch_est 키 {sorted(c.latch_est)} — file · md5 · source")
    if c.latch_s2r:
        if not c.latch_obs or c.release_timed_steps < 1 or not c.withdraw or c.carry:
            raise RhPlaceError("latch_s2r 계약 = 관측 래치 · release_timed_steps ≥ 1 · withdraw · 이송 없음")
        if set(c.withdraw) != {"clear_m", "max_steps", "hand_open"} or float(c.withdraw["clear_m"]) <= 0 \
                or int(c.withdraw["max_steps"]) < 1:
            raise RhPlaceError(f"withdraw {c.withdraw} — clear_m > 0 · max_steps ≥ 1 · hand_open")
    elif c.latch_est or c.release_timed_steps or c.withdraw or c.latch_arm_free:
        raise RhPlaceError("latch_est · release_timed_steps · withdraw · latch_arm_free 는 latch_s2r 계약에서만")
    if c.holder_range and not all(len(c.holder_range.get(k, ())) == 2 and c.holder_range[k][0] <= c.holder_range[k][1]
                                  for k in ("x", "y")):
        raise RhPlaceError(f"holder_range {c.holder_range} — x · y 각각 [lo, hi]")


def latch_dz_of(c: PlaceContract) -> float:
    """고른 물체의 래치 높이(hdgp _latch_dz_table — 덮기 없으면 latch_obs.dz)."""
    return float(c.latch_dz_objs.get(c.active_object or "", c.latch_obs["dz"]))


#: 실물 홀더가 학습 범위(holder_range) 밖으로 나가도 되는 거리 — 놓기 노드 holder_tol_m 기본 · 에피소드 --plan 검사가 같이 쓴다
HOLDER_TOL_M = 0.02
#: ★10.29 에피소드 카드가 놓기 job 하나에 넓혀 줄 수 있는 허용의 상한(m). 오타로 범위 검사가 꺼지지 않게 막는다
HOLDER_TOL_MAX_M = 0.10


def holder_tol_of(value) -> float:
    """holder_tol_m(m) 값 검사: 0 < 값 ≤ HOLDER_TOL_MAX_M 이면 float, 아니면 RhPlaceError(카드 · 노드 파라미터가 같이 쓴다)."""
    try:
        tol = float(value)
    except (TypeError, ValueError) as exc:
        raise RhPlaceError(f"holder_tol_m {value!r} 는 m 단위 수여야 한다") from exc
    if not 0.0 < tol <= HOLDER_TOL_MAX_M:
        raise RhPlaceError(f"holder_tol_m {tol} 는 0 보다 크고 {HOLDER_TOL_MAX_M} m 이하여야 한다")
    return tol


def holder_refusals(c: PlaceContract, holder_pos: Sequence[float], tol_m: float) -> tuple[list, dict]:
    """실물 홀더 원점이 학습 목표 홀더 범위에서 tol_m 넘게 밖이면 거부 이유. (이유, 축별 밖으로 나간 양 m)."""
    if not c.holder_range:
        return [], {}
    h = np.asarray(holder_pos, float)
    out, excess = [], {}
    for i, k in enumerate(("x", "y")):
        lo, hi = (float(v) for v in c.holder_range[k])
        e = max(lo - h[i], h[i] - hi, 0.0)
        excess[k] = round(e, 4)
        if e > tol_m:
            out.append(f"홀더 {k} {h[i]:+.3f} 가 학습 범위 [{lo:+.3f}, {hi:+.3f}] 에서 {e * 100:.1f} cm 밖(허용 {tol_m * 100:.1f} cm)")
    return out, excess


def load_contract(path: str | Path) -> PlaceContract:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise RhPlaceError(f"{path}: schema {raw.get('schema')!r} ≠ {SCHEMA}")
    raw["sides"] = {r: A.RaSide(**s) for r, s in raw["sides"].items()}
    c = PlaceContract(**raw)
    validate(c)
    return c


def seat_dz_of(c: PlaceContract) -> float:
    """홀더 원점 → 자리(컵 원점) 높이. 다물체면 고른 물체의 바닥으로(hdgp seat_pos: HOLDER_FLOOR_Z − bottom_z, env 별 물체)."""
    if c.object_names and c.active_object:
        return HOLDER_FLOOR_Z - float(c.object_geometry[c.active_object]["bottom_z"])
    return c.seat_dz


def origin_z_of(c: PlaceContract) -> float:
    """물체 원점 높이(바닥 → 원점) — hdgp _geo_origin. 단일 물체 계약은 seat_dz 에서 되돌린다."""
    if c.object_names and c.active_object:
        return float(c.object_geometry[c.active_object]["origin_offset_z"])
    return c.seat_dz - HOLDER_FLOOR_Z


def seat_goal(c: PlaceContract, holder_pos: Sequence[float]) -> A.Goal:
    return A.Goal(np.asarray(holder_pos, float) + np.array([0.0, 0.0, seat_dz_of(c)]), np.array([1.0, 0.0, 0.0, 0.0]))


def hand_action_of(c: PlaceContract, hand_q: Sequence[float]) -> np.ndarray:
    """손 목표 → 행동값(sim _carry_hand · _a_open 과 같은 linear_grip_raw 역사상)."""
    s = c.side()
    o, g = np.asarray(s.hand_open, float), np.asarray(s.hand_grip, float)
    lo, hi = np.minimum(o, g), np.maximum(o, g)
    return np.clip(2.0 * (np.asarray(hand_q, float) - lo) / np.maximum(hi - lo, 1e-6) - 1.0, -1.0, 1.0)


def open_hand_action(c: PlaceContract) -> np.ndarray:
    """편 손의 행동값 — sim _a_open(linear_grip_raw 역사상): (2·(open − lo)/(hi − lo) − 1), lo/hi = min/max(open, grip)."""
    return hand_action_of(c, c.side().hand_open)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """회전 행렬 → (w, x, y, z), 가장 큰 대각 성분으로 나눈다(Shepperd)."""
    m = np.asarray(R, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def is_free(c: PlaceContract, tactile_n: Sequence[float], joint_force: Sequence[float] | None,
            joint_free_g: float = JOINT_FREE_G) -> bool:
    """손이 컵에서 떨어졌다 — 손끝 촉각 전부 < 놓음 힘이고 관절 힘(있으면) 전부 < 문턱."""
    if float(np.max(np.asarray(tactile_n, float))) >= c.release_force_n:
        return False
    return joint_force is None or float(np.max(np.asarray(joint_force, float))) < joint_free_g


def start_refusals(c: PlaceContract, meas: Mapping) -> list:
    m = meas["arm"]
    if not grasp_signal(m.tactile_n, m.joint_force, AttachCfg(force_n=c.release_force_n)):
        return ["cup is not grasped (thumb AND another finger) — place starts from the aglt hand-off"]
    return []


class PlaceChain:
    """rh_place 한 스텝 — 붙은 컵 · 홀더 자리 목표로 관측 → 정책(놓은 뒤에는 스크립트) → rh_aglt 디코더."""

    def __init__(self, c: PlaceContract, policy, *, joint_free_g: float = JOINT_FREE_G, open_min_rad: float = OPEN_MIN_RAD) -> None:
        self.c, self.policy, self.joint_free_g, self.open_min_rad = c, policy, joint_free_g, open_min_rad
        self.base = c                                   # 파일 계약 — 다물체면 reset 마다 select_object 로 self.c 를 고른다
        self.object_name: str | None = None             # ★10.12 노드 파라미터 object(잡기 단계와 같은 물체)
        self.dec = A.RaDecoder(c)
        self.prev = np.zeros(c.action_dim)
        self.step_i, self.free_streak, self.settle_t = 0, 0, -1
        self.goal: A.Goal | None = None
        self.rel: tuple[np.ndarray, np.ndarray] | None = None   # 손바닥 좌표계의 컵 (위치, 쿼터니언)
        self.start_q: np.ndarray | None = None
        self._a_open = open_hand_action(c)
        s = c.side()
        self._close_dir = np.sign(np.asarray(s.hand_grip, float) - np.asarray(s.hand_open, float))
        self._fingers = [i for i, j in enumerate(s.hand_joints) if "thumb" not in j]
        self.hand0: np.ndarray | None = None
        self.opened, self.tact_max, self.jf_max = 0.0, 0.0, None
        self.latch = False                              # 관측 래치(latch_obs) — 리셋까지 유지
        self.fk = None                                  # ★10.13 이송: 노드가 넣는 UrdfChainFK(손바닥 자코비안용)
        self.carry: PC.Carry | None = None
        self.rc_chain = None                            # ★10.25 준비 자세 이송: 노드가 넣는 (ready_carry 패키지, Mover 사슬)
        self.rcarry: RCY.ReadyCarry | None = None
        self.q_g: np.ndarray | None = None              # block_follow_open: 마지막으로 쥐었던 틱의 실측(칸 단위)
        names = list(s.hand_joints)
        self._rel_idx = [next(i for i, n in enumerate(names) if n.endswith(HB.AXIS_SUFFIX[a])) for a in RELEASE_AXES]
        # ★10.22 newton_05(latch_s2r) — 래치 추정기(노드가 넣는 place_latch_est.LatchEst) · 래치 뒤 틱 · 빼기 끝
        self.est = None
        self.latch_t, self.latch_geo_n = 0, 0
        self.latch_cup: tuple[np.ndarray, np.ndarray] | None = None    # 이번 틱 래치 입력 컵(월드)
        self.released_cup: tuple[np.ndarray, np.ndarray] | None = None  # 놓은 틱의 래치 컵(빼기 거리 기준, 고정)
        self.clearance: float | None = None
        self.outcome: str | None = None                # "clear" | "withdraw_timeout" — 빼기 끝(stop)
        self.attach_info: dict = {}

    @property
    def settling(self) -> bool:
        return self.settle_t >= 0

    @property
    def done(self) -> bool:
        if self.c.withdraw:                             # ★10.22 정책 빼기 — 손이 컵에서 떨어지거나 빼기 시간이 다 되면
            return self.outcome is not None
        return self.settle_t >= self.c.settle_steps

    def reset(self, meas: Mapping, *, holder: Sequence[float] | None, held: tuple | None,
              allow_measured: bool = False, attach: tuple | None = None) -> None:
        """attach = (rel_p, rel_q wxyz) 손바닥 좌표 컵 — 계약 attach_source bundle 이면 반드시(넘김 묶음), fpp 면 무시하고
        지금 측정 컵(FP++ 한 장)에서 잡는다."""
        m = meas["arm"]
        if holder is None:
            raise RhPlaceError("holder pose missing — /objects/cup_holder_<k>/pose 가 와야 목표(자리)를 안다")
        self.c = A.select_object(self.base, self.object_name)   # 물체를 모르면 거부(조용한 0 one-hot 금지)
        s = self.c.side()
        hand_meas = np.array([m.hand_q[j] for j in s.hand_joints], float)
        if held is None and not allow_measured:
            raise RhPlaceError("held target missing — pd 가 붙잡은 마지막 joint_target 이 없다(aglt 뒤에 띄울 것, "
                               "실측에서 시작하려면 allow_measured)")
        arm0, hand0 = (m.arm_q, hand_meas) if held is None else held
        if len(arm0) != len(s.arm_joints) or len(hand0) != len(s.hand_joints):
            raise RhPlaceError("held target 은 팔 7 · 손 6")
        # ★10.29 held 의 손은 aglt 배포 전용 squeeze 가 얹힌 명령이다. 이송(Mover) 동안은 그대로 쥐고(실기 bag 20261019_133007:
        # hand_err ≈ 0 이면 RH56F1 힘이 2~50 g 로 빠져 병을 떨굼), 정책이 시작하는 틱에 정책 손 목표를 그 틱 실측 q 로 맞춘다
        # (_policy_hand_from_meas). 학습 뱅크 hand_target ≈ q(err ≈ 0) — oneshot 16_s0_obsdump 에서 err 0.2~0.35 로 분포 밖이었다.
        self.dec.reset(arm_start=arm0, hand_start=hand0)
        self.hand0 = np.asarray(hand0, float).copy()
        self.opened, self.tact_max, self.jf_max = 0.0, 0.0, None
        self.goal = seat_goal(self.c, holder)
        if self.c.attach_source == "bundle":
            if attach is None:
                raise RhPlaceError("attach_source bundle — 잡기의 넘김 묶음(rel_p · rel_q)이 없다(FP++ 로 붙이지 않는다)")
            q = np.asarray(attach[1], float).reshape(4)
            self.rel = (np.asarray(attach[0], float).reshape(3).copy(), q / np.linalg.norm(q))
        else:
            R = np.asarray(m.palm_R, float)
            self.rel = (R.T @ (np.asarray(m.cup_pos, float) - np.asarray(m.palm_pos, float)),
                        _quat_mul(_quat_conj(_matrix_to_quat(R)), np.asarray(m.cup_quat, float)))
        if self.c.latch_est and self.est is None:
            raise RhPlaceError("래치 추정기 계약인데 추정기가 없다 — 노드가 chain.est 를 넣어야 한다")
        if self.est is not None:
            self.est.reset()
        self.latch_t, self.latch_geo_n = 0, 0
        self.latch_cup = self.released_cup = None
        self.clearance, self.outcome = None, None
        self.start_q = np.asarray(m.arm_q, float).copy()          # sim _start_q = 리셋 때 팔 실측(놓은 뒤 복귀 목표)
        self.prev = np.zeros(self.c.action_dim)
        self.step_i, self.free_streak, self.settle_t = 0, 0, -1
        self.latch, self.q_g = False, None
        self.carry = None
        if self.c.carry:
            if self.fk is None:
                raise RhPlaceError("이송(carry) 계약인데 FK 가 없다 — 노드가 chain.fk 를 넣어야 한다")
            cfg = PC.CarryCfg.from_dict(self.c.carry)
            self.carry = PC.Carry(cfg, PC.carry_target(holder, origin_z_of(self.c), cfg.dz), hand_action_of(self.c, hand0),
                                  self.c.k_arm, self.c.arm_ema)
        self.rcarry = None
        if self.c.ready_carry:                           # ★10.25 끝 자세를 지금 푼다(움직이지 않음) — 이송 틱은 _step_s2r
            if self.rc_chain is None:
                raise RhPlaceError("준비 자세 이송 계약인데 Mover 사슬이 없다 — 노드가 chain.rc_chain 을 넣어야 한다")
            cfg = RCY.ReadyCarryCfg.from_dict(self.c.ready_carry)
            self.rcarry = RCY.ReadyCarry(cfg, *self.rc_chain, k_arm=self.c.k_arm, arm_ema=self.c.arm_ema,
                                         hand_action=hand_action_of(self.c, hand0))
            pos, quat = self.cup_from_palm(m.palm_pos, m.palm_R)
            self.rcarry.start(np.asarray(m.arm_q, float), pos, quat, PC.carry_target(holder, origin_z_of(self.c), cfg.dz))
        else:                                            # 이송 없음 — 정책이 지금부터라 손 목표를 지금 실측으로
            self._policy_hand_from_meas(hand_meas)
        self._policy_reset()

    def _policy_hand_from_meas(self, hand_q: np.ndarray) -> None:
        """★10.29 정책 시작 틱 — 정책 자기 손 목표와 놓음(opened) 기준을 실측 q 로(학습 뱅크 hand_target ≈ q)."""
        st = self.dec.state["arm"]
        self.dec.state["arm"] = replace(st, hand_target=np.asarray(hand_q, float).copy())
        self.hand0 = np.asarray(hand_q, float).copy()

    def _policy_reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _place_begin(self, hand_q: np.ndarray) -> None:
        """★10.25 이송이 끝난 틱 — 다음 틱부터 정책. place_grid_eval rand 처럼 RNN · 직전 행동 0, 래치 추정기 상태 0(sim 은 이송 중
        추정기를 돌리지 않는다), 래치 · 시간 기준 놓음 셈은 처음부터."""
        self._policy_reset()
        self._policy_hand_from_meas(hand_q)
        self.prev = np.zeros(self.c.action_dim)
        if self.est is not None:
            self.est.reset()
        self.latch, self.latch_t, self.latch_geo_n = False, 0, 0
        self.latch_cup = None
        self.step_i = 0

    @property
    def carrying(self) -> bool:
        return self.rcarry is not None and self.rcarry.on

    def _carry_tick(self, meas: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
        """★10.25 준비 자세 이송 한 틱 — 행동 = Mover(실측 팔 q) · 손 = 리셋 손 목표. 관측은 기록용으로만 만든다."""
        c = self.c
        m = meas["arm"]
        pos, quat = self.cup_from_palm(m.palm_pos, m.palm_R)
        m = replace(m, cup_pos=pos, cup_quat=quat)
        obs = A.build_obs(c, m, self.dec, self.goal, self.prev, extra=self._extra())
        a = self.rcarry.step(np.asarray(m.arm_q, float))
        hand_q = np.array([m.hand_q[j] for j in c.side().hand_joints], float)
        targets = self.dec.step(a, active=True, tactile_n=m.tactile_n, hand_q=hand_q)
        self.tact_max = float(np.max(np.asarray(m.tactile_n, float)))
        self.jf_max = None if m.joint_force is None else float(np.max(np.asarray(m.joint_force, float)))
        self.prev = np.clip(a, -1.0, 1.0)
        if not self.rcarry.on:
            self._place_begin(hand_q)
        return obs, a, targets

    @staticmethod
    def _tilt_deg(quat: np.ndarray) -> float:
        """컵 z 축과 세계 z 사이 각 [deg] — hdgp acos(up_z)."""
        up = A.quat_apply(quat, np.array([0.0, 0.0, 1.0]))
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def _latch_now(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        """hdgp rh_place _latch_obs 의 지금 조건 — 지각 컵(손바닥 FK 부착) · 자리 목표."""
        L = self.c.latch_obs
        up = A.quat_apply(quat, np.array([0.0, 0.0, 1.0]))
        tilt = self._tilt_deg(quat)
        bz = float(self.c.object_geometry[self.c.active_object]["bottom_z"]) if self.c.object_names else 0.0
        bxy = (pos + up * bz)[:2] if L["bottom_xy"] else pos[:2]
        g = np.asarray(self.goal.pos, float)
        return bool(pos[2] - g[2] <= latch_dz_of(self.c) and np.linalg.norm(bxy - g[:2]) <= L["dxy"] and tilt <= L["tilt_deg"])

    def _free_block(self, q_meas: np.ndarray, q_star: np.ndarray) -> bool:
        """block_follow_open — 막힘 latch 없음 AND 모든 축이 목표를 따라감 AND 쥔 자세에서 X 이상 열림(sim 과 같은 식)."""
        blk = self.dec.block
        qq = blk.quantize(q_meas)
        if self.q_g is None or blk.grasped():          # 리셋 뒤 첫 틱은 지금 자세, 그 뒤는 쥔 틱마다
            self.q_g = qq.copy()
        i = self._rel_idx
        sgn = blk.sgn[i]
        opened = (self.q_g[i] - qq[i]) * sgn
        follow = (np.asarray(q_star, float)[i] - qq[i]) * sgn < blk.cfg.g_rad
        x = np.asarray(self.c.release_open_rad, float)
        need = x > 0.0 if self.c.release_open_zero_skip else np.ones_like(x, dtype=bool)   # 10.14 A안 — sim 과 같은 식
        return bool(not blk.latch.any() and follow.all() and np.all((opened >= x) | ~need))

    def cup_from_palm(self, palm_pos: np.ndarray, palm_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rel is None:
            raise RhPlaceError("reset first — 컵이 아직 손바닥에 붙지 않았다")
        R = np.asarray(palm_R, float)
        return np.asarray(palm_pos, float) + R @ self.rel[0], _quat_mul(_matrix_to_quat(R), self.rel[1])

    def latch_pose(self, palm_pos: np.ndarray, palm_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """래치 입력 컵(월드) — 추정기가 있으면 손바닥 FK ⊗ 보정 자세(sim _latch_pose), 없으면 붙인 자세."""
        if self.est is None:
            return self.cup_from_palm(palm_pos, palm_R)
        rp, rq = self.est.rel_pose(*self.rel)
        R = np.asarray(palm_R, float)
        return np.asarray(palm_pos, float) + R @ rp, _quat_mul(_matrix_to_quat(R), rq)

    def _extra(self) -> dict | None:
        if not self.c.extra_actor_obs:
            return None
        out = {"latch": 1.0 if self.latch else 0.0}
        out.update({k: 0.0 for k in NB_OBS if k in self.c.extra_actor_obs})    # neighbor_obs "zero"
        return out

    def _clearance(self, m) -> float:
        """hdgp _hand_clearance — 손바닥 · 손끝 6 점마다 max(컵 축 수평거리 − 반지름, 입구 위 높이)의 최솟값. 컵 = 놓은 틱의
        래치 컵(배포는 놓은 뒤 컵을 못 본다 — 홀더로 내려앉으면 실제 입구가 더 낮아 거리를 덜 센다, 보수 쪽)."""
        cup = np.asarray(self.released_cup[0], float)
        g = self.c.object_geometry.get(self.c.active_object or "", {}) if self.c.object_names else {}
        r = float(g.get("radius", 0.0))
        mouth = float(g.get("mouth_z", self.c.cup_half_height))
        pts = np.vstack([np.asarray(m.palm_pos, float).reshape(1, 3), np.asarray(m.tips, float).reshape(-1, 3)])
        radial = np.linalg.norm(pts[:, :2] - cup[None, :2], axis=1) - r
        above = pts[:, 2] - (cup[2] + mouth)
        return float(np.min(np.maximum(radial, above)))

    def _step_s2r(self, meas: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
        """★10.22 newton_05 한 틱 — sim 스텝 순서(pre_physics → 물리 → 보상 · 판정 → 관측)를 배포 틱(측정 → 관측 → 정책 → 행동)에:
        ① 직전 틱 행동의 결과(sim _get_rewards): 래치 뒤 틱 += 1 → release_timed_steps 면 놓음(settle_t 0) · 빼기 중이면
           손이 놓은 컵에서 clear_m 떨어졌거나 max_steps 면 끝
        ② 관측(컵 = 붙인 자세) → 추정기 한 틱(래치 칸은 갱신 전) → 래치 기하 조건 debounce 틱 연속 · 놓기 전이면 래치(유지)
        ③ 정책 → 래치 뒤 놓기 전 · 빼기 중 손 = 편 손 행동(팔은 정책) → 디코더. prev_action = 실행한 행동."""
        if self.carrying:                                   # ★10.25 준비 자세 이송 — 정책 · 래치는 이송 뒤
            return self._carry_tick(meas)
        c = self.c
        m = meas["arm"]
        if self.step_i > 0:                                 # ① 리셋 뒤 첫 틱에는 직전 행동이 없다
            was = self.settling
            if was:
                self.settle_t += 1
            elif self.latch:
                self.latch_t += 1
                if self.latch_t >= c.release_timed_steps:
                    self.settle_t = 0
                    self.released_cup = self.latch_cup
            if was and self.outcome is None:
                self.clearance = self._clearance(m)
                if self.clearance >= float(c.withdraw["clear_m"]):
                    self.outcome = "clear"
                elif self.settle_t >= int(c.withdraw["max_steps"]):
                    self.outcome = "withdraw_timeout"
        pos, quat = self.cup_from_palm(m.palm_pos, m.palm_R)
        m = replace(m, cup_pos=pos, cup_quat=quat)
        obs = A.build_obs(c, m, self.dec, self.goal, self.prev, extra=self._extra())    # ② 래치 칸 = 갱신 전
        if self.est is not None:
            self.est.step(obs)
        self.latch_cup = self.latch_pose(m.palm_pos, m.palm_R)
        now = self._latch_now(*self.latch_cup)
        self.latch_geo_n = self.latch_geo_n + 1 if now else 0
        if self.latch_geo_n >= int(c.latch_debounce) and not self.settling and not self.latch:
            self.latch = True
            obs = A.build_obs(c, m, self.dec, self.goal, self.prev, extra=self._extra())
        n_arm = len(c.side().arm_joints)
        a = np.clip(np.asarray(self.policy.forward(obs), float), -c.action_clip, c.action_clip)
        if (self.latch and not self.settling and c.latch_arm_free) or (self.settling and c.withdraw["hand_open"]):
            a = a.copy()
            a[n_arm:] = self._a_open
        hand_q = np.array([m.hand_q[j] for j in c.side().hand_joints], float)
        targets = self.dec.step(a, active=True, tactile_n=m.tactile_n, hand_q=hand_q)
        f = self._fingers
        self.opened = float(np.mean(self._close_dir[f] * (self.hand0[f] - targets["arm"][1][f])))
        self.tact_max = float(np.max(np.asarray(m.tactile_n, float)))
        self.jf_max = None if m.joint_force is None else float(np.max(np.asarray(m.joint_force, float)))
        self.prev = np.clip(a, -1.0, 1.0)
        self.step_i += 1
        return obs, a, targets

    def step(self, meas: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.c.latch_s2r:
            return self._step_s2r(meas)
        m = meas["arm"]
        pos, quat = self.cup_from_palm(m.palm_pos, m.palm_R)
        m = replace(m, cup_pos=pos, cup_quat=quat)
        if self.c.latch_obs and not self.settling:     # 정책 출력 전 같은 스텝의 지각 자세로(놓은 뒤에는 그대로)
            self.latch = self.latch or self._latch_now(pos, quat)
        extra = {"latch": 1.0 if self.latch else 0.0} if self.c.extra_actor_obs else None
        obs = A.build_obs(self.c, m, self.dec, self.goal, self.prev, extra=extra)
        hand_q = np.array([m.hand_q[j] for j in self.c.side().hand_joints], float)
        carrying = self.carry is not None and self.carry.on
        if self.settling:
            st = self.dec.state["arm"]
            a = np.concatenate([np.clip((self.start_q - st.arm_target) / self.c.k_arm, -1.0, 1.0), self._a_open])
        else:
            a = np.clip(np.asarray(self.policy.forward(obs), float), -self.c.action_clip, self.c.action_clip)
            if carrying:                                # sim: 정책은 돌고(LSTM 진행) 출력은 이송 행동으로 바뀐다
                a = self.carry.step(m.palm_pos, pos, quat, self._palm_jacobian(m.arm_q, hand_q), A.quat_apply)
        targets = self.dec.step(a, active=True, tactile_n=m.tactile_n, hand_q=hand_q)
        f = self._fingers
        self.opened = float(np.mean(self._close_dir[f] * (self.hand0[f] - targets["arm"][1][f])))
        self.tact_max = float(np.max(np.asarray(m.tactile_n, float)))
        self.jf_max = None if m.joint_force is None else float(np.max(np.asarray(m.joint_force, float)))
        if self.settling:
            self.settle_t += 1
        else:
            if self.c.release_rule == "block_follow_open":
                free = self._free_block(hand_q, targets["arm"][1])
            else:
                free = self.opened >= self.open_min_rad and is_free(self.c, m.tactile_n, m.joint_force, self.joint_free_g)
            free = free and not carrying                  # 10.13 이송 틱은 놓음 판정 없음(q_g 갱신은 위에서 그대로)
            self.free_streak = self.free_streak + 1 if free else 0
            gate = self.latch or not self.c.latch_obs     # 관측 래치가 켜진 뒤에만(sim release_requires_latch)
            if self.c.release_max_tilt_deg > 0.0:         # 10.13 놓기 기울기 게이트(sim: 놓음 틱의 컵 기울기, streak 는 그대로 센다)
                gate = gate and self._tilt_deg(quat) <= self.c.release_max_tilt_deg
            if self.free_streak >= self.c.release_steps and gate:
                self.settle_t = 0
        self.prev = np.clip(a, -1.0, 1.0)
        self.step_i += 1
        return obs, a, targets

    def _palm_jacobian(self, arm_q: Sequence[float], hand_q: np.ndarray) -> np.ndarray:
        """손바닥 몸체(sim palm_idx 와 같은 palm_body) 6D 자코비안 — 실측 팔 · 손으로 FK(physx 자코비안과 같은 지금 상태)."""
        def pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            p = self.fk.palm_pose(q, hand_q)
            return np.asarray(p.palm_pos, float), np.asarray(p.extra["palm_rot"], float)
        return PC.palm_jacobian(pose, arm_q)

    def as_dict(self) -> dict:
        return {"phase": "done" if self.done else ("withdraw" if self.c.withdraw else "settle") if self.settling
                else "carry" if (self.carry is not None and self.carry.on) or self.carrying else "latched" if self.latch else "place",
                **({"ready_carry": self.rcarry.as_dict()} if self.rcarry is not None else {}),
                "carry": None if self.carry is None else self.carry.as_dict(), "free_streak": self.free_streak,
                "settle_t": self.settle_t, "seat": None if self.goal is None else [round(float(v), 4) for v in self.goal.pos],
                # 첫 실기 문턱 정하기용(PLACE 검토) — 손 목표가 인계보다 연 양 · 손끝 최대 · 관절 힘 최대
                "opened": round(self.opened, 4), "tact_max": round(self.tact_max, 3), "latch": self.latch,
                "rule": self.c.release_rule,
                "jf_max": None if self.jf_max is None else round(self.jf_max, 1),
                **({"latch_t": self.latch_t, "latch_geo_n": self.latch_geo_n, "outcome": self.outcome,
                    "clearance": None if self.clearance is None else round(self.clearance, 4),
                    "latch_cup": None if self.latch_cup is None else [round(float(v), 4) for v in self.latch_cup[0]],
                    "est": None if self.est is None else self.est.as_dict(), "attach": self.attach_info}
                   if self.c.latch_s2r else {})}


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])
