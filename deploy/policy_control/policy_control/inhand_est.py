"""★10.21 손 안 컵 추정기 2 개(hdgp agnostic/modules/inhand_pose) — rcg_04 정책 폴더에 두면 배포 기본으로 켜진다.

10.21 사용자: "추정기들 붙여야 s2r 될것 같은 디폴트 세팅으로 해둬야함".
근거 = learn:grasp/lift sim 폐루프(rcg_04 격자, Newton, 팁 파지 제외 성공 %, 10.21 회신):
  출발    base(FP++ 끝까지) · frozen(지금까지의 배포 — 끊긴 순간 손바닥 기준 자세를 손바닥 FK 에 붙임) · gru_obs(이 모듈)
  ap_r     86.0 · 26.0 · 74.4        ready_r 79.3 · 31.1 · 78.4
  ap_l     97.6 · 31.3 · 92.7        ready_l 86.9 · 37.8 · 84.2
→ 끊긴 뒤 정책이 보는 컵 = 손바닥 FK ⊗ GRU 추정(손바닥 좌표). 기준 구현 = server our_source/rh_aglt_runs/source_cup/gru_loop_hook.py perceive().

두 추정기(정책 폴더 inhand/inhand.yaml · ckpt md5 핀)
  gru          runtime.InHandPoseEstimator(과제 관측판, prior_input) — 정책 틱마다 한 번. 입력 = 직전 틱 관측(이번에 받은 행동을 낸
               obs, 컵 칸은 ckpt 가 뺀다) + 사전값. 손바닥 · 컵은 이번 측정(= sim 의 스텝 뒤). 출력 = 손바닥 좌표 컵 자세.
  grasp_state  GraspStateEstimator(정적 MLP) — 끊긴 뒤 매 틱 쌓고(끊긴 틱에 새로), 들기 성공으로 끝날 때 끝 30 틱 평균을 넘김 묶음
               (grasp_handover/1 npz, logs/inhand/)으로 남긴다. 다음 정책(place · tilt)이 object_world(손바닥 FK, 묶음)로 쓴다.
               입력 손 명령 = 디코더의 rad 손 목표 q*(squeeze 전, sim env.hand_target 과 같은 뜻 — learn:grasp/lift 10.21).
끊김(FppDropLatch — sim 훅 · 과제 env · 배포 공용 클래스)
  지각 컵(노드가 고른 FP++ · 마지막 값 · 부착)과 손바닥 거리가 처음 drop_m 미만인 틱에 끊는다. 그 틱의 손바닥 기준 지각 자세가
  사전값이고, 에피소드 끝까지 끊긴 채다. 끊기기 전 사전값 = 매 틱 지각값. sim 은 참값 거리 d ~ U(6.5, 16) cm(가림 모사) — 배포는
  지각 거리 · drop_m 하나(실기 10.20: 오른손 7.8 cm 까지 live, 왼손 8.6 cm 에서 3.2 s 끊김 — 둘 다 그 범위 안).
  live = 리셋 뒤 첫 틱만 False(훅 episode_length_buf > 0). 경로 D 넘김(ap05 → rcg)에서도 GRU 상태를 잇는다(훅과 같다).
코드 = hdgp 고정 커밋 사본(hdgp_pin.snapshot, 공유 작업 트리를 다른 세션이 고쳐도 그대로) + 파일 md5 핀(둘 다 inhand.yaml).
"""
from __future__ import annotations

import datetime as _dt
import importlib
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Mapping

import numpy as np
import yaml

from . import hdgp_pin
from . import rh_aglt as A

INHAND_DIR = "inhand"
INHAND_YAML = "inhand.yaml"
PKG = "inhand_pose_hdgp"            # openarm 패키지 초기화(학습 과제 설정 import)를 거치지 않게 따로 이름을 붙인다
SIDES = {"right": "r", "left": "l"}
BUNDLE_DIR = Path(__file__).resolve().parents[3] / "logs" / "inhand"


class InhandError(A.RhAgltError):
    """추정기 설정 · 파일이 맞지 않는다 — 노드 reset 이 거부한다(움직이기 전에 멈춘다)."""


@dataclass(frozen=True)
class InhandSpec:
    path: Path                      # inhand.yaml
    side: str                       # r | l
    commit: str                     # hdgp 고정 커밋
    module: str                     # hdgp 루트 기준 inhand_pose 폴더
    module_md5: Mapping[str, str]   # 파일 이름 → md5
    gru: str                        # inhand/ 기준 파일
    gru_md5: str
    grasp_state: str
    grasp_state_md5: str
    drop_m: float

    @property
    def root(self) -> Path:
        return self.path.parent


def load_spec(path: str | Path) -> InhandSpec:
    p = Path(path)
    if not p.is_file():
        raise InhandError(f"추정기 설정 {p} 가 없다")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        hd, gru, gs = raw["hdgp"], raw["gru"], raw["grasp_state"]
        spec = InhandSpec(path=p.resolve(), side=str(raw["side"]), commit=str(hd["commit"]), module=str(hd["module"]),
                          module_md5={str(k): str(v) for k, v in dict(hd["md5"]).items()},
                          gru=str(gru["file"]), gru_md5=str(gru["md5"]),
                          grasp_state=str(gs["file"]), grasp_state_md5=str(gs["md5"]), drop_m=float(raw["drop_m"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise InhandError(f"{p}: side · drop_m · hdgp{{commit, module, md5}} · gru{{file, md5}} · grasp_state{{file, md5}} 가 필요하다 ({exc})") from exc
    if spec.side not in SIDES.values():
        raise InhandError(f"{p}: side 는 r 또는 l: {spec.side!r}")
    if not (np.isfinite(spec.drop_m) and 0.0 < spec.drop_m <= 0.3):
        raise InhandError(f"{p}: drop_m 은 0 < x ≤ 0.3 m: {spec.drop_m}")
    if not spec.module_md5:
        raise InhandError(f"{p}: hdgp.md5 가 비었다")
    return spec


def find_spec(*contracts) -> InhandSpec | None:
    """계약 파일들의 정책 폴더에서 처음 찾은 inhand/inhand.yaml(없으면 None) — 경로 D 는 넘겨받는 정책(rcg)을 먼저 준다."""
    for c in contracts:
        if not c:
            continue
        p = Path(c).resolve().parent / INHAND_DIR / INHAND_YAML
        if p.is_file():
            return load_spec(p)
    return None


def load_pkg(spec: InhandSpec, root: Path | None = None) -> ModuleType:
    """hdgp 고정 커밋 사본의 inhand_pose 폴더를 독립 패키지(PKG)로 읽는다. md5 가 하나라도 다르면 InhandError."""
    root = hdgp_pin.hdgp_tree() if root is None else Path(root)
    try:
        snap = hdgp_pin.snapshot(root, spec.commit, [spec.module])
        hdgp_pin.verify(snap / spec.module, spec.module_md5)
    except hdgp_pin.PinError as exc:
        raise InhandError(f"추정기 코드: {exc}") from exc
    d = snap / spec.module
    have = sys.modules.get(PKG)
    if have is not None:
        if Path(getattr(have, "__file__", "")).parent != d:
            raise InhandError(f"이 프로세스는 이미 다른 사본의 {PKG}({have.__file__})를 읽었다 — 노드를 다시 띄운다")
        return have
    s = importlib.util.spec_from_file_location(PKG, d / "__init__.py", submodule_search_locations=[str(d)])
    if s is None or s.loader is None:
        raise InhandError(f"{d} 를 읽지 못했다")
    pkg = importlib.util.module_from_spec(s)
    sys.modules[PKG] = pkg
    try:
        s.loader.exec_module(pkg)
        for sub in ("geometry", "latch", "runtime", "grasp_state"):
            importlib.import_module(f"{PKG}.{sub}")
    except Exception:
        for k in [k for k in sys.modules if k == PKG or k.startswith(PKG + ".")]:
            del sys.modules[k]
        raise
    return pkg


def ckpt_object_names(c: A.RaContract, ckpt_objects) -> dict:
    """계약 물체 이름(source240_real …) → ckpt 물체 이름(source240_cvx …). sim 충돌체 별칭(A.SIM_COLLIDER_ALIAS)으로 잇는다."""
    out = {}
    for name in c.object_names:
        hits = [o for o in ckpt_objects if o == name or A.SIM_COLLIDER_ALIAS.get(o) == name]
        if len(hits) != 1:
            raise InhandError(f"물체 {name!r} 를 ckpt 물체 {list(ckpt_objects)} 에서 하나로 찾지 못했다({hits})")
        out[name] = hits[0]
    return out


class InhandRun:
    """한 팔의 추정기 둘 + 끊김 래치. 노드가 정책 틱마다 perceive → (정책) → remember 순으로 부른다."""

    def __init__(self, spec: InhandSpec, contract: A.RaContract, pkg: ModuleType | None = None, *,
                 drop_m: float | None = None) -> None:
        import torch

        self.torch = torch
        self.spec = spec
        side = SIDES.get(contract.side().side)
        if side != spec.side:
            raise InhandError(f"{spec.path}: side {spec.side} ≠ 계약 {contract.side().side}")
        for f, want in ((spec.gru, spec.gru_md5), (spec.grasp_state, spec.grasp_state_md5)):
            p = spec.root / f
            if not p.is_file():
                raise InhandError(f"추정기 ckpt {p} 가 없다 — inhand.yaml 의 source 에서 받는다(rsync, md5 {want})")
            if hdgp_pin.md5(p) != want:
                raise InhandError(f"추정기 ckpt {p}: md5 {hdgp_pin.md5(p)} ≠ 고정 {want}")
        self.pkg = pkg if pkg is not None else load_pkg(spec)
        self.geo = self.pkg.geometry
        lay = A.actor_layout(contract)
        try:
            self.gru = self.pkg.runtime.InHandPoseEstimator(str(spec.root / spec.gru), 1, "cpu", layout=lay)
        except ValueError as exc:
            raise InhandError(f"GRU {spec.gru}: {exc}") from exc
        if self.gru.proprio or not self.gru.prior_input:
            raise InhandError(f"GRU {spec.gru}: 과제 관측판 · prior_input ckpt 만 받는다(proprio {self.gru.proprio} · "
                              f"prior_input {self.gru.prior_input})")
        dt = (self.gru.ck.get("meta") or {}).get("dt")
        if dt is not None and abs(float(dt) * float(contract.policy_hz) - 1.0) > 1e-3:
            raise InhandError(f"GRU {spec.gru}: 학습 dt {dt} ≠ 정책 1/{contract.policy_hz}")
        try:
            self.gs = self.pkg.grasp_state.GraspStateEstimator(str(spec.root / spec.grasp_state), 1, "cpu", side=spec.side)
        except ValueError as exc:
            raise InhandError(f"grasp_state {spec.grasp_state}: {exc}") from exc
        if self.gs.n_hand != len(contract.side().hand_joints):
            raise InhandError(f"grasp_state 손 관절 {self.gs.n_hand} ≠ 계약 {len(contract.side().hand_joints)}")
        self.names = ckpt_object_names(contract, self.gs.objects)
        self.drop_m = float(spec.drop_m if drop_m is None else drop_m)
        if not (np.isfinite(self.drop_m) and 0.0 < self.drop_m <= 0.3):
            raise InhandError(f"drop_m 은 0 < x ≤ 0.3 m: {self.drop_m}")
        self.width = int(self.gru.width)
        self.object: str | None = None
        self.reset(None)

    # ---------------------------------------------------------------- 에피소드
    def reset(self, object_name: str | None) -> None:
        """새 에피소드 — 래치 · GRU · grasp_state 를 비운다. object_name = 계약 물체(없으면 grasp_state 를 쌓지 않는다)."""
        if object_name is not None and object_name not in self.names:
            raise InhandError(f"물체 {object_name!r} 는 추정기 물체 {list(self.names)} 밖이다")
        self.object = object_name
        self.latch = self.pkg.latch.FppDropLatch(1, "cpu", self.drop_m)
        self.gru.reset(slice(None))
        self.gs.reset()
        self._oh = None if object_name is None else self.gs.onehot([self.names[object_name]])
        self.k = 0
        self.prev_obs: np.ndarray | None = None
        self.dist_m: float | None = None
        self.drop_tick: int | None = None
        self.est_p: np.ndarray | None = None
        self.est_q: np.ndarray | None = None
        self.hand_q: np.ndarray | None = None          # 마지막 perceive 의 손 관절(계약 순서 rad) — 넘김 묶음 hand_q
        self.bundle_path: str | None = None

    @property
    def dropped(self) -> bool:
        return bool(self.latch.dropped[0])

    def _t(self, x, n: int):
        return self.torch.as_tensor(np.asarray(x, float).reshape(1, n), dtype=self.torch.float32)

    def perceive(self, m: A.RaMeas, hand_target, hand_q) -> A.RaMeas:
        """이번 틱 측정 → 정책이 볼 측정(끊긴 뒤면 컵 = 손바닥 FK ⊗ GRU). hand_target = 디코더 q*(직전 틱 결과), hand_q = 계약 손 순서 rad."""
        torch = self.torch
        live = torch.tensor([self.k > 0])
        pp = self._t(m.palm_pos, 3)
        pq = self.geo.matrix_to_quat(torch.as_tensor(np.asarray(m.palm_R, float), dtype=torch.float32)[None])
        cp, cq = self._t(m.cup_pos, 3), self._t(m.cup_quat, 4)
        dist = (cp - pp).norm(dim=-1)
        self.dist_m = float(dist[0])
        L = self.latch
        new = L.update(dist, cp, cq, pp, pq, live)
        if bool(new[0]):
            self.drop_tick = self.k
            self.gs.reset()                            # 끊긴 순간부터 새로 쌓는다(학습 표본도 끊김 뒤만 — grasp_handover_hook)
        ob = torch.zeros(1, self.width) if self.prev_obs is None else self._t(self.prev_obs, self.width)
        ep, eq = self.gru.step(ob, live, prior=(L.prior_p, L.prior_q))
        if self._oh is not None:
            self.gs.step(L.dropped.clone(), self._oh, self._t(hand_q, self.gs.n_hand), self._t(hand_target, self.gs.n_hand),
                         L.prior_p, L.prior_q, L.palm0_p, L.palm0_q, pp, pq)
        self.k += 1
        self.hand_q = np.asarray(hand_q, float).reshape(-1).copy()
        self.est_p, self.est_q = ep[0].numpy().astype(float), eq[0].numpy().astype(float)
        if not self.dropped:
            return m
        wp, wq = self.geo.world_pose(pp, pq, ep, eq)
        return replace(m, cup_pos=wp[0].numpy().astype(float), cup_quat=wq[0].numpy().astype(float))

    def remember(self, obs) -> None:
        """이번 틱에 정책이 본 관측 — 다음 틱 GRU 입력(sim obs_buf: 그 스텝 행동을 낸 관측)."""
        o = np.asarray(obs, float).reshape(-1)
        if o.size != self.width:
            raise InhandError(f"관측 폭 {o.size} ≠ GRU layout {self.width}")
        self.prev_obs = o.copy()

    # ---------------------------------------------------------------- 넘김 묶음
    def bundle(self, hand_q=None) -> dict | None:
        """끊긴 뒤 쌓은 grasp_state 의 넘김 묶음(끝 창 평균). 끊기지 않았거나 물체 · 손 관절을 모르면 None.
        hand_q 없으면 마지막 perceive 의 손 관절."""
        hand_q = self.hand_q if hand_q is None else hand_q
        if not self.dropped or self._oh is None or hand_q is None:
            return None
        return self.gs.bundle([self.names[self.object]], self._t(hand_q, self.gs.n_hand))

    def save_bundle(self, hand_q, meta: Mapping, out_dir: Path | None = None) -> str | None:
        """넘김 묶음을 grasp_handover/1 npz 로 쓴다 → 경로(묶음이 없으면 None). 파일 이름 = 시각 · 노드 · 에피소드.
        hand_q 없으면 마지막 perceive 의 손 관절."""
        b = self.bundle(hand_q)
        if b is None:
            return None
        now = _dt.datetime.now()
        d = Path(out_dir or BUNDLE_DIR) / now.strftime("%Y%m%d")
        tag = "_".join(str(meta.get(k, "")) for k in ("node", "episode") if meta.get(k) not in (None, ""))
        path = d / f"{now.strftime('%H%M%S')}_{tag or 'inhand'}.npz"
        full = {"spec": str(self.spec.path), "gru_md5": self.spec.gru_md5, "grasp_state_md5": self.spec.grasp_state_md5,
                "hdgp_commit": self.spec.commit, "drop_m": self.drop_m, "drop_tick": self.drop_tick, "ticks": self.k,
                "object_contract": self.object, **dict(meta)}
        self.pkg.grasp_state.save_bundle(path, b, full)
        self.bundle_path = str(path)
        return self.bundle_path

    def as_dict(self) -> dict:
        r = lambda v, n=4: None if v is None else [round(float(x), n) for x in np.asarray(v).reshape(-1)]   # noqa: E731
        return {"on": True, "dropped": self.dropped, "drop_m": self.drop_m, "dist_m": None if self.dist_m is None else round(self.dist_m, 4),
                "drop_tick": self.drop_tick, "ticks": self.k, "est_rel_p": r(self.est_p), "est_rel_q": r(self.est_q),
                "prior_rel_p": r(self.latch.prior_p[0].numpy()), "gs_steps": int(self.gs.cnt[0]), "bundle": self.bundle_path}
