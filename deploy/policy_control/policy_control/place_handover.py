"""★10.22 place 시작 컵 자세 = 잡기가 남긴 grasp_state 넘김 묶음(hdgp modules/inhand_pose/grasp_state.py grasp_handover/1).

learn:place 10.21 인계 · 10.22 확정: newton_05 는 "FP++ 는 처음부터 없음, 파지 자세와 인계받은 컵 자세에서 시작" — 리셋(인계)
순간의 FP++ 스냅숏 대신 묶음의 손바닥 좌표 rel_p · rel_q(wxyz) 를 붙인 자세(sim _pl_att_pos · _pl_att_quat)로 쓴다.
끊긴 순간 고정 자세(64~81 mm 어긋남)는 쓰지 않는다. 묶음은 잡기 노드(rh_aglt_node_<팔>, inhand_est.save_bundle)가 들기 성공으로
끝날 때 logs/inhand/<날짜>/<시각>_<노드>_<에피소드>.npz 로 남긴다.

auto 고르기(노드 파라미터 bundle:=auto): 그 팔 잡기 노드의 가장 새 묶음 중
  · 만든 지 max_age_s 안 · 이 place 노드가 아직 쓰지 않은 것 · valid(끝 창이 다 참) · 같은 팔 · 같은 물체
  · 묶음의 손 관절(들기 끝 실측)이 지금 손 실측과 hand_tol_rad 안 — pd 가 그 뒤로 손을 붙든 채인가(다른 잡기가 끼지 않았나)
하나라도 어긋나면 reset 거부(조용히 FP++ 로 돌아가지 않는다).
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import _paths

FORMAT = "grasp_handover/1"
QUAT_ORDER = "wxyz"
TENSORS = ("rel_p", "rel_q", "obj_axis_palm", "grasp_height", "hand_q", "n_steps", "valid")
BUNDLE_DIR = _paths.SIM2REAL / "logs" / "inhand"
SIDES = {"right": "r", "left": "l"}


class HandoverError(ValueError):
    pass


@dataclass(frozen=True)
class Bundle:
    path: Path
    side: str                   # r | l
    object_contract: str        # 잡기 계약 물체(배포 이름, 예 source240_real)
    rel_p: np.ndarray           # (3,) 손바닥 좌표 컵 원점
    rel_q: np.ndarray           # (4,) wxyz
    hand_q: np.ndarray          # (6,) 들기 끝 손 관절 실측(계약 손 순서 rad)
    valid: bool
    n_steps: int
    created: float              # epoch s
    meta: dict

    def as_dict(self) -> dict:
        return {"path": str(self.path), "object": self.object_contract, "rel_p": [round(float(v), 4) for v in self.rel_p],
                "rel_q": [round(float(v), 4) for v in self.rel_q], "n_steps": self.n_steps, "valid": self.valid,
                "age_s": round(time.time() - self.created, 1), "node": self.meta.get("node"), "episode": self.meta.get("episode")}


def load(path: str | Path) -> Bundle:
    p = Path(path)
    try:
        z = np.load(str(p), allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        arr = {k: np.asarray(z[k]) for k in TENSORS}
    except (OSError, KeyError, ValueError) as exc:
        raise HandoverError(f"넘김 묶음 {p} 를 못 읽는다: {exc}") from exc
    if meta.get("format") != FORMAT or meta.get("quat_order") != QUAT_ORDER or meta.get("frame", "palm") != "palm":
        raise HandoverError(f"{p}: 형식 {meta.get('format')} · {meta.get('quat_order')} · {meta.get('frame')} — {FORMAT} · {QUAT_ORDER} · palm")
    if arr["rel_p"].shape != (1, 3) or arr["rel_q"].shape != (1, 4):
        raise HandoverError(f"{p}: 한 팔 묶음(행 1)이 아니다 {arr['rel_p'].shape}")
    q = arr["rel_q"][0].astype(float)
    if not (np.all(np.isfinite(arr["rel_p"])) and np.all(np.isfinite(q)) and abs(float(np.linalg.norm(q)) - 1.0) < 1e-3):
        raise HandoverError(f"{p}: rel_p · rel_q 가 유한한 단위 쿼터니언이 아니다")
    try:
        created = _dt.datetime.fromisoformat(str(meta["created"])).timestamp()
    except (KeyError, ValueError):
        created = p.stat().st_mtime
    return Bundle(path=p, side=str(meta.get("side")), object_contract=str(meta.get("object_contract") or ""),
                  rel_p=arr["rel_p"][0].astype(float), rel_q=q / np.linalg.norm(q),
                  hand_q=np.asarray(arr["hand_q"], float).reshape(-1), valid=bool(np.asarray(arr["valid"]).reshape(-1)[0]),
                  n_steps=int(np.asarray(arr["n_steps"]).reshape(-1)[0]), created=float(created), meta=meta)


def check(b: Bundle, *, side: str, object_name: str | None, hand_q: Sequence[float], hand_tol_rad: float,
          max_age_s: float, now: float | None = None) -> list:
    """이 place 시작에 쓸 수 없는 이유(빈 목록 = 쓴다)."""
    out = []
    want = SIDES.get(side, side)
    if b.side != want:
        out.append(f"묶음 팔 {b.side} ≠ {want}")
    if not b.valid:
        out.append(f"묶음 valid False(끝 창 {b.n_steps} 틱 — 끊긴 뒤 충분히 쌓이지 않았다)")
    if object_name and b.object_contract and b.object_contract != object_name:
        out.append(f"묶음 물체 {b.object_contract} ≠ place 물체 {object_name}")
    age = (time.time() if now is None else now) - b.created
    if max_age_s > 0 and age > max_age_s:
        out.append(f"묶음이 {age:.0f} s 전 것(> {max_age_s:.0f} s) — 이번 잡기의 묶음이 아니다")
    h = np.asarray(hand_q, float).reshape(-1)
    if h.size != b.hand_q.size:
        out.append(f"손 관절 {h.size} ≠ 묶음 {b.hand_q.size}")
    else:
        err = float(np.abs(h - b.hand_q).max())
        if err > hand_tol_rad:
            out.append(f"지금 손 실측이 묶음(들기 끝)과 {err:.3f} rad 다르다(> {hand_tol_rad}) — 그 사이 손이 움직였다")
    return out


def candidates(side: str, node: str | None = None, root: Path = BUNDLE_DIR, days: int = 2) -> list:
    """그 팔 잡기 노드의 묶음 파일 — 새 것부터(오늘 · 어제 폴더)."""
    tag = node or f"rh_aglt_node_{side}"
    today = _dt.date.today()
    dirs = [Path(root) / (today - _dt.timedelta(days=d)).strftime("%Y%m%d") for d in range(days)]
    files = [f for d in dirs if d.is_dir() for f in d.glob(f"*_{tag}_*.npz")]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def pick(side: str, *, object_name: str | None, hand_q: Sequence[float], hand_tol_rad: float, max_age_s: float,
         used: Iterable[str] = (), root: Path = BUNDLE_DIR) -> Bundle:
    """auto — 가장 새 묶음 하나만 본다(그보다 옛 묶음으로 내려가지 않는다: 이번 잡기가 묶음을 못 남겼으면 거부)."""
    files = candidates(side, root=root)
    if not files:
        raise HandoverError(f"{side} 잡기 노드의 넘김 묶음이 {root} 에 없다 — 경로 F(rcg04 + 추정기)로 들어 올린 뒤에 place")
    f = files[0]
    if str(f) in set(used):
        raise HandoverError(f"가장 새 묶음 {f.name} 은 이 노드가 이미 썼다 — 새로 잡아 들어 올린 뒤에 place")
    b = load(f)
    bad = check(b, side=side, object_name=object_name, hand_q=hand_q, hand_tol_rad=hand_tol_rad, max_age_s=max_age_s)
    if bad:
        raise HandoverError(f"넘김 묶음 {f.name}: " + " · ".join(bad))
    return b
