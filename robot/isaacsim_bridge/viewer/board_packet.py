"""조작 패널(board_jog) -> 보드 뷰어(board_viewer) UDP 계약. 순수 파이썬(ROS·Isaac 무관, py3.10/3.11 공용).

127.0.0.1:47813, UTF-8 JSON 한 줄, 30 Hz. 좌표는 로봇 base(URDF body_root) 기준 m, 각은 rad.

    {"t": float,                                   # 송신 시각(epoch s)
     "names": [str x 13],                          # canonical: r_aj_1..7 + 손 구동 6(r_hj_thumb_1, r_hj_thumb_2, r_hj_{index,middle,ring,pinky}_1)
     "q_show": [float x len(names)],               # 보여 줄 자세. 실측 우선(발행 여부 무관), 없으면 가상; state_source로 구분
     "q_cmd": [float x len(names)] | null,         # 조그 목표(유지 보정 제외). null 이면 고스트를 숨긴다
     "boards": {"charuco": {"T_base_board": 4x4}}, # 보드 자세(board_jog BOARD_CONVENTION). charuco 외 키는 무시
     "target_xyz_base": [3] | null,                # 목표 교차점
     "tip_xyz_base": [3],                          # 실측(또는 dry 가상) 검지 끝 = <p>_hl_index_tip 원점
     "contact": bool,                              # 접촉 표시등
     "mode": "dry" | "pub"}

손 종속(mimic) 관절은 보내지 않는다 — 뷰어가 자산 URDF <mimic> 으로 펼친다. 왼팔 이름(l_*)도 같은 규칙으로 받는다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

DEFAULT_PORT = 47813
MAX_BYTES = 8192
MAX_JOINTS = 64
MODES = ("dry", "pub")


@dataclass(frozen=True)
class BoardPacket:
    t: float
    names: tuple
    q_show: tuple
    q_cmd: tuple | None
    T_base_board: tuple | None          # 4x4 행 튜플. 보드가 없으면 None
    target: tuple | None
    tip: tuple
    contact: bool
    mode: str
    tip_goal: tuple | None = None      # tip pose(base), optional for old senders
    state_source: str = "unknown"
    tip_meas: tuple | None = None


def _num(v, what: str) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
        raise ValueError(f"{what} 가 유한한 수가 아니다: {v!r}")
    return float(v)


def _vec(v, n: int, what: str) -> tuple:
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"{what} 는 길이 {n} 배열이어야 한다: {v!r}")
    return tuple(_num(x, what) for x in v)


def _rigid(T) -> tuple:
    if not isinstance(T, list) or len(T) != 4:
        raise ValueError("T_base_board 는 4x4 여야 한다")
    rows = tuple(_vec(r, 4, "T_base_board 행") for r in T)
    R = [r[:3] for r in rows[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i] * R[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-3:
                raise ValueError("T_base_board 회전부가 직교 행렬이 아니다")
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if det < 0.0 or rows[3] != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("T_base_board 가 강체 변환이 아니다(det<0 또는 마지막 행 ≠ 0 0 0 1)")
    return rows


def decode(data: bytes) -> BoardPacket:
    """데이터그램 -> BoardPacket. 틀리면 ValueError(뷰어는 그 패킷만 버린다)."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"데이터그램 {len(data)} B > {MAX_BYTES} B")
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON 해석 실패: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("최상위가 객체가 아니다")
    missing = {"t", "names", "q_show", "tip_xyz_base", "contact", "mode"} - set(obj)
    if missing:
        raise ValueError(f"필드 누락: {sorted(missing)}")
    names = obj["names"]
    if (not isinstance(names, list) or not names or len(names) > MAX_JOINTS or len(set(names)) != len(names)
            or not all(isinstance(n, str) and n for n in names)):
        raise ValueError(f"names 가 비었거나 중복·빈 이름·{MAX_JOINTS} 초과: {names!r}")
    n = len(names)
    if obj.get("state_source", "unknown") not in ("measured", "virtual", "unknown"):
        raise ValueError("state_source must be measured, virtual, or unknown")
    q_cmd = obj.get("q_cmd")
    target = obj.get("target_xyz_base")
    boards = obj.get("boards") or {}
    if not isinstance(boards, dict):
        raise ValueError("boards 는 객체여야 한다")
    board = boards.get("charuco")
    if board is not None and not (isinstance(board, dict) and "T_base_board" in board):
        raise ValueError("boards.charuco 에 T_base_board 가 없다")
    if not isinstance(obj["contact"], bool):
        raise ValueError(f"contact 는 bool: {obj['contact']!r}")
    if obj["mode"] not in MODES:
        raise ValueError(f"mode 는 {MODES} 중 하나: {obj['mode']!r}")
    return BoardPacket(
        t=_num(obj["t"], "t"), names=tuple(names), q_show=_vec(obj["q_show"], n, "q_show"),
        q_cmd=None if q_cmd is None else _vec(q_cmd, n, "q_cmd"),
        T_base_board=None if board is None else _rigid(board["T_base_board"]),
        target=None if target is None else _vec(target, 3, "target_xyz_base"),
        tip=_vec(obj["tip_xyz_base"], 3, "tip_xyz_base"), contact=obj["contact"], mode=obj["mode"],
        tip_goal=None if obj.get("T_tip_goal") is None else _rigid(obj["T_tip_goal"]),
        tip_meas=None if obj.get("T_tip_meas") is None else _rigid(obj["T_tip_meas"]),
        state_source=obj.get("state_source", "unknown"))


def encode(t: float, names, q_show, q_cmd, T_base_board, target, tip, contact: bool, mode: str) -> bytes:
    """송신 쪽(패널·가짜 송신기)용. 넣은 값을 decode 로 한 번 검사하고 돌려준다."""
    obj = {"t": float(t), "names": list(names), "q_show": [float(v) for v in q_show],
           "q_cmd": None if q_cmd is None else [float(v) for v in q_cmd],
           "boards": {} if T_base_board is None else {"charuco": {"T_base_board": [[float(v) for v in r] for r in T_base_board]}},
           "target_xyz_base": None if target is None else [float(v) for v in target],
           "tip_xyz_base": [float(v) for v in tip], "contact": bool(contact), "mode": mode}
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    decode(data)
    return data


def state_audit(pkt, joint_names, applied_positions, physical_tip_world, usd_tip_world,
                origin, received_at, applied_at, audit_origin):
    """Compare decoded input to simulator readback; no FK model is substituted.

    Physical tip comes from the articulation tensor API; USD tip is a separate
    stage readback (Fabric may update these on different schedules).
    """
    names, positions = list(joint_names), list(applied_positions)
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("readback joint names/positions must match and be unique")
    applied = {n: _num(v, "applied position") for n, v in zip(names, positions)}
    reference = dict(zip(pkt.names, pkt.q_show))
    errors = {n: applied[n] - q for n, q in reference.items() if n in applied}
    base_origin = _vec(list(origin), 3, "origin")

    def base_tip(world):
        return None if world is None else [v - o for v, o in zip(_vec(list(world), 3, "world tip"), base_origin)]

    physical, usd = base_tip(physical_tip_world), base_tip(usd_tip_world)

    def error_mm(tip):
        return None if tip is None else 1000 * math.sqrt(sum((a - b) ** 2 for a, b in zip(tip, pkt.tip)))

    return {"audit_origin": audit_origin, "packet_t": pkt.t, "received_at": received_at,
            "applied_at": applied_at, "state_source": pkt.state_source, "mode": pkt.mode,
            "input_q_show": reference, "applied_positions": applied,
            "missing_joints": [n for n in reference if n not in applied],
            "joint_errors_rad": errors, "joint_max_abs_error_rad": max(map(abs, errors.values()), default=None),
            "packet_tip_base_m": list(pkt.tip), "physical_tip_base_m": physical,
            "usd_tip_base_m": usd, "physical_tip_error_mm": error_mm(physical),
            "usd_tip_error_mm": error_mm(usd)}
