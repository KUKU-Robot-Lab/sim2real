"""RH56F1 EtherCAT — 마스터(tools/ethercat/rh56f1_ecat_master.c)와 주고받는 형식 · PDO 해석 · 명령 합치기. 순수(ROS 없음).

10.02 사용자: RS485 를 더 쓰지 않고 손을 EtherCAT 으로 제어한다(정책 제어 포함). 배선: 손 하나 = NIC 하나
(오른손 USB-C 랜 · 왼손 내장 랜, config/rh56f1_ports.yaml). PDO 는 매뉴얼 RH56F1-User-ManualV1.2 §2.6 표 50:
  입력 76 × INT16 (0x6000:01~4C)  POSACT · ANGLEACT · FORCEACT · CURACT · ERROR · STATUS · TEMP 각 6(슬롯 순 새끼부터)
                                   + 손가락 5 × (법선 · 접선 · 방향 · 근접 L · 근접 H) + 손바닥 3 × (법선 · 접선 · 방향)
  출력 19 × INT16 (0x7000:01~13)  ENABLE_SET · ANGLESET 6 · FORCESET 6 · SPEEDSET 6
각도 레지스터 단위 · 슬롯 순서는 RS485 레지스터와 같다 — config/rh56f1_hand_map.yaml 변환표를 그대로 쓴다.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Sequence

N_IN, N_OUT, N = 76, 19, 6
STATE_MAGIC = 0x31534852   # "RHS1"
CMD_MAGIC = 0x31434852     # "RHC1"
CMD_HEARTBEAT, CMD_ANGLE, CMD_FORCE, CMD_SPEED, CMD_ENABLE, CMD_MODE, CMD_ANGLE_ADM = 0, 1, 2, 3, 4, 5, 6
FLAG_OP, FLAG_ENABLED, FLAG_COMMANDED, FLAG_NODE_OK, FLAG_STOPPING = 0x01, 0x02, 0x04, 0x08, 0x10

_STATE = struct.Struct(f"<IIQHHHHIHH{N_IN}h{N_OUT}h")
_CMD = struct.Struct("<IHH6i")
STATE_SIZE, CMD_SIZE = _STATE.size, _CMD.size

#: 매뉴얼 2.5.11 각도 범위(슬롯 순) · 2.5.12 힘(g) · 2.5.13 속도 — 마스터도 같은 값으로 자른다
ANGLE_LO = (900, 900, 900, 900, 1100, 600)
ANGLE_HI = (1740, 1740, 1740, 1740, 1350, 1800)
FORCE_MAX, SPEED_MAX = 1000, 4000
#: 10.03 실측 — 손 펌웨어(LAN9252 SSC)가 이보다 빠른 PDO 주기에서는 OP 요청도 명령도 읽지 못한다(500 OK · 750 · 1000 무시)
MAX_CYCLE_HZ = 500.0
SLOT_FINGERS = ("pinky_1", "ring_1", "middle_1", "index_1", "thumb_2", "thumb_1")   # 벤더 드라이버와 같은 순서 · 이름
#: 매뉴얼 표 46 상태 코드(255 는 표에 없다 — 10.02 왼손 전원 뒤 명령 전 값으로 관찰)
STATUS_TEXT = {0: "펴는 중", 1: "쥐는 중", 2: "위치 도달 정지", 3: "힘 제어 정지", 5: "전류 보호 정지",
               6: "구속 정지", 7: "고장 정지"}


class EcatError(ValueError):
    pass


def _u16(v: int) -> int:
    """INT16 로 온 부호 없는 값(방향 0~359 · 65535 = 접촉 없음, 근접 32 비트의 반쪽)."""
    return int(v) & 0xFFFF


@dataclass(frozen=True)
class EcatState:
    seq: int
    t_ns: int
    al_state: int
    al_code: int
    flags: int
    wkc_bad: int
    cycles: int
    rtt_max_us: int
    late_max_us: int
    inputs: tuple
    outputs: tuple

    def _six(self, k: int) -> list[int]:
        return list(self.inputs[k * N:(k + 1) * N])

    @property
    def position(self) -> list[int]:
        return self._six(0)

    @property
    def angle(self) -> list[int]:
        return self._six(1)

    @property
    def force(self) -> list[int]:
        return self._six(2)

    @property
    def current(self) -> list[int]:
        return self._six(3)

    @property
    def error(self) -> list[int]:
        return self._six(4)

    @property
    def status(self) -> list[int]:
        return self._six(5)

    @property
    def temperature(self) -> list[int]:
        return self._six(6)

    def touch(self) -> dict[str, list[int]]:
        """벤더 TouchData1 과 같은 필드. 손가락 순서는 새끼부터(벤더와 같다). palm_data 는 RS485 9 칸과 뜻이 다르다:
        EtherCAT 은 손바닥 3 영역 × (법선 · 접선 · 방향)."""
        f = self.inputs[42:67]
        fingers = [f[i * 5:(i + 1) * 5] for i in range(5)]
        return {"finger_forces": [x[0] for x in fingers],
                "finger_tangentials": [x[1] for x in fingers],
                "finger_angles": [_u16(x[2]) for x in fingers],
                "finger_proximity": [(_u16(x[4]) << 16) | _u16(x[3]) for x in fingers],
                "palm_data": list(self.inputs[67:76])}

    @property
    def is_op(self) -> bool:
        return bool(self.flags & FLAG_OP)

    @property
    def commanded(self) -> bool:
        return bool(self.flags & FLAG_COMMANDED)

    def summary(self) -> dict:
        return {"seq": self.seq, "op": self.is_op, "al_state": self.al_state, "al_code": self.al_code,
                "enabled": bool(self.flags & FLAG_ENABLED), "commanded": self.commanded,
                "node_ok": bool(self.flags & FLAG_NODE_OK), "stopping": bool(self.flags & FLAG_STOPPING),
                "wkc_bad": self.wkc_bad, "cycles": self.cycles, "rtt_max_us": self.rtt_max_us,
                "late_max_us": self.late_max_us, "angle": self.angle, "target": list(self.outputs[1:7]),
                "error": self.error, "status": self.status,
                "status_text": [STATUS_TEXT.get(s, f"표에 없음({s})") for s in self.status],
                "temperature": self.temperature}


def unpack_state(buf: bytes) -> EcatState:
    if len(buf) != STATE_SIZE:
        raise EcatError(f"상태 {len(buf)} B ≠ {STATE_SIZE}")
    v = _STATE.unpack(buf)
    if v[0] != STATE_MAGIC:
        raise EcatError(f"상태 magic 0x{v[0]:08x}")
    return EcatState(seq=v[1], t_ns=v[2], al_state=v[3], al_code=v[4], flags=v[5], wkc_bad=v[6], cycles=v[7],
                     rtt_max_us=v[8], late_max_us=v[9], inputs=tuple(v[10:10 + N_IN]), outputs=tuple(v[10 + N_IN:]))


def pack_state(**kw) -> bytes:
    """테스트 · 가짜 마스터용."""
    inputs = list(kw.get("inputs", [0] * N_IN))
    outputs = list(kw.get("outputs", [0] * N_OUT))
    return _STATE.pack(STATE_MAGIC, kw.get("seq", 0), kw.get("t_ns", 0), kw.get("al_state", 8), kw.get("al_code", 0),
                       kw.get("flags", FLAG_OP), kw.get("wkc_bad", 0), kw.get("cycles", 0), kw.get("rtt_max_us", 0),
                       kw.get("late_max_us", 0), *inputs, *outputs)


def pack_cmd(kind: int, values: Sequence[int] = (0, 0, 0, 0, 0, 0)) -> bytes:
    vals = [int(x) for x in values]
    if len(vals) != N:
        raise EcatError(f"명령 값 {len(vals)} 개 ≠ {N}")
    return _CMD.pack(CMD_MAGIC, int(kind), N, *vals)


def unpack_cmd(buf: bytes) -> tuple[int, list[int]]:
    if len(buf) != CMD_SIZE:
        raise EcatError(f"명령 {len(buf)} B ≠ {CMD_SIZE}")
    v = _CMD.unpack(buf)
    if v[0] != CMD_MAGIC:
        raise EcatError(f"명령 magic 0x{v[0]:08x}")
    return v[1], list(v[3:])


def clip_angle(values: Sequence[int]) -> list[int]:
    """-1 은 그대로(그 축 유지), 나머지는 매뉴얼 범위로."""
    out = []
    for i, v in enumerate(values):
        v = int(v)
        out.append(-1 if v < 0 else max(ANGLE_LO[i], min(ANGLE_HI[i], v)))
    return out


def clip_limit(values: Sequence[int], hi: int) -> list[int]:
    return [-1 if int(v) < 0 else max(0, min(hi, int(v))) for v in values]


@dataclass
class CommandBook:
    """ROS 명령 → 마스터 명령. -1 은 그 축 직전 목표(없으면 마스터가 지금 각도)로 둔다 — 마스터에도 같은 규칙이 있다."""
    target: list[int] = field(default_factory=lambda: [-1] * N)

    def angle(self, values: Sequence[int]) -> bytes:
        if len(values) != N:
            raise EcatError(f"각도 {len(values)} 개 ≠ {N}")
        new = clip_angle(values)
        self.target = [n if n >= 0 else t for n, t in zip(new, self.target)]
        return pack_cmd(CMD_ANGLE, new)

    def angle_target(self, values: Sequence[int]) -> bytes:
        """/hand_<s>/angle_target: 같은 각도 목표 + 마스터의 손가락별 어드민턴스(rh56f1_admittance.h)."""
        if len(values) != N:
            raise EcatError(f"각도 {len(values)} 개 ≠ {N}")
        new = clip_angle(values)
        self.target = [n if n >= 0 else t for n, t in zip(new, self.target)]
        return pack_cmd(CMD_ANGLE_ADM, new)

    @staticmethod
    def force(values: Sequence[int]) -> bytes:
        if len(values) != N:
            raise EcatError(f"힘 {len(values)} 개 ≠ {N}")
        return pack_cmd(CMD_FORCE, clip_limit(values, FORCE_MAX))

    @staticmethod
    def speed(values: Sequence[int]) -> bytes:
        if len(values) != N:
            raise EcatError(f"속도 {len(values)} 개 ≠ {N}")
        return pack_cmd(CMD_SPEED, clip_limit(values, SPEED_MAX))

    @staticmethod
    def mode(values: Sequence[int]) -> bytes:
        """손가락 동작 모드(0 위치 · 1 힘 폐루프), -1 = 그대로. 마스터의 SDO 스레드가 OP 중에 쓴다."""
        if len(values) != N:
            raise EcatError(f"모드 {len(values)} 개 ≠ {N}")
        if not all(int(v) in (-1, 0, 1, 2) for v in values):
            raise EcatError(f"모드는 -1 · 0 · 1 · 2: {list(values)}")
        return pack_cmd(CMD_MODE, [int(v) for v in values])

    @staticmethod
    def heartbeat() -> bytes:
        return pack_cmd(CMD_HEARTBEAT)


def sample_time_ns(ros_now_ns: int, mono_now_ns: int, sample_mono_ns: int) -> int:
    """마스터가 PDO 를 받은 순간(CLOCK_MONOTONIC)을 ROS 시계로 — 10.03 bag 정렬용(노드가 받은 시각이 아니라 하드웨어 샘플 시각).
    마스터 · 노드는 같은 호스트의 같은 단조 시계라 차이만 빼면 된다. 미래 시각이면(시계 이상) 지금으로."""
    age = mono_now_ns - int(sample_mono_ns)
    return ros_now_ns - age if age >= 0 else ros_now_ns


def joint_names(side: str) -> list[str]:
    p = "r" if side == "right" else "l"
    return [f"{p}_hj_{f}" for f in SLOT_FINGERS]


CURRENT_LIMIT_RANGE = (100, 1500)   # mA, 매뉴얼 2.5.7 (100 미만은 빈손 동작도 세운다 — 10.06 빈손 ≤ 320 mA)
FINGER_MODES = (0, 1, 2)            # 매뉴얼 2.5.21: 0 속도 · 힘 보호 · 1 힘 폐루프 · 2 임피던스


def _six(value, name: str, valid) -> list[int]:
    vals = [int(value)] * 6 if isinstance(value, (int, float)) else [int(v) for v in value]
    if len(vals) != 6 or not all(v == -1 or valid(v) for v in vals):
        raise EcatError(f"{name}: 값 하나 또는 여섯 개(새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전, -1 = 안 씀)")
    return vals


def protection_argv(cfg: dict) -> list[str]:
    """손 보호 설정(SDO 0x2000, 마스터가 OP 전에 쓰고 다시 읽어 확인) — 10.06 컵 쥐기 뒤 손 먹통 대책."""
    argv: list[str] = []
    if bool(cfg.get("clear_error", False)):
        argv.append("--clear-error")
    if bool(cfg.get("force_calibrate", False)):   # 빈손에서만 — 6 s 동안 손가락이 움직인다(매뉴얼 2.5.6)
        argv.append("--force-calibrate")
    if cfg.get("current_limit_ma") is not None:
        lo, hi = CURRENT_LIMIT_RANGE
        vals = _six(cfg["current_limit_ma"], "current_limit_ma", lambda v: lo <= v <= hi)
        argv += ["--current-limit", ",".join(map(str, vals))]
    if cfg.get("finger_mode") is not None:
        vals = _six(cfg["finger_mode"], "finger_mode", lambda v: v in FINGER_MODES)
        argv += ["--finger-mode", ",".join(map(str, vals))]
    return argv


MODE_PREFIX = "[master] MODE "


def parse_mode_line(line: str) -> dict | None:
    """마스터가 모드를 쓴 뒤 찍는 한 줄 → {"finger_mode": [6], "ms": .., "ok": ..} (아니면 None)."""
    import json
    i = line.find(MODE_PREFIX + "{")
    if i < 0:
        return None
    try:
        return json.loads(line[i + len(MODE_PREFIX):])
    except ValueError:
        return None


SDO_PREFIX = "[master] SDO "


def parse_sdo_line(line: str) -> dict | None:
    """마스터가 시작마다 찍는 손 보호 설정 한 줄 → dict (아니면 None)."""
    import json
    i = line.find(SDO_PREFIX + "{")
    if i < 0:
        return None
    try:
        return json.loads(line[i + len(SDO_PREFIX):])
    except ValueError:
        return None


def master_argv(binary: str, ifname: str, master_sock: str, node_sock: str, cfg: dict, no_op: bool) -> list[str]:
    """마스터 실행 인자. cfg = rh56f1_ports.yaml 의 ethercat 블록."""
    argv = [binary, "--ifname", ifname, "--master-sock", master_sock, "--node-sock", node_sock,
            "--hz", str(float(cfg.get("cycle_hz", 1000))), "--state-hz", str(float(cfg.get("state_hz", 100))),
            "--speed", str(int(cfg.get("speed", 2000))), "--force", str(int(cfg.get("force", 600))),
            "--enable-value", str(int(cfg.get("enable_value", 1))),
            "--hb-timeout-ms", str(int(cfg.get("hb_timeout_ms", 500))),
            "--op-timeout-ms", str(int(cfg.get("op_timeout_ms", 3000)))]
    if bool(cfg.get("op_enable", False)):
        argv.append("--op-enable")
    if float(cfg.get("cycle_hz_op", 0) or 0) > 0:
        if not float(cfg.get("state_hz", 100)) <= float(cfg["cycle_hz_op"]) <= 4000:
            raise EcatError("cycle_hz_op 는 state_hz 이상 4000 이하")
        argv += ["--hz-op", str(float(cfg["cycle_hz_op"]))]
    if int(cfg.get("sync_type", -1)) >= 0:
        argv += ["--sync-type", str(int(cfg["sync_type"]))]
    argv += protection_argv(cfg)
    adm = cfg.get("admittance")
    if adm is not None and bool(adm.get("enabled", True)):
        from policy_control.rh56f1_admittance import AdmParams
        try:
            argv += ["--adm", AdmParams.from_cfg(adm).argv()]
        except ValueError as e:
            raise EcatError(str(e)) from e
    if not 50 <= float(cfg.get("cycle_hz", 1000)) <= 4000:
        raise EcatError("cycle_hz 는 50~4000")
    if not 0 < float(cfg.get("state_hz", 100)) <= float(cfg.get("cycle_hz", 1000)):
        raise EcatError("state_hz 는 0 초과 cycle_hz 이하")
    for key in ("cycle_hz", "cycle_hz_op"):          # ★10.03 실측: 750 · 1000 Hz 에서 손 MCU 가 OP 요청 · 명령을 읽지 못한다(조용히 무시)
        if float(cfg.get(key, 0) or 0) > MAX_CYCLE_HZ:
            raise EcatError(f"{key} {float(cfg[key]):g} > {MAX_CYCLE_HZ:g} Hz — 이 손은 그보다 빠르면 명령을 무시한다(10.03 실측)")
    for key in ("cycle_hz", "cycle_hz_op"):          # 마스터는 N 주기마다 상태를 보낸다 — 안 나눠떨어지면 주기가 틀어진다(10.03)
        hz = float(cfg.get(key, 0) or 0)
        if hz > 0 and abs(hz / float(cfg.get("state_hz", 100)) - round(hz / float(cfg.get("state_hz", 100)))) > 1e-9:
            raise EcatError(f"{key} {hz:g} 가 state_hz {float(cfg.get('state_hz', 100)):g} 로 나눠떨어지지 않는다")
    if not 0 <= int(cfg.get("speed", 2000)) <= SPEED_MAX or not 0 <= int(cfg.get("force", 600)) <= FORCE_MAX:
        raise EcatError(f"speed 0~{SPEED_MAX} · force 0~{FORCE_MAX}")
    return argv + (["--no-op"] if no_op else [])
