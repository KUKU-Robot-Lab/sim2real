"""RH56F1 EtherCAT(10.02 사용자: RS485 대신) — 마스터 형식 · PDO 해석 · 명령 합치기 · 소켓 연결. 실기 없이."""
from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest
import yaml

from policy_control import rh56f1_ecat as E
from policy_control import rh56f1_ecat_node as NODE

PC = Path(__file__).resolve().parents[2] / "deploy" / "policy_control"
MASTER_C = Path(__file__).resolve().parents[2] / "tools" / "ethercat" / "rh56f1_ecat_master.c"

#: 10.02 arm4090 오른손 SAFE_OP 실측 첫 입력(ecat_rh56f1 safeop)
RIGHT_INPUTS = ([142, 138, 102, 110, 106, 1234] + [1744, 1741, 1743, 1744, 1350, 948] + [45, -3, 18, 27, -8, -36]
                + [0] * 6 + [0] * 6 + [2] * 6 + [42, 42, 42, 42, 38, 40]
                + [0, 0, -1, 0, 5, 0, 0, 0, 0, 2, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]
                + [0, 0, -1, 0, 0, -1, 0, 0, -1])


def test_struct_sizes_match_the_c_master():
    """C 구조체(#pragma pack 1)와 바이트가 같아야 한다 — 필드를 바꾸면 양쪽을 같이."""
    src = MASTER_C.read_text()
    assert "int16_t in[N_IN];" in src and "int16_t out[N_OUT];" in src and "int32_t v[6];" in src
    assert E.STATE_SIZE == 4 + 4 + 8 + 2 * 4 + 4 + 2 * 2 + 2 * E.N_IN + 2 * E.N_OUT == 222
    assert E.CMD_SIZE == 4 + 2 + 2 + 4 * 6 == 32
    assert f"#define N_IN {E.N_IN}" in src and f"#define N_OUT {E.N_OUT}" in src
    assert "0x31534852u" in src and "0x31434852u" in src
    assert "sizeof(state_msg) == 222" in src and "sizeof(cmd_msg) == 32" in src     # 컴파일 때 C 쪽이 확인
    for lo, hi in zip(E.ANGLE_LO, E.ANGLE_HI):
        assert str(lo) in src and str(hi) in src


def test_state_round_trip_decodes_the_manual_table_50_layout():
    buf = E.pack_state(seq=7, flags=E.FLAG_OP | E.FLAG_COMMANDED, inputs=RIGHT_INPUTS, rtt_max_us=84)
    s = E.unpack_state(buf)
    assert s.seq == 7 and s.is_op and s.commanded and s.rtt_max_us == 84
    assert s.angle == [1744, 1741, 1743, 1744, 1350, 948]
    assert s.force == [45, -3, 18, 27, -8, -36]
    assert s.status == [2] * 6 and s.temperature == [42, 42, 42, 42, 38, 40]
    t = s.touch()
    assert set(t) == {"finger_forces", "finger_tangentials", "finger_angles", "finger_proximity", "palm_data"}
    assert all(len(t[k]) == 5 for k in ("finger_forces", "finger_tangentials", "finger_angles", "finger_proximity"))
    assert len(t["palm_data"]) == 9
    assert t["finger_angles"][0] == 0xFFFF                        # 방향 -1(INT16) = 65535 '접촉 없음'
    assert t["finger_forces"][0] == 0 and t["finger_forces"][1] == 0
    assert s.summary()["status_text"][0] == "위치 도달 정지"


def test_proximity_is_two_unsigned_halves():
    inp = [0] * E.N_IN
    inp[42 + 3], inp[42 + 4] = -1, 0x00F4                        # 손가락 1: L 0xFFFF · H 0x00F4
    assert E.unpack_state(E.pack_state(inputs=inp)).touch()["finger_proximity"][0] == (0xF4 << 16) | 0xFFFF


def test_bad_frames_are_refused():
    with pytest.raises(E.EcatError):
        E.unpack_state(b"x" * 10)
    bad = bytearray(E.pack_state())
    bad[0] ^= 0xFF
    with pytest.raises(E.EcatError, match="magic"):
        E.unpack_state(bytes(bad))
    with pytest.raises(E.EcatError):
        E.pack_cmd(E.CMD_ANGLE, [1, 2, 3])


def test_angle_commands_clip_to_the_manual_range_and_keep_minus_one():
    book = E.CommandBook()
    kind, vals = E.unpack_cmd(book.angle([2000, 100, -1, 1500, 1500, 500]))
    assert kind == E.CMD_ANGLE and vals == [1740, 900, -1, 1500, 1350, 600]
    assert book.target == [1740, 900, -1, 1500, 1350, 600]
    book.angle([-1, 1000, -1, -1, -1, -1])
    assert book.target == [1740, 1000, -1, 1500, 1350, 600]          # -1 축은 직전 목표


def test_force_and_speed_commands_clip_to_hardware_limits():
    assert E.unpack_cmd(E.CommandBook.force([600, 2000, -1, 0, 1000, -5]))[1] == [600, 1000, -1, 0, 1000, -1]
    assert E.unpack_cmd(E.CommandBook.speed([2000, 9000, -1, 0, 4000, 1]))[1] == [2000, 4000, -1, 0, 4000, 1]
    assert E.unpack_cmd(E.CommandBook.heartbeat())[0] == E.CMD_HEARTBEAT


def test_joint_names_match_the_vendor_driver_slot_order():
    assert E.joint_names("right") == ["r_hj_pinky_1", "r_hj_ring_1", "r_hj_middle_1", "r_hj_index_1",
                                      "r_hj_thumb_2", "r_hj_thumb_1"]
    hmap = yaml.safe_load((PC / "config" / "rh56f1_hand_map.yaml").read_text())
    assert list(E.SLOT_FINGERS) == hmap["slot_order"]


def test_port_file_gives_one_nic_per_hand_and_a_valid_master_command():
    ports = yaml.safe_load((PC / "config" / "rh56f1_ports.yaml").read_text())
    ifr, cfg = NODE.ecat_config(ports, "right")
    ifl, _ = NODE.ecat_config(ports, "left")
    assert (ifr, ifl) == ("enx00e04c6806e1", "enp6s0")
    argv = E.master_argv("/m", ifr, "/a", "/b", cfg, no_op=False)
    assert argv[:3] == ["/m", "--ifname", ifr] and "--no-op" not in argv
    assert argv[argv.index("--hz") + 1] == "500.0" and argv[argv.index("--speed") + 1] == "2000"
    assert E.master_argv("/m", ifr, "/a", "/b", cfg, no_op=True)[-1] == "--no-op"
    assert "--op-enable" not in argv and "--sync-type" not in argv                     # 실험 손잡이 기본 끔
    exp = E.master_argv("/m", ifr, "/a", "/b", dict(cfg, op_enable=True, sync_type=1), no_op=False)
    assert "--op-enable" in exp and exp[exp.index("--sync-type") + 1] == "1"
    assert "--hz-op" not in argv                                                          # 10.03 기본 꺼짐
    fast = E.master_argv("/m", ifr, "/a", "/b", dict(cfg, cycle_hz=250, cycle_hz_op=500), no_op=False)
    assert fast[fast.index("--hz-op") + 1] == "500.0"
    for k, v in (("cycle_hz_op", 1000), ("cycle_hz_op", 750), ("cycle_hz", 1000)):          # 10.03: 손이 명령을 무시
        with pytest.raises(E.EcatError, match="무시"):
            E.master_argv("/m", ifr, "/a", "/b", dict(cfg, **{k: v}), no_op=False)
    with pytest.raises(E.EcatError, match="cycle_hz_op"):
        E.master_argv("/m", ifr, "/a", "/b", dict(cfg, cycle_hz_op=50), no_op=False)
    with pytest.raises(E.EcatError, match="나눠떨어지지"):
        E.master_argv("/m", ifr, "/a", "/b", dict(cfg, state_hz=200), no_op=False)         # 500/200 → 166.7 Hz 였다

    with pytest.raises(E.EcatError, match="같은 NIC"):
        NODE.ecat_config({"right": ports["right"], "left": dict(ports["left"], ifname=ifr)}, "right")
    with pytest.raises(E.EcatError, match="transport"):
        NODE.ecat_config({"right": {"transport": "rs485"}}, "right")
    with pytest.raises(E.EcatError):
        E.master_argv("/m", ifr, "/a", "/b", dict(cfg, cycle_hz=10), no_op=False)
    with pytest.raises(E.EcatError):
        E.master_argv("/m", ifr, "/a", "/b", dict(cfg, force=5000), no_op=False)

def test_rates_line_up_for_high_speed_control():
    """10.03 사용자: EtherCAT 은 고속 제어가 목적 — 손 명령 · pd · 상태 발행이 정책(60 Hz)을 막지 않게."""
    from policy_control.pd_law import load_pd_config
    hmap = yaml.safe_load((PC / "config" / "rh56f1_hand_map.yaml").read_text())
    ports = yaml.safe_load((PC / "config" / "rh56f1_ports.yaml").read_text())
    for name in ("pd_rh56f1.yaml", "pd_rh56f1_exec.yaml", "pd_rh56f1_fake.yaml"):
        cfg = load_pd_config(PC / "config" / name)
        assert cfg.pd_hz == 120.0 and abs(cfg.settle.gain * cfg.pd_hz - 1.0) < 0.01     # 정착 시정수 유지
    assert hmap["command_max_hz"] >= 120.0                                # 손 명령이 pd 틱마다 나갈 수 있다
    assert ports["ethercat"]["state_hz"] >= 2 * 120.0 / 1.2              # 상태가 pd 보다 빠르다
    assert ports["ethercat"]["cycle_hz"] >= 4 * ports["ethercat"]["state_hz"] / 2


def test_master_link_exchanges_states_and_commands_with_a_fake_master(tmp_path):
    """노드 쪽 소켓 배선: 명령은 master.sock 으로, 상태는 node.sock 으로."""
    link = NODE.MasterLink("right", sock_dir=str(tmp_path))
    fake = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    fake.bind(link.master_sock)
    fake.settimeout(2.0)
    got = []

    def master():
        for _ in range(2):
            buf = fake.recv(64)
            got.append(E.unpack_cmd(buf))
        fake.sendto(E.pack_state(seq=3, inputs=RIGHT_INPUTS), link.node_sock)

    th = threading.Thread(target=master)
    th.start()
    assert link.send(E.CommandBook.heartbeat())
    assert link.send(E.CommandBook().angle([1500] * 6))
    s = None
    for _ in range(10):
        s = link.recv()
        if s is not None:
            break
    th.join()
    assert [k for k, _ in got] == [E.CMD_HEARTBEAT, E.CMD_ANGLE]
    assert got[1][1] == [1500, 1500, 1500, 1500, 1350, 1500]
    assert s is not None and s.seq == 3 and s.angle[0] == 1744
    fake.close()
    link.stop()
    assert not Path(link.node_sock).exists()


def test_send_without_a_master_counts_errors_instead_of_raising(tmp_path):
    link = NODE.MasterLink("left", sock_dir=str(tmp_path))
    assert not link.send(E.CommandBook.heartbeat())
    assert link.send_errors == 1
    link.stop()


def test_master_source_keeps_the_safety_rules():
    """실기 안전 규칙이 C 쪽에서 빠지지 않게 — 문자열로 지킨다(컴파일 · 실행은 arm4090)."""
    src = MASTER_C.read_text()
    assert "PR_SET_PDEATHSIG" in src                              # 노드가 죽으면 같이 끝난다
    assert "if (!c.commanded) hold(&c, in);" in src                # 첫 명령 전 · 노드 끊김 · 정지 = 제자리
    assert "out[OUT_ENABLE] = (int16_t)((c->commanded || c->hold_enable) ? c->enable_value : 0);" in src
    assert "int op_enable = 0, sync_type = -1" in src                # 실험 손잡이는 기본 끔
    assert "ECT_COEDET_SDOCA" in src and "EXPECT_ID 0x9252" in src
    assert "if (v < 0) v = c->commanded ? c->target[i] : in[IN_ANGLE + i];" in src
    # 10.02: 요청 값을 상태 변수에 덮지 않는다 · OP 는 한 번 요청하고 3 s 기다린다
    assert "ec_slave[1].state = EC_STATE_OPERATIONAL" not in src and "t0 - op_req_t > (uint64_t)op_timeout_ms * 1000000ull" in src


def test_sample_time_is_the_hardware_moment_on_the_ros_clock():
    """10.03 bag 정렬: 노드가 받은 시각이 아니라 마스터가 PDO 를 받은 시각."""
    assert E.sample_time_ns(ros_now_ns=10_000_000_000, mono_now_ns=500_000_000, sample_mono_ns=497_000_000) == 9_997_000_000
    assert E.sample_time_ns(10_000_000_000, 500, 900) == 10_000_000_000          # 미래(시계 이상)면 지금


def test_hand_protection_settings_reach_the_master():
    """10.06 왼손 컵 쥐기: 합 6.5 A 뒤 손 먹통 — 전류 보호 · 오류 지우기를 OP 전에 SDO 로 쓴다."""
    cfg = yaml.safe_load((PC / "config" / "rh56f1_ports.yaml").read_text())["ethercat"]
    argv = E.master_argv("/m", "eth0", "/a", "/b", cfg, no_op=False)
    assert "--clear-error" in argv
    assert argv[argv.index("--current-limit") + 1] == "800,800,800,800,800,800"
    assert "--finger-mode" not in argv  # 임피던스는 실기 확인 뒤
    six = E.protection_argv({"current_limit_ma": [700, 700, 700, 700, -1, 900], "finger_mode": 2})
    assert six == ["--current-limit", "700,700,700,700,-1,900", "--finger-mode", "2,2,2,2,2,2"]
    for bad in ({"current_limit_ma": 50}, {"current_limit_ma": 2000}, {"current_limit_ma": [800] * 5},
                {"finger_mode": 3}):
        with pytest.raises(E.EcatError):
            E.protection_argv(bad)


def test_master_reports_the_hand_protection_it_read():
    line = ('[INFO]: [master] SDO {"current_limit_ma": [800, 800, 800, 800, 800, 800], "finger_mode": [0, 0, 0, 0, 0, 0], '
            '"default_speed": [1000, null, 1000, 1000, 1000, 1000], "default_force_g": [500, 500, 500, 500, 500, 500]}')
    sdo = E.parse_sdo_line(line)
    assert sdo["current_limit_ma"] == [800] * 6 and sdo["default_speed"][1] is None
    assert E.parse_sdo_line("[master] OP") is None and E.parse_sdo_line("[master] SDO {broken") is None


def test_master_source_refuses_op_without_the_requested_protection():
    src = MASTER_C.read_text()
    assert "if (sdo_setup(clear_error, current_limit, finger_mode) != 0) goto out_ec;" in src
    assert src.index("sdo_setup(clear_error") < src.index("ec_config_map(&IOmap);")  # PREOP, OP 전
    for sub in ("SDO_CLEAR_ERROR 0x03", "SDO_CURRENT_LIMIT 0x07", "SDO_FINGER_MODE 0x1B"):
        assert sub in src
