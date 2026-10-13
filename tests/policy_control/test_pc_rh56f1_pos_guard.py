"""10.14 위치(angle_set) 축 보호 — 원시 힘 1,300 g · 전류 750 mA 를 0.3 s 넘게 넘으면 그 손가락을 지금 각도에 고정."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from policy_control import rh56f1_ecat as E

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "tools/ethercat/rh56f1_ecat_master.c"
HEADER = REPO / "tools/ethercat/rh56f1_pos_guard.h"
PORTS = REPO / "deploy/policy_control/config/rh56f1_ports.yaml"
DT = 0.001   # 마스터 1 kHz


def _gcc():
    if shutil.which("gcc") is None:
        pytest.skip("gcc 없음")


def _step_harness(tmp_path, force_g, current_ma, hold_s):
    """줄마다 'target actual force current' 를 읽어 'sent tripped pinned' 를 쓴다(슬롯 3, dt 1 ms)."""
    _gcc()
    h = tmp_path / "g.c"
    h.write_text(
        '#include <stdio.h>\n#include "rh56f1_pos_guard.h"\n'
        "int main(void){pos_guard_params_t p={%g,%g,%g}; pos_guard_state_t s={{0}}; int t,a; double f,c;\n"
        ' while(scanf("%%d %%d %%lf %%lf",&t,&a,&f,&c)==4){int tr=0; int o=pos_guard_step(&p,&s,3,%g,t,a,f,c,&tr);'
        ' printf("%%d %%d %%d\\n",o,tr,s.pinned[3]);} return 0;}\n' % (force_g, current_ma, hold_s, DT))
    exe = tmp_path / "g"
    subprocess.run(["gcc", "-O2", "-Wall", "-Werror", "-o", str(exe), str(h), "-I", str(HEADER.parent)], check=True)

    def run(rows):
        txt = "".join(f"{t} {a} {f} {c}\n" for t, a, f, c in rows)
        out = subprocess.run([str(exe)], input=txt, capture_output=True, text=True, check=True).stdout.split("\n")
        return [tuple(int(v) for v in line.split()) for line in out if line]
    return run


def test_a_short_spike_passes_but_a_sustained_squeeze_pins_the_finger_where_it_is(tmp_path):
    run = _step_harness(tmp_path, 1300, 750, 0.3)
    spike = [(1000, 1300, 1500, 900)] * 299 + [(1000, 1300, 400, 200)] * 5    # 0.299 s — 충돌 순간
    assert all(o == 1000 and tr == 0 for o, tr, _ in run(spike))
    out = run([(1000, 1300, 1400, 100)] * 400)                                # 힘만 0.4 s
    k = next(i for i, r in enumerate(out) if r[1] == 1)
    assert 299 <= k <= 300                                                      # 0.3 s 에 한 번
    assert all(o == 1000 for o, _, _ in out[:k]) and all(o == 1300 and p == 1 for o, _, p in out[k:])
    assert sum(tr for _, tr, _ in out) == 1


def test_current_alone_also_pins(tmp_path):
    run = _step_harness(tmp_path, 1300, 750, 0.3)
    out = run([(1000, 1250, 200, 800)] * 350)
    assert out[-1] == (1250, 0, 1)


def test_pinned_finger_never_closes_further_but_opens_when_the_target_opens(tmp_path):
    run = _step_harness(tmp_path, 1300, 750, 0.3)
    rows = [(1000, 1300, 1400, 0)] * 310                 # 고정 1300
    rows += [(950, 1300, 1400, 0)] * 50                  # 더 닫으라 해도 1300
    rows += [(1300, 1300, 1400, 0)] * 5                  # 고정 각도 그대로도 1300
    rows += [(1500, 1300, 1400, 0)]                      # 여는 쪽 → 풀린다
    rows += [(1500, 1350, 300, 0)] * 10
    out = run(rows)
    assert all(o == 1300 for o, _, _ in out[300:365])
    assert out[365] == (1500, 0, 0) and all(o == 1500 and p == 0 for o, _, p in out[366:])


def test_the_push_side_is_remembered_for_an_axis_that_closes_upward(tmp_path):
    run = _step_harness(tmp_path, 0, 750, 0.3)            # 힘 문턱 끔 · 레지스터 ↑ 로 민 경우
    out = run([(1700, 1400, 0, 900)] * 310 + [(1750, 1400, 0, 900)] * 5 + [(1200, 1400, 0, 0)])
    assert out[305][0] == 1400 and out[312][0] == 1400   # 민 쪽(↑)은 막힌다
    assert out[-1] == (1200, 0, 0)                       # 반대쪽(↓)이면 풀린다


def test_both_thresholds_zero_is_the_pre_10_14_behaviour(tmp_path):
    run = _step_harness(tmp_path, 0, 0, 0.3)
    assert all(o == 900 and p == 0 for o, _, p in run([(900, 1300, 3000, 1500)] * 500))


def test_the_master_guards_only_angle_set_axes_and_resets_before_commands():
    src = MASTER.read_text()
    i = src.index("static void control(")
    body = src[i:src.index("\n}\n", i)]
    hold = body.index("if (!c->commanded)")
    assert body.index("pos_guard_reset(&g_pos_guard_state, i)", hold) < body.index("continue", hold)
    pos = body.index("if (!c->adm_on[i])")
    assert pos < body.index("pos_guard_step(") < body.index("adm_step(")
    assert "(double)in[IN_FORCE + i], (double)in[IN_CURRENT + i]" in body   # 원시 힘(기준선 · 배율 전)
    assert '{"pos-guard", 1, 0, \'P\'}' in src and "parse_pos_guard(optarg, &g_pos_guard)" in src


def test_the_master_parses_the_launcher_string_and_refuses_bad_ones(tmp_path):
    _gcc()
    src = MASTER.read_text()
    s = src.index("static int parse_pos_guard(")
    fn = src[s:src.index("\n}\n", s) + 3]
    h = tmp_path / "p.c"
    h.write_text('#include <stdio.h>\n#include <stdlib.h>\n#include "rh56f1_pos_guard.h"\n' + fn +
                 'int main(int c, char **v){pos_guard_params_t g={0}; if(!parse_pos_guard(v[1], &g)) {puts("BAD"); return 0;} '
                 'printf("%g %g %g\\n", g.force_g, g.current_ma, g.hold_s); return 0;}\n')
    exe = tmp_path / "p"
    subprocess.run(["gcc", "-O2", "-Wall", "-o", str(exe), str(h), "-I", str(HEADER.parent)], check=True)
    run = lambda a: subprocess.run([str(exe), a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    eth = yaml.safe_load(PORTS.read_text())["ethercat"]
    assert run(E.pos_guard_arg(eth["pos_guard"])) == "1300 750 0.3"
    assert run("0,0,0.3") == "0 0 0.3"
    for bad in ("100,750,0.3", "1300,50,0.3", "1300,750,0", "1300,750,3", "1300,750", "1300,750,0.3,", "a,750,0.3"):
        assert run(bad) == "BAD", bad


def test_ports_turn_it_on_and_an_old_binary_is_refused(tmp_path):
    eth = yaml.safe_load(PORTS.read_text())["ethercat"]
    assert eth["pos_guard"] == {"force_g": 1300, "current_ma": 750, "hold_s": 0.3}   # 사용자 10.14 값
    new = tmp_path / "new_master"
    new.write_bytes(b"\x7fELF...adm-base\0...adm-fscale\0...pos-guard\0")
    for side in ("right", "left", None):
        argv = E.master_argv(str(new), "eth0", "/a", "/b", eth, no_op=False, side=side)
        assert argv[argv.index("--pos-guard") + 1] == "1300,750,0.3"
    off = {k: v for k, v in eth.items() if k != "pos_guard"}
    assert "--pos-guard" not in E.master_argv(str(new), "eth0", "/a", "/b", off, no_op=False, side="right")
    old = tmp_path / "old_master"
    old.write_bytes(b"\x7fELF...adm-base\0...adm-fscale\0")
    with pytest.raises(E.EcatError, match="pos-guard"):
        E.master_argv(str(old), "eth0", "/a", "/b", eth, no_op=False, side="right")


@pytest.mark.parametrize("bad", [{"force_g": 100}, {"current_ma": 2000}, {"force_g": 1300, "hold_s": 0}, {"force_g": float("nan")},
                                 [1300, 750]])
def test_a_bad_block_is_refused(bad):
    with pytest.raises(E.EcatError):
        E.pos_guard_arg(bad)
