"""10.12 어드민턴스 빈손 힘 기준선 — 마스터가 force_actual 에서 각도별 b 를 빼고(쉼 값도 뺀 힘으로) 쓴다."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_control import rh56f1_ecat as E

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "tools/ethercat/rh56f1_ecat_master.c"
HEADER = REPO / "tools/ethercat/rh56f1_admittance.h"
TABLE = REPO / "deploy/policy_control/config/rh56f1_adm_baseline.yaml"
PORTS = REPO / "deploy/policy_control/config/rh56f1_ports.yaml"


def _table():
    return yaml.safe_load(TABLE.read_text())


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_committed_table_is_zero_open_and_grows_when_closing(side):
    t = _table()
    k = np.asarray(t["knots"], float)
    g = np.asarray(t[side]["slots"], float)
    assert g.shape == (6, k.size) and np.all(np.diff(k) > 0)
    assert np.all(g[:, k >= 1725] == 0.0)                         # 편 손 = 0(쉼 값이 편 손에서 잡혀도 이중으로 안 뺀다)
    assert np.all(np.diff(g, axis=1) <= 1e-9)                      # 열수록(레지스터 ↑) 작거나 같다
    assert np.all(g[4:] == 0.0)                                    # 엄지굽힘 · 회전은 빼지 않는다(근거 부족 · 어드민턴스 끔)
    assert 140.0 <= g[:4, 0].min() and g[:4, 0].max() <= 240.0     # 빈손 주먹 145~236 g 범위


def test_the_ports_file_turns_it_on_for_both_hands():
    eth = yaml.safe_load(PORTS.read_text())["ethercat"]
    assert eth["adm_baseline"] == str(TABLE.relative_to(REPO))
    for side in ("right", "left"):
        argv = E.master_argv("/m", "eth0", "/a", "/b", eth, no_op=False, side=side)
        val = argv[argv.index("--adm-base") + 1].split(",")
        n = int(val[0])
        assert len(val) == 1 + 7 * n
    assert "--adm-base" not in E.master_argv("/m", "eth0", "/a", "/b", eth, no_op=False)          # side 모르면 안 붙인다
    assert "--adm-base" not in E.master_argv("/m", "eth0", "/a", "/b", {k: v for k, v in eth.items() if k != "adm_baseline"},
                                             no_op=False, side="right")                           # 줄을 지우면 10.12 전


@pytest.mark.parametrize("bad", [
    {"knots": [900, 900], "right": {"slots": [[0, 0]] * 6}},
    {"knots": [900, 1000], "right": {"slots": [[0, 0]] * 5}},
    {"knots": [900, 1000], "right": {"slots": [[0, -1]] * 6}},
    {"knots": [900], "right": {"slots": [[0]] * 6}},
])
def test_a_bad_table_is_refused(tmp_path, bad):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(E.EcatError):
        E.adm_base_arg(p, "right")


def test_the_master_subtracts_the_baseline_before_the_bias_and_the_law():
    src = MASTER.read_text()
    i = src.index("static void control(")
    body = src[i:src.index("\n}\n", i)]
    sub = body.index("adm_base_g(&g_adm_base, i, in[IN_ANGLE + i])")
    assert sub < body.index("bias[i] +=") and sub < body.index("adm_step(")                   # 쉼 값 · 법칙 모두 뺀 힘
    assert '{"adm-base", 1, 0, \'B\'}' in src and "parse_adm_base(optarg, &g_adm_base)" in src


def test_the_c_interpolation_matches_numpy(tmp_path):
    if shutil.which("gcc") is None:
        pytest.skip("gcc 없음")
    t = _table()
    k = np.asarray(t["knots"], float)
    g = np.asarray(t["right"]["slots"], float)
    harness = tmp_path / "h.c"
    init = ",".join(f"{v:g}" for v in k)
    rows = ",".join("{" + ",".join(f"{v:g}" for v in r) + "}" for r in g)
    harness.write_text('#include <stdio.h>\n#include "rh56f1_admittance.h"\n'
                       f"int main(void){{adm_base_t b={{{k.size},{{{init}}},{{{rows}}}}}; double r; int i;\n"
                       'while(scanf("%d %lf",&i,&r)==2) printf("%.6f\\n", adm_base_g(&b,i,r)); return 0;}\n')
    exe = tmp_path / "h"
    subprocess.run(["gcc", "-O2", "-o", str(exe), str(harness), "-I", str(HEADER.parent), "-lm"], check=True)
    q = [(i, r) for i in range(6) for r in (850, 900, 912.5, 1001, 1333.3, 1700, 1740, 1800)]
    out = subprocess.run([str(exe)], input="\n".join(f"{i} {r}" for i, r in q), capture_output=True, text=True, check=True)
    got = np.array([float(x) for x in out.stdout.split()])
    want = np.array([np.interp(r, k, g[i]) for i, r in q])
    assert got == pytest.approx(want, abs=1e-6)
    empty = tmp_path / "e.c"
    empty.write_text('#include <stdio.h>\n#include "rh56f1_admittance.h"\nint main(void){adm_base_t b={0}; '
                     'printf("%g\\n", adm_base_g(&b,2,950)); return 0;}\n')
    subprocess.run(["gcc", "-o", str(tmp_path / "e"), str(empty), "-I", str(HEADER.parent), "-lm"], check=True)
    assert subprocess.run([str(tmp_path / "e")], capture_output=True, text=True).stdout.strip() == "0"   # 표 없음 = 10.12 전


def _parser_harness(tmp_path):
    """마스터의 parse_adm_base 를 떼어 헤더와 같이 빌드 — SOEM 없이 인자 해석만 본다."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc 없음")
    src = MASTER.read_text()
    i = src.index("static int parse_adm_base(")
    fn = src[i:src.index("\n}\n", i) + 3]
    h = tmp_path / "p.c"
    h.write_text('#include <stdio.h>\n#include <stdlib.h>\n#include "rh56f1_admittance.h"\n' + fn +
                 "int main(int c, char **v){adm_base_t b={0}; if(!parse_adm_base(v[1],&b)){puts(\"BAD\");return 0;}"
                 'printf("%d %g %g %g\\n", b.n, b.knot[b.n-1], b.g[0][0], b.g[5][b.n-1]); return 0;}\n')
    exe = tmp_path / "p"
    subprocess.run(["gcc", "-O2", "-Wall", "-o", str(exe), str(h), "-I", str(HEADER.parent), "-lm"], check=True)
    return lambda arg: subprocess.run([str(exe), arg], capture_output=True, text=True, check=True).stdout.strip()


def test_the_master_parses_the_launcher_string_and_refuses_bad_ones(tmp_path):
    parse = _parser_harness(tmp_path)
    arg = E.adm_base_arg(TABLE, "right")
    t = _table()
    n, last, g00, g5 = parse(arg).split()
    assert int(n) == len(t["knots"]) and float(last) == t["knots"][-1]
    assert float(g00) == pytest.approx(t["right"]["slots"][0][0]) and float(g5) == t["right"]["slots"][5][-1]
    for bad in ["2,900,900," + ",".join(["0"] * 12),                   # 매듭 오름차순 아님
                "2,900,1000," + ",".join(["0"] * 11),                  # 값 개수 모자람
                "2,900,1000," + ",".join(["0"] * 11 + ["2500"]),       # 2000 g 넘음
                "1,900," + ",".join(["0"] * 6),                        # 매듭 1 개
                "2,900,1000," + ",".join(["0"] * 12) + ",",            # 끝 쉼표
                "2,900,1000,x" + ",0" * 11]:
        assert parse(bad) == "BAD", bad


def test_an_old_master_build_runs_without_the_baseline_instead_of_dying(tmp_path, capsys):
    cfg = {k: v for k, v in yaml.safe_load(PORTS.read_text())["ethercat"].items() if k != "adm_fscale"}   # 10.13 배율은 옛 빌드면 거부(아래)
    old = tmp_path / "old_master"
    old.write_bytes(b"\x7fELF...adm\0current-limit\0")
    new = tmp_path / "new_master"
    new.write_bytes(b"\x7fELF...adm\0adm-base\0current-limit\0")
    a_old = E.master_argv(str(old), "eth0", "/a", "/b", cfg, no_op=False, side="right")
    a_new = E.master_argv(str(new), "eth0", "/a", "/b", cfg, no_op=False, side="right")
    assert "--adm-base" not in a_old and "--adm-base" in a_new
    assert "옛 빌드" in capsys.readouterr().out


# ★10.13 손가락별 힘 입력 배율(--adm-fscale) ---------------------------------------------------------------------------

def _fscale_harness(tmp_path, scale, guard, cases):
    """헤더 adm_fscale_g 를 C 로 돌려 (슬롯, 원시 c) → 출력. f_max 800 · deadband 40(계약 기본)."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc 없음")
    src = tmp_path / "fs.c"
    rows = "".join(f'printf("%.6f\\n", adm_fscale_g(&q, &p, {i}, {c}));\n' for i, c in cases)
    src.write_text('#include <stdio.h>\n#include "rh56f1_admittance.h"\nint main(void){adm_params_t p = ADM_DEFAULTS; '
                   f'adm_fscale_t q = {{{{{",".join(map(str, scale))}}}, {guard}}};\n{rows}return 0;}}\n')
    exe = tmp_path / "fs"
    subprocess.run(["gcc", "-O2", "-Wall", "-o", str(exe), str(src), "-I", str(HEADER.parent), "-lm"], check=True)
    return [float(x) for x in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()]


def test_fscale_one_is_identity_and_scale_only_discounts(tmp_path):
    out = _fscale_harness(tmp_path, [1, 1, 0.45, 0.4, 0.7, 1], 1000,
                          [(0, 500.0), (3, 100.0), (3, 500.0), (3, -20.0), (2, 300.0), (4, 300.0)])
    assert out[0] == pytest.approx(500.0)                  # 배율 1 → 그대로(10.13 전 동작)
    assert out[1] == pytest.approx(40.0)                   # 검지 0.4
    assert out[2] == pytest.approx(200.0)
    assert out[3] == pytest.approx(-8.0)                   # 음수(쉼 값 아래)도 같은 배율 — 쉼 값이 같은 단위로 잡힌다
    assert out[4] == pytest.approx(135.0) and out[5] == pytest.approx(210.0)


def test_fscale_guard_restores_f_max_at_raw_guard_and_never_exceeds_raw(tmp_path):
    cs = [600.0, 750.0, 900.0, 1000.0, 1100.0, 1300.0, 2000.0]
    out = _fscale_harness(tmp_path, [1, 1, 1, 0.4, 1, 1], 1000, [(3, c) for c in cs])
    by = dict(zip(cs, out))
    assert by[600.0] == pytest.approx(240.0)               # 문턱에서 먼 곳은 할인 그대로
    assert by[1000.0] == pytest.approx(840.0)              # 원시 1000 g → f_max 800 + deadband 40(빨리 여는 구간 시작)
    assert all(o <= c + 1e-9 for c, o in by.items())       # 원시 힘을 넘지 않는다
    assert all(b >= a for a, b in zip(out, out[1:]))       # 단조 — 힘이 늘면 입력도 는다(출렁임 없음)
    assert by[2000.0] == pytest.approx(2000.0)             # 멀리 넘으면 원시 그대로
    off = _fscale_harness(tmp_path, [1, 1, 1, 0.4, 1, 1], 0, [(3, 1500.0)])
    assert off[0] == pytest.approx(600.0)                  # guard 0 = 끔


def test_master_parses_fscale_and_rejects_bad_values(tmp_path):
    src = MASTER.read_text()
    assert '{"adm-fscale", 1, 0, \'F\'}' in src and "parse_adm_fscale(optarg, &g_adm_fscale)" in src
    i = src.index("static void control(")
    body = src[i:src.index("static void request_modes", i)]
    assert body.index("adm_fscale_g(") < body.index("g_adm_state.bias[i] +=")   # 쉼 값도 배율 뒤 단위로
    if shutil.which("gcc") is None:
        pytest.skip("gcc 없음")
    s = src.index("static int parse_adm_fscale")
    fn = src[s:src.index("static void usage", s)]
    h = tmp_path / "p.c"
    h.write_text('#include <stdio.h>\n#include <stdlib.h>\n#include "rh56f1_admittance.h"\n' + fn +
                 'int main(int c, char **v){adm_fscale_t f; if(!parse_adm_fscale(v[1], &f)) {puts("BAD"); return 0;} '
                 'printf("%g %g %g\\n", f.scale[3], f.scale[0], f.guard_g); return 0;}\n')
    exe = tmp_path / "p"
    subprocess.run(["gcc", "-O2", "-Wall", "-o", str(exe), str(h), "-I", str(HEADER.parent), "-lm"], check=True)
    run = lambda a: subprocess.run([str(exe), a], capture_output=True, text=True, check=True).stdout.strip()
    assert run("1,1,0.45,0.4,0.7,1,1000") == "0.4 1 1000"
    assert run("1,1,1,1,1,1,0") == "1 1 0"
    for bad in ("1,1,1,0,1,1,1000", "1,1,1,1.2,1,1,1000", "1,1,1,1,1,1,100", "1,1,1,1,1,1", "1,1,1,1,1,1,1000,", "a,1,1,1,1,1,0"):
        assert run(bad) == "BAD", bad


def test_ports_fscale_reaches_master_per_side_and_old_binary_is_refused(tmp_path):
    eth = yaml.safe_load(PORTS.read_text())["ethercat"]
    assert eth["adm_fscale"]["right"][3] < 1.0             # 실험 값이 실려 있다(지우면 10.13 전 동작)
    new = tmp_path / "new_master"
    new.write_bytes(b"\x7fELF...adm-base\0...adm-fscale\0")
    for side in ("right", "left"):
        argv = E.master_argv(str(new), "eth0", "/a", "/b", eth, no_op=False, side=side)
        val = argv[argv.index("--adm-fscale") + 1].split(",")
        assert [float(v) for v in val[:6]] == [float(v) for v in eth["adm_fscale"][side]]
        assert float(val[6]) == float(eth["adm_fscale"]["guard_g"])
    assert "--adm-fscale" not in E.master_argv(str(new), "eth0", "/a", "/b", eth, no_op=False)      # side 모르면 안 붙인다
    old = tmp_path / "old_master"
    old.write_bytes(b"\x7fELF...adm-base\0")
    with pytest.raises(E.EcatError, match="adm-fscale"):
        E.master_argv(str(old), "eth0", "/a", "/b", eth, no_op=False, side="right")
    bad = {**eth, "adm_fscale": {"guard_g": 1000, "right": [1, 1, 1, 1.5, 1, 1]}}
    with pytest.raises(E.EcatError):
        E.master_argv(str(new), "eth0", "/a", "/b", bad, no_op=False, side="right")
