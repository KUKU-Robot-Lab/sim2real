"""pd 보상 조건 비교 실행기(tilt_comp_ab) · 분석기 — 조건 yaml · 덮어쓰기 · 순서 · 단계 · 재생 줄 · 손끝 지표(합성 배열)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"
sys.path.insert(0, str(TOOLS))


def _load(name):
    s = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


ab = _load("tilt_comp_ab")
rep = _load("tilt_comp_ab_report")


def test_conditions_alternate_and_reject_bad_order():
    assert ab.conditions(6, "ABAB") == list("ABABAB")
    assert ab.conditions(3, "ab") == list("ABA")
    for bad in ((0, "AB"), (2, "AC"), (2, "")):
        with pytest.raises(ValueError):
            ab.conditions(*bad)


def test_default_conds_yaml_reproduces_old_ab(tmp_path):
    spec = ab.load_conds(ab.CONDS)
    assert spec["order"] == "AB" and list(spec["conditions"]) == ["A", "B"]
    a = ab.cycle_steps(spec["conditions"]["A"], tmp_path, ab.TRAJ, 1.0)
    b = ab.cycle_steps(spec["conditions"]["B"], tmp_path, ab.TRAJ, 1.0)
    assert [s.id for s in a] == [s.id for s in b]
    diff = [(x.id, set(y.argv) - set(x.argv)) for x, y in zip(a, b) if x != y]
    assert diff == [("tilt", {"--tilt-comp"})]
    ids = [s.id for s in a]
    # 차렷 → 홈 → tilt 시작(오른팔 → 왼팔, 계획 뒤 재생) → 정지 → tilt → 홈 → 차렷 순서
    order = ["fist_left", "to_home_right", "to_home_left", "plan_tilt_start_right", "to_tilt_start_right",
             "plan_tilt_start_left", "to_tilt_start_left", "pre_tilt_hold", "tilt", "back_left", "back_right", "to_rest_left"]
    assert [ids.index(k) for k in order] == sorted(ids.index(k) for k in order)
    assert all("--reverse" in s.argv for s in a if s.id.startswith(("back_", "to_rest_")))
    assert all("--execute" not in s.argv for s in a if s.id.startswith("plan_tilt_start_"))   # 계획 · 검사만, 재생은 따로


def test_replay_args_go_to_every_replay_by_default(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("conditions:\n  I: {replay_args: [--tilt-comp]}\n")
    steps = ab.cycle_steps(ab.load_conds(y)["conditions"]["I"], tmp_path, ab.TRAJ, 1.0)
    replays = [s for s in steps if s.argv and s.argv[1].endswith("replay_to_pd.py")]
    assert len(replays) == 9 and all("--tilt-comp" in s.argv for s in replays)      # home · tilt_start · tilt · back · rest
    assert not any("--tilt-comp" in s.argv for s in steps if s not in replays)


@pytest.mark.parametrize("body, why", [
    ("conditions: {}", "비었다"),
    ("conditions: {A: {foo: 1}}", "모르는 키"),
    ("conditions: {A: {replay_args: [--reverse]}}", "이 도구가 정하는"),
    ("conditions: {A: {replay_on: [tilt, pour]}}", "replay_on"),
    ("conditions: {A: {pd: {tilt_comp.left.payload: [0, 0, 0, 0]}}}", "--payload-kg"),
    ("conditions: {'a/b': {}}", "이름"),
    ("conditions: {a: {}, A: {}}", "대소문자"),
])
def test_conds_yaml_rejects_bad_conditions(tmp_path, body, why):
    y = tmp_path / "c.yaml"
    y.write_text(body)
    with pytest.raises(ValueError, match=why):
        ab.load_conds(y)


def test_overrides_apply_by_dotted_path_and_reject_missing_or_wrong_length():
    cfg = {"gravity": {"payload": {"left": [0.2, 0, 0, 0.1]}}, "tilt_comp": {"left": {"ki": [1] * 7}}}
    out = ab.apply_overrides(cfg, {"gravity.payload.left": [0.3, 0, 0, 0.1], "tilt_comp.left.ki": [2] * 7})
    assert out["gravity"]["payload"]["left"][0] == 0.3 and out["tilt_comp"]["left"]["ki"] == [2] * 7
    assert cfg["gravity"]["payload"]["left"][0] == 0.2                       # 원본은 그대로
    for bad in ({"gravity.payload.lft": [0] * 4}, {"nope.x": 1}, {"tilt_comp.left.ki": [1] * 6}, {"gravity.payload.left.x": 1}):
        with pytest.raises(KeyError):
            ab.apply_overrides(cfg, bad)


def test_overrides_accept_per_side_list_for_list_key():
    cfg = {"gravity": {"scale": [1.0] * 7}}
    sc = {"left": [1.0, 1.24, 1.1, 1.0, 1.0, 1.0, 0.87], "right": [1.0] * 4 + [0.73, 1.0, 0.88]}
    assert ab.apply_overrides(cfg, {"gravity.scale": sc})["gravity"]["scale"] == sc
    for bad in ({"left": [1.0] * 6}, {"lft": [1.0] * 7}, {}):
        with pytest.raises(KeyError):
            ab.apply_overrides(cfg, {"gravity.scale": bad})


def test_cond_pd_yaml_copies_real_pd_config_and_passes_strict_loader(tmp_path):
    base = ab.PD_DEFAULT["real"]
    assert ab.cond_pd_yaml(base, "A", {}, tmp_path) == base                  # 덮어쓰기 없으면 원본(재기동 없음)
    p = ab.cond_pd_yaml(base, "B", {"gravity.payload.left": [0.3, 0.014342, 0.008226, 0.132814]}, tmp_path)
    import yaml
    assert p.parent == tmp_path and yaml.safe_load(p.read_text())["gravity"]["payload"]["left"][0] == 0.3
    with pytest.raises(ValueError):                                         # pd 로더가 거부하는 형식(PdConfigError)
        ab.cond_pd_yaml(base, "C", {"gravity.cap_nm": "많이"}, tmp_path)


def test_conditions_by_name_list_and_pd_switch_count():
    assert ab.conditions(4, "base,grav", ["base", "grav"]) == ["base", "grav", "base", "grav"]
    assert ab.conditions(3, "abc", ["A", "B", "C"]) == ["A", "B", "C"]
    spec = {"conditions": {"A": {"pd": {}}, "B": {"pd": {"x": 1}}, "C": {"pd": {"x": 2}}}}
    assert ab.pd_switches(list("ABAB"), spec) == 4 and ab.pd_switches(list("AA"), spec) == 0
    assert ab.pd_switches(list("BBCC"), spec) == 3                           # 처음 · B→C · 끝에 원래로


def test_resolve_carry_takes_latest_per_side(tmp_path):
    for d, side in (("101010_setting_x", "right"), ("101020_setting_x", "left"), ("101030_setting_x", "right")):
        f = tmp_path / d / f"carry_{side}" / "carry_cmd.npz"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"")
    argv = ab.resolve_carry(("x", "--npz", "<carry_right>"), tmp_path)
    assert argv[2].endswith("101030_setting_x/carry_right/carry_cmd.npz")
    assert ab.resolve_carry(("<carry_left>",), tmp_path)[0].endswith("101020_setting_x/carry_left/carry_cmd.npz")
    with pytest.raises(FileNotFoundError):
        ab.resolve_carry(("<carry_left>",), tmp_path / "none")


def test_parse_replay_summary_lines():
    lines = ["[replay] 무발행\n", "[replay] 발행 1101 프레임 · 늦음(> 40 ms) 0 · 최대 지연 3.2 ms · 간격 p95 20.4 ms\n",
             "[tilt_now] ...\n", "[replay] 발행 1 프레임 · 늦음(> 40 ms) 2 · 최대 지연 81.0 ms · 간격 p95 nan ms\n"]
    got = ab.parse_replay(lines)
    assert got[0] == {"n": 1101, "late_thr_ms": 40, "late": 0, "max_lag_ms": 3.2, "p95_ms": 20.4}
    assert got[1]["late"] == 2 and np.isnan(got[1]["p95_ms"]) and ab.parse_replay(["x"]) == []


def test_run_cycle_stops_at_first_failure():
    steps = [ab.Step(f"s{i}", kind="still") for i in range(5)]
    seen = []

    def do(s):
        seen.append(s.id)
        return ab.Result(s.id, 0.0, 0.0, 1 if s.id == "s2" else 0)
    out = ab.run_cycle(steps, do)
    assert seen == ["s0", "s1", "s2"] and out[-1].rc == 1


def test_fake_flag_only_adds_exact_zero_guard_bypass(tmp_path):
    def checks(fake):
        return [s.argv for s in ab.cycle_steps({}, tmp_path, ab.TRAJ, 1.0, fake=fake) if s.id.startswith("check_")]
    assert all("--allow-exact-zero" not in a for a in checks(False))
    assert all("--allow-exact-zero" in a for a in checks(True)) and len(checks(True)) == 4


def test_watchdog_only_hold_gets_grace_other_hold_stops_at_once(monkeypatch):
    """워치독만인 HOLD 는 HOLD_GRACE_S 안이면 넘기고 풀리면 기록, 다른 사유 HOLD 는 바로 멈춤."""
    w = object.__new__(ab.Watch)
    w.status, w.hold_since, w.blips = {s: None for s in ab.SIDES}, {s: None for s in ab.SIDES}, []
    now = [100.0]
    monkeypatch.setattr(ab.time, "time", lambda: now[0])
    w._on_status("left", '{"phase": "HOLD", "reasons": ["watchdog: target stale 0.254 s > 0.25 s"]}')
    assert w.hold_sides() == []
    now[0] += ab.HOLD_GRACE_S + 0.1
    assert w.hold_sides() == ["left"]
    w._on_status("left", '{"phase": "TRACKING", "ok": true}')
    assert w.hold_sides() == [] and w.blips and w.blips[0].startswith("left HOLD")
    w._on_status("right", '{"phase": "HOLD", "reasons": ["tracking error 0.2 rad > 0.15"]}')
    assert w.hold_sides() == ["right"]


# pd payload 실행 중 변경 · 계획만 출력은 test_console_bringup.py


# ---------------------------------------------------------------- 분석기(합성 배열)

def test_joint_error_is_measured_minus_applied_without_removing_sag():
    ta = np.linspace(0, 10, 1201)
    qa = np.tile(np.linspace(0, 1, 1201)[:, None], (1, 7))
    tm = np.linspace(0, 10, 7501)
    qm = np.interp(tm, ta, qa[:, 0])[:, None] + np.radians([0, 0, 0, -1.0, 0, 0, 0])   # j4 1° 처짐(상수)
    e = rep.joint_err_deg(tm, qm, ta, qa, (2.0, 4.0))
    assert e[3] == pytest.approx(-1.0, abs=0.05) and abs(e[0]) < 0.05   # ZOH 반 틱 지연만큼
    assert rep.joint_err_deg(tm, qm, ta, qa, (20.0, 21.0)) is None


def test_tip_error_uses_fk_difference():
    fk = lambda q: np.block([[np.eye(3), np.asarray(q[:3])[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])  # noqa: E731
    t = np.linspace(0, 2, 200)
    qa = np.zeros((200, 7))
    qm = qa + np.array([0, 0, -0.004, 0, 0, 0, 0])
    z, n = rep.tip_err_mm(t, qm, t, qa, (0.0, 2.0), fk)
    assert z == pytest.approx(-4.0) and n == pytest.approx(4.0)


def test_tau_stats_and_limit():
    lim = [0, 0, 0, 1.5, 0, 0, 0]
    rows = [(t, {"tilt_comp_tau": [0, 0, 0, 0.1 * t, 0, 0, 0]}) for t in range(20)] + [(30.0, {})]
    s = rep.tau_stats(rows, (0, 25), lim)
    assert s["n"] == 20 and s["max_nm"][3] == pytest.approx(1.9) and s["limit_hit"] == [False] * 3 + [True] + [False] * 3
    assert rep.tau_stats(rows, (0, 25), None)["limit_hit"] is None          # 그 팔 tilt_comp 블록 없음
    assert rep.tau_stats([(1.0, {})], (0, 2), lim) == {"n": 0, "max_nm": None, "limit_hit": None}


def test_goal_fails_when_one_right_joint_exceeds():
    ok = [0.1] * 7
    assert rep.goal_ok(ok, ok, 2.0)
    assert not rep.goal_ok(ok, [0, 0, 0, 0, -0.31, 0, 0], 2.0)              # 오른 j5 하나만 0.3° 넘음
    assert not rep.goal_ok(ok, ok, 3.5) and not rep.goal_ok(ok, None, 2.0)


def test_summary_mean_range_and_goal():
    def row(c, cond, j4, tip, rj6=0.0):
        d = {"left_deg": [0, 0, 0, j4, 0, 0, 0], "right_deg": [0, 0, 0, 0, 0, rj6, 0], "tip_mm": tip}
        return {"cycle": c, "condition": cond, "notes": [], "pre": dict(d), "max": dict(d)}
    rows = [row(1, "A", -1.0, 10.0), row(2, "B", -0.2, 2.0), row(3, "A", -1.2, 12.0), row(4, "B", -0.1, 1.0, rj6=0.5)]
    s = rep.summarize(rows)
    a = s["A"]["최대 기울기 왼 |오차| 최대 [°]"]
    assert a["mean"] == pytest.approx(1.1) and (a["lo"], a["hi"], a["n"]) == (pytest.approx(1.0), pytest.approx(1.2), 2)
    assert s["B"]["최대 기울기 오른 |오차| 최대 [°]"]["mean"] == pytest.approx(0.25)
    assert not s["A"]["goal_pass_max"] and s["B"]["goal_pass_max"] and s["B"]["goal_pass_pre"]
    md = rep.render(rows, s, Path("x"))
    assert "c02 B" in md and "하중" in md and "+0.50" in md and "B 평균 [범위]" in md


# ---------------------------------------------------------------- 구간별 손끝 · 관절 지표

def test_err_stats_tip_zero_when_measured_equals_command_and_splits_move_still():
    from policy_control.tin_live import palm_fk
    fk = palm_fk("left")
    t = np.linspace(0, 4, 3001)
    q = np.zeros((3001, 7)) + np.radians([10, 20, -10, 60, 5, -5, 10])
    q[1500:, 3] += np.radians(20) * np.clip((t[1500:] - 2.0) / 1.0, 0, 1)    # 2~3 s 지령이 움직인다, 그 밖은 정지
    e = rep.err_stats((t, q.copy()), (t, q), (0.0, 4.0), fk)
    for ph in ("move", "still"):
        assert e[ph]["tip_max_mm"] == pytest.approx(0.0, abs=1e-9) and max(e[ph]["max_deg"]) == pytest.approx(0.0, abs=1e-9)
    assert 0.8 < e["move"]["n"] / 750 < 1.2 and e["still"]["n"] > e["move"]["n"]
    qm = q + np.radians([0, 0, 0, -1.0, 0, 0, 0])                            # j4 1° 처짐 → 손끝이 수 mm 움직인다
    e = rep.err_stats((t, qm), (t, q), (0.0, 1.9), fk)
    assert e["move"] is None and e["still"]["mean_deg"][3] == pytest.approx(-1.0, abs=1e-6)
    assert 1.0 < e["still"]["tip_mean_mm"] < 20.0 and e["still"]["tip_max_mm"] >= e["still"]["tip_p95_mm"] >= e["still"]["tip_mean_mm"] - 1e-9
    assert rep.err_stats((t, q), (t, q), (10.0, 11.0), fk) is None


def test_verdict_is_paired_by_cycle():
    a = rep.stat([1.0, 1.4, 1.2])
    assert rep.verdict(a, rep.stat([1.1, 1.3, 1.25])) == "구분 안 됨 +0.02 [-0.10~+0.10]"   # 쌍 차가 0 을 넘나든다
    assert rep.verdict(a, rep.stat([3.0, 3.1, 3.3])) == "+1.93 [+1.70~+2.10]"
    # 회차끼리 같이 흔들리면(조건 간 범위는 겹쳐도) 짝 차이로는 구분된다
    assert rep.verdict(rep.stat([1.0, 2.0, 3.0]), rep.stat([1.2, 2.2, 3.2])).startswith("+0.20")
    assert rep.verdict(a, rep.stat([3.0, 3.1])) == "사이클 부족" and rep.verdict(None, a) == "-"
    assert rep.verdict(a, rep.stat([3.0, None, 3.1])) == "사이클 부족"       # 빈 회차는 짝에서 빠진다


def test_fk_pick_is_time_based_and_few_tip_samples_blank():
    for hz in (750, 250):
        t = np.arange(0, 1, 1 / hz)
        k = rep.fk_pick(t)
        assert 70 <= len(k) <= 76 and np.all(np.diff(t[k]) > 0.011)         # 빈도와 상관없이 약 13 ms 마다
    assert len(rep.fk_pick(np.zeros(0))) == 0
    fk = lambda q: np.eye(4)  # noqa: E731
    t = np.linspace(0, 0.02, 20)                                              # 20 ms → 손끝 표본 2개
    q = np.zeros((20, 7))
    e = rep.err_stats((t, q), (t, q), (0.0, 0.02), fk)
    assert e["still"]["tip_n"] < 3 and e["still"]["tip_mean_mm"] is None and e["still"]["max_deg"] == [0.0] * 7
    assert rep.tip_err_mm(t, q, t, q, (0.0, 0.02), fk) is None


def test_step_tables_and_late_replay_warning():
    def es(tip):
        return {"move": {"n": 10, "mean_deg": [0.1] * 7, "p95_deg": [0.2] * 7, "max_deg": [0.3] * 7,
                         "tip_mean_mm": tip, "tip_p95_mm": tip + 1, "tip_max_mm": tip + 2}, "still": None}
    rows = [{"cycle": i, "condition": c, "notes": [], "steps": {"tilt": {"left": es(tip), "right": es(1.0)}},
             "replay": [{"step": "tilt", "n": 1101, "late_thr_ms": 40, "late": late, "max_lag_ms": 3.0, "p95_ms": 20.0}]}
            for i, (c, tip, late) in enumerate((("A", 5.0, 0), ("B", 2.0, 0), ("A", 5.2, 0), ("B", 2.1, 3),
                                                ("A", 5.1, 0), ("B", 2.0, 0)), 1)]
    steps = rep.summarize_steps(rows)
    cell = steps["tilt"]["left"]["move"]
    assert cell["A"]["tip_mean_mm"]["mean"] == pytest.approx(5.1) and cell["B"]["joint_max_worst"]["mean"] == pytest.approx(0.3)
    md = "\n".join(rep.render_steps(rows, steps))
    assert "| tilt | left | 이동 | 5.10 [5.00~5.20]" in md and "-3.07 [-3.10~-3.00]" in md   # 쌍 차 평균 [범위]
    assert "| tilt | right | 이동 |" in md and "구분 안 됨" in md and "다중 비교 보정 없음" in md
    assert "★늦은 프레임 있음: c04 tilt 3" in md and "3 ★" in md
    top = rep.render_summary(rows)
    assert top[0].startswith("## 요약") and any(x.startswith("- left 이동: A 5.10") and "B−A -3.07" in x for x in top)
    assert not any("정지" in x for x in top[1:])                             # 정지 칸이 없으면 줄도 없다
    rows[1]["steps"]["tilt"]["left"]["move"] = None                          # 빈 회차 → 짝 둘 → 사이클 부족
    assert "사이클 부족" in rep.render_summary(rows)[2]


# ---------------------------------------------------------------- 조건 pd yaml 로 pd 다시 띄우기(가짜 프로세스 · 콘솔)

class FakePdWorld:
    """떠 있는 pd(팔별 pd_config) · 콘솔 유닛 · 우리가 띄운 프로세스를 흉내 — 실제 프로세스 · ROS 없음."""

    def __init__(self, monkeypatch, base, units=True, la=None):
        from types import SimpleNamespace
        self.la = {"fake": "true"} if la is None else la
        self.api_down = self.post_err = self.kill_err = None
        self.cfg = {s: str(base) for s in ab.SIDES}
        self.units = {f"pd_arm_{s}#1": {"alive": True} for s in ab.SIDES} if units else {}
        self.posted, self.released, self.launched, self.killed = [], [], [], []
        self.w = SimpleNamespace(status={"right": {"phase": "TRACKING"}, "left": {"phase": "IDLE"}})
        mp = monkeypatch
        mp.setattr(ab, "pd_proc", lambda s="left": None if self.cfg[s] is None else (
            100 + ab.SIDES.index(s), {"pd_config": self.cfg[s], "execute": True, "sides": s, "contract": "c.json",
                                      "robot": "r.yaml", "stage": "full"}))
        mp.setattr(ab, "proc_env", lambda pid: {"ROS_DOMAIN_ID": "97"})
        mp.setattr(ab, "launch_args", lambda pid: dict(self.la))
        mp.setenv("ROS_DOMAIN_ID", "97")
        self.domain = 97
        mp.setattr(ab.subprocess, "run", self.run)
        mp.setattr(ab.subprocess, "Popen", self.popen)
        mp.setattr(ab.os, "killpg", self.killpg)
        mp.setattr(ab.time, "sleep", lambda s: None)

    # 콘솔 API
    def get(self, path):
        if self.api_down:
            raise ab.CB.BringupError("연결 거부")
        return {"session": {"units": self.units, "profile": {"id": "rh56f1_fake", "domain": self.domain}}}

    def post(self, path, data, method="POST"):
        if self.post_err:
            raise self.post_err
        self.posted.append(data)
        self.units[data["key"]]["alive"] = False
        self.cfg[data["key"].split("#")[0][len("pd_arm_"):]] = None

    lease = release = lambda self: None

    def run(self, argv, **kw):
        side = argv[argv.index("--side") + 1]
        self.released.append(side)
        self.w.status[side] = {"phase": "IDLE"}
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0)

    def popen(self, argv, **kw):
        from types import SimpleNamespace
        side = next(a.split(":=")[1] for a in argv if a.startswith("sides:="))
        self.cfg[side] = next(a.split(":=", 1)[1] for a in argv if a.startswith("pd_config:="))
        self.w.status[side] = {"phase": "IDLE"}
        self.launched.append(argv)
        assert kw.get("start_new_session") and kw["env"] == {"ROS_DOMAIN_ID": "97"}
        return SimpleNamespace(pid=500 + ab.SIDES.index(side), poll=lambda: None, returncode=None)

    def killpg(self, pid, sig):
        if self.kill_err:
            raise self.kill_err
        self.killed.append(pid)
        self.cfg[ab.SIDES[pid - 500]] = None


def test_pd_switch_restarts_only_on_change_releases_engaged_and_restores(tmp_path, monkeypatch):
    base = ab.PD_DEFAULT["real"]
    fw = FakePdWorld(monkeypatch, base)
    spec = {"conditions": {"A": {"pd": {}}, "B": {"pd": {"gravity.payload.left": [0.3, 0.014342, 0.008226, 0.132814]}}}}
    sw = ab.PdSwitch(spec, tmp_path, fw.w, api=fw)
    assert sw.ensure("A") == [] and fw.launched == []                         # 원래 yaml 이면 손대지 않는다
    assert sw.ensure("B") == ["right", "left"]
    assert fw.released == ["right"]                                         # engage 된 팔만 release, IDLE 은 그대로
    assert [d["key"] for d in fw.posted] == ["pd_arm_right#1", "pd_arm_left#1"]   # 처음 pd 는 콘솔 유닛으로 내린다
    argv = fw.launched[0]
    assert f"pd_config:={tmp_path / 'pd_yaml' / 'pd_B_pd_rh56f1_exec.yaml'}" in argv and argv[:2] == ["ros2", "launch"]
    assert {"execute:=true", "sides:=right", "contract:=c.json", "robot:=r.yaml", "stage:=full", "fake:=true"} <= set(argv)
    cfg_b = str(tmp_path / "pd_yaml" / "pd_B_pd_rh56f1_exec.yaml")
    assert sw.ensure("B") == [] and sw.left_as() == {"right": cfg_b, "left": cfg_b}
    assert sw.restore() == ["right", "left"] and fw.killed == [500, 501]    # 우리가 띄운 pd 는 프로세스 그룹으로
    assert len(fw.posted) == 2 and sw.left_as() == {} and f"pd_config:={base}" in fw.launched[-1]
    fw.w.status["left"] = {"phase": "HOLD"}
    n = len(fw.launched)
    with pytest.raises(ab.PdSwitchError, match="left pd HOLD"):
        sw.ensure("B")                                                       # 한 팔이라도 HOLD 면 어느 팔도 다시 띄우지 않는다
    assert len(fw.launched) == n and fw.cfg["left"] == str(base) and fw.released[-1] != "left"


def test_pd_switch_releases_both_arms_before_any_console_unit_off(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    fw.w.status["left"] = {"phase": "TRACKING"}          # 바뀌는 팔이 하나여도 두 팔 다 IDLE 이어야 콘솔이 끈다
    spec = {"conditions": {"A": {"pd": {}}, "B": {"pd": {"gravity.payload.left": [0.3, 0.014342, 0.008226, 0.132814]}}}}
    sw = ab.PdSwitch(spec, tmp_path, fw.w, api=fw)
    post = fw.post
    fw.post = lambda path, data, method="POST": (
        pytest.fail("IDLE 아닌 팔이 있는데 끄려 했다") if any(fw.w.status[s]["phase"] != "IDLE" for s in ab.SIDES)
        else post(path, data))
    sw.ensure("B")
    assert sorted(fw.released) == ["left", "right"]


def test_pd_switch_unit_off_retries_only_stale_pd_refusal(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    sw = ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw)
    calls = []
    post = fw.post

    def flaky(path, data, method="POST"):
        calls.append(data["key"])
        if len(calls) == 1:
            raise ab.CB.BringupError("/api/unit: pd 가 TRACKING 다 — 먼저 PD 해제를 할 것")
        return post(path, data)
    fw.post = flaky
    sw.ensure("B")
    assert calls[:2] == ["pd_arm_right#1", "pd_arm_right#1"]
    fw2 = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    fw2.post_err = ab.CB.BringupError("연결 거부")
    with pytest.raises(ab.PdSwitchError, match="연결 거부"):
        ab.PdSwitch(SPEC_B, tmp_path, fw2.w, api=fw2).ensure("B")


def test_pd_switch_refuses_pd_not_owned_by_console(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"], units=False)
    with pytest.raises(ab.PdSwitchError, match="콘솔 유닛"):
        ab.PdSwitch({"conditions": {"B": {"pd": {"gravity.cap_nm": 20.0}}}}, tmp_path, fw.w, api=fw)
    assert fw.posted == fw.launched == fw.released == []


SPEC_B = {"conditions": {"A": {"pd": {}}, "B": {"pd": {"gravity.payload.left": [0.3, 0.014342, 0.008226, 0.132814]}}}}


@pytest.mark.parametrize("la, why", [({"use_source": "true"}, "use_source"), ({"params_file": "x.yaml"}, "params_file")])
def test_pd_switch_refuses_source_or_extra_params_launch(tmp_path, monkeypatch, la, why):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"], la=la)
    with pytest.raises(ab.PdSwitchError, match=why):
        ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw)


def test_pd_switch_fake_comes_from_original_launch(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"], la={})            # 원래 pd 가 fake:=true 없이 떴다
    ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw).ensure("B")
    assert fw.launched and not any(x == "fake:=true" for argv in fw.launched for x in argv)


def test_pd_switch_console_api_down_is_reported_separately(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    fw.api_down = True
    with pytest.raises(ab.PdSwitchError, match="콘솔 API 응답 없음"):
        ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw)


def test_pd_switch_errors_mid_restart_become_pd_switch_error_and_missing_pd_is_revived(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    sw = ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw)
    real_popen = fw.popen
    calls = {"n": 0}

    def popen_fails_once(argv, **kw):                  # 오른팔: 콘솔 pd 를 내린 뒤 기동이 OSError
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("ros2 없음")
        return real_popen(argv, **kw)
    monkeypatch.setattr(ab.subprocess, "Popen", popen_fails_once)
    with pytest.raises(ab.PdSwitchError, match="right pd 다시 띄우기 중 오류"):
        sw.ensure("B")
    assert sw.missing() == ["right"] and sw.left_as()["right"].startswith("pd 없음") and sw.touched
    assert sw.revive() == {"right": "원래 yaml 로 다시 띄움"} and sw.missing() == []
    assert f"pd_config:={ab.PD_DEFAULT['real']}" in fw.launched[-1]
    assert sw.revive() == {}                                                  # 다 떠 있으면 아무것도 안 한다


def test_pd_switch_console_post_error_and_killpg_errors(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    sw = ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw)
    fw.post_err = OSError("콘솔 끊김")
    with pytest.raises(ab.PdSwitchError, match="다시 띄우기 중 오류"):
        sw.ensure("B")
    fw.post_err = None
    sw.ensure("B")
    assert set(sw.ours) == {"right", "left"}
    fw.kill_err = PermissionError("남의 그룹")                               # killpg 실패 → 우리 pd 를 잊지 않는다
    with pytest.raises(ab.PdSwitchError):
        sw.restore()
    assert "right" in sw.ours
    fw.kill_err = ProcessLookupError()                                         # 그룹이 벌써 없다 → pd 가 내려갔는지만 본다
    fw.cfg["right"] = fw.cfg["left"] = None
    assert sw.restore() == ["right", "left"] and fw.cfg["right"] == str(ab.PD_DEFAULT["real"])


class FakeChild:
    """재생 자식 — 출력 한 줄, wait 는 정한 횟수만큼 시간 초과."""

    def __init__(self, timeouts=0):
        self.stdout, self.signals, self.killed, self.timeouts, self.rc = iter(["[replay] x\n"]), [], False, timeouts, None

    def poll(self):
        return self.rc

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self, timeout=None):
        if self.timeouts:
            self.timeouts -= 1
            raise ab.subprocess.TimeoutExpired("x", timeout)
        self.rc = -2
        return self.rc

    def kill(self):
        self.killed = True


@pytest.mark.parametrize("boom", [KeyboardInterrupt(), RuntimeError("예외")])
@pytest.mark.parametrize("timeouts", [0, 1])
def test_exec_step_ends_child_on_any_exception(monkeypatch, boom, timeouts):
    ch = FakeChild(timeouts)
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *a, **k: ch)

    def wait_child(*a):
        raise boom
    monkeypatch.setattr(ab, "_wait_child", wait_child)
    w = SimpleNamespace(blips=[])
    with pytest.raises(type(boom)):
        ab.exec_step(ab.Step("tilt", ["py", "replay_to_pd.py"]), w, True)
    assert ch.signals == [ab.signal.SIGINT] and ch.killed == bool(timeouts)   # SIGINT → 5 s 안 끝나면 kill


def test_end_child_skips_finished_and_kills_on_timeout():
    ch = FakeChild(timeouts=1)
    ab._end_child(ch, grace=10.0)
    assert ch.signals == [ab.signal.SIGINT] and ch.killed
    ch = FakeChild()
    ch.rc = 0
    ab._end_child(ch)
    assert ch.signals == [] and not ch.killed


def test_exec_step_warns_when_replay_summary_line_missing(monkeypatch, capsys):
    ch = FakeChild()
    ch.rc = 0
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *a, **k: ch)
    monkeypatch.setattr(ab, "_wait_child", lambda step, w, e, p, t0: ab.Result(step.id, t0, t0, 0, ""))
    r = ab.exec_step(ab.Step("tilt", ["py", "deploy/policy_control/tools/replay_to_pd.py"]), SimpleNamespace(blips=[]), True)
    assert "재생 끝 줄" in r.note and r.extra == {"tail": ["[replay] x"]}   # 출력 끝 줄은 늘 남긴다


def test_experiment_with_pd_lost_mid_restart_skips_payload_restore_and_exits_5(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    real_switch = ab.PdSwitch
    monkeypatch.setattr(ab, "PdSwitch", lambda spec, out, w, api=None: real_switch(spec, out, w, api=fw))
    monkeypatch.setattr(ab, "md5", lambda p: "x")
    monkeypatch.setattr(ab, "git_head", lambda: {"git": "x", "dirty": False})
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("ros2 없음")))
    sent, seen = [], {}
    fw.w.payload = lambda side="left": [0.117, 0, 0, 0]
    fw.w.set_payload = lambda kg, side="left", timeout=0: sent.append(kg)

    def cycles(a, conds, traj, out, w, base):          # 사이클 앞 다시 띄우기에서 왼팔 pd 를 내린 뒤 Ctrl-C
        fw.cfg["left"] = None
        raise KeyboardInterrupt
    monkeypatch.setattr(ab, "_cycles", cycles)
    monkeypatch.setattr(ab, "write_meta", lambda out, **kw: seen.update(kw))
    a = SimpleNamespace(payload_kg=0.0, rate_scale=1.0, cycles=2, keep_bringup=True, spec=SPEC_B)
    with pytest.raises(KeyboardInterrupt):
        ab._experiment(a, ["A", "B"], Path(ab.PD_DEFAULT["real"]), tmp_path, fw.w, None)
    assert sent == [0.0]                                                     # 적용만, 왼팔 pd 가 없어 복원 대기는 건너뜀
    pd = seen["pd_switch"]
    assert pd["missing"] == ["left"] and pd["revived"]["left"].startswith("★다시 띄우기 실패")
    assert seen["payload"]["restored"] is False and "pd 없음" in seen["payload"]["restore_error"]

    fw.cfg["left"] = str(ab.PD_DEFAULT["real"])
    monkeypatch.setattr(ab, "_cycles", lambda *x: (fw.cfg.__setitem__("left", None), "멈춤")[1])
    assert ab._experiment(a, ["A", "B"], Path(ab.PD_DEFAULT["real"]), tmp_path, fw.w, None) == 5



def _td_world(tmp_path, monkeypatch, cycles_result, keep=False):
    """_experiment 끝의 teardown — 콘솔 쪽(CB.teardown)은 받은 인자만 적는다. 오른팔은 TRACKING 으로 시작."""
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    real_switch = ab.PdSwitch
    monkeypatch.setattr(ab, "PdSwitch", lambda spec, out, w, api=None: real_switch(spec, out, w, api=fw))
    monkeypatch.setattr(ab, "md5", lambda p: "x")
    monkeypatch.setattr(ab, "git_head", lambda: {"git": "x", "dirty": False})
    fw.w.payload = lambda side="left": [0.117, 0, 0, 0]
    fw.w.set_payload = lambda kg, side="left", timeout=0: None
    monkeypatch.setattr(ab, "_cycles", lambda *x: cycles_result)
    seen, calls = {}, []
    monkeypatch.setattr(ab, "write_meta", lambda out, **kw: seen.update(kw))

    def cb_teardown(api, stages, out=print, *, keep_drivers, retry_s=0.0, poll=0.5):
        calls.append((list(stages), keep_drivers))
        return {"off": [], "rewound": [f"pd_arm_{s}" for s in ab.SIDES], "failed": [], "drivers_kept": keep_drivers}
    monkeypatch.setattr(ab.CB, "teardown", cb_teardown)
    a = SimpleNamespace(payload_kg=0.0, rate_scale=1.0, cycles=2, keep_bringup=keep, spec=SPEC_B)
    rc = ab._experiment(a, ["A"], Path(ab.PD_DEFAULT["real"]), tmp_path, fw.w, None)
    return fw, seen, calls, rc


def test_experiment_finished_releases_both_arms_then_tears_down_with_drivers(tmp_path, monkeypatch):
    fw, seen, calls, rc = _td_world(tmp_path, monkeypatch, None)
    assert rc == 0 and fw.released == ["right"]                              # 잡고 있던 팔만 release → 두 팔 IDLE
    (stages, keep), = calls
    assert keep is None and {"hand_left", "hand_right", "pd_arm_left", "pd_arm_right", "pd_load_left"} <= set(stages)
    assert seen["teardown"]["drivers_kept"] is None


def test_experiment_stopped_keeps_drivers_and_pd_of_arm_still_held(tmp_path, monkeypatch):
    fw, seen, calls, rc = _td_world(tmp_path, monkeypatch, "Ctrl-C")
    assert rc == 2 and fw.released == []                                     # 중단이면 release 하지 않는다(붙든 채)
    (stages, keep), = calls
    assert keep and "끝까지" in keep and "pd_arm_right" not in stages and {"hand_right", "pd_arm_left"} <= set(stages)
    fw2 = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    monkeypatch.setattr(ab.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=1))   # release 실패
    ab.teardown(SimpleNamespace(), fw2.w, None, finished=True)
    assert "release 실패" in calls[-1][1] and "pd_arm_right" not in calls[-1][0]   # 끝까지 갔어도 drivers 남김


def test_experiment_keep_bringup_tears_down_nothing(tmp_path, monkeypatch):
    fw, seen, calls, rc = _td_world(tmp_path, monkeypatch, None, keep=True)
    assert rc == 0 and calls == [] and fw.released == [] and seen["teardown"] == {"skipped": "--keep-bringup"}


def test_pd_switch_refuses_console_of_other_domain(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    fw.domain = 50                                                           # 실기 콘솔(8091)에 붙었다
    with pytest.raises(ab.PdSwitchError, match="도메인 50"):
        ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw)
    assert fw.posted == []


def test_pd_proc_sees_only_this_shell_domain(monkeypatch):
    import os
    import subprocess
    import time
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "--ros-args", "-r", "__node:=pd_node_left",
                          "-p", "pd_config:=x.yaml", "-p", "sides:=left"], env={**os.environ, "ROS_DOMAIN_ID": "77"})
    try:
        time.sleep(0.3)
        monkeypatch.setenv("ROS_DOMAIN_ID", "77")
        got = ab.pd_proc("left")
        assert got is not None and got[0] == p.pid and got[1]["pd_config"] == "x.yaml"
        monkeypatch.setenv("ROS_DOMAIN_ID", "78")
        assert ab.pd_proc("left") is None or ab.pd_proc("left")[0] != p.pid
    finally:
        p.kill()
        p.wait()


def test_pd_switch_interrupt_while_pd_comes_up_takes_launch_down_then_revive(tmp_path, monkeypatch):
    fw = FakePdWorld(monkeypatch, ab.PD_DEFAULT["real"])
    sw = ab.PdSwitch(SPEC_B, tmp_path, fw.w, api=fw)
    st = {"rc": None, "sig": []}

    def slow_popen(argv, **kw):                         # launch 가 아직 pd 노드를 못 띄웠다
        return SimpleNamespace(pid=600, poll=lambda: st["rc"], returncode=None)

    def killpg(pid, sig):
        st["sig"].append((pid, sig))
        st["rc"] = -2

    def sleep(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(ab.subprocess, "Popen", slow_popen)
    monkeypatch.setattr(ab.os, "killpg", killpg)
    monkeypatch.setattr(ab.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        sw.ensure("B")
    assert st["sig"] == [(600, ab.signal.SIGINT)] and "right" not in sw.ours and sw.missing() == ["right"]
    monkeypatch.setattr(ab.subprocess, "Popen", fw.popen)
    monkeypatch.setattr(ab.time, "sleep", lambda s: None)
    assert sw.revive() == {"right": "원래 yaml 로 다시 띄움"} and sw.missing() == []


def test_joint_table_has_14_rows_with_paired_diff():
    def row(c, cond, off):
        seg = {"still": {"mean_deg": [off] * 7, "p95_deg": [abs(off) + 0.1] * 7}}
        return {"cycle": c, "condition": cond, "steps": {"s": {"left": seg, "right": seg}}, "notes": []}
    rows = [row(1, "A", 0.5), row(2, "B", 0.2), row(3, "A", 0.6), row(4, "B", 0.25), row(5, "A", 0.55), row(6, "B", 0.2)]
    lines = rep.render_joints(rows)
    body = [ln for ln in lines if ln.startswith("| left |") or ln.startswith("| right |")]
    assert len(body) == 14
    assert "B 줄어듦" in body[0] and "-0.33 [-0.35~-0.30]" in body[0]
    assert rep.render_joints(rows[:2])[-1].count("사이클 부족") == 3


# --- 자세 유지 측정(--pose, tools/pose_hold.py) ---
ph = _load("pose_hold")
POSE_YAML = TOOLS.parent / "config" / "pd_comp_ab" / "pose_hold.yaml"


def test_single_name_order():
    assert ab.conditions(None, "base", ["base"]) == ["base"]
    assert ab.conditions(None, "ABA", ["A", "B"]) == list("ABA")


def test_pose_yaml_and_window_shape():
    spec = ph.load_pose(POSE_YAML)
    assert [p["name"] for p in ph.selected(spec, "open")][:3] == ["home", "tilt_start", "max_tilt"]
    assert len(ph.selected(spec, "open")) == 6 and len(ph.selected(spec, "closed")) == 3
    q = np.linspace(-0.5, 0.5, 14)
    f = ph.window(q, [5, 6, 7], -1, spec)
    off = f - q
    assert np.allclose(f[0], q) and np.allclose(f[-1], q)
    assert np.isclose(off[:, [4, 11]].min(), -np.radians(3.0)) and np.allclose(off[:, [0, 1, 2, 3, 7, 8, 9, 10]], 0)
    hold = round(spec["hold_s"] / ph.DT)
    assert np.allclose(f[-hold:], q) and not np.allclose(f[-hold - 2], q)


def test_pose_refs_reject_drift(tmp_path):
    bad = tmp_path / "p.yaml"
    bad.write_text(POSE_YAML.read_text(encoding="utf-8").replace("-69.48, 11.61", "-69.18, 11.61", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="home"):
        ph.make(ph.load_pose(bad), "open", tmp_path / "o", ab.TRAJ, ab.CONTRACT, ab.PD_DEFAULT["fake"])


def test_pose_make_files(tmp_path):
    m = ph.make(ph.load_pose(POSE_YAML), "object", tmp_path, ab.TRAJ, ab.CONTRACT, ab.PD_DEFAULT["fake"])
    assert len(m["windows"]) == 12 and {w["d"] for w in m["windows"]} == {1, -1}
    grip = [1.2, 0.24, 1.08, 1.08, 0.85, 0.85]
    r, l = (np.load(tmp_path / f"hand_grip_{s}.npz")["arm_target"] for s in ("right", "left"))
    assert r.shape[1] == 26 and np.allclose(r[-1, 14:20], grip) and np.allclose(l[-1, 14:26], grip * 2)
    assert np.allclose(r[0, 20:26], r[-1, 20:26]) and not np.allclose(r[-1, 20:26], grip)   # 오른손 닫는 동안 왼손 그대로
    to = np.load(tmp_path / "tilt_to_max.npz")["arm_target"]
    assert np.allclose(to[-1], np.load(ab.TRAJ)["arm_target"][768])


@pytest.mark.parametrize("grip", ["open", "closed", "object"])
def test_pose_steps_order(tmp_path, grip):
    spec = ph.load_pose(POSE_YAML)
    cond = ab.load_conds(POSE_YAML)["conditions"]["base"]
    ids = [s.id for s in ph.pose_steps(ab, spec, grip, cond, tmp_path / "c", ab.TRAJ, 1.0, True)]
    assert ids[:2] == ["engage_right", "engage_left"] and ids[-1] == "still_end"
    assert ids.count("plan_tilt_start_right") == 1 and ids.count("back_left") == 1
    assert ids.index("win_tilt_start_4") < ids.index("tilt_to_max") < ids.index("win_max_tilt_1") < ids.index("tilt_from_max") < ids.index("back_left")
    assert sum(i.startswith("win_") for i in ids) == 4 * len(ph.selected(spec, grip))
    hand = [i for i in ids if i.startswith(("hand_open_", "hand_grip_", "hand_release_", "hand_home_open", "enter_"))]
    #: ★11.18 사용자 "손 home 값은 편 손" — 홈에 서 있는 동안은 계약 home_hand 로 끝난다(옛 hand_home_fist).
    assert hand == {"open": [],
                    "closed": ["hand_grip_right", "hand_grip_left", "hand_release_right", "hand_release_left"],
                    "object": ["hand_open_right", "enter_hand_grip_right", "hand_grip_right",
                               "hand_open_left", "enter_hand_grip_left", "hand_grip_left",
                               "enter_hand_release_right", "hand_release_right",
                               "enter_hand_release_left", "hand_release_left", "hand_home_open"]}[grip]
    if grip != "open":   # 닫고 나서 확인(정착) 뒤 측정, 측정 뒤에 편다
        assert ids.index("still_hand_grip_left") < ids.index("win_home_1") < ids.index(hand[-1 if grip == "closed" else 6])
    assert ("hand_c3_j7flip_back" in ids) == (grip == "open")


def test_pose_steps_held_sends_no_hand_command_and_ends_at_home(tmp_path):
    """★11.18 --grip held: 쥔 채 홈에서 — 손 단계 · 차렷 · 주먹(pd_hand_path) · 손 열 재생이 하나도 없다."""
    spec = ph.load_pose(POSE_YAML)
    cond = ab.load_conds(POSE_YAML)["conditions"]["base"]
    steps = ph.pose_steps(ab, spec, "held", cond, tmp_path / "c", ab.TRAJ, 1.0, True)
    ids = [s.id for s in steps]
    assert ids[:7] == ["engage_right", "engage_left", "check_at_home_right", "settle_home_right",
                       "check_at_home_left", "settle_home_left", "still_home"]
    assert ids[-1] == "settle_back_left" and not any(i.startswith(("check_rest_", "to_rest_", "to_home_", "still_end")) for i in ids)
    assert not any(i.startswith(("hand_", "enter_", "fist_")) for i in ids)
    argv = " ".join(" ".join(map(str, s.argv)) for s in steps)
    assert not any(k in argv for k in ("pd_hand", "_hj_", "hand_")), "손 지령(pd hand 서비스 · 손 관절 재생)이 없어야 한다"
    assert sum(i.startswith("win_") for i in ids) == 4 * len(ph.selected(spec, "held"))
    chk = next(s for s in steps if s.id == "check_at_home_right").argv
    assert "--at" in chk and chk[chk.index("--at") + 1] == "end"


def test_held_cycles_swap_only_pd_and_keep_it_engaged(tmp_path, monkeypatch):
    """held: 조건이 바뀌는 사이클 앞에서만 pd 교체(ensure), 끝에 release · 되돌리기 없음."""
    calls = []
    sw = SimpleNamespace(ensure=lambda c: calls.append(("ensure", c)) or [], restore=lambda: calls.append(("restore",)),
                         left_as=lambda: {})
    spec = {"conditions": {"A": {"pd": {}, "replay_args": [], "replay_on": list(ab.SEGMENTS)},
                           "B": {"pd": {"gravity.scale": [1.0] * 7}, "replay_args": [], "replay_on": list(ab.SEGMENTS)}}}
    a = SimpleNamespace(spec=spec, pd_switch=sw, payload_kg=0.0, rate_scale=1.0, fake=True)
    monkeypatch.setattr(ab, "POSE", {"spec": ph.load_pose(POSE_YAML), "grip": "held"})
    monkeypatch.setattr(ab, "record", lambda *x, **k: 0)
    monkeypatch.setattr(ab, "running_pd_config", lambda s: None)
    monkeypatch.setattr(ab, "run_one", lambda *x, **k: True)
    monkeypatch.setattr(ab.subprocess, "run", lambda argv, **k: calls.append(tuple(argv)) or SimpleNamespace(returncode=0))
    w = SimpleNamespace(payload=lambda s: [0.0], set_payload=lambda kg: None, stop_event=lambda why: None)
    assert ab._cycles(a, ["A", "B", "A"], ab.TRAJ, tmp_path, w, {}) is None
    assert calls == [("ensure", "A"), ("ensure", "B"), ("ensure", "A")]     # pd_release · restore 없음


def test_pose_report_rows(tmp_path):
    t = np.arange(0, 20, 0.01)
    q = np.zeros((len(t), 7))
    arms = {s: {"meas": (t, q + np.radians(0.5)), "applied": (t, q)} for s in ("left", "right")}
    meta = {"steps": [{"id": "win_home_1", "rc": 0, "t0": 0.0, "t1": 10.0}], "pose": ""}
    assert rep.pose_rows(meta, arms) == []
    import json
    f = tmp_path / "windows.json"
    f.write_text(json.dumps({"windows": [{"id": "win_home_1", "pose": "home", "d": 1, "rep": 1, "grip": "open", "hold_s": 3.0}]}))
    meta["pose"] = str(f)
    r = rep.pose_rows(meta, arms)
    assert len(r) == 1 and np.allclose(r[0]["left"], 0.5) and r[0]["t"] == pytest.approx([7.2, 9.7])
    md = "\n".join(rep.render_pose([{"condition": "base", "cycle": 1, "pose": r}]))
    assert "d 별 평균" in md and "+0.500" in md


def test_adopt_ours_takes_pd_left_by_previous_run(tmp_path, monkeypatch):
    """★11.18 지난 실행 meta 의 ours_pgid 에 있는 pd 만 이어받고, 원래 yaml 은 사본 이름에서 되찾는다."""
    import json
    T = ab
    cfg = tmp_path / "deploy/policy_control/config"
    cfg.mkdir(parents=True)
    (cfg / "pd_rh56f1_exec.yaml").write_text("x: 1\n")
    run = tmp_path / "logs/runs/261118/tilt_comp_ab/021904"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"pd_switch": {"ours_pgid": {"right": 4242}}}))
    monkeypatch.setattr(T, "SIM2REAL", tmp_path)
    monkeypatch.setattr(T, "_pgid", lambda pid: 4242)
    got = T.adopt_ours("right", 1, run / "pd_yaml/pd_B_pd_rh56f1_exec.yaml")
    assert got is not None and got[0].pid == 4242 and got[1] == cfg / "pd_rh56f1_exec.yaml"
    assert T.adopt_ours("left", 1, run / "pd_yaml/pd_B_pd_rh56f1_exec.yaml") is None
    monkeypatch.setattr(T, "_pgid", lambda pid: 9999)
    assert T.adopt_ours("right", 1, run / "pd_yaml/pd_B_pd_rh56f1_exec.yaml") is None
