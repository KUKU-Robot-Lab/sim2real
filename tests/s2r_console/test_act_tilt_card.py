"""★11.19 act_tilt(ACT 정책 붓기) — 미션 등록 · 카드 설정(ckpt · max_vel) 치환 · 범위 거부."""
from pathlib import Path

import pytest
import yaml

from s2r_console import actions as A
from s2r_console.console import ConsoleError

REPO = Path(__file__).resolve().parents[2]


def _run(kind):
    return yaml.safe_load((REPO / "config" / f"mission_rh56f1_{kind}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["control", "control_cam5090", "fake"])
def test_act_tilt_stages_are_registered_with_dry_run_never_touching_real(kind):
    m = _run(kind)
    stages = {s["id"]: s for s in m["stages"]}
    assert stages["act_tilt_dry"].get("touches_real", False) is False   # fake 는 이 키를 쓰지 않는다
    assert stages["act_tilt"].get("touches_real", False) is (kind != "fake")
    assert set(stages["act_tilt_dry"]["needs"]) == {"pd_arm_right", "pd_arm_left"}
    # ★11.19 act: 정책이 목도 움직인다 — 끝에 목을 되돌리려면 목 드라이버(head_home)가 떠 있어야 한다
    assert set(stages["act_tilt"]["needs"]) == {"pd_arm_right", "pd_arm_left", "head_home"}


def test_real_act_tilt_publishes_only_with_execute_after_a_manual_check():
    run = _run("control")["run"]
    dry, = run["act_tilt_dry"]
    assert not dry.get("execute_args") and "--execute" not in dry["argv"]
    check, go = run["act_tilt"]
    assert check.get("manual") is True
    assert go["execute_args"] == ["--execute"] and go["argv"] == dry["argv"]
    assert go["argv"][3:] == ["act_tilt", A.ACT_TILT_CKPTS[0], "--max-vel", "0.5"]
    # ★11.19 act: 브리지는 목을 되돌리지 않는다 — 같은 셸에서 abort 여도 head_home 과 같은 명령으로 되돌린다(--execute 일 때만)
    head_home, = [c for c in run["head_home"] if c.get("execute_args")]
    script = go["argv"][2]
    assert " ".join(f'"{a}"' for a in head_home["argv"]) + " --execute" in script
    assert "exec " not in script and script.rstrip().endswith("exit $rc")
    fake, = [c for c in _run("fake")["run"]["act_tilt"] if c.get("execute_args")]
    assert "head_move.py" not in " ".join(fake["argv"])        # fake 는 머리가 없다


def test_card_parameters_swap_ckpt_and_max_vel_only_in_the_bridge_command():
    argv = tuple(_run("control")["run"]["act_tilt"][1]["argv"]) + ("--execute",)
    out = A.act_tilt_argv(argv, {"ckpt": "010000", "max_vel": 0.3})
    assert out[:3] == argv[:3] and out[3:] == ("act_tilt", "010000", "--max-vel", "0.3", "--execute")
    assert A.act_tilt_argv(("bash", "-lc", "true"), {"ckpt": "010000"}) == ("bash", "-lc", "true")
    assert A.act_tilt_argv(argv, {}) == argv


@pytest.mark.parametrize("bad", [{"ckpt": "020000"}, {"ckpt": "../x"}, {"max_vel": 0}, {"max_vel": 1.6},
                                 {"max_vel": "0.5"}, {"max_vel": True}])
def test_out_of_range_card_parameters_are_refused(bad):
    with pytest.raises(ConsoleError):
        A.act_tilt_parameters(bad)



# ── ★11.19 카드 UI — policy 탭 실험 목록의 ACT 칸(experiments.py · console.js) ──
def test_card_item_appears_with_the_same_choices_the_server_accepts():
    from s2r_console import experiments as XV
    ids = [s["id"] for s in _run("fake")["stages"]]
    v = XV.view(REPO, ids)
    assert not v.get("error")
    act, = [i for i in v["items"] if i["group"] == "act"]
    assert act["id"] == "act_tilt" and list(act["sides"]) == ["both"] and act["sides"]["both"]["action"] == "act_tilt"
    c = act["act"]
    assert (c["dry"], c["run"]) == A.ACT_TILT
    assert tuple(c["ckpts"]) == A.ACT_TILT_CKPTS and tuple(c["max_vel"]) == A.ACT_TILT_MAX_VEL
    assert A.ACT_TILT_MAX_VEL[0] < c["default_vel"] <= A.ACT_TILT_MAX_VEL[1]
    assert "last" in c                                      # 실행 로그는 캐시 밖에서 매번 붙는다
    assert not [i for i in XV.view(REPO, [s for s in ids if s != "act_tilt"])["items"] if i["group"] == "act"]


def test_last_run_summary_reads_the_newest_folder_and_flags_a_missing_summary(tmp_path):
    import json
    from s2r_console import experiments as XV
    assert XV.act_tilt_last(tmp_path / "nope") is None and XV.act_tilt_last(tmp_path) is None
    (tmp_path / "20261119_1001").mkdir()
    (tmp_path / "20261119_1001" / "summary.json").write_text(json.dumps(
        {"result": "ok", "steps": 300, "published": 0, "seconds": 10.2, "latency_ms": {"p50": 9.1, "p95": 12.0, "max": 30.5},
         "max_target_minus_measured_rad": None, "max_image_age_s": 0.05, "max_state_age_s": 0.01, "max_vel": 0.5, "extra": 1}))
    last = XV.act_tilt_last(tmp_path)
    assert last["run"] == "20261119_1001" and last["result"] == "ok" and last["latency_ms"]["p95"] == 12.0
    assert "extra" not in last and last["count"] == 1 and not last.get("missing")
    (tmp_path / "20261119_1102").mkdir()                    # 브리지가 summary.json 전에 죽은 실행
    last = XV.act_tilt_last(tmp_path)
    assert last["run"] == "20261119_1102" and last["missing"] is True and last["count"] == 2
    (tmp_path / "20261119_1203").mkdir()
    (tmp_path / "20261119_1203" / "summary.json").write_text("{깨짐")
    assert XV.act_tilt_last(tmp_path)["missing"] is True and "summary.json" in XV.act_tilt_last(tmp_path)["error"]


def test_card_renders_dry_run_and_execute_with_the_chosen_settings():
    from test_console_workspace_web import _node
    _node(r"""
const act = {dry:'act_tilt_dry', run:'act_tilt', ckpts:['050000','010000'], max_vel:[0,1.5], default_vel:0.5,
  last:{run:'20261119_1001', path:'/x/act_tilt_runs/20261119_1001', count:2, result:'abort', error:'state stale', steps:120, published:0,
        seconds:4.2, latency_ms:null, max_target_minus_measured_rad:0.012, max_image_age_s:0.2, max_state_age_s:0.01, max_vel:0.5}};
const X = {items:[{id:'act_tilt', group:'act', key:'ACT', label:'ACT tilt', summary:'ACT 정책 붓기', act,
  sides:{both:{action:'act_tilt', dry:'act_tilt_dry', runner:null, steps:[], problems:[], summary:'ACT 정책 붓기'}}}], policies:{}};
const session = {profile:{domain_class:'real'}, experiments:X, episode_runners:[], mission:{rows:[]}};
const calls = [];
vm.runInContext(`S = {session: ${JSON.stringify(session)}}; xpSel = {id:'act_tilt'};
  S.actions = [{id:'act_tilt_dry', enabled:true, reasons:[]}, {id:'act_tilt', enabled:false, reasons:['양팔 pd TRACKING 아님']}];
  render = () => {}; prepareAction = (id, extra) => { globalThis.__calls.push([id, extra]); };`, Object.assign(ctx, {__calls: calls}));
vm.runInContext('renderExperiments(S.session)', ctx);
assert.match(el('xp-list').innerHTML, /data-arg="act_tilt"[\s\S]*ACT/);
let main = el('xp-main').innerHTML;
assert.match(main, /data-act="act-tilt" data-arg="act_tilt_dry"\s+title="">/);                  // dry run 은 열림
assert.match(main, /data-act="act-tilt" data-arg="act_tilt" disabled/);              // 실행은 게이트 이유로 막힘
assert.match(main, /양팔 pd TRACKING 아님/);
assert.match(main, /btn btn-real" data-act="act-tilt" data-arg="act_tilt"/);
assert.match(main, /aria-pressed="true">50k/);
assert.match(main, /summary\.json|abort/); assert.match(main, /state stale/); assert.match(main, /총 2회/);
vm.runInContext(`acts['act-ckpt']('010000'); actDraft = {...actDraft, vel:'0.3'}; acts['act-tilt']('act_tilt_dry')`, ctx);
assert.deepEqual(JSON.parse(JSON.stringify(calls)), [['act_tilt_dry', {ckpt:'010000', max_vel:0.3}]]);
vm.runInContext(`actDraft = {...actDraft, vel:'2'}; renderExperiments(S.session)`, ctx);
main = el('xp-main').innerHTML;
assert.match(main, /data-arg="act_tilt_dry" disabled/);                               // 범위 밖 속도면 두 버튼 다 막힘
assert.match(main, /aria-invalid="true"/);
assert.doesNotMatch(main, /지금 실행할 수 없다/);
""")
