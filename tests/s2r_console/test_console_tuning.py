"""★11.09 tuning 탭 — 정지 마찰 실험 폴더 진행(tuning_view)과 화면 단계(console.js tuSteps · renderTuning)."""
import json
from pathlib import Path

from test_console_workspace_web import WEB, _node


def _exp(root: Path, name: str, trials, runs):
    e = root / "analysis/friction" / name
    (e / "setup").mkdir(parents=True)
    (e / "result").mkdir()
    (e / "setup" / "trials.txt").write_text("".join(f"{t}\n" for t in trials))
    (e / "setup" / f"pd_rh56f1_stationary_{name.rsplit('_', 1)[-1]}_off.yaml").write_text("")
    for t, summary in runs.items():
        (e / "runs" / t).mkdir(parents=True)
        if summary is not None:
            (e / "runs" / t / "summary.json").write_text(json.dumps(summary))
    return e


def test_tuning_view_reads_folders_runs_and_builds_runbook_commands(tmp_path):
    from s2r_console import tuning_view as TU
    _exp(tmp_path, "261109_F1off_right", ["a", "b", "c"],
         {"a": {"pass": True, "eligible": True, "excluded_reasons": []},
          "b": {"pass": False, "eligible": False, "excluded_reasons": ["tracking_error"]}})
    _exp(tmp_path, "261109_F3_left", ["x", "y"], {"x": {"pass": True}})
    v = TU._build(tmp_path, "261109")
    e = v["experiments"]["261109_F1off_right"]
    assert (e["side"], e["test"], e["cfg"], e["n"], e["done"], e["passed"]) == ("right", "F1off", "off", 3, 2, 1)
    assert e["stopped"]["id"] == "b" and "tracking_error" in e["stopped"]["why"] and e["next"] == "c"
    f1 = v["sides"]["right"][0]
    assert f1["key"] == "F1off" and f1["folders"] == ["261109_F1off_right"]
    assert f1["new_name"] == "261109_F1off_right_b"              # 같은 폴더를 다시 쓰지 않는다
    new = f1["commands"]["new"]
    assert new["prepare"] == "analysis/friction/common/scripts/new_experiment.sh 261109_F1off_right_b right off F1 2.0 3"
    assert "pd_rh56f1_stationary_right_off.yaml sides:=right execute:=true stage:=full" in new["pd"]
    assert new["pd"].count("source ") == 3 and "ROS_DOMAIN_ID=50" in new["pd"]
    assert "--side right --only pd_engage --hold-s 10 --execute --approve pd_engage" in new["engage"]
    assert "$(cat $E/setup/trials.txt)" in new["trials"] and "|| break" in new["trials"]
    # 이어서 하면 남은 trial 만(RUNBOOK 4 절)
    assert "for t in c;" in f1["commands"]["261109_F1off_right"]["trials"]
    f3l = next(r for r in v["sides"]["left"] if r["key"] == "F3")
    assert f3l["folders"] == ["261109_F3_left"] and f3l["new_name"] == "261109_F3_left_b"
    assert [r["key"] for r in v["sides"]["left"]] == ["F1off", "F3", "F4", "F1on", "F2x05", "F2x1", "F2x2"]
    assert "fs_i200 F2 2.0 3" in v["sides"]["left"][-1]["commands"]["new"]["prepare"]
    assert TU.view(tmp_path / "nowhere")["experiments"] == {}


def test_tuning_tab_shows_the_step_to_do_now_and_hold_button_only_when_ramping():
    html = (WEB / "index.html").read_text()
    assert 'data-arg="tuning"' in html and 'id="tab-tuning"' in html
    _node(r"""
const cmd = {prepare:'P', pd:'PD', engage:'EN', hold:'H', trials:'TR'};
const row = {key:'F1off', label:'F1 끔', cfg:'off', stage:'F1', offset_deg:'2.0', repeats:'3', trials:24,
  folders:['261109_F1off_right'], new_name:'261109_F1off_right_b', commands:{new:cmd, '261109_F1off_right':cmd}};
const exp = {name:'261109_F1off_right', path:'analysis/friction/261109_F1off_right', prepared:true, n:24, done:0, passed:0,
  next:'t1', stopped:null, complete:false, trials:[{id:'t1', state:'todo', why:''}], result:[]};
const st = (phase) => vm.runInContext(`tuSteps('right', ${JSON.stringify(row)}, ${JSON.stringify(exp)}, '${phase}')`, ctx);
const now = (phase) => st(phase).find((x) => x.state === 'now');
assert.equal(now('').n, 3);                         // PD 상태 없음 → PD 기동
assert.equal(now('IDLE').n, 4);
const ramp = now('RAMPING'); assert.equal(ramp.n, 5); assert.equal(ramp.buttons.join(), 'pd_hold:right');
assert.equal(now('TRACKING').n, 6); assert.equal(now('TRACKING').cmd, 'TR');
assert.equal(vm.runInContext(`tuSteps('right', ${JSON.stringify(row)}, null, '')`, ctx).find((x) => x.state === 'now').cmd, 'P');
const stopped = {...exp, done:1, stopped:{id:'t1', why:'tracking_error'}};
const s6 = vm.runInContext(`tuSteps('right', ${JSON.stringify(row)}, ${JSON.stringify(stopped)}, 'TRACKING')`, ctx)[5];
assert.equal(s6.state, 'stop'); assert.equal(s6.cmd, ''); assert.match(s6.say, /tracking_error/);
const session = {nodes:[{name:'pd_right', stale:false, status:{phase:'RAMPING'}}],
  tuning:{runbook:'analysis/friction/common/RUNBOOK.md', experiments:{'261109_F1off_right':exp}, sides:{right:[row], left:[]}}};
vm.runInContext(`renderTuning(${JSON.stringify(session)})`, ctx);
assert.match(el('tu-pick').innerHTML, /F1 끔/); assert.match(el('tu-pick').innerHTML, /0\/24/);
const h = el('tu-steps').innerHTML;
assert.match(h, /data-arg="pd_hold:right">팔 그 자리 붙들기/); assert.match(h, /▶ 지금/); assert.match(h, /data-arg="pd_release:right"/);
""")
