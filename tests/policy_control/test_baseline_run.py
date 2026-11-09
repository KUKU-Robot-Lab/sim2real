"""baseline_run: 실행 때 폴더 번호 고르기와 snapshot(ROS 없이)."""
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("baseline_run", ROOT / "deploy/policy_control/tools/baseline_run.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def test_next_run_dir_takes_next_free_number(tmp_path):
    assert br.next_run_dir(tmp_path, "20261108").name == "baseline_20261108_01"
    for n in ("baseline_20261108_01", "baseline_20261108_03", "baseline_20261107_09"):
        (tmp_path / "runs" / n).mkdir(parents=True)
    assert br.next_run_dir(tmp_path, "20261108").name == "baseline_20261108_04"


def test_snapshot_records_policies_pd_and_commit(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    snap = br.write_snapshot(out, "episode_grasp_f_right", ROOT / "config/mission_rh56f1_control.yaml",
                             ROOT / "config/episodes/grasp_f_right.yaml")
    assert json.loads((out / "snapshot.json").read_text()) == snap
    assert len(snap["sim2real_commit"]) == 40 and isinstance(snap["sim2real_dirty"], bool)
    rcg04 = snap["policies"]["rh56f1/aglt/right_rcg04"][0]           # 넘겨받는 정책(handoff)도 잡는다
    contract = json.loads((ROOT / rcg04["contract"]).read_text())
    assert rcg04["checkpoint_md5"] == contract["checkpoint_md5"] and rcg04["ep"] == 2800
    assert "rh56f1/aglt/right_approach_ap05" in snap["policies"]
    exec_pd = yaml.safe_load((ROOT / "deploy/policy_control/config/pd_rh56f1_exec.yaml").read_text())
    assert snap["pd_configs"]["pd_exec"]["gains"] == exec_pd["gains"]
    copies = {p.name for p in (out / "copies").iterdir()}
    assert "mission__mission_rh56f1_control.yaml" in copies and "pd_exec__pd_rh56f1_exec.yaml" in copies
    assert "episode__grasp_f_right.yaml" in copies


def test_mission_episode_starts_and_stops_baseline():
    mission = yaml.safe_load((ROOT / "config/mission_rh56f1_control.yaml").read_text())
    cmds = mission["run"]["episode_grasp_f_right"]
    argvs = [c.get("argv", []) for c in cmds]
    start = [i for i, a in enumerate(argvs) if "baseline_run.py" in " ".join(a) and "start" in a]
    stop = [i for i, a in enumerate(argvs) if "baseline_run.py" in " ".join(a) and "stop" in a]
    assert len(start) == 1 and len(stop) == 1 and start[0] < stop[0]
    assert not cmds[start[0]].get("background")                     # 바로 끝난다 → 배경 번호가 밀리지 않는다
    assert argvs[start[0]][argvs[start[0]].index("--sides") + 1] == "right"
