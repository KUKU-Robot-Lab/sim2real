"""baseline_run: snapshot(ROS 없이)."""
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("baseline_run", ROOT / "deploy/policy_control/tools/baseline_run.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


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


def test_mission_episode_has_no_stationary_recorder_and_pd_topics_go_to_bag():
    """★11.10 사용자: 에피소드 옆 stationary_recorder 는 뺀다(코어 1.3개 · 4.5 GB) — 그것만 갖던 PD 토픽은 bag 의 pd 묶음으로."""
    mission = yaml.safe_load((ROOT / "config/mission_rh56f1_control.yaml").read_text())
    argvs = [" ".join(c.get("argv", [])) for c in mission["run"]["episode_grasp_f_right"]]
    assert not any("baseline_run.py" in a or "stationary_recorder" in a for a in argvs)
    assert any("rh56f1_record.sh" in a and "start" in a for a in argvs)
    sh = (ROOT / "deploy/policy_control/tools/rh56f1_record.sh").read_text()
    assert "$OUT/pd' $PD" in sh and "for k in arm hand pd camera" in sh
    for t in ("/pd_state_broadcaster/joint_states", "/pd_temp_broadcaster/dynamic_joint_states", "/policy_control/estop",
              "_forward_${c}_controller/commands", "/policy_control/pd_$s/$t"):
        assert t in sh
