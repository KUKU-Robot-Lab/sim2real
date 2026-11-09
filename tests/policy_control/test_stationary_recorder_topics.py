"""stationary_recorder 가 에피소드 경계·estop 토픽을 pd_node 와 같은 이름·래치 QoS 로 받는지."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("stationary_recorder", ROOT / "deploy/policy_control/tools/stationary_recorder.py")
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)


def test_episode_and_estop_topics_match_pd_node_and_are_latched():
    pd_node = (ROOT / "deploy/policy_control/policy_control/pd_node.py").read_text()
    for side in ("right", "left"):
        spec_ = rec.topic_spec(side)
        assert spec_["/policy_control/episode"] == "std_msgs/msg/String"
        assert spec_[f"/policy_control/{side}/episode"] == "std_msgs/msg/String"
        assert spec_["/policy_control/estop"] == "std_msgs/msg/Bool"
        for topic in ("/policy_control/episode", f"/policy_control/{side}/episode", "/policy_control/estop"):
            assert rec.is_latched(topic)
    assert 'f"{NS}/episode"' in pd_node and 'f"{NS}/{s}/episode"' in pd_node and 'f"{NS}/estop"' in pd_node
    assert not rec.is_latched("/joint_states")
