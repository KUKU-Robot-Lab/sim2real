from types import SimpleNamespace as C

from vlm_supervisor.tools.fake_stack import plan


def cmd(argv=(), *, manual=False, stop=()):
    return C(argv=tuple(argv), manual=manual, stop=tuple(stop), background=True, note="")


def test_plan_keeps_swap_stop_drops_teardown_and_runner():
    stages = [
        ("pd", (cmd(["pd", "x"]), cmd(stop=["pd#0"]), cmd(["pd", "y"]))),
        ("ep", (cmd(["policy"]), cmd(["episode_runner_node.py"]), cmd(["ep_cmd"], manual=True), cmd(stop=["ep#0"]))),
    ]
    assert [k for k, _ in plan(stages, ["episode_runner_node.py"])] == ["pd#0", "pd#1", "pd#2", "ep#0"]
