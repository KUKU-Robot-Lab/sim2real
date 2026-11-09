"""★11.09 pd_release --ensure — JTC active · forward 3종 inactive 판정."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/policy_control/tools"))
import pd_release  # noqa: E402

FWD = [f"right_forward_{k}_controller" for k in ("position", "velocity", "effort")]
JTC = "right_joint_trajectory_controller"


def test_jtc_problem():
    ok = {JTC: "active", **{n: "inactive" for n in FWD}}
    assert pd_release.jtc_problem(ok, FWD, JTC) == ""
    assert pd_release.jtc_problem(None, FWD, JTC) == pd_release.NO_RESPONSE       # 응답 없음 = 모름
    assert pd_release.jtc_problem({}, FWD, JTC) == f"{JTC}=없음"                # 빈 목록 = 실제 어긋남
    left = {JTC: "inactive", **{n: "active" for n in FWD}}         # 11.09 17:4x pd 만 SIGTERM 한 뒤
    assert pd_release.jtc_problem(left, FWD, JTC).split(", ") == [f"{JTC}=inactive", *(f"{n}=active" for n in FWD)]
    assert pd_release.jtc_problem({n: "inactive" for n in FWD}, FWD, JTC) == f"{JTC}=없음"
