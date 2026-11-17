"""★11.18 고정 놓기 튜닝 탭(place_tune) — tune 설정 검사 · place_fixed 줄 바꾸기 · jog 누적 · 미션 단계와 맞물림."""
import json
import sys
from pathlib import Path

import pytest
import yaml

from mission_stages import Command
from s2r_console import place_tune as PT
from s2r_console.console import ConsoleError

ROOT = Path(__file__).resolve().parents[2]
STAMP = "261118_101500_123"


@pytest.fixture(autouse=True)
def _clear():
    PT.disarm()
    yield
    PT.disarm()


class _Lease:
    def __init__(self, token="tok"):
        self.token = token

    def check(self, token):
        from s2r_console.lease import LeaseError
        if token != self.token:
            raise LeaseError("다른 lease")


class _Con:
    def __init__(self, token="tok"):
        self.lease = _Lease(token)


def _arm(cup="blue", token="tok", now=None):
    now = __import__("time").time() if now is None else now
    with PT._lock:
        PT._armed.update(cup=cup, side=PT.CUPS[cup][0], since=now, seen=now, who="t", token=token)


def test_parse_jog_one_axis_allowed_values():
    assert PT.parse_jog("0,5,0,0") == [0, 5, 0, 0] and PT.parse_jog("0,0,0,-1") == [0, 0, 0, -1]
    for bad in ("1,1,0,0", "2,0,0,0", "0,0,0,3", "0,0,0,0", "a,0,0,0", "1,0,0"):
        with pytest.raises(ConsoleError):
            PT.parse_jog(bad)


def test_take_only_on_place_fixed_stage_and_jog_needs_arm():
    p = {"tune": "save", "stamp": STAMP, "side": "right"}
    assert PT.take("place_fixed_blue_descend_right", p) == {"tune": "save", "stamp": STAMP} and p == {"side": "right"}
    assert PT.take("tilt_now", {}) == {}
    for aid, prm in (("tilt_now", {"tune": "save", "stamp": STAMP}),
                     ("place_fixed_blue_descend_left", {"tune": "save", "stamp": STAMP}),
                     ("place_fixed_blue_descend_right", {"tune": "save", "stamp": "now"}),
                     ("place_fixed_blue_descend_right", {"tune": "save", "stamp": STAMP, "jog": "1,0,0,0"}),
                     ("place_fixed_blue_descend_right", {"tune": "jog", "stamp": STAMP, "jog": "1,0,0,0"})):   # 안 켬
        with pytest.raises(ConsoleError):
            PT.take(aid, dict(prm))
    jog = {"tune": "jog", "stamp": STAMP, "jog": "1,0,0,0"}
    _arm("pink")
    with pytest.raises(ConsoleError):                 # 다른 컵에 켜져 있다
        PT.take("place_fixed_blue_descend_right", dict(jog), _Con())
    _arm("blue")
    assert PT.take("place_fixed_blue_descend_right", dict(jog), _Con())["jog"] == "1,0,0,0"
    with pytest.raises(ConsoleError):                 # 켠 lease 와 지금 lease 가 다르다 → 거부 · 끔
        PT.take("place_fixed_blue_descend_right", dict(jog), _Con("other"))
    assert not PT.armed()
    _arm("blue")
    with pytest.raises(ConsoleError):                 # 콘솔(lease) 없이 jog 못 함
        PT.take("place_fixed_blue_descend_right", dict(jog))


def test_arm_expires_on_ttl_and_poll_gone_and_executes():
    _arm("blue", now=1000.0)
    assert PT.armed("blue", now=1000.0 + PT.POLL_GONE_S - 1)
    assert not PT.armed("blue", now=1000.0 + PT.POLL_GONE_S + 1)       # 화면 폴링이 끊겼다
    _arm("blue", now=1000.0)
    for t in range(1001, 1000 + int(PT.ARM_TTL_S), 5):
        PT._touch("seen", float(t))
    assert not PT.armed("blue", now=1000.0 + PT.ARM_TTL_S + 1)         # 폴링이 있어도 TTL
    for aid, prm, off in (("place_fixed_blue_descend_right", {"tune": "jog"}, False),
                          ("place_fixed_blue_descend_right", {}, True), ("place_fixed_blue_all_right", {}, True),
                          ("tilt_now", {}, True), ("place_fixed_blue_descend_right", {"tune": "save"}, True),
                          ("place_fixed_blue_descend_right", {"tune": "resume"}, True), ("head_control", {}, False)):
        _arm("blue")
        PT.executed(aid, prm)
        assert bool(PT.armed()) is not off, aid


def test_set_armed_needs_lease_token():
    with pytest.raises(ConsoleError):
        PT.set_armed(_Con(), {"cup": "blue", "on": True}, "who", None)


def _result(repo, cup, op, stamp, **r):
    d = PT.out_dir(repo, cup, op, stamp)
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"mode": op, "execute": True, "rc": 0, **r}))


def _check(repo, cup, op, stamp, holder_mm, yaw):
    d = PT.out_dir(repo, cup, op, stamp) / "plan"
    d.mkdir(parents=True)
    (d / "check.json").write_text(json.dumps({"jog_anchor_mm": {"holder_mm": holder_mm}, "jog_anchor_deg": {"yaw_deg": yaw}}))


def test_anchor_from_last_successful_executed_jog(tmp_path):
    _result(tmp_path, "blue", "jog", "261118_100000_000", jog="5,0,0,0", time=10.0)
    _check(tmp_path, "blue", "jog", "261118_100000_000", [5.0, 0.0, 0.0], 0.0)
    _result(tmp_path, "blue", "jog", "261118_100200_000", jog="0,1,0,0", time=12.0)
    _check(tmp_path, "blue", "jog", "261118_100200_000", [5.0, 1.0, 0.0], -1.0)
    _result(tmp_path, "blue", "jog", "261118_100400_000", jog="0,5,0,0", time=14.0, rc=2)        # 검사 실패 — 안 움직였다
    _result(tmp_path, "blue", "jog", "261118_100500_000", jog="0,5,0,0", time=15.0, execute=False)
    assert PT.anchor_from(PT._results(tmp_path, "blue")) == [5.0, 1.0, 0.0, -1.0]
    _result(tmp_path, "blue", "save", "261118_100600_000", jog=None, time=16.0)
    assert PT.anchor_from(PT._results(tmp_path, "blue")) is None                                   # 저장 뒤 기준이 바뀜


def test_broken_result_json_is_listed_not_counted(tmp_path):
    d = PT.out_dir(tmp_path, "blue", "jog", "261118_100000_000")
    d.mkdir(parents=True)
    (d / "result.json").write_text("{broken")
    rows = PT._results(tmp_path, "blue")
    assert len(rows) == 1 and rows[0]["rc"] is None and "깨짐" in rows[0]["error"]
    assert PT.anchor_from(rows) is None


def test_transform_rewrites_only_place_fixed_line(tmp_path):
    mission = yaml.safe_load((ROOT / "config/mission_rh56f1_fake.yaml").read_text(encoding="utf-8"))
    raw = mission["runbook"]["place_fixed_blue_descend_right"] if "runbook" in mission else None
    argv = next((tuple(c["argv"]) for c in (raw or []) if "place_fixed.py" in " ".join(c["argv"])),
                ("python3", "{repo}/deploy/policy_control/tools/place_fixed.py", "--cup", "blue", "--until", "descend"))
    cmd = Command(argv=(*argv, "--execute"))
    aid = "place_fixed_blue_descend_right"
    assert PT.transform(aid, {}, cmd, tmp_path) is cmd
    manual = Command(argv=("bash", "-lc", "true"))
    assert PT.transform(aid, {"tune": "save", "stamp": STAMP}, manual, tmp_path) is manual
    out = PT.transform(aid, {"tune": "save", "stamp": STAMP}, cmd, tmp_path).argv
    assert "--until" not in out and "--save-offset" in out and "--execute" in out
    assert out[out.index("--out") + 1] == str(tmp_path / "logs/runs/261118/place_fixed/blue_tune_save_101500_123")
    out = PT.transform(aid, {"tune": "jog", "stamp": STAMP, "jog": "5,0,0,0"}, cmd, tmp_path).argv
    assert out[out.index("--jog") + 1] == "5,0,0,0" and "--jog-cum" not in out
    assert PT.transform(aid, {"tune": "jog", "stamp": STAMP, "jog": "5,0,0,0"}, cmd, tmp_path).argv == out   # prepare = execute


def test_cups_match_planner():
    sys.path.insert(0, str(ROOT / "deploy/policy_control/tools"))
    import plan_place_fixed as PF
    assert {k: v[1] for k, v in PT.CUPS.items()} == PF.CUPS


def test_resume_refused_after_unsaved_jog(tmp_path):
    """저장 안 한 jog 뒤 마저 놓기는 prepare(take) 에서 400 — 교시값이 사라지지 않게. 저장 · 새 실행 하강 뒤에는 통과."""
    con, resume = _Con(), {"tune": "resume", "stamp": STAMP}
    con.repo = tmp_path
    descend = tmp_path / "logs/runs/261118/place_fixed/blue_095900"
    descend.mkdir(parents=True)
    (descend / "result.json").write_text(json.dumps({"mode": "descend", "execute": True, "rc": 0, "time": 9.0}))
    assert PT.take("place_fixed_blue_descend_right", dict(resume), con)["tune"] == "resume"
    _result(tmp_path, "blue", "jog", "261118_100000_000", time=10.0)
    _result(tmp_path, "blue", "jog", "261118_100100_000", time=11.0, execute=False)      # 드라이런은 상관없다
    with pytest.raises(ConsoleError) as e:
        PT.take("place_fixed_blue_descend_right", dict(resume), con)
    assert e.value.code == 400 and "먼저 저장" in str(e.value)
    with pytest.raises(ConsoleError):                 # 콘솔 없이는 판단 못 하니 막는다
        PT.take("place_fixed_blue_descend_right", dict(resume))
    _result(tmp_path, "blue", "save", "261118_100200_000", time=12.0, rc=1)              # 실패한 저장은 안 친다
    assert PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))
    _result(tmp_path, "blue", "save", "261118_100300_000", time=13.0)
    assert PT.take("place_fixed_blue_descend_right", dict(resume), con)["tune"] == "resume"
    _result(tmp_path, "blue", "jog", "261118_100400_000", time=14.0)
    assert PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))
    (descend / "result.json").write_text(json.dumps({"mode": "descend", "execute": True, "rc": 0, "time": 15.0}))
    assert not PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))                   # 새 하강 = 기준이 바뀜


def test_summary_reasons_are_sentences(tmp_path):
    _result(tmp_path, "blue", "resume", STAMP, time=10.0, rc=2)
    d = PT.out_dir(tmp_path, "blue", "resume", STAMP) / "plan"
    d.mkdir()
    (d / "check.json").write_text(json.dumps({"resume_pos_mm": {"ok": False}, "cup_table_m": {"ok": False},
                                              "new_key": {"ok": False}, "release_xy_m": {"ok": True}}))
    s = PT._summary(PT._results(tmp_path, "blue")[0])
    assert s["failed"] == ["resume_pos_mm", "cup_table_m", "new_key"]
    assert s["reasons"] == ["저장한 계획 위치와 지금 손 위치가 다름", "컵이 탁자에 닿음", "new_key"]


def _jog_dir(repo, cup, stamp, meta=None, result=None):
    """result.json 없거나 깨진 jog 런 — 재생 중 estop · SIGTERM · 크래시."""
    d = PT.out_dir(repo, cup, "jog", stamp)
    d.mkdir(parents=True)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    if result is not None:
        (d / "result.json").write_text(result)
    return d


def _descend(repo, t, mode="descend"):
    d = repo / "logs/runs/261118/place_fixed" / f"blue_0959{int(t):02d}"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"mode": mode, "execute": True, "rc": 0, "time": t}))


def test_interrupted_or_failed_executed_jog_blocks_resume(tmp_path):
    _descend(tmp_path, 9.0)
    _jog_dir(tmp_path, "blue", "261118_100000_000", meta={"mode": "jog", "execute": True})   # result.json 없음
    assert PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))
    assert "없음" in PT._results(tmp_path, "blue")[0]["error"]


def test_jog_rc2_unknown_execute_and_broken_block(tmp_path):
    _descend(tmp_path, 9.0)
    _result(tmp_path, "blue", "jog", "261118_100000_000", time=10.0, rc=2)
    assert PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))
    _descend(tmp_path, 11.0)                                                    # 새 하강 뒤에는 풀림
    assert not PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))
    _jog_dir(tmp_path, "blue", "261118_100100_000", result="{broken")           # meta 없음 = 실행으로 간주
    assert PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))


def test_dry_run_jog_without_result_ignored(tmp_path):
    _descend(tmp_path, 9.0)
    _jog_dir(tmp_path, "blue", "261118_100000_000", meta={"mode": "jog", "execute": False})
    _result(tmp_path, "blue", "jog", "261118_100100_000", time=1e12, rc=2, execute=False)
    assert not PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))


def test_successful_all_clears_pending(tmp_path):
    _result(tmp_path, "blue", "jog", "261118_100000_000", time=10.0)
    assert PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))
    _descend(tmp_path, 11.0, mode="all")
    assert not PT.unsaved_jog(PT._results(tmp_path, "blue", name="*"))


def test_pink_jog_does_not_block_blue_resume(tmp_path):
    con = _Con()
    con.repo = tmp_path
    _result(tmp_path, "pink", "jog", "261118_100000_000", time=10.0)
    _jog_dir(tmp_path, "pink", "261118_100100_000")
    assert PT.unsaved_jog(PT._results(tmp_path, "pink", name="*"))
    assert PT.take("place_fixed_blue_descend_right", {"tune": "resume", "stamp": STAMP}, con)["tune"] == "resume"


def test_summary_reasons_deduplicated(tmp_path, monkeypatch):
    _result(tmp_path, "blue", "resume", STAMP, time=10.0, rc=2)
    d = PT.out_dir(tmp_path, "blue", "resume", STAMP)
    (d / "plan").mkdir()
    (d / "plan" / "check.json").write_text(json.dumps({"x": {"ok": False}, "x2": {"ok": False}}))
    monkeypatch.setattr(PT, "REASONS", {**PT.REASONS, "x": "같은 문장", "x2": "같은 문장"})
    assert PT._summary(PT._results(tmp_path, "blue")[0])["reasons"] == ["같은 문장"]
