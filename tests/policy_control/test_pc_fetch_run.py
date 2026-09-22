"""fetch_run: 서버 학습 런에서 계약 생성 입력만 골라 내려받는 도구.

순수 선택 로직(체크포인트·sim meta·파일 계획·매니페스트 비교)만 여기서 잠근다.
ssh/rsync 는 주입된 러너로 대체하므로 네트워크도 서버도 필요 없다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "deploy" / "policy_control" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fetch_run as F  # noqa: E402

NN = [
    "open-short_b_pour_fab.pth",
    "last_open-short_b_pour_fab_ep_2400_rew_35000.5.pth",
    "last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth",
    "last_open-short_b_pour_fab_ep_2450_rew_-12.0.pth",
]
TOP = [
    "nn", "params", "summaries", "videos", "test_history.md",
    "trace_i18_ep2500_adr30_64env.npz",
    "trace_i18_ep2500_adr30_64env_meta.json",
]
LISTING = {"": TOP, "nn": NN, "params": ["env.yaml", "agent.yaml"]}


# ------------------------------------------------------------------ 체크포인트 선택

def test_parse_checkpoints_splits_best_from_epoch_tagged():
    got = {c.name: (c.epoch, c.rew) for c in F.parse_checkpoints(NN)}
    assert got["open-short_b_pour_fab.pth"] == (None, None)
    assert got["last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth"] == (2500, 36408.105)
    assert got["last_open-short_b_pour_fab_ep_2450_rew_-12.0.pth"] == (2450, -12.0)


def test_pick_checkpoint_best_takes_the_file_without_an_epoch_suffix():
    assert F.pick_checkpoint(NN, "best") == "open-short_b_pour_fab.pth"


def test_pick_checkpoint_best_fails_loudly_when_there_is_none():
    with pytest.raises(F.FetchError) as e:
        F.pick_checkpoint([n for n in NN if n.startswith("last_")], "best")
    assert "best" in str(e.value)


def test_pick_checkpoint_last_takes_the_highest_epoch_not_the_highest_reward():
    assert F.pick_checkpoint(NN, "last") == "last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth"


def test_pick_checkpoint_by_epoch():
    assert F.pick_checkpoint(NN, "ep:2400") == "last_open-short_b_pour_fab_ep_2400_rew_35000.5.pth"


def test_pick_checkpoint_missing_epoch_lists_the_nearby_candidates():
    with pytest.raises(F.FetchError) as e:
        F.pick_checkpoint(NN, "ep:9999")
    msg = str(e.value)
    assert "2500" in msg and "2450" in msg


def test_pick_checkpoint_accepts_an_exact_filename_and_rejects_an_unknown_one():
    assert F.pick_checkpoint(NN, NN[0]) == NN[0]
    with pytest.raises(F.FetchError):
        F.pick_checkpoint(NN, "nope.pth")


def test_pick_checkpoint_refuses_to_guess_when_the_query_is_empty():
    with pytest.raises(F.FetchError):
        F.pick_checkpoint(NN, "")


# ------------------------------------------------------------------ sim meta / trace 선택

def test_pick_sim_meta_prefers_the_one_matching_the_checkpoint_epoch():
    top = TOP + ["trace_i18_ep1000_adr30_64env_meta.json"]
    got = F.pick_sim_meta(top, "last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth")
    assert got == "trace_i18_ep2500_adr30_64env_meta.json"


def test_pick_sim_meta_falls_back_to_the_only_one_when_the_epoch_does_not_match():
    got = F.pick_sim_meta(TOP, "open-short_b_pour_fab.pth")
    assert got == "trace_i18_ep2500_adr30_64env_meta.json"


def test_pick_sim_meta_refuses_to_guess_between_several():
    top = TOP + ["trace_i18_ep1000_adr30_64env_meta.json"]
    with pytest.raises(F.FetchError) as e:
        F.pick_sim_meta(top, "open-short_b_pour_fab.pth")
    assert "ep1000" in str(e.value) and "ep2500" in str(e.value)


def test_pick_sim_meta_says_what_to_run_when_the_run_has_none():
    with pytest.raises(F.FetchError) as e:
        F.pick_sim_meta(["nn", "params"], "open-short_b_pour_fab.pth")
    assert "play.py" in str(e.value)


def test_pick_trace_takes_the_npz_beside_the_chosen_meta():
    assert F.pick_trace(TOP, "trace_i18_ep2500_adr30_64env_meta.json") == "trace_i18_ep2500_adr30_64env.npz"


# ------------------------------------------------------------------ 파일 계획

def test_plan_files_maps_the_remote_run_onto_the_local_contract_layout():
    plan = F.plan_files(LISTING, checkpoint="ep:2500", sim_meta="auto", trace="none")
    assert [(f.remote_rel, f.local_rel) for f in plan] == [
        ("params/env.yaml", "params/env.yaml"),
        ("params/agent.yaml", "params/agent.yaml"),
        ("nn/last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth",
         "nn/last_open-short_b_pour_fab_ep_2500_rew_36408.105.pth"),
        ("test_history.md", "test_history.md"),
        ("trace_i18_ep2500_adr30_64env_meta.json", "trace_meta.json"),
    ]


def test_plan_files_takes_the_test_history_because_it_holds_the_training_commit():
    plan = F.plan_files(LISTING, checkpoint="best", sim_meta="auto", trace="none")
    assert any(f.local_rel == "test_history.md" for f in plan)
    listing = {**LISTING, "": [n for n in TOP if n != "test_history.md"]}
    plan = F.plan_files(listing, checkpoint="best", sim_meta="auto", trace="none")
    assert not any(f.local_rel == "test_history.md" for f in plan)   # 없으면 그냥 건너뛴다


def test_sim_meta_none_skips_the_meta_for_single_arm_runs():
    """pour 계열만 sim meta 가 필요하다. grasp 계열 런에는 아예 없다."""
    listing = {**LISTING, "": ["nn", "params", "test_history.md"]}
    plan = F.plan_files(listing, checkpoint="best", sim_meta="none", trace="none")
    assert not any(f.local_rel == "trace_meta.json" for f in plan)
    with pytest.raises(F.FetchError):                       # auto 면 없다고 알려준다
        F.plan_files(listing, checkpoint="best", sim_meta="auto", trace="none")


def test_trace_without_a_meta_is_refused():
    with pytest.raises(F.FetchError) as e:
        F.plan_files(LISTING, checkpoint="best", sim_meta="none", trace="auto")
    assert "sim meta" in str(e.value)


def test_run_commit_is_read_from_the_test_history(tmp_path):
    (tmp_path / "test_history.md").write_text(
        "# f1_fresh\n- **Commit**: `ccfe4a5a` — fix(grasp_fj): ...\n")
    assert F.run_commit(tmp_path) == "ccfe4a5a"


def test_run_commit_is_empty_when_there_is_no_history(tmp_path):
    assert F.run_commit(tmp_path) == ""


def test_plan_files_puts_exactly_one_pth_under_nn_so_the_builder_never_has_to_guess():
    plan = F.plan_files(LISTING, checkpoint="best", sim_meta="auto", trace="none")
    assert sum(f.local_rel.startswith("nn/") for f in plan) == 1


def test_plan_files_adds_the_trace_only_when_asked():
    plan = F.plan_files(LISTING, checkpoint="ep:2500", sim_meta="auto", trace="auto")
    assert any(f.local_rel == "trace.npz" for f in plan)


def test_plan_files_requires_both_param_dumps():
    listing = {**LISTING, "params": ["env.yaml"]}
    with pytest.raises(F.FetchError) as e:
        F.plan_files(listing, checkpoint="best", sim_meta="auto", trace="none")
    assert "agent.yaml" in str(e.value)


# ------------------------------------------------------------------ 매니페스트 / 멱등성

def _manifest(tmp_path: Path) -> dict:
    return {"host": "server", "remote_dir": "/r/t2r_i18", "hdgp_commit": "deadbee",
            "files": [{"local_rel": "params/env.yaml", "remote_rel": "params/env.yaml", "sha256": "aa"},
                      {"local_rel": "nn/x.pth", "remote_rel": "nn/x.pth", "sha256": "bb"}]}


def test_manifest_is_current_only_when_local_and_remote_both_match(tmp_path):
    man = _manifest(tmp_path)
    local = {"params/env.yaml": "aa", "nn/x.pth": "bb"}
    remote = {"params/env.yaml": "aa", "nn/x.pth": "bb"}
    assert F.manifest_is_current(man, local, remote) is True
    assert F.manifest_is_current(man, {**local, "nn/x.pth": "cc"}, remote) is False
    assert F.manifest_is_current(man, local, {**remote, "nn/x.pth": "cc"}) is False
    assert F.manifest_is_current(man, {"params/env.yaml": "aa"}, remote) is False


def test_manifest_is_current_is_false_when_the_plan_asks_for_more_than_it_covers():
    """trace 를 추가로 요청했는데 매니페스트가 4개짜리 그대로면 '최신'이 아니다."""
    man = _manifest(Path("."))
    local = {"params/env.yaml": "aa", "nn/x.pth": "bb"}
    assert F.manifest_is_current(man, local, dict(local), want=["params/env.yaml", "nn/x.pth"]) is True
    assert F.manifest_is_current(man, local, dict(local), want=["params/env.yaml", "nn/x.pth",
                                                                "trace.npz"]) is False


def test_manifest_is_current_without_remote_hashes_checks_the_local_side_only():
    """서버에 못 붙었을 때 — 내려받은 것이 온전한지만 본다."""
    man = _manifest(Path("."))
    assert F.manifest_is_current(man, {"params/env.yaml": "aa", "nn/x.pth": "bb"}, None) is True
    assert F.manifest_is_current(man, {"params/env.yaml": "aa", "nn/x.pth": "zz"}, None) is False


# ------------------------------------------------------------------ 러너 주입 (ssh/rsync 대역)

class FakeRunner:
    """ssh/rsync 대역. 실행된 명령을 기록하고 미리 정한 출력을 돌려준다."""

    def __init__(self, listing=LISTING, hashes=None, commit="deadbee", fail=False):
        self.listing, self.hashes, self.commit, self.fail = listing, hashes or {}, commit, fail
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if self.fail:
            raise F.RemoteError("ssh: Could not resolve hostname nosuch")
        joined = " ".join(argv)
        if "rev-parse" in joined:
            return self.commit
        if "sha256sum" in joined:
            return "\n".join(f"{h}  {p}" for p, h in self.hashes.items())
        if "find" in joined or "ls" in joined:
            out = [f"{n}" for n in self.listing[""]]
            out += [f"nn/{n}" for n in self.listing["nn"]]
            out += [f"params/{n}" for n in self.listing["params"]]
            return "\n".join(out)
        return ""


def test_dry_run_writes_nothing_and_never_calls_rsync(tmp_path):
    r = FakeRunner()
    rc = F.main(["--run", "t2r_i18", "--checkpoint", "ep:2500", "--out", str(tmp_path / "pour_i18"),
                 "--dry-run"], runner=r)
    assert rc == 0
    assert not (tmp_path / "pour_i18").exists()
    assert not any(c[0] == "rsync" for c in r.calls)


def test_list_writes_nothing_and_shows_the_checkpoints(tmp_path, capsys):
    r = FakeRunner()
    rc = F.main(["--run", "t2r_i18", "--list", "--out", str(tmp_path / "pour_i18")], runner=r)
    assert rc == 0
    out = capsys.readouterr().out
    assert "2500" in out and "36408.105" in out
    assert not (tmp_path / "pour_i18").exists()
    assert not any(c[0] == "rsync" for c in r.calls)


def test_offline_with_a_good_manifest_verifies_locally_and_succeeds(tmp_path):
    out = tmp_path / "pour_i18"
    (out / "params").mkdir(parents=True)
    (out / "params" / "env.yaml").write_text("x")
    man = {"host": "server", "remote_dir": "/r", "hdgp_commit": "deadbee",
           "files": [{"local_rel": "params/env.yaml", "remote_rel": "params/env.yaml",
                      "sha256": F.sha256_file(out / "params" / "env.yaml")}]}
    (out / "fetch.json").write_text(json.dumps(man))
    rc = F.main(["--run", "t2r_i18", "--out", str(out)], runner=FakeRunner(fail=True))
    assert rc == 0


def test_offline_without_a_manifest_fails_and_prints_a_copyable_command(tmp_path, capsys):
    rc = F.main(["--run", "t2r_i18", "--out", str(tmp_path / "pour_i18")], runner=FakeRunner(fail=True))
    assert rc == 2
    assert "scp" in capsys.readouterr().out


def test_offline_with_a_corrupt_local_file_fails(tmp_path):
    out = tmp_path / "pour_i18"
    (out / "params").mkdir(parents=True)
    (out / "params" / "env.yaml").write_text("x")
    man = {"host": "server", "remote_dir": "/r", "hdgp_commit": "deadbee",
           "files": [{"local_rel": "params/env.yaml", "remote_rel": "params/env.yaml", "sha256": "00" * 32}]}
    (out / "fetch.json").write_text(json.dumps(man))
    rc = F.main(["--run", "t2r_i18", "--out", str(out)], runner=FakeRunner(fail=True))
    assert rc == 3


def test_the_server_is_only_ever_read(tmp_path):
    """서버에서 도는 명령은 읽기뿐이어야 한다 — 학습 중 호스트에서 쓰는 도구다."""
    r = FakeRunner()
    F.main(["--run", "t2r_i18", "--checkpoint", "ep:2500", "--out", str(tmp_path / "o"), "--dry-run"], runner=r)
    banned = ("rm", "mv", "cp", "tee", "truncate", ">", "python", "train", "nvidia-smi")
    for call in r.calls:
        joined = " ".join(call)
        assert not any(b in joined.split() or f" {b} " in joined for b in banned), joined


# ------------------------------------------------------------------ deploy/policies/ 등록 (09.21)

def _hashes_for(plan):
    return {f.remote_rel: "aa" for f in plan}


class CopyRunner(FakeRunner):
    """rsync 를 실제 파일 쓰기로 흉내낸다 — 해시가 맞아야 _fetch 가 제자리로 옮긴다."""

    BODY = b"x"

    def __call__(self, argv, **kw):
        if argv[0] == "rsync":
            self.calls.append(list(argv))
            Path(argv[-1]).write_bytes(self.BODY)
            return ""
        return super().__call__(argv, **kw)


def _body_hashes(listing, **kw):
    import hashlib
    h = hashlib.sha256(CopyRunner.BODY).hexdigest()
    return {f.remote_rel: h for f in F.plan_files(listing, **kw)}


def test_local_host_runs_bash_not_ssh_and_rsync_has_no_host_prefix(tmp_path):
    kw = dict(checkpoint="ep:2500", sim_meta="none", trace="none")
    r = CopyRunner(hashes=_body_hashes(LISTING, **kw))
    rc = F.main(["--host", "local", "--root", "~/src", "--run", "one", "--checkpoint", "ep:2500",
                 "--sim-meta", "none", "--out", str(tmp_path / "p")], runner=r)
    assert rc == 0
    assert not any(c[0] == "ssh" for c in r.calls)
    assert any(c[:2] == ["bash", "-c"] for c in r.calls)
    srcs = [c[-2] for c in r.calls if c[0] == "rsync"]
    assert srcs and all(":" not in s and not s.startswith("~") for s in srcs), srcs


def test_remote_host_still_uses_ssh_and_a_host_prefix(tmp_path):
    kw = dict(checkpoint="ep:2500", sim_meta="none", trace="none")
    r = CopyRunner(hashes=_body_hashes(LISTING, **kw))
    assert F.main(["--run", "one", "--checkpoint", "ep:2500", "--sim-meta", "none",
                   "--out", str(tmp_path / "p")], runner=r) == 0
    assert any(c[0] == "ssh" for c in r.calls)
    assert all(c[-2].startswith("server:") for c in r.calls if c[0] == "rsync")


def test_default_destination_is_the_policies_folder():
    assert F.POLICIES.name == "policies" and F.POLICIES.parent == Path(F.__file__).resolve().parents[2]
    r = FakeRunner()
    F.main(["--run", "t2r_i18", "--checkpoint", "ep:2500", "--dry-run"], runner=r)      # 아무것도 쓰지 않는다
    assert not (F.POLICIES / "t2r_i18").exists()


def test_card_stub_is_written_once_and_never_overwritten(tmp_path):
    out = tmp_path / "p"
    kw = dict(checkpoint="ep:2500", sim_meta="none", trace="none")
    argv = ["--run", "one", "--checkpoint", "ep:2500", "--sim-meta", "none", "--out", str(out)]
    assert F.main(argv, runner=CopyRunner(hashes=_body_hashes(LISTING, **kw))) == 0
    card = out / F.CARD
    assert "id: p" in card.read_text() and "status: candidate" in card.read_text()
    card.write_text("id: p\nstatus: deployed\n")
    assert F.main(argv + ["--force"], runner=CopyRunner(hashes=_body_hashes(LISTING, **kw))) == 0
    assert card.read_text() == "id: p\nstatus: deployed\n"


def test_source_readme_is_carried_when_present():
    listing = {**LISTING, "": [*LISTING[""], "README.md"]}
    plan = F.plan_files(listing, checkpoint="ep:2500", sim_meta="none", trace="none")
    assert F.FetchFile("README.md", F.NOTES_LOCAL) in plan
    assert not any(f.remote_rel == "README.md"
                   for f in F.plan_files(LISTING, checkpoint="ep:2500", sim_meta="none", trace="none"))
