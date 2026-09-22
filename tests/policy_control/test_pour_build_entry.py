"""contract_build dispatch for the pour_bimanual family + the CLI (--sim-meta is required)."""
import importlib.util
import json
from pathlib import Path

import pytest

from policy_control import contract_build as B
from policy_control.pour_contract import FAMILY, PourContract, PourContractError, load_contract
from pour_trace_util import FIX, contract

TOOL = Path(__file__).resolve().parents[2] / "policy_control/tools/build_deploy_contract.py"


def _tool():
    spec = importlib.util.spec_from_file_location("build_deploy_contract", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_detect_family_recognises_the_pour_dump():
    assert B.detect_family(FIX / "params/env.yaml") == FAMILY == "pour_bimanual"


def test_build_contract_refuses_the_pour_family_and_names_the_entry_point():
    with pytest.raises(PourContractError, match="build_pour"):
        B.build_contract(FIX)


def test_build_pour_equals_the_direct_builder_with_the_asset_taken_from_the_dump():
    c = B.build_pour(FIX, FIX / "trace_meta.json")
    assert isinstance(c, PourContract) and c == contract()


def test_build_pour_requires_an_existing_sim_meta(tmp_path):
    with pytest.raises(PourContractError, match="--trace_out"):
        B.build_pour(FIX, None)
    with pytest.raises(PourContractError, match="--trace_out"):
        B.build_pour(FIX, tmp_path / "nope.json")


def test_build_pour_rejects_an_unknown_asset():
    with pytest.raises(PourContractError, match="asset"):
        B.build_pour(FIX, FIX / "trace_meta.json", asset="not_an_asset")


def test_cli_writes_a_loadable_pour_contract(tmp_path):
    out = tmp_path / "deploy_contract.json"
    rc = _tool().main(["--run", str(FIX), "--sim-meta", str(FIX / "trace_meta.json"), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["family"] == FAMILY
    assert load_contract(out) == contract()


def test_cli_without_sim_meta_fails_clearly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _tool().main(["--run", str(FIX), "--out", str(tmp_path / "c.json")])
    assert "--sim-meta" in str(exc.value)


def test_cli_default_output_is_pour_contract_json_next_to_the_run(tmp_path):
    import shutil
    run = tmp_path / "run"
    shutil.copytree(FIX / "params", run / "params")
    rc = _tool().main(["--run", str(run), "--sim-meta", str(FIX / "trace_meta.json")])
    assert rc == 0
    assert not (run / "deploy_contract.json").exists()  # that name belongs to the single-arm DeployContract schema
    assert json.loads((run / "pour_contract.json").read_text())["family"] == FAMILY


def _run_with_ckpts(tmp_path, names):
    import shutil
    run = tmp_path / "run"
    shutil.copytree(FIX / "params", run / "params")
    (run / "nn").mkdir()
    for n in names:
        (run / "nn" / n).write_bytes(b"not-a-real-checkpoint:" + n.encode())
    return run


def test_build_pour_picks_the_only_checkpoint_in_nn(tmp_path):
    run = _run_with_ckpts(tmp_path, ["pour.pth"])
    c = B.build_pour(run, FIX / "trace_meta.json")
    assert c.checkpoint == str(run / "nn/pour.pth") and c.checkpoint_md5


def test_build_pour_refuses_to_guess_between_checkpoints(tmp_path):
    run = _run_with_ckpts(tmp_path, ["a.pth", "b.pth"])
    with pytest.raises(SystemExit) as exc:
        B.build_pour(run, FIX / "trace_meta.json")
    assert "--checkpoint" in str(exc.value)
    c = B.build_pour(run, FIX / "trace_meta.json", checkpoint=run / "nn/b.pth")
    assert c.checkpoint.endswith("b.pth")
