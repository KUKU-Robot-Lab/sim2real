"""tools/contract_doc.py: the pour_bimanual section is generated, so it survives regeneration."""
import importlib.util
from pathlib import Path

import pytest

from policy_control.pour_build import FILL_NOTE
from policy_control.pour_contract import save_contract
from pour_trace_util import contract

SIM2REAL = Path(__file__).resolve().parents[2]
TOOL = SIM2REAL / "policy_control/tools/contract_doc.py"
RIGHT_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"
HAND_DOC = "RUNBOOK_pour_bimanual.md"


def _tool():
    spec = importlib.util.spec_from_file_location("contract_doc", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def pour_json(tmp_path):
    path = tmp_path / "pour_contract.json"
    save_contract(contract(), path)
    return path


def test_render_pour_lists_layout_sides_and_fabric(pour_json):
    text = _tool().render_any(pour_json)
    assert "pour_bimanual" in text and "open-short_b_pour_fab" in text
    assert "obs 223" in text and "action 18" in text
    for cls in ("OpenArmTeoslloPoseFabric", "OpenArmTeoslloLeftPoseFabric"):
        assert cls in text
    for seg in ("arm_qd", "joint_err", "fill_level", "prev_actions"):
        assert seg in text
    assert "| src |" in text and "| rcv |" in text


def test_render_pour_segment_offsets_end_at_obs_dim(pour_json):
    text = _tool().render_any(pour_json)
    c = contract()
    assert f"[{c.obs_dim - c.action_dim}:{c.obs_dim}]" in text


def test_fill_level_means_the_source_cup(pour_json):
    assert "SOURCE cup" in FILL_NOTE and "receiver" not in FILL_NOTE.lower()
    assert FILL_NOTE in _tool().render_any(pour_json)


def test_main_writes_pour_section_and_runbook_link(pour_json, tmp_path):
    out = tmp_path / "doc.md"
    mod = _tool()
    assert mod.main([str(pour_json), "--out", str(out)]) == 0
    text = out.read_text()
    assert "pour_bimanual" in text and HAND_DOC in text
    assert (SIM2REAL / "docs" / HAND_DOC).is_file()


@pytest.mark.skipif(not RIGHT_JSON.exists(), reason="right_g1 contract not present")
def test_single_arm_render_is_unchanged_and_mixes_with_pour(pour_json, tmp_path):
    mod = _tool()
    from policy_control import contract as C
    # 제목은 계약 **파일** 로 구분한다 — 같은 런의 변종 계약과 asset 계약들이 겹치지 않게.
    src = mod._rel(RIGHT_JSON)
    single = mod.render(C.load_contract(RIGHT_JSON), src)
    assert mod.render_any(RIGHT_JSON) == single
    assert single.splitlines()[0].endswith(f"`{src}`")
    out = tmp_path / "doc.md"
    assert mod.main([str(RIGHT_JSON), str(pour_json), "--out", str(out)]) == 0
    text = out.read_text()
    assert single in text and "pour_bimanual" in text
