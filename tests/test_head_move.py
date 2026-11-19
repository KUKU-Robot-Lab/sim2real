import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import head_move  # noqa: E402


def test_extrinsics_cmd_deg_prefers_provenance_then_ticks_and_none_when_missing(tmp_path):
    p = tmp_path / "live.yaml"
    assert head_move.extrinsics_cmd_deg(p) is None
    p.write_text("head_pose: {pan_tick: 1811, tilt_tick: 2834}\nprovenance: {cmd_deg: [-20.856, 69.055]}\n")
    assert head_move.extrinsics_cmd_deg(p) == (-20.856, 69.055)
    p.write_text("head_pose: {pan_tick: 2048, tilt_tick: 3072}\n")
    assert head_move.extrinsics_cmd_deg(p) == (0.0, 90.0)
