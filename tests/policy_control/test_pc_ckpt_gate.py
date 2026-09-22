"""체크포인트 게이트: trace 에서 직접 센 숫자로 실기 진입 자격을 판정한다.

합성 trace 로 각 기준이 실제로 무는지 본다. 실물 i18 trace 가 있으면 알려진 판정과도 대조한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "policy_control" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import ckpt_gate as G  # noqa: E402

T, E = 40, 3


def synth(**over):
    """전 항목이 통과하는 기본 trace. 키워드로 한 축만 망가뜨려 그 게이트만 무는지 본다."""
    up = np.zeros((T, E, 3), np.float32)
    up[..., 2] = np.cos(np.radians(over.pop("tilt_deg", 80.0)))
    up[..., 0] = -np.sin(np.radians(over.pop("tilt_deg_x", 80.0)))     # -x 로 붓는다
    src = np.zeros((T, E, 3), np.float32)
    src[..., 1] = -0.16
    src[..., 0] = np.linspace(0, over.pop("travel", 0.10), T)[:, None]
    rcv = np.zeros((T, E, 3), np.float32)
    rcv[..., 1] = 0.16
    z = {"success": np.ones((T, E), np.float32),
         "in_target": np.full((T, E), 0.9, np.float32),
         "spill": np.zeros((T, E), np.float32),
         "src_cup_pos": src, "rcv_cup_pos": rcv, "src_cup_up": up,
         "fill_level": np.full((T, E), 0.6, np.float32)}
    z.update(over)
    return _Z(z)


class _Z(dict):
    """np.load 가 돌려주는 객체 흉내 — `files` 속성이 필요하다."""

    @property
    def files(self):
        return list(self)


def verdict(rows):
    return {r.name: r.ok for r in rows if r.kind == "gate"}


def test_a_clean_run_passes_every_gate():
    assert all(verdict(G.evaluate(synth())).values())


def test_low_success_is_caught():
    z = synth(success=np.zeros((T, E), np.float32))
    assert verdict(G.evaluate(z))["success_ever"] is False


def test_low_transfer_is_caught_even_when_success_is_set():
    z = synth(in_target=np.full((T, E), 0.2, np.float32))
    v = verdict(G.evaluate(z))
    assert v["in_target_max"] is False and v["success_ever"] is True


def test_spill_uses_the_last_row():
    z = synth(spill=np.concatenate([np.zeros((T - 1, E)), np.full((1, E), 0.5)]).astype(np.float32))
    assert verdict(G.evaluate(z))["spill_last"] is False


def test_cups_too_close_is_caught():
    rcv = np.zeros((T, E, 3), np.float32)
    rcv[..., 1] = -0.10                      # 소스(-0.16)와 6 cm
    z = synth(rcv_cup_pos=rcv)
    assert verdict(G.evaluate(z))["min_cup_dist_m"] is False


def test_over_tilt_is_caught():
    assert verdict(G.evaluate(synth(tilt_deg=150.0)))["src_tilt_peak_deg"] is False


def test_tilt_below_the_limit_passes():
    assert verdict(G.evaluate(synth(tilt_deg=100.0)))["src_tilt_peak_deg"] is True


def test_large_excursion_is_caught():
    assert verdict(G.evaluate(synth(travel=0.40)))["src_xy_excursion_m"] is False


def test_pouring_towards_the_body_is_caught():
    z = synth()
    z["src_cup_up"] = z["src_cup_up"].copy()
    z["src_cup_up"][..., 0] *= -1.0          # +x 로 붓는다 = 몸통 쪽
    assert verdict(G.evaluate(z))["pour_dir_neg_x"] is False


def test_crossing_the_centre_line_is_caught():
    src = np.zeros((T, E, 3), np.float32)
    src[..., 1] = np.linspace(-0.16, 0.12, T)[:, None]     # 수신 쪽으로 넘어간다
    assert verdict(G.evaluate(synth(src_cup_pos=src)))["src_crosses_centre"] is False


def test_centre_line_follows_the_receiver_side():
    """수신 컵이 -y 에 있으면 '넘어감'의 부호도 뒤집혀야 한다."""
    src = np.zeros((T, E, 3), np.float32)
    src[..., 1] = np.linspace(0.16, -0.12, T)[:, None]
    rcv = np.zeros((T, E, 3), np.float32)
    rcv[..., 1] = -0.16
    assert verdict(G.evaluate(synth(src_cup_pos=src, rcv_cup_pos=rcv)))["src_crosses_centre"] is False


def test_report_names_every_failing_gate():
    txt = G.report(G.evaluate(synth(tilt_deg=150.0, travel=0.40)))
    assert "탈락" in txt and "src_tilt_peak_deg" in txt and "src_xy_excursion_m" in txt


def test_report_says_pass_when_nothing_bites():
    assert "판정: 통과" in G.report(G.evaluate(synth()))


def test_context_rows_never_decide_the_verdict():
    rows = G.evaluate(synth())
    assert {r.name for r in rows if r.kind == "context"} >= {"envs", "steps"}
    assert all(r.ok for r in rows if r.kind == "context")


I18 = Path(__file__).resolve().parents[2] / "logs/policy/pour_i18/trace.npz"


@pytest.mark.skipif(not I18.is_file(), reason="i18 trace 가 없다 (fetch_run.py --trace auto)")
def test_the_real_i18_trace_is_rejected_for_the_known_three_reasons():
    v = verdict(G.evaluate(np.load(I18)))
    assert v["success_ever"] and v["in_target_max"] and v["spill_last"] and v["min_cup_dist_m"]
    assert not v["src_tilt_peak_deg"], "i18 소스 컵 최대 기울기가 기준 아래로 내려왔다면 기준을 다시 봐야 한다"
    assert not v["src_xy_excursion_m"]
    assert not v["src_crosses_centre"]
