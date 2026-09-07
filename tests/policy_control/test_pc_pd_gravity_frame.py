"""pd 중력 페이로드는 **체인 마지막 링크 프레임**으로 들어가야 한다 (ROS 없음).

2026-09-07 우팔 실기: pd 의 `model_tau_ff` 로 팔을 든 자세를 넘기자 j7 이 33.7° 내려앉아
손이 테이블에 닿았다. 원인은 프레임 불일치다 —

  · `pd_dg5f_m.yaml` 의 payload COM 은 **palm_ee 프레임** 값이다(주석에 그렇게 적혀 있다).
  · `chain_from_urdf(tip=…palm_ee)` 는 고정 링크를 마지막 **움직이는** 링크(`r_al_7`)로 접는다.
  · `with_payload(chain, mass, com)` 은 그 com 을 **마지막 링크 프레임**에서 해석한다.

⇒ 손목→손바닥의 고정 변환 [0.028, 0, 0.1633] + z축 −90° 가 통째로 빠져, 손 질량이
   손목에서 4.8 cm 지점에 얹혔다(실제 17.0 cm). 모멘트암 3.1배 축소 → 손목 τ_ff 가
   검증된 모델의 38~40 % 밖에 안 나왔다(실기 실측 대조).
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from policy_control import pd_gravity as G
from policy_control.contract import load_contract
from policy_control.pd_law import load_pd_config

from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[2]
CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
PD_YAML = SIM2REAL / "policy_control/config/pd_dg5f_m.yaml"
needs = pytest.mark.skipif(not (CONTRACT.exists() and PD_YAML.exists()), reason="계약/pd 설정 없음")

# 2026-09-07 실기 실측: 검증된 gravity_comp_node 가 이 자세에서 실제로 송출한 τ [N·m]
REAL_POSE_DEG = [-11.25, 14.26, -3.25, 98.74, -2.33, 5.58, -24.71]
REAL_TAU = np.array([7.76, 1.67, 2.24, 9.54, -0.05, -0.78, 2.74])


def _chain(side: str):
    cfg = load_pd_config(PD_YAML)
    blk = G.block_for_side(cfg.gravity, side)
    contract = load_contract(CONTRACT)
    return G.make_gravity(blk, contract, n_joints=7, side=side)


@needs
@pytest.mark.parametrize("side", ["right", "left"])
def test_hand_mass_sits_a_hand_length_from_the_wrist(side):
    model = _chain(side)
    com = np.asarray(model.chain.links[-1].com, dtype=float)
    reach = float(np.linalg.norm(com))
    assert reach > 0.12, (
        f"{side}: 마지막 링크 무게중심이 손목에서 {reach*100:.1f} cm 다 — 손이 달린 팔에서 "
        "불가능한 값이다. payload COM 이 palm_ee 프레임 그대로 들어갔는지 확인할 것")


@needs
def test_wrist_gravity_matches_the_measured_validated_model():
    """j6·j7 이 실기에서 무너진 자리 — 검증된 모델과 대조한다."""
    tau = np.asarray(_chain("right")(np.radians(REAL_POSE_DEG)), dtype=float)
    for j in (5, 6):                                     # j6, j7 (0-based)
        ratio = abs(tau[j]) / abs(REAL_TAU[j])
        assert 0.85 < ratio < 1.2, (
            f"j{j+1}: pd {tau[j]:+.2f} vs 검증모델 {REAL_TAU[j]:+.2f} N·m (비율 {ratio:.2f}) — "
            "손목 중력보상이 어긋난다")


@needs
def test_large_joints_stay_matched():
    """큰 관절은 원래 맞았다 — 고치면서 깨뜨리지 말 것."""
    tau = np.asarray(_chain("right")(np.radians(REAL_POSE_DEG)), dtype=float)
    for j in (0, 2, 3):                                  # j1, j3, j4
        ratio = abs(tau[j]) / abs(REAL_TAU[j])
        assert 0.85 < ratio < 1.2, f"j{j+1}: pd {tau[j]:+.2f} vs {REAL_TAU[j]:+.2f} (비율 {ratio:.2f})"


@needs
def test_payload_is_only_the_links_beyond_a_movable_joint():
    """★이중 계상 금지. 손 전체(1.763 kg) 중 mount·adapter·base·palm 0.889 kg 은 고정 링크라
    체인이 이미 싣는다. payload 는 가동 관절 너머 손가락(0.8737 kg)뿐이어야 한다."""
    cfg = load_pd_config(PD_YAML)
    for side in ("left", "right"):
        mass = float(G.block_for_side(cfg.gravity, side).payload[0])
        assert 0.80 < mass < 0.95, (
            f"{side}: payload {mass} kg — 손 전체(1.763)를 얹으면 고정 링크 0.889 kg 을 "
            "두 번 세어 중력토크가 1.3배가 된다")


@needs
def test_fake_config_payload_matches_the_real_one():
    """fake 로 검증한 것이 실기와 같은 중력이어야 한다."""
    real = load_pd_config(PD_YAML)
    fake = load_pd_config(SIM2REAL / "policy_control/config/pd_dg5f_m_fake.yaml")
    for side in ("left", "right"):
        a = tuple(float(v) for v in G.block_for_side(real.gravity, side).payload)
        b = tuple(float(v) for v in G.block_for_side(fake.gravity, side).payload)
        assert a == b, f"{side}: 실기 {a} ≠ fake {b}"
