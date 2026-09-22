"""실제로 등록된 pour 런(픽스처가 아니라 `fetch_run.py` 로 받은 것)이 sim 을 재현하는가.

픽스처 i11 과 달리 여기 런은 ADR 를 올린 play 세션이라 **관측 노이즈·지연이 켜져 있다** —
그래서 obs 재구성 파리티는 걸 수 없고(그건 `test_pour_golden.py` 가 ADR0 픽스처로 한다),
노이즈와 무관한 두 가지만 본다:

  1. 계약 + 체크포인트가 trace 의 obs → action 을 재현하는가 (배포 대상 쌍이 맞는가)
  2. 디코더가 trace 의 palm/hand 목표를 재현하는가 (액션 해석이 맞는가)

런 디렉터리가 없으면 통째로 skip — 이 저장소는 143 MB trace 를 담지 않는다.
받는 법: `policy_control/tools/fetch_run.py --run t2r_i18 --checkpoint ep:2500 --trace auto
          --out logs/policy/pour_i18`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_control.pour_decoder import PourDecoder, SideInputs
from policy_control.pour_obs import clip_obs
from pour_trace_util import contract, trace

SIM2REAL = Path(__file__).resolve().parents[2]
RUN = Path(os.environ.get("POUR_RUN_DIR", SIM2REAL / "logs/policy/pour_i18"))
NEEDED = ("trace.npz", "trace_meta.json", "params/agent.yaml", "params/env.yaml")
ENVS = (0, 33, 63)           # 64 env 중 표본 — 전수는 느리기만 하고 새로 걸리는 게 없다
STEPS = 850                  # 에피소드 900, 끝머리 리셋 행을 피한다

pytestmark = pytest.mark.skipif(
    not all((RUN / n).exists() for n in NEEDED) or not list((RUN / "nn").glob("*.pth"))
    if (RUN / "nn").is_dir() else True,
    reason=f"등록된 pour 런이 없다: {RUN} (fetch_run.py 로 받는다)")


@pytest.fixture(scope="module")
def ctx():
    """배포되는 **산출물**(pour_contract.json)을 본다 — 없으면 런 덤프에서 새로 만든다.

    CLI 가 만든 계약에는 체크포인트 경로·md5 가 박혀 있고, 런 덤프에서 직접 빌드한 것에는 없다.
    실기가 읽는 것은 전자다.
    """
    saved = RUN / "pour_contract.json"
    if saved.is_file():
        from policy_control.pour_contract import load_contract
        c = load_contract(saved)
    else:
        c = contract(RUN)
    z, meta = trace(RUN)
    return c, z, meta


def test_the_run_dir_holds_exactly_one_checkpoint(ctx):
    """계약 생성기가 후보를 고르지 않아도 되게 — fetch_run 의 규약이다."""
    assert len(list((RUN / "nn").glob("*.pth"))) == 1


def test_contract_matches_the_deployed_layout(ctx):
    c = ctx[0]
    assert (c.obs_dim, c.action_dim) == (223, 18)
    assert [(s.role, s.side) for s in c.sides] == [("src", "right"), ("rcv", "left")]
    assert c.asset == "openarm_dg5f-m-short_bi_rl"


def test_trace_shape_agrees_with_the_contract(ctx):
    c, z, _ = ctx
    assert z["obs_next"].shape[2] == c.obs_dim
    assert z["actions"].shape[2] == c.action_dim


def test_checkpoint_md5_is_recorded_and_matches_the_fetch_manifest(ctx):
    import json
    man = RUN / "fetch.json"
    if not man.is_file() or not (RUN / "pour_contract.json").is_file():
        pytest.skip("fetch.json 또는 pour_contract.json 이 없다")
    files = json.loads(man.read_text())["files"]
    md5 = {f["local_rel"]: f.get("md5") for f in files}
    ck = next((RUN / "nn").glob("*.pth"))
    assert ctx[0].checkpoint_md5 == md5[f"nn/{ck.name}"]


def test_actor_reproduces_trace_actions(ctx):
    """계약이 만든 obs 클립 + 이 체크포인트 = sim 이 그때 낸 액션."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(SIM2REAL / "scripts"))
    from policy_loader import RLGamesActorPolicy

    c, z, _ = ctx
    ck = next((RUN / "nn").glob("*.pth"))
    policy = RLGamesActorPolicy(str(RUN / "params/agent.yaml"), str(ck), obs_dim=c.obs_dim,
                                action_dim=c.action_dim, device="cpu", action_clip=None)
    worst = 0.0
    for e in ENVS:
        obs = clip_obs(c, z["obs_next"][:STEPS, e])
        mu = policy.get_action(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
        err = np.abs(np.clip(mu, -1, 1) - np.clip(z["actions"][1:STEPS + 1, e], -1, 1)).max()
        worst = max(worst, float(err))
    print(f"[{RUN.name}] actor 재현 max err {worst:.3e}")
    assert worst < 1e-5


def test_decoder_reproduces_trace_targets(ctx):
    """액션 → palm/손 목표 해석이 sim 과 같은가. 관측 노이즈와 무관한 경로다."""
    c, z, _ = ctx
    worst: dict[str, float] = {}
    for e in ENVS:
        dec = PourDecoder(c)
        for t in range(STEPS):
            p = max(t - 1, 0)
            inputs = {s.role: SideInputs(float(z[f"{s.role}_close_gate"][p, e]),
                                         z[f"{s.role}_f_mid"][p, e], z[f"{s.role}_f_dist"][p, e])
                      for s in c.sides}
            out = dec.step(z["actions"][t, e], inputs)
            for i, s in enumerate(c.sides):
                r = s.role
                for k, v, ref in (("palm_cmd", out[r].palm_cmd, z["palm_cmd"][t, e, i]),
                                  ("palm_target", out[r].palm_target, z[f"{r}_palm_tgt"][t, e]),
                                  ("hand_target", out[r].hand_target, z[f"{r}_syn_target"][t, e])):
                    key = f"{r}/{k}"
                    worst[key] = max(worst.get(key, 0.0), float(np.abs(v - ref).max()))
    print(f"[{RUN.name}] decoder max err:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert max(worst.values()) < 1e-5, worst
