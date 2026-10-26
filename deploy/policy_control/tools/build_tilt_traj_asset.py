#!/usr/bin/env python3
"""tilt 1차(tilt_traj 궤적, 잔차 0) 배포 자산을 sim 과 같은 입력으로 만든다 — server(hdgp · IsaacLab venv)에서 한 번 돌린다.

10.26 learn:e2e-aglt 요청(grasp2tilt, hdgp 24bfd4dc). 실기 계획기 plan_tilt_traj.py 는 Isaac 없이 hdgp tilt_traj 만으로
궤적을 푼다. 그때 필요한 '파지와 무관한 고정 입력'(TiltSetup)을 sim Context 에서 그대로 꺼내 npz 로 둔다.
  · q0 = 붓기 시작 자세(rh56f1_pour_start.yaml) · th_tab/q_tab = D4 기울기별 오른팔 관절 표(single734_t120 · collect_q_A_g25)
  · mouth_ref = CMA 가 최종 파라미터를 찾을 때 쓴 기준점(= tilt_rl/v9/cma<컵>/grasps 파지 0 의 받는 컵 입구).
    sim 은 '--grasps 폴더 파지 0' 으로 정해서 실행마다 다르다(grasp2tilt 491 mm · v10 뱅크 506~508 mm · CMA 501~503 mm, z).
    두 컵이 같이 옮겨지는 값이라 컵 사이 관계는 같고 월드 붓기 자리만 움직인다. 배포는 CMA 값으로 고정한다.
  · T_in 기본값 = e2e 잡기→이송 짝(grasp2tilt grasps<컵>) 중 손마다 중앙값에 가장 가까운 짝(실측 파지가 없을 때 쓴다).
  · ref_* = 위 입력 + 홀더 yaml 로 푼 기준 궤적(계획기 동등성 검사용).

    PYTHONPATH=source/openarm:scripts/tools/tilt_newton ~/rl_ws/IsaacLab3/.venv/bin/python <이 파일> --out /tmp/traj_v10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HOME = os.path.expanduser("~")
I = f"{HOME}/rl_ws/videos/pour/_work/cma_ts/inputs"
TR = f"{HOME}/rl_ws/our_source/tilt_rl"
G2T = f"{HOME}/rl_ws/our_source/rh_aglt_runs/eval/1026_grasp2tilt"
CUPS = {
    "240": {"params": f"{TR}/v10/cma240/final_params.json", "cma_grasps": f"{TR}/v9/cma240/grasps", "e2e_grasps": f"{G2T}/grasps240"},
    "200": {"params": f"{TR}/v10b/final_params_200.json", "cma_grasps": f"{TR}/v9/cma200/grasps", "e2e_grasps": f"{G2T}/grasps200"},
}
SRC_MOUTH_Z = {"240": 0.155, "200": 0.115}   # tilt_newton_batch.SRC
HDGP_FILES = (
    "source/openarm/openarm/agnostic/modules/tilt_traj/__init__.py",
    "source/openarm/openarm/agnostic/modules/tilt_traj/plan.py",
    "source/openarm/openarm/agnostic/modules/tilt_traj/traj.py",
    "source/openarm/openarm/agnostic/modules/tilt_traj/clearance.py",
    "source/openarm/openarm/agnostic/modules/tilt_traj/fast_ik.py",
    "source/openarm/openarm/agnostic/modules/ready_carry/__init__.py",
    "source/openarm/openarm/agnostic/modules/ready_carry/chain.py",
    "source/openarm/openarm/agnostic/modules/ready_carry/posture.py",
    "assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf",
    "scripts/tools/tilt_newton/tilt_primitive_ts.py",
    "scripts/tools/tilt_newton/tilt_gpu.py",
    "scripts/tools/tilt_newton/holder_clear.py",
)


def md5(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def tin_feat(T: np.ndarray) -> np.ndarray:
    """T_in (N,4,4) → (N,6) = 위치[m] + 0.05 × 회전 벡터(hdgp e2e_chain/grasp2tilt.tin_feat 와 같은 척도)."""
    from scipy.spatial.transform import Rotation
    return np.concatenate([T[:, :3, 3], 0.05 * Rotation.from_matrix(T[:, :3, :3]).as_rotvec()], -1)


def nominal_pair(Tr, Tl) -> int:
    """손마다 T_in 특징 중앙값까지 거리 합이 가장 작은 짝."""
    d = 0.0
    for T in (Tr, Tl):
        f = tin_feat(T.double().numpy())
        d = d + np.linalg.norm(f - np.median(f, 0), axis=1)
    return int(np.argmin(d))


def main() -> int:
    global HDGP
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--hdgp", default=f"{HOME}/rl_ws/hdgp")
    ap.add_argument("--holders", default=f"{I}/cup_holder_poses_5090_v1.yaml", help="기준 궤적용(sim grasp2tilt 와 같은 옛 배치)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    HDGP = os.path.abspath(a.hdgp)
    sys.path[:0] = [os.path.join(HDGP, "source", "openarm"), os.path.join(HDGP, "scripts", "tools", "tilt_newton")]
    import torch
    import tilt_gpu as TG
    import tilt_primitive_ts as TS
    from openarm.agnostic.modules.tilt_traj import build_batch, params_tensor

    commit = subprocess.run(["git", "-C", HDGP, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    os.makedirs(a.out, exist_ok=True)
    meta = {"tool": "sim2real build_tilt_traj_asset.py", "hdgp_commit": commit,
            "hdgp_md5": {f: md5(os.path.join(HDGP, f)) for f in HDGP_FILES},
            "inputs_md5": {os.path.basename(p): md5(p) for p in (f"{I}/rh56f1_pour_start.yaml", a.holders,
                                                                 f"{I}/d4/single734_t120.npz", f"{I}/d4/collect_q_A_g25.npz")},
            "cups": {}}
    for cup, c in CUPS.items():
        kw = dict(start_yaml=f"{I}/rh56f1_pour_start.yaml", holders_yaml=a.holders, d4_dir=f"{I}/d4",
                  geo=TS.Geo(src_mouth_z=SRC_MOUTH_Z[cup]))
        g = c["cma_grasps"]
        ctx = TS.Context(f"{g}/trace_r.npz", 0, 0, f"{g}/trace_l.npz", 0, 0, **kw)
        s = TG.setup_from_context(ctx, "cpu")
        s.fast, s.cuda_graph = False, False
        Tr, Tl = TG.grasps_T_in(ctx, c["e2e_grasps"])
        k = nominal_pair(Tr, Tl)
        p = json.load(open(c["params"]))
        params = TS.fill(p.get("params", p))
        tb = build_batch(s, params_tensor([params], s.device), Tr[k:k + 1].double(), Tl[k:k + 1].double())
        L = int(tb.length[0])
        geo = {f: getattr(s.geo, f) for f in s.geo.__dataclass_fields__}
        clear = {f: getattr(s.clear, f) for f in s.clear.__dataclass_fields__}
        np.savez(os.path.join(a.out, f"setup_{cup}.npz"),
                 q0_r=s.q0_r.numpy(), q0_l=s.q0_l.numpy(), mouth_ref=s.mouth_ref.numpy(),
                 th_tab=s.th_tab.numpy(), q_tab=s.q_tab.numpy(),
                 T_in_r=Tr[k].double().numpy(), T_in_l=Tl[k].double().numpy(),
                 ref_holders=s.holders.numpy(), ref_q_r=tb.q_r[0, :L].numpy(), ref_q_l=tb.q_l[0, :L].numpy(),
                 ref_edges=tb.knots.edges[0].numpy(),
                 geo_json=json.dumps(geo), clear_json=json.dumps(clear))
        with open(os.path.join(a.out, f"params_{cup}.json"), "w") as fh:
            json.dump({"src": cup, "params": params, "from": c["params"], "from_md5": md5(c["params"])}, fh, indent=1)
        info = {n: round(float(v[0]), 4) for n, v in tb.info.items()}
        meta["cups"][cup] = {"params_from": c["params"], "params_md5": md5(c["params"]), "mouth_ref_from": f"{g} 파지 0",
                             "mouth_ref_mm": [round(float(x) * 1000, 1) for x in s.mouth_ref],
                             "T_in_from": f"{c['e2e_grasps']} 짝 {k}", "knots": L, "T_s": round(L * s.geo.dt, 2),
                             "edges_s": [round(float(x), 3) for x in tb.knots.edges[0]], "ref_info": info}
        print(f"[asset] {cup}: 짝 {k} · knot {L} ({L * s.geo.dt:.1f} s) · mouth_ref {meta['cups'][cup]['mouth_ref_mm']} mm · {info}", flush=True)
    with open(os.path.join(a.out, "asset_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
