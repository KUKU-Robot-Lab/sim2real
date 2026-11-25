#!/usr/bin/env python3
"""잰 치수로 병(회전체) CAD 를 만든다 — CAD 가 없는 시판 병을 FP++ 에 등록할 때 fpp_object.py add 의 입력.

    python3 scripts/ops/bottle_mesh.py powerade600 <out.stl>
    python3 scripts/ops/fpp_object.py add <out.stl> --name powerade --colors blue --origin-above-bottom 0.085 --units m --class 39
"""
import sys

import numpy as np
import trimesh

#: 병 → 옆선 (바닥에서 높이 m, 반지름 m), 아래부터
PROFILES = {
    # ★11.18 파워에이드 600 ml — 사용자 실측: 높이 22 cm · 가장 큰 지름 6.5 cm · 허리 5.3 cm.
    #   허리 · 라벨 · 어깨 · 뚜껑의 높이는 사진으로 어림했다(허리 가운데 바닥 위 10 cm) — 실측이 오면 여기만 고친다
    "powerade600": [(0.000, 0.0270), (0.006, 0.0325), (0.045, 0.0325), (0.075, 0.0295), (0.100, 0.0265), (0.120, 0.0295),
                    (0.135, 0.0325), (0.175, 0.0325), (0.190, 0.0240), (0.200, 0.0160), (0.202, 0.0185), (0.220, 0.0185)],
}


def bottle(profile) -> trimesh.Trimesh:
    z, r = np.asarray(profile, float).T
    line = np.column_stack([np.r_[0.0, r, 0.0], np.r_[z[0], z, z[-1]]])     # 바닥 · 뚜껑을 축까지 닫는다
    m = trimesh.creation.revolve(line, sections=48)
    assert m.is_watertight and np.allclose(m.extents, [2 * r.max(), 2 * r.max(), z[-1] - z[0]], atol=1e-4), m.extents
    return m


if __name__ == "__main__":
    bottle(PROFILES[sys.argv[1]]).export(sys.argv[2])
