#!/bin/bash
# ★11.18 글로벌 D435i(camera_up.sh global · /global_camera/camera/*)로 두 번째 FP++ 묶음 컨테이너 fppg_<group> 를 띄운다.
#   머리 쪽(fpp_<group> · /perception_plus_plus/* · /objects/*)은 건드리지 않는다. 출력은 S2R_FPP_NS 아래
#   (기본 /perception_plus_plus_global/<물체>/{pose,live_pose} · /perception_plus_plus_global/snapshot/{status,cmd}).
# usage: fpp_global_up.sh <group> [extrinsics_yaml|-] [object ...]
#   group = config/objects.yaml · objects.d 의 fpp.group(source240_blue · source240_pink · source240_orange · shaker_c …)
#   extrinsics_yaml = 글로벌 카메라 base←카메라(주면 작업 영역 밖 후보를 버린다, '-' 또는 생략이면 거르지 않는다)
#   object = 그 묶음 중 등록할 물체만(생략이면 묶음 전부)
# 내리기: fpp_down.sh fppg_<group>   (fpp_down.sh all 은 '^fpp_' 만 내리므로 이 컨테이너는 남는다)
source "$(dirname "$0")/common.sh"
GROUP=${1:?group}; EXT=${2:--}; shift; shift || true
export S2R_FPP_NS="${S2R_FPP_NS:-/perception_plus_plus_global}"
[ "$EXT" = - ] || [ -f "$EXT" ] || { echo "extrinsics missing: $EXT" >&2; exit 1; }
YAML="$LOGDIR/group_global_$GROUP.yaml"
python3 - "$GROUP" "$EXT" "$@" >"$YAML" <<'EOF'
import os, sys
sys.path.insert(0, os.environ["SIM2REAL"] + "/scripts")
from object_registry import load_registry, render_group_yaml
group, ext, only = sys.argv[1], sys.argv[2], sys.argv[3:]
print(render_group_yaml(load_registry(), group, camera_yaml=None if ext == "-" else ext, only=only or None,
                        camera_ns="/global_camera/camera",
                        stand_top_from_rgb=True, footprint_pad=0.04), end="")
EOF
exec bash "$(dirname "$0")/fpp_group_up.sh" "$GROUP" "$YAML" "fppg_$GROUP"
