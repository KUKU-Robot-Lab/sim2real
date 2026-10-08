#!/bin/bash
# 묶음 FP++ 컨테이너 하나 — 같은 모양 · 다른 색 물체 여럿을 한 번 찍는다(10.08 사용자, scripts/nodes/fpp_snapshot_node.py).
# usage: fpp_group_up.sh <group> <yaml_host_path>   (yaml = object_registry.render_group_yaml, 인지 런처가 쓴다)
# 이미지 · 마운트 · CPU 배치는 fpp_up.sh 와 같다. 다른 점: 추적 노드 대신 sim2real scripts 를 /opt/s2r/scripts 에 붙여
# 한 번 찍기 노드를 돌린다(perception_plus_plus 저장소는 고치지 않는다). 이름 fpp_<group>.
source "$(dirname "$0")/common.sh"
GROUP=${1:?group}; YAML=${2:?yaml path}
[ -f "$YAML" ] || { echo "yaml missing: $YAML" >&2; exit 1; }
YAML="$(realpath "$YAML")"
docker rm -f "fpp_$GROUP" >/dev/null 2>&1 || true
CACHE="${FPP_CACHE:-$HOME/.cache/fpp}"
mkdir -p "$CACHE/torch" "$CACHE/warp"
GENERAL_CPUS="$(python3 "$SIM2REAL/deploy/policy_control/policy_control/cpu_plan.py" --general 2>/dev/null || true)"
CPUSET=()
[ -n "$GENERAL_CPUS" ] && CPUSET=(--cpuset-cpus "$GENERAL_CPUS")
docker run -d --name "fpp_$GROUP" --network host --ipc=host --gpus all "${CPUSET[@]}" -e ROS_DOMAIN_ID=126 -e ROS_LOCALHOST_ONLY=1 \
  -v $PPP/perception_plus_plus_core/detection/yolo.py:/workspace/perception_plus_plus/perception_plus_plus_core/detection/yolo.py:ro \
  -v $PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:/workspace/perception_plus_plus/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:ro \
  -v $PPP/assets/meshes:/workspace/perception_plus_plus/assets/meshes:ro \
  -v $SIM2REAL/assets/meshes:/workspace/perception_plus_plus/assets/s2r_meshes:ro \
  -v $SIM2REAL/scripts:/opt/s2r/scripts:ro \
  -v "$YAML":/opt/params/group_"$GROUP".yaml:ro \
  -v "$CACHE/torch":/home/perception/.cache/torch -v "$CACHE/warp":/home/perception/.cache/warp \
  perception-plus-plus:humble-cup bash -lc "
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    cd /workspace/perception_plus_plus
    exec python3 /opt/s2r/scripts/nodes/fpp_snapshot_node.py --config /opt/params/group_$GROUP.yaml" \
  >/dev/null
echo "fpp_$GROUP up${GENERAL_CPUS:+ · cpuset $GENERAL_CPUS}"
