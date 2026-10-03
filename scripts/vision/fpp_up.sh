#!/bin/bash
# 물체 하나의 FP++ 컨테이너 기동. usage: fpp_up.sh <name> <yaml_host_path>
# 이미지는 baked 라 패치 파일은 파일 단위 바인드 마운트(09.02 규약). 이름 fpp_<name>.
source "$(dirname "$0")/common.sh"
NAME=${1:?name}; YAML=${2:?yaml path}
[ -f "$YAML" ] || { echo "yaml missing: $YAML" >&2; exit 1; }
YAML="$(realpath "$YAML")"          # docker -v 는 절대 경로만 받는다 — 런처는 홈 기준 상대 경로를 넘긴다
docker rm -f "fpp_$NAME" >/dev/null 2>&1 || true
# 10.01: 컨테이너가 뜰 때마다 resnet50/18(약 140 MB)을 내려받았다(이미지에 없음, 캐시가 컨테이너와 함께 사라짐) — 오프라인이면 FP++ 가
#  못 뜬다. 호스트 캐시를 붙인다. 컨테이너 사용자 perception 은 uid 1000 = 호스트 사용자(arm4090 · vision-3090 같음).
CACHE="${FPP_CACHE:-$HOME/.cache/fpp}"
mkdir -p "$CACHE/torch" "$CACHE/warp"
# ★10.03 FP++ 는 CPU 를 코어 3.5 개쯤 쓴다(홀더 하나 추적, arm4090 실측) — EtherCAT 마스터 실시간 코어와 그 형제 스레드를
#  비켜 간다. 코어 목록은 이 PC 에서 cpu_plan 이 정한다(고정 안 하는 PC 면 빈 값 → 제한 없음). 끄기 S2R_CPU_PIN=0.
GENERAL_CPUS="$(python3 "$SIM2REAL/deploy/policy_control/policy_control/cpu_plan.py" --general 2>/dev/null || true)"
CPUSET=()
[ -n "$GENERAL_CPUS" ] && CPUSET=(--cpuset-cpus "$GENERAL_CPUS")
docker run -d --name "fpp_$NAME" --network host --ipc=host --gpus all "${CPUSET[@]}" -e ROS_DOMAIN_ID=126 -e ROS_LOCALHOST_ONLY=1 \
  -v $PPP/perception_plus_plus_core/detection/yolo.py:/workspace/perception_plus_plus/perception_plus_plus_core/detection/yolo.py:ro \
  -v $PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:/workspace/perception_plus_plus/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:ro \
  -v $PPP/ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/node.py:/opt/perception_plus_plus/lib/python3.10/site-packages/perception_plus_plus_ros/node.py:ro \
  -v $PPP/assets/meshes:/workspace/perception_plus_plus/assets/meshes:ro \
  -v $SIM2REAL/assets/meshes:/workspace/perception_plus_plus/assets/s2r_meshes:ro \
  -v "$YAML":/opt/params/"$NAME".yaml:ro \
  -v "$CACHE/torch":/home/perception/.cache/torch -v "$CACHE/warp":/home/perception/.cache/warp \
  perception-plus-plus:humble-cup bash -lc "
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    cd /workspace/perception_plus_plus
    exec ros2 launch perception_plus_plus_ros cup_tracking.launch.py parameters_file:=/opt/params/$NAME.yaml" \
  >/dev/null
echo "fpp_$NAME up${GENERAL_CPUS:+ · cpuset $GENERAL_CPUS}"
