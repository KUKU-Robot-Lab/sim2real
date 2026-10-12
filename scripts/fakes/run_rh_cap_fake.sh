#!/usr/bin/env bash
# rh_cap_node fake 리허설(도메인 97) — $1 = grasp|seat
set +u
PH=$1
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=97
cd "$(dirname "$0")/../.."
C=tests/policy_control/fixtures/rh_cap_${PH}_contract.json
LOG=/tmp/capdeploy/fake_${PH}
mkdir -p $LOG
python3 scripts/fakes/fake_rh_cap_plant.py --contract $C --arm-q=-0.6,0.2026,0.6538,1.7608,0.3791,0.2,0.6646 --grip-close ${GRIP:-0.5} > $LOG/plant.log 2>&1 &
PP=$!
sleep 3
${PY:-.venv/bin/python} deploy/policy_control/policy_control/rh_cap_node.py --ros-args -r __node:=rh_cap_node_fake -p ns:=right_cap \
  -p contract:=$C -p robot:=deploy/policy_control/config/robots/rh56f1_right_fake.yaml -p device:=cpu \
  -p allow_measured_start:=true > $LOG/node.log 2>&1 &
NP=$!
sleep 12
echo "== reset"; timeout 60 ros2 service call /policy_control/right_cap/episode/reset std_srvs/srv/Trigger "{}" 2>&1 | tail -2
echo "== start"; timeout 30 ros2 service call /policy_control/right_cap/episode/start std_srvs/srv/Trigger "{}" 2>&1 | tail -2
sleep 15
echo "== status"; timeout 5 ros2 topic echo --once /policy_control/status/rh_cap_node_fake 2>&1 | head -5
sleep 2
kill $NP $PP 2>/dev/null; wait 2>/dev/null
echo "== node log"; grep -v "^$" $LOG/node.log | tail -15
echo "== plant log"; tail -6 $LOG/plant.log
