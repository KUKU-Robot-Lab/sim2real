#!/usr/bin/env bash
# New friction experiment folder + offline plan (prepare only, sends 0 commands).
# usage: new_experiment.sh <name> <side> <cfg> <stage> <offset_deg> <repeats>
#   e.g. new_experiment.sh 261109_F1off_right right off F1 2.0 3
#   fast: MOTION=sweep|step new_experiment.sh ... (offset = step size; repeats usually 1)
# cfg = off | fs | fs_i050 | fs_i100 | fs_i200  (pd_rh56f1_stationary_<side>_<cfg>.yaml)
set -euo pipefail
[ $# -eq 6 ] || { sed -n 2,5p "$0"; exit 2; }
name=$1 side=$2 cfg=$3 stage=$4 off=$5 rep=$6
cd "$(dirname "$0")/../../../.."   # sim2real
F=analysis/friction; E=$F/$name; I=$F/common/inputs
C=deploy/policy_control/config/stationary_1108/pd_rh56f1_stationary_${side}_${cfg}.yaml
CT=logs/policy/asset_openarm_rh56f1_bi_rl/deploy_contract.json
[ -e "$E" ] && { echo "exists: $E (use a new name, e.g. ${name}_b)"; exit 1; }
[ -f "$C" ] || { echo "no PD config: $C"; exit 1; }
[ -f "$CT" ] || { echo "no contract: $CT"; exit 1; }
if [ "$side" = right ]; then extra="" pose=right_fit_01 scene=$I/scene_right_fit01_head_from_attention.json
else extra=--mirror-right pose=left_mirror_fit_01 scene=$I/right_pose_set_v1/scene_complete_attention_left_home.json; fi
mkdir -p "$E/setup" "$E/runs" "$E/result"
sed -e "s/{{NAME}}/$name/" -e "s/{{SIDE}}/$side/" -e "s/{{STAGE}}/$stage/" -e "s/{{CFG}}/$cfg/" \
    -e "s/{{OFF}}/$off/" -e "s/{{REP}}/$rep/" -e "s/{{DATE}}/$(date '+%F %T')/" $F/_template/README.md > "$E/README.md"
cp "$C" "$CT" "$E/setup/"
{ git rev-parse HEAD; git status --short; } > "$E/setup/git_head.txt"
git diff > "$E/setup/git_diff.patch"
.venv/bin/python deploy/policy_control/tools/stationary_tracking.py prepare \
  --poses $I/right_pose_set_v1/poses.json --scene $scene --pd-config "$C" --contract "$CT" \
  --profile ../robot_control/src/robot_control/profiles/openarm_rh56f1.yaml \
  --env-yaml deploy/policy_control/config/home_path/rh56f1_world_env.yaml \
  --side $side $extra --stage $stage --moved-joints ${MOVED_JOINTS:-4} --offset-deg $off --repeats $rep \
  --only-pose $pose --motion "${MOTION:-stationary}" --out "$E/setup/plan" > "$E/setup/prepare.log" 2>&1
.venv/bin/python -c "import json,sys;[print(t['id']) for t in json.load(open(sys.argv[1]))['trials']]" \
  "$E/setup/plan/plan.json" > "$E/setup/trials.txt"
echo "$E ready: $(wc -l < "$E/setup/trials.txt") trials"
