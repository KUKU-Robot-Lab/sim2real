#!/bin/bash
source "$(dirname "$0")/common.sh"
pkill -f "scripts/nodes/fpp_pose_tx.py" || true
echo "pose tx down"
