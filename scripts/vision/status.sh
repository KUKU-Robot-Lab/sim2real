#!/bin/bash
# 마지막 줄이 JSON. 카메라 hz 는 로컬 런처가 DDS 로 직접 잰다(여기선 프로세스 유무만).
# 09.29: 컨테이너가 Up 이어도 추적 노드는 CUDA OOM 으로 죽어 있을 수 있다(학습과 GPU 공유) — 로그의 마지막
#        Traceback 오류 줄(crashes)과 GPU 메모리(gpu)를 같이 낸다. 콘솔 상태창의 FP++ 칸이 읽는다.
source "$(dirname "$0")/common.sh"
python3 - <<'EOF'
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.environ["SIM2REAL"], "scripts"))
def up(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
out = subprocess.run(["docker", "ps", "-a", "--filter", "name=^fpp_", "--format", "{{.Names}}\t{{.Status}}"],
                     capture_output=True, text=True).stdout
containers = dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line)
crashes = {}
try:
    from perception_launcher_core import last_crash
    for name, status in containers.items():
        if status.startswith("Up"):
            log = subprocess.run(["docker", "logs", "--tail", "200", name], capture_output=True, text=True, timeout=10)
            crashes[name] = last_crash(log.stdout + log.stderr)
except Exception as exc:  # noqa: BLE001 — 상태 보고는 죽지 않는다
    crashes = {"_": f"crash scan failed: {exc}"[:160]}
gpu = None
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10).stdout.splitlines()[0].split(",")
    gpu = {"used_mib": int(q[0]), "total_mib": int(q[1])}
except Exception:  # noqa: BLE001
    pass
print(json.dumps({"camera_up": up("realsense2_camera_node"), "containers": containers,
                  "viewer_up": up("cup_view_stream.py"), "pose_tx_up": up("fpp_pose_tx.py"),
                  "crashes": {k: v for k, v in crashes.items() if v}, "gpu": gpu}))
EOF
