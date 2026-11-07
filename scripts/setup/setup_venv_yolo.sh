#!/usr/bin/env bash
# 목표 마스크 학생 A 의 검출 노드(scripts/nodes/head_mask_node.py) 전용 파이썬 환경 .venv-yolo.
#
# 왜: ultralytics 가 torch · opencv-python 을 끌어와 .venv-student(학생 추론)의 torch 를 바꿀 수 있다. 검출만 따로 둔다.
# 방식: setup_student_env.sh 와 같이 ROS python3.10 위 --system-site-packages(rclpy 상속), torch 는 같은 cu128 핀.
#       ultralytics 는 1105_det 학습 버전(8.4.202). numpy 는 ROS(cv_bridge 등) 와 맞춘 1.26.4.
#
# 사용:
#   bash scripts/setup/setup_venv_yolo.sh
#   source /opt/ros/humble/setup.bash && .venv-yolo/bin/python scripts/nodes/head_mask_node.py --policy-dir <마스크 학생 묶음>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$HERE/.venv-yolo"
INDEX="https://download.pytorch.org/whl/cu128"

TORCH_SPEC="torch==2.7.1+cu128"
TV_SPEC="torchvision==0.22.1+cu128"
NUMPY_SPEC="numpy==1.26.4"
UL_SPEC="ultralytics==8.4.202"

if command -v uv >/dev/null; then
  uv venv --python /usr/bin/python3.10 --system-site-packages "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match)
else
  /usr/bin/python3.10 -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  PIP=("$VENV/bin/python" -m pip install -q)
fi
"${PIP[@]}" --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
  "$TORCH_SPEC" "$TV_SPEC" "$NUMPY_SPEC" "$UL_SPEC" "PyYAML>=6.0.2" pytest

echo "== 검증 =="
set +u; source /opt/ros/humble/setup.bash; set -u
"$VENV/bin/python" - <<'PY'
import sys, numpy, torch, cv2, ultralytics, yaml, rclpy  # noqa: F401
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert ultralytics.__version__ == "8.4.202", ultralytics.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
print(f"  python {sys.version.split()[0]} | numpy {numpy.__version__} | torch {torch.__version__} | cv2 {cv2.__version__} "
      f"| ultralytics {ultralytics.__version__}")
print(f"  CUDA {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} | rclpy OK")
PY
