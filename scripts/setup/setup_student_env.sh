#!/usr/bin/env bash
# 깊이 + RGB 학생(arch v3) 전용 파이썬 환경 .venv-student — 배포 .venv(deploy 세션 소유)와 따로 둔다.
#
# 왜: hdgp student/model.py 가 torchvision(ResNet18 두 벌)을 쓰는데 .venv 에는 torch 만 있다. .venv 를 고치면
#     배포 세션 환경이 바뀌므로 같은 torch 핀 위에 torchvision 을 더한 별도 venv 를 만든다.
# 방식: setup_deploy_env.sh 와 같이 ROS python3.10 위 --system-site-packages(rclpy 상속), torch 는 같은 cu128 핀.
#
# 사용:
#   bash scripts/setup/setup_student_env.sh
#   source /opt/ros/humble/setup.bash && .venv-student/bin/python scripts/nodes/head_depth_node.py --policy-dir <학생 폴더>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$HERE/.venv-student"
INDEX="https://download.pytorch.org/whl/cu128"

# torch 는 배포 .venv · Isaac 학습과 같은 핀. torchvision 0.22.1 = torch 2.7.1 짝.
TORCH_SPEC="torch==2.7.1+cu128"
TV_SPEC="torchvision==0.22.1+cu128"
NUMPY_SPEC="numpy==1.26.4"
SCIPY_SPEC="scipy==1.13.1"

if command -v uv >/dev/null; then
  uv venv --python /usr/bin/python3.10 --system-site-packages "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match)
else
  /usr/bin/python3.10 -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  PIP=("$VENV/bin/python" -m pip install -q)
fi
"${PIP[@]}" --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
  "$TORCH_SPEC" "$TV_SPEC" "$NUMPY_SPEC" "$SCIPY_SPEC" "PyYAML>=6.0.2" pytest

echo "== 검증 =="
set +u; source /opt/ros/humble/setup.bash; set -u
"$VENV/bin/python" - <<'PY'
import sys, numpy, torch, torchvision, yaml, rclpy  # noqa: F401
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torchvision.__version__.startswith("0.22.1"), torchvision.__version__
print(f"  python {sys.version.split()[0]} | numpy {numpy.__version__} | torch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"  CUDA {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} | rclpy OK")
PY
