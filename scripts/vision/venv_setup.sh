#!/bin/bash
# 인지 PC 의 sim2real 전용 파이썬(.venv) 만들기(멱등). usage: bash scripts/vision/venv_setup.sh
# ★10.13 vision-3090: ~/.local 의 numpy 2.2(opencv-python 4.13 이 끌고 옴)가 시스템 scipy 1.8 을 깨서
#   (scipy.optimize · scipy.spatial ImportError) 컵홀더 노드 · 영상 기록기(_base_from_camera)가 시작하자마자 죽는다.
#   5090 .venv 와 같은 numpy 1.26.4 · scipy 1.13.1 로 맞춘다. opencv-python-headless 4.14 는 numpy 2 를 요구해 4.11.
#   --system-site-packages: ROS(rclpy · 메시지)는 /opt/ros 를 그대로 쓴다. venv 안 패키지가 ~/.local 보다 앞선다.
set -eo pipefail
SIM2REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SIM2REAL"
[ -x .venv/bin/python ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q --disable-pip-version-check \
  numpy==1.26.4 scipy==1.13.1 opencv-python-headless==4.11.0.86 PyYAML==6.0.3 pytest==8.3.5
.venv/bin/python -c "
import numpy, scipy, scipy.optimize, scipy.spatial.transform, cv2, yaml
assert numpy.__version__ == '1.26.4', numpy.__file__
print('venv ok · numpy', numpy.__version__, '· scipy', scipy.__version__, '· cv2', cv2.__version__)"
