"""hdgp 레거시 태스크에서 가져와 **고정한** 상수 — sim2real 이 hdgp 의 옛 태스크 패키지를 import 하지 않게.

09.29 사용자: hdgp 정리("일주일 넘게 안 쓴 rh56f1 · tesollo · gripper 레거시 태스크 27개를 main 에서 삭제", hdgp 1eb205d4)
뒤 sim2real 이 상수 몇 개를 얻으려고 `openarm.tesollo.left.grasp_v1` · `openarm.gripper.left.grasp_sensor(_v2)` 를
import 하던 곳이 모두 깨졌다 — DG-5F 실기 미션의 preflight(계약 재생성)가 첫 단계에서 멈췄다.
값은 hdgp 태그 `archive/legacy-tasks-20260929`(커밋 2ec0c035)의 파일에서 그대로 옮겼다. 줄 번호는 그 커밋 기준.
"""
from __future__ import annotations

# ---------------------------------------------------------------- DG-5F 좌우 거울 부호
# source/openarm/openarm/tesollo/left/grasp_v1/grasp_left_preset.py:52-61 (07-28 URDF FK 확정)
#: 오른팔 관절 → 왼팔 관절 부호 (j1..j7)
ARM_MIRROR_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
#: 오른손 20 관절(thumb · index · middle · ring · pinky, 각 _1.._4) → 왼손 부호
HAND_MIRROR_SIGN = (
    -1.0, -1.0, -1.0, -1.0,   # thumb  (X,Z,X,X)
    -1.0, 1.0, 1.0, 1.0,      # index  (X,Y,Y,Y)
    -1.0, 1.0, 1.0, 1.0,      # middle
    -1.0, 1.0, 1.0, 1.0,      # ring
    -1.0, -1.0, 1.0, 1.0,     # pinky  (Z,X,Y,Y)
)


# ---------------------------------------------------------------- 좌 스톡 그리퍼(gripper_left 계약, 레거시)
class GripperLeftPreset:
    """source/openarm/openarm/gripper/left/grasp_sensor/grasp_left_preset.py 의 값(줄 번호)."""

    GRIPPER_DRIVE_JOINT = "l_hj_gripper_1"                              # :49
    GRIPPER_JOINT_NAMES = ["l_hj_gripper_1", "l_hj_gripper_2"]          # :50
    GRIPPER_BASE_BODY = "l_hl_gripper_base"                             # :58
    GRIPPER_OPEN_POS = 0.044                                            # :70
    LEFT_ARM_HOME_JOINT_POS = {"l_aj_1": -0.0136, "l_aj_2": -0.3757, "l_aj_3": -0.0010, "l_aj_4": 0.9336,
                               "l_aj_5": -0.4655, "l_aj_6": 0.0003, "l_aj_7": -0.3306}   # :481-489
    GRASP_HEIGHT_BAND = (0.010, 0.085)                                  # :645 (v1)
    PALM_BOX_X = (0.22, 0.60)                                           # :826
    PALM_BOX_Y = (0.10, 0.43)                                           # :827
    PALM_BOX_Z = (0.16, 0.60)                                           # :838
    FABRIC_VEL_FF_SCALE = 1.0                                           # :868
    FABRIC_PARAMS_FILENAME = "openarm_gripper_left_pose_params.yaml"    # :959 (env HDGP_FABRIC_PARAMS 기본값)
    FABRIC_DECIMATION = 2                                               # :982
    FABRIC_DAMPING_GAIN = 10.0                                          # :983
    FABRIC_ROBOT_DIR = "openarm_tesollo_sensor_left_gripper"            # :984
    FABRIC_WORLD_FILENAME = "open_gripper_left_boxes_no_table"          # :985


class GripperLeftV2Preset:
    """source/openarm/openarm/gripper/left/grasp_sensor_v2/v2_preset.py:84."""

    GRASP_HEIGHT_BAND = (0.075, 0.135)
