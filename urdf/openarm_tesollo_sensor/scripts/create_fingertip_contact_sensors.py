"""Create fingertip contact sensors for the Tesollo right hand in Isaac Sim.

Usage in Isaac Sim Script Editor:
    1. Open the stage that contains openarm_tesollo_sensor.usd.
    2. Adjust ROBOT_ROOT if the robot is not under /World/openarm_tesollo_sensor.
    3. Run this file with "Window > Script Editor".

This script:
    - attaches PhysX contact report API to each fingertip rigid body
    - creates an Isaac Contact Sensor prim under each fingertip
    - stores real-sensor protocol metadata as custom USD attributes
"""

from omni.isaac.core.utils.stage import get_current_stage
import omni.kit.commands
from pxr import Gf, PhysxSchema, Sdf


ROBOT_ROOT = "/World/openarm_tesollo_sensor"
TIP_COUNT = 5

CONTACT_SENSOR_NAME = "Contact_Sensor"
SENSOR_PERIOD = 0.0
MIN_THRESHOLD_N = 0.001
MAX_THRESHOLD_N = 30.0

# Real sensor protocol metadata for downstream tooling.
FORCE_LIMIT_N = 30.0
TORQUE_LIMIT_NM = 250.0
FORCE_SCALE = 10.0
TORQUE_SCALE = 10.0
BYTE_ORDER = "big"
DATA_ORDER = "fx,fy,fz,tx,ty,tz"
SENSOR_CODE = "0x05"


def _set_custom_attr(prim, name, value_type, value):
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, value_type, custom=True)
    attr.Set(value)


def _find_robot_root_from_tip(stage):
    """Resolve the robot root by searching for rl_dg_1_tip in the current stage."""
    candidate_suffix = "/rl_dg_1_tip"
    matches = []
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if path_str.endswith(candidate_suffix):
            matches.append(path_str[: -len(candidate_suffix)])

    if not matches:
        raise RuntimeError(
            "Could not find any prim ending with '/rl_dg_1_tip'. "
            "Check whether the USD was imported and expanded in the current stage."
        )

    if len(matches) > 1:
        print("Multiple robot roots detected. Using the first match:")
        for item in matches:
            print(f"  - {item}")

    return matches[0]


def configure_fingertip_contact_sensor(tip_path: str) -> None:
    stage = get_current_stage()
    tip_prim = stage.GetPrimAtPath(tip_path)
    if not tip_prim.IsValid():
        raise RuntimeError(f"Missing fingertip prim: {tip_path}")

    # Needed to expose raw contact data from the rigid body.
    report_api = PhysxSchema.PhysxContactReportAPI.Apply(tip_prim)
    report_api.CreateThresholdAttr().Set(0.0)

    sensor_prim_path = f"{tip_path}/{CONTACT_SENSOR_NAME}"
    sensor_prim = stage.GetPrimAtPath(sensor_prim_path)
    if not sensor_prim.IsValid():
        success, _ = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path=CONTACT_SENSOR_NAME,
            parent=tip_path,
            sensor_period=SENSOR_PERIOD,
            min_threshold=MIN_THRESHOLD_N,
            max_threshold=MAX_THRESHOLD_N,
            translation=Gf.Vec3d(0.0, 0.0, 0.0),
            radius=-1.0,
        )
        if not success:
            raise RuntimeError(f"Failed to create contact sensor for {tip_path}")
        sensor_prim = stage.GetPrimAtPath(sensor_prim_path)

    # Mirror the real protocol settings into USD as custom attributes so the
    # values stay attached to the asset even though Isaac Contact Sensor itself
    # only consumes force thresholds and period.
    _set_custom_attr(sensor_prim, "user:sensor_code", Sdf.ValueTypeNames.String, SENSOR_CODE)
    _set_custom_attr(sensor_prim, "user:force_limit_n", Sdf.ValueTypeNames.Float, FORCE_LIMIT_N)
    _set_custom_attr(sensor_prim, "user:torque_limit_nm", Sdf.ValueTypeNames.Float, TORQUE_LIMIT_NM)
    _set_custom_attr(sensor_prim, "user:force_scale", Sdf.ValueTypeNames.Float, FORCE_SCALE)
    _set_custom_attr(sensor_prim, "user:torque_scale", Sdf.ValueTypeNames.Float, TORQUE_SCALE)
    _set_custom_attr(sensor_prim, "user:byte_order", Sdf.ValueTypeNames.String, BYTE_ORDER)
    _set_custom_attr(sensor_prim, "user:data_order", Sdf.ValueTypeNames.String, DATA_ORDER)

    print(f"Configured: {sensor_prim_path}")


def main() -> None:
    stage = get_current_stage()
    robot_root = ROBOT_ROOT
    if not stage.GetPrimAtPath(f"{robot_root}/rl_dg_1_tip").IsValid():
        robot_root = _find_robot_root_from_tip(stage)
        print(f"Resolved ROBOT_ROOT -> {robot_root}")

    for finger_idx in range(1, TIP_COUNT + 1):
        configure_fingertip_contact_sensor(f"{robot_root}/rl_dg_{finger_idx}_tip")

    print("All fingertip contact sensors configured.")


if __name__ == "__main__":
    main()
