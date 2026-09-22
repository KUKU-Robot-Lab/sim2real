import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_calib = os.path.join(
        get_package_share_directory("isaacsim_bridge"),
        "config",
        "rh56f1_hand_calibration.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hands",
                default_value="[right, left]",
                description="Which RH56F1 hands to bridge (subset of right/left).",
            ),
            DeclareLaunchArgument(
                "calibration_config",
                default_value=default_calib,
                description="YAML with per-joint register/radian calibration.",
            ),
            DeclareLaunchArgument(
                "merged_hand_states_topic",
                default_value="/rh56f1/joint_states",
                description="Topic where measured hand state (radians) is republished.",
            ),
            Node(
                package="isaacsim_bridge",
                executable="rh56f1_hand_bridge",
                name="rh56f1_hand_bridge",
                output="screen",
                parameters=[
                    {
                        "hands": LaunchConfiguration("hands"),
                        "calibration_config": LaunchConfiguration("calibration_config"),
                        "merged_hand_states_topic": LaunchConfiguration(
                            "merged_hand_states_topic"
                        ),
                    }
                ],
            ),
        ]
    )
