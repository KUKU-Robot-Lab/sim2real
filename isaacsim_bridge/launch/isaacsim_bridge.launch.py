import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    hardware_launch = os.path.join(
        repo_dir,
        "integrated_control",
        "launch",
        "openarm_left_gripper_right_dg5_real.launch.py",
    )

    with_hardware = LaunchConfiguration("with_hardware")
    with_recorder = LaunchConfiguration("with_recorder")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "with_hardware",
                default_value="false",
                description="Start real hardware bringup together with the bridge.",
            ),
            DeclareLaunchArgument("left_can_interface", default_value="can1"),
            DeclareLaunchArgument("right_can_interface", default_value="can0"),
            DeclareLaunchArgument("dg5f_right_ip", default_value="169.254.186.72"),
            DeclareLaunchArgument("dg5f_right_port", default_value="502"),
            DeclareLaunchArgument(
                "with_recorder",
                default_value="false",
                description="Start sim-vs-real joint error recorder together with the bridge.",
            ),
            DeclareLaunchArgument(
                "recorder_output_path",
                default_value="/tmp/isaacsim_joint_error.csv",
            ),
            DeclareLaunchArgument(
                "recorder_real_joint_states_topic",
                default_value="/isaacsim/joint_states",
            ),
            DeclareLaunchArgument(
                "recorder_sim_joint_states_topic",
                default_value="/isaacsim/sim_joint_states",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(hardware_launch),
                condition=IfCondition(with_hardware),
                launch_arguments={
                    "left_can_interface": LaunchConfiguration("left_can_interface"),
                    "right_can_interface": LaunchConfiguration("right_can_interface"),
                    "dg5f_right_ip": LaunchConfiguration("dg5f_right_ip"),
                    "dg5f_right_port": LaunchConfiguration("dg5f_right_port"),
                }.items(),
            ),
            Node(
                package="isaacsim_bridge",
                executable="bridge_node",
                name="isaacsim_bridge",
                output="screen",
            ),
            Node(
                package="isaacsim_bridge",
                executable="joint_error_recorder",
                name="joint_error_recorder",
                output="screen",
                condition=IfCondition(with_recorder),
                parameters=[
                    {
                        "output_path": LaunchConfiguration("recorder_output_path"),
                        "real_joint_states_topic": LaunchConfiguration(
                            "recorder_real_joint_states_topic"
                        ),
                        "sim_joint_states_topic": LaunchConfiguration(
                            "recorder_sim_joint_states_topic"
                        ),
                    }
                ],
            ),
        ]
    )
