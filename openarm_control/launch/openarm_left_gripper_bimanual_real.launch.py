import os

import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPENARM_XACRO_PATH = os.path.join(REPO_DIR, "urdf", "openarm_left_gripper_bimanual_real.xacro")
OPENARM_CONTROLLERS_PATH = os.path.join(
    REPO_DIR, "integrated_control", "config", "openarm_left_gripper_bimanual_controllers.yaml"
)
RVIZ_CONFIG_PATH = os.path.join(get_package_share_directory("openarm_description"), "rviz", "bimanual.rviz")


def openarm_nodes_spawner(
    context: LaunchContext,
    left_can_interface,
    right_can_interface,
    use_fake_hardware,
):
    robot_description = xacro.process_file(
        OPENARM_XACRO_PATH,
        mappings={
            "left_can_interface": context.perform_substitution(left_can_interface),
            "right_can_interface": context.perform_substitution(right_can_interface),
            "use_fake_hardware": context.perform_substitution(use_fake_hardware),
        },
    ).toprettyxml(indent="  ")

    robot_description_param = {"robot_description": robot_description}

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="openarm_robot_state_publisher",
            output="screen",
            parameters=[robot_description_param],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[robot_description_param, OPENARM_CONTROLLERS_PATH],
        ),
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
                    output="screen",
                )
            ],
        ),
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=[
                        "left_joint_trajectory_controller",
                        "right_joint_trajectory_controller",
                        "-c",
                        "/controller_manager",
                    ],
                    output="screen",
                )
            ],
        ),
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=["left_gripper_controller", "-c", "/controller_manager"],
                    output="screen",
                )
            ],
        ),
        TimerAction(
            period=1.5,
            actions=[
                Node(
                    package="rviz2",
                    executable="rviz2",
                    name="rviz2",
                    arguments=["-d", RVIZ_CONFIG_PATH],
                    output="screen",
                )
            ],
        ),
    ]


def generate_launch_description():
    left_can_interface = LaunchConfiguration("left_can_interface")
    right_can_interface = LaunchConfiguration("right_can_interface")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")

    openarm_loader = OpaqueFunction(
        function=openarm_nodes_spawner,
        args=[left_can_interface, right_can_interface, use_fake_hardware],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("left_can_interface", default_value="can1"),
            DeclareLaunchArgument("right_can_interface", default_value="can0"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            openarm_loader,
        ]
    )
