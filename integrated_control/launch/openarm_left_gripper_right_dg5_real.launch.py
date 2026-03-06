import os

import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPENARM_XACRO_PATH = os.path.join(
    REPO_DIR, "urdf", "openarm_left_gripper_bimanual_real.xacro"
)
OPENARM_CONTROLLERS_PATH = os.path.join(
    REPO_DIR, "integrated_control", "config", "openarm_left_gripper_bimanual_controllers.yaml"
)
RVIZ_CONFIG_PATH = os.path.join(get_package_share_directory("openarm_description"), "rviz", "bimanual.rviz")
DG5_RIGHT_DRIVER_LAUNCH = os.path.join(get_package_share_directory("dg5f_driver"), "launch", "dg5f_right_driver.launch.py")


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
    dg5f_right_ip = LaunchConfiguration("dg5f_right_ip")
    dg5f_right_port = LaunchConfiguration("dg5f_right_port")
    dg5f_fingertip_sensor = LaunchConfiguration("dg5f_fingertip_sensor")
    dg5f_io = LaunchConfiguration("dg5f_io")

    openarm_loader = OpaqueFunction(
        function=openarm_nodes_spawner,
        args=[
            left_can_interface,
            right_can_interface,
            use_fake_hardware,
        ],
    )

    dg5_right_loader = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(DG5_RIGHT_DRIVER_LAUNCH),
        launch_arguments={
            "delto_ip": dg5f_right_ip,
            "delto_port": dg5f_right_port,
            "fingertip_sensor": dg5f_fingertip_sensor,
            "io": dg5f_io,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "left_can_interface",
                default_value="can1",
                description="SocketCAN interface for the left OpenArm arm",
            ),
            DeclareLaunchArgument(
                "right_can_interface",
                default_value="can0",
                description="SocketCAN interface for the right OpenArm arm",
            ),
            DeclareLaunchArgument(
                "use_fake_hardware",
                default_value="false",
                description="Use fake hardware for OpenArm instead of real CAN hardware",
            ),
            DeclareLaunchArgument(
                "dg5f_right_ip",
                default_value="169.254.186.72",
                description="IP address for the right DG5F hand",
            ),
            DeclareLaunchArgument(
                "dg5f_right_port",
                default_value="502",
                description="Port for the right DG5F hand",
            ),
            DeclareLaunchArgument(
                "dg5f_fingertip_sensor",
                default_value="false",
                description="Enable right DG5F fingertip F/T sensors",
            ),
            DeclareLaunchArgument(
                "dg5f_io",
                default_value="false",
                description="Enable right DG5F IO interface",
            ),
            openarm_loader,
            dg5_right_loader,
        ]
    )
