from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "urdf_path",
            default_value="/home/user/rl_ws/sim2real/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf",
        ),
        DeclareLaunchArgument("root_link", default_value="openarm_left_link0"),
        DeclareLaunchArgument("tip_link", default_value="openarm_left_hand_tcp"),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
        DeclareLaunchArgument(
            "target_pose_topic", default_value="/openarm/left_arm/eef_target"
        ),
        DeclareLaunchArgument(
            "trajectory_topic",
            default_value="/left_joint_trajectory_controller/joint_trajectory",
        ),
        DeclareLaunchArgument("trajectory_time_sec", default_value="0.2"),
        Node(
            package="openarm_eef_control",
            executable="left_arm_eef_controller",
            output="screen",
            parameters=[{
                "urdf_path": LaunchConfiguration("urdf_path"),
                "root_link": LaunchConfiguration("root_link"),
                "tip_link": LaunchConfiguration("tip_link"),
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "target_pose_topic": LaunchConfiguration("target_pose_topic"),
                "trajectory_topic": LaunchConfiguration("trajectory_topic"),
                "trajectory_time_sec": LaunchConfiguration("trajectory_time_sec"),
            }],
        ),
    ])
