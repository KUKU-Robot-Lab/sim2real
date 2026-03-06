import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


DG5_RIGHT_DRIVER_LAUNCH = os.path.join(get_package_share_directory("dg5f_driver"), "launch", "dg5f_right_driver.launch.py")


def generate_launch_description():
    dg5f_right_ip = LaunchConfiguration("dg5f_right_ip")
    dg5f_right_port = LaunchConfiguration("dg5f_right_port")
    dg5f_fingertip_sensor = LaunchConfiguration("dg5f_fingertip_sensor")
    dg5f_io = LaunchConfiguration("dg5f_io")

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
            DeclareLaunchArgument("dg5f_right_ip", default_value="169.254.186.72"),
            DeclareLaunchArgument("dg5f_right_port", default_value="502"),
            DeclareLaunchArgument("dg5f_fingertip_sensor", default_value="false"),
            DeclareLaunchArgument("dg5f_io", default_value="false"),
            dg5_right_loader,
        ]
    )
