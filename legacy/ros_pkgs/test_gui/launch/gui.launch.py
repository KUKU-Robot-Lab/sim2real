import os
import argparse
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

def generate_launch_description():
    # param_dir = LaunchConfiguration(
    #     'param_dir',
    #     default=os.path.join(
    #         get_package_share_directory('atr_gui'),
    #         'param',
    #         'object_box_config.yaml'
    #     )
    # )

    gui_node = Node(
        package="test_gui",
        executable="test_gui",
        name="test_gui",
        output="screen",
    )
    return LaunchDescription([gui_node])
