# Copyright 2021 Abrar Rahman Protyasha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur16e", description="Type/series of used UR robot."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )
    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="ur_bringup",
            description='Package with the controller\'s configuration in "config" folder. \
        Usually the argument is not set, it enables use of a custom setup.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="ur_description",
            description="Description package with robot URDF/XACRO files. Usually the argument \
        is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="ur.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="Enable fake command interfaces for sensors used for simple simulations. \
            Used only if 'use_fake_hardware' parameter is true.",
        )
    )

    # Track-ik specific arguments
    # declared_arguments.append(
    #     DeclareLaunchArgument(
    #         "num_samples",
    #         default_value="1000",
    #         description="number of random target 6D pose samples.",
    #     )
    # )
    declared_arguments.append(
        DeclareLaunchArgument(
            "chain_start",
            default_value="base_link",
            description="link of chain start",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "chain_end",
            default_value="wrist_3_link",
            description="link of chain end.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "timeout",
            default_value="0.005",
            description="enable time to solve",
        )
    )

    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    # General arguments
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    prefix = LaunchConfiguration("prefix")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")

    joint_limit_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "joint_limits.yaml"]
    )
    kinematics_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "default_kinematics.yaml"]
    )
    physical_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "physical_parameters.yaml"]
    )
    visual_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "visual_parameters.yaml"]
    )

    # num_samples = LaunchConfiguration('num_samples')
    chain_start = LaunchConfiguration('chain_start')
    chain_end = LaunchConfiguration('chain_end')
    timeout = LaunchConfiguration('timeout')

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),
            " ",
            "joint_limit_params:=",
            joint_limit_params,
            " ",
            "kinematics_params:=",
            kinematics_params,
            " ",
            "physical_params:=",
            physical_params,
            " ",
            "visual_params:=",
            visual_params,
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=",
            ur_type,
            " ",
            "prefix:=",
            prefix,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "fake_sensor_commands:=",
            fake_sensor_commands,
            " ",
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # pkg_share = FindPackageShare('trac_ik_examples').find('trac_ik_examples')
    # urdf_file = os.path.join(pkg_share, 'launch', 'pr2.urdf')
    # with open(urdf_file, 'r') as infp:
    #     robot_desc = infp.read()

    ik_node = Node(
        package="ur_eef_control",
        executable="ur16e_compliance2",
        output="screen",
        parameters=[robot_description,
            {
                # 'robot_description': robot_desc,
                # 'num_samples': num_samples,
                'chain_start': chain_start,
                'chain_end': chain_end,
                'timeout': timeout,
            }
        ],
    )

    nodes_to_start = [
        ik_node,
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)

    # return LaunchDescription(
    #     [
    #         DeclareLaunchArgument('num_samples', default_value='1000'),
    #         DeclareLaunchArgument('chain_start', default_value='torso_lift_link'),
    #         DeclareLaunchArgument('chain_end', default_value='r_wrist_roll_link'),
    #         DeclareLaunchArgument('timeout', default_value='0.005'),
    #         Node(
    #             package='trac_ik_examples',
    #             executable='ik_tests',
    #             output='screen',
    #             parameters=[
    #                 {
    #                     'robot_description': robot_desc,
    #                     'num_samples': num_samples,
    #                     'chain_start': chain_start,
    #                     'chain_end': chain_end,
    #                     'timeout': timeout,
    #                 }
    #             ],
    #         ),
    #     ]
    # )


"""
Alternative ways to obtain the robot description:

1. Using `xacro` with command substitution:
xacro_file = os.path.join(urdf_dir, 'test.urdf.xacro')
robot_desc = launch.substitutions.Command(f'xacro {xacro_file}')

2. Using `xacro` API:
xacro_file = os.path.join(urdf_dir, 'test.urdf.xacro')
robot_desc = xacro.process_file(xacro_file).toprettyxml(indent='    ')

3. Using `xacro` with `subprocess` utils:
xacro_file = os.path.join(urdf_dir, 'test.urdf.xacro')
p = subprocess.Popen(['xacro', xacro_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
robot_desc, stderr = p.communicate()
"""
