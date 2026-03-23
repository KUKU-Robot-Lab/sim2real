cmake_minimum_required(VERSION 3.8)
project(openarm_eef_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(urdf REQUIRED)

add_executable(right_arm_eef_controller src/right_arm_eef_controller.cpp)
add_executable(left_arm_eef_controller src/left_arm_eef_controller.cpp)
target_compile_features(right_arm_eef_controller PUBLIC cxx_std_17)
target_compile_features(left_arm_eef_controller PUBLIC cxx_std_17)
ament_target_dependencies(
  right_arm_eef_controller
  geometry_msgs
  kdl_parser
  orocos_kdl
  rclcpp
  sensor_msgs
  trajectory_msgs
  urdf
)
ament_target_dependencies(
  left_arm_eef_controller
  geometry_msgs
  kdl_parser
  orocos_kdl
  rclcpp
  sensor_msgs
  trajectory_msgs
  urdf
)

install(TARGETS right_arm_eef_controller left_arm_eef_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
