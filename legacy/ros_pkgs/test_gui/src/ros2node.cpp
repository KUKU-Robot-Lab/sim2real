#include "test_gui/ros2node.hpp"

#include <chrono>

// All source files that use ROS logging should define a file-specific
// static const rclcpp::Logger named LOGGER, located at the top of the file
// and inside the namespace with the narrowest scope (if there is one)
static const rclcpp::Logger LOGGER = rclcpp::get_logger("test_gui");

namespace
{
constexpr double kLeftGripperOpen = 0.04;
constexpr double kLeftGripperClosed = 0.0;
constexpr double kRightFingerCloseRad = 1.3962634;  // 80 deg
}

Ros2Node::Ros2Node()
  : rclcpp::Node("ros2_node")
{
  publisher_ = this->create_publisher<std_msgs::msg::String>("publish_topic", 10);
  left_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/openarm/left_arm/eef_target", 10);
  right_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/openarm/right_arm/eef_target", 10);
  right_hand_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
    "/dg5f_right/dg5f_right_controller/joint_trajectory", 10);
  left_gripper_client_ = rclcpp_action::create_client<control_msgs::action::GripperCommand>(
    this, "/left_gripper_controller/gripper_cmd");

  //sub
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
              "/joint_states",
              qos_profile,
              std::bind(&Ros2Node::jointStateCallback, this, _1)
              );
//   gripCrpose_sub = this->create_subscription<std_msgs::msg::Int32MultiArray>(
//               "recv_grip",
//               qos_profile,
//               std::bind(&Ros2Node::gripCrPoseCallback, this, _1)
//               );

  // Service

  // init data
//   gripCrData[0] = gripCrData[1] = 0;


  // box setup


}

void Ros2Node::publish_message(std::string message)
{
  std_msgs::msg::String ros2_message;
  ros2_message.data = message;
  publisher_->publish(ros2_message);
}

void Ros2Node::publish_left_target_pose(const geometry_msgs::msg::Pose & pose)
{
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = "world";
  msg.pose = pose;
  left_pose_pub_->publish(msg);
}

void Ros2Node::publish_right_target_pose(const geometry_msgs::msg::Pose & pose)
{
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = "world";
  msg.pose = pose;
  right_pose_pub_->publish(msg);
}

std::vector<double> Ros2Node::left_joint_degrees() const
{
  return left_joint_deg_;
}

std::vector<double> Ros2Node::right_joint_degrees() const
{
  return right_joint_deg_;
}

void Ros2Node::command_left_gripper_from_ui(int ui_value)
{
  if (!left_gripper_client_) {
    RCLCPP_WARN(LOGGER, "Left gripper action client is not available");
    return;
  }

  if (!left_gripper_client_->wait_for_action_server(std::chrono::milliseconds(200))) {
    RCLCPP_WARN(LOGGER, "Left gripper action server is not ready");
    return;
  }

  const double ratio = std::clamp(static_cast<double>(ui_value) / 255.0, 0.0, 1.0);
  const double position = kLeftGripperOpen + (kLeftGripperClosed - kLeftGripperOpen) * ratio;

  control_msgs::action::GripperCommand::Goal goal;
  goal.command.position = position;
  goal.command.max_effort = 20.0;
  left_gripper_client_->async_send_goal(goal);
}

void Ros2Node::command_right_gripper_from_ui(int ui_value)
{
  const double ratio = std::clamp(static_cast<double>(ui_value) / 255.0, 0.0, 1.0);

  trajectory_msgs::msg::JointTrajectory traj;
  traj.header.stamp = this->now();
  traj.joint_names = right_hand_joint_names();

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions.assign(traj.joint_names.size(), 0.0);
  for (size_t finger = 0; finger < 5; ++finger) {
    const size_t base = finger * 4;
    point.positions[base + 2] = ratio * kRightFingerCloseRad;
    point.positions[base + 3] = ratio * kRightFingerCloseRad;
  }
  point.time_from_start = rclcpp::Duration::from_seconds(0.2);
  traj.points.push_back(point);
  right_hand_pub_->publish(traj);
}

void Ros2Node::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  left_joint_deg_.assign(6, 0.0);
  right_joint_deg_.assign(6, 0.0);

  for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
    const auto & name = msg->name[i];
    const double deg = RAD2DEG(msg->position[i]);

    if (name == "openarm_left_joint1") left_joint_deg_[0] = deg;
    else if (name == "openarm_left_joint2") left_joint_deg_[1] = deg;
    else if (name == "openarm_left_joint3") left_joint_deg_[2] = deg;
    else if (name == "openarm_left_joint4") left_joint_deg_[3] = deg;
    else if (name == "openarm_left_joint5") left_joint_deg_[4] = deg;
    else if (name == "openarm_left_joint6") left_joint_deg_[5] = deg;
    else if (name == "openarm_right_joint1") right_joint_deg_[0] = deg;
    else if (name == "openarm_right_joint2") right_joint_deg_[1] = deg;
    else if (name == "openarm_right_joint3") right_joint_deg_[2] = deg;
    else if (name == "openarm_right_joint4") right_joint_deg_[3] = deg;
    else if (name == "openarm_right_joint5") right_joint_deg_[4] = deg;
    else if (name == "openarm_right_joint6") right_joint_deg_[5] = deg;
  }
}

std::vector<std::string> Ros2Node::right_hand_joint_names()
{
  return {
    "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
    "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
    "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
    "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
    "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"
  };
}
