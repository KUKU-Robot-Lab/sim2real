#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "kdl/chain.hpp"
#include "kdl/chainfksolverpos_recursive.hpp"
#include "kdl/chainiksolverpos_nr_jl.hpp"
#include "kdl/chainiksolvervel_pinv.hpp"
#include "kdl/frames.hpp"
#include "kdl/jntarray.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "urdf/model.h"

namespace
{
std::string read_file(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream.is_open()) {
    return "";
  }

  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}
}

class LeftArmEefController : public rclcpp::Node
{
public:
  LeftArmEefController()
  : Node("left_arm_eef_controller")
  {
    declare_parameter<std::string>(
      "urdf_path", "/home/user/rl_ws/sim2real/urdf/openarm_tesollo_sensor/openarm_tesollo_sensor.urdf");
    declare_parameter<std::string>("root_link", "openarm_left_link0");
    declare_parameter<std::string>("tip_link", "openarm_left_hand_tcp");
    declare_parameter<std::string>("joint_state_topic", "/joint_states");
    declare_parameter<std::string>("target_pose_topic", "/openarm/left_arm/eef_target");
    declare_parameter<std::string>(
      "trajectory_topic", "/left_joint_trajectory_controller/joint_trajectory");
    declare_parameter<double>("trajectory_time_sec", 0.2);
    declare_parameter<int>("ik_max_iterations", 200);
    declare_parameter<double>("ik_eps", 1e-5);

    const auto urdf_path = get_parameter("urdf_path").as_string();
    const auto root_link = get_parameter("root_link").as_string();
    const auto tip_link = get_parameter("tip_link").as_string();

    if (!load_model(urdf_path, root_link, tip_link)) {
      throw std::runtime_error("failed to initialize KDL chain from URDF");
    }

    trajectory_time_sec_ = get_parameter("trajectory_time_sec").as_double();

    const auto joint_state_topic = get_parameter("joint_state_topic").as_string();
    const auto target_pose_topic = get_parameter("target_pose_topic").as_string();
    const auto trajectory_topic = get_parameter("trajectory_topic").as_string();

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic, 20,
      std::bind(&LeftArmEefController::joint_state_callback, this, std::placeholders::_1));
    target_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      target_pose_topic, 10,
      std::bind(&LeftArmEefController::target_pose_callback, this, std::placeholders::_1));
    trajectory_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(trajectory_topic, 10);

    RCLCPP_INFO(
      get_logger(),
      "Left arm EEF controller ready. root=%s tip=%s target_topic=%s trajectory_topic=%s",
      root_link.c_str(), tip_link.c_str(), target_pose_topic.c_str(), trajectory_topic.c_str());
  }

private:
  bool load_model(const std::string & urdf_path, const std::string & root_link, const std::string & tip_link)
  {
    const auto urdf_xml = read_file(urdf_path);
    if (urdf_xml.empty()) {
      RCLCPP_ERROR(get_logger(), "Failed to read URDF: %s", urdf_path.c_str());
      return false;
    }

    if (!urdf_model_.initString(urdf_xml)) {
      RCLCPP_ERROR(get_logger(), "Failed to parse URDF model from %s", urdf_path.c_str());
      return false;
    }

    KDL::Tree tree;
    if (!kdl_parser::treeFromString(urdf_xml, tree)) {
      RCLCPP_ERROR(get_logger(), "Failed to build KDL tree from URDF");
      return false;
    }

    if (!tree.getChain(root_link, tip_link, chain_)) {
      RCLCPP_ERROR(
        get_logger(), "Failed to extract KDL chain from %s to %s", root_link.c_str(),
        tip_link.c_str());
      return false;
    }

    joint_names_.clear();
    lower_limits_.clear();
    upper_limits_.clear();

    for (const auto & segment : chain_.segments) {
      const auto & joint = segment.getJoint();
      if (joint.getType() == KDL::Joint::None) {
        continue;
      }

      const auto joint_name = joint.getName();
      const auto urdf_joint = urdf_model_.getJoint(joint_name);
      if (!urdf_joint || !urdf_joint->limits) {
        RCLCPP_ERROR(get_logger(), "Missing limits for joint %s", joint_name.c_str());
        return false;
      }

      joint_names_.push_back(joint_name);
      lower_limits_.push_back(urdf_joint->limits->lower);
      upper_limits_.push_back(urdf_joint->limits->upper);
    }

    if (joint_names_.empty()) {
      RCLCPP_ERROR(get_logger(), "No movable joints found in chain");
      return false;
    }

    RCLCPP_INFO(get_logger(), "Loaded KDL chain with %zu joints", joint_names_.size());
    return true;
  }

  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    const auto count = std::min(msg->name.size(), msg->position.size());
    for (size_t i = 0; i < count; ++i) {
      latest_positions_[msg->name[i]] = msg->position[i];
    }
  }

  void target_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (latest_positions_.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for joint states");
      return;
    }

    KDL::JntArray seed(joint_names_.size());
    KDL::JntArray lower(joint_names_.size());
    KDL::JntArray upper(joint_names_.size());

    for (size_t i = 0; i < joint_names_.size(); ++i) {
      const auto it = latest_positions_.find(joint_names_[i]);
      if (it == latest_positions_.end()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "Joint state for %s not available yet",
          joint_names_[i].c_str());
        return;
      }
      seed(i) = it->second;
      lower(i) = lower_limits_[i];
      upper(i) = upper_limits_[i];
    }

    KDL::Frame target_frame(
      KDL::Rotation::Quaternion(
        msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z,
        msg->pose.orientation.w),
      KDL::Vector(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z));

    const auto max_iterations = get_parameter("ik_max_iterations").as_int();
    const auto eps = get_parameter("ik_eps").as_double();

    KDL::ChainFkSolverPos_recursive fk_solver(chain_);
    KDL::ChainIkSolverVel_pinv vel_solver(chain_);
    KDL::ChainIkSolverPos_NR_JL ik_solver(chain_, lower, upper, fk_solver, vel_solver, max_iterations, eps);

    KDL::JntArray result(joint_names_.size());
    const int rc = ik_solver.CartToJnt(seed, target_frame, result);
    if (rc < 0) {
      RCLCPP_WARN(
        get_logger(), "IK solve failed with code %d for target frame '%s'", rc,
        msg->header.frame_id.c_str());
      return;
    }

    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = now();
    traj.joint_names = joint_names_;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.resize(joint_names_.size());
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      point.positions[i] = std::clamp(result(i), lower_limits_[i], upper_limits_[i]);
    }

    point.time_from_start = rclcpp::Duration::from_seconds(trajectory_time_sec_);
    traj.points.push_back(point);

    trajectory_pub_->publish(traj);
  }

  urdf::Model urdf_model_;
  KDL::Chain chain_;
  std::vector<std::string> joint_names_;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  std::unordered_map<std::string, double> latest_positions_;
  double trajectory_time_sec_{0.2};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LeftArmEefController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
