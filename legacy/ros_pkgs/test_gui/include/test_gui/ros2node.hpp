#pragma once
#define BOOST_BIND_NO_PLACEHOLDERS
#include "control_msgs/action/gripper_command.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <QObject>
#include <QDebug>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include "test_gui/config.hpp"
#include "geometry_msgs/msg/twist.hpp"

//#include "main_gui.hpp"

using std::placeholders::_1;

class Ros2Node : public rclcpp::Node
{
//    Q_OBJECT
	public:
		Ros2Node();
        //pub
		void publish_message(std::string message);
        void publish_left_target_pose(const geometry_msgs::msg::Pose & pose);
        void publish_right_target_pose(const geometry_msgs::msg::Pose & pose);
        void command_left_gripper_from_ui(int ui_value);
        void command_right_gripper_from_ui(int ui_value);
        std::vector<double> left_joint_degrees() const;
        std::vector<double> right_joint_degrees() const;

        //sub
        // void gripCrPoseCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
        // int gripCrData[2];

        // Service client

        //function

        //data

	private:
        //publish
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr left_pose_pub_;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr right_pose_pub_;
        rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr right_hand_pub_;
        rclcpp_action::Client<control_msgs::action::GripperCommand>::SharedPtr left_gripper_client_;
        //subscribe
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
        void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
        std::vector<double> left_joint_deg_;
        std::vector<double> right_joint_deg_;

        static std::vector<std::string> right_hand_joint_names();
		// rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr gripCrpose_sub;

        // Service client

};
