#include <chrono>
#include <map>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>
#include <math.h>

#include "kdl/chainiksolverpos_nr_jl.hpp"
#include "rclcpp/rclcpp.hpp"
#include "trac_ik/trac_ik.hpp"

#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"

#define DEG2RAD(x)                          (x * 0.01745329252)  // *PI/180
#define RAD2DEG(x)                          (x * 57.2957795131)  // *180/PI

using std::placeholders::_1;

class URjointTest : public rclcpp::Node{
  public:
    URjointTest();
    //pub function
    void publish_target_joints(sensor_msgs::msg::JointState target_j);
    void publish_ft_data(geometry_msgs::msg::WrenchStamped ft_data);
    void publish_result(std_msgs::msg::Float64MultiArray result);

    //sub function
    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr joint_);
    void dposeCallback(const geometry_msgs::msg::Twist::SharedPtr dPose);
    void ft_1Callback(const geometry_msgs::msg::WrenchStamped::SharedPtr ft_1);
    void cplCheckCallback(const std_msgs::msg::Int32::SharedPtr c_check);

    // Function

    // Data
    std::vector<std::string> joint_name;
    std::vector<double> joint_position, joint_velocity, joint_effort;
    geometry_msgs::msg::Twist desired_pose;
    geometry_msgs::msg::Wrench ft_1;
    geometry_msgs::msg::WrenchStamped ft_data;
    int cpl_enable, cpl_enable_check;

  private:
    // Publsher
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr ur_jointarget_pub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_data_pub;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr result_pub;

    // Subscribe
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr ur_jointstate_sub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ur_desired_pose_sub;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_1_sub;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr cpl_check_sub;

};

static const rclcpp::Logger LOGGER = rclcpp::get_logger("ur16e_joint_test_1");

URjointTest::URjointTest()
  : rclcpp::Node("ur16e_compliance2")
{
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
  // Pub
  ur_jointarget_pub = this->create_publisher<sensor_msgs::msg::JointState>("joint_command", qos_profile);
  ft_data_pub = this->create_publisher<geometry_msgs::msg::WrenchStamped>("ft_data_revised", qos_profile);
  result_pub = this->create_publisher<std_msgs::msg::Float64MultiArray>("ur_result", qos_profile);

  // Sub
  ur_jointstate_sub = this->create_subscription<sensor_msgs::msg::JointState>(
              "joint_states",
              qos_profile,
              std::bind(&URjointTest::jointCallback, this, _1)
              );
  ur_desired_pose_sub = this->create_subscription<geometry_msgs::msg::Twist>(
    "eef_target_d_twist", qos_profile, std::bind(&URjointTest::dposeCallback, this, _1));
  ft_1_sub = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
    "ft_data", qos_profile, std::bind(&URjointTest::ft_1Callback, this, _1));
  cpl_check_sub = this->create_subscription<std_msgs::msg::Int32>(
    "cpl_check", qos_profile, std::bind(&URjointTest::cplCheckCallback, this, _1));

  // Init data
  std::vector<std::string> check_ur_j = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
  joint_name = check_ur_j;

  joint_position.resize(6);
  joint_velocity.resize(6);
  joint_effort.resize(6);

  cpl_enable = cpl_enable_check = 0;

}

void URjointTest::publish_target_joints(sensor_msgs::msg::JointState target_j){

  this->ur_jointarget_pub->publish(target_j);

}

void URjointTest::publish_ft_data(geometry_msgs::msg::WrenchStamped ft){
  ft.header.stamp = this->now();
  this->ft_data_pub->publish(ft);
}

void URjointTest::publish_result(std_msgs::msg::Float64MultiArray rs){
  this->result_pub->publish(rs);
}

void URjointTest::jointCallback(const sensor_msgs::msg::JointState::SharedPtr joint_){
//    RCLCPP_INFO(LOGGER, "joint %s", joint_->name);

    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            if(joint_name[i] == joint_->name[j]){
                joint_position[i] = joint_->position[j];
                joint_velocity[i] = joint_->velocity[j];
                joint_effort[i] = joint_->effort[j];
            }
        }
//        RCLCPP_INFO(LOGGER, "joint [%d] %lf deg",i, RAD2DEG(joint_position[i]));
    }

}

void URjointTest::dposeCallback(const geometry_msgs::msg::Twist::SharedPtr dPose){
  this->desired_pose = *dPose;
  // RCLCPP_INFO_STREAM(LOGGER,
  // "d Pose: linear x: " << this->desired_pose.linear.x<<" y: "<<this->desired_pose.linear.y<<" z: "<<this->desired_pose.linear.z);
  // RCLCPP_INFO_STREAM(LOGGER,
  // "d Pose: angular x: " << this->desired_pose.angular.x<<" y: "<<this->desired_pose.angular.y<<" z: "<<this->desired_pose.angular.z);
}

void URjointTest::ft_1Callback(const geometry_msgs::msg::WrenchStamped::SharedPtr ft_1_data){

  this->ft_1 = ft_1_data->wrench;
}

void URjointTest::cplCheckCallback(const std_msgs::msg::Int32::SharedPtr c_check){
  if (this->cpl_enable_check == 0 && c_check->data == 1){
    this->cpl_enable_check = c_check->data;
  }

  if(this->cpl_enable_check == 1 && c_check->data == 0){
    if(this->cpl_enable == 1){
      this->cpl_enable = 0;
      RCLCPP_INFO_STREAM(this->get_logger(),"Not compliance mode");
    }
    else{
      this->cpl_enable = 1;
      RCLCPP_INFO_STREAM(this->get_logger(),"Compliance mode");
    }
    this->cpl_enable_check = 0;
  }

}

double constrain_v(double v, double v_max, double v_min){
  if (v > v_max){
    v = v_max;
  }
  else if(v < v_min){
    v = v_min;
  }
  return v;
}

// Main script *************************************************************************************

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // auto node = rclcpp::Node::make_shared("ur16e_joint_test");
  auto sub_node = std::make_shared<URjointTest>();
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

  // Init data
  std::vector<std::string> joint_name;
  std::vector<double> joint_position, joint_velocity, joint_effort;
  std::string chain_start, chain_end, urdf_xml;
  double timeout;

  sub_node->declare_parameter<double>("timeout", 0.005);
  sub_node->declare_parameters<std::string>(
    std::string(),       // parameters are not namespaced
    std::map<std::string, std::string>{
    {"chain_start", std::string()},
    {"chain_end", std::string()},
    {"robot_description", std::string()},
  });

  sub_node->get_parameter("timeout", timeout);
  sub_node->get_parameter("chain_start", chain_start);
  sub_node->get_parameter("chain_end", chain_end);
  sub_node->get_parameter("robot_description", urdf_xml);

  if (chain_start.empty() || chain_end.empty()) {
    RCLCPP_FATAL(LOGGER, "Missing chain info in launch file");
    exit(-1);
  }

  // Create a JointState message
  sensor_msgs::msg::JointState joint_state;
  double current_joints[6] = {0,};
  double target_joints[6] = {0,};

  joint_state.name.resize(6);
  joint_state.position.resize(6);
  joint_state.effort.resize(6);
  joint_state.velocity.resize(6);

  std::vector<std::string> check_ur_j = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
  std::vector<double> default_joints = {-1.29955968, -1.96147873, -1.62985075, -1.1210595,   1.57079633,  0.27123665};

  joint_state.name = check_ur_j;
  joint_state.position = default_joints;

  // TRAC-IK setting start
  double eps = 1e-5;
  // This constructor parses the URDF loaded in rosparm urdf_xml into the
  // needed KDL structures.  We then pull these out to compare against the KDL
  // IK solver.
  TRAC_IK::TRAC_IK tracik_solver(chain_start, chain_end, urdf_xml, timeout, eps);

  KDL::Chain chain;
  KDL::JntArray ll, ul;  // lower joint limits, upper joint limits

  bool valid = tracik_solver.getKDLChain(chain);

  if (!valid) {
    RCLCPP_ERROR(sub_node->get_logger(), "There was no valid KDL chain found");
    return 0;
  }

  valid = tracik_solver.getKDLLimits(ll, ul);

  if (!valid) {
    RCLCPP_ERROR(sub_node->get_logger(), "There were no valid KDL joint limits found");
    return 0;
  }

  //check Dof
  assert(chain.getNrOfJoints() == ll.data.size());
  assert(chain.getNrOfJoints() == ul.data.size());

  RCLCPP_INFO(sub_node->get_logger(), "Using %d joints", chain.getNrOfJoints());

  // Set up KDL IK
  KDL::ChainFkSolverPos_recursive fk_solver(chain);  // Forward kin. solver

  // Create Nominal chain configuration midway between all joint limits
  KDL::JntArray nominal(chain.getNrOfJoints());

  for (uint j = 0; j < nominal.data.size(); j++) {
    nominal(j) = (ll(j) + ul(j)) / 2.0;
  }

  KDL::JntArray result;
  KDL::Frame end_effector_pose; // Desired eef - Pose
  KDL::Frame current_effector_pose;
  KDL::Frame current_effector_pose_r;
  KDL::Frame desired_effector_pose;
  KDL::Frame compliance_pose;
  int rc;
  KDL::JntArray q(chain.getNrOfJoints()); // Initial joints
  KDL::JntArray q_current(chain.getNrOfJoints()); // Initial joints

  for (int i=0; i<chain.getNrOfJoints(); i++){
    q(i) = default_joints[i];
  }

  std::vector<double> vec2(q.data.data(), q.data.data()+q.data.size());
  RCLCPP_INFO_STREAM(sub_node->get_logger(),"Init Joint q: ");
  for(int i=0; i<chain.getNrOfJoints(); i++){
    RCLCPP_INFO_STREAM(sub_node->get_logger(),"joint " << i+1<<" "<<vec2[i]);
  }

  fk_solver.JntToCart(q, end_effector_pose); // Set initial joint pose

  // Solve Inverse kinematics using Trac-IK
  rc = tracik_solver.CartToJnt(nominal, end_effector_pose, result);

  if (rc < 0){
    RCLCPP_ERROR(sub_node->get_logger(), "There were no valid end effector pose");
  }

  // Wrench set
  KDL::Wrench ft_sensor_wrench;
  KDL::Wrench base_wrench;

  // Compliance Set
  double x_kp, r_kp;
  int S_cpl[6] = {1, 1, 1, 1, 1, 1};  //dof
  KDL::Wrench target_F;  // 0,0,0,0,0,0
  // target_F.force(2) = -50.0;
  x_kp = 0.0000005;   // 0.0000008  0000005
  r_kp = 0.00001;     // 0.00001   00001
  sub_node->cpl_enable = 0;

  // Init setting

  int loop_c = 0;

  rclcpp::WallRate loop_rate(100);
  while(rclcpp::ok()){
    rclcpp::spin_some(sub_node);

    // // Get F/T Sensor data <- this data is the compensation of Compliant solver forces
    sub_node->ft_data.wrench.force.x = sub_node->ft_1.force.x;
    sub_node->ft_data.wrench.force.y = sub_node->ft_1.force.y;
    sub_node->ft_data.wrench.force.z = sub_node->ft_1.force.z;
    sub_node->ft_data.wrench.torque.x = sub_node->ft_1.torque.x;
    sub_node->ft_data.wrench.torque.y = sub_node->ft_1.torque.y;
    sub_node->ft_data.wrench.torque.z = sub_node->ft_1.torque.z;
    // // sub_node->publish_ft_data(sub_node->ft_data);

    // Get current Pose
    for(int i=0; i<chain.getNrOfJoints(); i++){
      q_current(i) = sub_node->joint_position[i];
    }
    fk_solver.JntToCart(q_current, current_effector_pose_r); // Set current joint pose
    // temp test!!
    fk_solver.JntToCart(q, current_effector_pose); // Set current joint pose

    ft_sensor_wrench.force.x(sub_node->ft_data.wrench.force.x);
    ft_sensor_wrench.force.y(sub_node->ft_data.wrench.force.y);
    ft_sensor_wrench.force.z(sub_node->ft_data.wrench.force.z);
    ft_sensor_wrench.torque.x(sub_node->ft_data.wrench.torque.x);
    ft_sensor_wrench.torque.y(sub_node->ft_data.wrench.torque.y);
    ft_sensor_wrench.torque.z(sub_node->ft_data.wrench.torque.z);

    geometry_msgs::msg::WrenchStamped ft_s_w;
    ft_s_w.wrench.force.x   = ft_sensor_wrench(0);
    ft_s_w.wrench.force.y   = ft_sensor_wrench(1);
    ft_s_w.wrench.force.z   = ft_sensor_wrench(2);
    ft_s_w.wrench.torque.x  = ft_sensor_wrench(3);
    ft_s_w.wrench.torque.y  = ft_sensor_wrench(4);
    ft_s_w.wrench.torque.z  = ft_sensor_wrench(5);
    // sub_node->publish_ft_data(ft_s_w);

    // data publish

    if(loop_c >= 100){//-- loop_rate x 100 = 5hz
      //ros2 result topic publish
      std_msgs::msg::Float64MultiArray ft_result;
      ft_result.data.push_back(1);
      ft_result.data.push_back(ft_s_w.wrench.force.x);
      ft_result.data.push_back(ft_s_w.wrench.force.y);
      ft_result.data.push_back(ft_s_w.wrench.force.z);
      ft_result.data.push_back(ft_s_w.wrench.torque.x);
      ft_result.data.push_back(ft_s_w.wrench.torque.y);
      ft_result.data.push_back(ft_s_w.wrench.torque.z);
      ft_result.data.push_back(2);
      ft_result.data.push_back(current_effector_pose_r.p.data[0]);
      ft_result.data.push_back(current_effector_pose_r.p.data[1]);
      ft_result.data.push_back(current_effector_pose_r.p.data[2]);
      sub_node->publish_result(ft_result);

      loop_c = 0;
    }

    ft_s_w.wrench.force.x = constrain_v(ft_s_w.wrench.force.x, 200, -200);
    ft_s_w.wrench.force.y = constrain_v(ft_s_w.wrench.force.y, 200, -200);
    ft_s_w.wrench.force.z = constrain_v(ft_s_w.wrench.force.z, 200, -200);
    ft_s_w.wrench.torque.x = constrain_v(ft_s_w.wrench.torque.x, 30, -30);
    ft_s_w.wrench.torque.y = constrain_v(ft_s_w.wrench.torque.y, 30, -30);
    ft_s_w.wrench.torque.z = constrain_v(ft_s_w.wrench.torque.z, 30, -30);


    // Get desired Sub Pose
    KDL::Vector dp(sub_node->desired_pose.linear.x, sub_node->desired_pose.linear.y, sub_node->desired_pose.linear.z);
    KDL::Rotation dr = KDL::Rotation::RPY(sub_node->desired_pose.angular.x, sub_node->desired_pose.angular.y, sub_node->desired_pose.angular.z);
    // Change local rotation
    desired_effector_pose.p = current_effector_pose.M * dp;
    desired_effector_pose.M = dr;

    // Compliance Control
    // postion
    KDL::Vector cpl_dp(S_cpl[0] * (-1) * (target_F.force(0) - ft_s_w.wrench.force.x),
                        S_cpl[1] * (-1) * (target_F.force(1) - ft_s_w.wrench.force.y),
                        S_cpl[2] * (-1) * (target_F.force(2) - ft_s_w.wrench.force.z));
    compliance_pose.p = x_kp * cpl_dp;
    // Change local rotation to base rotation
    compliance_pose.p = current_effector_pose.M * compliance_pose.p;
    // rotation
    compliance_pose.M = KDL::Rotation::RPY(S_cpl[3] * r_kp * (-1) * (target_F.torque(0) - ft_s_w.wrench.torque.x),
                                          S_cpl[4] * r_kp * (-1) * (target_F.torque(1) - ft_s_w.wrench.torque.y),
                                          S_cpl[5] * r_kp * (-1) * (target_F.torque(2) - ft_s_w.wrench.torque.z));

    // RCLCPP_INFO_STREAM(sub_node->get_logger(),"position f "<<compliance_pose.p.x() << " " << compliance_pose.p.y() << " "<< compliance_pose.p.z());
    if(sub_node->cpl_enable){
      // RCLCPP_INFO_STREAM(sub_node->get_logger(), "T BA's rotation");
      // RCLCPP_INFO_STREAM(sub_node->get_logger(), compliance_pose.M.data[0]<<" "<<compliance_pose.M.data[1]<<" "<<compliance_pose.M.data[2]);
      // RCLCPP_INFO_STREAM(sub_node->get_logger(), compliance_pose.M.data[3]<<" "<<compliance_pose.M.data[4]<<" "<<compliance_pose.M.data[5]);
      // RCLCPP_INFO_STREAM(sub_node->get_logger(), compliance_pose.M.data[6]<<" "<<compliance_pose.M.data[7]<<" "<<compliance_pose.M.data[8]);
      end_effector_pose.M = current_effector_pose.M * desired_effector_pose.M * compliance_pose.M;
      end_effector_pose.p = current_effector_pose.p + desired_effector_pose.p + compliance_pose.p;
    }
    else{
      end_effector_pose.M = current_effector_pose.M * desired_effector_pose.M;
      end_effector_pose.p = current_effector_pose.p + desired_effector_pose.p;
    }

    // RCLCPP_INFO_STREAM(sub_node->get_logger(),"end_effector_pose p "<<end_effector_pose.p.x() << " " << end_effector_pose.p.y() << " "<< end_effector_pose.p.z());

    // Get target joints q
    // Solve Inverse kinematics using Trac-IK
    rc = tracik_solver.CartToJnt(q, end_effector_pose, result);

    if (rc < 0){
      RCLCPP_ERROR(sub_node->get_logger(), "There were no valid end effector pose");
    }

    joint_state.header.stamp = sub_node->get_clock()->now();
    for(int i=0;i <6;i++){
      joint_state.position[i] = result(i);
      // RCLCPP_INFO_STREAM(LOGGER, i+1<<" q: "<<q(i)<<" result: "<<result(i));
    }
    // RCLCPP_INFO_STREAM(LOGGER,"T Pose x :" << end_effector_pose.p.x() << " y :"<< end_effector_pose.p.y() <<" z :"<<end_effector_pose.p.z());
    q = result;

    sub_node->publish_target_joints(joint_state);

    loop_rate.sleep();
    loop_c++;
  }



  return 0;
}
