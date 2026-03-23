#include "../include/test_gui/main_gui.hpp"
#include "../resources/ui_main_gui.h"

namespace
{
constexpr double kLinearStep = 0.01;
constexpr double kAngularStepDeg = 5.0;

geometry_msgs::msg::Quaternion quaternion_from_rpy_deg(double roll_deg, double pitch_deg, double yaw_deg)
{
    tf2::Quaternion quaternion;
    quaternion.setRPY(DEG2RAD(roll_deg), DEG2RAD(pitch_deg), DEG2RAD(yaw_deg));
    quaternion.normalize();

    geometry_msgs::msg::Quaternion q_msg;
    q_msg.x = quaternion.x();
    q_msg.y = quaternion.y();
    q_msg.z = quaternion.z();
    q_msg.w = quaternion.w();
    return q_msg;
}

void rpy_deg_from_quaternion(const geometry_msgs::msg::Quaternion & quaternion, double & roll_deg, double & pitch_deg, double & yaw_deg)
{
    tf2::Quaternion tf_quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
    tf2::Matrix3x3 rotation(tf_quaternion);
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    rotation.getRPY(roll, pitch, yaw);
    roll_deg = RAD2DEG(roll);
    pitch_deg = RAD2DEG(pitch);
    yaw_deg = RAD2DEG(yaw);
}
}

main_gui::main_gui(const std::shared_ptr<Ros2Node>& ros2_node, QWidget* parent)
  : QMainWindow(parent)
  , ros2_node(ros2_node)
  , ui(new Ui::main_gui)
{
    ui->setupUi(this);

    //icon set
    setWindowIcon(QIcon(QString::fromStdString(ament_index_cpp::get_package_share_directory("test_gui")) + "/resources/robot-icons-30500.png"));

    //timer count
    m_timer = new QTimer;
    connect(m_timer, SIGNAL(timeout()), this, SLOT(on_timer_count()));
    m_timer->start(50); //50ms

    //log model
    net_model = new QStringListModel(this);
    ctr_model = new QStringListModel(this);

    //data init
    this->ui->LE_TCPPORT1->setText("9999");
    ui->LE_J1_deg->setText("");
    ui->LE_J1_deg->setText("");
    ui->LE_J2_deg->setText("");
    ui->LE_J3_deg->setText("");
    ui->LE_J4_deg->setText("");
    ui->LE_J5_deg->setText("");
    ui->LE_J6_deg->setText("");
    ui->LE_target_ee_x->setText("");
    ui->LE_target_ee_y->setText("");
    ui->LE_target_ee_z->setText("");
    ui->LE_target_ee_roll->setText("");
    ui->LE_target_ee_pitch->setText("");
    ui->LE_target_ee_yaw->setText("");
    ui->LE_error_pitch->setText("");
    ui->LE_error_roll->setText("");
    ui->spinBox_gripper_pos->setValue(0);
    ui->spinBox_gripper_speed->setValue(0);
    ui->spinBox_gripper_power->setValue(255);
    ui->spinBox_gripper_pos_2->setValue(0);
    ui->spinBox_gripper_speed_2->setValue(0);
    ui->spinBox_gripper_power_2->setValue(255);

    set_arm_init_pose(LEFT);
    set_arm_init_pose(RIGHT);
    sync_arm_pose_fields();

    connect(ui->BTN_ee_targetmove, &QPushButton::clicked, this, [this]() { publish_arm_pose(LEFT); });
    connect(ui->BTN_ee_targetmove_2, &QPushButton::clicked, this, [this]() { publish_arm_pose(RIGHT); });
    connect(ui->BTN_ee_poseupd, &QPushButton::clicked, this, [this]() { target_poses_[LEFT] = pose_from_fields(LEFT); });
    connect(ui->BTN_ee_poseupd_2, &QPushButton::clicked, this, [this]() { target_poses_[RIGHT] = pose_from_fields(RIGHT); });
    connect(ui->BTN_ee_init, &QPushButton::clicked, this, [this]() { set_arm_init_pose(LEFT); publish_arm_pose(LEFT); });
    connect(ui->BTN_ee_init_2, &QPushButton::clicked, this, [this]() { set_arm_init_pose(RIGHT); publish_arm_pose(RIGHT); });
    connect(ui->BTN_ee_stop, &QPushButton::clicked, this, [this]() { publish_arm_pose(LEFT); });
    connect(ui->BTN_ee_stop_2, &QPushButton::clicked, this, [this]() { publish_arm_pose(RIGHT); });

    connect(ui->BTN_ee_x_up, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 0, kLinearStep); });
    connect(ui->BTN_ee_x_down, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 0, -kLinearStep); });
    connect(ui->BTN_ee_y_up, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 1, kLinearStep); });
    connect(ui->BTN_ee_y_down, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 1, -kLinearStep); });
    connect(ui->BTN_ee_z_up, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 2, kLinearStep); });
    connect(ui->BTN_ee_z_down, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 2, -kLinearStep); });
    connect(ui->BTN_ee_roll_up, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 3, kAngularStepDeg); });
    connect(ui->BTN_ee_roll_down, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 3, -kAngularStepDeg); });
    connect(ui->BTN_ee_pitch_up, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 4, kAngularStepDeg); });
    connect(ui->BTN_ee_pitch_down, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 4, -kAngularStepDeg); });
    connect(ui->BTN_ee_yaw_up, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 5, kAngularStepDeg); });
    connect(ui->BTN_ee_yaw_down, &QPushButton::clicked, this, [this]() { adjust_arm_pose(LEFT, 5, -kAngularStepDeg); });

    connect(ui->BTN_ee_x_up_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 0, kLinearStep); });
    connect(ui->BTN_ee_x_down_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 0, -kLinearStep); });
    connect(ui->BTN_ee_y_up_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 1, kLinearStep); });
    connect(ui->BTN_ee_y_down_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 1, -kLinearStep); });
    connect(ui->BTN_ee_z_up_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 2, kLinearStep); });
    connect(ui->BTN_ee_z_down_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 2, -kLinearStep); });
    connect(ui->BTN_ee_roll_up_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 3, kAngularStepDeg); });
    connect(ui->BTN_ee_roll_down_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 3, -kAngularStepDeg); });
    connect(ui->BTN_ee_pitch_up_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 4, kAngularStepDeg); });
    connect(ui->BTN_ee_pitch_down_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 4, -kAngularStepDeg); });
    connect(ui->BTN_ee_yaw_up_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 5, kAngularStepDeg); });
    connect(ui->BTN_ee_yaw_down_2, &QPushButton::clicked, this, [this]() { adjust_arm_pose(RIGHT, 5, -kAngularStepDeg); });

    connect(ui->spinBox_gripper_pos, qOverload<int>(&QSpinBox::valueChanged), ui->PB_Grip_pos, &QProgressBar::setValue);
    connect(ui->spinBox_gripper_pos_2, qOverload<int>(&QSpinBox::valueChanged), ui->PB_Grip_pos_2, &QProgressBar::setValue);
    connect(ui->spinBox_gripper_pos, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        ui->LE_Grip_state->setText(value > 127 ? "Close" : "Open");
    });
    connect(ui->spinBox_gripper_pos_2, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        ui->LE_Grip_state_2->setText(value > 127 ? "Close" : "Open");
    });
    connect(ui->BTN_target_grip, &QPushButton::clicked, this, [this]() {
        this->ros2_node->command_left_gripper_from_ui(ui->spinBox_gripper_pos->value());
    });
    connect(ui->BTN_target_nogrip, &QPushButton::clicked, this, [this]() {
        ui->spinBox_gripper_pos->setValue(0);
        this->ros2_node->command_left_gripper_from_ui(0);
    });
    connect(ui->BTN_target_grip_2, &QPushButton::clicked, this, [this]() {
        this->ros2_node->command_right_gripper_from_ui(ui->spinBox_gripper_pos_2->value());
    });
    connect(ui->BTN_target_nogrip_2, &QPushButton::clicked, this, [this]() {
        ui->spinBox_gripper_pos_2->setValue(0);
        this->ros2_node->command_right_gripper_from_ui(0);
    });

    ui->LE_Grip_state->setText("Open");
    ui->LE_Grip_state_2->setText("Open");

}

main_gui::~main_gui()
{
    delete ui;
}

//void main_gui::publish_button_clicked()
//{
////  if (!lineedit->text().isEmpty()){
//    // Make sure to convert from QString to std::string
////    ros2_node->publish_message(lineedit->text().toUtf8().constData());
////  }
//    qDebug()<<"test 123";
//}

void error(const char *msg){
    qDebug()<<msg;
}

void main_gui::on_timer_count(){
//    std::cout<<"timer test"<<std::endl;
    FachData();
}

void main_gui::FachData(){

    // Update data
    sync_joint_fields();
}

void main_gui::sync_arm_pose_fields()
{
    set_fields_from_pose(LEFT, target_poses_[LEFT]);
    set_fields_from_pose(RIGHT, target_poses_[RIGHT]);
}

void main_gui::sync_joint_fields()
{
    const auto left_joints = ros2_node->left_joint_degrees();
    const auto right_joints = ros2_node->right_joint_degrees();

    if (left_joints.size() >= 6) {
        set_line_edit_value(ui->LE_J1_deg, left_joints[0]);
        set_line_edit_value(ui->LE_J2_deg, left_joints[1]);
        set_line_edit_value(ui->LE_J3_deg, left_joints[2]);
        set_line_edit_value(ui->LE_J4_deg, left_joints[3]);
        set_line_edit_value(ui->LE_J5_deg, left_joints[4]);
        set_line_edit_value(ui->LE_J6_deg, left_joints[5]);
    }

    if (right_joints.size() >= 6) {
        set_line_edit_value(ui->LE_J1_deg_2, right_joints[0]);
        set_line_edit_value(ui->LE_J2_deg_2, right_joints[1]);
        set_line_edit_value(ui->LE_J3_deg_2, right_joints[2]);
        set_line_edit_value(ui->LE_J4_deg_2, right_joints[3]);
        set_line_edit_value(ui->LE_J5_deg_2, right_joints[4]);
        set_line_edit_value(ui->LE_J6_deg_2, right_joints[5]);
    }
}

geometry_msgs::msg::Pose main_gui::pose_from_fields(int arm_index) const
{
    geometry_msgs::msg::Pose pose;

    if (arm_index == LEFT) {
        pose.position.x = line_edit_value(ui->LE_target_ee_x);
        pose.position.y = line_edit_value(ui->LE_target_ee_y);
        pose.position.z = line_edit_value(ui->LE_target_ee_z);
        pose.orientation = quaternion_from_rpy_deg(
            line_edit_value(ui->LE_target_ee_roll),
            line_edit_value(ui->LE_target_ee_pitch),
            line_edit_value(ui->LE_target_ee_yaw));
    } else {
        pose.position.x = line_edit_value(ui->LE_target_ee_x_2);
        pose.position.y = line_edit_value(ui->LE_target_ee_y_2);
        pose.position.z = line_edit_value(ui->LE_target_ee_z_2);
        pose.orientation = quaternion_from_rpy_deg(
            line_edit_value(ui->LE_target_ee_roll_2),
            line_edit_value(ui->LE_target_ee_pitch_2),
            line_edit_value(ui->LE_target_ee_yaw_2));
    }

    return pose;
}

void main_gui::set_fields_from_pose(int arm_index, const geometry_msgs::msg::Pose & pose)
{
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
    rpy_deg_from_quaternion(pose.orientation, roll_deg, pitch_deg, yaw_deg);

    if (arm_index == LEFT) {
        set_line_edit_value(ui->LE_target_ee_x, pose.position.x);
        set_line_edit_value(ui->LE_target_ee_y, pose.position.y);
        set_line_edit_value(ui->LE_target_ee_z, pose.position.z);
        set_line_edit_value(ui->LE_target_ee_roll, roll_deg);
        set_line_edit_value(ui->LE_target_ee_pitch, pitch_deg);
        set_line_edit_value(ui->LE_target_ee_yaw, yaw_deg);
    } else {
        set_line_edit_value(ui->LE_target_ee_x_2, pose.position.x);
        set_line_edit_value(ui->LE_target_ee_y_2, pose.position.y);
        set_line_edit_value(ui->LE_target_ee_z_2, pose.position.z);
        set_line_edit_value(ui->LE_target_ee_roll_2, roll_deg);
        set_line_edit_value(ui->LE_target_ee_pitch_2, pitch_deg);
        set_line_edit_value(ui->LE_target_ee_yaw_2, yaw_deg);
    }
}

void main_gui::publish_arm_pose(int arm_index)
{
    target_poses_[arm_index] = pose_from_fields(arm_index);
    set_fields_from_pose(arm_index, target_poses_[arm_index]);

    if (arm_index == LEFT) {
        ros2_node->publish_left_target_pose(target_poses_[arm_index]);
    } else {
        ros2_node->publish_right_target_pose(target_poses_[arm_index]);
    }
}

void main_gui::adjust_arm_pose(int arm_index, int axis, double delta)
{
    target_poses_[arm_index] = pose_from_fields(arm_index);

    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
    rpy_deg_from_quaternion(target_poses_[arm_index].orientation, roll_deg, pitch_deg, yaw_deg);

    switch (axis) {
    case 0: target_poses_[arm_index].position.x += delta; break;
    case 1: target_poses_[arm_index].position.y += delta; break;
    case 2: target_poses_[arm_index].position.z += delta; break;
    case 3: roll_deg += delta; break;
    case 4: pitch_deg += delta; break;
    case 5: yaw_deg += delta; break;
    default: break;
    }

    target_poses_[arm_index].orientation = quaternion_from_rpy_deg(roll_deg, pitch_deg, yaw_deg);
    set_fields_from_pose(arm_index, target_poses_[arm_index]);
    publish_arm_pose(arm_index);
}

void main_gui::set_arm_init_pose(int arm_index)
{
    geometry_msgs::msg::Pose pose;
    pose.orientation = quaternion_from_rpy_deg(0.0, 0.0, 0.0);

    if (arm_index == LEFT) {
        pose.position.x = 0.45;
        pose.position.y = 0.20;
        pose.position.z = 0.65;
    } else {
        pose.position.x = 0.45;
        pose.position.y = -0.20;
        pose.position.z = 0.65;
    }

    target_poses_[arm_index] = pose;
    set_fields_from_pose(arm_index, pose);
}

double main_gui::line_edit_value(const QLineEdit * line_edit, double fallback)
{
    bool ok = false;
    const double value = line_edit->text().toDouble(&ok);
    return ok ? value : fallback;
}

void main_gui::set_line_edit_value(QLineEdit * line_edit, double value)
{
    line_edit->setText(QString::number(value, 'f', 3));
}


void main_gui::on_actionConnect_Panel_triggered()
{
    ui->dockWidget->show();
}
