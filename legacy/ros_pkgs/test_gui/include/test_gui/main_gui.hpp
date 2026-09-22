#pragma once

#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include <QBoxLayout>
#include <QTimer>
#include <QStringListModel>
#include <QString>
#include <QObject>
#include <QTabWidget>
#include <QMessageBox>
#include <QDebug>
#include <QLineEdit>
#include <array>

#include <geometry_msgs/msg/pose.hpp>

#include "ros2node.hpp"
#include "test_gui/config.hpp"

#include <string>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>
#include <math.h>
#include "ament_index_cpp/get_package_share_directory.hpp"


QT_BEGIN_NAMESPACE
namespace Ui { class main_gui; }
QT_END_NAMESPACE

class main_gui : public QMainWindow
{
    Q_OBJECT
public:
    explicit main_gui(const std::shared_ptr<Ros2Node>&  ros2_node, QWidget* parent = nullptr);
    ~main_gui() override;

    void FachData();
    void log_t(QStringListModel &t_model,QString msg);
    void sync_arm_pose_fields();
    void sync_joint_fields();
    geometry_msgs::msg::Pose pose_from_fields(int arm_index) const;
    void set_fields_from_pose(int arm_index, const geometry_msgs::msg::Pose & pose);
    void publish_arm_pose(int arm_index);
    void adjust_arm_pose(int arm_index, int axis, double delta);
    void set_arm_init_pose(int arm_index);
    static double line_edit_value(const QLineEdit * line_edit, double fallback = 0.0);
    static void set_line_edit_value(QLineEdit * line_edit, double value);

    //init data

public Q_SLOTS:

private slots:
    void on_timer_count();
    void on_actionConnect_Panel_triggered();

private:

    Ui::main_gui *ui;

//    void publish_button_clicked();

    const std::shared_ptr<Ros2Node> ros2_node;

    QPushButton* publish_button;
    QTimer* m_timer;
    QStringListModel *net_model, *ctr_model;
    std::array<geometry_msgs::msg::Pose, 2> target_poses_;

};
