#include <zmq.hpp>
#include <future>
#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "SaiModel.h"
#include "SaiPrimitives.h"
#include <yaml-cpp/yaml.h>

using namespace Eigen;

const std::string zeromq_server = "ipc:///tmp/zmq_motion_force_server";
const int ROBOT_GRIPPER_JOINTS = 7;
int NUM_ENVS = 1;
int NUM_WORKERS = 1;
const string robot_file = std::string(URDF_PATH) + "/scenes/panda_arm.urdf";

// Preallocate thread-local robot instances
std::vector<std::shared_ptr<SaiModel::SaiModel>> robot_pool;
std::vector<std::shared_ptr<SaiPrimitives::MotionForceTask>> mft_pool;
std::vector<std::shared_ptr<SaiPrimitives::JointTask>> jt_pool;

void updateRobotState(std::shared_ptr<SaiModel::SaiModel> robot,
                      const float* qpos, const float* qvel,
                      std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task) {

    VectorXd robot_q = Map<const VectorXf>(qpos, ROBOT_GRIPPER_JOINTS).cast<double>();
    VectorXd robot_dq = Map<const VectorXf>(qvel, ROBOT_GRIPPER_JOINTS).cast<double>();

    robot->setQ(robot_q);
    robot->setDq(robot_dq);
    robot->updateModel();
}

void compute_robot_joint_torques(std::shared_ptr<SaiModel::SaiModel> robot,
                                  std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task,
                                  std::shared_ptr<SaiPrimitives::JointTask> joint_task,
                                  float* torques_out, const float* des_cart_pos, const float* des_cart_orient, const float* force_or_motion_axis, const float* force_dim, const float* desired_force) {

    Vector3d d_force = Vector3d(desired_force[0], desired_force[1], desired_force[2]);
    motion_force_task->setGoalForce(d_force);
    Vector3d f_or_m_axis = Vector3d(force_or_motion_axis[0], force_or_motion_axis[1], force_or_motion_axis[2]);
    motion_force_task->parametrizeForceMotionSpaces(static_cast<int>(*force_dim),f_or_m_axis);

    Vector3d desired_pos = Vector3d(des_cart_pos[0], des_cart_pos[1], des_cart_pos[2]);

    std::cout << "desired_pos: " << desired_pos << std::endl;

    std::cout << "desired_force: " << d_force << std::endl;

    std::cout << "force or motion axis: " << f_or_m_axis << std::endl;
    
    std::cout << "force dim: " << static_cast<int>(*force_dim) << std::endl;

    VectorXd sent_quat(4);

    sent_quat << des_cart_orient[0], des_cart_orient[1], des_cart_orient[2], des_cart_orient[3];

    std::cout << "quat: " << sent_quat << std::endl;

    motion_force_task->setGoalPosition(desired_pos);
    Eigen::Quaterniond qd(static_cast<double>(des_cart_orient[3]),
                      static_cast<double>(des_cart_orient[0]),
                      static_cast<double>(des_cart_orient[1]),
                      static_cast<double>(des_cart_orient[2]));
    Matrix3d orient = qd.normalized().toRotationMatrix();
    motion_force_task->setGoalOrientation(orient);

    motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
    joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
    VectorXd control_torques = motion_force_task->computeTorques() + joint_task->computeTorques() + robot->jointGravityVector();

    Map<VectorXf>(torques_out, ROBOT_GRIPPER_JOINTS) = control_torques.cast<float>();
}

int main() {
    std::cout << "Starting New Motion Force Torque Server ..." << std::endl;

    // Parse YAML parallel config file

    YAML::Node parallel_config = YAML::LoadFile("./apps/zeromq/parallel_config.yaml");

    NUM_ENVS = parallel_config["NUM_ENVS"].as<int>();
    NUM_WORKERS = parallel_config["NUM_WORKERS"].as<int>();

    std::cout << "NUM ENVS : " << NUM_ENVS << " NUM_WORKERS : " << NUM_WORKERS << std::endl;

    robot_pool.resize(NUM_ENVS);
    mft_pool.resize(NUM_ENVS);
    jt_pool.resize(NUM_ENVS);



    // Initialize per-env robots and tasks
    for (int i = 0; i < NUM_ENVS; ++i) {
        robot_pool[i] = std::make_shared<SaiModel::SaiModel>(robot_file, false);
        Vector3d control_point = Vector3d(0, 0, 0.2015);
        Affine3d control_frame = Affine3d::Identity();
        control_frame.translation() = control_point;
        mft_pool[i] = std::make_shared<SaiPrimitives::MotionForceTask>(robot_pool[i], "link7", control_frame);
        jt_pool[i] = std::make_shared<SaiPrimitives::JointTask>(robot_pool[i]);
        mft_pool[i]->disableInternalOtg();
    }

    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind(zeromq_server);

    while (true) {
        zmq::message_t request;
        socket.recv(request, zmq::recv_flags::none);

        if (request.size() != NUM_ENVS * 28 * sizeof(float)) {
            std::cerr << "Unexpected message size : " << request.size() << " " << NUM_ENVS * 28 * sizeof(float)<< std::endl;
            continue;
        }

        const float* joint_data = static_cast<const float*>(request.data());
        float output_torques[NUM_ENVS * ROBOT_GRIPPER_JOINTS];

        NUM_WORKERS = std::min(NUM_WORKERS, NUM_ENVS);

        std::vector<std::thread> workers;
        int chunk_size = (NUM_ENVS + NUM_WORKERS - 1) / NUM_WORKERS;

        for (int w = 0; w < NUM_WORKERS; ++w) {
            workers.emplace_back([&, w]() {
                int start = w * chunk_size;
                int end = std::min(start + chunk_size, NUM_ENVS);
                for (int i = start; i < end; ++i) {
                    const float* pos = joint_data + i *28;
                    const float* orient = pos + 3;
                    const float* qpos = orient + 4;
                    const float* qvel = qpos + 7;
                    const float* force_or_motion_axis = qvel + 7;
                    const float* force_dim = force_or_motion_axis + 3;
                    const float* desired_force = force_dim + 1;
                
                    updateRobotState(robot_pool[i], qpos, qvel, mft_pool[i]);
                    compute_robot_joint_torques(robot_pool[i], mft_pool[i], jt_pool[i], output_torques + i * ROBOT_GRIPPER_JOINTS, pos, orient, force_or_motion_axis, force_dim, desired_force);
                }
            });
        }

        for (auto& t : workers) t.join();

        zmq::message_t reply(sizeof(output_torques));
        std::memcpy(reply.data(), output_torques, sizeof(output_torques));
        socket.send(reply, zmq::send_flags::none);
    }
}