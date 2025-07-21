#include <zmq.hpp>
#include <future>
#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "SaiModel.h"
#include "SaiPrimitives.h"

using namespace Eigen;

const std::string zeromq_server = "ipc:///tmp/zmq_torque_server";
const int ROBOT_GRIPPER_JOINTS = 7;
const int num_envs = 5;
const int num_workers = 2;
const string robot_file = std::string(URDF_PATH) + "/scenes/rizon4spayload.urdf";

// Preallocate thread-local robot instances
std::vector<std::shared_ptr<SaiModel::SaiModel>> robot_pool(num_envs);
std::vector<std::shared_ptr<SaiPrimitives::MotionForceTask>> mft_pool(num_envs);
std::vector<std::shared_ptr<SaiPrimitives::JointTask>> jt_pool(num_envs);

void updateRobotState(std::shared_ptr<SaiModel::SaiModel> robot,
                      const float* qpos, const float* qvel,
                      const float* des_cart_pos, const float* des_cart_orient,
                      std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task) {

    VectorXd robot_q = Map<const VectorXf>(qpos, ROBOT_GRIPPER_JOINTS).cast<double>();
    VectorXd robot_dq = Map<const VectorXf>(qvel, ROBOT_GRIPPER_JOINTS).cast<double>();
    Eigen::Quaterniond qd(static_cast<double>(des_cart_orient[0]),
                      static_cast<double>(des_cart_orient[1]),
                      static_cast<double>(des_cart_orient[2]),
                      static_cast<double>(des_cart_orient[3]));
    Matrix3d orient = qd.normalized().toRotationMatrix();

    robot->setQ(robot_q);
    robot->setDq(robot_dq);
    robot->updateModel();

    motion_force_task->setGoalPosition(Vector3d(des_cart_pos[0], des_cart_pos[1], des_cart_pos[2]));
    motion_force_task->setGoalOrientation(orient);
}

void compute_robot_joint_torques(std::shared_ptr<SaiModel::SaiModel> robot,
                                  std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task,
                                  std::shared_ptr<SaiPrimitives::JointTask> joint_task,
                                  float* torques_out) {

    motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
    joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
    VectorXd control_torques = motion_force_task->computeTorques() + joint_task->computeTorques() + robot->jointGravityVector();

    Map<VectorXf>(torques_out, ROBOT_GRIPPER_JOINTS) = control_torques.cast<float>();
}

int main() {
    // Initialize per-env robots and tasks
    for (int i = 0; i < num_envs; ++i) {
        robot_pool[i] = std::make_shared<SaiModel::SaiModel>(robot_file, false);
        mft_pool[i] = std::make_shared<SaiPrimitives::MotionForceTask>(robot_pool[i], "link7", Affine3d::Identity());
        jt_pool[i] = std::make_shared<SaiPrimitives::JointTask>(robot_pool[i]);
        mft_pool[i]->disableInternalOtg();
    }

    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind(zeromq_server);

    while (true) {
        zmq::message_t request;
        socket.recv(request, zmq::recv_flags::none);

        if (request.size() != num_envs * 21 * sizeof(float)) {
            std::cerr << "Unexpected message size : " << request.size() << std::endl;
            continue;
        }

        const float* joint_data = static_cast<const float*>(request.data());
        float output_torques[num_envs * ROBOT_GRIPPER_JOINTS];

        int num_workers = std::min(num_workers, num_envs);

        std::vector<std::thread> workers;
        int chunk_size = (num_envs + num_workers - 1) / num_workers;

        for (int w = 0; w < num_workers; ++w) {
            workers.emplace_back([&, w]() {
                int start = w * chunk_size;
                int end = std::min(start + chunk_size, num_envs);
                for (int i = start; i < end; ++i) {
                    const float* pos = joint_data + i * 7;
                    const float* orient = joint_data + i * 7 + 3;
                    const float* qpos = joint_data + num_envs * 7 + i * 7;
                    const float* qvel = joint_data + num_envs * 14 + i * 7;

                    updateRobotState(robot_pool[i], qpos, qvel, pos, orient, mft_pool[i]);
                    compute_robot_joint_torques(robot_pool[i], mft_pool[i], jt_pool[i], output_torques + i * ROBOT_GRIPPER_JOINTS);
                }
            });
        }

        for (auto& t : workers) t.join();


        zmq::message_t reply(sizeof(output_torques));
        std::memcpy(reply.data(), output_torques, sizeof(output_torques));
        socket.send(reply, zmq::send_flags::none);
    }
}






