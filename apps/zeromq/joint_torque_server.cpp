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
const int num_envs = 1;
int num_workers = 1;
const string robot_file = std::string(URDF_PATH) + "/scenes/panda_arm.urdf";

// Preallocate thread-local robot instances
std::vector<std::shared_ptr<SaiModel::SaiModel>> robot_pool(num_envs);
std::vector<std::shared_ptr<SaiPrimitives::MotionForceTask>> mft_pool(num_envs);
std::vector<std::shared_ptr<SaiPrimitives::JointTask>> jt_pool(num_envs);

void updateRobotState(std::shared_ptr<SaiModel::SaiModel> robot,
                      const float* qpos, const float* qvel,
                      std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task) {

    VectorXd robot_q = Map<const VectorXf>(qpos, ROBOT_GRIPPER_JOINTS).cast<double>();
    VectorXd robot_dq = Map<const VectorXf>(qvel, ROBOT_GRIPPER_JOINTS).cast<double>();

    // std::cout << "Robot q: " << robot_q << std::endl;
    // std::cout << "Robot dq: " << robot_dq << std::endl;

    robot->setQ(robot_q);
    robot->setDq(robot_dq);
    robot->updateModel();
}

void compute_robot_joint_torques(std::shared_ptr<SaiModel::SaiModel> robot,
                                  std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task,
                                  std::shared_ptr<SaiPrimitives::JointTask> joint_task,
                                  float* torques_out, const float* des_cart_pos, const float* des_cart_orient) {

    Vector3d goalPos = Vector3d(des_cart_pos[0], des_cart_pos[1], des_cart_pos[2]);

    motion_force_task->setGoalPosition(Vector3d(des_cart_pos[0], des_cart_pos[1], des_cart_pos[2]));
    Eigen::Quaterniond qd(static_cast<double>(des_cart_orient[3]),
                      static_cast<double>(des_cart_orient[0]),
                      static_cast<double>(des_cart_orient[1]),
                      static_cast<double>(des_cart_orient[2]));
    Matrix3d orient = qd.normalized().toRotationMatrix();

    // std::cout << "goal orient: " <<  orient << std::endl;
    // std::cout << "goal pos: " << goalPos << std::endl;
    motion_force_task->setGoalOrientation(orient);

    motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
    joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
    VectorXd control_torques = motion_force_task->computeTorques() + joint_task->computeTorques() + robot->jointGravityVector();

    Map<VectorXf>(torques_out, ROBOT_GRIPPER_JOINTS) = control_torques.cast<float>();
}

int main() {
    std::cout << "Starting Joint Torque Server ..." << std::endl;
    // Initialize per-env robots and tasks
    for (int i = 0; i < num_envs; ++i) {
        robot_pool[i] = std::make_shared<SaiModel::SaiModel>(robot_file, false);
        Vector3d control_point = Vector3d(0, 0, 0.2015);
        Affine3d control_frame = Affine3d::Identity();
        control_frame.translation() = control_point;
        mft_pool[i] = std::make_shared<SaiPrimitives::MotionForceTask>(robot_pool[i], "link7", control_frame);
        jt_pool[i] = std::make_shared<SaiPrimitives::JointTask>(robot_pool[i]);
        mft_pool[i]->setPosControlGains(400, 40, 0);
	    mft_pool[i]->setOriControlGains(400, 40, 0);
        mft_pool[i]->disableInternalOtg();
    }

    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind(zeromq_server);

    int request_count = 0;

    while (true) {
        zmq::message_t request;
        socket.recv(request, zmq::recv_flags::none);

        if (request.size() != num_envs * 21 * sizeof(float)) {
            std::cerr << "Unexpected message size : " << request.size() << std::endl;
            continue;
        }

        // std::cout << "request count: " << request_count << std::endl;
        // request_count += 1;

        const float* joint_data = static_cast<const float*>(request.data());
        float output_torques[num_envs * ROBOT_GRIPPER_JOINTS];

        num_workers = std::min(num_workers, num_envs);

        std::vector<std::thread> workers;
        int chunk_size = (num_envs + num_workers - 1) / num_workers;

        for (int w = 0; w < num_workers; ++w) {
            workers.emplace_back([&, w]() {
                int start = w * chunk_size;
                int end = std::min(start + chunk_size, num_envs);
                for (int i = start; i < end; ++i) {
                    const float* pos = joint_data + i * 21;
                    const float* orient = pos + 3;
                    const float* qpos = orient + 4;
                    const float* qvel = qpos + 7;

                    updateRobotState(robot_pool[i], qpos, qvel, mft_pool[i]);
                    compute_robot_joint_torques(robot_pool[i], mft_pool[i], jt_pool[i], output_torques + i * ROBOT_GRIPPER_JOINTS, pos, orient);
                }
            });
        }

        for (auto& t : workers) t.join();


        zmq::message_t reply(sizeof(output_torques));
        std::memcpy(reply.data(), output_torques, sizeof(output_torques));
        socket.send(reply, zmq::send_flags::none);
    }
}






