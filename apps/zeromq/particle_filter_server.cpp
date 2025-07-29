/**
 * @file particle_filter_server.cpp
 * @author William Chong (wmchong@stanford.edu)
 * @brief 
 * @version 0.1
 * @date 2024-07-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <zmq.hpp>
#include <queue>
#include <future>
#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <ForceSpaceParticleFilter.h>
#include <yaml-cpp/yaml.h>

using namespace Eigen;

const std::string zeromq_server = "ipc:///tmp/zmq_pf_server";
const std::string yaml_file = "./resources/pfilter_settings.yaml";
const int ROBOT_INFO = 12;  // motion command (3) + force command (3) + measured velocity (3) + measured force (3)
const int num_envs = 5;
int num_workers = 1;

/*
    YAML file parameters
*/
int N_PARTICLES = 1000;
double FILTER_FREQ = 50;  // hz 
int QUEUE_SIZE = 20;
double F_LOW = 0;
double F_HIGH = 5;
double V_LOW = 1e-2;
double V_HIGH = 7e-2;
double F_LOW_ADD = 5;
double F_HIGH_ADD = 10;
double V_LOW_ADD = 2e-2;
double V_HIGH_ADD = 1e-1;

/*
    Particle filter output
*/
struct PFilterOutput {
    Vector3d force_or_motion_axis;
    int force_space_dimension;

    PFilterOutput(const Vector3d& force_or_motion_axis = Vector3d::Zero(),
                  const int force_space_dimension = 0) :
                  force_or_motion_axis(force_or_motion_axis),
                  force_space_dimension(force_space_dimension) {}
};

// struct PFilterOutput {
//     Matrix3d sigma_force;
//     Matrix3d sigma_motion;
//     Vector3d force_or_motion_axis;
//     std::vector<Vector3d> force_axes;
//     bool flag_force_to_free;
//     int force_space_dimension;

//     PFilterOutput(const Matrix3d& sigma_force,
//                   const Matrix3d& sigma_motion,
//                   const Vector3d& force_or_motion_axis,
//                   const std::vector<Vector3d>& force_axes,
//                   const bool flag_force_to_free,
//                   const int force_space_dimension) :
//                   sigma_force(sigma_force),
//                   sigma_motion(sigma_motion),
//                   force_or_motion_axis(force_or_motion_axis),
//                   force_axes(force_axes),
//                   flag_force_to_free(flag_force_to_free),
//                   force_space_dimension(force_space_dimension) {}
// };

//------------------------------------------------------------------------------
/**
 * @brief Checks if all elements in a queue are the same
 */
template<typename T>
bool allElementsSame(const std::queue<T>& q) {
	if (q.empty()) {
		return true; // An empty queue has all elements the same (technically)
	}

	// Get the first element
	T firstElement = q.front();

	// Iterate through the queue
	std::queue<T> tempQueue = q; // Create a copy of the original queue
	while (!tempQueue.empty()) {
		// If any element is different from the first element, return false
		if (tempQueue.front() != firstElement) {
			return false;
		}
		tempQueue.pop(); // Remove the front element
	}

	return true; // All elements are the same
}

//------------------------------------------------------------------------------
Eigen::Vector3d floatPtrToVector3d(const float* data) {
    if (!data) {
        throw std::invalid_argument("Null pointer passed to floatPtrToVector3d.");
    }
    return Eigen::Vector3d(static_cast<double>(data[0]),
                           static_cast<double>(data[1]),
                           static_cast<double>(data[2]));
}

// Preallocate thread-local pfilter instances
std::vector<std::shared_ptr<WireConnectAgent::ForceSpaceParticleFilter>> pfilter_pool(num_envs);
std::vector<std::queue<int>> force_dimension_queue(num_envs);
std::vector<PFilterOutput> pfilter_output(num_envs);

//------------------------------------------------------------------------------
void updateParticleFilter(std::shared_ptr<WireConnectAgent::ForceSpaceParticleFilter> pfilter,
                          PFilterOutput& pfilter_output, 
                          std::queue<int>& force_dimension_queue,
                          const Vector3d& motion_control,
                          const Vector3d& force_control,
                          const Vector3d& measured_velocity,
                          const Vector3d& measured_force) {

    // std::cout << "motion control: " << motion_control.transpose() << "\n";
    // std::cout << "force control: " << force_control.transpose() << "\n";
    // std::cout << "measured velocity: " << measured_velocity.transpose() << "\n";
    // std::cout << "measured force: " << measured_force.transpose() << "\n";

    pfilter->update(motion_control, force_control, measured_velocity, measured_force);

    pfilter_output.force_space_dimension = pfilter->getForceSpaceDimension();
    pfilter_output.force_or_motion_axis = pfilter->getForceOrMotionAxis();
    force_dimension_queue.pop();
    force_dimension_queue.push(pfilter_output.force_space_dimension);
    // pfilter_output.flag_filter_force_to_free = false;

    // std::cout << pfilter_output.force_or_motion_axis.transpose() << "\n";

    // bool all_elements_same = allElementsSame(force_dimension_queue);

    // if (pfilter_output.force_space_dimension == 0) {
    //     // make sure that past classifications must be 0 to be free-space, otherwise keep previous sigma
    //     if (all_elements_same && pfilter_output.force_space_dimension == 0) {
    //         // pfilter_output.flag_filter_force_to_free = true;
    //         // pfilter_output.sigma_force = pfilter.getSigmaForce();
    //         // pfilter_output.sigma_motion = Matrix3d::Identity() - pfilter_output.sigma_force;
    //         // pfilter_output.force_axes = pfilter.getForceAxes();
    //         pfilter_output.force_or_motion_axis = pfilter->getForceOrMotionAxis();
    //     }
    // } else {
    //     // pfilter_output.sigma_force = pfilter.getSigmaForce(); 
    //     // pfilter_output.sigma_motion = Matrix3d::Identity() - sigma_force;
    //     // pfilter_output.force_axes = pfilter.getForceAxes();
    //     pfilter_output.force_or_motion_axis = pfilter->getForceOrMotionAxis();
    // }
}

//------------------------------------------------------------------------------
void resetParticleFilter(std::shared_ptr<WireConnectAgent::ForceSpaceParticleFilter> pfilter) {
    pfilter->reset();
}

//------------------------------------------------------------------------------
void resetForceDimensionQueue(std::queue<int>& force_dimension_queue) {
    const int queue_size = force_dimension_queue.size();

    while (!force_dimension_queue.empty()) {
            force_dimension_queue.pop();
        }

    // initialize classification queue with free-space 
    for (int i = 0; i < queue_size; ++i) {
        force_dimension_queue.push(0);  
    }
}

//------------------------------------------------------------------------------
void getFilterOutput(int& force_dimension,
                     Vector3d& force_or_motion_axis,
                     float* filter_out) {
    VectorXf output(4);
    output(0) = static_cast<float>(force_dimension);                   // set first element
    output.segment<3>(1) = force_or_motion_axis.cast<float>();        // set next 3 elements

    Map<VectorXf>(filter_out, 4) = output;  // copy to raw float*
}

// void updateRobotState(std::shared_ptr<SaiModel::SaiModel> robot,
//                       const float* qpos, const float* qvel,
//                       std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task) {

//     VectorXd robot_q = Map<const VectorXf>(qpos, ROBOT_GRIPPER_JOINTS).cast<double>();
//     VectorXd robot_dq = Map<const VectorXf>(qvel, ROBOT_GRIPPER_JOINTS).cast<double>();

//     robot->setQ(robot_q);
//     robot->setDq(robot_dq);
//     robot->updateModel();
// }

// void compute_robot_joint_torques(std::shared_ptr<SaiModel::SaiModel> robot,
//                                   std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task,
//                                   std::shared_ptr<SaiPrimitives::JointTask> joint_task,
//                                   float* torques_out, const float* des_cart_pos, const float* des_cart_orient) {

//     motion_force_task->setGoalPosition(Vector3d(des_cart_pos[0], des_cart_pos[1], des_cart_pos[2]));
//     Eigen::Quaterniond qd(static_cast<double>(des_cart_orient[0]),
//                       static_cast<double>(des_cart_orient[1]),
//                       static_cast<double>(des_cart_orient[2]),
//                       static_cast<double>(des_cart_orient[3]));
//     Matrix3d orient = qd.normalized().toRotationMatrix();
//     motion_force_task->setGoalOrientation(orient);

//     motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
//     joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
//     VectorXd control_torques = motion_force_task->computeTorques() + joint_task->computeTorques() + robot->jointGravityVector();

//     Map<VectorXf>(torques_out, ROBOT_GRIPPER_JOINTS) = control_torques.cast<float>();
// }

int main() {
    std::cout << "Starting Particle Filter Server ..." << std::endl;

    // Process controller yaml file 
	YAML::Node config = YAML::LoadFile("./resources/pfilter_settings.yaml");

    // Parse YAML config file
    if (config["filter"]) {
		YAML::Node current_node = config["filter"];
		N_PARTICLES = current_node["n_particles"].as<double>();
		FILTER_FREQ = current_node["filter_freq"].as<double>();
		QUEUE_SIZE = current_node["queue_size"].as<double>();
		F_LOW = current_node["f_low"].as<double>();
		F_HIGH = current_node["f_high"].as<double>();
		V_LOW = current_node["v_low"].as<double>();
		V_HIGH = current_node["v_high"].as<double>();
		F_LOW_ADD = current_node["f_low_add"].as<double>();
		F_HIGH_ADD = current_node["f_high_add"].as<double>();
		V_LOW_ADD = current_node["v_low_add"].as<double>();
		V_HIGH_ADD = current_node["v_high_add"].as<double>();
	}

    // Initialize per-env particle filters
    for (int i = 0; i < num_envs; ++i) {
        pfilter_pool[i] = std::make_shared<WireConnectAgent::ForceSpaceParticleFilter>(N_PARTICLES);
        pfilter_pool[i]->setParameters(0, 0.025, 0.3, 0.05);  // from paper implementation
        pfilter_pool[i]->setWeightingParameters(F_LOW, F_HIGH, V_LOW, V_HIGH, F_LOW_ADD, F_HIGH_ADD, V_LOW_ADD, V_HIGH_ADD);

        for (int j = 0; j < QUEUE_SIZE; ++j) {
            force_dimension_queue[i].push(0);  // init with free-space classification 
        }
    }

    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind(zeromq_server);

    while (true) {
        zmq::message_t request;
        socket.recv(request, zmq::recv_flags::none);

        bool reset_filter = false;
        if (request.size() == sizeof(float)) {
            std::cout << "Reset particle filter message received\n";
            reset_filter = true;

            for (auto& pfilter : pfilter_pool) {
                pfilter->reset();
            }

            float filter_output[4];
            zmq::message_t reply(sizeof(filter_output));
            std::memcpy(reply.data(), filter_output, sizeof(filter_output));
            socket.send(reply, zmq::send_flags::none);

            continue;

        } else if (request.size() != num_envs * ROBOT_INFO * sizeof(float)) {
            std::cerr << "Unexpected message size : " << request.size() << std::endl;
            continue;
        }

        const float* force_data = static_cast<const float*>(request.data());
        float filter_output[num_envs * 4];

        num_workers = std::min(num_workers, num_envs);

        std::vector<std::thread> workers;
        int chunk_size = (num_envs + num_workers - 1) / num_workers;

        for (int w = 0; w < num_workers; ++w) {
            workers.emplace_back([&, w]() {
                int start = w * chunk_size;
                int end = std::min(start + chunk_size, num_envs);
                for (int i = start; i < end; ++i) {

                    const float* motion_control = force_data + i * 3 + 0;
                    const float* force_control = force_data + i * 3 + 3;
                    const float* measured_velocity = force_data + i * 3 + 6;
                    const float* measured_force = force_data + i * 3 + 9;

                    // update particle filter 
                    updateParticleFilter(pfilter_pool[i], 
                                            pfilter_output[i], 
                                            force_dimension_queue[i], 
                                            floatPtrToVector3d(motion_control), 
                                            floatPtrToVector3d(force_control), 
                                            floatPtrToVector3d(measured_velocity), 
                                            floatPtrToVector3d(measured_force));

                    // send pfilter output
                    // force dimension, force or motion axis 
                    getFilterOutput(pfilter_output[i].force_space_dimension, pfilter_output[i].force_or_motion_axis, filter_output + i * 4);                    
                }
            });
        }

        for (auto& t : workers) t.join();

        zmq::message_t reply(sizeof(filter_output));
        std::memcpy(reply.data(), filter_output, sizeof(filter_output));
        socket.send(reply, zmq::send_flags::none);
    }
}






