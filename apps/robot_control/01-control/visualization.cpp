/**
 * @file visualization.cpp
 * @author William Chong (wmchong@stanford.edu)
 * @brief 
 * @version 0.1
 * @date 2022-12-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

/*
	Graphically displays the current robot state and the proxy.
	- Log for training:
		- State: Actual robot end-effector position and orientation 
				 Desired robot end-effector position and orientation 
		- Action: Haptic robot end-effector proxy position and orientation 
		- Images: 96 x 96 (RGB)
	- Simulation environment reset with button press
*/

#include <math.h>
#include <signal.h>

#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <fstream>
#include <filesystem>

#include "Sai2Graphics.h"
#include "Sai2Model.h"
#include "Sai2Simulation.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"

#include "redis_keys.h"

bool fSimulationRunning = false;
void sighandler(int) { fSimulationRunning = false; }

using namespace std;
using namespace Eigen;
using namespace chai3d;

// specify urdf and robots 
const string world_file = "./resources/blank_world.urdf";
const string robot_file = "./resources/panda_arm.urdf";
const string robot_name = "panda";
// const string proxy_object_name = "proxy_frame";
const string camera_name = "camera_fixed";
const string recording_camera_name = "camera_recording";

// specify last link parameters 
const string control_link = "end-effector";
const Vector3d control_point = Vector3d(0, 0, 0.17);
const Vector3d sensor_pos_in_link = Vector3d(0, 0, 0.045 + 0.107);
const string proxy_name = "proxy_end_effector";

// simulation thread
void simulation(shared_ptr<Sai2Model::Sai2Model> robot, 
				shared_ptr<Sai2Simulation::Sai2Simulation> sim);
void logThread(const Vector3d& ee_pos_init, const Matrix3d& ee_ori_init);

// global
mutex mutex_update;

enum State {
	FREE_SPACE = 0,
	CONTACT
};

int main(int argc, char *argv[]) {
	cout << "Loading URDF world model file: " << world_file << endl;

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// connect redis client
	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();

	// // overwrite redis keys with simulation keys 
	// JOINT_ANGLES_KEY = "sai2::simulation::sensors::q";
	// JOINT_VELOCITIES_KEY = "sai2::simulation::sensors::dq";
	// JOINT_TORQUES_COMMANDED_KEY = "sai2::simulation::actuators::fgc";
	// FORCE_SENSOR_KEY = "sai2::simulation::sensors::force_moment";
	// ROBOT_RUNNING_KEY = "sai2::simulation::running";

	// load graphics scene
	auto graphics = make_shared<Sai2Graphics::Sai2Graphics>(world_file);
	// graphics->showTransparency(true, robot_name, 0.95);
	graphics->setBackgroundColor(66.0/255, 135.0/255, 245.0/255);
	// graphics->addUIForceInteraction(robot_name);
	// graphics->setBackgroundColor(1, 1, 1);
	// graphics->showLinkFrame(true, robot_name, ee_link_name, 0.15);  // can add frames for different links
	graphics->showObjectLinkFrame(true, proxy_name, 0.15);
	// graphics->showObjectTransparency(true, "Hole", 0.5);

	// graphics->addFrameBuffer("camera_recording", 96, 96);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file, false);
	int dof = robot->dof();
	VectorXd q_init(dof);
	q_init << 0, -25, 0, -135, 0, 105, 0;
	q_init *= M_PI / 180;
	robot->setQ(q_init);
	robot->setDq(VectorXd::Zero(dof));
	robot->updateModel();

	Affine3d control_transform = Affine3d::Identity();
	control_transform.translation() = control_point;

	// get initial ee values
	Vector3d ee_pos_init = robot->position(control_link, control_point);
	Matrix3d ee_ori_init = robot->rotation(control_link);

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);
	sim->setJointPositions(robot_name, robot->q());
	sim->setJointVelocities(robot_name, robot->dq());

    // set contact parameters
    sim->setCollisionRestitution(0.0);
    sim->setCoeffFrictionStatic(0.0);
    sim->setCoeffFrictionDynamic(0.0);	

	// create simulated force sensor
	Affine3d T_sensor = Affine3d::Identity();
	T_sensor.translation() = sensor_pos_in_link;
	sim->addSimulatedForceSensor(robot_name, control_link, T_sensor, 15.0);
	graphics->addForceSensorDisplay(sim->getAllForceSensorData()[0]);

	// set proxy information 
	Affine3d proxy_pose = Affine3d::Identity();
	// proxy_pose.translation() = robot->position(control_link, Vector3d::Zero());
	proxy_pose.translation() = robot->position(control_link, control_point);
	proxy_pose.linear() = robot->rotation(control_link);
	redis_client.setEigen(PROXY_POS_KEY, proxy_pose.translation());
	redis_client.setEigen(PROXY_ORI_KEY, proxy_pose.linear());
	redis_client.setEigen(PROXY_GHOST_POS_KEY, proxy_pose.translation());

	// camera filename convention
	// std::string filename = redis_client.get(FILENAME_KEY);
	// int cnt = 0;

	// // start the simulation thread first
	// thread sim_thread(simulation, robot, sim);

	// // start the logger thread 
	// thread log_thread(logThread, ee_pos_init, ee_ori_init);
	
	// while window is open:
	while (graphics->isWindowOpen()) {
		// proxy_pose.translation() = redis_client.getEigen(PROXY_GHOST_POS_KEY);
		proxy_pose.translation() = redis_client.getEigen(PROXY_POS_KEY);
		// Vector3d diff_vector = robot->position(control_link, Vector3d::Zero()) - robot->position(control_link, control_point);
		// proxy_pose.translation() += diff_vector;
		proxy_pose.linear() = redis_client.getEigen(PROXY_ORI_KEY);
		graphics->updateObjectGraphics(proxy_name, proxy_pose);
		graphics->updateRobotGraphics(robot_name, redis_client.getEigen(JOINT_ANGLES_KEY));
		// graphics->updateDisplayedForceSensor(sim->getAllForceSensorData()[0]);
		// for (const auto sensor_data : sim->getAllForceSensorData()) {
		// 	graphics->updateDisplayedForceSensor(sensor_data);
		// }
		graphics->renderGraphicsWorld();
		
		// // save camera frame and log data if controller cycle 
		// graphics->writeFrameBuffer("camera_recording", filename + "/image" + to_string(cnt));
		// cnt++;
	}

	// stop simulation
	fSimulationRunning = false;
	// sim_thread.join();
	// log_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
void simulation(shared_ptr<Sai2Model::Sai2Model> robot, 
			    shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	// initialize 
	fSimulationRunning = true;

	// containers 
	VectorXd control_torques = VectorXd::Zero(robot->dof());
	VectorXd robot_q = robot->q();
	VectorXd robot_dq = robot->dq();
	VectorXd force_moment = Vector6d::Zero();

	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, control_torques);
	redis_client.setInt(ROBOT_RUNNING_KEY, 1);
	redis_client.setInt(SIM_RESET_KEY, 0);  

	// redis_client.addToReceiveGroup(JOINT_TORQUES_COMMANDED_KEY, control_torques);

	// redis_client.addToSendGroup(JOINT_ANGLES_KEY, robot_q);
	// redis_client.addToSendGroup(JOINT_VELOCITIES_KEY, robot_dq);
	// redis_client.addToSendGroup(FORCE_SENSOR_KEY, force_moment);

	// create a timer
	double sim_freq = 2000;  
	Sai2Common::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);
	sim->enableGravityCompensation(true);
	// sim->disableJointLimits(robot_name);

	int simulation_counter = 0;

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// redis_client.receiveAllFromGroup();
		control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);

		// // check for reset
		// if (redis_client.getInt(MOUSE_B0)) {
		// 	// sim->resetWorld(world_file);
		// 	redis_client.setInt(MOUSE_B0, 0);
		// 	redis_client.setInt(SIM_RESET_KEY, 1);
		// 	// continue;
		// }

		sim->setJointTorques(robot_name, control_torques);
		sim->integrate();

		{
			std::lock_guard<mutex> lock(mutex_update);
			robot->setQ(sim->getJointPositions(robot_name));
			robot->setDq(sim->getJointVelocities(robot_name));
			force_moment.head(3) = sim->getAllForceSensorData()[0].force_local_frame;
			force_moment.tail(3) = sim->getAllForceSensorData()[0].moment_local_frame;
		}

		// redis_client.sendAllFromGroup();
		redis_client.setEigen(JOINT_ANGLES_KEY, robot->q());
		redis_client.setEigen(JOINT_VELOCITIES_KEY, robot->dq());
		redis_client.setEigen(FORCE_SENSOR_KEY, force_moment);
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
	redis_client.setInt(ROBOT_RUNNING_KEY, 0);
}

//------------------------------------------------------------------------------
void logThread(const Vector3d& ee_pos_init, const Matrix3d& ee_ori_init) {

	// redis client
	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();
	redis_client.setInt(LOGGER_START_KEY, 0);
	redis_client.setInt(LOGGER_STOP_KEY, 0);

	// proxy pose 
	Vector3d proxy_pos = ee_pos_init;
	Matrix3d proxy_ori = ee_ori_init;

	// logger 
	bool flag_is_running = false;
	double log_freq = 1000;
	int log_cnt = 0;
	std::string folder_path = "recorded_trajectories";
	if (!std::filesystem::exists(folder_path)) {
		std::filesystem::create_directory(folder_path);
	}
	std::string log_fname = "./recorded_trajectories/traj";
	Sai2Common::Logger logger(log_fname, true);
	logger.addToLog(proxy_pos, "proxy_pos");
	logger.addToLog(proxy_ori, "proxy_ori");

	Sai2Common::LoopTimer timer(1000.0);
	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// get proxy information
		proxy_pos = redis_client.getEigen(PROXY_POS_KEY);
		proxy_ori = redis_client.getEigen(PROXY_ORI_KEY);

		// receieve log stop key 
		if (redis_client.getInt(LOGGER_STOP_KEY)) {
			redis_client.setInt(LOGGER_STOP_KEY, 0);
			if (flag_is_running) {
				std::cout << "Stopping log\n";
				logger.stop();
				flag_is_running = false;
			} else {
				std::cout << "Please start the log first before stopping\n";
			}
		}

		if (redis_client.getInt(LOGGER_START_KEY)) {
			redis_client.setInt(LOGGER_START_KEY, 0);
			if (!flag_is_running) {
				std::cout << "Starting new log\n";
				std::string new_fname = log_fname + std::to_string(log_cnt);
				if (log_cnt != 0) {
					logger.newFileStart(new_fname, log_freq);
				} else {
					logger.start(log_freq);
				}
				flag_is_running = true;
				log_cnt++;
			} else {
				std::cout << "Please stop the current log first\n";
			}
		}
	}

	timer.stop();
	cout << "\nLogger loop timer stats:\n";
	timer.printInfoPostRun();
	logger.stop();
}

// //------------------------------------------------------------------------------
// Affine3d getNewObjectPose(const std::string object_name, const Eigen::Affine3d object_pose,
// 						  const Vector3d mouse_pos, const Vector3d mouse_ori) {
// 	// constants 
// 	const double linear_sf = 1;
// 	const double angular_sf = 1;
// 	const double dt = 0.001;

// 	Matrix3d delta_ori_in_user_frame = AngleAxisd(mouse_ori(0) * dt * angular_sf, Vector3d::UnitX()).toRotationMatrix() * \
// 									   AngleAxisd(mouse_ori(1) * dt * angular_sf, Vector3d::UnitY()).toRotationMatrix() * \
// 									   AngleAxisd(mouse_ori(2) * dt * angular_sf, Vector3d::UnitZ()).toRotationMatrix();
// 	Matrix3d delta_ori_in_base_frame = R_user_to_base * delta_ori_in_user_frame * R_user_to_base.transpose();
// 	Affine3d delta_pose;
// 	delta_pose.linear() = delta_ori_in_base_frame;
// 	delta_pose.translation() += R_user_to_base * mouse_pos * dt * linear_sf;

// 	return delta_pose * object_pose;
// }

// //------------------------------------------------------------------------------
// Matrix3d getRotationMatrix(const Vector3d& ori) {
// 	return AngleAxisd(ori(0), Vector3d::UnitX()).toRotationMatrix() * \
// 			AngleAxisd(ori(1), Vector3d::UnitY()).toRotationMatrix() * \
// 			AngleAxisd(ori(2), Vector3d::UnitZ()).toRotationMatrix();
// }

// //------------------------------------------------------------------------------
// Eigen::Matrix3d rotationMatrixToVector(const Eigen::Vector3d& v, const Eigen::Vector3d& v_prime) {
//     // Normalize vectors
//     Eigen::Vector3d v_normalized = v.normalized();
//     Eigen::Vector3d v_prime_normalized = v_prime.normalized();

//     // Compute rotation axis and angle
//     Eigen::Vector3d k = v_normalized.cross(v_prime_normalized);
//     double theta = std::acos(v_normalized.dot(v_prime_normalized));

//     // Compute rotation matrix using the Eigen library
//     Eigen::Matrix3d K;
//     K << 0, -k.z(), k.y(),
//          k.z(), 0, -k.x(),
//          -k.y(), k.x(), 0;

//     Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity() +
//                                        std::sin(theta) * K +
//                                        (1 - std::cos(theta)) * K * K;

//     return rotation_matrix;
// }