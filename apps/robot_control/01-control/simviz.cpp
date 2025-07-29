/**
 * @file simviz.cpp
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
	During training, provides the simulation environment.
	- Log for training (100 Hz recording):
		- State: Absolute tool pose
		- Action: Absolute proxy pose, gripper width, moment control mode (?)
		- Image: 96 x 96 x 4 (RGBA)
	- To collect multiple trajectories:
		- Simulation reset, stop controller for the user to manually start the controller again 
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
#include "graphics_extension/MultiWorldView.h"
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
const string world_file = "./resources/world.urdf";
const string robot_file = "./resources/panda_arm_hand.urdf";
const string robot_name = "panda";
const string proxy_robot_name = "panda_proxy";
const string camera_name = "camera_fixed";
const string recording_camera_name = "camera_recording";

// specify control parameters for visualization 
const string control_link = "link7";
const Vector3d control_point = Vector3d(0, 0, 0.107 + 0.017 + 0.028 + 0.035 + 0.097);
const Vector3d sensor_point = Vector3d(0, 0, 0.045);
const string proxy_name = "proxy_end_effector";
Affine3d T_sensor = Affine3d::Identity();

// specify dynamic objects 
const double BOX_WIDTH = 0.0635;  // 2.5 in
const std::vector<std::string> dynamic_object_names {"box1", "box2", "box3"};
const std::vector<Vector3d> initial_box_positions {Vector3d(0.25, 0.25, BOX_WIDTH/2 + 0.5), 
												   Vector3d(0.5, 0.5, BOX_WIDTH/2 + 0.5),
												   Vector3d(0.25, -0.25, BOX_WIDTH/2 + 0.5)};

// simulation thread
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, 
				std::shared_ptr<Sai2Simulation::Sai2Simulation> sim,
				std::shared_ptr<Sai2Graphics::MultiWorldView> graphics);
void logThread();
Vector6d transformForce(const Vector3d& force_sensor_location,
						const Vector6d& right_gripper_force_moment,
						const Vector3d& right_gripper_force_location,
					    const Vector6d& left_gripper_force_moment,
						const Vector3d& left_gripper_force_location);
// global
mutex mutex_update;
const int width = 96;
const int height = 96;
VectorXd q_init(9);

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

	// overwrite redis keys with simulation keys 
	JOINT_ANGLES_KEY = "sai2::simulation::sensors::q";
	JOINT_VELOCITIES_KEY = "sai2::simulation::sensors::dq";
	JOINT_TORQUES_COMMANDED_KEY = "sai2::simulation::actuators::fgc";
	FORCE_SENSOR_KEY = "sai2::simulation::sensors::force_moment";
	ROBOT_RUNNING_KEY = "sai2::simulation::running";

	// load multi-world graphics scene 
	std::vector<std::string> world_file_vec = {"resources/proxy_world.urdf", "resources/world.urdf"};
	std::vector<std::string> camera_name_vec = {"camera_fixed", "camera_on_arm"};
	auto graphics_multiworld = std::make_shared<Sai2Graphics::MultiWorldView>(world_file_vec, camera_name_vec, false);

	// graphics modifications for the first world (proxy world)
	graphics_multiworld->getGraphics(0)->setBackgroundColor(66.0/255, 135.0/255, 245.0/255);
	graphics_multiworld->getGraphics(0)->showObjectLinkFrame(true, proxy_name, 0.15);
	graphics_multiworld->getGraphics(0)->showTransparency(true, proxy_robot_name, 0.2);

	// graphics modifications for the second world (actual world)
	graphics_multiworld->getGraphics(1)->setBackgroundColor(66.0/255, 135.0/255, 245.0/255);
	graphics_multiworld->getGraphics(1)->addFrameBuffer("camera_recording", width, height);
	graphics_multiworld->getGraphics(1)->addFrameBuffer("camera_on_arm", width, height);

	// set camera on arm fov
	Affine3d camera_on_robot_pose = Affine3d::Identity();
    camera_on_robot_pose.translation() = Vector3d(0.066, 0, 0.107 + 0.030);
    camera_on_robot_pose.linear() = AngleAxisd(3 * M_PI / 2, Vector3d::UnitZ()).toRotationMatrix();
	graphics_multiworld->getGraphics(1)->setCameraFov("camera_on_arm", 65 * M_PI / 180);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file, false);
	int dof = robot->dof();
	// q_init = robot->q();
	q_init << 0, -25, 0, -135, 0, 105, 0, 0.04, -0.04;
	q_init.head(7) *= M_PI / 180;
	robot->setQ(q_init);
	robot->setDq(VectorXd::Zero(dof));
	robot->updateModel();

	// load simulation world
	auto sim = make_shared<Sai2Simulation::Sai2Simulation>(world_file);
	sim->setJointPositions(robot_name, q_init);
	sim->setJointVelocities(robot_name, 0 * q_init);

    // set contact parameters
    sim->setCollisionRestitution(0.0);
    sim->setCoeffFrictionStatic(5.0);
    sim->setCoeffFrictionDynamic(0.1);	

	// // set object poses
	// Affine3d object_pose = Affine3d::Identity();
	// for (int i = 0; i < dynamic_object_names.size(); ++i) { 
	// 	object_pose.translation() = initial_box_positions[i];
	// 	sim->setObjectPose(dynamic_object_names[i], object_pose);
	// }

	// create simulated force sensor for the right and left grippers
	T_sensor.translation() = sensor_point;
	sim->addSimulatedForceSensor(robot_name, "rightfinger", T_sensor, 10.0);  // 15 hz cut-off 
	sim->addSimulatedForceSensor(robot_name, "leftfinger", T_sensor, 10.0);
	sim->addSimulatedForceSensor(robot_name, "link7", T_sensor, 10.0);
	graphics_multiworld->getGraphics(0)->addForceSensorDisplay(sim->getAllForceSensorData()[0]);
	graphics_multiworld->getGraphics(0)->addForceSensorDisplay(sim->getAllForceSensorData()[1]);
	graphics_multiworld->getGraphics(0)->addForceSensorDisplay(sim->getAllForceSensorData()[2]);

	// create a manual force sensor for the wrist point sensor to be updated by the left and right gripper data
	Sai2Model::ForceSensorData force_sensor_data_wrist_point;
	force_sensor_data_wrist_point.robot_name = robot_name;
	force_sensor_data_wrist_point.link_name = "link7";
	force_sensor_data_wrist_point.transform_in_link = T_sensor;
	force_sensor_data_wrist_point.force_local_frame = Vector3d::Zero();
	force_sensor_data_wrist_point.moment_local_frame = Vector3d::Zero();
	force_sensor_data_wrist_point.force_world_frame = Vector3d::Zero();
	force_sensor_data_wrist_point.moment_world_frame = Vector3d::Zero();

	// set proxy information 
	Affine3d proxy_pose = Affine3d::Identity();
	proxy_pose.translation() = robot->position(control_link, control_point);
	proxy_pose.linear() = robot->rotation(control_link);
	redis_client.setEigen(PROXY_POS_KEY, proxy_pose.translation());
	redis_client.setEigen(PROXY_ORI_KEY, proxy_pose.linear());
	// redis_client.setEigen(PROXY_GHOST_POS_KEY, proxy_pose.translation());
	redis_client.setDouble(GRIPPER_CURRENT_WIDTH_KEY, std::abs(robot->q()(8) - robot->q()(7)));

	// start the simulation thread 
	thread sim_thread(simulation, robot, sim, graphics_multiworld);

	// start the logger thread 
	thread log_thread(logThread);
	
	// while window is open:
	while (graphics_multiworld->isWindowOpen()) {

		// update proxy pose 
		proxy_pose.translation() = redis_client.getEigen(PROXY_POS_KEY);
		proxy_pose.linear() = redis_client.getEigen(PROXY_ORI_KEY);
		graphics_multiworld->getGraphics(0)->updateObjectGraphics(proxy_name, proxy_pose);

		// update wrist force sensor (only displays world frame force data)
		Vector6d wrist_force_moment = redis_client.getEigen(FORCE_SENSOR_KEY);
		force_sensor_data_wrist_point.force_world_frame = wrist_force_moment.head(3);
		force_sensor_data_wrist_point.moment_world_frame = wrist_force_moment.tail(3);

		{
			std::lock_guard<mutex> lock(mutex_update);
			// graphics_multiworld->getGraphics(0)->updateDisplayedForceSensor(sim->getAllForceSensorData()[0]);
			// graphics_multiworld->getGraphics(0)->updateDisplayedForceSensor(sim->getAllForceSensorData()[1]);
			// graphics_multiworld->getGraphics(0)->updateDisplayedForceSensor(force_sensor_data_wrist_point);
			graphics_multiworld->getGraphics(0)->updateRobotGraphics(robot_name, robot->q());
			graphics_multiworld->getGraphics(0)->updateRobotGraphics(proxy_robot_name, robot->q());
			graphics_multiworld->getGraphics(1)->updateRobotGraphics(robot_name, robot->q());
			for (int i = 0; i < dynamic_object_names.size(); ++i) {
				graphics_multiworld->getGraphics(0)->updateObjectGraphics(dynamic_object_names[i], sim->getObjectPose(dynamic_object_names[i]));
				graphics_multiworld->getGraphics(1)->updateObjectGraphics(dynamic_object_names[i], sim->getObjectPose(dynamic_object_names[i]));
			}
			// move arm camera
			graphics_multiworld->getGraphics(1)->setCameraOnRobot("camera_on_arm", robot_name, "link7", camera_on_robot_pose);
		}
		graphics_multiworld->renderGraphicsWorld();

		// send camera frame to redis data (96 x 96 x 4)
		// will be replaced by a redis get method from the camera itself 
		// graphics_recorded->updateRobotGraphics(robot_name, robot->q());
		// save camera frame and log data if controller cycle 
		// graphics->writeFrameBuffer("camera_recording", filename + "/image" + to_string(cnt));
		// cnt++;	
		auto static_binary_frame_data = graphics_multiworld->getGraphics(1)->getFrameBuffer("camera_recording");
		auto moving_binary_frame_data = graphics_multiworld->getGraphics(1)->getFrameBuffer("camera_on_arm");
		redis_client.set(STATIC_CAMERA_FRAME_KEY, static_binary_frame_data);
		redis_client.set(MOVING_CAMERA_FRAME_KEY, moving_binary_frame_data);
	}

	// stop simulation
	fSimulationRunning = false;
	sim_thread.join();
	log_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
// Will be changed to only visualization during the experiment 
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, 
			    std::shared_ptr<Sai2Simulation::Sai2Simulation> sim,
				std::shared_ptr<Sai2Graphics::MultiWorldView> graphics_multiworld) {
	// initialize 
	fSimulationRunning = true;

	// containers 
	VectorXd control_torques = VectorXd::Zero(robot->dof());
	VectorXd robot_q = robot->q();
	VectorXd robot_dq = robot->dq();
	Vector6d right_gripper_force_moment = Vector6d::Zero();
	Vector6d left_gripper_force_moment = Vector6d::Zero();
	Sai2Model::ForceSensorData right_gripper_force_data, left_gripper_force_data;
	Vector6d force_moment = Vector6d::Zero();
	Vector3d force_sensor_point, right_gripper_sensor_point, left_gripper_sensor_point;
	Vector3d control_position = robot->position(control_link, control_point);
	Matrix3d control_orientation = robot->rotation(control_link);
	double gripper_width = 0.08;

	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, control_torques);
	redis_client.setInt(ROBOT_RUNNING_KEY, 1);
	redis_client.setInt(SIM_RESET_KEY, 0); 
	redis_client.setEigen(CONTROL_POS_KEY, control_position);
	redis_client.setEigen(CONTROL_ORI_KEY, control_orientation);
	redis_client.setInt(CONTROLLER_RESET_KEY, 0);

	// redis_client.addToReceiveGroup(JOINT_TORQUES_COMMANDED_KEY, control_torques);

	// redis_client.addToSendGroup(JOINT_ANGLES_KEY, robot_q);
	// redis_client.addToSendGroup(JOINT_VELOCITIES_KEY, robot_dq);
	// redis_client.addToSendGroup(FORCE_SENSOR_KEY, force_moment);

	// create a timer
	double sim_freq = 2000;  
	Sai2Common::LoopTimer timer(sim_freq, 1e6);
	sim->setTimestep(1.0 / sim_freq);
	sim->enableGravityCompensation(true);
	// sim->disableJointLimits(robot_name);

	int simulation_counter = 0;

	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// check for world reset key 
		if (redis_client.getInt(SIM_RESET_KEY)) {
			std::cout << "Reset world\n";
			sim->resetWorld(world_file);
			sim->addSimulatedForceSensor(robot_name, "rightfinger", T_sensor, 10.0);
			sim->addSimulatedForceSensor(robot_name, "leftfinger", T_sensor, 10.0);
			sim->addSimulatedForceSensor(robot_name, "link7", T_sensor, 10.0);

			sim->setJointPositions(robot_name, q_init);
			sim->setJointVelocities(robot_name, 0 * q_init);

			// set contact parameters
			sim->setCollisionRestitution(0.0);
			sim->setCoeffFrictionStatic(2.0);
			sim->setCoeffFrictionDynamic(0.1);	

			sim->enableGravityCompensation(true);
			sim->enableJointLimits(robot_name);

			// {
			// 	// reset proxy pose
			// 	std::lock_guard<mutex> lock(mutex_update);
			// 	redis_client.setEigen(PROXY_POS_KEY, robot->position(control_link, control_point));
			// 	redis_client.setEigen(PROXY_ORI_KEY, robot->rotation(control_link));	
			// }

			redis_client.setInt(SIM_RESET_KEY, 0);
			redis_client.setInt(CONTROLLER_RESET_KEY, 1);
			control_torques.setZero();
			redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, control_torques);
			continue;
		}

		// check if controller reset key is active
		if (redis_client.getInt(CONTROLLER_RESET_KEY) == 1) {
			// continue;
		}

		// redis_client.receiveAllFromGroup();
		control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);

		sim->setJointTorques(robot_name, control_torques);
		sim->integrate();

		{
			std::lock_guard<mutex> lock(mutex_update);
			robot->setQ(sim->getJointPositions(robot_name));
			robot->setDq(sim->getJointVelocities(robot_name));
			robot->updateModel();
			// force sensor data
			auto force_data = sim->getAllForceSensorData();
			for (auto force : force_data) {
				if (force.link_name == "rightfinger") {
					right_gripper_force_data = force;
				} else if (force.link_name == "leftfinger") {
					left_gripper_force_data = force;
				}
			}
			right_gripper_sensor_point = robot->position("rightfinger", Vector3d(0, 0, 0));
			left_gripper_sensor_point = robot->position("leftfinger", Vector3d(0, 0, 0));
			force_sensor_point = robot->position(control_link, sensor_point);
			control_position = robot->position(control_link, control_point);
			control_orientation = robot->rotation(control_link);
			gripper_width = std::abs(robot->q()(7) - robot->q()(8));
		}

		// perform compensation if object is gripped
		// zero out force in gripper axis direction
		// zero out moment in the rotational direction 
		// if (redis_client.getDouble(GRIPPER_CURRENT_WIDTH_KEY) < 0.075) {
		// 	right_gripper_force_data.force_local_frame(2) = 0;
		// 	left_gripper_force_data.force_local_frame(2) = 0;
		// 	right_gripper_force_data.moment_local_frame.setZero();
		// 	left_gripper_force_data.moment_local_frame.setZero();

		// 	// update world frame values 
		// 	Eigen::Matrix3d R_base_sensor = robot->rotationInWorld(control_link);

		// 	right_gripper_force_data.force_world_frame = R_base_sensor * right_gripper_force_data.force_local_frame;
		// 	right_gripper_force_data.moment_world_frame = R_base_sensor * right_gripper_force_data.moment_local_frame;
		// 	left_gripper_force_data.force_world_frame = R_base_sensor * left_gripper_force_data.force_local_frame;
		// 	left_gripper_force_data.moment_world_frame = R_base_sensor * left_gripper_force_data.moment_local_frame;
		// }

		right_gripper_force_moment.head(3) = right_gripper_force_data.force_world_frame;
		right_gripper_force_moment.tail(3) = 0 * right_gripper_force_data.moment_world_frame;
		left_gripper_force_moment.head(3) = left_gripper_force_data.force_world_frame;
		left_gripper_force_moment.tail(3) = 0 * left_gripper_force_data.moment_world_frame;

		force_moment = transformForce(force_sensor_point,
									  right_gripper_force_moment, 
									  right_gripper_sensor_point,
									  left_gripper_force_moment,
									  left_gripper_sensor_point);

		// just doing force control, so just send forces (moments are disregarded)
		// force_moment.head(3) = right_gripper_force_moment.head(3) + left_gripper_force_moment.head(3);

		// redis_client.sendAllFromGroup();
		redis_client.setEigen(JOINT_ANGLES_KEY, robot->q());
		redis_client.setEigen(JOINT_VELOCITIES_KEY, robot->dq());
		redis_client.setDouble(GRIPPER_CURRENT_WIDTH_KEY, gripper_width);
		redis_client.setEigen(FORCE_SENSOR_KEY, - force_moment);  
		redis_client.setEigen(CONTROL_POS_KEY, control_position);
		redis_client.setEigen(CONTROL_ORI_KEY, control_orientation);
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
	redis_client.setInt(ROBOT_RUNNING_KEY, 0);
}

//------------------------------------------------------------------------------
void logThread() {

	// redis client
	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();
	redis_client.setInt(LOGGER_START_KEY, 0);
	redis_client.setInt(LOGGER_STOP_KEY, 0);

	// setup logger variables
	Vector3d proxy_pos = Vector3d::Zero();
	Matrix3d proxy_ori = Matrix3d::Identity();
	Vector3d control_pos = Vector3d::Zero();
	Matrix3d control_ori = Matrix3d::Identity();
	double gripper_width = 0.08;
	std::vector<unsigned char> camera_binary_data = {};	

	// logger 
	bool flag_is_running = false;
	double log_freq = 1000;
	int log_cnt = 0;
	std::string folder_path = "recorded_trajectories";
	if (!std::filesystem::exists(folder_path)) {
		std::filesystem::create_directory(folder_path);
	}
	std::string log_fname = folder_path + "/traj" + std::to_string(log_cnt);
	Sai2Common::Logger logger(log_fname, false);
	logger.addToLog(proxy_pos, "proxy_pos");
	logger.addToLog(proxy_ori, "proxy_ori");
	logger.addToLog(control_pos, "control_pos");
	logger.addToLog(control_ori, "control_ori");
	logger.addToLog(gripper_width, "gripper_width");

	// binary camera frame data setup
	std::string binary_save_folder = folder_path + "/camera/";
	std::string binary_save_path = binary_save_folder + "traj0/";
	if (!std::filesystem::exists(binary_save_folder)) {
		std::filesystem::create_directory(binary_save_folder);
	}
	if (!std::filesystem::exists(binary_save_path)) {
		std::filesystem::create_directory(binary_save_path);
	}
	int binary_fname_counter = 0;
	int loop_counter = 0;	

	Sai2Common::LoopTimer timer(1000.0, 1e6);
	while (fSimulationRunning) {
		timer.waitForNextLoop();

		// get logger information
		proxy_pos = redis_client.getEigen(PROXY_POS_KEY);
		proxy_ori = redis_client.getEigen(PROXY_ORI_KEY);
		control_pos = redis_client.getEigen(CONTROL_POS_KEY);
		control_ori = redis_client.getEigen(CONTROL_ORI_KEY);
		gripper_width = redis_client.getDouble(GRIPPER_CURRENT_WIDTH_KEY);

		// save binary image data every 30 fps 
		if (flag_is_running && (loop_counter % 30 == 0)) {
			for (int i = 0; i < 2; ++i) {
				std::string camera_frame_key_string = STATIC_CAMERA_FRAME_KEY;
				std::string binary_save_path = binary_save_path + "static_camera_";
				if (i == 1) {
					camera_frame_key_string = MOVING_CAMERA_FRAME_KEY;
					binary_save_path = binary_save_path + "moving_camera_";
				}
				std::vector<unsigned char> binary_camera_data = redis_client.getBinary(camera_frame_key_string);
				unsigned char* data = new unsigned char[binary_camera_data.size()];
				std::copy(binary_camera_data.begin(), binary_camera_data.end(), data);
				std::FILE* fp = std::fopen((binary_save_path + std::to_string(loop_counter) + ".bin").c_str(), "wb");
				std::fwrite(data, 4, width * height, fp);
				std::fclose(fp);
			}
		}

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

		// receive log start key                                   
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
				binary_save_path = binary_save_folder + "/traj" + std::to_string(binary_fname_counter) + '/';
				if (!std::filesystem::exists(binary_save_path)) {
					std::filesystem::create_directory(binary_save_path);
				}
				flag_is_running = true;
				binary_fname_counter++;
				log_cnt++;
				loop_counter = -1;
			} else {
				std::cout << "Please stop the current log first\n";
			}
		}

		loop_counter++;
	}

	timer.stop();
	cout << "\nLogger loop timer stats:\n";
	timer.printInfoPostRun();
	logger.stop();
}

//------------------------------------------------------------------------------

/**
 * @brief Transform right and left gripper forces to the perceived force sensor force at the wrist point.
 * 			F_{wrist} = F_{left} + F_{right} 
 * 			M_{wrist} = r_{left} x F_{left} + r_{right} x F_{right}
 * 
 * @param force_sensor_location 
 * @param right_gripper_force 
 * @param right_gripper_force_location 
 * @param left_gripper_force 
 * @param left_gripper_force_location 
 * @return Vector6d 
 */
Vector6d transformForce(const Vector3d& force_sensor_location,
						const Vector6d& right_gripper_force_moment,
						const Vector3d& right_gripper_force_location,
					    const Vector6d& left_gripper_force_moment,
						const Vector3d& left_gripper_force_location) {
	Vector3d net_force = right_gripper_force_moment.head(3) + left_gripper_force_moment.head(3);
	Vector3d net_moment = (right_gripper_force_location - force_sensor_location).cross(Vector3d(right_gripper_force_moment.head(3))) + \
							(left_gripper_force_location - force_sensor_location).cross(Vector3d(left_gripper_force_moment.head(3))) + \
							right_gripper_force_moment.tail(3) + left_gripper_force_moment.tail(3);
	Vector6d force_sensor_reading;
	force_sensor_reading.head(3) = net_force;
	force_sensor_reading.tail(3) = net_moment;
	return force_sensor_reading;
}

