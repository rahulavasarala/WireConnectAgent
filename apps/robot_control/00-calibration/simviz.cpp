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
const string world_file = "./resources/world.urdf";
const string robot_file = "./resources/panda_arm.urdf";
const string robot_name = "panda";
const string camera_name = "camera_fixed";

// specify control parameters for visualization 
const string control_link = "link7";

// simulation thread
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, 
				std::shared_ptr<Sai2Simulation::Sai2Simulation> sim);

// global
mutex mutex_update;
VectorXd q_init(7);

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

	// load graphics
	auto graphics = std::make_shared<Sai2Graphics::Sai2Graphics>(world_file);

	// load robots
	auto robot = make_shared<Sai2Model::Sai2Model>(robot_file, false);
	int dof = robot->dof();
	q_init << 0, -25, 0, -135, 0, 105, 0;
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
    sim->setCoeffFrictionStatic(0.0);
    sim->setCoeffFrictionDynamic(0.0);	

	// start the simulation thread 
	thread sim_thread(simulation, robot, sim);

	// while window is open:
	while (graphics->isWindowOpen()) {
		{
			std::lock_guard<mutex> lock(mutex_update);
			graphics->updateRobotGraphics(robot_name, robot->q());
		}
		graphics->renderGraphicsWorld();

	}

	// stop simulation
	fSimulationRunning = false;
	sim_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
// Will be changed to only visualization during the experiment 
void simulation(std::shared_ptr<Sai2Model::Sai2Model> robot, 
			    std::shared_ptr<Sai2Simulation::Sai2Simulation> sim) {
	// initialize 
	fSimulationRunning = true;
	VectorXd control_torques = VectorXd::Zero(robot->dof());

	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, control_torques);

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

		// redis_client.receiveAllFromGroup();
		control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);

		sim->setJointTorques(robot_name, control_torques);
		sim->integrate();

		{
			std::lock_guard<mutex> lock(mutex_update);
			robot->setQ(sim->getJointPositions(robot_name));
			robot->setDq(sim->getJointVelocities(robot_name));
			robot->updateModel();
		}

		// redis_client.sendAllFromGroup();
		redis_client.setEigen(JOINT_ANGLES_KEY, robot->q());
		redis_client.setEigen(JOINT_VELOCITIES_KEY, robot->dq());
	}
	timer.stop();
	cout << "\nSimulation loop timer stats:\n";
	timer.printInfoPostRun();
	redis_client.setInt(ROBOT_RUNNING_KEY, 0);
}


