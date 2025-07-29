/**
 * @file controller.cpp
 * @author William Chong (wmchong@stanford.edu)
 * @brief 
 * @version 0.1
 * @date 2024-07-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// external includes 
#include "Sai2Graphics.h"
#include "Sai2Model.h"
#include "Sai2Primitives.h"
#include "filters/ButterworthFilter.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"
#include "redis/keys/chai_haptic_devices_driver.h"
#include "../../include/algorithms/ForceSpaceParticleFilter.h"
#include "../../include/algorithms/FirstOrderLowPassFilter.h"
#include "../../include/algorithms/RotationLowPassFilter.h"
#include <yaml-cpp/yaml.h>
#include "redis_keys.h"

// c++ includes
#include <iostream>
#include <string>
#include <mutex>
#include <atomic>
#include <tinyxml2.h>
#include <map>
#include <type_traits>

#include <signal.h>
bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;
using namespace Sai2Common::ChaiHapticDriverKeys;	

// constants 

namespace {
	// initialization
	VectorXd Q_INIT(7);
	VectorXd Q_MID(7);
	bool FLAG_SIMULATION = false;
	Matrix3d R_USER_TO_BASE = Matrix3d::Identity();
	double DEVICE_ROTATION = 0;
	double QTOL = 0.2;
	double QTOL_INT = 0.8;
	int PRIMITIVE_FLAG = 0;

	// proxy
	Vector3d CONTROL_POINT = Vector3d(0, 0, 0.107 + 0.014 + 0.017 + 0.028 + 0.035 + 0.097);
	Vector3d ORIGINAL_CONTROL_POINT = CONTROL_POINT;
	Vector3d LOAD_CONTROL_POINT = Vector3d(0, 0, 0);
	Vector3d SENSOR_POINT = Vector3d(0, 0, 0.289);
	double KP_FORCE_PROXY = 250;
	double KP_MOMENT_PROXY = 5;
	double MAX_FORCE_NORM = 20;
	double MIN_FORCE_NORM = 1;
	double MAX_PROXY_RADIUS = 0.05;
	double MAX_MOMENT_NORM = 2;
	double MAX_PROXY_ORI_ANGLE = 15;
	double FRAME_RESET_TIME = 0.25;
	double OUTER_RADIUS = 0.05;
	double KV_MOMENT_TO_VELOCITY = 1.0;
	double KI_MOMENT_TO_VELOCITY = 1.0;
	double MAX_VELOCITY_NORM = 0.1;
	int MOMENT_CONTROL_MODE = 0;
	int ZERO_CROSSING_MODE = 0;
	double FORCE_BLENDING_TIME = 0.1;  // amount of time to transition from current desired force to proxy force
	double KV_ORI_PROXY_TO_ANGULAR_VELOCITY = 10;
	double MAX_ANGULAR_VELOCITY = 0.3;  // rad/s
	double MOMENT_GAIN_FOR_LOAD = 5;  // gain for moment gain 

	// filter parameters 
	int N_PARTICLES = 1000;
	double FILTER_FREQ = 50;
	int QUEUE_SIZE = 20;
	double F_LOW = 0;
	double F_HIGH = 5;
	double V_LOW = 1e-2;
	double V_HIGH = 7e-2;
	double F_LOW_ADD = 5;
	double F_HIGH_ADD = 10;
	double V_LOW_ADD = 2e-2;
	double V_HIGH_ADD = 1e-1;

	// 1, 4, 0.01, 0.07, 5, 10, 0.02, 0.1

	// op space
	double KP_POS = 100;
	double KV_POS = 20;
	double KI_POS = 0;
	double KP_ORI = 100;
	double KP_ORI_LOW = 20;
	double KV_ORI = 20;
	double KI_ORI = 0;
	double KP_FORCE = 0.05;
	double KV_FORCE = 10;
	double KI_FORCE = 1.0;
	double KP_MOMENT = 0.05;
	double KV_MOMENT = 10;
	double KI_MOMENT = 1.0;
	double BIE_ADDITION = 0.25;

	// joint space
	double KP_JOINT = 400;
	double KV_JOINT = 30;
	double KI_JOINT = 0;

	// haptic
	double POS_SCALE_FACTOR = 1.5;
	double ORI_SCALE_FACTOR = 1;
	double FORCE_REDUCTION = 0.3;

	// force
	double FORCE_SLOPE = 1;
	double POS_TOL = 1e-2;
	double VEL_TOL = 1e-3;
	double WAIT_TIME = 0.2;

	// object
	double OBJECT_MASS = 0.5;
	Vector3d OBJECT_COM = Vector3d(0, 0, 0);
	Vector3d OBJECT_INERTIA = Vector3d(0, 0, 0);
	bool ACTIVE_GRIPPER_CONTROL = true;
}

// map of flags for key presses
namespace {
	map<int, bool> key_pressed = {
		{GLFW_KEY_P, false},
		{GLFW_KEY_L, false},
		{GLFW_KEY_W, false},
		{GLFW_KEY_O, false},
		{GLFW_KEY_Q, false},
		{GLFW_KEY_W, false},
		{GLFW_KEY_E, false},
	};
	map<int, bool> key_was_pressed = key_pressed;
}

// control specifications 
const string calibration_fname = "../../calibration/bias_measurement.xml";
std::string robot_file = "./resources/panda_arm_hand.urdf";
const std::string world_file = "./resources/proxy_world.urdf";
const std::string control_link = "link7";

// state machine
enum State {
	POSTURE = 0, 
	MOTION,
	FRAME_RESET,
	RESET,
	PICKUP_POSTURE,
	LOADING_POSTURE,
	PAYLOAD_DETECTION
};

// function declarations
void getKeyPressAndGraphics();
void particle_filter();
void payloadDetection();
Matrix3d getRotationMatrix(const Vector3d& ori);
Eigen::Matrix3d rotationMatrixToVector(const Eigen::Vector3d& v, const Eigen::Vector3d& v_prime);
Vector3d linearInterp(const Vector3d& start, 
					  const Vector3d& end, 
					  const double curr_time,
					  const double start_time,
					  const double end_time);
Matrix3d slerp(const Matrix3d& start,
			   const Matrix3d& end, 
			   const double curr_time,
			   const double start_time,
			   const double end_time);
template<typename T>
void clearQueue(std::queue<T>& q);
template<typename T>
bool allElementsSame(const std::queue<T>& q);
template<typename T>
void printQueue(const std::queue<T>& q);
Eigen::Matrix3d reOrthogonalizeRotationMatrix(const Eigen::Matrix3d& mat);

Vector3d getNonlinearForce(const Vector3d& penetration, const double& f_max, const double& k_vir) {
	double beta = k_vir / f_max;
	return f_max * (1 - exp(- beta * penetration.norm())) * penetration / penetration.norm();
}

std::tuple<double, Vector3d, Eigen::Matrix<double, 6, 1>> readBiasXML(const string& path_to_bias_file) {
	// initialize variables 
	VectorXd sensor_bias = VectorXd::Zero(6);
	Vector3d com = Vector3d::Zero();
	double mass;

	tinyxml2::XMLDocument doc;
	doc.LoadFile(path_to_bias_file.c_str());
	if (!doc.Error()) {
		cout << "Loading bias file file ["+path_to_bias_file+"]." << endl;
		try {
			// load mass 
			std::stringstream mass_string(doc.FirstChildElement("mass")->
				Attribute("value"));
			mass_string >> mass;
			std::cout << "Load mass: " << mass << "\n";

			// load com
			std::stringstream com_string(doc.FirstChildElement("com")->
				Attribute("value"));
			com_string >> com(0);
			com_string >> com(1);
			com_string >> com(2);
			std::cout << "Load com: " << com << "\n";

			// force bias			
			std::stringstream bias(doc.FirstChildElement("force_bias")->
				Attribute("value"));
			bias >> sensor_bias(0);
			bias >> sensor_bias(1);
			bias >> sensor_bias(2);
			bias >> sensor_bias(3);
			bias >> sensor_bias(4);
			bias >> sensor_bias(5);
			std::stringstream ss; ss << sensor_bias.transpose();
			cout << "Sensor bias : "+ss.str() << endl;
		}
		catch( const std::exception& e ) { // reference to the base of a polymorphic object
			std::cout << e.what(); // information from length_error printed
			cout << "WARNING : Failed to parse bias file." << endl;
		}
	} else {
		cout << "WARNING : Could not load bias file ["+path_to_bias_file+"]" << endl;
		doc.PrintError();
	}
	return std::make_tuple(mass, com, sensor_bias);
}

Eigen::Matrix3d computeRotationMatrix(double angle, const Eigen::Vector3d& axis);
Eigen::Matrix3d minimumRotationMatrix(const Eigen::Vector3d& u, const Eigen::Vector3d& v);
double angleBetweenVectors(const Eigen::Vector3d& u, const Eigen::Vector3d& v);
Eigen::Vector3d axisOfRotation(const Eigen::Vector3d& u, const Eigen::Vector3d& v);
Eigen::Vector3d projectOntoPlane(const Eigen::Vector3d& v, const Eigen::Vector3d& n);

Eigen::Matrix3d computeConstrainedOrientation(
    const Eigen::Matrix3d& R0,       // Initial orientation
    const Eigen::Matrix3d& Rd,       // Desired final orientation
    const Eigen::Vector3d& v) {      // Constraint axis (must be non-zero)
    
    // Ensure v is normalized
    Eigen::Vector3d v_normalized = v.normalized();

    // Compute relative rotation
    Eigen::Matrix3d R_rel = R0.transpose() * Rd;

    // Extract the rotation axis (using the anti-symmetric part)
    Eigen::Vector3d axis_rel(
        R_rel(2, 1) - R_rel(1, 2),
        R_rel(0, 2) - R_rel(2, 0),
        R_rel(1, 0) - R_rel(0, 1)
    );
    axis_rel /= 2.0; // Normalize the axis magnitude

    // Compute rotation angle
    double theta_rel = std::acos(std::clamp((R_rel.trace() - 1.0) / 2.0, -1.0, 1.0));

    // Project the rotation axis onto the constrained plane
    Eigen::Vector3d axis_parallel = axis_rel.dot(v_normalized) * v_normalized;
    Eigen::Vector3d axis_perpendicular = axis_rel - axis_parallel;

    // If axis_perpendicular is close to zero, no valid rotation is possible
    if (axis_perpendicular.norm() < 1e-6) {
		std::cout << "axis is aligned; quitting\n";
        return R0; // Return the initial orientation if constrained rotation is not possible
    }

    Eigen::Vector3d axis_new = axis_perpendicular.normalized();

    // Reconstruct the constrained rotation matrix
    Eigen::Matrix3d skew;
    skew << 0, -axis_new.z(), axis_new.y(),
            axis_new.z(), 0, -axis_new.x(),
           -axis_new.y(), axis_new.x(), 0;

    Eigen::Matrix3d R_new = Eigen::Matrix3d::Identity()
                            + std::sin(theta_rel) * skew
                            + (1 - std::cos(theta_rel)) * (skew * skew);

    // Compute the constrained final orientation
    Eigen::Matrix3d Rf = R0 * R_new;

    return Rf;
}

// Function to constrain rotation
Eigen::Matrix3d constrainRotationBetweenOrientations(const Eigen::Matrix3d& R_start, const Eigen::Matrix3d& R_end, const Eigen::Vector3d& axis) {
    // Ensure the axis is a unit vector
    Eigen::Vector3d unit_axis = axis.normalized();

    // Compute the relative rotation matrix
    Eigen::Matrix3d R_rel = R_end * R_start.transpose();

    // Extract axis-angle representation of the relative rotation
    Eigen::AngleAxisd angle_axis_rel(R_rel);
    double theta_rel = angle_axis_rel.angle();
    Eigen::Vector3d axis_rel = angle_axis_rel.axis();

    // Project the relative axis onto the plane orthogonal to the constrained axis
    Eigen::Vector3d constrained_axis = axis_rel - (axis_rel.dot(unit_axis)) * unit_axis;
	// std::cout << "constrained axis: " << constrained_axis.transpose() << "\n";

    if (constrained_axis.norm() < 1e-6) {
        // If the constrained axis is near zero, return the starting orientation
        return R_start;
    }
    constrained_axis.normalize();

	// align constrained axis to input axis
	if (constrained_axis.dot(unit_axis) < 0) {
		constrained_axis *= -1;
	}

    // Construct the constrained relative rotation
    Eigen::AngleAxisd constrained_rotation(theta_rel, constrained_axis);
    Eigen::Matrix3d R_constrained_rel = constrained_rotation.toRotationMatrix();

    // Compute the constrained ending orientation
    Eigen::Matrix3d R_constrained_end = R_constrained_rel * R_start;

    return R_constrained_end;
}

// Particle filter parameters 
mutex mutex_pfilter;
mutex mutex_keypressed;

std::atomic<int> reset_filter_flag(0);  // flag to reset the particle filter
std::atomic<int> pause_filter_flag(0);  // flag to pause the particle filter 

// const int n_particles = 1000;
// MatrixXd particle_positions_to_redis = MatrixXd::Zero(3, n_particles);
int force_space_dimension = 0;
int prev_force_space_dimension = 0;
Matrix3d sigma_force = Matrix3d::Zero();
Matrix3d sigma_motion = Matrix3d::Identity();
Vector3d force_or_motion_axis = Vector3d::Zero();
std::vector<Vector3d> force_axes = {};

Vector3d motion_control_pfilter;
Vector3d force_control_pfilter;
Vector3d measured_velocity_pfilter;
Vector3d measured_force_pfilter;

queue<Vector3d> pfilter_motion_control_buffer;
queue<Vector3d> pfilter_force_control_buffer;
queue<Vector3d> pfilter_sensed_force_buffer;
queue<Vector3d> pfilter_sensed_velocity_buffer;
double freq_ratio_filter_control;
bool flag_filter_force_to_free = false;
std::queue<int> force_dimension_queue;  // all elements in queue must be 0 to switch to free space

// moment-tracking frame items 
Matrix3d moment_home_frame = Matrix3d::Identity();
Matrix3d moment_current_frame = Matrix3d::Identity();

// payload-related items 
std::atomic<bool> switch_orientation_control_flag(false);
bool grasped_object = false;
bool prev_grasped_object = false;
bool flag_payload_estimation_needed = false;
double payload_force_estimation_value = 0;
double payload_start_time = 0;
bool first_loop_payload_detection = true;
mutex mutex_payload;

/*
	Functions 
*/
int sign(const double& x) {
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

AngleAxisd orientationDiffAngleAxis(const Matrix3d& goal_orientation,
									const Matrix3d& current_orientation,
									const double scaling_factor = 1.0) {
	if (scaling_factor < 0 || scaling_factor > 1) {
		throw std::runtime_error(
			"Scaling factor must be between 0 and 1 in "
			"scaledOrientationErrorFromAngleAxis");
	}

	// expressed in base frame common to goal and current orientation
	AngleAxisd current_orientation_from_goal_orientation_aa(
		current_orientation * goal_orientation.transpose());

	return AngleAxisd(
		scaling_factor * current_orientation_from_goal_orientation_aa.angle(),
		current_orientation_from_goal_orientation_aa.axis());
}

int main(int argc, char *argv[]) {

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// process controller yaml file 
	YAML::Node config = YAML::LoadFile("./resources/controller_settings.yaml");

	if (config["setup"]) {
		YAML::Node current_node = config["setup"];
		std::vector<double> q_init_vec = current_node["q_init"].as<std::vector<double>>();
		Q_INIT << q_init_vec[0], q_init_vec[1], q_init_vec[2], q_init_vec[3], q_init_vec[4], \
								q_init_vec[5], q_init_vec[6];
		std::vector<double> q_mid_vec = current_node["q_mid"].as<std::vector<double>>();
		Q_MID << q_mid_vec[0], q_mid_vec[1], q_mid_vec[2], q_mid_vec[3], q_mid_vec[4], \
								q_mid_vec[5], q_mid_vec[6];					
		FLAG_SIMULATION = current_node["simulation"].as<bool>();
		DEVICE_ROTATION = current_node["device_rotation"].as<double>() * M_PI / 180;
		R_USER_TO_BASE = AngleAxisd(DEVICE_ROTATION, \
										Vector3d::UnitZ()).toRotationMatrix();
		QTOL = current_node["q_tol"].as<double>();
		QTOL_INT = current_node["q_tol_int"].as<double>();
		PRIMITIVE_FLAG = current_node["primitive_flag"].as<int>();
	} else {
		Q_INIT << -1.1768,-1.07863,0.770571,-2.43178,-0.983503,3.8256,1.63483;
		Q_MID << 0, 0, 0, -1.52105, 0, 1.9862, 0;
		FLAG_SIMULATION = true;
		DEVICE_ROTATION = 0;
		R_USER_TO_BASE = AngleAxisd(DEVICE_ROTATION, \
										Vector3d::UnitZ()).toRotationMatrix();
		QTOL = 0.2;
	}

	if (config["proxy"]) {
		YAML::Node current_node = config["proxy"];
		if (current_node) {
			std::vector<double> control_vec = current_node["control_point"].as<std::vector<double>>();
			CONTROL_POINT = Vector3d(control_vec[0], control_vec[1], control_vec[2]);
			ORIGINAL_CONTROL_POINT = CONTROL_POINT;

			std::vector<double> load_control_vec = current_node["load_control_point"].as<std::vector<double>>();
			LOAD_CONTROL_POINT = Vector3d(load_control_vec[0], load_control_vec[1], load_control_vec[2]);

			std::vector<double> sensor_vec = current_node["sensor_point"].as<std::vector<double>>();
			SENSOR_POINT = Vector3d(sensor_vec[0], sensor_vec[1], sensor_vec[2]);

			KP_FORCE_PROXY = current_node["kp_force"].as<double>();
			KP_MOMENT_PROXY = current_node["kp_moment"].as<double>();
			MAX_FORCE_NORM = current_node["max_force"].as<double>();
			MIN_FORCE_NORM = current_node["min_force"].as<double>();
			MAX_MOMENT_NORM = current_node["max_moment"].as<double>();
			FRAME_RESET_TIME = current_node["frame_reset_time"].as<double>();
			OUTER_RADIUS = current_node["outer_radius"].as<double>();
			MAX_PROXY_RADIUS = MAX_FORCE_NORM / KP_FORCE_PROXY + OUTER_RADIUS;   
			KV_MOMENT_TO_VELOCITY = current_node["kv_moment_to_velocity"].as<double>();
			KI_MOMENT_TO_VELOCITY = current_node["ki_moment_to_velocity"].as<double>();
			MAX_VELOCITY_NORM = current_node["max_velocity"].as<double>();
			MOMENT_CONTROL_MODE = current_node["moment_control_mode"].as<int>();
			ZERO_CROSSING_MODE = current_node["zero_crossing_mode"].as<int>();
			FORCE_BLENDING_TIME = current_node["force_blending_time"].as<double>();
			KV_ORI_PROXY_TO_ANGULAR_VELOCITY = current_node["kv_orientation_proxy_to_angular_velocity"].as<double>();
			MAX_ANGULAR_VELOCITY = current_node["max_angular_velocity_in_contact"].as<double>();
			MOMENT_GAIN_FOR_LOAD = current_node["moment_gain_for_load"].as<double>();
		}
	}

	if (config["filter"]) {
		YAML::Node current_node = config["filter"];
		N_PARTICLES = current_node["n_particles"].as<double>();
		FILTER_FREQ = current_node["filter_freq"].as<double>();
		QUEUE_SIZE = current_node["queue_size"].as<double>();
		freq_ratio_filter_control = FILTER_FREQ / 1000;
		F_LOW = current_node["f_low"].as<double>();
		F_HIGH = current_node["f_high"].as<double>();
		V_LOW = current_node["v_low"].as<double>();
		V_HIGH = current_node["v_high"].as<double>();
		F_LOW_ADD = current_node["f_low_add"].as<double>();
		F_HIGH_ADD = current_node["f_high_add"].as<double>();
		V_LOW_ADD = current_node["v_low_add"].as<double>();
		V_HIGH_ADD = current_node["v_high_add"].as<double>();
	}

	if (config["op_space"]) {
		YAML::Node current_node = config["op_space"];
		KP_POS = current_node["kp_pos"].as<double>();
		KV_POS = current_node["kv_pos"].as<double>();
		KI_POS = current_node["ki_pos"].as<double>();
		KP_ORI = current_node["kp_ori"].as<double>();
		KP_ORI_LOW = current_node["kp_ori_low"].as<double>();
		KV_ORI = current_node["kv_ori"].as<double>();
		KI_ORI = current_node["ki_ori"].as<double>();
		KP_FORCE = current_node["kp_force"].as<double>();
		KV_FORCE = current_node["kv_force"].as<double>();
		KI_FORCE = current_node["ki_force"].as<double>();
		KP_MOMENT = current_node["kp_moment"].as<double>();
		KV_MOMENT = current_node["kv_moment"].as<double>();
		KI_MOMENT = current_node["ki_moment"].as<double>();
		BIE_ADDITION = current_node["bie_addition"].as<double>();
	}

	if (config["joint_space"]) {
		YAML::Node current_node = config["joint_space"];
		KP_JOINT = current_node["kp_joint"].as<double>();
		KV_JOINT = current_node["kv_joint"].as<double>();
		KI_JOINT = current_node["ki_joint"].as<double>();
	}

	if (config["haptic"]) {
		YAML::Node current_node = config["haptic"];
		POS_SCALE_FACTOR = current_node["pos_scale_factor"].as<double>();
		ORI_SCALE_FACTOR = current_node["ori_scale_factor"].as<double>();
		FORCE_REDUCTION = current_node["force_reduction"].as<double>();
	}

	if (config["payload_estimation"]) {
		YAML::Node current_node = config["payload_estimation"];
		FORCE_SLOPE = current_node["force_slope"].as<double>();
		POS_TOL = current_node["position_tol"].as<double>();
		VEL_TOL = current_node["velocity_tol"].as<double>();
		WAIT_TIME = current_node["wait_time"].as<double>();
	}

	if (config["object"]) {
		YAML::Node current_node = config["object"];
		std::vector<double> com_vec = current_node["item_com"].as<std::vector<double>>();
		OBJECT_COM = Vector3d(com_vec[0], com_vec[1], com_vec[2]);
		OBJECT_MASS = current_node["item_mass"].as<double>();
		std::vector<double> inertia_vec = current_node["item_inertia"].as<std::vector<double>>();
		OBJECT_INERTIA = Vector3d(inertia_vec[0], inertia_vec[1], inertia_vec[2]);
		ACTIVE_GRIPPER_CONTROL = current_node["active_gripper_control"].as<bool>();
	}

	// if not simulation, then robot is 7 dof
	if (!FLAG_SIMULATION) {
		robot_file = "./resources/panda_arm.urdf";
	}

	// initial state 
	int state = POSTURE;
	string controller_status = "1";
	
	// start redis client
	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// redis keys overwrite for simulation 
	if (FLAG_SIMULATION) {
		JOINT_ANGLES_KEY = "sai2::simulation::sensors::q";
		JOINT_VELOCITIES_KEY = "sai2::simulation::sensors::dq";
		JOINT_TORQUES_COMMANDED_KEY = "sai2::simulation::actuators::fgc";
		FORCE_SENSOR_KEY = "sai2::simulation::sensors::force_moment";
		ROBOT_RUNNING_KEY = "sai2::simulation::running";
	}

	/*
		Robot information 
	*/
	auto robot = std::make_shared<Sai2Model::Sai2Model>(robot_file);
	VectorXd robot_q = redis_client.getEigen(JOINT_ANGLES_KEY);
	VectorXd robot_dq = redis_client.getEigen(JOINT_VELOCITIES_KEY);
	robot->setQ(robot_q);
	robot->setDq(robot_dq);
	robot->updateModel();

	int dof = robot->dof();
	VectorXd control_torques = VectorXd::Zero(dof);

	/*
		Control information
	*/
	Affine3d control_transform = Affine3d::Identity();
	control_transform.translation() = CONTROL_POINT;
	auto motion_force_task = std::make_shared<Sai2Primitives::MotionForceTask>(robot, control_link, control_transform);  // controls the robot proxy
	motion_force_task->disableInternalOtg();
	// motion_force_task->disablePassivity();
	motion_force_task->handleAllSingularitiesAsType1(true);
	// motion_force_task->enableVelocitySaturation(0.3, M_PI / 3);

	// velocity-controlled setup through integrated desired velocity to position 
	motion_force_task->setPosControlGains(KP_POS, KV_POS, 0);
	motion_force_task->setOriControlGains(KP_ORI, KV_ORI, 0);
	// motion_force_task->setForceControlGains(0.4, 10.0, 1.0);  // default gains 
	// motion_force_task->setMomentControlGains(0.4, 10.0, 1.0);
	motion_force_task->setForceControlGains(KP_FORCE, KV_FORCE, KI_FORCE);   
	motion_force_task->setMomentControlGains(KP_MOMENT, KV_MOMENT, KI_MOMENT);
	motion_force_task->enableZeroForceCrossing();
	motion_force_task->enableZeroMomentCrossing();
	motion_force_task->enableForceControlTaskDamping();
	motion_force_task->enableMomentControlTaskDamping();
	// motion_force_task->setSingularityHandlingBounds(6e-3, 6e-2);
	// motion_force_task->setForceControlDeadband(0.1);
	// motion_force_task->setMomentControlDeadband(0.1);
	// motion_force_task->enableForceSensorCompensation();
	// motion_force_task->enableForceInertiaShaping();

	// redundancy completion 
	std::cout << "Desired starting joint posture:\n" << Q_INIT.transpose() << "\n";
	auto joint_task = std::make_shared<Sai2Primitives::JointTask>(robot);
	joint_task->setGoalPosition(Q_INIT);
	// joint_task->disableInternalOtg();
	// joint_task->enableVelocitySaturation(M_PI / 4);
	bool integrator_on = false;
	VectorXd previous_joint_error = 0 * joint_task->getGoalPosition();

	/*
		Sensor information
	*/
	Affine3d sensor_transform = Affine3d::Identity();
	sensor_transform.translation() = SENSOR_POINT;
	motion_force_task->setForceSensorFrame(control_link, sensor_transform);

	// force sensor calibration values 
	auto sensor_data = readBiasXML(calibration_fname);
	double load_mass = std::get<0>(sensor_data);
	Vector3d load_com = std::get<1>(sensor_data);
	VectorXd force_bias = std::get<2>(sensor_data);
	Vector3d load_inertia = Vector3d::Zero();

	if (FLAG_SIMULATION) {
		load_mass = 0;
		load_com.setZero();
		force_bias.setZero();
	}

	double original_load_mass = load_mass;
	Vector3d original_load_inertia = Vector3d::Zero();
	Vector3d original_load_com = load_com;
	Vector3d starting_ee_pos;

	VectorXd initial_force_moment_bias = VectorXd::Zero(6);
	Vector6d force_moment = Vector6d::Zero();

	// /*
	// 	Haptic device setup
	// */
	// std::cout << "Haptic Device Setup\n";
	// Sai2Primitives::HapticDeviceController::DeviceLimits device_limits(
	// 	redis_client.getEigen(createRedisKey(MAX_STIFFNESS_KEY_SUFFIX, 0)),
	// 	redis_client.getEigen(createRedisKey(MAX_DAMPING_KEY_SUFFIX, 0)),
	// 	redis_client.getEigen(createRedisKey(MAX_FORCE_KEY_SUFFIX, 0)));
	// Matrix3d device_base_rotation_in_world = AngleAxisd(DEVICE_ROTATION, Vector3d::UnitZ()).toRotationMatrix();
	// auto haptic_controller =
	// 	make_shared<Sai2Primitives::HapticDeviceController>(
	// 		device_limits, robot->transformInWorld(control_link, CONTROL_POINT),
	// 		Affine3d::Identity(),
	// 		device_base_rotation_in_world);
	// haptic_controller->setScalingFactors(POS_SCALE_FACTOR, ORI_SCALE_FACTOR);
	// haptic_controller->setReductionFactorForce(FORCE_REDUCTION);
	// // haptic_controller->setVariableDampingGainsPos(vector<double>{0.15, 0.25},
	// // 											  vector<double>{2, 20});
	// haptic_controller->setVariableDampingGainsPos(vector<double>{0.05, 0.15},
	// 											  vector<double>{2, 20});
	// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::HOMING);
	// haptic_controller->disableOrientationTeleop();

	// Sai2Primitives::HapticControllerInput haptic_input;
	// Sai2Primitives::HapticControllerOutput haptic_output;
	// bool haptic_button_was_pressed = false;
	// int haptic_button_is_pressed = 0;
	// redis_client.setInt(createRedisKey(SWITCH_PRESSED_KEY_SUFFIX, 0),
	// 					haptic_button_is_pressed);
	// redis_client.setInt(createRedisKey(USE_GRIPPER_AS_SWITCH_KEY_SUFFIX, 0), 1);

	// // create bilateral teleop POPC
	// auto POPC_teleop = make_shared<Sai2Primitives::POPCBilateralTeleoperation>(
	// 	motion_force_task, haptic_controller, 0.001);

	// // setup redis communication
	// redis_client.addToSendGroup(createRedisKey(COMMANDED_FORCE_KEY_SUFFIX, 0),
	// 							haptic_output.device_command_force);
	// redis_client.addToSendGroup(createRedisKey(COMMANDED_TORQUE_KEY_SUFFIX, 0),
	// 							haptic_output.device_command_moment);

	// redis_client.addToReceiveGroup(createRedisKey(POSITION_KEY_SUFFIX, 0),
	// 							   haptic_input.device_position);
	// redis_client.addToReceiveGroup(createRedisKey(ROTATION_KEY_SUFFIX, 0),
	// 							   haptic_input.device_orientation);
	// redis_client.addToReceiveGroup(
	// 	createRedisKey(LINEAR_VELOCITY_KEY_SUFFIX, 0),
	// 	haptic_input.device_linear_velocity);
	// redis_client.addToReceiveGroup(
	// 	createRedisKey(ANGULAR_VELOCITY_KEY_SUFFIX, 0),
	// 	haptic_input.device_angular_velocity);
	// redis_client.addToReceiveGroup(createRedisKey(SWITCH_PRESSED_KEY_SUFFIX, 0),
	// 							   haptic_button_is_pressed);

	/*
		Proxy information 
	*/
	std::cout << "Proxy Setup\n";
	Vector3d global_proxy_pos = robot->position(control_link, CONTROL_POINT);
	Matrix3d global_proxy_ori = robot->rotation(control_link);
	Vector3d prev_global_proxy_pos = global_proxy_pos;
	Matrix3d prev_global_proxy_ori = global_proxy_ori;
	Vector3d starting_proxy_position, ending_proxy_position;
	Matrix3d starting_proxy_ori, ending_proxy_ori;

	// frame reset containers 
	bool flag_frame_reset = false;
	bool flag_frame_reset_first_loop = false; 
	bool flag_ori_reset = false;
	double starting_time_pos_reset, ending_time_pos_reset;
	double total_time_pos_reset = FRAME_RESET_TIME;
	double starting_time_ori_reset, ending_time_ori_reset;
	double total_time_ori_reset = FRAME_RESET_TIME;

	// open and initialize gripper 
	bool gripper_is_open = true;
	if (ACTIVE_GRIPPER_CONTROL) {
		redis_client.set(GRIPPER_MODE_KEY, "o");
		redis_client.setDouble(GRIPPER_DESIRED_WIDTH_KEY, 0.08);
		redis_client.setInt(GRIPPER_GRASP_SUCCESSFUL_KEY, 0);
		redis_client.setInt(GRIPPER_MOVE_SUCCESSFUL_KEY, 0);
	}

	// force space particle filter 
	// int moment_control_mode = 0;  // 1 for moment control using relative orientation 
	bool flag_in_contact = false;
	bool flag_prev_contact = false;
	int loop_force_space_dimension = force_space_dimension;
	Matrix3d loop_sigma_force = sigma_force;
	Matrix3d loop_sigma_motion = sigma_motion;
	bool loop_flag_force_to_free = flag_filter_force_to_free;
	Vector3d loop_force_or_motion_axis = force_or_motion_axis;
	std::vector<Vector3d> loop_force_axes = force_axes;

	Vector3d integrated_moment_error = Vector3d::Zero();
	Vector3d prev_moment_error = Vector3d::Zero();

	bool flag_force_blending = false;
	double force_transition_start_time = 0;
	Vector3d force_before_contact = Vector3d::Zero();
	bool flag_ori_frame_reset = false;
	bool flag_ori_frame_reset_first_loop = false;
    bool zero_moment_control_flag = false;
	bool prev_zero_moment_control_flag = false;

	double frame_time_offset = 0;

	Vector3d desired_position_offset_from_moving_control_point = Vector3d::Zero();

	bool hold_robot_pose = false;  // used in contact to free space, to hold the robot's curernt pose until frame reset is finished 

	// low pass filter
	auto task_point_force_moment_filter = Sai2Common::ButterworthFilter(5, 1000);  
	task_point_force_moment_filter.initializeFilter(VectorXd::Zero(3));
	auto projected_task_point_force_moment_filter = Sai2Common::ButterworthFilter(5, 1000);  
	projected_task_point_force_moment_filter.initializeFilter(VectorXd::Zero(3));
	// VectorXd low_pass_force_moment = VectorXd::Zero(6);
	// auto force_sensor_lpf = ButterworthLowPass(5, 1000, 6);

	// orientation low pass filter
	auto rotation_lpf = RotationLowPassFilter(5, 1000);
	auto normal_force_lpf = Sai2Common::ButterworthFilter(2, 1000);  // tune normal force low pass filter 
	normal_force_lpf.initializeFilter(Vector3d::Zero());

	/*
		Initialize keys not already initialized 
	*/
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, control_torques);
	redis_client.setInt(CONTROLLER_RESET_KEY, 0);
	redis_client.setInt(ZERO_MOMENT_CONTROL_KEY, 0);
	redis_client.setInt(DIFFUSION_START_KEY, 0);

	/*
		Start threads 
	*/
	// start particle filter loop
	thread particle_filter_thread(particle_filter);

	// start key read thread
	thread key_read_thread(getKeyPressAndGraphics);

	// create a timer
	Sai2Common::LoopTimer timer(1000, 1e9);  // second argment: wait in nanoseconds before starting loop
	double start_time = timer.elapsedTime(); // secs
	bool first_loop = true;
	unsigned long long counter = 0;
	runloop = true;
	std::cout << "Starting Control\n"; 
	
	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		double time = timer.elapsedTime() - start_time;

		// execute redis read callback
		// redis_client.receiveAllFromGroup();

		// get diffusion output 
		global_proxy_pos = redis_client.getEigen(DIFFUSION_PROXY_POS_KEY);
		global_proxy_ori = redis_client.getEigen(DIFFUSION_PROXY_ORI_KEY);
		zero_moment_control_flag = redis_client.getInt(ZERO_MOMENT_CONTROL_KEY);
		double desired_gripper_pos = redis_client.getDouble(DIFFUSION_GRIPPER_POS_KEY);

		// compute haptic control
		// haptic_input.robot_position = robot->positionInWorld(control_link, CONTROL_POINT);
		// haptic_input.robot_orientation = robot->rotationInWorld(control_link);
		// haptic_input.robot_position = global_proxy_pos;
		// haptic_input.robot_orientation = global_proxy_ori;
		// haptic_input.robot_linear_velocity =
		// 	robot->linearVelocityInWorld(control_link, CONTROL_POINT);
		// haptic_input.robot_angular_velocity =
		// 	robot->angularVelocityInWorld(control_link);
		// haptic_input.robot_sensed_force =
		// 	motion_force_task->getSensedForceControlWorldFrame();
		// haptic_input.robot_sensed_moment =
		// 	motion_force_task->getSensedMomentControlWorldFrame();

		// get sensor feedback
		if (!FLAG_SIMULATION) {
			robot_q.head(7) = redis_client.getEigen(JOINT_ANGLES_KEY);
			robot_dq.head(7) = redis_client.getEigen(JOINT_VELOCITIES_KEY);
		} else {
			robot_q = redis_client.getEigen(JOINT_ANGLES_KEY);
			robot_dq = redis_client.getEigen(JOINT_VELOCITIES_KEY);	
		}
		force_moment = redis_client.getEigen(FORCE_SENSOR_KEY);

		// update robot
		robot->setQ(robot_q);
		robot->setDq(robot_dq);
		VectorXd dynamic_bias_torques = VectorXd::Zero(dof);
		if (!FLAG_SIMULATION) {
			MatrixXd M = redis_client.getEigen(MASS_MATRIX_KEY);
			// dynamic_bias_torques = redis_client.getEigen(CORIOLIS_KEY);
			M(4, 4) += BIE_ADDITION;
			M(5, 5) += BIE_ADDITION;
			M(6, 6) += BIE_ADDITION;

			// add load mass contribution (original load mass is already compensated for)
			if (grasped_object && ACTIVE_GRIPPER_CONTROL) {
				MatrixXd Jv_load = robot->Jv(control_link, load_com);
				MatrixXd Jw_load = robot->Jw(control_link);
				M += (load_mass - original_load_mass) * Jv_load.transpose() * Jv_load + Jw_load.transpose() * load_inertia.asDiagonal() * Jw_load;
				dynamic_bias_torques = - Jv_load.transpose() * (load_mass - original_load_mass) * Vector3d(0, 0, -9.81);
			}
			robot->updateModel(M);

		} else {
			robot->updateModel();
			dynamic_bias_torques = robot->coriolisForce();
		}

		// get kinematic data
		Vector3d ee_pos_original = robot->position(control_link, ORIGINAL_CONTROL_POINT);
		Vector3d ee_pos = robot->position(control_link, CONTROL_POINT);
		Matrix3d ee_ori = robot->rotation(control_link);
		Vector3d ee_lin_vel_original = robot->linearVelocity(control_link, ORIGINAL_CONTROL_POINT);
		Vector3d ee_lin_vel = robot->linearVelocity(control_link, CONTROL_POINT);
		Vector3d ee_ang_vel = robot->angularVelocity(control_link);

		redis_client.setEigen(EE_POS_KEY, ee_pos);
		redis_client.setEigen(EE_POS_ORIGINAL_KEY, ee_pos_original);
		redis_client.setEigen(EE_ORI_KEY, ee_ori);
		redis_client.setEigen(EE_LIN_VEL_ORIGINAL_KEY, ee_lin_vel_original);
		redis_client.setEigen(EE_LIN_VEL_KEY, ee_lin_vel);
		redis_client.setEigen(EE_ANG_VEL_KEY, ee_ang_vel);
		redis_client.setInt(GRASPED_OBJECT_KEY, grasped_object);

		/*
			*******************************************
			*******************************************
						Force sensor 
			*******************************************
			*******************************************
		*/

		// remove calibrated bias 
		force_moment -= force_bias;

		// remove load mass bias 
		Matrix3d R_sensor_to_link = Matrix3d::Identity();
		Matrix3d R_link_to_base = robot->rotation(control_link);  // from link to inertial frame 
		Matrix3d R_sensor_to_base = R_link_to_base * R_sensor_to_link;  // sensor to inertial frame rotation 
		Affine3d T_sensor = Affine3d::Identity();
		T_sensor.translation() = SENSOR_POINT;
		Vector3d tool_force = load_mass * R_sensor_to_base.transpose() * Vector3d(0, 0, -9.81);
		Vector3d tool_moment = load_com.cross(tool_force);
		force_moment.head(3) += tool_force;
		force_moment.tail(3) += tool_moment;

		// // remove force sensor start-up bias 
		// if (first_loop) {
		// 	if (!FLAG_SIMULATION && state == MOTION) {
		// 		initial_force_moment_bias = force_moment;
		// 		first_loop = false;
		// 	}
		// } 
		force_moment -= 1 * initial_force_moment_bias;

		// low pass filter : NOTE, UNSTABLE CLOSED-LOOP FORCE CONTROL WITH THIS (SEE REFERENCE)
		// VectorXd force_moment_filtered = force_sensor_lpf.update(force_moment);
		// force_moment = force_sensor_lpf.update(force_moment);

		// receive force sensor information 
		if (state == MOTION) {
			motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));
			// motion_force_task->updateSensedForceAndMoment(force_moment_filtered.head(3), force_moment_filtered.tail(3));
		}

		// LPF for task point force-moment signal 
		Vector3d filtered_task_point_force = task_point_force_moment_filter.update(motion_force_task->getSensedForceControlWorldFrame());
		Vector3d filtered_projected_task_point_force = projected_task_point_force_moment_filter.update(loop_sigma_force * motion_force_task->getSensedForceControlWorldFrame());

		/*
			*******************************************
			*******************************************
					Particle filter update 
			*******************************************
			*******************************************
		*/
		// filter update mutex
		if (reset_filter_flag.load() == 0) {			
			std::lock_guard<mutex> lock(mutex_pfilter);
			loop_force_space_dimension = force_space_dimension;
			loop_force_or_motion_axis = force_or_motion_axis;
			loop_sigma_force = sigma_force;
			loop_sigma_motion = sigma_motion;
			loop_flag_force_to_free = flag_filter_force_to_free;
			loop_force_axes = force_axes;
			// std::cout << "loop force axes\n";
			// for (auto axis : force_axes) {
				// std::cout << axis.transpose() << "\n";
			// }
		}

		/*
			*******************************************
			*******************************************
						Object grasping 
			*******************************************
			*******************************************
		*/

		if (ACTIVE_GRIPPER_CONTROL) {

			// push forward
			prev_grasped_object = grasped_object;

			// execute gripper action
			if (state == MOTION) {
				if (desired_gripper_pos > 0.07 && grasped_object) {
					redis_client.set(GRIPPER_MODE_KEY, "o");
					redis_client.setDouble(GRIPPER_DESIRED_WIDTH_KEY, 0.08);
				} else if (desired_gripper_pos <= 0.01 && !grasped_object) {
					redis_client.set(GRIPPER_MODE_KEY, "g");
					redis_client.setInt(GRIPPER_DESIRED_WIDTH_KEY, 0);
				}
			}

			// if grasping and releasing motion, then clutch the haptic and hold the robot 
			if ((redis_client.get(GRIPPER_MODE_KEY) == "g" && grasped_object == false) || \
					(redis_client.get(GRIPPER_MODE_KEY) == "o" && grasped_object == true)) {					
				// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);
				// haptic_controller->setOutputGoal(ee_pos, ee_ori);
				motion_force_task->reInitializeTask();
				// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));

				// reset control
				// motion_force_task->setGoalForce(Vector3d::Zero());
				// motion_force_task->setGoalMoment(Vector3d::Zero());
				// motion_force_task->setClosedLoopForceControl(false);
				// motion_force_task->setClosedLoopMomentControl(false);
				// motion_force_task->parametrizeMomentRotMotionSpaces(0);
				// motion_force_task->setOriControlGains(0, KV_ORI, 0);
			}

			// if grasping and releasing motion is successful, then change logic 
			if (redis_client.get(GRIPPER_MODE_KEY) == "g" && redis_client.get(GRIPPER_GRASP_SUCCESSFUL_KEY) == "1") {
				if (!prev_grasped_object) {
					std::cout << time << ": GRASPED OBJECT\n";
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::MOTION_MOTION);
					grasped_object = true;
					redis_client.set(GRIPPER_GRASP_SUCCESSFUL_KEY, "0");

					// reset particle filter queues
					clearQueue(pfilter_motion_control_buffer);
					clearQueue(pfilter_force_control_buffer);
					clearQueue(pfilter_motion_control_buffer);
					clearQueue(pfilter_sensed_force_buffer);
					motion_force_task->reInitializeTask();
					// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));
					reset_filter_flag.store(1);
				}
			} else if (redis_client.get(GRIPPER_MODE_KEY) == "o" && redis_client.get(GRIPPER_MOVE_SUCCESSFUL_KEY) == "1") {
				if (prev_grasped_object) {
					std::cout << time << ": RELEASED OBJECT\n";
					// set haptic clutch to prevent haptic force jerk when moving ?
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::MOTION_MOTION);
					grasped_object = false;
					redis_client.set(GRIPPER_MOVE_SUCCESSFUL_KEY, "0");

					// reset particle filter queues
					clearQueue(pfilter_motion_control_buffer);
					clearQueue(pfilter_force_control_buffer);
					clearQueue(pfilter_motion_control_buffer);
					clearQueue(pfilter_sensed_force_buffer);
					motion_force_task->reInitializeTask();
					// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));
					reset_filter_flag.store(1);

					// standard reset
					motion_force_task->setClosedLoopForceControl(false);
					motion_force_task->setClosedLoopMomentControl(false);
					motion_force_task->parametrizeForceMotionSpaces(0);
					motion_force_task->parametrizeMomentRotMotionSpaces(0);
					motion_force_task->setOriControlGains(KP_ORI, KV_ORI, 0);
					motion_force_task->reInitializeTask();
					// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));
					// flag_in_contact = false;

					// reset global proxy pos and orientation to current ee pos and orientation
					// global_proxy_pos = ee_pos;
					// global_proxy_ori = ee_ori;
					// flag_in_contact = false;

					// reset original load mass and com parameters 
					std::cout << time << ": RESETTING LOAD PARAMETERS\n";
					load_mass = original_load_mass;
					load_inertia = original_load_inertia;
					load_com = original_load_com; 		

					// change control point
					motion_force_task->setControlPoint(CONTROL_POINT);	
					desired_position_offset_from_moving_control_point = Vector3d::Zero();
					motion_force_task->setForceSensorFrame(control_link, sensor_transform);
					motion_force_task->reInitializeTask();
					// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));

					// set zero desired force and moment
					motion_force_task->setGoalForce(Vector3d::Zero());
					motion_force_task->setGoalMoment(Vector3d::Zero());

					// trigger frame reset
					loop_force_space_dimension = 0;
					flag_in_contact = true;
					loop_flag_force_to_free = true;
				}
				// re-tare haptic force values 
				// initial_force_moment_bias = force_moment;
			}
			
			// // if object isn't grasped, then restore original load mass and com 
			// if (grasped_object == false && prev_grasped_object == true) {
			// 	std::cout << time << ": RESETTING LOAD PARAMETERS\n";
			// 	load_mass = original_load_mass;
			// 	load_inertia = original_load_inertia;
			// 	load_com = original_load_com; 		

			// 	// change control point
			// 	motion_force_task->setControlPoint(CONTROL_POINT);	
			// 	desired_position_offset_from_moving_control_point = Vector3d::Zero();
			// 	motion_force_task->setForceSensorFrame(control_link, sensor_transform);
			// 	motion_force_task->reInitializeTask();
			// 	// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));

			// 	// set zero desired force and moment
			// 	motion_force_task->setGoalForce(Vector3d::Zero());
			// 	motion_force_task->setGoalMoment(Vector3d::Zero());

			// 	// reset particle filter queues
			// 	clearQueue(pfilter_motion_control_buffer);
			// 	clearQueue(pfilter_force_control_buffer);
			// 	clearQueue(pfilter_motion_control_buffer);
			// 	clearQueue(pfilter_sensed_force_buffer);
			// 	// motion_force_task->reInitializeTask();
			// 	reset_filter_flag.store(1);
			// }

			// track desired offset if grasped object 
			if (grasped_object) {
				desired_position_offset_from_moving_control_point = \
					robot->positionInWorld(control_link, LOAD_CONTROL_POINT) - robot->positionInWorld(control_link, CONTROL_POINT);
			}

			// update task given added payload information 
			if (grasped_object == true && prev_grasped_object == false) {
				// starting_ee_pos = ee_pos;
				// state = PAYLOAD_DETECTION;
				// flag_payload_estimation_needed = false;
				// std::cout << "Starting payload estimation\n";
				std::cout << time << ": ADDED PAYLOAD OF MASS " << OBJECT_MASS << " AND COM " \
					<< OBJECT_COM.transpose() << " AND INERTIA " << OBJECT_INERTIA.transpose() << "\n";
				load_mass = original_load_mass + OBJECT_MASS;
				load_inertia = OBJECT_INERTIA;			
				load_com = (original_load_mass * original_load_com + OBJECT_MASS * OBJECT_COM) / (load_mass);
				flag_payload_estimation_needed = false;

				// change control point 
				std::cout << "CHANGING CONTROL POINT TO " << LOAD_CONTROL_POINT.transpose() << "\n";
				motion_force_task->setControlPoint(LOAD_CONTROL_POINT);
				desired_position_offset_from_moving_control_point = \
					robot->positionInWorld(control_link, LOAD_CONTROL_POINT) - robot->positionInWorld(control_link, CONTROL_POINT);	
				motion_force_task->setForceSensorFrame(control_link, sensor_transform);

				// reset particle filter queues
				clearQueue(pfilter_motion_control_buffer);
				clearQueue(pfilter_force_control_buffer);
				clearQueue(pfilter_motion_control_buffer);
				clearQueue(pfilter_sensed_force_buffer);
				motion_force_task->reInitializeTask();
				// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));
				reset_filter_flag.store(1);
			}

		}

		/*
			*******************************************
			*******************************************
					Contact event handling 
			*******************************************
			*******************************************
		*/

		// handle motion force task transitions if NOT in state estimation state
		if (state != PAYLOAD_DETECTION) {
			// robot detects contact 
			if (loop_force_space_dimension > 0) {
				/*
				********************************************************************************************************
					FREE SPACE TO CONTACT SPACE TRANSITION 
				********************************************************************************************************
				*/
				if (!flag_in_contact) {
					motion_force_task->setForceControlGains(KP_FORCE, KV_FORCE, KI_FORCE);
					// motion_force_task->setClosedLoopForceControl(false);
					motion_force_task->setClosedLoopForceControl(true);
					zero_moment_control_flag = false;
					prev_zero_moment_control_flag = false;
					motion_force_task->parametrizeMomentRotMotionSpaces(0);
					// motion_force_task->setOriControlGains(KP_ORI_LOW, KV_MOMENT, 0);

					flag_in_contact = true;

					// blend desired force to prevent jump : maybe not necessary 
					// flag_force_blending = true;
					force_before_contact = motion_force_task->getSensedForceControlWorldFrame();
					force_transition_start_time = time;

				} else if (!flag_ori_frame_reset && zero_moment_control_flag == false && prev_zero_moment_control_flag == true) {
					// orientation slerp from zero moment control to orientation control in contact
					flag_ori_frame_reset_first_loop = true; 
					flag_ori_frame_reset = true; 
					starting_time_ori_reset = time;
					ending_time_ori_reset = starting_time_ori_reset + total_time_ori_reset;
					starting_proxy_ori = global_proxy_ori;
					ending_proxy_ori = ee_ori;

					// // disable haptic input orientation
					// if (!haptic_controller->getOrientationTeleopEnabled()) {
					// 	haptic_controller->disableOrientationTeleop();
					// }					
				}

			} else if (loop_force_space_dimension == 0) {
				/*
				********************************************************************************************************
					CONTACT SPACE TO FREE SPACE TRANSITION 
				********************************************************************************************************
				*/
				// if ((flag_in_contact && loop_flag_force_to_free) || flag_trigger_reset_from_load_change) {
				if (flag_in_contact && loop_flag_force_to_free) {
					motion_force_task->setClosedLoopForceControl(false);
					motion_force_task->setClosedLoopMomentControl(false);
					motion_force_task->parametrizeForceMotionSpaces(0);
					motion_force_task->parametrizeMomentRotMotionSpaces(0);
					motion_force_task->setOriControlGains(KP_ORI, KV_ORI, 0);
					motion_force_task->reInitializeTask();
					// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));
					flag_in_contact = false;

					// instantaneous reset
					// global_proxy_pos = ee_pos;
					// global_proxy_pos_ghost = robot->position(control_link, Vector3d::Zero());

					// interpolated frame reset 
					flag_frame_reset_first_loop = true; 
					flag_frame_reset = true;
					starting_time_pos_reset = time;
					ending_time_pos_reset = starting_time_pos_reset + total_time_pos_reset;  // T seconds of interpolation window 
					starting_proxy_position = global_proxy_pos;
					ending_proxy_position = ee_pos;
					// ending_proxy_position = starting_proxy_position;
					
					starting_time_ori_reset = time;
					ending_time_ori_reset = starting_time_ori_reset + total_time_ori_reset;
					starting_proxy_ori = global_proxy_ori;
					ending_proxy_ori = ee_ori;
					// ending_proxy_ori = starting_proxy_ori;

					// override the current orientation frame reset 
					flag_ori_frame_reset = false;

					// hold robot pose 
					hold_robot_pose = true;
				}
			} 

			/*
			********************************************************************************************************
				FRAME RESET LOGIC  
			********************************************************************************************************
			*/
			// motion force task update 
			if (flag_frame_reset) {
				if (time > ending_time_pos_reset) {
					flag_frame_reset = false;
					integrated_moment_error.setZero();
					motion_force_task->reInitializeTask();
					// motion_force_task->updateSensedForceAndMoment(force_moment.head(3), force_moment.tail(3));
					global_proxy_pos = ee_pos;
					global_proxy_ori = ee_ori;
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::MOTION_MOTION);
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);
					// overwrite the last output of the haptic controller 
					// if (!haptic_controller->getOrientationTeleopEnabled()) {
					// 	haptic_controller->enableOrientationTeleop();
					// }

					// reset force blending 
					flag_force_blending = false;
					force_before_contact.setZero();

					// stop holding robot 
					hold_robot_pose = false;

				} else {
					std::cout << time << ": FRAME RESET\n";
					// motion_force_task->disableInternalOtg();
					// ending_proxy_position = robot->position(control_link, control_point);
					// ending_proxy_ori = robot->rotation(control_link);

					// // interpolate frame reset (ADJUST STARTING TIME SUCH THAT INTERPOLATION STARTS FROM ZERO)
					// if (flag_frame_reset_first_loop) {
					// 	frame_time_offset = time;
					// 	flag_frame_reset_first_loop = false;
					// }
					global_proxy_pos = linearInterp(starting_proxy_position, ending_proxy_position, time, starting_time_pos_reset, ending_time_pos_reset); 
					global_proxy_ori = slerp(starting_proxy_ori, ending_proxy_ori, time, starting_time_ori_reset, ending_time_ori_reset);

					// debug
					// std::cout << "Global proxy ori from slerp: \n" << global_proxy_ori << "\n";

					// gain schedule the orientation from low to high gains 
					// double alpha = std::clamp((time - starting_time_pos_reset) / (ending_time_pos_reset - starting_time_pos_reset), 0., 1.);
					// motion_force_task->setPosControlGains(KP_FORCE_PROXY + std::pow(alpha, 2) * (KP_POS - KP_FORCE_PROXY), 15);
					// motion_force_task->setOriControlGains(KP_ORI_LOW + std::pow(alpha, 2) * (KP_ORI - KP_ORI_LOW), KV_ORI);
					// motion_force_task->setOriControlGains(kp_ori_high, 20);

					// blending force control to free-space control 

					// set haptic clutch
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);

					// // zero out forces
					// haptic_input.robot_sensed_force =
					// 	0 * motion_force_task->getSensedForceControlWorldFrame();
					// haptic_input.robot_sensed_moment =
					// 	0 * motion_force_task->getSensedMomentControlWorldFrame();
				}
			}

			// slerp of orientation when switching between IN CONTACT moment and orientation control
			if (flag_ori_frame_reset && !flag_frame_reset) {
				if (time > ending_time_ori_reset) {
					flag_ori_frame_reset = false;
					integrated_moment_error.setZero();
					// motion_force_task->reInitializeTask();
					// global_proxy_pos = robot->position(control_link, CONTROL_POINT);
					
					global_proxy_ori = ee_ori;

					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::MOTION_MOTION);
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);
					// overwrite the last output of the haptic controller 

					// // enable haptic input orientation
					// if (!haptic_controller->getOrientationTeleopEnabled()) {
					// 	haptic_controller->enableOrientationTeleop();
					// }

					// reset force blending 
					flag_force_blending = false;
					force_before_contact.setZero();

				} else {
					std::cout << time << ": ORIENTATION FRAME RESET\n";
					// motion_force_task->disableInternalOtg();
					// ending_proxy_position = robot->position(control_link, control_point);
					// ending_proxy_ori = robot->rotation(control_link);

					// if (flag_ori_frame_reset_first_loop) {
					// 	frame_time_offset = timer.elapsedTime() - start_time;  // get time offset when calling the function 
					// 	starting_time_ori_reset = time;
					// 	ending_time_ori_reset = time + total_time_ori_reset;
					// 	flag_ori_frame_reset_first_loop = false;
					// }

					// interpolate frame reset
					// global_proxy_pos = linearInterp(starting_proxy_position, ending_proxy_position, time, starting_time_pos_reset, ending_time_pos_reset); 
					global_proxy_ori = slerp(starting_proxy_ori, ending_proxy_ori, time, starting_time_ori_reset, ending_time_ori_reset);

					// // disable haptic input orientation first time 
					// if (!haptic_controller->getOrientationTeleopEnabled()) {
					// 	haptic_controller->disableOrientationTeleop();
					// }

					// gain schedule the orientation from low to high gains 
					// double alpha = std::clamp((time - starting_time_pos_reset) / (ending_time_pos_reset - starting_time_pos_reset), 0., 1.);
					// motion_force_task->setPosControlGains(KP_FORCE_PROXY + std::pow(alpha, 2) * (KP_POS - KP_FORCE_PROXY), 15);
					// motion_force_task->setOriControlGains(KP_ORI_LOW + std::pow(alpha, 2) * (KP_ORI - KP_ORI_LOW), KV_ORI);
					// motion_force_task->setOriControlGains(kp_ori_high, 20);

					// blending force control to free-space control 

					// set haptic clutch
					// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);

					// // zero out forces
					// haptic_input.robot_sensed_force =
					// 	0 * motion_force_task->getSensedForceControlWorldFrame();
					// haptic_input.robot_sensed_moment =
					// 	0 * motion_force_task->getSensedMomentControlWorldFrame();
				}
			}
		}

		/*
			*******************************************
			*******************************************
					Main state machine 
			*******************************************
			*******************************************
		*/
		if (state == POSTURE) {
			joint_task->updateTaskModel(MatrixXd::Identity(dof, dof));
			control_torques = joint_task->computeTorques();

			if (joint_task->goalPositionReached(QTOL_INT) && !integrator_on) {
				integrator_on = true;
				joint_task->resetIntegrators();
				joint_task->setGains(KP_JOINT, KV_JOINT, KI_JOINT);
			}

			// integrator reset 			
			VectorXd current_joint_error = joint_task->getCurrentError();
			for (int i = 0; i < robot->dof(); ++i) {
				if (sign(current_joint_error(i)) != sign(previous_joint_error(i))) {
					joint_task->resetIntegratorsByIndex(i);
				}
			}
			previous_joint_error = current_joint_error;

			global_proxy_pos = ee_pos;
			global_proxy_ori = ee_ori;
			redis_client.setEigen(PROXY_POS_KEY, global_proxy_pos);
			redis_client.setEigen(PROXY_ORI_KEY, global_proxy_ori);

			// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);	
			// haptic_controller->setOutputGoal(ee_pos, ee_ori);
			// haptic_output = haptic_controller->computeHapticControl(haptic_input);  // homing input 

			redis_client.setInt(GRIPPER_GRASP_SUCCESSFUL_KEY, 0);
			redis_client.setInt(GRIPPER_MOVE_SUCCESSFUL_KEY, 0);
			redis_client.setInt(CONTROLLER_START_KEY, 0);  // this will reset the policy until starting 

			if (joint_task->goalPositionReached(QTOL) && redis_client.getInt(DIFFUSION_START_KEY)) {
				std::cout << "State To Motion\n";
				state = MOTION;
				motion_force_task->reInitializeTask();
				joint_task->reInitializeTask();  			
				joint_task->setGains(KP_JOINT, KV_JOINT, 0 * KI_JOINT);
				// joint_task->setGoalPosition(Q_MID);  // test (this causes the jump)
				joint_task->disableInternalOtg();

				global_proxy_pos = ee_pos;
				global_proxy_ori = ee_ori;
				redis_client.setEigen(PROXY_POS_KEY, global_proxy_pos);
				redis_client.setEigen(PROXY_ORI_KEY, global_proxy_ori);

				// reset particle filter queues
				clearQueue(pfilter_motion_control_buffer);
				clearQueue(pfilter_force_control_buffer);
				clearQueue(pfilter_motion_control_buffer);
				clearQueue(pfilter_sensed_force_buffer);

				// // haptic controller reset 
				// haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::MOTION_MOTION);
				// haptic_controller->setDeviceControlGains(200.0, 15.0);
				// haptic_controller->enableOrientationTeleop();
				// cout << "Haptic device homed" << endl;

				// set haptic clutch flag
				redis_client.set(HAPTIC_CLUTCH_KEY, "0");

				// remove force sensor start-up bias 
				if (first_loop) {
					// if (!FLAG_SIMULATION && state == MOTION) {
						initial_force_moment_bias = 0 * force_moment;
						std::cout << "Initial force moment bias: \n" << initial_force_moment_bias.transpose() << "\n";
						first_loop = false;
					// }
				} 

				continue;
			}
		} else if (state == MOTION) {

			// integrator reset 			
			VectorXd current_joint_error = joint_task->getCurrentError();
			for (int i = 0; i < robot->dof(); ++i) {
				if (sign(current_joint_error(i)) != sign(previous_joint_error(i))) {
					joint_task->resetIntegratorsByIndex(i);
				}
			}
			previous_joint_error = current_joint_error;
			prev_zero_moment_control_flag = zero_moment_control_flag;

			// haptic_output = haptic_controller->computeHapticControl(haptic_input);				

			// /*
			// ********************************************************************************************************
			// 	HAPTIC CONTROLLER BUTTON LOGIC
			// ********************************************************************************************************
			// */
			// // state machine for button presses
			// if (haptic_controller->getHapticControlType() == Sai2Primitives::HapticControlType::HOMING &&
			// 	haptic_controller->getHomed() && haptic_button_is_pressed) {
			// 	haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::MOTION_MOTION);
			// 	haptic_controller->setDeviceControlGains(200.0, 15.0);
			// 	std::cout << "Haptic device homed\n";
			// }

			// // hold haptic button to do zero moment control in contact, and clutch 
			// if (haptic_controller->getHapticControlType() == Sai2Primitives::HapticControlType::MOTION_MOTION &&
			// 		haptic_button_is_pressed && !haptic_button_was_pressed) {
			// 	if (!flag_in_contact && !flag_frame_reset && !flag_ori_frame_reset) {
			// 		haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::CLUTCH);

			// 		// set haptic clutch flag 
			// 		redis_client.set(HAPTIC_CLUTCH_KEY, "1");

			// 	} else if (flag_in_contact) {
			// 		std::cout << time << ": ENABLING ZERO MOMENT CONTROL IN CONTACT\n";
			// 		if (!zero_moment_control_flag) {
			// 			zero_moment_control_flag = true;
			// 		}
			// 	}					
			// } else if (haptic_controller->getHapticControlType() == Sai2Primitives::HapticControlType::MOTION_MOTION &&
			// 		!haptic_button_is_pressed && haptic_button_was_pressed) {
			// 	if (flag_in_contact) {
			// 		std::cout << time << ": DISABLE ZERO MOMENT CONTROL IN CONTACT\n";
			// 		if (zero_moment_control_flag) {
			// 			zero_moment_control_flag = false;
			// 		}
			// 	}
			// } else if (haptic_controller->getHapticControlType() == Sai2Primitives::HapticControlType::CLUTCH &&
			// 		!haptic_button_is_pressed && haptic_button_was_pressed) {
				
			// 	if (!flag_in_contact && !flag_frame_reset && !flag_ori_frame_reset) {
			// 		haptic_controller->setHapticControlType(Sai2Primitives::HapticControlType::MOTION_MOTION);

			// 		// set haptic clutch flag 
			// 		redis_client.set(HAPTIC_CLUTCH_KEY, "0");
			// 	} 
			// } else if (switch_orientation_control_flag) {
			// 	if (haptic_controller->getOrientationTeleopEnabled()) {
			// 		cout << "Disabling orientation teleoperation" << endl;	
			// 		haptic_controller->disableOrientationTeleop();
			// 	} else {
			// 		cout << "Enabling orientation teleoperation" << endl;
			// 		haptic_controller->enableOrientationTeleop();
			// 	}
			// 	switch_orientation_control_flag = false;
			// } 

			// haptic_button_was_pressed = haptic_button_is_pressed;

			// /*
			// ********************************************************************************************************
			// 	UPDATE GLOBAL PROXY POS AND ORI BASED ON HAPTIC OUTPUT, DEPENDING ON ROBOT STATE 
			// ********************************************************************************************************
			// */
			// // update global proxy position and orientation with haptic input
			// if (flag_frame_reset) {
			// 	// pass through
			// } else if (flag_ori_frame_reset) {
			// 	// only update position, but not orientation
			// 	global_proxy_pos = haptic_output.robot_goal_position;
			// 	// haptic_controller->setOutputGoalOrientation(ee_ori);
			// } else {
			// 	// update position and orientation
			// 	global_proxy_pos = haptic_output.robot_goal_position;
			// 	// if (!zero_moment_control_flag) {
			// 		global_proxy_ori = haptic_output.robot_goal_orientation;  // global proxy orientation is filtered through the zero moment control 
			// 	// }
			// }

			/*
			********************************************************************************************************
				UPDATE PARTICLE FILTER BASED ON MOTION 
			********************************************************************************************************
			*/
			pfilter_motion_control_buffer.push(loop_sigma_motion * (global_proxy_pos + desired_position_offset_from_moving_control_point - motion_force_task->getCurrentPosition()) * freq_ratio_filter_control);
			pfilter_force_control_buffer.push(loop_sigma_force * (global_proxy_pos + desired_position_offset_from_moving_control_point - motion_force_task->getCurrentPosition()) * freq_ratio_filter_control);

			if (reset_filter_flag.load() == 0) {
				motion_force_task->parametrizeForceMotionSpaces(loop_force_space_dimension, loop_force_or_motion_axis, true);
			}

			if (grasped_object) {
				pfilter_sensed_velocity_buffer.push(robot->linearVelocity(control_link, LOAD_CONTROL_POINT) * freq_ratio_filter_control);
			} else {
				pfilter_sensed_velocity_buffer.push(robot->linearVelocity(control_link, CONTROL_POINT) * freq_ratio_filter_control);
			}
			pfilter_sensed_force_buffer.push(motion_force_task->getSensedForceControlWorldFrame() * freq_ratio_filter_control);

			motion_control_pfilter += pfilter_motion_control_buffer.back();
			force_control_pfilter += pfilter_force_control_buffer.back();
			measured_velocity_pfilter += pfilter_sensed_velocity_buffer.back();
			measured_force_pfilter += pfilter_sensed_force_buffer.back();

			if (pfilter_motion_control_buffer.size() > 1 / freq_ratio_filter_control) {
				motion_control_pfilter -= pfilter_motion_control_buffer.front();
				force_control_pfilter -= pfilter_force_control_buffer.front();
				measured_velocity_pfilter -= pfilter_sensed_velocity_buffer.front();
				measured_force_pfilter -= pfilter_sensed_force_buffer.front();

				pfilter_motion_control_buffer.pop();
				pfilter_force_control_buffer.pop();
				pfilter_sensed_velocity_buffer.pop();
				pfilter_sensed_force_buffer.pop();			
			}

			/*
			********************************************************************************************************
				FORCE AND MOMENT CONTACT HANDLING 
			********************************************************************************************************
			*/
			// in contact, use proxy for desired force/moment
			// in free-space, set motion commnds
			Vector3d proxy_error = Vector3d::Zero();
			if (flag_in_contact) {

				// update force and linear motion 
				if (grasped_object) {
					motion_force_task->setGoalPosition(global_proxy_pos + desired_position_offset_from_moving_control_point);
				} else {
					motion_force_task->setGoalPosition(global_proxy_pos);
				}
				proxy_error = global_proxy_pos + desired_position_offset_from_moving_control_point - motion_force_task->getCurrentPosition();

				// set desired force with saturation (in each direction only)
				Vector3d desired_force = Vector3d::Zero();
				if (flag_force_blending) {
					double force_transition_time = time - force_transition_start_time;
					double alpha_force_blending = std::clamp(force_transition_time / FORCE_BLENDING_TIME, 0., 1.);
					std::cout << "alpha: " << alpha_force_blending << "\n";
					desired_force = std::pow(alpha_force_blending, 1) * KP_FORCE_PROXY * proxy_error + \
										std::pow(1 - alpha_force_blending, 1) * force_before_contact;
					std::cout << "force before contact: " << force_before_contact.transpose() << "\n";

					if (force_transition_time > FORCE_BLENDING_TIME) {
						flag_force_blending = false;
					}
				} else {
					desired_force = KP_FORCE_PROXY * proxy_error;
				}				

				// overwrite with nonlinear spring force scaling 
				desired_force = getNonlinearForce(proxy_error, MAX_FORCE_NORM, KP_FORCE_PROXY);

				// L2 norm saturation
				Vector3d saturated_desired_force = desired_force;
				if (desired_force.norm() > MAX_FORCE_NORM) {
					saturated_desired_force = MAX_FORCE_NORM * (desired_force / desired_force.norm());
				}
				if (desired_force.norm() < MIN_FORCE_NORM) {
					saturated_desired_force = MIN_FORCE_NORM * (desired_force / desired_force.norm());
				}

				// // L1 norm saturation 
				// Vector3d desired_force_direction = desired_force / desired_force.norm();
				// Vector3d saturated_desired_force = Vector3d::Zero();
				// std::vector<Vector3d> force_axes = {};
				// for (int i = 0; i < 3; ++i) {
				// 	force_axes.push_back(loop_sigma_force.col(i));
				// 	double force_in_direction = std::abs(desired_force.dot(force_axes[i]));
				// 	if (force_in_direction > MAX_FORCE_NORM) {
				// 		force_in_direction = MAX_FORCE_NORM;
				// 	} else if (force_in_direction < MIN_FORCE_NORM) {
				// 		force_in_direction = MIN_FORCE_NORM;
				// 	}
				// 	saturated_desired_force += force_in_direction * force_axes[i];					
				// }				
				// // project saturated l1 norm force back to desired force direction 
				// saturated_desired_force = std::abs((desired_force_direction.dot(saturated_desired_force))) * desired_force_direction;

				motion_force_task->setGoalForce(saturated_desired_force);

				// saturate proxy within outer sphere 
				// TODO: saturate in a sphere in just the force directions (?)
				// if (proxy_error.norm() > MAX_PROXY_RADIUS) {
					// global_proxy_pos = motion_force_task->getCurrentPosition() + \
					// 			MAX_PROXY_RADIUS * (proxy_error / proxy_error.norm());
					// global_proxy_pos = ee_pos + MAX_PROXY_RADIUS * (proxy_error / proxy_error.norm());
				// }

/*
					Moment/orientation control				
					- If zero moment control, then orientation follows the end-effector
						- Need to update the haptic orientation goal 
					- If orientation control, then orientation controls
				*/
			    if (zero_moment_control_flag) {

					std::cout << time << ": MOMENT CONTROL IN CONTACT\n";
					redis_client.set(ZERO_MOMENT_CONTROL_KEY, "1");

					if (!prev_zero_moment_control_flag) {
						// haptic_controller->disableOrientationTeleop();
						rotation_lpf.reset(ee_ori);
					}

					motion_force_task->setClosedLoopMomentControl(true);
					motion_force_task->setMomentControlGains(KP_MOMENT, KV_MOMENT, KI_MOMENT);
					motion_force_task->setOriControlGains(KP_ORI, KV_ORI, 0);

					// characterize axes of moment control based on force space dimension assuming surface-surface constraint 

					// enforce one axis of orientation control always (take the largest force direction)
					// Vector3d projected_force = loop_sigma_force * motion_force_task->getSensedForceControlWorldFrame();
					// Vector3d projected_force = loop_sigma_force * filtered_task_point_force;
					// Vector3d normalized_projected_force = projected_force.normalized();  // get normalized projected force direction
					// Vector3d normalized_projected_force = - filtered_projected_task_point_force.normalized();

					Vector3d normalized_projected_force = Vector3d::Zero();

					if (PRIMITIVE_FLAG == 0) {
						/*
							Truncate force readings based on the force threshold from particle filter
						*/
						// Vector3d control_point_forces = loop_sigma_force * motion_force_task->getSensedForceControlWorldFrame();
						Vector3d control_point_forces = motion_force_task->getSensedForceControlWorldFrame();
						// Vector3d control_point_forces = filtered_task_point_force;
						for (int ii = 0; ii < 3; ++ii) {
							double alpha = std::clamp((std::abs(control_point_forces(ii)) - F_LOW) / (F_HIGH - F_LOW), 0.0, 1.0);
							control_point_forces(ii) = std::pow(alpha, 2) * control_point_forces(ii);
						}
						normalized_projected_force = - control_point_forces.normalized();
						std::cout << time << ": ZERO MOMENT MOTION AXIS: " << normalized_projected_force.transpose() << "\n";

						// Vector3d normalized_projected_force = - filtered_task_point_force.normalized();
						// Vector3d normalized_projected_force = - motion_force_task->getSensedForceControlWorldFrame().normalized();

						// low pass filter the normalized projected force
						VectorXd filtered_normalized_projected_force = normal_force_lpf.update(normalized_projected_force).normalized();
						motion_force_task->parametrizeMomentRotMotionSpaces(2, filtered_normalized_projected_force, true);  // two axes at a time (surface - surface)
						motion_force_task->setGoalMoment(Vector3d::Zero());
						normalized_projected_force = filtered_normalized_projected_force;  // set to the filtered value 
						std::cout << "Controlled Axis: " << normalized_projected_force.transpose() << "\n";

					} else if (PRIMITIVE_FLAG == 1) {

						motion_force_task->parametrizeMomentRotMotionSpaces(2, Vector3d(0, 1, 0), true); 
						motion_force_task->setGoalMoment(Vector3d::Zero());
						normalized_projected_force = Vector3d(0, 1, 0);  // hard-code the normalized projected force 
						std::cout << "Controlled Axis: " << normalized_projected_force.transpose() << "\n";

					} else if (PRIMITIVE_FLAG == 2) {

						motion_force_task->parametrizeMomentRotMotionSpaces(3, Vector3d(1, 0, 0), true);  // reset override
						motion_force_task->setGoalMoment(Vector3d::Zero());
						normalized_projected_force.setZero();
						
					}

					// /*
					// 	Truncate force readings based on the force threshold from particle filter
					// */
					// // Vector3d control_point_forces = loop_sigma_force * motion_force_task->getSensedForceControlWorldFrame();
					// Vector3d control_point_forces = motion_force_task->getSensedForceControlWorldFrame();
					// // Vector3d control_point_forces = filtered_task_point_force;
					// for (int ii = 0; ii < 3; ++ii) {
					// 	double alpha = std::clamp((std::abs(control_point_forces(ii)) - F_LOW) / (F_HIGH - F_LOW), 0.0, 1.0);
					// 	control_point_forces(ii) = std::pow(alpha, 2) * control_point_forces(ii);
					// }
					// Vector3d normalized_projected_force = - control_point_forces.normalized();
					// std::cout << time << ": ZERO MOMENT MOTION AXIS: " << normalized_projected_force.transpose() << "\n";

					// // Vector3d normalized_projected_force = - filtered_task_point_force.normalized();
					// // Vector3d normalized_projected_force = - motion_force_task->getSensedForceControlWorldFrame().normalized();

					// // low pass filter the normalized projected force
					// VectorXd filtered_normalized_projected_force = normal_force_lpf.update(normalized_projected_force).normalized();

					// // if (!normalized_projected_force.isZero()) {
					// if (!filtered_normalized_projected_force.isZero()) {
					// 	if (PRIMITIVE_FLAG == 0) {
					// 		// motion_force_task->parametrizeMomentRotMotionSpaces(2, normalized_projected_force, true);  // reset override 
					// 		motion_force_task->parametrizeMomentRotMotionSpaces(2, filtered_normalized_projected_force, true);  // two axes at a time (surface - surface)
					// 		// motion_force_task->parametrizeMomentRotMotionSpaces(1, filtered_normalized_projected_force, true);  // one axis at a time
					// 		// motion_force_task->parametrizeMomentRotMotionSpaces(1, Vector3d(1, 0, 0), true);  // plug hard-code
					// 	} else if (PRIMITIVE_FLAG == 1) {
					// 		motion_force_task->parametrizeMomentRotMotionSpaces(2, Vector3d(1, 0, 0), true);  
					// 		normalized_projected_force = Vector3d(1, 0, 0);  // hard-code the normalized projected force 
					// 	} else if (PRIMITIVE_FLAG == 2) {
					// 		motion_force_task->parametrizeMomentRotMotionSpaces(3, Vector3d(1, 0, 0), true);  // reset override
					// 	}
					// }
					// // motion_force_task->parametrizeMomentRotMotionSpaces(2, Vector3d(0, 0, 1));
					// motion_force_task->setGoalMoment(Vector3d::Zero());

					// global_proxy_ori = constrainRotationBetweenOrientations(global_proxy_ori, ee_ori, normalized_projected_force);

					// proxy orientation follows the end-effector after blending only in orthogonal axes 
					// if (!flag_ori_frame_reset && prev_zero_moment_control_flag != false) {
					if (!flag_ori_frame_reset) {
						if (normalized_projected_force.isZero()) {
							global_proxy_ori = ee_ori;
						} else {
							// global_proxy_ori = ee_ori;
							// haptic_controller->setOutputGoalOrientation(ee_ori);
							// haptic_controller->disableOrientationTeleop();
							// global_proxy_ori = computeConstrainedOrientation(ee_ori, global_proxy_ori, Vector3d(0, 0, 1));
							// global_proxy_ori = constrainRotationBetweenOrientations(global_proxy_ori, ee_ori, Vector3d(0, 0, 1));
							// global_proxy_ori = computeConstrainedOrientation(global_proxy_ori, ee_ori, normalized_projected_force);
							global_proxy_ori = computeConstrainedOrientation(ee_ori, global_proxy_ori, normalized_projected_force);  // start from current orientation, and go to desired orientation 
							// haptic_controller->setOutputGoalOrientation(global_proxy_ori);
						}

						// low pass filtered
						global_proxy_ori = rotation_lpf.filter(global_proxy_ori);  // log orientation change
						// motion_force_task->setGoalOrientation(global_proxy_ori);  // update orientation based on constrained motion direction
						// global_proxy_ori = ee_ori;
						// haptic_controller->setOutputGoalOrientation(global_proxy_ori);
						// global_proxy_ori = constrainRotationBetweenOrientations(global_proxy_ori, ee_ori, Vector3d(0, 0, 1));
					}

				} else {
					// orientation control in contact 
					std::cout << time << ": ORIENTATION CONTROL IN CONTACT\n";
					redis_client.set(ZERO_MOMENT_CONTROL_KEY, "0");

					if (flag_ori_frame_reset || prev_zero_moment_control_flag) {
						// global_proxy_ori interpolation still in use 
						// continue;
						normal_force_lpf.initializeFilter(Vector3d::Zero());
					} else {
						// if (!haptic_controller->getOrientationTeleopEnabled()) {
							// haptic_controller->enableOrientationTeleop();
							// global_proxy_ori = ee_ori;
							// haptic_controller->setOutputGoalOrientation(ee_ori);
							// motion_force_task->parametrizeMomentRotMotionSpaces(0, Vector3d::Zero(), true);
							// motion_force_task->setOriControlGains(KP_ORI_LOW, KV_MOMENT, 0);
							normal_force_lpf.initializeFilter(Vector3d::Zero());
						// } 

						motion_force_task->setOriControlGains(KP_ORI_LOW, KV_MOMENT, 0);
						motion_force_task->parametrizeMomentRotMotionSpaces(0, Vector3d::Zero(), true);
						motion_force_task->setGoalOrientation(global_proxy_ori);
					}
				}
			} else {
				// free space motion 
				if (grasped_object) {
					motion_force_task->setGoalPosition(global_proxy_pos + desired_position_offset_from_moving_control_point);
					motion_force_task->setGoalOrientation(global_proxy_ori);
				} else {
					motion_force_task->setGoalPosition(global_proxy_pos);
					motion_force_task->setGoalOrientation(global_proxy_ori);
				}
			}

			// compute torques 
			try {
				if (FLAG_SIMULATION) {
					// motion_force_task->updateTaskModel(MatrixXd::Identity(dof, dof));
					// gripper_joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
					// joint_task->updateTaskModel(gripper_joint_task->getTaskAndPreviousNullspace());
					// control_torques = motion_force_task->computeTorques() + gripper_joint_task->computeTorques() + joint_task->computeTorques();
				} else {
					if (hold_robot_pose) {
						motion_force_task->reInitializeTask();
					}
					motion_force_task->updateTaskModel(MatrixXd::Identity(dof, dof));
					joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
					control_torques = motion_force_task->computeTorques() + joint_task->computeTorques(); 
				}
			} catch (...) {
				std::cout << "Torque computation error \n";
				control_torques.setZero();
			}

			// // debug
			// std::cout << "Global proxy ori at end of cycle: \n" << global_proxy_ori << "\n";

			// publish proxy data for recording 
			redis_client.setEigen(PROXY_POS_KEY, global_proxy_pos);
			redis_client.setEigen(PROXY_ORI_KEY, global_proxy_ori);

			redis_client.setInt(FORCE_SPACE_DIMENSION_KEY, loop_force_space_dimension);
			redis_client.setEigen(FORCE_OR_MOTION_AXIS_KEY, loop_force_or_motion_axis);
		} 

		// execute redis write callback
		// redis_client.sendAllFromGroup();
		redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 1 * (control_torques + dynamic_bias_torques));
		if (isnan(control_torques(0))) {
			redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * control_torques);  // back to floating
			std::cout << "nan torques; setting to 0 torques\n";
		}

		// set calibrated force reading 
		redis_client.setEigen("sai2::FrankaPanda::Romeo::calibrated_force_moment", force_moment);
		VectorXd control_frame_force_moment(6);
		control_frame_force_moment.head(3) = motion_force_task->getSensedForceControlWorldFrame();
		control_frame_force_moment.tail(3) = motion_force_task->getSensedMomentControlWorldFrame();
		redis_client.setEigen("sai2::FrankaPanda::Romeo::control_calibrated_force_moment", control_frame_force_moment);
		redis_client.setEigen("sai2::FrankaPanda::Romeo::sigma_force", loop_sigma_force);

		// set force reading with filter 
		redis_client.setEigen("sai2::FrankaPanda::Romeo::control_calibrated_force_moment_with_filter", control_frame_force_moment * flag_in_contact);
		redis_client.setEigen("sai2::FrankaPanda::Romeo::calibrated_force_moment_with_filter", force_moment * flag_in_contact);

		// goal force
		redis_client.setEigen("sai2::FrankaPanda::Romeo::desired_forces", motion_force_task->getGoalForce());

		// state flags 
		redis_client.setInt(IS_IN_CONTACT_KEY, flag_in_contact);
		redis_client.setInt(IS_HOLDING_OBJECT_KEY, grasped_object);

		// controller start key for diffusion
		if (state == MOTION) {
			redis_client.setInt(CONTROLLER_START_KEY, 1);
		}

		// if (redis_client.getInt(ROBOT_RUNNING_KEY) == 0) {
		// 	runloop = false;
		// }

		counter++;
	}

	double end_time = timer.elapsedTime();
	std::cout << "Controller run stats:\n---\n";
	timer.printInfoPostRun();
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * control_torques);  // back to floating

	redis_client.setEigen(createRedisKey(COMMANDED_FORCE_KEY_SUFFIX, 0),
						  Vector3d::Zero());
	redis_client.setEigen(createRedisKey(COMMANDED_TORQUE_KEY_SUFFIX, 0),
						  Vector3d::Zero());
	redis_client.setInt(createRedisKey(USE_GRIPPER_AS_SWITCH_KEY_SUFFIX, 0), 0);
	particle_filter_thread.join();
	key_read_thread.join();

	return 0;
}

//------------------------------------------------------------------------------
Matrix3d getRotationMatrix(const Vector3d& ori) {
	return (AngleAxisd(ori(0), Vector3d::UnitX()) * \
			AngleAxisd(ori(1), Vector3d::UnitY()) * \
			AngleAxisd(ori(2), Vector3d::UnitZ())).toRotationMatrix();
}

//------------------------------------------------------------------------------
Eigen::Matrix3d rotationMatrixToVector(const Eigen::Vector3d& v, const Eigen::Vector3d& v_prime) {
    // Normalize vectors
    Eigen::Vector3d v_normalized = v.normalized();
    Eigen::Vector3d v_prime_normalized = v_prime.normalized();

    // Compute rotation axis and angle
    Eigen::Vector3d k = v_normalized.cross(v_prime_normalized);
    double theta = std::acos(v_normalized.dot(v_prime_normalized));

    // Compute rotation matrix using the Eigen library
    Eigen::Matrix3d K;
    K << 0, -k.z(), k.y(),
         k.z(), 0, -k.x(),
         -k.y(), k.x(), 0;

    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity() +
                                       std::sin(theta) * K +
                                       (1 - std::cos(theta)) * K * K;

    return rotation_matrix;
}

//------------------------------------------------------------------------------
void particle_filter() {
	// // start redis client for particles
	// auto redis_client_particles = RedisClient();
	// redis_client_particles.connect();

	unsigned long long pf_counter = 0;

	// create particle filter
	auto pfilter = ForceSpaceParticleFilter(N_PARTICLES);
	pfilter.setParameters(0, 0.025, 0.3, 0.05);
	// pfilter.setWeightingParameters(2, 6, 0.01, 0.07, 5, 10, 0.02, 0.1);
	// pfilter.setWeightingParameters(1, 4, 0.01, 0.07, 5, 10, 0.02, 0.1);  // F low, F high, v low, v high, F low add, F high add, v low add, v high add 
	pfilter.setWeightingParameters(F_LOW, F_HIGH, V_LOW, V_HIGH, F_LOW_ADD, F_HIGH_ADD, V_LOW_ADD, V_HIGH_ADD);  // F low, F high, v low, v high, F low add, F high add, v low add, v high add 
	Vector3d evals = Vector3d::Zero();
	Matrix3d evecs = Matrix3d::Identity();

	// initialize classification queue with free-space 
	for (int i = 0; i < QUEUE_SIZE; ++i) {
		force_dimension_queue.push(0);  
	}

	// initialize force axes
	std::vector<Vector3d> force_axes = {};

	// create a timer
	Sai2Common::LoopTimer timer(FILTER_FREQ, 1e6);
	double current_time = 0;
	double prev_time = 0;
	// double dt = 0;
	double start_time = timer.elapsedTime(); // secs

	while (runloop) {
		timer.waitForNextLoop();

		// filter reset 
		if (reset_filter_flag.load() == 1) {

			pfilter.reset();

			while (!force_dimension_queue.empty()) {
				force_dimension_queue.pop();
			}

			// initialize classification queue with free-space 
			for (int i = 0; i < QUEUE_SIZE; ++i) {
				force_dimension_queue.push(0);  
			}

			{
				std::lock_guard<mutex> lock(mutex_pfilter);
				sigma_force.setZero();
				sigma_motion.setIdentity();
				force_space_dimension = 0;
				force_or_motion_axis = Vector3d(0, 0, 1);
				force_axes = {};
			}
			reset_filter_flag.store(0);
			continue;
		}

		pfilter.update(motion_control_pfilter, force_control_pfilter, measured_velocity_pfilter, measured_force_pfilter);

		{
			std::lock_guard<mutex> lock(mutex_pfilter);
			prev_force_space_dimension = force_space_dimension;
			force_space_dimension = pfilter.getForceSpaceDimension();
			force_dimension_queue.pop();
			force_dimension_queue.push(force_space_dimension);
			flag_filter_force_to_free = false;			
		}

		bool all_elements_same = allElementsSame(force_dimension_queue);

		{
			std::lock_guard<mutex> lock(mutex_pfilter);	
			if (force_space_dimension == 0) {
				// make sure that past classifications must be 0 to be free-space, otherwise keep previous sigma
				if (all_elements_same && sigma_motion.norm() != 0) {
					flag_filter_force_to_free = true;
					sigma_force = pfilter.getSigmaForce();
					sigma_motion = Matrix3d::Identity() - sigma_force;
					force_axes = pfilter.getForceAxes();
					force_or_motion_axis = pfilter.getForceOrMotionAxis();
				}
			} else {
				sigma_force = pfilter.getSigmaForce(); 
				sigma_motion = Matrix3d::Identity() - sigma_force;
				force_axes = pfilter.getForceAxes();
				// std::cout << "force axes\n";
				// for (auto axis : force_axes) {
					// std::cout << axis.transpose() << "\n";
				// }
				force_or_motion_axis = pfilter.getForceOrMotionAxis();
			}
			// std::cout << "sigma motion: \n" << sigma_motion << "\n";
			// std::cout << "sigma force: \n" << sigma_force << "\n";
		}

		// debug
		// std::cout << "Force axes from pfilter loop\n";
		// for (auto axis: force_axes) {
			// std::cout << axis.transpose() << "\n";
		// }

		// for(int i=0 ; i<n_particles ; i++)
		// {
		// 	particle_positions_to_redis.col(i) = pfilter->_particles[i];
		// }
		// redis_client_particles.setEigenMatrixJSON(PARTICLE_POSITIONS_KEY, particle_positions_to_redis);

		pf_counter++;
	}

	std::cout << "Particle filter stats: \n---\n";
	timer.printInfoPostRun();
}

//------------------------------------------------------------------------------
void getKeyPressAndGraphics() {

	auto redis_client = Sai2Common::RedisClient();
	redis_client.connect();

	auto graphics = std::make_shared<Sai2Graphics::Sai2Graphics>(world_file);
	graphics->setBackgroundColor(66.0/255, 135.0/255, 245.0/255);
	// graphics->showLinkFrame(true, "panda_proxy", "end-effector", 0.15);	
	// graphics->showLinkFrame(true, "proxy_end_effector", 0.15);
	graphics->showObjectLinkFrame(true, "proxy_end_effector", 0.15);

	// // add label for control commands 
	// // Create a font object
	// chai3d::cFontPtr font = chai3d::NEW_CFONTCALIBRI20();

	// // Create a new label
	// chai3d::cLabel* label = new chai3d::cLabel(font);

	// // Set the text of the label
	// label->setText("Hello, CHAI3D!");

	// // Set the color of the label (optional)
	// label->m_fontColor.setWhite();

	// // Position the label in the scene (e.g., in 2D screen space relative to the camera)
	// label->setLocalPos(20, 40); // (x, y) in pixels

	// // Attach the label to the camera (or another parent object, such as the world)
	// graphics->getCamera("camera_fixed")->m_frontLayer->addChild(label);

	bool gripper_is_open = true;

	// graphics timer
	Sai2Common::LoopTimer graphicsTimer(30.0, 1e6);

	while (graphics->isWindowOpen()) {
		graphicsTimer.waitForNextLoop();

		for (auto& key : key_pressed) {
			key_pressed[key.first] = graphics->isKeyPressed(key.first);
		}

		// recording input 
		if (key_pressed.at(GLFW_KEY_P) && !key_was_pressed.at(GLFW_KEY_P)) {
			redis_client.setInt(LOGGER_START_KEY, 1);
			cout << "---------------------\nstarting recording\n-------------------------\n";
		}

		if (key_pressed.at(GLFW_KEY_L) && !key_was_pressed.at(GLFW_KEY_L)) {
			redis_client.setInt(LOGGER_STOP_KEY, 1);
			cout << "----------------------\nstopping recording\n-------------------------\n";
		}

		// gripper input 
		if (key_pressed.at(GLFW_KEY_E) && !key_was_pressed.at(GLFW_KEY_E)) {
			if (gripper_is_open) {
				cout << "closing gripper" << endl;
				redis_client.set(GRIPPER_MODE_KEY, "g");
				redis_client.setInt(GRIPPER_DESIRED_WIDTH_KEY, 0);
				gripper_is_open = false;
			} else {
				cout << "opening gripper" << endl;
				redis_client.set(GRIPPER_MODE_KEY, "o");
				redis_client.setDouble(GRIPPER_DESIRED_WIDTH_KEY, 0.08);
				gripper_is_open = true;
			}
		} 

		// orientation control input switch 
		if (key_pressed.at(GLFW_KEY_Q) && !key_was_pressed.at(GLFW_KEY_Q)) {
			switch_orientation_control_flag = true;
		}

		key_was_pressed = key_pressed;

		VectorXd robot_q = VectorXd::Zero(9);
		robot_q.head(7) = redis_client.getEigen(JOINT_ANGLES_KEY);
		double gripper_width = redis_client.getDouble(GRIPPER_CURRENT_WIDTH_KEY);		
		robot_q(7) = gripper_width / 2;
		robot_q(8) = - gripper_width / 2;
		graphics->updateRobotGraphics("panda_proxy", robot_q);

		Vector3d proxy_pos = redis_client.getEigen(PROXY_POS_KEY);
		Matrix3d proxy_ori = redis_client.getEigen(PROXY_ORI_KEY);
		Affine3d proxy_transform;
		proxy_transform.translation() = proxy_pos;
		proxy_transform.linear() = proxy_ori;
		graphics->updateObjectGraphics("proxy_end_effector", proxy_transform);

		graphics->renderGraphicsWorld();
	}

	runloop = false;
}

//------------------------------------------------------------------------------
/**
 * @brief Computes the payload mass and center of mass 
*/
void payloadDetection() {
	// assume that gripper if closed (but non-zero), and object is detected by vision

	// apply vision-based mass compensation 

	// recursively determine mass and center of mass from motion 
}

//------------------------------------------------------------------------------
/**
 * @brief Linear interpolation
 */
Vector3d linearInterp(const Vector3d& start, 
					  const Vector3d& end, 
					  const double curr_time,
					  const double start_time,
					  const double end_time) {
	return start + (end - start) * std::clamp((curr_time - start_time) / (end_time - start_time), 0., 1.);
}

//------------------------------------------------------------------------------
/**
 * @brief Rotation matrix interpolation 
 */
Matrix3d slerp(const Matrix3d& start,
			   const Matrix3d& end, 
			   const double curr_time,
			   const double start_time,
			   const double end_time) {
    Quaterniond q1(start), q2(end);

    // Ensure shortest path
    if (q1.dot(q2) < 0)
        q2.coeffs() = -q2.coeffs();

	double alpha = std::pow(std::clamp((curr_time - start_time) / (end_time - start_time), 0., 1.), 1);
	
	// std::cout << "SLERP BLENDING COEFFICIENT: " << alpha << "\n";

    Quaterniond qInterpolated = q1.slerp(alpha, q2);
    // Quaterniond qInterpolated = q2.slerp(alpha, q1);

    return qInterpolated.toRotationMatrix();
}

// Matrix3d slerp(const Matrix3d& start,
//                const Matrix3d& end, 
//                const double curr_time,
//                const double start_time,
//                const double end_time) {
//     Quaterniond q1(start), q2(end);

//     // Ensure shortest path
//     if (q1.dot(q2) < 0)
//         q2.coeffs() = -q2.coeffs();

//     // Calculate normalized time
//     double t = std::clamp((curr_time - start_time) / (end_time - start_time), 0.0, 1.0);

//     // Apply sinusoidal easing for even smoother interpolation
//     double tSmooth = 0.5 - 0.5 * std::cos(t * M_PI);

//     Quaterniond qInterpolated = q1.slerp(tSmooth, q2);

//     return qInterpolated.toRotationMatrix();
// }

//------------------------------------------------------------------------------
/*
 * @brief Clears queue 
*/
template<typename T>
void clearQueue(std::queue<T>& q) {
	while (!q.empty()) {
		q.pop();
	}
}

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
/**
 * @brief Prints queue elements
 */
template<typename T>
void printQueue(const std::queue<T>& q) {   
	// Create a copy of the queue since we'll be modifying it
	std::queue<T> tempQueue = q;

	// Print the contents of the queue
	std::cout << "Queue: ";
	while (!tempQueue.empty()) {
		std::cout << tempQueue.front() << " ";
		tempQueue.pop();
	}
	std::cout << std::endl;
}

//------------------------------------------------------------------------------
// Function to re-orthogonalize a rotation matrix
Eigen::Matrix3d reOrthogonalizeRotationMatrix(const Eigen::Matrix3d& mat) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d correctedMat = svd.matrixU() * svd.matrixV().transpose();
    return correctedMat;
}

//------------------------------------------------------------------------------
// Function to compute rotation matrix with axis-angle convention 
Eigen::Matrix3d computeRotationMatrix(double angle, const Eigen::Vector3d& axis) {
    // Normalize the axis
    Eigen::Vector3d normalizedAxis = axis.normalized();

    // Compute the skew-symmetric matrix K
    Eigen::Matrix3d K;
    K << 0, -normalizedAxis.z(), normalizedAxis.y(),
         normalizedAxis.z(), 0, -normalizedAxis.x(),
        -normalizedAxis.y(), normalizedAxis.x(), 0;

    // Compute the rotation matrix using Rodrigues' formula
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity()
                        + std::sin(angle) * K
                        + (1 - std::cos(angle)) * (K * K);

    return R;
}

//------------------------------------------------------------------------------
Eigen::Matrix3d minimumRotationMatrix(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
    // Normalize the input vectors
    Eigen::Vector3d u_norm = u.normalized();
    Eigen::Vector3d v_norm = v.normalized();

    // Compute the cross product and dot product
    Eigen::Vector3d k = u_norm.cross(v_norm);
    double cosTheta = u_norm.dot(v_norm);

    // Handle edge cases
    if (cosTheta > 0.9999) {
        // u and v are almost identical
        return Eigen::Matrix3d::Identity();
    } else if (cosTheta < -0.9999) {
        // u and v are opposite
        // Find an arbitrary orthogonal vector
        Eigen::Vector3d orthogonal = (u_norm.x() != 0 || u_norm.y() != 0)
                                         ? Eigen::Vector3d(-u_norm.y(), u_norm.x(), 0).normalized()
                                         : Eigen::Vector3d(0, -u_norm.z(), u_norm.y()).normalized();
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(M_PI, orthogonal).toRotationMatrix();
        return R;
    }

    // Compute the skew-symmetric matrix K
    Eigen::Matrix3d K;
    K << 0, -k.z(), k.y(),
         k.z(), 0, -k.x(),
        -k.y(), k.x(), 0;

    // Compute the rotation matrix using Rodrigues' formula
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + K + K * K * ((1 - cosTheta) / (1 - cosTheta * cosTheta));

    return R;
}

//------------------------------------------------------------------------------
double angleBetweenVectors(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
    // Compute the dot product
    double dotProduct = u.dot(v);

    // Compute the magnitudes of the vectors
    double magnitudeU = u.norm();
    double magnitudeV = v.norm();

    // Compute the cosine of the angle
    double cosTheta = dotProduct / (magnitudeU * magnitudeV);

    // Clamp the cosine value to the range [-1, 1] to handle numerical inaccuracies
    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));

    // Return the angle in radians
    return std::acos(cosTheta);
}

//------------------------------------------------------------------------------
Eigen::Vector3d axisOfRotation(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
    // Compute the cross product
    Eigen::Vector3d crossProduct = u.cross(v);

    // Check if the vectors are parallel (or zero-length)
    if (crossProduct.norm() < 1e-6) {
        throw std::runtime_error("Vectors are collinear or one of them is zero. No unique axis of rotation.");
    }

    // Normalize the cross product to get the axis
    return crossProduct.normalized();
}

//------------------------------------------------------------------------------
Eigen::Vector3d projectOntoPlane(const Eigen::Vector3d& v, const Eigen::Vector3d& n) {
    // Normalize the normal vector
    Eigen::Vector3d nNorm = n.normalized();

    // Compute the projection
    Eigen::Vector3d vProj = v - (v.dot(nNorm)) * nNorm;

    return vProj;
}