/**
 * @file redis_keys.h
*/

#pragma once 
#include <string>

std::string ROBOT_NAME = "4s-Oberon";

// robot-related keys 
std::string JOINT_ANGLES_KEY = "sai::sensors::" + ROBOT_NAME + "::sensors::q";
std::string JOINT_VELOCITIES_KEY = "sai::sensors::" + ROBOT_NAME + "::Romeo::sensors::dq";
std::string JOINT_TORQUES_COMMANDED_KEY = "sai::comands::" + ROBOT_NAME + "::control_torques";
std::string FORCE_SENSOR_KEY = "sai::sensors::" + ROBOT_NAME + "::ft_sensor::force";
std::string MOMENT_SENSOR_KEY = "sai::sensors::" + ROBOT_NAME + "::ft_sensor::moment";
std::string MASS_MATRIX_KEY = "sai::sensors::" + ROBOT_NAME + "model::mass_matrix";
std::string CORIOLIS_KEY = "sai::sensors::" + ROBOT_NAME + "model::coriolis";
std::string ROBOT_RUNNING_KEY = "";

