/**
 * @file redis_keys.h
 * @brief Contains all redis keys for simulation and control
 * 
 */

#pragma once

#include <string>

std::string ROBOT_NAME = "4s-Oberon";

// robot-related keys 
std::string JOINT_ANGLES_KEY = "sai::sensors::" + ROBOT_NAME + "::joint_positions";
std::string JOINT_VELOCITIES_KEY = "sai::sensors::" + ROBOT_NAME + "::joint_velocities";
std::string JOINT_TORQUES_COMMANDED_KEY = "sai::commands::" + ROBOT_NAME + "::control_torques";
std::string FORCE_SENSOR_KEY = "sai::sensors::" + ROBOT_NAME + "::ft_sensor::tcp_force";
std::string MOMENT_SENSOR_KEY = "sai::sensors::" + ROBOT_NAME + "::ft_sensor::tcp_moment";
std::string MASS_MATRIX_KEY = "sai::sensors::" + ROBOT_NAME + "::model::mass_matrix";
std::string CORIOLIS_KEY = "sai::sensors::" + ROBOT_NAME + "::model::coriolis";

// gripper-related keys 
std::string GRIPPER_MODE_KEY = "sai::" + ROBOT_NAME + "::gripper::mode";
std::string GRIPPER_DESIRED_WIDTH_KEY = "sai::" + ROBOT_NAME + "::gripper::desired_width";
std::string GRIPPER_CURRENT_WIDTH_KEY = "sai::sensors::" + ROBOT_NAME + "::gripper::width";
std::string GRIPPER_GRASP_SUCCESSFUL_KEY = "sai::" + ROBOT_NAME + "::gripper::grasp_successful";
std::string GRIPPER_MOVE_SUCCESSFUL_KEY = "sai::" + ROBOT_NAME + "::gripper::move_successful";

// application-related keys 
std::string ROBOT_RUNNING_KEY = "sai2::FrankaPanda::Romeo::RobotDriver::running";
std::string CONTROL_POS_KEY = "sai2::FrankaPanda::Romeo::control::position";
std::string CONTROL_ORI_KEY = "sai2::FrankaPanda::Romeo::control::orientation";
std::string EE_POS_ORIGINAL_KEY = "sai2::FrankaPanda::Romeo::ee_pos_original";
std::string EE_POS_KEY = "sai2::FrankaPanda::Romeo::ee_pos";
std::string EE_ORI_KEY = "sai2::FrankaPanda::Romeo::ee_ori";
std::string EE_LIN_VEL_ORIGINAL_KEY = "sai2::FrankaPanda::Romeo::ee_lin_vel_original";
std::string EE_LIN_VEL_KEY = "sai2::FrankaPanda::Romeo::ee_lin_vel";
std::string EE_ANG_VEL_KEY = "sai2::FrankaPanda::Romeo::ee_ang_vel";
std::string HAPTIC_CLUTCH_KEY = "sai2::Haptic::clutch";

// camera-related keys 
std::string CAMERA_POS_KEY = "sai2::FrankaPanda::Romeo::camera_pos";
std::string CAMERA_ORI_KEY = "sai2::FrankaPanda::Romeo::camera_ori";

// environment-related keys 
std::string ENVIRONMENT_NORMAL_VECTOR_KEY = "sai2::FrankaPanda::Romeo::environment_normal";
std::string RESET_VOXEL_KEY = "sai2::FrankaPanda::Romeo::reset_voxel";

// normal map (image) key 
std::string NORMAL_MAP_KEY = "sai2::FrankaPanda::Romeo::normal_map";

// diffusion-related keys 
std::string DIFFUSION_PROXY_POS_KEY = "diffusion::position";
std::string DIFFUSION_PROXY_ORI_KEY = "diffusion::orientation";
std::string DIFFUSION_GRIPPER_POS_KEY = "diffusion::gripper";
std::string DIFFUSION_RESET_KEY = "diffusion::reset_action_queue";
std::string DIFFUSION_START_KEY = "diffusion::start";
std::string DIFFUSION_FORCE_OR_MOTION_AXIS_KEY = "diffusion::force_or_motion_axis";
std::string DIFFUSION_FORCE_SPACE_DIMENSION_KEY = "diffusion::force_space_dimension";
std::string DIFFUSION_DESIRED_FORCE_KEY = "diffusion::desired_force";
std::string CONTROLLER_START_KEY = "controller::start";

// mouse-related keys 
const std::string MOUSE_POS = "pymouse::pos";
const std::string MOUSE_ORI = "pymouse::ori";
const std::string RESET_KEY = "sai2::FrankaPanda::reset";  // controller reset
const std::string LOGGER_START_KEY = "sai2::Logger::start";  // logger start
const std::string LOGGER_STOP_KEY = "sai2::Logger::stop";  // logger stop
const std::string GRIPPER_OPEN_KEY = "sai2::gripper::open";  // gripper open
const std::string GRIPPER_CLOSE_KEY = "sai2::gripper::close";  // gripper close 

// proxy logging  
std::string PROXY_JOINT_ANGLES_KEY = "sai2::FrankaPanda::Proxy::sensors::q";
std::string PROXY_POS_KEY = "sai2::FrankaPanda::Proxy::proxy_pos";
std::string PROXY_ORI_KEY = "sai2::FrankaPanda::Proxy::proxy_ori";
std::string PROXY_GHOST_POS_KEY = "sai2::FrankaPanda::Proxy::proxy_pos_ghost";
std::string MOMENT_CONTROL_KEY = "sai2::FrankaPanda::Proxy::moment_control_mode";

// data logging
std::string STATIC_CAMERA_FRAME_KEY = "sai2::perception::static_camera_frame";
std::string MOVING_CAMERA_FRAME_KEY = "sai2::perception::moving_camera_frame";
// std::string SIM_RESET_KEY = "sai2::FrankaPanda::simulation::reset";
std::string SIM_RESET_KEY = "sai2::FrankaPanda::simulation::reset";
std::string CONTROLLER_RESET_KEY = "sai2::FrankaPanda::controller::reset";
std::string SIM_START_KEY = "sai2::FrankaPanda::start";
// std::string LOGGER_START_KEY = "sai2::Logger::start";
// std::string LOGGER_STOP_KEY = "sai2::Logger::stop";
std::string FORCE_SPACE_DIMENSION_KEY = "sai2::FrankaPanda::Proxy::force_space_dimension";
std::string FORCE_OR_MOTION_AXIS_KEY = "sai2::FrankaPanda::Proxy::force_or_motion_axis";
std::string ZERO_MOMENT_CONTROL_KEY = "sai2::FrankaPanda::Romeo::zero_moment_control";
std::string GRASPED_OBJECT_KEY = "sai2::FrankaPanda::gripper::grasped_object";

// const std::string ROBOT_STATE = "robot_state";
// const std::string FRAME_POS = "pymouse::frame_pos";
// const std::string FRAME_ORI = "pymouse::frame_ori";
// const std::string ROT_AXIS_POINT = "pymouse::rot_axis_point";
// const std::string ROT_AXIS = "pymouse::rot_axis";

// debug
std::string CONTROL_FRAME_POS_KEY = "sai2::FrankaPanda::control_frame_pos";
std::string CONTROL_FRAME_ORI_KEY = "sai2::FrankaPanda::control_frame_ori";

// state-related keys 
std::string IS_IN_CONTACT_KEY = "sai2::FrankaPanda::is_in_contact_flag";
std::string IS_HOLDING_OBJECT_KEY = "sai2::FrankaPanda::is_holding_object_flag";