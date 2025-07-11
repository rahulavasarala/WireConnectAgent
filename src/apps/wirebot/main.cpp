// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstring>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include "redis/RedisClient.h"
#include <iostream>
#include <string>

#include <sw/redis++/redis++.h>
#include<redis/RedisClient.h>
using namespace sw::redis;

#include "SaiModel.h"
#include "SaiPrimitives.h"
#include "timer/LoopTimer.h"
#include "redis/RedisClient.h"
#include "redis_keys.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

//redis client
SaiCommon::RedisClient redis_client;

// globals
const string mujoco_file = std::string(URDF_PATH) + "/scenes/rizon4sgripper.xml";
const string robot_file = std::string(URDF_PATH) + "/scenes/rizon4sgripper.urdf";
const string robot_name = "rizon4s";
std::shared_ptr<SaiModel::SaiModel> robot;
std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task;
std::shared_ptr<SaiPrimitives::JointTask> joint_task;

const string forces = "rizon4s::sensed_forces";
const string torques = "rizon4s::sensed_torques";

const int ROBOT_GRIPPER_JOINTS = 13;

Vector3d START_POS = Vector3d(0.3, 0.3, 0.3);
Matrix3d START_ORIENTATION = (Matrix3d() << 
    1,  0,  0,
    0, -1,  0,
    0,  0, -1).finished();
int GRIPPER_START_POS = 0;

// global kinematics
Vector3d x_start;

// MuJoCo data structures
mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract camera
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
        // flag_start_simulation = false;
    } else if (act == GLFW_PRESS && key == GLFW_KEY_P) {
        // flag_start_simulation = true;
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos) {
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right) {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
    {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left)
    {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else
    {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

//Get the camera rotation matrix from the mjmodel
Matrix3d getCameraRotationMatrix(const mjModel* m, int cam_id) {
    mjtNum quat[4];
    mju_copy4(quat, m->cam_quat + 4 * cam_id);


    mjtNum mujoco_mat[9];
    mju_quat2Mat(mujoco_mat, quat);

    Matrix3d R;

    R << mujoco_mat[0], mujoco_mat[1], mujoco_mat[2],
         mujoco_mat[3], mujoco_mat[4], mujoco_mat[5],
         mujoco_mat[6], mujoco_mat[7], mujoco_mat[8];

    return R;
}

std::vector<float> sampleDepth(float * depth, int width, int height,  int stride) {
    std::vector<float> depth_flat;
    for(int i = 0; i < width*height; i += stride) {
        depth_flat.push_back(depth[i]);
    }

    return depth_flat;
}

std::vector<float> flattenPointCloud(std::vector<Vector3d> cloud) {

    std::vector<float> buffer;
    for(int i = 0; i < cloud.size(); i++) {
        buffer.push_back(cloud[i](0));
        buffer.push_back(cloud[i](1));
        buffer.push_back(cloud[i](2));
    }

    return buffer;
}

//Generate I think we should have a relative point cloud with respect to the camera
std::vector<Vector3d> generatePointCloud(const mjModel* m, const mjData* d, const float* depth_buffer, int width, int height, int xstride, int ystride, int cam_id, bool useLocal) {

    std::vector<Vector3d> cloud;

    mjtNum cam_pos[3];
    mju_copy3(cam_pos, m->cam_pos + 3 * cam_id);

    float near = m->vis.map.znear;
    float far = m->vis.map.zfar;

    float fov = m->cam_fovy[cam_id] * M_PI/180.0f;
    float fx = 0.5f * width/std::tan(fov/2.0f);
    float fy = 0.5f * height/std::tan(fov/2.0f);

    float cx = width/2.0f;
    float cy = height/2.0f;

    Matrix3d R = Matrix3d::Identity();
    
    if (!useLocal) {
        R = getCameraRotationMatrix(m, cam_id);
    }

    int count = 0;

    for(int y = 0; y < height; y += ystride) {
        for(int x = 0; x < width; x += xstride) {
            
            int idx = y * width + x;

            float z_buf = depth_buffer[idx];

            if(z_buf >= 1.0f) {
                continue;
            }

            float z = near * far / (far - (far - near)* z_buf);
            float x_cam = (x - cx) * z / fx;
            float y_cam = (y - cy)*z/fy;

            Vector3d p_cam(x_cam, -y_cam, -z);

            if (useLocal) {
                cloud.push_back(p_cam);
                continue;
            }

            Vector3d p_world = R * p_cam + Vector3d(cam_pos[0], cam_pos[1], cam_pos[2]);
            cloud.push_back(p_world);
        }
    }

    return cloud;
}

// controller callback
void controller_callback(const mjModel* m, mjData* d);
void updateRobotState(std::shared_ptr<SaiModel::SaiModel> robot, const mjModel* m, const mjData* d);\

void init_redis() {
    redis_client.setEigen(DESIRED_CARTESIAN_POSITION, START_POS);
    redis_client.setEigen(DESIRED_CARTESIAN_ORIENTATION, START_ORIENTATION);
    redis_client.setInt(DESIRED_GRIPPER_POSITION, GRIPPER_START_POS);
}

// main function
int main(int argc, char* argv[])
{

    //connect to redis
    redis_client.connect();

    // debug info
    std::cout << "Mujoco xml: " << mujoco_file << "\n";

    char error[1000] = "Could not load binary model";
    m = mj_loadXML(mujoco_file.c_str(), 0, error, 1000);

    // make data
    d = mj_makeData(m);

    // Disable joint limits by setting the range to very large values
    for (int i = 0; i < m->njnt; ++i) {
        m->jnt_range[2 * i] = -1e10;     // Lower limit
        m->jnt_range[2 * i + 1] = 1e10; // Upper limit
    }

    std::cout << "Joint limits disabled!" << std::endl;

    // create robot and controller
    robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
    std::cout << "Robot DOF: " << robot->dof() << "\n";
    std::cout << "MJ DOF: " << m->nq << "\n"; 
    std::string control_link = "link7";
    Vector3d control_point = Vector3d(0, 0, 0.224);
    Affine3d control_frame = Affine3d::Identity();
    control_frame.translation() = control_point;
    motion_force_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link, control_frame);
    motion_force_task->disableInternalOtg();
    joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);

    // set initial state 
    VectorXd q_init = robot->q();
    q_init.head(13) << 0, -90, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    q_init.head(13) *= M_PI / 180;
    std::cout << "Initial joint configuration: " << q_init.transpose() << "\n";
    for (int i = 0; i < 7; ++i) {
        d->qpos[i] = q_init(i);
        d->qvel[i] = 0;
    }

    updateRobotState(robot, m, d);
    motion_force_task->reInitializeTask();
    joint_task->reInitializeTask();
    x_start = robot->positionInWorld(control_link, control_point);
    std::cout << "x start: " << x_start.transpose() << std::endl;

    init_redis();

    // Set the control callback (global)
    mjcb_control = controller_callback;

    // init GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    //create the camera buffer size:
    int width = 1200;
    int height = 900;

    // float* depth = new float[width* height];

    // int cam_id = mj_name2id(m, mjOBJ_CAMERA, "joint7_cam");
    // cam.type = mjCAMERA_FIXED;
    // cam.fixedcamid = cam_id;

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window))
    {
        mjtNum simstart = d->time;
        int cnt = 0;

        // t1 = high_resolution_clock::now();
        // if (flag_start_simulation) {
        while (d->time - simstart < 1.0 / 60.0)
        {
            cnt++;
            mj_step(m, d);  // simulate until 1/60 seconds goes by for visual rendering (this calls the controller at sim rate)
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        //Point Cloud Code ---------------------------------------------------------
        // mjr_readPixels(nullptr, depth, mjrRect{280, 210, width, height}, &con);

        // std::vector<Vector3d> cloud = generatePointCloud(m, d, depth, width, height, 10, 10, cam_id, true);
        // std::vector<float> buffer = flattenPointCloud(cloud);

        // std::cout << "Buffer size is: " << cloud.size() << std::endl;

        // std::string blob(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));

        // redis.set("point_cloud", blob);
        //Point Cloud Code ----------------------------------------------------------

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);
        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return 1;
}

void update_redis(std::shared_ptr<SaiModel::SaiModel> robot) {
    Vector3d currentPosition = motion_force_task->getCurrentPosition();
    Matrix3d currentOrientation = motion_force_task->getCurrentOrientation();

    redis_client.setEigen(CURRENT_CARTESIAN_POSITION, currentPosition);
    redis_client.setEigen(CURRENT_CARTESIAN_ORIENTATION, currentOrientation);
}

// ---------------------------------------
void updateRobotState(std::shared_ptr<SaiModel::SaiModel> robot, const mjModel* m, const mjData* d) {

    VectorXd robot_q(ROBOT_GRIPPER_JOINTS), robot_dq(ROBOT_GRIPPER_JOINTS);

    for (int i = 0; i < ROBOT_GRIPPER_JOINTS; ++i) {
        robot_q(i) = d->qpos[i];
    }

    for (int i = 0; i < ROBOT_GRIPPER_JOINTS; ++i) {
        robot_dq(i) = d->qvel[i];
    }

    // set and update robot
    robot->setQ(robot_q);
    robot->setDq(robot_dq);
    robot->updateModel();
}

// ---------------------------------------
// need to add safety checks to see whether the input data from redis is good
void controller_callback(const mjModel* m, mjData* d) {

    // update robot state
    updateRobotState(robot, m, d);

    //update redis --------------------
    update_redis(robot);
    //update redis ------------------------

    //have code to get the goal position and orientation of the robot
    MatrixXd g_p = redis_client.getEigen(DESIRED_CARTESIAN_POSITION);
    Vector3d goal_position = g_p.col(0).template head<3>();
    Matrix3d goal_orientation = redis_client.getEigen(DESIRED_CARTESIAN_ORIENTATION).topLeftCorner<3,3>();
    int goal_gripper_pos = redis_client.getInt(DESIRED_GRIPPER_POSITION);

    // get time 
    double time = d->time;
    double t_wait = 2;  // wait time for passive compensation -> active compensation 

    // compute control torque
	VectorXd control_torques = VectorXd::Zero(robot->dof());

    // if (time - t_wait > total_duration) {
    //     throw runtime_error("Simulation finished\n");
    // }

    if (time > t_wait) {
        // set goals 
        double freq = 1;
        double t_elapsed = time - t_wait;
        motion_force_task->setGoalPosition(goal_position);
        motion_force_task->setGoalOrientation(goal_orientation);

        // compute torques 
        motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
        joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
        control_torques = motion_force_task->computeTorques() + joint_task->computeTorques() + robot->jointGravityVector();
    } else {
        control_torques = robot->jointGravityVector();
    }

    // set torques 
    for (int i = 0; i < ROBOT_GRIPPER_JOINTS; ++i) {
        d->ctrl[i] = control_torques(i);  // set actuated joint torques 
    }

    d->ctrl[7] = goal_gripper_pos; // how to control the gripper
}

