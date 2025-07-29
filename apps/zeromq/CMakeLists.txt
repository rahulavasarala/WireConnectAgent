# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MUJOCO_SAI_CONTROL_BINARY_DIR}/zeromq)
# ADD_EXECUTABLE (joint_torque_server joint_torque_server.cpp ${MUJOCO_SAI_COMMON_SOURCE})
# ADD_EXECUTABLE (motion_force_torque_server motion_force_torque_server.cpp ${MUJOCO_SAI_COMMON_SOURCE})
ADD_EXECUTABLE (particle_filter_server particle_filter_server.cpp ${MUJOCO_SAI_COMMON_SOURCE})

# and link the library against the executable
# TARGET_LINK_LIBRARIES (joint_torque_server ${MUJOCO_SAI_COMMON_LIBRARIES})
# TARGET_LINK_LIBRARIES (motion_force_torque_server ${MUJOCO_SAI_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (particle_filter_server ${MUJOCO_SAI_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY pfilter_settings.yaml DESTINATION ${APP_RESOURCE_DIR})