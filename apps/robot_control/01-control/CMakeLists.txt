# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MUJOCO_SAI_CONTROL_BINARY_DIR}/robot_control/01-control)
ADD_EXECUTABLE (controller_grasping controller.cpp ${MUJOCO_SAI_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (controller_grasping ${MUJOCO_SAI_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY urdf/proxy_world.urdf urdf/blank_world.urdf urdf/world.urdf urdf/panda_arm_hand.urdf urdf/panda_arm.urdf urdf/panda_arm_proxy.urdf controller_settings.yaml DESTINATION ${APP_RESOURCE_DIR})