# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MUJOCO_SAI_CONTROL_BINARY_DIR}/robot_control/00-calibration)
ADD_EXECUTABLE (calibration calibration.cpp ${MUJOCO_SAI_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_calibration simviz.cpp ${MUJOCO_SAI_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (calibration ${MUJOCO_SAI_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_calibration ${MUJOCO_SAI_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY urdf/panda_arm.urdf urdf/world.urdf DESTINATION ${APP_RESOURCE_DIR})