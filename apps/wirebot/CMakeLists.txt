set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MUJOCO_SAI_CONTROL_BINARY_DIR}/wirebot)
add_executable(wirebot main.cpp)
target_link_libraries(wirebot ${MUJOCO_SAI_COMMON_LIBRARIES})