# SAI-SIMULATION
set(SAI-SIMULATION_DIR "../core/sai-simulation/build")
find_package(SAI-SIMULATION REQUIRED)
include_directories(${SAI-SIMULATION_INCLUDE_DIRS})
add_definitions(${SAI-SIMULATION_DEFINITIONS})

# SAI-GRAPHICS
set(SAI-GRAPHICS_DIR "../core/sai-graphics/build")
find_package(SAI-GRAPHICS REQUIRED)
include_directories(${SAI-GRAPHICS_INCLUDE_DIRS})
add_definitions(${SAI-GRAPHICS_DEFINITIONS})

# SAI-COMMON
set(SAI-COMMON_DIR "../core/sai-common/build")
find_package(SAI-COMMON REQUIRED)
include_directories(${SAI-COMMON_INCLUDE_DIRS})

# SAI-URDF
set(SAI-URDF_DIR "../core/sai-urdfreader/build")
find_package(SAI-URDF REQUIRED)
include_directories(${SAI-URDF_INCLUDE_DIRS})

# SAI-MODEL
set(SAI-MODEL_DIR "../core/sai-model/build")
find_package(SAI-MODEL REQUIRED)
include_directories(${SAI-MODEL_INCLUDE_DIRS})
add_definitions(${SAI-MODEL_DEFINITIONS})

# SAI-PRIMITIVES
set(SAI-PRIMITIVES_DIR "../core/sai-primitives/build")
find_package(SAI-PRIMITIVES REQUIRED)
include_directories(${SAI-PRIMITIVES_INCLUDE_DIRS})
add_definitions(${SAI-PRIMITIVES_DEFINITIONS})

# Link libraries
set(MUJOCO_SAI_COMMON_LIBRARIES
    ${SAI-MODEL_LIBRARIES}
    ${SAI-GRAPHICS_LIBRARIES}
    ${SAI-SIMULATION_LIBRARIES}
    ${SAI-COMMON_LIBRARIES}
    ${SAI-URDF_LIBRARIES}
    ${SAI-PRIMITIVES_LIBRARIES}
)