# SAI2-SIMULATION
set(SAI2-SIMULATION_DIR "../core/sai2-simulation/build")
find_package(SAI2-SIMULATION REQUIRED)
include_directories(${SAI2-SIMULATION_INCLUDE_DIRS})
add_definitions(${SAI2-SIMULATION_DEFINITIONS})

# SAI2-GRAPHICS
set(SAI2-GRAPHICS_DIR "../core/sai2-graphics/build")
find_package(SAI2-GRAPHICS REQUIRED)
include_directories(${SAI2-GRAPHICS_INCLUDE_DIRS})
add_definitions(${SAI2-GRAPHICS_DEFINITIONS})

# SAI2-COMMON
set(SAI2-COMMON_DIR "../core/sai2-common/build")
find_package(SAI2-COMMON REQUIRED)
include_directories(${SAI2-COMMON_INCLUDE_DIRS})

# SAI2-URDF
set(SAI2-URDF_DIR "../core/sai2-urdfreader/build")
find_package(SAI2-URDF REQUIRED)
include_directories(${SAI2-URDF_INCLUDE_DIRS})

# SAI2-MODEL
set(SAI2-MODEL_DIR "../core/sai2-model/build")
find_package(SAI2-MODEL REQUIRED)
include_directories(${SAI2-MODEL_INCLUDE_DIRS})
add_definitions(${SAI2-MODEL_DEFINITIONS})

# SAI2-PRIMITIVES
set(SAI2-PRIMITIVES_DIR "../core/sai2-primitives/build")
find_package(SAI2-PRIMITIVES REQUIRED)
include_directories(${SAI2-PRIMITIVES_INCLUDE_DIRS})
add_definitions(${SAI2-PRIMITIVES_DEFINITIONS})

# Link libraries
set(MUJOCO_SAI_COMMON_LIBRARIES
    ${SAI2-MODEL_LIBRARIES}
    ${SAI2-GRAPHICS_LIBRARIES}
    ${SAI2-SIMULATION_LIBRARIES}
    ${SAI2-COMMON_LIBRARIES}
    ${SAI2-URDF_LIBRARIES}
    ${SAI2-PRIMITIVES_LIBRARIES}
)