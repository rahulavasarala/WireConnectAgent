cmake_minimum_required (VERSION 3.5)
project (WireConnectingAgent)

SET(CMAKE_BUILD_TYPE "Debug")  
# SET(CMAKE_BUILD_TYPE "Release")  

# set common dependencies
# - eigen3
find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})

# include Chai3d 
# -- this is required by Chai3d to pick the correct path for OpenGL headers
FIND_PACKAGE(CHAI3D REQUIRED)
ADD_DEFINITIONS(${CHAI3D_DEFINITIONS})
if (CMAKE_SYSTEM_NAME MATCHES Darwin) 
  ADD_DEFINITIONS(-D_MACOSX -DMACOSX)
elseif (CMAKE_SYSTEM_NAME MATCHES Linux)
  ADD_DEFINITIONS(-D_LINUX -DLINUX)
endif () #NOTE: Windows is currently unsupported
LINK_DIRECTORIES(${CHAI3D_LIBRARY_DIRS})
INCLUDE_DIRECTORIES(${CHAI3D_INCLUDE_DIRS})

# - SAI2-SIMULATION
set(SAI-SIMULATION_DIR "../core/sai-simulation/build")
find_package(SAI-SIMULATION REQUIRED)
include_directories(${SAI-SIMULATION_INCLUDE_DIRS})
add_definitions(${SAI-SIMULATION_DEFINITIONS})

# - SAI2-GRAPHICS
set(SAI-GRAPHICS_DIR "../core/sai-graphics/build")
FIND_PACKAGE(SAI-GRAPHICS REQUIRED)
include_directories(${SAI-GRAPHICS_INCLUDE_DIRS})
add_definitions(${SAI-GRAPHICS_DEFINITIONS})

# - SAI2-COMMON
set(SAI-COMMON_DIR "../core/sai-common/build")
FIND_PACKAGE(SAI-COMMON REQUIRED)
include_directories(${SAI-COMMON_INCLUDE_DIRS})

# - SAI2-URDF
set(SAI-URDF_DIR "../core/sai-urdfreader/build")
FIND_PACKAGE(SAI-URDF REQUIRED)
include_directories(${SAI-URDF_INCLUDE_DIRS})

# - SAI2-MODEL
set(SAI-MODEL_DIR "../core/sai-model/build")
find_package(SAI-MODEL REQUIRED)
include_directories(${SAI-MODEL_INCLUDE_DIRS})
add_definitions(${SAI-MODEL_DEFINITIONS})

# - SAI2-PRIMITIVES
set(SAI-PRIMITIVES_DIR "../core/sai-primitives/build")
find_package(SAI-PRIMITIVES REQUIRED)
include_directories(${SAI-PRIMITIVES_INCLUDE_DIRS})
add_definitions(${SAI-PRIMITIVES_DEFINITIONS})

# - hiredis
find_library(HIREDIS_LIBRARY hiredis)

# - redis plus plus
find_library(REDIS_PLUS_PLUS_LIBRARY redis++)

# - json
find_package(nlohmann_json 3.11.3 REQUIRED)

# - mujoco
find_package(mujoco REQUIRED)

# - zmq
find_library(ZeroMQ_LIBRARY
    NAMES zmq
    PATHS /usr/local/lib /opt/homebrew/lib
    REQUIRED
)

# === Find cppzmq (header-only C++ binding) ===
find_path(ZeroMQ_INCLUDE_DIR
    NAMES zmq.hpp
    PATHS /usr/local/include /opt/homebrew/include
    REQUIRED
)

include_directories(${ZeroMQ_INCLUDE_DIR})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (${CMAKE_SYSTEM_NAME} MATCHES Linux)
	add_compile_options(-I/usr/include -I/usr/local/include -O3)
elseif (${CMAKE_SYSTEM_NAME} MATCHES Darwin)
	add_compile_options(-I/usr/include -I/usr/local/include -I/opt/homebrew/include -w -O3)
endif()

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# set reference folders
add_definitions(-DURDF_PATH=\"${PROJECT_SOURCE_DIR}/models\")

# define libraries for linking 
set(MUJOCO_SAI_COMMON_LIBRARIES
	${SAI-MODEL_LIBRARIES}
	${SAI-GRAPHICS_LIBRARIES}
	${SAI-SIMULATION_LIBRARIES}
	${SAI-COMMON_LIBRARIES}
	${SAI-URDF_LIBRARIES}
	${CHAI3D_LIBRARIES}
	${HIREDIS_LIBRARY}
	${REDIS_PLUS_PLUS_LIBRARY}
	${SAI-PRIMITIVES_LIBRARIES}
	mujoco::mujoco
	${ZeroMQ_LIBRARY}
	)

set (MUJOCO_SAI_CONTROL_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(src/apps)