cmake_minimum_required (VERSION 3.5)
project (WireConnectingAgent)

SET(CMAKE_BUILD_TYPE "Debug")  
# SET(CMAKE_BUILD_TYPE "Release")  

option(USE_SAI2 "Use SAI2-* libraries instead of SAI-* libraries" OFF)

if(USE_SAI2)
    message(STATUS "Using SAI2 libraries")
    message(STATUS "DEBUG: Current source directory is: ${CMAKE_SOURCE_DIR}")
    include(${CMAKE_SOURCE_DIR}/cmake/FindSAI2.cmake)
else()
    message(STATUS "Using SAI libraries")
    message(STATUS "DEBUG: Current source directory is: ${CMAKE_SOURCE_DIR}")
    include(${CMAKE_SOURCE_DIR}/cmake/FindSAI.cmake)
endif()

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

# - hiredis
find_library(HIREDIS_LIBRARY hiredis REQUIRED)

# - redis plus plus
find_library(REDIS_PLUS_PLUS_LIBRARY redis++ REQUIRED)

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

# - yamlcpp
find_package(yaml-cpp REQUIRED)

# === Find cppzmq (header-only C++ binding) ===
find_path(ZeroMQ_INCLUDE_DIR
    NAMES zmq.h
    PATHS /usr/local/include /opt/homebrew/include 
    REQUIRED
)

include_directories(${ZeroMQ_INCLUDE_DIR})

# custom library 
add_library(wire-connect-agent STATIC ${PROJECT_SOURCE_DIR}/src/ForceSpaceParticleFilter.cpp)

# set reference folders
add_definitions(-DURDF_PATH=\"${PROJECT_SOURCE_DIR}/models\")

# add more libraries 
list(APPEND MUJOCO_SAI_COMMON_LIBRARIES
		${CHAI3D_LIBRARIES}
		${HIREDIS_LIBRARY}
		${REDIS_PLUS_PLUS_LIBRARY}
		mujoco::mujoco
		${ZeroMQ_LIBRARY}
    yaml-cpp::yaml-cpp
		wire-connect-agent 
	)

set (MUJOCO_SAI_CONTROL_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(apps)
# add_subdirectory(test)