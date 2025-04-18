cmake_minimum_required(VERSION 3.24)
project(VoxelGrid_Downsampling)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# Include directories
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Add definitions
add_definitions(${PCL_DEFINITIONS})

# Create executable
add_executable(voxel_downsample voxel_downsample.cpp)

# Link libraries
target_link_libraries(voxel_downsample
    ${PCL_LIBRARIES}
)

# Add OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(voxel_downsample OpenMP::OpenMP_CXX)
endif()
