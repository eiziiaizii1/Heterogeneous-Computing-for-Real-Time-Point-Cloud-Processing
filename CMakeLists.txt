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

# Create serial version
add_executable(voxel_grid_serial voxel_grid_serial.cpp)
target_link_libraries(voxel_grid_serial ${PCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(voxel_grid_serial OpenMP::OpenMP_CXX)
endif()


# Create OpenMP version
add_executable(voxel_grid_omp voxel_grid_omp.cpp)
target_link_libraries(voxel_grid_omp ${PCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(voxel_grid_omp OpenMP::OpenMP_CXX)
endif()
