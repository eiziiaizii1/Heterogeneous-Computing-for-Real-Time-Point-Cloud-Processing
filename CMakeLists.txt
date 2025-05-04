cmake_minimum_required(VERSION 3.24)
project(VoxelGrid_Downsampling)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -march=native")

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



# CUDA version
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA found. Enabling CUDA support.")
    enable_language(CUDA)
    
    # Set CUDA architectures (adjust for your GPU)
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures")  # For Turing GPUs
    
    # Add CUDA executable with .cu extension
    add_executable(voxel_grid_cuda voxel_grid_cuda.cu)
    
    set_target_properties(voxel_grid_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    target_include_directories(voxel_grid_cuda PRIVATE
        ${CUDA_INCLUDE_DIRS}
    )
    
    target_link_libraries(voxel_grid_cuda
        ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
    )
else()
    message(WARNING "CUDA not found. CUDA target will not be built.")
endif()