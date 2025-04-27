# Heterogeneous Computing for Real-Time Point Cloud Processing

## Overview
This project leverages heterogeneous computing systems (CPU+GPU) to optimize real-time processing of LiDAR point cloud data. The implementation focuses on efficient workload distribution across hardware platforms to ensure maximum performance and responsiveness for computationally intensive operations such as data pre-processing, feature extraction, clustering, segmentation, and object detection.

## Motivation
Real-time processing of LiDAR data is crucial for:
- Autonomous vehicles
- Robotic navigation systems
- Smart-city infrastructures
- Disaster response operations

These applications demand high computational power with low latency, making heterogeneous computing architectures ideal for handling massive point cloud datasets promptly and efficiently.

## Key Objectives
- Implement point cloud processing algorithms optimized for heterogeneous computing environments
- Develop effective workload distribution strategies across CPUs, GPUs, and accelerators
- Integrate frameworks like OpenCL and CUDA to facilitate seamless collaboration among computing resources
- Benchmark system performance (throughput, latency, scalability, and resource utilization)
- Use profiling tools to identify and resolve performance bottlenecks
- Validate the system using standard LiDAR datasets (KITTI, NuScenes)

## Getting Started

### Prerequisites
- **CMake** ≥ 3.24
- **C++ Compiler** supporting C++17
- **GCC/G++** 11.4.0
- **CUDA Toolkit** ≥ 12.4
- **OpenMP** 4.5+
- **PCL (Point Cloud Library)** ≥ 1.8
- **Eigen3** (recent version)

### Building the Project
```bash
mkdir build
cd build
cmake ..
make
```

### Running the Application
```bash
cd build
./voxel_grid_type --input path/to/pointcloud.pcd --output downsampled.pcd --voxel-size 0.1
```

## Features
- Voxel-based point cloud downsampling
- Hardware-accelerated point cloud processing
- Dynamic workload distribution between CPU and GPU

## Contributors
- BUSE NUR İLERİ
- MAHMUT ESAT ÖZHÖLÇEK
- AZİZ ÖNDER