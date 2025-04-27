#include <iostream>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <Eigen/Core>
#include <algorithm> // std::sort için

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Hash combine fonksiyonu - Eksik olan bu fonksiyon eklendi
template <typename T>
__host__ __device__ 
size_t hash_combine(size_t seed, const T& v) {
    return seed ^ (std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

// CUDA'da std::hash yerine basit bir hash fonksiyonu
__device__ 
size_t hash_int(int val) {
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = (val >> 16) ^ val;
    return val;
}

// CUDA için hash_combine yardımcı fonksiyonu
__device__ 
size_t cuda_hash_combine(size_t seed, int val) {
    return seed ^ (hash_int(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

// Structure to hold point data for CUDA processing
struct PointData {
    float x, y, z;
};

// Structure to hold voxel grid cell data
struct VoxelData {
    int64_t voxel_idx;  // Combined voxel index
    float x, y, z;      // Point coordinates
    
    // Add constructor with parameters
    __host__ __device__
    VoxelData() : voxel_idx(0), x(0), y(0), z(0) {}
    
    __host__ __device__
    VoxelData(int64_t idx, float x_, float y_, float z_) 
        : voxel_idx(idx), x(x_), y(y_), z(z_) {}
};

// CPU implementation of voxel grid downsampling (as a fallback)
pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, 
    float voxel_size) 
{
    auto start = std::chrono::high_resolution_clock::now();
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (input_cloud->empty() || voxel_size <= 0) {
        std::cerr << "Invalid input cloud or voxel size!" << std::endl;
        return output_cloud;
    }
    
    std::unordered_map<int64_t, std::vector<size_t>> voxel_map;
    
    // CPU implementasyonunda (voxelGridDownsample fonksiyonu içinde)
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        const auto& point = input_cloud->points[i];
        
        // Skip invalid points (NaN or Inf)
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }
        
        // Calculate voxel indices
        int voxel_x = static_cast<int>(std::floor(point.x / voxel_size));
        int voxel_y = static_cast<int>(std::floor(point.y / voxel_size));
        int voxel_z = static_cast<int>(std::floor(point.z / voxel_size));
        
        // Hash fonksiyonu kullanarak voxel indeksini oluştur
        size_t combined_idx = 0;
        combined_idx = hash_combine(combined_idx, voxel_x);
        combined_idx = hash_combine(combined_idx, voxel_y);
        combined_idx = hash_combine(combined_idx, voxel_z);
        
        // Voxel indeksi olarak kullan
        int64_t voxel_idx = static_cast<int64_t>(combined_idx);
        
        voxel_map[voxel_idx].push_back(i);
    }
    // Second pass: Calculate centroids
    output_cloud->points.reserve(voxel_map.size());
    
    for (const auto& entry : voxel_map) {
        const auto& indices = entry.second;
        
        // Calculate centroid
        float sum_x = 0, sum_y = 0, sum_z = 0;
        for (size_t idx : indices) {
            sum_x += input_cloud->points[idx].x;
            sum_y += input_cloud->points[idx].y;
            sum_z += input_cloud->points[idx].z;
        }
        
        pcl::PointXYZ centroid;
        centroid.x = sum_x / indices.size();
        centroid.y = sum_y / indices.size();
        centroid.z = sum_z / indices.size();
        
        output_cloud->points.push_back(centroid);
    }
    
    output_cloud->width = output_cloud->points.size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "CPU VoxelGrid downsampling completed in " << duration << " ms" << std::endl;
    std::cout << "Points before: " << input_cloud->size() << std::endl;
    std::cout << "Points after: " << output_cloud->size() << std::endl;
    std::cout << "Reduction ratio: " << (1.0f - static_cast<float>(output_cloud->size()) / static_cast<float>(input_cloud->size())) * 100.0f << "%" << std::endl;
    
    return output_cloud;
}

// CUDA kernel to compute voxel indices for each point
__global__ void computeVoxelIndices(
    const PointData* points,
    VoxelData* voxel_data,
    int num_points,
    float voxel_size,
    int* valid_points_mask) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {
        float x = points[idx].x;
        float y = points[idx].y;
        float z = points[idx].z;
        
        // Check for NaN points
        if (isfinite(x) && isfinite(y) && isfinite(z)) {
            // Calculate voxel indices
            int voxel_x = static_cast<int>(floorf(x / voxel_size));
            int voxel_y = static_cast<int>(floorf(y / voxel_size));
            int voxel_z = static_cast<int>(floorf(z / voxel_size));
            
            // CUDA için hash_combine fonksiyonu kullanarak voxel indeksini oluştur
            size_t combined_idx = 0;
            combined_idx = cuda_hash_combine(combined_idx, voxel_x);
            combined_idx = cuda_hash_combine(combined_idx, voxel_y);
            combined_idx = cuda_hash_combine(combined_idx, voxel_z);
            
            // int64_t tipine dönüştür (gerekirse)
            int64_t voxel_idx = static_cast<int64_t>(combined_idx);
            
            voxel_data[idx].voxel_idx = voxel_idx;
            voxel_data[idx].x = x;
            voxel_data[idx].y = y;
            voxel_data[idx].z = z;
            valid_points_mask[idx] = 1;
        } else {
            // Mark invalid points
            voxel_data[idx].voxel_idx = INT64_MIN;
            valid_points_mask[idx] = 0;
        }
    }
}

// CUDA-based voxel grid downsampling function
pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsampleCUDA(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, 
    float voxel_size) 
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create output cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (input_cloud->empty() || voxel_size <= 0) {
        std::cerr << "Invalid input cloud or voxel size!" << std::endl;
        return output_cloud;
    }

    int num_points = input_cloud->size();
    std::cout << "Processing " << num_points << " points with CUDA..." << std::endl;

    // Determine minimum and maximum points for our bounding box
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(*input_cloud, min_point, max_point);
    
    std::cout << "Point cloud bounds: " << std::endl;
    std::cout << "Min: [" << min_point[0] << ", " << min_point[1] << ", " << min_point[2] << "]" << std::endl;
    std::cout << "Max: [" << max_point[0] << ", " << max_point[1] << ", " << max_point[2] << "]" << std::endl;

    // Prepare data for CUDA - host memory allocation
    PointData* h_points = new PointData[num_points];
    
    // Transfer point data to host array
    for (int i = 0; i < num_points; ++i) {
        h_points[i].x = input_cloud->points[i].x;
        h_points[i].y = input_cloud->points[i].y;
        h_points[i].z = input_cloud->points[i].z;
    }
    
    // Allocate device memory
    PointData* d_points;
    VoxelData* d_voxel_data;
    int* d_valid_points_mask;
    
    CUDA_CHECK(cudaMalloc((void**)&d_points, num_points * sizeof(PointData)));
    CUDA_CHECK(cudaMalloc((void**)&d_voxel_data, num_points * sizeof(VoxelData)));
    CUDA_CHECK(cudaMalloc((void**)&d_valid_points_mask, num_points * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_points, h_points, num_points * sizeof(PointData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_valid_points_mask, 0, num_points * sizeof(int)));
    
    // Launch kernel to compute voxel indices
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    
    computeVoxelIndices<<<blocksPerGrid, threadsPerBlock>>>(
        d_points,
        d_voxel_data,
        num_points,
        voxel_size,
        d_valid_points_mask);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy valid points mask back to host to count valid points
    int* h_valid_points_mask = new int[num_points];
    CUDA_CHECK(cudaMemcpy(h_valid_points_mask, d_valid_points_mask, num_points * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Count valid points
    int valid_points = 0;
    for (int i = 0; i < num_points; i++) {
        valid_points += h_valid_points_mask[i];
    }
    
    std::cout << "Valid points: " << valid_points << " of " << num_points << std::endl;
    
    if (valid_points == 0) {
        // Clean up memory
        delete[] h_points;
        delete[] h_valid_points_mask;
        CUDA_CHECK(cudaFree(d_points));
        CUDA_CHECK(cudaFree(d_voxel_data));
        CUDA_CHECK(cudaFree(d_valid_points_mask));
        return output_cloud;
    }
    
    // Copy voxel data back to host
    VoxelData* h_voxel_data = new VoxelData[num_points];
    CUDA_CHECK(cudaMemcpy(h_voxel_data, d_voxel_data, num_points * sizeof(VoxelData), cudaMemcpyDeviceToHost));
    
    // Clean up device memory that is no longer needed
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_voxel_data));
    CUDA_CHECK(cudaFree(d_valid_points_mask));
    
    // Process valid points on CPU (instead of using Thrust which is causing issues)
    std::vector<VoxelData> valid_voxels;
    valid_voxels.reserve(valid_points);
    
    for (int i = 0; i < num_points; i++) {
        if (h_valid_points_mask[i] == 1) {
            valid_voxels.push_back(h_voxel_data[i]);
        }
    }
    
    // Sort by voxel index using CPU
    std::sort(valid_voxels.begin(), valid_voxels.end(), 
             [](const VoxelData& a, const VoxelData& b) {
                 return a.voxel_idx < b.voxel_idx;
              });
    
    // Compute centroids for each voxel
    std::vector<pcl::PointXYZ> centroids;
    
    if (!valid_voxels.empty()) {
        int64_t current_voxel = valid_voxels[0].voxel_idx;
        float sum_x = valid_voxels[0].x;
        float sum_y = valid_voxels[0].y;
        float sum_z = valid_voxels[0].z;
        int point_count = 1;
        
        for (size_t i = 1; i < valid_voxels.size(); ++i) {
            if (valid_voxels[i].voxel_idx == current_voxel) {
                sum_x += valid_voxels[i].x;
                sum_y += valid_voxels[i].y;
                sum_z += valid_voxels[i].z;
                point_count++;
            } else {
                pcl::PointXYZ centroid;
                centroid.x = sum_x / point_count;
                centroid.y = sum_y / point_count;
                centroid.z = sum_z / point_count;
                centroids.push_back(centroid);
                
                current_voxel = valid_voxels[i].voxel_idx;
                sum_x = valid_voxels[i].x;
                sum_y = valid_voxels[i].y;
                sum_z = valid_voxels[i].z;
                point_count = 1;
            }
        }
        
        // Add the last centroid
        pcl::PointXYZ centroid;
        centroid.x = sum_x / point_count;
        centroid.y = sum_y / point_count;
        centroid.z = sum_z / point_count;
        centroids.push_back(centroid);
    }
    
    // Copy centroids to output cloud
    output_cloud->points.reserve(centroids.size());
    output_cloud->width = centroids.size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;
    
    for (const auto& c : centroids) {
        output_cloud->points.push_back(c);
    }
    
    // Clean up remaining host memory
    delete[] h_points;
    delete[] h_valid_points_mask;
    delete[] h_voxel_data;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "CUDA VoxelGrid downsampling completed in " << duration << " ms" << std::endl;
    std::cout << "Points before: " << input_cloud->size() << std::endl;
    std::cout << "Points after: " << output_cloud->size() << std::endl;
    std::cout << "Reduction ratio: " << (1.0f - static_cast<float>(output_cloud->size()) / static_cast<float>(input_cloud->size())) * 100.0f << "%" << std::endl;
    
    return output_cloud;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string input_file = "input.pcd";
    std::string output_file = "downsampled.pcd";
    float voxel_size = 0.1f; // 10cm default voxel size
    bool use_cuda = true;    // Use CUDA by default
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                input_file = argv[++i];
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        } else if (arg == "-v" || arg == "--voxel-size") {
            if (i + 1 < argc) {
                voxel_size = std::stof(argv[++i]);
            }
        } else if (arg == "--cpu") {
            use_cuda = false;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -i, --input FILE       Input point cloud file (default: input.pcd)" << std::endl;
            std::cout << "  -o, --output FILE      Output point cloud file (default: downsampled.pcd)" << std::endl;
            std::cout << "  -v, --voxel-size SIZE  Voxel size for downsampling (default: 0.1)" << std::endl;
            std::cout << "  --cpu                  Use CPU implementation instead of CUDA" << std::endl;
            std::cout << "  -h, --help             Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    std::cout << "Loading point cloud from: " << input_file << std::endl;
    
    // Determine file type and read
    std::string extension = input_file.substr(input_file.find_last_of(".") + 1);
    if (extension == "pcd") {
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1) {
            std::cerr << "Error loading PCD file." << std::endl;
            return -1;
        }
    } else if (extension == "ply") {
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(input_file, *cloud) == -1) {
            std::cerr << "Error loading PLY file." << std::endl;
            return -1;
        }
    } else {
        std::cerr << "Unsupported file format. Please use .pcd or .ply files." << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << cloud->size() << " points." << std::endl;
    
    // Check if CUDA is available
    int cuda_device_count;
    cudaError_t cuda_status = cudaGetDeviceCount(&cuda_device_count);
    
    if (cuda_status != cudaSuccess || cuda_device_count == 0) {
        std::cerr << "No CUDA devices found. Falling back to CPU implementation." << std::endl;
        use_cuda = false;
    } else {
        std::cout << "Found " << cuda_device_count << " CUDA device(s)" << std::endl;
        
        // Print device properties
        for (int i = 0; i < cuda_device_count; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            std::cout << "Device " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        }
        
        // Select first device
        CUDA_CHECK(cudaSetDevice(0));
    }
    
    // Benchmark total execution time
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Perform voxel grid downsampling
    std::cout << "Performing voxel grid downsampling with voxel size: " << voxel_size << std::endl;
    std::cout << "Using " << (use_cuda ? "CUDA" : "CPU") << " implementation" << std::endl;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud;
    
    if (use_cuda) {
        downsampled_cloud = voxelGridDownsampleCUDA(cloud, voxel_size);
    } else {
        downsampled_cloud = voxelGridDownsample(cloud, voxel_size);
    }
    
    // Save the downsampled cloud
    extension = output_file.substr(output_file.find_last_of(".") + 1);
    
    std::cout << "Saving downsampled point cloud to: " << output_file << std::endl;
    if (extension == "pcd") {
        pcl::io::savePCDFile(output_file, *downsampled_cloud);
    } else if (extension == "ply") {
        pcl::io::savePLYFile(output_file, *downsampled_cloud);
    } else {
        std::cerr << "Unsupported output format. Using default PCD format." << std::endl;
        pcl::io::savePCDFile("downsampled.pcd", *downsampled_cloud);
        std::cout << "File saved as downsampled.pcd" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    
    std::cout << "Total execution time: " << total_duration << " ms" << std::endl;
    
    return 0;
}