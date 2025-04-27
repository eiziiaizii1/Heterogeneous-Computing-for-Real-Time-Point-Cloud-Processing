#include <iostream>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h> // Include this for getMinMax3D
#include <Eigen/Core>
#include <omp.h>
#include <mutex>
#include <atomic>

// Hash function for 3D grid coordinates
struct VoxelGridHasher {
    std::size_t operator()(const Eigen::Vector3i& key) const {
        // Pairing function to create a unique hash from 3D coordinates
        // Using Cantor pairing function extended to 3D
        std::size_t h1 = ((key[0] + key[1]) * (key[0] + key[1] + 1)) / 2 + key[1];
        return ((h1 + key[2]) * (h1 + key[2] + 1)) / 2 + key[2];
    }
};

// Class for equality comparison of Eigen::Vector3i in the hash map
struct VoxelGridEqualTo {
    bool operator()(const Eigen::Vector3i& k1, const Eigen::Vector3i& k2) const {
        return k1[0] == k2[0] && k1[1] == k2[1] && k1[2] == k2[2];
    }
};

// Use double precision for the voxel centroids and accumulation
typedef std::unordered_map<Eigen::Vector3i, std::pair<Eigen::Vector3d, int>, VoxelGridHasher, VoxelGridEqualTo> VoxelMap;

// Global variables to store voxel origin consistently between implementations
Eigen::Vector3d g_voxel_origin;
bool g_voxel_origin_set = false;

// Function to consistently calculate voxel indices
inline Eigen::Vector3i calculateVoxelIndices(
    const pcl::PointXYZ& point, 
    const double voxel_size, 
    const Eigen::Vector3d& origin) 
{
    // Use double precision for calculations
    double vx = (static_cast<double>(point.x) - origin[0]) / voxel_size;
    double vy = (static_cast<double>(point.y) - origin[1]) / voxel_size;
    double vz = (static_cast<double>(point.z) - origin[2]) / voxel_size;
    
    // Use floor to ensure correct handling of negative coordinates
    int voxel_x = static_cast<int>(std::floor(vx));
    int voxel_y = static_cast<int>(std::floor(vy));
    int voxel_z = static_cast<int>(std::floor(vz));
    
    return Eigen::Vector3i(voxel_x, voxel_y, voxel_z);
}

// Custom VoxelGrid downsample function (serial implementation with double precision)
pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, 
    double voxel_size) 
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create output cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (input_cloud->empty() || voxel_size <= 0) {
        std::cerr << "Invalid input cloud or voxel size!" << std::endl;
        return output_cloud;
    }

    // Determine minimum and maximum points for our bounding box
    Eigen::Vector4f min_point_f, max_point_f;
    pcl::getMinMax3D(*input_cloud, min_point_f, max_point_f);
    
    // Convert to double precision
    Eigen::Vector3d min_point(min_point_f[0], min_point_f[1], min_point_f[2]);
    Eigen::Vector3d max_point(max_point_f[0], max_point_f[1], max_point_f[2]);
    
    std::cout << "Point cloud bounds: " << std::endl;
    std::cout << "Min: [" << min_point[0] << ", " << min_point[1] << ", " << min_point[2] << "]" << std::endl;
    std::cout << "Max: [" << max_point[0] << ", " << max_point[1] << ", " << max_point[2] << "]" << std::endl;

    // Set global voxel grid origin if not already set
    if (!g_voxel_origin_set) {
        g_voxel_origin = min_point;
        g_voxel_origin_set = true;
    }
    
    // Use the global voxel origin for consistency
    const Eigen::Vector3d& voxel_origin = g_voxel_origin;
    std::cout << "Using voxel origin: [" << voxel_origin[0] << ", " << voxel_origin[1] << ", " << voxel_origin[2] << "]" << std::endl;

    // Hash map to store voxel indices and their associated points (using double precision)
    // Key: voxel grid coordinates, Value: (sum of points, count of points)
    VoxelMap voxel_map;

    // Process each point in the input cloud
    for (const auto& point : input_cloud->points) {
        // Skip NaN points
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }

        // Calculate voxel indices for this point using the helper function
        Eigen::Vector3i voxel_idx = calculateVoxelIndices(point, voxel_size, voxel_origin);
        
        // Store point with double precision
        Eigen::Vector3d point_vector(static_cast<double>(point.x), 
                                    static_cast<double>(point.y), 
                                    static_cast<double>(point.z));

        // Add point to corresponding voxel
        auto voxel_iter = voxel_map.find(voxel_idx);
        if (voxel_iter != voxel_map.end()) {
            // Voxel exists, update sum and count
            voxel_iter->second.first += point_vector;
            voxel_iter->second.second++;
        } else {
            // New voxel, initialize with this point
            voxel_map[voxel_idx] = std::make_pair(point_vector, 1);
        }
    }

    // Prepare output cloud
    output_cloud->points.reserve(voxel_map.size());
    output_cloud->width = voxel_map.size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;

    // Calculate centroids and add to output cloud
    for (const auto& voxel : voxel_map) {
        // Calculate centroid with double precision
        Eigen::Vector3d centroid = voxel.second.first / static_cast<double>(voxel.second.second);
        
        // Convert back to float for PCL point
        pcl::PointXYZ new_point;
        new_point.x = static_cast<float>(centroid[0]);
        new_point.y = static_cast<float>(centroid[1]);
        new_point.z = static_cast<float>(centroid[2]);
        output_cloud->points.push_back(new_point);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Serial VoxelGrid downsampling completed in " << duration << " ms" << std::endl;
    std::cout << "Points before: " << input_cloud->size() << std::endl;
    std::cout << "Points after: " << output_cloud->size() << std::endl;
    std::cout << "Reduction ratio: " << (1.0 - static_cast<double>(output_cloud->size()) / static_cast<double>(input_cloud->size())) * 100.0 << "%" << std::endl;
    
    return output_cloud;
}

// OpenMP parallelized VoxelGrid downsample function with double precision
pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsampleOMP(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, 
    double voxel_size,
    int num_threads = 0) 
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create output cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (input_cloud->empty() || voxel_size <= 0) {
        std::cerr << "Invalid input cloud or voxel size!" << std::endl;
        return output_cloud;
    }

    // Set number of threads if specified
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    int max_threads = omp_get_max_threads();
    std::cout << "Using " << max_threads << " threads for parallel processing" << std::endl;

    // Determine minimum and maximum points for our bounding box if voxel origin not set
    if (!g_voxel_origin_set) {
        Eigen::Vector4f min_point_f, max_point_f;
        pcl::getMinMax3D(*input_cloud, min_point_f, max_point_f);
        g_voxel_origin = Eigen::Vector3d(min_point_f[0], min_point_f[1], min_point_f[2]);
        g_voxel_origin_set = true;
    }
    
    // Use the global voxel origin for consistency between runs and implementations
    const Eigen::Vector3d& voxel_origin = g_voxel_origin;
    std::cout << "Using voxel origin: [" << voxel_origin[0] << ", " << voxel_origin[1] << ", " << voxel_origin[2] << "]" << std::endl;

    // Each thread has its own voxel map to avoid contention
    std::vector<VoxelMap> thread_voxel_maps(max_threads);
    
    // Process points in parallel
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_voxel_map = thread_voxel_maps[thread_id];
        
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < input_cloud->points.size(); i++) {
            const auto& point = input_cloud->points[i];
            
            // Skip NaN points
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }

            // Calculate voxel indices for this point using the helper function
            Eigen::Vector3i voxel_idx = calculateVoxelIndices(point, voxel_size, voxel_origin);
            
            // Store point with double precision
            Eigen::Vector3d point_vector(static_cast<double>(point.x), 
                                        static_cast<double>(point.y), 
                                        static_cast<double>(point.z));

            // Add point to thread's local voxel map
            auto voxel_iter = local_voxel_map.find(voxel_idx);
            if (voxel_iter != local_voxel_map.end()) {
                // Voxel exists, update sum and count
                voxel_iter->second.first += point_vector;
                voxel_iter->second.second++;
            } else {
                // New voxel, initialize with this point
                local_voxel_map[voxel_idx] = std::make_pair(point_vector, 1);
            }
        }
    }

    // Merge thread-local voxel maps
    VoxelMap merged_voxel_map;
    for (const auto& local_map : thread_voxel_maps) {
        for (const auto& voxel : local_map) {
            auto merged_iter = merged_voxel_map.find(voxel.first);
            if (merged_iter != merged_voxel_map.end()) {
                // Voxel exists in merged map, combine results
                merged_iter->second.first += voxel.second.first;
                merged_iter->second.second += voxel.second.second;
            } else {
                // New voxel in merged map
                merged_voxel_map[voxel.first] = voxel.second;
            }
        }
    }

    // Prepare output cloud
    output_cloud->points.reserve(merged_voxel_map.size());
    output_cloud->width = merged_voxel_map.size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;

    // Calculate centroids and add to output cloud
    std::vector<pcl::PointXYZ> output_points(merged_voxel_map.size());
    
    // Convert map to vector for parallel processing
    std::vector<std::pair<Eigen::Vector3i, std::pair<Eigen::Vector3d, int>>> voxel_vector;
    voxel_vector.reserve(merged_voxel_map.size());
    for (const auto& voxel : merged_voxel_map) {
        voxel_vector.push_back(voxel);
    }
    
    // Process voxels in parallel (no sorting)
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < voxel_vector.size(); i++) {
        // Calculate centroid with double precision
        Eigen::Vector3d centroid = voxel_vector[i].second.first / static_cast<double>(voxel_vector[i].second.second);
        
        // Convert back to float for PCL point
        pcl::PointXYZ new_point;
        new_point.x = static_cast<float>(centroid[0]);
        new_point.y = static_cast<float>(centroid[1]);
        new_point.z = static_cast<float>(centroid[2]);
        output_points[i] = new_point;
    }
    
    // Add all points to output cloud
    output_cloud->points.insert(output_cloud->points.end(), output_points.begin(), output_points.end());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "OMP Parallel VoxelGrid downsampling completed in " << duration << " ms" << std::endl;
    std::cout << "Points before: " << input_cloud->size() << std::endl;
    std::cout << "Points after: " << output_cloud->size() << std::endl;
    std::cout << "Reduction ratio: " << (1.0 - static_cast<double>(output_cloud->size()) / static_cast<double>(input_cloud->size())) * 100.0 << "%" << std::endl;
    
    return output_cloud;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string input_file = "input.pcd";
    std::string output_file = "downsampled.pcd";
    double voxel_size = 0.1; // 10cm default voxel size
    int num_threads = 0; // 0 means use default OpenMP thread count
    
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
                voxel_size = std::stod(argv[++i]); // Changed to double
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -i, --input FILE       Input point cloud file (default: input.pcd)" << std::endl;
            std::cout << "  -o, --output FILE      Output point cloud file (default: downsampled.pcd)" << std::endl;
            std::cout << "  -v, --voxel-size SIZE  Voxel size for downsampling (default: 0.1)" << std::endl;
            std::cout << "  -t, --threads NUM      Number of OpenMP threads (default: system)" << std::endl;
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
    
    // Benchmark total execution time
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Reset global voxel origin before processing
    g_voxel_origin_set = false;
    
    // Perform serial voxel grid downsampling
    std::cout << "\n=== SERIAL IMPLEMENTATION ===" << std::endl;
    std::cout << "Performing voxel grid downsampling with voxel size: " << voxel_size << std::endl;
    auto serial_start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr serial_downsampled = voxelGridDownsample(cloud, voxel_size);
    auto serial_end = std::chrono::high_resolution_clock::now();
    auto serial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serial_end - serial_start).count();
    
    // Perform parallel voxel grid downsampling
    std::cout << "\n=== PARALLEL IMPLEMENTATION ===" << std::endl;
    std::cout << "Performing parallel voxel grid downsampling with voxel size: " << voxel_size << std::endl;
    auto parallel_start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr parallel_downsampled = voxelGridDownsampleOMP(cloud, voxel_size, num_threads);
    auto parallel_end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(parallel_end - parallel_start).count();
    
    // Performance comparison
    std::cout << "\n=== PERFORMANCE COMPARISON ===" << std::endl;
    std::cout << "Serial processing time: " << serial_duration << " ms" << std::endl;
    std::cout << "Parallel processing time: " << parallel_duration << " ms" << std::endl;
    
    if (serial_duration > 0) { // Avoid division by zero
        double speedup = static_cast<double>(serial_duration) / static_cast<double>(parallel_duration);
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        std::cout << "Efficiency: " << (speedup / omp_get_max_threads()) * 100.0 << "%" << std::endl;
    }

    // Save the downsampled point cloud (using both versions)
    extension = output_file.substr(output_file.find_last_of(".") + 1);

    std::cout << "\nSaving downsampled point clouds..." << std::endl;
    if (extension == "pcd") {
        // Save serial version with "_serial" suffix
        std::string serial_output = output_file.substr(0, output_file.find_last_of(".")) + "_serial.pcd";
        pcl::io::savePCDFile(serial_output, *serial_downsampled);
        std::cout << "Serial version saved to: " << serial_output << std::endl;
        
        // Save parallel version with "_parallel" suffix
        std::string parallel_output = output_file.substr(0, output_file.find_last_of(".")) + "_parallel.pcd";
        pcl::io::savePCDFile(parallel_output, *parallel_downsampled);
        std::cout << "Parallel version saved to: " << parallel_output << std::endl;
    } else if (extension == "ply") {
        // Save serial version with "_serial" suffix
        std::string serial_output = output_file.substr(0, output_file.find_last_of(".")) + "_serial.ply";
        pcl::io::savePLYFile(serial_output, *serial_downsampled);
        std::cout << "Serial version saved to: " << serial_output << std::endl;
        
        // Save parallel version with "_parallel" suffix
        std::string parallel_output = output_file.substr(0, output_file.find_last_of(".")) + "_parallel.ply";
        pcl::io::savePLYFile(parallel_output, *parallel_downsampled);
        std::cout << "Parallel version saved to: " << parallel_output << std::endl;
    } else {
        std::cerr << "Unsupported output format. Using default PCD format." << std::endl;
        pcl::io::savePCDFile("downsampled_serial.pcd", *serial_downsampled);
        pcl::io::savePCDFile("downsampled_parallel.pcd", *parallel_downsampled);
        std::cout << "Files saved as downsampled_serial.pcd and downsampled_parallel.pcd" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    
    std::cout << "\nTotal execution time: " << total_duration << " ms" << std::endl;
    
    return 0;
}