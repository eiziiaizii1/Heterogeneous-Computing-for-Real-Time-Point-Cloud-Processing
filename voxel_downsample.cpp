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

// Custom VoxelGrid downsample function (serial implementation)
pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridDownsample(
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

    // Determine minimum and maximum points for our bounding box
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(*input_cloud, min_point, max_point);
    
    std::cout << "Point cloud bounds: " << std::endl;
    std::cout << "Min: [" << min_point[0] << ", " << min_point[1] << ", " << min_point[2] << "]" << std::endl;
    std::cout << "Max: [" << max_point[0] << ", " << max_point[1] << ", " << max_point[2] << "]" << std::endl;

    // Hash map to store voxel indices and their associated points
    // Key: voxel grid coordinates, Value: (sum of points, count of points)
    std::unordered_map<Eigen::Vector3i, std::pair<Eigen::Vector3f, int>, VoxelGridHasher, VoxelGridEqualTo> voxel_map;

    // Process each point in the input cloud
    for (const auto& point : input_cloud->points) {
        // Skip NaN points
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }

        // Calculate voxel indices for this point
        int voxel_x = static_cast<int>(std::floor(point.x / voxel_size));
        int voxel_y = static_cast<int>(std::floor(point.y / voxel_size));
        int voxel_z = static_cast<int>(std::floor(point.z / voxel_size));
        
        Eigen::Vector3i voxel_idx(voxel_x, voxel_y, voxel_z);
        Eigen::Vector3f point_vector(point.x, point.y, point.z);

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
        // Calculate centroid
        Eigen::Vector3f centroid = voxel.second.first / static_cast<float>(voxel.second.second);
        
        // Add centroid to output cloud
        pcl::PointXYZ new_point;
        new_point.x = centroid[0];
        new_point.y = centroid[1];
        new_point.z = centroid[2];
        output_cloud->points.push_back(new_point);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "VoxelGrid downsampling completed in " << duration << " ms" << std::endl;
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
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -i, --input FILE       Input point cloud file (default: input.pcd)" << std::endl;
            std::cout << "  -o, --output FILE      Output point cloud file (default: downsampled.pcd)" << std::endl;
            std::cout << "  -v, --voxel-size SIZE  Voxel size for downsampling (default: 0.1)" << std::endl;
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
    
    // Perform voxel grid downsampling
    std::cout << "Performing voxel grid downsampling with voxel size: " << voxel_size << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud = voxelGridDownsample(cloud, voxel_size);
    
    // Save the original and downsampled cloud
    extension = output_file.substr(output_file.find_last_of(".") + 1);
    
    std::cout << "Saving downsampled point cloud to: " << output_file << std::endl;
    if (extension == "pcd") {
        pcl::io::savePCDFile(output_file, *downsampled_cloud);
        // Save original with "_original" suffix
        /*
        std::string original_output = output_file.substr(0, output_file.find_last_of(".")) + "_original.pcd";
        pcl::io::savePCDFile(original_output, *cloud);
        std::cout << "Original point cloud saved to: " << original_output << std::endl;
        */
    } else if (extension == "ply") {
        pcl::io::savePLYFile(output_file, *downsampled_cloud);
        // Save original with "_original" suffix
        /*
        std::string original_output = output_file.substr(0, output_file.find_last_of(".")) + "_original.ply";
        pcl::io::savePLYFile(original_output, *cloud);
        std::cout << "Original point cloud saved to: " << original_output << std::endl;
        */
    } else {
        std::cerr << "Unsupported output format. Using default PCD format." << std::endl;
        pcl::io::savePCDFile("downsampled.pcd", *downsampled_cloud);
        pcl::io::savePCDFile("original.pcd", *cloud);
        std::cout << "Files saved as downsampled.pcd and original.pcd" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    
    std::cout << "Total execution time: " << total_duration << " ms" << std::endl;
    
    return 0;
}