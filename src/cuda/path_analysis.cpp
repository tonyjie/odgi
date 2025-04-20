#include "layout.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <map>
#include <set>

namespace cuda {

struct path_stats_t {
    uint32_t path_idx;
    uint32_t step_count;
    std::set<uint32_t> unique_nodes;
    uint32_t unique_node_count;
};

void analyze_paths(const path_data_t &path_data, const std::string &output_file) {
    std::cout << "=== Path Analysis for NUMA Optimization ===" << std::endl;
    std::cout << "Total number of paths: " << path_data.path_count << std::endl;
    std::cout << "Total number of steps across all paths: " << path_data.total_path_steps << std::endl;
    
    // Collect statistics for each path
    std::vector<path_stats_t> path_stats;
    uint64_t total_steps = 0;
    uint64_t min_steps = UINT64_MAX;
    uint64_t max_steps = 0;
    
    for (uint32_t p = 0; p < path_data.path_count; p++) {
        path_stats_t stats;
        stats.path_idx = p;
        stats.step_count = path_data.paths[p].step_count;
        
        // Track unique nodes in this path
        std::set<uint32_t> unique_nodes;
        for (uint32_t s = 0; s < stats.step_count; s++) {
            uint32_t node_id = path_data.paths[p].elements[s].node_id;
            unique_nodes.insert(node_id);
        }
        
        stats.unique_nodes = unique_nodes;
        stats.unique_node_count = unique_nodes.size();
        
        path_stats.push_back(stats);
        
        // Update statistics
        total_steps += stats.step_count;
        min_steps = std::min(min_steps, (uint64_t)stats.step_count);
        max_steps = std::max(max_steps, (uint64_t)stats.step_count);
    }
    
    // Sort paths by step count (descending) for easier partitioning
    std::sort(path_stats.begin(), path_stats.end(), 
              [](const path_stats_t &a, const path_stats_t &b) {
                  return a.step_count > b.step_count;
              });
    
    // Calculate average and median
    double avg_steps = (double)total_steps / path_data.path_count;
    uint64_t median_steps;
    if (path_data.path_count % 2 == 0) {
        median_steps = (path_stats[path_data.path_count/2-1].step_count + 
                         path_stats[path_data.path_count/2].step_count) / 2;
    } else {
        median_steps = path_stats[path_data.path_count/2].step_count;
    }
    
    std::cout << "Step count statistics:" << std::endl;
    std::cout << "  Average steps per path: " << avg_steps << std::endl;
    std::cout << "  Median steps per path: " << median_steps << std::endl;
    std::cout << "  Min steps in a path: " << min_steps << std::endl;
    std::cout << "  Max steps in a path: " << max_steps << std::endl;
    
    // Analyze node distribution
    std::map<uint32_t, uint32_t> node_frequency;
    for (const auto &stats : path_stats) {
        for (uint32_t node_id : stats.unique_nodes) {
            node_frequency[node_id]++;
        }
    }
    
    uint32_t total_unique_nodes = node_frequency.size();
    std::cout << "Total unique nodes across all paths: " << total_unique_nodes << std::endl;
    
    // Find nodes that appear in many paths
    std::vector<std::pair<uint32_t, uint32_t>> sorted_nodes;
    for (const auto &pair : node_frequency) {
        sorted_nodes.push_back(pair);
    }
    
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
              [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
                  return a.second > b.second;
              });
    
    std::cout << "Top 10 most shared nodes:" << std::endl;
    for (int i = 0; i < std::min(10u, (uint32_t)sorted_nodes.size()); i++) {
        std::cout << "  Node " << sorted_nodes[i].first 
                  << " appears in " << sorted_nodes[i].second 
                  << " paths (" << (double)sorted_nodes[i].second / path_data.path_count * 100.0 
                  << "%)" << std::endl;
    }
    
    // Try different path partitioning approaches
    std::cout << "\nNUMA Partitioning Analysis:" << std::endl;
    
    // Approach 1: Greedy partitioning by step count
    uint64_t half_steps = total_steps / 2;
    uint64_t current_steps = 0;
    int split_idx = 0;
    
    for (int i = 0; i < path_stats.size(); i++) {
        current_steps += path_stats[i].step_count;
        if (current_steps > half_steps) {
            split_idx = i;
            break;
        }
    }
    
    uint64_t numa1_steps = current_steps;
    uint64_t numa2_steps = total_steps - current_steps;
    uint32_t numa1_paths = split_idx + 1;
    uint32_t numa2_paths = path_data.path_count - numa1_paths;
    
    std::cout << "Greedy partitioning by step count:" << std::endl;
    std::cout << "  NUMA Node 1: " << numa1_paths << " paths with " << numa1_steps 
              << " steps (" << (double)numa1_steps / total_steps * 100.0 << "%)" << std::endl;
    std::cout << "  NUMA Node 2: " << numa2_paths << " paths with " << numa2_steps 
              << " steps (" << (double)numa2_steps / total_steps * 100.0 << "%)" << std::endl;
    
    // Check node overlap between partitions
    std::set<uint32_t> numa1_nodes;
    std::set<uint32_t> numa2_nodes;
    
    for (int i = 0; i < numa1_paths; i++) {
        for (uint32_t node_id : path_stats[i].unique_nodes) {
            numa1_nodes.insert(node_id);
        }
    }
    
    for (int i = numa1_paths; i < path_stats.size(); i++) {
        for (uint32_t node_id : path_stats[i].unique_nodes) {
            numa2_nodes.insert(node_id);
        }
    }
    
    // Find intersection
    std::vector<uint32_t> common_nodes;
    std::set_intersection(numa1_nodes.begin(), numa1_nodes.end(),
                          numa2_nodes.begin(), numa2_nodes.end(),
                          std::back_inserter(common_nodes));
    
    uint32_t overlap_count = common_nodes.size();
    double overlap_percent = (double)overlap_count / total_unique_nodes * 100.0;
    
    std::cout << "Node overlap analysis:" << std::endl;
    std::cout << "  NUMA Node 1 unique nodes: " << numa1_nodes.size() << std::endl;
    std::cout << "  NUMA Node 2 unique nodes: " << numa2_nodes.size() << std::endl;
    std::cout << "  Overlapping nodes: " << overlap_count 
              << " (" << overlap_percent << "%)" << std::endl;
    
    // Write detailed path info to a CSV file for further analysis
    if (!output_file.empty()) {
        std::ofstream outfile(output_file);
        if (outfile.is_open()) {
            outfile << "path_idx,original_idx,step_count,unique_node_count,numa_node\n";
            
            for (int i = 0; i < path_stats.size(); i++) {
                outfile << i << "," 
                        << path_stats[i].path_idx << "," 
                        << path_stats[i].step_count << "," 
                        << path_stats[i].unique_node_count << "," 
                        << (i < numa1_paths ? 1 : 2) << "\n";
            }
            
            outfile.close();
            std::cout << "\nDetailed path information written to " << output_file << std::endl;
        } else {
            std::cerr << "Error: Could not open output file " << output_file << std::endl;
        }
    }
}

// Function to return path partition assignments for NUMA optimization
std::vector<int> partition_paths_for_numa(const path_data_t &path_data, int numa_count) {
    // Default to 2 NUMA nodes if not specified
    if (numa_count <= 0) numa_count = 2;
    
    // Store step counts for each path
    std::vector<std::pair<uint32_t, uint32_t>> path_sizes; // (path_idx, step_count)
    for (uint32_t p = 0; p < path_data.path_count; p++) {
        path_sizes.push_back({p, path_data.paths[p].step_count});
    }
    
    // Sort by step count (descending)
    std::sort(path_sizes.begin(), path_sizes.end(),
              [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
                  return a.second > b.second;
              });
    
    // Initialize NUMA node loads
    std::vector<uint64_t> numa_loads(numa_count, 0);
    
    // Result vector: maps original path_idx to NUMA node (0-based)
    std::vector<int> path_to_numa(path_data.path_count, -1);
    
    // Greedy assignment: assign each path to the least loaded NUMA node
    for (const auto &path : path_sizes) {
        // Find least loaded NUMA node
        int target_numa = 0;
        uint64_t min_load = numa_loads[0];
        
        for (int n = 1; n < numa_count; n++) {
            if (numa_loads[n] < min_load) {
                min_load = numa_loads[n];
                target_numa = n;
            }
        }
        
        // Assign path to this NUMA node
        path_to_numa[path.first] = target_numa;
        numa_loads[target_numa] += path.second;
    }
    
    // Report distribution
    std::cout << "Path distribution across NUMA nodes:" << std::endl;
    for (int n = 0; n < numa_count; n++) {
        int path_count = std::count(path_to_numa.begin(), path_to_numa.end(), n);
        std::cout << "  NUMA Node " << n << ": " << path_count << " paths, " 
                  << numa_loads[n] << " steps ("
                  << (double)numa_loads[n] / path_data.total_path_steps * 100.0 << "%)" << std::endl;
    }
    
    return path_to_numa;
}

} // namespace cuda 