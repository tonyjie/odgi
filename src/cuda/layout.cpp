#include "layout.h"
//#include <cuda.h>
#include <assert.h>
#include <iomanip>
#include <sched.h>

namespace cuda {

// ===== path_analysis functions =====
struct path_stats_t {
    uint32_t path_idx;
    uint32_t step_count;
    std::set<uint32_t> unique_nodes;
    uint32_t unique_node_count;
};

void analyze_paths(const cuda::node_data_t &node_data, const path_data_t &path_data, const std::string &output_file) {
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
            if (path_data.paths[p].elements != NULL) {
                uint32_t node_id = path_data.paths[p].elements[s].node_id;
                unique_nodes.insert(node_id);
            } else {
                // raise error
                std::cerr << "Path " << p << " has no elements" << std::endl;
                exit(1);
            }
        }
        
        // add some check print
        std::cout << "Path " << p << " has " << stats.step_count << " steps" << std::endl;

        stats.unique_nodes = unique_nodes;
        stats.unique_node_count = unique_nodes.size();
        
        path_stats.push_back(stats);
        
        // Update statistics
        total_steps += stats.step_count;
        min_steps = std::min(min_steps, (uint64_t)stats.step_count);
        max_steps = std::max(max_steps, (uint64_t)stats.step_count);
    }
    
    // print if unique nodes are running corredctly
    std::cout << "Passed unique nodes check" << std::endl;

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
    // print node_count
    std::cout << "node_count: " << node_data.node_count << std::endl;
    
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
// ===== path_analysis functions =====




void cpu_layout(cuda::layout_config_t config, double *etas, double *zetas, cuda::node_data_t &node_data, cuda::path_data_t &path_data, const std::vector<int> &path_numa_assignments) {
    int nbr_threads = config.nthreads;
    std::cout << "cuda cpu layout (" << nbr_threads << " threads)" << std::endl;
    
    bool use_numa_partitioning = !path_numa_assignments.empty() && path_numa_assignments.size() == path_data.path_count;
    if (use_numa_partitioning) {
        std::cout << "Using NUMA-aware path partitioning" << std::endl;
    }
    
    // Each NUMA node will need its own paths distribution for random selection
    std::vector<std::vector<uint64_t>> numa_path_dist;
    if (use_numa_partitioning) {
        // Calculate the number of NUMA nodes (max value in assignments + 1)
        int numa_count = 0;
        for (int numa : path_numa_assignments) {
            numa_count = std::max(numa_count, numa + 1);
        }
        numa_path_dist.resize(numa_count);
        
        // Build path distribution for each NUMA node
        for (uint32_t p = 0; p < path_data.path_count; p++) {
            int numa = path_numa_assignments[p];
            numa_path_dist[numa].push_back(uint64_t(path_data.paths[p].step_count));
        }
    }

    // If not using NUMA partitioning, use a single distribution for all threads
    std::vector<uint64_t> path_dist;
    if (!use_numa_partitioning) {
        for (int p = 0; p < path_data.path_count; p++) {
            path_dist.push_back(uint64_t(path_data.paths[p].step_count));
        }
    }

#pragma omp parallel num_threads(nbr_threads)
    {
        int tid = omp_get_thread_num();
        int numa_node = (use_numa_partitioning && nbr_threads > 1) ? (tid / (nbr_threads / 2)) : 0;
        
        // Set thread affinity to the appropriate NUMA node
        if (use_numa_partitioning) {
            #ifdef _OPENMP
            // Bind each thread to cores on its NUMA node
            cpu_set_t mask;
            CPU_ZERO(&mask);
            
            // For a dual-socket system with 24 cores per socket:
            // - NUMA node 0: cores 0-23
            // - NUMA node 1: cores 24-47
            int core_id;
            if (numa_node == 0) {
                // Bind to a core on NUMA node 0 (0-23)
                core_id = tid % (nbr_threads / 2);
            } else {
                // Bind to a core on NUMA node 1 (24-47)
                core_id = 24 + (tid % (nbr_threads / 2));
            }
            
            CPU_SET(core_id, &mask);
            
            // Apply the affinity mask
            if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
                #pragma omp critical
                {
                    std::cerr << "Thread " << tid << " failed to set CPU affinity to core " << core_id << " on NUMA node " << numa_node << std::endl;
                }
            } else {
                // Only print a few thread bindings to avoid excessive output
                bool should_print = (tid < 2) || (tid == nbr_threads/2) || (tid == nbr_threads/2 - 1) || (tid >= nbr_threads - 2);
                if (should_print) {
                    #pragma omp critical
                    {
                        std::cout << "Thread " << tid << " bound to core " << core_id << " on NUMA node " << numa_node << std::endl;
                    }
                }
            }
            #endif
        }
        
        // Thread's discrete distribution either based on all paths or just its NUMA node's paths
        std::discrete_distribution<> rand_path;
        if (use_numa_partitioning) {
            // Check if this NUMA node has any paths
            if (!numa_path_dist[numa_node].empty()) {
                rand_path = std::discrete_distribution<>(
                    numa_path_dist[numa_node].begin(), 
                    numa_path_dist[numa_node].end()
                );
            } else {
                // If no paths for this NUMA node, default to all paths (should not happen)
                std::cout << "Warning: NUMA node " << numa_node << " has no paths assigned" << std::endl;
                rand_path = std::discrete_distribution<>(path_dist.begin(), path_dist.end());
            }
        } else {
            rand_path = std::discrete_distribution<>(path_dist.begin(), path_dist.end());
        }

        XoshiroCpp::Xoshiro256Plus gen(9399220 + tid);
        std::uniform_int_distribution<uint64_t> flip(0, 1);

        const int steps_per_thread = config.min_term_updates / nbr_threads;

        for (int iter = 0; iter < config.iter_max; iter++ ) {
            // synchronize all threads before each iteration
#pragma omp barrier
            for (int step = 0; step < steps_per_thread; step++ ) {
                // get path
                uint32_t path_idx; 
                if (use_numa_partitioning) {
                    // When using NUMA partitioning, need to map the distribution's output to actual path indices
                    int local_path_idx = rand_path(gen);
                    
                    // Convert local path index to global path index based on NUMA assignments
                    int count = 0;
                    for (uint32_t p = 0; p < path_data.path_count; p++) {
                        if (path_numa_assignments[p] == numa_node) {
                            if (count == local_path_idx) {
                                path_idx = p;
                                break;
                            }
                            count++;
                        }
                    }
                } else {
                    path_idx = rand_path(gen);
                }
                
                path_t p = path_data.paths[path_idx];
                if (p.step_count < 2) {
                    continue;
                }

                std::uniform_int_distribution<uint32_t> rand_step(0, p.step_count-1);

                uint32_t s1_idx = rand_step(gen);
                uint32_t s2_idx;
                if (iter >= config.first_cooling_iteration || flip(gen)) {
                    if (s1_idx > 0 && flip(gen) || s1_idx == p.step_count-1) {
                        // go backward
                        uint32_t jump_space = std::min(config.space, s1_idx);
                        uint32_t space = jump_space;
                        if (jump_space > config.space_max) {
                            space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
                        }
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, config.theta, zetas[space]);
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                        uint32_t z_i = (uint32_t) z(gen);
                        s2_idx = s1_idx - z_i;
                    } else {
                        // go forward
                        uint32_t jump_space = std::min(config.space, p.step_count - s1_idx - 1);
                        uint32_t space = jump_space;
                        if (jump_space > config.space_max) {
                            space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
                        }
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, config.theta, zetas[space]);
                        dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                        uint32_t z_i = (uint32_t) z(gen);
                        s2_idx = s1_idx + z_i;
                    }
                } else {
                    do {
                        s2_idx = rand_step(gen);
                    } while (s1_idx == s2_idx);
                }

                assert(s1_idx < p.step_count);
                assert(s2_idx < p.step_count);

                uint32_t n1_id = p.elements[s1_idx].node_id;
                int64_t n1_pos_in_path = p.elements[s1_idx].pos;
                bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
                n1_pos_in_path = std::abs(n1_pos_in_path);

                uint32_t n2_id = p.elements[s2_idx].node_id;
                int64_t n2_pos_in_path = p.elements[s2_idx].pos;
                bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
                n2_pos_in_path = std::abs(n2_pos_in_path);

                uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
                bool n1_use_other_end = flip(gen);
                if (n1_use_other_end) {
                    n1_pos_in_path += uint64_t{n1_seq_length};
                    n1_use_other_end = !n1_is_rev;
                } else {
                    n1_use_other_end = n1_is_rev;
                }

                uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
                bool n2_use_other_end = flip(gen);
                if (n2_use_other_end) {
                    n2_pos_in_path += uint64_t{n2_seq_length};
                    n2_use_other_end = !n2_is_rev;
                } else {
                    n2_use_other_end = n2_is_rev;
                }

                double term_dist = std::abs(static_cast<double>(n1_pos_in_path) - static_cast<double>(n2_pos_in_path));

                if (term_dist < 1e-9) {
                    term_dist = 1e-9;
                }

                double w_ij = 1.0 / term_dist;

                double mu = etas[iter] * w_ij;
                if (mu > 1.0) {
                    mu = 1.0;
                }

                double d_ij = term_dist;

                int n1_offset = n1_use_other_end? 2: 0;
                int n2_offset = n2_use_other_end? 2: 0;

                std::atomic<float> *x1 = &node_data.nodes[n1_id].coords[n1_offset];
                std::atomic<float> *x2 = &node_data.nodes[n2_id].coords[n2_offset];
                std::atomic<float> *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
                std::atomic<float> *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];

                double dx = float(x1->load() - x2->load());
                double dy = float(y1->load() - y2->load());

                if (dx == 0.0) {
                    dx = 1e-9;
                }

                double mag = sqrt(dx * dx + dy * dy);
                double delta = mu * (mag - d_ij) / 2.0;

                double r = delta / mag;
                double r_x = r * dx;
                double r_y = r * dy;

                x1->store(x1->load() - float(r_x));
                y1->store(y1->load() - float(r_y));
                x2->store(x2->load() + float(r_x));
                y2->store(y2->load() + float(r_y));
            }
        }
    }
}


void cuda_layout(layout_config_t config, const odgi::graph_t &graph, std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y) {

#ifdef cuda_layout_profiling
    auto start = std::chrono::high_resolution_clock::now();
#endif

    std::cout << "Hello world from CUDA host" << std::endl;
    std::cout << "iter_max: " << config.iter_max << std::endl;
    std::cout << "first_cooling_iteration: " << config.first_cooling_iteration << std::endl;
    std::cout << "min_term_updates: " << config.min_term_updates << std::endl;
    std::cout << "size of node_t: " << sizeof(node_t) << std::endl;
    std::cout << "theta: " << config.theta << std::endl;

    // First count how many paths and steps we have
    uint32_t node_count = graph.get_node_count();
    std::cout << "node_count: " << node_count << std::endl;
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() == node_count);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    uint32_t path_count = graph.get_path_count();
    std::cout << "path_count: " << path_count << std::endl;
    
    // Count total path steps
    uint64_t total_path_steps = 0;
    vector<odgi::path_handle_t> path_handles{};
    path_handles.reserve(path_count);
    graph.for_each_path_handle(
        [&] (const odgi::path_handle_t& p) {
            path_handles.push_back(p);
            total_path_steps += graph.get_step_count(p);
        });
    std::cout << "total_path_steps: " << total_path_steps << std::endl;

    // Get path lengths first
    vector<uint32_t> path_lengths(path_count);
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        path_lengths[path_idx] = graph.get_step_count(p);
    }
    
    // Create initial path assignments based on their sizes
    std::vector<int> path_numa_assignments(path_count, -1);
    uint64_t numa0_steps = 0;
    uint64_t numa1_steps = 0;
        
    // Greedily assign paths to NUMA nodes to balance workload
    // Keep adding paths to NUMA0 until we exceed half of total steps
    uint64_t half_total_steps = total_path_steps / 2;
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        uint32_t step_count = path_lengths[path_idx];
        
        if (numa0_steps < half_total_steps) {
            path_numa_assignments[path_idx] = 0;
            numa0_steps += step_count;
        } else {
            path_numa_assignments[path_idx] = 1;
            numa1_steps += step_count;
        }
    }
    
    std::cout << "NUMA path assignments:" << std::endl;
    std::cout << "  NUMA node 0: " << std::count(path_numa_assignments.begin(), path_numa_assignments.end(), 0) 
              << " paths, " << numa0_steps << " steps (" 
              << (double)numa0_steps / total_path_steps * 100.0 << "%)" << std::endl;
    std::cout << "  NUMA node 1: " << std::count(path_numa_assignments.begin(), path_numa_assignments.end(), 1) 
              << " paths, " << numa1_steps << " steps ("
              << (double)numa1_steps / total_path_steps * 100.0 << "%)" << std::endl;

    std::cout << "Paths on NUMA node 0: ";
    for (int i = 0; i < path_count; i++) {
        if (path_numa_assignments[i] == 0) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    std::cout << "Paths on NUMA node 1: ";
    for (int i = 0; i < path_count; i++) {
        if (path_numa_assignments[i] == 1) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    // Now determine node assignments based on path usage frequency
    std::cout << "Calculating node NUMA assignments based on usage frequency..." << std::endl;
    
    // Create a vector to count how often each node is accessed by paths from each NUMA node
    std::vector<uint32_t> node_numa0_count(node_count, 0);
    std::vector<uint32_t> node_numa1_count(node_count, 0);
    std::vector<int> node_numa_assignments(node_count, -1);
    
    // First, we need to allocate path_data for the counting
    cuda::path_data_t temp_path_data;
    temp_path_data.path_count = path_count;
    temp_path_data.total_path_steps = total_path_steps;
    temp_path_data.paths = (cuda::path_t*) malloc(path_count * sizeof(cuda::path_t));
    temp_path_data.element_array = (path_element_t*) malloc(total_path_steps * sizeof(path_element_t));
    
    // Initialize path data for counting (simplified without NUMA awareness)
    uint32_t first_step_counter = 0;
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        temp_path_data.paths[path_idx].step_count = path_lengths[path_idx];
        temp_path_data.paths[path_idx].first_step_in_path = first_step_counter;
        
        if (path_lengths[path_idx] == 0) {
            temp_path_data.paths[path_idx].elements = NULL;
        } else {
            path_element_t *cur_path = &temp_path_data.element_array[first_step_counter];
            temp_path_data.paths[path_idx].elements = cur_path;
            
            odgi::path_handle_t p = path_handles[path_idx];
            odgi::step_handle_t s = graph.path_begin(p);
            int64_t pos = 1;
            
            for (int step_idx = 0; step_idx < path_lengths[path_idx]; step_idx++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                
                uint32_t node_id = graph.get_id(h) - 1;
                cur_path[step_idx].node_id = node_id;
                
                // Count this node access for the NUMA node the path is assigned to
                if (path_numa_assignments[path_idx] == 0) {
                    node_numa0_count[node_id]++;
                } else {
                    node_numa1_count[node_id]++;
                }
                
                // Move to next step
                if (graph.has_next_step(s)) {
                    s = graph.get_next_step(s);
                } else if (!(step_idx == path_lengths[path_idx]-1)) {
                    std::cerr << "Error: Path " << path_idx << " has fewer steps than expected" << std::endl;
                    break;
                }
            }
        }
        
        first_step_counter += path_lengths[path_idx];
    }
    
    // Now assign each node to the NUMA node that uses it more frequently
    for (uint32_t n = 0; n < node_count; n++) {
        if (node_numa0_count[n] >= node_numa1_count[n]) {
            node_numa_assignments[n] = 0;
        } else {
            node_numa_assignments[n] = 1;
        }
    }
    
    // Calculate and print statistics about node assignments
    uint32_t numa0_nodes = std::count(node_numa_assignments.begin(), node_numa_assignments.end(), 0);
    uint32_t numa1_nodes = std::count(node_numa_assignments.begin(), node_numa_assignments.end(), 1);
    
    std::cout << "Node NUMA assignments:" << std::endl;
    std::cout << "  NUMA node 0: " << numa0_nodes << " nodes (" 
              << (double)numa0_nodes / node_count * 100.0 << "%)" << std::endl;
    std::cout << "  NUMA node 1: " << numa1_nodes << " nodes ("
              << (double)numa1_nodes / node_count * 100.0 << "%)" << std::endl;
    
    // Clean up temporary path data
    free(temp_path_data.paths);
    free(temp_path_data.element_array);

    // Now allocate memory with NUMA awareness
    // 1. Create eta array (small, not NUMA critical)
    double *etas = (double*) malloc(config.iter_max * sizeof(double));
    const int32_t iter_max = config.iter_max;
    const int32_t iter_with_max_learning_rate = config.iter_with_max_learning_rate;
    const double w_max = 1.0;
    const double eps = config.eps;
    const double eta_max = config.eta_max;
    const double eta_min = eps / w_max;
    const double lambda = log(eta_max / eta_min) / ((double) iter_max - 1);
    for (int32_t i = 0; i < config.iter_max; i++) {
        double eta = eta_max * exp(-lambda * (std::abs(i - iter_with_max_learning_rate)));
        etas[i] = isnan(eta)? eta_min : eta;
    }

    // 2. Allocate node and path data structures
    // We'll initialize these with NUMA awareness
    cuda::node_data_t node_data;
    node_data.node_count = node_count;
    node_data.nodes = (cuda::node_t*) malloc(node_count * sizeof(cuda::node_t));
    
    cuda::path_data_t path_data;
    path_data.path_count = path_count;
    path_data.total_path_steps = total_path_steps;
    path_data.paths = (cuda::path_t*) malloc(path_count * sizeof(cuda::path_t));
    
    // Pre-calculate first_step positions (just a counter, no NUMA concerns)
    first_step_counter = 0;
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        path_data.paths[path_idx].step_count = path_lengths[path_idx];
        path_data.paths[path_idx].first_step_in_path = first_step_counter;
        first_step_counter += path_lengths[path_idx];
    }
    
    // Allocate element array - will be touched with NUMA awareness
    path_data.element_array = (path_element_t*) malloc(total_path_steps * sizeof(path_element_t));

    // NUMA-aware initialization for both node_data and path_data.element_array
    std::cout << "Performing NUMA-aware initialization..." << std::endl;
    
    // For path data: instead of relying on thread-iteration assignment,
    // we will explicitly handle NUMA assignments
    
    // First initialize all nodes with appropriate NUMA awareness
#pragma omp parallel num_threads(config.nthreads)
    {
        int tid = omp_get_thread_num();
        int numa_node = (tid < config.nthreads / 2) ? 0 : 1;
        
        // Bind thread to the right NUMA node
        #ifdef _OPENMP
        cpu_set_t mask;
        CPU_ZERO(&mask);
        
        int core_id;
        if (numa_node == 0) {
            // Bind to a core on NUMA node 0 (0-23)
            core_id = tid;
        } else {
            // Bind to a core on NUMA node 1 (24-47)
            core_id = 24 + (tid - config.nthreads/2);
        }
        
        CPU_SET(core_id, &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
        #endif
        
        // Initialize nodes based on their NUMA assignment
#pragma omp for
        for (uint32_t n = 0; n < node_count; n++) {
            // Only touch nodes that belong to this NUMA node
            if (node_numa_assignments[n] == numa_node) {
                // Get handle for this node
                const handlegraph::handle_t h = graph.get_handle(n + 1, false);
                
                // First-touch: initialize node sequence length
                node_data.nodes[n].seq_length = graph.get_length(h);
                
                // First-touch: initialize node coordinates
                node_data.nodes[n].coords[0].store(float(X[n * 2].load()));
                node_data.nodes[n].coords[1].store(float(Y[n * 2].load()));
                node_data.nodes[n].coords[2].store(float(X[n * 2 + 1].load()));
                node_data.nodes[n].coords[3].store(float(Y[n * 2 + 1].load()));
            }
        }
    }
    
    // 2. Now handle path initialization with explicit NUMA management
    // We'll use two separate parallel sections, one for each NUMA node
    
    // First initialize NUMA node 0 paths
#pragma omp parallel num_threads(config.nthreads/2)
    {
        int tid = omp_get_thread_num();
        int numa_node = 0;
        
        // Bind thread to NUMA node 0
        #ifdef _OPENMP
        cpu_set_t mask;
        CPU_ZERO(&mask);
        int core_id = tid;
        CPU_SET(core_id, &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
        #endif
        
        // Initialize paths assigned to NUMA node 0
#pragma omp for
        for (int path_idx = 0; path_idx < path_count; path_idx++) {
            if (path_numa_assignments[path_idx] == 0) {
                // First-touch path metadata on NUMA 0
                volatile uint32_t step_count = path_data.paths[path_idx].step_count;
                volatile uint64_t first_step = path_data.paths[path_idx].first_step_in_path;
                
                odgi::path_handle_t p = path_handles[path_idx];
                step_count = path_data.paths[path_idx].step_count;
                uint32_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;
                
                if (step_count == 0) {
                    path_data.paths[path_idx].elements = NULL;
                    std::cout << "Path " << path_idx << " has 0 steps" << std::endl;
                } else {
                    path_element_t *cur_path = &path_data.element_array[first_step_in_path];
                    path_data.paths[path_idx].elements = cur_path;
                    
                    odgi::step_handle_t s = graph.path_begin(p);
                    int64_t pos = 1;
                    
                    // Initialize all elements in this path
                    for (int step_idx = 0; step_idx < step_count; step_idx++) {
                        odgi::handle_t h = graph.get_handle_of_step(s);
                        
                        cur_path[step_idx].node_id = graph.get_id(h) - 1;
                        cur_path[step_idx].pidx = uint32_t(path_idx);
                        
                        // Store position negative when handle reverse
                        if (graph.get_is_reverse(h)) {
                            cur_path[step_idx].pos = -pos;
                        } else {
                            cur_path[step_idx].pos = pos;
                        }
                        pos += graph.get_length(h);
                        
                        // Get next step
                        if (graph.has_next_step(s)) {
                            s = graph.get_next_step(s);
                        } else if (!(step_idx == step_count-1)) {
                            // Should never be reached
                            std::cout << "Error: Here should be another step" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    // Now initialize NUMA node 1 paths
#pragma omp parallel num_threads(config.nthreads/2)
    {
        int tid = omp_get_thread_num();
        int numa_node = 1;
        
        // Bind thread to NUMA node 1
        #ifdef _OPENMP
        cpu_set_t mask;
        CPU_ZERO(&mask);
        int core_id = 24 + tid; // Cores 24+ for NUMA node 1
        CPU_SET(core_id, &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
        #endif
        
        // Initialize paths assigned to NUMA node 1
#pragma omp for
        for (int path_idx = 0; path_idx < path_count; path_idx++) {
            if (path_numa_assignments[path_idx] == 1) {
                // First-touch path metadata on NUMA 1
                volatile uint32_t step_count = path_data.paths[path_idx].step_count;
                volatile uint64_t first_step = path_data.paths[path_idx].first_step_in_path;
                
                
                odgi::path_handle_t p = path_handles[path_idx];
                step_count = path_data.paths[path_idx].step_count;
                uint32_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;
                
                if (step_count == 0) {
                    path_data.paths[path_idx].elements = NULL;
                    std::cout << "Path " << path_idx << " has 0 steps" << std::endl;
                } else {
                    path_element_t *cur_path = &path_data.element_array[first_step_in_path];
                    path_data.paths[path_idx].elements = cur_path;
                    
                    odgi::step_handle_t s = graph.path_begin(p);
                    int64_t pos = 1;
                    
                    // Initialize all elements in this path
                    for (int step_idx = 0; step_idx < step_count; step_idx++) {
                        odgi::handle_t h = graph.get_handle_of_step(s);
                        
                        cur_path[step_idx].node_id = graph.get_id(h) - 1;
                        cur_path[step_idx].pidx = uint32_t(path_idx);
                        
                        // Store position negative when handle reverse
                        if (graph.get_is_reverse(h)) {
                            cur_path[step_idx].pos = -pos;
                        } else {
                            cur_path[step_idx].pos = pos;
                        }
                        pos += graph.get_length(h);
                        
                        // Get next step
                        if (graph.has_next_step(s)) {
                            s = graph.get_next_step(s);
                        } else if (!(step_idx == step_count-1)) {
                            // Should never be reached
                            std::cout << "Error: Here should be another step" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "NUMA-aware initialization complete." << std::endl;

    // cache zipf zetas (small, not NUMA critical)
    auto start_zeta = std::chrono::high_resolution_clock::now();
    double *zetas;
    uint64_t zetas_cnt = ((config.space <= config.space_max)? config.space : (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;
    std::cout << "zetas_cnt: " << zetas_cnt << std::endl;
    std::cout << "space_max: " << config.space_max << std::endl;
    std::cout << "config.space: " << config.space << std::endl;
    std::cout << "config.space_quantization: " << config.space_quantization_step << std::endl;

    zetas = (double*) malloc(zetas_cnt * sizeof(double));
    double zeta_tmp = 0.0;
    for (uint64_t i = 1; i < config.space + 1; i++) {
        zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / i, config.theta);
        if (i <= config.space_max) {
            zetas[i] = zeta_tmp;
        }
        if (i >= config.space_max && (i - config.space_max) % config.space_quantization_step == 0) {
            zetas[config.space_max + 1 + (i - config.space_max) / config.space_quantization_step] = zeta_tmp;
        }
    }
    auto end_zeta = std::chrono::high_resolution_clock::now();
    uint32_t duration_zeta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_zeta - start_zeta).count();
    std::cout << "Zeta precompute took " << duration_zeta_ms << "ms" << std::endl;

    auto start_compute = std::chrono::high_resolution_clock::now();

    // Now analyze the paths to get detailed node distribution info
    // analyze_paths(node_data, path_data, "path_analysis.csv");
    // std::cout << "Detailed path analysis complete.\n" << std::endl;

    // CPU Layout with NUMA-aware path assignments
    cpu_layout(config, etas, zetas, node_data, path_data, path_numa_assignments);

    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "CPU cache-optimized layout compute took " << duration_compute_ms << "ms" << std::endl;

    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &(node_data.nodes[node_idx]);
        // coords[0], coords[1], coords[2], coords[3] are stored consecutively. 
        std::atomic<float> *coords = n->coords;
        // check if coordinates valid (not NaN or infinite)
        for (int i = 0; i < 4; i++) {
            if (!isfinite(coords[i].load())) {
                std::cout << "WARNING: invalid coordiate" << std::endl;
            }
        }
        X[node_idx * 2].store(double(coords[0].load()));
        Y[node_idx * 2].store(double(coords[1].load()));
        X[node_idx * 2 + 1].store(double(coords[2].load()));
        Y[node_idx * 2 + 1].store(double(coords[3].load()));
        //std::cout << "coords of " << node_idx << ": [" << X[node_idx*2] << "; " << Y[node_idx*2] << "] ; [" << X[node_idx*2+1] << "; " << Y[node_idx*2+1] <<"]\n";
    }

    // get rid of CUDA data structures
    free(etas);
    free(node_data.nodes);
    free(path_data.paths);
    free(path_data.element_array);
    free(zetas);

#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout << "CPU cache-optimized layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
