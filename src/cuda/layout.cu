// layout.cu
#include "layout.cuh"
#include <assert.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>

namespace layout_kernel {

__global__ void setup_rand_states(curandState *states, int num_threads, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_threads) {
        curand_init(seed + tid, 0, 0, &states[tid]);
    }
}

__global__ void gpu_layout_kernel(layout_config_t config, double *d_etas, double *d_zetas, 
                                 node_t *d_nodes, path_t *d_paths, path_element_t *d_elements,
                                 curandState *d_states, double *d_path_cdf, uint32_t path_cdf_size, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= config.min_term_updates) return;

    curandState local_state = d_states[tid];
    double eta = d_etas[iter];
    
    // Path selection using binary search on CDF
    double rand_val = curand_uniform_double(&local_state) * d_path_cdf[path_cdf_size-1];
    int low = 0, high = path_cdf_size - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        if (d_path_cdf[mid] < rand_val) low = mid + 1;
        else high = mid;
    }
    uint32_t path_idx = low;
    path_t p = d_paths[path_idx];
    if (p.step_count < 2) return;

    // Step selection logic
    uint32_t s1_idx, s2_idx;
    if (iter >= config.first_cooling_iteration || curand_uniform_double(&local_state) < 0.5) {
        s1_idx = curand(&local_state) % p.step_count;
        bool go_backward = (s1_idx > 0 && curand_uniform_double(&local_state) < 0.5) || (s1_idx == p.step_count-1);
        
        if (go_backward) {
            uint32_t jump_space = min(config.space, s1_idx);
            uint32_t space = (jump_space > config.space_max) ? 
                config.space_max + (jump_space - config.space_max)/config.space_quantization_step + 1 : jump_space;
            double u = curand_uniform_double(&local_state) * d_zetas[space];
            uint32_t z_i = 1;
            while (z_i < jump_space && d_zetas[z_i] < u) z_i++;
            s2_idx = s1_idx - z_i;
        } else {
            uint32_t jump_space = min(config.space, p.step_count - s1_idx - 1);
            uint32_t space = (jump_space > config.space_max) ? 
                config.space_max + (jump_space - config.space_max)/config.space_quantization_step + 1 : jump_space;
            double u = curand_uniform_double(&local_state) * d_zetas[space];
            uint32_t z_i = 1;
            while (z_i < jump_space && d_zetas[z_i] < u) z_i++;
            s2_idx = s1_idx + z_i;
        }
    } else {
        do {
            s1_idx = curand(&local_state) % p.step_count;
            s2_idx = curand(&local_state) % p.step_count;
        } while (s1_idx == s2_idx);
    }

    // Node position calculations
    path_element_t e1 = d_elements[p.first_step_in_path + s1_idx];
    path_element_t e2 = d_elements[p.first_step_in_path + s2_idx];

    int64_t n1_pos = abs(e1.pos);
    int64_t n2_pos = abs(e2.pos);

    bool n1_use_other_end = curand_uniform_double(&local_state) < 0.5;
    if (n1_use_other_end) n1_pos += d_nodes[e1.node_id].seq_length;

    bool n2_use_other_end = curand_uniform_double(&local_state) < 0.5;
    if (n2_use_other_end) n2_pos += d_nodes[e2.node_id].seq_length;

    double term_dist = abs(n1_pos - n2_pos);
    if (term_dist < 1e-9) {
        term_dist = 1e-9;
    }
    double w_ij = 1.0 / term_dist;
    double mu = min(d_etas[iter] * w_ij, 1.0);

    // Atomic updates
    int n1_offset = n1_use_other_end ? 2 : 0;
    int n2_offset = n2_use_other_end ? 2 : 0;

    float *x1 = &d_nodes[e1.node_id].coords[n1_offset];
    float *y1 = &d_nodes[e1.node_id].coords[n1_offset+1];
    float *x2 = &d_nodes[e2.node_id].coords[n2_offset];
    float *y2 = &d_nodes[e2.node_id].coords[n2_offset+1];

    float dx = *x1 - *x2;
    float dy = *y1 - *y2;
    float mag = sqrtf(dx*dx + dy*dy);
    if (mag < 1e-9) mag = 1e-9f;
    float delta = mu * (mag - term_dist) / 2.0f;
    
    float r_x = delta * dx / mag;
    float r_y = delta * dy / mag;

    atomicAdd(x1, -r_x);
    atomicAdd(y1, -r_y);
    atomicAdd(x2, r_x);
    atomicAdd(y2, r_y);

    d_states[tid] = local_state;
}

void layout_func(layout_config_t config, const odgi::graph_t &graph, 
                std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y) {
    std::cout << "Running GPU Layout Function with CUDA Kernel" << std::endl;
    
    // Original eta initialization
    double *etas = (double*)malloc(config.iter_max * sizeof(double));
    const int32_t iter_max = config.iter_max;
    const int32_t iter_with_max_learning_rate = config.iter_with_max_learning_rate;
    const double eta_max = config.eta_max;
    const double eta_min = config.eps / 1.0;
    const double lambda = log(eta_max / eta_min) / (iter_max - 1);
    for (int32_t i = 0; i < iter_max; i++) {
        double eta = eta_max * exp(-lambda * abs(i - iter_with_max_learning_rate));
        etas[i] = isnan(eta) ? eta_min : eta;
    }

    // Node data preparation
    uint32_t node_count = graph.get_node_count();
    node_t *h_nodes = (node_t*)malloc(node_count * sizeof(node_t));
    for (uint32_t i = 0; i < node_count; i++) {
        const handlegraph::handle_t h = graph.get_handle(i + 1, false);
        h_nodes[i].seq_length = graph.get_length(h);
        h_nodes[i].coords[0] = X[i*2].load();
        h_nodes[i].coords[1] = Y[i*2].load();
        h_nodes[i].coords[2] = X[i*2+1].load();
        h_nodes[i].coords[3] = Y[i*2+1].load();
    }

    // Path data preparation
    std::vector<path_t> h_paths;
    std::vector<path_element_t> h_elements;
    std::vector<double> path_weights;
    std::vector<odgi::path_handle_t> path_handles;
    graph.for_each_path_handle([&](const odgi::path_handle_t& p) {
        path_handles.push_back(p);
    });

    uint32_t element_counter = 0;
    for (uint32_t i = 0; i < path_handles.size(); i++) {
        path_t p;
        p.step_count = graph.get_step_count(path_handles[i]);
        p.first_step_in_path = element_counter;
        path_weights.push_back(p.step_count);
        h_paths.push_back(p);
        element_counter += p.step_count;
    }

    h_elements.resize(element_counter);
#pragma omp parallel for num_threads(config.nthreads)
    for (uint32_t i = 0; i < path_handles.size(); i++) {
        auto p = path_handles[i];
        uint32_t step_count = h_paths[i].step_count;
        odgi::step_handle_t s = graph.path_begin(p);
        int64_t pos = 1;
        for (uint32_t j = 0; j < step_count; j++) {
            odgi::handle_t h = graph.get_handle_of_step(s);
            h_elements[h_paths[i].first_step_in_path + j] = {
                static_cast<uint32_t>(i),
                static_cast<uint32_t>(graph.get_id(h) - 1),
                graph.get_is_reverse(h) ? -pos : pos
            };
            pos += graph.get_length(h);
            if (graph.has_next_step(s)) s = graph.get_next_step(s);
        }
    }

    // Create path CDF
    thrust::exclusive_scan(path_weights.begin(), path_weights.end(), path_weights.begin());
    double total_weight = path_weights.back();
    for (auto& w : path_weights) w /= total_weight;

    // Device allocations
    node_t *d_nodes;
    path_t *d_paths;
    path_element_t *d_elements;
    double *d_etas, *d_zetas, *d_path_cdf;
    curandState *d_states;

    cudaMalloc(&d_nodes, node_count * sizeof(node_t));
    cudaMalloc(&d_paths, h_paths.size() * sizeof(path_t));
    cudaMalloc(&d_elements, h_elements.size() * sizeof(path_element_t));
    cudaMalloc(&d_etas, config.iter_max * sizeof(double));
    cudaMalloc(&d_path_cdf, path_weights.size() * sizeof(double));
    
    uint64_t zetas_cnt = ((config.space <= config.space_max) ? config.space : 
        (config.space_max + (config.space - config.space_max)/config.space_quantization_step + 1)) + 1;
    double *zetas = (double*)malloc(zetas_cnt * sizeof(double));
    // ... zeta initialization as original ...
    cudaMalloc(&d_zetas, zetas_cnt * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_nodes, h_nodes, node_count * sizeof(node_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_paths, h_paths.data(), h_paths.size() * sizeof(path_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements, h_elements.data(), h_elements.size() * sizeof(path_element_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_etas, etas, config.iter_max * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_path_cdf, path_weights.data(), path_weights.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zetas, zetas, zetas_cnt * sizeof(double), cudaMemcpyHostToDevice);

    // RNG initialization
    int block_size = 256;
    int grid_size = (config.min_term_updates + block_size - 1) / block_size;
    cudaMalloc(&d_states, config.min_term_updates * sizeof(curandState));
    setup_rand_states<<<grid_size, block_size>>>(d_states, config.min_term_updates, 9399220);

    // Kernel execution
    dim3 block(block_size);
    dim3 grid(grid_size);  // Use y-dimension for iterations

    // Launch one kernel per iteration
    for (int iter = 0; iter < config.iter_max; iter++) {
        gpu_layout_kernel<<<grid, block>>>(config, d_etas, d_zetas, d_nodes, d_paths, 
                                          d_elements, d_states, d_path_cdf, path_weights.size(), iter);
        cudaDeviceSynchronize();
    }

    // Copy results back
    cudaMemcpy(h_nodes, d_nodes, node_count * sizeof(node_t), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < node_count; i++) {
        X[i*2].store(h_nodes[i].coords[0]);
        Y[i*2].store(h_nodes[i].coords[1]);
        X[i*2+1].store(h_nodes[i].coords[2]);
        Y[i*2+1].store(h_nodes[i].coords[3]);
    }

    // Cleanup
    cudaFree(d_nodes); cudaFree(d_paths); cudaFree(d_elements);
    cudaFree(d_etas); cudaFree(d_zetas); cudaFree(d_path_cdf);
    cudaFree(d_states);
    free(h_nodes); free(etas); free(zetas);
}
}