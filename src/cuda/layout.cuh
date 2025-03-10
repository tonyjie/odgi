// layout.cuh
#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <random>
#include <math.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <atomic>

#include "odgi.hpp"
#include "XoshiroCpp.hpp"
#include "dirty_zipfian_int_distribution.h"

namespace layout_kernel {
    struct node_t {
        float coords[4];
        int32_t seq_length;
    };

    struct path_element_t {
        uint32_t pidx;
        uint32_t node_id;
        int64_t pos;
    };

    struct path_t {
        uint32_t step_count;
        uint64_t first_step_in_path;
        path_element_t *elements;
    };

    struct layout_config_t {
        uint64_t iter_max;
        uint64_t min_term_updates;
        double eta_max;
        double eps;
        int32_t iter_with_max_learning_rate;
        uint32_t first_cooling_iteration;
        double theta;
        uint32_t space;
        uint32_t space_max;
        uint32_t space_quantization_step;
        int nthreads;
    };

    __global__ void gpu_layout_kernel(layout_config_t config, double *d_etas, double *d_zetas, 
                                     node_t *d_nodes, path_t *d_paths, path_element_t *d_elements,
                                     curandState *d_states, double *d_path_cdf, uint32_t path_cdf_size);
    
    __global__ void setup_rand_states(curandState *states, int num_threads, unsigned int seed);
    
    void layout_func(layout_config_t config, const odgi::graph_t &graph, 
                    std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y);
}