#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <random>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sstream>
#include <iomanip>

#include "odgi.hpp"
#include "XoshiroCpp.hpp"
#include "dirty_zipfian_int_distribution.h"


#define cuda_layout_profiling


namespace cuda {


struct node_t {
    int32_t seq_length;
    float coords[4];
};
struct node_data_t {
    uint32_t node_count;
    node_t *nodes;
};


struct path_element_t {
    uint32_t node_id;
    int64_t pos;    // if position negative: reverse orientation
};

struct path_t {
    uint32_t step_count;
    uint64_t first_step_in_path;  // precomputed position in path
    path_element_t *elements;
};

struct path_data_t {
    uint32_t path_count;
    uint32_t total_path_steps;
    path_t *paths;
};


// TODO: make parameters constant?
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


void cuda_layout(layout_config_t config, const odgi::graph_t &graph, std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y);

}
