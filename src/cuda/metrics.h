#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <random>
#include <math.h>
#include <sstream>
#include <iomanip>
#include <curand.h>
#include <curand_kernel.h>

#include "odgi.hpp"
#include "algorithms/layout.hpp"


namespace cuda { 

#define BLOCK_SIZE 1024
#define SM_COUNT 84
#define SAMPLE_FACTOR 100 // for sampled-path-stress

struct Point 
{ 
	double x; 
	double y; 
}; 

struct __attribute__((aligned(8))) node_t {
    float coords[4]; // coords[0]: start.x, coords[1]: start.y, coords[2]: end.x, coords[3]: end.y
    int32_t seq_length;
};

struct node_data_t {
    uint32_t node_count;
    node_t *nodes;
};

struct __attribute__((aligned(8))) path_element_t {
    uint32_t pidx;
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
    uint64_t total_path_steps;
    path_t *paths;
    path_element_t *element_array;
};

struct curandStateXORWOWCoalesced_t {
    unsigned int d[BLOCK_SIZE];
    unsigned int w0[BLOCK_SIZE];
    unsigned int w1[BLOCK_SIZE];
    unsigned int w2[BLOCK_SIZE];
    unsigned int w3[BLOCK_SIZE];
    unsigned int w4[BLOCK_SIZE];
};
typedef struct curandStateXORWOWCoalesced_t curandStateCoalesced_t;


void cuda_node_crossing(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout);

void cuda_sampled_path_stress(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout, int nthreads);

void cuda_all_pair_path_stress(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout, int nthreads);

} // namespace cuda