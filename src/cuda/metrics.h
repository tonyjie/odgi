#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <random>
#include <math.h>
#include <sstream>
#include <iomanip>

#include "odgi.hpp"
#include "algorithms/layout.hpp"


namespace cuda { 

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


void cuda_node_crossing(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout);

void cuda_path_stress(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout);

} // namespace cuda