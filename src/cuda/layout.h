#pragma once
#include <iostream>
#include <chrono>
#include <vector>

#include "odgi.hpp"


#define cuda_layout_profiling


namespace cuda {


struct node_t {
    int32_t seq_length;
    double coords[4];
};


void cuda_layout(const odgi::graph_t &graph, std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y);

}
