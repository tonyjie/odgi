#include "layout.h"
#include <cuda.h>


namespace cuda {

__global__ void cuda_device_layout(cuda::layout_config_t config, double *etas, cuda::node_data_t &node_data, cuda::path_data_t &path_data) {
    int32_t device_id = blockIdx.x * blockDim.x + threadIdx.x;
    node_t *n = &node_data.nodes[device_id];

    //printf("Hello World from CUDA device: %i %f %f %f %f\n", n->seq_length, n->coords[0] ,n->coords[1] ,n->coords[2] ,n->coords[3]);
    printf("CUDA device %i: step_count: %i\n", device_id, path_data.paths[device_id].step_count);
}


void cpu_layout(cuda::layout_config_t config, double *etas, cuda::node_data_t &node_data, cuda::path_data_t &path_data) {
    std::cout << "cuda cpu layout" << std::endl;
    int nbr_threads = 40;
    std::vector<uint64_t> path_dist;
    for (int p = 0; p < path_data.path_count; p++) {
        path_dist.push_back(uint64_t(path_data.paths[p].step_count));
    }

#pragma omp parallel num_threads(nbr_threads)
    {
        int tid = omp_get_thread_num();

        XoshiroCpp::Xoshiro256Plus gen(9399220 + tid);
        std::uniform_int_distribution<uint64_t> flip(0, 1);
        std::discrete_distribution<> rand_path(path_dist.begin(), path_dist.end());

        const int steps_per_thread = config.min_term_updates / nbr_threads;

        for (int iter = 0; iter < config.iter_max; iter++ ) {
            // synchronize all threads before each iteration
#pragma omp barrier
            for (int step = 0; step < steps_per_thread; step++ ) {
                // get path
                uint32_t path_idx = rand_path(gen);
                path_t p = path_data.paths[path_idx];

                std::uniform_int_distribution<uint32_t> rand_step(0, p.step_count-1);

                uint32_t s1_idx = rand_step(gen);
                // TODO implement cooling with zipf distribution
                uint32_t s2_idx;
                do {
                    s2_idx = rand_step(gen);
                } while (s1_idx == s2_idx);

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

                double *x1 = &node_data.nodes[n1_id].coords[n1_offset];
                double *x2 = &node_data.nodes[n2_id].coords[n2_offset];
                double *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
                double *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];

                double dx = *x1 - *x2;
                double dy = *y1 - *y2;
                if (dx == 0.0) {
                    dx = 1e-9;
                }

                double mag = sqrt(dx * dx + dy * dy);
                double delta = mu * (mag - d_ij) / 2.0;
                //double delta_abs = std::abs(delta);

                // TODO implement delta max stop functionality
                double r = delta / mag;
                double r_x = r * dx;
                double r_y = r * dy;

                *x1 -= r_x;
                *y1 -= r_y;
                *x2 += r_x;
                *y2 += r_y;
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
    std::cout << "min_term_updates: " << config.min_term_updates << std::endl;
    std::cout << "size of node_t: " << sizeof(node_t) << std::endl;

    // create eta array
    double *etas;
    cudaMallocManaged(&etas, config.iter_max * sizeof(double));

    const int32_t iter_max = config.iter_max;
    const int32_t iter_with_max_learning_rate = config.iter_with_max_learning_rate;
    const double w_max = 1.0;
    const double eps = config.eps;
    const double eta_max = config.eta_max;
    const double eta_min = eps / w_max;
    const double lambda = log(eta_max / eta_min) / ((double) iter_max - 1);
    for (int32_t i = 0; i < config.iter_max; i++) {
        etas[i] = eta_max * exp(-lambda * (std::abs(i - iter_with_max_learning_rate)));
    }


    // create node data structure
    // consisting of sequence length and coords
    uint32_t node_count = graph.get_node_count();
    std::cout << "node_count: " << node_count << std::endl;
    // TODO handle cases when min_node_id != 1
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() == node_count);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    cuda::node_data_t node_data;
    node_data.node_count = node_count;
    cudaMallocManaged(&node_data.nodes, node_count * sizeof(cuda::node_t));
    // TODO parallelise with openmp
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        assert(graph.has_node(node_idx));
        cuda::node_t *n_tmp = &node_data.nodes[node_idx];

        // sequence length
        const handlegraph::handle_t h = graph.get_handle(node_idx + 1, false);
        // NOTE: unable store orientation (reverse), since this information is path dependent
        n_tmp->seq_length = graph.get_length(h);

        // copy random coordinates
        n_tmp->coords[0] = X[node_idx * 2].load();
        n_tmp->coords[1] = Y[node_idx * 2].load();
        n_tmp->coords[2] = X[node_idx * 2 + 1].load();
        n_tmp->coords[3] = Y[node_idx * 2 + 1].load();
    }


    // create path data structure
    uint32_t path_count = graph.get_path_count();
    cuda::path_data_t path_data;
    path_data.path_count = path_count;
    cudaMallocManaged(&path_data.paths, node_count * sizeof(cuda::path_t));

    vector<odgi::path_handle_t> path_handles{};
    path_handles.reserve(path_count);
    graph.for_each_path_handle(
        [&] (const odgi::path_handle_t& p) {
            path_handles.push_back(p);
        });

    // TODO parallelise with openmp
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        std::cout << graph.get_path_name(p) << ": " << graph.get_step_count(p) << std::endl;

        int step_count = graph.get_step_count(p);
        path_data.paths[path_idx].step_count = step_count;

        cuda::path_element_t *cur_path;
        cudaMallocManaged(&cur_path, step_count * sizeof(path_element_t));
        path_data.paths[path_idx].elements = cur_path;

        odgi::step_handle_t s = graph.path_begin(p);
        int64_t pos = 1;
        // Iterate through path
        for (int step_idx = 0; step_idx < step_count; step_idx++) {
            odgi::handle_t h = graph.get_handle_of_step(s);
            //std::cout << graph.get_id(h) << std::endl;

            cur_path[step_idx].node_id = graph.get_id(h);
            // store position negative when handle reverse
            if (graph.get_is_reverse(h)) {
                cur_path[step_idx].pos = -pos;
            } else {
                cur_path[step_idx].pos = pos;
            }
            pos += graph.get_length(h);

            // get next step
            if (graph.has_next_step(s)) {
                s = graph.get_next_step(s);
            } else if (!(step_idx == step_count-1)) {
                // should never be reached
                std::cout << "Error: Here should be another step" << std::endl;
            }
        }
    }



//#define USE_CUDA
#ifdef USE_CUDA
    cuda_device_layout<<<1,10>>>(config, etas, node_data, path_data);
    cudaDeviceSynchronize();
#else
    auto start_compute = std::chrono::high_resolution_clock::now();
    cpu_layout(config, etas, node_data, path_data);
    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "CUDA layout compute took " << duration_compute_ms << "ms" << std::endl;
#endif



    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &node_data.nodes[node_idx];
        //assert(n->coords[0] == X[node_idx * 2].load());
        //assert(n->coords[1] == Y[node_idx * 2].load());
        //assert(n->coords[2] == X[node_idx * 2 + 1].load());
        //assert(n->coords[3] == Y[node_idx * 2 + 1].load());

        X[node_idx * 2].store(n->coords[0]);
        Y[node_idx * 2].store(n->coords[1]);
        X[node_idx * 2 + 1].store(n->coords[2]);
        Y[node_idx * 2 + 1].store(n->coords[3]);
    }


    // get rid of CUDA data structures
    cudaFree(node_data.nodes);
    for (int i = 0; i < path_count; i++) {
        cudaFree(path_data.paths[i].elements);
    }
    cudaFree(path_data.paths);


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CUDA layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
