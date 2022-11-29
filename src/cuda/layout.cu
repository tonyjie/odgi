#include "layout.h"
#include <cuda.h>


namespace cuda {

__global__ void cuda_device_init(curandState *rnd_state) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(42, tid, 0, &rnd_state[threadIdx.x]);
}

__global__ void cuda_device_layout(int iter, cuda::layout_config_t config, curandState *rnd_state, double eta, cuda::node_data_t node_data, cuda::path_data_t path_data, int *counter) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // get path
    uint32_t path_idx = blockIdx.x % path_data.path_count;
    path_t p = path_data.paths[path_idx];

    uint32_t s1_idx = (uint32_t)(ceil((curand_uniform(rnd_state + threadIdx.x)*float(p.step_count))) - 1.0);

    // TODO: implement cooling with zipf distribution
    uint32_t s2_idx;
    do {
        s2_idx = (uint32_t)(ceil((curand_uniform(rnd_state + threadIdx.x)*float(p.step_count))) - 1.0);
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
    bool n1_use_other_end = (curand_uniform(rnd_state + threadIdx.x) <= 0.5)? true: false;
    if (n1_use_other_end) {
        n1_pos_in_path += uint64_t{n1_seq_length};
        n1_use_other_end = !n1_is_rev;
    } else {
        n1_use_other_end = n1_is_rev;
    }

    uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
    bool n2_use_other_end = (curand_uniform(rnd_state + threadIdx.x) <= 0.5)? true: false;
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

    double mu = eta * w_ij;
    if (mu > 1.0) {
        mu = 1.0;
    }

    double d_ij = term_dist;

    int n1_offset = n1_use_other_end? 2: 0;
    int n2_offset = n2_use_other_end? 2: 0;

    float *x1 = &node_data.nodes[n1_id].coords[n1_offset];
    float *x2 = &node_data.nodes[n2_id].coords[n2_offset];
    float *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
    float *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];

    double dx = double(*x1 - *x2);
    double dy = double(*y1 - *y2);

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

    atomicAdd(x1, -r_x);
    atomicAdd(y1, -r_y);
    atomicAdd(x2, r_x);
    atomicAdd(y2, r_y);

    atomicAdd(counter, 1);
}


void cpu_layout(cuda::layout_config_t config, double *etas, cuda::node_data_t &node_data, cuda::path_data_t &path_data) {
    int nbr_threads = config.nthreads;
    std::cout << "cuda cpu layout (" << nbr_threads << " threads)" << std::endl;
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

#define profiling
#ifdef profiling
        auto total_duration_dist = std::chrono::duration<double>::zero(); // total time on computing distance: in seconds
        auto total_duration_sgd = std::chrono::duration<double>::zero(); // total time on SGD: in seconds
        // detailed analysis on different parts of Updating Coordinates Part
        auto total_duration_compute_first = std::chrono::duration<double>::zero();
        auto total_duration_load = std::chrono::duration<double>::zero();
        auto total_duration_compute_second = std::chrono::duration<double>::zero();
        auto total_duration_store = std::chrono::duration<double>::zero();
        // detailed analysis on different parts of Getting Distance Part
        auto total_duration_one_step_gen = std::chrono::duration<double>::zero();
        auto total_duration_two_step_gen = std::chrono::duration<double>::zero();
        auto total_duration_get_distance = std::chrono::duration<double>::zero();


        std::chrono::high_resolution_clock::time_point start_dist;
        std::chrono::high_resolution_clock::time_point end_dist;
        std::chrono::high_resolution_clock::time_point start_sgd;
        std::chrono::high_resolution_clock::time_point one_step_gen;
        std::chrono::high_resolution_clock::time_point two_step_gen;

        // detailed analysis on Updating Coordinates part
        std::chrono::high_resolution_clock::time_point before_load;
        std::chrono::high_resolution_clock::time_point after_load;
        std::chrono::high_resolution_clock::time_point before_store;
        std::chrono::high_resolution_clock::time_point after_store;
#endif

        for (int iter = 0; iter < config.iter_max; iter++ ) {
            // synchronize all threads before each iteration
#pragma omp barrier
            for (int step = 0; step < steps_per_thread; step++ ) {
#ifdef profiling
                start_dist = std::chrono::high_resolution_clock::now();
#endif
                // get path
                uint32_t path_idx = rand_path(gen);
                path_t p = path_data.paths[path_idx];

                std::uniform_int_distribution<uint32_t> rand_step(0, p.step_count-1);

                uint32_t s1_idx = rand_step(gen);
                // TODO implement cooling with zipf distribution
#ifdef profiling
                one_step_gen = std::chrono::high_resolution_clock::now();
                total_duration_one_step_gen += std::chrono::duration_cast<std::chrono::nanoseconds>(one_step_gen - start_dist);
#endif
                uint32_t s2_idx;
                do {
                    s2_idx = rand_step(gen);
                } while (s1_idx == s2_idx);

#ifdef profiling
                two_step_gen = std::chrono::high_resolution_clock::now();
                total_duration_two_step_gen += std::chrono::duration_cast<std::chrono::nanoseconds>(two_step_gen - one_step_gen);
#endif
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
#ifdef profiling
                end_dist = std::chrono::high_resolution_clock::now();
                total_duration_get_distance += std::chrono::duration_cast<std::chrono::nanoseconds>(end_dist - two_step_gen);

                total_duration_dist += std::chrono::duration_cast<std::chrono::nanoseconds>(end_dist - start_dist);

                start_sgd = std::chrono::high_resolution_clock::now();
#endif

                double w_ij = 1.0 / term_dist;

                double mu = etas[iter] * w_ij;
                if (mu > 1.0) {
                    mu = 1.0;
                }

                double d_ij = term_dist;

                int n1_offset = n1_use_other_end? 2: 0;
                int n2_offset = n2_use_other_end? 2: 0;

#ifdef profiling
                before_load = std::chrono::high_resolution_clock::now();
                total_duration_compute_first += std::chrono::duration_cast<std::chrono::nanoseconds>(before_load - start_sgd);
#endif
                float *x1 = &node_data.nodes[n1_id].coords[n1_offset];
                float *x2 = &node_data.nodes[n2_id].coords[n2_offset];
                float *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
                float *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];

                double dx = float(*x1 - *x2);
                double dy = float(*y1 - *y2);
#ifdef profiling
                after_load = std::chrono::high_resolution_clock::now();
                total_duration_load += std::chrono::duration_cast<std::chrono::nanoseconds>(after_load - before_load);
#endif
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

#ifdef profiling
                before_store = std::chrono::high_resolution_clock::now();
                total_duration_compute_second += std::chrono::duration_cast<std::chrono::nanoseconds>(before_store - after_load);
#endif
                *x1 -= float(r_x);
                *y1 -= float(r_y);
                *x2 += float(r_x);
                *y2 += float(r_y);
#ifdef profiling
                after_store = std::chrono::high_resolution_clock::now();
                total_duration_store += std::chrono::duration_cast<std::chrono::nanoseconds>(after_store - before_store);
                total_duration_sgd += std::chrono::duration_cast<std::chrono::nanoseconds>(after_store - start_sgd);
#endif
            }
        }

#ifdef profiling
        std::stringstream msg;
        msg << "Thread[" << tid << "]: Dataloading time = " << total_duration_dist.count() << " sec;\t" << "Compute time = " << total_duration_sgd.count() << " sec." << std::endl;

        msg << std::left
            << std::setw(40) << "Getting Distance Part Breakdown: " << std::endl
            << std::setw(20) << "[0] One Step Gen: "
            << std::setw(10) << total_duration_one_step_gen.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[1] Two Steps Gen: "
            << std::setw(10) << total_duration_two_step_gen.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[2] Get Distance: "
            << std::setw(10) << total_duration_get_distance.count()
            << std::setw(10) << " sec."
            << std::endl;

        msg << std::setw(40) << "Updating Coordinate Part Breakdown: " << std::endl
            << std::setw(20) << "[0] First Compute: "
            << std::setw(10) << total_duration_compute_first.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[1] Load Pos: "
            << std::setw(10) << total_duration_load.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[2] Second Compute: "
            << std::setw(10) << total_duration_compute_second.count()
            << std::setw(10)  << " sec;"
            << std::setw(20) << "[3] Update Pos: "
            << std::setw(10) << total_duration_store.count()
            << std::setw(10)  << " sec."
            << std::endl << std::endl;

        std::cerr << msg.str();
#endif

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
        // TODO check for nan values (when iter_max == 1)
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
        n_tmp->coords[0] = float(X[node_idx * 2].load());
        n_tmp->coords[1] = float(Y[node_idx * 2].load());
        n_tmp->coords[2] = float(X[node_idx * 2 + 1].load());
        n_tmp->coords[3] = float(Y[node_idx * 2 + 1].load());
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



    auto start_compute = std::chrono::high_resolution_clock::now();
#define USE_GPU
#ifdef USE_GPU
    std::cout << "cuda gpu layout" << std::endl;

    // TODO use different block_size and/or block_nbr when computing small pangenome to prevent NaN coordinates
    const int block_size = 1024;
    const int block_nbr = (config.min_term_updates + block_size - 1) / block_size;
    curandState *rnd_state;
    cudaMallocManaged(&rnd_state, block_size * sizeof(curandState));
    cuda_device_init<<<1, block_size>>>(rnd_state);

    // TODO remove counter
    int *counter;
    cudaMallocManaged(&counter, sizeof(int));
    *counter = 0;

    for (int iter = 0; iter < config.iter_max; iter++) {
        cuda_device_layout<<<block_nbr, block_size>>>(iter, config, rnd_state, etas[iter], node_data, path_data, counter);
        cudaError_t error = cudaDeviceSynchronize();
        // TODO check for error
        std::cout << "CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }

    std::cout << "thread counter: " << *counter << std::endl;
#else
    cpu_layout(config, etas, node_data, path_data);
#endif
    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "CUDA layout compute took " << duration_compute_ms << "ms" << std::endl;



    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &node_data.nodes[node_idx];
        //assert(n->coords[0] == X[node_idx * 2].load());
        //assert(n->coords[1] == Y[node_idx * 2].load());
        //assert(n->coords[2] == X[node_idx * 2 + 1].load());
        //assert(n->coords[3] == Y[node_idx * 2 + 1].load());

        X[node_idx * 2].store(double(n->coords[0]));
        Y[node_idx * 2].store(double(n->coords[1]));
        X[node_idx * 2 + 1].store(double(n->coords[2]));
        Y[node_idx * 2 + 1].store(double(n->coords[3]));
    }


    // get rid of CUDA data structures
    cudaFree(node_data.nodes);
    for (int i = 0; i < path_count; i++) {
        cudaFree(path_data.paths[i].elements);
    }
    cudaFree(path_data.paths);
#ifdef USE_GPU
    cudaFree(rnd_state);
    cudaFree(counter);
#endif


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CUDA layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
