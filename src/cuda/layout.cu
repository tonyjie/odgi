#include "layout.h"
#include <cuda.h>
#include <assert.h>


namespace cuda {

__global__ void cuda_device_init(curandState *rnd_state) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(42+tid, tid, 0, &rnd_state[threadIdx.x]);
}

__device__ double compute_zeta(uint32_t n, double theta) {
    double ans = 0.0;
    for (uint32_t i = 1; i <= n; i++) {
        ans += pow(1.0 / double(i), theta);
    }
    return ans;
}

__device__ uint32_t cuda_rnd_zipf(curandState *rnd_state, uint32_t n, double theta, double zeta2, double zetan) {
    // TODO Compute zetan on GPU (with exact pow, instead of dirtyzipfian pow)
    double alpha = 1.0 / (1.0 - theta);
    double eta = (1.0 - pow(2.0 / double(n), 1.0 - theta)) / (1.0 - zeta2 / zetan);

    double u = curand_uniform(rnd_state);
    double uz = u * zetan;

    int64_t val = 0;
    if (uz < 1.0) val = 1;
    else if (uz < 1.0 + pow(0.5, theta)) val = 2;
    else val = 1 + int64_t(double(n) * pow(eta * u - eta + 1.0, alpha));

    if (val > n) {
        //printf("WARNING: val: %ld, n: %u\n", val, uint32_t(n));
        // TODO Fix sometimes val == n+1
        val--;
    }
    assert(val >= 0);
    assert(val <= n);
    return uint32_t(val);
}


__global__ void cuda_device_layout(int iter, cuda::layout_config_t config, curandState *rnd_state, double eta, double *zetas, cuda::node_data_t node_data,
        cuda::path_data_t path_data) { //, int *counter) {
    // TODO pipeline step kernel; get nodes and distance for next step (hide memory access time?)
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO improve discrete distribution selection
    // select path
    int32_t offset = (int32_t)(ceil((curand_uniform(rnd_state) * (gridDim.x + 1.0))) - 1.0);
    uint32_t step_idx = ((uint32_t) (offset * blockDim.x + threadIdx.x)) % (uint32_t) path_data.total_path_steps;
    uint32_t path_idx;  // = blockIdx.x % path_data.path_count;
    uint32_t start_step_path = 0;
    for (int pidx = 0; pidx < path_data.path_count; pidx++) {
        if (step_idx >= start_step_path && step_idx < start_step_path + path_data.paths[pidx].step_count) {
            path_idx = pidx;
            break;
        } else {
            start_step_path += path_data.paths[pidx].step_count;
        }
    }

    path_t p = path_data.paths[path_idx];
    if (p.step_count < 2) {
        return;
    }
    assert(p.step_count > 1);

    uint32_t s1_idx = (uint32_t)(ceil((curand_uniform(rnd_state + threadIdx.x)*float(p.step_count))) - 1.0);
    uint32_t s2_idx;


    __shared__ bool cooling;
    if (tid % 32 == 0) {
        cooling = (iter >= config.first_cooling_iteration) || (curand_uniform(rnd_state + threadIdx.x) <= 0.5);
    }
    __syncwarp();

    if (cooling) {
        bool backward;
        uint32_t jump_space;
        if (s1_idx > 0 && (curand_uniform(rnd_state + threadIdx.x) <= 0.5) || s1_idx == p.step_count-1) {
            // go backward
            backward = true;
            jump_space = min(config.space, s1_idx);
        } else {
            // go forward
            backward = false;
            jump_space = min(config.space, p.step_count - s1_idx - 1);
        }
        uint32_t space = jump_space;
        if (jump_space > config.space_max) {
            space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
        }

        uint32_t z_i = cuda_rnd_zipf(&rnd_state[threadIdx.x], jump_space, config.theta, zetas[2], zetas[space]);

        if (backward) {
            if (!(z_i <= s1_idx)) {
                printf("Error (thread %i): %u - %u\n", threadIdx.x, s1_idx, z_i);
                printf("Jumpspace %u, theta %f, zeta %f\n", jump_space, config.theta, zetas[space]);
            }
            assert(z_i <= s1_idx);
        } else {
            if (!(z_i <= p.step_count - s1_idx - 1)) {
                printf("Error (thread %i): %u + %u, step_count %u\n", threadIdx.x, s1_idx, z_i, p.step_count);
                printf("Jumpspace %u, theta %f, zeta %f\n", jump_space, config.theta, zetas[space]);
            }
            assert(s1_idx + z_i < p.step_count);
        }

        s2_idx = backward? s1_idx - z_i: s1_idx + z_i;
    } else {
        do {
            s2_idx = (uint32_t)(ceil((curand_uniform(rnd_state + threadIdx.x)*float(p.step_count))) - 1.0);
        } while (s1_idx == s2_idx);
    }
    assert(s1_idx < p.step_count);
    assert(s2_idx < p.step_count);
    assert(s1_idx != s2_idx);


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
    double x1_val = double(atomicAdd(x1, 0.0));
    double x2_val = double(atomicAdd(x2, 0.0));
    double y1_val = double(atomicAdd(y1, 0.0));
    double y2_val = double(atomicAdd(y2, 0.0));

    double dx = x1_val - x2_val;
    double dy = y1_val - y2_val;

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
    // TODO check current value before updating
    atomicExch(x1, float(x1_val - r_x));
    atomicExch(x2, float(x2_val + r_x));
    atomicExch(y1, float(y1_val - r_y));
    atomicExch(y2, float(y2_val + r_y));
    //atomicAdd(counter, 1);
}


void cpu_layout(cuda::layout_config_t config, double *etas, double *zetas, cuda::node_data_t &node_data, cuda::path_data_t &path_data) {
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

//#define profiling
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
                if (p.step_count < 2) {
                    continue;
                }

                std::uniform_int_distribution<uint32_t> rand_step(0, p.step_count-1);

                uint32_t s1_idx = rand_step(gen);
#ifdef profiling
                one_step_gen = std::chrono::high_resolution_clock::now();
                total_duration_one_step_gen += std::chrono::duration_cast<std::chrono::nanoseconds>(one_step_gen - start_dist);
#endif
                uint32_t s2_idx;
                if (iter + 1 >= config.first_cooling_iteration || flip(gen)) {
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
#ifdef profiling
                two_step_gen = std::chrono::high_resolution_clock::now();
                total_duration_two_step_gen += std::chrono::duration_cast<std::chrono::nanoseconds>(two_step_gen - one_step_gen);
#endif
                // TODO check if those asserts are triggered for CPU / dirtyzipfian distribution
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
    std::cout << "first_cooling_iteration: " << config.first_cooling_iteration << std::endl;
    std::cout << "min_term_updates: " << config.min_term_updates << std::endl;
    std::cout << "size of node_t: " << sizeof(node_t) << std::endl;
    std::cout << "theta: " << config.theta << std::endl;

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
        double eta = eta_max * exp(-lambda * (std::abs(i - iter_with_max_learning_rate)));
        etas[i] = isnan(eta)? eta_min : eta;
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
        // TODO Check assert; why is it failing?
        //assert(graph.has_node(node_idx));
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
    path_data.total_path_steps = 0;
    cudaMallocManaged(&path_data.paths, path_count * sizeof(cuda::path_t));

    vector<odgi::path_handle_t> path_handles{};
    path_handles.reserve(path_count);
    graph.for_each_path_handle(
        [&] (const odgi::path_handle_t& p) {
            path_handles.push_back(p);
            path_data.total_path_steps += graph.get_step_count(p);
        });

#pragma omp parallel for num_threads(config.nthreads)
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        // TODO: sort paths for uniform distribution? Largest should not just be next to each other
        odgi::path_handle_t p = path_handles[path_idx];
        //std::cout << graph.get_path_name(p) << ": " << graph.get_step_count(p) << std::endl;

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

            cur_path[step_idx].node_id = graph.get_id(h) - 1;
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


    // cache zipf zetas
    double *zetas;
    // TODO use uint32_t
    // TODO move to GPU
    uint64_t zetas_cnt = ((config.space <= config.space_max)? config.space : (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;
    std::cout << "zetas_cnt: " << zetas_cnt << std::endl;
    std::cout << "space_max: " << config.space_max << std::endl;
    cudaMallocManaged(&zetas, zetas_cnt * sizeof(double));
    uint64_t last_quantized_i = 0;
    // TODO parallelise with openmp?
    for (uint64_t i = 1; i < config.space + 1; i++) {
        uint64_t quantized_i = i;
        uint64_t compressed_space = i;
        if (i > config.space_max) {
            quantized_i = config.space_max + (i - config.space_max) / config.space_quantization_step + 1;
            compressed_space = config.space_max + ((i - config.space_max) / config.space_quantization_step) * config.space_quantization_step;
        }

        if (quantized_i != last_quantized_i) {
            dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, compressed_space, config.theta);
            zetas[quantized_i] = z_p.zeta();
            last_quantized_i = quantized_i;
        }
    }



    auto start_compute = std::chrono::high_resolution_clock::now();
#define USE_GPU
#ifdef USE_GPU
    std::cout << "cuda gpu layout" << std::endl;
    std::cout << "total-path_steps: " << path_data.total_path_steps << std::endl;

    // TODO use different block_size and/or block_nbr when computing small pangenome to prevent NaN coordinates
    const uint64_t block_size = 1024;
    uint64_t block_nbr = (config.min_term_updates + block_size - 1) / block_size;
    std::cout << "block_nbr: " << block_nbr << " block_size: " << block_size << std::endl;
    curandState *rnd_state;
    // TODO increase number of curandState objects; each thread in SM own curandState?
    std::cout << "sizeof curandState: " << sizeof(curandState) << std::endl;
    cudaMallocManaged(&rnd_state, block_size * sizeof(curandState));
    cuda_device_init<<<1, block_size>>>(rnd_state);

    // TODO remove counter
    //int *counter;
    //error = cudaMallocManaged(&counter, sizeof(int));
    //std::cout << "[6] CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    //*counter = 0;

    for (int iter = 0; iter < config.iter_max; iter++) {
        cuda_device_layout<<<block_nbr, block_size>>>(iter, config, rnd_state, etas[iter], zetas, node_data, path_data); //, counter);
        cudaError_t error = cudaDeviceSynchronize();
        // TODO check for error
        std::cout << "CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }

    //std::cout << "thread counter: " << *counter << std::endl;
#else
    cpu_layout(config, etas, zetas, node_data, path_data);
#endif
    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "CUDA layout compute took " << duration_compute_ms << "ms" << std::endl;



    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &node_data.nodes[node_idx];

        for (int i = 0; i < 4; i++) {
            float coord = n->coords[i];
            if (!isfinite(coord)) {
                std::cout << "WARNING: invalid coordiate" << std::endl;
                coord = 0.0;
            }
            switch (i) {
                case 0:
                    X[node_idx * 2].store(double(coord));
                    break;
                case 1:
                    Y[node_idx * 2].store(double(coord));
                    break;
                case 2:
                    X[node_idx * 2 + 1].store(double(coord));
                    break;
                case 3:
                    Y[node_idx * 2 + 1].store(double(coord));
                    break;
            }
        }
        //std::cout << "coords of " << node_idx << ": [" << X[node_idx*2] << "; " << Y[node_idx*2] << "] ; [" << X[node_idx*2+1] << "; " << Y[node_idx*2+1] <<"]\n";
    }


    // get rid of CUDA data structures
    cudaFree(node_data.nodes);
    for (int i = 0; i < path_count; i++) {
        cudaFree(path_data.paths[i].elements);
    }
    cudaFree(path_data.paths);
    cudaFree(zetas);
#ifdef USE_GPU
    cudaFree(rnd_state);
    //cudaFree(counter);
#endif


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CUDA layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
