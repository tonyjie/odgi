#include "layout.h"
//#include <cuda.h>
#include <assert.h>


namespace cuda {

/*
__global__ void cuda_device_init(curandState_t *rnd_state_tmp, curandStateCoalesced_t *rnd_state) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(42+tid, tid, 0, &rnd_state[tid]);
}

__device__ double compute_zeta(uint32_t n, double theta) {
    double ans = 0.0;
    for (uint32_t i = 1; i <= n; i++) {
        ans += pow(1.0 / double(i), theta);
    }
    return ans;
}

// this function uses the cuda operation __powf, which is a faster but less precise alternative to the pow operation
__device__ uint32_t cuda_rnd_zipf(curandState *rnd_state, uint32_t n, double theta, double zeta2, double zetan) {
    double alpha = 1.0 / (1.0 - theta);
    double denominator = 1.0 - zeta2 / zetan;
    if (denominator == 0.0) {
        denominator = 1e-9;
    }
    double eta = (1.0 - __powf(2.0 / double(n), 1.0 - theta)) / (denominator);

    // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
    double u = 1.0 - curand_uniform(rnd_state);
    double uz = u * zetan;

    int64_t val = 0;
    if (uz < 1.0) val = 1;
    else if (uz < 1.0 + __powf(0.5, theta)) val = 2;
    else val = 1 + int64_t(double(n) * __powf(eta * u - eta + 1.0, alpha));

    if (val > n) {
        //printf("WARNING: val: %ld, n: %u\n", val, uint32_t(n));
        val--;
    }
    assert(val >= 0);
    assert(val <= n);
    return uint32_t(val);
}


static __device__ __inline__ uint32_t __mysmid(){
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void cuda_device_layout(int iter, cuda::layout_config_t config, curandState *rnd_state, double eta, double *zetas, cuda::node_data_t node_data,
        cuda::path_data_t path_data, uint32_t *pidx_array, int64_t *pos_array, uint32_t *node_id_array, float *x_coords, float *y_coords, int32_t *seq_length_array) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smid = __mysmid();
    assert(smid < 84);
    curandState *thread_rnd_state = &rnd_state[smid * 1024 + threadIdx.x];

    // select path
    // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
    uint32_t step_idx = uint32_t(floor((1.0 - curand_uniform(thread_rnd_state)) * float(path_data.total_path_steps)));
    assert(step_idx < path_data.total_path_steps);

    // find path of step of specific thread with LUT (threads in warp pick same path)
    uint32_t path_idx = pidx_array[step_idx];


    path_t p = path_data.paths[path_idx];
    if (p.step_count < 2) {
        return;
    }
    assert(p.step_count > 1);

    // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
    uint32_t s1_idx = uint32_t(floor((1.0 - curand_uniform(thread_rnd_state)) * float(p.step_count)));
    assert(s1_idx < p.step_count);
    uint32_t s2_idx;

    bool cooling = (iter >= config.first_cooling_iteration) || (curand_uniform(thread_rnd_state) <= 0.5);
    if (cooling) {
        if (s1_idx > 0 && (curand_uniform(thread_rnd_state) <= 0.5) || s1_idx == p.step_count-1) {
            // go backward
            uint32_t jump_space = min(config.space, s1_idx);
            uint32_t space = jump_space;
            if (jump_space > config.space_max) {
                space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
            }

            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state, jump_space, config.theta, zetas[2], zetas[space]);
            if (!(z_i <= s1_idx)) {
                printf("Error (thread %i): %u - %u\n", threadIdx.x, s1_idx, z_i);
                printf("Jumpspace %u, theta %f, zeta %f\n", jump_space, config.theta, zetas[space]);
            }
            assert(z_i <= s1_idx);
            s2_idx = s1_idx - z_i;
        } else {
            // go forward
            uint32_t jump_space = min(config.space, p.step_count - s1_idx - 1);
            uint32_t space = jump_space;
            if (jump_space > config.space_max) {
                space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
            }

            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state, jump_space, config.theta, zetas[2], zetas[space]);
            if (!(z_i <= p.step_count - s1_idx - 1)) {
                printf("Error (thread %i): %u + %u, step_count %u\n", threadIdx.x, s1_idx, z_i, p.step_count);
                printf("Jumpspace %u, theta %f, zeta %f\n", jump_space, config.theta, zetas[space]);
            }
            assert(s1_idx + z_i < p.step_count);
            s2_idx = s1_idx + z_i;
        }
    } else {
        do {
            s2_idx = uint32_t(floor((1.0 - curand_uniform(thread_rnd_state)) * float(p.step_count)));
        } while (s1_idx == s2_idx);
    }
    assert(s1_idx < p.step_count);
    assert(s2_idx < p.step_count);
    assert(s1_idx != s2_idx);


    uint32_t n1_id = node_id_array[p.first_step_in_path + s1_idx];
    int64_t n1_pos_in_path = pos_array[p.first_step_in_path + s1_idx];
    bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
    n1_pos_in_path = std::abs(n1_pos_in_path);

    uint32_t n2_id = node_id_array[p.first_step_in_path + s2_idx];
    int64_t n2_pos_in_path = pos_array[p.first_step_in_path + s2_idx];
    bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
    n2_pos_in_path = std::abs(n2_pos_in_path);

    uint32_t n1_seq_length = seq_length_array[n1_id];
    bool n1_use_other_end = (curand_uniform(thread_rnd_state) <= 0.5)? true: false;
    if (n1_use_other_end) {
        n1_pos_in_path += uint64_t{n1_seq_length};
        n1_use_other_end = !n1_is_rev;
    } else {
        n1_use_other_end = n1_is_rev;
    }

    uint32_t n2_seq_length = seq_length_array[n2_id];
    bool n2_use_other_end = (curand_uniform(thread_rnd_state) <= 0.5)? true: false;
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

    int n1_offset = n1_use_other_end? 1: 0;
    int n2_offset = n2_use_other_end? 1: 0;

    float *x1 = &x_coords[n1_id * 2 + n1_offset];
    float *x2 = &x_coords[n2_id * 2 + n2_offset];
    float *y1 = &y_coords[n1_id * 2 + n1_offset];
    float *y2 = &y_coords[n2_id * 2 + n2_offset];
    double x1_val = double(*x1);
    double x2_val = double(*x2);
    double y1_val = double(*y1);
    double y2_val = double(*y2);

    double dx = x1_val - x2_val;
    double dy = y1_val - y2_val;

    if (dx == 0.0) {
        dx = 1e-9;
    }

    double mag = sqrt(dx * dx + dy * dy);
    double delta = mu * (mag - d_ij) / 2.0;
    //double delta_abs = std::abs(delta);

    double r = delta / mag;
    double r_x = r * dx;
    double r_y = r * dy;
    atomicExch(x1, float(x1_val - r_x));
    atomicExch(x2, float(x2_val + r_x));
    atomicExch(y1, float(y1_val - r_y));
    atomicExch(y2, float(y2_val + r_y));
}
*/


void cpu_layout(cuda::layout_config_t config, double *etas, double *zetas, cuda::node_data_t &node_data, cuda::path_data_t &path_data,
        uint32_t *pidx_array, int64_t *pos_array, uint32_t *node_id_array, std::vector<std::atomic<double>> &x_coords, std::vector<std::atomic<double>> &y_coords, int32_t *seq_length_array) {
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
        //std::discrete_distribution<> rand_path(path_dist.begin(), path_dist.end());

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
                std::uniform_int_distribution<uint32_t> rand_total_steps(0, path_data.total_path_steps-1);
                uint32_t step_idx = rand_total_steps(gen);

                uint32_t path_idx = pidx_array[step_idx];
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
#ifdef profiling
                two_step_gen = std::chrono::high_resolution_clock::now();
                total_duration_two_step_gen += std::chrono::duration_cast<std::chrono::nanoseconds>(two_step_gen - one_step_gen);
#endif
                assert(s1_idx < p.step_count);
                assert(s2_idx < p.step_count);

                uint32_t n1_id = node_id_array[p.first_step_in_path + s1_idx];
                int64_t n1_pos_in_path = pos_array[p.first_step_in_path + s1_idx];
                bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
                n1_pos_in_path = std::abs(n1_pos_in_path);

                uint32_t n2_id = node_id_array[p.first_step_in_path + s2_idx];
                int64_t n2_pos_in_path = pos_array[p.first_step_in_path + s2_idx];
                bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
                n2_pos_in_path = std::abs(n2_pos_in_path);

                uint32_t n1_seq_length = seq_length_array[n1_id];
                bool n1_use_other_end = flip(gen);
                if (n1_use_other_end) {
                    n1_pos_in_path += uint64_t{n1_seq_length};
                    n1_use_other_end = !n1_is_rev;
                } else {
                    n1_use_other_end = n1_is_rev;
                }

                uint32_t n2_seq_length = seq_length_array[n2_id];
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

                int n1_offset = n1_use_other_end? 1: 0;
                int n2_offset = n2_use_other_end? 1: 0;

#ifdef profiling
                before_load = std::chrono::high_resolution_clock::now();
                total_duration_compute_first += std::chrono::duration_cast<std::chrono::nanoseconds>(before_load - start_sgd);
#endif
                std::atomic<double> *x1 = &x_coords[n1_id * 2 + n1_offset];
                std::atomic<double> *x2 = &x_coords[n2_id * 2 + n2_offset];
                std::atomic<double> *y1 = &y_coords[n1_id * 2 + n1_offset];
                std::atomic<double> *y2 = &y_coords[n2_id * 2 + n2_offset];

                double dx = double(x1->load() - x2->load());
                double dy = double(y1->load() - y2->load());
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

                double r = delta / mag;
                double r_x = r * dx;
                double r_y = r * dy;

#ifdef profiling
                before_store = std::chrono::high_resolution_clock::now();
                total_duration_compute_second += std::chrono::duration_cast<std::chrono::nanoseconds>(before_store - after_load);
#endif
                x1->store(x1->load() - double(r_x));
                y1->store(y1->load() - double(r_y));
                x2->store(x2->load() + double(r_x));
                y2->store(y2->load() + double(r_y));
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
    //std::cout << "size of node_t: " << sizeof(node_t) << std::endl;
    std::cout << "theta: " << config.theta << std::endl;

    // create eta array
    double *etas;
    //cudaMallocManaged(&etas, config.iter_max * sizeof(double));
    etas = (double*) malloc(config.iter_max * sizeof(double));

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
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() == node_count);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    cuda::node_data_t node_data;
    node_data.node_count = node_count;
    //cudaMallocManaged(&node_data.nodes, node_count * sizeof(cuda::node_t));

    //double *x_coords;
    //double *y_coords;
    //cudaMallocManaged(&x_coords, node_count * 2 * sizeof(double));
    //cudaMallocManaged(&y_coords, node_count * 2 * sizeof(double));
    std::vector<std::atomic<double>> x_coords(2 * node_count);
    std::vector<std::atomic<double>> y_coords(2 * node_count);
    int32_t *seq_length_array;
    seq_length_array = (int32_t*) malloc(node_count * sizeof(int32_t));
    //cudaMallocManaged(&seq_length_array, node_count * sizeof(int32_t));
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        //assert(graph.has_node(node_idx));
        //cuda::node_t *n_tmp = &node_data.nodes[node_idx];

        // sequence length
        const handlegraph::handle_t h = graph.get_handle(node_idx + 1, false);
        // NOTE: unable store orientation (reverse), since this information is path dependent
        //n_tmp->seq_length = graph.get_length(h);
        seq_length_array[node_idx] = graph.get_length(h);

        // copy random coordinates
        x_coords[node_idx * 2].store(double(X[node_idx * 2].load()));
        y_coords[node_idx * 2].store(double(Y[node_idx * 2].load()));
        x_coords[node_idx * 2 + 1].store(double(X[node_idx * 2 + 1].load()));
        y_coords[node_idx * 2 + 1].store(double(Y[node_idx * 2 + 1].load()));
    }


    // create path data structure
    uint32_t path_count = graph.get_path_count();
    cuda::path_data_t path_data;
    path_data.path_count = path_count;
    path_data.total_path_steps = 0;
    //cudaMallocManaged(&path_data.paths, path_count * sizeof(cuda::path_t));
    path_data.paths = (cuda::path_t*) malloc(path_count * sizeof(cuda::path_t));

    vector<odgi::path_handle_t> path_handles{};
    path_handles.reserve(path_count);
    graph.for_each_path_handle(
        [&] (const odgi::path_handle_t& p) {
            path_handles.push_back(p);
            path_data.total_path_steps += graph.get_step_count(p);
        });
    //cudaMallocManaged(&path_data.element_array, path_data.total_path_steps * sizeof(path_element_t));

    // npi_iv in original implementation
    uint32_t *pidx_array;
    //cudaMallocManaged(&pidx_array, path_data.total_path_steps * sizeof(uint32_t));
    pidx_array = (uint32_t*) malloc(path_data.total_path_steps * sizeof(uint32_t));

    int64_t *pos_array;
    //cudaMallocManaged(&pos_array, path_data.total_path_steps * sizeof(int64_t));
    pos_array = (int64_t*) malloc(path_data.total_path_steps * sizeof(int64_t));

    uint32_t *node_id_array;
    //cudaMallocManaged(&node_id_array, path_data.total_path_steps * sizeof(uint32_t));
    node_id_array = (uint32_t*) malloc(path_data.total_path_steps * sizeof(uint32_t));

    // get length and starting position of all paths
    uint32_t first_step_counter = 0;
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        int step_count = graph.get_step_count(p);
        path_data.paths[path_idx].step_count = step_count;
        path_data.paths[path_idx].first_step_in_path = first_step_counter;
        first_step_counter += step_count;
    }

#pragma omp parallel for num_threads(config.nthreads)
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        //std::cout << graph.get_path_name(p) << ": " << graph.get_step_count(p) << std::endl;

        uint32_t step_count = path_data.paths[path_idx].step_count;
        uint32_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;
        //if (step_count == 0) {
            //path_data.paths[path_idx].elements = NULL;

        //} else {
        if (step_count > 0) {
            //path_element_t *cur_path = &path_data.element_array[first_step_in_path];
            //path_data.paths[path_idx].elements = cur_path;

            odgi::step_handle_t s = graph.path_begin(p);
            int64_t pos = 1;
            // Iterate through path
            for (int step_idx = 0; step_idx < step_count; step_idx++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                //std::cout << graph.get_id(h) << std::endl;

                //cur_path[step_idx].node_id = graph.get_id(h) - 1;
                node_id_array[first_step_in_path + step_idx] = graph.get_id(h) - 1;
                //cur_path[step_idx].pidx = uint32_t(path_idx);
                pidx_array[first_step_in_path + step_idx] = uint32_t(path_idx);
                // store position negative when handle reverse
                if (graph.get_is_reverse(h)) {
                    //cur_path[step_idx].pos = -pos;
                    pos_array[first_step_in_path + step_idx] = -pos;
                } else {
                    //cur_path[step_idx].pos = pos;
                    pos_array[first_step_in_path + step_idx] = pos;
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
    }


    // cache zipf zetas
    auto start_zeta = std::chrono::high_resolution_clock::now();
    double *zetas;
    uint64_t zetas_cnt = ((config.space <= config.space_max)? config.space : (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;
    std::cout << "zetas_cnt: " << zetas_cnt << std::endl;
    std::cout << "space_max: " << config.space_max << std::endl;
    std::cout << "config.space: " << config.space << std::endl;
    std::cout << "config.space_quantization: " << config.space_quantization_step << std::endl;

    //cudaMallocManaged(&zetas, zetas_cnt * sizeof(double));
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
//#define USE_GPU
#ifdef USE_GPU
    /*
    std::cout << "cuda gpu layout" << std::endl;
    std::cout << "total-path_steps: " << path_data.total_path_steps << std::endl;

    const uint64_t block_size = BLOCK_SIZE;
    uint64_t block_nbr = (config.min_term_updates + block_size - 1) / block_size;
    std::cout << "block_nbr: " << block_nbr << " block_size: " << block_size << std::endl;
    curandState *rnd_state;
    std::cout << "sizeof curandState: " << sizeof(curandState) << std::endl;
    cudaError_t tmp_error = cudaMallocManaged(&rnd_state, 84 * block_size * sizeof(curandState));
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
    cuda_device_init<<<84, block_size>>>(rnd_state);
    tmp_error = cudaDeviceSynchronize();
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;

    for (int iter = 0; iter < config.iter_max; iter++) {
        cuda_device_layout<<<block_nbr, block_size>>>(iter, config, rnd_state, etas[iter], zetas, node_data, path_data, pidx_array, pos_array, node_id_array, x_coords, y_coords, seq_length_array);
        cudaError_t error = cudaDeviceSynchronize();
        std::cout << "CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }
    */

#else
    cpu_layout(config, etas, zetas, node_data, path_data, pidx_array, pos_array, node_id_array, x_coords, y_coords, seq_length_array);
#endif
    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "CPU cache-optimized layout compute took " << duration_compute_ms << "ms" << std::endl;



    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        //cuda::node_t *n = &(node_data.nodes[node_idx]);
        // coords[0], coords[1], coords[2], coords[3] are stored consecutively. 
        /*
        float *coords = n->coords;
        // check if coordinates valid (not NaN or infinite)
        for (int i = 0; i < 4; i++) {
            if (!isfinite(coords[i])) {
                std::cout << "WARNING: invalid coordiate" << std::endl;
            }
        }
        */
        X[node_idx * 2].store(double(x_coords[node_idx * 2].load()));
        Y[node_idx * 2].store(double(y_coords[node_idx * 2].load()));
        X[node_idx * 2 + 1].store(double(x_coords[node_idx * 2 + 1].load()));
        Y[node_idx * 2 + 1].store(double(y_coords[node_idx * 2 + 1].load()));
        //std::cout << "coords of " << node_idx << ": [" << X[node_idx*2] << "; " << Y[node_idx*2] << "] ; [" << X[node_idx*2+1] << "; " << Y[node_idx*2+1] <<"]\n";
    }


    // get rid of CUDA data structures
    free(etas);
    //free(node_data.nodes);
    free(path_data.paths);
    //free(path_data.element_array);
    free(zetas);

    free(pidx_array);
    free(pos_array);
    free(node_id_array);
    //free(x_coords);
    //free(y_coords);
    free(seq_length_array);
#ifdef USE_GPU
    //cudaFree(rnd_state);
#endif


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout << "CPU cache-optimized layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
