#include "layout.h"
#include <cuda.h>
#include <assert.h>

// #define METRIC

namespace cuda {

// compute the metric "node stress" and print out during each iteration. 
// help to guide the converging process
#ifdef METRIC

__global__ void cuda_compute_metric(float *stress_partial_sum, cuda::node_data_t node_data, uint32_t node_count) {
    /*
    stress_partial_sum includes the partial sum of the stress from each block. stress_partial_sum.size() = numBlocks
    This kernel function first computes the stress of each node, and then computes the partial sum of the stress of each block with a strided reduction algorithm. 
    */
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 1. compute stress of each node
    float node_stress = 0.0;
    float dist_layout = 0.0;
    uint32_t dist_nuc = 0;
    if (tid < node_count) {
        dist_nuc = node_data.nodes[tid].seq_length;
        // dist_layout = sqrt( (x_start - x_end)^2 + (y_start - y_end)^2 )
        dist_layout = sqrt( pow(node_data.nodes[tid].coords[0] - node_data.nodes[tid].coords[2], 2) + \
                                pow(node_data.nodes[tid].coords[1] - node_data.nodes[tid].coords[3], 2) );
        node_stress = pow( (dist_layout - dist_nuc) / dist_nuc, 2);
    }
    __syncthreads();

    // // print out each code's stress
    // if (threadIdx.x == 0) {
    //     printf("tid: %d, dist_nuc: %d, dist_layout: %f, node_stress: %f\n", tid, dist_nuc, dist_layout, node_stress);
    // }

    // 2. Store the stress of each node in the shared memory
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = node_stress;
    __syncthreads();

    // 3. Reduction in the shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // 4. Store the partial sum of the stress of each block in the global memory
    if (threadIdx.x == 0) {
        stress_partial_sum[blockIdx.x] = sdata[0];
    }
}



#endif

// standard curand_init function
__global__ void cuda_device_init_std(curandState *rnd_state) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, tid, 0, &rnd_state[tid]);
}



__global__ void cuda_device_init(curandState_t *rnd_state_tmp, curandStateCoalesced_t *rnd_state) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // initialize curandState with original curand implementation
    // curand_init(42+tid, tid, 0, &rnd_state_tmp[tid]);
    curand_init(9399220+tid, tid, 0, &rnd_state_tmp[tid]);
    // copy to coalesced data structure
    rnd_state[blockIdx.x].d[threadIdx.x] = rnd_state_tmp[tid].d;
    rnd_state[blockIdx.x].w0[threadIdx.x] = rnd_state_tmp[tid].v[0];
    rnd_state[blockIdx.x].w1[threadIdx.x] = rnd_state_tmp[tid].v[1];
    rnd_state[blockIdx.x].w2[threadIdx.x] = rnd_state_tmp[tid].v[2];
    rnd_state[blockIdx.x].w3[threadIdx.x] = rnd_state_tmp[tid].v[3];
    rnd_state[blockIdx.x].w4[threadIdx.x] = rnd_state_tmp[tid].v[4];
}

__device__ double curand_uniform_double_coalesced(curandStateCoalesced_t *state, uint32_t thread_id) {
    // generate 32 bit pseudorandom value with XORWOW generator (see paper "Xorshift RNGs" by George Marsaglia);
    // also used in curand library (see curand_kernel.h)

    // x = curand(state)
    uint32_t t;
    t = state->w0[thread_id] ^ (state->w0[thread_id] >> 2);
    state->w0[thread_id] = state->w1[thread_id];
    state->w1[thread_id] = state->w2[thread_id];
    state->w2[thread_id] = state->w3[thread_id];
    state->w3[thread_id] = state->w4[thread_id];
    state->w4[thread_id] = (state->w4[thread_id] ^ (state->w4[thread_id] << 4)) ^ (t ^ (t << 1));
    state->d[thread_id] += 362437;
    uint32_t x = state->d[thread_id] + state->w4[thread_id];

    // y = curand(state)
    t = state->w0[thread_id] ^ (state->w0[thread_id] >> 2);
    state->w0[thread_id] = state->w1[thread_id];
    state->w1[thread_id] = state->w2[thread_id];
    state->w2[thread_id] = state->w3[thread_id];
    state->w3[thread_id] = state->w4[thread_id];
    state->w4[thread_id] = (state->w4[thread_id] ^ (state->w4[thread_id] << 4)) ^ (t ^ (t << 1));
    state->d[thread_id] += 362437;
    uint32_t y = state->d[thread_id] + state->w4[thread_id];

    // convert to float; see curand_uniform.h
    return _curand_uniform_double_hq(x, y);
}

__device__ uint32_t curand_coalesced(curandStateCoalesced_t *state, uint32_t thread_id) {
    // generate 32 bit pseudorandom value with XORWOW generator (see paper "Xorshift RNGs" by George Marsaglia);
    // also used in curand library (see curand_kernel.h)
    uint32_t t;
    t = state->w0[thread_id] ^ (state->w0[thread_id] >> 2);
    state->w0[thread_id] = state->w1[thread_id];
    state->w1[thread_id] = state->w2[thread_id];
    state->w2[thread_id] = state->w3[thread_id];
    state->w3[thread_id] = state->w4[thread_id];
    state->w4[thread_id] = (state->w4[thread_id] ^ (state->w4[thread_id] << 4)) ^ (t ^ (t << 1));
    state->d[thread_id] += 362437;

    uint32_t rnd_uint = state->d[thread_id] + state->w4[thread_id];
    
    return rnd_uint;
}

// use two uint32_t to form a uint64_t
__device__ uint64_t curand_coalesced_uint64(curandStateCoalesced_t *state, uint32_t thread_id) {
    uint32_t upper = curand_coalesced(state, thread_id);
    uint32_t lower = curand_coalesced(state, thread_id);
    uint64_t rnd_uint64 = (uint64_t(upper) << 32) | uint64_t(lower);
    return rnd_uint64;
}

__device__ float curand_uniform_coalesced(curandStateCoalesced_t *state, uint32_t thread_id) {
    // generate 32 bit pseudorandom value with XORWOW generator (see paper "Xorshift RNGs" by George Marsaglia);
    // also used in curand library (see curand_kernel.h)
    uint32_t t;
    t = state->w0[thread_id] ^ (state->w0[thread_id] >> 2);
    state->w0[thread_id] = state->w1[thread_id];
    state->w1[thread_id] = state->w2[thread_id];
    state->w2[thread_id] = state->w3[thread_id];
    state->w3[thread_id] = state->w4[thread_id];
    state->w4[thread_id] = (state->w4[thread_id] ^ (state->w4[thread_id] << 4)) ^ (t ^ (t << 1));
    state->d[thread_id] += 362437;

    uint32_t rnd_uint = state->d[thread_id] + state->w4[thread_id];

    // convert to float; see curand_uniform.h
    return _curand_uniform(rnd_uint);
}


__device__ double compute_zeta(uint32_t n, double theta) {
    double ans = 0.0;
    for (uint32_t i = 1; i <= n; i++) {
        ans += pow(1.0 / double(i), theta);
    }
    return ans;
}

// this function uses the cuda operation __powf, which is a faster but less precise alternative to the pow operation
__device__ uint32_t cuda_rnd_zipf(curandStateCoalesced_t *rnd_state, uint32_t n, double theta, double zeta2, double zetan) {
    double alpha = 1.0 / (1.0 - theta);
    double denominator = 1.0 - zeta2 / zetan;
    if (denominator == 0.0) {
        denominator = 1e-9;
    }
    double eta = (1.0 - __powf(2.0 / double(n), 1.0 - theta)) / (denominator);

    // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
    double u = 1.0 - curand_uniform_coalesced(rnd_state, threadIdx.x);
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

__global__ void cuda_device_layout(int iter, cuda::layout_config_t config, curandStateCoalesced_t *rnd_state, double eta, double *zetas, cuda::node_data_t node_data,
        cuda::path_data_t path_data, uint32_t *path_hist, curandState *rnd_state_std) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smid = __mysmid();
    assert(smid < 84);
    curandStateCoalesced_t *thread_rnd_state = &rnd_state[smid];

    __shared__ bool cooling[32];
    if (threadIdx.x % 32 == 1) {
        cooling[threadIdx.x / 32] = (iter >= config.first_cooling_iteration) || (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5);
    }

    // select path: each thread in a warp selects the same path
    __shared__ uint32_t first_step_idx[32];
    if (threadIdx.x % 32 == 0) {
        // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
        // first_step_idx[threadIdx.x / 32] = uint32_t(floor((1.0 - curand_uniform_coalesced(thread_rnd_state, threadIdx.x)) * float(path_data.total_path_steps)));
        // first_step_idx[threadIdx.x / 32] = uint32_t(round(((double)1.0 - curand_uniform_double_coalesced(thread_rnd_state, threadIdx.x)) * double(path_data.total_path_steps)));

        // first_step_idx[threadIdx.x / 32] = uint32_t( curand_uniform_double_coalesced(thread_rnd_state, threadIdx.x) * (double(path_data.total_path_steps) + 0.9999999999) );

        assert(first_step_idx[threadIdx.x / 32] < path_data.total_path_steps);
    }
    __syncwarp();

    // // find path of step of specific thread with LUT (threads in warp pick same path)
    // uint32_t step_idx = first_step_idx[threadIdx.x / 32];
    // uint32_t path_idx = path_data.element_array[step_idx].pidx;


    // each thread select its own path
    // uint32_t step_idx = uint32_t(floor((1.0 - curand_uniform_double_coalesced(thread_rnd_state, threadIdx.x)) * float(path_data.total_path_steps)));
    // uint32_t step_idx = curand_coalesced(thread_rnd_state, threadIdx.x) % path_data.total_path_steps;
    uint32_t step_idx = uint32_t( curand_coalesced_uint64(thread_rnd_state, threadIdx.x) % uint64_t(path_data.total_path_steps) );

    // use standard curand_uniform to select path
    // uint32_t step_idx = uint32_t(floor((1.0 - curand_uniform_double(&rnd_state_std[tid])) * double(path_data.total_path_steps)));

    // uint32_t step_idx = curand(&rnd_state_std[tid]) % path_data.total_path_steps;

    uint32_t path_idx = path_data.element_array[step_idx].pidx;
    // if (threadIdx.x == 0) {
    //     printf("tid: %d, path_idx: %d\n", tid, path_idx);
    // }

    // count this path selection, saved in a histogram
    atomicAdd(&path_hist[path_idx], 1);
    return; // continue to the next for loop (to speedup the process)


    path_t p = path_data.paths[path_idx];
    if (p.step_count < 2) {
        return;
    }
    assert(p.step_count > 1);

    // restrict the number of concurrent threads within a warp for the path with very small number of steps (e.g. num_steps = 4), which is fewer than one warp (32 threads)
    // [FAIL] only add this doesn't work. 
    // if (p.step_count < 32) {
    //     // only the first few threads in the warp are allowed to run. (threadIdx.x % 32) is the thread id within a warp
    //     if ((threadIdx.x % 32) >= p.step_count) {
    //         return;
    //     } 
    // }



    // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
    uint32_t s1_idx = uint32_t(floor((1.0 - curand_uniform_coalesced(thread_rnd_state, threadIdx.x)) * float(p.step_count)));
    assert(s1_idx < p.step_count);
    uint32_t s2_idx;

    if (cooling[threadIdx.x / 32]) {
        bool backward;
        uint32_t jump_space;
        if (s1_idx > 0 && (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5) || s1_idx == p.step_count-1) {
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

        uint32_t z_i = cuda_rnd_zipf(thread_rnd_state, jump_space, config.theta, zetas[2], zetas[space]);

        /*
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
        */

        s2_idx = backward? s1_idx - z_i: s1_idx + z_i;
    } else {
        do {
            s2_idx = uint32_t(floor((1.0 - curand_uniform_coalesced(thread_rnd_state, threadIdx.x)) * float(p.step_count)));
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
    bool n1_use_other_end = (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5)? true: false;
    if (n1_use_other_end) {
        n1_pos_in_path += uint64_t{n1_seq_length};
        n1_use_other_end = !n1_is_rev;
    } else {
        n1_use_other_end = n1_is_rev;
    }

    uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
    bool n2_use_other_end = (curand_uniform_coalesced(thread_rnd_state, threadIdx.x) <= 0.5)? true: false;
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


void cpu_layout(cuda::layout_config_t config, double *etas, double *zetas, cuda::node_data_t &node_data, cuda::path_data_t &path_data, std::atomic<uint32_t> *path_hist) {
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

                // count this path selection, saved in a histogram. Then continue to the next for loop (to speedup the process)
                path_hist[path_idx].fetch_add(1);
                continue;


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
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() == node_count);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    cuda::node_data_t node_data;
    node_data.node_count = node_count;
    cudaMallocManaged(&node_data.nodes, node_count * sizeof(cuda::node_t));
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
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
    cudaMallocManaged(&path_data.element_array, path_data.total_path_steps * sizeof(path_element_t));

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
        if (step_count == 0) {
            path_data.paths[path_idx].elements = NULL;
        } else {
            path_element_t *cur_path = &path_data.element_array[first_step_in_path];
            path_data.paths[path_idx].elements = cur_path;

            odgi::step_handle_t s = graph.path_begin(p);
            int64_t pos = 1;
            // Iterate through path
            for (int step_idx = 0; step_idx < step_count; step_idx++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                //std::cout << graph.get_id(h) << std::endl;

                cur_path[step_idx].node_id = graph.get_id(h) - 1;
                cur_path[step_idx].pidx = uint32_t(path_idx);
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
    }


    // cache zipf zetas
    auto start_zeta = std::chrono::high_resolution_clock::now();
    double *zetas;
    uint64_t zetas_cnt = ((config.space <= config.space_max)? config.space : (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;
    std::cout << "zetas_cnt: " << zetas_cnt << std::endl;
    std::cout << "space_max: " << config.space_max << std::endl;
    std::cout << "config.space: " << config.space << std::endl;
    std::cout << "config.space_quantization: " << config.space_quantization_step << std::endl;

    cudaMallocManaged(&zetas, zetas_cnt * sizeof(double));
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
#define USE_GPU
// #define USE_CPU

#ifdef USE_GPU
    std::cout << "cuda gpu layout" << std::endl;
    std::cout << "total-path_steps: " << path_data.total_path_steps << std::endl;

    const uint64_t block_size = BLOCK_SIZE;
    uint64_t block_nbr = (config.min_term_updates + block_size - 1) / block_size;
    // uint64_t block_nbr = (config.min_term_updates / 10 + block_size - 1) / block_size;
    std::cout << "block_nbr: " << block_nbr << " block_size: " << block_size << std::endl;
    curandState_t *rnd_state_tmp;
    curandStateCoalesced_t *rnd_state;
    cudaError_t tmp_error = cudaMallocManaged(&rnd_state_tmp, SM_COUNT * block_size * sizeof(curandState_t));
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
    tmp_error = cudaMallocManaged(&rnd_state, SM_COUNT * sizeof(curandStateCoalesced_t));
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
    cuda_device_init<<<SM_COUNT, block_size>>>(rnd_state_tmp, rnd_state);
    tmp_error = cudaDeviceSynchronize();
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
    cudaFree(rnd_state_tmp);
#endif

#ifdef METRIC
    int blockSize = 1024;
    int sharedMemSize = blockSize * sizeof(float);
    int numBlocks = (node_count + blockSize - 1) / blockSize;
    // create StressPartialSum array
    float *stress_partial_sum;
    cudaMallocManaged(&stress_partial_sum, numBlocks * sizeof(float));
#endif

#ifdef USE_GPU

    // create a histogram vector to record the path selection
    uint32_t *path_hist;
    cudaMallocManaged(&path_hist, path_count * sizeof(uint32_t));
    // initialize with 0
    for (int i = 0; i < path_count; i++) {
        path_hist[i] = 0;
    }

    // use standard curandState for each thread
    curandState *rnd_state_std;
    // tmp_error = cudaMallocManaged(&rnd_state_std, block_nbr * block_size * sizeof(curandState));
    // std::cout << "CudaMallocManaged rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
    // cuda_device_init_std<<<block_nbr, block_size>>>(rnd_state_std);
    // tmp_error = cudaDeviceSynchronize();
    // std::cout << "Standard rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;

    for (int iter = 0; iter < config.iter_max; iter++) {
        cuda_device_layout<<<block_nbr, block_size>>>(iter, config, rnd_state, etas[iter], zetas, node_data, path_data, path_hist, rnd_state_std);
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cout << "CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
        } else {
            std::cout << "Iteration[" << iter << "] ";
        }
#endif        
        // add a metric computing function here to print out the metric interactively during the layout process
#ifdef METRIC
        cuda_compute_metric<<<numBlocks, blockSize, sharedMemSize>>>(stress_partial_sum, node_data, node_count);
        error = cudaDeviceSynchronize();
        // std::cout << "CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;        
        // sum up the partial sum
        float stress_sum = 0.0;
        #pragma omp parallel for reduction(+:stress_sum)
        for (int i = 0; i < numBlocks; i++) {
            stress_sum += stress_partial_sum[i];
            // std::cout << "stress_partial_sum[" << i << "]: " << stress_partial_sum[i] << std::endl;
        }
        // normalized by node_count
        stress_sum /= node_count;
        std::cout << "Node Stress: " << stress_sum;
#endif
#ifdef USE_GPU
        std::cout << std::endl;
        // cudaError_t error = cudaDeviceSynchronize();
        // std::cout << "CUDA Error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }
#endif
#ifdef USE_CPU
    // create a path_hist vector to record the path selection. Should be atomic
    std::atomic<uint32_t> *path_hist = new std::atomic<uint32_t>[path_count];
    // initialize with 0
    for (int i = 0; i < path_count; i++) {
        path_hist[i] = 0;
    }
    cpu_layout(config, etas, zetas, node_data, path_data, path_hist);
#endif

    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    // std::cout << "CUDA layout compute took " << duration_compute_ms << "ms" << std::endl;
    // in seconds
    std::cout << "CUDA layout kernel time: " << duration_compute_ms / 1000.0 << "s" << std::endl;


    // combine the path histogran with path_handle
    struct PathInfo {
        odgi::path_handle_t path_handle;
        uint32_t path_id;
        uint32_t path_hit_count;
    };

    std::vector<PathInfo> path_info;
    path_info.reserve(path_count);

    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        PathInfo p;
        p.path_handle = path_handles[path_idx];
        p.path_id = path_idx;
        p.path_hit_count = path_hist[path_idx];
        path_info.push_back(p);
    }


    // sort path_info by path_hit_count, from the smallest to the largest
    std::sort(path_info.begin(), path_info.end(), [](const PathInfo &a, const PathInfo &b) {
        return a.path_hit_count < b.path_hit_count;
    });


    // print path info
    std::cout << "Path info: " << std::endl;
    for (int i = 0; i < path_count; i++) {
        std::cout << "Path[" << path_info[i].path_id << "] " << 
        graph.get_path_name(path_info[i].path_handle) << ": " << "hit count: " << path_info[i].path_hit_count << "; num_steps: " << graph.get_step_count(path_info[i].path_handle) << std::endl;
    }
    


    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &(node_data.nodes[node_idx]);
        // coords[0], coords[1], coords[2], coords[3] are stored consecutively. 
        float *coords = n->coords;
        // check if coordinates valid (not NaN or infinite)
        for (int i = 0; i < 4; i++) {
            if (!isfinite(coords[i])) {
                std::cout << "WARNING: invalid coordiate" << std::endl;
            }
        }
        X[node_idx * 2].store(double(coords[0]));
        Y[node_idx * 2].store(double(coords[1]));
        X[node_idx * 2 + 1].store(double(coords[2]));
        Y[node_idx * 2 + 1].store(double(coords[3]));
        //std::cout << "coords of " << node_idx << ": [" << X[node_idx*2] << "; " << Y[node_idx*2] << "] ; [" << X[node_idx*2+1] << "; " << Y[node_idx*2+1] <<"]\n";
    }


    // get rid of CUDA data structures
    cudaFree(etas);
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(zetas);
#ifdef USE_GPU
    cudaFree(rnd_state);
    cudaFree(rnd_state_std);
#endif


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Entire Layout function time: " << duration_ms / 1000.0 << "s" << std::endl;
#endif

    return;
}

}
