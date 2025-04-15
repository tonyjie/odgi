#include "layout.h"
#include <cuda.h>
#include <assert.h>
#include "cuda_runtime_api.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace cuda {

__global__ void cuda_device_init(curandState_t *rnd_state_tmp, curandStateCoalesced_t *rnd_state) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // initialize curandState with original curand implementation
    curand_init(42+tid, tid, 0, &rnd_state_tmp[tid]);
    // copy to coalesced data structure
    rnd_state[blockIdx.x].d[threadIdx.x] = rnd_state_tmp[tid].d;
    rnd_state[blockIdx.x].w0[threadIdx.x] = rnd_state_tmp[tid].v[0];
    rnd_state[blockIdx.x].w1[threadIdx.x] = rnd_state_tmp[tid].v[1];
    rnd_state[blockIdx.x].w2[threadIdx.x] = rnd_state_tmp[tid].v[2];
    rnd_state[blockIdx.x].w3[threadIdx.x] = rnd_state_tmp[tid].v[3];
    rnd_state[blockIdx.x].w4[threadIdx.x] = rnd_state_tmp[tid].v[4];
}

/**
 * @brief: Return 32-bits of pseudorandomness from an XORWOW generator. from "curand_kernel.h"
 * For some use cases, we don't need floating point uniform distribution. So we don't need to call `curand_uniform_coalesced` as below. We shall use this function. 
 * \param state - Pointer to state to update
 * \param thread_id - Thread id
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
*/
__device__ 
unsigned int curand_coalesced(curandStateCoalesced_t *state, uint32_t thread_id) {
    // Return 32-bits of pseudorandomness from an XORWOW generator. 
    uint32_t t;
    t = (state->w0[thread_id] ^ (state->w0[thread_id] >> 2));
    state->w0[thread_id] = state->w1[thread_id];
    state->w1[thread_id] = state->w2[thread_id];
    state->w2[thread_id] = state->w3[thread_id];
    state->w3[thread_id] = state->w4[thread_id];
    state->w4[thread_id] = (state->w4[thread_id] ^ (state->w4[thread_id] << 4)) ^ (t ^ (t << 1));
    state->d[thread_id] += 362437;    
    return state->w4[thread_id] + state->d[thread_id];
}

__device__
float curand_uniform_coalesced(curandStateCoalesced_t *state, uint32_t thread_id) {
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

__device__ 
void update_pos_gpu(int64_t &n1_pos_in_path, uint32_t &n1_id, int &n1_offset,
                    int64_t &n2_pos_in_path, uint32_t &n2_id, int &n2_offset,
                    double eta, 
                    cuda::node_data_t &node_data) 
{
    double term_dist = fabs((double)n1_pos_in_path - (double)n2_pos_in_path);
    if (term_dist < 1e-9) {
        term_dist = 1e-9;
    }

    double w_ij = 1.0 / term_dist;
    double mu = eta * w_ij;
    if (mu > 1.0) {
        mu = 1.0;
    }

    float *x1 = &node_data.nodes[n1_id].coords[n1_offset];
    float *x2 = &node_data.nodes[n2_id].coords[n2_offset];
    float *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
    float *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];

    double x1_val = (double)(*x1);
    double x2_val = (double)(*x2);
    double y1_val = (double)(*y1);
    double y2_val = (double)(*y2);

    double dx = x1_val - x2_val;
    double dy = y1_val - y2_val;

    if (dx == 0.0) {
        dx = 1e-9;
    }

    double mag = sqrt(dx * dx + dy * dy);
    double delta = mu * (mag - term_dist) * 0.5;
    double r = delta / mag;
    double r_x = r * dx;
    double r_y = r * dy;

    // Overwriting style: last writer wins
    atomicExch(x1, (float)(x1_val - r_x));
    atomicExch(x2, (float)(x2_val + r_x));
    atomicExch(y1, (float)(y1_val - r_y));
    atomicExch(y2, (float)(y2_val + r_y)); 
}

// ------------------------------------------------------------------
// Single-Launch Multi-Iteration Kernel
// ------------------------------------------------------------------
__global__ 
void gpu_layout_kernel(const cuda::layout_config_t config,
                                  cuda::curandStateCoalesced_t *rnd_state,
                                  const double * __restrict__ etas,
                                  const double * __restrict__ zetas,
                                  cuda::node_data_t node_data,
                                  cuda::path_data_t path_data,
                                  const int sm_count, 
                                  int iter)
{
    // Thread info
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smid = __mysmid();

    // Instead of asserting, we can safely return if SM count is exceeded
    // assert(smid < sm_count);
    if (smid >= (uint32_t)sm_count) {
        return;
    }

    // This thread's random state pointer
    cuda::curandStateCoalesced_t* thread_rnd_state = &rnd_state[smid];

    // One boolean per warp in this block
    __shared__ bool cooling[BLOCK_SIZE / WARP_SIZE];


    // Gather warp info
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // Let exactly one thread in each warp set the boolean
    if (laneId == 0) {
        // same logic: if we've passed first_cooling_iteration, cooling = true,
        // or randomly if not
        bool warp_cool = (iter >= (int)config.first_cooling_iteration) ||
            ((curand_coalesced(thread_rnd_state, threadIdx.x) % 2) == 0);
        cooling[warpId] = warp_cool;
    }
    // ensure warp sees the updated value
    __syncwarp();  // warp-level sync is enough, but __syncthreads() also works

    bool do_cooling = cooling[warpId];
    double eta = etas[iter];

    // Select a random step in any path
    // random index into path_data.element_array
    uint32_t step_idx = curand_coalesced(thread_rnd_state, threadIdx.x)
                        % path_data.total_path_steps;

    // Access the path index in that step
    uint32_t path_idx = path_data.element_array[step_idx].pidx;
    cuda::path_t p = path_data.paths[path_idx];

    // If the chosen path has too few steps, skip
    if (p.step_count < 2) {
        return;
    }
    // In performance builds, you may remove next assert
    // assert(p.step_count > 1);

    // pick a random step s1 within path p
    uint32_t s1_idx = curand_coalesced(thread_rnd_state, threadIdx.x)
                        % p.step_count;
    // assert(s1_idx < p.step_count);

    uint32_t s2_idx;
    if (do_cooling) {
        // “Cooling” logic from original code
        bool backward;
        uint32_t jump_space;
        uint32_t path_last = p.step_count - 1;
        // pick forward/backward:
        // the original code’s condition can be simplified but is left intact
        uint32_t r = curand_coalesced(thread_rnd_state, threadIdx.x) % 2;
        if ((s1_idx > 0 && r == 0) || (s1_idx == path_last)) {
            backward = true;
            jump_space = min(config.space, s1_idx);
        } else {
            backward = false;
            jump_space = min(config.space, p.step_count - s1_idx - 1);
        }

        // clamp jump_space if bigger than space_max
        uint32_t space = jump_space;
        if (jump_space > config.space_max) {
            space = config.space_max +
                (jump_space - config.space_max) / config.space_quantization_step + 1;
        }

        uint32_t z_i = cuda_rnd_zipf(thread_rnd_state,
                                        jump_space,
                                        config.theta,
                                        zetas[2],
                                        zetas[space]);
        s2_idx = backward ? (s1_idx - z_i) : (s1_idx + z_i);
    } else {
        // non-cooling logic
        do {
            s2_idx = curand_coalesced(thread_rnd_state, threadIdx.x)
                        % p.step_count;
        } while (s1_idx == s2_idx);
    }
    // assert(s2_idx < p.step_count);

    // Retrieve node info
    uint32_t n1_id = p.elements[s1_idx].node_id;
    int64_t n1_pos_in_path = p.elements[s1_idx].pos;
    bool n1_is_rev = (n1_pos_in_path < 0);
    n1_pos_in_path = llabs(n1_pos_in_path);

    uint32_t n2_id = p.elements[s2_idx].node_id;
    int64_t n2_pos_in_path = p.elements[s2_idx].pos;
    bool n2_is_rev = (n2_pos_in_path < 0);
    n2_pos_in_path = llabs(n2_pos_in_path);

    uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
    bool n1_use_other_end =
        ((curand_coalesced(thread_rnd_state, threadIdx.x) % 2) == 0);

    if (n1_use_other_end) {
        n1_pos_in_path += (uint64_t)n1_seq_length;
        n1_use_other_end = !n1_is_rev;
    } else {
        n1_use_other_end = n1_is_rev;
    }
    int n1_offset = n1_use_other_end ? 2 : 0;

    uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
    bool n2_use_other_end =
        ((curand_coalesced(thread_rnd_state, threadIdx.x) % 2) == 0);

    if (n2_use_other_end) {
        n2_pos_in_path += (uint64_t)n2_seq_length;
        n2_use_other_end = !n2_is_rev;
    } else {
        n2_use_other_end = n2_is_rev;
    }
    int n2_offset = n2_use_other_end ? 2 : 0;

    // Finally, update positions
    update_pos_gpu(n1_pos_in_path, n1_id, n1_offset,
                    n2_pos_in_path, n2_id, n2_offset,
                    eta, node_data);

}


void gpu_layout(layout_config_t config, const odgi::graph_t &graph,
                std::vector<std::atomic<double>> &X,
                std::vector<std::atomic<double>> &Y) {
    std::cout << "===== Use GPU to compute odgi-layout =====" << std::endl;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;
    
    // Create eta array in managed memory.
    double *etas;
    cudaMallocManaged(&etas, config.iter_max * sizeof(double));
    const int32_t iter_max = config.iter_max;
    const int32_t iter_with_max_learning_rate = config.iter_with_max_learning_rate;
    const double w_max = 1.0;
    const double eps = config.eps;
    const double eta_max = config.eta_max;
    const double eta_min = eps / w_max;
    const double lambda = log(eta_max / eta_min) / (double(iter_max) - 1);
    for (int32_t i = 0; i < config.iter_max; i++) {
        double eta = eta_max * exp(-lambda * (std::abs(i - iter_with_max_learning_rate)));
        etas[i] = isnan(eta) ? eta_min : eta;
    }
    
    // Allocate and fill node data using the new SoA layout.
    uint32_t node_count = graph.get_node_count();
    cuda::node_data_t node_data;
    node_data.node_count = node_count;
    cudaMallocManaged(&node_data.nodes, node_count * sizeof(cuda::node_t));

    // Initialize node coords
    for (int node_idx = 0; node_idx < (int)node_count; node_idx++) {
        cuda::node_t *n_tmp = &node_data.nodes[node_idx];
        auto h = graph.get_handle(node_idx + 1, false);
        n_tmp->seq_length = graph.get_length(h);
        n_tmp->coords[0] = float(X[node_idx*2].load());
        n_tmp->coords[1] = float(Y[node_idx*2].load());
        n_tmp->coords[2] = float(X[node_idx*2 + 1].load());
        n_tmp->coords[3] = float(Y[node_idx*2 + 1].load());
    }
    
    // Create path data structure.
    uint32_t path_count = graph.get_path_count();
    path_data_t path_data;
    path_data.path_count = path_count;
    path_data.total_path_steps = 0;
    cudaMallocManaged(&path_data.paths, path_count * sizeof(path_t));
    
    std::vector<odgi::path_handle_t> path_handles;
    path_handles.reserve(path_count);
    graph.for_each_path_handle([&](const odgi::path_handle_t &p) {
        path_handles.push_back(p);
        path_data.total_path_steps += graph.get_step_count(p);
    });
    cudaMallocManaged(&path_data.element_array, path_data.total_path_steps * sizeof(path_element_t));
    
    uint64_t first_step_counter = 0;
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
        uint32_t step_count = path_data.paths[path_idx].step_count;
        uint64_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;
        if (step_count == 0) {
            path_data.paths[path_idx].elements = nullptr;
        } else {
            path_element_t *cur_path = &path_data.element_array[first_step_in_path];
            path_data.paths[path_idx].elements = cur_path;
            
            odgi::step_handle_t s = graph.path_begin(p);
            int64_t pos = 1;
            for (int step_idx = 0; step_idx < step_count; step_idx++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                cur_path[step_idx].node_id = graph.get_id(h) - 1;
                cur_path[step_idx].pidx = static_cast<uint32_t>(path_idx);
                cur_path[step_idx].pos = graph.get_is_reverse(h) ? -pos : pos;
                pos += graph.get_length(h);
                if (graph.has_next_step(s)) {
                    s = graph.get_next_step(s);
                }
            }
        }
    }
    
    // Cache zipf zetas.
    auto start_zeta = std::chrono::high_resolution_clock::now();
    double *zetas;
    uint64_t zetas_cnt = ((config.space <= config.space_max) ? config.space : 
                          (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;
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
    
    // Set up random states.
    const uint64_t block_size = BLOCK_SIZE;
    uint64_t block_nbr = (config.min_term_updates + block_size - 1) / block_size; 
    
    curandState_t *rnd_state_tmp;
    curandStateCoalesced_t *rnd_state;
    CUDACHECK(cudaMallocManaged(&rnd_state_tmp, sm_count * block_size * sizeof(curandState_t)));
    CUDACHECK(cudaMallocManaged(&rnd_state, sm_count * sizeof(curandStateCoalesced_t)));
    cuda_device_init<<<sm_count, block_size>>>(rnd_state_tmp, rnd_state);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    cudaFree(rnd_state_tmp);
    
    // Launch the persistent kernel.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int iter = 0; iter < config.iter_max; iter++) {
        gpu_layout_kernel<<<block_nbr, block_size>>>(config, rnd_state, etas, zetas, node_data, path_data, sm_count, iter);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total CUDA kernel time: " << milliseconds << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy back node coordinates to host vectors.
    for (int node_idx = 0; node_idx < (int)node_count; node_idx++) {
        cuda::node_t *n = &node_data.nodes[node_idx];
        float *coords = n->coords;
        // check for validity if desired
        X[node_idx * 2].store(double(coords[0]));
        Y[node_idx * 2].store(double(coords[1]));
        X[node_idx * 2 + 1].store(double(coords[2]));
        Y[node_idx * 2 + 1].store(double(coords[3]));
    }
    
    // Free memory.
    cudaFree(etas);
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(zetas);
    cudaFree(rnd_state);
}

}