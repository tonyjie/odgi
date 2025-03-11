#include "layout.h"
#include <cuda.h>
#include <assert.h>
#include "cuda_runtime_api.h"

// #define CREATE_DIV

#define O1_CODE

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

__global__ void cuda_device_init(curandState *rnd_state) {
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

// o1
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>

// Comment out or define this to control debug statements
// #define DEBUG_KERNEL

// Example struct if combining x/y
struct float2coords {
    float2* data; // data size: node_count * 2 (for both orientations)
};


// o1

__global__ void cuda_device_layout(
    int iter,
    cuda::layout_config_t config,
    curandState *rnd_state,
    double eta,
    double *zetas,
    cuda::node_data_t node_data,
    cuda::path_data_t path_data,
    uint32_t *pidx_array,
    int64_t *pos_array,
    uint32_t *node_id_array,
    float2coords coords,
    int32_t *seq_length_array,
    int sm_count)
{
    // --------------------------------------------------------------------
    // Thread/SM indexing and random state
    // --------------------------------------------------------------------
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smid = __mysmid();  // custom function
#ifdef DEBUG_KERNEL
    if (smid >= sm_count) {
        printf("Error: smid %u out of range %d\n", smid, sm_count);
        return;
    }
#endif
    curandState *thread_rnd_state = &rnd_state[smid * 1024 + threadIdx.x];

    // --------------------------------------------------------------------
    // Randomly pick step and path
    // --------------------------------------------------------------------
    uint32_t step_idx = curand(thread_rnd_state) % path_data.total_path_steps;
    uint32_t path_idx = pidx_array[step_idx];
    path_t p = path_data.paths[path_idx];

    if (p.step_count < 2) {
        // trivial path, skip
        return;
    }

    // pick s1_idx, s2_idx
    uint32_t s1_idx = curand(thread_rnd_state) % p.step_count;
    uint32_t s2_idx;

    // Cooling or not
    bool cooling = (iter >= config.first_cooling_iteration) ||
                   ((curand(thread_rnd_state) % 2) == 0);

    if (cooling) {
        // Branch can cause warp divergence, but we keep it for correctness
        bool go_backward = false;
        // Weighted condition for going backward
        if ((s1_idx > 0 && (curand(thread_rnd_state) % 2 == 0)) ||
            (s1_idx == p.step_count - 1)) {
            go_backward = true;
        }

        // Compute jump space
        uint32_t jump_space =
            go_backward
                ? min(config.space, s1_idx)
                : min(config.space, p.step_count - s1_idx - 1);

        uint32_t space = jump_space;
        if (jump_space > config.space_max) {
            space = config.space_max +
                    (jump_space - config.space_max) / config.space_quantization_step + 1;
        }

        // sample from zipf
        uint32_t z_i = cuda_rnd_zipf(thread_rnd_state,
                                     jump_space,
                                     config.theta,
                                     zetas[2],
                                     zetas[space]);

#ifdef DEBUG_KERNEL
        if (go_backward) {
            if (!(z_i <= s1_idx)) {
                printf("Error (thread %i): s1_idx=%u z_i=%u\n",
                       threadIdx.x, s1_idx, z_i);
            }
        } else {
            if (!(z_i <= p.step_count - s1_idx - 1)) {
                printf("Error (thread %i): %u + %u > step_count\n",
                       threadIdx.x, s1_idx, z_i);
            }
        }
#endif
        // pick s2
        s2_idx = go_backward ? (s1_idx - z_i) : (s1_idx + z_i);
    } else {
        // Non-cooling branch
        do {
            s2_idx = curand(thread_rnd_state) % p.step_count;
        } while (s1_idx == s2_idx);
    }

    // --------------------------------------------------------------------
    // Lookup node IDs and positions
    // --------------------------------------------------------------------
    uint32_t n1_id = node_id_array[p.first_step_in_path + s1_idx];
    int64_t n1_pos_in_path = pos_array[p.first_step_in_path + s1_idx];
    bool n1_is_rev = n1_pos_in_path < 0;
    if (n1_is_rev) n1_pos_in_path = -n1_pos_in_path;

    uint32_t n2_id = node_id_array[p.first_step_in_path + s2_idx];
    int64_t n2_pos_in_path = pos_array[p.first_step_in_path + s2_idx];
    bool n2_is_rev = n2_pos_in_path < 0;
    if (n2_is_rev) n2_pos_in_path = -n2_pos_in_path;

    // Decide whether we attach at other end
    uint32_t n1_seq_length = seq_length_array[n1_id];
    bool n1_use_other_end = (curand(thread_rnd_state) % 2 == 0);
    if (n1_use_other_end) {
        n1_pos_in_path += uint64_t(n1_seq_length);
        n1_use_other_end = !n1_is_rev;
    } else {
        n1_use_other_end = n1_is_rev;
    }

    uint32_t n2_seq_length = seq_length_array[n2_id];
    bool n2_use_other_end = (curand(thread_rnd_state) % 2 == 0);
    if (n2_use_other_end) {
        n2_pos_in_path += uint64_t(n2_seq_length);
        n2_use_other_end = !n2_is_rev;
    } else {
        n2_use_other_end = n2_is_rev;
    }

    // --------------------------------------------------------------------
    // Compute layout influences
    // --------------------------------------------------------------------
    double term_dist = fabs(double(n1_pos_in_path) - double(n2_pos_in_path));
    if (term_dist < 1e-9) {
        term_dist = 1e-9;
    }

    double w_ij = 1.0 / term_dist;
    double mu = eta * w_ij;
    if (mu > 1.0) mu = 1.0;
    double d_ij = term_dist;

    int n1_offset = (n1_use_other_end ? 1 : 0);
    int n2_offset = (n2_use_other_end ? 1 : 0);

    // --------------------------------------------------------------------
    // Load coords (float2) for each node+offset
    // --------------------------------------------------------------------
    float2 c1 = coords.data[n1_id * 2 + n1_offset];
    float2 c2 = coords.data[n2_id * 2 + n2_offset];

    double dx = double(c1.x) - double(c2.x);
    double dy = double(c1.y) - double(c2.y);

    if (fabs(dx) < 1e-12) dx = 1e-9;

    double mag = sqrt(dx * dx + dy * dy);
    double delta = mu * (mag - d_ij) / 2.0;

    if (mag < 1e-12) {
        // If the nodes are extremely close, skip or do a small fix
        return;
    }

    double r = delta / mag;
    double r_x = r * dx;
    double r_y = r * dy;

    // --------------------------------------------------------------------
    // Update partial movement using atomicAdd

    // use atomicExch
    atomicExch(&(coords.data[n1_id * 2 + n1_offset].x), float(double(c1.x) - r_x));
    atomicExch(&(coords.data[n1_id * 2 + n1_offset].y), float(double(c1.y) - r_y));
    atomicExch(&(coords.data[n2_id * 2 + n2_offset].x), float(double(c2.x) + r_x));
    atomicExch(&(coords.data[n2_id * 2 + n2_offset].y), float(double(c2.y) + r_y));
}



// o3-mini
/*
// Optimized kernel: note the new signature uses __restrict__ and const qualifiers
__global__ void cuda_device_layout(int iter,
    const layout_config_t config,
    curandState * __restrict__ rnd_state,
    double eta,
    const double * __restrict__ zetas,
    node_data_t node_data,
    path_data_t path_data,
    const uint32_t * __restrict__ pidx_array,
    const int64_t * __restrict__ pos_array,
    const uint32_t * __restrict__ node_id_array,
    float * __restrict__ x_coords,
    float * __restrict__ y_coords,
    const int32_t * __restrict__ seq_length_array,
    int sm_count)
{
    // thread id and SM id in current grid configuration
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smid = __mysmid();  // Assumes __mysmid() returns the SM ID (and that only one block per SM executes concurrently)
    curandState *thread_rnd_state = &rnd_state[smid * blockDim.x + threadIdx.x];

    // For better memory behavior, have all threads in a warp pick the same “step” index.
    const unsigned int laneId = threadIdx.x & 31;  // threadIdx.x % 32
    uint32_t step_idx;
    if (laneId == 0)
    {
        step_idx = curand(thread_rnd_state) % path_data.total_path_steps;
    }
    step_idx = __shfl_sync(0xFFFFFFFF, step_idx, 0);

    // Look up the path index from the LUT (all threads in the warp have the same step_idx)
    uint32_t path_idx = pidx_array[step_idx];
    path_t p = path_data.paths[path_idx];
    if (p.step_count < 2)
        return;  // Nothing to do if path too short

    // Generate the first index in the path (each thread gets its own s1 index)
    uint32_t s1_idx = curand(thread_rnd_state) % p.step_count;
    uint32_t s2_idx;

    // Decide whether to use a cooling move.
    // (We use bit masking to check a coin flip instead of % 2; note the iter check remains)
    bool cooling = (iter >= config.first_cooling_iteration) || ((curand(thread_rnd_state) & 1u) == 0);

    if (cooling)
    {
        // Branch into backward/forward changes; note that we combine the coin flip into a temporary.
        bool go_backward = false;
        if (s1_idx > 0 && ((curand(thread_rnd_state) & 1u) == 0))
            go_backward = true;
        if (s1_idx == p.step_count - 1)
            go_backward = true;  // must go backward if at last step

        if (go_backward)
        {
            uint32_t jump_space = min(config.space, s1_idx);
            uint32_t space = jump_space;
            if (jump_space > config.space_max)
            {
                space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
            }
            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state, jump_space, config.theta, zetas[2], zetas[space]);
            assert(z_i <= s1_idx);
            s2_idx = s1_idx - z_i;
        }
        else
        {
            uint32_t jump_space = min(config.space, p.step_count - s1_idx - 1);
            uint32_t space = jump_space;
            if (jump_space > config.space_max)
            {
                space = config.space_max + (jump_space - config.space_max) / config.space_quantization_step + 1;
            }
            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state, jump_space, config.theta, zetas[2], zetas[space]);
            assert(s1_idx + z_i < p.step_count);
            s2_idx = s1_idx + z_i;
        }
    }
    else
    {
        // When not cooling, choose another random step (ensure s2_idx != s1_idx)
        do {
            s2_idx = curand(thread_rnd_state) % p.step_count;
        } while (s1_idx == s2_idx);
    }

    // Cache the base offset (first step in this path) for use in several accesses.
    uint64_t base = p.first_step_in_path;

    // Retrieve node IDs and positions for the two steps.
    uint32_t n1_id = node_id_array[base + s1_idx];
    int64_t n1_pos_in_path = pos_array[base + s1_idx];
    bool n1_is_rev = (n1_pos_in_path < 0);
    n1_pos_in_path = (n1_pos_in_path < 0) ? -n1_pos_in_path : n1_pos_in_path;

    uint32_t n2_id = node_id_array[base + s2_idx];
    int64_t n2_pos_in_path = pos_array[base + s2_idx];
    bool n2_is_rev = (n2_pos_in_path < 0);
    n2_pos_in_path = (n2_pos_in_path < 0) ? -n2_pos_in_path : n2_pos_in_path;

    // Decide on “other-end” use via a coin flip.
    uint32_t n1_seq_length = seq_length_array[n1_id];
    bool n1_use_other_end = ((curand(thread_rnd_state) & 1u) == 0);
    if (n1_use_other_end)
    {
        n1_pos_in_path += n1_seq_length;
        n1_use_other_end = !n1_is_rev;
    }
    else
    {
        n1_use_other_end = n1_is_rev;
    }

    uint32_t n2_seq_length = seq_length_array[n2_id];
    bool n2_use_other_end = ((curand(thread_rnd_state) & 1u) == 0);
    if (n2_use_other_end)
    {
        n2_pos_in_path += n2_seq_length;
        n2_use_other_end = !n2_is_rev;
    }
    else
    {
        n2_use_other_end = n2_is_rev;
    }

    // Compute the “term distance” and weight.
    double term_dist = fabs(double(n1_pos_in_path) - double(n2_pos_in_path));
    if (term_dist < 1e-9)
        term_dist = 1e-9;
    double w_ij = 1.0 / term_dist;
    double mu = eta * w_ij;
    if (mu > 1.0)
        mu = 1.0;
    double d_ij = term_dist;

    // Determine coordinate offsets depending on whether the “other-end” is used.
    int n1_offset = n1_use_other_end ? 1 : 0;
    int n2_offset = n2_use_other_end ? 1 : 0;

    // Compute pointers to the coordinate values. (Each node has two coordinate entries.)
    float *x1 = &x_coords[n1_id * 2 + n1_offset];
    float *x2 = &x_coords[n2_id * 2 + n2_offset];
    float *y1 = &y_coords[n1_id * 2 + n1_offset];
    float *y2 = &y_coords[n2_id * 2 + n2_offset];
    double x1_val = double(*x1);
    double x2_val = double(*x2);
    double y1_val = double(*y1);
    double y2_val = double(*y2);

    // Compute the displacement between the two nodes.
    double dx = x1_val - x2_val;
    double dy = y1_val - y2_val;
    if (dx == 0.0)
        dx = 1e-9;
    double mag = sqrt(dx * dx + dy * dy);
    double delta = mu * (mag - d_ij) / 2.0;
    double r = delta / mag;
    double r_x = r * dx;
    double r_y = r * dy;

    // Update the coordinates with atomic operations.
    atomicExch(x1, float(x1_val - r_x));
    atomicExch(x2, float(x2_val + r_x));
    atomicExch(y1, float(y1_val - r_y));
    atomicExch(y2, float(y2_val + r_y));
}
*/


// original
/*
__global__ void cuda_device_layout(int iter, cuda::layout_config_t config, curandState *rnd_state, double eta, double *zetas, cuda::node_data_t node_data,
        cuda::path_data_t path_data, uint32_t *pidx_array, int64_t *pos_array, uint32_t *node_id_array, float *x_coords, float *y_coords, int32_t *seq_length_array, int sm_count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smid = __mysmid();
    assert(smid < sm_count);
    curandState *thread_rnd_state = &rnd_state[smid * 1024 + threadIdx.x];

    // select path
    uint32_t step_idx = curand(thread_rnd_state) % path_data.total_path_steps;
    assert(step_idx < path_data.total_path_steps);

    // find path of step of specific thread with LUT (threads in warp pick same path)
    uint32_t path_idx = pidx_array[step_idx];


    path_t p = path_data.paths[path_idx];
    if (p.step_count < 2) {
        return;
    }
    assert(p.step_count > 1);

    // INFO: curand_uniform generates random values between 0.0 (excluded) and 1.0 (included)
    uint32_t s1_idx = curand(thread_rnd_state) % p.step_count;
    assert(s1_idx < p.step_count);
    uint32_t s2_idx;

    bool cooling = (iter >= config.first_cooling_iteration) || (curand(thread_rnd_state) % 2 == 0);

    if (cooling) {
        if (s1_idx > 0 && (curand(thread_rnd_state) % 2 == 0) || s1_idx == p.step_count - 1) {
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
            s2_idx = curand(thread_rnd_state) % p.step_count;
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
    bool n1_use_other_end = (curand(thread_rnd_state) % 2 == 0) ? true: false;
    if (n1_use_other_end) {
        n1_pos_in_path += uint64_t{n1_seq_length};
        n1_use_other_end = !n1_is_rev;
    } else {
        n1_use_other_end = n1_is_rev;
    }

    uint32_t n2_seq_length = seq_length_array[n2_id];
    bool n2_use_other_end = (curand(thread_rnd_state) % 2 == 0) ? true: false;
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
        uint32_t *pidx_array, int64_t *pos_array, uint32_t *node_id_array, float *x_coords, float *y_coords, int32_t *seq_length_array) {
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
                float *x1 = &x_coords[n1_id * 2 + n1_offset];
                float *x2 = &x_coords[n2_id * 2 + n2_offset];
                float *y1 = &y_coords[n1_id * 2 + n1_offset];
                float *y2 = &y_coords[n2_id * 2 + n2_offset];

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
    // get cuda device property, and get the SM count
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    std::cout << "SM count: " << sm_count << std::endl;

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


    int32_t *seq_length_array;
    cudaMallocManaged(&seq_length_array, node_count * sizeof(int32_t));
#ifdef O1_CODE    
    float2 *coords_array = nullptr;
    cudaMallocManaged(&coords_array, node_count * 2 * sizeof(float2));
#else
    float *x_coords;
    float *y_coords;
    cudaMallocManaged(&x_coords, node_count * 2 * sizeof(float));
    cudaMallocManaged(&y_coords, node_count * 2 * sizeof(float));
#endif

    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        // sequence length
        const handlegraph::handle_t h = graph.get_handle(node_idx + 1, false);
        // NOTE: unable store orientation (reverse), since this information is path dependent
        seq_length_array[node_idx] = graph.get_length(h);

#ifdef O1_CODE   
        float xF = float(X[node_idx * 2].load());
        float yF = float(Y[node_idx * 2].load());
        coords_array[node_idx * 2 + 0] = make_float2(xF, yF);

        float xF2 = float(X[node_idx * 2 + 1].load());
        float yF2 = float(Y[node_idx * 2 + 1].load());
        coords_array[node_idx * 2 + 1] = make_float2(xF2, yF2);
#else
        // copy random coordinates
        x_coords[node_idx * 2] = float(X[node_idx * 2].load());
        y_coords[node_idx * 2] = float(Y[node_idx * 2].load());
        x_coords[node_idx * 2 + 1] = float(X[node_idx * 2 + 1].load());
        y_coords[node_idx * 2 + 1] = float(Y[node_idx * 2 + 1].load());
#endif
    }

#ifdef O1_CODE
    // Wrap our float2* into a float2coords struct
    cuda::float2coords coords_container;
    coords_container.data = coords_array;
#endif

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

    // npi_iv in original implementation
    uint32_t *pidx_array;
    cudaMallocManaged(&pidx_array, path_data.total_path_steps * sizeof(uint32_t));

    int64_t *pos_array;
    cudaMallocManaged(&pos_array, path_data.total_path_steps * sizeof(int64_t));

    uint32_t *node_id_array;
    cudaMallocManaged(&node_id_array, path_data.total_path_steps * sizeof(uint32_t));

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

        uint32_t step_count = path_data.paths[path_idx].step_count;
        uint32_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;

        if (step_count > 0) {
            odgi::step_handle_t s = graph.path_begin(p);
            int64_t pos = 1;
            // Iterate through path
            for (int step_idx = 0; step_idx < step_count; step_idx++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                node_id_array[first_step_in_path + step_idx] = graph.get_id(h) - 1;
                pidx_array[first_step_in_path + step_idx] = uint32_t(path_idx);
                // store position negative when handle reverse
                if (graph.get_is_reverse(h)) {
                    pos_array[first_step_in_path + step_idx] = -pos;
                } else {
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
    double *zetas;
    uint64_t zetas_cnt = ((config.space <= config.space_max)? config.space : (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;

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


#define USE_GPU
#ifdef USE_GPU
    std::cout << "cuda gpu layout" << std::endl;

    const uint64_t block_size = BLOCK_SIZE;
    uint64_t block_nbr = (config.min_term_updates + block_size - 1) / block_size;
    std::cout << "block_nbr: " << block_nbr << " block_size: " << block_size << std::endl;
    curandState *rnd_state;
    CUDACHECK(cudaMallocManaged(&rnd_state, sm_count * block_size * sizeof(curandState)));
    cuda_device_init<<<sm_count, block_size>>>(rnd_state);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

// with o1 kernel
#ifdef O1_CODE
    for (int iter = 0; iter < (int)config.iter_max; iter++) {
        cuda_device_layout<<<block_nbr, block_size>>>(
            iter,
            config,
            rnd_state,
            etas[iter],
            zetas,                // assume you have your zipf array
            node_data,            // e.g. node_data_t
            path_data,            // e.g. path_data_t
            pidx_array,           // device pointer
            pos_array,            // device pointer
            node_id_array,        // device pointer
            coords_container,     // float2coords
            seq_length_array,     // device pointer
            sm_count
        );
    }

#else
    for (int iter = 0; iter < config.iter_max; iter++) {
        cuda_device_layout<<<block_nbr, block_size>>>(iter, config, rnd_state, etas[iter], zetas, node_data, path_data, pidx_array, pos_array, node_id_array, x_coords, y_coords, seq_length_array, sm_count);
    }

#endif

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total CUDA kernel time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

#ifdef O1_CODE
    // ---------------------------------------------------------------------
    // Copy results from coords_array back into X, Y
    // ---------------------------------------------------------------------
    for (int node_idx = 0; node_idx < (int)node_count; node_idx++) {
        float2 cF  = coords_array[node_idx * 2 + 0];
        float2 cF2 = coords_array[node_idx * 2 + 1];
        X[node_idx * 2].store(double(cF.x));
        Y[node_idx * 2].store(double(cF.y));
        X[node_idx * 2 + 1].store(double(cF2.x));
        Y[node_idx * 2 + 1].store(double(cF2.y));
    }
#else
    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        X[node_idx * 2].store(double(x_coords[node_idx * 2]));
        Y[node_idx * 2].store(double(y_coords[node_idx * 2]));
        X[node_idx * 2 + 1].store(double(x_coords[node_idx * 2 + 1]));
        Y[node_idx * 2 + 1].store(double(y_coords[node_idx * 2 + 1]));
    }
#endif

    // get rid of CUDA data structures
    cudaFree(etas);
    cudaFree(path_data.paths);
    cudaFree(zetas);

    cudaFree(pidx_array);
    cudaFree(pos_array);
    cudaFree(node_id_array);
#ifdef O1_CODE
    cudaFree(coords_array);
#else
    cudaFree(x_coords);
    cudaFree(y_coords);
#endif
    cudaFree(seq_length_array);
#ifdef USE_GPU
    cudaFree(rnd_state);
#endif

    return;
}

}
