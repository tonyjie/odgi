#include "metrics.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include "cuda_runtime_api.h"

// #define LOG_STRESS
// for geometric mean of each term's stress. 

// #define COUNT_TIME
// #define DEBUG
// #define DEBUG_CHR16
// #define DEBUG_BLOCK

// #define DISTRIBUTION

#define PRINT_INFO

#define STDDEV


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


// __global__ void cuda_hello_device() {
//     printf("Hello World from CUDA device\n");
// }




// Given three collinear points p, q, r, the function checks if 
// point q lies on line segment 'pr' 
__device__
bool onSegment(Point p, Point q, Point r) 
{ 
	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && 
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y)) 
	return true; 

	return false; 
} 

// To find orientation of ordered triplet (p, q, r). 
// The function returns following values 
// 0 --> p, q and r are collinear 
// 1 --> Clockwise 
// 2 --> Counterclockwise 
__device__
int orientation(Point p, Point q, Point r) 
{ 
	// See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
	// for details of below formula. 
	int val = (q.y - p.y) * (r.x - q.x) - 
			(q.x - p.x) * (r.y - q.y); 

	if (val == 0) return 0; // collinear 

	return (val > 0)? 1: 2; // clock or counterclock wise 
} 

// returns True if pangenome graph node a and b intersect
__device__
bool doIntersect(cuda::node_t a, cuda::node_t b) 
{
    Point p1 = {a.coords[0], a.coords[1]};
    Point q1 = {a.coords[2], a.coords[3]};
    Point p2 = {b.coords[0], b.coords[1]};
    Point q2 = {b.coords[2], b.coords[3]};
	// Find the four orientations needed for general and 
	// special cases 
	int o1 = orientation(p1, q1, p2); 
	int o2 = orientation(p1, q1, q2); 
	int o3 = orientation(p2, q2, p1); 
	int o4 = orientation(p2, q2, q1); 

	// General case 
	if (o1 != o2 && o3 != o4) 
		return true; 

	// Special Cases 
	// p1, q1 and p2 are collinear and p2 lies on segment p1q1 
	if (o1 == 0 && onSegment(p1, p2, q1)) return true; 

	// p1, q1 and q2 are collinear and q2 lies on segment p1q1 
	if (o2 == 0 && onSegment(p1, q2, q1)) return true; 

	// p2, q2 and p1 are collinear and p1 lies on segment p2q2 
	if (o3 == 0 && onSegment(p2, p1, q2)) return true; 

	// p2, q2 and q1 are collinear and q1 lies on segment p2q2 
	if (o4 == 0 && onSegment(p2, q1, q2)) return true; 

	return false; // Doesn't fall in any of the above cases 
} 


// kernel function to check the number of node-crossing
__global__ 
void compute_node_crossing(cuda::node_data_t node_data, uint32_t N, int* blockSums) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    if (tid < N) {
        // count number of node-crossing for node pair (tid, i). This is computed by each thread. 
        for (int i = tid + 1; i < N; i++) {
            if (doIntersect(node_data.nodes[tid], node_data.nodes[i])) {
                count++;
            }
        }
    }

    extern __shared__ int sharedCounts[]; // shared memory for each block, used for reduction
    sharedCounts[threadIdx.x] = count;

    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedCounts[threadIdx.x] += sharedCounts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Store the partial sum
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sharedCounts[0];
    }
}


void cuda_node_crossing(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout) {
    printf("CUDA kernel to compute node crossing\n");


	// ========= Preprocessing: prepare data structure =========
    uint32_t node_count = graph.get_node_count();
    // std::cout << "node_count: " << node_count << std::endl;
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

        // copy coordinates
        n_tmp->coords[0] = layout.coords(h).x;              // start.x
        n_tmp->coords[1] = layout.coords(h).y;              // start.y
        n_tmp->coords[2] = layout.coords(graph.flip(h)).x;  // end.x
        n_tmp->coords[3] = layout.coords(graph.flip(h)).y;  // end.y
    }    

    // uint64_t node_pair_count = node_count * (node_count - 1) / 2;
    
    int blockSize = 1024;
    int sharedMemSize = blockSize * sizeof(int);
    int numBlocks = (node_count + blockSize - 1) / blockSize;

    // std::cout << "numBlocks: " << numBlocks << ", blockSize: " << blockSize << std::endl;

    int* blockSums; // partial sum for each block
    cudaMallocManaged(&blockSums, numBlocks * sizeof(int));

    compute_node_crossing<<<numBlocks, blockSize, sharedMemSize>>>(node_data, node_count, blockSums);
    // check for errors
    cudaDeviceSynchronize();

    // Sum the block sums
    uint64_t total_node_crossing = 0;
    #pragma openmp parallel for reduction(+:total_node_crossing)
    for (int i = 0; i < numBlocks; i++) {
        total_node_crossing += blockSums[i];
        // cout << "block " << i << " has " << blockSums[i] << " node-crossing" << endl;
    }
    std::cout << "total_node_crossing: " << total_node_crossing << std::endl;

    // Free memory
    cudaFree(node_data.nodes);
    cudaFree(blockSums);


    return;
}

/*
* coalesced version of curand(). Generates a 32-bit pseudorandom value for each thread.
*/
__device__ 
uint32_t cuda_coalesced_metric(curandStateCoalesced_t *state, uint32_t thread_id) {
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
    return state->d[thread_id] + state->w4[thread_id];
}

/*
* CUDA kernel to initialize curandStateCoalesced_t
*/
__global__ 
void cuda_device_init_metric(curandState_t *rnd_state_tmp, curandStateCoalesced_t *rnd_state) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // initialize curandState with original curand implementation
    curand_init(42+tid, tid, 0, &rnd_state_tmp[tid]);
    // curand_init(clock64()+tid, tid, 0, &rnd_state_tmp[tid]);
    // copy to coalesced data structure
    rnd_state[blockIdx.x].d[threadIdx.x] = rnd_state_tmp[tid].d;
    rnd_state[blockIdx.x].w0[threadIdx.x] = rnd_state_tmp[tid].v[0];
    rnd_state[blockIdx.x].w1[threadIdx.x] = rnd_state_tmp[tid].v[1];
    rnd_state[blockIdx.x].w2[threadIdx.x] = rnd_state_tmp[tid].v[2];
    rnd_state[blockIdx.x].w3[threadIdx.x] = rnd_state_tmp[tid].v[3];
    rnd_state[blockIdx.x].w4[threadIdx.x] = rnd_state_tmp[tid].v[4];
}


static __device__ __inline__ uint32_t __mysmid(){
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

#ifdef DISTRIBUTION
/*
* CUDA kernel to compute the sampled path stress and count the distribution as well
*/
__global__
void compute_sampled_path_stress_distribution(cuda::node_data_t node_data, cuda::path_data_t path_data, uint64_t total_term_count, curandStateCoalesced_t *rnd_state, double *blockSums, int *ignr_sums, unsigned long long int *hist_stress) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double stress = 0; // each threaed counts the stress of a node-pair
    int ignr_count = 0; // count the number of node-pairs that are directly connected

    if (tid < total_term_count) { // if the thread is doing something

        uint32_t smid = __mysmid();
        assert(smid < 84);
        curandStateCoalesced_t *thread_rnd_state = &rnd_state[smid];

        // select path
        uint32_t step_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % path_data.total_path_steps;
        uint32_t path_idx = path_data.element_array[step_idx].pidx;

        path_t p = path_data.paths[path_idx];

        uint32_t s1_idx, s2_idx, n1_id, n2_id;
        int n1_offset, n2_offset;
        double term_dist, layout_dist; // ground-truth and layout distance of a node-pair

        if (p.step_count == 1) { // single-node path
            // TODO: count this specific node's stress
            n1_id = p.elements[0].node_id;
            n2_id = n1_id;
            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length; 
            term_dist = n1_seq_length; // ground-truth distance
            n1_offset = 0; 
            n2_offset = 2;
        } // end of single-node path 
        else { // normal scenario

            s1_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % p.step_count;
            assert(s1_idx < p.step_count);
            do {
                s2_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % p.step_count;
            } while (s2_idx == s1_idx);
            
            assert(s1_idx < p.step_count);
            assert(s2_idx < p.step_count);
            assert(s1_idx != s2_idx);

            n1_id = p.elements[s1_idx].node_id;
            int64_t n1_pos_in_path = p.elements[s1_idx].pos;
            bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
            n1_pos_in_path = std::abs(n1_pos_in_path);

            n2_id = p.elements[s2_idx].node_id;
            int64_t n2_pos_in_path = p.elements[s2_idx].pos;
            bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
            n2_pos_in_path = std::abs(n2_pos_in_path);    
            
            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
            bool n1_use_other_end = (cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % 2 == 0) ? true: false;
            if (n1_use_other_end) {
                n1_pos_in_path += uint64_t(n1_seq_length);
                n1_use_other_end = !n1_is_rev;
            } else {
                n1_use_other_end = n1_is_rev;
            }

            uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
            bool n2_use_other_end =(cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % 2 == 0) ? true: false;
            if (n2_use_other_end) {
                n2_pos_in_path += uint64_t(n2_seq_length);
                n2_use_other_end = !n2_is_rev;
            } else {
                n2_use_other_end = n2_is_rev;
            }    

            term_dist = std::abs(n1_pos_in_path - n2_pos_in_path); // ground-truth distance of a node-pair
            // if (term_dist < 1e-9) { // we want to skip these connected nodes, since their ground-truth distance is 0, which leads to too large stress for a single term
            //     term_dist = 100;
            // }
            
            n1_offset = n1_use_other_end ? 2: 0;
            n2_offset = n2_use_other_end ? 2: 0;

            // if (threadIdx.x == 0) {
            //     // printf("tid: %d, term_dist: %f\n", tid, term_dist);
            // }

        } // end of normal scenario

        if (term_dist < 1e-9) { // we want to skip these connected nodes, since their ground-truth distance is 0, which leads to too large stress for a single term
            stress = 0;
            ignr_count++;
        } else { // normal case: node-pair not directly connected

            float *x1 = &node_data.nodes[n1_id].coords[n1_offset];
            float *x2 = &node_data.nodes[n2_id].coords[n2_offset];
            float *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
            float *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];
            double x1_val = double(*x1);
            double x2_val = double(*x2);
            double y1_val = double(*y1);
            double y2_val = double(*y2);    

            layout_dist = std::sqrt(std::pow(x1_val - x2_val, 2) + std::pow(y1_val - y2_val, 2)); // layout distance of a node-pair
            // compute the stress
            stress = std::pow( ((layout_dist - term_dist) / term_dist), 2);
            // reversed version of stress
            // if (layout_dist < 1e-4) {
            //     layout_dist = 1e-4;
            // }
            // stress = std::pow( ((layout_dist - term_dist) / layout_dist), 2);
            // stress = std::pow( ((layout_dist - term_dist) / (layout_dist + term_dist)), 2); // symmetric stress
        }

#ifdef DEBUG
    // if stress is very large, print out the node-pair
    if (stress > 1e8) {
    // if (term_dist == 1) {
        // get the current path
        printf("path_idx: %d, path_step_count: %d\n", path_idx, p.step_count);

        printf("tid: %ld, stress: %f, term_dist: %f, layout_dist: %f, n1_id: %d, n2_id: %d, n1_offset: %d, n2_offset: %d\n", tid, stress, term_dist, layout_dist, n1_id, n2_id, n1_offset, n2_offset);
    }
#endif

    // check where the stress falls into the hist_stress
    // region: 0-0.1, 0.1-1, 1-10, 10-100, 1e2-1e3, 1e3-1e4, 1e4-1e5, 1e5-1e6, 1e6-1e7, 1e7-1e8, 1e8-1e9, >1e9
    if (stress < 0.1) {
        atomicAdd(&hist_stress[0], 1);
    } else if (stress < 1) {
        atomicAdd(&hist_stress[1], 1);
    } else if (stress < 10) {
        atomicAdd(&hist_stress[2], 1);
    } else if (stress < 100) {
        atomicAdd(&hist_stress[3], 1);
    } else if (stress < 1e3) {
        atomicAdd(&hist_stress[4], 1);
    } else if (stress < 1e4) {
        atomicAdd(&hist_stress[5], 1);
    } else if (stress < 1e5) {
        atomicAdd(&hist_stress[6], 1);
    } else if (stress < 1e6) {
        atomicAdd(&hist_stress[7], 1);
    } else if (stress < 1e7) {
        atomicAdd(&hist_stress[8], 1);
    } else if (stress < 1e8) {
        atomicAdd(&hist_stress[9], 1);
    } else if (stress < 1e9) {
        atomicAdd(&hist_stress[10], 1);
    } else {
        atomicAdd(&hist_stress[11], 1);
    }


    } // end of if the thread is doing something


    // now each thread has a stress value, we need to sum up the stress of all node-pairs
    // NEXT: Then we need to sum up the stress of all node-pairs
    extern __shared__ double sharedStress[];

#ifdef LOG_STRESS
    // now we count geometric mean. so need to take log. Have to take log(1+stress) to avoid log(0) -> -inf
    sharedStress[threadIdx.x] = log(stress + 1);
#else
    sharedStress[threadIdx.x] = stress;
#endif
    // DEBUG: print it out
    // if (threadIdx.x == 0) {
    //     printf("tid: %d, log(stress+1): %f\n", tid, log(stress+1));
    // }

    __syncthreads();



    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedStress[threadIdx.x] += sharedStress[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // Store the partial sum
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sharedStress[0];
    }

    // sum up the number of node-pairs that are directly connected
    extern __shared__ int sharedIgnrCount[];
    sharedIgnrCount[threadIdx.x] = ignr_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedIgnrCount[threadIdx.x] += sharedIgnrCount[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ignr_sums[blockIdx.x] = sharedIgnrCount[0];
    }

}

#endif

#ifdef STDDEV
/*
* CUDA kernel to compute the STDDEV for sampled path stress
*/
__global__
void compute_sampled_stress_stddev(cuda::node_data_t node_data, cuda::path_data_t path_data, uint64_t total_term_count, curandStateCoalesced_t *rnd_state, double *blockSums, int *ignr_sums, double mean_stress) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double stress = 0; // each threaed counts the stress of a node-pair
    int ignr_count = 0; // count the number of node-pairs that are directly connected

    if (tid < total_term_count) { // if the thread is doing something

        uint32_t smid = __mysmid();
        assert(smid < 84);
        curandStateCoalesced_t *thread_rnd_state = &rnd_state[smid];

        // select path
        uint32_t step_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % path_data.total_path_steps;
        uint32_t path_idx = path_data.element_array[step_idx].pidx;

        path_t p = path_data.paths[path_idx];

        uint32_t s1_idx, s2_idx, n1_id, n2_id;
        int n1_offset, n2_offset;
        double term_dist, layout_dist; // ground-truth and layout distance of a node-pair

        if (p.step_count == 1) { // single-node path
            // TODO: count this specific node's stress
            n1_id = p.elements[0].node_id;
            n2_id = n1_id;
            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length; 
            term_dist = n1_seq_length; // ground-truth distance
            n1_offset = 0; 
            n2_offset = 2;
        } // end of single-node path 
        else { // normal scenario

            s1_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % p.step_count;
            assert(s1_idx < p.step_count);
            do {
                s2_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % p.step_count;
            } while (s2_idx == s1_idx);
            
            assert(s1_idx < p.step_count);
            assert(s2_idx < p.step_count);
            assert(s1_idx != s2_idx);

            n1_id = p.elements[s1_idx].node_id;
            int64_t n1_pos_in_path = p.elements[s1_idx].pos;
            bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
            n1_pos_in_path = std::abs(n1_pos_in_path);

            n2_id = p.elements[s2_idx].node_id;
            int64_t n2_pos_in_path = p.elements[s2_idx].pos;
            bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
            n2_pos_in_path = std::abs(n2_pos_in_path);    
            
            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
            bool n1_use_other_end = (cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % 2 == 0) ? true: false;
            if (n1_use_other_end) {
                n1_pos_in_path += uint64_t(n1_seq_length);
                n1_use_other_end = !n1_is_rev;
            } else {
                n1_use_other_end = n1_is_rev;
            }

            uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
            bool n2_use_other_end =(cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % 2 == 0) ? true: false;
            if (n2_use_other_end) {
                n2_pos_in_path += uint64_t(n2_seq_length);
                n2_use_other_end = !n2_is_rev;
            } else {
                n2_use_other_end = n2_is_rev;
            }    

            term_dist = std::abs(n1_pos_in_path - n2_pos_in_path); // ground-truth distance of a node-pair
            // if (term_dist < 1e-9) { // we want to skip these connected nodes, since their ground-truth distance is 0, which leads to too large stress for a single term
            //     term_dist = 100;
            // }
            
            n1_offset = n1_use_other_end ? 2: 0;
            n2_offset = n2_use_other_end ? 2: 0;

            // if (threadIdx.x == 0) {
            //     // printf("tid: %d, term_dist: %f\n", tid, term_dist);
            // }

        } // end of normal scenario

        if (term_dist < 1e-9) { // we want to skip these connected nodes, since their ground-truth distance is 0, which leads to too large stress for a single term
            stress = 0;
            ignr_count++;
        } else { // normal case: node-pair not directly connected

            float *x1 = &node_data.nodes[n1_id].coords[n1_offset];
            float *x2 = &node_data.nodes[n2_id].coords[n2_offset];
            float *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
            float *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];
            double x1_val = double(*x1);
            double x2_val = double(*x2);
            double y1_val = double(*y1);
            double y2_val = double(*y2);    

            layout_dist = std::sqrt(std::pow(x1_val - x2_val, 2) + std::pow(y1_val - y2_val, 2)); // layout distance of a node-pair
            // compute the stress
            stress = std::pow( ((layout_dist - term_dist) / term_dist), 2);

            // [STDEV] compute standard deviation. Here for simplicty just reuse the stress variable. 
            stress = std::pow( (stress - mean_stress), 2);

        }

#ifdef DEBUG
    // if stress is very large, print out the node-pair
    if (stress > 1e16) {
    // if (term_dist == 1) {
        // get the current path
        printf("path_idx: %d, path_step_count: %d\n", path_idx, p.step_count);

        printf("tid: %ld, stress: %f, term_dist: %f, layout_dist: %f, n1_id: %d, n2_id: %d, n1_offset: %d, n2_offset: %d\n", tid, stress, term_dist, layout_dist, n1_id, n2_id, n1_offset, n2_offset);
    }
#endif

    } // end of if the thread is doing something


    // now each thread has a stress value, we need to sum up the stress of all node-pairs
    // NEXT: Then we need to sum up the stress of all node-pairs
    extern __shared__ double sharedStress[];

#ifdef LOG_STRESS
    // now we count geometric mean. so need to take log. Have to take log(1+stress) to avoid log(0) -> -inf
    sharedStress[threadIdx.x] = log(stress + 1);
#else
    sharedStress[threadIdx.x] = stress;
#endif
    // DEBUG: print it out
    // if (threadIdx.x == 0) {
    //     printf("tid: %d, log(stress+1): %f\n", tid, log(stress+1));
    // }

    __syncthreads();



    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedStress[threadIdx.x] += sharedStress[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // Store the partial sum
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sharedStress[0];
    }

    // sum up the number of node-pairs that are directly connected
    extern __shared__ int sharedIgnrCount[];
    sharedIgnrCount[threadIdx.x] = ignr_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedIgnrCount[threadIdx.x] += sharedIgnrCount[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ignr_sums[blockIdx.x] = sharedIgnrCount[0];
    }

}
#endif



/*
* CUDA kernel to compute the sampled path stress
*/
__global__
void compute_sampled_path_stress(cuda::node_data_t node_data, cuda::path_data_t path_data, uint64_t total_term_count, curandStateCoalesced_t *rnd_state, double *blockSums, int *ignr_sums) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double stress = 0; // each threaed counts the stress of a node-pair
    int ignr_count = 0; // count the number of node-pairs that are directly connected

    if (tid < total_term_count) { // if the thread is doing something

        uint32_t smid = __mysmid();
        assert(smid < 84);
        curandStateCoalesced_t *thread_rnd_state = &rnd_state[smid];

        // select path
        uint32_t step_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % path_data.total_path_steps;
        uint32_t path_idx = path_data.element_array[step_idx].pidx;

        path_t p = path_data.paths[path_idx];

        uint32_t s1_idx, s2_idx, n1_id, n2_id;
        int n1_offset, n2_offset;
        double term_dist, layout_dist; // ground-truth and layout distance of a node-pair

        if (p.step_count == 1) { // single-node path
            // TODO: count this specific node's stress
            n1_id = p.elements[0].node_id;
            n2_id = n1_id;
            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length; 
            term_dist = n1_seq_length; // ground-truth distance
            n1_offset = 0; 
            n2_offset = 2;
        } // end of single-node path 
        else { // normal scenario

            s1_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % p.step_count;
            assert(s1_idx < p.step_count);
            do {
                s2_idx = cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % p.step_count;
            } while (s2_idx == s1_idx);
            
            assert(s1_idx < p.step_count);
            assert(s2_idx < p.step_count);
            assert(s1_idx != s2_idx);

            n1_id = p.elements[s1_idx].node_id;
            int64_t n1_pos_in_path = p.elements[s1_idx].pos;
            bool n1_is_rev = (n1_pos_in_path < 0)? true: false;
            n1_pos_in_path = std::abs(n1_pos_in_path);

            n2_id = p.elements[s2_idx].node_id;
            int64_t n2_pos_in_path = p.elements[s2_idx].pos;
            bool n2_is_rev = (n2_pos_in_path < 0)? true: false;
            n2_pos_in_path = std::abs(n2_pos_in_path);    
            
            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
            bool n1_use_other_end = (cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % 2 == 0) ? true: false;
            if (n1_use_other_end) {
                n1_pos_in_path += uint64_t(n1_seq_length);
                n1_use_other_end = !n1_is_rev;
            } else {
                n1_use_other_end = n1_is_rev;
            }

            uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
            bool n2_use_other_end =(cuda_coalesced_metric(thread_rnd_state, threadIdx.x) % 2 == 0) ? true: false;
            if (n2_use_other_end) {
                n2_pos_in_path += uint64_t(n2_seq_length);
                n2_use_other_end = !n2_is_rev;
            } else {
                n2_use_other_end = n2_is_rev;
            }    

            term_dist = std::abs(n1_pos_in_path - n2_pos_in_path); // ground-truth distance of a node-pair
            // if (term_dist < 1e-9) { // we want to skip these connected nodes, since their ground-truth distance is 0, which leads to too large stress for a single term
            //     term_dist = 100;
            // }
            
            n1_offset = n1_use_other_end ? 2: 0;
            n2_offset = n2_use_other_end ? 2: 0;

            // if (threadIdx.x == 0) {
            //     // printf("tid: %d, term_dist: %f\n", tid, term_dist);
            // }

        } // end of normal scenario

        if (term_dist < 1e-9) { // we want to skip these connected nodes, since their ground-truth distance is 0, which leads to too large stress for a single term
            stress = 0;
            ignr_count++;
        } else { // normal case: node-pair not directly connected

            float *x1 = &node_data.nodes[n1_id].coords[n1_offset];
            float *x2 = &node_data.nodes[n2_id].coords[n2_offset];
            float *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
            float *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];
            double x1_val = double(*x1);
            double x2_val = double(*x2);
            double y1_val = double(*y1);
            double y2_val = double(*y2);    

            layout_dist = std::sqrt(std::pow(x1_val - x2_val, 2) + std::pow(y1_val - y2_val, 2)); // layout distance of a node-pair
            // compute the stress
            stress = std::pow( ((layout_dist - term_dist) / term_dist), 2);
            // reversed version of stress
            // if (layout_dist < 1e-4) {
            //     layout_dist = 1e-4;
            // }
            // stress = std::pow( ((layout_dist - term_dist) / layout_dist), 2);
            // stress = std::pow( ((layout_dist - term_dist) / (layout_dist + term_dist)), 2); // symmetric stress
        }

#ifdef DEBUG
    // if stress is very large, print out the node-pair
    if (stress > 1e8) {
    // if (term_dist == 1) {
        // get the current path
        printf("path_idx: %d, path_step_count: %d\n", path_idx, p.step_count);

        printf("tid: %ld, stress: %f, term_dist: %f, layout_dist: %f, n1_id: %d, n2_id: %d, n1_offset: %d, n2_offset: %d\n", tid, stress, term_dist, layout_dist, n1_id, n2_id, n1_offset, n2_offset);
    }
#endif

    } // end of if the thread is doing something


    // now each thread has a stress value, we need to sum up the stress of all node-pairs
    // NEXT: Then we need to sum up the stress of all node-pairs
    extern __shared__ double sharedStress[];

#ifdef LOG_STRESS
    // now we count geometric mean. so need to take log. Have to take log(1+stress) to avoid log(0) -> -inf
    sharedStress[threadIdx.x] = log(stress + 1);
#else
    sharedStress[threadIdx.x] = stress;
#endif
    // DEBUG: print it out
    // if (threadIdx.x == 0) {
    //     printf("tid: %d, log(stress+1): %f\n", tid, log(stress+1));
    // }

    __syncthreads();



    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedStress[threadIdx.x] += sharedStress[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // Store the partial sum
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sharedStress[0];
    }

    // sum up the number of node-pairs that are directly connected
    extern __shared__ int sharedIgnrCount[];
    sharedIgnrCount[threadIdx.x] = ignr_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedIgnrCount[threadIdx.x] += sharedIgnrCount[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ignr_sums[blockIdx.x] = sharedIgnrCount[0];
    }

}

/*
* CUDA kernel to compute the ALL-PAIR path stress. 
*/
__global__
void compute_all_path_stress(cuda::node_data_t node_data, cuda::path_data_t path_data, uint64_t total_path_steps, double *blockSums, int *ignr_sums) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double stress_thread = 0; // the accumulated stress for this thread
    int ignr_count = 0; // count the number of node-pairs that are directly connected

    if (tid >= total_path_steps) { 
        // do nothing, just wait
    } 
    else { // if the thread is doing something

        int num_step_pair = 0; // the number of step-pairs chosen by this thread

        uint32_t s1_idx, s2_idx, n1_id, n2_id;
        int n1_offset, n2_offset;
        bool n1_use_other_end, n2_use_other_end;
        double term_dist, layout_dist; // ground-truth and layout distance of a node-pair


        // map each thread with one step "s". For that thread, it compute the stress of all step-pairs (s, t) where t > s, within that path
        uint32_t path_idx = path_data.element_array[tid].pidx;
        path_t p = path_data.paths[path_idx];
        s1_idx = tid - path_data.paths[path_idx].first_step_in_path;
        s2_idx = s1_idx + 1;

        if (p.step_count == 1) { // single-node path
            n1_id = p.elements[0].node_id;
            n2_id = n1_id;
            uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
            term_dist = n1_seq_length;
            n1_offset = 0;
            n2_offset = 2;

            double x1_val = double(node_data.nodes[n1_id].coords[n1_offset]);
            double x2_val = double(node_data.nodes[n2_id].coords[n2_offset]);
            double y1_val = double(node_data.nodes[n1_id].coords[n1_offset + 1]);
            double y2_val = double(node_data.nodes[n2_id].coords[n2_offset + 1]);

            layout_dist = std::sqrt(std::pow(x1_val - x2_val, 2) + std::pow(y1_val - y2_val, 2)); // layout distance of a node-pair
            // compute the stress
            stress_thread = std::pow( ((layout_dist - term_dist) / term_dist), 2);
        } // end of single-node path

        else { // normal case

            while (s2_idx < p.step_count) { // (s1, s2) is within the same path
                // ===== START: compute the stress of step-pair (s1, s2) =====
                // Let's start with only computing one end of the step-pair

                double stress = 0; // the stress of this step-pair

                // just let them to be false right now. Actually there are 4 cases for each step-pair. 
                // iterate through all 4 cases
                for (int i = 0; i < 4; i++) { // iterate through all 4 cases: (start, start), (start, end), (end, start), (end, end)
                    n1_use_other_end = (i & 1) ? true: false;
                    n2_use_other_end = (i & 2) ? true: false;
                
                    n1_id = p.elements[s1_idx].node_id;
                    int64_t n1_pos_in_path = p.elements[s1_idx].pos;
                    bool n1_is_rev = (n1_pos_in_path < 0) ? true: false;
                    n1_pos_in_path = std::abs(n1_pos_in_path);

                    n2_id = p.elements[s2_idx].node_id;
                    int64_t n2_pos_in_path = p.elements[s2_idx].pos;
                    bool n2_is_rev = (n2_pos_in_path < 0) ? true: false;
                    n2_pos_in_path = std::abs(n2_pos_in_path);

                    uint32_t n1_seq_length = node_data.nodes[n1_id].seq_length;
                    if (n1_use_other_end) {
                        n1_pos_in_path += uint64_t(n1_seq_length);
                        n1_use_other_end = !n1_is_rev;
                    } else {
                        n1_use_other_end = n1_is_rev;
                    }

                    uint32_t n2_seq_length = node_data.nodes[n2_id].seq_length;
                    if (n2_use_other_end) {
                        n2_pos_in_path += uint64_t(n2_seq_length);
                        n2_use_other_end = !n2_is_rev;
                    } else {
                        n2_use_other_end = n2_is_rev;
                    }

                    term_dist = std::abs(n1_pos_in_path - n2_pos_in_path); // ground-truth distance of a node-pair

                    n1_offset = n1_use_other_end ? 2: 0;
                    n2_offset = n2_use_other_end ? 2: 0;
                
                    if (term_dist < 1e-9) { // we want to skip these connected nodes, since their ground-truth distance is 0, which leads to too large stress for a single term
                        stress += 0;
                        ignr_count++;
                    } else { // normal case: node-pair not directly connected
                        double x1_val = double(node_data.nodes[n1_id].coords[n1_offset]);
                        double x2_val = double(node_data.nodes[n2_id].coords[n2_offset]);
                        double y1_val = double(node_data.nodes[n1_id].coords[n1_offset + 1]);
                        double y2_val = double(node_data.nodes[n2_id].coords[n2_offset + 1]);

                        layout_dist = std::sqrt(std::pow(x1_val - x2_val, 2) + std::pow(y1_val - y2_val, 2)); // layout distance of a node-pair
                        // compute the stress
                        stress += std::pow( ((layout_dist - term_dist) / term_dist), 2);

#ifdef DEBUG_ALL_PAIR
                        // DEBUG
                        if (stress == 0 && stress > 1e6) {
                            printf("path_idx: %d, s1_idx: %d, s2_idx: %d, term_dist: %f, layout_dist: %f, n1_id: %d, n2_id: %d, n1_offset: %d, n2_offset: %d\n", path_idx, s1_idx, s2_idx, term_dist, layout_dist, n1_id, n2_id, n1_offset, n2_offset);
                        }
#endif                        

                    }
                } // end of iterating through all 4 cases
                // ===== END: compute the stress of step-pair (s1, s2) =====
                stress_thread += stress;
                s2_idx++; // move to the next step-pair
                num_step_pair++;
            }
        } // end of normal case
    
    } // end of if the thread is doing something


    // print out the stress
    // if (blockIdx.x == 0 || blockIdx.x == 1) {
    //     printf("tid: %d, stress_thread: %f, num_step_pair: %d\n", tid, stress_thread, num_step_pair);
    // }


    // ===== Reduction =====
    extern __shared__ double sharedStress[];
    sharedStress[threadIdx.x] = stress_thread;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedStress[threadIdx.x] += sharedStress[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sharedStress[0];
    }

    // sum up the number of node-pairs that are directly connected
    extern __shared__ int sharedIgnrCount[];
    sharedIgnrCount[threadIdx.x] = ignr_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedIgnrCount[threadIdx.x] += sharedIgnrCount[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ignr_sums[blockIdx.x] = sharedIgnrCount[0];
    }



    return;
}



/*
* CUDA host function to compute the SAMPLED path stress: layout_node pairs within each path, and sums up. 
*/
void cuda_sampled_path_stress(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout, int nthreads) {
    printf("CUDA kernel to compute sampled path stress\n");

    // get cuda device property, and get the SM count
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    std::cout << "SM count: " << sm_count << std::endl;

#ifdef COUNT_TIME
    // start time
    auto start_preprocess = std::chrono::high_resolution_clock::now();
#endif    

    // Preprocessing: prepare data structure
    uint32_t node_count = graph.get_node_count();
    // std::cout << "node_count: " << node_count << std::endl;
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

        // copy coordinates
        n_tmp->coords[0] = layout.coords(h).x;              // start.x
        n_tmp->coords[1] = layout.coords(h).y;              // start.y
        n_tmp->coords[2] = layout.coords(graph.flip(h)).x;  // end.x
        n_tmp->coords[3] = layout.coords(graph.flip(h)).y;  // end.y
    }  
    // Preprocessing: prepare path structure
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
    uint64_t first_step_counter = 0;
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        int step_count = graph.get_step_count(p);
        path_data.paths[path_idx].step_count = step_count;
        path_data.paths[path_idx].first_step_in_path = first_step_counter;
        first_step_counter += step_count;
    }

#ifdef DEBUG_MORE
    // print the short path which has less than 5 steps
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        if (path_data.paths[path_idx].step_count < 5) {
            std::cout << std::endl << "path[" << path_idx << "]" << ", step_count: " << path_data.paths[path_idx].step_count << std::endl;
            // print their step position
            odgi::step_handle_t s = graph.path_begin(path_handles[path_idx]);
            for (int step_idx = 0; step_idx < path_data.paths[path_idx].step_count; step_idx++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                // print its length
                uint64_t len = graph.get_length(h);
                std::cout << "node[" << graph.get_id(h) - 1 << "], len: " << len << std::endl;

                // get next step
                if (graph.has_next_step(s)) {
                    s = graph.get_next_step(s);
                }
            }
        }
    }


#endif

#ifdef DEBUG_DETAIL


#ifdef DEBUG_CHR16
    // print the length of path[0]
    std::cout << "Path[0] length: " << graph.get_step_count(path_handles[0]) << std::endl;
    // go through path[0], print out the node_id and its length
    odgi::step_handle_t s = graph.path_begin(path_handles[0]);
    bool begin_count = false;
    int step_cnt_between = 0;
    int nuc_length_between = 0;
    for (int step_idx = 0; step_idx < graph.get_step_count(path_handles[0]); step_idx++) {
        odgi::handle_t h = graph.get_handle_of_step(s);
        // std::cout << "node[" << graph.get_id(h) - 1 << "], len: " << graph.get_length(h) << std::endl;

        if (graph.get_id(h) - 1 == 2670508) {
            std::cout << "node[2670508]: " << ", len: " << graph.get_length(h) << ", is_reverse: " << graph.get_is_reverse(h) << std::endl;
            begin_count = false;
        }

        if (begin_count) {
            step_cnt_between++;
            nuc_length_between += graph.get_length(h);
            std::cout << "node[" << graph.get_id(h) - 1 << "], len: " << graph.get_length(h) << ", is_reverse: " << graph.get_is_reverse(h) << std::endl;
        }

        // when first see node[2670508], start to count the number of steps, until see node[2670505]
        if (graph.get_id(h) - 1 == 2670505) {
            std::cout << "node[2670505]: " << ", len: " << graph.get_length(h) << ", is_reverse: " << graph.get_is_reverse(h) << std::endl;
            begin_count = true;
        }

        // get next step
        if (graph.has_next_step(s)) {
            s = graph.get_next_step(s);
        }
    }

    std::cout << "step_cnt_between: " << step_cnt_between << ", nuc_length_between: " << nuc_length_between << std::endl;


    // Path[436]
    // print its step and its length, and its reverse
    std::cout << "Path[436] length: " << graph.get_step_count(path_handles[436]) << std::endl;
    // go through path[0], print out the node_id and its length
    s = graph.path_begin(path_handles[436]);
    for (int step_idx = 0; step_idx < graph.get_step_count(path_handles[436]); step_idx++) {
        odgi::handle_t h = graph.get_handle_of_step(s);
        std::cout << "node[" << graph.get_id(h) - 1 << "], len: " << graph.get_length(h) << ", is_reverse: " << graph.get_is_reverse(h) << std::endl;
        // get next step
        if (graph.has_next_step(s)) {
            s = graph.get_next_step(s);
        }
    }


    // check from path[0], the layout distance of these node-pairs

    std::cout << "Path[0] critical area..." << std::endl;
    int node_0 = 2670505;
    int node_1 = 2670506;
    int node_2 = 2670508;

    auto node_0_start = layout.coords(graph.get_handle(node_0 + 1, false)); // the left end of this node
    auto node_0_end = layout.coords(graph.get_handle(node_0 + 1, true)); // the right end of this node
    auto node_1_start = layout.coords(graph.get_handle(node_1 + 1, false)); // the left end of this node
    auto node_1_end = layout.coords(graph.get_handle(node_1 + 1, true)); // the right end of this node
    auto node_2_start = layout.coords(graph.get_handle(node_2 + 1, false)); // the left end of this node
    auto node_2_end = layout.coords(graph.get_handle(node_2 + 1, true)); // the right end of this node

    double node_0_within_dist = odgi::algorithms::layout::coord_dist(node_0_start, node_0_end);
    double node_0_node_1_dist = odgi::algorithms::layout::coord_dist(node_0_end, node_1_start);
    double node_1_within_dist = odgi::algorithms::layout::coord_dist(node_1_start, node_1_end);
    double node_1_node_2_dist = odgi::algorithms::layout::coord_dist(node_1_end, node_2_start);
    double node_2_within_dist = odgi::algorithms::layout::coord_dist(node_2_start, node_2_end);

    // print out the layout distance
    std::cout << "node_0_within_dist: " << node_0_within_dist << std::endl;
    std::cout << "node_0_node_1_dist: " << node_0_node_1_dist << std::endl;
    std::cout << "node_1_within_dist: " << node_1_within_dist << std::endl;
    std::cout << "node_1_node_2_dist: " << node_1_node_2_dist << std::endl;
    std::cout << "node_2_within_dist: " << node_2_within_dist << std::endl;

    // check from path[436], the layout distance of these node-pairs
    std::cout << "Path[436] critical area..." << std::endl;
    int node_3 = 2670507;

    // for path[436] which is all reversed
    auto node_2_start_436 = node_2_end;
    auto node_2_end_436 = node_2_start;
    auto node_3_start_436 = layout.coords(graph.get_handle(node_3 + 1, true)); // the right end of this node
    auto node_3_end_436 = layout.coords(graph.get_handle(node_3 + 1, false)); // the left end of this node
    auto node_0_start_436 = node_0_end;
    auto node_0_end_436 = node_0_start;

    double node_2_node_3_dist = odgi::algorithms::layout::coord_dist(node_2_end_436, node_3_start_436);
    double node_3_within_dist = odgi::algorithms::layout::coord_dist(node_3_start_436, node_3_end_436);
    double node_3_node_0_dist = odgi::algorithms::layout::coord_dist(node_3_end_436, node_0_start_436);

    // print out the layout distance
    std::cout << "node_2_within_dist: " << node_2_within_dist << std::endl;
    std::cout << "node_2_node_3_dist: " << node_2_node_3_dist << std::endl;
    std::cout << "node_3_within_dist: " << node_3_within_dist << std::endl;
    std::cout << "node_3_node_0_dist: " << node_3_node_0_dist << std::endl;
    std::cout << "node_0_within_dist: " << node_0_within_dist << std::endl;
#endif


    // For the path with 3 steps
    // int node_id = 2670508;
    // // I want to check for node[1], how many paths it is in
    // for (int path_idx = 0; path_idx < path_count; path_idx++) {
    //     odgi::path_handle_t p = path_handles[path_idx];
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         if (graph.get_id(h) - 1 == node_id) {
    //             std::cout << "path[" << path_idx << "]: " << " has node[" << node_id << "]" << std::endl;
    //         }
    //     });
    // }

    // // check another node_id
    // node_id = 2670507;
    // for (int path_idx = 0; path_idx < path_count; path_idx++) {
    //     odgi::path_handle_t p = path_handles[path_idx];
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         if (graph.get_id(h) - 1 == node_id) {
    //             std::cout << "path[" << path_idx << "]: " << " has node[" << node_id << "]" << std::endl;
    //         }
    //     });
    // }
    // // check another node_id
    // node_id = 2670505;
    // for (int path_idx = 0; path_idx < path_count; path_idx++) {
    //     odgi::path_handle_t p = path_handles[path_idx];
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         if (graph.get_id(h) - 1 == node_id) {
    //             std::cout << "path[" << path_idx << "]: " << " has node[" << node_id << "]" << std::endl;
    //         }
    //     });
    // }


    // For the path with 4 steps

    // node_id = 1964057;
    // for (int path_idx = 0; path_idx < path_count; path_idx++) {
    //     odgi::path_handle_t p = path_handles[path_idx];
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         if (graph.get_id(h) - 1 == node_id) {
    //             std::cout << "path[" << path_idx << "]: " << " has node[" << node_id << "]" << std::endl;
    //         }
    //     });
    // }

    // node_id = 1964055;
    // for (int path_idx = 0; path_idx < path_count; path_idx++) {
    //     odgi::path_handle_t p = path_handles[path_idx];
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         if (graph.get_id(h) - 1 == node_id) {
    //             std::cout << "path[" << path_idx << "]: " << " has node[" << node_id << "]" << std::endl;
    //         }
    //     });
    // }

    // node_id = 1964054; 
    // for (int path_idx = 0; path_idx < path_count; path_idx++) {
    //     odgi::path_handle_t p = path_handles[path_idx];
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         if (graph.get_id(h) - 1 == node_id) {
    //             std::cout << "path[" << path_idx << "]: " << " has node[" << node_id << "]" << std::endl;
    //         }
    //     });
    // }

    // node_id = 1964052; 
    // for (int path_idx = 0; path_idx < path_count; path_idx++) {
    //     odgi::path_handle_t p = path_handles[path_idx];
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         if (graph.get_id(h) - 1 == node_id) {
    //             std::cout << "path[" << path_idx << "]: " << " has node[" << node_id << "]" << std::endl;
    //         }
    //     });
    // }

    // exit(0);
#endif



#pragma omp parallel for num_threads(nthreads)
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        //std::cout << graph.get_path_name(p) << ": " << graph.get_step_count(p) << std::endl;

        uint32_t step_count = path_data.paths[path_idx].step_count;
        uint64_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;
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
#ifdef PRINT_INFO
    std::cout << "Preprocessing done" << std::endl;
#endif
    uint64_t total_term_count = path_data.total_path_steps * SAMPLE_FACTOR;
    const uint32_t block_size = BLOCK_SIZE;
    uint64_t block_nbr = (total_term_count + block_size - 1) / block_size;

    std::cout << "total_term_count: " << total_term_count << ", block_nbr: " << block_nbr << ", block_size: " << block_size << std::endl;

    // initialize random states
    curandState_t *rnd_state_tmp;
    curandStateCoalesced_t *rnd_state;
    CUDACHECK(cudaMallocManaged(&rnd_state_tmp, sm_count * block_size * sizeof(curandState_t)));
    CUDACHECK(cudaMallocManaged(&rnd_state, sm_count * sizeof(curandStateCoalesced_t)));
    cuda_device_init_metric<<<sm_count, block_size>>>(rnd_state_tmp, rnd_state);
    CUDACHECK(cudaGetLastError());  
    CUDACHECK(cudaDeviceSynchronize());
    cudaFree(rnd_state_tmp);

#ifdef COUNT_TIME
    // end time of preprocessing
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    uint32_t duration_preprocess = std::chrono::duration_cast<std::chrono::seconds>(end_preprocess - start_preprocess).count();
    std::cout << "Preprocessing time: " << (float)duration_preprocess << " s" << std::endl;

    // start time of kernel
    auto start_kernel = std::chrono::high_resolution_clock::now();
#endif

    // blockSums: partial sum for each block
    double *blockSums;
    cudaMallocManaged(&blockSums, block_nbr * sizeof(double));
    // ignrSums: partial sum for each block to count the number of node-pairs that are directly connected. We skip the stress of these node-pair, since their stress could be too large.
    int *ignrSums;
    cudaMallocManaged(&ignrSums, block_nbr * sizeof(int));

#ifdef DISTRIBUTION
    // create a histogram for each stress term. Within the kernel, we will compute the stress of each term, and send it into the correct bin of histogram
    // region: 0-0.1, 0.1-1, 1-10, 10-100, 1e2-1e3, 1e3-1e4, 1e4-1e5, 1e5-1e6, 1e6-1e7, 1e7-1e8, 1e8-1e9, >1e9
    // std::vector<int> hist_stress(12, 0);
    // this historgram should be sent into the kernel, so that each thread can update the histogram
    
    unsigned long long int *hist_stress;
    CUDACHECK(cudaMallocManaged(&hist_stress, 12 * sizeof(unsigned long long int)));
    for (int i = 0; i < 12; i++) {
        hist_stress[i] = 0;
    }

#endif

    int sharedMemSize = block_size * sizeof(double);

#ifdef DISTRIBUTION
    compute_sampled_path_stress_distribution<<<block_nbr, block_size, sharedMemSize>>>(node_data, path_data, total_term_count, rnd_state, blockSums, ignrSums, hist_stress);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    // print out the hist_stress
    std::cout << "===== [Distribution] Histogram for stress =====" << std::endl;
    for (int i = 0; i < 12; i++) {
        if (i == 0) {
            std::cout << "0-0.1: " << hist_stress[i] << std::endl;
        }
        else if (i == 11) {
            std::cout << ">1e9: " << hist_stress[i] << std::endl;
        } else {
            std::cout << pow(10, i - 2) << "-" << pow(10, i - 1) << ": " << hist_stress[i] << std::endl;
        }
    }
    // free
    cudaFree(hist_stress);
#else
    // compute the sampled path stress
    compute_sampled_path_stress<<<block_nbr, block_size, sharedMemSize>>>(node_data, path_data, total_term_count, rnd_state, blockSums, ignrSums);
    CUDACHECK(cudaGetLastError());
    // check for errors
    CUDACHECK(cudaDeviceSynchronize());
#endif

#ifdef COUNT_TIME
    // end time of kernel
    auto end_kernel = std::chrono::high_resolution_clock::now();
    uint32_t duration_kernel = std::chrono::duration_cast<std::chrono::seconds>(end_kernel - start_kernel).count();
    // std::cout << "Kernel time: " << (float)duration_kernel_ms / 1000 << " s" << std::endl;
    std::cout << "Kernel time: " << duration_kernel << " s" << std::endl;
#endif

#ifdef DISTRIBUTION
    // create a histogram for the blockSums
    // Average stress
    // region: 0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-N
    // std::vector<int> hist(10, 0);
    // for (int i = 0; i < block_nbr; i++) {
    //     int idx = int(blockSums[i] / 10 / BLOCK_SIZE);
    //     if (idx >= 10) {
    //         idx = 9;
    //     }
    //     hist[idx]++;
    // }

    // print out the histogram
    // std::cout << "===== [Distribution] Histogram for blockSums =====" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     std::cout << i * 10 << "-" << (i + 1) * 10 << ": " << hist[i] << std::endl;
    // }
    // std::cout << std::endl;

    // region: 0-1; 1-10; 10-100; 100-1000; 1000-10000; 10000+
    std::vector<int> hist(6, 0);
    for (int i = 0; i < block_nbr; i++) {
        int idx = int(blockSums[i] / BLOCK_SIZE);
        if (idx < 1) {
            hist[0]++;
        } else if (idx < 10) {
            hist[1]++;
        } else if (idx < 100) {
            hist[2]++;
        } else if (idx < 1000) {
            hist[3]++;
        } else if (idx < 10000) {
            hist[4]++;
        } else {
            hist[5]++;
        }
    }
    // print out the histogram
    std::cout << "===== [Distribution] Histogram for blockSums =====" << std::endl;
    std::cout << "0-1: " << hist[0] << std::endl;
    std::cout << "1-10: " << hist[1] << std::endl;
    std::cout << "10-100: " << hist[2] << std::endl;
    std::cout << "100-1000: " << hist[3] << std::endl;
    std::cout << "1000-10000: " << hist[4] << std::endl;
    std::cout << "10000+: " << hist[5] << std::endl;


#endif


    // Sum the block sums
    double total_path_stress = 0;
    #pragma openmp parallel for reduction(+:total_path_stress)
    for (int i = 0; i < block_nbr; i++) {
        total_path_stress += blockSums[i];
#ifdef DEBUG_BLOCK
        // DEBUG: print the blockSums
        // std::cout << "block " << i << " has " << blockSums[i] << " stress" << std::endl;
        
        // if blockSums are very large, print out the blockSums
        if (blockSums[i] > block_size * 1000) {
            std::cout << "block " << i << " has average of " << blockSums[i] / BLOCK_SIZE << " stress" << std::endl;
        }
#endif
    }

    // Sum the block sums for ignrSums
    int total_ignr_count = 0;
    #pragma openmp parallel for reduction(+:total_ignr_count)
    for (int i = 0; i < block_nbr; i++) {
        total_ignr_count += ignrSums[i];
    }

    // normalized by total_term_count
    total_path_stress /= (total_term_count - total_ignr_count);
#ifdef LOG_STRESS
    // exponentiate to get geometric mean
    total_path_stress = exp(total_path_stress);
#endif
    std::cout << "path_stress: " << total_path_stress << std::endl;
    std::cout << "total_ignr_count: " << total_ignr_count << std::endl;

#ifdef STDDEV
    double mean_stress = total_path_stress;
    // another pass to compute the standard deviation

    std::cout << "Compute STDDEV......" << std::endl;
    // initialize random states
    CUDACHECK(cudaMallocManaged(&rnd_state_tmp, sm_count * block_size * sizeof(curandState_t)));
    CUDACHECK(cudaMallocManaged(&rnd_state, sm_count * sizeof(curandStateCoalesced_t)));
    cuda_device_init_metric<<<sm_count, block_size>>>(rnd_state_tmp, rnd_state); 
    CUDACHECK(cudaGetLastError());   
    CUDACHECK(cudaDeviceSynchronize());
    cudaFree(rnd_state_tmp);

    // change to compute STDDEV
    compute_sampled_stress_stddev<<<block_nbr, block_size, sharedMemSize>>>(node_data, path_data, total_term_count, rnd_state, blockSums, ignrSums, mean_stress);
    CUDACHECK(cudaGetLastError());
    // check for errors
    CUDACHECK(cudaDeviceSynchronize());

    // Sum the block sums
    double total_std_dev = 0;
    #pragma openmp parallel for reduction(+:total_std_dev)
    for (int i = 0; i < block_nbr; i++) {
        total_std_dev += blockSums[i];
#ifdef DEBUG_BLOCK
        // DEBUG: print the blockSums
        // std::cout << "block " << i << " has " << blockSums[i] << " stress" << std::endl;
        
        // if blockSums are very large, print out the blockSums
        if (blockSums[i] > block_size * 1000) {
            std::cout << "block " << i << " has " << blockSums[i] << " stress" << std::endl;
        }
#endif
    }


    // Sum the block sums for ignrSums
    total_ignr_count = 0;
    #pragma openmp parallel for reduction(+:total_ignr_count)
    for (int i = 0; i < block_nbr; i++) {
        total_ignr_count += ignrSums[i];
    }    

    // final stddev
    double stddev = std::sqrt(total_std_dev / (total_term_count - total_ignr_count));

    std::cout << "stddev: " << stddev << std::endl;
    std::cout << "ignr_count: " << total_ignr_count << std::endl;

    // 95% confidence interval
    double stddev_div_sqrtn = stddev / std::sqrt(total_term_count - total_ignr_count);
    double conf_interval = 1.96 * stddev_div_sqrtn;
    double conf_interval_low = mean_stress - conf_interval;
    double conf_interval_high = mean_stress + conf_interval;

    std::cout << "conf interval: " << conf_interval << std::endl;
    std::cout << "95% confidence interval: [" << conf_interval_low << ", " << conf_interval_high << "]" << std::endl;

#endif
    

    // free memory
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(rnd_state);
    cudaFree(blockSums);
    cudaFree(ignrSums);


    return;
}

/*
* CUDA host function to compute the ALL-PAIR path stress: layout_node pairs within each path, and sums up.
*/
void cuda_all_pair_path_stress(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout, int nthreads) {
    std::cout << "CUDA kernel to compute all-pair path stress" << std::endl;
    
    // ===== Preprocessing: prepare data structure for nodes =====
    uint32_t node_count = graph.get_node_count();
    // std::cout << "node_count: " << node_count << std::endl;
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

        // copy coordinates
        n_tmp->coords[0] = layout.coords(h).x;              // start.x
        n_tmp->coords[1] = layout.coords(h).y;              // start.y
        n_tmp->coords[2] = layout.coords(graph.flip(h)).x;  // end.x
        n_tmp->coords[3] = layout.coords(graph.flip(h)).y;  // end.y
    }  
    // ===== Preprocessing: prepare data structure for paths =====
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
    uint64_t first_step_counter = 0;
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        int step_count = graph.get_step_count(p);
        path_data.paths[path_idx].step_count = step_count;
        path_data.paths[path_idx].first_step_in_path = first_step_counter;
        first_step_counter += step_count;
    }

#pragma omp parallel for num_threads(nthreads)
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        odgi::path_handle_t p = path_handles[path_idx];
        //std::cout << graph.get_path_name(p) << ": " << graph.get_step_count(p) << std::endl;

        uint32_t step_count = path_data.paths[path_idx].step_count;
        uint64_t first_step_in_path = path_data.paths[path_idx].first_step_in_path;
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

    // print some info for the graph
    std::cout << "node_count: " << node_count << ", path_count: " << path_count << ", total_path_steps: " << path_data.total_path_steps << std::endl;

    // count the total number of step-pairs within each path
    // For each path, there are (N = path_data.paths[path_idx].step_count) steps; there are N*(N-1)/2 step-pairs
    uint64_t total_term_count = 0;
    for (int path_idx = 0; path_idx < path_count; path_idx++) {
        uint32_t step_count = path_data.paths[path_idx].step_count;
        // std::cout << "path[" << path_idx << "]: " << step_count << " steps" << std::endl;
        total_term_count += step_count * (step_count - 1) / 2;
    }
    // there are 4 cases for each step-pair
    total_term_count *= 4;
    std::cout << "total_term_count: " << total_term_count << std::endl;


    // map each thread withi one step "s". For that thread, it compute the stress of all step-pairs (s, t) where t > s, within that path
    const uint32_t block_size = BLOCK_SIZE;
    uint64_t block_nbr = (path_data.total_path_steps + block_size - 1) / block_size;
    std::cout << "block_nbr: " << block_nbr << ", block_size: " << block_size << std::endl;

    // blockSums: partial sum for each block
    double *blockSums;
    cudaMallocManaged(&blockSums, block_nbr * sizeof(double));
    // ignrSums: partial sum for each block to count the number of node-pairs that are directly connected. We skip the stress of these node-pair, since their stress could be too large.
    int *ignrSums;
    cudaMallocManaged(&ignrSums, block_nbr * sizeof(int));

    int sharedMemSize = block_size * sizeof(double);
    // kernel: compute the all-pair path stress
    compute_all_path_stress<<<block_nbr, block_size, sharedMemSize>>>(node_data, path_data, path_data.total_path_steps, blockSums, ignrSums);
    cudaDeviceSynchronize();
    // check errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }

    // Sum the block sums
    double total_path_stress = 0;
    #pragma openmp parallel for reduction(+:total_path_stress)
    for (int i = 0; i < block_nbr; i++) {
        total_path_stress += blockSums[i];
#ifdef DEBUG_ALL_PAIR
        std::cout << "block " << i << " has " << blockSums[i] << " stress" << std::endl;
#endif
    }

    // Sum the block sums for ignrSums
    int total_ignr_count = 0;
    #pragma openmp parallel for reduction(+:total_ignr_count)
    for (int i = 0; i < block_nbr; i++) {
        total_ignr_count += ignrSums[i];
    }

    // normalized by total_term_count
    total_path_stress /= (total_term_count - total_ignr_count);
    std::cout << "path_stress: " << total_path_stress << std::endl;
    std::cout << "total_ignr_count: " << total_ignr_count << std::endl;


    // free memory
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(blockSums);
    cudaFree(ignrSums);
}

} // namespace cuda