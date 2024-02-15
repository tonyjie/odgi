#include "metrics.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>

// #define LOG_STRESS
// for geometric mean of each term's stress. 

// #define COUNT_TIME
// #define DEBUG

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


/*
* CUDA host function to compute the path stress: ALL layout_node pairs within each path, and sums up. 
*/
void cuda_path_stress(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout, int nthreads) {
    printf("CUDA kernel to compute path stress\n");

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

#ifdef DEBUG
    std::cout << "total_path_steps: " << path_data.total_path_steps << std::endl;
    uint64_t total_term_count_check = path_data.total_path_steps * SAMPLE_FACTOR;
    std::cout << "total_term_count_check: " << total_term_count_check << std::endl;
    exit(0);
#endif

    // get length and starting position of all paths
    uint32_t first_step_counter = 0;
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

    std::cout << "Preprocessing done" << std::endl;

    uint64_t total_term_count = path_data.total_path_steps * SAMPLE_FACTOR;
    const uint32_t block_size = BLOCK_SIZE;
    uint64_t block_nbr = (total_term_count + block_size - 1) / block_size;

    std::cout << "total_term_count: " << total_term_count << ", block_nbr: " << block_nbr << ", block_size: " << block_size << std::endl;

    // initialize random states
    curandState_t *rnd_state_tmp;
    curandStateCoalesced_t *rnd_state;
    cudaError_t tmp_error = cudaMallocManaged(&rnd_state_tmp, SM_COUNT * block_size * sizeof(curandState_t));        
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
    tmp_error = cudaMallocManaged(&rnd_state, SM_COUNT * sizeof(curandStateCoalesced_t));
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
    cuda_device_init_metric<<<SM_COUNT, block_size>>>(rnd_state_tmp, rnd_state);    
    tmp_error = cudaDeviceSynchronize();
    std::cout << "rnd state CUDA Error: " << cudaGetErrorName(tmp_error) << ": " << cudaGetErrorString(tmp_error) << std::endl;
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


    int sharedMemSize = block_size * sizeof(double);
    // compute the sampled path stress
    compute_sampled_path_stress<<<block_nbr, block_size, sharedMemSize>>>(node_data, path_data, total_term_count, rnd_state, blockSums, ignrSums);
    // check for errors
    cudaDeviceSynchronize();
    // check errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }

#ifdef COUNT_TIME
    // end time of kernel
    auto end_kernel = std::chrono::high_resolution_clock::now();
    uint32_t duration_kernel = std::chrono::duration_cast<std::chrono::seconds>(end_kernel - start_kernel).count();
    // std::cout << "Kernel time: " << (float)duration_kernel_ms / 1000 << " s" << std::endl;
    std::cout << "Kernel time: " << duration_kernel << " s" << std::endl;
#endif

    // Sum the block sums
    double total_path_stress = 0;
    #pragma openmp parallel for reduction(+:total_path_stress)
    for (int i = 0; i < block_nbr; i++) {
        total_path_stress += blockSums[i];
        // DEBUG: print the blockSums
        // std::cout << "block " << i << " has " << blockSums[i] << " stress" << std::endl;
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

    

    // free memory
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(rnd_state);
    cudaFree(blockSums);
    cudaFree(ignrSums);


    return;
}


} // namespace cuda