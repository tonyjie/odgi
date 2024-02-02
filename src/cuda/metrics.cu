#include "metrics.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>

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
    if (tid >= N) {
        return;
    }
    
    // count number of node-crossing for node pair (tid, i). This is computed by each thread. 
    int count = 0;
    for (int i = tid + 1; i < N; i++) {
        if (doIntersect(node_data.nodes[tid], node_data.nodes[i])) {
            count++;
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

        // copy random coordinates
        n_tmp->coords[0] = layout.coords(h).x;              // start.x
        n_tmp->coords[1] = layout.coords(h).y;              // start.y
        n_tmp->coords[2] = layout.coords(graph.flip(h)).x;  // end.x
        n_tmp->coords[3] = layout.coords(graph.flip(h)).y;  // end.y
    }    

    uint64_t node_pair_count = node_count * (node_count - 1) / 2;
    
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


    return;
}

/*
* CUDA host function to compute the path stress: ALL layout_node pairs within each path, and sums up. 
*/
void cuda_path_stress(const odgi::graph_t &graph, odgi::algorithms::layout::Layout &layout) {
    printf("CUDA kernel to compute path stress\n");

    // Preprocessing: prepare data structure
    





    return;
}


} // namespace cuda