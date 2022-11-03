#include "layout.h"
#include <cuda.h>


namespace cuda {

__global__ void cuda_device_layout(cuda::node_t *node_data) {
    int32_t device_id = blockIdx.x * blockDim.x + threadIdx.x;
    node_t *n = &node_data[device_id];

    printf("Hello World from CUDA device: %i %f %f %f %f\n", n->seq_length, n->coords[0] ,n->coords[1] ,n->coords[2] ,n->coords[3]);
}


void cuda_layout(const odgi::graph_t &graph, std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y) {

#ifdef cuda_layout_profiling
    auto start = std::chrono::high_resolution_clock::now();
#endif


    std::cout << "Hello world from CUDA host" << std::endl;
    std::cout << "size of node_t: " << sizeof(node_t) << std::endl;

    // create node array
    // consisting of sequence length and coords
    uint32_t node_count = graph.get_node_count();
    std::cout << "node_count: " << node_count << std::endl;
    // TODO handle cases when min_node_id != 1
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() == node_count);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    cuda::node_t *node_data;
    cudaMallocManaged(&node_data, node_count * sizeof(cuda::node_t));
    // TODO parallelise with openmp
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        assert(graph.has_node(node_idx));
        cuda::node_t *n_tmp = &node_data[node_idx];

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


    // TODO: Transfer path_t data



    cuda_device_layout<<<1,10>>>(node_data);
    cudaDeviceSynchronize();


    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &node_data[node_idx];
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
    cudaFree(node_data);


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CUDA layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
