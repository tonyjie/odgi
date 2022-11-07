#include "layout.h"
#include <cuda.h>


namespace cuda {

__global__ void cuda_device_layout(cuda::layout_config_t config, cuda::node_data_t &node_data, cuda::path_data_t &path_data) {
    int32_t device_id = blockIdx.x * blockDim.x + threadIdx.x;
    node_t *n = &node_data.nodes[device_id];

    //printf("Hello World from CUDA device: %i %f %f %f %f\n", n->seq_length, n->coords[0] ,n->coords[1] ,n->coords[2] ,n->coords[3]);
    printf("CUDA device %i: step_count: %i\n", device_id, path_data.paths[device_id].step_count);
}


void cpu_layout(cuda::layout_config_t config, cuda::node_data_t &node_data, cuda::path_data_t &path_data) {
    int nbr_threads = 40;

#pragma omp parallel for num_threads(nbr_threads)
    for (int tid = 0; tid < nbr_threads; tid++) {

        node_t *n = &node_data.nodes[tid];
        printf("CPU device %i: step_count: %i\n", tid, path_data.paths[tid].step_count);
    }
}


void cuda_layout(layout_config_t config, const odgi::graph_t &graph, std::vector<std::atomic<double>> &X, std::vector<std::atomic<double>> &Y) {

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
        n_tmp->coords[0] = X[node_idx * 2].load();
        n_tmp->coords[1] = Y[node_idx * 2].load();
        n_tmp->coords[2] = X[node_idx * 2 + 1].load();
        n_tmp->coords[3] = Y[node_idx * 2 + 1].load();
    }


    // TODO: Transfer path_t data
    uint32_t path_count = graph.get_path_count();
    cuda::path_data_t path_data;
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



//#define USE_CUDA
#ifdef USE_CUDA
    cuda_device_layout<<<1,10>>>(config, node_data, path_data);
    cudaDeviceSynchronize();
#else
    cpu_layout(config, node_data, path_data);
#endif



    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &node_data.nodes[node_idx];
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
    cudaFree(node_data.nodes);
    for (int i = 0; i < path_count; i++) {
        cudaFree(path_data.paths[i].elements);
    }
    cudaFree(path_data.paths);


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CUDA layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
