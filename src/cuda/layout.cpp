#include "layout.h"
//#include <cuda.h>
#include <assert.h>


namespace cuda {

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

        for (int iter = 0; iter < config.iter_max; iter++ ) {
            // synchronize all threads before each iteration
#pragma omp barrier
            for (int step = 0; step < steps_per_thread; step++ ) {
                // get path
                uint32_t path_idx = rand_path(gen);
                path_t p = path_data.paths[path_idx];
                if (p.step_count < 2) {
                    continue;
                }

                std::uniform_int_distribution<uint32_t> rand_step(0, p.step_count-1);

                uint32_t s1_idx = rand_step(gen);
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

                double w_ij = 1.0 / term_dist;

                double mu = etas[iter] * w_ij;
                if (mu > 1.0) {
                    mu = 1.0;
                }

                double d_ij = term_dist;

                int n1_offset = n1_use_other_end? 2: 0;
                int n2_offset = n2_use_other_end? 2: 0;

                std::atomic<float> *x1 = &node_data.nodes[n1_id].coords[n1_offset];
                std::atomic<float> *x2 = &node_data.nodes[n2_id].coords[n2_offset];
                std::atomic<float> *y1 = &node_data.nodes[n1_id].coords[n1_offset + 1];
                std::atomic<float> *y2 = &node_data.nodes[n2_id].coords[n2_offset + 1];

                double dx = float(x1->load() - x2->load());
                double dy = float(y1->load() - y2->load());

                if (dx == 0.0) {
                    dx = 1e-9;
                }

                double mag = sqrt(dx * dx + dy * dy);
                double delta = mu * (mag - d_ij) / 2.0;
                //double delta_abs = std::abs(delta);

                double r = delta / mag;
                double r_x = r * dx;
                double r_y = r * dy;

                x1->store(x1->load() - float(r_x));
                y1->store(y1->load() - float(r_y));
                x2->store(x2->load() + float(r_x));
                y2->store(y2->load() + float(r_y));

            }
        }

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
    node_data.nodes = (cuda::node_t*) malloc(node_count * sizeof(cuda::node_t));
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        //assert(graph.has_node(node_idx));
        cuda::node_t *n_tmp = &node_data.nodes[node_idx];

        // sequence length
        const handlegraph::handle_t h = graph.get_handle(node_idx + 1, false);
        // NOTE: unable store orientation (reverse), since this information is path dependent
        n_tmp->seq_length = graph.get_length(h);

        // copy random coordinates
        n_tmp->coords[0].store(float(X[node_idx * 2].load()));
        n_tmp->coords[1].store(float(Y[node_idx * 2].load()));
        n_tmp->coords[2].store(float(X[node_idx * 2 + 1].load()));
        n_tmp->coords[3].store(float(Y[node_idx * 2 + 1].load()));
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
    path_data.element_array = (path_element_t*) malloc(path_data.total_path_steps * sizeof(path_element_t));

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

    // CPU Layout
    cpu_layout(config, etas, zetas, node_data, path_data);

    auto end_compute = std::chrono::high_resolution_clock::now();
    uint32_t duration_compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute - start_compute).count();
    std::cout << "CPU cache-optimized layout compute took " << duration_compute_ms << "ms" << std::endl;


    // copy coords back to X, Y vectors
    for (int node_idx = 0; node_idx < node_count; node_idx++) {
        cuda::node_t *n = &(node_data.nodes[node_idx]);
        // coords[0], coords[1], coords[2], coords[3] are stored consecutively. 
        std::atomic<float> *coords = n->coords;
        // check if coordinates valid (not NaN or infinite)
        for (int i = 0; i < 4; i++) {
            if (!isfinite(coords[i].load())) {
                std::cout << "WARNING: invalid coordiate" << std::endl;
            }
        }
        X[node_idx * 2].store(double(coords[0].load()));
        Y[node_idx * 2].store(double(coords[1].load()));
        X[node_idx * 2 + 1].store(double(coords[2].load()));
        Y[node_idx * 2 + 1].store(double(coords[3].load()));
        //std::cout << "coords of " << node_idx << ": [" << X[node_idx*2] << "; " << Y[node_idx*2] << "] ; [" << X[node_idx*2+1] << "; " << Y[node_idx*2+1] <<"]\n";
    }


    // get rid of CUDA data structures
    free(etas);
    free(node_data.nodes);
    free(path_data.paths);
    free(path_data.element_array);
    free(zetas);


#ifdef cuda_layout_profiling
    auto end = std::chrono::high_resolution_clock::now();
    uint32_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout << "CPU cache-optimized layout took " << duration_ms << "ms" << std::endl;
#endif

    return;
}

}
