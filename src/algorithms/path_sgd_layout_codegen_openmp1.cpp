#include "path_sgd_layout.hpp"
#include "algorithms/layout.hpp"
#include <omp.h>

namespace odgi {
    namespace algorithms {

        void path_linear_sgd_layout(const PathHandleGraph &graph,
                                    const xp::XP &path_index,
                                    const std::vector<path_handle_t> &path_sgd_use_paths,
                                    const uint64_t &iter_max,
                                    const uint64_t &iter_with_max_learning_rate,
                                    const uint64_t &min_term_updates,
                                    const double &delta,
                                    const double &eps,
                                    const double &eta_max,
                                    const double &theta,
                                    const uint64_t &space,
                                    const uint64_t &space_max,
                                    const uint64_t &space_quantization_step,
                                    const double &cooling_start,
                                    const uint64_t &nthreads,
                                    const bool &progress,
                                    const bool &snapshot,
                                    const std::string &snapshot_prefix,
                                    std::vector<std::atomic<double>> &X,
                                    std::vector<std::atomic<double>> &Y) {

            uint64_t first_cooling_iteration = std::floor(cooling_start * (double)iter_max);

            bool at_least_one_path_with_more_than_one_step = false;
            for (auto &path : path_sgd_use_paths) {
                if (path_index.get_path_step_count(path) > 1) {
                    at_least_one_path_with_more_than_one_step = true;
                    break;
                }
            }

            if (at_least_one_path_with_more_than_one_step) {
                double w_min = 1.0 / eta_max;
                double w_max = 1.0;
                std::vector<double> etas = path_linear_sgd_layout_schedule(w_min, w_max, iter_max,
                                                                           iter_with_max_learning_rate, eps);

                std::vector<double> zetas((space <= space_max ? space : space_max + (space - space_max) / space_quantization_step + 1)+1);
                double zeta_tmp = 0.0;
                for (uint64_t i = 1; i < space + 1; i++) {
                    zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / i, theta);
                    if (i <= space_max) zetas[i] = zeta_tmp;
                    if (i >= space_max && (i - space_max) % space_quantization_step == 0) {
                        zetas[space_max + 1 + (i - space_max)/space_quantization_step] = zeta_tmp;
                    }
                }

                std::atomic<uint64_t> term_updates(0);
                std::atomic<uint64_t> iteration(0);
                std::atomic<bool> work_todo(true);
                std::atomic<double> eta(etas.front());
                std::atomic<double> adj_theta(theta);
                std::atomic<bool> cooling(false);
                std::atomic<double> Delta_max(0);

                const sdsl::bit_vector &np_bv = path_index.get_np_bv();
                const sdsl::int_vector<> &nr_iv = path_index.get_nr_iv();
                const sdsl::int_vector<> &npi_iv = path_index.get_npi_iv();

                omp_set_num_threads(nthreads);

                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    XoshiroCpp::Xoshiro256Plus gen(9399220 + tid);
                    std::uniform_int_distribution<uint64_t> dis_step(0, np_bv.size()-1);
                    std::uniform_int_distribution<uint64_t> flip(0, 1);
                    uint64_t term_updates_local = 0;

                    while (work_todo.load(std::memory_order_relaxed)) {
                        // Worker logic
                        uint64_t step_index = dis_step(gen);
                        uint64_t path_i = npi_iv[step_index];
                        path_handle_t path = as_path_handle(path_i);

                        size_t path_step_count = path_index.get_path_step_count(path);
                        if (path_step_count == 1) continue;

                        step_handle_t step_a, step_b;
                        as_integers(step_a)[0] = path_i;
                        size_t s_rank = nr_iv[step_index] - 1;
                        as_integers(step_a)[1] = s_rank;

                        if (cooling.load() || flip(gen)) {
                            if (s_rank > 0 && flip(gen) || s_rank == path_step_count-1) {
                                // go backward
                                uint64_t jump_space = std::min(space, (uint64_t) s_rank);
                                uint64_t space = jump_space;
                                if (jump_space > space_max){
                                    space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                                }
                                dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, zetas[space]);
                                dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                                uint64_t z_i = z(gen);
                                //assert(z_i <= path_space);
                                as_integers(step_b)[0] = as_integer(path);
                                as_integers(step_b)[1] = s_rank - z_i;
                            } else {
                                // go forward
                                uint64_t jump_space = std::min(space, (uint64_t) (path_step_count - s_rank - 1));
                                uint64_t space = jump_space;
                                if (jump_space > space_max){
                                    space = space_max + (jump_space - space_max) / space_quantization_step + 1;
                                }
                                dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, theta, zetas[space]);
                                dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                                uint64_t z_i = z(gen);
                                //assert(z_i <= path_space);
                                as_integers(step_b)[0] = as_integer(path);
                                as_integers(step_b)[1] = s_rank + z_i;
                            }
                        } else {
                            // sample randomly across the path
                            std::uniform_int_distribution<uint64_t> rando(0, graph.get_step_count(path)-1);
                            as_integers(step_b)[0] = as_integer(path);
                            as_integers(step_b)[1] = rando(gen);
                        }                        

                        handle_t term_i = path_index.get_handle_of_step(step_a);
                        handle_t term_j = path_index.get_handle_of_step(step_b);
                        uint64_t term_i_length = graph.get_length(term_i);
                        uint64_t term_j_length = graph.get_length(term_j);

                        // adjust the positions to the node starts
                        size_t pos_in_path_a = path_index.get_position_of_step(step_a);
                        size_t pos_in_path_b = path_index.get_position_of_step(step_b);

                        // determine which end we're working with for each node
                        bool term_i_is_rev = graph.get_is_reverse(term_i);
                        bool use_other_end_a = flip(gen); // 1 == +; 0 == -
                        if (use_other_end_a) {
                            pos_in_path_a += term_i_length;
                            // flip back if we were already reversed
                            use_other_end_a = !term_i_is_rev;
                        } else {
                            use_other_end_a = term_i_is_rev;
                        }
                        bool term_j_is_rev = graph.get_is_reverse(term_j);
                        bool use_other_end_b = flip(gen); // 1 == +; 0 == -
                        if (use_other_end_b) {
                            pos_in_path_b += term_j_length;
                            // flip back if we were already reversed
                            use_other_end_b = !term_j_is_rev;
                        } else {
                            use_other_end_b = term_j_is_rev;
                        }

                        // establish the term distance
                        double term_dist = std::abs(
                                static_cast<double>(pos_in_path_a) - static_cast<double>(pos_in_path_b));

                        if (term_dist == 0) {
                            term_dist = 1e-9;
                        }
                        double term_weight = 1.0 / (double) term_dist;
                        double w_ij = term_weight;
                        double mu = eta.load() * w_ij;
                        if (mu > 1) {
                            mu = 1;
                        }
                        // actual distance in graph
                        double d_ij = term_dist;
                        // identities
                        uint64_t i = number_bool_packing::unpack_number(term_i);
                        uint64_t j = number_bool_packing::unpack_number(term_j);
                        // distance == magnitude in our 2D situation
                        uint64_t offset_i = 0;
                        uint64_t offset_j = 0;
                        if (use_other_end_a) {
                            offset_i += 1;
                        }
                        if (use_other_end_b) {
                            offset_j += 1;
                        }

                        // Atomic updates
                        double dx = X[2*i + offset_i].load(std::memory_order_relaxed) - 
                                    X[2*j + offset_j].load(std::memory_order_relaxed);
                        double dy = Y[2*i + offset_i].load(std::memory_order_relaxed) - 
                                    Y[2*j + offset_j].load(std::memory_order_relaxed);
                        if (dx == 0) {
                            dx = 1e-9; // avoid nan
                        }
                        double mag = sqrt(dx*dx + dy*dy);
                        double Delta = mu * (mag - d_ij) / 2;
                        double Delta_abs = std::abs(Delta);
                        while (Delta_abs > Delta_max.load()) {
                            Delta_max.store(Delta_abs);
                        }
                        // calculate update
                        double r = Delta / mag;
                        double r_x = r * dx;
                        double r_y = r * dy;

                        X[2*i + offset_i].store(X[2*i + offset_i].load(std::memory_order_relaxed) - r_x, 
                                                std::memory_order_relaxed);
                        Y[2*i + offset_i].store(Y[2*i + offset_i].load(std::memory_order_relaxed) - r_y, 
                                                std::memory_order_relaxed);
                        X[2*j + offset_j].store(X[2*j + offset_j].load(std::memory_order_relaxed) + r_x, 
                                                std::memory_order_relaxed);
                        Y[2*j + offset_j].store(Y[2*j + offset_j].load(std::memory_order_relaxed) + r_y, 
                                                std::memory_order_relaxed);

                        // Batch progress updates
                        if (++term_updates_local >= 1000) {
                            term_updates.fetch_add(term_updates_local, std::memory_order_relaxed);
                            term_updates_local = 0;
                        }

                        // Check termination conditions
                        #pragma omp master
                        {
                            if (term_updates.load(std::memory_order_relaxed) >= min_term_updates) {
                                uint64_t current_iter = iteration.load(std::memory_order_relaxed);
                                if (current_iter >= iter_max) {
                                    work_todo.store(false, std::memory_order_relaxed);
                                }
                                else {
                                    // Update parameters
                                    eta.store(etas[current_iter+1], std::memory_order_relaxed);
                                    if (current_iter >= first_cooling_iteration) {
                                        adj_theta.store(0.001, std::memory_order_relaxed);
                                        cooling.store(true, std::memory_order_relaxed);
                                    }
                                    iteration.fetch_add(1, std::memory_order_relaxed);
                                    term_updates.store(0, std::memory_order_relaxed);
                                    
                                    // Early stopping check
                                    if (Delta_max.load(std::memory_order_relaxed) <= delta) {
                                        work_todo.store(false, std::memory_order_relaxed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ... (keep path_linear_sgd_layout_schedule implementation identical)
        std::vector<double> path_linear_sgd_layout_schedule(const double &w_min,
                                                            const double &w_max,
                                                            const uint64_t &iter_max,
                                                            const uint64_t &iter_with_max_learning_rate,
                                                            const double &eps) {

            double eta_max = 1.0 / w_min;
            double eta_min = eps / w_max;
            double lambda = log(eta_max / eta_min) / ((double) iter_max - 1);

            // initialize step sizes
            std::vector<double> etas;
            etas.reserve(iter_max+1);

            for (int64_t t = 0; t <= iter_max; t++) {
                etas.push_back(eta_max * exp(-lambda * (abs(t - (int64_t) iter_with_max_learning_rate))));
            }
            return etas;
        }

#ifdef USE_GPU
        void path_linear_sgd_layout_gpu(const PathHandleGraph &graph,
                                    const xp::XP &path_index,
                                    const std::vector<path_handle_t> &path_sgd_use_paths,
                                    const uint64_t &iter_max,
                                    const uint64_t &iter_with_max_learning_rate,
                                    const uint64_t &min_term_updates,
                                    const double &delta,
                                    const double &eps,
                                    const double &eta_max,
                                    const double &theta,
                                    const uint64_t &space,
                                    const uint64_t &space_max,
                                    const uint64_t &space_quantization_step,
                                    const double &cooling_start,
                                    const uint64_t &nthreads,
                                    const bool &progress,
                                    const bool &snapshot,
                                    const std::string &snapshot_prefix,
                                    std::vector<std::atomic<double>> &X,
                                    std::vector<std::atomic<double>> &Y) {
            cuda::layout_config_t config;
            config.iter_max = iter_max;
            config.min_term_updates = min_term_updates;
            config.eta_max = eta_max;
            config.eps = eps;
            config.iter_with_max_learning_rate = (int32_t)  iter_with_max_learning_rate;
            config.first_cooling_iteration = std::floor(cooling_start * (double)iter_max);
            config.theta = theta;
            config.space = uint32_t(space);
            config.space_max = uint32_t(space_max);
            config.space_quantization_step = uint32_t(space_quantization_step);
            config.nthreads = nthreads;
            cuda::gpu_layout(config, dynamic_cast<const odgi::graph_t&>(graph), X, Y);
            return;
        }
#endif

    }
}