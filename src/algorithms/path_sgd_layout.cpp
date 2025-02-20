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

    }
}