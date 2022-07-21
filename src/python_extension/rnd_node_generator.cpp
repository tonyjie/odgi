#include "rnd_node_generator.h"

namespace python_extension {

    RndNodeGenerator::RndNodeGenerator(odgi::graph_t &graph, double zipf_theta, uint64_t space_max, uint64_t space_quantization_step) : _graph{graph} {
        this->_path_index.from_handle_graph(this->_graph, 1);
        this->_nr_iv = &(this->_path_index.get_nr_iv());
        this->_npi_iv = &(this->_path_index.get_npi_iv());

        this->_rng_gen = XoshiroCpp::Xoshiro256Plus(42);
        this->_dis_step = std::uniform_int_distribution<uint64_t>(0, this->_path_index.get_np_bv().size() - 1);
        this->_flip = std::uniform_int_distribution<uint64_t>(0, 1);


        this->_theta = zipf_theta;
        uint64_t max_path_step_count = 0;
        this->_graph.for_each_path_handle(
                [&] (const handlegraph::path_handle_t &path) {
                    max_path_step_count = std::max(max_path_step_count, this->_path_index.get_path_step_count(path));
                });
        this->_space = max_path_step_count;
        this->_space_max = space_max;
        this->_space_quantization_step = space_quantization_step;

        // implemented as in path_sgd_layout.cpp
        this->_zetas = std::vector<double>((this->_space <= this->_space_max ? this->_space : this->_space_max + (this->_space - this->_space_max) / this->_space_quantization_step + 1)+1);
        uint64_t last_quantized_i = 0;
        for (uint64_t i = 1; i < this->_space+1; ++i) {
            uint64_t quantized_i = i;
            uint64_t compressed_space = i;
            if (i > this->_space_max){
                quantized_i = this->_space_max + (i - this->_space_max) / this->_space_quantization_step + 1;
                compressed_space = this->_space_max + ((i - this->_space_max) / this->_space_quantization_step) * this->_space_quantization_step;
            }

            if (quantized_i != last_quantized_i){
                dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, compressed_space, this->_theta);
                this->_zetas[quantized_i] = z_p.zeta();
                last_quantized_i = quantized_i;
            }
        }
    }


    RndNodeGenerator::~RndNodeGenerator() {
    }

    random_nodes_pack_t RndNodeGenerator::get_random_node_pack(bool cooling) {
        // generate random nodes similar to worker thread in path_sgd_layout.cpp
        uint64_t step_idx = this->_dis_step(this->_rng_gen);

        uint64_t path_idx = (*this->_npi_iv)[step_idx];
        handlegraph::path_handle_t path = handlegraph::as_path_handle(path_idx);
        uint64_t path_step_count = this->_path_index.get_path_step_count(path);

        uint64_t s_rank0 = (*this->_nr_iv)[step_idx] - 1;
        uint64_t s_rank1 = 0;

        if (cooling || this->_flip(this->_rng_gen)) {
            if (s_rank0 > 0 && this->_flip(this->_rng_gen) || s_rank0 == path_step_count-1) {
                // go backward
                uint64_t jump_space = std::min(this->_space, s_rank0);
                uint64_t space = jump_space;
                if (jump_space > this->_space_max){
                    space = this->_space_max + (jump_space - this->_space_max) / this->_space_quantization_step + 1;
                }
                dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, this->_theta, this->_zetas[space]);
                dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                uint64_t z_i = z(this->_rng_gen);
                s_rank1 = s_rank0 - z_i;
            } else {
                // go forward
                uint64_t jump_space = std::min(this->_space, path_step_count - s_rank0 - 1);
                uint64_t space = jump_space;
                if (jump_space > this->_space_max){
                    space = this->_space_max + (jump_space - this->_space_max) / this->_space_quantization_step + 1;
                }
                dirtyzipf::dirty_zipfian_int_distribution<uint64_t>::param_type z_p(1, jump_space, this->_theta, this->_zetas[space]);
                dirtyzipf::dirty_zipfian_int_distribution<uint64_t> z(z_p);
                uint64_t z_i = z(this->_rng_gen);
                s_rank1 = s_rank0 + z_i;
            }
        } else {
            std::uniform_int_distribution<uint64_t> rando(0, this->_graph.get_step_count(path)-1);
            // repeat until s_rank0 & s_rank1 different nodes (in same path)
            do {
                s_rank1 = rando(this->_rng_gen);
            } while (s_rank0 == s_rank1);
        }

        // sort: s_rank0 < s_rank1
        if (s_rank0 > s_rank1) {
            uint64_t tmp = s_rank0;
            s_rank0 = s_rank1;
            s_rank1 = tmp;
        }
        // assert(s_rank0 < s_rank1);

        handlegraph::step_handle_t step_a, step_b;
        as_integers(step_a)[0] = path_idx;
        as_integers(step_a)[1] = s_rank0;

        as_integers(step_b)[0] = path_idx;
        as_integers(step_b)[1] = s_rank1;


        handlegraph::handle_t term_i = this->_path_index.get_handle_of_step(step_a);
        uint64_t term_i_length = this->_graph.get_length(term_i);
        uint64_t pos_in_path_a = this->_path_index.get_position_of_step(step_a);

        bool term_i_is_rev = this->_graph.get_is_reverse(term_i);
        bool use_other_end_a = this->_flip(this->_rng_gen); // 1 == +; 0 == -
        if (use_other_end_a) {
            pos_in_path_a += term_i_length;
            // flip back if we were already reversed
            use_other_end_a = !term_i_is_rev;
        } else {
            use_other_end_a = term_i_is_rev;
        }
        uint64_t id_n0 = this->_graph.get_id(term_i);


        handlegraph::handle_t term_j = this->_path_index.get_handle_of_step(step_b);
        uint64_t term_j_length = this->_graph.get_length(term_j);
        uint64_t pos_in_path_b = this->_path_index.get_position_of_step(step_b);

        bool term_j_is_rev = this->_graph.get_is_reverse(term_j);
        bool use_other_end_b = this->_flip(this->_rng_gen); // 1 == +; 0 == -
        if (use_other_end_b) {
            pos_in_path_b += term_j_length;
            // flip back if we were already reversed
            use_other_end_b = !term_j_is_rev;
        } else {
            use_other_end_b = term_j_is_rev;
        }

        uint64_t id_n1 = this->_graph.get_id(term_j);


        double distance = std::abs(static_cast<double>(pos_in_path_b) - static_cast<double>(pos_in_path_a));
        if (distance == 0.0) {
            distance = 1e-9;
        }


        random_nodes_pack_t pack;
        pack.id_n0 = id_n0;
        pack.id_n1 = id_n1;
        pack.vis_p_n0 = use_other_end_a? 1 : 0;
        pack.vis_p_n1 = use_other_end_b? 1 : 0;
        pack.distance = distance;
        return pack;
    }

    uint64_t RndNodeGenerator::get_max_path_length(void) {
        return this->_space;
    }
}
