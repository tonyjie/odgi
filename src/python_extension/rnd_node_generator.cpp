#include "rnd_node_generator.h"

namespace python_extension {

    RndNodeGenerator::RndNodeGenerator(odgi::graph_t &graph) : _graph{graph} {
        this->_path_index.from_handle_graph(this->_graph, 1);
        this->_nr_iv = &(this->_path_index.get_nr_iv());
        this->_npi_iv = &(this->_path_index.get_npi_iv());

        this->_rng_gen = XoshiroCpp::Xoshiro256Plus(42);
        this->_dis_step = std::uniform_int_distribution<uint64_t>(0, this->_path_index.get_np_bv().size() - 1);
    }

    RndNodeGenerator::~RndNodeGenerator() {
    }

    random_nodes_pack_t RndNodeGenerator::get_random_node_pack(void) {
        // generate random nodes similar to worker thread in path_sgd_layout.cpp
        uint64_t step_idx = this->_dis_step(this->_rng_gen);

        uint64_t path_idx = (*this->_npi_iv)[step_idx];
        handlegraph::path_handle_t path = handlegraph::as_path_handle(path_idx);

        uint64_t s_rank0 = (*this->_nr_iv)[step_idx] - 1;
        uint64_t s_rank1 = 0;

        std::uniform_int_distribution<uint64_t> rando(0, this->_graph.get_step_count(path)-1);
        // repeat until s_rank0 & s_rank1 different nodes (in same path)
        do {
            s_rank1 = rando(this->_rng_gen);
        } while (s_rank0 == s_rank1);

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
        uint64_t id_n0 = this->_graph.get_id(term_i);

        handlegraph::handle_t term_j = this->_path_index.get_handle_of_step(step_b);
        uint64_t id_n1 = this->_graph.get_id(term_j);

        uint64_t p_dis = s_rank1 - s_rank0;


        random_nodes_pack_t pack;
        pack.id_n0 = id_n0;
        pack.id_n1 = id_n1;
        pack.p_dis = p_dis;
        return pack;
    }

}
