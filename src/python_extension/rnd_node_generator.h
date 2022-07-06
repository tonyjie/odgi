#pragma once

#include "odgi.hpp"
#include "xp.hpp"
#include "XoshiroCpp.hpp"

namespace python_extension {

    struct random_nodes_pack_t {
        uint64_t id_n0;
        uint64_t id_n1;
        uint64_t p_dis;
    };

    class RndNodeGenerator {
        public:
        RndNodeGenerator(odgi::graph_t &graph);
        ~RndNodeGenerator();

        random_nodes_pack_t get_random_node_pack(void);

        private:
        odgi::graph_t &_graph;
        xp::XP _path_index;
        const sdsl::int_vector<> *_nr_iv;
        const sdsl::int_vector<> *_npi_iv;

        XoshiroCpp::Xoshiro256Plus _rng_gen;
        std::uniform_int_distribution<uint64_t> _dis_step;
    };


}
