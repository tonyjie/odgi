#include <iostream>
#include <vector>

#include "odgi.hpp"
#include "algorithms/layout.hpp"
#include "algorithms/draw.hpp"
#include "weakly_connected_components.hpp"

namespace python_extension {

    void generate_layout_file(odgi::graph_t &graph, std::vector<double> x_final, std::vector<double> y_final, string layout_file_name);

}
