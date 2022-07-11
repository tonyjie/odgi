#include <iostream>
#include <vector>

#include "odgi.hpp"
#include <pybind11/numpy.h>
#include "algorithms/layout.hpp"
#include "algorithms/draw.hpp"
#include "weakly_connected_components.hpp"

namespace py = pybind11;

namespace python_extension {

    void generate_layout_file(py::array_t<double> coords_np, string layout_file_name, odgi::graph_t &graph) {
        // similar to layout_main.cpp
        int size = coords_np.shape(0) * coords_np.shape(1);
        std::vector<double> x_final(size);
        std::vector<double> y_final(size);

        // transfer coordinates from numpy to std::vector
        auto coords = coords_np.unchecked<3>();
        for (int e = 0; e < coords.shape(0); e++) {
            x_final[2*e] = coords(e, 0, 0);
            x_final[2*e+1] = coords(e, 1, 0);
            y_final[2*e] = coords(e, 0, 1);
            y_final[2*e+1] = coords(e, 1, 1);
        }

        // put border around it? same code as in layout_main.cpp
        std::vector<std::vector<handlegraph::handle_t>> weak_components = odgi::algorithms::weakly_connected_component_vectors(&graph);

        double border = 1000.0;
        double curr_y_offset = border;
        std::vector<odgi::algorithms::coord_range_2d_t> component_ranges;
        for (auto& component : weak_components) {
            component_ranges.emplace_back();
            auto& component_range = component_ranges.back();
            for (auto& handle : component) {
                uint64_t pos = 2 * handlegraph::number_bool_packing::unpack_number(handle);
                for (uint64_t j = pos; j <= pos+1; ++j) {
                    component_range.include(x_final[j], y_final[j]);
                }
            }
            component_range.x_offset = component_range.min_x - border;
            component_range.y_offset = curr_y_offset - component_range.min_y;
            curr_y_offset += component_range.height() + border;
        }

        for (uint64_t num_component = 0; num_component < weak_components.size(); ++num_component) {
            auto& component_range = component_ranges[num_component];

            for (auto& handle :  weak_components[num_component]) {
                uint64_t pos = 2 * handlegraph::number_bool_packing::unpack_number(handle);

                for (uint64_t j = pos; j <= pos+1; ++j) {
                    x_final[j] -= component_range.x_offset;
                    y_final[j] += component_range.y_offset;
                }
            }
        }


        // write layout file
        odgi::algorithms::layout::Layout lay(x_final, y_final);
        ofstream f(layout_file_name);
        lay.serialize(f);
        f.close();
    }

}
