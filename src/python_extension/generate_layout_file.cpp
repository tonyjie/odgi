#include "generate_layout_file.h"


namespace python_extension {

    void generate_layout_file(odgi::graph_t &graph, std::vector<double> x_final, std::vector<double> y_final, string layout_file_name) {
        // similar to layout_main.cpp
        // single numpy array already converted to x_final and y_final in pythonffi.cpp
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
