// odgi ffi calls C-API functions
//
// Copyright © 2022 Pjotr Prins

#include "odgi.hpp"
#include "odgi-api.h"

// Pybind11
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <vector>

PYBIND11_MAKE_OPAQUE(ograph_t);
PYBIND11_MAKE_OPAQUE(oRndNodeGenerator);
PYBIND11_MAKE_OPAQUE(python_extension::random_nodes_pack_t);

namespace py = pybind11;

PYBIND11_MODULE(odgi_ffi, m)
{
    py::class_<ograph_t>(m, "opaque_graph pointer for FFI");
    // py::class_<path_handle_t>(m, "opaque path_handle_t for FFI");
    // py::class_<handle_t>(m, "opaque handle_t for FFI");
    py::class_<step_handle_t>(m, "step_handle_t for FFI");
    py::class_<oRndNodeGenerator>(m, "rnd node generator");
    py::class_<python_extension::random_nodes_pack_t>(m, "package of two random nodes in graph") ;
    m.def("odgi_version", &odgi_version, "Get the odgi library build version");
    m.def("odgi_long_long_size", &odgi_long_long_size);
    m.def("odgi_handle_i_size", &odgi_handle_i_size);
    m.def("odgi_step_handle_i_size", &odgi_step_handle_i_size);
    m.def("odgi_test_uint128", &odgi_test_uint128);
    m.def("odgi_load_graph", &odgi_load_graph);
    m.def("odgi_free_graph", &odgi_free_graph);
    m.def("odgi_get_node_count", &odgi_get_node_count);
    m.def("odgi_max_node_id", &odgi_max_node_id);
    m.def("odgi_min_node_id", &odgi_min_node_id);
    m.def("odgi_get_path_count", &odgi_get_path_count);
    m.def("odgi_for_each_path_handle", &odgi_for_each_path_handle);
    m.def("odgi_for_each_handle", &odgi_for_each_handle);
    m.def("odgi_follow_edges", &odgi_follow_edges);
    m.def("odgi_edge_first_handle", &odgi_edge_first_handle);
    m.def("odgi_edge_second_handle", &odgi_edge_second_handle);
    m.def("odgi_has_node", &odgi_has_node);
    m.def("odgi_get_sequence", &odgi_get_sequence);
    m.def("odgi_get_id", &odgi_get_id);
    m.def("odgi_get_is_reverse", &odgi_get_is_reverse);
    m.def("odgi_get_length", &odgi_get_length);
    m.def("odgi_get_path_name", &odgi_get_path_name);
    m.def("odgi_has_path", &odgi_has_path);
    m.def("odgi_path_is_empty", &odgi_path_is_empty);
    m.def("odgi_get_path_handle", &odgi_get_path_handle);
    m.def("odgi_get_step_count", &odgi_get_step_count);
    m.def("odgi_get_step_in_path_count", &odgi_get_step_in_path_count);
    m.def("odgi_get_handle_of_step", &odgi_get_handle_of_step);
    m.def("odgi_get_path", &odgi_get_path);
    m.def("odgi_path_begin", &odgi_path_begin);
    m.def("odgi_path_end", &odgi_path_end);
    m.def("odgi_path_back", &odgi_path_back);
    m.def("odgi_step_path_id", &odgi_step_path_id);
    m.def("odgi_step_is_reverse", &odgi_step_is_reverse);
    m.def("odgi_step_prev_id", &odgi_step_prev_id);
    m.def("odgi_step_prev_rank", &odgi_step_prev_rank);
    m.def("odgi_step_next_id", &odgi_step_next_id);
    m.def("odgi_step_next_rank", &odgi_step_next_rank);
    m.def("odgi_step_eq", &odgi_step_eq);
    m.def("odgi_path_front_end", &odgi_path_front_end);
    m.def("odgi_get_next_step", &odgi_get_next_step);
    m.def("odgi_get_previous_step", &odgi_get_previous_step);
    m.def("odgi_has_edge", &odgi_has_edge);
    m.def("odgi_is_path_front_end", &odgi_is_path_front_end);
    m.def("odgi_is_path_end", &odgi_is_path_end);
    m.def("odgi_has_next_step", &odgi_has_next_step);
    m.def("odgi_has_previous_step", &odgi_has_previous_step);
    m.def("odgi_get_path_handle_of_step", &odgi_get_path_handle_of_step);
    m.def("odgi_for_each_step_in_path", &odgi_for_each_step_in_path);
    m.def("odgi_for_each_step_on_handle", &odgi_for_each_step_on_handle);
    m.def("odgi_create_rnd_node_generator", &odgi_create_rnd_node_generator);
    m.def("odgi_RNG_get_max_path_length", &odgi_RNG_get_max_path_length);
    m.def("odgi_get_random_node_pack", &odgi_get_random_node_pack);
    m.def("odgi_RNP_get_id_n0", &odgi_RNP_get_id_n0, "Get id of node0 from random_nodes_pack.");
    m.def("odgi_RNP_get_id_n1", &odgi_RNP_get_id_n1, "Get id of node1 from random_nodes_pack.");
    m.def("odgi_RNP_get_vis_p_n0", &odgi_RNP_get_vis_p_n0, "Get chosen visualization point for node n0 (0 or 1).");
    m.def("odgi_RNP_get_vis_p_n1", &odgi_RNP_get_vis_p_n1, "Get chosen visualization point for node n1 (0 or 1).");
    m.def("odgi_RNP_get_distance", &odgi_RNP_get_distance, "Get distance between random nodes in random_nodes_pack.");
    m.def("odgi_generate_layout_file",
          [](const ograph_t graph, py::array_t<double> coords_np, string layout_file_name) {
              // similar to layout_main.cpp
              // numpy array converted here to keep pybind11 dependencies to this file
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
              odgi_generate_layout_file(graph, x_final, y_final, layout_file_name);
          });
    m.def("odgi_get_random_node_numpy_batch",
          [](oRndNodeGenerator RNoG, int batch_size, bool cooling) {
              int64_t *i = new int64_t[batch_size];
              int64_t *j = new int64_t[batch_size];
              int64_t *vis_i = new int64_t[batch_size];
              int64_t *vis_j = new int64_t[batch_size];
              double *d = new double[batch_size];

              for (int idx = 0; idx < batch_size; idx++) {
                  python_extension::random_nodes_pack_t p = RNoG->get_random_node_pack(cooling);
                  i[idx] = p.id_n0;
                  j[idx] = p.id_n1;
                  vis_i[idx] = p.vis_p_n0;
                  vis_j[idx] = p.vis_p_n1;
                  d[idx] = p.distance;
              }

              py::array_t<int64_t> i_np = py::array_t<int64_t>(batch_size, i);
              py::array_t<int64_t> j_np = py::array_t<int64_t>(batch_size, j);
              py::array_t<int64_t> vis_i_np = py::array_t<int64_t>(batch_size, vis_i);
              py::array_t<int64_t> vis_j_np = py::array_t<int64_t>(batch_size, vis_j);
              py::array_t<double> d_np = py::array_t<double>(batch_size, d);

              py::tuple ret_tuple = py::make_tuple(i_np, j_np, vis_i_np, vis_j_np, d_np);

              delete i;
              delete j;
              delete vis_i;
              delete vis_j;
              delete d;

              return ret_tuple;
          },
          py::return_value_policy::move);
}
