#include "subcommand.hpp"
#include <iostream>
#include "odgi.hpp"
#include "args.hxx"
#include <omp.h>
#include "algorithms/layout.hpp"
#include "algorithms/tension/tension_bed_records_queued_writer.hpp"
#include <numeric>
#include "progress.hpp"

namespace odgi {

using namespace odgi::subcommand;

int main_tension(int argc, char **argv) {

    // trick argument parser to do the right thing with the subcommand
    for (uint64_t i = 1; i < argc - 1; ++i) {
        argv[i] = argv[i + 1];
    }
    std::string prog_name = "odgi tension";
    argv[0] = (char *) prog_name.c_str();
    --argc;

    args::ArgumentParser parser(
        "evaluate the tension of a graph helping to locate structural variants and abnormalities");
	args::Group mandatory_opts(parser, "[ MANDATORY ARGUMENTS ]");
    args::ValueFlag<std::string> dg_in_file(mandatory_opts, "FILE", "load the graph from this file", {'i', "idx"});
	args::ValueFlag<std::string> layout_in_file(mandatory_opts, "FILE", "read the layout coordinates from this .lay format file produced by odgi sort or odgi layout", {'c', "coords-in"});
	args::Group tension_opts(parser, "[ Tension Options ]");
	args::ValueFlag<double> window_size(tension_opts, "N", "window size in bases in which each tension is calculated, DEFAULT: 1kb", {'w', "window-size"});
	// args::ValueFlag<std::string> tsv_out_file(tension_opts, "FILE", "write the tension intervals to this TSV file", {'t', "tsv"});
	args::Flag node_sized_windows(tension_opts, "node-sized-windows", "instead of manual window sizes, each window has the size of the node of the step we are currently iterating", {'n', "node-sized-windows"});
	args::Flag pangenome_mode(tension_opts, "run tension in pangenome mode", "calculate the tension for each node of the pangenome: node tension is the sum of the tension of all steps visiting that node. Results are written in TSV format to stdout. 1st col: node identifier. 2nd col: tension=(path_layout_dist/path_nuc_dist). 3rd col: 2nd_col/#steps_on_node. (DEFAULT: ENABLED)", {'p', "pangenome-mode"});
	args::Group threading_opts(parser, "[ Threading ]");
	args::ValueFlag<uint64_t> nthreads(parser, "N", "number of threads to use for parallel phases", {'t', "threads"});
	args::Group processing_info_opts(parser, "[ Processing Information ]");
	args::Flag progress(processing_info_opts, "progress", "display progress", {'P', "progress"});
	args::Group program_info_opts(parser, "[ Program Information ]");
	args::HelpFlag help(program_info_opts, "help", "display this help summary", {'h', "help"});

    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    if (argc == 1) {
        std::cout << parser;
        return 1;
    }

    uint64_t thread_count = 1;
    if (args::get(nthreads)) {
        omp_set_num_threads(args::get(nthreads));
        thread_count = args::get(nthreads);
    }

    if (!dg_in_file) {
        std::cerr
                << "[odgi tension] error: Please specify an input file from where to load the graph via -i=[FILE], --idx=[FILE]."
                << std::endl;
        return 1;
    }

    graph_t graph;
    assert(argc > 0);
    std::string infile = args::get(dg_in_file);
    if (infile.size()) {
        if (infile == "-") {
            graph.deserialize(std::cin);
        } else {
            ifstream f(infile.c_str());
            graph.deserialize(f);
            f.close();
        }
    }

    algorithms::layout::Layout layout;
    if (layout_in_file) {
        auto& infile = args::get(layout_in_file);
        if (infile.size()) {
            if (infile == "-") {
                layout.load(std::cin);
            } else {
                ifstream f(infile.c_str());
                layout.load(f);
                f.close();
            }
        }
    }


	// ======= check the length of each path in the layout ==========
	std::vector<odgi::path_handle_t> paths;
	graph.for_each_path_handle([&] (const odgi::path_handle_t &p) {
		paths.push_back(p);
	});

	struct PathInfo {
        path_handle_t path_handle;
        uint64_t path_nuc_dist;
        double path_layout_len;
        double path_len_error;
        // Constructor
        PathInfo(path_handle_t path_handle, uint64_t path_nuc_dist, double path_layout_len, double path_len_error)
            : path_handle(path_handle), path_nuc_dist(path_nuc_dist), path_layout_len(path_layout_len), path_len_error(path_len_error) {};
    };
    // vector to save the "tolopogy error" for each path. Each vector is a (<string> path_name, <uint64_t> path_nuc_dist, <double> path_layout_len, <double> path_len_error)
    std::vector<PathInfo> path_topology;


    #pragma omp parallel for schedule(static, 1) num_threads(thread_count)
	for (auto p : paths) {
        uint64_t path_nuc_dist = 0;
        odgi::algorithms::xy_d_t start_p, end_p; // start point of the first step in the path; end point of the last step in the path
        double path_layout_len = 0;

		graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
            // count the length of each path
			odgi::handle_t h = graph.get_handle_of_step(s);
			uint64_t step_nuc_dist = graph.get_length(h);
            path_nuc_dist += step_nuc_dist;

            // if this is the first step
            if (!graph.has_previous_step(s)) {
                // std::cout << "first step: " << endl;
                // start point of the first step (ignore reverse right now)
                start_p = layout.coords(h);
            }

            // if this is the last step
            if (!graph.has_next_step(s)) {
                // std::cout << "last step: " << endl;
                end_p = layout.coords(graph.flip(h));
            }
            

		});

        path_layout_len = odgi::algorithms::layout::coord_dist(start_p, end_p);
        // normalized squared error
        double path_len_error = pow((((double)path_layout_len - (double)path_nuc_dist) / (double)path_nuc_dist), 2);
        // std::cout << "path: " << graph.get_path_name(p) << " nuc_dist: " << path_nuc_dist << "; layout_len: " << path_layout_len << "; error: " << path_len_error << std::endl;

        path_topology.push_back(PathInfo(p, path_nuc_dist, path_layout_len, path_len_error));
    }

    // compute the average path topology error
    double sum_path_topology_error = 0;
    for (auto& p : path_topology) {
        sum_path_topology_error += p.path_len_error;
    }
    double avg_path_topology_error = sum_path_topology_error / (double)path_topology.size();

	std::cout << "average path topology error: " << avg_path_topology_error << std::endl;

    // show top-10 path topology error

    std::sort(path_topology.begin(), path_topology.end(), [](const PathInfo& a, const PathInfo& b) {
        return a.path_len_error > b.path_len_error;
    });



    // investigate into the first five path (weird)
    // std::cout << "==== Check the path with largest error =====" << std::endl;
    // for (uint64_t i = 0; i < 1; i++) {
    //     std::cout << "path: " << graph.get_path_name(path_topology[i].path_handle) << " nuc_dist: " << path_topology[i].path_nuc_dist << "; layout_len: " << path_topology[i].path_layout_len << "; error: " << path_topology[i].path_len_error << std::endl;
    //     graph.for_each_step_in_path(path_topology[i].path_handle, [&](const odgi::step_handle_t &s) {
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         // the layout coordinates of the node -- layout distance
    //         odgi::algorithms::xy_d_t h_coords_start;
    //         odgi::algorithms::xy_d_t h_coords_end;
    //         h_coords_start = layout.coords(h);
    //         h_coords_end = layout.coords(graph.flip(h));
    //         double within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
    //         double node_to_node_dist = 0; 
    //         if (graph.has_previous_step(s)) {
    //             odgi::step_handle_t prev_s = graph.get_previous_step(s);
    //             odgi::handle_t prev_h = graph.get_handle_of_step(prev_s);
    //             odgi::algorithms::xy_d_t prev_h_coords_start;
    //             odgi::algorithms::xy_d_t prev_h_coords_end;
    //             prev_h_coords_start = layout.coords(prev_h);
    //             prev_h_coords_end = layout.coords(graph.flip(prev_h));
    //             node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_end, h_coords_start);
    //         }


    //         std::cout << "node: " << graph.get_id(h) << " length: " << graph.get_length(h) << " within_node_dist: " << within_node_dist << " node_to_node_dist: " << node_to_node_dist << std::endl;
    //     }
    //     );
    //     std::cout << std::endl;
    // }


    // path_handle_t error_path = path_topology[0].path_handle;
    // // how many steps in this path
    // graph.for_each_step_in_path(error_path, [&](const odgi::step_handle_t &s) {
    //     odgi::handle_t h = graph.get_handle_of_step(s);
    //     std::cout << "node: " << graph.get_id(h) << " length: " << graph.get_length(h) << std::endl;
    // }
    // );


    std::cout << "top-10 path topology error: " << std::endl;
    // the largest top-10 path topology error
    // for (uint64_t i = 0; i < path_topology.size() - 1; i++) {
	for (uint64_t i = 0; i < 10; i++) {
        // how many steps in this path
        uint64_t num_steps = 0;
        graph.for_each_step_in_path(path_topology[i].path_handle, [&](const odgi::step_handle_t &s) {
            num_steps += 1;
        }
        );
        std::cout << "path: " << graph.get_path_name(path_topology[i].path_handle) << " nuc_dist: " << path_topology[i].path_nuc_dist << "; layout_len: " << path_topology[i].path_layout_len << "; error: " << path_topology[i].path_len_error << 
        " num_steps: " << num_steps << std::endl;
    }


    // show the least top-10 path topology error
    // std::cout << "least top-10 path topology error: " << std::endl;
    // for (uint64_t i = path_topology.size() - 1; i > path_topology.size() - 11; i--) {
    //     // how many steps in this path
    //     uint64_t num_steps = 0;
    //     graph.for_each_step_in_path(path_topology[i].path_handle, [&](const odgi::step_handle_t &s) {
    //         num_steps += 1;
    //     }
    //     );
    //     std::cout << "path: " << graph.get_path_name(path_topology[i].path_handle) << " nuc_dist: " << path_topology[i].path_nuc_dist << "; layout_len: " << path_topology[i].path_layout_len << "; error: " \\
    //     << path_topology[i].path_len_error << "num_steps: " << num_steps << std::endl;
    // }



    // std::vector<odgi::path_handle_t> paths;
    // graph.for_each_path_handle([&] (const odgi::path_handle_t &p) {
    //     paths.push_back(p);
    // });


    // double sum_stress_squared_dist_weight = 0;
    // uint32_t num_steps_iterated = 0;

    // #pragma omp parallel for schedule(static, 1) num_threads(thread_count) reduction(+:sum_stress_squared_dist_weight, num_steps_iterated)
    // for (auto p: paths) {
    //     double path_layout_dist;
    //     uint64_t path_nuc_dist;
    //     graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
    //         path_layout_dist = 0;
    //         path_nuc_dist = 0;
    //         odgi::handle_t h = graph.get_handle_of_step(s);
    //         odgi::algorithms::xy_d_t h_coords_start;
    //         odgi::algorithms::xy_d_t h_coords_end;
    //         if (graph.get_is_reverse(h)) {
    //             h_coords_start = layout.coords(graph.flip(h));
    //             h_coords_end = layout.coords(h);
    //         } else {
    //             h_coords_start = layout.coords(h);
    //             h_coords_end = layout.coords(graph.flip(h));
    //         }
    //         // TODO refactor into function start
    //         // did we hit the first step?
    //         if (graph.has_previous_step(s)) {
    //             odgi::step_handle_t prev_s = graph.get_previous_step(s);
    //             odgi::handle_t prev_h = graph.get_handle_of_step(prev_s);
    //             odgi::algorithms::xy_d_t prev_h_coords_start;
    //             odgi::algorithms::xy_d_t prev_h_coords_end;
    //             if (graph.get_is_reverse(prev_h)) {
    //                 prev_h_coords_start = layout.coords(graph.flip(prev_h));
    //                 prev_h_coords_end = layout.coords(prev_h);
    //             } else {
    //                 prev_h_coords_start = layout.coords(prev_h);
    //                 prev_h_coords_end = layout.coords(graph.flip(prev_h));
    //             }
    //             double within_node_dist = 0;
    //             double from_node_to_node_dist = 0;
    //             if (!graph.get_is_reverse(prev_h)) {
    //                 /// f + f
    //                 if (!graph.get_is_reverse(h)) {
    //                     within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
    //                     from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_end, h_coords_start);
    //                 } else {
    //                     /// f + r
    //                     within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
    //                     from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_end, h_coords_end);
    //                 }
    //             } else {
    //                 /// r + r
    //                 if (graph.get_is_reverse(h)) {
    //                     within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
    //                     from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_start, h_coords_end);
    //                 } else {
    //                     /// r + f
    //                     within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
    //                     from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_start,
    //                                                                             h_coords_start);
    //                 }
    //             }
    //             path_layout_dist += within_node_dist;
    //             path_layout_dist += from_node_to_node_dist;
    //             uint64_t nuc_dist = graph.get_length(h);
    //             path_nuc_dist += nuc_dist;
    //             // cur_window_end += nuc_dist;
    //         } else {
    //             // we only take a look at the current node
    //             /// f
    //             if (!graph.get_is_reverse(h)) {
    //                 path_layout_dist += odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
    //             } else {
    //                 /// r
    //                 path_layout_dist += odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
    //             }
    //             uint64_t nuc_dist = graph.get_length(h);
    //             path_nuc_dist += nuc_dist;
    //             // cur_window_end += nuc_dist;
    //         } // TODO refactor into function end

    //         sum_stress_squared_dist_weight += pow((((double)path_layout_dist - (double)path_nuc_dist) / (double)path_nuc_dist), 2); // weight = 1 / (d*d)

    //         num_steps_iterated += 1;
    //     });
    // }

    // double stress_result = sum_stress_squared_dist_weight / (double)num_steps_iterated;

	// std::cout << "stress: " << stress_result << std::endl;
// */

// }	




    return 0;
}

static Subcommand odgi_tension("tension", "evaluate the tension of a graph helping to locate structural variants and abnormalities",
                            PIPELINE, 3, main_tension);


}
