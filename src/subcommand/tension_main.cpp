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
	// option to control if compute the new or old node-stress
	args::Flag new_node_stress(tension_opts, "new-node-stress", "compute the new node-stress", {'n', "new-node-stress"});
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

	if (new_node_stress) {
		cout << "compute the new node-stress" << endl;


		// save a vector of all the node handle_t
		std::vector<handle_t> nodes;
		graph.for_each_handle([&](const handle_t& h) {
			nodes.push_back(h);
		});

		double sum_stress = 0;
		// iterate through all the nodes
		#pragma omp parallel for schedule(static, 1) num_threads(thread_count) reduction(+:sum_stress)
		for (auto& h : nodes) {
			odgi::algorithms::xy_d_t h_coords_start;
			odgi::algorithms::xy_d_t h_coords_end;
			// get the node nucleotide length
			uint64_t dist_nuc = graph.get_length(h);
			// get the node layout distance
			h_coords_start = layout.coords(h);
			h_coords_end = layout.coords(graph.flip(h));
			double dist_layout = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
			// compute the node-stress
			double node_stress = pow((((double)dist_layout - (double)dist_nuc) / (double)dist_nuc), 2);
			// add the node-stress to the sum
			sum_stress += node_stress;
		}
		
		// normalized by the number of nodes
		sum_stress = sum_stress / (double)graph.get_node_count();
		cout << "stress: " << sum_stress << endl;

	} 



	else {

	// The original node-stress. There are two differences compared to the above new node-stress. 
	// 1. First iterate through paths, then iterate through each step (node) in the path. 
	// 2. Not only consider the start-point and end-point of one node, but also consider the relationship to the previous node. (And if they are reversed or not. )
		cout << "compute the old node-stress" << endl;

		std::vector<odgi::path_handle_t> paths;
		graph.for_each_path_handle([&] (const odgi::path_handle_t &p) {
			paths.push_back(p);
		});


		double sum_stress_squared_dist_weight = 0;
		uint32_t num_steps_iterated = 0;

		#pragma omp parallel for schedule(static, 1) num_threads(thread_count) reduction(+:sum_stress_squared_dist_weight, num_steps_iterated)
		for (auto p: paths) {
			double path_layout_dist;
			uint64_t path_nuc_dist;
			graph.for_each_step_in_path(p, [&](const odgi::step_handle_t &s) {
				path_layout_dist = 0;
				path_nuc_dist = 0;
				odgi::handle_t h = graph.get_handle_of_step(s);
				odgi::algorithms::xy_d_t h_coords_start;
				odgi::algorithms::xy_d_t h_coords_end;
				if (graph.get_is_reverse(h)) {
					h_coords_start = layout.coords(graph.flip(h));
					h_coords_end = layout.coords(h);
				} else {
					h_coords_start = layout.coords(h);
					h_coords_end = layout.coords(graph.flip(h));
				}
				// TODO refactor into function start
				// did we hit the first step?
				if (graph.has_previous_step(s)) {
					odgi::step_handle_t prev_s = graph.get_previous_step(s);
					odgi::handle_t prev_h = graph.get_handle_of_step(prev_s);
					odgi::algorithms::xy_d_t prev_h_coords_start;
					odgi::algorithms::xy_d_t prev_h_coords_end;
					if (graph.get_is_reverse(prev_h)) {
						prev_h_coords_start = layout.coords(graph.flip(prev_h));
						prev_h_coords_end = layout.coords(prev_h);
					} else {
						prev_h_coords_start = layout.coords(prev_h);
						prev_h_coords_end = layout.coords(graph.flip(prev_h));
					}
					double within_node_dist = 0;
					double from_node_to_node_dist = 0;
					if (!graph.get_is_reverse(prev_h)) {
						/// f + f
						if (!graph.get_is_reverse(h)) {
							within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
							from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_end, h_coords_start);
						} else {
							/// f + r
							within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
							from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_end, h_coords_end);
						}
					} else {
						/// r + r
						if (graph.get_is_reverse(h)) {
							within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
							from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_start, h_coords_end);
						} else {
							/// r + f
							within_node_dist = odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
							from_node_to_node_dist = odgi::algorithms::layout::coord_dist(prev_h_coords_start,
																					h_coords_start);
						}
					}
					path_layout_dist += within_node_dist;
					path_layout_dist += from_node_to_node_dist;
					uint64_t nuc_dist = graph.get_length(h);
					path_nuc_dist += nuc_dist;
					// cur_window_end += nuc_dist;
				} else {
					// we only take a look at the current node
					/// f
					if (!graph.get_is_reverse(h)) {
						path_layout_dist += odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);
					} else {
						/// r
						path_layout_dist += odgi::algorithms::layout::coord_dist(h_coords_end, h_coords_start);
					}
					uint64_t nuc_dist = graph.get_length(h);
					path_nuc_dist += nuc_dist;
					// cur_window_end += nuc_dist;
				} // TODO refactor into function end

				sum_stress_squared_dist_weight += pow((((double)path_layout_dist - (double)path_nuc_dist) / (double)path_nuc_dist), 2); // weight = 1 / (d*d)

				num_steps_iterated += 1;
			});
		}

		double stress_result = sum_stress_squared_dist_weight / (double)num_steps_iterated;

		std::cout << "stress: " << stress_result << std::endl;

	}




    return 0;
}

static Subcommand odgi_tension("tension", "evaluate the tension of a graph helping to locate structural variants and abnormalities",
                            PIPELINE, 3, main_tension);


}
