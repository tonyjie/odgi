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



/*
    double window_size_ = 1;
    if (window_size) {
        window_size_ = args::get(window_size);
    }
    if ((node_sized_windows && window_size) || (pangenome_mode && window_size) || (pangenome_mode && node_sized_windows)) {
        std::cerr
                << "[odgi tension] error: Please specify only one of -w=[N], --window-size=[N] or -n, --node-sized-windows or -p, --pangenome-mode."
                << std::endl;
        return 1;
    }

    vector<path_handle_t> paths;
    graph.for_each_path_handle([&] (const path_handle_t &p) {
       paths.push_back(p);
    });


	if (node_sized_windows || window_size) {
		std::unique_ptr<algorithms::progress_meter::ProgressMeter> progress_meter;
		if (progress) {
			progress_meter = std::make_unique<algorithms::progress_meter::ProgressMeter>(
					paths.size(), "[odgi::tension::main] BED Progress:");
		}
		algorithms::bed_records_class bed;
		bed.open_writer();
#pragma omp parallel for schedule(static, 1) num_threads(thread_count)
		for (auto p: paths) {
			std::string path_name = graph.get_path_name(p);
			uint64_t cur_window_start = 1;
			uint64_t cur_window_end = 0;
			double path_layout_dist = 0;
			uint64_t path_nuc_dist = 0;
			graph.for_each_step_in_path(p, [&](const step_handle_t &s) {
				handle_t h = graph.get_handle_of_step(s);
				algorithms::xy_d_t h_coords_start;
				algorithms::xy_d_t h_coords_end;
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
					step_handle_t prev_s = graph.get_previous_step(s);
					handle_t prev_h = graph.get_handle_of_step(prev_s);
					algorithms::xy_d_t prev_h_coords_start;
					algorithms::xy_d_t prev_h_coords_end;
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
							within_node_dist = algorithms::layout::coord_dist(h_coords_start, h_coords_end);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_end, h_coords_start);
						} else {
							/// f + r
							within_node_dist = algorithms::layout::coord_dist(h_coords_start, h_coords_end);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_end, h_coords_end);
						}
					} else {
						/// r + r
						if (graph.get_is_reverse(h)) {
							within_node_dist = algorithms::layout::coord_dist(h_coords_end, h_coords_start);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_start, h_coords_end);
						} else {
							/// r + f
							within_node_dist = algorithms::layout::coord_dist(h_coords_end, h_coords_start);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_start,
																					h_coords_start);
						}
					}
					path_layout_dist += within_node_dist;
					path_layout_dist += from_node_to_node_dist;
					uint64_t nuc_dist = graph.get_length(h);
					path_nuc_dist += nuc_dist;
					cur_window_end += nuc_dist;
				} else {
					// we only take a look at the current node
					/// f
					if (!graph.get_is_reverse(h)) {
						path_layout_dist += algorithms::layout::coord_dist(h_coords_start, h_coords_end);
					} else {
						/// r
						path_layout_dist += algorithms::layout::coord_dist(h_coords_end, h_coords_start);
					}
					uint64_t nuc_dist = graph.get_length(h);
					path_nuc_dist += nuc_dist;
					cur_window_end += nuc_dist;
				} // TODO refactor into function end
				// we add a new bed entry for each step
				if (node_sized_windows) {
					double path_layout_nuc_dist_ratio = (double) path_layout_dist / (double) path_nuc_dist;
					bed.append(path_name,
							   (cur_window_start - 1),
							   cur_window_end,
							   path_layout_dist,
							   path_nuc_dist,
							   path_layout_nuc_dist_ratio);
					cur_window_start = cur_window_end + 1;
					cur_window_end = cur_window_start - 1;
					path_layout_dist = 0;
					path_nuc_dist = 0;
					// we only add a new entry of the current node exceeds the window size
				} else if ((cur_window_end - cur_window_start + 1) >= window_size_) {
					double path_layout_nuc_dist_ratio = (double) path_layout_dist / (double) path_nuc_dist;
					bed.append(path_name,
							   (cur_window_start - 1),
							   cur_window_end,
							   path_layout_dist,
							   path_nuc_dist,
							   path_layout_nuc_dist_ratio);
					cur_window_start = cur_window_end + 1;
					cur_window_end = cur_window_start - 1;
					path_layout_dist = 0;
					path_nuc_dist = 0;
				}
			});
			/// we have to add the last window
			// we add a new bed entry for each step
			if (!node_sized_windows) {
				double path_layout_nuc_dist_ratio = (double) path_layout_dist / (double) path_nuc_dist;
				bed.append(path_name,
						   (cur_window_start - 1),
						   cur_window_end,
						   path_layout_dist,
						   path_nuc_dist,
						   path_layout_nuc_dist_ratio);
			}
			if (progress) {
				progress_meter->increment(1);
			}
		}
		bed.close_writer();
		if (progress) {
			progress_meter->finish();
		}
		// TODO we want the pangenome tension
		// we want to sum up the tension!
		// tension = (lay/nuc);
		// if (tension < 1) { tension = 1/tension };
	} else {
		std::vector<double> node_tensions(graph.get_node_count() + 1, 0.0);
		std::unique_ptr<algorithms::progress_meter::ProgressMeter> progress_meter;
		if (progress) {
			progress_meter = std::make_unique<algorithms::progress_meter::ProgressMeter>(
					paths.size(), "[odgi::tension::main] Pangenome Mode Progress:");
		}
		// std::cout << "TEST\tPANGENOME\tMODE" << std::endl;
#pragma omp parallel for schedule(static, 1) num_threads(thread_count) shared(node_tensions)
		for (auto p: paths) {
			double path_layout_dist;
			uint64_t path_nuc_dist;
			graph.for_each_step_in_path(p, [&](const step_handle_t &s) {
				path_layout_dist = 0;
				path_nuc_dist = 0;
				handle_t h = graph.get_handle_of_step(s);
				algorithms::xy_d_t h_coords_start;
				algorithms::xy_d_t h_coords_end;
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
					step_handle_t prev_s = graph.get_previous_step(s);
					handle_t prev_h = graph.get_handle_of_step(prev_s);
					algorithms::xy_d_t prev_h_coords_start;
					algorithms::xy_d_t prev_h_coords_end;
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
							within_node_dist = algorithms::layout::coord_dist(h_coords_start, h_coords_end);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_end, h_coords_start);
						} else {
							/// f + r
							within_node_dist = algorithms::layout::coord_dist(h_coords_start, h_coords_end);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_end, h_coords_end);
						}
					} else {
						/// r + r
						if (graph.get_is_reverse(h)) {
							within_node_dist = algorithms::layout::coord_dist(h_coords_end, h_coords_start);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_start, h_coords_end);
						} else {
							/// r + f
							within_node_dist = algorithms::layout::coord_dist(h_coords_end, h_coords_start);
							from_node_to_node_dist = algorithms::layout::coord_dist(prev_h_coords_start,
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
						path_layout_dist += algorithms::layout::coord_dist(h_coords_start, h_coords_end);
					} else {
						/// r
						path_layout_dist += algorithms::layout::coord_dist(h_coords_end, h_coords_start);
					}
					uint64_t nuc_dist = graph.get_length(h);
					path_nuc_dist += nuc_dist;
					// cur_window_end += nuc_dist;
				} // TODO refactor into function end
				double tension = (double)path_layout_dist / (double)path_nuc_dist;
				if (tension < 1.0) {
					tension = 1 / tension;
				}

				node_tensions[graph.get_id(h)] = node_tensions[graph.get_id(h)] + tension;
			});
			if (progress) {
				progress_meter->increment(1);
			}
		}
		if (progress) {
			progress_meter->finish();
		}
		graph.for_each_handle([&](const handle_t &h) {
			uint64_t n_id = graph.get_id(h);
			double handle_tension = node_tensions[n_id];
			uint64_t step_count_h = graph.get_step_count(h);
			double handle_tension_norm = handle_tension / (double)step_count_h;
			std::cout << n_id << "\t" << handle_tension << "\t" << handle_tension_norm << std::endl;
		});
	}
*/


// What we need
// void metrics_compute(odgi::algorithms::layout::Layout &layout, 
//                      const handlegraph::PathHandleGraph &graph, 
//                      int thread_count,
//                      double &stress_result) {



// /*
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
// */

// }	




    return 0;
}

static Subcommand odgi_tension("tension", "evaluate the tension of a graph helping to locate structural variants and abnormalities",
                            PIPELINE, 3, main_tension);


}
