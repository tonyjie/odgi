#include "subcommand.hpp"
#include <iostream>
#include "odgi.hpp"
#include "args.hxx"
#include <omp.h>
#include "algorithms/layout.hpp"
#include "algorithms/tension/tension_bed_records_queued_writer.hpp"
#include <numeric>
#include "progress.hpp"
#include "cuda/metrics.h"

namespace odgi {

using namespace odgi::subcommand;

// ========================= check number of node crossings. =========================
// Here we only consider the nodes in the pangenome graph. They are segments in the layout. 

struct Point 
{ 
	double x; 
	double y; 
}; 

// Given three collinear points p, q, r, the function checks if 
// point q lies on line segment 'pr' 
bool onSegment(Point p, Point q, Point r) 
{ 
	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && 
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y)) 
	return true; 

	return false; 
} 

// To find orientation of ordered triplet (p, q, r). 
// The function returns following values 
// 0 --> p, q and r are collinear 
// 1 --> Clockwise 
// 2 --> Counterclockwise 
int orientation(Point p, Point q, Point r) 
{ 
	// See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
	// for details of below formula. 
	int val = (q.y - p.y) * (r.x - q.x) - 
			(q.x - p.x) * (r.y - q.y); 

	if (val == 0) return 0; // collinear 

	return (val > 0)? 1: 2; // clock or counterclock wise 
} 

// The main function that returns true if line segment 'p1q1' 
// and 'p2q2' intersect. 
bool doIntersect(Point p1, Point q1, Point p2, Point q2) 
{ 
	// Find the four orientations needed for general and 
	// special cases 
	int o1 = orientation(p1, q1, p2); 
	int o2 = orientation(p1, q1, q2); 
	int o3 = orientation(p2, q2, p1); 
	int o4 = orientation(p2, q2, q1); 

	// General case 
	if (o1 != o2 && o3 != o4) 
		return true; 

	// Special Cases 
	// p1, q1 and p2 are collinear and p2 lies on segment p1q1 
	if (o1 == 0 && onSegment(p1, p2, q1)) return true; 

	// p1, q1 and q2 are collinear and q2 lies on segment p1q1 
	if (o2 == 0 && onSegment(p1, q2, q1)) return true; 

	// p2, q2 and p1 are collinear and p1 lies on segment p2q2 
	if (o3 == 0 && onSegment(p2, p1, q2)) return true; 

	// p2, q2 and q1 are collinear and q1 lies on segment p2q2 
	if (o4 == 0 && onSegment(p2, q1, q2)) return true; 

	return false; // Doesn't fall in any of the above cases 
} 
// ========================= End of check number of node crossings. =========================



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
	// option to control if we want to compute the step-stress (only count each within-node distance, but iterate though path, then iterate through steps within path -- the same as old-node-stress)
	args::Flag step_stress(tension_opts, "step-stress", "compute the step-stress", {'s', "step-stress"});
	// option for old-node-stress
	args::Flag old_node_stress(tension_opts, "old-node-stress", "compute the old node-stress", {'o', "old-node-stress"});
	// option to control if we want to compute the node-crossing (node is pangenome graph node, not visualization node. Each node is a segment in the layout. )
	args::Flag node_crossing(tension_opts, "node-crossing", "compute the number of node-crossing", {'x', "node-crossing"});
	// avg_path_len_error
    args::Flag avg_path_len_error(tension_opts, "avg-path-len-error", "compute the average path length error", {'e', "avg-path-len-error"});
    // path_stress
    args::Flag path_stress(tension_opts, "path-stress", "compute the path-stress", {'p', "path-stress"});

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

	// Save the path of the graph
	std::vector<odgi::path_handle_t> paths;
	graph.for_each_path_handle([&] (const odgi::path_handle_t &p) {
		paths.push_back(p);
	});


// ===== 1. Check the number of node crossings. =====
	if (node_crossing) {
        
#ifdef CPU_NODE_CROSSING
		cout << "compute the number of node-crossing using CPUs" << endl;

		// save a vector of all the node handle_t
		std::vector<handle_t> nodes;
		graph.for_each_handle([&](const handle_t& h) {
			nodes.push_back(h);
		});

		// iterate through each node pair
		uint64_t num_node_crossing = 0;
		#pragma omp parallel for schedule(static, 1) num_threads(thread_count) reduction(+:num_node_crossing)
		for (uint64_t i = 0; i < nodes.size(); i++) {
			for (uint64_t j = i + 1; j < nodes.size(); j++) {
				// get the node coordinates
				odgi::algorithms::xy_d_t node_i_coords_start;
				odgi::algorithms::xy_d_t node_i_coords_end;
				odgi::algorithms::xy_d_t node_j_coords_start;
				odgi::algorithms::xy_d_t node_j_coords_end;
				node_i_coords_start = layout.coords(nodes[i]);
				node_i_coords_end = layout.coords(graph.flip(nodes[i]));
				node_j_coords_start = layout.coords(nodes[j]);
				node_j_coords_end = layout.coords(graph.flip(nodes[j]));
				// check if the two nodes are crossing
				if (doIntersect({node_i_coords_start.x, node_i_coords_start.y}, {node_i_coords_end.x, node_i_coords_end.y}, \
								{node_j_coords_start.x, node_j_coords_start.y}, {node_j_coords_end.x, node_j_coords_end.y})) {
					num_node_crossing += 1;
				}
			}
		}

		cout << "node-crossing: " << num_node_crossing << " / " << nodes.size() * (nodes.size() - 1) / 2 << " = " << (double)num_node_crossing / (double)(nodes.size() * (nodes.size() - 1) / 2) << endl;
#endif

		// cout << "===== GPU version to compute # of node crossings =====" << endl;
		// // GPU version of computing node-crossing
		// // O(N*N) complexity. For Chr1 with N=1.1e7, it takes 131min. 

        cout << "===== GPU version to compute # of node crossings =====" << endl;
		cuda::cuda_node_crossing(graph, layout);

	}

// ===== 2. Check the average path length error =====
    struct PathInfo {
        path_handle_t path_handle;
        uint64_t path_nuc_dist;
        double path_layout_len;
        double path_len_error;
        // Constructor
        PathInfo(path_handle_t path_handle, uint64_t path_nuc_dist, double path_layout_len, double path_len_error)
            : path_handle(path_handle), path_nuc_dist(path_nuc_dist), path_layout_len(path_layout_len), path_len_error(path_len_error) {};
    };

    if (avg_path_len_error) {
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

        // DEBUG: show top-10 path topology error
        // std::sort(path_topology.begin(), path_topology.end(), [](const PathInfo& a, const PathInfo& b) {
        //     return a.path_len_error > b.path_len_error;
        // });

        // std::cout << "top-10 path topology error: " << std::endl;
        // // the largest top-10 path topology error
        // // for (uint64_t i = 0; i < path_topology.size() - 1; i++) {
        // for (uint64_t i = 0; i < 10; i++) {
        //     // how many steps in this path
        //     uint64_t num_steps = 0;
        //     graph.for_each_step_in_path(path_topology[i].path_handle, [&](const odgi::step_handle_t &s) {
        //         num_steps += 1;
        //     }
        //     );
        //     std::cout << "path: " << graph.get_path_name(path_topology[i].path_handle) << " nuc_dist: " << path_topology[i].path_nuc_dist << "; layout_len: " << path_topology[i].path_layout_len << "; error: " << path_topology[i].path_len_error << 
        //     " num_steps: " << num_steps << std::endl;
        // }

        // further DEBUG: investigate into the first path (weird)
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

    } // end of avg_path_len_error


    // ===== 3. Check the new node-stress =====
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
		cout << "new node stress: " << sum_stress << endl;

	} // end of new_node_stress
	
    // ===== 4. Check the step-stress =====
	if (step_stress) {
		// so-called "step-stress". Only consider the within-node distance.
		// Only one difference compared to the above new node-stress. 
		// 1. First iterate through each step (node) in the path, then iterate through paths. So there are some over-counting instead of iterate though all the nodes directly. 
		// We want to check which difference (1) or (2) made the change on the results. 
		cout << "compute the step-stress" << endl;

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
				odgi::handle_t h = graph.get_handle_of_step(s);
				odgi::algorithms::xy_d_t h_coords_start;
				odgi::algorithms::xy_d_t h_coords_end;
				// get the node nucleotide length
				path_nuc_dist = graph.get_length(h);
				// get the node layout distance
				h_coords_start = layout.coords(h);
				h_coords_end = layout.coords(graph.flip(h));
				path_layout_dist = odgi::algorithms::layout::coord_dist(h_coords_start, h_coords_end);

				sum_stress_squared_dist_weight += pow((((double)path_layout_dist - (double)path_nuc_dist) / (double)path_nuc_dist), 2); // weight = 1 / (d*d)

				num_steps_iterated += 1;
			});
		}

		double stress_result = sum_stress_squared_dist_weight / (double)num_steps_iterated;

		std::cout << "step stress: " << stress_result << std::endl;
	} // end of step_stress



    // ===== 5. Check the old node-stress =====
    if (old_node_stress) {
	// The original node-stress. There are two differences compared to the above new node-stress. 
	// 1. First iterate through paths, then iterate through each step (node) in the path. 
	// 2. Not only consider the start-point and end-point of one node, but also consider the relationship to the previous node. (And if they are reversed or not. )
		cout << "compute the old node-stress" << endl;

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
    } // end of old_node_stress

    
    // ===== 6. Check the Path-Stress using GPU =====
    // Path stress considers ALL the layout_node pairs within each path, and sums up the stress. 
    // Its O(P*N*N) complexity requires the GPU kernel. 
    if (path_stress) {
        cout << "compute the path-stress using GPU" << endl;
        cuda::cuda_path_stress(graph, layout);
    }



    return 0;
}

static Subcommand odgi_tension("tension", "evaluate the tension of a graph helping to locate structural variants and abnormalities",
                            PIPELINE, 3, main_tension);


}
