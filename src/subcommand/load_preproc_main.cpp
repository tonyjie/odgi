#include "subcommand.hpp"
#include <iostream>
#include "odgi.hpp"
#include "args.hxx"
#include "algorithms/path_sgd_layout.hpp"
#include "utils.hpp"

// #define debug_load_preproc
#define debug_show_step_count
#define debug_gen_csv_array
namespace odgi {

using namespace odgi::subcommand;

int main_load_preproc(int argc, char **argv) {

    // trick argumentparser to do the right thing with the subcommand
    for (uint64_t i = 1; i < argc - 1; ++i) {
        argv[i] = argv[i + 1];
    }
    std::string prog_name = "odgi layout";
    argv[0] = (char *) prog_name.c_str();
    --argc;

    args::ArgumentParser parser(
        "Load the graph (.og), and preprocess it for the 2D layout algorithm. The output is some required arrays, which will be stored in the .txt files.");
    args::Group mandatory_opts(parser, "[ MANDATORY OPTIONS ]");
    args::ValueFlag<std::string> dg_in_file(mandatory_opts, "FILE", "Load the succinct variation graph in ODGI format from this *FILE*. The file name usually ends with *.og*. It also accepts GFAv1, but the on-the-fly conversion to the ODGI format requires additional time!", {'i', "idx"});
    args::Group files_io_opts(parser, "[ Files IO ]");
    args::ValueFlag<std::string> pos_out_file(files_io_opts, "FILE", "Write the positions array of each path to this FILE (.txt).", {'p', "pos"});
    args::ValueFlag<std::string> vis_id_out_file(files_io_opts, "FILE", "Write the vis_id array of each path to this FILE (.txt).", {'v', "vis_id"});
    args::ValueFlag<std::string> vis_dist_csv_file(files_io_opts, "FILE", "Write the vis_dist array to this FILE (.csv).", {'d', "vis_dist_csv"});
    // args::ValueFlag<std::string> node_length_out_file(files_io_opts, "FILE", "Write the node_length array to this FILE (.txt).", {'n', "node_length"}); # don't need it for Pytorch implementation
    args::ValueFlag<std::string> init_layout_out_file(files_io_opts, "FILE", "Write the initial layout coordinates of all the visualization nodes to this FILE (.txt).", {"init_layout"});
    args::ValueFlag<std::string> config_out_file(files_io_opts, "FILE", "Write the configurations of the algorithm to this FILE (.txt), including the learning rate settings (etas)", {'c', "config"});
    args::ValueFlag<std::string> p_sgd_in_file(files_io_opts, "FILE",
                                               "Specify a line separated list of paths to sample from for the on the fly term generation process in the path guided 2D SGD (default: sample from all paths).",
                                               {'f', "path-sgd-use-paths"});
    args::Group threading_opts(parser, "[ Threading ]");
    args::ValueFlag<uint64_t> nthreads(threading_opts, "N",
                                       "Number of threads to use for parallel operations.",
                                       {'t', "threads"});
    args::Group processing_info_opts(parser, "[ Processsing Information ]");
    args::Flag progress(processing_info_opts, "progress", "Write the current progress to stderr.", {'P', "progress"});    
    args::Group program_info_opts(parser, "[ Program Information ]");
    args::HelpFlag help(program_info_opts, "help", "Print a help summary for odgi layout.", {'h', "help"});

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

    if (!dg_in_file) {
        std::cerr
            << "[odgi::load-preproc] error: Please specify an input file from where to load the graph via -i=[FILE], --idx=[FILE]."
            << std::endl;
        return 1;
    }

    const uint64_t num_threads = nthreads ? args::get(nthreads) : 1;

	graph_t graph;
    assert(argc > 0);
    if (!args::get(dg_in_file).empty()) {
        std::string infile = args::get(dg_in_file);
        if (infile == "-") {
            graph.deserialize(std::cin);
        } else {
			utils::handle_gfa_odgi_input(infile, "layout", args::get(progress), num_threads, graph);
        }
    }    

    if (!graph.is_optimized()) {
		std::cerr << "[odgi::layout] error: the graph is not optimized. Please run 'odgi sort' using -O, --optimize." << std::endl;
		exit(1);
    }

    if (!pos_out_file) {
        std::cerr
            << "[odgi::load-preproc] error: Please specify an output file to write the positions array of each path via -p=[FILE], --pos=[FILE]."
            << std::endl;
        return 1;
    }

    if (!vis_id_out_file) {
        std::cerr
            << "[odgi::load-preproc] error: Please specify an output file to write the vis_id array of each path via -v=[FILE], --vis_id=[FILE]."
            << std::endl;
        return 1;
    }

    if (!vis_dist_csv_file) {
        std::cerr
            << "[odgi::load-preproc] error: Please specify an output file to write the vis_id array of each path via -d=[FILE], --vis_dist_csv=[FILE]."
            << std::endl;
        return 1;
    }


    // if (!node_length_out_file) {
    //     std::cerr
    //         << "[odgi::load-preproc] error: Please specify an output file to write the node_length array via -n=[FILE], --node_length=[FILE]."
    //         << std::endl;
    //     return 1;
    // }

    if (!init_layout_out_file) {
        std::cerr
            << "[odgi::load-preproc] error: Please specify an output file to write the node_length array via --init_layout=[FILE]."
            << std::endl;
        return 1;
    }

    if (!config_out_file) {
        std::cerr
            << "[odgi::load-preproc] error: Please specify an output file to write the configurations of the algorithm via -c=[FILE], --config=[FILE]."
            << std::endl;
        return 1;
    }


    // open the output files
    auto& pos_out_f = args::get(pos_out_file);
    std::ofstream f_pos(pos_out_f.c_str());

    auto& vis_id_out_f = args::get(vis_id_out_file);
    std::ofstream f_vis_id(vis_id_out_f.c_str());

    auto& vis_dist_csv_out_f = args::get(vis_dist_csv_file);
    std::ofstream f_vis_dist_csv(vis_dist_csv_out_f.c_str());

    // auto& node_length_out_f = args::get(node_length_out_file);
    // std::ofstream f_node_length(node_length_out_f.c_str());

    auto& init_layout_out_f = args::get(init_layout_out_file);
    std::ofstream f_init_layout(init_layout_out_f.c_str());

    auto& config_out_f = args::get(config_out_file);
    std::ofstream f_config(config_out_f.c_str());

#ifdef debug_gen_csv_array
    // here we give each visualization node a unique ID. 
    // 2 visualization nodes for each node would have distance of the corresponding node length
    // 2 visualization nodes for each edge would have distance of 0.01 (small value)
    // the other pair of visualization nodes are not directly connected

    // get node_length
    uint64_t node_cnt = graph.get_node_count();
    // std::cout << node_cnt << endl;
    graph.for_each_handle([&](const handle_t &handle) {
        uint32_t len = graph.get_length(handle);
        uint64_t node_id = graph.get_id(handle);
        
        // std::cout << "node " << node_id << ": " << len << std::endl;
    });

    uint64_t vis_node_count = node_cnt * 2;
    // generate a matrix of size vis_node_count * vis_node_count, initialized with 0. 
    // use std::vector
    // vis_node distance matrix
    std::vector<std::vector<double>> vis_node_dist;
    vis_node_dist.resize(vis_node_count);
    for (int32_t i = 0; i < vis_node_count; i++) {
        vis_node_dist[i].resize(vis_node_count);
    }
    // initialize the matrix with 0
    for (int32_t i = 0; i < vis_node_count; i++) {
        for (int32_t j = 0; j < vis_node_count; j++) {
            vis_node_dist[i][j] = 0;
        }
    }
    
    // fill in with the node_length
    graph.for_each_handle([&](const handle_t &handle) {
        uint32_t len = graph.get_length(handle);
        uint64_t node_id = graph.get_id(handle);
        uint64_t vis_id = (node_id - 1) * 2;
        vis_node_dist[vis_id][vis_id + 1] = len;
        vis_node_dist[vis_id + 1][vis_id] = len;
    });


    // fill in with the connected vis node: the endpoint of prev_node to the startpoint of current_node are connected
    // iterate through steps in the path
    graph.for_each_path_handle([&](const path_handle_t &path) {
        bool first_step = true;
        uint64_t prev_vis_id_end; // the endpoint of prev_node
        uint64_t curr_vis_id_start; // the startpoint of current_node
        graph.for_each_step_in_path(path, [&](const step_handle_t &step) {
            handle_t handle = graph.get_handle_of_step(step);
            uint64_t node_id = graph.get_id(handle);
            bool is_rev = graph.get_is_reverse(handle);
            uint32_t len = graph.get_length(handle);
            uint64_t vis_id = (node_id - 1) * 2; // startpoint of the current node, if is_rev = False
            uint64_t vis_id_rev = (node_id - 1) * 2 + 1; // startpoint of the current node, if is_rev = True
            curr_vis_id_start = is_rev ? vis_id_rev : vis_id; // startpoint of current_node

            if (!first_step) { // skip the first_step in the path
                vis_node_dist[prev_vis_id_end][curr_vis_id_start] = 0.01;
                vis_node_dist[curr_vis_id_start][prev_vis_id_end] = 0.01;
                
                // print the prev_vis_id_end and curr_vis_id_start
                // std::cout << "prev_vis_id_end: " << prev_vis_id_end << "; curr_vis_id_start: " << curr_vis_id_start << std::endl;
            }
            prev_vis_id_end = is_rev ? vis_id : vis_id_rev; // update the endpoint of prev_node
            first_step = false;
        });
    });

    // print out the whole matrix
    // for (int32_t i = 0; i < vis_node_count; i++) {
    //     for (int32_t j = 0; j < vis_node_count; j++) {
    //         std::cout << vis_node_dist[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // generate CSV array
    // first row:   , 0, 1, 2, ..., vis_node_count - 1
    // second row: 0, vis_node_dist[0][0], vis_node_dist[0][1], ..., vis_node_dist[0][vis_node_count - 1]
    // third row:  1, vis_node_dist[1][0], vis_node_dist[1][1], ..., vis_node_dist[1][vis_node_count - 1]
    // ...
    // last row:   vis_node_count - 1, vis_node_dist[vis_node_count - 1][0], vis_node_dist[vis_node_count - 1][1], ..., vis_node_dist[vis_node_count - 1][vis_node_count - 1]

    // first row
    f_vis_dist_csv << std::setw(6) << " ";
    for (int32_t i = 0; i < vis_node_count; i++) {
        // fix the printed interval of each column
        f_vis_dist_csv << ", " << std::setw(6) << i;
    }
    f_vis_dist_csv << endl;
    // begin with second row
    for (int32_t i = 0; i < vis_node_count; i++) {
        f_vis_dist_csv << std::setw(6) << i;
        for (int32_t j = 0; j < vis_node_count; j++) {
            f_vis_dist_csv << ", " << std::setw(6) << vis_node_dist[i][j];
        }
        f_vis_dist_csv << endl;
    }

#endif

    // get positions & vis_id
    // iterate through paths
    f_pos << graph.get_path_count() << endl;
#ifdef debug_show_step_count
    int path_count_idx = 0;
#endif
    graph.for_each_path_handle([&](const path_handle_t &path) {
        string path_name = graph.get_path_name(path);
        int32_t step_count = graph.get_step_count(path);

        f_pos << step_count << endl;

#ifdef debug_show_step_count
        path_count_idx ++;
        cerr << "path_count_idx: " << path_count_idx << "; path: " << path_name << "; step_count: " << step_count << endl;
#endif

        // int32_t positions[step_count * 2];
        // int32_t vis_id[step_count * 2];
        int32_t* positions = new int32_t[step_count * 2];
        int32_t* vis_id = new int32_t[step_count * 2];

        // show the size of vis_id
        // cerr << "vis_id size: " << sizeof(vis_id) << endl;
        // cerr << "positions size: " << sizeof(positions) << endl;

        int32_t idx = 0;
        int32_t current_pos = 0;
        
        // iterate through steps in the path
        graph.for_each_step_in_path(path, [&](const step_handle_t &step) {
            handle_t handle = graph.get_handle_of_step(step);
            uint64_t node_id = graph.get_id(handle);
            bool is_rev = graph.get_is_reverse(handle);
            uint32_t len = graph.get_length(handle);
#ifdef debug_load_preproc
            cerr << "node_id: " << node_id << " is_rev: " << is_rev << " len: " << len << endl;
#endif
            // vis_id
            vis_id[idx] = (node_id - 1) * 2;
            vis_id[idx + 1] = (node_id - 1) * 2 + 1;

            // positions array varies depending on the orientation of the node
            if (!is_rev) {
                positions[idx] = current_pos;
                current_pos += len;
                positions[idx + 1] = current_pos;
            } else {
                positions[idx + 1] = current_pos;
                current_pos += len;
                positions[idx] = current_pos;
            }

            idx += 2;
        });
        
        for (int32_t i = 0; i < step_count * 2; i++) {
            f_pos << positions[i] << " ";
        }
        f_pos << endl;
        for (int32_t i = 0; i < step_count * 2; i++) {
            f_vis_id << vis_id[i] << " ";
        }
        f_vis_id << endl;

        delete[] positions;
        delete[] vis_id;
    });

    // get node_length
    // f_node_length << graph.get_node_count() << endl;
    // graph.for_each_handle([&](const handle_t &handle) {
    //     uint32_t len = graph.get_length(handle);
    //     f_node_length << len << " ";
    // });
    // f_node_length << endl;

    // get init_layout
    size_t node_count = graph.get_node_count();

    std::random_device dev;
    std::mt19937 rng(dev());
    std::normal_distribution<double> gaussian_noise(0,  sqrt(node_count * 2));

    double *graph_X = new double[node_count * 2];
    double *graph_Y = new double[node_count * 2];

    uint64_t len = 0;
    graph.for_each_handle([&](const handle_t &h) {
        uint64_t pos = 2 * number_bool_packing::unpack_number(h);
        graph_X[pos] = len;
        graph_Y[pos] = gaussian_noise(rng);
        len += graph.get_length(h);
        graph_X[pos + 1] = len;
        graph_Y[pos + 1] = gaussian_noise(rng);
        }
    );
    // write to f_init_layout: first line is graph_X; second line is graph_Y
    for (int32_t i = 0; i < node_count * 2; i++) {
        f_init_layout << graph_X[i] << " ";
    }
    f_init_layout << endl;
    for (int32_t i = 0; i < node_count * 2; i++) {
        f_init_layout << graph_Y[i] << " ";
    }
    f_init_layout << endl;
    


    // get configs
    std::vector<path_handle_t> path_sgd_use_paths;
    if (p_sgd_in_file) {
        std::string buf;
        std::ifstream use_paths(args::get(p_sgd_in_file).c_str());
        while (std::getline(use_paths, buf)) {
            // check if the path is actually in the graph, else print an error and exit 1
            if (graph.has_path(buf)) {
                path_sgd_use_paths.push_back(graph.get_path_handle(buf));
            } else {
                std::cerr << "[odgi::layout] error: path '" << buf
                          << "' as was given by -f=[FILE], --path-sgd-use-paths=[FILE]"
                    " is not present in the graph. Please remove this path from the file and restart 'odgi sort'.";
            }
        }
        use_paths.close();
    } else {
        graph.for_each_path_handle(
            [&](const path_handle_t &path) {
                path_sgd_use_paths.push_back(path);
            });
    }


    std::function<uint64_t(const std::vector<path_handle_t> &)> get_max_path_step_count
        = [&](const std::vector<path_handle_t> &path_sgd_use_paths) {
              uint64_t max_path_step_count = 0;
              for (auto& path : path_sgd_use_paths) {
                  max_path_step_count = std::max(max_path_step_count, graph.get_step_count(path));
              }
              return max_path_step_count;
          };

    uint64_t max_path_step_count = get_max_path_step_count(path_sgd_use_paths);
    double eta_max = (double) max_path_step_count * max_path_step_count;
    double w_min = 1.0 / eta_max;
    double w_max = 1.0;
    int32_t iter_max = 30; // actually total iteration is 31. Start from [0, 30]. 
    int32_t iter_with_max_learning_rate = 0; 
    double eps = 0.01;

    std::vector<double> etas = odgi::algorithms::path_linear_sgd_layout_schedule(w_min, w_max, iter_max, iter_with_max_learning_rate, eps);

    std::function<uint64_t(const std::vector<path_handle_t> &)> get_sum_path_step_count
        = [&](const std::vector<path_handle_t> &path_sgd_use_paths) {
            uint64_t sum_path_step_count = 0;
            for (auto& path : path_sgd_use_paths) {
                sum_path_step_count += graph.get_step_count(path);
            }
            return sum_path_step_count;
        };

    uint64_t min_term_updates;
    uint64_t sum_path_step_count = get_sum_path_step_count(path_sgd_use_paths);
    min_term_updates = 10 * sum_path_step_count;

    // write to f_config
    // f_config << min_term_updates << endl;
    // f_config << iter_max + 1 << endl;
    // write the total number of nodes in the graph to the config file
    f_config << graph.get_node_count() << endl;
    for (int32_t i = 0; i < etas.size(); i++) {
        f_config << etas[i] << " ";
    }

    // close ofstream
    f_pos.close();
    f_vis_id.close();
    // f_node_length.close();
    f_init_layout.close();
    f_config.close();



    return 0;
}

static Subcommand odgi_load_preproc("load-preproc", "Load the graph (.og), and preprocess it for the 2D layout algorithm. The output is some required arrays, which will be stored in the .txt files. ",
                               PIPELINE, 3, main_load_preproc);

}