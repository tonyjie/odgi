#include "subcommand.hpp"
#include <iostream>
#include "odgi.hpp"
#include "args.hxx"
#include "algorithms/draw.hpp"
#include "algorithms/layout.hpp"
#include "utils.hpp"
#include "position.hpp"
#include "subgraph/region.hpp"
#include "subgraph/extract.hpp"
#include "split.hpp"

namespace odgi {

using namespace odgi::subcommand;

int main_draw_postproc(int argc, char**argv) { // draw postprocess

    // trick argumentparser to do the right thing with the subcommand
    for (uint64_t i = 1; i < argc - 1; ++i) {
        argv[i] = argv[i + 1];
    }
    std::string prog_name = "odgi draw-postproc";
    argv[0] = (char *) prog_name.c_str();
    --argc;

    args::ArgumentParser parser(
        "Draw 2D layouts of the graph given the 2D coordinates file (.txt). This is a postprocessing command combining the last step of [odgi layout] and [odgi draw].");    
    args::Group mandatory_opts(parser, "[ MANDATORY OPTIONS ]");
    args::ValueFlag<std::string> dg_in_file(mandatory_opts, "FILE", "Load the succinct variation graph in ODGI format from this *FILE*. The file name usually ends with *.og*. It also accepts GFAv1, but the on-the-fly conversion to the ODGI format requires additional time!", {'i', "idx"});
    // Read fron .txt file, instead of .lay file
    args::ValueFlag<std::string> layout_in_file(mandatory_opts, "FILE", "Read the layout coordinates from this .txt format FILE produced by the 2D path-SGD algorithm.", {'c', "coords-in"});
    args::Group files_io_opts(parser, "[ Files IO ]");
    args::ValueFlag<std::string> tsv_out_file(files_io_opts, "FILE", "Write the TSV layout plus displayed annotations to this FILE.", {'T', "tsv"});
    args::ValueFlag<std::string> svg_out_file(files_io_opts, "FILE", "Write an SVG rendering to this FILE.", {'s', "svg"});
    args::ValueFlag<std::string> png_out_file(files_io_opts, "FILE", "Write a rasterized PNG rendering to this FILE.", {'p', "png"});
    args::Group visualizations_opts(parser, "[ Visualization Options ]");
    args::ValueFlag<uint64_t> png_height(visualizations_opts, "FILE", "Height of PNG rendering (default: 1000).", {'H', "png-height"});
    args::ValueFlag<uint64_t> png_border(visualizations_opts, "FILE", "Size of PNG border in bp (default: 10).", {'E', "png-border"});
    args::Flag color_paths(visualizations_opts, "color-paths", "Color paths (in PNG output).", {'C', "color-paths"});
    args::ValueFlag<double> render_scale(visualizations_opts, "N", "Image scaling (default 1.0).", {'R', "scale"});
    args::ValueFlag<double> render_border(visualizations_opts, "N", "Image border (in approximate bp) (default 100.0).", {'B', "border"});
    args::ValueFlag<double> png_line_width(visualizations_opts, "N", "Line width (in approximate bp) (default 0.0).", {'w', "line-width"});
    args::ValueFlag<double> png_path_line_spacing(visualizations_opts, "N", "Spacing between path lines in PNG layout (in approximate bp) (default 0.0).", {'S', "path-line-spacing"});
    args::ValueFlag<std::string> _path_bed_file(visualizations_opts, "FILE",
                                                "Color the nodes based on the input annotation in the given BED FILE. "
                                                "Colors are derived from the 4th column, if present, else from the path name."
                                                "If the 4th column value is in the format 'string#RRGGBB', the RRGGBB color (in hex notation) will be used.",
                                                {'b', "bed-file"});
    args::Group threading(parser, "[ Threading ]");
	args::ValueFlag<uint64_t> nthreads(threading, "N", "Number of threads to use for parallel operations.", {'t', "threads"});
	args::Group processing_info_opts(parser, "[ Processing Information ]");
	args::Flag progress(processing_info_opts, "progress", "Write the current progress to stderr.", {'P', "progress"});
    args::Group program_info_opts(parser, "[ Program Information ]");
    args::HelpFlag help(program_info_opts, "help", "Print a help message for odgi draw.", {'h', "help"});

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
            << "[odgi::draw] error: please specify an input file from where to load the graph via -i=[FILE], --idx=[FILE]."
            << std::endl;
        return 1;
    }

	if (!layout_in_file) {
		std::cerr
				<< "[odgi::draw] error: please specify an input file from where to load the layout from via -c=[FILE], --coords-in=[FILE]."
				<< std::endl;
		return 1;
	}

    if (!tsv_out_file && !svg_out_file && !png_out_file) {
        std::cerr
            << "[odgi::draw] error: please specify an output file to where to store the layout via -p/--png=[FILE], -s/--svg=[FILE], -T/--tsv=[FILE]"
            << std::endl;
        return 1;
    }

	const uint64_t num_threads = args::get(nthreads) ? args::get(nthreads) : 1;

	graph_t graph;
    assert(argc > 0);
    {
        const std::string infile = args::get(dg_in_file);
        if (!infile.empty()) {
            if (infile == "-") {
                graph.deserialize(std::cin);
            } else {
                utils::handle_gfa_odgi_input(infile, "draw", args::get(progress), num_threads, graph);
            }
        }
    }

    if (graph.max_node_id() >= graph.get_node_count() + 1){
        std::cerr << "[odgi::draw] error: the node IDs are not compacted. Please run 'odgi sort' using -O, --optimize to optimize the graph." << std::endl;
        exit(1);
    }

     // handle targets from BED
    std::vector<odgi::path_range_t> path_ranges;
    std::vector<algorithms::color_t> node_id_to_color;
    if (_path_bed_file && !args::get(_path_bed_file).empty()) {
        std::ifstream bed_in(args::get(_path_bed_file));
        std::string line;
        while (std::getline(bed_in, line)) {
            add_bed_range(path_ranges, graph, line);
        }

        if (!path_ranges.empty()) {
            node_id_to_color.resize(graph.get_node_count() + 1, algorithms::COLOR_LIGHTGRAY);

            for (auto &path_range : path_ranges) {
                graph_t subgraph;

                const path_handle_t path_handle = path_range.begin.path;

                // If there is no `name` information, hash the path_name to get a color
                algorithms::color_t path_color = algorithms::hash_color(graph.get_path_name(path_handle));
                if (!path_range.name.empty()) {
                    auto vals = split(path_range.name, '#');
                    if (vals.size() == 2 && vals[1].length() == 6) {
                        // Colors are given in RRGGBB in the BED file, but they are taken in BBGGRR, so we need to switch BB/RR

                        char temp = vals[1][0];
                        vals[1][0] = vals[1][4];
                        vals[1][4] = temp;
                        temp = vals[1][1];
                        vals[1][1] = vals[1][5];
                        vals[1][5] = temp;

                        unsigned int x;
                        std::stringstream ss;
                        ss << std::hex << vals[1];
                        ss >> x;

                        path_color = {0xff000000 + x};
                    } else {
                        path_color = algorithms::hash_color(path_range.name);
                    }
                }

                algorithms::for_handle_in_path_range(
                        graph, path_handle, path_range.begin.offset, path_range.end.offset,
                        [&](const handle_t& handle) {
                            node_id_to_color[graph.get_id(handle)] = path_color;
                        });
            }
        }
    }

    const uint64_t _png_height = png_height ? args::get(png_height) : 1000;
    const double _png_line_width = png_line_width ? args::get(png_line_width) : 0;
    const bool _color_paths = args::get(color_paths);
    const double _png_path_line_spacing = png_path_line_spacing ? args::get(png_path_line_spacing) : 0.0;
    const double svg_scale = !render_scale ? 1.0 : args::get(render_scale);
    uint64_t max_node_depth = 0;
    graph.for_each_handle(
        [&](const handle_t& h) {
            max_node_depth = std::max(graph.get_step_count(h), max_node_depth);
        });
    const double border_bp = !render_border ? std::max(100.0, _png_line_width * max_node_depth) : args::get(render_border);


    // when the code comes to the difference
    size_t num_nodes = graph.get_node_count();

    std::vector<double> X_final(num_nodes * 2);
    std::vector<double> Y_final(num_nodes * 2);
 

    // Load the coord file into vector array X, Y
    if (layout_in_file) {
        auto& infile = args::get(layout_in_file);
        if (!infile.empty()) {
            std::ifstream f(infile.c_str());
            for (size_t i = 0; i < num_nodes * 2; i++) {
                f >> X_final[i];
            }
            for (size_t i = 0; i < num_nodes * 2; i++) {
                f >> Y_final[i];
            }
            f.close();
        }
    }

    // refine order by weakly connected components [Code steal from src/subcommand/layout_main.cpp]
    std::vector<std::vector<handlegraph::handle_t>> weak_components = algorithms::weakly_connected_component_vectors(&graph);

    //uint64_t num_components_on_each_dimension = std::ceil(sqrt(weak_components.size()));
    //std::cerr << " num_components_on_each_dimension " << num_components_on_each_dimension << std::endl;

    double border = 1000.0;
    double curr_y_offset = border;
    std::vector<algorithms::coord_range_2d_t> component_ranges;
    for (auto& component : weak_components) {
        component_ranges.emplace_back();
        auto& component_range = component_ranges.back();
        for (auto& handle : component) {
            uint64_t pos = 2 * number_bool_packing::unpack_number(handle);
            for (uint64_t j = pos; j <= pos+1; ++j) {
                component_range.include(X_final[j], Y_final[j]);
            }
        }
        component_range.x_offset = component_range.min_x - border;
        component_range.y_offset = curr_y_offset - component_range.min_y;
        curr_y_offset += component_range.height() + border;
    }

    for (uint64_t num_component = 0; num_component < weak_components.size(); ++num_component) {
        auto& component_range = component_ranges[num_component];

        for (auto& handle :  weak_components[num_component]) {
            uint64_t pos = 2 * number_bool_packing::unpack_number(handle);

            for (uint64_t j = pos; j <= pos+1; ++j) {
                X_final[j] -= component_range.x_offset;
                Y_final[j] += component_range.y_offset;
            }
        }
    }

    // get layout file
    algorithms::layout::Layout layout(X_final, Y_final);

    if (tsv_out_file) {
        auto& outfile = args::get(tsv_out_file);
        if (!outfile.empty()) {
            if (outfile == "-") {
                layout.to_tsv(std::cout);
            } else {
                ofstream f(outfile.c_str());
                layout.to_tsv(f);
                f.close();
            }
        }
    }

    if (svg_out_file) {
        auto& outfile = args::get(svg_out_file);
        ofstream f(outfile.c_str());
        // todo could be done with callbacks
        std::vector<double> X = layout.get_X();
        std::vector<double> Y = layout.get_Y();
        algorithms::draw_svg(f, X, Y, graph, svg_scale, border_bp);
        f.close();    
    }

    if (png_out_file) {
        auto& outfile = args::get(png_out_file);
        // todo could be done with callbacks
        std::vector<double> X = layout.get_X();
        std::vector<double> Y = layout.get_Y();
        algorithms::draw_png(outfile, X, Y, graph, 1.0, border_bp, 0, _png_height, _png_line_width, _png_path_line_spacing, _color_paths, node_id_to_color);
    }    

    return 0;
}

static Subcommand odgi_draw_postproc("draw-postproc", "Draw 2D layouts of the graph given the 2D coordinates file (.txt). This is a postprocessing command combining the last step of [odgi layout] and [odgi draw].",
                                        PIPELINE, 3, main_draw_postproc);

}