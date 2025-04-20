#include "layout.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> [output_file]" << std::endl;
        return 1;
    }
    
    std::string graph_file = argv[1];
    std::string output_file = (argc > 2) ? argv[2] : "path_analysis.csv";
    
    std::cout << "Loading graph from: " << graph_file << std::endl;
    
    // Load the graph
    try {
        odgi::graph_t graph;
        if (!graph.load(graph_file)) {
            std::cerr << "Error loading graph file: " << graph_file << std::endl;
            return 1;
        }
        
        std::cout << "Graph loaded successfully." << std::endl;
        std::cout << "Node count: " << graph.get_node_count() << std::endl;
        std::cout << "Path count: " << graph.get_path_count() << std::endl;
        
        // Initialize configuration
        cuda::layout_config_t config;
        config.iter_max = 30;
        config.min_term_updates = 100000;
        config.eta_max = 1.0;
        config.eps = 0.01;
        config.iter_with_max_learning_rate = 0;
        config.first_cooling_iteration = 10;
        config.theta = 0.6;
        config.space = 100;
        config.space_max = 100;
        config.space_quantization_step = 10;
        config.nthreads = 48; // Use all threads for the test
        
        // Run the layout with analysis
        std::vector<std::atomic<double>> X(graph.get_node_count() * 2);
        std::vector<std::atomic<double>> Y(graph.get_node_count() * 2);
        
        // Initialize random coordinates
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        for (size_t i = 0; i < X.size(); i++) {
            X[i].store(dist(gen));
            Y[i].store(dist(gen));
        }
        
        // Run the standard algorithm (without NUMA optimization) for comparison
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "\n=== Running standard algorithm (for comparison) ===" << std::endl;
        
        cuda::cuda_layout(config, graph, X, Y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "\nStandard algorithm completed in " << duration << " ms." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 