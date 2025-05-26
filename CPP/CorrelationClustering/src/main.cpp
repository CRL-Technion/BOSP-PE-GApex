#include "Utils.h"
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <unordered_map>

using namespace std;

int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    std::vector<string> objective_files;
    std::string cluster_map_file;

    // ==========================================================================
    // Parsing the supported line arguments options
    // ==========================================================================
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("map,m", po::value< std::vector<string> >(&objective_files)->multitoken(), "files for edge weight")
        ("clusters,c", po::value<std::string>()->default_value(""), "clusters mapping file")
        ("eps,e", po::value<double>()->default_value(0), "approximation factor")
        ("super_edges_file,s", po::value<std::string>()->required(), "Name of the super-edges output file")
        ("logging_file,l", po::value<std::string>()->default_value(""), "logging file")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    po::notify(vm);

    std::string logFile = vm["logging_file"].as<std::string>();
    std::string superEdgesFile = vm["super_edges_file"].as<std::string>();

    // ==========================================================================
    // Loading the bi-objective graph files
    // ==========================================================================
    size_t graph_size;
    std::vector<Edge> edges;
    int objectives_count = objective_files.size();
    double eps = vm["eps"].as<double>();
    std::vector<double> approx_factor = std::vector<double>(objectives_count, eps);

    std::cout << "Loading the following objective graph files:" << endl;
    for (auto file : objective_files) {
        std::cout << file << std::endl;
    }

    if (load_gr_files(objective_files, edges, graph_size) == false) {
        std::cout << "Failed to load gr files" << std::endl;
        return -1;
    }
    std::cout << "Graph Size (# vertices): " << graph_size << std::endl;

    // Constructing the graph's adjacency matrix
    std::cout << "Constructing the forward and backwards adjacency matrices ..." << endl;
    AdjacencyMatrix graph(graph_size, edges);
    AdjacencyMatrix inv_graph(graph_size, edges, true);

    // ==========================================================================
    // Loading the clustering analysis file
    // ==========================================================================
    ClusterMapping clusters_map;
    std::cout << "Reading clusters meta-file " << vm["clusters"].as<std::string>() << " ... ";
    load_clusters_mapping(vm["clusters"].as<std::string>(), clusters_map);
    std::cout << "Number of clusters is: " << clusters_map.size() << std::endl;    

    // ==========================================================================
    // Computing all-pairs shortest-path between all boundary vertices
    // ==========================================================================
    std::cout << "Starting clusters analysis for edge contraction:" << std::endl;
    int boundary_vertices_minimal_count = 2;
    std::vector<int> boundary_vertices;
    std::vector<ContractedEdge> contractedEdges;
    std::vector<Edge> incontractedEdges;
    double progress;
    double compression_ratio;
    double last_progress = -1;
    long int total_runtime_ms = 0;
    
    // Boolean lookup table to mark which vertices are boundary vertices
    std::vector<bool> current_cluster_boundary_vertices(graph_size + 1, false);

    // Array of boundary vertices' IDs
    std::vector<int> current_cluster_boundary_vertices_ids;

    for (int cluster_id = 0; cluster_id < clusters_map.size(); cluster_id++)
    {
        progress = (double)cluster_id / clusters_map.size();
        if (progress - last_progress > 0.01)
        {
            updateProgressBar(cluster_id, clusters_map.size());
            last_progress = progress;
        }

        // Creating a lookup table of all vertices inside the cluster
        std::vector<bool> lookupTable(graph_size + 1, false);
        int nodes_in_cluster = 0;
        for (int node_id : clusters_map[cluster_id])
        {
            lookupTable[node_id] = true;
            nodes_in_cluster++;
        }

        // Initializing the current cluster's boundary nodes lookup table
        for (int k = 0; k < current_cluster_boundary_vertices.size(); k++)
        {
            current_cluster_boundary_vertices[k] = false;
        }
        
        // Identifying the current cluster's boundary nodes
        int boundary_vertices_count = get_boundary_vertices_enhanced(current_cluster_boundary_vertices, 
            graph, inv_graph, lookupTable);

        // Asserting that are sufficient number of boundary nodes (the cluster is significant enough)
        if (boundary_vertices_count < boundary_vertices_minimal_count)
        {
            continue;
        }

        // Adding the current cluster's identified boundary nodes to the global boundary nodes list
        current_cluster_boundary_vertices_ids.clear();
        for (size_t k = 0; k < graph_size + 1; k++)
        {
            if (current_cluster_boundary_vertices[k])
            {
                current_cluster_boundary_vertices_ids.push_back(k);
                boundary_vertices.push_back(k);
            }
        }
        
        // Invoking all-(boundary)-pairs Dijkstra search for edge contraction
        std::vector<ContractedEdge> currentClusterContractedEdges;
        std::vector<Edge> currentClusterIncontractablePaths;
        std::vector<std::vector<int>> stats;
        
        total_runtime_ms +=            
            all_pairs_shortest_paths(clusters_map[cluster_id], current_cluster_boundary_vertices_ids,
            graph, approx_factor, currentClusterContractedEdges, stats, currentClusterIncontractablePaths);

        // Adding the newly computed super-edges to the general super-edges list
        contractedEdges.insert(contractedEdges.end(),
            currentClusterContractedEdges.begin(), currentClusterContractedEdges.end());

        incontractedEdges.insert(incontractedEdges.end(),
            currentClusterIncontractablePaths.begin(), currentClusterIncontractablePaths.end());
    }
    updateProgressBar(clusters_map.size(), clusters_map.size());
    std::cout << std::endl;

    // Writing the log file comprised of paths that were contracted yes/no and total time
    std::cout << "Writing log file ... ";
    std::ofstream logFileStrm(logFile);
    logFileStrm << total_runtime_ms << std::endl;
    logFileStrm << contractedEdges.size() << std::endl;
    logFileStrm << incontractedEdges.size() << std::endl;
    for (Edge e : incontractedEdges)
    {
        logFileStrm << e.source << "," << e.target << ",0" <<std::endl;
    }
    for (ContractedEdge e : contractedEdges)
    {
        logFileStrm << e.source << "," << e.target << ",1" << std::endl;
    }
    logFileStrm.close();
    std::cout << "Done" << std::endl;

    // Exporting boundary nodes list to file
    std::cout << "Exporting boundary vertices (" << boundary_vertices.size() << ") to file ... ";
    std::ofstream outFile(R"(boundary_vertices.txt)");
    for (int node : boundary_vertices)
    {
        outFile << node << std::endl;
    }
    outFile.close();
    std::cout << "Done" << std::endl;

    // Exporting the super-edges to file
    std::cout << "Exporting super edges (" << contractedEdges.size() << ") to file ... ";
    export_contracted_edges(superEdgesFile, contractedEdges);
    std::cout << "Done" << std::endl;

    // Reporting the total contracted paths length
    std::cout << "Contracted paths needed storage: " << graph.total_length_of_paths * 8.0 / 1024 / 1024 / 1024 << " [Gigabytes]\n";

    return 0;
}