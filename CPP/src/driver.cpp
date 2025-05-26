#include <iostream>
#include <memory>
#include <time.h>
#include <fstream>

#include "ShortestPathHeuristic.h"
#include "Utils/Definitions.h"
#include "Utils/IOUtils.h"
#include "Utils/Logger.h"
#include "BOAStar.h"
#include "PPA.h"
#include "SingleCriteria.h"
#include "ApexSearch.h"
#include "ApexPE.h"
#include "NAMOA.h"

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

using namespace std;

const std::string resource_path = "resources/";
const std::string output_path = "output/";
const MergeStrategy DEFAULT_MERGE_STRATEGY = MergeStrategy::SMALLER_G2;
std::string alg_variant = "";


// Simple example to demonstarte the usage of the algorithm

void single_run_map(size_t graph_size, AdjacencyMatrix& graph, AdjacencyMatrix&inv_graph, size_t source, size_t target, std::ofstream& output, std::string algorithm, MergeStrategy ms, LoggerPtr logger, 
    double eps, unsigned int time_limit, bool target_bounding, bool inflate_h, LoggerPtr optimal_paths_logger)
{
    // Compute heuristic
    std::cout << "Start Computing Heuristic...";
    ShortestPathHeuristic sp_heuristic(target, graph_size, inv_graph);
    std::cout << "Done\n";

    // Yaron 16-03-23 : Inflating heuristics according to a user-defined eps (eps>=0)
    if (inflate_h)
    {
        //sp_heuristic.inflate_h_by_eps(eps);
        //std::cout << "Heuristic was inflated by a factor of " << (1 + eps) << std::endl;
        sp_heuristic.inflate_h_by_eps(0.01);
        std::cout << "Heuristic was inflated by a factor of " << (1 + 0.01) << std::endl;
    }

    using std::placeholders::_1;
    Heuristic heuristic = std::bind( &ShortestPathHeuristic::operator(), sp_heuristic, _1);

    // Sorting edges according to f values
    std::cout << "Start sorting edges by f-values...";
    graph.sort_by_f_values(heuristic);
    std::cout << "Done\n";

    // Populating the regular and super edges adjacency matrices
    std::cout << "Start splitting to regular/super edges adj. matrices...";
    graph.populate_regular_and_super_edges_matrices();

    SolutionSet solutions;
    int num_exp, num_gen;
    auto runtime = std::clock();

    std::unique_ptr<AbstractSolver> solver;
    if (algorithm == "PPA"){
        Pair<double> eps_pair({eps, eps});
        solver = std::make_unique<PPA>(graph, eps_pair, logger);
    }else if (algorithm == "BOA"){
        Pair<double> eps_pair({eps, eps});
        solver = std::make_unique<BOAStar>(graph, eps_pair, logger);
    }else if (algorithm == "NAMOAdr"){
        EPS eps_vec (graph.get_num_of_objectives(), eps);
        solver = std::make_unique<NAMOAdr>(graph, eps_vec, logger);
        // ((ApexSearch*)solver.get())->set_merge_strategy(ms);
    }else if (algorithm == "Apex"){
        EPS eps_vec (graph.get_num_of_objectives(), eps);
        solver = std::make_unique<ApexSearch>(graph, eps_vec, sp_heuristic.cost_to_target, target_bounding, logger, optimal_paths_logger);
        ((ApexSearch*)solver.get())->set_merge_strategy(ms);
    }
    else if (algorithm == "GApex_PE") {
        EPS eps_vec(graph.get_num_of_objectives(), eps);
        solver = std::make_unique<ApexPESearch>(graph, eps_vec, sp_heuristic.cost_to_target, target_bounding, logger, optimal_paths_logger);
        ((ApexPESearch*)solver.get())->set_merge_strategy(ms);
    }
    else{
        std::cerr << "unknown solver name" << std::endl;
        exit(-1);
    }
    auto start =std::clock();
    (*solver)(source, target, heuristic, solutions, time_limit);
    runtime = std::clock() - start;

    std::cout << "Generations count: " << solver->get_num_generation() << std::endl;
    std::cout << "Node expansions count: " << solver->get_num_expansion() << std::endl;
    std::cout << "Number of solutions: " << solutions.size() << std::endl;
    std::cout << "Runtime [sec]: " <<  ((double) runtime) / CLOCKS_PER_SEC<< std::endl;
    num_exp = solver->get_num_expansion();
    num_gen = solver->get_num_generation();
    for (auto sol: solutions){
        std::cout << *sol << std::endl;
    }


    output << algorithm << "-" << alg_variant << " (" << eps << ")" << "\t"
           << source << "\t" << target << "\t"
           << num_gen << "\t"
           << num_exp << "\t"
           << solutions.size() << "\t"
           << (double) runtime / CLOCKS_PER_SEC
           << std::endl;

    std::cout << "-----End Single Example-----" << std::endl;
}

void single_run_map(size_t graph_size, std::vector<Edge> & edges, size_t source, size_t target, std::string output_file, std::string algorithm, MergeStrategy ms, 
    LoggerPtr logger, double eps, int time_limit, bool target_bounding, bool inflate_h, LoggerPtr optimal_paths_logger)
{
    AdjacencyMatrix graph(graph_size, edges);
    AdjacencyMatrix inv_graph(graph_size, edges, true);
    std::ofstream stats;
    stats.open(output_path + output_file, std::fstream::app);

    single_run_map(graph_size, graph, inv_graph, source, target, 
        stats, algorithm, ms, logger, eps, time_limit, target_bounding, inflate_h, optimal_paths_logger);
 }

void run_query(size_t graph_size, std::vector<Edge> & edges, std::string query_file, std::string output_file, std::string algorithm, MergeStrategy ms, LoggerPtr logger, 
    double eps, int time_limit, bool inflate_h, bool target_bounding) {
    std::ofstream stats;
    stats.open(output_path + output_file, std::fstream::app);


    std::vector<std::pair<size_t, size_t>> queries;
    if (load_queries(query_file, queries) == false) {
        std::cout << "Failed to load queries file" << std::endl;
        return;
    }

    // Build graphs
    AdjacencyMatrix graph(graph_size, edges);
    AdjacencyMatrix inv_graph(graph_size, edges, true);

    size_t query_count = 0;
    for (auto iter = queries.begin(); iter != queries.end(); ++iter) {

        query_count++;
        std::cout << "Started Query: " << query_count << "/" << queries.size() << std::endl;
        size_t source = iter->first;
        size_t target = iter->second;

        single_run_map(graph_size, graph, inv_graph, source, target, stats, algorithm, ms, 
            logger, eps, time_limit, target_bounding, inflate_h, nullptr);
    }

}

int main(int argc, char** argv){
    namespace po = boost::program_options;

    std::vector<string> objective_files;

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("start,s", po::value<int>()->default_value(-1), "start location")
        ("goal,g", po::value<int>()->default_value(-1), "goal location")
        ("query,q", po::value<std::string>()->default_value(""), "query file")
        ("map,m",po::value< std::vector<string> >(&objective_files)->multitoken(), "files for edge weight")
        ("eps,e", po::value<double>()->default_value(0), "approximation factor")
        ("merge", po::value<std::string>()->default_value(""), "strategy for merging apex node pair: SMALLER_G2, RANDOM or MORE_SLACK")
        ("algorithm,a", po::value<std::string>()->default_value("Apex"), "solvers (BOA, PPA or Apex search)")
        ("inflate_h,h", po::value<bool>()->default_value(false), "flag (true/false) whether to inflate the heuristics by the given epsilon")
        ("target_bounding,b", po::value<bool>()->default_value(false), "Target cost bounding (true/false)")
        ("cutoffTime,t", po::value<int>()->default_value(300), "cutoff time (seconds)")
        ("output,o", po::value<std::string>()->required(), "Name of the output file")
        ("logging_file,l", po::value<std::string>()->default_value(""), "logging file" )
        ("optimal_paths_log,p", po::value<std::string>()->default_value(""), "optimal paths log file")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    po::notify(vm);
    srand((int)time(0));

    if (vm["query"].as<std::string>() != ""){
        if (vm["start"].as<int>() != -1 || vm["goal"].as<int>() != -1){
            std::cerr << "query file and start/goal cannot be given at the same time !" << std::endl;
            return -1;
        }
    }
    
    LoggerPtr logger = nullptr;
    LoggerPtr optimal_paths_logger = nullptr;


    if (vm["logging_file"].as<std::string>() != ""){
        logger = new Logger(vm["logging_file"].as<std::string>());
    }
    if (vm["optimal_paths_log"].as<std::string>() != "") {
        optimal_paths_logger = new Logger(vm["optimal_paths_log"].as<std::string>());
    }
    
    // Load files
    size_t graph_size;
    std::vector<Edge> edges;

    for (auto file:objective_files){
        std::cout << file << std::endl;
    }


    if (load_gr_files(objective_files, edges, graph_size) == false) {
        std::cout << "Failed to load gr files" << std::endl;
        return -1;
    }

    std::cout << "Graph Size: " << graph_size << std::endl;

    // Build graphs
    MergeStrategy ms = DEFAULT_MERGE_STRATEGY;
    alg_variant = vm["merge"].as<std::string>();

    if (vm["merge"].as<std::string>() != "" && vm["algorithm"].as<std::string>()!= "Apex"){
        alg_variant = "";
        std::cout << "WARNING: merge strategy with non-apex search" << std::endl;
    }else if(vm["merge"].as<std::string>() == "SMALLER_G2"){
        ms = MergeStrategy::SMALLER_G2;
    }else if(vm["merge"].as<std::string>() == "SMALLER_G2_FIRST"){
        ms = MergeStrategy::SMALLER_G2_FIRST;
    }else if(vm["merge"].as<std::string>() == "RANDOM"){
        ms = MergeStrategy::RANDOM;
    }else if(vm["merge"].as<std::string>() == "MORE_SLACK"){
        ms = MergeStrategy::MORE_SLACK;
    }else if(vm["merge"].as<std::string>() == "REVERSE_LEX"){
        ms = MergeStrategy::REVERSE_LEX;
    }else{
        std::cerr << "unknown merge strategy" << std::endl;
    }

    // Determining command lind flags
    bool inflate_h = vm["inflate_h"].as<bool>();
    bool target_bounding = vm["target_bounding"].as<bool>();

    if (vm["query"].as<std::string>() != ""){
        run_query(graph_size, edges, vm["query"].as<std::string>(), vm["output"].as<std::string>(), vm["algorithm"].as<std::string>(), ms, logger, vm["eps"].as<double>(), 
            vm["cutoffTime"].as<int>(), inflate_h, target_bounding);
    } else{
        single_run_map(graph_size, edges, vm["start"].as<int>(), vm["goal"].as<int>(), vm["output"].as<std::string>(), vm["algorithm"].as<std::string>(), ms, logger, 
            vm["eps"].as<double>(), vm["cutoffTime"].as<int>(), target_bounding, inflate_h, optimal_paths_logger);
    }

    delete(logger);

    return 0;
}
