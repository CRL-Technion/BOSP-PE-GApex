#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include "ApexSearch.h"


ApexSearch::ApexSearch(const AdjacencyMatrix &adj_matrix, EPS eps, 
    std::unordered_map<int, std::vector<CostsVector>>& _cost_to_target,
    bool target_bounding, const LoggerPtr logger, const LoggerPtr optimal_paths_logger) :
    AbstractSolver(adj_matrix, eps, logger, optimal_paths_logger),
    num_of_objectives(adj_matrix.get_num_of_objectives())
{
    expanded.resize(this->adj_matrix.size()+1);
    cost_to_target = _cost_to_target;
    target_bounding_enabled = target_bounding;
}

void ApexSearch::insert(ApexPathPairPtr &ap, APQueue &queue) {
    std::list<ApexPathPairPtr> &relevant_aps = queue.get_open(ap->id);
    for (auto existing_ap = relevant_aps.begin(); existing_ap != relevant_aps.end(); ++existing_ap) 
    {
        if ((*existing_ap)->is_active == false) 
        {
            relevant_aps.erase(existing_ap);
            continue;
        }

        if (ap->update_nodes_by_merge_if_bounded(*existing_ap, this->eps, ms) == true) 
        {
            // pp and existing_pp were merged successfuly into pp
            // std::cout << "merge!" << std::endl;
            if ((ap-> apex!= (*existing_ap)->apex) ||
                (ap-> path_node!= (*existing_ap)->path_node)) {
                // If merged_pp == existing_pp we avoid inserting it to keep the queue as small as possible.
                // existing_pp is deactivated and not removed to avoid searching through the heap
                // (it will be removed on pop and ignored)
                (*existing_ap)->is_active = false;
                relevant_aps.erase(existing_ap);
                queue.insert(ap);
            }
            // both apex and path_node are equal -> ap is dominated
            return;
        }
    }
    queue.insert(ap);
}


void ApexSearch::merge_to_solutions(const ApexPathPairPtr &ap, ApexPathSolutionSet &solutions) {
    for (auto existing_solution = solutions.begin(); existing_solution != solutions.end(); ++existing_solution) {
        if ((*existing_solution)->update_nodes_by_merge_if_bounded(ap, this->eps, ms) == true) {
            return;
        }
    }
    solutions.push_back(ap);
    // std::cout << "update solution checker" << std::endl;
    solution_dom_checker->add_node(ap);
}


bool ApexSearch::is_dominated(ApexPathPairPtr ap)
{
    dom_check_counter++; // yaron
    if (local_dom_checker->is_dominated(ap))
    {
        dom_check_local++;
        return true;
    }

    if (solution_dom_checker->is_dominated(ap))
    {
        dom_check_solution++;
        return true;
    }

    return false;
}


void ApexSearch::operator()(size_t source, size_t target, Heuristic &heuristic, SolutionSet &solutions, unsigned int time_limit) 
{

    init_search();

    auto start_time = std::clock();
    if (num_of_objectives == 2){
        local_dom_checker = std::make_unique<LocalCheck>(eps, this->adj_matrix.size());
        solution_dom_checker = std::make_unique<SolutionCheck>(eps);
    }else{
        local_dom_checker = std::make_unique<LocalCheckLinear>(eps, this->adj_matrix.size());
        solution_dom_checker = std::make_unique<SolutionCheckLinear>(eps);
    }

    target_bounds_dom_checker = std::make_unique<TargetBoundsChecker>(eps, num_of_objectives);

    this->start_logging(source, target);

    ApexPathSolutionSet ap_solutions;
    ApexPathPairPtr   ap;
    ApexPathPairPtr   next_ap;

    // Saving all the unused PathPairPtrs in a vector improves performace for some reason
    // std::vector<ApexPathPairPtr> closed;

    // Vector to hold mininum cost of 2nd criteria per node
    // std::vector<size_t> min_g2(this->adj_matrix.size()+1, MAX_COST);
    
    // Init open heap
    APQueue open(this->adj_matrix.size()+1);
    NodePtr source_node = std::make_shared<Node>(source, std::vector<size_t>(num_of_objectives, 0), heuristic(source));
    ap = std::make_shared<ApexPathPair>(source_node, source_node, heuristic, nullptr);
    open.insert(ap);

    std::vector<size_t> temp_cost_vec(2, 0);
    std::vector<size_t> temp_h(2, 0);
    auto search_start_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> meas_start;

    while (open.empty() == false) {
        if ((std::clock() - start_time)/CLOCKS_PER_SEC > time_limit){
            for (auto solution = ap_solutions.begin(); solution != ap_solutions.end(); ++solution) {
                solutions.push_back((*solution)->path_node);
            }

            this->end_logging(solutions, false);
            return;
        }
        
        // Pop min from queue and process
        ap = open.pop();
        num_generation +=1;  

        // Optimization: PathPairs are being deactivated instead of being removed so we skip them.
        if (ap->is_active == false) 
        {
            num_inactive_pop += 1;
            continue;
        }
        
        auto meas_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(meas_end - meas_start);
        apex_creation_total_time += std::chrono::duration_cast<std::chrono::microseconds>(duration).count() * 1e-6;

        // Dominance check
        if (is_dominated(ap)){
            continue;
        }

        //  min_g2[ap->id] = ap->bottom_right->g[1];
        local_dom_checker->add_node(ap);

        num_expansion += 1;

        //expanded[ap->id].push_back(ap);

        if (ap->id == target) 
        {
            if(this->logger->first_solution_expansion_count < 0)
            { 
                this->logger->first_solution_expansion_count = num_expansion;
            }
            this->merge_to_solutions(ap, ap_solutions);
            continue;
        }

        // Check to which neighbors we should extend the paths
        const std::vector<Edge> &outgoing_edges = adj_matrix[ap->id];        

        for(auto p_edge = outgoing_edges.begin(); p_edge != outgoing_edges.end(); p_edge++) 
        {
            Edge edge = (*p_edge);
            // Local domination check
            for (int k = 0; k < num_of_objectives; k++)
            {
                temp_cost_vec[k] = ap->apex->g[k] + edge.cost[k];
            }
            if (local_dom_checker->is_dominated_lite(edge.target, temp_cost_vec))
            {
                continue;
            }
            // Solution domination check
            temp_h = heuristic(edge.target);
            for (int k = 0; k < num_of_objectives; k++)
            {
                temp_cost_vec[k] += temp_h[k];
            }
            if (solution_dom_checker->is_dominated_lite(edge.target, temp_cost_vec))
            {
                continue;
            }
            
            // Prepare extension of path pair
            next_ap = std::make_shared<ApexPathPair>(ap, *p_edge);

            // If not dominated extend path pair and push to queue
            // Creation is defered after dominance check as it is
            // relatively computational heavy and should be avoided if possible
            this->insert(next_ap, open);
        }
    }
    auto search_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(search_end_time - search_start_time);
    long int total_runtime_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

    this->logger->total_runtime_ms = total_runtime_ms;
    std::cout << "Search Runtime [sec]: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Number of inactive nodes cycles: " << this->num_inactive_pop << std::endl;
    std::cout << "Total spent in apex creation: " << this->apex_creation_total_time <<
        " [seconds] Percentage of total time = " << this->apex_creation_total_time / duration.count()/1e-6 << std::endl;
    std::cout << "First solution @ expansion count = " << this->logger->first_solution_expansion_count << std::endl;
    std::cout << "Open Heap max size = " << open.get_max_size() << std::endl;

    // Pair solutions is used only for logging, as we need both the solutions for testing reasons
    for (auto solution = ap_solutions.begin(); solution != ap_solutions.end(); ++solution) {
        solutions.push_back((*solution)->path_node);

    }

    // Adding number of expansions and generations to the JSON log file
    this->end_logging(solutions, true, this->get_num_expansion(), this->get_num_generation());
}

std::string ApexSearch::get_solver_name() 
{
    std::string alg_variant;
    if (ms == MergeStrategy::SMALLER_G2){
        alg_variant ="-s2";
    } else if ( ms == MergeStrategy::SMALLER_G2_FIRST){
        alg_variant ="-s2f";
    } else if (ms == MergeStrategy::RANDOM){
        alg_variant ="-r";
    } else if (ms == MergeStrategy::MORE_SLACK){
        alg_variant ="-ms";
    } else if (ms == MergeStrategy::REVERSE_LEX){
        alg_variant ="-rl";
    }
    return "Apex" + alg_variant;
}

void ApexSearch::init_search(){
    AbstractSolver::init_search();
    expanded.clear();
    expanded.resize(this->adj_matrix.size()+1);
}
