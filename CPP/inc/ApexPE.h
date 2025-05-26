#pragma once
#include "Utils/Definitions.h"
#include "Utils/Logger.h"
#include "Utils/MapQueue.h"
#include"DominanceChecker.h"
#include "AbstractSolver.h"
#include <unordered_map>

class ApexPESearch: public AbstractSolver {
protected:
    std::unordered_map<int, std::vector<CostsVector>> cost_to_target;
    size_t num_of_objectives;
    MergeStrategy ms=MergeStrategy::SMALLER_G2;
    bool target_bounding_enabled = false;

    std::unique_ptr<DominanceChecker> local_dom_checker;
    std::unique_ptr<DominanceChecker> solution_dom_checker;

    //std::vector<std::list<ApexPathPairPtr>>   closed_map;

    virtual void insert(ApexPathPairPtr &pp, APQueue &queue);
    bool is_dominated(ApexPathPairPtr ap);
    void merge_to_solutions(const ApexPathPairPtr &pp, ApexPathSolutionSet &solutions);
    std::vector<std::vector<ApexPathPairPtr>> expanded;
    void init_search();

    bool last_ap_inserted = false;

public:
    long int elapsed_time_ms = 0;
    long int dom_check_counter = 0;
    long int dom_check_local = 0;
    long int dom_check_solution = 0;

    virtual std::string get_solver_name();

    void set_merge_strategy(MergeStrategy new_ms){ms = new_ms;}
    ApexPESearch(const AdjacencyMatrix &adj_matrix, EPS eps, 
        std::unordered_map<int, std::vector<CostsVector>>& cost_to_target,
        bool target_bounding,
        const LoggerPtr logger=nullptr,
        const LoggerPtr optimal_paths_logger = nullptr);
    virtual void operator()(size_t source, size_t target, Heuristic &heuristic, SolutionSet &solutions, unsigned int time_limit=UINT_MAX) override;

    double apex_creation_total_time;
    size_t heap_insert_count = 0;

    bool verbose = false;
};

