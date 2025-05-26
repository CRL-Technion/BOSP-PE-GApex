#pragma once

#include "Utils/Definitions.h"
#include "Utils/Logger.h"

class AbstractSolver {
protected:
    const AdjacencyMatrix   &adj_matrix;
    // Pair<double>            eps;
    EPS eps;

    size_t num_expansion = 0;
    size_t num_generation= 0;
    size_t num_inactive_pop = 0;

    virtual void init_search(){
        num_expansion = 0;
        num_generation = 0;
    }

    const LoggerPtr         logger;
    const LoggerPtr         optimal_paths_logger;
    void start_logging(size_t source, size_t target);
    void end_logging(SolutionSet &solutions, bool succ=true);
    void end_logging(SolutionSet& solutions, bool succ, int n_expansions, int n_generations);
    void log_optimal_paths(ApexPathSolutionSet& solutions, std::string filename);

public:
    virtual std::string get_solver_name() = 0;

    size_t get_num_expansion(){return num_expansion;}
    size_t get_num_generation(){return num_generation;}

    virtual void operator()(size_t source, size_t target, Heuristic &heuristic, SolutionSet &solutions, unsigned int time_limit=UINT_MAX) = 0;


    AbstractSolver(const AdjacencyMatrix &adj_matrix, EPS eps, 
        const LoggerPtr logger, const LoggerPtr optimal_paths_logger): adj_matrix(adj_matrix),
        eps(eps), logger(logger), optimal_paths_logger(optimal_paths_logger) {}
    AbstractSolver(const AdjacencyMatrix& adj_matrix, EPS eps,
        const LoggerPtr logger) : adj_matrix(adj_matrix),
        eps(eps), logger(logger), optimal_paths_logger(nullptr) {}
    virtual ~AbstractSolver(){}
};
