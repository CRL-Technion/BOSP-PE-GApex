#pragma once

#include "Utils/Definitions.h"
#include <set>

class DominanceChecker {
protected:
    EPS eps;
public:
    virtual ~DominanceChecker(){};
    DominanceChecker(EPS eps):eps(eps){};

    virtual bool is_dominated(ApexPathPairPtr node) = 0;

    virtual void add_node(ApexPathPairPtr ap) = 0;

    virtual bool is_dominated_lite(size_t id, std::vector<size_t> cost_vec) = 0;
    
};


class SolutionCheck: public DominanceChecker {
    ApexPathPairPtr last_solution = nullptr;
public:

    SolutionCheck(EPS eps):DominanceChecker(eps){};

    virtual bool is_dominated(ApexPathPairPtr node);

    virtual void add_node(ApexPathPairPtr ap){ last_solution = ap;};

    virtual bool is_dominated_lite(size_t id, std::vector<size_t> cost_vec);
};

class SolutionCheckLinear: public DominanceChecker {
    std::list<ApexPathPairPtr> solutions;

public:

    SolutionCheckLinear(EPS eps):DominanceChecker(eps){};

    virtual bool is_dominated(ApexPathPairPtr node);

    virtual void add_node(ApexPathPairPtr ap);

    virtual bool is_dominated_lite(size_t id, std::vector<size_t> cost_vec);
};

class TargetBoundsChecker {
    std::list<CostsVector> bounds;
    EPS eps;
    size_t num_of_calls = 0;
    size_t num_of_prunes = 0;
    CostsVector cost_to_check;
    int objectives_count;
    std::set<size_t> ids;

public:

    TargetBoundsChecker(EPS eps, int num_of_objectives) : eps(eps)
    {
        cost_to_check = std::vector<size_t>(num_of_objectives, 0);
        objectives_count = num_of_objectives;
    };

    bool is_dominated(ApexPathPairPtr cost_to_check);
    void add_node(CostsVector new_cost);

    bool is_A_dominated_by_B(CostsVector A, CostsVector B);
    void print_stats() 
    { 
        if (num_of_calls > 0) {
            std::cout << (double)num_of_prunes / num_of_calls << std::endl;
        }
        else
        {
            std::cout << 0 << std::endl;

        }
    }
};


class LocalCheck: public DominanceChecker {

protected:
    std::vector<size_t> min_g2;

public:

    LocalCheck(EPS eps, size_t graph_size):DominanceChecker(eps), min_g2(graph_size + 1, MAX_COST) {};

    virtual bool is_dominated(ApexPathPairPtr node);

    virtual bool is_dominated_lite(size_t id, std::vector<size_t> cost_vec);

    virtual void add_node(ApexPathPairPtr ap);

};

class LocalCheckLinear: public DominanceChecker {

protected:
    std::vector<std::list<ApexPathPairPtr>> min_g2;

public:

    LocalCheckLinear(EPS eps, size_t graph_size):DominanceChecker(eps), min_g2(graph_size + 1) {};

    virtual bool is_dominated(ApexPathPairPtr node);

    virtual void add_node(ApexPathPairPtr ap);

    virtual bool is_dominated_lite(size_t id, std::vector<size_t> cost_vec);
};

class LocalCheckFull : public DominanceChecker {

protected:
    std::vector<std::list<ApexPathPairPtr>> existing_aps;

public:

    LocalCheckFull(EPS eps, size_t graph_size) :DominanceChecker(eps), existing_aps(graph_size + 1) {};

    virtual bool is_dominated(ApexPathPairPtr node);

    virtual void add_node(ApexPathPairPtr ap);

    virtual bool is_dominated_lite(size_t id, std::vector<size_t> cost_vec);
};


class GCL {

protected:
    std::vector<std::list<NodePtr>> gcl;

public:

    GCL(size_t graph_size):gcl(graph_size + 1) {};

    virtual bool is_dominated(NodePtr node);

    virtual void add_node(NodePtr node);
};


