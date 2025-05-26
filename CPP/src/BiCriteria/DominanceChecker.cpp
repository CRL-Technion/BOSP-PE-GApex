#include "DominanceChecker.h"


bool SolutionCheck::is_dominated(ApexPathPairPtr node){
    if (last_solution == nullptr){
        return false;
    }
    if (is_bounded(node->apex, last_solution->path_node, eps)){
        assert(last_solution->update_apex_by_merge_if_bounded(node->apex, eps));
        // std::cout << "solution dom" << std::endl;
        return true;
    }
    return false;
}

bool SolutionCheck::is_dominated_lite(size_t id, std::vector<size_t> cost_vec){
    if (last_solution == nullptr) {
        return false;
    }
    auto node = last_solution->path_node;
    for (int i = 0; i < cost_vec.size(); i++) {
        if (node->f[i] > (1 + eps[i]) * cost_vec[i]) {
            return false;
        }
    }
    return true;
}

bool SolutionCheckLinear::is_dominated_lite(size_t id, std::vector<size_t> cost_vec) 
{
    std::cout << "SolutionCheckLinear::is_dominated_lite not implemented!\n";
    return false;
}

bool LocalCheck::is_dominated(ApexPathPairPtr node){
    return (node->apex->g[1] >= min_g2[node->id]);
}

bool LocalCheck::is_dominated_lite(size_t id, std::vector<size_t> cost_vec){
    return (cost_vec[1] >= min_g2[id]);
}

void LocalCheck::add_node(ApexPathPairPtr ap){
    auto id = ap->id;
    assert(min_g2[ap->id] > ap->apex->g[1]);
    if (!(min_g2[ap->id] > ap->apex->g[1]))
    {
        std::cout << "assert problem!\n";
    }
    min_g2[ap->id] = ap->apex->g[1];
}

bool LocalCheckLinear::is_dominated(ApexPathPairPtr node){
    for (auto ap:min_g2[node->id]){
        if (is_dominated_dr(node->apex, ap->apex)){
            assert(node->apex->f[0] >= ap->apex->f[0]);
            return true;
        }
    }
    return false;
}

void LocalCheckLinear::add_node(ApexPathPairPtr ap){
    auto id = ap->id;
    for (auto it = min_g2[id].begin(); it != min_g2[id].end(); ){
        // TODO remove it for performance
        assert(! is_dominated_dr(ap->apex, (*it)->apex  ));
        if (is_dominated_dr((*it)->apex, ap->apex)){
            it = min_g2[id].erase(it);
        } else {
            it ++;
        }
    }

    min_g2[ap->id].push_front(ap);
}

bool LocalCheckLinear::is_dominated_lite(size_t id, std::vector<size_t> cost_vec)
{
    std::cout << "LocalCheckLinear::is_dominated_lite() not implemented!\n";
    return false;
}

bool SolutionCheckLinear::is_dominated(ApexPathPairPtr node){
    for (auto ap: solutions){
        // if (is_bounded(node->apex, ap->path_node, eps)){
        if (ap->update_apex_by_merge_if_bounded(node->apex, eps)){
            // assert(ap->update_apex_by_merge_if_bounded(node->apex, eps));
            return true;
        }
    }
    return false;
}

void SolutionCheckLinear::add_node(ApexPathPairPtr ap){
    for (auto it = solutions.begin(); it != solutions.end(); ){
        if (is_dominated_dr((*it)->path_node, ap->path_node)){
            it = solutions.erase(it);
        } else {
            it ++;
        }
    }
    solutions.push_front(ap);
}


bool TargetBoundsChecker::is_A_dominated_by_B(CostsVector A, CostsVector B)
{
    int dom_counter = 0;
    for (int i = 0; i < objectives_count; i++)
    {
        if (B[i] < (1.0 + 0.0 * eps[i]) * A[i])
        {
            dom_counter++;
        }
    }
    return ( dom_counter == objectives_count);
}

bool TargetBoundsChecker::is_dominated(ApexPathPairPtr node)
{    
    if (bounds.size() > 0)
    {
        num_of_calls++;
        for (int i = 0; i < objectives_count; i++)
        {
            cost_to_check[i] = node->apex->f[i];
        }
        for (auto cost : bounds)
        {
            if (is_A_dominated_by_B(cost_to_check, cost))
            {
                num_of_prunes++;
                return true;
            }
        }
        return false;
    }
    
    return false;
}

void TargetBoundsChecker::add_node(CostsVector new_cost) 
{
    for (auto it = bounds.begin(); it != bounds.end(); ) 
    {
        if(is_A_dominated_by_B((*it), new_cost)) 
        {
            it = bounds.erase(it);
        }
        else 
        {
            it++;
        }
    }
    bounds.push_front(new_cost);
}

bool LocalCheckFull::is_dominated(ApexPathPairPtr node)
{
    for (auto ap : existing_aps[node->id]) {
        if (is_dominated_full(node->apex, ap->apex)) 
        {
            return true;
        }
    }
    return false;
}

void LocalCheckFull::add_node(ApexPathPairPtr ap)
{
    auto id = ap->id;

    for (auto it = existing_aps[id].begin(); it != existing_aps[id].end(); ) 
    {
        if (is_dominated_full((*it)->apex, ap->apex)) 
        {
            it = existing_aps[id].erase(it);
        }
        else 
        {
            it++;
        }
    }

    existing_aps[ap->id].push_front(ap);
}

bool LocalCheckFull::is_dominated_lite(size_t id, std::vector<size_t> cost_vec)
{
    return false;
}
