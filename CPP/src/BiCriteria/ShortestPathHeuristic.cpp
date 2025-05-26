#include <limits>
#include <memory>
#include <algorithm>

#include "ShortestPathHeuristic.h"


ShortestPathHeuristic::ShortestPathHeuristic(size_t source, size_t graph_size, const AdjacencyMatrix &adj_matrix)
    : source(source), all_nodes(graph_size+1, nullptr) {
    size_t num_of_objectives = adj_matrix.get_num_of_objectives();
    size_t i = 0;
    for (auto node_iter = this->all_nodes.begin(); node_iter != this->all_nodes.end(); node_iter++)
    {
        *node_iter = std::make_shared<Node>(i++, std::vector<size_t>(num_of_objectives, 0), 
            std::vector<size_t>(num_of_objectives, MAX_COST));
    }

    for (int j=0; j < num_of_objectives; j ++)
    {
        // Initialize the cross-objective (different than j)
        /*
        for (int k = 0; k < num_of_objectives; k++)
        {
            if (j != k)
            {
                for (auto node_iter = this->all_nodes.begin(); node_iter != this->all_nodes.end(); node_iter++)
                {
                    (*node_iter)->g[k] = 0;
                }
            }
        }
        */
        // Computing the shortest-paths in respect to objective #j
        compute(j, adj_matrix);
    }
}


std::vector<size_t> ShortestPathHeuristic::operator()(size_t node_id) {
    return this->all_nodes[node_id]->h;
}


// Implements Dijkstra shortest path algorithm per cost_idx cost function
void ShortestPathHeuristic::compute(size_t cost_idx, const AdjacencyMatrix &adj_matrix) 
{
    int objectives_count = this->all_nodes[this->source]->h.size();

    // Init all heuristics to MAX_COST
    for (auto node_iter = this->all_nodes.begin(); node_iter != this->all_nodes.end(); node_iter++) 
    {
        (*node_iter)->h[cost_idx] = MAX_COST;
        (*node_iter)->parent = nullptr; // Initialize predecessor pointers

        /*
        // Reseting the cross-objective temp costs
        for (int cross_obj_ind = 0; cross_obj_ind < objectives_count; cross_obj_ind++)
        {
            if (cross_obj_ind != cost_idx)
            {
                (*node_iter)->temp_h[cross_obj_ind] = 0;
            }
        }
        */
    }

    NodePtr node;
    NodePtr next;

    // Init open heap
    Node::more_than_specific_heurisitic_cost more_than(cost_idx);
    std::vector<NodePtr> open;
    std::make_heap(open.begin(), open.end(), more_than);

    this->all_nodes[this->source]->h[cost_idx] = 0;
    open.push_back(this->all_nodes[this->source]);
    std::push_heap(open.begin(), open.end(), more_than);

    

    while (open.empty() == false) {
        // Pop min from queue and process
        std::pop_heap(open.begin(), open.end(), more_than);
        node = open.back();
        open.pop_back();

        // Check to which neighbors we should extend the paths
        const std::vector<Edge> &outgoing_edges = adj_matrix[node->id];
        for(auto p_edge = outgoing_edges.begin(); p_edge != outgoing_edges.end(); p_edge++) {
            next = this->all_nodes[p_edge->target];

            // Dominance check
            if (next->h[cost_idx] <= (node->h[cost_idx]+p_edge->cost[cost_idx])) {
                continue;
            }

            // If not dominated push to queue
            next->h[cost_idx] = node->h[cost_idx] + p_edge->cost[cost_idx];
            next->parent = node; // Update predecessor
            
            /*
            // Updating the cross-objective costs
            for (int i = 0; i < objectives_count; i++)
            {
                if( i != cost_idx)
                {
                    next->temp_h[i] = node->temp_h[i] + p_edge->cost[i];
                }
            }
            */
            
            open.push_back(next);
            std::push_heap(open.begin(), open.end(), more_than);
        }
    }
    
    // After search is completed, computing the cost-to-target bounds
    /*
    for (auto node_iter = this->all_nodes.begin(); node_iter != this->all_nodes.end(); node_iter++) 
    {
        //std::vector<size_t> node_cost_to_target = std::vector<size_t>(objectives_count, 0);
        CostsVector node_cost_to_target = CostsVector(objectives_count, 0);
        NodePtr current_node = *node_iter;
        size_t node_id = current_node->id;
        
        // Asserting that target is reachable from this node
        if (current_node->h[cost_idx] == MAX_COST)
        {
            continue;
        }
        // Copying the cost of the objective function that was currently considered in the above Dijkstra
        node_cost_to_target[cost_idx] = current_node->h[cost_idx];

        // Updating the cross-objective costs
        for (int cross_obj_ind = 0; cross_obj_ind < objectives_count; cross_obj_ind++)
        {
            if (cross_obj_ind != cost_idx)
            {
                node_cost_to_target[cross_obj_ind] = current_node->temp_h[cross_obj_ind];
            }
        }
        
        /*
        // Reconstructing the optimal path to compute the cost of the cross-objective functions costs
        while (true) 
        {
            if (current_node->parent == nullptr)
            {
                break;
            }
            NodePtr prev_node = current_node->parent;

            for (int cost_id = 0; cost_id < objectives_count; cost_id++)
            {
                if(cost_idx != cost_id)
                { 
                    node_cost_to_target[cost_id] += (current_node->temp_h[cost_id] - prev_node->temp_h[cost_id]);
                }
            }
            current_node = current_node->parent;
        }
        

        // Checking if this is the first update of the ordered_map
        auto it = cost_to_target.find(node_id);
        if (it != cost_to_target.end()) 
        {
            // Key found, adding new target bound derived from the currently optimized objective
            cost_to_target[node_id].push_back(node_cost_to_target);
        }
        else 
        {
            // Key not found, creating the initial vector (first objective function)
            std::vector<CostsVector> target_bounds;
            target_bounds.push_back(node_cost_to_target);
            cost_to_target[node_id] = target_bounds;
        }
    }
    */
}
