#ifndef UTILS_DEFINITIONS_H
#define UTILS_DEFINITIONS_H

#include <map>
#include <vector>
#include <array>
#include <list>
#include <iostream>
#include <limits>
#include <functional>
#include <memory>
#include <climits>
#include "boost/heap/pairing_heap.hpp"
#include "boost/heap/priority_queue.hpp"


#ifndef DEBUG
#define DEBUG 1
#endif
typedef std::vector<size_t> CostsVector;

//static double apex_creation_total_time;
//double get_apex_creation_time();
//void reset_apex_creation_time();


const size_t MAX_COST = std::numeric_limits<size_t>::max();

template<typename T>
using Pair      = std::array<T, 2>;

template<typename T>
std::ostream& operator<<(std::ostream &stream, const Pair<T> pair) {
    stream << "[" << pair[0] << ", " << pair[1] << "]";
    return stream;
}

using Heuristic = std::function<std::vector<size_t>(size_t)>;

enum EdgeTypeEnum
{
    REGULAR_EDGE,
    SUPER_EDGE
};

// Structs and classes
struct Edge {
    size_t          source;
    size_t          target;
    std::vector<size_t>    cost;
    std::vector<size_t>    edge_apex;
    EdgeTypeEnum edge_type = REGULAR_EDGE;

    Edge(size_t source, size_t target, std::vector<size_t> cost, EdgeTypeEnum edge_type = REGULAR_EDGE) :
        source(source), target(target), cost(cost), edge_type(edge_type), edge_apex(cost) {}

    Edge(size_t source, size_t target, std::vector<size_t> cost, EdgeTypeEnum edge_type, std::vector<size_t> edge_apex) :
        source(source), target(target), cost(cost), edge_apex(edge_apex), edge_type(edge_type)
    {
        assert(edge_type == SUPER_EDGE);
    }

    Edge inverse() {
        return Edge(this->target, this->source, this->cost, this->edge_type);
    }

    // Operator< to allow sorting by cost vector
    bool operator<(const Edge& other) const {
        return this->cost < other.cost;
    }
};
std::ostream& operator<<(std::ostream &stream, const Edge &edge);


// Graph representation as adjacency matrix
class AdjacencyMatrix {
private:
    std::vector<std::vector<Edge>> matrix;
    size_t                         graph_size;
    size_t num_of_objectives = 0;
    std::vector<bool> boundary_vertices;

    // Spliting the adjacency matrix for regular and super edges
    std::vector<std::vector<Edge>> regular_edges_matrix;
    std::vector<std::vector<Edge>> super_edges_matrix;
        
public:
    AdjacencyMatrix() = default;
    AdjacencyMatrix(size_t graph_size, std::vector<Edge> &edges, bool inverse=false);
    void add(Edge edge);
    void sort_by_f_values(Heuristic& heuristic);
    void populate_regular_and_super_edges_matrices();
    size_t size(void) const;
    size_t get_num_of_objectives() const;
    const std::vector<Edge>& operator[](size_t vertex_id) const;
    const bool is_boundary_vertex(size_t vertex_id) const;
    const std::vector<Edge>& get_regular_edges(size_t vertex_id) const;
    const std::vector<Edge>& get_super_edges(size_t vertex_id) const;
  
    friend std::ostream& operator<<(std::ostream &stream, const AdjacencyMatrix &adj_matrix);
};


struct Node;
struct PathPair;
struct ApexPathPair;
using NodePtr       = std::shared_ptr<Node>;
using PathPairPtr   = std::shared_ptr<PathPair>;
using ApexPathPairPtr   = std::shared_ptr<ApexPathPair>;
using SolutionSet   = std::vector<NodePtr>;
using PPSolutionSet = std::vector<PathPairPtr>;
using ApexPathSolutionSet = std::vector<ApexPathPairPtr>;

using EPS = std::vector<double>;

struct Node {
    size_t          id;
    std::vector<size_t>    g;
    std::vector<size_t>    h;
    std::vector<size_t>    f;
    NodePtr         parent;

    // yaron 21-04-2024
    //std::vector<size_t>    temp_h;    
    
    Node(size_t id, std::vector<size_t> g, std::vector<size_t> h, NodePtr parent = nullptr)
        : id(id), g(g), h(h), f(g.size()), parent(parent)
    {
        for (int i = 0; i < g.size(); i++) {
            f[i] = g[i] + h[i];
        }

        // yaron
        //temp_h = std::vector<size_t>(g.size(), 0);
    };

    struct more_than_specific_heurisitic_cost {
        size_t cost_idx;

        more_than_specific_heurisitic_cost(size_t cost_idx) : cost_idx(cost_idx) {};
        bool operator()(const NodePtr &a, const NodePtr &b) const;
    };

    struct more_than_combined_heurisitic {
        double factor;

        more_than_combined_heurisitic(double factor) : factor(factor) {};
        bool operator()(const NodePtr &a, const NodePtr &b) const;
    };


    struct more_than_full_cost {
        bool operator()(const NodePtr &a, const NodePtr &b) const;
    };

    enum LEX_ORDER {LEX0, LEX1};
    struct more_than_lex{
        Node::LEX_ORDER order;
        more_than_lex(Node::LEX_ORDER order) : order(order) {};
        bool operator()(const NodePtr &a, const NodePtr &b) const;
    };


    struct compare_lex1
    {
        bool operator()(const NodePtr n1, const NodePtr n2) const
        {
            if (n1->f[0] != n2->f[0]){
                return n1->f[0] > n2->f[0];
            }
            return n1->f[1] > n2->f[1];
        }
    };
    friend std::ostream& operator<<(std::ostream &stream, Node &node);
};

struct PathPair {
    size_t      id;
    NodePtr     top_left;
    NodePtr     bottom_right;
    NodePtr     parent;
    bool        is_active=true;

    PathPair(const NodePtr &top_left, const NodePtr &bottom_right)
        : id(top_left->id), top_left(top_left), bottom_right(bottom_right), parent(top_left->parent) {};

    bool update_nodes_by_merge_if_bounded(const PathPairPtr &other, const Pair<double> eps);
    bool update_nodes_by_merge_if_bounded_keep_track(const PathPairPtr &other, const Pair<double> eps, std::list<NodePtr>& pruned_list);
    bool update_nodes_by_merge_if_bounded2(const PathPairPtr &other, const Pair<double> eps);

    bool if_merge_bounded(const PathPairPtr &other, const Pair<double> eps)  const;


    struct more_than_full_cost {
 bool operator()(const PathPairPtr &a, const PathPairPtr &b) const;
    };

    friend std::ostream& operator<<(std::ostream &stream, const PathPair &pp);
};

enum MergeStrategy {SMALLER_G2, RANDOM, MORE_SLACK, SMALLER_G2_FIRST, REVERSE_LEX};

struct ApexPathPair {
    size_t      id; // state of the node
    NodePtr     apex;
    NodePtr     path_node;
    ApexPathPairPtr     parent;
    bool        is_active=true;   

    size_t out_regular_edge_ind = 0;
    size_t out_super_edge_ind = 0;

    Heuristic& h;

    ApexPathPair(const NodePtr &apex, const NodePtr &path_node, Heuristic& h, const ApexPathPairPtr &pp_parent)
        : apex(apex), path_node(path_node) , parent(pp_parent), h(h), id(apex->id){};

    ApexPathPair(const ApexPathPairPtr parent, const Edge& egde);


    bool update_nodes_by_merge_if_bounded(const ApexPathPairPtr &other, const EPS eps, MergeStrategy s=MergeStrategy::SMALLER_G2);
    bool update_apex_by_merge_if_bounded(const NodePtr &other_apex, const EPS eps);

    // bool if_merge_bounded(const ApexPathPairPtr &other, const EP S eps)  const;


    struct more_than_full_cost {
        bool operator()(const ApexPathPairPtr &a, const ApexPathPairPtr &b) const;
    };

    friend std::ostream& operator<<(std::ostream &stream, const ApexPathPair &pp);
};

bool is_bounded(NodePtr apex, NodePtr node,  const EPS eps);
bool is_bounded(NodePtr apex, NodePtr node);
bool is_dominated_dr(NodePtr apex, NodePtr node);
bool is_dominated_dr(NodePtr apex, NodePtr node, const EPS eps);
bool is_dominated_full(NodePtr apex, NodePtr node);
bool is_dominated_full(NodePtr apex, NodePtr node, const EPS eps);


class Interval{
public:
    double eps = 0;
    NodePtr top_left;
    NodePtr bottom_right;
    std::shared_ptr<std::list<NodePtr>> to_expand;

    Interval(){};
    Interval(const NodePtr top_left, const NodePtr bottom_right, std::shared_ptr<std::list<NodePtr>> to_expand);
};

std::ostream& operator<<(std::ostream& os, const Interval& interval);


using IntervalList   = std::vector<Interval>;

typedef boost::heap::priority_queue<NodePtr , boost::heap::compare<Node::compare_lex1> > heap_open_t;


#endif //UTILS_DEFINITIONS_H
