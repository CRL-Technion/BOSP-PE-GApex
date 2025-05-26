#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <unordered_map>



struct Edge {
    size_t          source;
    size_t          target;
    //std::vector<size_t>    cost;
    std::vector<double>    cost;
   

    Edge(size_t source, size_t target, std::vector<double> cost) : 
        source(source), target(target), cost(cost) 
    {}
    Edge inverse() {
        return Edge(this->target, this->source, this->cost);
    }
};

struct ContractedEdge
{
    int source;
    int target;
    std::vector<double> edge_cost;
    std::vector<double> edge_apex;
};

typedef std::vector<std::vector<int>> ClusterMapping;
typedef std::vector<std::vector<double>> CrossObjectiveCost;
typedef std::vector<std::vector<CrossObjectiveCost>> AllPairsCostsTensor;

// Graph representation as adjacency matrix
class AdjacencyMatrix {
private:
    std::vector<std::vector<Edge>> matrix;
    size_t                         graph_size;
    size_t num_of_objectives = 0;

public:
    AdjacencyMatrix() = default;
    AdjacencyMatrix(size_t graph_size, std::vector<Edge>& edges, bool inverse = false);
    void add(Edge edge);
    size_t size(void) const;
    size_t get_num_of_objectives() const;
    const std::vector<Edge>& operator[](size_t vertex_id) const;

    friend std::ostream& operator<<(std::ostream& stream, const AdjacencyMatrix& adj_matrix);

    std::vector<std::vector<Edge>> get_matrix() { return matrix; }

    long int total_length_of_paths = 0;
};


void updateProgressBar(int progress, int total);
void updateProgressBar(int progress, int total, double ratio);
bool load_gr_files(std::vector<std::string> gr_files, std::vector<Edge>& edges_out, size_t& graph_size);
bool load_clusters_mapping(std::string filename, ClusterMapping& clusters_map);
bool build_lookup_table(std::unordered_set<int>& lut, std::vector<int> cluster_nodes);
bool get_boundary_vertices(std::vector<int>& boundary_vertices, AdjacencyMatrix graph, const std::unordered_set<int>lookupTable);
int  get_boundary_vertices_enhanced(std::vector<bool>& boundary_vertices,
     AdjacencyMatrix& graph, AdjacencyMatrix& inverse_graph, const std::vector<bool>& lookupTable);
long int all_pairs_shortest_paths(std::vector<int>& cluster_nodes, std::vector<int>& boundary_vertices,
     AdjacencyMatrix& graph, std::vector<double> approx_factor, 
     std::vector<ContractedEdge>& contractedEdges, std::vector<std::vector<int>>& stats, std::vector<Edge>& IncontractablePaths);
int get_edge_id(int source, int target, AdjacencyMatrix graph);
bool normalize_edge_costs(std::vector<Edge>& edges);
bool export_contractability_vs_pathlength_stats(std::string filename, std::vector<std::vector<int>> stats);
bool export_contracted_edges(std::string filename, std::vector<ContractedEdge> contractedEdges);