import json
import random
import time
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

def read_clusters_metafile(clusters_metafile):
    correlated_nodes = []
    node_ind = 0
    with open(clusters_metafile, "r") as f:
        for line in f:
            if len(correlated_nodes) == 0:
                nodes_count = int(line.rstrip('\n'))
                correlated_nodes = np.zeros((nodes_count, 2))
                continue

            cluster_id, node = line.rstrip('\n').split(' ')
            correlated_nodes[node_ind, :] = [int(cluster_id), int(node)]
            node_ind += 1

    return correlated_nodes

def partition_analysis(edges_list, partition, report_file):
    data_list = [(key, value) for key, value in partition.items()]
    data_array = np.array(data_list)
    clusters = np.unique(data_array[:, 1])

    with open(report_file, 'w') as file:
        for cluster in clusters:
            ind = np.where(data_array[:, 1] == cluster)
            ind = np.array(ind[0])
            nodes_in_cluster = data_array[ind, 0]
            relevant_edges_ind = np.where(np.isin(edges_list[:, 0], nodes_in_cluster))
            relevant_edges_ind = np.array(relevant_edges_ind[0])
            count = len(edges_list[relevant_edges_ind, 2])
            mean = round(np.mean(edges_list[relevant_edges_ind, 2]), 2)
            median = round(np.median(edges_list[relevant_edges_ind, 2]), 2)
            stddev = round(np.std(edges_list[relevant_edges_ind, 2]), 2)

            file.write(f'{cluster.astype(int)},{count},{mean},{median},{stddev}\n')

def find_farthest_nodes(new_coords_filename, n=1):
    coords = read_coords_file(new_coords_filename)

    # Compute the pairwise Euclidean distance matrix
    distance_matrix = cdist(coords[:, 0:2], coords[:, 0:2], metric='euclidean')

    # Get the unique distances in the distance matrix and sort them in descending order
    unique_distances = np.unique(distance_matrix)
    sorted_distances = np.sort(unique_distances)[::-1]

    if n > len(sorted_distances):
        raise ValueError(
            f"Requested the {n}th farthest distance, but there are only {len(sorted_distances)} unique distances.")

    # The nth farthest distance
    nth_farthest_distance = sorted_distances[n - 1]

    # Find the indices of the nth farthest distance in the distance matrix
    result = np.where(distance_matrix == nth_farthest_distance)
    indices = list(zip(result[0], result[1]))

    # Select the first occurrence, as distances are symmetric and appear twice
    nth_farthest_indices = indices[0]

    return coords[nth_farthest_indices[0].astype(int), 2].astype(int), coords[nth_farthest_indices[1].astype(int), 2].astype(int)

def contracted_edges_similarity_analysis(coords_filename, contracted_edges_file):
    # Read data
    coordinates = read_coords_file(coords_filename)

    # Reading contracted edges output file
    contracted_edges = []
    edge_ind = 0
    with open(contracted_edges_file, "r") as f:
        for line in f:
            if len(contracted_edges) == 0:
                edges_count = int(line.rstrip('\n'))
                contracted_edges = np.zeros((edges_count, 5))
                continue

            source, target, cost1, cost2 = line.rstrip('\n').split(',')
            ind_s = np.where(coordinates[:, 2] == int(source))
            ind_s = np.array(ind_s[0])
            ind_t = np.where(coordinates[:, 2] == int(target))
            ind_t = np.array(ind_t[0])
            dist = np.linalg.norm(coordinates[ind_s, 0:2] - coordinates[ind_t, 0:2])

            contracted_edges[edge_ind, :] = [int(source), int(target), int(cost1), int(cost2), dist]
            edge_ind += 1

    trials = 0
    success = 0
    while True:
        start = np.random.choice(contracted_edges[:, 0]).astype(int)
        ind = np.where(contracted_edges[:, 0] == start)
        ind = np.array(ind[0])
        edges = contracted_edges[ind, :]

        # Compute all the mutual distances between the targets
        sub_coords = np.zeros((edges.shape[0], 2))
        for i in range(edges.shape[0]):
            ind = np.where(coordinates[:, 2] == edges[i, 1])
            ind = np.array(ind[0])
            sub_coords[i] = coordinates[ind, 0:2]

        distance_matrix = cdist(sub_coords, sub_coords, metric='euclidean')

        # Setting all elements on diagonal to INF
        min_dim = min(distance_matrix.shape)
        indices = np.arange(min_dim)
        distance_matrix[indices, indices] = np.inf

        # Randomly choose some target edge
        target_ind = np.random.choice(np.arange(edges.shape[0]))

        # Find its closest among the edges bundle
        target_neighbor_ind = np.argmin(distance_matrix[target_ind, :])
        dist = distance_matrix[target_ind, target_neighbor_ind]
        c1_costs = edges[(target_ind, target_neighbor_ind), 2]
        c2_costs = edges[(target_ind, target_neighbor_ind), 3]
        c1_width = pareto_width(max(c1_costs), min(c1_costs))
        c2_width = pareto_width(max(c2_costs), min(c2_costs))

        trials += 1
        needed_width = min(c1_width, c2_width)
        if needed_width < 0.01:
            success += 1

        # c1_costs = contracted_edges[ind, 2]
        # c2_costs = contracted_edges[ind, 3]
        # c1_width = pareto_width(max(c1_costs), min(c1_costs))
        # c2_width = pareto_width(max(c2_costs), min(c2_costs))
        print(f'Distance = {dist}, width = {needed_width}, success ratio = {success / trials}')
        # plt.figure()
        # plt.hist(c1_costs, bins = 100)
        # plt.scatter(1 + np.arange(len(c1_costs)), c1_costs)
        # plt.grid()
        # plt.show()

def pareto_width(high_cost, low_cost):
    return (high_cost - low_cost) / low_cost

def sparse_adjacency_matrix(edges_list):
    rows = edges_list[:, 0].astype(int) - 1
    columns = edges_list[:, 1].astype(int) - 1
    return coo_matrix((edges_list[:, 2].astype(int), (rows, columns))).tocsr()

def adjacency_matrix(vertices_count, edges_list):
    adj_matrix = np.zeros((vertices_count, vertices_count))
    for i in range(edges_list.shape[0]):
        start_node = int(edges_list[i, 0])
        end_node = int(edges_list[i, 1])
        cost = int(edges_list[i, 2])
        adj_matrix[start_node, end_node] = cost

    return adj_matrix

def generate_multiple_correlated_graph(distance_filename, time_filename, coords_filename,
                                       clusters_count, new_distance_gr_filename, new_time_gr_filename, new_coords_filename,
                                       clusters_metafile, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
    # Reading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]

    # Randomizing the entire graph
    distance_cost, time_cost = generate_correlated_vectors(edges_count, target_correlation=0.99, std_scaler=3000, offset=100)
    c1_graph[:, -1] = distance_cost
    c2_graph[:, -1] = time_cost

    coordinates = read_coords_file(coords_filename)

    vertices_count = coordinates.shape[0]
    vertices_set = set(coordinates[:, 2])

    # Computing the original correlation between the two cost functions
    original_corr = pearson_correlation(c1_graph[:, 2], c2_graph[:, 2])
    print(f'Original Correlation is {original_corr}')

    # Filtering only the desired area of interest
    if min_lon is not None:
        inside_nodes = np.array(np.where(
            (coordinates[:, 0] >= min_lon) & (coordinates[:, 0] <= max_lon) &
            (coordinates[:, 1] >= min_lat) & (coordinates[:, 1] <= max_lat)))

        inside_nodes = inside_nodes.reshape(inside_nodes.shape[1])

        coordinates = coordinates[inside_nodes, :]

        inside_nodes += 1

        valid_start_nodes = np.where(np.in1d(c1_graph[:, 0], inside_nodes))[0]
        valid_end_nodes = np.where(np.in1d(c1_graph[:, 1], inside_nodes))[0]
        intersection = np.intersect1d(valid_start_nodes, valid_end_nodes)
        c1_graph = c1_graph[intersection, :]
        c2_graph = c2_graph[intersection, :]

        vertices_count = coordinates.shape[0]
        edges_count = c1_graph.shape[0]
        vertices_set = set(coordinates[:, 2])

    # Computing the geographic east and north extent
    east_width = max(coordinates[:, 0]) - min(coordinates[:, 0])
    north_width = max(coordinates[:, 1]) - min(coordinates[:, 1])
    sw_corner_east = min(coordinates[:, 0])
    sw_corner_north = min(coordinates[:, 1])

    # clusters_center_x = sw_corner_east + np.random.uniform(0, 1, clusters_count) * east_width
    # clusters_center_y = sw_corner_north + np.random.uniform(0, 1, clusters_count) * north_width
    # clusters_radius = np.random.uniform(east_width / 100, north_width / 100, clusters_count)


    updated_nodes = np.array([]).reshape(-1, 2)
    clusters_correlation = 0.99
    cluster_id = 0
    for cluster_id in range(clusters_count):
        done = False
        while not done:
            clusters_center_x = sw_corner_east + np.random.uniform(0, 1) * east_width
            clusters_center_y = sw_corner_north + np.random.uniform(0, 1) * north_width
            clusters_radius = np.random.uniform(east_width / 15, north_width / 15)

            # Retrieving the list of nodes indices that reside inside the bounding circle
            nodes = points_inside_circle(coordinates[:, 0:2], [clusters_center_x, clusters_center_y],
                                         clusters_radius)
            # nodes = points_inside_circle(coordinates[:, 0:2],
            #                              [clusters_center_x[cluster_id], clusters_center_y[cluster_id]],
            #                              clusters_radius[cluster_id])
            done = nodes.shape[0] > 10

        print(f'Cluster {cluster_id + 1} / {clusters_count} (nodes count = {nodes.shape[0]}) ...')

        # Keeping track of nodes to be updated
        nodes_with_corr = np.hstack((nodes.reshape(-1, 1), np.ones((len(nodes), 1)) * clusters_correlation))
        updated_nodes = np.concatenate((updated_nodes, nodes_with_corr))

        # Updating the weight of all the edges that leave the desired nodes
        edges_to_be_updated = np.array([])
        for node in nodes:
            edges_to_be_updated = np.concatenate((edges_to_be_updated, np.argwhere(c1_graph[:, 0] == node).flatten()))
        '''
        AB = [[0.1, 10000],
              [5, 0],
              [2, 0],
              [1.5, 30000],
              [5, 40000],
              [2, 40000],
              ]
        '''

        AB = [[4, 10000],
              [2, 0],
              [1, 0],
              ]

        if len(edges_to_be_updated) > 0:
            random_row = random.choice(AB)
            chosenA = random_row[0]
            chosenB = random_row[1]

            distance_cost, time_cost = generate_correlated_vectors(len(edges_to_be_updated),
                                                                   target_correlation=clusters_correlation,
                                                                   std_scaler=3000, offset=100, A=[chosenA, chosenA],
                                                                   B=[chosenB, chosenB])

            # distance_cost, time_cost = generate_correlated_vectors(len(edges_to_be_updated),
            #                                                        target_correlation=clusters_correlation,
            #                                                        std_scaler=3000, offset=100, A=[0.01, 10], B=[-10000, 10000])

            c1_graph[edges_to_be_updated.astype(int), 2] = distance_cost
            c2_graph[edges_to_be_updated.astype(int), 2] = time_cost

        vertices_set = vertices_set - set(coordinates[nodes, 2])

    # Adding extra boundary as 0-correlation region
    add_extra_frame = False
    if(add_extra_frame):
        frame_east_width = east_width / 10
        frame_north_width = north_width / 10
        frame_ind = np.array(np.where(
            (coordinates[:, 0] < min_lon + frame_east_width) | (coordinates[:, 0] > max_lon - frame_east_width) |
            (coordinates[:, 1] < min_lat + frame_north_width) | (coordinates[:, 1] > max_lat - frame_north_width)))

        frame_ind = frame_ind.reshape(frame_ind.shape[1])
        frame_nodes = coordinates[frame_ind, 2]

        # Keeping track of nodes to be updated
        nodes_with_corr = np.hstack((frame_nodes.reshape(-1, 1), np.ones((len(frame_nodes), 1)) * clusters_correlation))
        updated_nodes = np.concatenate((updated_nodes, nodes_with_corr))

        # Updating the weight of all the edges that leave the desired nodes
        edges_to_be_updated = np.array([])
        for node in frame_nodes:
            edges_to_be_updated = np.concatenate((edges_to_be_updated, np.argwhere(c1_graph[:, 0] == node).flatten()))

        if len(edges_to_be_updated) > 0:
            distance_cost, time_cost = generate_correlated_vectors(len(edges_to_be_updated),
                                                                   target_correlation=clusters_correlation)

            # Shifting and scaling the data so all value will be non-negative
            min_value = 0
            max_value = 10e3
            distance_cost = np.clip(((abs(min(distance_cost)) + distance_cost) * 1000).astype(int), min_value,
                                    max_value)
            time_cost = np.clip(((abs(min(time_cost)) + time_cost) * 1000).astype(int), min_value, max_value)


            c1_graph[edges_to_be_updated.astype(int), 2] = distance_cost
            c2_graph[edges_to_be_updated.astype(int), 2] = time_cost

        vertices_set = vertices_set - set(frame_nodes)


    # Exporting new files
    export_gr_file(c1_graph, vertices_count, edges_count, new_distance_gr_filename,
                   'Distance graph for multiple correlated clusters')
    export_gr_file(c2_graph, vertices_count, edges_count, new_time_gr_filename,
                   'Time graph for multiple correlated clusters')
    export_coords_file(coordinates, vertices_count, new_coords_filename,
                       'Filtered coords file')

    clusters_mapping = np.zeros((len(vertices_set), 2))
    clusters_mapping[:, 0] = cluster_id
    clusters_mapping[:, 1] = np.array(list(vertices_set)).astype(int)

    export_clusters_file(clusters_mapping, clusters_metafile)

    plt.figure()
    plt.scatter(c1_graph[:, -1]/max(c1_graph[:, -1]), c2_graph[:, -1]/max(c2_graph[:, -1]))
    plt.axis('equal')
    plt.show()

def convert_partition_to_clusters_metafile(clusters_metafile, ratio_graph, partition, cluster_ratio_thr=None):
    start_time = time.time()
    data_list = [(key, value) for key, value in partition.items()]
    data_array = np.array(data_list)
    clusters = np.unique(data_array[:, 1])
    exported_clusters_count = 0

    if cluster_ratio_thr is None:
        cluster_ratio_thr = float('inf')

    for cluster_id in clusters:
        print(f'Cluster {cluster_id}/{len(clusters)}')
        ind = np.where(data_array[:, 1] == cluster_id)
        ind = np.array(ind[0])
        nodes_in_cluster = data_array[ind, 0]
        relevant_edges_ind = np.where(np.isin(ratio_graph[:, 0], nodes_in_cluster))
        relevant_edges_ind = np.array(relevant_edges_ind[0])
        count = len(nodes_in_cluster)
        stddev = round(np.std(ratio_graph[relevant_edges_ind, 2]), 2)

        # Checking that cluster statistics is salient enough
        if count > 50 and stddev < cluster_ratio_thr:
            cluster_mapping = np.zeros((count, 2))
            cluster_mapping[:, 0] = cluster_id
            cluster_mapping[:, 1] = nodes_in_cluster

            if exported_clusters_count == 0:
                exported_clusters_mapping = cluster_mapping
            else:
                exported_clusters_mapping = np.vstack((exported_clusters_mapping, cluster_mapping))

            exported_clusters_count += 1

    export_clusters_file(exported_clusters_mapping, clusters_metafile)
    end_time = time.time()
    print(f'Cluster partition export to metafile completed in {end_time-start_time} [sec]')


def inter_clusters_cost_approximation(clusters_metafile, timeout, epsilon, output_dir,
                                      distance_filename, time_filename, path2ApexExe, path2CorrClusteringExe,
                                      super_edges_file):

    apex_query_filename = rf'{output_dir}\query.txt'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Running Method 1 (Single-objectives queries)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'ICCA-method-1 start...')
    log_file = rf'{output_dir}\AllPairsDijkstra.txt'
    command_line = f'{path2CorrClusteringExe} -m {distance_filename} {time_filename} \
                                    -e {epsilon} -l {log_file} -c {clusters_metafile} -s {super_edges_file}'
    os.system(command_line)

    # Analyzing the number of contracted and incontracted paths for first iteration only
    with open(log_file, "r") as f:
        f.readline().rstrip('\n')
        contracted_paths_count = f.readline().rstrip('\n')
        incontracted_paths_count = f.readline().rstrip('\n')

        contracted_paths = np.zeros((int(contracted_paths_count), 2))
        incontracted_paths = np.zeros((int(incontracted_paths_count), 2))
        contracted_paths_ind = 0
        incontracted_paths_ind = 0

        for line in f:
            source, target, result = line.rstrip('\n').split(',')
            if int(result) == 0:
                incontracted_paths[incontracted_paths_ind, 0] = int(source)
                incontracted_paths[incontracted_paths_ind, 1] = int(target)
                incontracted_paths_ind += 1
            else:
                contracted_paths[contracted_paths_ind, 0] = int(source)
                contracted_paths[contracted_paths_ind, 1] = int(target)
                contracted_paths_ind += 1

    # Create a query for method 2
    contracted_paths = contracted_paths.astype(int)
    incontracted_paths = incontracted_paths.astype(int)

    if incontracted_paths.shape[0] > 0:
        write_query_file(incontracted_paths, apex_query_filename)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Running Method 2 (A*pex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f'ICCA-method-2 start...')
        apex_log_file = rf'{output_dir}\Apex_output.txt'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                                   -e {epsilon} -h 0 -q {apex_query_filename} -a Apex -o output.txt -l {apex_log_file} -t {timeout}'
        os.system(command_line)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Updating the super-edges file if method2 was invoked
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        method1_contracted_edges = []
        edge_ind = 0

        # Reading the original super_edges file generated by method-1
        with open(super_edges_file, "r") as f:
            for line in f:
                if len(method1_contracted_edges) == 0:
                    edges_count = int(line.rstrip('\n'))
                    method1_contracted_edges = np.zeros((edges_count, 6))
                    continue

                source, target, cost1, apex1, cost2, apex2 = line.rstrip('\n').split(',')
                method1_contracted_edges[edge_ind, :] = [int(source), int(target), int(cost1), int(apex1), int(cost2), int(apex2)]
                edge_ind += 1

        # Reading the results of method-2
        method2_contracted_edges = np.empty((0, 6))
        edge_ind = 0
        with open(apex_log_file, 'r') as file:
            log = json.load(file)

        solutions_count = len(log)

        for k in range(solutions_count):
            source = log[k]['source']
            target = log[k]['target']
            for apex in log[k]['finish_info']['solutions']:
                bi_cost = apex['full_cost']
                cost1 = bi_cost[0]
                cost2 = bi_cost[1]
                apex1 = cost1
                apex2 = cost2
                new_edge = np.array([[int(source), int(target), int(cost1), int(apex1), int(cost2),
                                      int(apex2)]])
                method2_contracted_edges = np.vstack((method2_contracted_edges, new_edge))

        # Integrating the two methods super-edges to a single file
        super_edges = np.vstack((method1_contracted_edges, method2_contracted_edges))
        super_edges_count = super_edges.shape[0]
        with open(super_edges_file, 'w') as file:
            file.write(f'{super_edges_count}\n')
            for i in range(super_edges_count):
                file.write(f'{int(super_edges[i, 0])},{int(super_edges[i, 1])},{int(super_edges[i, 2])},{int(super_edges[i, 3])},{int(super_edges[i, 4])},{int(super_edges[i, 5])}\n')


def convert_partition_to_clusters_metafile_enhanced(clusters_metafile, node2cluster, cluster2node):
    start_time = time.time()
    exported_clusters_count = 0
    clusters_ids = list(cluster2node.keys())

    next_counter_report = 0
    progress = 0
    for i in range(len(clusters_ids)):
        if progress >= next_counter_report:
            print(f'{round(progress)}%')
            next_counter_report += 1
        progress = i / len(clusters_ids) * 100

        cluster_id = clusters_ids[i]
        count = len(cluster2node[cluster_id])
        cluster_mapping = np.zeros((count, 2))
        cluster_mapping[:, 0] = cluster_id
        cluster_mapping[:, 1] = cluster2node[cluster_id]

        if exported_clusters_count == 0:
            exported_clusters_mapping = cluster_mapping
        else:
            exported_clusters_mapping = np.vstack((exported_clusters_mapping, cluster_mapping))

        exported_clusters_count += 1

    export_clusters_file(exported_clusters_mapping, clusters_metafile)
    end_time = time.time()
    print(f'Cluster partition export to metafile completed in {end_time-start_time} [sec]')

def generate_triobjective_multiple_correlated_graph(distance_filename, time_filename, coords_filename,
                                       clusters_count, new_distance_gr_filename, new_time_gr_filename,
                                       new_coords_filename, new_fuel_gr_filename,
                                       clusters_metafile, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
    # Reading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]
    coordinates = read_coords_file(coords_filename)

    # Computing the original correlation between the two cost functions
    original_corr = pearson_correlation(c1_graph[:, 2], c2_graph[:, 2])

    # Filtering only the desired area of interest
    if min_lon is not None:
        inside_nodes = np.array(np.where(
            (coordinates[:, 0] >= min_lon) & (coordinates[:, 0] <= max_lon) &
            (coordinates[:, 1] >= min_lat) & (coordinates[:, 1] <= max_lat)))

        inside_nodes = inside_nodes.reshape(inside_nodes.shape[1])

        coordinates = coordinates[inside_nodes, :]

        inside_nodes += 1

        valid_start_nodes = np.where(np.in1d(c1_graph[:, 0], inside_nodes))[0]
        valid_end_nodes = np.where(np.in1d(c1_graph[:, 1], inside_nodes))[0]
        intersection = np.intersect1d(valid_start_nodes, valid_end_nodes)
        c1_graph = c1_graph[intersection, :]
        c2_graph = c2_graph[intersection, :]

        vertices_count = coordinates.shape[0]
        edges_count = c1_graph.shape[0]
        vertices_set = set(coordinates[:, 2])

    # Computing the geographic east and north extent
    east_width = max(coordinates[:, 0]) - min(coordinates[:, 0])
    north_width = max(coordinates[:, 1]) - min(coordinates[:, 1])
    sw_corner_east = min(coordinates[:, 0])
    sw_corner_north = min(coordinates[:, 1])

    clusters_center_x = sw_corner_east + np.random.uniform(0, 1, clusters_count) * east_width
    clusters_center_y = sw_corner_north + np.random.uniform(0, 1, clusters_count) * north_width

    clusters_radius = np.random.uniform(east_width / 30, east_width / 20, clusters_count)

    updated_nodes = np.array([]).reshape(-1, 2)
    clusters_correlation = 0
    cluster_id = 0
    for cluster_id in range(clusters_count):
        print(f'Cluster {cluster_id + 1} / {clusters_count}...')
        # Retrieving the list of nodes indices that reside inside the bounding circle
        nodes = points_inside_circle(coordinates[:, 0:2], [clusters_center_x[cluster_id], clusters_center_y[cluster_id]],
                                     clusters_radius[cluster_id])

        # Keeping track of nodes to be updated
        nodes_with_corr = np.hstack((nodes.reshape(-1, 1), np.ones((len(nodes), 1)) * clusters_correlation))
        updated_nodes = np.concatenate((updated_nodes, nodes_with_corr))

        # Updating the weight of all the edges that leave the desired nodes
        edges_to_be_updated = np.array([])
        for node in nodes:
            edges_to_be_updated = np.concatenate((edges_to_be_updated, np.argwhere(c1_graph[:, 0] == node).flatten()))

        if len(edges_to_be_updated) > 0:
            distance_cost, time_cost = generate_correlated_vectors(len(edges_to_be_updated), target_correlation=clusters_correlation)
            c1_graph[edges_to_be_updated.astype(int), 2] = distance_cost
            c2_graph[edges_to_be_updated.astype(int), 2] = time_cost

        vertices_set = vertices_set - set(coordinates[nodes, 2])

    # Exporting new files
    export_gr_file(c1_graph, vertices_count, edges_count, new_distance_gr_filename,
                   'Distance graph for multiple correlated clusters')
    export_gr_file(c2_graph, vertices_count, edges_count, new_time_gr_filename,
                   'Time graph for multiple correlated clusters')
    export_coords_file(coordinates, vertices_count, new_coords_filename,
                       'Filtered NY coords file')

    clusters_mapping = np.zeros((len(vertices_set), 2))
    clusters_mapping[:, 0] = cluster_id
    clusters_mapping[:, 1] = np.array(list(vertices_set)).astype(int)

    export_clusters_file(clusters_mapping, clusters_metafile)


def load_graph(filename):
    edge_ind = 0
    with open(filename, "r") as f:
        for line in f:
            # Inputting the graph dimensions and allocating data structures
            if line[0] == 'p':
                _, _, vertices_count_str, edges_count_str = line.rstrip('\n').split(' ')
                vertices_count = int(vertices_count_str)
                edges_count = int(edges_count_str)
                graph = np.zeros((edges_count, 3))
                continue

            # Inputting each edge data
            if line[0] == 'a':
                _, start_node, end_node, cost = line.rstrip('\n').split(' ')
                graph[edge_ind, 0] = int(start_node)
                graph[edge_ind, 1] = int(end_node)
                graph[edge_ind, 2] = cost
                edge_ind += 1

    return graph, vertices_count

def points_inside_circle(points, center, radius):
    """
    Identify points inside a circle.

    Parameters:
    - points: Nx2 array of (x, y) coordinates of points
    - center: (x, y) coordinates of the circle's center
    - radius: Radius of the circle

    Returns:
    - List of indices of points inside the circle
    """
    distances = np.linalg.norm(points - center, axis=1)
    indices_inside = np.where(distances <= radius)[0]
    return indices_inside

def create_sparse_adj_matrix(edge_list, num_nodes):
    # Create a COO sparse matrix from edge_list
    row_indices = [edge[0] for edge in edge_list]
    col_indices = [edge[1] for edge in edge_list]
    weights = [edge[2] for edge in edge_list]

    sparse_adj_matrix = coo_matrix((weights, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    return sparse_adj_matrix

def spectral_clustering(edge_list, num_clusters):
    # Extract nodes and create a mapping from node indices to consecutive integers
    nodes = list(set(node for edge in edge_list for node in edge))
    node_indices = {node: i for i, node in enumerate(nodes)}

    # Create a sparse adjacency matrix
    sparse_adj_matrix = create_sparse_adj_matrix(edge_list, len(nodes))

    # Compute the Laplacian matrix
    laplacian_matrix = csgraph.laplacian(sparse_adj_matrix, normed=False)

    # Perform a partial eigendecomposition using sparse eigensolver
    num_eigenvectors = min(num_clusters + 1, len(nodes) - 1)  # Choose the number of eigenvectors to compute
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=num_eigenvectors, which='SM')  # 'SM' for smallest magnitude

    print('After eigsh, invoking SpectralClustering...')
    # Use the eigenvectors for spectral clustering
    spectral_model = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', eigen_solver='arpack')
    cluster_labels = spectral_model.fit_predict(eigenvectors[:, 1:num_clusters+1])  # Exclude the first eigenvector (constant)

    return nodes, cluster_labels

def graph_agglomerative_clustering(distance_filename, time_filename, eps):
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)

    print('Building graph for networkx...')
    c1_graph = c1_graph[0:10000, :]
    G = nx.DiGraph()
    for i in range(c1_graph.shape[0]):
        start_node = int(c1_graph[i, 0])
        end_node = int(c1_graph[i, 1])
        cost = int(c1_graph[i, 2])
        G.add_edge(start_node, end_node, weight=cost)

    print('Running Floyd-Warshall...')
    t1 = time.time()
    result_matrix = nx.floyd_warshall(G)
    t2 = time.time()
    print(f'Floyd-Warshall took {t2-t1} [sec]')

def read_queries_file(query_file):
    queries = []
    with open(query_file, "r") as f:
        for line in f:
            start, goal = line.rstrip('\n').split(',')
            queries.append([int(start), int(goal)])

    return np.array(queries)

def read_coords_file(coords_filename):
    # Reading the coords file
    with open(coords_filename, "r") as f:
        row_id = 0
        for line in f:
            if line[0] == 'p':
                _, _, _, _, vertices_count = line.rstrip('\n').split(' ')
                coordinates = np.zeros((int(vertices_count), 3))
            if line[0] == 'v':
                _, node, x, y = line.rstrip('\n').split(' ')
                coordinates[row_id, :] = [int(x), int(y), int(node)]
                row_id += 1

    return coordinates

def plot_graph(adjacency_filename, coords_filename, clusters_metafile=None, boundary_nodes_file=None):
    # Reading the adjacency graph structure
    edge_ind = 0
    with open(adjacency_filename, "r") as f:
        for line in f:
            # Inputting the graph dimensions and allocating data structures
            if line[0] == 'p':
                _, _, vertices_count_str, edges_count_str = line.rstrip('\n').split(' ')
                vertices_count = int(vertices_count_str)
                edges_count = int(edges_count_str)
                graph = np.zeros((edges_count, 2))
                continue

            # Inputting each edge data
            if line[0] == 'a':
                _, graph[edge_ind, 0], graph[edge_ind, 1], \
                    _ = line.rstrip('\n').split(' ')
                edge_ind += 1

    # Reading the coords file
    coordinates = read_coords_file(coords_filename)

    # Reading the clusters metafile
    correlated_nodes = []
    node_ind = 0
    if clusters_metafile is not None:
        with open(clusters_metafile, "r") as f:
            for line in f:
                if len(correlated_nodes) == 0:
                    nodes_count = int(line.rstrip('\n'))
                    correlated_nodes = np.zeros((nodes_count, 2))
                    continue

                cluster_id, node = line.rstrip('\n').split(' ')
                correlated_nodes[node_ind, :] = [int(cluster_id), int(node)]
                node_ind += 1


        # Retrieving the coordinates row corresponding to the clusters' nodes ids
        ind = np.where(np.isin(coordinates[:, 2], correlated_nodes[:, 1]))
        ind = np.array(ind[0])

    marker_size = 0.5
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red', marker='.', s=marker_size, label='Unclustered Nodes')
    if clusters_metafile is not None:
        plt.scatter(coordinates[ind, 0], coordinates[ind, 1],
                    color='blue', marker='.', s=marker_size, label=f'Correlated-Clustered Nodes')

    # Plotting the boundary nodes
    if boundary_nodes_file is not None:
        bnd_nodes_id = read_boundary_nodes(boundary_nodes_file)

        # Retrieving the coordinates row corresponding to the clusters' nodes ids
        bnd_ind = np.where(np.isin(coordinates[:, 2], bnd_nodes_id))
        bnd_ind = np.array(bnd_ind[0])
        plt.scatter(coordinates[bnd_ind, 0], coordinates[bnd_ind, 1],
                    color='green', marker='.', s=10, label='Boundary Nodes')

    plt.grid()
    plt.axis('equal')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(fontsize=8)
    plt.title('Graph with multiple correlations regions', fontsize=14, fontweight='bold')
    plt.show()


def read_boundary_nodes(boundary_nodes_file):
    bnd_nodes_id = np.array([])
    with open(boundary_nodes_file, "r") as f:
        for line in f:
            node = line.rstrip('\n').split(' ')
            if bnd_nodes_id.size == 0:
                bnd_nodes_id = np.array(int(node[0]))
            else:
                bnd_nodes_id = np.vstack((bnd_nodes_id, int(node[0])))

    return bnd_nodes_id


def export_gr_file(graph, vertices_count, edges_count, filename,
                   header_string, first_super_edge_ind=None):
    with open(filename, 'w') as file:
        # Write header
        file.write('c ' + header_string + '\n')

        # Write problem line (optional)
        file.write("p sp {} {}\n".format(vertices_count, edges_count))

        # Write edge lines
        if first_super_edge_ind is None:
            first_super_edge_ind = graph.shape[0]
        for i in range(edges_count):
            if i <= first_super_edge_ind:
                file.write("a {} {} {}\n".format(graph[i, 0].astype(int), graph[i, 1].astype(int), graph[i, 2].astype(int)))
            else:
                file.write("s {} {} {} {}\n".format(graph[i, 0].astype(int), graph[i, 1].astype(int),
                                                    graph[i, 2].astype(int), graph[i, 3].astype(int)))

def export_coords_file(coordinates, vertices_count, new_coords_filename, header_string):
    with open(new_coords_filename, 'w') as file:
        # Write header
        file.write('c ' + header_string + '\n')

        # Write problem line (optional)
        file.write("p aux sp co {}\n".format(vertices_count))

        # Write nodes' coordinates
        for i in range(vertices_count):
            file.write("v {} {} {}\n".format(coordinates[i, 2].astype(int), coordinates[i, 0].astype(int), coordinates[i, 1].astype(int)))

def export_clusters_file(clusters, filename):
    with open(filename, 'w') as file:
        file.write("{}\n".format(clusters.shape[0]))
        for i in range(clusters.shape[0]):
            file.write("{} {}\n".format(clusters[i, 0].astype(int), clusters[i, 1].astype(int)))

def testbench_A(input_filename, output_dir, corr_vec, samples_per_corr, path2ApexExe):
    # Reading the input graph structure
    edge_ind = 0
    with open(input_filename, "r") as f:
        for line in f:
            # Inputting the graph dimensions and allocating data structures
            if line[0] == 'p':
                _, _, vertices_count_str, edges_count_str = line.rstrip('\n').split(' ')
                vertices_count = int(vertices_count_str)
                edges_count = int(edges_count_str)
                graph = np.zeros((edges_count, 3))
                continue

            # Inputting each edge data
            if line[0] == 'a':
                _, graph[edge_ind, 0], graph[edge_ind, 1], \
                    _ = line.rstrip('\n').split(' ')
                edge_ind += 1

        # Scanning through requested target correlations
        for target_corr in corr_vec:
            # Generating two costs vector with desired linear correlation
            distance_cost, time_cost = generate_correlated_vectors(edges_count, target_corr)

            # Shifting and scaling the data so all value will be non-negative
            min_value = 0
            max_value = 10e3
            distance_cost = np.clip(((abs(min(distance_cost)) + distance_cost) * 1000).astype(int), min_value, max_value)
            time_cost = np.clip(((abs(min(time_cost)) + time_cost) * 1000).astype(int), min_value, max_value)

            # debug printing
            new_correlation = pearson_correlation(distance_cost, time_cost)
            print(f'Current correlation is {new_correlation}')

            # Generating the new distance graph file
            new_distance_filename = output_dir + f'\Distance_corr_{int(round(target_corr*100))}.gr'
            with open(new_distance_filename, 'w') as file:
                # Write header
                file.write(f'c Distance graph adjusted to correlation = {target_corr}\n')

                # Write problem line (optional)
                file.write("p sp {} {}\n".format(vertices_count, edges_count))

                # Write edge lines
                for i in range(edges_count):
                    file.write("a {} {} {}\n".format(graph[i, 0].astype(int), graph[i, 1].astype(int), distance_cost[i]))

            # Generating the new time graph file
            new_time_filename = output_dir + f'\Time_corr_{int(round(target_corr*100))}.gr'
            with open(new_time_filename, 'w') as file:
                # Write header
                file.write(f'c Time graph adjusted to correlation = {target_corr}\n')

                # Write problem line (optional)
                file.write("p sp {} {}\n".format(vertices_count, edges_count))

                # Write edge lines
                for i in range(edges_count):
                    file.write("a {} {} {}\n".format(graph[i, 0].astype(int), graph[i, 1].astype(int), time_cost[i]))

            # Performing multiple A*pex invocations for the same correlation
            for i in range(samples_per_corr):
                print(f'Iteration {i}:')
                ready = False
                while not ready:
                    startNode = np.random.randint(min(graph[:, 0]), max(graph[:, 0]))
                    goalNode = np.random.randint(min(graph[:, 0]), max(graph[:, 0]))
                    if abs(startNode - goalNode) > vertices_count * 0.5:
                        ready = True
                log_file = f'{output_dir}\log_corr_{int(round(target_corr*100))}_iter_{i+1}.json'
                command_line = f'{path2ApexExe}  -m {new_distance_filename} {new_time_filename} \
                               -e 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file}'
                os.system(command_line)

def fix_json_file(filename):
    tmp_filename = filename + '_'
    blocks_number = 0
    current_block = 1
    with open(filename, "r") as f_in:
        with open(tmp_filename, "w") as f_out:
            f_in.readline()
            for line in f_in:
                if line.find('start_time') != -1:
                    blocks_number += 1

    if blocks_number == 1:
        return

    with open(filename, "r") as f_in:
        with open(tmp_filename, "w") as f_out:
            f_in.readline()
            for line in f_in:
                if current_block == blocks_number:
                    f_out.write(line)
                elif line[0] == ']':
                    current_block += 1

    os.rename(filename, filename + '_old')
    os.rename(tmp_filename, filename)

def analyze_corr_vs_optimality_ratio(output_dir, correlations, samples):
    results = {}
    for corr in correlations:
        results[corr] = np.array([])
        for sample in range(1, samples + 1):
            log_file = f'{output_dir}\log_corr_{int(round(corr * 100))}_iter_{sample}.json'
            with open(log_file, 'r') as file:
                log = json.load(file)
            n_solutions = log[0]['finish_info']['amount_of_solutions']
            solutions = np.zeros((n_solutions, 2))
            for i in range(n_solutions):
                solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']

            # Computing the width in respect to the 1st cost function
            best_cost2_sol_ind = np.argmin(solutions[:, 1])
            cost1_at_best_cost2 = solutions[best_cost2_sol_ind, 0]
            best_cost1 = min(solutions[:, 0])
            needed_epsilon_cost1 = (cost1_at_best_cost2 - best_cost1) / best_cost1

            # Computing the width in respect to the 2nd cost function
            best_cost1_sol_ind = np.argmin(solutions[:, 0])
            cost2_at_best_cost1 = solutions[best_cost1_sol_ind, 1]
            best_cost2 = min(solutions[:, 1])
            needed_epsilon_cost2 = (cost2_at_best_cost1 - best_cost2) / best_cost2

            needed_epsilon = max([needed_epsilon_cost1, needed_epsilon_cost2])

            if needed_epsilon < 5:
                results[corr] = np.append(results[corr], needed_epsilon)

    for corr in results.keys():
        plt.scatter(np.ones(results[corr].shape) * corr, results[corr], color='grey', marker='o', s=5)

    plt.plot(correlations, [max(results[corr]) for corr in results.keys()], color='red', linewidth=3)
    # plt.plot(correlations, [np.percentile(results[corr], 99) for corr in results.keys()], color='red', linewidth=3)

    # Adding labels and title
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel(r'Needed Approximation Factor ($\varepsilon$)')
    plt.title('Pareto-Optimal front width vs. Correlation')
    plt.grid()
    plt.show()

def pearson_correlation(vector1, vector2):
    # Standardize the vectors
    std_vector1 = (vector1 - np.mean(vector1)) / np.std(vector1)
    std_vector2 = (vector2 - np.mean(vector2)) / np.std(vector2)

    # Calculate the current correlation
    return np.corrcoef(std_vector1, std_vector2)[0, 1]


def generate_correlated_vectors(size, target_correlation,
                                std_scaler=1, offset=1, A=[1, 1], B=[0, 0]):
    # Generate uncorrelated random variables
    random_variables = np.random.randn(2, size) * std_scaler

    # Create the covariance matrix from the target correlation
    covariance_matrix = np.array([[1, target_correlation], [target_correlation, 1]])

    # Perform Cholesky decomposition on the covariance matrix
    cholesky_matrix = np.linalg.cholesky(covariance_matrix)

    # Transform the uncorrelated variables to be correlated
    correlated_variables = np.dot(cholesky_matrix, random_variables)

    vector1 = correlated_variables[0, :]
    vector2 = correlated_variables[1, :]

    # Randomize the slope (A) and intercept (B)
    A = np.random.uniform(A[0], A[1])  # Random positive slope
    B = np.random.uniform(B[0], B[1])  # Random intercept

    # Extract the correlated vectors, adding bias to avoid negative values
    vector1 = vector1 - min(vector1) + offset
    vector2 = vector2 - min(vector2) + offset

    # Adjust vector2 to follow Y = AX + B relationship
    vector1 = A * vector1 + B

    return vector1.astype(int), vector2.astype(int)

def generate_correlated_graph(gr_filename, out_distance_filename, out_time_filename, target_corr):
    # Reading the input graph
    edge_ind = 0
    with open(gr_filename, "r") as f:
        for line in f:
            # Inputting the graph dimensions and allocating data structures
            if line[0] == 'p':
                _, _, vertices_count_str, edges_count_str = line.rstrip('\n').split(' ')
                vertices_count = int(vertices_count_str)
                edges_count = int(edges_count_str)
                graph = np.zeros((edges_count, 3))
                continue

            # Inputting each edge data
            if line[0] == 'a':
                _, graph[edge_ind, 0], graph[edge_ind, 1], \
                    _ = line.rstrip('\n').split(' ')
                edge_ind += 1

        # Generating two costs vector with desired linear correlation
        distance_cost, time_cost = generate_correlated_vectors(edges_count, target_corr)

        # Shifting and scaling the data so all value will be non-negative
        distance_cost = ((abs(min(distance_cost)) + distance_cost) * 1000).astype(int)
        time_cost = ((abs(min(time_cost)) + time_cost) * 1000).astype(int)

        # debug check new correlation
        new_correlation = pearson_correlation(distance_cost, time_cost)
        print(f'Correlation check is {new_correlation}')

        # Generating the new distance graph file
        with open(out_distance_filename, 'w') as file:
            # Write header
            file.write(f'c Distance graph adjusted to correlation = {target_corr}\n')

            # Write problem line (optional)
            file.write("p sp {} {}\n".format(vertices_count, edges_count))

            # Write edge lines
            for i in range(edges_count):
                file.write("a {} {} {}\n".format(graph[i, 0].astype(int), graph[i, 1].astype(int), distance_cost[i]))

        # Generating the new time graph file
        with open(out_time_filename, 'w') as file:
            # Write header
            file.write(f'c Time graph adjusted to correlation = {target_corr}\n')

            # Write problem line (optional)
            file.write("p sp {} {}\n".format(vertices_count, edges_count))

            # Write edge lines
            for i in range(edges_count):
                file.write("a {} {} {}\n".format(graph[i, 0].astype(int), graph[i, 1].astype(int), time_cost[i]))

def plot_path_length_vs_contractability(stats_file):
    paths_count = -1
    index = 0
    with open(stats_file, "r") as f:
        for line in f:
            if paths_count < 0:
                paths_count = line.rstrip('\n')
                paths_count = int(paths_count)
                data = np.zeros((paths_count, 2))
                continue

            path_length, contracted = line.rstrip('\n').split(',')
            path_length = int(path_length)
            is_contracted = int(contracted)
            data[index,0] = path_length
            data[index, 1] = is_contracted
            index += 1

    success_percent = round(data[:, 1].sum() / data[:, 1].size * 100)
    fail_percent = 100 - success_percent
    not_contracted_arr = data[data[:, 1] == 0, 0]
    contracted_arr = data[data[:, 1] == 1, 0]
    bins_no = 100
    plt.hist(not_contracted_arr, bins = bins_no, color='red', alpha=0.5, label=f'Fail {fail_percent}%', density=False)
    plt.hist(contracted_arr,  bins = bins_no, color='lime', alpha=0.5, label=f'Success {success_percent}%', density=False)
    plt.xlabel('Path Length')
    plt.ylabel('Number of Paths')
    plt.title(f'Edge Contractability vs. Path Length\nNumber of Paths = {data.shape[0]}', fontweight='bold')
    plt.grid()
    plt.legend()
    plt.show()


def testbench_B(distance_filename, time_filename, coords_filename, output_dir, samples, path2ApexExe):
    # Reading the input graph structure
    edge_ind = 0
    with open(distance_filename, "r") as f:
        for line in f:
            # Inputting the graph dimensions and allocating data structures
            if line[0] == 'p':
                _, _, vertices_count_str, edges_count_str = line.rstrip('\n').split(' ')
                vertices_count = int(vertices_count_str)
                edges_count = int(edges_count_str)
                graph = np.zeros((edges_count, 3))
                continue

            # Inputting each edge data
            if line[0] == 'a':
                _, graph[edge_ind, 0], graph[edge_ind, 1], \
                    _ = line.rstrip('\n').split(' ')
                edge_ind += 1

        # Performing multiple A*pex invocations for the same correlation
        coordinates = read_coords_file(coords_filename)
        # Single-type edges
        min_lat = 4.089e7
        max_lat = 4.097e7
        min_lon = -7.401e7
        max_lon = -7.395e7

        # Multi-type edges
        # min_lat = 4.050e7
        # max_lat = 4.058e7
        # min_lon = -7.439e7
        # max_lon = -7.433e7

        inside_nodes = np.array(np.where(
            (coordinates[:, 0] >= min_lon) & (coordinates[:, 0] <= max_lon) &
            (coordinates[:, 1] >= min_lat) & (coordinates[:, 1] <= max_lat)))

        inside_nodes = inside_nodes.reshape(inside_nodes.shape[1])

        coordinates = coordinates[inside_nodes, :]

        for i in range(samples):
            print(f'Iteration {i}:')
            ready = False
            while not ready:
                # startNode = np.random.randint(min(graph[:, 0]), max(graph[:, 0]))
                # goalNode = np.random.randint(min(graph[:, 0]), max(graph[:, 0]))
                startNode = np.random.choice(coordinates[:, 2]).astype(int)
                goalNode = np.random.choice(coordinates[:, 2]).astype(int)
                ready = True
            log_file = f'{output_dir}\log_iter_{i+1}.json'
            command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                           -e 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file}'
            os.system(command_line)

            # Analyzing the Pareto-set width
            with open(log_file, 'r') as file:
                log = json.load(file)
            n_solutions = log[0]['finish_info']['amount_of_solutions']
            solutions = np.zeros((n_solutions, 2))
            for i in range(n_solutions):
                solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']

            # Computing the width in respect to the 1st cost function
            best_cost2_sol_ind = np.argmin(solutions[:, 1])
            cost1_at_best_cost2 = solutions[best_cost2_sol_ind, 0]
            best_cost1 = min(solutions[:, 0])
            needed_epsilon_cost1 = (cost1_at_best_cost2 - best_cost1) / best_cost1

            # Computing the width in respect to the 2nd cost function
            best_cost1_sol_ind = np.argmin(solutions[:, 0])
            cost2_at_best_cost1 = solutions[best_cost1_sol_ind, 1]
            best_cost2 = min(solutions[:, 1])
            needed_epsilon_cost2 = (cost2_at_best_cost1 - best_cost2) / best_cost2

            needed_epsilon = max([needed_epsilon_cost1, needed_epsilon_cost2])
            print('=========================================================================================')
            print(f'-------------> Epsilons = ({round(needed_epsilon_cost1, 2)},{round(needed_epsilon_cost2, 2)}), Needed Epsilon is {round(needed_epsilon, 2)}')
            print('=========================================================================================')


def read_cluster_nodes(clusters_metafile):
    cluster_nodes_id = []
    ind = 0
    with open(clusters_metafile, "r") as f:
        for line in f:
            if len(cluster_nodes_id) == 0:
                count = int(line.rstrip('\n'))
                cluster_nodes_id = np.zeros(count)
                continue

            _, node = line.rstrip('\n').split(' ')
            cluster_nodes_id[ind] = int(node)
            ind += 1

    return cluster_nodes_id

def testbench_D(distance_filename, time_filename, cntrcted_distance_filename, cntrcted_time_filename,
                coords_filename, clusters_metafile, output_dir, samples, path2ApexExe, epsilon):
    # Reading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]
    coordinates = read_coords_file(coords_filename)
    cluster_nodes_id = read_cluster_nodes(clusters_metafile)
    non_cluster_nodes_ids = np.setdiff1d(coordinates[:, 2], cluster_nodes_id)

    # Performing multiple A*pex invocations for the same correlation
    for i in range(samples):
        print(f'Iteration {i}:')
        ready = False
        while not ready:
            startNode = int(np.random.choice(non_cluster_nodes_ids))
            goalNode  = int(np.random.choice(non_cluster_nodes_ids))
            start_crdnts_id = np.argwhere(coordinates[:, 2] == startNode)
            goal_crdnts_id = np.argwhere(coordinates[:, 2] == goalNode)
            if np.linalg.norm(coordinates[start_crdnts_id, 0:2]-coordinates[goal_crdnts_id, 0:2]) > 0.3e7:
                # debug
                print(f'Start = ({coordinates[start_crdnts_id, 0]},{coordinates[start_crdnts_id, 1]})')
                print(f'Goal = ({coordinates[goal_crdnts_id, 0]},{coordinates[goal_crdnts_id, 1]})')
                ready = True

        # A*pex invocation WITHOUT edge contraction
        print('============ WITHOUT CONTRACTION ============')
        log_file = f'{output_dir}\log_without_contraction_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                               -e {epsilon} -h 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file}'
        os.system(command_line)

        # A*pex invocation WITH edge contraction
        print('============ WITH CONTRACTION ============')
        log_file = f'{output_dir}\log_with_contraction_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {cntrcted_distance_filename} {cntrcted_time_filename} \
                                       -e {epsilon} -h 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file}'
        os.system(command_line)

def testbench_E(distance_filename, time_filename, coords_filename, output_dir, samples, path2ApexExe, epsilon):
    # Reading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]
    coordinates = read_coords_file(coords_filename)

    # Performing multiple A*pex invocations for the same correlation
    for i in range(samples):
        print(f'Iteration {i}:')
        ready = False
        while not ready:
            startNode = int(np.random.choice(coordinates[:, 2]))
            goalNode  = int(np.random.choice(coordinates[:, 2]))
            start_crdnts_id = np.argwhere(coordinates[:, 2] == startNode)
            goal_crdnts_id = np.argwhere(coordinates[:, 2] == goalNode)
            if np.linalg.norm(coordinates[start_crdnts_id, 0:2]-coordinates[goal_crdnts_id, 0:2]) > 1e5:
                # debug
                print(f'Start = ({coordinates[start_crdnts_id, 0]},{coordinates[start_crdnts_id, 1]})')
                print(f'Goal = ({coordinates[goal_crdnts_id, 0]},{coordinates[goal_crdnts_id, 1]})')
                ready = True

        # A*pex invocation WITHOUT epsilon and WITHOUT heuristics inflation
        print('============ A ============')
        log_file = f'{output_dir}\log_A_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                       -e 0 -h 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file}'
        os.system(command_line)

        # A*pex invocation WITH epsilon and WITHOUT heuristics inflation
        print('============ B ============')
        log_file = f'{output_dir}\log_B_iter_{i+1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                       -e {epsilon} -h 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file}'
        os.system(command_line)

        # A*pex invocation WITH epsilon and WITH heuristics inflation
        print('============ C ============')
        log_file = f'{output_dir}\log_C_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                               -e {epsilon} -h 1 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file}'
        os.system(command_line)

def testbench_F(distance_filename, time_filename, coords_filename, output_dir, samples, path2ApexExe):
    # Reading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    coordinates = read_coords_file(coords_filename)

    # Performing multiple A*pex invocations for the same correlation
    for i in range(samples):
        print(f'Iteration {i}:')
        ready = False
        while not ready:
            startNode = int(np.random.choice(coordinates[:, 2]))
            goalNode  = int(np.random.choice(coordinates[:, 2]))
            start_crdnts_id = np.argwhere(coordinates[:, 2] == startNode)
            goal_crdnts_id = np.argwhere(coordinates[:, 2] == goalNode)
            if np.linalg.norm(coordinates[start_crdnts_id, 0:2]-coordinates[goal_crdnts_id, 0:2]) > 1e5:
                # debug
                print(f'Start = ({coordinates[start_crdnts_id, 0]},{coordinates[start_crdnts_id, 1]})')
                print(f'Goal = ({coordinates[goal_crdnts_id, 0]},{coordinates[goal_crdnts_id, 1]})')
                ready = True

        log_file_orig = f'{output_dir}\log_original_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                       -e 0 -h 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file_orig}'
        os.system(command_line)

        log_file_inf = f'{output_dir}\log_inflated_iter_{i+1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                       -e 0 -h 1 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file_inf}'
        os.system(command_line)

        log_file_inf = f'{output_dir}\log_apex_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                               -e 0.01 -h 0 -s {startNode} -g {goalNode} -a Apex -o output.txt -l {log_file_inf}'
        os.system(command_line)

        # Reading original results
        try:
            with open(log_file_orig, 'r') as file:
                log = json.load(file)
            n_solutions = log[0]['finish_info']['amount_of_solutions']
            A_solutions = np.zeros((n_solutions, 2))
            for i in range(n_solutions):
                A_solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']

            # Reading inflated results
            with open(log_file_inf, 'r') as file:
                log = json.load(file)
            n_solutions = log[0]['finish_info']['amount_of_solutions']
            B_solutions = np.zeros((n_solutions, 2))
            for i in range(n_solutions):
                B_solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']
        except:
            continue
        else:
            A_eps_dominates_by_B = is_approx_pareto_set_bound_true_pareto_set(B_solutions, A_solutions, 0.01)

            if not A_eps_dominates_by_B:
                print('PROBLEM!')
                return

def write_query_file(queries, query_filename):
    with open(query_filename, 'w') as file:
        for i in range(queries.shape[0]):
            file.write("{},{}\n".format(int(queries[i, 0]), int(queries[i, 1])))


def testbench_G(distance_filename, time_filename, coords_filename, output_dir, samples, path2ApexExe,
                epsilon, query_filename, timeout):
    # Loading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]
    coordinates = read_coords_file(coords_filename)

    queries = np.zeros((samples, 2))
    queries[:, 0] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    queries[:, 1] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    write_query_file(queries, query_filename)

    # A*pex WITHOUT heuristics inflation
    log_file = f'{output_dir}\log_A.json'
    command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                   -e {epsilon} -h 0 -q {query_filename} -a Apex -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

    # A*pex WITH heuristics inflation
    log_file = f'{output_dir}\log_B.json'
    command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                   -e 0 -h 1 -q {query_filename} -a Apex -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

    # A*pex baseline (no epsilon and no heuristics inflation)
    log_file = f'{output_dir}\log_C.json'
    command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                       -e 0 -h 0 -q {query_filename} -a Apex -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

def testbench_H(distance_filename, time_filename, coords_filename, output_dir, samples, path2ApexExe,
                epsilon, query_filename, timeout):
    # Loading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]
    coordinates = read_coords_file(coords_filename)

    queries = np.zeros((samples, 2))
    queries[:, 0] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    queries[:, 1] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    write_query_file(queries, query_filename)

    # A*pex with first ordering
    log_file = f'{output_dir}\log_A.json'
    command_line = f'{path2ApexExe}  -m {distance_filename} {time_filename} \
                   -e {epsilon} -h 0 -b 0 -q {query_filename} -a Apex -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

    # A*pex with second ordering
    log_file = f'{output_dir}\log_B.json'
    command_line = f'{path2ApexExe}  -m {time_filename} {distance_filename} \
                   -e {epsilon} -h 0 -b 0 -q {query_filename} -a Apex -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

def testbench_I(distance_filename, time_filename, coords_filename, output_dir, samples, path2ApexExe,
                epsilon, query_filename, timeout):
    # Loading the graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]
    coordinates = read_coords_file(coords_filename)

    if not os.path.exists(query_filename):
        queries = np.zeros((samples, 2))
        queries[:, 0] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
        queries[:, 1] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
        write_query_file(queries, query_filename)

    # A*pex
    log_file = f'{output_dir}\log_Apex_{epsilon}.json'
    command_line = f'{path2ApexExe}  -m {time_filename} {distance_filename} \
                   -e {epsilon} -h 0 -b 0 -q {query_filename} -a Apex -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

    # BOA
    log_file = f'{output_dir}\log_BOA_{epsilon}.json'
    command_line = f'{path2ApexExe}  -m {time_filename} {distance_filename} \
                   -e {epsilon} -h 0 -b 0 -q {query_filename} -a BOA -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)


def analyze_testbench_F(output_dir, samples):
    runtime_res = np.zeros((samples, 2))
    expansions_res = np.zeros((samples, 2))
    generations_res = np.zeros((samples, 2))

    for sample in range(1, samples + 1):
        # Reading the JSON of without contraction
        log_file = f'{output_dir}\log_apex_iter_{sample}.json'
        with open(log_file, 'r') as file:
            log = json.load(file)
        expansions_res[sample - 1, 0] = log[0]['finish_info']['number_of_expansions']
        generations_res[sample - 1, 0] = log[0]['finish_info']['number_of_generations']
        runtime_res[sample - 1, 0] = log[0]['total_runtime(ms)']

        # Reading the JSON of with contraction
        log_file = f'{output_dir}\log_inflated_iter_{sample}.json'
        with open(log_file, 'r') as file:
            log = json.load(file)
        expansions_res[sample - 1, 1] = log[0]['finish_info']['number_of_expansions']
        generations_res[sample - 1, 1] = log[0]['finish_info']['number_of_generations']
        runtime_res[sample - 1, 1] = log[0]['total_runtime(ms)']

    plt.figure()
    nbins = 100
    plt.hist((expansions_res[:, 1] - expansions_res[:, 0]) / expansions_res[:, 0] * 100, bins=nbins)
    plt.xlabel('% Difference of states expansions count (Inflated - A*pex)')
    plt.ylabel('Count')
    plt.title(f'State Expansions Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist((generations_res[:, 1] - generations_res[:, 0]) / generations_res[:, 0] * 100, bins=nbins)
    plt.xlabel('% Difference of generations count (Inflated - A*pex)')
    plt.ylabel('Count')
    plt.title(f'Generations Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist(runtime_res[:, 1] - runtime_res[:, 0], bins=nbins)
    plt.xlabel('Difference of runtime (Inflated - A*pex) [ms]')
    plt.ylabel('Count')
    plt.title(f'Runtime Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.show()

def analyze_testbench_F(output_dir, samples):
    runtime_res = np.zeros((samples, 2))
    expansions_res = np.zeros((samples, 2))
    generations_res = np.zeros((samples, 2))

    for sample in range(1, samples + 1):
        # Reading the JSON of without contraction
        log_file = f'{output_dir}\log_apex_iter_{sample}.json'
        with open(log_file, 'r') as file:
            log = json.load(file)
        expansions_res[sample - 1, 0] = log[0]['finish_info']['number_of_expansions']
        generations_res[sample - 1, 0] = log[0]['finish_info']['number_of_generations']
        runtime_res[sample - 1, 0] = log[0]['total_runtime(ms)']

        # Reading the JSON of with contraction
        log_file = f'{output_dir}\log_inflated_iter_{sample}.json'
        with open(log_file, 'r') as file:
            log = json.load(file)
        expansions_res[sample - 1, 1] = log[0]['finish_info']['number_of_expansions']
        generations_res[sample - 1, 1] = log[0]['finish_info']['number_of_generations']
        runtime_res[sample - 1, 1] = log[0]['total_runtime(ms)']

    plt.figure()
    nbins = 100
    plt.hist((expansions_res[:, 1] - expansions_res[:, 0]) / expansions_res[:, 0] * 100, bins=nbins)
    plt.xlabel('% Difference of states expansions count (Inflated - A*pex)')
    plt.ylabel('Count')
    plt.title(f'State Expansions Difference Histogram, samples count = {samples}')
    plt.grid()


def analyze_apex_performance(output_dir, samples):
    runtime_res = np.zeros((samples, 2))
    expansions_res = np.zeros((samples, 2))
    generations_res = np.zeros((samples, 2))
    for sample in range(1, samples + 1):
        # Reading the JSON of without contraction
        log_file = f'{output_dir}\log_without_contraction_iter_{sample}.json'
        with open(log_file, 'r') as file:
            log = json.load(file)
        expansions_res[sample - 1, 0] = log[0]['finish_info']['number_of_expansions']
        generations_res[sample - 1, 0] = log[0]['finish_info']['number_of_generations']
        runtime_res[sample -1, 0] = log[0]['total_runtime(ms)']

        # Reading the JSON of with contraction
        log_file = f'{output_dir}\log_with_contraction_iter_{sample}.json'
        with open(log_file, 'r') as file:
            log = json.load(file)
        expansions_res[sample - 1, 1] = log[0]['finish_info']['number_of_expansions']
        generations_res[sample - 1, 1] = log[0]['finish_info']['number_of_generations']
        runtime_res[sample - 1, 1] = log[0]['total_runtime(ms)']

    # a = np.argmax(generations_res[:, 1])
    # b = np.argmax(runtime_res[:, 1])

    plt.figure()
    nbins = 100
    plt.hist((expansions_res[:, 1] - expansions_res[:, 0]) / expansions_res[:, 0] * 100, bins= nbins)
    plt.xlabel('% Difference of states expansions count (Augmented - Original)')
    plt.ylabel('Count')
    plt.title(f'State Expansions Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist((generations_res[:, 1] - generations_res[:, 0]) / generations_res[:, 0] * 100, bins=nbins)
    plt.xlabel('% Difference of generations_res count (Augmented - Original)')
    plt.ylabel('Count')
    plt.title(f'generations_res Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist(runtime_res[:, 1] - runtime_res[:, 0], bins= nbins)
    plt.xlabel('Difference of runtime (Augmented - Original) [ms]')
    plt.ylabel('Count')
    plt.title(f'Runtime Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.show()

def analyze_heuristics_inflation(output_dir, samples, epsilon):
    runtime_res = np.zeros((samples, 3))
    expansions_res = np.zeros((samples, 3))
    for sample in range(1, samples + 1):

        try:
            # Reading log of no epsilon and no heuristics inflation (baseline)
            log_file = f'{output_dir}\log_A_iter_{sample}.json'
            with open(log_file, 'r') as file:
                log = json.load(file)
            expansions_res[sample - 1, 0] = log[0]['finish_info']['number_of_expansions']
            runtime_res[sample - 1, 0] = log[0]['total_runtime(ms)']
            n_solutions = log[0]['finish_info']['amount_of_solutions']
            A_solutions = np.zeros((n_solutions, 2))
            for i in range(n_solutions):
                A_solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']

            # Reading log of with epsilon and no heuristics inflation
            log_file = f'{output_dir}\log_B_iter_{sample}.json'
            with open(log_file, 'r') as file:
                log = json.load(file)
            expansions_res[sample - 1, 1] = log[0]['finish_info']['number_of_expansions']
            runtime_res[sample - 1, 1] = log[0]['total_runtime(ms)']
            n_solutions = log[0]['finish_info']['amount_of_solutions']
            B_solutions = np.zeros((n_solutions, 2))
            for i in range(n_solutions):
                B_solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']

            # Reading log of with epsilon and with heuristics inflation
            log_file = f'{output_dir}\log_C_iter_{sample}.json'
            with open(log_file, 'r') as file:
                log = json.load(file)
            expansions_res[sample - 1, 2] = log[0]['finish_info']['number_of_expansions']
            runtime_res[sample - 1, 2] = log[0]['total_runtime(ms)']
            n_solutions = log[0]['finish_info']['amount_of_solutions']
            C_solutions = np.zeros((n_solutions, 2))
            for i in range(n_solutions):
                C_solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']
        except:
            continue
        else:
            print(f'Processing sample {sample}')
            # Asserting that A is epsilon-dominated by B
            A_eps_dominates_by_B = is_approx_pareto_set_bound_true_pareto_set(B_solutions, A_solutions, epsilon)

            if not A_eps_dominates_by_B:
                print('Problem at file ', log_file)
                return

            # Asserting that A is epsilon-dominated by C
            A_eps_dominates_by_C = is_approx_pareto_set_bound_true_pareto_set(C_solutions, A_solutions, epsilon)
            print(A_eps_dominates_by_C)

def approximation_slack(approx_set, precise_set, eps):
    width = []
    for real_sol in precise_set:
        for approx_sol in approx_set:
            if approx_sol[0] <= (1+eps) * real_sol[0] and approx_sol[0] >= real_sol[0]:
                width.append(pareto_width(approx_sol[0], real_sol[0]))
            if approx_sol[1] <= (1 + eps) * real_sol[1] and approx_sol[1] >= real_sol[1]:
                width.append(pareto_width(approx_sol[1], real_sol[1]))

    return 1 - np.array(width).mean() / eps

def is_approx_pareto_set_bound_true_pareto_set(approx_set, precise_set, eps):
    for real_sol in precise_set:
        is_eps_dominated = False
        for approx_sol in approx_set:
            if (approx_sol[0] <= (1+eps) * real_sol[0]) and (approx_sol[1] <= (1 + eps) * real_sol[1]):
                is_eps_dominated = True
                break
        if not is_eps_dominated:
            return False

    return True

def analyze_testbench_G(output_dir, query_file, samples, epsilon, coords_filename):
    runtime_res = np.full((samples, 3), np.nan)
    expansions_res = np.full((samples, 3), np.nan)
    generations_res = np.full((samples, 3), np.nan)
    distance_res = np.zeros(samples)
    start_goal_id_diff =np.zeros(samples)
    approx_slack_res = np.full((samples, 2), np.nan)

    coordinates = read_coords_file(coords_filename)
    queries = read_queries_file(query_file)

    # Reading the JSON file of A*pex+eps without heuristics inflation
    log_A_file = f'{output_dir}\log_A.json'
    with open(log_A_file, 'r') as file:
        logA = json.load(file)

    # Reading the JSON file of A*pex+eps without heuristics inflation
    log_B_file = f'{output_dir}\log_B.json'
    with open(log_B_file, 'r') as file:
        logB = json.load(file)

    # Reading the JSON file of A*pex+eps without heuristics inflation
    log_C_file = f'{output_dir}\log_C.json'
    with open(log_C_file, 'r') as file:
        logC = json.load(file)

    for sample in range(0, samples):
        if logA[sample]['finish_info']['status'] == 'Success':
            expansions_res[sample, 0] = logA[sample]['finish_info']['number_of_expansions']
            generations_res[sample, 0] = logA[sample]['finish_info']['number_of_generations']
            runtime_res[sample, 0] = logA[sample]['total_runtime(ms)']

        if logB[sample]['finish_info']['status'] == 'Success':
            expansions_res[sample, 1] = logB[sample]['finish_info']['number_of_expansions']
            generations_res[sample, 1] = logB[sample]['finish_info']['number_of_generations']
            runtime_res[sample, 1] = logB[sample]['total_runtime(ms)']

        if logC[sample]['finish_info']['status'] == 'Success':
            expansions_res[sample, 2] = logC[sample]['finish_info']['number_of_expansions']
            generations_res[sample, 2] = logC[sample]['finish_info']['number_of_generations']
            runtime_res[sample, 2] = logC[sample]['total_runtime(ms)']

        startNode, goalNode = queries[sample, :]
        distance_res[sample] = np.linalg.norm(coordinates[startNode - 1, 0:2] - coordinates[goalNode - 1, 0:2])
        start_goal_id_diff[sample] = abs(goalNode - startNode)

        # Reading the full pareto-set
        n_solutions = logC[sample]['finish_info']['amount_of_solutions']
        sol_set_C = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_C[i, :] = logC[sample]['finish_info']['solutions'][i]['full_cost']

        # Reading the A*pex approximated pareto-set
        n_solutions = logA[sample]['finish_info']['amount_of_solutions']
        sol_set_A = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_A[i, :] = logA[sample]['finish_info']['solutions'][i]['full_cost']

        # Reading the inflated heuristics approximated pareto-set
        n_solutions = logB[sample]['finish_info']['amount_of_solutions']
        sol_set_B = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_B[i, :] = logB[sample]['finish_info']['solutions'][i]['full_cost']

        # Asserting epsilon-domination
        C_eps_dominates_by_A = is_approx_pareto_set_bound_true_pareto_set(sol_set_A, sol_set_C, epsilon)
        C_eps_dominates_by_B = is_approx_pareto_set_bound_true_pareto_set(sol_set_B, sol_set_C, epsilon)

        if not (C_eps_dominates_by_A and C_eps_dominates_by_B):
            print(f'Warning! sample {sample} has non-dominancy issues!')
        else:
            # Computing approximation slack approx_slack_res
            approx_slack_res[sample, 0] = approximation_slack(sol_set_A, sol_set_C, epsilon)
            approx_slack_res[sample, 1] = approximation_slack(sol_set_B, sol_set_C, epsilon)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Difference Histograms of num. of expansions, num.
    # of generations and runtime difference
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ind = np.where(~np.isnan(expansions_res).any(axis=1))[0]

    plt.figure()
    nbins = 100
    plt.hist(approx_slack_res[ind, 0], bins=nbins)
    plt.xlabel('Approximation slack of A*pex')
    plt.ylabel('Count')
    plt.title('Approximation slack of A*pex')
    plt.grid()

    plt.figure()
    nbins = 100
    plt.hist(approx_slack_res[ind, 1], bins=nbins)
    plt.xlabel('Approximation slack of inflated heuristics')
    plt.ylabel('Count')
    plt.title('Approximation slack of inflated heuristics')
    plt.grid()

    plt.figure()
    nbins = 100
    plt.hist(approx_slack_res[ind, 1] - approx_slack_res[ind, 0], bins=nbins)
    plt.xlabel(r'Approximation Slack Difference normalized by $\varepsilon$ (Inflated - A*pex)')
    plt.ylabel('Count')
    plt.title(f'Approximation Slack Difference')
    plt.grid()

    plt.figure()
    nbins = 1000
    plt.hist(expansions_res[ind, 1] - expansions_res[ind, 0], bins=nbins)
    plt.xlabel('Difference of states expansions count (Inflated - A*pex)')
    plt.ylabel('Count')
    plt.title(f'State Expansions Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist(generations_res[ind, 1] - generations_res[ind, 0], bins=nbins)
    plt.xlabel('Difference of generations count (Inflated - A*pex)')
    plt.ylabel('Count')
    plt.title(f'Generations Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist(runtime_res[ind, 1] - runtime_res[ind, 0], bins=nbins)
    plt.xlabel('Difference of runtime (Inflated - A*pex) [ms]')
    plt.ylabel('Count')
    plt.title(f'Runtime Difference Histogram, samples count = {samples}')
    plt.grid()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Scatter plots of expansion difference, generations
    # diference and runtime difference vs. Euclidean distance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    marker_size = 3
    marker_color = 'red'
    marker = '.'

    plt.figure()
    plt.scatter(distance_res[ind], expansions_res[ind, 1] - expansions_res[ind, 0], color=marker_color, marker=marker, s=marker_size)
    plt.xlabel('Euclidean Distance [m]')
    plt.ylabel('Difference of states expansions count (Inflated - A*pex)')
    plt.title(f'State Expansions Difference vs. Euclidean distance')
    plt.grid()

    plt.figure()
    plt.scatter(distance_res[ind], generations_res[ind, 1] - generations_res[ind, 0], color=marker_color, marker=marker,
                s=marker_size)
    plt.xlabel('Euclidean Distance [m]')
    plt.ylabel('Difference of generations count (Inflated - A*pex)')
    plt.title(f'Generations Difference vs. Euclidean distance')
    plt.grid()

    plt.figure()
    plt.scatter(distance_res[ind], runtime_res[ind, 1] - runtime_res[ind, 0], color=marker_color, marker=marker,
                s=marker_size)
    plt.xlabel('Euclidean Distance [m]')
    plt.ylabel('Runtime difference [ms] (Inflated - A*pex)')
    plt.title(f'Runtime Difference vs. Euclidean distance')
    plt.grid()

    plt.figure()
    plt.scatter(start_goal_id_diff[ind], runtime_res[ind, 1] - runtime_res[ind, 0], color=marker_color, marker=marker,
                s=marker_size)
    plt.xlabel('start_goal_id_diff')
    plt.ylabel('Runtime difference [ms] (Inflated - A*pex)')
    plt.title(f'Runtime Difference vs. start_goal_id_diff')
    plt.grid()

    plt.show()

def analyze_apex_vs_boa(samples, epsilon_ar, output_dirs):
    n_eps = len(epsilon_ar)
    n_dirs = len(output_dirs)
    runtime_res = np.full((n_dirs, n_eps, samples, 2), np.nan)
    expansions_res = np.full((n_dirs, n_eps, samples, 2), np.nan)
    generations_res = np.full((n_dirs, n_eps, samples, 2), np.nan)

    for i_output_dir in range(n_dirs):
        for i_eps in range(n_eps):
            output_dir = output_dirs[i_output_dir]
            epsilon = epsilon_ar[i_eps]
            log_A_filename = f'{output_dir}\log_Apex_{epsilon}.json'
            with open(log_A_filename, 'r') as file:
                logA = json.load(file)

            log_B_filename = f'{output_dir}\log_BOA_{epsilon}.json'
            with open(log_B_filename, 'r') as file:
                logB = json.load(file)

            for sample in range(0, samples):
                if logA[sample]['finish_info']['status'] == 'Success':
                    expansions_res[i_output_dir, i_eps, sample, 0] = logA[sample]['finish_info']['number_of_expansions']
                    generations_res[i_output_dir, i_eps, sample, 0] = logA[sample]['finish_info']['number_of_generations']
                    runtime_res[i_output_dir, i_eps, sample, 0] = max(1, logA[sample]['total_runtime(ms)'])

                if logB[sample]['finish_info']['status'] == 'Success':
                    expansions_res[i_output_dir, i_eps, sample, 1] = logB[sample]['finish_info']['number_of_expansions']
                    generations_res[i_output_dir, i_eps, sample, 1] = logB[sample]['finish_info']['number_of_generations']
                    runtime_res[i_output_dir, i_eps, sample, 1] = max(1, logB[sample]['total_runtime(ms)'])

    # Comparing Lex1 vs Lex2
    eps_i = 1
    plt.figure()
    plt.subplot(1, 2, 1)
    delta_time = expansions_res[0, eps_i, :, 0] / expansions_res[1, eps_i, :, 0]
    plt.plot(delta_time)
    plt.plot([0, samples], [1, 1], linestyle='--', color='black')
    plt.grid()
    plt.title(f'Running time ratio of LEX1/LEX2 for A*pex (eps = {epsilon_ar[eps_i]})')
    plt.xlabel('Sample')
    plt.ylabel('Running time ratio (LEX1/LEX2)')

    plt.subplot(1, 2, 2)
    delta_time = expansions_res[0, eps_i, :, 1] / expansions_res[1, eps_i, :, 1]
    plt.plot(delta_time)
    plt.plot([0, samples], [1, 1], linestyle='--', color='black')
    plt.grid()
    plt.title(f'Running time ratio of LEX1/LEX2 for BOA* (eps = {epsilon_ar[eps_i]})')
    plt.xlabel('Sample')
    plt.ylabel('Running time ratio (LEX1/LEX2)')
    plt.tight_layout()

    marker_size = 50
    line_styles = ['-', '--']
    # Number of nodes expansions
    plt.figure()
    for i_output_dir in range(n_dirs):
        plt.scatter(epsilon_ar, np.nanmean(expansions_res[i_output_dir, :, :, 0], axis=1),
                    facecolors='none', edgecolors='blue', marker='o', s=marker_size)
        plt.plot(epsilon_ar, np.nanmean(expansions_res[i_output_dir, :, :, 0], axis=1),
                    color='blue', label=f'A*pex LEX{i_output_dir+1}', linestyle=line_styles[i_output_dir])
        plt.scatter(epsilon_ar, np.nanmean(expansions_res[i_output_dir, :, :, 1], axis=1),
                    facecolors='none', edgecolors='red', marker='^', s=marker_size)
        plt.plot(epsilon_ar, np.nanmean(expansions_res[i_output_dir, :, :, 1], axis=1),
                 color='red', label=f'BOA* LEX{i_output_dir+1}', linestyle=line_styles[i_output_dir])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('Number of queue expansions')
    plt.title(f'State Expansions A*pex vs BOA*')
    plt.grid(True)
    plt.legend()

    # Number of queue generations
    plt.figure()
    for i_output_dir in range(n_dirs):
        plt.scatter(epsilon_ar, np.nanmean(generations_res[i_output_dir, :, :, 0], axis=1),
                    facecolors='none', edgecolors='blue', marker='o', s=marker_size)
        plt.plot(epsilon_ar, np.nanmean(generations_res[i_output_dir, :, :, 0], axis=1),
                 color='blue', label=f'A*pex LEX{i_output_dir + 1}', linestyle=line_styles[i_output_dir])
        plt.scatter(epsilon_ar, np.nanmean(generations_res[i_output_dir, :, :, 1], axis=1),
                    facecolors='none', edgecolors='red', marker='^', s=marker_size)
        plt.plot(epsilon_ar, np.nanmean(generations_res[i_output_dir, :, :, 1], axis=1),
                 color='red', label=f'BOA* LEX{i_output_dir + 1}', linestyle=line_styles[i_output_dir])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('Number of queue generations')
    plt.title(f'Queue generations A*pex vs BOA*')
    plt.grid(True)
    plt.legend()

    # Running time
    plt.figure()
    for i_output_dir in range(n_dirs):
        plt.scatter(epsilon_ar, np.nanmean(runtime_res[i_output_dir, :, :, 0], axis=1),
                    facecolors='none', edgecolors='blue', marker='o', s=marker_size)
        plt.plot(epsilon_ar, np.nanmean(runtime_res[i_output_dir, :, :, 0], axis=1),
                 color='blue', label=f'A*pex LEX{i_output_dir + 1}', linestyle=line_styles[i_output_dir])
        plt.scatter(epsilon_ar, np.nanmean(runtime_res[i_output_dir, :, :, 1], axis=1),
                    facecolors='none', edgecolors='red', marker='^', s=marker_size)
        plt.plot(epsilon_ar, np.nanmean(runtime_res[i_output_dir, :, :, 1], axis=1),
                 color='red', label=f'BOA* LEX{i_output_dir + 1}', linestyle=line_styles[i_output_dir])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('Running time [ms]')
    plt.title(f'Running Time A*pex vs BOA*')
    plt.grid(True)
    plt.legend()

    plt.show()

def test_triangle_inequality(gr_filename):
    graph, vertices_count = load_graph(gr_filename)
    flag = True
    while flag:
        vertex = np.random.choice(graph[:, 0])
        ind = np.array(np.where(graph[:, 0] == vertex))
        ind = ind.reshape(ind.shape[1])
        if len(ind) > 1:
            # abs(AC) <= abs(AB) + abs(BC)
            A = vertex
            targets = set(graph[ind, 1].astype(int))
            B = np.random.choice(list(targets))
            targets.discard(B)
            if len(targets) == 0:
                continue
            C = np.random.choice(list(targets))
            targets.discard(C)

            # Check if B and C are linked
            BC_a = graph[:, 0] == B
            BC_b = graph[:, 1] == C
            BC = np.array(np.where(np.logical_and(BC_a, BC_b)))
            if BC.any() > 0:
                BC = BC.reshape(BC.shape[1])

                AB_a = graph[:, 0] == A
                AB_b = graph[:, 1] == B
                AB = np.array(np.where(np.logical_and(AB_a, AB_b)))
                AB = AB.reshape(AB.shape[1])

                AC_a = graph[:, 0] == A
                AC_b = graph[:, 1] == C
                AC = np.array(np.where(np.logical_and(AC_a, AC_b)))
                AC = AC.reshape(AC.shape[1])

                # check inequality
                if (graph[AB[0], 2] + graph[BC[0], 2]) < (graph[AC[0], 2] - 10):
                    print('Triangle Inequality is not satisfied!')
                    print(f'AB = {graph[AB[0], 2]}, BC = {graph[BC[0], 2]}, AC = {graph[AC[0], 2]}')
                    print('gap = ', graph[AC[0], 2] - (graph[AB[0], 2] + graph[BC[0], 2]))
                    flag = False
                else:
                    print('.')

def correlation_graph_preprocessing(distance_filename, time_filename, coords_filename,
                              new_distance_filename, new_time_filename,
                              new_coords_filename, clusters_metafile, contracted_edges_file):

    """
    Performs a full graph preprocessing for edge contraction.
    Steps:
    1. Connected Components Analysis for correlated clusters
    2. Generating a clustering report file, used by the Dijkstra C++
    3. Edge contraction by running all-pairs Dijkstra

    """

def generate_contracted_graph(distance_filename, time_filename, coords_filename,
                              new_distance_filename, new_time_filename,
                              new_coords_filename, clusters_metafile, contracted_edges_file):

    # Read original graph
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    coordinates = read_coords_file(coords_filename)

    # Read clusters metafile
    correlated_nodes = []
    node_ind = 0
    clusters_ids = []
    with open(clusters_metafile, "r") as f:
        for line in f:
            if len(correlated_nodes) == 0:
                nodes_count = int(line.rstrip('\n'))
                correlated_nodes = np.zeros((nodes_count, 2))
                continue

            current_cluster_id, node = line.rstrip('\n').split(' ')
            if current_cluster_id not in clusters_ids:
                clusters_ids.append(current_cluster_id)
            correlated_nodes[node_ind, :] = [int(current_cluster_id), int(node)]
            node_ind += 1

    # Reading contracted edges output file
    contracted_edges = []
    edge_ind = 0
    with open(contracted_edges_file, "r") as f:
        for line in f:
            if len(contracted_edges) == 0:
                edges_count = int(line.rstrip('\n'))
                contracted_edges = np.zeros((edges_count, 6))
                continue

            source, target, cost1, apex1, cost2, apex2 = line.rstrip('\n').split(',')
            contracted_edges[edge_ind, :] = [int(source), int(target), int(cost1), int(apex1), int(cost2), int(apex2)]
            edge_ind += 1

    # Creating the new coordinates table : keeping vertices that either NOT part of any cluster or they're some cluster boundary nodes
    outside_clusters_ind = np.where(~np.isin(coordinates[:, 2], correlated_nodes[:, 1]))
    outside_clusters_ind = np.array(outside_clusters_ind[0])
    boundary_ind = np.where(np.isin(coordinates[:, 2], contracted_edges[:, 0]))
    boundary_ind = np.array(boundary_ind[0])
    nodes_to_keep = np.hstack((outside_clusters_ind, boundary_ind))
    new_coordinates = coordinates[nodes_to_keep, :]

    # Removing all edges that both start and target vertices and within some cluster
    source_is_part_of_cluster = np.isin(c1_graph[:, 0], correlated_nodes[:, 1])
    target_is_part_of_cluster = np.isin(c1_graph[:, 1], correlated_nodes[:, 1])
    edges_to_keep = ~np.logical_and(source_is_part_of_cluster, target_is_part_of_cluster)
    new_c1_graph = c1_graph[edges_to_keep, :]
    new_c2_graph = c2_graph[edges_to_keep, :]

    # Add a column of -1 for the edge apex as default for regular edges
    new_c1_graph = np.hstack((new_c1_graph, np.full((new_c1_graph.shape[0], 1), -1)))
    new_c2_graph = np.hstack((new_c2_graph, np.full((new_c2_graph.shape[0], 1), -1)))

    # Adding the contracted edges
    index_of_first_super_edge = new_c1_graph.shape[0] - 1
    new_c1_graph = np.vstack((new_c1_graph, contracted_edges[:, [0, 1, 2, 3]]))
    new_c2_graph = np.vstack((new_c2_graph, contracted_edges[:, [0, 1, 4, 5]]))

    # Exporting new files
    edges_count = new_c2_graph.shape[0]
    vertices_count = new_coordinates.shape[0]
    export_gr_file(new_c1_graph, vertices_count, edges_count, new_distance_filename,
                   'Distance graph after inner-cluster edge contraction', index_of_first_super_edge)
    export_gr_file(new_c2_graph, vertices_count, edges_count, new_time_filename,
                   'Time graph after inner-cluster edge contraction', index_of_first_super_edge)
    export_coords_file(new_coordinates, vertices_count, new_coords_filename,
                       'Coords table after inner-cluster edge contraction')


def analyze_testbench_H(query_file, samples, epsilon, coords_filename,
                        log_A_filename, log_B_filename):
    runtime_res = np.full((samples, 2), np.nan)
    expansions_res = np.full((samples, 2), np.nan)
    generations_res = np.full((samples, 2), np.nan)
    distance_res = np.zeros(samples)
    start_goal_id_diff = np.zeros(samples)
    approx_slack_res = np.full((samples, 2), np.nan)

    coordinates = read_coords_file(coords_filename)
    queries = read_queries_file(query_file)

    # log_A_file = f'{output_dir}\log_A.json'
    with open(log_A_filename, 'r') as file:
        logA = json.load(file)

    # log_B_file = f'{output_dir}\log_B.json'
    with open(log_B_filename, 'r') as file:
        logB = json.load(file)

    for sample in range(0, samples):
        if logA[sample]['finish_info']['status'] == 'Success':
            expansions_res[sample, 0] = logA[sample]['finish_info']['number_of_expansions']
            generations_res[sample, 0] = logA[sample]['finish_info']['number_of_generations']
            runtime_res[sample, 0] = max(1,logA[sample]['total_runtime(ms)'])

        if logB[sample]['finish_info']['status'] == 'Success':
            expansions_res[sample, 1] = logB[sample]['finish_info']['number_of_expansions']
            generations_res[sample, 1] = logB[sample]['finish_info']['number_of_generations']
            runtime_res[sample, 1] = max(1,logB[sample]['total_runtime(ms)'])

        startNode, goalNode = queries[sample, :]
        distance_res[sample] = np.linalg.norm(coordinates[startNode - 1, 0:2] - coordinates[goalNode - 1, 0:2])
        start_goal_id_diff[sample] = abs(goalNode - startNode)

        # Reading the A*pex approximated pareto-set
        n_solutions = logA[sample]['finish_info']['amount_of_solutions']
        sol_set_A = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_A[i, :] = logA[sample]['finish_info']['solutions'][i]['full_cost']

        # Reading the inflated heuristics approximated pareto-set
        n_solutions = logB[sample]['finish_info']['amount_of_solutions']
        sol_set_B = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_B[i, :] = logB[sample]['finish_info']['solutions'][i]['full_cost']

        # Asserting epsilon-domination
        if sol_set_A.shape[0] != sol_set_B.shape[0]:
            print(f'Warning!')

        # if abs(runtime_res[sample, 0] - runtime_res[sample, 1]) > 3000:
        #     if runtime_res[sample, 0] < runtime_res[sample, 1]:
        #         print('Lex1 is superior')
        #     else:
        #         print('Lex2 is superior')
        #         plot_pareto_set(sol_set_A)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Difference Histograms of num. of expansions, num.
    # of generations and runtime difference
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ind = np.where(~np.isnan(expansions_res).any(axis=1))[0]

    plt.figure()
    nbins = 100
    plt.hist((expansions_res[ind, 0] - expansions_res[ind, 1])/expansions_res[ind, 1] * 100, bins=nbins)
    plt.xlabel('Difference of states expansions count')
    plt.ylabel('Count')
    plt.title(f'State Expansions Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist((generations_res[ind, 0] - generations_res[ind, 1])/generations_res[ind, 1]*100, bins=nbins)
    plt.xlabel('Difference of generations count')
    plt.ylabel('Count')
    plt.title(f'Generations Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.figure()
    plt.hist((runtime_res[ind, 0] - runtime_res[ind, 1])/runtime_res[ind, 1] * 100, bins=nbins)
    plt.xlabel('Difference of runtime [ms]')
    plt.ylabel('Count')
    plt.title(f'Runtime Difference Histogram, samples count = {samples}')
    plt.grid()

    plt.show()

def test_yaron(output_dir, epsilon, samples):
    for sample in range(0, samples):
        # Reading the JSON file of A*pex+eps without heuristics inflation
        log_A_file = f'{output_dir}\log_apex_iter_{sample + 1}.json'
        with open(log_A_file, 'r') as file:
            logA = json.load(file)

        # Reading the JSON file of A*pex+eps without heuristics inflation
        log_B_file = f'{output_dir}\log_inflated_iter_{sample + 1}.json'
        with open(log_B_file, 'r') as file:
            logB = json.load(file)

        # Reading the JSON file of A*pex+eps without heuristics inflation
        log_C_file = f'{output_dir}\log_original_iter_{sample + 1}.json'
        with open(log_C_file, 'r') as file:
            logC = json.load(file)

        # Reading the full pareto-set
        n_solutions = logC[0]['finish_info']['amount_of_solutions']
        sol_set_C = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_C[i, :] = logC[0]['finish_info']['solutions'][i]['full_cost']

        # Reading the A*pex approximated pareto-set
        n_solutions = logA[0]['finish_info']['amount_of_solutions']
        sol_set_A = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_A[i, :] = logA[0]['finish_info']['solutions'][i]['full_cost']

        # Reading the inflated heuristics approximated pareto-set
        n_solutions = logB[0]['finish_info']['amount_of_solutions']
        sol_set_B = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_B[i, :] = logB[0]['finish_info']['solutions'][i]['full_cost']

        # Asserting epsilon-domination
        C_eps_dominates_by_A = is_approx_pareto_set_bound_true_pareto_set(sol_set_A, sol_set_C, epsilon)
        C_eps_dominates_by_B = is_approx_pareto_set_bound_true_pareto_set(sol_set_B, sol_set_C, epsilon)

        if not (C_eps_dominates_by_A and C_eps_dominates_by_B):
            print('PROBLEM!')
            # return

def test_PFs(output_dir, log_name_A, log_name_B, log_name_C, epsilon, samples):
    for iter in range(samples):
        # Reference (exact Pareto Frontier)
        log_C_file = rf'{output_dir}\{log_name_C}_iter_{iter + 1}.json'
        with open(log_C_file, 'r') as file:
            logC = json.load(file)

        # Approximate PF 1
        log_B_file = rf'{output_dir}\{log_name_B}_iter_{iter+1}.json'
        with open(log_B_file, 'r') as file:
            logB = json.load(file)

        # Approximate PF 1
        log_A_file = rf'{output_dir}\{log_name_A}_iter_{iter + 1}.json'
        with open(log_A_file, 'r') as file:
            logA = json.load(file)

        # Reading the full pareto-set
        n_solutions = logC[0]['finish_info']['amount_of_solutions']
        sol_set_C = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_C[i, :] = logC[0]['finish_info']['solutions'][i]['full_cost']

        # Reading the Approx-PF-1
        n_solutions = logA[0]['finish_info']['amount_of_solutions']
        sol_set_A = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_A[i, :] = logA[0]['finish_info']['solutions'][i]['full_cost']

        # Reading the Approx-PF-2
        n_solutions = logB[0]['finish_info']['amount_of_solutions']
        sol_set_B = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            sol_set_B[i, :] = logB[0]['finish_info']['solutions'][i]['full_cost']

        # Asserting epsilon-domination
        C_eps_dominated_by_A = is_approx_pareto_set_bound_true_pareto_set(sol_set_A, sol_set_C, epsilon)
        C_eps_dominated_by_B = is_approx_pareto_set_bound_true_pareto_set(sol_set_B, sol_set_C, epsilon)

        if not (C_eps_dominated_by_A and C_eps_dominated_by_B):
            print('PROBLEM!')
            # return


def add_contracted_edges_to_graphs(original_distance_filename, original_time_filename,
                                   new_distance_filename, new_time_filename,
                                   contracted_edges_file, augmentation_ratio= 1):
    # Load original graphs
    c1_graph, vertices_count = load_graph(original_distance_filename)
    c2_graph, _ = load_graph(original_time_filename)
    edges_count = c1_graph.shape[0]
    contracted_edges_count = -1
    edge_ind = -1
    first_cntrcted_edge_id = edges_count

    # Reading the contracted edges data
    with open(contracted_edges_file, "r") as f:
        for line in f:
            if contracted_edges_count < 0:
                # Inputting the graph dimensions and allocating data structures
                contracted_edges_count = line.rstrip('\n')
                contracted_edges_count = int(contracted_edges_count)
                new_c1_graph = np.zeros((edges_count + contracted_edges_count, 3))
                new_c2_graph = np.zeros((edges_count + contracted_edges_count, 3))
                new_c1_graph[0:edges_count, :] = c1_graph
                new_c2_graph[0:edges_count, :] = c2_graph
                edge_ind = edges_count
                edges_count += contracted_edges_count
                continue

            # Adding the contracted edge to distance and time graphs
            start_node, target_node, cost1, cost2  = line.rstrip('\n').split(',')
            new_c1_graph[edge_ind ,0] = int(start_node)
            new_c1_graph[edge_ind, 1] = int(target_node)
            new_c1_graph[edge_ind, 2] = int(cost1)

            new_c2_graph[edge_ind, 0] = int(start_node)
            new_c2_graph[edge_ind, 1] = int(target_node)
            new_c2_graph[edge_ind, 2] = int(cost2)

            edge_ind += 1

    # Diluting the number of augmented edges if necessary
    if augmentation_ratio < 1:
        cntrcted_edges_to_add = round(contracted_edges_count * augmentation_ratio)
        cntrcted_edges_ids = np.random.choice(np.arange(first_cntrcted_edge_id, edges_count),
                                              size=cntrcted_edges_to_add, replace=False)
        regular_edges_ids = np.arange(0, first_cntrcted_edge_id)
        indices = np.hstack((regular_edges_ids, cntrcted_edges_ids))
        new_c1_graph = new_c1_graph[indices, :]
        new_c2_graph = new_c2_graph[indices, :]
        edges_count = len(indices)

    # Writing new graph files (augmented with contracted edges)
    export_gr_file(new_c1_graph, vertices_count, edges_count, new_distance_filename,
                   'Distance graph augmented with contracted edges')
    export_gr_file(new_c2_graph, vertices_count, edges_count, new_time_filename,
                   'Time graph for augmented with contracted edges')

def plot_correlations(distance_filename, time_filename):
    # Load original graphs
    c1_graph, vertices_count = load_graph(distance_filename)
    c2_graph, _ = load_graph(time_filename)
    edges_count = c1_graph.shape[0]

    plt.figure()
    plt.scatter(np.arange(edges_count), c2_graph[: ,2] / (c1_graph[:, 2]+100),
                marker='.', s=0.5)
    plt.grid()

    plt.figure()
    plt.hist(c2_graph[: ,2] / (c1_graph[:, 2] + 100))
    plt.grid()
    plt.show()

def load_optimal_paths(paths_filename):
    paths = {}
    with open(paths_filename, "r") as f:
        line = f.readline()
        n_paths = int(line.rstrip('\n'))
        for i in range(n_paths):
            line = f.readline()
            n_steps = int(line.rstrip('\n'))
            path = []
            for j in range(n_steps):
                line = f.readline()
                path.append(int(line.rstrip('\n')))
            paths[i] = path

    return paths

def plot_trajectories(coords_filename, paths_filename, boundary_nodes_file=None):
    global current_index
    # Read data
    coordinates = read_coords_file(coords_filename)
    paths = load_optimal_paths(paths_filename)
    if boundary_nodes_file is not None:
        bnd_nodes_id = read_boundary_nodes(boundary_nodes_file)

    # Converting nodes ids to coordinates
    current_index = 0
    trajectories = {}
    for i in range(len(paths)):
        trajectory = []
        for j in range(len(paths[i])):
            ind = np.where(coordinates[:, 2] == paths[i][j])
            ind = np.array(ind[0])
            if len(trajectory) == 0:
                trajectory = coordinates[ind, 0:2]
            else:
                trajectory = np.vstack((trajectory, coordinates[ind, 0:2]))
        trajectories[i] = trajectory

    # Initialize the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom space to fit the buttons

    # Function to update the plot
    def plot_trajectory():
        index = current_index
        ax.clear()
        ax.scatter(coordinates[:, 0], coordinates[:, 1], color='purple', marker='.', s=1)
        ax.plot(trajectories[index][:, 0], trajectories[index][:, 1], linewidth=3)
        if boundary_nodes_file is not None:
            # Retrieving the coordinates row corresponding to the clusters' nodes ids
            bnd_ind = np.where(np.isin(coordinates[:, 2], bnd_nodes_id))
            bnd_ind = np.array(bnd_ind[0])
            ax.scatter(coordinates[bnd_ind, 0], coordinates[bnd_ind, 1],
                        color='green', marker='.', s=50, label='Boundary Nodes')

        ax.set_title(f"Trajectory {index + 1}")
        ax.axis('equal')
        ax.grid()
        plt.draw()

    # Initial plot
    plot_trajectory()

    # Button click functions
    def next_trajectory(event):
        global current_index
        current_index = (current_index + 1) % len(trajectories)
        plot_trajectory()

    def prev_trajectory(event):
        global current_index
        current_index = (current_index - 1) % len(trajectories)
        plot_trajectory()

    # Add buttons
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next_trajectory)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(prev_trajectory)

    plt.show()

def plot_pareto_set(solutions):
    plt.scatter(solutions[:, 0] / max(solutions[:, 0]), solutions[:, 1] / max(solutions[:, 1]), color='blue', marker='o')

    # Adding labels and title
    plt.xlabel('Distance cost')
    plt.ylabel('Time cost')
    plt.title('Set of Pareto-Optimal solutions')

    plt.grid()

    plt.show()

def Alg_vs_Alg_query_mode(distance_filename_orig, time_filename_orig,
               distance_filename_contr, time_filename_contr,
               coords_filename_orig, coords_filename_contr,
               alg_str_orig, alg_str_contr,
               log_name_orig, log_name_contr,
               output_dir, samples, path2ApexExe, epsilon, query_filename, timeout):

    # Reading coordinates data file
    coordinates = read_coords_file(coords_filename_contr)

    # Generating the random query and storing it to file
    queries = np.zeros((samples, 2))
    queries[:, 0] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    queries[:, 1] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    write_query_file(queries, query_filename)

    # Executing original algorithm
    log_file = rf'{output_dir}\{log_name_orig}.json'
    command_line = f'{path2ApexExe}  -m {distance_filename_orig} {time_filename_orig} \
                   -e {epsilon} -h 0 -b 0 -q {query_filename} -a {alg_str_orig} -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

    # Executing contracted algorithm
    log_file = rf'{output_dir}\{log_name_contr}.json'
    command_line = f'{path2ApexExe}  -m {distance_filename_contr} {time_filename_contr} \
                   -e {epsilon} -h 0 -b 0 -q {query_filename} -a {alg_str_contr} -o output.txt -l {log_file} -t {timeout}'
    os.system(command_line)

def Alg_vs_Alg_single_mode(distance_filename_orig, time_filename_orig,
               distance_filename_contr, time_filename_contr,
               coords_filename_orig, coords_filename_contr,
               alg_str_orig, alg_str_contr,
               log_name_orig, log_name_contr,
               output_dir, samples, path2ApexExe, epsilon, query_filename, timeout):

    # Reading coordinates data file
    coordinates = read_coords_file(coords_filename_contr)

    # Generating the random query and storing it to file
    queries = np.zeros((samples, 2))
    queries[:, 0] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    queries[:, 1] = np.random.choice(coordinates[:, 2], size=samples, replace=False)
    write_query_file(queries, query_filename)

    for i in range(samples):
        startNode = queries[i, 0].astype(int)
        goalNode = queries[i, 1].astype(int)

        # Executing original algorithm
        log_file = rf'{output_dir}\{log_name_orig}_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename_orig} {time_filename_orig} \
                       -e {epsilon} -s {startNode} -g {goalNode} -h 0 -b 0 -a {alg_str_orig} -o output.txt -l {log_file} -t {timeout}'
        os.system(command_line)

        # Executing contracted algorithm
        log_file = rf'{output_dir}\{log_name_contr}_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename_contr} {time_filename_contr} \
                               -e {epsilon} -s {startNode} -g {goalNode} -h 0 -b 0 -a {alg_str_contr} -o output.txt -l {log_file} -t {timeout}'
        os.system(command_line)

def compute_exact_pareto_frontier(distance_filename_orig, time_filename_orig,
                                  log_name_orig, output_dir, path2ApexExe, query_filename, timeout):

    # Generating the random query and storing it to file
    queries = read_queries_file(query_filename)

    for i in range(queries.shape[0]):
        startNode = queries[i, 0].astype(int)
        goalNode = queries[i, 1].astype(int)

        # Executing A*pex algorithm with zero epsilon for exact PF computation
        log_file = rf'{output_dir}\{log_name_orig}_iter_{i + 1}.json'
        command_line = f'{path2ApexExe}  -m {distance_filename_orig} {time_filename_orig} \
                               -e 0 -s {startNode} -g {goalNode} -h 0 -b 0 -a Apex -o output.txt -l {log_file} -t {timeout}'
        os.system(command_line)

def generate_strings_for_DIMACS_instance(dimacs_root_path, instance):
    # original data files
    distance_filename_orig = fr'{dimacs_root_path}\{instance}\USA-road-d.{instance}.gr'
    time_filename_orig = fr'{dimacs_root_path}\{instance}\USA-road-t.{instance}.gr'
    coords_filename_orig = fr'{dimacs_root_path}\{instance}\USA-road-d.{instance}.co'

    # contracted data files
    distance_filename_contr = fr'{dimacs_root_path}\{instance}\USA-road-d.{instance}_contracted.gr'
    time_filename_contr = fr'{dimacs_root_path}\{instance}\USA-road-t.{instance}_contracted.gr'
    coords_filename_contr = fr'{dimacs_root_path}\{instance}\USA-road-d.{instance}_contracted.co'

    return distance_filename_orig, time_filename_orig, coords_filename_orig, \
        distance_filename_contr, time_filename_contr, coords_filename_contr


class LogAnalysis(object):
    def __init__(self, filename):
        with open(filename, 'r') as file:
            log = json.load(file)
        n_solutions = log[0]['finish_info']['amount_of_solutions']
        self.solutions = np.zeros((n_solutions, 2))
        for i in range(n_solutions):
            self.solutions[i, :] = log[0]['finish_info']['solutions'][i]['full_cost']

    def plot_pareto_set(self):
        plt.scatter(self.solutions[:, 0] / max(self.solutions[:, 0]), self.solutions[:, 1] / max(self.solutions[:, 1]), color='blue', marker='o')

        # Adding labels and title
        plt.xlabel('Distance cost')
        plt.ylabel('Time cost')
        plt.title('Set of Pareto-Optimal solutions')

        plt.grid()

        plt.show()



