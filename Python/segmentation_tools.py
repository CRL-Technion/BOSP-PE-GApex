import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay
from sklearn.cluster import SpectralClustering
from RANSAC_Multiple_Line_Detection import *
import networkx as nx
import community as community_louvain
import time
import random


def generate_graph(num_vertices, branching_factor):
    points = np.random.rand(num_vertices, 2)
    delaunay = Delaunay(points)
    branching_stats = np.zeros(num_vertices)
    neighbors = {i: set() for i in range(num_vertices)}

    for simplex in delaunay.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                pt1, pt2 = simplex[i], simplex[j]
                distance = np.linalg.norm(points[pt1] - points[pt2])
                # Add random positive noise to the distance
                noise = np.random.uniform(0.1, 0.5)
                distance += noise
                neighbors[pt1].add((distance, pt2))
                neighbors[pt2].add((distance, pt1))

    for i in range(num_vertices):
        while len(neighbors[i]) < branching_factor:
            potential_neighbor = random.randint(0, num_vertices - 1)
            if potential_neighbor != i and potential_neighbor not in neighbors[i]:
                # Compute distance to potential neighbor
                distance = np.linalg.norm(points[i] - points[potential_neighbor])
                # Add random positive noise to the distance
                noise = 0 # np.random.uniform(0.01, 0.1)
                distance += noise
                neighbors[i].add((distance, potential_neighbor))
                neighbors[potential_neighbor].add((distance, i))

    edges = []
    for i in range(num_vertices):
        if len(neighbors[i]) > branching_factor:
            neighbors[i] = random.choices(list(neighbors[i]), k=branching_factor)
        branching_stats[i] = len(neighbors[i])
        for distance, neighbor in neighbors[i]:
            edges.append((i, neighbor, distance))

    return points, edges, branching_stats


def save_to_files(points, edges, vertices_file, edges_file):
    with open(vertices_file, 'w') as vf:
        vf.write(f'{points.shape[0]}\n')
        for i, (x, y) in enumerate(points):
            vf.write(f"{i} {x} {y}\n")

    with open(edges_file, 'w') as ef:
        ef.write(f'{len(edges)}\n')
        for src, tgt, distance in edges:
            ef.write(f"{src} {tgt} {distance}\n")


def find_connected_components_in_graph(ratio_graph, desired_ratio, tolerance):
    nodes = set()

    # Get the unique nodes
    sources = np.unique(ratio_graph[:, 0])

    # Scan over unique source nodes and test for ratio coherency
    next_counter_report = 0
    counter = 0
    progress = 0
    for source in sources:
        if progress >= next_counter_report:
            print(f'{round(progress)}%')
            next_counter_report += 1

        counter += 1
        progress = counter / len(sources) * 100
        ind = np.where(ratio_graph[:, 0] == source)
        ind = np.array(ind[0])
        if not np.any(abs(ratio_graph[ind, 2] - desired_ratio) > tolerance):
            nodes.add(source)

    # Create the undirected graph that will be used to identify the connected components
    G = nx.Graph()
    for i in range(ratio_graph.shape[0]):
        source = ratio_graph[i, 0].astype(int)
        target = ratio_graph[i, 1].astype(int)
        if source in nodes and target in nodes:
            G.add_edge(source, target)

    # Find connected components in each sub-graph
    components = list(nx.connected_components(G))
    partition = {}
    for i in range(len(components)):
        for node in list(components[i]):
            partition[node] = i

    return partition

def find_connected_components_in_graph_enhanced(ratio_graph, desired_ratio, tolerance):

    max_vertex_id = max(np.hstack((ratio_graph[:, 0], ratio_graph[:, 1]))).astype(int)
    potential_nodes = -1 * np.ones(max_vertex_id + 1)

    # Scan over edges and test for ratio coherency
    print('Beginning scan of edges cost ratio coherency...')
    next_counter_report = 0
    progress = 0
    for i in range(ratio_graph.shape[0]):
        if progress >= next_counter_report:
            print(f'{round(progress)}%')
            next_counter_report += 1
        progress = i / ratio_graph.shape[0] * 100

        source = ratio_graph[i, 0].astype(int)
        if potential_nodes[source] == 0:
            continue
        edge_cost = ratio_graph[i, 2]
        if abs(edge_cost - desired_ratio) < tolerance:
            potential_nodes[source] = 1
        else:
            potential_nodes[source] = 0

    # Create the undirected graph that will be used to identify the connected components
    print('Beginning of NetworkX graph population...')
    G = nx.Graph()
    next_counter_report = 0
    progress = 0
    for i in range(ratio_graph.shape[0]):
        if progress >= next_counter_report:
            print(f'{round(progress)}%')
            next_counter_report += 1
        progress = i / ratio_graph.shape[0] * 100

        source = ratio_graph[i, 0].astype(int)
        target = ratio_graph[i, 1].astype(int)
        if potential_nodes[source] == 1 and potential_nodes[target] == 1:
            G.add_edge(source, target)

    # Find connected components in each sub-graph
    print('Beginning of connected components analysis...')
    minimal_nodes_for_single_component = 50
    components = list(nx.connected_components(G))
    partition = {}
    reverse_partition = {}
    for i in range(len(components)):
        print(f'Component {i+1}/{len(components)} with {len(components[i])} nodes:')
        if len(components[i]) < minimal_nodes_for_single_component:
            continue
        for node in list(components[i]):
            partition[node] = i
        reverse_partition[i] = list(components[i])

    return partition, reverse_partition

def find_connected_components_in_graph_enhanced(ratio_graph, desired_ratio, tolerance):

    max_vertex_id = max(np.hstack((ratio_graph[:, 0], ratio_graph[:, 1]))).astype(int)
    potential_nodes = -1 * np.ones(max_vertex_id + 1)

    # Scan over edges and test for ratio coherency
    print('Beginning scan of edges cost ratio coherency...')
    next_counter_report = 0
    progress = 0
    for i in range(ratio_graph.shape[0]):
        if progress >= next_counter_report:
            print(f'{round(progress)}%')
            next_counter_report += 1
        progress = i / ratio_graph.shape[0] * 100

        source = ratio_graph[i, 0].astype(int)
        if potential_nodes[source] == 0:
            continue
        edge_cost = ratio_graph[i, 2]
        if abs(edge_cost - desired_ratio) < tolerance:
            potential_nodes[source] = 1
        else:
            potential_nodes[source] = 0

    # Create the undirected graph that will be used to identify the connected components
    print('Beginning of NetworkX graph population...')
    G = nx.Graph()
    next_counter_report = 0
    progress = 0
    for i in range(ratio_graph.shape[0]):
        if progress >= next_counter_report:
            print(f'{round(progress)}%')
            next_counter_report += 1
        progress = i / ratio_graph.shape[0] * 100

        source = ratio_graph[i, 0].astype(int)
        target = ratio_graph[i, 1].astype(int)
        if potential_nodes[source] == 1 and potential_nodes[target] == 1:
            G.add_edge(source, target)

    # Find connected components in each sub-graph
    print('Beginning of connected components analysis...')
    minimal_nodes_for_single_component = 50
    components = list(nx.connected_components(G))
    partition = {}
    reverse_partition = {}
    for i in range(len(components)):
        print(f'Component {i+1}/{len(components)} with {len(components[i])} nodes:')
        if len(components[i]) < minimal_nodes_for_single_component:
            continue
        for node in list(components[i]):
            partition[node] = i
        reverse_partition[i] = list(components[i])

    return partition, reverse_partition

def delineate_correlated_clusters(costs_graph, detected_lines, delta,
                                  minimal_nodes_for_single_component=50,
                                  maximal_nodes_for_single_component=100000):

    max_vertex_id = max(np.hstack((costs_graph[:, 0], costs_graph[:, 1]))).astype(int)
    potential_nodes = -1 * np.ones(max_vertex_id + 1)

    partition = {}
    reverse_partition = {}

    potential_nodes = -1 * np.ones(max_vertex_id + 1)

    # Scan over edges and test for ratio coherency
    for desired_line in detected_lines:
        print(f'Beginning clustering for mode = {desired_line}...')
        # potential_nodes = -1 * np.ones(max_vertex_id + 1)
        next_counter_report = 0
        progress = 0
        for i in range(costs_graph.shape[0]):
            if progress >= next_counter_report:
                # print(f'{round(progress)}%')
                next_counter_report += 1
            progress = i / costs_graph.shape[0] * 100

            source = costs_graph[i, 0].astype(int)
            if potential_nodes[source] == 0 or potential_nodes[source] == 2:
                continue
            edge_cost = costs_graph[i, 2]

            # Similarity condition
            if dist_perp(desired_line, costs_graph[i, 2:4]) < delta:
                potential_nodes[source] = 1
            else:
                potential_nodes[source] = 0

        # Create the undirected graph that will be used to identify the connected components
        print('Beginning of NetworkX graph population...')
        G = nx.Graph()
        next_counter_report = 0
        progress = 0
        for i in range(costs_graph.shape[0]):
            if progress >= next_counter_report:
                # print(f'{round(progress)}%')
                next_counter_report += 1
            progress = i / costs_graph.shape[0] * 100

            source = costs_graph[i, 0].astype(int)
            target = costs_graph[i, 1].astype(int)
            if potential_nodes[source] == 1 and potential_nodes[target] == 1:
                G.add_edge(source, target)

        # Find connected components in each sub-graph
        print('Beginning of connected components analysis...')
        components = list(nx.connected_components(G))

        if len(reverse_partition) == 0:
            last_cluster_id = 0
        else:
            last_cluster_id = max(reverse_partition.keys()) + 1

        for i in range(len(components)):
            current_cluster_id = last_cluster_id + i
            # print(f'Component {i+1}/{len(components)} with {len(components[i])} nodes:')
            if len(components[i]) < minimal_nodes_for_single_component or \
                    len(components[i]) > maximal_nodes_for_single_component:
                continue
            for node in list(components[i]):
                partition[node] = current_cluster_id

                # Marking source and target vertices so they cannot be mistakenly re-clustered again
                potential_nodes[node] = 2

            reverse_partition[current_cluster_id] = list(components[i])

    return partition, reverse_partition

def adjacency_list_to_nx_graph(edges_list, coords):
    G = nx.Graph()
    for i in range(edges_list.shape[0]):
        u = edges_list[i, 0].astype(int)
        v = edges_list[i, 1].astype(int)
        G.add_edge(u, v)
        G[u][v]['weight'] = edges_list[i, 2]

    pos = {}
    for i in range(coords.shape[0]):
        pos[coords[i, 2].astype(int)] = coords[i, 0:2]

    return G, pos

def generate_weighted_planar_graph(num_nodes, means, std_devs):
    # Generate three distinct sets of points
    points_cluster1 = np.random.rand(num_nodes // 3, 2) * [1.0, 0.3]
    points_cluster2 = np.random.rand(num_nodes // 3, 2) * [1.0, 0.3] + [0.0, 0.35]
    points_cluster3 = np.random.rand(num_nodes // 3, 2) * [1.0, 0.3] + [0.0, 0.7]
    points = np.vstack((points_cluster1, points_cluster2, points_cluster3))

    # Create a Delaunay triangulation
    tri = Delaunay(points)

    # Create the graph from the triangulation
    G = nx.Graph()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

    # Assign weights based on the cluster regions
    cluster_boundaries = [len(points_cluster1), len(points_cluster1) + len(points_cluster2)]
    for (u, v) in G.edges():
        if u < cluster_boundaries[0] and v < cluster_boundaries[0]:
            G[u][v]['weight'] = round(np.random.normal(means[0], std_devs[0]), 2)
        elif u < cluster_boundaries[1] and v < cluster_boundaries[1]:
            G[u][v]['weight'] = round(np.random.normal(means[1], std_devs[1]), 2)
        elif u >= cluster_boundaries[1] and v >= cluster_boundaries[1]:
            G[u][v]['weight'] = round(np.random.normal(means[2], std_devs[2]), 2)
        else:
            # Assign a high weight for edges between different clusters to discourage connections
            G[u][v]['weight'] = 100000

    return G, points


def plot_nx_graph(G, points):
    pos = {i: points[i] for i in range(len(points))}
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def generate_graph_with_modes():
    G = nx.Graph()

    # Example nodes and edges with costs having small perturbations around 4 modes
    nodes = range(10)
    edges = [
        (0, 1, np.random.normal(1, 0.1)), (1, 2, np.random.normal(1, 0.1)), (2, 3, np.random.normal(1, 0.1)),
        (3, 4, np.random.normal(2, 0.1)), (4, 5, np.random.normal(2, 0.1)), (5, 6, np.random.normal(2, 0.1)),
        (6, 7, np.random.normal(3, 0.1)), (7, 8, np.random.normal(3, 0.1)), (8, 9, np.random.normal(3, 0.1)),
        (0, 9, np.random.normal(4, 0.1)), (1, 8, np.random.normal(4, 0.1))
    ]

    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)

    return G


def louvain_clustering_on_edge_costs(G):
    start_time = time.time()
    # Use the Louvain method to partition the graph
    partition = community_louvain.best_partition(G, weight='weight', resolution=1)
    end_time = time.time()
    print(f'Louvain graph clustering completed in {end_time-start_time} [sec]')
    return partition


def plot_graph_with_partitions(G, partition):
    # Plot the graph with the partition
    pos = nx.spring_layout(G)  # Compute graph layout
    cmap = plt.get_cmap('viridis')

    # Draw nodes with colors corresponding to their partition
    for community in set(partition.values()):
        nodes = [node for node in partition.keys() if partition[node] == community]
        nx.draw_networkx_nodes(G, pos, nodes, node_size=100,
                               node_color=[cmap(community / len(set(partition.values())))])

    # Draw edges with their weights
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    plt.show()

def Louvain_costs_correlation_clustering(edges_list, coordinates):
    nodes = np.union1d(edges_list[:, 0], edges_list[:, 1])

    if nodes.shape[0] > coordinates.shape[0]:
        valid_ind = np.isin(nodes, coordinates[:, 2])
        nodes = nodes[valid_ind, :]
        source_valid = np.isin(nodes, edges_list[:, 0])
        target_valid = np.isin(nodes, edges_list[:, 1])
        ind = np.logical_and(source_valid, target_valid)
        edges_list = edges_list[ind, :]
    elif nodes.shape[0] < coordinates.shape[0]:
        valid_ind = np.isin(coordinates[:, 2], nodes)
        invalid_nodes = coordinates[~valid_ind, 2]
        source_invalid = np.isin(edges_list[:, 0], invalid_nodes)
        target_invalid = np.isin(edges_list[:, 1], invalid_nodes)
        invalid_ind = np.logical_and(source_invalid, target_invalid)
        edges_list = edges_list[~invalid_ind, :]
        coordinates = coordinates[valid_ind, :]
        node_ind = np.isin(nodes, coordinates[:, 2])
        nodes = nodes[node_ind]

    # Construct a nx graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges_list)

    # Apply the Louvain method to the graph
    start_time = time.time()
    partition = louvain_clustering_on_edge_costs(G)
    end_time = time.time()
    print(f'Finished in {end_time - start_time}')

    # Plot the clustered data
    sorted_indices = np.argsort(coordinates[:, 2])

    # Use the indices to sort the entire matrix
    sorted_coords = coordinates[sorted_indices]

    items = list(partition.items())
    labels = np.array(items)

    labels = labels[0:sorted_coords.shape[0], :]

    plt.figure()
    colors = labels[:, 1]
    X = sorted_coords[:, 0]
    Y = sorted_coords[:, 1]
    plt.scatter(X, Y, c=colors, cmap='jet', s=5)#, alpha=0.2, edgecolors=None)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.grid()
    plt.tight_layout()
    plt.show()

def Louvain_example():
    # Create a sample graph
    G = nx.karate_club_graph()

    # Apply the Louvain method
    partition = community_louvain.best_partition(G)

    # Draw the graph with the partition
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    cmap = plt.get_cmap('viridis')

    # Draw nodes with colors corresponding to their partition
    for community in set(partition.values()):
        nodes = [node for node in partition.keys() if partition[node] == community]
        nx.draw_networkx_nodes(G, pos, nodes, node_size=100,
                               node_color=[cmap(community / len(set(partition.values())))])

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

def segment_example():
    # Parameters
    image_size = (100, 100)
    square1_pos = (20, 20)
    square1_size = (20, 20)
    square1_value = 0.5
    square2_pos = (60, 60)
    square2_size = (20, 20)
    square2_value = 0.7

    # Create a random noise background
    np.random.seed(42)
    image = np.random.rand(*image_size).astype(np.float32)

    # Add two squares with different grayscale values
    image[square1_pos[0]:square1_pos[0] + square1_size[0],
    square1_pos[1]:square1_pos[1] + square1_size[1]] = square1_value
    image[square2_pos[0]:square2_pos[0] + square2_size[0],
    square2_pos[1]:square2_pos[1] + square2_size[1]] = square2_value

    # Reshape the image for clustering
    X = image.reshape(-1, 1)
    coords = np.indices(image_size).reshape(2, -1).T


    # Vectorized calculation of the combined affinity matrix using a Gaussian kernel
    def combined_gaussian_kernel(X, coords, sigma_intensity=0.1, sigma_spatial=0.2):
        num_pixels = X.shape[0]

        # Compute intensity differences
        intensity_diff = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

        # Compute spatial differences
        spatial_diff = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)

        # Calculate the affinity matrix
        affinity_matrix = np.exp(- (intensity_diff ** 2) / (2 * sigma_intensity ** 2)
                                 - (spatial_diff ** 2) / (2 * sigma_spatial ** 2))
        return affinity_matrix


    affinity_matrix = combined_gaussian_kernel(X, coords)

    # Compute the Laplacian matrix
    laplacian = csgraph.laplacian(affinity_matrix, normed=True)

    # Compute the first k eigenvectors of the Laplacian matrix
    n_clusters = 3  # Background and two squares
    _, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM')

    # Perform k-means clustering on the eigenvectors
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(eigenvectors)

    # Reshape the labels back to the image shape
    segmented_image = labels.reshape(image_size)

    # Plot the original and segmented images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(segmented_image, cmap='gray')
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')

    plt.show()

def label_propagation_communities(G):
    partition = nx.community.label_propagation_communities(G)
    return partition

def apply_spectral_clustering(G, num_clusters=2):
    # Convert graph to adjacency matrix
    adj_matrix = nx.to_numpy_array(G)

    # Apply Spectral Clustering
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(adj_matrix)
    partition = {}
    for node, cluster in zip(G.nodes(), labels):
        partition[node] = cluster

    return partition

def plot_graph_communities(G, pos, partition, save2tiff=False):
    edge_labels = {(u, v): round(G[u][v]['weight'], 2) for u, v in G.edges()}

    node_color = 'lightblue'
    if partition:
        node_color = [partition[node] for node in G.nodes()]

    if save2tiff:
        fig = plt.figure()
        ax = fig.add_subplot()
        nx.draw(G, pos, with_labels=False, node_color=node_color, node_size=100, cmap=plt.cm.jet, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=partition, font_size=8, font_color="k", ax=ax)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.savefig('network_graph.tiff', format='tiff')
        plt.close(fig)

    else:
        nx.draw(G, pos, with_labels=False, node_color=node_color, node_size=100, cmap=plt.cm.jet)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
        nx.draw_networkx_labels(G, pos, labels=partition, font_size=8, font_color="k")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.grid()
        plt.show()