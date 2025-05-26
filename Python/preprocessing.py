from DIMACS_tools import *
from segmentation_tools import *
from RANSAC_Multiple_Line_Detection import *
import time
import numpy as np

if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Definitions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DIMACS_root_path = 'D:\Thesis\DIMACS'
    instance = 'NY'
    path2CorrClusteringExe = r"D:\Thesis\A-pex\multiobj\x64\Release\CorrelationClustering.exe"
    path2ApexExe = r'D:\Thesis\A-pex\multiobj\x64\Release\multiobj.exe'
    epsilon = 0.01 # approximation factor used for each objective function
    delta = 0.001 # for checking if a cost point (c1,c2) conforms to a line
    timeout = 3600  # [sec]
    super_edges_file = r'D:\Thesis\Python\Super_Edges.csv'
    output_dir = rf'D:\temp'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reading instance datafiles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    total_time_start = time.time()

    # Generating the standard DIMACS instance filenames to be loaded
    distance_filename_orig, time_filename_orig, coords_filename_orig, \
        distance_filename_contr, time_filename_contr, coords_filename_contr = \
        generate_strings_for_DIMACS_instance(DIMACS_root_path, instance)

    # Loading DIMACS instance
    c1_graph, _ = load_graph(distance_filename_orig)
    c2_graph, _ = load_graph(time_filename_orig)
    coords = read_coords_file(coords_filename_orig)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting map geographic display
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if False:
        plt.figure()
        plt.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.5)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(instance)
        plt.axis('equal')
        plt.grid()
        plt.tight_layout()
        plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting costs ratio histogram
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if False:
        plt.figure()
        plt.hist(c1_graph[:, 2] / c2_graph[:, 2], bins=50)
        plt.xlabel(r'$\frac{C_{1}}{C_{2}}$')
        plt.ylabel('Number of edges')
        plt.grid()
        plt.title(r'$\frac{C_{1}}{C_{2}}$')
        plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RANSAC multiple linear relationships detector
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time = time.time()
    decim_factor = 1000  # Decimating the data speeding-up computation
    x_max = max(c1_graph[::decim_factor, -1])
    y_max = max(c2_graph[::decim_factor, -1])
    x_decim = c1_graph[::decim_factor, -1] / x_max
    y_decim = c2_graph[::decim_factor, -1] / y_max

    detected_lines = ransac_multiple_lines(x_decim, y_decim, delta=delta,
                                           num_hypotheses=100, min_inlier_threshold=0, min_samples=0, max_iter=5)
    end_time = time.time()
    print(f'RANSAC finished in {end_time - start_time} [sec]')
    print(detected_lines)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot RANSAC detected lines with data points
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if False:
        plt.figure(figsize=(6, 6))
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.5)

        x = c1_graph[:, -1] / x_max
        y = c2_graph[:, -1] / y_max

        for k, (a, b) in enumerate(detected_lines):
            # min_x, max_x = min(x), max(x)
            min_x = 0
            max_x = 1
            if k == 0:
                plt.plot([min_x, max_x], [a * min_x + b, a * max_x + b], label='Detected\nCorrelations',
                         color='r', linestyle='-', linewidth=2)
            else:
                plt.plot([min_x, max_x], [a * min_x + b, a * max_x + b], label=None,
                         color='r', linestyle='-', linewidth=2)

        # Scatter plot with transparency
        plt.scatter(x, y, s=30, alpha=0.5, color='b', label='Data points')

        # Customize axes labels and title
        plt.xlabel(r'Normalized $C_1$', fontsize=25)
        plt.ylabel(r'Normalized $C_2$', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.legend(fontsize=18, loc='lower right', frameon=True, fancybox=True, framealpha=0.8)
        plt.tight_layout()
        plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Correlated clusters delineation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time = time.time()
    costs_graph = np.hstack((c1_graph[:, 0].reshape(-1, 1), c1_graph[:, 1].reshape(-1, 1),
                             (c1_graph[:, -1]/x_max).reshape(-1, 1), (c2_graph[:, -1]/y_max).reshape(-1, 1)))

    partition, reverse_partition = delineate_correlated_clusters(costs_graph, detected_lines, delta,
                                                                 minimal_nodes_for_single_component=50,
                                                                 maximal_nodes_for_single_component=10000)
    clusters_metafile = rf"{DIMACS_root_path}\{instance}\{instance}_clusters_metafile.txt"
    convert_partition_to_clusters_metafile_enhanced(clusters_metafile, partition, reverse_partition)
    end_time = time.time()
    print(f'Clustering finished in {end_time - start_time} [sec]')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Inter Clusters Approximation Cost computation (super-edges)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time = time.time()
    inter_clusters_cost_approximation(clusters_metafile, timeout, epsilon, output_dir, distance_filename_orig,
                                      time_filename_orig, path2ApexExe, path2CorrClusteringExe, super_edges_file)
    end_time = time.time()
    print(f'ICCA finished in {end_time - start_time} [sec]')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generating a new, generalized graph, enriched with super-edges
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time = time.time()
    generate_contracted_graph(distance_filename_orig, time_filename_orig, coords_filename_orig,
                              distance_filename_contr, time_filename_contr,
                              coords_filename_contr, clusters_metafile, super_edges_file)
    end_time = time.time()
    print(f'New graph generation finished in {end_time - start_time} [sec]')

    total_time_end = time.time()
    print('=======================================================')
    print(f'Preprocessing finished in {total_time_end - total_time_start} [sec]')
    print('=======================================================')
