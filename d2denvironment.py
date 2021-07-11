import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_clusters(mode, num_nodes, num_clusters, overlap):
    # Parameters for cluster configuration
    overlap_factor = overlap
    max_cluster_size = int(num_nodes/num_clusters * (1 + overlap_factor))
    min_cluster_size = 3
    cluster_set = [[] for _ in range(num_clusters)]      

    # Configure cluster sizes. Choose values randomly for a given max and min cluser_size
    if num_clusters > 1: # Centralized Fed vs Clustered D2D
        if mode == 'd2d':
            cluster_size = 2 * np.ones(num_clusters)
            cluster_sizes = [int(x) for x in cluster_size]
        else:
            cluster_sizes = np.random.randint(min_cluster_size, max_cluster_size, size=num_clusters, dtype = int)
        # Assign nodes to each cluster randomly. Looped over cluster sizes
        for i in range(num_clusters):
            cluster_set[i] = np.random.choice(num_nodes, cluster_sizes[i], replace = False)
    else:
        cluster_set[0] = list(range(num_nodes))
        
    # Initiate main graph    
    cluster_graph = nx.Graph()
    # Generate Graph
    for i in range(len(cluster_set)):
        temp = nx.complete_graph(cluster_set[i])
        cluster_graph = nx.compose(cluster_graph, temp)
        del temp
    
    # Calucluate Laplacian
#     L = nx.normalized_laplacian_matrix(cluster_graph)
#     e = np.linalg.eigvals(L.A)
    
    return cluster_set, cluster_graph


def plot_graph_EVhist(graph, hist_limit):
    nx.draw(graph, with_labels = True, font_weight = 'bold')
    plt.title('Clustered D2D Setting')
    plt.savefig('Cluster Configuration')
    
    laplacian = nx.normalized_laplacian_matrix(graph)
    L = nx.normalized_laplacian_matrix(graph)
    e = np.linalg.eigvals(L.A)
    print("Largest eigenvalue:", max(e))
    print("Smallest eigenvalue:", min(e))
    plt.hist(e, bins=100)  # histogram with 100 bins
    plt.xlim(0, hist_limit)  # eigenvalues between 0 and 2
    plt.savefig('Eigen Value: Histogram')
    
    
    
    
    
    