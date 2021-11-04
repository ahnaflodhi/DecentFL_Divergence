import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_clusters(mode, num_nodes, num_clusters, overlap):
    # Parameters for cluster configuration
    overlap_factor = overlap
    max_cluster_size = int(num_nodes/num_clusters * (1 + overlap_factor))
    min_cluster_size = 3
    cluster_set = []      

    # Configure cluster sizes. Choose values randomly for a given max and min cluser_size
    if num_clusters > 1: # Centralized Fed vs Clustered D2D
        if mode == 'Gossip':
            cluster_size = 2 * np.ones(num_clusters)
            cluster_sizes = [int(x) for x in cluster_size]
            # Assign nodes to each cluster randomly. Looped over cluster sizes
            for i in range(num_clusters):
                cluster_set.append(list(np.random.choice(num_nodes, cluster_sizes[i], replace = False)))
        else:
            cluster_sizes = []
            while sum(cluster_sizes) != num_nodes:
                cluster_sizes = np.random.randint(min_cluster_size, max_cluster_size, size=num_clusters, dtype = int)
            main_node_list = list(range(num_nodes))
            node_list = list(range(num_nodes))
            random.shuffle(node_list)
            for size in cluster_sizes:
                temp = random.sample(node_list, size)
                node_list = [item for item in node_list if item not in temp]
                add_factor = np.random.randint(1, 5)
                add = random.sample(list(range(num_nodes)),  add_factor)
                for add_node in add:
                    if add_node not in temp:
                        temp.append(add_node)
                cluster_set.append(temp)
                
#             cluster_set = [[8, 2, 10, 17, 6, 2, 1, 14], [1, 13, 18, 3, 16], [7, 3, 11, 16, 4, 7, 16, 10], [14, 9, 12, 19, 5, 15, 0, 4, 16, 1, 10]]
#             cluster_set = [[27, 12, 13, 9], [12, 29, 18, 8, 2], [28, 21, 0, 13], [25, 1, 6], [25, 4, 7, 17], [26, 18, 12, 4], [24, 11, 25], [20, 12, 10]]
            
    elif num_clusters == 1:
        cluster_set = list(range(num_nodes))
    
    cluster_graph = nx.Graph()
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
    
    
    
    
    
    