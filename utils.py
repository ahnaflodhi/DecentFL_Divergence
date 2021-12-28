import random

def constrained_sum(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    divider = []
    while 1 in divider or len(divider) == 0:
        dividers = sorted(random.sample(range(1, total), n - 1))
        divider = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    return divider


# Deprecated versions
#def generate_clusters(self):
#         cluster_set = []
#         cluster_sizes = []
#         # Ensure total nodes across clusters are equal to the total nodes
#         while sum(cluster_sizes) != self.num_nodes:
#             cluster_sizes = list(np.random.randint(self.min_size, self.max_size, size=self.num_clusters, dtype = int))
#         node_list = list(range(self.num_nodes))
#         random.shuffle(node_list)
#         for size in cluster_sizes:
#             temp = random.sample(node_list, size)
#             node_list = [item for item in node_list if item not in temp]
#             add_factor = np.random.randint(2, 6)
#             add = random.sample(list(range(self.num_nodes)),  add_factor)
#             for add_node in add:
#                 if add_node not in temp:
#                     temp.append(add_node)
#             cluster_set.append(temp)
#         self.cluster_set = cluster_set
#         print(f'The generated cluster set is {self.cluster_set}')


#def create_graph(self):
#         cluster_graph = nx.Graph()
#         for i in range(len(self.cluster_set)):
#             temp = nx.random_regular_graph(3, self.cluster_set[i], 4)
#             cluster_graph = nx.compose(cluster_graph, temp)
#             del temp
#         self.graph = cluster_graph


#Prev Model
#             if mode == 'hd2d': 
#                 # Create Hierarchical Servers
#                 modes[mode].form_serverset(environment.num_servers)
#                 # Assign Nodes
#                 node_list = list(range(num_nodes))
#                 for i in range(environment.num_servers):
#                     print(node_list)
#                     modes[mode].serverset[i].harchy_servers_allnodes(environment.cluster_ids[i], environment.cluster_set, node_list)
#                     node_list = [item for item in node_list if item not in modes[mode].serverset[i].node_ids]
                                 
#                     print(f'The nodes assigned to Server-{i} are {modes[mode].serverset[i].node_ids}')
#                 # Assign server list to Master Server
#                 modes[mode].serverset[-1].node_ids = list(range(environment.num_servers))
#                 print(f'The nodes assigned to Global Server are {modes[mode].serverset[-1].node_ids}')
            
#             if mode == 'hfl':
#                 modes[mode] = copy.deepcopy(modes['hd2d'])