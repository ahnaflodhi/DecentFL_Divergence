import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import copy
import heapq
import pickle

import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

from DNN import *
from devices import Nodes, Servers

class system_model:
    """
    Creates the requisite System model / environment.
    Generates clusters, defines neighborhood and graph.
    Creates dictionary for records.
    """    
    def __init__(self, num_nodes, num_clusters):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.max_size = 10
        self.min_size = 4
        
        self.generate_clusters() # Generate Network Layout
        self.create_graph() # Generate graph
        self.map_neighborhood() # Map node neighborhood
        self.graph_stats() # Graph statistics
        self.create_servers() # Create hierarchical servers.
    
    def generate_clusters(self):
        cluster_set = []
        cluster_sizes = []
        # Ensure total nodes across clusters are equal to the total nodes
        while sum(cluster_sizes) != self.num_nodes:
            cluster_sizes = list(np.random.randint(self.min_size, self.max_size, size=self.num_clusters, dtype = int))
        node_list = list(range(self.num_nodes))
        random.shuffle(node_list)
        for size in cluster_sizes:
            temp = random.sample(node_list, size)
            node_list = [item for item in node_list if item not in temp]
            add_factor = np.random.randint(2, 6)
            add = random.sample(list(range(self.num_nodes)),  add_factor)
            for add_node in add:
                if add_node not in temp:
                    temp.append(add_node)
            cluster_set.append(temp)
        self.cluster_set = cluster_set
        print(f'The generated cluster set is {self.cluster_set}')
        
    def create_graph(self):
        cluster_graph = nx.Graph()
        for i in range(len(self.cluster_set)):
            temp = nx.complete_graph(self.cluster_set[i])
            cluster_graph = nx.compose(cluster_graph, temp)
            del temp
        self.graph = cluster_graph
        nx.draw(self.graph, with_labels = True, font_weight = 'bold')
        plt.title('Network graph')
        plt.savefig('Network_graph')
            
    def map_neighborhood(self):
        neighborhood = []
        for node in range(self.num_nodes):
            temp = [neighbor for neighbor in self.graph.neighbors(node)]
            neighborhood.append(temp)
        self.neighborhood_map = neighborhood

    def graph_stats(self, plot_fig = False):
        self.Lp = nx.normalized_laplacian_matrix(self.graph)
        self.ev = np.linalg.eigvals(self.Lp.A)
#         print("Largest eigenvalue:", max(self.ev))
#         print("Smallest eigenvalue:", min(self.ev))
        if plot_fig == True:
            plt.figure()
            plt.hist(self.ev, bins=100)  # histogram with 100 bins
            plt.xlim(0, 5)  # eigenvalues between 0 and 2
            plt.savefig('Eigen Value: Histogram')
    
    def create_servers(self):
#         self.num_servers = np.random.randint(int(np.ceil(self.num_clusters/5)), int(np.floor(self.num_clusters * 0.5)))
#         self.num_servers = int(np.ceil(self.num_clusters/2))
        self.num_servers = 2
#         target_clusters = []
#         while sum(target_clusters) != self.num_clusters:
#             print('Creating cluster allocations for Servers')
#             target_clusters = list(np.random.randint(2, self.num_clusters, self.num_servers))
        target_clusters = [3, 2]
        print('Servers and Server_targets configured')
        
        self.cluster_ids = []
        cluster_list = list(range(self.num_clusters))
        random.shuffle(cluster_list)
        for i in target_clusters:
            temp = cluster_list[:i]
            cluster_list = [item for item in cluster_list if item not in temp]
            self.cluster_ids.append(temp)

    def workermodels(self, base_model):
        self.model_list = []
        for i in range(self.num_nodes):
            self.model_list.append(base_model.cuda())        
        
           
class FL_modes(Nodes):
    modes_list = []
    #'d2d', dataset, num_epochs, num_nodes, base_model, num_labels, in_ch, traindata, train_dist, testdata, test_dist, dataset, batch_size, env.neighborhood_map, env.Lp
    def __init__(self, name, dataset, num_epochs, num_rounds, num_nodes, dist, base_model, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, batch_size, nhood, env_Lp, num_clusters):
        
        #Passing static values for neighborhood
        # modes receives the neighborhood for all the nodes whereas nodes class requires neighborhood for the node only
        super().__init__(1, base_model, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, dataset, batch_size, [0,0], [0,0])
        self.name = name
        self.dataset = dataset
        self.dist = dist
        self.epochs = num_epochs
        self.rounds = num_rounds
        self.base_model = base_model
        self.cfl_model = copy.deepcopy(self.base_model).cuda()
        self.batch_size = batch_size
        if self.name != 'sgd':
            self.num_clusters = num_clusters
            self.default_weights(env_Lp)
            self.num_nodes = num_nodes
            self.form_nodeset(num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, nhood)
                               
            self.cluster_trgloss = {cluster_id:[] for cluster_id in range(self.num_clusters)}
            self.cluster_trgacc = {cluster_id:[] for cluster_id in range(self.num_clusters)}
            self.cluster_testloss = {cluster_id:[] for cluster_id in range(self.num_clusters)}
            self.cluster_testacc = {cluster_id:[] for cluster_id in range(self.num_clusters)}
        
        # Mode records
        self.avgtrgloss = []
        self.avgtestloss = []
        self.avgtrgacc = []
        self.avgtestacc = []
        
        FL_modes.modes_list.append(self.name)
        
    # node_id, base_model, num_labels, in_channels, traindata, trg_dist, testdata, test_dist, dataset,
    def form_nodeset(self, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, nhood):
        # base_model, num_labels, in_channels, traindata, trg_dist, testdata, test_dist, dataset, batch_size, node_neighborhood
        self.nodeset = []
        for idx in range(self.num_nodes):
            node_n_nhood = nhood[idx]
#             node_n_nhood.append(idx)
            self.nodeset.append(Nodes(idx, self.base_model, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, 
                                      self.dataset, self.batch_size, node_n_nhood, self.weightset))
    
    def form_serverset(self, num_servers):
        self.serverset = []
        for idx in range(num_servers):
            self.serverset.append(Servers(idx, self.base_model))
            
        #Append 1-additioal server to act as a Global server
        self.serverset.append(Servers(num_servers, self.base_model))
                      
    def default_weights(self, Laplacian):
        self.weightset = Laplacian.toarray()

    def update_round(self):
        temp_loss = []
        temp_acc = []
        for node in self.nodeset:
            node.local_update(self.epochs)
            temp_loss.append(node.trgloss[-1])
        self.avgtrgloss.append(sum(temp_loss)/self.num_nodes)

#     def update_round(self, env_models):
#         """
#         Loading a single set of environment models based on cuda.
#         Loading local dictionary on them to update and copy back to local model.
#         """       
#         temp_loss = []
#         temp_acc = []
#         for i, node in enumerate(self.nodeset):
#             env_models[i].load_state_dict(node.model.state_dict())
#             node.local_update(env_models[i], self.epochs)
#             node.model.load_state_dict(env_models[i].state_dict())
#             temp_loss.append(node.trgloss[-1])
#         self.avgtrgloss.append(sum(temp_loss)/self.num_nodes)
            
    def test_round(self, cluster_set):
        temp_acc = []
        temp_loss = []
        for node in self.nodeset:
            node.node_test()
            temp_acc.append(node.testacc[-1])
            temp_loss.append(node.testloss[-1])
        self.avgtestacc.append(sum(temp_acc)/self.num_nodes)
        self.avgtestloss.append(sum(temp_loss)/self.num_nodes)
        
        self.global_avgs()
        self.cluster_avgs(cluster_set)
            
    def ranking_round(self, rnd):
        for node in self.nodeset:
            node.neighborhood_divergence(self.nodeset, self.cfl_model, normalize = True)
            node.nhood_ranking(rnd)
            
    def aggregate_round(self, weightage = 'equal'):
        for node in self.nodeset:
            node.aggregate_nhood(self.nodeset, weightage)            
        
    def random_aggregate_round(self, weightage = 'equal'):
        node_pairs = []
        node_list = list(range(len(self.nodeset)))
        while len(node_list) > 1:
            temp =  random.sample(node_list, 2)
            node_list = [item for item in node_list if item not in temp]
            node_pairs.append(temp)
        for node_pair in node_pairs:
            scale = {node:1.0 for node in node_pair}
            aggregate(self.nodeset, node_pair, scale)
            
    def cfl_aggregate_round(self, weightage = 'equal'):
        if weightage == 'equal':
            scale = {i:1.0 for i in range(len(self.nodeset))}
        elif weightage == 'proportional':
            scale = {i:self.nodeset[i].divergence_dict[i][-1] for i in range(len(self.nodeset))}
        
        agg_model = aggregate(self.nodeset, list(range(len(self.nodeset))), scale)
        self.cfl_model.load_state_dict(agg_model.state_dict())
                          
        for node in self.nodeset:
            node.model.load_state_dict(self.cfl_model.state_dict())
            
    def server_aggregate(self, cluster_id, cluster_set):
        ref_dict = self.nodeset[0].model.state_dict()
        for cluster in cluster_id:
            for layer in ref_dict.keys():
                ref_dict[layer] = torch.stack([self.nodeset[node].model.state_dict()[layer].float() for node in cluster_set[cluster]], 0).mean(0)
            
            
    def global_avgs(self):
        temp_trgloss = []
        temp_trgacc = []
        temp_testacc = []
        temp_testloss = []

        for node in self.nodeset:
            temp_trgloss.append(node.trgloss[-1])
            temp_testloss.append(node.testloss[-1])
            temp_testacc.append(node.testacc[-1])
            
        self.trgloss.append(sum(temp_trgloss)/self.num_nodes)
        self.testloss.append(sum(temp_testloss)/self.num_nodes)
        self.testacc.append(sum(temp_testacc)/self.num_nodes)
        
    
    def cluster_avgs(self, cluster_set):
        for cluster_id, cluster_nodes in enumerate(cluster_set):
            temp_trgloss = []
            temp_trgacc = []
            temp_testacc = []
            temp_testloss = []
            for node in cluster_nodes:
                temp_trgloss.append(self.nodeset[node].trgloss[-1])
                temp_testloss.append(self.nodeset[node].testloss[-1])
                temp_testacc.append(self.nodeset[node].testacc[-1])
            self.cluster_trgloss[cluster_id].append(sum(temp_trgloss)/self.num_nodes)
            self.cluster_testloss[cluster_id].append(sum(temp_testloss)/self.num_nodes)
            self.cluster_testacc[cluster_id].append(sum(temp_testacc)/self.num_nodes)
                
        


            
        
    
            
    
        
        
        
            

    
    
        
          
        
        
        
        
            
        
        
        
    
            
            
        