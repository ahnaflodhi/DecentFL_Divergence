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
from utils import constrained_sum

class system_model:
    """
    Creates the requisite System model / environment.
    Generates clusters, defines neighborhood and graph.
    Creates dictionary for records.
    """    
    def __init__(self, num_nodes, num_clusters, prob_int = 0.95, prob_ext = 0.02):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.num_servers  = 3
        self.max_size = int((self.num_nodes / self.num_clusters) + 3) 
        self.min_size = 4
        self.prob_int = prob_int #Probability of intra-parition edges
        self.prob_ext = prob_ext #Probability of inter-partition edges
        
        self.generate_clusters() # Generate Network Layout
        self.create_graph() # Generate graph
        self.map_neighborhood() # Map node neighborhood
        self.graph_stats() # Graph statistics
        self.create_servers() # Create hierarchical servers.
        self.cluster_head_select() #Select Cluster heads. Applicable should only be flag based
    
    def generate_clusters(self):
        self.cluster_sizes = []
        while sum(self.cluster_sizes) != self.num_nodes:
            self.cluster_sizes = np.random.randint(self.min_size, self.max_size, self.num_clusters)
            
        self.cluster_set = []
        for i, _ in enumerate(self.cluster_sizes):
            temp = list(range(sum(self.cluster_sizes[:i]), sum(self.cluster_sizes[:i+1])))
            self.cluster_set.append(temp)
        print(f'The generated cluster set is {self.cluster_set}')
        
    def create_graph(self):
        self.graph = nx.random_partition_graph(self.cluster_sizes, self.prob_int, self.prob_ext)
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
        target_indx = constrained_sum(self.num_servers, self.num_clusters)
        self.cluster_ids = []
        cluster_list = list(range(self.num_clusters))
        random.shuffle(cluster_list)
        for i in target_indx:
            temp = cluster_list[:i]
            cluster_list = [item for item in cluster_list if item not in temp]
            self.cluster_ids.append(temp)            
        print(f'With {self.cluster_ids} servers and server_targets have been configured')    
    
    def cluster_head_select(self):
        self.cluster_heads = []
        for cluster in self.cluster_set:
            temp = list(random.sample(cluster, 1))
            self.cluster_heads.append(temp)
            
class FL_modes(Nodes):
    modes_list = []
    #'d2d', dataset, num_epochs, num_nodes, base_model, num_labels, in_ch, traindata, train_dist, testdata, test_dist, dataset, batch_size, env.neighborhood_map, env.Lp
    def __init__(self, name, dataset, num_epochs, num_rounds, num_nodes, dist, base_model, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, batch_size, nhood, env_Lp, num_clusters, **kwargs):
        
        # Passing static values for neighborhood
        # modes receives the neighborhood for all the nodes whereas nodes class requires neighborhood for the node only
        # Kwargs include d2d_agg_flg, ch_agg_flg, hserver_agg_flg, 
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
            
            # Aggregation Flags
            self.d2d_agg_flg = kwargs['d2d_agg_flg']
            self.ch_agg_flg = kwargs['ch_agg_flg']
            self.hserver_agg_flg = kwargs['hserver_agg_flg']
            self.inter_ch_agg_flg = kwargs['inter_ch_agg_flg']
                        
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
        
#     def __getstate__(self):
#         """Returns objects to be pickled"""
#         data = self.__dict__.copy()
#         list = ['avgtrgacc', 'avgtrgloss', 'avgtestacc', 'avgtestloss', 'trgacc', 'trgloss', 'testacc', 'testloss', 
#                 'cluster_trgacc', 'cluster_trgloss', 'cluster_testacc', 'cluster_testloss',
#                'degree', 'dist', 'divergence_dict', 'divergence_conv_dict', 'divergence_fc_dict']
#         del data['nodeset]
#         return state       
    
        
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
            
    def nhood_aggregate_round(self, weightage = 'equal'):
        for node in self.nodeset:
            node.aggregate_nodes(self.nodeset, weightage)            
        
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
    
    def clshead_aggregate_round(self, cluster_head, cluster_set, prop, weightage = 'equal'):
        self.nodeset[cluster_head].aggregate_nodes(self.nodeset, weightage, cluster_set = cluster_set, agg_prop = 0.6)
        # Load CH model on all cluster nodes
        for node in cluster_set:
            self.nodeset[node].model.load_state_dict(self.nodeset[cluster_head].model.state_dict())
            
    def inter_ch_aggregate_round(self, cluster_heads, weigtage = 'equal'):
        random.shuffle(cluster_heads)
        ch_pairs = []
        ch_list = [ch[0] for ch in cluster_heads]
        if len(ch_list) % 2 != 0:
            while len(ch_list) > 1:
                temp = random.sample(ch_list, 2)
                ch_list = [ch for ch in ch_list if ch not in temp]
                ch_pairs.append(temp)
        else:
            while len(ch_list) != 0:
                temp = random.sample(ch_list, 2)
                ch_list = [ch for ch in ch_list if ch not in temp]
                ch_pairs.append(temp)
                
        for ch_pair in ch_pairs:
            scale = {ch:1.0 for ch in ch_list}
            aggregate(self.nodeset, ch_pair, scale)
            self.nodeset[ch_pair[1]].model.load_state_dict(self.nodeset[ch_pair[0]].model.state_dict())
        
    def cfl_aggregate_round(self, prop,  weightage = 'equal'):
        if weightage == 'equal':
            scale = {i:1.0 for i in range(len(self.nodeset))}
        elif weightage == 'proportional':
            scale = {i:self.nodeset[i].divergence_dict[i][-1] for i in range(len(self.nodeset))}
        nodelist = list(range(len(self.nodeset)))
        agg_count =int(np.floor(prop * len(nodelist)))
        if agg_count < 1:
            agg_count = 1
        sel_nodes = random.sample(nodelist, agg_count)
        agg_model = aggregate(self.nodeset, sel_nodes, scale)
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