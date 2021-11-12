from DNN import *
import heapq
import numpy as np
from data_utils import DataSubset
import copy

import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

class Nodes:
    """
    Generates node status and recording dictionaries
    """
    
    def __init__(self, node_idx, base_model, num_labels, in_channels, traindata, trg_dist, testdata, test_dist, dataset, batch_size, node_neighborhood,
                 network_weights, lr = 0.01, wt_init = True):
        #Node properties
        self.idx = node_idx
        self.batch_size = batch_size
        self.neighborhood = node_neighborhood
        self.ranked_nhood = node_neighborhood
        self.degree = len(self.neighborhood)
        self.weights = network_weights[self.idx]
        
        # Dataset and data dist related
        self.trainset = trg_dist[self.idx]
        self.trainloader = DataLoader(DataSubset(traindata, trg_dist, self.idx), batch_size = batch_size)
        self.testset = test_dist[self.idx]
        self.testloader = DataLoader(DataSubset(testdata, test_dist, self.idx))
        self.base_model_selection(base_model, wt_init, lr)
        
        # Recorders
        self.trgloss = []
        self.trgacc = []
        self.testloss = []
        self.testacc = []
        
        # Appending self-idx to record CFL divergence
        # Divergence Targets
        div_targets = self.neighborhood
        div_targets.append(self.idx)
        self.divergence_dict = {node:[] for node in div_targets}
        self.divergence_conv_dict = {node:[] for node in div_targets}
        self.divergence_fc_dict = {node:[] for node in div_targets}        
        
    def base_model_selection(self, base_model, wt_init, lr):
        # Same weight initialization
        if wt_init == True:
            self.model = copy.deepcopy(base_model).cuda()
        else:
            self.model = Net(num_labels, in_channels, dataset).cuda()
        self.opt = optim.SGD(self.model.parameters(), lr = lr)

    def local_update(self, model, num_epochs):
        node_update(model, self.opt, self.trainloader, self.trgloss, self.trgacc, num_epochs)
        
    def node_test(self):
        test_loss, test_acc = test(self.model, self.testloader)
        self.testloss.append(test_loss)
        self.testacc.append(test_acc)
#         print(f'Accuracy for node{self.idx} is {test_acc:0.5f}')
        
        
    def nhood_ranking(self, rnd, sort_crit = 'total', sort_scope= 'last', sort_type = 'min'):
        if sort_crit == 'total':
            self.apply_ranking(self.divergence_dict, rnd, sort_scope, sort_type)
        elif sort_crit == 'conv':
            self.apply_ranking(self.divergence_conv_dict, rnd, sort_scope, sort_type)
        elif sort_crit == 'fc':
            self.apply_ranking(self.divergence_fc_dict, rnd, sort_scope, sort_type)
            
         
    def apply_ranking(self, target, rnd, sort_scope, sort_type):
        # Target is the metric (divergence, KL, WS) to apply ranking on.
        if rnd == 0:
            self.ranked_nhood = self.neighborhood
        else:
            if sort_scope == 'last':
                # Sort Scope : Number of previous rounds to base ranking metric on
                prev_performance = {neighbor:divergence[-1] for neighbor, divergence in target.items()}
                
                if sort_type == 'min':
#                     sorted_nhood ={k: v for k, v in sorted(prev_performance.items(), key=lambda item: item[1])}
                    sorted_nhood = heapq.nsmallest(len(self.neighborhood), prev_performance.items(), key = lambda i:i[1])
    
                elif sort_type == 'max':
                    sorted_nhood = heapq.nlargest(len(self.neighborhood), prev_performance.items(), key = lambda i:i[1])
                
            self.ranked_nhood = [node for node, _ in sorted_nhood]
            
                    
    def neighborhood_divergence(self, nodeset, cfl_model, mode ='internode', normalize = False):
                
        for target_node in self.neighborhood:
            target_model = nodeset[target_node].model
            total_div, conv_div, fc_div = self.internode_divergence(target_model)
            
            self.divergence_dict[target_node].append(total_div)
            self.divergence_conv_dict[target_node].append(conv_div)
            self.divergence_fc_dict[target_node].append(fc_div)
        
        total_div_cfl, conv_div_cfl, fc_div_cfl = self.internode_divergence(cfl_model, mode = 'cfl_div')
        self.divergence_dict[self.idx].append(total_div_cfl)
        self.divergence_conv_dict[self.idx].append(conv_div_cfl)
        self.divergence_fc_dict[self.idx].append(fc_div_cfl)            
            
        if normalize == True:
            self.normalize_divergence()

    
    def internode_divergence(self, target_model, mode = 'internode'):
        total_div = 0
        conv_div = 0
        fc_div  = 0
        
        if mode == 'internode':        
            ref_wt = extract_weights(self.model)
            target_wt = extract_weights(target_model)
            
        elif mode == 'cfl_div':
            ref_wt = extract_weights(target_model)
            target_wt = extract_weights(self.model)
            
        for layer in ref_wt.keys():
            if 'weight' not in layer:
                continue
            diff = torch.linalg.norm(ref_wt[layer] - target_wt[layer]).item()
            total_div += diff

            if 'conv' in layer:
                conv_div += diff
            if 'fc' in layer:
                fc_div += diff
        return total_div, conv_div, fc_div
      
    def normalize_divergence(self):
        temp = self.neighborhood.append(self.idx)
        for neighbor in temp:
            norm_factor_total = np.linalg.norm(self.divergence_dict[neighbor])
            self.divergence_dict[neighbor] = self.divergence_dict[neighbor] / norm_factor
            
            norm_factor_conv = np.linalg.norm(self.divergence_conv_dict[neighbor])
            self.divergence_conv_dict[neighbor] = self.divergence_conv_dict[neighbor] / norm_factor_conv
            
            norm_factor_fc = np.linalg.norm(self.divergence_fc_dict[neighbor])
            self.divergence_fc_dict[neighbor] = self.divergence_fc_dict[neighbor] / norm_factor_fc
    
    
    def weight_update(self, network_weights):
        pass
        
    def aggregate_nhood(self, nodeset, agg_count = 'default'):
        #Choosing the #agg_count number of highest ranked nodes for aggregation
        if agg_count == 'default':
            agg_scope = len(self.ranked_nhood)
        else:
            agg_scope = agg_count
        agg_targets = self.ranked_nhood[:agg_scope]
        agg_targets.append(self.idx)
        agg_model = aggregate(nodeset, agg_targets)
        self.model.load_state_dict(agg_model.state_dict())
        
    def aggregate_random(self, nodeset):
        target_id = self.idx
        while target_id == self.idx:
            target_id = random.sample(list(range(len(nodeset))), 1)[0]
        node_list = [self.idx, target_id]
        agg_model = aggregate(nodeset, node_list)
        self.model.load_state_dict(agg_model.state_dict())
        
        
class Servers:
    idx = 0
    def __init__(self, idx, model, records = False):
        self.idx = Servers.idx
        self.model = copy.deepcopy(model)
        Servers.idx += 1
        if records == True:
            self.avgtrgloss = []
            self.avgtrgacc = []
            self.avgtestloss = []
            self.avgtestacc =[]
        
    def harchy_servers(self, cluster_ids, cluster_set):
        self.clusters = cluster_ids
        self.node_ids = []
        for cluster_id in self.clusters:
            for node in cluster_set[cluster_id]:
                self.node_ids.append(node)
                
    def aggregate_servers(self, server_set, nodeset):
        global_model = aggregate(server_set, self.node_ids)
        self.model.load_state_dict(global_model.state_dict())
        for server in server_set:
            server.model.load_state_dict(self.model.state_dict())
            
        for node in nodeset:
            node.model.load_state_dict(self.model.state_dict())
        
    def aggregate_clusters(self, nodeset):
        server_agg_model = aggregate(nodeset, self.node_ids)
        self.model.load_state_dict(server_agg_model.state_dict())