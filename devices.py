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
                 network_weights, lr = 0.01, wt_init = True, role = 'node'):
        #Node properties
        self.idx = node_idx
        self.batch_size = batch_size
        self.neighborhood = node_neighborhood
        self.ranked_nhood = node_neighborhood
        self.degree = len(self.neighborhood)
        self.weights = network_weights[self.idx]
        self.role = role
        
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

    def local_update(self, num_epochs):
        node_update(self.model, self.opt, self.trainloader, self.trgloss, self.trgacc, num_epochs)
        print(f'Node {self.idx} : Trg Loss = {self.trgloss}  Test Acc : {self.trgacc}')
                
    def node_test(self):
        test_loss, test_acc = test(self.model, self.testloader)
        self.testloss.append(test_loss)
        self.testacc.append(test_acc)
#         print(f'Accuracy for node{self.idx} is {test_acc:0.5f}')
                    
    def neighborhood_divergence(self, nodeset, cfl_model, mode ='internode', normalize = True):     
        for target_node in self.neighborhood:
            target_model = nodeset[target_node].model
            total_div, conv_div, fc_div = self.internode_divergence(target_model)
#             print(total_div, conv_div, fc_div)
#             print(self.divergence_dict)
            self.divergence_dict[target_node].append(total_div)
            self.divergence_conv_dict[target_node].append(conv_div)
            self.divergence_fc_dict[target_node].append(fc_div)           
            
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
        total_div_factor = sum([self.divergence_dict[nhbr][-1] for nhbr in self.divergence_dict.keys()])
        conv_div_factor = sum([self.divergence_conv_dict[nhbr][-1] for nhbr in self.divergence_conv_dict.keys()])
        fc_div_factor = sum([self.divergence_fc_dict[nhbr][-1] for nhbr in self.divergence_fc_dict.keys()])
        
        for nhbr in self.divergence_dict.keys():
            self.divergence_dict[nhbr][-1] = self.divergence_dict[nhbr][-1] / total_div_factor
            self.divergence_conv_dict[nhbr][-1] = self.divergence_conv_dict[nhbr][-1] / conv_div_factor
            self.divergence_fc_dict[nhbr][-1] = self.divergence_fc_dict[nhbr][-1] / fc_div_factor
    
    def nhood_ranking(self, rnd, sort_crit = 'total', sort_scope= 1 , sort_type = 'min'):
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
            prev_performance = {neighbor:sum(divergence[-sort_scope:]) for neighbor, divergence in target.items()}
            
            if sort_type == 'min':
#                     sorted_nhood ={k: v for k, v in sorted(prev_performance.items(), key=lambda item: item[1])}
                sorted_nhood = heapq.nsmallest(len(self.neighborhood), prev_performance.items(), key = lambda i:i[1])

            elif sort_type == 'max':
                sorted_nhood = heapq.nlargest(len(self.neighborhood), prev_performance.items(), key = lambda i:i[1])
                
            self.ranked_nhood = [node for node, _ in sorted_nhood]
    
    def scale_update(self, weightage):
        # Aggregation Weights
        if weightage == 'equal':
            # Same weights applied to all aggregation
            scale = {node:1.0 for node in self.neighborhood}
        elif weightage == 'proportional':
            # Divergence-based weights applied to respective models.
            scale = {node:self.divergence_dict[node][-1] for node in self.neighborhood}
        return scale
            
    def aggregate_nodes(self, nodeset, weightage, cluster_set = None, agg_prop = 0.6):
        #Choosing the #agg_count number of highest ranked nodes for aggregation
        # If Node aggregating Nhood
        if cluster_set == None:
            agg_scope = int(np.floor(agg_prop * len(self.ranked_nhood)))
            if agg_scope >= 1 and agg_scope <= len(self.ranked_nhood):
                try:
                    agg_targets = self.ranked_nhood[:agg_scope]
                    agg_targets.append(self.idx)
                except:
                    print(f'Agg_scope:{agg_scope} does not conform to neighborhood {len(self.ranked_nhood)}')
        # If CH aggregating Cluster    
        else:
            agg_scope = int(np.floor(agg_prop * len(cluster_set)))
            if agg_scope >= 1 and agg_scope <= len(cluster_set):
                try:
                    # No need to add self index since cluster-head id already included in cluster-set
                    agg_targets = random.sample(cluster_set, agg_scope)
                except:
                    print(f'Agg_scope {agg_scope} does not conform to size of Cluster_Set {len(cluster_set)}')
        
        scale = self.scale_update(weightage)            
        agg_model = aggregate(nodeset, agg_targets, scale)
        self.model.load_state_dict(agg_model.state_dict())
        
    def aggregate_random(self, nodeset, weightage):
        target_id = self.idx

        while target_id == self.idx:
            target_id = random.sample(list(range(len(nodeset))), 1)[0]
        node_list = [self.idx, target_id]
        if weightage  == 'equal':
            scale = {node:1.0 for node in node_list}
        elif weightage == 'proportional':
            scale = {node:1.0 for node in node_list}
                     
        agg_model = aggregate(nodeset, node_list, scale)
        self.model.load_state_dict(agg_model.state_dict())
        
        
class Servers:
    idx = 0
    def __init__(self, server_id, model, records = False):
        self.idx = server_id
        self.model = copy.deepcopy(model).cuda()
        Servers.idx += 1
        if records == True:
            self.avgtrgloss = []
            self.avgtrgacc = []
            self.avgtestloss = []
            self.avgtestacc =[]
        
    def harchy_servers_allnodes(self, cluster_ids, cluster_set, node_list):
        self.clusters = cluster_ids
        self.node_ids = []
        if len(cluster_ids) <= 1:
            raise Exception("Min cluster assignment for aggregation should be 2 or more")
        else:
            temp = []
            for cluster_id in cluster_ids:
                temp = temp + cluster_set[cluster_id]
            self.node_ids = temp
             
    def aggregate_servers(self, server_set, nodeset):
        scale = {server_id:1.0 for server_id in range(len(server_set)) }
        global_model = aggregate(server_set, self.node_ids, scale)
        self.model.load_state_dict(global_model.state_dict())
        for server in server_set:
            server.model.load_state_dict(self.model.state_dict())
            
        for node in nodeset:
            node.model.load_state_dict(self.model.state_dict())
        
    def aggregate_clusters(self, nodeset, prop):
        nodelist = random.sample(self.node_ids, int(prop*len(self.node_ids)))
        scale = {node:1.0 for node in nodelist}
        server_agg_model = aggregate(nodeset, nodelist, scale)
        self.model.load_state_dict(server_agg_model.state_dict())