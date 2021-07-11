import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
    
from DNN import *
from data_utils import *
from d2denvironment import *

def federate(modes, num_rounds, num_epochs, num_nodes, cluster_def, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, wt_init = 'same'):
    #### Initialize DNN models at each node. 
    """ Instantiate the models using same weights or different weights (default wt_init = 'same').
        The node-model dictionary(model_collection) is then created.
        A randomly chosen cluster is required to perform local updates and aggregate cluster models after num_epochs and 
        the process is repeated num_cycles times.
    """
    
    source_data = type(traindata)
    if 'mnist' in str(source_data):
        dataset = 'mnist'
    elif 'cifar' in str(source_data):
        dataset = 'cifar'
        
    # Same weight initialization
    if wt_init == 'same':       
        same_wt_basemodel = Net(num_labels, in_channels, dataset)
        model_dict = {i:copy.deepcopy(same_wt_basemodel) for i in range(num_nodes)}
    elif wt_init == 'diff':
        model_dict = {i:Net(num_labels, in_channels, dataset) for i in range(num_nodes)}
    
    recorder = {key:[] for key in range(num_nodes)}
    ## Model Dictionary for each of the Fed Learning Modes
    ## Model Dictionary Initialization

    for key in modes:
        mode_model_dict = {key:None}
        mode_trgloss_dict = {key:None}
        mode_testloss_dict = {key:None}
        mode_acc_dict = {key:None}

    divergence_dict = {key:None for key in range(num_rounds)}
    
    # Create separate copies for each mode
    for mode in modes:
        if mode != 'sgd':
            mode_model_dict[mode] = copy.deepcopy(model_dict)
            mode_trgloss_dict[mode] = copy.deepcopy(recorder)
            mode_testloss_dict[mode] = copy.deepcopy(recorder)
            mode_acc_dict[mode] = copy.deepcopy(recorder)
        elif mode == 'sgd':
            mode_model_dict[mode] = None
            mode_trgloss_dict[mode] = []
            mode_testloss_dict[mode] = []
            mode_acc_dict[mode] = []
            
    
    for rnd in range(num_rounds):
        for mode in modes:
           #  Run local update on models for each mode
            if mode != 'sgd':
                if rnd == 0:
                    if mode == 'd2d_clus':
                        num_clusters = cluster_def[0]
                    elif mode == 'd2d':
                        num_clusters = cluster_def[1]
                    elif mode == 'centr_fed':
                        num_clusters = cluster_def[2]
                    else:
                        raise NotImplementedError('Environment Mode not correctly set. Choose either cluster_d2d or centr_fed')
                
                cluster_set, cluster_graph = generate_clusters(mode, num_nodes, num_clusters, overlap = 0.75)
                updated_model_dict, loss_dict, cluster_loss = local_update(num_epochs, mode_model_dict[mode], num_labels, in_channels, dataset, traindata, traindata_dist)

                for node, loss in loss_dict.items():
                    mode_trgloss_dict[mode][node].append(loss)
                print('Local update for all nodes for mode-%s completed' %(mode))
                print('Average train loss pre-aggregate %0.3g' %(cluster_loss/len(model_dict)))

                # Model Aggregation
                model_aggregation(cluster_set, updated_model_dict, mode_model_dict, mode, dataset, testdata, testdata_dist, num_labels, in_channels)

                # Model testing: Accuracy and Loss calculation
                test_losses, test_accs = model_testing(mode_model_dict[mode], testdata, testdata_dist)

                for node, test_loss in test_losses.items():
                    mode_testloss_dict[mode][node].append(test_loss)

                for node, test_acc in test_accs.items():
                    mode_acc_dict[mode][node].append(test_acc)

                print(f'Cycle {rnd} for mode {mode} completed')
        else:
            # Execute SGD based learning for the same setting ( LR, epochs, full data)

            sgd_model = Net(num_labels, in_channels, dataset).cuda()
            trainloader = DataLoader(traindata)
            testloader = DataLoader(testdata)
            sgd_model_optim = optim.SGD(sgd_model.parameters(), lr = 0.01)
            sgdtrgloss = client_update(sgd_model, sgd_model_optim, trainloader, num_epochs)

            sgdtestloss, sgdtestacc = test(sgd_model, DataLoader(testdata))
                    
            mode_model_dict['sgd'] = copy.deepcopy(sgd_model)
            mode_trgloss_dict[mode].append(sgdtrgloss)
            mode_testloss_dict[mode].append(sgdtestloss)
            mode_acc_dict[mode].append(sgdtestacc)
            print('SGD Training completed-Moving to calculate Divergence')
        
        
        sgd_divergence = calculate_divergence(modes, mode_model_dict, cluster_set, num_nodes)
        divergence_dict[rnd] = sgd_divergence
            
    return mode_model_dict, mode_acc_dict, mode_trgloss_dict, mode_testloss_dict, divergence_dict
            
                      
def local_update(num_epochs, model_dict, num_labels, in_channels, dataset, train, train_dist): 
    client_models = [Net(num_labels, in_channels, dataset).cuda() for _ in range(len(model_dict))]
#     client_models_dict = dict(zip(list(range(len(client_models))), client_models))
    opt = [optim.SGD(model.parameters(), lr = 0.001) for model in client_models]
#     opt_dict = dict(zip(list(range(len(model_dict))), opt))
    for node, model in model_dict.items():
            client_models[node].load_state_dict(model_dict[node].state_dict())       

    loss_dict = {key: [] for key in range(len(model_dict))}
    cumm_loss = 0
    for node, client_model in enumerate(client_models):
        loss = client_update(client_model, opt[node], DataLoader(DataSubset(train, train_dist, node)), num_epochs)
        loss_dict[node].append(loss)
        cumm_loss += loss         
       
    return client_models, loss_dict, cumm_loss
        

def model_aggregation(cluster_set, model_dict, main_dict, mode, dataset, test, test_dist, num_labels, in_channels):
    test_loss_dict = {key: [] for key in range(len(model_dict))}
    test_acc_dict = {key: [] for key in range(len(model_dict))}
#     if mode =='d2d_clus' or mode == 'centr_fed':
#         agg_order = random.sample(range(0, len(cluster_set)), len(cluster_set))
#     else:
#         raise NotImplementedError('D2D/P2P aggregation methods not implemented')
    agg_order = random.sample(range(0, len(cluster_set)), len(cluster_set))
    aggregated_models = [Net(num_labels, in_channels, dataset) for i in range(len(cluster_set))]
    
    for cluster_id in agg_order:
        cluster_nodes = cluster_set[cluster_id]
        cluster_models = [model_dict[i] for i in cluster_nodes]
        # server aggregate
        aggregated_model = server_aggregate(cluster_models, num_labels, in_channels, dataset)
        
        for node in cluster_nodes:
            main_dict[mode][node].load_state_dict(aggregated_model.state_dict())
    print(f'Aggregation for mode {mode} completed')
        

def model_testing(model_dict, testdata, test_dist):
    loss_dict = {k:None for k in range(len(model_dict))}
    acc_dict = {k:None for k in range(len(model_dict))}
    for node in range(len(model_dict)):
        loss, acc = test(model_dict[node].cuda(), DataLoader(DataSubset(testdata, test_dist, node)))          
        loss_dict[node] = loss
        acc_dict[node] = acc
    print('Model_testing completed')
    return loss_dict, acc_dict
               