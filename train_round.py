import numpy as np
import random
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
    
from DNN import *
from data_utils import *
from d2denvironment import *

def federate(dataset, modes, num_rounds, num_epochs, num_nodes, cluster_def, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, batch_size, 
             test_batch_size, num_servers, wt_init = 'same'):
    #### Initialize DNN models at each node. 
    """ Instantiate the models using same weights or different weights (default wt_init = 'same').
        The node-model dictionary(model_collection) is then created.
        A randomly chosen cluster is required to perform local updates and aggregate cluster models after num_epochs and 
        the process is repeated num_cycles times.
    """
    #modes, dataset, num_labels, in_channels, num_nodes, num_rounds, wt_init
    mode_model_dict, mode_trgloss_dict, mode_testloss_dict, mode_avgloss_dict, mode_acc_dict, mode_avgacc_dict, divergence_dict = dict_creator(modes, dataset, 
                                                                                                               num_labels, in_channels, num_nodes, num_rounds, wt_init)
    
    
    num_clusters = cluster_def['D2D_Clus']
    print('Making clusters')
    cluster_set, cluster_graph = generate_clusters(modes[0], num_nodes, num_clusters, overlap = 0.75)
    print(f'The cluster configuration is {cluster_set}')
    # Main Local update and federation round
    for rnd in range(num_rounds):
        for mode in modes:
            print(f'Starting update and aggregation for {mode}')
           #  Run local update on models for each mode
            # Distributed modes processing here
            if mode != 'SGD':
    #                 if rnd == 0 and mode == 'D2D_Clus':
    #                     num_clusters = cluster_def[mode]
    #                     cluster_set, cluster_graph = generate_clusters(mode, num_nodes, num_clusters, overlap = 0.75)
    #                     print(f'The cluster configuration for {mode} is {cluster_set}')
                if rnd == 0 and mode == 'HD2D':
                    num_clusters = cluster_def[mode]
                    target_clusters = []
                    target_clusters = random.sample(list(range(2, int(num_clusters*0.5))), num_servers-1)
                    target_clusters.append(num_clusters - sum(target_clusters))                               
                    server_targets = []
                    cluster_list = list(range(num_clusters))
                    for i in range(num_servers):
                        temp = random.sample(cluster_list, target_clusters[i])
                        cluster_list = [item for item in cluster_list if item not in temp]
                        server_targets.append(temp)
                    print(f'Server set with {server_targets} generated')

    #                 if mode != 'D2D_Clus':       
                #num_epochs, model_dict, node_loss, num_labels, in_channels, train, train_dist, batch_size
                client_models = local_update(dataset, num_epochs, mode_model_dict[mode], mode_trgloss_dict[mode], num_labels,
                                         in_channels, traindata, traindata_dist, rnd, batch_size)
                print(f'Update for mode {mode} with {dataset} and {num_epochs} epochs in round-{rnd} finished')
                load_dict(mode_model_dict[mode], client_models)
                del client_models

                cluster_loss = {key:[] for key in range(len(cluster_set))}
                global_avg_loss = 0
                for cluster_id, cluster_nodes in enumerate(cluster_set):
                    cluster_avg_loss = 0 
                    for node in cluster_nodes:
                        avg_loss = 0
                        for epoch in range(num_epochs):
                            avg_loss += mode_trgloss_dict[mode][node][-1*(epoch + 1)]
                        avg_loss = avg_loss / num_epochs
    #                         print(f'For mode {mode}, the average loss for node-{node} is {avg_loss:0.5f}')
                        cluster_avg_loss += avg_loss
                    cluster_avg_loss = cluster_avg_loss / len(cluster_nodes)
                    print(f'For mode {mode}, the average cluster loss for cluster-{cluster_id} is {cluster_avg_loss:0.5f}')
                    cluster_loss[cluster_id].append(cluster_avg_loss)
                    global_avg_loss += cluster_avg_loss
                global_avg_loss = global_avg_loss / num_clusters
                print(f'For mode {mode}, the average global loss for {global_avg_loss:0.5f}')
                mode_avgloss_dict[mode].append(global_avg_loss)

                # Model Aggregation
                #(cluster_set, main_dict, dataset, num_labels, in_channels)
    #                 if mode == 'D2D_Clus':
    #                     model_aggregation(cluster_set, mode_model_dict[mode], mode)
                print('Entering Aggregation')
                if mode == 'D2D' or mode == 'HD2D':
                    agg_neighborhood = []
                    for node in range(num_nodes):
                        neighborhood = [neighbors for neighbors in cluster_graph.neighbors(node)]
                        neighborhood.append(node)
                        agg_neighborhood.append(neighborhood)
                    model_aggregation(agg_neighborhood, mode_model_dict[mode], mode)
                    if mode == 'HD2D' and rnd % 5 == 0:
                        #num_labels, in_channels, dataset, main_dict, mode, server_targets
                        harchy_aggregation(num_labels, in_channels, dataset, mode_model_dict[mode], mode, server_targets)
                else:
                    aggregation_set, aggregation_graph = generate_clusters(mode, num_nodes, cluster_def[mode], overlap = 0.75)
                    model_aggregation(aggregation_set, mode_model_dict[mode], mode)
                                
                # Model testing: Accuracy and Loss calculation
                model_testing(mode_model_dict[mode], testdata, testdata_dist, mode_acc_dict[mode], mode_avgacc_dict[mode])

                print(f'Average Accuracy in round-{rnd} for mode {mode} is {mode_avgacc_dict[mode][-1]} \n')

        # SGD processing here
            elif mode == 'SGD':
                # Execute SGD based learning for the same setting ( LR, epochs, full data)s
                # num_labels, in_channels, dataset, traindata, testdata, model,  num_epochs, trg_loss, mode_acc, divergence, batch_size
                sgd_model = sgd_stage(num_labels, in_channels, dataset, traindata, testdata, mode_model_dict[mode], num_epochs,
                                      mode_trgloss_dict[mode], mode_acc_dict[mode], mode_avgacc_dict[mode], divergence_dict, batch_size)
                mode_model_dict[mode].load_state_dict(sgd_model.state_dict())
                avg_sgdloss =0
                for epoch in range(num_epochs):
                    avg_sgdloss += mode_trgloss_dict[mode][-1*(epoch + 1)]
                avg_sgdloss = avg_sgdloss / num_epochs
                mode_avgloss_dict[mode].append(avg_sgdloss)

            print(f'Training and aggregation completed-Moving to calculate Divergence')
            calculate_divergence(modes, mode_model_dict, cluster_set, num_nodes, divergence_dict)
            print(f'Divergence for round-{rnd} completed')

        if rnd > 0 and rnd % 5 == 0:
            state = [mode_acc_dict, mode_avgacc_dict, mode_trgloss_dict, mode_avgloss_dict, divergence_dict, cluster_set, cluster_graph]
            file_name = dataset.upper() + '_' + str(num_nodes) + '_' + str(num_epochs)
            file = opne(file_name, 'wb')
            pickle.dump(state, file)
            file.close()
            
    return mode_model_dict, cluster_set, mode_acc_dict, mode_avgacc_dict, mode_trgloss_dict, mode_avgloss_dict, divergence_dict, cluster_set, cluster_graph

def load_dict(target_models, source_models):
    """
    Input: target_model_dict, source_model_dict
    Takes a dictionary of target and source models.
    Loads the state of source models onto target models.
    """
    for node, model in source_models.items():
        target_models[node].load_state_dict(model.state_dict())
                      
def local_update(dataset, num_epochs, model_dict, node_trgloss, num_labels, in_channels, train, train_dist, rnd, batch_size):
    client_models = {node:Net(num_labels, in_channels, dataset).cuda() for node in range(len(model_dict))}
    opt = [optim.SGD(model.parameters(), lr = 0.01) for _, model in client_models.items()]
    
    load_dict(client_models, model_dict)    
    for node, client_model in client_models.items():
        #client_model, optimizer, train_loader, loss_list, num_epochs):
        client_update(client_model, opt[node], DataLoader(DataSubset(train, train_dist, node)), node_trgloss[node], num_epochs)
#         print(f' Loss for node {node} in the round-epoch {rnd} is {node_trgloss[node][-1]:0.5f}')
    return client_models

def model_aggregation(cluster_set, main_dict, mode):
    agg_order = random.sample(range(len(cluster_set)), len(cluster_set))
    
    for cluster_id in agg_order:
        cluster_nodes = cluster_set[cluster_id]
        # server aggregate
        aggregated_model = server_aggregate(main_dict, cluster_nodes)
        
        if mode == 'D2D':
            main_dict[cluster_id].load_state_dict(aggregated_model.state_dict())
        else:
            for node in cluster_nodes:
                main_dict[node].load_state_dict(aggregated_model.state_dict())
        
        del aggregated_model
        
def harchy_aggregation(num_labels, in_channels, dataset, main_dict, mode, server_targets):
    server_models = {server:Net(num_labels, in_channels, dataset) for server in range(len(server_targets))}
    
    for i, nodelist in enumerate(server_targets):
        agg_model =  server_aggregate(main_dict, nodelist)
        server_models[i].load_state_dict(agg_model.state_dict())
    
    global_model = server_aggregate(server_models, list(range(len(server_models))))
    
    for node in range(len(main_dict)):
        main_dict[node].load_state_dict(global_model.state_dict())
    

def sgd_stage(num_labels, in_channels, dataset, traindata, testdata, model,  num_epochs, trg_loss, mode_acc, avg_acc_dict, divergence, batch_size):
    trainloader = DataLoader(traindata, batch_size = batch_size)
    testloader = DataLoader(testdata)
    sgd_model = Net(num_labels, in_channels, dataset).cuda()
    sgd_model.load_state_dict(model.state_dict())
    
    sgd_model_optim = optim.SGD(sgd_model.parameters(), lr = 0.01)
    client_update(sgd_model, sgd_model_optim, trainloader, trg_loss, num_epochs)

    loss, acc = test(sgd_model, DataLoader(testdata))
    mode_acc.append(acc)
    avg_acc_dict.append(acc)
    print(f'Accuracy for SGD is {acc:0.5f}')

    return sgd_model
        
def model_testing(model_dict, testdata, test_dist, acc_dict, avg_acc_dict):
    avg_acc = 0
    for node, model in model_dict.items():
        loss, acc = test(model, DataLoader(DataSubset(testdata, test_dist, node)))
        acc_dict[node].append(acc)
#         print(f'Accuracy for node{node} is {acc:0.5f}')
        avg_acc += acc
    avg_acc = avg_acc / len(model_dict)
    avg_acc_dict.append(avg_acc)
#         print(f'Test loss for node{node} is {loss:0.5f}')
               