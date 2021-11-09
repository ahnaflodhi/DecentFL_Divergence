import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, num_classes, in_ch, dataset):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        if dataset == 'mnist' or dataset == 'fashion':
            self.fc1 = nn.Linear(2000, 1000)
            self.fc2 = nn.Linear(1000, num_classes)
        elif dataset == 'cifar':
            self.fc1 = nn.Linear(2880, 1440)
            self.fc2 = nn.Linear(1440, num_classes)
            
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
def client_update(client_model, optimizer, train_loader, loss_list, num_epochs):
    client_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float()
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
#             if batch_idx % 500 == 0:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %(epoch + 1, batch_idx  + 1, running_loss / 500))
#                 loss_list.append(running_loss/500)
#                 runnin_loss = 0.0
        epoch_loss = sum(batch_loss) / len(batch_loss)
        loss_list.append(epoch_loss)
#     return client_model

def server_aggregate(client_models, node_list):
    agg_model = copy.deepcopy(client_models[0])
    ref_dict = agg_model.state_dict()
    for k in ref_dict.keys():
        ref_dict[k] = torch.stack([client_models[node].state_dict()[k].float() for node in node_list], 0).mean(0)
    agg_model.load_state_dict(ref_dict)
    return agg_model

def model_checker(model1, model2):
    models_differ = 0
    for modeldata1, modeldata2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(modeldata1[1], modeldata2[1]):
            pass
        else:
            models_differ += 1
            if (modeldata1[0] ==  modeldata2[0]):
                print("Mismatch at ", modeldata1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match')


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

def extract_weights(model):
    weights = {}
    for key in model.state_dict():
        if 'weight' not in key:
            continue
        weights[key] = model.state_dict()[key]
    return weights

def model_checker(model1, model2):
    models_differ = 0
    for modeldata1, modeldata2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(modeldata1[1], modeldata2[1]):
            pass
        else:
            models_differ += 1
            if (modeldata1[0] ==  modeldata2[0]):
                print("Mismatch at ", modeldata1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match')

def calculate_divergence(modes, main_model_dict, cluster_set, num_nodes, divergence_results):
    centr_fed_ref = np.random.randint(0, num_nodes)
    for mode in modes:
        basemodel_keys = main_model_dict[mode][0].state_dict().keys()
        break
     # Structure of Dictionary   
    # divergence_results {mode: {node:{layer:[divergence for each round]}}       
                                      
    ref_model = main_model_dict['SGD']
    ref_weight = extract_weights(ref_model)
    
    for mode in modes:
        if mode != 'SGD':
            for target_node in range(num_nodes):
                target_model = main_model_dict[mode][target_node].cuda()
                target_weight = extract_weights(target_model)
                for layer in ref_weight.keys():                          
                    divergence_results[mode][target_node][layer].append(torch.linalg.norm(ref_weight[layer] - target_weight[layer]))

    
    
#     centr_fed_ref = np.random.randint(0, num_nodes)
#     for mode in modes:
#         basemodel_keys = main_model_dict[mode][0].state_dict().keys()
#         break
#     diverg_recorder = {key:[] for key in basemodel_keys}
#     # Dictionary to be accessed by mode-key-divergence (at the end of each round)
#     sgdmode_divergence_results = {mode:diverg_recorder for mode in modes if mode != 'SGD'}
#     # Do not inlude a list of nodes for SGD model
#     sample_nodes = {mode:[] for mode in modes if mode != 'SGD'}
#     for mode, _ in sample_nodes.items():
#         for cluster_id in range(len(cluster_set)):
#             sample_nodes[mode].append(random.choice(cluster_set[cluster_id]))
    
#     # Intra-cluster same-mode divergence
#     # i and j are node indices.
#     ref_model = main_model_dict['SGD']
#     ref_weight = extract_weights(ref_model)
#     for mode in modes:
#         if mode != 'SGD':
#             for target_id in sample_nodes[mode]:
#                 target_model = main_model_dict[mode][target_id].cuda()
#                 target_weight = extract_weights(target_model)
#                 for key in ref_weight.keys():                          
#                     sgdmode_divergence_results[mode][key].append(torch.linalg.norm(ref_weight[key] - target_weight[key]))
    return  divergence_results    
    
                     