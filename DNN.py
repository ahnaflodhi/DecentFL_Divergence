import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import random
import copy
import heapq


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, num_classes, in_ch, dataset):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        if dataset == 'mnist' or dataset == 'fashion':
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, num_classes)
        elif dataset == 'cifar':
            self.fc1 = nn.Linear(12544, 128)
            self.fc2 = nn.Linear(128, num_classes)
            
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
   
    def visualize(feat, labels, epoch):
        plt.ion()
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()
        for i in range(10):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
        plt.xlim(xmin=-8,xmax=8)
        plt.ylim(ymin=-8,ymax=8)
        plt.text(-7.8,7.3,"epoch=%d" % epoch)
        plt.savefig('./images/epoch=%d.jpg' % epoch)
        plt.draw()
        plt.pause(0.001)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
# self.model, self.opt, self.trainset, self.trainloader, self.trgloss, self.trgacc, num_epochs   
def node_update(client_model, optimizer, train_loader, record_loss, record_acc, num_epochs):
    client_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_loss = []
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float()
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
#             correct += (output == target).float().sum()/len(data)
            batch_loss.append(loss.item())
#             if batch_idx % 500 == 0:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %(epoch + 1, batch_idx  + 1, running_loss / 500))
#                 loss_list.append(running_loss/500)
#                 runnin_loss = 0.0
#         accuracy = 100 * correct.item() / batch_idx
        epoch_loss = sum(batch_loss) / len(batch_loss)
        record_loss.append(epoch_loss)
#         record_acc.append(accuracy)
#     return client_model

def aggregate(model_list, node_list):
    agg_model = copy.deepcopy(model_list[0].model)
    ref_dict = agg_model.state_dict()
    for k in ref_dict.keys():
        ref_dict[k] = torch.stack([model_list[node].model.state_dict()[k].float() for node in node_list], 0).mean(0)
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


def clustering_divergence(model_dict, cluster_graph, num_nodes):
    divergence_results = []
    neighborhood = []
    basemodel_keys = model_dict[0].state_dict().keys()
    div_recorder = {}
    div_recorder_conv = {}
    div_recorder_fc = {}
    
    for node in range(num_nodes):
        temp = [neighbor for neighbor in cluster_graph.neighbors(node)]
        neighborhood.append(temp)
        div_recorder[node] = {neighbor:None for neighbor in temp}
        div_recorder_conv[node] = {neighbor:None for neighbor in temp}
        div_recorder_fc[node] = {neighbor:None for neighbor in temp}
        
        
    for ref_node, neighbor_nodes in enumerate(neighborhood):
        for neighbor in neighbor_nodes:
            total_diff = 0
            conv_diff = 0
            fc_diff = 0
            for layer in model_dict[ref_node].state_dict():
                if 'weight' not in layer:
                    continue
                diff = torch.linalg.norm(model_dict[ref_node].state_dict()[layer] - model_dict[neighbor].state_dict()[layer])
                total_diff += diff.item()
                if 'conv' in layer:
                    conv_diff += diff.item()
                elif 'fc' in layer:
                    fc_diff += diff.item()
                
            div_recorder[ref_node][neighbor] = total_diff
            div_recorder_conv[ref_node][neighbor] = conv_diff
            div_recorder_fc[ref_node][neighbor] = fc_diff
    
     #Normalize
    div_recorder = normalize_div(div_recorder)
    div_recorder_conv = normalize_div(div_recorder_conv)
    div_recorder_fc = normalize_div(div_recorder_fc)
            
    return div_recorder, div_recorder_conv, div_recorder_fc

def normalize_div(div_dict):
    for node in range(len(div_dict)):
        temp = []
        print(div_dict[node])
        for _ , val in div_dict[node].items():
            temp.append(val)
            norm_factor = np.linalg.norm(temp)
            
        for neighbor, _ in div_dict[node].items():
            div_dict[node][neighbor] = div_dict[node][neighbor] / norm_factor
    return div_dict

def revise_neighborhood(div_dict, n, sort_type = 'min'):
    revised_neighborhood = []
    for node in range(len(div_dict)):
        if sort_type == 'min':
            temp = heapq.nsmallest(n, div_dict[node].items() , key=lambda i: i[1])
        elif sort_type == 'max':
            temp = heapq.nlargest(n, div_dict[node].items() , key=lambda i: i[1])
        temp_nodes = [max_node[0] for max_node in temp]
        revised_neighborhood.append(temp_nodes)
    return revised_neighborhood
    
                     