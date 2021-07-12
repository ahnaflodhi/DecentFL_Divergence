import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from network import *
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, num_classes, in_ch, dataset):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        if dataset == 'mnist':
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, num_classes)
        elif dataset == 'cifar':
            self.fc1 = nn.Linear(500, 250)
            self.fc2 = nn.Linear(250, num_classes)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        reshaped = torch.prod(torch.tensor(x.shape), 0)
        x = x.view(-1, reshaped.item())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)
    
    
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
    
def client_update(client_model, optimizer, train_loader, num_epoch):
    client_model.train()
    for e in tqdm(range(num_epoch)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data =data.float()
#             print(data.shape)
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()

def server_aggregate(client_models, num_labels, in_channels, dataset):
    base_model = Net(num_labels, in_channels, dataset)
    base_dict = base_model.state_dict()
    for k in base_dict.keys():
        base_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    base_model.load_state_dict(base_dict)
    return base_model

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

def calculate_divergence(modes, main_model_dict, cluster_set, num_nodes):
    centr_fed_ref = np.random.randint(0, num_nodes)
    intramode_divergence_results = {mode:[] for mode in modes if mode != 'sgd'}
    sgdmode_divergence_results = {mode:[] for mode in modes if mode != 'sgd'}
    # Do not inlude a list of nodes for SGD model
    sample_nodes = {mode:[] for mode in modes if mode != 'sgd'}
    samples = []
    for mode, _ in sample_nodes.items():
        for cluster_id in range(len(cluster_set)):
            samples.append(random.choice(cluster_set[cluster_id]))
        sample_nodes[mode] = samples
    
    # Intra-cluster same-mode divergence
    # i and j are node indices.
    ref_model = main_model_dict['sgd']
    ref_weight = extract_weights(ref_model)
    for mode in modes:
        if mode != 'sgd':
            for target_id in sample_nodes[mode]:
                target_model = main_model_dict[mode][target_id].cuda()
                target_weight = extract_weights(target_model)
                for key in ref_weight.keys():                          
                    sgdmode_divergence_results[mode].append(torch.linalg.norm(ref_weight[key] - target_weight[key]))
    print('SGD Divergence concluded')
    return  sgdmode_divergence_results    
    
                          
        
#         if mode != 'sgd' and mode != 'centr_fed':
#         # Same number of sample nodes as cluster_sets
#             for i in range(len(sample_nodes[mode])):
#                 j = i+1
#                 if j <= len(sample_nodes[mode]):
#                     print(f'i-{i} and j-{j}')
#                     ref_model = main_model_dict[mode][sample_nodes[mode][i]]
#                     target_model = main_model_dict[mode][sample_nodes[mode][j]]
#                     ref_weight = extract_weights(ref_model)
#                     target_weight = extract_weights(target_model)
#                     for key in ref_weight.keys():
#                         intramode_divergence_results[mode].append(torch.linalg.norm(ref_weight[key] - target_weight[key]))
#                     j += 1
#             print('Non-SGD divergence Concluded')
#         else:
#             ref_model = main_model_dict[mode]
#             ref_weight = extract_weight(ref_model)
#             for mode in modes:
#                 if mode != 'sgd':
#                     for target_id in sample_nodes[mode]:
#                         target_model = main_model_dict[mode][target_id]
#                         target_weight = extract_weight(target_model)
#                         for key in ref_weight.keys():                          
#                             sgdmode_divergence_results[mode].append(torch.linalg.norm(ref_weight[key] - target_weight[key]))
#             print('SGD Divergence concluded')
                                

#     return  intramode_divergence_results, sgdmode_divergence_results
            
            