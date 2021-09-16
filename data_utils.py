import copy
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

from DNN import *

class DataSubset(Dataset):
    """
    Takes the dataset, distribution list and node as arguments.
    """
    
    def __init__(self, dataset, datadist, node):
        self.dataset = dataset
        self.datadist = datadist
        self.indx = list(self.datadist[node])       
    
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.indx[item]]
        return torch.tensor(image), torch.tensor(label)

def dataset_select(dataset):
    ## MNIST
    if dataset == 'mnist':
        ### Choose transforms
        transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,)),
                                    ])

        ### Create Train and Test Dataets
        traindata = torchvision.datasets.MNIST("../data/", train = True, download = True, transform = transform)
        testdata = torchvision.datasets.MNIST(root = '../data/', train = False, download = True, transform = transform)

    ## CIFAR
    elif dataset == 'cifar':
        ### Choose transforms
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        ###
        traindata = torchvision.datasets.CIFAR10(root='../data/', train = True, download = True, transform = transform)
        testdata = torchvision.datasets.CIFAR10(root = '../data/', train = False, download = True, transform = transform)
        
    else:
        raise NotImplementedError
      
    return traindata, testdata

def dict_creator(modes, dataset, num_labels, in_channels, num_nodes, num_rounds, wt_init):
        
    # Same weight initialization
    if wt_init == 'same':       
        same_wt_basemodel = Net(num_labels, in_channels, dataset)
        model_dict = {i:copy.deepcopy(same_wt_basemodel).cuda() for i in range(num_nodes)}
    elif wt_init == 'diff':
        model_dict = {i:Net(num_labels, in_channels, dataset).cuda() for i in range(num_nodes)}
    
    recorder = {key:[] for key in range(num_nodes)}
    ## Model Dictionary for each of the Fed Learning Modes
    ## Model Dictionary Initialization

    mode_model_dict = {key:None for key in modes}
    mode_trgloss_dict = {key:None for key in modes}
    mode_testloss_dict = {key:None for key in modes}
    mode_avgloss_dict = {key:[] for key in modes}
    mode_acc_dict = {key:None for key in modes}
    mode_avgacc_dict = {key:[] for key in modes}
    divergence_dict = {key:None for key in range(num_rounds)}
    
    # Create separate copies for each mode
    for mode in modes:
        if mode != 'sgd':
            mode_model_dict[mode] = copy.deepcopy(model_dict)
            mode_trgloss_dict[mode] = copy.deepcopy(recorder)
            mode_testloss_dict[mode] = copy.deepcopy(recorder)
            mode_acc_dict[mode] = copy.deepcopy(recorder)
        elif mode == 'sgd':
            mode_model_dict[mode] = copy.deepcopy(same_wt_basemodel).cuda()
            mode_trgloss_dict[mode] = []
            mode_testloss_dict[mode] = []
            mode_acc_dict[mode] = []
    return mode_model_dict, mode_trgloss_dict, mode_testloss_dict, mode_avgloss_dict, mode_acc_dict, mode_avgacc_dict, divergence_dict

        
