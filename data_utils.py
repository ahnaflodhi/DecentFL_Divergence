import copy
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

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

class DataSubset(Dataset):
    def __init__(self, dataset, datadist, node):
        self.dataset = dataset
        self.datadist = datadist
        self.indx = list(self.datadist[node])       
    
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.indx[item]]
        return torch.tensor(image), torch.tensor(label)
    
class MainData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
