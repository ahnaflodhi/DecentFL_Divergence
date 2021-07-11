import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

#The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
from data_utils import dataset_select, DataSubset  # Returns the train and test sets for the chosen dataset; dataset_select and class DataSubset
from data_dist import * # (Returns the dictionary of nodes/data partitions for both iid and nidd) )
from DNN import * # (Returns Network, client update, aggregate)
from d2denvironment import *  #(returns cluster_set, graph, Laplacian and Eigenvalues)
from train_round import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', type = str, default = 'mnist', help='Name of the dataset : mnist or cifar')
parser.add_argument('-n', type = int, default = 30, help='Number of ndoes')
parser.add_argument('-c', type = int, default = 5, help='Number of clusters')
parser.add_argument('-e', type = int, default = 5, help='Number of clusters')
parser.add_argument('-r', type = int, default = 5, help='Number of federation rounds')
parser.add_argument('-o', type = float, default = 0.75, help='Overlap factor in cluser boundaries')
parser.add_argument('-s', type = int, default =50, help = ' Shard size for Non-IID distribution')
args = parser.parse_args()

dataset = args.d
nodes = args.n
clusters = args.c
epochs = args.e
rounds = args.r
overlap_factor = args.o
shards =args.s

def D2DFL(dataset, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap):
    # Step 1: Define parameters for the environment, dataset and dataset distribution
    if dataset == 'mnist' or dataset == 'cifar': # Num labels will depend on the class in question
        num_labels = 10
    else:
        raise NotImplementedError('The required dataset has not been implemented')

    #### Step 2: Import Dataset partitioned into train and testsets
    # Call data_select from data_utils
    traindata, testdata = dataset_select(dataset)

    #### Step 3: Divide data among the nodes according to the distribution IID or non-IID
    # Call data_iid/ data_noniid from data_dist
    # partitions = data_iid(train, num_nodes)
    # train_dist = data_iid(traindata, num_nodes)
    train_dist = data_noniid(traindata, num_nodes, shard_size)
    test_dist = data_iid(testdata, num_nodes)

    ### Step 4: Instantiate DNN models (same weight or different weights) at all nodes. Choose random cluster, update local model and aggregate in the cluster
    cs, cg = generate_clusters(num_nodes, num_clusters, overlap)

    ### Step5: Call federate function to start training
    model_collection, test_acc, trg_loss, test_loss = federate(num_rounds, num_epochs, num_nodes, cs, num_labels,
                                     traindata, train_dist, testdata, test_dist)

    return model_collection, test_acc, trg_loss, test_loss

if __name__ == "__main__":
    models, test_acc, trg_loss, test_loss = D2DFL(dataset, nodes, clusters, rounds, epochs, shards, overlap_factor)