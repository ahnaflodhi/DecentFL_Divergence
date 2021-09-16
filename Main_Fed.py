import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

#The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import argparse
import pickle
from data_utils import dataset_select, DataSubset  # Returns the train and test sets for the chosen dataset; dataset_select and class DataSubset
from data_dist import * # (Returns the dictionary of nodes/data partitions for both iid and nidd) )
from DNN import * # (Returns Network, client update, aggregate)
from d2denvironment import *  #(returns cluster_set, graph, Laplacian and Eigenvalues)
from train_round import federate
from results_plots import plot_metric, save_results, plot_avgs

parser = argparse.ArgumentParser()
parser.add_argument('-b', type = int, default = 8, help = 'Batch size for the dataset')
parser.add_argument('-t', type = int, default = 8, help = 'Batch size for the test dataset')
parser.add_argument('-d', type = str, default = 'mnist', help='Name of the dataset : mnist or cifar')
parser.add_argument('-n', type = int, default = 30, help='Number of ndoes')
parser.add_argument('-c', type = int, default = 5, help='Number of clusters')
parser.add_argument('-e', type = int, default = 2, help='Number of epochs')
parser.add_argument('-r', type = int, default = 5, help='Number of federation rounds')
parser.add_argument('-o', type = float, default = 0.75, help='Overlap factor in cluser boundaries')
parser.add_argument('-s', type = int, default =50, help = ' Shard size for Non-IID distribution')
parser.add_argument('-m', type = str, default = 'niid', help = 'Data distribution mode (IID, non-IID, 1-class and 2-class non-IID: iid, niid, niid1 or niid2.')
args = parser.parse_args()

dataset = args.d
batch_size = args.b
nodes = args.n
clusters = args.c
epochs = args.e
rounds = args.r
overlap_factor = args.o
shards =args.s
dist_mode = args.m
test_batch_size = args.t

modes = ['d2d_clus', 'd2d', 'centr_fed','sgd']
# modes = ['d2d_clus', 'sgd']
num_clusters = [clusters, int(nodes/2), 1, None]
# num_clusters = [clusters, None]
num_clusters = dict(zip(modes, num_clusters))

def D2DFL(dataset, batch_size, test_batch_size, mode_list, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist):
    # Step 1: Define parameters for the environment, dataset and dataset distribution
    if dataset == 'mnist': # Num labels will depend on the class in question
        num_labels = 10
        in_ch = 1
    elif dataset == 'cifar':
        num_labels = 10
        in_ch =3        

    #### Step 2: Import Dataset partitioned into train and testsets
    # Call data_select from data_utils
    traindata, testdata = dataset_select(dataset)

    #### Step 3: Divide data among the nodes according to the distribution IID or non-IID
    # Call data_iid/ data_noniid from data_dist
    if dist == 'iid':
        train_dist = data_iid(traindata, num_nodes)
    elif dist == 'niid':
        train_dist = data_noniid(traindata, num_nodes, shard_size)
    elif dist == 'niid1':
        skew = 1
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew, shard_size)
    elif dist == 'niid2':
        skew = 2
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew, shard_size)
    
    test_dist = data_iid(testdata, num_nodes)

    ### Step5: Call federate function to start training
    #mode_model_dict, cluster_set, mode_acc_dict, mode_trgloss_dict, mode_avgloss_dict, divergence_dict
    model_collection, cluster_set, test_acc, test_avgacc, trg_loss, avg_loss, divergence_dict = federate(dataset, mode_list, num_rounds, num_epochs, num_nodes, num_clusters,
                                                                       num_labels, in_ch, traindata, train_dist, testdata, test_dist, batch_size, test_batch_size)
    
    return model_collection, cluster_set, test_acc, test_avgacc, trg_loss, avg_loss, divergence_dict

if __name__ == "__main__":
    #mode_model_dict, cluster_set, mode_acc_dict, mode_trgloss_dict, mode_avgloss_dict, mode_testloss_dict, divergence_dict
    model_collection, cluster_set, test_acc, test_avgacc, trg_loss, avg_loss, divergence_dict = D2DFL(dataset, batch_size, test_batch_size, modes,  nodes, num_clusters, 
                                                                                         rounds, epochs, shards, overlap_factor, dist_mode)
    #dataset_label, distmode, test_acc, trg_loss, avg_loss, divergence_dict, cluster_set, num_nodes, cluster, epochs
    #saves in order test_acc, trg_loss, avg_loss, divergence_dict, cluster_set
    save_results(dataset, dist_mode, test_acc, test_avgacc, trg_loss, avg_loss, divergence_dict, cluster_set, nodes, clusters, epochs) 
    plot_metric(test_acc, modes, dataset, 'Acc', dist_mode, cluster_set, nodes, epochs, clusters)
    plot_metric(trg_loss, modes, dataset, 'Loss', dist_mode, cluster_set, nodes, epochs, clusters)
    plot_avgs(avg_loss, modes, dataset, 'Average Loss', dist_mode, nodes, epochs, clusters)
    plot_avgs(test_avgacc, modes, dataset, 'Average Accuracy', dist_mode, nodes, epochs, clusters)
    
    