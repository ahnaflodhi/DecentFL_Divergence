import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

#The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import pickle
import time
from data_utils import * # Returns the train and test sets for the chosen dataset; dataset_select and class DataSubset
from data_dist import * # (Returns the dictionary of nodes/data partitions for both iid and nidd) )
from DNN import * # (Returns Network, client update, aggregate)
from env_sysmodel import system_model, FL_modes
from devices import Nodes, Servers
from plots import plot_testacc

parser = argparse.ArgumentParser()
parser.add_argument('-b', type = int, default = 8, help = 'Batch size for the dataset')
parser.add_argument('-t', type = int, default = 8, help = 'Batch size for the test dataset')
parser.add_argument('-d', type = str, default = 'mnist', help='Name of the dataset : mnist, cifar10 or fashion')
parser.add_argument('-n', type = int, default = 30, help='Number of nodes')
parser.add_argument('-c', type = int, default = 5, help='Number of clusters')
parser.add_argument('-e', type = int, default = 1, help='Number of epochs')
parser.add_argument('-r', type = int, default = 30, help='Number of federation rounds')
parser.add_argument('-o', type = float, default = 0.75, help='Overlap factor in cluser boundaries')
parser.add_argument('-s', type = int, default = 50, help = ' Shard size for Non-IID distribution')
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

modes= {'d2d':None, 'hd2d':None, 'gossip':None, 'cfl':None, 'sgd':None}

def D2DFL(dataset, batch_size, test_batch_size, mode_list, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist):
    # Step 1: Define parameters for the environment, dataset and dataset distribution
    if dataset == 'mnist': # Num labels will depend on the class in question
        location = '../data/'
        num_labels = 10
        in_ch = 1
    elif dataset == 'cifar':
        location = '../data/'
        num_labels = 10
        in_ch = 3
    elif dataset == 'fashion':
        location = '../data/'
        num_labels = 10
        in_ch = 1
    
    base_model = Net(num_labels, in_ch, dataset)
    
    #### Step 2: Import Dataset partitioned into train and testsets
    # Call data_select from data_utils
    traindata, testdata = dataset_select(dataset, location)

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
    
    # Step 4: Create Environment
    environment = system_model(num_nodes, num_clusters)
    print(f'Number of servers is {environment.num_servers}')
    # Generate Worker models on GPU to be used by all modes: self.model_list
#     environment.workermodels(base_model)
    
    for mode in modes.keys():
        if mode != 'sgd':
            # Creates Nodeset and other attributes for each mode in modes
            modes[mode] = FL_modes(mode, dataset, num_epochs, num_rounds, num_nodes, dist, base_model, num_labels, in_ch, traindata, train_dist, 
                                   testdata, test_dist, batch_size, environment.neighborhood_map, environment.Lp, num_clusters)
            if mode == 'hd2d': 
                # Create Hierarchical Servers
                modes[mode].form_serverset(environment.num_servers)
                # Assign Nodes
                for i in range(environment.num_servers):
                    modes[mode].serverset[i].harchy_servers(environment.cluster_ids[i], environment.cluster_set)
                    print(f'The nodes assigned to Server-{i} are {modes[mode].serverset[i].node_ids}')
                # Assign server list to Master Server
                modes[mode].serverset[-1].node_ids = list(range(environment.num_servers))
                print(f'The nodes assigned to Global Server are {modes[mode].serverset[-1].node_ids}')
                
        elif mode == 'sgd':
            sgd_model = base_model.cuda()
            modes[mode] = Servers(0, sgd_model, records = True)
            sgd_optim = optim.SGD(modes[mode].model.parameters(), lr = 0.01)
            sgd_trainloader = DataLoader(traindata, batch_size = batch_size)
            sgd_testloader =  DataLoader(testdata)
            
        
    ### Step5: Call federate function to start training
    #mode_model_dict, cluster_set, mode_ acc_dict, mode_trgloss_dict, mode_avgloss_dict, divergence_dict    
    for rnd in range(num_rounds):        
        for mode in modes:
            print(f'Starting with mode {mode} in round-{rnd}')
            if mode != 'sgd':
                #1- Local Update
                print(f'Memory status before update r:{rnd}-mode {mode} Mbs-{torch.cuda.memory_allocated()/(1024 * 1024)}')
                modes[mode].update_round()
                print(f'Memory status after update r:{rnd}-mode {mode} Mbs-{torch.cuda.memory_allocated()/(1024 * 1024)}')
                
                #2-Update ranking
                modes[mode].ranking_round(rnd)
                
                #3 Test round
                modes[mode].test_round(environment.cluster_set)
                
                #4 Aggregate from neighborhood
                print(f'Starting Aggregation in round{rnd} for mode {mode}')
                if mode == 'd2d':
                    modes[mode].aggregate_round()
                elif mode == 'hd2d':
                    modes[mode].aggregate_round()
                    if rnd % 5 == 0:
                        for i in range(environment.num_servers):
                            modes[mode].serverset[i].aggregate_clusters(modes[mode].nodeset)
                        
                        #Final Server Aggregation
                        modes[mode].serverset[-1].aggregate_servers(modes[mode].serverset[:-1], modes[mode].nodeset)
                elif mode == 'gossip':
                    modes[mode].random_aggregate_round()
                    
                elif mode == 'cfl':
                    modes[mode].cfl_aggregate_round()
            
            elif mode == 'sgd':
                node_update(modes[mode].model, sgd_optim, sgd_trainloader, modes[mode].avgtrgloss, modes[mode].avgtrgacc, num_epochs)
                loss, acc = test(modes[mode].model,sgd_testloader)
                modes[mode].avgtestloss.append(loss)
                modes[mode].avgtestacc.append(acc)
        
        if rnd % 5 == 0:
            filename = dataset.upper() + '_' + dist.upper()  + '_' +'n'+ str(num_nodes)  + '_' + 'c' + str(num_clusters)  + '_' +'e' + str(num_epochs) + '_' + 'r' + str(num_rounds)
            with open(filename, 'wb') as f:
                pickle.dump(modes, f)
            
    folder = '../Results'        
    file_name = title_gen(dataset, dist, num_nodes, num_clusters, num_epochs, num_rounds)
    file_path = os.path.join(folder, file_name)
    with open(file_path, 'wb') as ffinal:
        pickle.dump(modes, ffinal)
    
    plot_testacc(folder, file_name)
                
        
                
def title_gen(dataset, dist, num_nodes, num_clusters, num_epochs, num_rounds):
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = 'Final_' + dataset.upper() + '_' + dist.upper()  + '_' +'n'+ str(num_nodes)  + '_' + 'c' + str(num_clusters)  + '_' +'e' + str(num_epochs) + '_' + 'r' + str(num_rounds) + '_' + timestr
    return filename

                

if __name__ == "__main__":
    #mode_model_dict, cluster_set, mode_acc_dict, mode_trgloss_dict, mode_avgloss_dict, mode_testloss_dict, divergence_dict
    mode_state = D2DFL(dataset, batch_size, test_batch_size, modes,  nodes, clusters, rounds, epochs, shards, overlap_factor, dist_mode)

    
    