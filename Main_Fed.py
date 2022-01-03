import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

#The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
parser.add_argument('-n', type = int, default = 40, help='Number of nodes')
parser.add_argument('-c', type = int, default = 7, help='Number of clusters')
parser.add_argument('-e', type = int, default = 2, help='Number of epochs')
parser.add_argument('-r', type = int, default = 30, help='Number of federation rounds')
parser.add_argument('-o', type = float, default = 0.75, help='Overlap factor in cluser boundaries')
parser.add_argument('-s', type = int, default = 50, help = ' Shard size for Non-IID distribution')
parser.add_argument('-prop', type = float, default = 1.0, help = 'Proportion of nodes chosen for aggregation in CFL/HFL: 0.0-1.0')
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
proportion = args.prop

modes_list = {'d2d':None, 'hd2d':None, 'hfl': None, 'chd2d':None, 'hch_d2d': None, 'gossip': None, 'hgossip':None, 'cfl': None, 'sgd': None}

def D2DFL(dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist, prop):
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
    
    # Select Cluster Heads
    cluster_heads = [random.sample((environment.cluster_set[i]),1)[0] for i in range(environment.num_clusters)]    
    
    # Mode Dictionary
    # name, dataset, num_epochs, num_rounds, num_nodes, dist, base_model, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, batch_size, nhood, env_Lp, num_clusters
    base_params = { 'dataset' : dataset, 'num_epochs' : num_epochs, 'num_rounds' : num_rounds, 'num_nodes' : num_nodes, 'dist' : dist, 'base_model' : base_model,
                   'num_labels' : num_labels, 'in_channels' : in_ch, 'traindata' : traindata, 'traindata_dist' : train_dist, 'testdata' : testdata, 
                   'testdata_dist' : test_dist, 'batch_size' : batch_size, 'nhood' : environment.neighborhood_map, 'env_Lp' : environment.Lp,
                   'num_clusters' : num_clusters}   
    """
    Add Flags if another mode is added.
    Flags for gossip, cfl and sgd remain None.
    """
    d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    hd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    hfl_flags = {'d2d_agg_flg' : False, 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    chd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    hch_d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': True, 'inter_ch_agg_flg': True}
    gossip_flg = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    hgossip_flg = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    cfl_flg = {'d2d_agg_flg' : 'CServer', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    
    flag_dict = {'d2d': d2d_flags, 'hd2d': hd2d_flags, 'hfl': hfl_flags, 'chd2d':chd2d_flags, 'hch_d2d': hch_d2d_flags, 
                 'gossip':gossip_flg, 'hgossip':hgossip_flg, 'cfl':cfl_flg, 'sgd':None}
    
    # Combine base params and special flags for all modes under mode_params
    mode_params = {mode:None for mode in modes.keys()}
    for mode in modes.keys():
        if flag_dict[mode] != None:
            mode_params[mode] = {**base_params, **flag_dict[mode]}
        else:
            mode_params[mode] = base_params
        mode_params[mode]['name'] = mode
        
    for mode in modes.keys():
        if mode != 'sgd':
            # Creates Nodeset and other attributes for each mode in modes
            modes[mode] = FL_modes(** mode_params[mode])

        elif mode == 'sgd':
            sgd_model = base_model.cuda()
            modes[mode] = Servers(0, sgd_model, records = True)
            sgd_optim = optim.SGD(modes[mode].model.parameters(), lr = 0.01)
            sgd_trainloader = DataLoader(traindata, batch_size = batch_size)
            sgd_testloader =  DataLoader(testdata)
        
    # Form once hierarchy config
    # Copy that to all valid modes for accurate comparison
    target_modes = copy.deepcopy(list(modes.keys()))
    target_modes.remove('sgd')

    for mode in modes.keys():
        if mode != 'sgd':
            # Check Hierarchical Aggregation Flag
            if modes[mode].hserver_agg_flg == True:
            # Create Hierarchical Servers
                modes[mode].form_serverset(environment.num_servers)
                # Assign Nodes
                node_list = list(range(num_nodes))
                for i in range(environment.num_servers):
                    modes[mode].serverset[i].harchy_servers_allnodes(environment.cluster_ids[i], environment.cluster_set, node_list)
                    node_list = [item for item in node_list if item not in modes[mode].serverset[i].node_ids]
                    print(f'The nodes assigned to Server-{i} are {modes[mode].serverset[i].node_ids}')

                # Assign server list to Master Server
                modes[mode].serverset[-1].node_ids = list(range(environment.num_servers))
                print(f'The nodes assigned to Global Server are {modes[mode].serverset[-1].node_ids}')

                target_modes.remove(mode)
                for dest_mode in target_modes:
                    if modes[dest_mode].hserver_agg_flg == True:
                        modes[dest_mode] = copy.deepcopy(modes[mode])
                break
            else:
                target_modes.remove(mode)
        
    ### Step5: Call federate function to start training
    #mode_model_dict, cluster_set, mode_ acc_dict, mode_trgloss_dict, mode_avgloss_dict, divergence_dict    
    for rnd in range(num_rounds):        
        for mode in modes.keys():
            print(f'Starting with mode {mode} in round-{rnd}')
            if mode != 'sgd':
                #1- Local Update
                print(f'Memory status before update r:{rnd}-mode {mode} Mbs-{torch.cuda.memory_allocated()/(1024 * 1024)}')
                modes[mode].update_round()
                print(f'Memory status after update r:{rnd}-mode {mode} Mbs-{torch.cuda.memory_allocated()/(1024 * 1024)}')
                
                #3 Test round
                modes[mode].test_round(environment.cluster_set)
                                                
                #2-Update ranking
                if rnd % 3 == 0:
                    modes[mode].ranking_round(rnd)
                
                # Base Flags
                # 'd2d_agg_flg', 'ch_agg_flg', 'hserver_agg_flg', 'inter_ch_agg_flg'
                #4 Aggregate from neighborhood
                print(f'Starting Local Aggregation in round{rnd} for mode {mode}')
                if modes[mode].d2d_agg_flg == 'D2D':
                    modes[mode].nhood_aggregate_round()
                    
                elif modes[mode].d2d_agg_flg == 'Random':
                    modes[mode].random_aggregate_round()
                
                elif modes[mode].d2d_agg_flg == 'CServer':
                    modes[mode].cfl_aggregate_round(prop)
                
                if rnd % 3 == 0:
                    if modes[mode].ch_agg_flg == True:
                        print(f'Entering Cluster Head Aggregation for mode-{mode} in round-{rnd}')
                        for i in range(environment.num_clusters):
                            cluster_head = environment.cluster_heads[i][0]
                            modes[mode].clshead_aggregate_round(cluster_head, environment.cluster_set[i], prop)
                            
                    if modes[mode].inter_ch_agg_flg == True:
                        modes[mode].inter_ch_aggregate_round(environment.cluster_heads)
                        
                if rnd % 3 == 0:
                    if modes[mode].hserver_agg_flg == True: 
                        print(f'Entering Hierarchical Aggregation for mode-{mode} in round-{rnd}')
                        for i in range(environment.num_servers):
                            modes[mode].serverset[i].aggregate_clusters(modes[mode].nodeset, prop)
                        
                        #Final Server Aggregation
                        modes[mode].serverset[-1].aggregate_servers(modes[mode].serverset[:-1], modes[mode].nodeset)
                    
            
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
    
#     plot_testacc(folder, file_name)
                
        
def title_gen(dataset, dist, num_nodes, num_clusters, num_epochs, num_rounds):
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = 'Final_' + dataset.upper() + '_' + dist.upper()  + '_' +'n'+ str(num_nodes)  + '_' + 'c' + str(num_clusters)  + '_' +'e' + str(num_epochs) + '_' + 'r' + str(num_rounds) + '_' + timestr
    return filename

                

if __name__ == "__main__":
#     dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist
    mode_state = D2DFL(dataset, batch_size, test_batch_size, modes_list,  nodes, clusters, rounds, epochs, shards, overlap_factor, dist_mode, proportion)

    
    
