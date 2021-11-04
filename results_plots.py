import time
import matplotlib.pyplot as plt
import pickle
import copy
import networkx as nx

def title_gen(label, dataset_label, mode, distmode, num_nodes, cluster, epochs):
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = './Results/No_Strag/' + label + '_' + dataset_label +'_'+  mode + '_' + distmode + '_' + 'n' + str(num_nodes) + '_' + 'c' + str(cluster) + '_' + 'e' + str(epochs) + '_' + timestr
    return filename

def final_list(cluster_set, num_nodes):
    # Remove nodes not part of any cluster
    nodes_list= list(range(num_nodes))
    exclude = []
    for node in nodes_list:
        if any(node in cluster_list for cluster_list in cluster_set):
            continue
        else:
            exclude.append(node)
    final_nodes = copy.deepcopy(nodes_list)
    for absent_nodes in exclude:
        final_nodes.remove(absent_nodes)
    return final_nodes

def plot_metric(metric, modes, dataset_label, label, distmode, cluster_set, num_nodes, epochs, cluster):
    timestr = time.strftime("%Y%m%d-%H%M")
    final_nodes = final_list(cluster_set, num_nodes)
    for mode in modes:
        if mode != 'sgd':
            fig = plt.figure(figsize = (15,10))
            for node in final_nodes:
                plt.grid()
                plt.plot(metric[mode][node])
                plt.title('%s for mode %s' %(label, mode))
                filename = title_gen(label, dataset_label, mode, distmode, num_nodes, cluster, epochs)
#             filename = './Results/' + label + '_' + dataset_label +'_'+  mode + '_' + distmode + '_' + 'n' + str(num_nodes) + '_' + 'c' + str(cluster) + '_' + 'e' + str(epochs) + '_' + timestr
            plt.savefig(filename)
            
def plot_avgs(metric, modes, dataset_label, label, distmode, num_nodes, epochs,  cluster):
    timestr = time.strftime("%Y%m%d-%H%M")
    for mode in modes:
        fig = plt.figure(figsize = (15,10))
        plt.grid()
        plt.plot(metric[mode])
        plt.title('%s for mode %s' %(label, mode))
        filename = title_gen(label, dataset_label, mode, distmode, num_nodes, cluster, epochs)

#         filename = './Results/' + label + '_' + dataset_label +'_'+  mode + '_' + distmode + '_' +  'n' + str(num_nodes) + '_' + 'c' + str(cluster) + '_' + 'e' + str(epochs) + '_' + timestr
        plt.savefig(filename)
# dataset, dist_mode, test_acc, trg_loss, avg_loss, divergence_dict, cluster_set            
def save_results(dataset_label, distmode, test_acc, test_avgacc, trg_loss, avg_loss, divergence_dict, cluster_set, num_nodes, cluster, epochs):
    timestr = time.strftime("%Y%m%d-%H%M")
    folder = './Results_2/' 
    filename =  dataset_label + '_' + distmode + '_' + 'n' + str(num_nodes) + '_' + 'c' + str(cluster) + '_' + 'e' + str(epochs) + '_' +  timestr
    final_path = folder +filename
    results = [test_acc, trg_loss, avg_loss, divergence_dict, test_avgacc, cluster_set]
#     results = {'test_acc':test_acc, 'trg_loss':trg_loss, 'avg_loss':avg_loss, 'divergence_dict':divergence_dict, 'test_avgacc':test_avgacc, 'cluster_set':cluster_set}
    f = open(final_path, 'wb')
    pickle.dump(results, f)
    return filename
    
def plot_bar(file_name, file_path, dest_path):
    """ 
    Plots results for previous mode of saved results (final round accuracy/no average accuracy recorder.
    All such results are saved in ./Results/Results-2/.
    Takes file_name, path to file and path of the destination where images are to be saved.
    """
    file = file_path + file_name
    with open(file, 'rb') as f:
        state =  pickle.load(f)
    num_nodes = len(state[0]['d2d_clus'])
    labels = ['mnist', 'cifar']
    
    cats = file_name.split('_')
        
    for dataset in labels:
        if dataset in cats[0]:
            label = dataset
            break
            
    dists = ['niid2', 'niid1', 'niid']
    for label_dist in dists:
        if label_dist in cats[1]:
            dist = label_dist
            break
        else:
            dist = 'iid'
            
    gap = 0.35
    keys = np.array(list(range(num_nodes)))
    fig1 = plt.figure(figsize = (20, 10))
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    plt.grid()
    plt.bar(keys-gap/2, state[0]['d2d_clus'].values(), gap-0.2, label = 'd2d_clus')
    plt.bar(keys, state[0]['d2d'].values(), gap-0.2)
    plt.bar(keys+gap/2, state[0]['centr_fed'].values(), gap-0.2)
    plt.legend(['Clustered D2D', 'D2D', 'Centr FL'])
    title_acc = 'Accuracy Comparison for ' + label.upper() + ' : ' + dist.upper()
    plt.title(title_acc, size = 30)
    plt.xlabel('Nodes', size = 20)
    plt.ylabel('Accuracies', size = 20)
    plt.savefig(dest_path + title_acc + cats[2])
    
    for mode in state[0].keys():
        if mode != 'sgd':
            fig = plt.figure(figsize = (15,10))
            plt.grid()
            title_loss = 'Loss Comparison for ' + mode.upper() + ' ' + label.upper() + ' : ' + dist.upper()
            for node in state[1][mode].keys():
                plt.plot(state[1][mode][node])
                plt.xlabel('Aggregation Rounds', size = 20)
                plt.ylabel('Training Loss', size = 20)
                plt.title(title_loss, size = 30)
            plt.savefig(dest_path + title_loss + cats[2])
    
    fig = plt.figure(figsize  = (15, 10))
    plt.grid()
    
    for mode in state[0].keys():
        title_avgloss = 'Average Loss Comparison for ' + label.upper() + ' : ' + dist.upper()
        if mode != 'sgd':
            plt.plot(state[2][mode])
        plt.legend([mode for mode in state[0].keys() if mode != 'sgd'])
        plt.title(title_avgloss, size = 25)
        plt.xlabel('Aggregation Rounds', size = 20)
        plt.ylabel('Average Training Loss for Nodes', size = 20)
        plt.savefig(dest_path + title_avgloss + cats[2])
        
    nonsgd_modes = ['d2d_clus', 'd2d', 'centr_fed']
    divergence = {mode:[] for mode in nonsgd_modes}
    for mode in nonsgd_modes:
        for i in range(len(state[3][0][mode])):
            divergence[mode].append(state[3][0][mode][i].item())  
    keys = np.array(list(range(len(divergence[mode]))))
    i = -0.3
    plt.figure(figsize = (25, 10))
    plt.bar( keys+i, divergence['d2d_clus'], 0.2)
    plt.bar( keys, divergence['d2d'], 0.2)
    plt.legend(['D2D_Clus', 'D2D'])
    plt.xlabel('Rounds', size = 20)
    title_div = 'Divergence Comparison for ' + label.upper() + ' : ' + dist.upper()
    plt.ylabel(title_div, size =20)
    plt.title(title_div, size =30)
    plt.savefig(dest_path + title_div + cats[2])

def plot_cluster(graph, dataset, dist_mode, nodes, epochs, num_clusters):    
    nx.draw(graph, with_labels = True, font_weight = 'bold')
    plt.title('Clustered D2D Setting' + + 'n' + nodes + '_' + 'e' + epochs + '_' + 'c' + num_clusters + 
              dataset + '_' + distmode + '_' + 'n' + nodes + '_' + 'e' + epochs + '_' + 'c' + num_clusters )
    file_name = './Results/No_Strag/'+ dataset + '_' + dist_mode + '_' + 'n' + nodes + '_' + 'e' + epochs + '_' + 'c' + num_clusters
    plt.savefig(file_name)
    
