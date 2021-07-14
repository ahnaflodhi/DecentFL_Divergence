import time
import matplotlib.pyplot as plt

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

def plot_metric(metric, modes, dataset_label, label, distmode, cluster_set, num_nodes):
    timestr = time.strftime("%Y%m%d-%H%M")
    final_nodes = final_list(cluster_set, num_nodes)
    for mode in modes:
        if mode != 'sgd':
            fig = plt.figure(figsize = (15,10))
            for node in final_nodes:
                plt.plot(metric[mode][node])
                plt.title('%s for mode %s' %(label, mode))
                plt.grid()
            filename = './Results/' + label + '_' + dataset_label +'_'+  mode + '_'+ '_' + distmode + '_' + timestr
            plt.savefig(filename)
            
def save_results(dataset_label, distmode, test_acc, trg_loss, divergence_dict, cluster_set):
    timestr = time.strftime("%Y%m%d-%H%M")
    filename = './Results/results'  + dataset_label +'_'+  mode + '_'+ '_' + distmode + '_' + timestr
    results = [test_acc, trg_loss, divergece_dict, cluster_set]
    f = open(filename, 'wb')
    pickle.dump(results, f)
    
