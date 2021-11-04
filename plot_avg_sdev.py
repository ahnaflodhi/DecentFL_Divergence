import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import statistics
from results_plots import final_list


def plot_avgsdev(metric, file_name, folder, dest_path, active_ref = True):
    file = os.path.join(folder, file_name)
    with open(file, 'rb') as f:
        state =  pickle.load(f)
    if len(state) < 6:
        return
    labels = ['mnist', 'cifar', 'fashion']    
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

    modes = ['D2D_Clus', 'D2D', 'HD2D', 'Gossip', 'Centr_Fed']
    ref_mode = 'SGD'
    metrics = ['test_acc', 'trg_loss', 'avg_loss', 'divergence', 'avg_acc', 'cluster_set', 'cluster_avgacc']
    
    metric_order = {metric:order for order, metric in enumerate(metrics)}
   
    title_div = metric.upper() + '_' +  label.upper() + '_' + dist.upper() +'-' + cats[2].upper() +'_' + cats[3] + '_' + cats[4]
    if metric == 'test_acc' or metric == 'trg_loss':
        mode_roundstat = {mode:[] for mode in modes}
        mode_roundupper = {mode:[] for mode in modes}
        mode_roundlower = {mode:[] for mode in modes}
        mode_round_avgstat = {mode:[] for mode in modes}
        
        figure = plt.figure(figsize = (15, 10))
        for mode in modes:
            # List of nodes part of clusters
            nodes_list = final_list(state[5], len(state[metric_order[metric]][mode]))

            for rnd in range(len(state[metric_order[metric]][mode][0])):
                round_stat = []
                round_mean = 0
                round_sdev = 0

                for node in nodes_list:
                    round_stat.append(state[metric_order[metric]][mode][node][rnd])

                mode_roundstat[mode].append(round_stat)
                round_mean = statistics.mean(round_stat)
                round_sdev = statistics.stdev(round_stat)

                mode_round_avgstat[mode].append(round_mean)
                mode_roundupper[mode].append(round_mean + round_sdev)
                mode_roundlower[mode].append(round_mean - round_sdev)

            plt.grid()
            plt.plot(mode_round_avgstat[mode])
            plt.fill_between(list(range(len(mode_roundupper[mode]))), mode_roundupper[mode], mode_roundlower[mode], alpha = 0.3)

        if active_ref:
            plt.plot(state[metric_order[metric]][ref_mode])
            modes.append(ref_mode)
            plt.legend(modes)
            modes.remove(ref_mode)
        else:
            plt.legend(modes)
        plt.xlabel('Rounds', size = 16)
        plt.ylabel(metric.upper(), size = 16)
        plt.title(title_div, size = 20)
        filename = title_div + '_' + cats[5]
        save_path = os.path.join(dest_path, filename)
        plt.savefig(save_path)
        plt.close(figure)
            
    elif metric == 'avg_loss' or metric == 'avg_acc':
        "Plotting node averages"
        figure = plt.figure(figsize = (15,10))
        plt.grid()
        for mode in modes:
            plt.plot(state[metric_order[metric]][mode], linewidth = 4)
            plt.xlabel('Rounds', fontsize = 16)
            plt.ylabel(metric.upper(), fontsize = 16)
            plt.title(title_div, size = 20)
        if active_ref:
            plt.plot(state[metric_order[metric]][ref_mode], linewidth = 4)
            modes.append(ref_mode)
            plt.legend(modes, fontsize = 16)
            modes.remove(ref_mode)
        else:
            plt.legend(modes, fontsize = 14)

        filename = title_div + '_' + cats[5]
        save_path = os.path.join(dest_path, filename)
        plt.savefig(save_path)        
        plt.close(figure)
        
    elif 'divergence' in metric:
        num_rounds = len(state[metric_order[metric]])
        modes = list(state[metric_order[metric]][0].keys())
        divergence = {mode:None for mode in modes}
        figure = plt.figure(figsize = (15,10))

        for mode in modes:
            nn_divergence = {key:[] for key in state[metric_order[metric]][0][mode].keys()}
            for nn_key in state[metric_order[metric]][0][mode].keys():
                if 'weight' in nn_key:
                    for rnd in range(num_rounds):
                        round_divergence = 0
                        for i in range(len(state[metric_order[metric]][0][mode][nn_key])):
                            round_divergence += state[metric_order[metric]][rnd][mode][nn_key][i].item()
                        round_divergence = round_divergence / i
                        nn_divergence[nn_key].append(round_divergence)
            divergence[mode] = nn_divergence
            legends = []
            for keys in nn_divergence.keys():
                if 'weight' in keys:
                    plt.plot(divergence[mode][keys])
                    legends.append(keys)
            plt.legend(legends)
            plt.title(title_div, size = 20)
        filename = title_div + '_' + cats[5]
        save_path = os.path.join(dest_path, filename)
        plt.savefig(save_path)        
        plt.close(figure)
        
    elif 'cluster_avgacc' in metric:
        
        cluster_set = state[5]
        cluster_avgacc = {mode:None for mode in modes}
        cluster_avgstat = {cluster_id:[] for cluster_id in range(len(cluster_set))}
        
        for mode in modes:
            for cluster_id in range(len(cluster_set)):
                for rnd in range(len(state[0][mode][0])):
                    acc = 0
                    for node in cluster_set[cluster_id]:
                        acc += state[0][mode][node][rnd]
                    acc = acc / len(cluster_set[cluster_id])
                    cluster_avgstat[cluster_id].append(acc)
            cluster_avgacc[mode] = cluster_avgstat
        
        return cluster_avgacc
    
    elif metric == 'cluster_graph':
        cluster_set = state[5]
        figure = plt.figure(figsize = (7.5,5))
        cluster_graph = nx.Graph()
        for i in range(len(cluster_set)):
            temp = nx.complete_graph(cluster_set[i])
            cluster_graph = nx.compose(cluster_graph, temp)
            del temp
        nx.draw(cluster_graph, with_labels = True, font_weight = 'bold')
        plt.title(title_div, size = 15)
        filename = title_div + '_' + cats[5]
        save_path = os.path.join(dest_path, filename)
        plt.savefig(save_path)        
        plt.close(figure)