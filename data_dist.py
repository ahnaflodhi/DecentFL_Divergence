import copy
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

def data_iid(dataset, num_nodes):
    """
    Sample I.I.D. client data for the selected dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """  
    num_users = num_nodes
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def data_noniid(dataset, num_users, shard_size):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    Assign shards(chunks) of data to each client
    :param dataset: The dataset being used
    :param num_users: Number of nodes
    :param shard_size: Number of images in a shard
    
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
#     # Making shard_size a perfect multiple of dataset size
#     # Variable datasize 
#     while not (len(dataset) // shard_size) == 0:
#         up_factor = len(dataset) // shard_size

        
#     num_imgs = shard_size
#     num_shards = int(len(dataset)//num_imgs)
    # 50,000 training imgs --> 50 imgs/shard X 1200 shards
    num_imgs = shard_size
    num_shards = int(len(dataset) / num_imgs) 
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype = int) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.targets.numpy()
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users
    

def data_noiid_skewed(dataset, num_nodes, num_classes, class_assn = None, niid = False, real = True):     
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """    
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    
    ## Regular Class Assignment
    ## Assign class_per_node = num_classes
    if not niid or real or class_assn == None:
        class_per_node = num_classes
    class_num_nodes = [class_per_node for _ in range(num_nodes)]
    
    for i in range(num_classes):
        selected_clients = []
        for client in range(num_nodes):
            if class_num_client[client] > 0:
                selected_clients.append(client)
        if niid and not real:
            selected_clients = selected_clients[:int(num_nodes/num_classes*class_per_client)]