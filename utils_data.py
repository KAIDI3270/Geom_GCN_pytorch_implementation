import os
import re

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import ShuffleSplit
import torch_geometric as pyg
from dgl import graph
import dgl

import utils

def load_data(dataset_name, 
              splits_file_path=None, 
              train_percentage=None, 
              val_percentage=None, 
              embedding_mode=None,
              embedding_method=None,
              embedding_method_graph=None, 
              embedding_method_space=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = utils.load_data(dataset_name)  # adj in scipy.sparse.csr.csr_matrix,  features in scipy.sparse.lil.lil_matrix, labels in numpy ndarray
        labels = np.argmax(labels, axis=-1)                             # numpy ndarray of shape (num_nodes,)
        features = features.todense()                                   # features now in numpy ndarray type with shape (num_nodes, num_dimension)
        edge_index = np.asarray(list(adj.nonzero()))
          
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name, f'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()          # each line has : node_id, feature, label with tab separations
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')                    # remove white spaces at the end of each line using rstrip()
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)       # dict : key is node_id, value is numpy array holding node_feature
                    graph_labels_dict[int(line[0])] = int(line[2])                                              # dict : key is node_id, value is integer holding node_label

        # got node features, labels. Time to get edge index
        edge_source = []
        edge_dest = []
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()                        # each line has two node_ids with tab separations
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')                        # remove white spaces at the end of each line using rstrip()
                assert (len(line) == 2)
                edge_source.append(int(line[0]))
                edge_dest.append(int(line[1]))
                
        edge_index = np.asarray([edge_source, edge_dest])
        features = dict(sorted(graph_node_features_dict.items()))
        features = np.asarray([v for k,v in features.items()])          # numpy ndarray of shape (num_node, feature_dimension) holding node feature information
        labels = dict(sorted(graph_labels_dict.items()))
        labels = np.asarray([v for k,v in labels.items()])              # numpy ndarray of shape (num_node, ) holding node label information

    features = utils.preprocess_features(features)
    edge_relation = np.repeat(-1, edge_index.shape[1])
    edge_index_in_tuple = list(zip(edge_index[0], edge_index[1])) 
    
    if not embedding_mode:
        #g = graph(adj + sp.eye(adj.shape[0]))
        raise NotImplementedError
    else:
        if embedding_mode == 'ExperimentTwoAll':
            embedding_file_path = os.path.join('embedding_method_combinations_all', f'outf_nodes_relation_{dataset_name}all_embedding_methods.txt')
        elif embedding_mode == 'ExperimentTwoPairs':
            embedding_file_path = os.path.join('embedding_method_combinations_in_pairs', f'outf_nodes_relation_{dataset_name}_graph_{embedding_method_graph}_space_{embedding_method_space}.txt')
        else:                                                                   # default to here
            embedding_file_path = os.path.join('structural_neighborhood', f'outf_nodes_space_relation_{dataset_name}_{embedding_method}.txt')
        space_and_relation_type_to_idx_dict = {}
        with open(embedding_file_path) as embedding_file:
            for line in embedding_file:
                # space : one of graph, latent_space
                # relation_type : one of 0~3 
                if line.rstrip() == 'node1,node2	space	relation_type':   # skip the first line (first line contains column names )
                    continue
                line = re.split(r'[\t,]', line.rstrip())
                assert (len(line) == 4)
                #assert (int(line[0]) in G and int(line[1]) in G)                                    # two nodes must be in the graph already
                assert (int(line[0])<features.shape[0] and int(line[1])<features.shape[0])           # two nodes must be in the graph already
                if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict: 
                    space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(             # space, relation을 하나의 index화  --> subgraph_idx
                        space_and_relation_type_to_idx_dict)
                    
                
                if (int(line[0]), int(line[1])) in edge_index_in_tuple:
                    index = edge_index_in_tuple.index((int(line[0]), int(line[1])))
                    edge_relation[index] = space_and_relation_type_to_idx_dict[(line[2], int(line[3]))]
                else :
                    edge_index = np.column_stack((edge_index, np.asarray([int(line[0]), int(line[1])])))
                    edge_relation = np.append(edge_relation, space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] )

        # remove edges that are in -1 space&relation index 
        # (some edges in the original graph can be removed when constructing new graph if it is not within radius R. Refer to Geom GCN paper.)
        index = np.argwhere(edge_relation == -1)
        edge_relation = np.delete(edge_relation, index)
        temp_index_0 = np.delete(edge_index[0], index)
        temp_index_1 = np.delete(edge_index[1], index)
        edge_index = np.asarray([temp_index_0,temp_index_1])
        edge_index_in_tuple = list(zip(edge_index[0], edge_index[1])) 
                
        # Now, time to add self loop edges
        num_nodes = features.shape[0]
        space_and_relation_type_to_idx_dict['self_loop'] = len(space_and_relation_type_to_idx_dict)     # self loop을 마지막 type으로 정의 
        for node in range(num_nodes):
            if (int(node),int(node)) in edge_index_in_tuple:
                index = edge_index_in_tuple.index((int(node), int(node)))
                edge_relation[index] = space_and_relation_type_to_idx_dict['self_loop']
            else: 
                edge_index = np.column_stack((edge_index, np.asarray([int(node), int(node)])))
                edge_relation = np.append(edge_relation, space_and_relation_type_to_idx_dict['self_loop'] )
                   
    if splits_file_path:
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        assert (train_percentage is not None and val_percentage is not None)
        assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

        if dataset_name in {'cora', 'citeseer'}:
            disconnected_node_file_path = os.path.join('unconnected_nodes', f'{dataset_name}_unconnected_nodes.txt')
            with open(disconnected_node_file_path) as disconnected_node_file:   # contains only one node_id per row
                disconnected_node_file.readline()
                disconnected_nodes = []
                for line in disconnected_node_file:
                    line = line.rstrip()                        # disconnected node_id
                    disconnected_nodes.append(int(line))

            disconnected_nodes = np.array(disconnected_nodes)                                       # numpy ndarray containing disconnected node_ids
            connected_nodes = np.setdiff1d(np.arange(features.shape[0]), disconnected_nodes)        # numpy ndarray containing connected node_ids

            connected_labels = labels[connected_nodes]                                              # numpy ndarray containing labels of connected node_ids

            ## split with connected ones 
            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(connected_labels), connected_labels))     
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(connected_labels[train_and_val_index]), connected_labels[train_and_val_index]))
            
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[connected_nodes[train_index]] = 1
            val_mask = np.zeros_like(labels)
            val_mask[connected_nodes[val_index]] = 1
            test_mask = np.zeros_like(labels)
            test_mask[connected_nodes[test_index]] = 1
        else:
            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(labels), labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[train_index] = 1
            val_mask = np.zeros_like(labels)
            val_mask[val_index] = 1
            test_mask = np.zeros_like(labels)
            test_mask[test_index] = 1

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    
    degrees = np.repeat(-1, features.shape[0])
    unique, counts = np.unique(edge_index[1], return_counts=True)
    assert unique.shape == counts.shape
    degrees[unique] = counts
    degrees = torch.from_numpy(degrees)
    norm = torch.pow(degrees, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.unsqueeze(1)   #  1 / (degree^0.5), torch Tensor of shape (num_nodes)

    features = torch.from_numpy(features).float() # float
    labels = torch.from_numpy(labels).long()       # long
    train_mask = torch.from_numpy(train_mask).bool()   # bool
    val_mask = torch.from_numpy(val_mask).bool()   # bool
    test_mask = torch.from_numpy(test_mask).bool() # bool
    
    # print(features)     # torch.Tensor of shape (num_node, dimension)
    # print(labels)       # torch.Tensor of shape (num_node, )
    # print(train_mask)   # bool type numpy ndarray of shape (num_node, )
    # print(num_features) # dimension of features
    # print(num_labels)   # number of classes

    edge_index = torch.from_numpy(edge_index)           # convert to torch Tensor
    edge_relation = torch.from_numpy(edge_relation)     # convert to torch Tensor
    

    return edge_index, edge_relation, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, norm, space_and_relation_type_to_idx_dict
