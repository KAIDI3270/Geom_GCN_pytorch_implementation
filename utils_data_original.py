import os
import re

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import graph
import dgl
from sklearn.model_selection import ShuffleSplit

import utils

cuda_num =3

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
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                f'out1_node_feature_label.txt')

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

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()                        # each line has two node_ids with tab separations
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')                        # remove white spaces at the end of each line using rstrip()
                assert (len(line) == 2)
                if int(line[0]) not in G:                               # if node is not in graph object, add it
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))                  # add edge to graph object

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))                                                         # extract adj      --> sparse matrix?
        features = np.array(                                                                                    # extract features --> numpy ndarray
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(                                                                                      # extract labels   --> numpy ndarray
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = utils.preprocess_features(features)
    if not embedding_mode:
        g = graph(adj + sp.eye(adj.shape[0]))
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
                assert (int(line[0]) in G and int(line[1]) in G)                                    # two nodes must be in the graph already
                if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict: 
                    space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(             # space, relation을 하나의 index화  --> subgraph_idx
                        space_and_relation_type_to_idx_dict)
                if G.has_edge(int(line[0]), int(line[1])):                                          # edge에 subgraph_idx 정보 추가하기 위한 삭제
                    G.remove_edge(int(line[0]), int(line[1]))
                G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                    (line[2], int(line[3]))])

        space_and_relation_type_to_idx_dict['self_loop'] = len(space_and_relation_type_to_idx_dict)     # self loop을 마지막 type으로 정의 
        for node in sorted(G.nodes()):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])       # self loop edge도 결국 subgraph_idx 부여하여 추가
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))                                                 # rebuild adj matrix (sparse matrix?)
        g = dgl.from_scipy(adj)

        for u, v, feature in G.edges(data='subgraph_idx'):                                              # subgraph_idx를 tensor type으로 수정하는 코드
            g.edges[g.edge_ids(u, v)].data['subgraph_idx'] = th.tensor([feature])


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

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    
    

    # print(features)     # torch.Tensor of shape (num_node, dimension)
    # print(labels)       # torch.Tensor of shape (num_node, )
    # print(train_mask)   # bool type numpy ndarray of shape (num_node, )
    # print(num_features) # dimension of features
    # print(num_labels)   # number of classes
    
    
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)  #  1 / (degree^0.5), torch Tensor of shape (num_nodes)
    # norm = norm.cuda(cuda_num)

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
