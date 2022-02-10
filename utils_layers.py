import dgl.function as fn
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg



class GeomGCN_layer(nn.Module):
    def __init__(self, in_feats, out_feats, num_divisions, num_heads, activation, dropout_prob, merge_method, device):
        super(GeomGCN_layer, self).__init__()
        
        # save parameter
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_divisions = num_divisions
        self.num_heads = num_heads
        self.activation = activation
        self.merge_method = merge_method
        self.norm = None
        self.device = device
        
        # nn        
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.linear_per_division = nn.ModuleList()
        for i in range(self.num_divisions * self.num_heads):
            self.linear_per_division.append(nn.Linear(in_feats, out_feats, bias=False))
        for i in range(self.num_divisions * self.num_heads):
            nn.init.xavier_uniform_(self.linear_per_division[i].weight)
        
        # save graph info
        self.edge_index = None
        self.edge_relation = None
        self.relation_dict_raw = None               # raw relation dictionary from dataset
        self.relation_dict = None                   # processed dictionary. holds relation indexes related to each relation type(self_loop, original, latent)
        self.sparse_adj_per_relation = None
        self.num_nodes = 0
        
        
        
    def set_edges(self, edge_index, edge_relation, relation_dict, num_nodes, norm):
        self.edge_index = edge_index ;    self.edge_relation = edge_relation;    self.relation_dict_raw = relation_dict;     self.num_nodes = num_nodes;       self.norm = norm
        
        graph_indexes = [] ;    latent_space_indexes = [];    self_loop_indexes = []
        
        relation_to_space_relation = {v: k for k, v in self.relation_dict_raw.items()}  # invert key and value of relation_dict
        num_relations = 0
        for k,v in relation_to_space_relation.items():
            num_relations += 1
            if v == 'self_loop':
                self_loop_indexes.append(k)
            elif v[0] == 'graph':
                latent_space_indexes.append(k)
            elif v[0] == 'latent_space':
                graph_indexes.append(k)
            else:
                raise NotImplementedError
        assert num_relations == self.num_divisions
        self.relation_dict = {'self_loop':self_loop_indexes, 'graph':graph_indexes, 'latent':latent_space_indexes}
        
        
        transposed_edge_index = torch.t(self.edge_index)
        edge_index_per_relation = []
        for edge_type in range(self.num_divisions):
            edge_index_for_specific_relation_index = ((self.edge_relation == edge_type).nonzero(as_tuple=True)[0])
            edges_for_specific_relation_index = transposed_edge_index[edge_index_for_specific_relation_index]
            edge_index_per_relation.append(torch.t(edges_for_specific_relation_index))
        
        
        sparse_adj_per_relation = []
        for edge_index in edge_index_per_relation:
            sparse_adj = torch.sparse_coo_tensor(indices = edge_index, values = torch.ones(edge_index.size(1)).float(), size = (num_nodes, num_nodes), device = self.device )
            sparse_adj_per_relation.append(sparse_adj)
        self.sparse_adj_per_relation = sparse_adj_per_relation
        
    def forward(self, features):
        sparse_adj_per_relation = self.sparse_adj_per_relation
        features = self.in_feats_dropout(features)   
        features = (features * self.norm).float()     # normalize by node degree. See the equation about 'e ^ v,l+1 _ (i,r)' at page 7  of https://arxiv.org/pdf/2002.05287.pdf
        attention_head_results = []
        for head_index in range(self.num_heads):
            result = []
            for division_index in range(self.num_divisions):
                result.append( torch.sparse.mm(sparse_adj_per_relation[division_index], self.linear_per_division[head_index * self.num_divisions + division_index](features)) )
            if self.merge_method == 'cat':
                aggregated_result = torch.cat(result, 1)
            else:
                aggregated_result = torch.mean(torch.stack(result, dim=-1), dim=-1)
            head_result = self.activation(aggregated_result * self.norm )
            attention_head_results.append(head_result)
            
        if self.merge_method == 'cat':
            result_final = torch.cat(attention_head_results, dim=1)                                # 각 head의 결과를 concat함
        else:
            result_final = torch.mean(torch.stack(attention_head_results), dim=0)

        return result_final 
       

class GeomGCN_model(nn.Module):                                                                # 2layer geom gcn
    def __init__(self, in_feats, hidden_feats, out_feats, num_divisions, num_heads_first, num_heads_two, 
                 dropout_rate, merge_method_one, merge_method_two, device):
        super(GeomGCN_model, self).__init__()  
        self.geomgcn1 = GeomGCN_layer(in_feats = in_feats,                      # in_feats
                                out_feats = hidden_feats,                       # out_feats
                                num_divisions = num_divisions,                  # num_divisions
                                num_heads = num_heads_first,
                                activation = F.relu,                            # activation
                                dropout_prob = dropout_rate,
                                merge_method = merge_method_one,
                                device = device)

        if merge_method_one == 'cat':       # default
            layer_one_ggcn_merge_multiplier = num_divisions
            layer_one_channel_merge_multiplier = num_heads_first
        else:
            layer_one_ggcn_merge_multiplier = 1
            layer_one_channel_merge_multiplier = 1
        # in_feats, out_feats, num_divisions, num_heads, activation, dropout_prob, merge_method
        self.geomgcn2 = GeomGCN_layer(in_feats = hidden_feats * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,          # in_feats
                                out_feats = out_feats,                                                                                         # out_feats
                                num_divisions = num_divisions,                                                                                 # num_divisions
                                num_heads = num_heads_two,                                                                                     # num_heads
                                activation = lambda x: x,                                                                                      # activation
                                dropout_prob = dropout_rate,                                                                                   # dropout_prob
                                merge_method = merge_method_two,
                                device = device)  
 
    def set_edges(self, edge_index, edge_relation, relation_dict, num_nodes, norm):
        self.geomgcn1.set_edges(edge_index, edge_relation, relation_dict, num_nodes, norm)
        self.geomgcn2.set_edges(edge_index, edge_relation, relation_dict, num_nodes, norm)

    def forward(self, features):
        x = self.geomgcn1(features)
        x = self.geomgcn2(x)
        return x

########################################################################################################################################################################
# Below is the original code
########################################################################################################################################################################

class GeomGCNSingleChannel(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, dropout_prob, merge):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.linear_for_each_division = nn.ModuleList()
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False))
        for i in range(self.num_divisions):
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight)
            
        self.activation = activation
        self.g = g
        self.subgraph_edge_list_of_list = self.get_subgraphs(self.g)
        self.merge = merge          # default to cat for layer 1, mean for layer 2
        self.out_feats = out_feats

    def get_subgraphs(self, g):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')     # u,v : node, eid : edge_id
        for i in range(g.number_of_edges()):
            subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])

        return subgraph_edge_list

    def forward(self, feature):
        in_feats_dropout = self.in_feats_dropout(feature)       
        self.g.ndata['h'] = in_feats_dropout                    # result of dropout

        for i in range(self.num_divisions):
            subgraph = self.g.edge_subgraph(self.subgraph_edge_list_of_list[i])
            subgraph.copy_from_parent()
            subgraph.ndata[f'Wh_{i}'] = self.linear_for_each_division[i](subgraph.ndata['h']) * subgraph.ndata['norm']      # nn.linear(dropout(in_feature)) / degree^0.5
            subgraph.update_all(message_func=fn.copy_u(u=f'Wh_{i}', out=f'm_{i}'),  # create message
                                reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))     # aggregate (not mean aggregation, but sum aggregation)
            subgraph.ndata.pop(f'Wh_{i}')
            subgraph.copy_to_parent()

        self.g.ndata.pop('h')
        
        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            if f'h_{i}' in self.g.node_attr_schemes():
                results_from_subgraph_list.append(self.g.ndata.pop(f'h_{i}'))
            else:
                results_from_subgraph_list.append(
                    torch.zeros((feature.size(0), self.out_feats), dtype=torch.float32, device=feature.device))

        if self.merge == 'cat':                                             # for layer 1
            h_new = torch.cat(results_from_subgraph_list, dim=-1)
        else:                                                               # for layer 2
            h_new = torch.mean(torch.stack(results_from_subgraph_list, dim=-1), dim=-1)
        h_new = h_new * self.g.ndata['norm']
        h_new = self.activation(h_new)
        return h_new


class GeomGCN(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge, channel_merge):
        super(GeomGCN, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(g, in_feats, out_feats, num_divisions, activation, dropout_prob, ggcn_merge))
        self.channel_merge = channel_merge              # default is concat('cat') for layer one, mean for layer two
        self.g = g

    def forward(self, feature):
        all_attention_head_outputs = [head(feature) for head in self.attention_heads]       # GeomGCNSingleChannel의 forward 결과들
        if self.channel_merge == 'cat':                     # default option
            return torch.cat(all_attention_head_outputs, dim=1)                                # 각 head의 결과를 concat함
        else:
            return torch.mean(torch.stack(all_attention_head_outputs), dim=0)


class GeomGCNNet(nn.Module):                                                                # 2layer geom gcn
    def __init__(self, g, num_input_features, num_output_classes, num_hidden, num_divisions, num_heads_layer_one,
                 num_heads_layer_two,
                 dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, layer_two_ggcn_merge,
                 layer_two_channel_merge):
        super(GeomGCNNet, self).__init__()  
        self.geomgcn1 = GeomGCN(g,                              # dgl graph
                                num_input_features,             # in_feats
                                num_hidden,                     # out_feats
                                num_divisions,                  # num_divisions
                                F.relu,                         # activation
                                num_heads_layer_one,            # num_heads
                                dropout_rate,                   # dropout_prob
                                layer_one_ggcn_merge,           # ggcn_merge            # default to concat
                                layer_one_channel_merge)        # channel_merge         # default to concat

        if layer_one_ggcn_merge == 'cat':       # default
            layer_one_ggcn_merge_multiplier = num_divisions
        else:
            layer_one_ggcn_merge_multiplier = 1

        if layer_one_channel_merge == 'cat':       # default
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1

        self.geomgcn2 = GeomGCN(g,                                                                                          # dgl graph
                                num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,          # in_feats
                                num_output_classes,                                                                         # out_feats
                                num_divisions,                                                                              # num_divisions
                                lambda x: x,                                                                                # activation
                                num_heads_layer_two,                                                                        # num_heads
                                dropout_rate,                                                                               # dropout_prob
                                layer_two_ggcn_merge,                                                                       # ggcn_merge            # default to mean
                                layer_two_channel_merge)                                                                    # channel_merge         # default to mean
        self.g = g  

    def forward(self, features):
        x = self.geomgcn1(features)
        x = self.geomgcn2(x)
        return x
