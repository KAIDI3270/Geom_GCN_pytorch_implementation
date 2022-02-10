import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg


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
                                reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))     # aggregate
            subgraph.ndata.pop(f'Wh_{i}')
            subgraph.copy_to_parent()

        self.g.ndata.pop('h')
        
        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            if f'h_{i}' in self.g.node_attr_schemes():
                results_from_subgraph_list.append(self.g.ndata.pop(f'h_{i}'))
            else:
                results_from_subgraph_list.append(
                    th.zeros((feature.size(0), self.out_feats), dtype=th.float32, device=feature.device))

        if self.merge == 'cat':                                             # for layer 1
            h_new = th.cat(results_from_subgraph_list, dim=-1)
        else:                                                               # for layer 2
            h_new = th.mean(th.stack(results_from_subgraph_list, dim=-1), dim=-1)
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
            return th.cat(all_attention_head_outputs, dim=1)                                # 각 head의 결과를 concat함
        else:
            return th.mean(th.stack(all_attention_head_outputs), dim=0)


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
