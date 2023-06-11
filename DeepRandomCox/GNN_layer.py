import math
import torch
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
from torch.nn.functional import one_hot, normalize, logsigmoid
import torch.nn as nn
import numpy as np
import pandas as pd

class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, feature_len, dim):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = dim
        self.gnn_layers = ModuleList([])
        if gnn in ['gcn', 'gat', 'sage', 'tag']:
            for i in range(n_layer):
                if gnn == 'gcn':
                    self.gnn_layers.append(GraphConv(in_feats=feature_len if i == 0 else dim,
                                                     out_feats=dim,
                                                     activation=None if i == n_layer - 1 else torch.relu))
                elif gnn == 'gat':
                    num_heads = 16  # make sure that dim is dividable by num_heads
                    self.gnn_layers.append(GATConv(in_feats=feature_len if i == 0 else dim,
                                                   out_feats=dim // num_heads,
                                                   activation=None if i == n_layer - 1 else torch.relu,
                                                   num_heads=num_heads))
                elif gnn == 'sage':
                    agg = 'pool'
                    self.gnn_layers.append(SAGEConv(in_feats=feature_len if i == 0 else dim,
                                                    out_feats=dim,
                                                    activation=None if i == n_layer - 1 else torch.relu,
                                                    aggregator_type=agg))
                elif gnn == 'tag':
                    hops = 2
                    self.gnn_layers.append(TAGConv(in_feats=feature_len if i == 0 else dim,
                                                   out_feats=dim,
                                                   activation=None if i == n_layer - 1 else torch.relu,
                                                   k=hops))
        elif gnn == 'sgc':
            self.gnn_layers.append(SGConv(in_feats=feature_len, out_feats=dim, k=n_layer))
        else:
            raise ValueError('unknown GNN model')
        self.pooling_layer = SumPooling()
        self.factor = None

    def forward(self, graph):
        h = graph.ndata['feature']
        for layer in self.gnn_layers:
            h = layer(graph, h)
            if self.gnn == 'gat':
                h = torch.reshape(h, [h.size()[0], -1])
        graph_embedding = h
        return graph_embedding