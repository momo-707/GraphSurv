import pandas as pd
import numpy as np
import torch
from .utils import _prepare_surv_data
from .utils import concordance_index
from DeepRandomCox.CoxNN import Coxnn, DeepCox_LossFunc
from DeepRandomCox.GNN_layer import GNN
import torch.nn as nn

class GraphSurv(nn.Module):
    def __init__(self, args, input_nodes):
        super(GraphSurv, self).__init__()
        torch.manual_seed(1234)
        torch.set_printoptions(profile="full")
        self.args = args
        self.GNN = GNN(args['gnn'], args['n_layer'], feature_len=3, dim=1)
        self.Coxnn = Coxnn(args, input_nodes)

    def forward(self, x):
        GNN_out = self.GNN(x)
        GNN_out = GNN_out.T
        risk = self.Coxnn(GNN_out)
        return risk