import os
import torch
import pickle
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DeepRandomCox.graphsurv import GraphSurv
from DeepRandomCox.CoxNN import DeepCox_LossFunc
from copy import deepcopy
from dgl.dataloading import GraphDataLoader
import pandas as pd
from dgl.data.utils import save_info, load_info


def test(args, data):
    dataloader = GraphDataLoader(data, batch_size=args.batch_size)
    #加载模型超参数
    path = 'saved/' + args.pretrained_model + '/'
    print('loading hyperparameters of pretrained model from ' + path + 'args.pkl')
    with open(path + 'args.pkl', 'rb') as f:
        hparams = pickle.load(f)

    print('loading pretrained model from ' + path + 'model.pt')
    model = GraphSurv(hparams, data.num_nodes)

    #加载模型
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path + 'model.pt'))
        model = model.cuda(args.gpu)
    else:
        model.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        num_samples = data.__len__()
        risks = torch.zeros([num_samples,1],dtype=torch.float)
        for id, graphs in enumerate(dataloader):
            risk = model(graphs)
            risks[id] = risk
        risks_save = risks.detach()
    df = pd.DataFrame({'risk': risks_save.reshape(-1)})
    directory = 'result/%s' % (args.dataset)
    if not os.path.exists('result/'):
        print('creating directory: result/')
        os.mkdir('result/')
    if not os.path.exists(directory):
        os.mkdir(directory)

    df.to_csv(directory + "/risk" + ".csv")