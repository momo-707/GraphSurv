# encoding: utf-8
import pandas as pd
import numpy as np
import os
import torch
import csv
from DeepRandomCox.graphsurv import GraphSurv
from DeepRandomCox.CoxNN import DeepCox_LossFunc
from DeepRandomCox.utils import concordance_index
from support import split_censor, split_data, sort_data, mkdir, plot_curve, cal_pval
from dgl.data.utils import makedirs, save_info, load_info
from dgl.dataloading import GraphDataLoader
from copy import deepcopy
import pickle



def train(args, data):
    test_save_set = []
    argsDict = args.__dict__

    path = 'data/cancer/' + argsDict['dataset'] + '/cache/' + '_info.pkl'
    status, time, num_nodes = load_info(path)['status'], load_info(path)['time'], load_info(path)['num_nodes']
    censor, censor_graphs, no_censor, no_censor_graphs = split_censor(data, status, time)

    for seed in range(argsDict['seed']):
        test_save_set.append(["hidden layer : " + str(argsDict['hidden_layers']) + " learning rate : " + str(argsDict['learning_rate']) + " seed : " + str(argsDict['seed'])])
        for fold_num in range(args.fold):
            #初始化一些用以保存数据的
            train_Cindex_list, train_curve = [], []
            test_Cindex_list, test_curve = [], []
            epoch_list = []

            train_data, train_time, test_data, test_time, test_index = split_data(seed, censor, censor_graphs, no_censor, no_censor_graphs, fold_num, nfold=5)
            train_samples, test_samples = len(train_data), len(test_data)
            sorted_idx, train_data, train_time = sort_data(train_data, train_time)
            train_dataloader = GraphDataLoader(train_data, batch_size=argsDict['batch_size'], shuffle=False)
            test_dataloader = GraphDataLoader(test_data, batch_size=argsDict['batch_size'])
            model = GraphSurv(argsDict, num_nodes)
            loss_func = DeepCox_LossFunc()
            optimizer = torch.optim.Adam(model.parameters(), lr=argsDict['learning_rate'], weight_decay=argsDict['l2'])
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 1000], gamma=argsDict['learning_rate_decay'])

            if torch.cuda.is_available():
                model = model.cuda(argsDict['gpu'])

            for i in range(argsDict['epoch_num']):
                # train
                model.train()
                risks = torch.zeros([train_samples,1],dtype=torch.float)
                for id, graphs in enumerate(train_dataloader):
                    risk = model(graphs)
                    risks[id] = risk
                loss = loss_func(risks, train_time)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #eval
                if i % argsDict['skip_num'] == 0:
                    model.eval()
                    risks_save = risks.detach()
                    train_Cindex = concordance_index(train_time, -risks_save.cpu().numpy())
                    train_Cindex_list.append(train_Cindex)
                    test_Cindex, test_risks = evaluate(model, test_dataloader, test_time)
                    test_Cindex_list.append(test_Cindex)
                    print('epoch: %d      train Cindex: %.4f      test Cindex: %.4f' % (i, train_Cindex, test_Cindex))
                    epoch_list.append(i)
                    model.train()

            train_curve.append(epoch_list)
            train_curve.append(train_Cindex_list)
            test_curve.append(epoch_list)
            test_curve.append(test_Cindex_list)
            test_save_set.append(test_Cindex_list)
            # evaluation on the test set
            test_Cindex, test_risks = evaluate(model, test_dataloader, test_time)
            print('fold %i final result on the test set: %.4f' % (fold_num, test_Cindex))
            # save the model to disk
            directory = 'saved/%s' % (argsDict['dataset'])
            if not os.path.exists('saved/'):
                print('creating directory: saved/')
                os.mkdir('saved/')

            if not os.path.exists(directory):
                os.mkdir(directory)

            model_params = deepcopy(model.state_dict())
            torch.save(model_params, directory + '/model.pt')
            with open(directory + '/args.pkl', 'wb') as f:
                pickle.dump(argsDict, f)

        ### write csv
    mkdir("result")
    with open("result/" + argsDict['dataset'] + "_Cindex" + ".csv", "w",
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test_save_set)

    df = pd.DataFrame({'index': test_index,
                        'risk': test_risks.reshape(-1)})
    df.to_csv("result/" + argsDict['dataset'] + "_risk" + ".csv")


def evaluate(model, dataloader, test_time):
    model.eval()
    with torch.no_grad():
        risks = torch.zeros([test_time.shape[0],1],dtype=torch.float)
        for id, graphs in enumerate(dataloader):
            risk = model(graphs)
            risks[id] = risk
        risks_save = risks.detach()
        cindex = concordance_index(test_time, -risks_save.cpu().numpy())
    return cindex, risks_save

### 参数
### 5-cv
### time-Death graph-done
### censor-experiment
### return a loss function Cindex graph-done
### p-value
