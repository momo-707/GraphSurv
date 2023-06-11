from support import import_omic_data, import_edge
import numpy as np
import os
import dgl
import torch
import pandas as pd
from dgl.data.utils import makedirs, save_info, load_info


#data: 输入一个数据集，Dataset返回gene图和label(survival time)
class MultiOmicsDataset(dgl.data.DGLDataset):
    def __init__(self, args, gene_exp=None, copy_num=None, meth=None, data_time=None, status=None, edges=None):
        self.args = args
        self.dataset = args.dataset
        self.gene_exp = gene_exp
        self.copy_num = copy_num
        self.meth = meth
        self.time = data_time
        self.status = status
        self.edges = edges
        self.num_nodes = 0
        self.graphs = []
        self.path = 'data/cancer/' + args.dataset + '/cache/'
        super().__init__(name='MultiOmics_')

    def to_gpu(self):
        if torch.cuda.is_available():
            self.graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.graphs]

    def save(self):
        print('saving ' + self.dataset + ' graph to ' + self.path + 'graphs.bin')
        dgl.save_graphs(self.path + 'graphs.bin', self.graphs)
        save_info(self.path + '_info.pkl', {'status': self.status, 'time':self.time, 'num_nodes':self.num_nodes})

    def load(self):
        print('loading ' + self.dataset + ' graphs from ' + self.path + 'graphs.bin')
        # graphs loaded from disk will have a default empty label set: [graphs, labels], so we only take the first item
        self.graphs = dgl.load_graphs(self.path + 'graphs.bin')[0]
        self.to_gpu()

    def process(self):
        print('transforming ' + self.dataset + ' data to DGL graphs')
        self.num_nodes = self.gene_exp.shape[1] #samples * genes
        for id, gene in enumerate(self.gene_exp):
            graph = dgl.graph((self.edges['node_idx_x'], self.edges['node_idx_y']), num_nodes=self.num_nodes)
            feature = np.vstack((gene, self.copy_num[id]))
            feature = np.vstack((feature, self.meth[id]))
            graph.ndata['feature'] = torch.tensor(feature, dtype=torch.float32).t()
            graph = dgl.add_self_loop(graph)
            self.graphs.append(graph)
        self.to_gpu()

    def has_cache(self):
        return os.path.exists(self.path + 'graphs.bin')

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

    def reset_index(self, index):
        return [self.graphs[i] for i in index]

    def graph_nodes(self):
        return self.num_nodes



def standardize(x):
    for j in range(x.shape[1]):
        mean = np.mean(x[:, j])
        if np.std(x[:, j]) != 0:
            x[:, j] = (x[:, j] - mean) / np.std(x[:, j])
    return x

def read_data(args):
    cancer_path = 'data/cancer/' + args.dataset + '/' + args.dataset + '.csv'
    graph_path = 'data/relationship_dataset/gene_relationship.csv'
    print('preprocessing %s data from %s' % (args.dataset, cancer_path))
    #处理features
    dataset_original = pd.read_csv(cancer_path)
    gene_relationship = pd.read_csv(graph_path, sep=',', header=None)
    omic_data, time, status = import_omic_data(dataset_original)
    features = omic_data[0:]
    features = features.drop('GeneSymbol', axis=1)
    features = features.values
    features = np.transpose(features)
    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_time = time.reshape(-1, 1)
    samples_num = features.shape[0] // 3
    if data_time.shape[0] == status.shape[0] == samples_num: print("维度已对齐, 总共有%d个样本" % (samples_num))
    gene_exp, copy_num, meth = features[0:samples_num], features[samples_num:2*samples_num], features[2*samples_num:]
    gene_exp = np.log2(gene_exp + 0.0001)
    if args.standardize:
        gene_exp, copy_num, meth = standardize(gene_exp), standardize(copy_num), standardize(meth)


    #处理edges
    edges = import_edge(omic_data, gene_relationship)
    return gene_exp, copy_num, meth, data_time, status, edges


def load_data(args):

    gene_exp, copy_num, meth, data_time, status, edges = read_data(args)
    cancer_dataset = MultiOmicsDataset(args, gene_exp, copy_num, meth, data_time, status, edges)

    return cancer_dataset
