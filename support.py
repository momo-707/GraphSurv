import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from lifelines.statistics import logrank_test
import torch
from lifelines import KaplanMeierFitter

def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'Folder create successfully !')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' Folader is exist')
        return False

def cal_pval(time, pred):
    event = np.zeros_like(time)
    event[time > 0] = 1
    pred_median = np.median(pred)
    risk_group = np.zeros_like(pred)
    risk_group[pred > pred_median] = 1

    group_lowrisk_time = time[risk_group==0].copy()
    glt = np.array(group_lowrisk_time)
    group_highrisk_time = time[risk_group==1].copy()
    ght = np.array(group_highrisk_time)
    group_lowrisk_event = event[risk_group==0].copy()
    gle = np.array(group_lowrisk_event)
    group_highrisk_event = event[risk_group==1].copy()
    ghe = np.array(group_highrisk_event)


    km = KaplanMeierFitter()
    km.fit(glt, gle)
    km.plot()
    km.survival_function_.plot()
    km.fit(ght, ghe)
    km.plot()

    print(glt.shape)

    results = logrank_test(group_lowrisk_time, group_highrisk_time, event_observed_A=group_lowrisk_event , event_observed_B=group_highrisk_event)
    #results.print_summary()
    return results.p_value

def get_risk(pred):
    pred_median = np.median(pred)
    risk_group = np.zeros_like(pred)
    risk_group[pred > pred_median] = 1
    return risk_group.astype(np.int)

def sort_data(X,Y):
    #将时间从小到大排序，时间越短的死亡风险越高
    T = - np.abs(np.squeeze(np.array(Y)))
    sorted_idx = np.argsort(T) #返回索引值数组
    X_final = [X[i] for i in sorted_idx]
    return sorted_idx, X_final, Y[sorted_idx]

def plot_curve(curve_data, title="train epoch-Cindex curve", x_label="epoch", y_label="Cindex"):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(curve_data[0], curve_data[1], color='black', markerfacecolor='black', marker='o', markersize=1)
    plt.show()
    ###print sorted survival time:

def split_censor(graphs, status, time):
    #划分censor数据和非cencor数据
    censor_index = np.where(status == 0)
    censor_index = censor_index[0]
    censor = time[censor_index]
    print("删失数据的个数为")
    print(len(censor))
    censor_graphs = graphs.reset_index(censor_index)

    no_censor_index = np.where(status > 0)
    no_censor_index = no_censor_index[0]
    no_censor = time[no_censor_index]
    print("非删失数据的个数为")
    print(len(no_censor))
    no_censor_graphs = graphs.reset_index(no_censor_index)
    return censor, censor_graphs, no_censor, no_censor_graphs

def split_data(seed, censor, censor_graphs, no_censor, no_censor_graphs, fold_num, nfold = 5):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)

    num = 0
    for train_index, test_index in kf.split(censor_graphs):
        train_censor = [censor_graphs[i] for i in train_index]
        train_Y1 = censor[train_index]
        test_censor = [censor_graphs[i] for i in test_index]
        test_Y1 = censor[test_index]
        if num == fold_num:
            test_index_all = test_index.tolist()
            break
        num += 1

    num = 0
    for train_index, test_index in kf.split(no_censor_graphs):
        train_no_censor = [no_censor_graphs[i] for i in train_index]
        train_Y2 = no_censor[train_index]
        test_no_censor = [no_censor_graphs[i] for i in test_index]
        test_Y2 = no_censor[test_index]
        if num == fold_num:
            test_index_all.extend(test_index.tolist())
            break
        num += 1

    train_censor.extend(train_no_censor)
    train_Y = np.vstack((train_Y1, train_Y2))
    test_censor.extend(test_no_censor)
    test_Y = np.vstack((test_Y1, test_Y2))

    return train_censor, train_Y, test_censor, test_Y, test_index_all

def import_omic_data(dataset_original):
    original_data = dataset_original.where(dataset_original.notnull(), 0)
    time = original_data.iloc[0,:]
    time = time.drop('Platform', 0).tolist()[1:]
    time = np.array(time)
    status = original_data.iloc[1, :]
    status = status.drop('Platform', 0).tolist()[1:]
    status = np.array(status)
    gene_exp = original_data.loc[original_data["Platform"] == "geneExp"]
    gene_exp = gene_exp.drop_duplicates(['GeneSymbol'], keep='first')
    gene_exp = gene_exp.drop('Platform', 1)
    copy_num = original_data.loc[original_data["Platform"] == "copyNumber"]
    copy_num = copy_num.drop_duplicates(['GeneSymbol'], keep='first')
    copy_num = copy_num.drop('Platform', 1)
    meth = original_data.loc[original_data["Platform"] == "methylation"]
    meth = meth.drop_duplicates(['GeneSymbol'], keep='first')
    meth = meth.drop('Platform', 1)
    df = pd.merge(gene_exp, copy_num, on=['GeneSymbol'], how='inner')
    omic_data = pd.merge(df, meth, on=['GeneSymbol'], how='inner')
    return omic_data, time, status


def import_edge(omic_data, convert):
    train_data = omic_data['GeneSymbol']
    print("train_data.size", train_data.shape)
    convert.columns = ['gene_x', 'gene_y']

    c = {'gene_name': train_data, 'node_idx': list(np.arange(len(train_data)))}
    expressions_id = pd.DataFrame(c)

    tmp1 = expressions_id.rename(columns={'gene_name': 'gene_x'})
    tmp_nodex = pd.merge(convert, tmp1, on='gene_x').drop_duplicates().reset_index(drop=True)

    tmp2 = expressions_id.rename(columns={'gene_name': 'gene_y'})
    adj_df = pd.merge(tmp_nodex, tmp2, on='gene_y').drop_duplicates().reset_index(drop=True)

    adj_df = adj_df[['node_idx_x', 'node_idx_y']]
    return adj_df