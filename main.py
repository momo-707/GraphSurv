import os
import argparse
import data_processing

import train
import test
import pandas as pd


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    #device
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5, help='the index of gpu device')
    parser.add_argument('--mode', type=str, default='train', help='trian mode or test mode')

    # args train_mode
    #'''
    parser.add_argument('--dataset', type=str, default='lihc', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--fold', type=int, default=1, help='fold')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.9999, help='learning rate decay')
    parser.add_argument('--epoch_num', type=int, default=50, help='number of epochs')
    parser.add_argument('--skip_num', type=int, default=5, help='skip num')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model')
    parser.add_argument('--n_layer', type=int, default=2, help='layer numbers for GNN')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--l1', type=float, default=3.4e-5, help='L1 regularization')
    parser.add_argument('--l2', type=float, default=1.2e-4, help='L2 regularization')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--standardize', type=bool, default=True, help='standardize or not')
    parser.add_argument('--batchnorm', type=bool, default=False, help='batchnorm or not')
    parser.add_argument('--momentum', type=float, default=0.7, help='momentum rate')
    parser.add_argument('--hidden_layers', type=list, default=[500, 200, 25, 1], help='hidden layers')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model to disk')
    parser.add_argument('--is_plot', type=bool, default=True, help='plot or not')
    #'''

    #args test_mode
    '''
    parser.add_argument('--pretrained_model', type=str, default='lihc', help='the pretrained model')
    parser.add_argument('--standardize', type=bool, default=True, help='standardize or not')
    parser.add_argument('--dataset', type=str, default='example', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for calling the pretrained model')
    '''

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    data = data_processing.load_data(args)
    if args.mode == 'train': 
        train.train(args, data)
    else:
        test.test(args, data)


if __name__ == '__main__':
    main()
