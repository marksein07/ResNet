#!/usr/bin/env python3

import argparse
import numpy as np
#from environment import Environment

#seed = 7704
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data

import torchvision
from torchvision import transforms

import numpy as np

from tensorboardX import SummaryWriter

import pickle

from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import shutil

from utils.utils import *
from model.ResNet import ResNet

print(torch.__version__)
# torch.manual_seed(1)



def parse():
    parser = argparse.ArgumentParser(description="DensetNet - 2020 by marksein07")
    #parser.add_argument('--optimizer')
    parser.add_argument('--memory_efficient', type=float, default=0.9, help='memory efficient on DenseNet')
    parser.add_argument('--bias', type=float, default=0.9, help='Apply bias on DenseNet')
    parser.add_argument('--num_init_features', type=int, default=16, help='init features number')
    parser.add_argument('--compression_rate', type=float, default=0.5, help='DenseNet compression rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--bottleneck_size', type=int, default=4, help='DenseNet bottleneck size')
    parser.add_argument('--layer_num', type=int, default=100, help='DenseNet layers number')
    parser.add_argument('--growth_rate', type=int, default=12, help='DenseNet growth rate')
    
    parser.add_argument('--batch_size', type=int,default=64, help='traing and testing batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer L2 penalty')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--cuda', type=str, default='0', help='GPU Index for training')
    parser.add_argument('--log', type=str, default='../result/log', help='tensorboard log directory')
    parser.add_argument('--result', type=str, default='../result', help='experiment result')
    parser.add_argument('--preceed', type=bool, default=False, help='whether load a pretrain model')
    parser.add_argument('--training_epoch', type=int, default=300, help='total training epoch')
    
    optimizer_group = parser.add_mutually_exclusive_group()
    optimizer_group.add_argument('--SGD', action='store_true', default=True, help='Choose SGD as optimizer')
    optimizer_group.add_argument('--Adam', action='store_true', default=False, help='Choose Adam as optimizer')
    optimizer_group.add_argument('--Adagrad', action='store_true', default=False, help='Choose Adagrad as optimizer')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    batch_size = args.batch_size
    preceed = args.preceed
    log = args.log
    device = torch.device('cuda:'+args.cuda)
    LR = args.learning_rate
    result=args.result
    if not os.path.isdir(result) :
        os.mkdir( result, 0o755 );
    if os.path.isdir(log) :
        shutil.rmtree(log)
    writer = SummaryWriter(log)

    trainloader, testloader = dataloader(batch_size)
    
    model = ResNet()
    
    if len(args.cuda) == 1 :
        model.to(device)      # Moves all model parameters and buffers to the GPU.
    elif len(args.cuda) > 1 :
        model = torch.nn.DataParallel(model,device_ids=[0,1]).to(device)
        
    if preceed :
        model.load_state_dict(torch.load("DenseNetL=100,k=12"))

    if args.Adam :
        opt = torch.optim.Adam
    elif args.Adagrad :
        opt = torch.optim.Adagrad
    elif args.SGD :
        opt = torch.optim.SGD

    optimizer = opt( model.parameters(),
                    lr           = args.learning_rate,
                    weight_decay = args.weight_decay,
                    momentum     = args.momentum )
    
    criterion = nn.CrossEntropyLoss()

    epoch = 0
    
    training_error_rate, testing_error_rate, \
    training_loss, testing_loss = train(model, optimizer, criterion, trainloader, testloader, 
                                        device, channel_normalization, writer, opt, LR, 
                                        max_epoch = args.training_epoch, )
    loss = dict({'Training':training_loss, 'Validation':testing_loss})
    error_rate = dict({'Training':training_error_rate, 'Validation':testing_error_rate})
    
    with open('../result/loss.pickle', 'wb') as file :
        pickle.dump(loss, file)
    with open('../result/error_rate.pickle', 'wb') as file :
        pickle.dump(error_rate, file)
    
    plot(loss, filename='../result/loss', ylabel='Loss', title='Loss')
    plot(error_rate, filename='../result/error_rate', ylabel='Error rate', 
         title='Error rate', log_scale=False)
    
if __name__ == '__main__':
    args = parse()
    run(args)
