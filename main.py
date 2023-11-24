# python 3
# -*- coding:utf-8 -*-
"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""
"""
Set <--data_dir> to the directory of the CWRU dataset firstly! (for example: './Datasets_dir/CWRU')
"""

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils import train_utils
import random
import numpy as np
import torch
import time

# Random setting
seed = 999
np.random.seed(seed)  # seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

mybool = lambda x: x.lower() in ['yes', 'true', 't', 'y', '1']


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # datasets parameters
    parser.add_argument('--data_dir', type=str, default='./Datasets_dir/CWRU',
                        help='the directory of the dataset')
    parser.add_argument('--data_name', type=str, default='CWRU', choices=['CWRU', ],
                        help='the name of the dataset')
    parser.add_argument('--data_type', type=str, default='time', choices=['time'],
                        help='the data_type of the dataset')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std'], default='mean-std',
                        help='data normalization methods')
    parser.add_argument('--data_signalsize', type=int, default=1024, help='the name of the data')
    parser.add_argument('--SNR', type=float, default=1000, help='activate when SNR in (-100,100) else set to None')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of dataloader workers')
    parser.add_argument('--test_size', type=float, default=0.3, help='for few-shot analysis')

    # models parameters
    parser.add_argument('--model_name', type=str, default='TFN_STTF',
                        choices=["Backbone_CNN", "Random_CNN", "TFN_STTF", "TFN_Chirplet", "TFN_Morlet"],
                        help='the model to be trained')
    parser.add_argument('--kernel_size', type=int, default=11, help='the kernel size of traditional conv layer')
    parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoint',
                        help='the directory to save the models and the results')

    # func-models parameters
    parser.add_argument('--mid_channel', type=int, default=32, help='the channel number of preprocessing layer')
    parser.add_argument('--clamp_flag', type=mybool, default="True", help='flag to limit the superparams of TFconv layer')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam', 'RMSprop'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=0, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='learning rate scheduler parameter for step and exp')  # 0.99
    parser.add_argument('--steps', type=str, default='1', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--save_model', type=mybool, default="False", help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=50, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=5, help='the interval of log training information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print("current dir: %s" % os.path.curdir)

    args = parse_args()

    # Process the args.SNR
    if args.SNR <= -1e2 or args.SNR >= 1e2:
        args.SNR = None


    # Prepare the saving path for the models
    sub_dir = args.model_name + '-' + args.data_name + '-' + args.data_type + '-' + datetime.strftime(datetime.now(),
                                                                                                      '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir).replace('\\', '/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("<args> {}: {}".format(k, v))

    # initialize the trainer
    trainer = train_utils(args, save_dir)
    trainer.setup()

    # train the model
    time_start_train = time.time()
    trainer.train()
    logging.info("<training time>: {:.3f}".format(time.time() - time_start_train))

    # plot the results
    trainer.plot_save()
