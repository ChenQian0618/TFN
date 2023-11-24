#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import Models
import Datasets
import matplotlib.pyplot as plt
from utils.mysummary import summary
import numpy as np
import scipy.io as io
import random
import matplotlib as mplb
import seaborn as sns

# set random seed
seed = 999
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class train_utils(object):
    def __init__(self, args, save_dir: str):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, models, loss and optimizer
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = 1
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        dataset = getattr(Datasets, args.data_name)

        self.datasets = {}
        subargs = {k: getattr(args, k) for k in ['data_dir', 'data_type', 'normlizetype', 'test_size']}
        (self.datasets['train'], self.datasets['val']), self.label_name = dataset(subargs) \
            .data_preprare(signal_size=args.data_signalsize, SNR=self.args.SNR)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        logging.info(f"dataset_train:{len(self.datasets['train']):d}, dataset_train:{len(self.datasets['val']):d}")

        # Define the models
        self.model = getattr(Models, args.model_name) \
            (in_channels=dataset.inputchannel, out_channels=dataset.num_classes, kernel_size=args.kernel_size,
             clamp_flag=args.clamp_flag, mid_channel=args.mid_channel)
        self.model.to(self.device)

        # summary the model and record model info
        try:
            info = summary(self.model, self.datasets['train'][0][0].shape, batch_size=-1, device="cuda")
            for item in info.split('\n'):
                logging.info(item)
        except:
            print('summary does not work!')

        self.criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                           weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # confusion matrix initialization
        self.c_matrix = {phase: np.zeros([dataset.num_classes, dataset.num_classes]) for phase in ['train', 'val']}

    def train(self):
        """
        Training process
        """

        args = self.args

        best_acc = 0.0
        step_start = time.time()

        self.Records = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                        "best_epoch": 0}  # epoch-wise records
        self.MinorRecords = {"train_loss": [], "train_acc": []}  # batch-wise records

        # Train the models via epochs
        for epoch in range(args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # Record the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                batch_acc = 0
                batch_loss = 0.0
                batch_count = 0

                # Set models to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                # batch-wise training or testing
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # confusion matrix calculation
                        if epoch == args.max_epoch - 1:
                            for i, j in zip(labels.detach().cpu().numpy(), pred.detach().cpu().numpy()):
                                self.c_matrix[phase][i][j] += 1

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()  # update the model parameters

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            # Print the training information
                            if (batch_idx + 1) % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info(
                                    '<tempinfo> Epoch: {:3d} [{:4d}/{:4d}],  step: {:3d},  Train Loss: {:.3f},  Train Acc: {:.2f},  {:8.1f} samples/sec,  {:.4f} sec/batch'
                                    .format(epoch, (batch_idx + 1) * len(inputs), len(self.dataloaders[phase].dataset),
                                            batch_idx + 1, batch_loss, batch_acc * 100, sample_per_sec, batch_time))
                                self.MinorRecords['train_loss'].append(batch_loss)
                                self.MinorRecords['train_acc'].append(batch_acc)
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0

                # Print the epoch information during the end of each epoch (both train and val)
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                logging.info('<info> Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc * 100, time.time() - epoch_start
                ))

                # Record the epoch information
                self.Records["%s_loss" % phase].append(epoch_loss)
                self.Records["%s_acc" % phase].append(epoch_acc)

                # save the best and the final model if needed
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        self.Records['best_epoch'] = epoch
                        logging.info("save best epoch {}, best acc {:.4f}".format(epoch, epoch_acc))
                        save_best_data_dir = os.path.join(self.save_dir,
                                                          'epoch{}-acc{:.4f}-best_model.pth'.format(epoch,
                                                                                                    epoch_acc * 100))
                        save_best_data = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                          'optimizer': self.optimizer.state_dict(), 'opt': self.args.opt}

                    if epoch == args.max_epoch - 1:
                        if args.save_model:
                            # save the best models according to the val accuracy
                            torch.save(save_best_data, save_best_data_dir)
                            # save the final models
                            logging.info("save final epoch {}, final acc {:.4f}".format(epoch, epoch_acc))
                            save_data = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                         'optimizer': self.optimizer.state_dict(), 'opt': self.args.opt}
                            torch.save(save_data,
                                       os.path.join(self.save_dir,
                                                    'epoch{}-acc{:.4f}-final_model.pth'.format(epoch, epoch_acc * 100)))
            # Update the learning rate each epoch
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # After training, save the records
        # stacks Record list to numpy when finished train process
        for k in self.Records.keys():
            self.Records[k] = np.array(self.Records[k])
        for k in self.MinorRecords.keys():
            self.MinorRecords[k] = np.array(self.MinorRecords[k])
        # save the records
        io.savemat(os.path.join(self.save_dir, "Records.mat"), self.Records)
        io.savemat(os.path.join(self.save_dir, "MinorRecords.mat"), self.MinorRecords)

        # log the best and final acc and loss, final acc is the mean of the last 5 epochs generally
        final_len = int(max(min(args.max_epoch * 0.5, 5), 1))
        info = "max train acc in epoch {:2d}: {:10.6f}\n".format(self.Records['train_acc'].argmax() + 1,
                                                                 self.Records['train_acc'].max()) \
               + "max val acc in epoch {:2d}: {:10.6f}\n".format(self.Records['val_acc'].argmax() + 1,
                                                                 self.Records['val_acc'].max()) \
               + "final train acc: %.6f\n final val acc: %.6f\n" \
               % (self.Records['train_acc'][-final_len:].mean(), self.Records['val_acc'][-final_len:].mean())
        for item in info.split('\n'):
            logging.info(item)
        with open(os.path.join(self.save_dir, 'acc output.txt'), 'w') as f:
            f.write(info)

    def plot_save(self):
        """
        plot confusion matrix and loss curve after training
        """
        # set color
        current_cmap = sns.color_palette("husl", 10)
        sns.set(style="white")
        sns.set(style="ticks", context="notebook", font='Times New Roman', palette=current_cmap, font_scale=1.5)

        # make dir
        self.save_dir_sub = os.path.join(self.save_dir, "postprosess")
        if not os.path.exists(self.save_dir_sub):
            os.makedirs(self.save_dir_sub)

        # plot confusion matrix
        mplb.rcParams['font.size'] = 12
        for phase in ['train', 'val']:
            f, ax = plt.subplots(figsize=(10, 8), dpi=100)
            sns.heatmap(self.c_matrix[phase], annot=True, ax=ax)
            ax.invert_yaxis()
            ax.set_xticklabels(self.label_name)
            ax.set_yticklabels(self.label_name)
            ax.set_xlabel('predict', fontsize=15)
            ax.set_ylabel('true', fontsize=15)
            ax.set_title('confusion matrix: %s' % phase, fontsize=18)
            f.savefig(os.path.join(self.save_dir_sub, "confusion_matrix_%s.jpg" % phase))

        # plot MinorRecords
        steps = np.arange(self.MinorRecords['train_loss'].shape[0]) + 1
        steps = steps / len(steps) * self.args.max_epoch
        fig = plt.figure(figsize=[10, 8], dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(steps, self.MinorRecords['train_loss'], "b-.d", markersize=3, linewidth=2)
        ax1.set_xlabel('epoch', fontfamily='monospace')
        ax1.set_ylabel('loss', fontfamily='monospace')
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(steps, self.MinorRecords['train_acc'] * 100, "b-.d", markersize=3, linewidth=2)
        ax2.legend(['train acc', 'val acc'], loc=5)
        ax2.set_xlabel('epoch', fontfamily='monospace')
        ax2.set_ylabel('acc', fontfamily='monospace')
        ax2.set_ylim([80, 100])
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir_sub, "Minor_loss_acc.jpg"))
