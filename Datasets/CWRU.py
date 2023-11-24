"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""

import pandas as pd
from Datasets.Dataset_utils.DatasetsBase import dataset
import Datasets.Dataset_utils.sequence_aug as aug1d
from Datasets.get_files.CWRU_get_files import get_files
from Datasets.get_files.generalfunciton import data_transforms1d
from Datasets.get_files.generalfunciton import balance_label
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
import pickle

# random seed
seed = 999
np.random.seed(seed)
random.seed(seed)


class CWRU(object):
    num_classes = 10
    inputchannel = 1

    def __init__(self, args):
        self.args = args # args = {'data_type': 'time', 'data_dir': './Datasets_dir/CWRU','test_size': 0.3,'normlizetype': 'mean-std'}

    def _preload(self,prefile_dir,args):
        """
        preload the data from the prefile_dir
        """
        # check if the prefile_dir exists
        if os.path.exists(prefile_dir):
            with open(prefile_dir, 'rb') as f:
                data_pd, storage_args, label_name = pickle.load(f)
                if storage_args == args: # check if the args is the same as the args in the prefile_dir, then preload the data
                    return data_pd,label_name
        # else, get the data and save it to the prefile_dir
        list_data, label_name = get_files(**args)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        with open(prefile_dir, 'wb') as f:
            pickle.dump((data_pd,args,label_name), f)
        return data_pd,label_name

    def data_preprare(self, signal_size=1024, SNR=None):
        test_size = self.args['test_size'] if 'test_size' in self.args.keys() else 0.3
        # preload the args
        temp_args = {'root':self.args['data_dir'], 'type':self.args['data_type'], 'signal_size':signal_size,
                     'downsample_rate':1,'SNR':SNR, 'load_condition':3}
        # preload the data
        data_pd, label_name = self._preload(os.path.join(self.args['data_dir'], 'data_buffer.pkl'), temp_args)
        # balance the label numbers to 450
        data_pd = balance_label(data_pd, 'label', 450)
        # split the data to train and val
        train_pd, val_pd = train_test_split(data_pd, test_size=test_size, random_state=40,
                                            stratify=data_pd["label"])
        # get the dataset
        train_dataset = dataset(list_data=train_pd, transform=data_transforms1d(aug1d, 'train', self.args['normlizetype']))
        val_dataset = dataset(list_data=val_pd, transform=data_transforms1d(aug1d, 'val', self.args['normlizetype']))
        return (train_dataset, val_dataset), label_name



if __name__ == '__main__': # check CWRU works well
    args = {'data_type': 'time', 'data_dir': './Datasets_dir/CWRU','normlizetype': 'mean-std'}
    cwru = CWRU(args)
    out = cwru.data_preprare()
    print(1)
