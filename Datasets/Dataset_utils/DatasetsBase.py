#!/usr/bin/python
# -*- coding:utf-8 -*-

from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, list_data, transform=None):
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        self.transforms = transform


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        label = self.labels[item]
        if self.transforms:
           seq = self.transforms(seq)
        return seq, label