"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""
import os
import numpy as np
from datetime import datetime
import pandas as pd


def ExtractInfo(filepath,append_acc = True):
    """
    extract the training info from the log file
    """
    start_time, record_time = None, None
    print(os.path.abspath(filepath))
    Dict = {'current_lr': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    Params = {'filename': os.path.split(os.path.split(filepath)[0])[-1]}
    with open(filepath, 'r',encoding='utf8') as f:
        temp = f.readline()
        flag_temp = True
        while '-----Epoch' not in temp:  # read params until epoch
            if ": " in temp[15:] and flag_temp:
                key = temp.split(": ")[0].split(" ")[-1]
                value = temp.split(": ")[1].replace("\n", '')
                Params[key] = value
            if '---------------------------------' in temp:
                flag_temp = False
            temp = f.readline()
        while temp != '':  # readlines until the end
            if '-----Epoch' in temp:  # read epoch info
                Dict['current_lr'].append(float(f.readline().split('current lr: ')[1].split(':')[-1].strip('\n[]')))
                temp = f.readline()
                if not start_time:
                    start_time = datetime.strptime(temp[:14], '%m-%d %H:%M:%S')
                while "<tempinfo>" in temp:  # skip tempinfo
                    temp = f.readline()
                Dict['train_loss'].append(float(temp.split('train-Loss: ')[1].split(' ')[0]))
                Dict['train_acc'].append(float(temp.split('train-Acc: ')[1].split(' ')[0].replace(',', '')))
                temp = f.readline()
                Dict['val_loss'].append(float(temp.split('val-Loss: ')[1].split(' ')[0]))
                Dict['val_acc'].append(float(temp.split('val-Acc: ')[1].split(' ')[0].replace(',', '')))
                end_time = datetime.strptime(temp[:14], '%m-%d %H:%M:%S')
            if '<training time>: ' in temp:  # extract training time
                record_time = float(temp.split('<training time>: ')[1].split(' ')[0].replace(',', ''))
            temp = f.readline()

    for key in Dict.keys():  # transform list to np.array
        if key != 'params':
            Dict[key] = np.array(Dict[key])
    Params['train_time'] = record_time if record_time else (end_time - start_time).total_seconds()
    if append_acc: # append max acc and final acc
        Params['max acc'] = max(Dict['val_acc'])
        final_len = int(max(min(len(Dict['val_acc'])*0.5,5),1))
        Params['final acc'] = Dict['val_acc'][-final_len:].mean()
    return Params,Dict


def acc2csv(savepath,data,focus_column = ['model_name']):
    """
    save the training info to excel
    """
    # transform data to dataframe
    df1 = (pd.DataFrame(data).sort_values(by=focus_column).reset_index(drop=True))
    df1 = df1.apply(pd.to_numeric,errors='ignore')

    # dataframe 2
    number = 3 if 'train_time' in df1.columns else 2
    col = df1.drop(['filename','checkpoint_dir'],axis=1).columns.to_list()[:-number]
    df2 = df1.drop(['filename','checkpoint_dir'],axis=1).set_index(col)
    df2 = pd.concat([df2.groupby(col).mean(),df2.groupby(col).std()],axis=1,keys=['mean','std']).reset_index()
    df2.columns = [f'{i} {j}'.strip() for i, j in df2.columns]
    # obtain dataframe 3 by focus_column
    try:
        temp = df1.loc[:,focus_column+['train_time','max acc', 'final acc']].set_index(focus_column)
    except:
        temp = df1.loc[:, focus_column + ['max acc', 'final acc']].set_index(focus_column)
    meanstddf = pd.concat([temp.groupby(focus_column).mean(),temp.groupby(focus_column).std()],axis=1,keys=['mean','std']).reset_index()
    df3 = meanstddf.set_index(focus_column)
    df3.columns.names = ['mean/std','max/final']

    # obtain dataframe 4 by focus_column
    df4 = meanstddf
    df4.columns = [f'{i} {j}'.strip() for i, j in df4.columns]

    df_save = {'df1': df1, 'df2': df2, 'df3': df3, 'df4': df4}

    # write to excel
    with pd.ExcelWriter(savepath) as writer:
        for item in ['df1', 'df2', 'df3', 'df4']:
            df_save[item].to_excel(writer, item)


if __name__ == '__main__':
    ExtractInfo(r'../../checkpoint/Acc-CWRU/Backbone_CNN-CWRU-time-1122-200445/training.log', append_acc=True)
