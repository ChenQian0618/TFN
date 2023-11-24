"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""
"""
This script is used to extract the training info from the log files and save to excel, and plot the accuracy figure.
"""

import os
from process_utils.processlib import ExtractInfo
from process_utils.processlib import acc2csv
from process_utils.PlotAccuracy import main as PlotaccMain

def main(root = '../checkpoint/Acc-CWRU',logname = "training.log"):
    # extract training info from log files
    subdirs = next(os.walk(root))[1]
    Info = []
    for subdir in subdirs:
        if subdir != "postfiles":
            filepath = os.path.join(root, subdir, logname)
            Params, Dict = ExtractInfo(filepath)
            Info.append(Params)
    #  save to excel and plot
    save_dir = os.path.join(root, 'postfiles')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    acc2csv(os.path.join(save_dir, "1-Acc_statistic.xlsx"), Info,focus_column = ['model_name','mid_channel'])
    PlotaccMain(os.path.join(save_dir, "1-Acc_statistic.xlsx"),focus_column = ['model_name','mid_channel'])

if __name__ == '__main__':
    # set the current directory
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print("current dir: %s" % os.path.curdir)
    # acc statistic
    main(root = '../checkpoint/Acc-CWRU')