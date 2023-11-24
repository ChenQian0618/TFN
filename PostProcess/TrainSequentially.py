"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""
"""
This script is used to train the models sequentially.
Before run this, please set <--data_dir> to the directory of the CWRU dataset in <main.py> first.
"""

import os, sys

if __name__ == '__main__':
    # set the current directory
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print("current dir: %s" % os.path.curdir)
    # prepare the command lines
    model_list = ["Backbone_CNN", "Random_CNN", "TFN_STTF", "TFN_Chirplet", "TFN_Morlet"]
    max_epoch = 50
    mid_channel = [16, 32, 64, 128]
    command_lines = []
    SNR = 1e3
    checkpoint_dir = os.path.abspath(r'..\checkpoint\Acc-CWRU').replace('\\', '/')
    data_name = 'CWRU'

    for item in model_list:
        for item2 in mid_channel:
            if item == "Backbone_CNN" and item2 != 16:
                continue
            line = f' --data_name {data_name:s} --SNR {SNR:.2f}' + \
                   f' --model_name {item:s} --mid_channel {str(item2):s}' \
                   f' --checkpoint_dir "{checkpoint_dir:s}" --save_model False  --max_epoch {max_epoch:d}'
            command_lines.append(line)

    """
    round: the number of times to run each model, and we set 2 for saving time.
    start: the index of the command line to start with. (default: 0)
    """
    round = 2;
    start = 0
    for i, item in enumerate(command_lines):
        for p in range(round):
            if i * round + p + 1 <= start:
                continue
            print('-----' * 10)
            print('process of total: {:^3d}/{:^3d} \ncommand line: {:s}'.format(i * round + p + 1,
                                                                                len(command_lines) * round, item))
            temp = '%s "%s/main.py" %s' % (sys.executable, os.path.abspath('../'), item)
            print("Commandline: %s" % temp)
            # run the command line in the terminal (use main.py to train the model)
            os.system(temp)
            print('-----' * 10)
