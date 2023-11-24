"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def setseaborn():
    # set color
    current_cmap = sns.color_palette("deep")
    sns.set(style="whitegrid")
    sns.set(style="ticks", context="notebook", font='Times New Roman', palette=current_cmap, font_scale=2)
    return current_cmap


def setdefault():
    mpl.rcParams['axes.linewidth'] = 0.4
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['figure.figsize'] = [8 / 2.54, 6 / 2.54]
    mpl.rcParams['figure.dpi'] = 1000


def setaxesLW(myaxes, axes_lw=1, tick_len=3, tick_lw=None):
    if not tick_lw:
        tick_lw = axes_lw / 3 * 2

    for item in ['top', 'left', 'bottom', 'right']:
        myaxes.spines[item].set_linewidth(axes_lw)
    myaxes.tick_params(width=tick_lw, length=tick_len)


def PlotAcc(Datas, labels, ticklabels, savename, BackboneData=None, width=0.2, ylim=[95, 100],
            title='The channel number of preprocessing layer', xlabel='Model', ylabel='Test accuracy (%)', ygap=1,
            elinewidth=0.7, ncol=5):
    cmap = np.array(setseaborn())
    setdefault()
    color_alpha = 1
    cmap = cmap * color_alpha + np.array([1.0, 1, 1]) * (1 - color_alpha)
    labelsize, ticksize, legend_size = 6, 6, 6
    err_kw = {'elinewidth': elinewidth, 'ecolor': 'k'}

    X = np.arange(1, Datas.shape[0] + 1)
    fig = plt.figure()
    ax = fig.add_subplot()
    Bars = []
    # plot Datas
    for i in range(Datas.shape[1]):
        temp = ax.bar(X + width * (i + 1 / 2 - Datas.shape[1] / 2), Datas[:, i, 0], width, label=labels[i],
                      color=cmap[i + 1], yerr=Datas[:, i, 1], error_kw=err_kw)
        Bars.append(temp)
    h, l = plt.gca().get_legend_handles_labels()
    r, c = int(np.ceil(len(l) / ncol)), ncol
    order = [c * (i % r) + (i // r) for i in range(len(l))]
    leg = ax.legend([h[i] for i in order], [l[i] for i in order], prop={'size': legend_size},
                    bbox_to_anchor=(0.5, 1.05), loc='lower center', borderaxespad=0, ncol=ncol)
    leg.set_title(title, prop={'size': legend_size})

    # plot BackboneData
    if BackboneData is None:
        start = 1 - width * Datas.shape[1] / 2 - 0.1
        end = len(ticklabels) + width * Datas.shape[1] / 2 + 0.1
    else:
        temp = ax.bar([0], BackboneData[0], width, color=cmap[0], yerr=BackboneData[1], error_kw=err_kw)
        Bars.append(temp)
        start = 0 - width / 2 * 1 - 0.1
        end = len(ticklabels) - 1 + width * Datas.shape[1] / 2 + 0.1
    # ax set
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.set_xticks(np.arange(len(ticklabels)) if not BackboneData is None else np.arange(len(ticklabels)) + 1)
    ax.set_xticklabels(ticklabels, fontsize=ticksize)
    ax.set_xlim([start, end])
    if ylim:
        ax.set_ylim(ylim)
    if ygap:
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, ygap))
        ax.set_yticklabels(['{:d}'.format(int(item)) for item in np.arange(ylim[0], ylim[1] + 0.01, ygap)],
                           fontsize=ticksize)
    else:
        ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', linewidth=0.5)
    setaxesLW(ax, 0.8)
    fig.tight_layout(pad=0.2)
    # save
    fig.savefig(savename.replace('.jpg', '-withlegend.jpg'))


def main(filedir, focus_column=['model_name', 'mid_channel']):
    mid_channel = [16, 32, 64, 128]
    # read data from excel
    temp = pd.read_excel(filedir, sheet_name='df4')
    # obtain data by focus_column
    data = temp.loc[:, focus_column + ['mean max acc', 'std max acc']]
    data = data.set_index(focus_column)
    # prepare model acc from data
    model_names = ["Random_CNN", "TFN_STTF", "TFN_Chirplet", "TFN_Morlet"]
    model_new_names = ["Backbone\n_CNN", "Random\n_CNN", "TFN\n_STTF", "TFN_\nChirplet", "TFN\n_Morlet"]
    backbone_name = "Backbone_CNN"
    CNN_Data = data.loc[backbone_name, :].to_numpy().mean(0).squeeze()
    Datas = []
    for item in model_names:
        Datas.append(data.loc[item, :].loc[mid_channel].to_numpy())
    Datas = np.array(Datas)
    # plot
    PlotAcc(Datas, ['16', '32', '64', '128'], model_new_names, os.path.join(os.path.split(filedir)[0], '2-TestAcc.jpg'),
            CNN_Data)
